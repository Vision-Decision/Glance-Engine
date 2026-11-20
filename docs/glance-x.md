# Glance-X models

The engine also serves the X series:
- [Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B): K=8, all three directions
- Glance-X2: same architecture as X-4B, training `.pt`
- [Glance-X-Lite](https://huggingface.co/Vision-Decision/Glance-X-Lite): 0.8B, K=2, `image_first` and `text_first`

The X models come from [glancex](https://github.com/Vision-Decision/Glance-X-Lite). They are box models with one
backbone and three reading directions. They load through the same `Glance` API and `glance-serve` server. The Glance v1
paths are unchanged.

```bash
pip install -e '.[gpu,serve,x]'
```

```python
from glance_engine import Box, Glance

gx = Glance.from_pretrained("Vision-Decision/Glance-X-Lite")   # BF16 on CUDA; "tiny-x" runs on CPU
image_id = gx.add_image("demo/data/search_large/211069.jpg")     # vision tower + image prefix, once
answers = gx.ask(image_id, [Box("the giraffe on the right"), Box("a red umbrella")])
answers[0].boxes                                                  # every target (multi-box head), normalized
gx.ask("demo/data/search_large/211069.jpg", [Box("the giraffe")], direction="text_first")
gx.scanner(max_vis=640).scan("a giraffe", images)                # cached bank; mode="text_first" reads the request first
```

## How the X backend differs from v1

| | Glance v1 | Glance-X ([`glance_engine/x/`](../glance_engine/x)) |
|---|---|---|
| Sequence | one causal sequence: chat header, image, question block | image stream and request stream; for K layers neither sees the other |
| Positions | consecutive | image M-RoPE from 0; request from the fixed `text_pos0` = 66 |
| Directions | image first | `image_first` (cacheable), `text_first`, `joint` where the checkpoint was trained in it |
| Questions | Box, YesNo, Choice, Score | Box only; answers add `boxes`, `box_scores`, `boxes_px` |
| Cache entry | GDN state, conv tail, K/V of every layer | the same, for the fusion layers K+1..N, plus the final image tokens |

`image_first` is split into `encode` and `answer` steps, the same way the v1 engine splits a prefill:
- `encode(images)` runs the image prefix once and keeps it in the prefix cache under the usual 64-hex image id.
- `answer(prefixes, rows)` answers any number of request blocks against any of those prefixes in one forward.

Each distinct request runs the modality stage (layers 1..K) once, and the result is shared by every image it is asked
about. Each block then continues from its own image's recurrent state and keys, using glancex's prefix/block kernels.

`text_first` and `joint` cannot be cached because their image tokens read the request. They run glancex's own forward.

On CUDA the encode step, the answer step and the `text_first`/`joint` forward are replayed as CUDA graphs, one per
shape bucket:
- images: 1–16
- rows: 1–256
- prefix and visual lengths: multiples of 64
- request length: a multiple of 8

Padding rows repeat a real image or request and are dropped. Padding tokens are zeroed or masked. Images are resized
and uploaded on worker threads and normalised on the GPU. Large banks run eagerly in chunks of 256.

## Exactness and speed (H200, BF16)

`python -m glance_engine.x.check --model <id>` compares the engine with `glancex.GlanceX(fast=False)` on the four demo
images × 16 requests, every trained direction. Raw reports:
[X-Lite](../outputs/glance_x/check_x_lite_h200.json), [X-4B](../outputs/glance_x/check_x_4b_h200.json).

| Gate | X-Lite | X-4B |
|---|---|---|
| `exact=True` (unmerged LoRA) vs glancex, every direction | identical (max \|ΔP\| 0, box 0 px) | identical, including `joint` |
| default (merged LoRA, graphs) vs glancex: max \|ΔP(present)\|, present agreement | 0.017, 64/64 | 0.012, 64/64 |
| cached follow-up vs fresh, cached scan vs fresh | identical | identical |

On a near-tie between two patches, BF16 noise can move the box centre to the neighbouring patch: up to 50 px on
X-Lite and 119 px on X-4B (`joint`). The median box error is 0.2–0.3 px. This is the
[same known limit](exactness.md#known-limit-bf16-box-centres-near-a-tie) as v1.

Warm medians in ms on one H200. glancex's fast path (merged LoRA, CUDA graphs) runs on the same inputs.

| Workload | X-Lite engine | X-Lite glancex | X-4B engine | X-4B glancex |
|---|---:|---:|---:|---:|
| One image, 1 request | **28.5** | 30.8 | **45.1** | 51.6 |
| One image, 32 requests | **32.8** | 42.5 | **61.0** | 90.3 |
| Follow-up on a cached image, 32 requests | 13.0 | — | 27.8 | — |
| Cached scan, 1 request × 256 images (X-4B: 128) | **58.1** | 62.3 | **75.0** | 79.5 |
| `text_first`, 1 request × 64 new images | **478** | 530 | **1,269** | 1,433 |

Not yet fast: encoding a bank runs the vision tower and prefix eagerly per 16-image graph, and copies each prefix out
of the batch. This took 4.0 s for 256 images on X-Lite and 5.0 s for 128 images on X-4B.

## Not supported for Glance-X

- YesNo, Choice and Score questions, and the video and clip endpoints (`/encode`, `/scan`, `/score_clips`). These
  need a Glance v1 model and return a 400 error.
- `fp8` and `w12` weight formats, and the SM120 fused kernels. The X backend uses glancex's PyTorch and FLA kernels.
- Glance-X-Flash. Its selective-joint and early-exit modules are not in the public glancex package.
