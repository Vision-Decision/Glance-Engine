# API reference

## Python

```python
from glance_engine import Glance, Box, YesNo, Choice, Score
```

### `Glance`

| Member | Description |
|---|---|
| `Glance.from_pretrained(model="Vision-Decision/Glance-4B-preview", *, device="cuda", fp8=False, w12=None, graphs=True, max_vis=None, cache_gib=8.0, max_tokens=8192, calibrated=True)` | Load a checkpoint and build the engine with the benchmark weight path (merged LoRA, fused projections, CUDA graphs). `model` is a release repo or directory (`glance_config.json`, `adapter_model.safetensors`, `heads.safetensors`), a training `.pt` file, `"org/repo:path/ckpt.pt"` on the Hub, or `"tiny"`. Real checkpoints need CUDA. |
| `Glance.tiny_x(**kw)` | Random tiny Glance-X (K=2, all three directions) in FP32 on CPU. See [Glance-X](glance-x.md). |
| `Glance.tiny(**kw)` | Random tiny Qwen3.5 model in FP32 on CPU. Same code path, meaningless answers. Downloads only the base config and processor. |
| `add_image(image) -> str` | Run the vision tower and prefill once; keep the prefix in the cache; return its 64-hex id. |
| `ask(image, questions, *, cache=True, direction=None) -> Answers` | Answer every question about one image in one forward. `image` is a PIL image, a path, an RGB `numpy` array or an id. With `cache=True`, `Answers.image_id` names the kept prefix. An unknown or evicted id raises `KeyError`. `direction` (Glance-X only): `"image_first"` (default), `"text_first"` or `"joint"`. |
| `ask_each(images, questions, *, cache=False, direction=None) -> list[Answers]` | Broadcast the same questions over several images in one batched call. |
| `forget(image_id)` | Drop a cached prefix. |
| `scanner(*, max_vis=256, alpha=0.1, q=0.1) -> Scanner` | One question over many images or clips (below). |
| `engine`, `worker` | The underlying `ScanEngine` and `DecisionWorker`. |
| `temperatures`, `calibrated` | Per-type temperatures from the release config (`presence`, `noul`, `choice`, `score`), and whether `ask` applies them. |

Images are resized to at most `max_vis` visual tokens (one token is 32×32 pixels; sides are multiples of 32; upscaling is capped at 2×), as during training. `Glance` is not thread-safe; serve concurrent clients with `glance-serve`, whose scheduler owns the device.

Glance-X checkpoints load through the same `from_pretrained`: a `glancex_config.json` release (for example `"Vision-Decision/Glance-X-Lite"`), a glancex training `.pt`, or `"tiny-x"`. `exact=True` keeps their LoRA unmerged. They answer `Box` questions only. `BoxAnswer` then also has `boxes`, `boxes_px` and `box_scores`, every target from the multi-box head, and `glance.scanner()` returns a Glance-X scanner with `mode="cached" | "fresh" | "text_first" | "joint"`. See [Glance-X](glance-x.md).

### Questions

| Class | Wire format | Answer |
|---|---|---|
| `Box(text)` | `{"type": "box", "text": ...}` | `BoxAnswer`: `p_present`, `box` [x1, y1, x2, y2] normalized to the image (not clipped, so edges can fall slightly outside 0..1), `box_px` in input pixels, `p_iou50` = P(IoU ≥ 0.5), `present` |
| `YesNo(question, short=False)` | `{"type": "noul", "instr": ...}` | `YesNoAnswer`: `p_yes`, `answer` |
| `Choice(question, options)` | `{"type": "choice", "instr": ..., "options": [...]}` | `ChoiceAnswer`: `probs` (option → probability, in order), `answer` |
| `Score(question, levels)` | `{"type": "score", "instr": ..., "options": [...]}` | `ChoiceAnswer` over the levels |

`short=True` selects the distilled short scan readout, which needs a checkpoint trained for it (`Vision-Decision/glance-train-ckpts:scan/short_v1/`). Plain dicts in wire format are accepted wherever a question is.

**Calibration.** Release checkpoints carry post-hoc temperatures fitted on held-out training-distribution splits. `ask` divides each head's logits by its temperature (presence for boxes; `noul`, `choice`, `score` for typed answers). `p_iou50` is not rescaled. Training `.pt` checkpoints carry no temperatures, so their answers are the raw head probabilities, the same values the HTTP server returns.

### `Scanner` ([`scan/api.py`](../scan/api.py))

```python
scanner = glance.scanner(max_vis=640)
r = scanner.scan("a giraffe", images)                 # first call on this list encodes the bank
r = scanner.scan("a toaster", images)                 # same list object: only the query runs
r = scanner.scan("a person falling", clips, readout="short")   # clips: lists of frames in time order
scanner.calibrate(negative_scans)                     # per-image presence logits of scans without the target
```

| Argument | Values |
|---|---|
| `mode` | `"cached"` (default: bank keyed by the list object and length), `"fresh"` (no bank, packed prefill), `"query_first"` (query text before the image; out of training distribution, for measurement) |
| `readout` | `"box"` (presence, box, P(IoU)), `"yes"` (yes/no block), `"short"` (distilled scan readout) |

Returns `{"images": [...], "set": {...}, "latency_ms": {...}, "mode", "n"}`. `set` has `any` (some P > 0.5), `argmax`, `curve` and `count_frames_naive`; after `calibrate`, also `none_conformal` (true when no image clears the max-statistic threshold; false alarms on target-free scans ≤ α on average over calibration draws), `top` (Benjamini–Hochberg selection at level `q`), `count_frames` and `count_events`.

### Lower-level modules

| Import | Contents |
|---|---|
| `glance_engine.engine` | `GlanceEngine`, `ScanEngine`, `FreshPack`, `packed_forward`, `merge_lora_`, `fuse_weights_`, `fp8_language_model_`, item builders |
| `glance_engine.serving` | `Scheduler`, `DecisionWorker`, `Decision`, `ImageRef`, `PrefixCache`, `application` |
| `glance_engine.banks` | `StoredBank`, `FORMATS`, `HostBank`, `encode` / `decode` for lossless state planes |
| `glance_engine.checkpoints` | `load`, `tiny`, `release_config`, `load_release_weights` |
| `glance_engine.kernels` | `KERNELS`, `report()`; `python -m glance_engine.kernels` |

`ScanEngine.encode(units, bank_format=..., tier="gpu" | "cpu", readout="typed")` and `ScanEngine.query(bank, question)` are the bank primitives; see [item 8](../outputs/engine_opt/08_bank_memory/summary.md#equivalence-and-reproduction) for the format and tier options.

## HTTP

```bash
glance-serve [--model REPO|DIR|CKPT.pt | --tiny] [--host 127.0.0.1] [--port 8002] [--cache-gib 8] \
             [--max-tokens 8192] [--wait-ms 0.5] [--max-batch 64] [--queue-limit 256] [--timeout 120] \
             [--calib calib.json] [--readout yes|short] [--no-graphs]
```

One server owns one GPU. It binds to loopback by default and drains accepted requests on shutdown. The HTTP server returns raw head probabilities (no temperature scaling).

### `POST /v1/decisions`

```json
{"image": "<base64 or data URL>", "cache": true, "max_vis": 640,
 "questions": [{"type": "box", "text": "the giraffe"},
               {"type": "noul", "instr": "Is this photo taken outdoors?"},
               {"type": "choice", "instr": "How many giraffes?", "options": ["one", "two", "three or more"]}]}
```

- `images` (list, ≤ 256) instead of `image` broadcasts the questions over every image.
- `image_ids` (from an earlier reply with `cache: true`) instead of image data skips upload, vision and prefill.
- Up to 64 questions per request and 26 options per question; `max_vis` between 16 and 4,096 (default 640).

```json
{"object": "glance.decisions", "model": "glance", "sec": 0.041,
 "results": [[{"p_present": 0.97, "box": [0.41, 0.38, 0.65, 0.81], "p_iou50": 0.88},
              {"probs": [0.03, 0.97]},
              {"probs": [0.12, 0.83, 0.05]}]],
 "image_ids": ["7b28aaa1…"]}
```

`results` is image-major, then question order. Typed `probs` follow label order: `[no, yes]` for `noul`, the supplied option order for `choice` and `score`. (Values above illustrate the shape.)

With a Glance-X model (`glance-serve --model Vision-Decision/Glance-X-Lite`):
- Requests may set `"direction": "image_first" | "text_first" | "joint"`. Only `image_first` is cached; v1 models reject the other two.
- Box results add `boxes` and `box_scores`.
- Non-box questions and the video endpoints return 400.

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | One user message with `image_url` data URLs and a yes/no question; the assistant message content is the decision results as JSON. No streaming. |
| `POST /encode` | `{"video", "clips": [[frame, ...], ...], "times", "dt"}`: build and keep a clip bank under a name |
| `POST /scan` | `{"video", "query", "ops": ["curve", "ever", "when", "count"], "alpha", "q"}` over a stored bank; `ever`/`when`/`count` carry guarantees only with `--calib` |
| `POST /score_clips` | Score clips directly (`kind`: `yesno` or `choice`), cached by content |
| `GET /health`, `/ready`, `/metrics` | Status, readiness (503 while draining or not started), admission counts, batch sizes, latency, cache bytes |

### Errors

| Status | Meaning |
|---|---|
| 400 | Invalid JSON, question, image or option |
| 400 | Malformed `image_ids` entry (not 64 hex characters) |
| 404 | Unknown or evicted `image_ids` entry; `/scan` on an unknown video |
| 413 | Request body over 64 MiB |
| 429 | Queue full |
| 503 | Draining, or a device OOM that could not be split |
| 504 | Request deadline (`--timeout`) exceeded |

Errors have the form `{"error": {"type": "...", "message": "..."}}`.
