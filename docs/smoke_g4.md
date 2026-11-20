# G4 smoke test of the release

The README paths, run once on a fresh Colab G4 VM on 2026-11-20 against the public checkpoint. This checks that the release works end to end on the benchmark hardware. It is not a benchmark: the image is a small COCO photo (360×270, far below the 640-token budget), and the CEILING runs remain the reference numbers ([performance](performance.md)).

**Setup.** NVIDIA RTX PRO 6000 Blackwell Server Edition, torch 2.14.1+cu130 (Colab image), transformers 5.23.0, flash-linear-attention 0.6.0, standard FLA stack (no `[sm120-prefill]`). Model: `Glance.from_pretrained("Vision-Decision/Glance-4B-preview")`. Colab's preinstalled torchao 0.10.0 makes peft refuse to import; `pip uninstall -y torchao` fixes it (as `speed/setup_vm.sh` does).

| Path | Result |
|---|---|
| `from_pretrained` (weights already in the HF cache) | 5.7 s to a ready engine |
| README quickstart, first `add_image` (includes Triton JIT and CUDA graph capture) | 27.5 s, once per process |
| Quickstart answers on `211069.jpg` | box "the giraffe on the right": P(present) 0.995, P(IoU≥0.5) 0.991, box on the right-hand giraffe; outdoors: P(yes) 0.997; how many giraffes: "two" 0.893 |
| Follow-up, 3 questions on the cached image | 10.5 ms (median of 30) |
| Follow-up, 32 box questions on the cached image | 19.3 ms |
| Uncached image, 1 / 32 box questions | 26.1 / 39.2 ms |
| Cached scan, 4 images at 640 tokens | "a giraffe": P(present) 0.988–0.994 on all four; "a toaster": ≤ 0.0042 on all four |
| Cached scan, the same 4 photos repeated to N = 256 | bank encode 14.4 s (first call, includes capture); query 81.1 ms (median of 10 changing questions after warm-up) |
| `examples/quickstart.py`, `examples/cached_scan.py` | exit 0, same answers |
| `glance-serve` + README curl | HTTP 200, two answers and an `image_ids` entry (first request 2.5 s, includes capture) |
| HTTP follow-up by `image_ids` / resent image (2 questions) | 9.7 / 31.6 ms round trip on loopback (server-side 9.3 / 31.2 ms), median of 30 |

First calls of each new shape (batch, prefix length, question count) capture CUDA graphs and compile Triton kernels, which takes seconds; the latencies above are after one warm-up of each shape. The first query of a new question length on a 4-image bank took 1.3–1.6 s for the same reason.

The script and driver are reproduced from the release branch: `glance_engine`, `speed`, `scan`, `examples` and `demo/data/search_large`, with glance-core v0.1.0 on `PYTHONPATH`.
