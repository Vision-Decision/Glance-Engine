# Exactness

Every optimization in this engine reorganizes the reference computation; none changes the model. "Same answer" has three different strengths here, and this page says which one applies where.

| Claim | Where it holds | Evidence |
|---|---|---|
| **Matches the reference to FP32 rounding** | Tiny random model in FP32 on CPU (GPU-only kernels are not exercised here): cached, fresh and query-first scans, packed variable-length prefill, clips and the Scanner API against each image run alone through the glance-core reference forward and the plain HF forward | Largest relative error 2.04×10⁻⁶ against a 10⁻⁴ gate. [Log](../outputs/engine_opt/09_numerics/raw/item9_default_cpu.log) · `pytest tests/test_equivalence_cpu.py` |
| **Bit-identical** | The box readout on fixed hidden states, across batch and padding layouts; lossless bank formats; W12 weight decoding (W12 GEMM outputs can differ in BF16 rounding) | 210/210 real-weight readout replays bit-identical. [Item 9](../outputs/engine_opt/09_numerics/summary.md) · Lossless banks reproduced dense-bank probabilities on 455 image and 5,419 clip decisions. [Item 8](../outputs/engine_opt/08_bank_memory/summary.md) · Every W12 weight reconstructed bitwise. [Item 5](../outputs/engine_opt/05_w12/summary.md) |
| **Within calibrated BF16 gates** | Real weights in BF16 on GPU, end to end | All decisions agree on the suites below (one exception on a separate panel, noted there); continuous outputs stay inside twice the spread the shipped BF16 model shows against itself. Details below. |

## Why BF16 needs gates, not bit-equality

The shipped BF16 model does not agree with itself bit for bit. Running the shipped reference with the same images, weights and question, and changing only the batch composition or padding, changes BF16 rounding in the backbone and therefore the hidden states. Any engine that batches differently inherits the same freedom. The gates are calibrated on that self-disagreement:

1. Run the shipped reference in five equivalent batch/padding layouts on 24 fixtures (ten layout pairs per fixture).
2. Record the largest difference of each output between layouts.
3. Accept an engine if its difference from the reference stays within **twice** that spread, and require every presence and typed decision to agree.

The frozen limits are in [`scan/equivalence_limits.py`](../scan/equivalence_limits.py):

| Quantity | Reference spread | Acceptance limit |
|---|---:|---:|
| Presence probability | 0.015852 | 0.031704 |
| Typed probability | 0.011677 | 0.023353 |
| Box coordinate (normalized), any centre | 0.212472 | 0.424945 |
| IoU probability, any centre | 0.203709 | 0.407418 |
| Box coordinate, matching centres | 0.018160 | 0.036320 |
| IoU probability, matching centres | 0.024313 | 0.048626 |
| Box coordinate at the engine's selected patch | 0.046973 | 0.093947 |

The matching-centre gates are 11.7× (box) and 8.4× (IoU) tighter than the raw limits, which is what makes them useful: an injected 0.04 error on one coordinate left all 234 decisions intact and still failed the 0.03632 same-centre gate. A one-token shift of the packed readout also failed. Both negative controls are part of the GPU suite. [Fault logs](../outputs/engine_opt/09_numerics/summary.md#verification)

### Results against the gates

| Check | Result | Source |
|---|---|---|
| Real-weight suite, both attention masks, standard and fused prefill | 234/234 decisions each, all continuous gates pass | [Item 9 verification](../outputs/engine_opt/09_numerics/summary.md#verification) |
| 60-image RefCOCO diagnostic, every BF16 engine variant vs shipped BF16 | 930/930 presence and typed decisions | [Diagnostic](../outputs/engine_opt/09_numerics/raw/item9_release_diag.json) |
| Scan benchmark: fresh, cached, separate-engine and native-clip paths | Every decision in 20 comparisons of 24 inputs | [Scan equivalence](../outputs/engine_opt/09_numerics/raw/item9_release_equiv.json) |
| Scheduler: interleaved concurrent requests vs isolated and vs shipped | 254/254 decisions each | [Interleaving](../outputs/engine_opt/09_numerics/raw/item9_release_interleaving.json) |

Decisions are not unconditionally identical. On a fresh Tv640 COCO panel, one of 192 presence comparisons crossed the 0.5 threshold: both engine heads returned 0.508259 where the shipped singleton returned 0.490466. [Presence audit](../outputs/engine_opt/09_numerics/raw/item9_release_presence_audit.json)

## Known limit: BF16 box centres near a tie

The box head picks a patch with a hard argmax, then regresses the box from that patch. When two patches score within a few hundredths of a logit, BF16 noise in the backbone can flip the choice, and the box moves to another object part or instance. The readout itself is not the cause: on fixed hidden states it is bit-identical across layouts (fixed-size FP32 projection blocks, a fixed FP64 reduction per patch, lowest index on an exact tie).

| Backbone precision | Centre switches across five layouts | Normalized-coordinate spread | Latency, 32 box questions |
|---|---:|---:|---:|
| BF16 (shipped and engine) | 28 / 420 | 0.212472 | 96.37 ms (shipped forward) |
| FP32 (control) | 0 / 420 | 0.000357 | 300.68 ms (3.12×) |

Presence and typed decisions matched on all 930 diagnostic questions (see the one threshold crossing above for a separate panel). The fused BF16 engine matched the shipped BF16 centre on 472 of 570 box questions; 97 of its 98 switches had reference margins at or below 0.06 logit. An FP32 backbone is a measured control today, not an engine mode; it is the candidate opt-in exact mode. [Item 9](../outputs/engine_opt/09_numerics/summary.md) · [FP32 control](../outputs/engine_opt/09_numerics/raw/item9_precise_fp32_probe.json) · [Timing](../outputs/engine_opt/09_numerics/raw/item9_controls_precision_latency.json)

**What this means in practice.** Use `p_present` and typed probabilities as decisions. Treat a box whose question is ambiguous (two similar objects, a large object with several plausible anchor patches) as one valid answer among near-ties; `p_iou50` is lower on such boxes.

## Lossy modes are separate

These are opt-in and reported separately; they are not covered by the gates above.

| Mode | Agreement with shipped BF16 | Source |
|---|---|---|
| FP8 language model (`fp8=True`) | 926/930 decisions | [Item 9 diagnostic](../outputs/engine_opt/09_numerics/summary.md#decision-and-center-agreement) |
| FP8 language and vision models | 923/930 decisions | same |
| Lossy bank formats (`state_bf16`, `state_fp8`, `state_int8`, `kv_fp8`, `state_kv_fp8`) | Flip counts per format | [Item 8](../outputs/engine_opt/08_bank_memory/summary.md) |

## Running the gates

CPU, tiny model (runs in CI):

```bash
pytest tests/test_equivalence_cpu.py          # or: python -m scan.test_scan_equivalence
```

GPU, real weights (one RTX PRO 6000; each takes minutes). `CKPT` is a training checkpoint such as `Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt`, downloaded locally:

```bash
python -m scan.test_scan_equivalence --real --ckpt $CKPT                               # 234 decisions + gates
python -m scan.test_scan_equivalence --real --ckpt $CKPT --fused-prefill optimized     # production prefill
python -m scan.test_scan_equivalence --real --ckpt $CKPT --inject small_fault          # must exit 1
python -m speed.diag_equiv --ckpt $CKPT --out diag.json                                # 930-decision diagnostic
python -m scan.bench_scan --parts equiv --ckpt $CKPT --out equiv.json                  # scan paths
python -m speed.test_scheduler_gpu --ckpt $CKPT --out interleaving.json                # concurrent requests
python -m scan.test_scan_equivalence --real --ckpt $CKPT --measure-reference ref.json  # recalibrate the spread
```

A passing suite prints `SCAN_EQUIVALENCE_OK`; the injected faults print `SCAN_EQUIVALENCE_FAIL`. [`scan/vm/numerics.sh`](../scan/vm/numerics.sh) runs the whole set with GPU locking and logging.
