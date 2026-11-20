# Performance

All numbers are from the frozen optimization series ([`outputs/engine_opt/CEILING.md`](../outputs/engine_opt/CEILING.md), tag `engine-opt-freeze-1`) and link to the raw result files there. Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4, SM120). Software: [`constraints/g4-benchmarks.txt`](../constraints/g4-benchmarks.txt).

## Summary

- **Against BF16 servers:** Glance has the lower median latency in all **65** workload cells, against both vLLM 0.34.0 and SGLang 0.5.23.
- **Against lossy vLLM (FP8 weights + MTP speculative decoding):** 24 cells have a matched full-output FP8+MTP run. vLLM wins **one**: a 4,096-token ScreenSpot-Pro screenshot with one box question, 262.4 ms against Glance's 277.9 ms. Glance runs BF16 throughout.
- **The size of the lead depends on the workload.** It is largest where decoding or repeated prefill dominates the baseline, and smallest where every engine is bound by the same prefill.

| Family | Cells | vLLM BF16 ÷ Glance | SGLang BF16 ÷ Glance | vLLM FP8+MTP ÷ Glance | Glance at % of modeled floor |
|---|---:|---|---|---|---|
| [One image, ≤640 visual tokens](../outputs/engine_opt/CEILING.md#single-image-640-visual-tokens) | 5 | 4.0–6.8× | 5.0–13.5× | 2.6–4.7× | 46–57% |
| [Follow-ups on a cached image](../outputs/engine_opt/CEILING.md#follow-up-questions-on-one-cached-image) | 6 | 6.4–19.1× | 14.1–33.6× | 3.9–11.0× | 38–56% |
| [GUI screenshots, 2k / 4k tokens](../outputs/engine_opt/CEILING.md#screenspot-pro-screenshots) | 6 | 1.3–3.9× | 1.8–9.6× | 0.94–2.9× | 52–60% |
| [Video, 32 / 64 frames](../outputs/engine_opt/CEILING.md#video-style-speed-workload-one-box-question-per-frame) | 2 | 1.6–1.9× | 1.8–2.3× | 1.2–1.3× | 73–74% |
| [Cached scan, N = 64 / 256](../outputs/engine_opt/CEILING.md#cached-scan-one-new-question-over-n-independent-prefixes) | 14 | 3.5–37.9× | 7.6–80.0× | 12.8–30.4× (5 cells) | 53–80% |
| [Fresh scan, N = 32 … 256](../outputs/engine_opt/CEILING.md#fresh-scan-one-yesno-question-over-n-new-frames) | 8 | 1.03–1.24× | 1.2–1.8× | no matched run | 74–78% |
| [Concurrent clients, C = 1 … 64](../outputs/engine_opt/CEILING.md#concurrent-clients) | 24 | 1.03–16.1× | 1.3–19.2× | no matched run | 12–77% |

Ratios are baseline median ÷ Glance median per cell, computed from the per-family tables in CEILING.md; the FP8+MTP column covers the 24 matched cells.

## How to read the numbers

- **Warm medians.** Each cell is the median of repeated warm calls (50 for single-image rows, 15 changing questions for cached scans, 10 calls for fresh scans, five bursts for clients). Cached rows exclude bank construction; follow-ups use a retained prefix.
- **Original → Final.** "Original" is the engine before the series (the fused sprint engine, the first scan bank and the tuned 4k-chunk fresh runs). "Final" is the most recent production measurement of each workload.
- **Same task for every engine.** vLLM and SGLang run the generative Qwen3.5-4B on the same resized pixels and question text, with prefix caching, CUDA graphs and continuous batching. A box answer is generated as up to 24 tokens; a yes/no scan generates one token with logprobs. Fresh-scan baselines reset the encoder cache between calls; cached-scan baselines see each image once before timing, the most favourable protocol for them.
- **Glance answers with its own model.** The baselines measure the time to the same kind of answer with the generative base model. They do not compare accuracy between the two models.
- **Floor.** "% of modeled floor" is 100 × (modeled hardware floor ÷ measured Glance time), from measured G4 peaks (401.7 BF16 TFLOP/s, 1,463 GB/s device bandwidth, 56.9 GB/s pinned H2D) and the useful work in each call. [Roofline](../outputs/engine_opt/ROOFLINE.md)
- **Bursts end at the last answer.** Client rows time a simultaneous burst until the last client has all its answers. In item 6, vLLM returned the *median* client earlier in the 64-client yes/no burst even though Glance finished the burst first. [Item 6](../outputs/engine_opt/06_scheduler/summary.md)

## Where the lead is small, and where Glance loses

- **The loss.** 4,096-token GUI screenshot, one box question: vLLM FP8+MTP 262.4 ms, Glance BF16 277.9 ms. At eight questions on the same screenshot Glance leads again (282.9 vs 299.5 ms at Q = 8). Glance is not using FP8 here; the optional FP8 language model exists but changes numerics ([exactness](exactness.md#lossy-modes-are-separate)). [Glance run](../outputs/engine_opt/03_lm_prefill/raw/item3_gui4096.json) · [vLLM run](../outputs/sprint_speed/raw/serve_vllm_fp8mtp_g4.json)
- **Fresh scans.** One new yes/no question over N new frames is pure prefill for every engine. vLLM BF16 takes 3–24% longer; the closest cell is Tv256, N = 256: 2,420.5 ms against 2,495.9 ms.
- **Cold yes/no bursts.** At 16 clients, 370.3 ms against vLLM's 381.7 ms.

## Largest remaining gaps to the hardware floor

From CEILING.md's final section:

- **64 clients × 32 fresh frames:** 9,695.5 ms against a 7,153.5 ms floor. Vision and upload are 1,431.2 ms above their floor; packed LM and readout 1,062.1 ms.
- **Fresh Tv256, N = 256:** 634.4 ms above the floor, mostly vision/upload (353.1 ms) and LM/readout (249.0 ms). Faster patch projections changed outputs on the numerical checks, so the reference patch convolution stays.
- **64 cached-box clients resending pixels:** 11.5% of the floor; the CPU spends 87.8 ms preparing and hashing resent images. Sending the returned image ids instead took 66.3 ms. [Image-id timings](../outputs/engine_opt/06_scheduler/raw/item6_production_ids.json)
- **256 cached four-frame clips:** 241.4 ms against a 127.2 ms floor; the reference-compatible GDN continuation is the largest stage gap.

## How the engine got here

Nine items, each with its own report, raw timings, equivalence checks and before/after controls. Highlights:

| Item | Change | Before → after |
|---|---|---|
| [1](../outputs/engine_opt/01_varlen_prefill/summary.md) | Packed variable-length fresh prefill | Fresh Tv128/N128: 2,586.2 → 931.1 ms |
| [2](../outputs/engine_opt/02_vision_prep/summary.md) | Compact RGB upload, fused preprocessing and vision ops | 933.3 → 774.2 ms |
| [3](../outputs/engine_opt/03_lm_prefill/summary.md) | Fused SwiGLU epilogue, optional SM120 GDN prefill | 771.7 → 598.6 ms |
| [4](../outputs/engine_opt/04_gdn_branch/summary.md) | Branch kernels read shared state and K/V in place | Cached Tv640/N256 yes/no: 300.69 → 177.43 ms |
| [5](../outputs/engine_opt/05_w12/summary.md) | Opt-in lossless W12 weights | One cached-image box query: 7.32 → 5.55 ms |
| [6](../outputs/engine_opt/06_scheduler/summary.md) | Cross-request scheduler and async HTTP server | 64 cold box clients: 2,384.6 → 1,439.4 ms |
| [7](../outputs/engine_opt/07_frames/summary.md) | Frame throughput, packed video uploads | 64 frames × one box: 773.6 → 587.7 ms |
| [8](../outputs/engine_opt/08_bank_memory/summary.md) | Compact lossless banks, pinned-host tier | Tv640 bank: 75.49 → 72.87 MiB/image |
| [9](../outputs/engine_opt/09_numerics/summary.md) | Deterministic box readout, tighter gates | Cached Tv640/N256 box: 95.13 → 96.71 ms (the price of determinism) |

[Full index](../outputs/engine_opt/SUMMARY.md) · [Research log](../outputs/engine_opt/FINDINGS.md) · [Reproduce](benchmarks.md)
