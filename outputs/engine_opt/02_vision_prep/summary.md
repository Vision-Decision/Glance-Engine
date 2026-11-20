# Item 2 · Fresh-frame preprocessing and vision

At **Tv=128, N=128**, the verified packed fresh path took **774.2 ms**, down from **933.3 ms** (17.0%). Preparation fell from 82.6 to 8.0 ms and vision including transfer fell from 263.8 to 187.3 ms. Glance finished 41.0 ms ahead of SGLang and 139.3 ms behind vLLM. At Tv=256, N=256, latency fell from 3,697.8 to 3,173.8 ms. [Before](raw/item2_before.json) · [Final](raw/item2_fast.json)

## E1b end to end

One broadcast yes/no question over resized RGB COCO frames. Glance timing begins with the resized PIL frames and ends with all answers on the host. The two visual budgets produce 117 and 252 merged tokens per frame. JPEG decoding and bicubic resize precede timing for every system. Each Glance row uses four warm-up calls and ten measured calls with rotating questions. The VM runner waits for an empty GPU process list before launching each process, serializes chains with a file lock and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Stage medians are independent. The final Glance sweep peaked at 13.60 GiB of allocated GPU memory. [Runner](../../../scan/vm/vision_prep.sh) · [Before](raw/item2_before.json) · [Final](raw/item2_fast.json) · [vLLM 0.34](raw/item2_cold_vllm.json) · [SGLang 0.5.23](raw/item2_serving_sglang.json)

| Tv | N | Before prep / vision / LM | Before total | Final prep / vision / LM | Final total | vLLM | SGLang |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 22.1 / 61.0 / 137.5 | 220.6 | 2.1 / 45.1 / 134.5 | **181.7** | 184.3 | 254.8 |
| 128 | 64 | 44.0 / 128.3 / 289.8 | 462.5 | 4.1 / 91.7 / 285.0 | **380.9** | 329.4 | 448.2 |
| 128 | 128 | 82.6 / 263.8 / 585.5 | 933.3 | 8.0 / 187.3 / 579.0 | **774.2** | 635.0 | 815.2 |
| 128 | 256 | 172.3 / 528.8 / 1,175.8 | 1,889.4 | 16.0 / 375.1 / 1,161.1 | **1,552.3** | 1,255.4 | 1,585.5 |
| 256 | 32 | 31.0 / 145.6 / 283.6 | 460.2 | 4.2 / 112.4 / 277.4 | **393.9** | 324.4 | 458.7 |
| 256 | 64 | 60.9 / 293.2 / 569.9 | 924.0 | 8.2 / 224.0 / 560.4 | **792.8** | 625.7 | 840.5 |
| 256 | 128 | 120.9 / 586.9 / 1,139.1 | 1,847.2 | 16.4 / 448.4 / 1,121.5 | **1,586.4** | 1,241.3 | 1,561.5 |
| 256 | 256 | 241.2 / 1,174.3 / 2,277.6 | 3,697.8 | 32.8 / 897.1 / 2,243.6 | **3,173.8** | 2,482.4 | 3,025.9 |

Glance improved in all 18 measured configurations. Across the eight large-batch medians, Glance finished ahead of SGLang in six and ahead of vLLM in one. The latter was Tv=128, N=32, with 181.7 ms for Glance, 184.3 ms for vLLM with the encoder reset and 183.7 ms for vLLM with unique content. [Glance](raw/item2_fast.json) · [vLLM](raw/item2_cold_vllm.json) · [Unique-content control](raw/item2_unique_vllm.json) · [SGLang](raw/item2_serving_sglang.json)

### Small batches

The serving measurements use the same full batch-size sweep as Glance, with eight measured calls per cell after three warm-ups. Both serving frameworks receive one request per frame. Glance and vLLM receive resized PIL inputs. SGLang receives paths to PNG files created before timing, and the request includes reading those files. The timed calls all reported zero mean cached prompt tokens. The vLLM figures use an explicit encoder-cache reset. SGLang flushes its cache and changes one pixel per request before timing. [Full vLLM sweep](raw/item2_cold_vllm.json) · [Full SGLang sweep](raw/item2_serving_sglang.json)

| Tv | N | Glance before | Glance final | vLLM | SGLang |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 34.6 | 33.4 | 21.4 | 31.0 |
| 128 | 2 | 37.5 | 35.2 | 35.4 | 54.2 |
| 128 | 4 | 41.2 | 38.1 | 42.9 | 66.0 |
| 128 | 8 | 57.7 | 50.4 | 61.4 | 98.0 |
| 128 | 16 | 113.1 | 94.7 | 102.4 | 157.2 |
| 128 | 32 | 220.6 | 181.7 | 184.3 | 254.8 |
| 256 | 1 | 37.3 | 35.5 | 24.9 | 40.3 |
| 256 | 2 | 41.7 | 38.7 | 43.8 | 64.6 |
| 256 | 4 | 55.0 | 50.8 | 59.4 | 92.3 |
| 256 | 8 | 101.9 | 90.9 | 101.9 | 157.8 |
| 256 | 16 | 212.5 | 185.8 | 175.4 | 241.7 |
| 256 | 32 | 460.2 | 393.9 | 324.4 | 458.7 |

### Serving cache control

The original vLLM runner cleared the prefix and multimodal processor caches. vLLM 0.34 has a separate encoder-cache reset. With a 64-image pool and the original resets, Tv=256, N=32 measured 237.2 ms. Adding the encoder reset to that small-pool run gave 327.5 ms. With a 256-image pool, the same cell measured 325.3 ms with the original resets, 324.4 ms with the encoder reset and 326.2 ms with the encoder reset and content-unique inputs. At the 128-frame headline the fresh vLLM rerun measured 635.0 ms, the unique-content control 635.5 ms and archived item 1 638.3 ms. The main table uses the complete-cache-reset run over the full pool. [Original small pool](raw/item2_small_vllm.json) · [Small pool with encoder reset](raw/item2_small_cold_vllm.json) · [Original reset, full pool](raw/item2_serving_vllm.json) · [Unique content](raw/item2_unique_vllm.json) · [Full resets](raw/item2_cold_vllm.json) · [Item-1 vLLM](../01_varlen_prefill/raw/item1_vllm.json) · [Item-1 SGLang](../01_varlen_prefill/raw/item1_sglang.json)

## Where the time went

The instrumented pass executes the eager tower so CUDA events can mark individual sub-stages. Each entry is a median of three calls after warm-up. Host columns use wall time. Device columns sum same-stream events over the actual chunks of a call. Events inside very short kernels can include submission gaps. The end-to-end sweep uses the E1b harness’s 131,072-patch graph limit. The attention column in the final profile measures elapsed time on the caller stream after the worker streams join. Summed worker events overlap in time. [Baseline instrumentation](raw/item2_profile_before.json) · [Final instrumentation](raw/item2_profile_fast.json) · [Profiler](../../../scan/profile_vision.py)

### Baseline sub-stages

| Tv | N | HF processor | Tokenize | Collate | H2D | Patch GEMM | Block modules | RoPE | Attention | Merger | Tower total | Scatter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 13.86 | 1.54 | 9.18 | 4.02 | 0.17 | 29.94 | 22.05 | 2.70 | 0.61 | 57.35 | 0.05 |
| 128 | 64 | 24.95 | 3.38 | 19.45 | 7.66 | 0.25 | 57.86 | 50.04 | 5.49 | 1.24 | 120.29 | 0.07 |
| 128 | 128 | 45.70 | 5.72 | 36.00 | 14.20 | 0.49 | 117.77 | 106.52 | 10.49 | 2.35 | 250.42 | 0.15 |
| 128 | 256 | 88.60 | 10.82 | 71.96 | 28.44 | 0.98 | 235.70 | 213.00 | 21.00 | 4.70 | 500.98 | 0.30 |
| 256 | 32 | 19.15 | 1.69 | 9.90 | 8.08 | 0.29 | 65.54 | 55.18 | 9.70 | 1.28 | 137.76 | 0.07 |
| 256 | 64 | 38.38 | 3.14 | 19.50 | 15.30 | 0.54 | 127.65 | 114.80 | 18.94 | 2.50 | 278.19 | 0.17 |
| 256 | 128 | 77.09 | 6.10 | 38.43 | 30.47 | 1.08 | 255.55 | 229.58 | 37.91 | 5.00 | 556.65 | 0.32 |
| 256 | 256 | 149.53 | 11.40 | 77.42 | 61.03 | 2.15 | 511.31 | 459.18 | 75.86 | 10.02 | 1,113.53 | 0.66 |

### Final sub-stages

| Tv | N | Gather RGB | Layout | H2D | GPU patchify | Patch embed | Block modules | RoPE | Attention | Merger | Tower total | Scatter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 2.12 | 0.03 | 0.23 | 0.20 | 4.37 | 29.98 | 1.31 | 8.57 | 0.61 | 46.61 | 0.03 |
| 128 | 64 | 4.02 | 0.03 | 0.44 | 0.50 | 8.77 | 58.33 | 3.85 | 14.87 | 1.24 | 92.09 | 0.05 |
| 128 | 128 | 7.86 | 0.04 | 0.86 | 1.31 | 17.52 | 118.10 | 8.19 | 28.02 | 2.36 | 186.48 | 0.13 |
| 128 | 256 | 15.76 | 0.09 | 1.72 | 2.62 | 35.04 | 236.37 | 16.39 | 55.98 | 4.71 | 373.22 | 0.27 |
| 256 | 32 | 4.11 | 0.03 | 0.47 | 0.57 | 9.48 | 65.63 | 3.91 | 27.69 | 1.28 | 113.70 | 0.06 |
| 256 | 64 | 8.14 | 0.03 | 0.92 | 1.43 | 18.83 | 127.94 | 8.85 | 54.80 | 2.50 | 226.21 | 0.15 |
| 256 | 128 | 16.34 | 0.06 | 1.84 | 2.86 | 37.69 | 256.02 | 17.70 | 109.63 | 5.01 | 452.62 | 0.29 |
| 256 | 256 | 32.42 | 0.12 | 3.70 | 5.72 | 75.39 | 512.30 | 35.41 | 219.01 | 10.04 | 905.35 | 0.58 |

Block modules include both layer norms, QKV, the attention output projection and the MLP. The tower total also includes residual additions, position addition and reshaping or copying between operations. Feature scatter belongs to the LM timer. The baseline HF processor includes normalization and patchification on the host. The final profile uses a warm metadata cache for its fixed question.

## Implementation and numerical checks

The GPU receives three RGB bytes per original pixel. The old transfer carried two temporal copies of three FP32 channels, or 24 bytes per pixel. A pinned host batch gathers the current frames in input order. A lookup table built by the configured HF processor supplies the normalized FP32 value of each of the 256 possible channel inputs. The GPU indexes that table and applies the processor's own patch permutation and temporal expansion. A bounded, 16-entry host cache stores the singleton collator output for a grid and question. Expanding those tensors preserves the reference positions, centers and valid-label mask. Every call reads its input pixels afresh. Native-aspect mixed grids, clip inputs and unsupported processors use the existing input path. [RGB preprocessing](../../../scan/vision_prep.py) · [Scan entry point](../../../scan/engine.py)

At 128 frames, Tv=128, the baseline spent 106.5 ms in rotary embedding and 10.5 ms in attention. The Triton rotary kernel loads both halves of Q and K, rotates them in FP32 with separate multiplies and sums, and writes BF16 once. The SDPA input comparisons were bitwise equal at both budgets. The original flattened patch GEMM differed from HF Conv3d by up to 0.03125 before the first transformer block. Adding bias separately reduced mean absolute error from 0.000344 to 0.00000050 at Tv=128, with the same maximum. The production path uses the reference convolution. The singleton speed-engine change also restores all 930 BF16 decision comparisons on the diagnostic. [Rotary implementation](../../../speed/vision_rotary.py) · [Vision tower](../../../speed/engine.py) · [Patch trials](raw/item2_patch.json) · [Speed diagnostic](raw/item2_diag_fused.json)

At the headline point, compact upload with the legacy tower measured 837.6 ms. Fusing rotary and restoring Conv3d while keeping batched SDPA reached 762.0 ms. That combination differed from matching HF features by as much as 4.03125 on the 32-frame Tv=256 test. Ordinary batch flash and HF’s per-image split-KV flash use different reduction orders. Preserving the singleton reduction on a serial stream measured 797.2 ms. Graph-replaying independent per-image reductions on worker streams brought this to 774.2 ms and preserved the HF tensor exactly. All dense projections still operate on the full chunk. Short grids use up to four streams through 32 images and up to eight at larger N. Longer grids use up to two. The packed LM chunk planner remains in control of peak workspace. [RGB only](raw/item2_prep_only.json) · [Batched SDPA trial](raw/item2_after.json) · [Feature deltas](raw/item2_features.json) · [Serial exact tower](raw/item2_final.json) · [Stream-count sweep](raw/item2_streams.json)

A mixed-length first-block test alternated 468/234 or 1,008/504 patches. Packed varlen attention took 0.052/0.101 ms versus 0.228/0.259 ms with individual SDPA calls. Both packed outputs differed from the reference by up to 0.0078125. The dense and varlen split-count controls also failed the bitwise check. The corrected paged-KV control produced larger errors with multiple splits. Mixed-grid scans retain the existing implementation. [Dense and varlen dispatch](raw/item2_attention.json) · [Paged and mixed-length controls](raw/item2_paged2.json)

The legacy tower dispatched CUTLASS `cutlass_80_tensorop_bf16` GEMMs on sm_120. Choosing cuBLASLt explicitly selected the same kernels. At eight Tv=128 frames the measured GEMM time was 7.35 ms with cuBLAS and 7.34 ms with cuBLASLt. Four Triton tile configurations compiled for `.target sm_120a` were tested at 12 actual ViT projection shapes. At 59,904 × 1,024 × 4,096, the deployed linear took 1.375 ms and the fastest native candidate took 1.640 ms. PyTorch was faster on 11 of 12 shapes, and the lone Triton lead was 0.0013 ms. The production tower retains the measured faster overall dispatch. [BLAS traces and batch-size sweep](raw/item2_tune.json) · [Native kernels and error measurements](raw/item2_dispatch.json)

The final strict feature test compared the complete tower output with HF on matching batches of 1, 8, 32, 64 and 128 COCO frames at each budget. All ten output tensors and all ten FP32 pixel tensors were bitwise identical. Packed IDs, positions, readout indices and head metadata also matched exactly. Both query-first and image-first layouts matched for box, short yes/no, choice and score questions. HF itself produced different features when the same frames went through its tower individually. [Direct feature checks](raw/item2_features_fast.json)

The tiny FP32 CPU suite passed. The real BF16 suite passed its existing continuous-output thresholds and all 234/234 decisions. The graph-enabled COCO/clip suite agreed on 480/480 decisions across 20 path/question combinations. At the largest production chunk for each budget, graph capture and replay with reversed input order also gave bitwise-identical features against fresh HF calls. Each grid retained one captured graph. [Graph feature checks](raw/item2_graph_features.json)

The 60-image speed diagnostic agreed on 930/930 decisions for all five exact BF16 engine variants versus shipped BF16. Its separately reported FP32 and optional FP8 comparisons use different numerical formats. No tolerance changed. [CPU](raw/item2_cpu.log) · [Real suite](raw/item2_real_fast.log) · [COCO and clips](raw/item2_equiv_fast.json) · [Speed diagnostic](raw/item2_diag_fused.json)

The largest headline gap now belongs to LM/readout. Its measured 579.0 ms plus the compact-input vision/transfer floor of 101.4 ms already sum to about 680.4 ms before host preparation. Reaching 629 ms on this serial workload also requires a faster LM/readout or stage overlap. The roadmap's packed-LM item owns that work.

The singleton patch and rotary change also affects the speed workload. Glance measured 41.3 ms at Q=1 with ≤640 visual tokens, 125.4/294.2 ms on the 2k/4k GUI calls and 2,630.6 ms for a cold burst of 64 serial clients. The updated ceiling map contains all touched rows. `GlanceEngine(..., optimized_singleton=False)` preserves the former singleton tower. [Single image](raw/item2_single_fused.json) · [2k GUI](raw/item2_fused_gui2048.json) · [4k GUI](raw/item2_fused_gui4096.json) · [Clients](raw/item2_clients_fused.json)

## Roofline

C0 measured 401.73 BF16 TFLOP/s, 1,463.21 GB/s of device copy bandwidth and 56.87 GB/s of pinned H2D bandwidth on this G4. Each compute stage uses the larger of its FLOP floor and compulsory-byte floor. The compact input changes only the modeled H2D payload, from 6,144 to 768 bytes per raw patch. The model retains the original ViT, LM and head arithmetic, plus one minimum dispatch. The host-preparation floor is zero in C0, so a percentage for host preparation is undefined. Percentages below are floor / achieved. [Measured peaks](../ROOFLINE.md) · [C0 call shapes and stage floors](../00_ceiling/raw/c0_floors.json) · [Compact-input floors](raw/item2_floors.json)

| Tv | N | H2D floor | ViT dense floor | ViT attention floor | Vision + H2D achieved / floor | Vision ceiling | LM/readout achieved / floor | LM ceiling | Total achieved / floor | Total ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 0.20 | 23.1 | 2.0 | 45.1 / 25.4 | 56.2% | 134.5 / 85.9 | 63.8% | 181.7 / 111.2 | 61.2% |
| 128 | 64 | 0.40 | 46.3 | 4.0 | 91.7 / 50.7 | 55.3% | 285.0 / 171.7 | 60.3% | 380.9 / 222.5 | 58.4% |
| 128 | 128 | 0.81 | 92.6 | 8.0 | 187.3 / 101.4 | 54.2% | 579.0 / 343.5 | 59.3% | 774.2 / 444.9 | 57.5% |
| 128 | 256 | 1.62 | 185.1 | 16.1 | 375.1 / 202.8 | 54.1% | 1,161.1 / 687.0 | 59.2% | 1,552.3 / 889.8 | 57.3% |
| 256 | 32 | 0.44 | 49.8 | 8.0 | 112.4 / 58.2 | 51.8% | 277.4 / 165.0 | 59.5% | 393.9 / 223.3 | 56.7% |
| 256 | 64 | 0.87 | 99.7 | 15.9 | 224.0 / 116.5 | 52.0% | 560.4 / 330.0 | 58.9% | 792.8 / 446.5 | 56.3% |
| 256 | 128 | 1.74 | 199.4 | 31.8 | 448.4 / 232.9 | 51.9% | 1,121.5 / 660.1 | 58.9% | 1,586.4 / 893.0 | 56.3% |
| 256 | 256 | 3.49 | 398.7 | 63.6 | 897.1 / 465.9 | 51.9% | 2,243.6 / 1,320.2 | 58.8% | 3,173.8 / 1,786.1 | 56.3% |

The eager profile also separates transfer, dense modules and attention. Each cell below gives achieved ms / modeled floor ms (percent of ceiling). The dense interval sums patch embedding, norms, projections, MLP and merger. Its floor counts dense multiply-adds and minimum traffic. The attention interval includes the worker-stream join. RoPE, residual addition and packing sit outside these two compute intervals.

| Tv | N | Pixel H2D | Dense modules | Attention |
|---:|---:|---:|---:|---:|
| 128 | 32 | 0.23 / 0.20 (90%) | 34.95 / 23.14 (66%) | 8.57 / 2.01 (23%) |
| 128 | 64 | 0.44 / 0.40 (92%) | 68.34 / 46.28 (68%) | 14.87 / 4.02 (27%) |
| 128 | 128 | 0.86 / 0.81 (94%) | 137.97 / 92.56 (67%) | 28.02 / 8.05 (29%) |
| 128 | 256 | 1.72 / 1.62 (94%) | 276.12 / 185.13 (67%) | 55.98 / 16.10 (29%) |
| 256 | 32 | 0.47 / 0.44 (93%) | 76.39 / 49.84 (65%) | 27.69 / 7.96 (29%) |
| 256 | 64 | 0.92 / 0.87 (94%) | 149.28 / 99.68 (67%) | 54.80 / 15.91 (29%) |
| 256 | 128 | 1.84 / 1.74 (95%) | 298.72 / 199.37 (67%) | 109.63 / 31.82 (29%) |
| 256 | 256 | 3.70 / 3.49 (94%) | 597.73 / 398.74 (67%) | 219.01 / 63.65 (29%) |

## Reproduction and authorship

From the synced checkout on engopt, use `bash scan/vm/vision_prep.sh before prep_only features_fast fast profile_before profile_fast real_fast equiv_fast diag_fused graph_features`. The fresh benchmark keeps `--fresh_impl varlen` throughout. `--vision_prep reference` selects the previous HF-preprocessed packed engine, `rgb` isolates preprocessing, and `fused` selects the production path. Use `cold_vllm serving_sglang` for the serving columns. The `small_vllm small_cold_vllm serving_vllm unique_vllm` stages reproduce the encoder-cache controls. The result archive is under `engine_opt/02_vision_prep` in `Vision-Decision/collection-scan-results`. The initial development driver logged a shell error after publishing its last stage, and one precision driver log is empty. The subsequent precision retry and the final production-stage logs record the completed runs. [Development driver](raw/item2_chain_dev.log) · [Precision retry](raw/item2_precision_retry.log) · [Final benchmark](raw/item2_fast.log)

### Authorship

Codex wrote the preprocessing, vision changes, profilers, stage runner and reports, and ran the benchmarks and equivalence checks. The reviewer supplied the existing engopt VM, environments, checkpoint and COCO data.
