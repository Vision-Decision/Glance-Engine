# BF16 language-model prefill on the G4

At 128 new frames and Tv=128, Glance took **598.6 ms** end to end, from **771.7 ms** on the same VM. LM and readout fell from 576.8 to 403.1 ms. The clean BF16 serving medians were 635.0 ms for vLLM and 815.2 ms for SGLang. The modeled LM/readout floor is 343.5 ms, and the optimized LM reaches 85.2% of that empirical ceiling. [Before](raw/item3_before.json) · [Production](raw/item3_accel_production.json) · [Stage model](../ROOFLINE.md)

The production sweep’s p90 was 608.4 ms. An independent run measured 600.2 ms at the same headline, and the standard-stack configuration measured 619.6 ms. Both configurations met the original 629 ms target. Against the clean BF16 serving table, the accelerated implementation wins all 18 fresh comparisons with SGLang and 16 of 18 with vLLM. At Tv=256, vLLM leads by 5.7 ms at 128 frames and 15.1 ms at 256. [Independent repeat](raw/item3_accel_headline.json) · [Standard stack](raw/item3_production.json) · [Serving baselines](../CEILING.md)

## Fresh frames

The timer starts with resized PIL images and ends with the requested answers on the host. Each Glance cell rotates ten questions after four warm-ups. The before and production sweeps use the same 256-image pool, packed token planner, checkpoint, BF16 weights, compact vision input and expandable-segments allocator. Each GPU process started after the shared lock and an empty GPU process check. The comparison servers use the item-2 measurements with encoder-cache reset or content-unique inputs. [Runner](../../../scan/vm/lm_prefill.sh) · [vLLM](../02_vision_prep/raw/item2_cold_vllm.json) · [SGLang](../02_vision_prep/raw/item2_serving_sglang.json)

| Tv / frames | Before total | Before LM | Prep | Vision + H2D / floor | LM + readout / floor | LM ceiling | Final total / floor | vLLM | SGLang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 / 32 | 181.2 | 134.1 | 2.1 | 45.2 / 25.4 | 98.9 / 85.9 | 86.8% | 146.3 / 111.2 | 184.3 | 254.8 |
| 128 / 64 | 379.6 | 284.1 | 4.1 | 92.1 / 50.7 | 200.4 / 171.7 | 85.7% | 296.6 / 222.5 | 329.4 | 448.2 |
| 128 / 128 | 771.7 | 576.8 | 8.0 | 187.6 / 101.4 | 403.1 / 343.5 | 85.2% | 598.6 / 444.9 | 635.0 | 815.2 |
| 128 / 256 | 1,548.8 | 1,158.1 | 15.9 | 376.2 / 202.8 | 810.3 / 687.0 | 84.8% | 1,202.6 / 889.8 | 1,255.4 | 1,585.5 |
| 256 / 32 | 393.5 | 277.0 | 4.2 | 112.8 / 58.2 | 194.3 / 165.0 | 84.9% | 311.2 / 223.3 | 324.4 | 458.7 |
| 256 / 64 | 792.2 | 559.9 | 8.3 | 224.3 / 116.5 | 389.8 / 330.0 | 84.7% | 622.3 / 446.5 | 625.7 | 840.5 |
| 256 / 128 | 1,585.9 | 1,121.3 | 16.4 | 449.2 / 232.9 | 781.5 / 660.1 | 84.5% | 1,247.0 / 893.0 | 1,241.3 | 1,561.5 |
| 256 / 256 | 3,172.9 | 2,243.3 | 32.7 | 898.4 / 465.9 | 1,566.3 / 1,320.2 | 84.3% | 2,497.5 / 1,786.1 | 2,482.4 | 3,025.9 |

Medians are in milliseconds. Stage medians can sum to a different value from the full-call median. The host-preparation floor in C0 is zero. At Tv=256, vision and transfer are the largest remaining stage gap for the large batches. [Full ceiling map](../CEILING.md)

The standard-stack result uses the native SwiGLU kernel with FLA chunk-64. The accelerated stack adds FlashQLA 0.1.3 and TileLang 0.1.15 for the packed GDN recurrence, preserving BF16 throughout. [Standard stack](raw/item3_production.json) · [Accelerated stack](raw/item3_accel_production.json)

| Tv | Frames | Standard LM | Accelerated LM | Standard total | Accelerated total | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 103.6 | 98.9 | 150.8 | 146.3 | 4.5 |
| 128 | 64 | 210.2 | 200.4 | 306.0 | 296.6 | 9.4 |
| 128 | 128 | 424.2 | 403.1 | 619.6 | 598.6 | 21.0 |
| 128 | 256 | 852.9 | 810.3 | 1,244.4 | 1,202.6 | 41.9 |
| 256 | 32 | 204.6 | 194.3 | 321.4 | 311.2 | 10.2 |
| 256 | 64 | 411.5 | 389.8 | 643.9 | 622.3 | 21.6 |
| 256 | 128 | 824.7 | 781.5 | 1,289.2 | 1,247.0 | 42.2 |
| 256 | 256 | 1,651.7 | 1,566.3 | 2,581.5 | 2,497.5 | 84.0 |

| Tv / frames | Before | Final | vLLM | SGLang |
|---|---:|---:|---:|---:|
| 128 / 1 | 33.6 | 12.7 | 21.4 | 31.0 |
| 128 / 2 | 34.9 | 16.3 | 35.4 | 54.2 |
| 128 / 4 | 38.4 | 25.5 | 42.9 | 66.0 |
| 128 / 8 | 51.0 | 45.4 | 61.4 | 98.0 |
| 128 / 16 | 94.5 | 84.9 | 102.4 | 157.2 |
| 256 / 1 | 36.1 | 17.4 | 24.9 | 40.3 |
| 256 / 2 | 39.2 | 26.8 | 43.8 | 64.6 |
| 256 / 4 | 50.6 | 45.7 | 59.4 | 92.3 |
| 256 / 8 | 91.0 | 81.5 | 101.9 | 157.8 |
| 256 / 16 | 185.8 | 153.7 | 175.4 | 241.7 |

## Kernels and the measured ceiling

The pre-optimization C0 model divides the headline LM work into 333.2 ms of dense projections, 7.56 ms of GDN, 2.10 ms of attention and 0.62 ms of FP32 heads. It uses the larger of useful FLOPs at 401.73 BF16 or 67.4 FP32 TFLOP/s and compulsory bytes at 1,463.21 GB/s. Compact vision and H2D add 101.4 ms. These rates were measured on engopt. Short individual GEMMs can exceed the square-GEMM calibration. [Peak microbenchmarks](../00_ceiling/raw/c0_peaks.json) · [Compact workload floors](../02_vision_prep/raw/item2_floors.json)

| Nominal visual tokens | Actual LM tokens | Original eager | Fused norm / conv | Direct SwiGLU | TMA / FLA | TMA / FlashQLA |
|---|---:|---:|---:|---:|---:|---:|
| 4096 | 4672 | 132.0 | 109.4 | 105.5 | 105.2 | 99.7 |
| 8192 | 9344 | 280.6 | 221.4 | 212.7 | 212.4 | 200.5 |
| 16384 | 18688 | 570.0 | 447.9 | 426.1 | 426.8 | 402.3 |
| 32768 | 37376 | 1,150.0 | 903.2 | 856.5 | 853.4 | 805.4 |

Each isolated call reuses the same real vision features, executes the unchunked LM and both heads, and reports a five-sample CUDA-event median after warm-up. The profiler pass is separate. The per-kernel file records the shape, call count, time, effective rate, percentage of its measured peak and the definition of its useful work. Tensor payload rates omit cache transactions and intermediate workspace traffic. [Kernel rates](raw/item3_kernel_rates.tsv) · [Before trace](raw/item3_profile_before.json) · [TMA / FLA trace](raw/item3_profile_production.json) · [TMA / FlashQLA trace](raw/item3_accel_profile_production.json)

The following timings use three additional instrumented calls. Nested CUDA events subtract the projection time from the mixer times. Dense includes the MLP, norms, gates and input assembly. The floor uses the exact 117 visual, 122 prefix and 24 question tokens per image in this probe. Each cell gives original ms / production ms / floor ms and the fraction of the empirical ceiling attained. [Original events](raw/item3_stages_before.json) · [TMA / FLA events](raw/item3_stages_final.json) · [TMA / FlashQLA events](raw/item3_accel_stages_final.json)

| LM tokens | Dense and residual | GDN | Attention | FP32 heads |
|---|---:|---:|---:|---:|
| 4672 | 95.28 / 87.81 / 83.01 (95%) | 33.02 / 8.02 / 1.88 (23%) | 3.48 / 1.77 / 0.52 (30%) | 0.63 / 0.62 / 0.16 (25%) |
| 9344 | 189.87 / 177.86 / 166.03 (93%) | 82.68 / 18.52 / 3.77 (20%) | 7.48 / 3.55 / 1.05 (29%) | 0.98 / 0.99 / 0.31 (32%) |
| 18688 | 383.60 / 355.71 / 332.06 (93%) | 168.52 / 37.57 / 7.53 (20%) | 16.85 / 7.01 / 2.09 (30%) | 1.85 / 1.87 / 0.62 (33%) |
| 37376 | 772.77 / 712.10 / 664.11 (93%) | 339.03 / 76.20 / 15.07 (20%) | 35.74 / 14.22 / 4.19 (29%) | 3.33 / 3.31 / 1.25 (38%) |

Nsight Compute sampled the first warmed launch for each kernel and launch configuration per input size, with normal cache and clock settings. The useful-work table covers all 1,512 kernel/shape groups across the original, TMA/FLA and TMA/FlashQLA traces. The DRAM table reports counted read-plus-write bytes divided by hardware duration. The production TMA SwiGLU launch at 18,688 rows took 4.197 ms and moved 188.8 GB/s, equivalent to 420.2 useful BF16 TFLOP/s across its two projections. At 18,688 tokens the direct-epilogue pass measured 1,420 GB/s for convolution, 1,473 GB/s for gated RMSNorm, 1,427 GB/s for the GDN key/value transform and 1,440 GB/s for its state update. The latter kernels materialize intermediate state across chunk passes. [Hardware rate by kernel and launch geometry](raw/item3_dram_rates.tsv) · [Direct counter sample](raw/item3_ncu_final_128.csv) · [Production counter sample](raw/item3_ncu_production_128.csv)

GDN output normalization in the original trace expanded both activations and gates to FP32, with separate powers, reductions, products, conversions and SiLU. At 18,688 tokens two copy kernels at the gated head width alone consumed 30.6 ms. The packed kernel reads the strided gate directly and preserves the BF16 rounding after normalization and learned scaling. A depthwise-convolution kernel writes contiguous Q, K and V, applies SiLU after the convolution’s BF16 cast, and normalizes Q/K per head. The packing code precomputes the sequence start for every token. The convolution cannot read across a frame boundary. [Packed prefill](../../../scan/varlen.py) · [CUDA kernels](../../../speed/lm_prefill.py)

Full attention normalizes each head and rotates the configured leading dimensions in one kernel. It runs causal attention over each sequence and overwrites only the bidirectional question rows with full-context attention. The gate consumes the strided QKV projection. Layer transitions fuse residual addition with the next RMSNorm, including the final norm. GDN beta and decay continue to use the fused gates. Query heads, key heads and value heads keep their checkpoint widths. The LM returns the original FP32 box and typed heads. [Shared engine](../../../speed/engine.py)

### Projection dispatch

PyTorch 2.14.1+cu130 and cuBLAS 13.1.1.3 selected CUTLASS 80 BF16 tensor-op kernels on compute capability 12.0. The probe used the real contiguous row-major activation and weight strides. The sampled pointers were 256-byte aligned. PyTorch had TF32 matmul disabled and BF16 reduced-precision reductions enabled. Its TF32 flag applies to FP32 operands. The native kernels used BF16 multiplicands and FP32 accumulators. The separate cuBLASLt sweep supplied column-major descriptors for the equivalent transposed product, BF16 operands/output, FP32 accumulation, and a 32 MiB workspace. It requested 16 heuristic results per shape and timed each returned algorithm after warm-up. At 18,688 rows the first heuristic won three of five distinct shapes, and all returned winners used algorithm ID 21. This establishes the selected candidates on this stack. The vendor ranking model is internal. [Backend traces](raw/item3_dispatch_final.json) · [Heuristic sweep](raw/item3_heuristic.log) · [cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/)

| Projection | K × N | cuBLAS ms | TFLOP/s | cuBLASLt ms | Best native plain ms |
|---|---:|---:|---:|---:|---:|
| gdn_in | 2,560 × 12,352 | 2.966 | 398 | 3.038 | 3.088 |
| gdn_out | 4,096 × 2,560 | 1.004 | 390 | 1.006 | 1.022 |
| qkv | 2,560 × 10,240 | 2.488 | 394 | 2.489 | 2.515 |
| attn_out | 4,096 × 2,560 | 1.001 | 391 | 0.999 | 1.022 |
| gate_up | 2,560 × 18,432 | 4.369 | 404 | 4.381 | 4.594 |
| down | 9,216 × 2,560 | 2.215 | 398 | 2.211 | 2.242 |

The first design removed most of the elementwise traffic and measured 649.0 ms end to end. The second computes SwiGLU before writing the gate/up projection to device memory. Its Triton dot products accumulate in FP32, round both halves to BF16, and apply the activation and multiply before storing the MLP intermediate. It uses the existing learned weight layout. At 18,688 rows the separate cuBLAS projection and pointwise gate took 5.197 ms in the dispatch sweep. The direct native epilogue took 4.555 ms, and the independent TMA sweep measured 4.255 ms at its fastest tile. The deployed tile’s PTX targets `sm_120a` and contains `cp.async.bulk.tensor.2d` copies and BF16-input, FP32-accumulating `mma.sync.aligned.m16n8k16` instructions. The deployed BF16 path uses TMA with a 128 × 64 × 32 tile for aligned SM120 gate/up projections of at least 4,096 rows. Other projection shapes retain the measured library dispatch. End to end, the direct variant measured 625.4 ms and the TMA variant measured 619.6 ms. At 18,688 fixed-feature tokens the two isolated LM medians differed by 0.7 ms. [First fusion sweep](raw/item3_after.json) · [Direct epilogue](raw/item3_dispatch_final.json) · [TMA tiles](raw/item3_tma.json) · [Compiled PTX target and instructions](raw/item3_native_ptx.json)

### Gated delta rule and graphs

The recurrence microbench uses checkpoint projections of normalized synthetic BF16 inputs, with 16 key heads, 32 value heads and 128-wide states. In the FLA chunk sweep, the FLA recurrence was fastest with packed segments and a 64-token chunk in all four tested batch/length combinations. At 128 × 146 tokens, 64/32/16-token chunks took 1.946/2.329/3.223 ms per GDN core. A dense batch at chunk 64 took 2.039 ms. The 24-layer core alone therefore needs about 46.7 ms at this shape, compared with the 7.56 ms ideal stage floor that includes convolution. The before and after profiles expose the intra-chunk solve, transformed vectors, state update and output contraction separately. [Chunk and layout sweep](raw/item3_gdn.json)

The persistent FLA recurrence took 2.389 ms at 128 × 146, compared with 1.946 ms for packed chunk-64. FlashQLA 0.1.3’s SM120 public entry encountered an unavailable context-parallel symbol. Calling its single-card stages directly exposed a hardware limit. Its default 128-wide value tile requested 112,640 bytes of shared memory per block, above the device’s 101,376-byte opt-in limit. [Persistent and single-card trials](raw/item3_gdn_single_card.json) · [Public entry diagnostic](raw/item3_gdn_optional.log)

Splitting the 128 value lanes into two independent 64-wide tiles preserved the state equations and fit the device limit. The single-card implementation uses chunkwise log-decay sums, a 32-token KKT solve and a fused state-update/output kernel. On 128 × 146 it took 0.906 ms per core, down from 1.946 ms with FLA chunk-64. Summing one counter sample per core kernel gives 2.548 GB of DRAM traffic across five FLA launches versus 0.634 GB across three FlashQLA launches. Across 24 layers the instrumented full GDN stage, including convolution and normalization, fell from 61.37 to 37.57 ms. At 128 × 281 it took 1.442 ms. Glance calls the versioned TileLang kernel directly with this tile, skips context-parallel setup and uses the checkpoint weights unchanged. [64-wide sweep](raw/item3_gdn_tile64.json) · [Accelerated DRAM counters](raw/item3_accel_ncu_production_128.csv) · [Full-model prototype](raw/item3_qla_model.json) · [Production adapter](../../../speed/flash_gdn.py)

Short, uniform fresh batches use a bounded CUDA graph. Every sequence has a fixed prefix and a right-hand question reserve. Causal GDN never reads future tokens. A device vector limits keys for bidirectional question attention, and readout selects only real rows. The key contains the mask mode, frame count and prefix length. Longer questions fall back to the eager packed path. The cache retains at most eight graphs, owns its sequence-index tensors, and is cleared with the engine. The larger-bucket control at one image took 19.25 ms eager and 8.97 ms replayed. At eight images it took 31.91 ms eager and 34.44 ms replayed. [Graph control](raw/item3_graphs.json) · [Final reuse and reverse-order checks](raw/item3_graphs32.json)

The final 32-slot question reserve is rounded up with the prefix to a multiple of 16. For 1/4/8 images, median eager LM times were 18.68/19.14/31.89 ms and graph times were 8.85/18.01/31.11 ms. With the test cap raised to 2,048 padded tokens, 6 captures served 32 checked calls across both masks, query variations, reverse frame order and returns to a prior bucket. Maximum probability drift against original eager was 0.000306. Total CUDA memory reserved by the process, including weights and the vision tower, peaked at 10.52 GiB in this sweep. The default cap is 1,024 padded tokens, keeping the small-latency benefit and avoiding a larger workspace for the sub-millisecond eight-frame gain. [Final graph sweep](raw/item3_graphs32.json)

With the cache capped at one graph, a 124-token question and a second batch shape both fell back to eager with bitwise-equal typed logits. Returning to the first shape reused its graph. All four calls kept one graph resident. [Overflow and cache limit checks](raw/item3_graphs_limits.json)

## Shared prefill workloads

Full image, GUI and follow-up rows use the existing speed harness. Each video frame has its own question and native aspect ratio. The answer-only multi-image path can discard prefix caches and use the packed LM. Callers retaining a prefix keep the cache-returning interface. The speed, GUI, concurrency and serving before values come from the matching engopt C0 or item-2 measurements. The client workload reports time to the last answer from a concurrent burst. [Single image](raw/item3_single.json) · [GUI 2k](raw/item3_gui2048.json) · [GUI 4k](raw/item3_gui4096.json) · [Video, FLA](raw/item3_video_packed.json) · [Video, FlashQLA](raw/item3_accel_video_packed.json) · [Follow-up](raw/item3_followup.json) · [Clients](raw/item3_clients.json)

| Workload | Before | Final | Total floor | Ceiling | LM / floor | vLLM | SGLang |
|---|---:|---:|---:|---:|---:|---:|---:|
| bs1_box_N1 | 41.3 | 39.5 | 17.5 | 44.3% | 27.4 / 11.8 | 166.5 | 187.3 |
| bs1_box_N8 | 43.6 | 41.8 | 19.1 | 45.8% | 29.4 / 13.4 | 221.0 | 395.8 |
| bs1_box_N32 | 59.7 | 57.5 | 24.8 | 43.1% | 44.1 / 19.1 | 334.0 | 658.0 |
| bs1_mixed_N8 | 52.3 | 50.5 | 22.9 | 45.4% | 37.7 / 17.1 | 199.2 | 375.3 |
| bs1_mixed_N32 | 89.3 | 86.7 | 39.6 | 45.6% | 72.6 / 33.9 | 278.2 | 655.4 |
| gui2048_box_N1 | 125.4 | 119.2 | 67.9 | 57.0% | 69.0 / 38.1 | 226.2 | 301.7 |
| gui2048_box_N8 | 128.9 | 122.6 | 69.5 | 56.7% | 71.5 / 39.7 | 311.7 | 566.9 |
| gui2048_box_N32 | 152.1 | 146.1 | 75.5 | 51.7% | 94.5 / 45.7 | 572.2 | 1,406.2 |
| gui4096_box_N1 | 294.2 | 277.9 | 165.8 | 59.7% | 143.0 / 76.1 | 353.7 | 489.0 |
| gui4096_box_N8 | 301.7 | 282.9 | 167.9 | 59.3% | 147.3 / 78.2 | 418.4 | 1,305.1 |
| gui4096_box_N32 | 325.4 | 311.9 | 174.9 | 56.1% | 172.1 / 83.4 | 566.4 | 2,398.4 |
| video_f32_N1 | 580.2 | 383.7 | 217.8 | 56.8% | 183.4 / 157.0 | 548.6 | 672.3 |
| video_f64_N1 | 1,157.0 | 771.4 | 435.8 | 56.5% | 371.5 / 314.3 | 933.1 | 1,046.3 |
| Tv640_N64_box | 34.5 | 34.5 | 16.7 | 48.4% | 34.4 / 16.7 | 899.1 | 1,556.6 |
| Tv640_N64_noul | 66.0 | 65.8 | 31.9 | 48.5% | 65.7 / 31.9 | 680.6 | 1,287.3 |
| Tv640_N256_box | 139.9 | 139.8 | 66.7 | 47.7% | 139.6 / 66.7 | 3,151.6 | 7,732.9 |
| Tv640_N256_noul | 300.5 | 300.7 | 127.5 | 42.4% | 300.6 / 127.5 | 2,679.9 | 5,670.7 |
| Tv256_N64_box | 33.0 | 33.0 | 16.1 | 49.0% | 32.9 / 16.1 | 653.1 | 592.2 |
| Tv256_N256_box | 124.1 | 124.1 | 64.4 | 51.9% | 124.0 / 64.4 | 2,950.1 | 2,981.4 |
| Tv64_N64_box | 32.2 | 32.2 | 15.8 | 49.2% | 32.1 / 15.8 | 401.3 | 446.1 |
| Tv64_N256_box | 120.4 | 120.4 | 63.3 | 52.6% | 120.3 / 63.3 | 1,112.6 | 1,440.6 |
| f4_v96_N64_noul | 60.7 | 60.8 | 31.2 | 51.4% | 60.7 / 31.2 | 342.7 | 620.7 |
| f4_v96_N256_noul | 272.9 | 273.2 | 124.9 | 45.7% | 273.0 / 124.9 | 1,355.1 | 2,203.7 |
| followup_box_N1 | 10.8 | 10.5 | 5.0 | 47.4% | 10.2 / 5.0 | 145.2 | 162.0 |
| followup_mixed_N1 | 10.8 | 10.5 | 5.0 | 47.5% | 10.2 / 5.0 | 145.1 | 161.7 |
| followup_box_N8 | 13.6 | 13.2 | 5.0 | 38.1% | 12.6 / 5.0 | 196.6 | 346.1 |
| followup_mixed_N8 | 19.5 | 19.0 | 5.6 | 29.7% | 18.2 / 5.6 | 175.3 | 342.2 |
| followup_box_N32 | 27.8 | 27.3 | 7.6 | 28.0% | 25.8 / 7.6 | 316.9 | 606.5 |
| followup_mixed_N32 | 56.9 | 56.1 | 22.3 | 39.8% | 53.8 / 22.3 | 254.9 | 561.8 |
| clients_1 | 41.6 | 39.5 | 17.4 | 43.9% | 27.4 / 11.7 | 167.3 | 182.6 |
| clients_4 | 166.0 | 158.2 | 69.0 | 43.6% | 110.1 / 46.6 | 262.7 | 318.4 |
| clients_16 | 660.3 | 632.7 | 276.5 | 43.7% | 440.7 / 186.9 | 577.1 | 719.0 |
| clients_64 | 2,630.6 | 2,526.6 | 1,100.9 | 43.6% | 1,758.1 / 744.0 | 1,835.9 | 2,301.9 |

## Equivalence and reproduction

The tiny FP32 suite passed on the Mac with both mask modes. The first CPU attempt on engopt selected GPU-only FLA with CUDA hidden and exited before its first comparison. The Mac run used the existing CPU environment and the same read-only model source. The standard calibrated BF16 suite passed 234/234 decisions. The graph-enabled COCO/clip benchmark matched all 480 decisions across 20 path/question combinations of 24 examples each. The speed diagnostic checks 930 decisions on 60 images per engine variant against the shipped BF16 model. All five exact engine variants reached 930/930. [CPU](raw/item3_cpu_local.log) · [Standard BF16](raw/item3_real.log) · [COCO and clips](raw/item3_equiv.json) · [Speed diagnostic](raw/item3_diag.json)

The additional merged-model test exercises the production fusions against the unmerged readout and plain HF forward. Its first run kept all 234 decisions but exceeded the existing normalized-coordinate limit (0.424945) on two causal bank comparisons, reaching 0.49224. Bank construction now uses the same validated image tower as fresh scan. The rerun passed all gates. The maximum causal cached coordinate difference fell to 0.1676, and the specific plain-HF image comparison fell to 0.00404. A control with the earlier merged LM crossed the stricter mixed-image coordinate limit of 0.00668645 in the separate-image path. Applying the image-tower change to native clips also kept their decisions but gave 0.03104 maximum cached typed-probability drift over 24 clips. Scoping the image-tower change to still images restored the native clip path and reduced this to 0.02309, inside the existing 0.02335 typed-probability limit. Both the final merged and standard real-weight suites passed all 234 decisions and continuous gates. [Final merged suite](raw/item3_real_release.log) · [Clip control](raw/item3_equiv_tower.json) · [Initial failure](raw/item3_real_fused.log) · [Earlier merged control](raw/item3_real_control.log) · [Corrected tower](raw/item3_real_tower.log)

The packed frame-specific video path matched 384/384 presence decisions from the earlier cache-producing engine on identical features, four rotations per size. The maximum probability differences were 0.01715 for presence and 0.09298 for IoU. The maximum normalized-coordinate difference was 0.07247. [Per-frame comparisons](raw/item3_video_equiv.json)

The identical-feature sweep compared the optimized packed LM with the original packed LM across 32, 64, 128 and 256 frames for box and yes/no questions. All 960 decisions agreed. Maximum typed probability drift was 0.01091. At 128 frames the box comparison moved a normalized coordinate by up to 0.32592 while every presence decision agreed. The checkpoint’s hard center selection matters when interpreting BF16 coordinate drift. [Identical-feature checks](raw/item3_fixed_features.json)

With the SM120 recurrence enabled, all 960 fixed-feature and 384 frame-specific decisions also agreed. Maximum fixed-feature drift was 0.00657 in typed probability, 0.01492 in presence probability and 0.32288 in normalized coordinates. The merged calibrated suite and the 24-example scan benchmark were repeated with the optional packages loaded. [Accelerated fixed features](raw/item3_accel_fixed_features.json) · [Video](raw/item3_accel_video_equiv.json) · [Calibrated suite](raw/item3_accel_real_release.log) · [Scan decisions](raw/item3_accel_equiv.json) · [Speed decisions](raw/item3_accel_diag.json)

Use the item-3 runner on an idle, synced engopt checkout. Use `--lm_prefill reference` for the original kernels in either benchmark, or `lm_prefill=False` on either engine constructor. The scan benchmark also accepts `pointwise` for the norm/conv fusions with library MLP and `direct` for the first native SwiGLU kernel. Select `fla` to keep the native TMA MLP and force the FLA GDN backend. Its default is `fused` (TMA on supported SM120 shapes). To reproduce the calibrated merged-versus-shipped check, run `python -m scan.test_scan_equivalence --real --fused-prefill optimized --ckpt /content/ckpt/v1/ckpt_final.pt` on engopt. Generate both per-kernel tables with `python -m scan.lm_profile_rates outputs/engine_opt/03_lm_prefill/raw`. The accelerated measurements use isolated packages on engopt. The standard-stack path uses PyTorch, Triton and FLA. Installing the `sm120-prefill` extra enables the 64-wide FlashQLA value tile on SM120 for supported packed calls of at least 4,096 rows. Launch with `FLA_FLASH_QLA=0` to keep FLA’s own automatic dispatcher on its calibrated chunk-64 implementation for other calls. The item runner applies this setting for `accel_*` stages. The consolidated validation file records counts and maximum deltas for both stacks. The HF result archive is under `engine_opt/03_lm_prefill` in `Vision-Decision/collection-scan-results`. [Validation totals](raw/item3_verification.json) [Runner](../../../scan/vm/lm_prefill.sh) · [GPU profiler](../../../scan/profile_lm.py)

## Authorship

Codex wrote the engine changes, Triton kernels, FlashQLA adapter, profilers, benchmark controls and reports, and ran the measurements. The reviewer provided the prepared engopt VM, checkpoint, datasets and prior baseline series. The worktree is uncommitted.
