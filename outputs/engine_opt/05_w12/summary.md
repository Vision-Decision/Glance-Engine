# Item 5 · Lossless W12 language projections on SM120

One box follow-up on a retained ≤640-token image took **7.04 ms with W12**, down from 8.67 ms with BF16. Querying one image in the scan bank took 5.55 ms, down from 7.32 ms. Eight box questions on one retained prefix took 9.84 ms with W12 and 10.11 ms with BF16. All 3,569,090,560 encoded weights recovered their original BF16 bits. The W12 engine agreed with BF16 on 2,460 mixed follow-up answers and 455 cached presence decisions. [Speed sweep](raw/item5_verified_w12.json) · [Bank sweep](raw/item5_verified_scan_w12.json) · [Paired accuracy](raw/item5_verified_accuracy.json)

## Where weight traffic matters

The measured G4 rates were 401.73 BF16 TFLOP/s and 1,463.2 GB/s. A full BF16 dense pass reads 7.138 GB, a 4.88 ms lower bound. The packed tensor payload reduces that bound to 3.67 ms. The compute floor is about 0.01777 ms per useful LM token. Ignoring activation traffic, the two bounds meet at about 275 BF16 rows or 207 fully packed rows. The pre-optimization item-4 events placed one box follow-up at 6.18 ms in the dense stage against a 4.88 ms floor. [Measured rates and model](../ROOFLINE.md) · [Prior stage measurements](../04_gdn_branch/raw/item4_stage_floors.json)

We measured the incremental cost of fetching real weights from device memory by replaying the same shape with rotating matrices and then reusing one L2-resident matrix. Each rotating group spans at least 256 MiB in W12 on a 128 MiB L2. Weighting the five matrix shapes by their layer counts gives the following sum of projection times. The percentage measures the extra time for cold weights, including cache and execution overlap. It excludes the residual cost of reading the hot operands. [Rotating and reused graph timings](raw/item5_final_micro_direct.json) · [Aggregation and hardware counters](raw/item5_weight_traffic.json)

| GEMM rows | Rotating BF16 ms | Reused BF16 ms | Increment attributable to cold weights |
|---:|---:|---:|---:|
| 16 | 5.39 | 2.61 | 51.6% |
| 64 | 5.44 | 2.47 | 54.7% |
| 128 | 5.92 | 3.85 | 35.0% |
| 256 | 7.10 | 6.66 | 6.1% |
| 512 | 12.29 | 12.01 | 2.3% |
| 1,024 | 22.39 | 22.19 | 0.9% |

The cached dense workload scales with the sum of question-block lengths across prefixes. In the existing table, 64 and 256 images with a box query have median useful row counts of 736 and 2,944. A cold single-image call evaluates both prefix and question tokens in the same projection. At the nominal 16k fresh workload, the representative packed call evaluates 18,688 rows. Nsight measured total DRAM traffic and kernel duration. That yields 382.5 useful TFLOP/s and 361 GB/s for its cuBLAS gate/up control, compared with 17.5 TFLOP/s and 1,341 GB/s at 16 rows. Item 3 measured the production fused SwiGLU at 4.197 ms. We retained its BF16 path for that workload. [Counter CSV](raw/item5_counters.csv) · [Invocation order](raw/item5_counters.json) · [Production prefill profile](../03_lm_prefill/summary.md)

## Storage and kernels

The offline encoder runs after LoRA merge and projection fusion. For each output channel it chooses a 16-binade window, stores sign and seven mantissa bits, and writes the original 16-bit pattern of every exception to a row-indexed sparse list. The on-disk tensors occupy **5,366,326,968 bytes**, or **12.02845 bits/weight**, including tile padding, row windows, row offsets and escapes. There are **281,134 escapes** (0.007877%). The serialized directory on engopt occupied 5,366,728,908 bytes including its manifest and file overhead. All 128 matrices passed the integer round trip. Installation also decodes each matrix on the GPU and compares every bit against the active fused checkpoint. [Per-matrix counts and round trips](raw/item5_encode.json) · [Offline codec and dispatch](../../../speed/w12.py)

| Executed projection | Matrices | Bits/weight | Escapes per million weights |
|---|---:|---:|---:|
| MLP gate/up | 32 | 12.01806 | 76.14 |
| MLP down | 32 | 12.00728 | 91.73 |
| GDN QKV/Z/B/A | 24 | 12.08014 | 70.44 |
| GDN output | 24 | 12.01227 | 78.11 |
| Attention QKV | 8 | 12.01813 | 78.22 |
| Attention output | 8 | 12.01259 | 88.17 |

The fused GDN input has 12,352 output channels and pads to 12,416 in 128-channel records. That tail accounts for most of its extra bits. Serving keeps the original 7.138 GB of dense weights for BF16 dispatch and adds 5.374 GB of packed buffers before allocator overhead and graph workspaces. Only selected skinny operators use the narrower weight stream. The public loader accepts an offline W12 directory, and creates the engine after selecting its precision. Its separate FP8 option now uses the same pre-construction conversion and separate BF16 GDN gates as the FP8 benchmark. [Scanner loader](../../../scan/api.py) · [Model construction](../../../scan/pilot_charades.py)

The production kernel adapts SpecPilot’s Apache-2.0 SM120 persistent stream-K implementation. It decodes a 12 KiB tile in registers, multiplies BF16 operands with tensor-core MMA, applies per-channel power-of-two scaling and corrects sparse escapes in FP32. Split-K workspaces and counters belong to the call and are captured in its CUDA graph. The graph path buckets short retained questions at 16, 32, 64, 96 and 128 rows. Larger questions retain the existing 64-row increment. Different reduction orders can change individual BF16 output bits. [CUDA implementation and provenance](../../../speed/specpilot/w12.cu) · [SpecPilot upstream](https://github.com/StargazerX0/SpecPilot/tree/1058dc29481150e66ae11b8e8cf55ed60caf254a) · [SpecPilot paper](https://arxiv.org/abs/2611.02501)

W12 dispatch uses only the probed row counts. The upper limits are 128 for fused gate/up, 96 for GDN input and attention QKV, 40 for MLP down and 24 for GDN/attention output. At 64 rows gate/up took 52.6 µs versus 72.1 µs BF16 in the first sweep, while the 2,560 × 4,096 output took 29.0 µs versus 18.6 µs. At 128 rows the wider stream-K gate/up measured 72.2 versus 76.3 µs. A second design loaded fragments directly into registers and reduced independent K splits in another launch. A third moved escape handling into the final epilogue to free shared memory for deeper prefetch. Stream-K had the lowest time in the gate/up microbenchmark and was the variant taken to end-to-end testing. At 16 rows the fused gate/up took 50.2 µs with stream-K, 53.3 µs with the best direct split in the second sweep and 54.6 µs with epilogue correction. Materializing BF16 before cuBLAS took 117.5 µs at eight rows. [Original kernels and decode control](raw/item5_micro.json) · [Direct splits and wide shapes](raw/item5_final_micro_direct.json) · [Epilogue comparison](raw/item5_micro_epilogue.json)

## End-to-end latency

All values are warm medians in milliseconds on the dedicated G4. Glance BF16 and W12 use the item-3/item-4 optimized standard FLA stack. FP8 uses the existing language-only W8A8 path with dynamic activation quantization. Its precision and prefill dispatch differ from BF16. The speed sweep rotates 30 calls per question regime and seven per video size. The scan sweep uses ten changing questions per regime. Cached rows exclude bank construction. [BF16 speed](raw/item5_bf16.json) · [FP8 speed](raw/item5_fp8.json) · [BF16 scan](raw/item5_scan_bf16.json) · [FP8 scan](raw/item5_scan_fp8.json) · [W12 speed](raw/item5_verified_w12.json) · [W12 scan](raw/item5_verified_scan_w12.json)

| Workload | BF16 | W12 | FP8 LM | vLLM BF16 | SGLang BF16 | W12 floor | Floor / W12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold, box, Q=1 | 37.83 | 37.53 | 34.93 | 166.5 | 187.3 | 17.47 | 46.5% |
| Cold, box, Q=8 | 38.54 | 38.63 | 36.98 | 221.0 | 395.8 | 19.12 | 49.5% |
| Cold, box, Q=32 | 48.61 | 48.80 | 49.68 | 334.0 | 658.0 | 24.63 | 50.5% |
| Cold, mixed, Q=8 | 45.95 | 46.19 | 42.54 | 199.2 | 375.3 | 22.89 | 49.6% |
| Cold, mixed, Q=32 | 69.07 | 69.13 | 74.64 | 278.2 | 655.4 | 39.56 | 57.2% |
| Follow-up, box, Q=1 | 8.67 | 7.04 | 9.47 | 145.2 | 162.0 | 3.77 | 53.5% |
| Follow-up, mixed, Q=1 | 8.65 | 7.03 | 9.47 | 145.1 | 161.7 | 3.77 | 53.6% |
| Follow-up, box, Q=8 | 10.11 | 9.84 | 11.60 | 196.6 | 346.1 | 4.53 | 46.1% |
| Follow-up, mixed, Q=8 | 14.65 | 14.66 | 16.37 | 175.3 | 342.2 | 5.63 | 38.4% |
| Follow-up, box, Q=32 | 19.20 | 19.20 | 22.69 | 316.9 | 606.5 | 7.61 | 39.6% |
| Follow-up, mixed, Q=32 | 39.47 | 39.54 | 48.47 | 254.9 | 561.8 | 22.31 | 56.4% |
| Video, 32 frames, 1 box per frame | 393.39 | 395.90 | 511.94 | 548.6 | 672.3 | 217.81 | 55.0% |
| Video, 64 frames, 1 box per frame | 795.87 | 799.74 | 1029.44 | 933.1 | 1046.3 | 435.78 | 54.5% |

| Workload | BF16 | W12 | FP8 LM | vLLM BF16 | SGLang BF16 | W12 floor | Floor / W12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bank Tv640, 1 image, box | 7.32 | 5.55 | 8.05 | 26.7 | 56.1 | 3.76 | 67.8% |
| Bank Tv640, 1 image, yes/no | 7.44 | 6.04 | 8.42 | 14.2 | 45.5 | 3.89 | 64.3% |
| Bank Tv640, 8 images, box | 8.40 | 7.02 | 9.72 | 175.7 | 283.3 | 4.54 | 64.7% |
| Bank Tv640, 8 images, yes/no | 9.79 | 9.81 | 12.32 | 34.9 | 147.0 | 5.46 | 55.6% |
| Bank Tv640, 32 images, box | 13.50 | 13.50 | 16.77 | 401.3 | 578.1 | 8.36 | 61.9% |
| Bank Tv640, 32 images, yes/no | 24.11 | 24.17 | 27.58 | 229.9 | 405.7 | 15.97 | 66.1% |
| Bank Tv640, 64 images, box | 23.48 | 23.50 | 27.85 | 899.1 | 1556.6 | 16.69 | 71.0% |
| Bank Tv640, 64 images, yes/no | 44.30 | 44.66 | 54.82 | 680.6 | 1287.3 | 31.90 | 71.4% |
| Bank Tv640, 256 images, box | 92.12 | 92.47 | 117.36 | 3151.6 | 7732.9 | 66.68 | 72.1% |
| Bank Tv640, 256 images, yes/no | 176.27 | 177.14 | 255.56 | 2679.9 | 5670.7 | 127.52 | 72.0% |

| Workload | BF16 | W12 | FP8 LM | vLLM BF16 | SGLang BF16 | W12 floor | Floor / W12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh Tv128, 1 frame, yes/no | 12.68 | 12.69 | 41.21 | 21.4 | 31.0 | 5.82 | 45.8% |
| Fresh Tv128, 2 frames, yes/no | 16.30 | 16.26 | 42.51 | 35.4 | 54.2 | 6.98 | 42.9% |
| Fresh Tv128, 4 frames, yes/no | 25.56 | 25.56 | 45.02 | 42.9 | 66.0 | 13.92 | 54.5% |
| Fresh Tv128, 8 frames, yes/no | 45.42 | 45.52 | 51.12 | 61.4 | 98.0 | 27.81 | 61.1% |
| Fresh Tv256, 1 frame, yes/no | 17.36 | 17.38 | 43.42 | 24.9 | 40.3 | 7.00 | 40.3% |
| Fresh Tv256, 2 frames, yes/no | 26.81 | 26.82 | 46.78 | 43.8 | 64.6 | 13.97 | 52.1% |
| Fresh Tv256, 4 frames, yes/no | 45.67 | 45.79 | 53.45 | 59.4 | 92.3 | 27.91 | 60.9% |
| Fresh Tv256, 8 frames, yes/no | 81.32 | 81.64 | 75.15 | 101.9 | 157.8 | 55.82 | 68.4% |

An independent repeat measured 7.05 ms for one box follow-up, 9.81 ms for eight and 7.02 ms for one box query over eight banked images. [Repeat speed sweep](raw/item5_release_w12.json) · [Repeat bank sweep](raw/item5_release_scan_w12.json)

The serving columns use the clean BF16 baselines for the same output and cache protocol. We remeasured the missing SGLang N=1/8/32 cached rows on engopt. The cold speed and video rows, large cached queries and small fresh scans dispatch BF16 throughout on this checkpoint. Differences between BF16 and W12-enabled wall clock on these rows measure run-to-run variation and wrapper/host costs. [New SGLang bank timings](raw/item5_sglang.json) · [Cached vLLM](../../scan/raw/scan-a/vllm_bf16.json) · [Cached SGLang](../00_ceiling/raw/c0_sglang_bank.json) · [Speed vLLM](../../sprint_speed/raw/serve_vllm_async_g4.json) · [Speed SGLang](../00_ceiling/raw/c0_sglang_clean.json) · [Fresh vLLM](../02_vision_prep/raw/item2_cold_vllm.json) · [Fresh SGLang](../02_vision_prep/raw/item2_serving_sglang.json)

## Stage measurements and floor

The floor uses the measured G4 rates and unpadded useful tokens for each call. For a dispatched matrix it charges the padded planes, FP32 row scales, CSR offsets, columns and corrections. Other matrices retain their BF16 byte cost. Fresh scans use the item-2 RGB transfer floor. Each stage uses the larger of its byte and FLOP bounds. Host prep has a zero modeled lower bound. The raw JSON contains per-stage measurements and ceilings for all 31 rows in the preceding tables. [Per-workload floors](raw/item5_workload_floors.json) · [W12 floor calculation](ceiling.py) · [Base floor model](../../../speed/ceiling_model.py)

| Workload | Prep ms | Vision + H2D, ms / floor | LM + readout, ms / floor | LM ceiling |
|---|---:|---:|---:|---:|
| Cold, box, Q=1 | 2.34 | 9.79 / 5.66 | 25.40 / 11.81 | 46.5% |
| Follow-up, box, Q=1 | 0.32 | — | 6.72 / 3.77 | 56.1% |
| Follow-up, box, Q=8 | 0.63 | — | 9.20 / 4.53 | 49.2% |
| Follow-up, mixed, Q=32 | 2.26 | — | 37.30 / 22.30 | 59.8% |
| Bank Tv640, 1 image, box | 0.09 | — | 5.46 / 3.76 | 68.9% |
| Bank Tv640, 8 images, box | 0.10 | — | 6.93 / 4.54 | 65.6% |
| Bank Tv640, 64 images, box | 0.11 | — | 23.39 / 16.69 | 71.3% |
| Bank Tv640, 256 images, yes/no | 0.16 | — | 176.99 / 127.52 | 72.1% |
| Fresh Tv128, 1 frame, yes/no | 0.48 | 2.89 / 0.79 | 9.32 / 5.02 | 53.9% |
| Fresh Tv256, 8 frames, yes/no | 1.11 | 27.27 / 14.56 | 53.26 / 41.26 | 77.5% |
| Video, 32 frames, 1 box per frame | 32.36 | 167.47 / 60.72 | 195.60 / 157.09 | 80.3% |
| Video, 64 frames, 1 box per frame | 66.20 | 338.63 / 121.49 | 394.96 / 314.23 | 79.6% |

We also timed the 128 dense projections on the first query in each branch regime with 11 warmed graph replays. The compact BF16 control uses the same question buckets as W12. Measurements include the native fused gate/up epilogue where the large-row implementation calls it. [Dense graph events and actual dispatch counts](raw/item5_verified_stages.json)

| Fixed query | BF16 rows → W12 rows | W12 GEMMs / 128 | BF16 ms | BF16 with W12 buckets, ms | W12 ms |
|---|---:|---:|---:|---:|---:|
| Follow-up, box, Q=1 | 64 → 16 | 128 | 6.17 | 5.86 | 4.75 |
| Follow-up, box, Q=8 | 128 → 128 | 32 | 6.75 | 6.76 | 6.53 |
| Follow-up, box, Q=32 | 448 → 448 | 0 | 12.49 | 12.49 | 12.49 |
| Bank Tv640, 1 image, box | 64 → 16 | 128 | 6.13 | 5.80 | 4.51 |
| Bank Tv640, 1 image, yes/no | 64 → 32 | 96 | 6.13 | 6.14 | 4.85 |
| Bank Tv640, 8 images, box | 128 → 96 | 64 | 6.73 | 6.60 | 5.43 |
| Bank Tv640, 32 images, box | 384 → 384 | 0 | 9.96 | 9.96 | 9.95 |
| Bank Tv640, 64 images, box | 704 → 704 | 0 | 16.72 | 16.72 | 16.72 |
| Bank Tv640, 256 images, yes/no | 6144 → 6144 | 0 | 112.17 | 112.50 | 112.60 |

For the first box query over one cached image, the 128 BF16 projections took 6.13 ms with the old 64-row graph and 5.80 ms with a 16-row graph. Keeping that 16-row graph and switching its projections to W12 lowered dense time to 4.51 ms. Over eight cached images, the corresponding times were 6.73, 6.60 and 5.43 ms. [Bucket-matched graph events](raw/item5_verified_stages.json)

## Accuracy

| Check | BF16 | W12 | FP8 LM |
|---|---:|---:|---:|
| Cold decisions vs fused BF16 | 930/930 | 930/930 | 926/930 |
| Retained-prefix mixed decisions vs fused BF16 | 2,460/2,460 | 2,460/2,460 | 2,444/2,460 |
| Cached presence decisions vs fused BF16 | 455/455 | 455/455 | 452/455 |
| RefCOCO Acc@0.5, 450 ground-truth boxes | 94.00% | 94.00% | 93.56% |
| gRefCOCO cached presence AUROC | 0.96970 | 0.96968 | 0.96899 |

The sprint diagnostic supplies 60 RefCOCO images, 930 mixed answers and 450 ground-truth boxes. Its positive-only box set cannot estimate presence AUROC. The AUROC row uses the first 60 images of the existing gRefCOCO scan subset at Tv≤640, with 146 present and 309 absent expressions, one cached question at a time and half credit for ties. W12's largest mixed-follow-up deltas were 0.000213 in presence probability, 0.006324 in typed probability and 0.001210 in normalized box coordinates. W12 passed the existing BF16 acceptance limits. [Paired evaluation](raw/item5_verified_accuracy.json) · [Sprint FP32 diagnostic](raw/item5_diag.json)

Tiny FP32 scan passed both masks. The CPU codec test covered every BF16 bit pattern, signed zeros, an incomplete tile and an invalid escape column. Standard BF16, merged BF16 and W12 each passed 234/234 calibrated real-weight decisions and continuous gates. The scan sweep matched the shipped single-image/clip reference on 480/480 decisions across 20 groups. The separate engine diagnostic reproduced 930/930 decisions for every BF16 engine variant against shipped BF16 (928/930 against FP32 for the shipped reference). [CPU scan](raw/item5_cpu.log) · [CPU codec](raw/item5_codec_cpu.log) · [Standard](raw/item5_real.log) · [Merged](raw/item5_merged.log) · [W12](raw/item5_verified_w12_real.log) · [Scan comparisons](raw/item5_verified_equiv.json)

The public `Scanner.load` entry point was also tested with one and eight images, fresh and cached calls, and box and yes/no readouts. W12 and FP8 each matched all 36 BF16 threshold decisions on this smoke test. The larger paired evaluation above measured the FP8 decision differences. [Public API results](raw/item5_verified_api.json) · [Offline export revalidation](raw/item5_verified_encode.log)

## Reproduction

The reported W12 latency and accuracy results use the `verified_` files. The `release_` latency files record the repeat. Earlier snapshots used narrower dispatch or different question buckets. The first `release_stages` probe omitted the 32 native fused SwiGLU calls at the largest batch. Its corrected `verified_stages` replacement measured all 128 projections for every row.

Run the following from the worktree root with engopt set up. The offline files live on the VM under `/content/item5-w12-v1`. Rerunning the encoder there validates those files against the merged checkpoint. The CUDA extension needs the CUDA toolkit, a C++ compiler and Ninja. The runner takes the shared exclusive GPU lock, checks for an empty compute-process list before every invocation and archives each result and log. To load through the Python API, use `Scanner.load(ckpt, w12="/content/item5-w12-v1")`. [Runner](../../../scan/vm/w12.sh) · [GPU result archive](https://huggingface.co/datasets/Vision-Decision/collection-scan-results/tree/main/engine_opt/05_w12)

```sh
MONO=~/Developer/decision-vison/.claude/worktrees/engine-opt bash scan/vm/sync.sh engopt
bash scan/vm/bg.sh engopt item5_reproduce 'bash scan/vm/w12.sh encode micro final_micro_direct micro_epilogue counters real merged diag bf16 fp8 scan_bf16 scan_fp8 sglang verified_w12_real verified_equiv verified_accuracy verified_stages verified_w12 verified_scan_w12 verified_api'
```

After pulling the raw measurements, run `PYTHONPATH=. python3 outputs/engine_opt/05_w12/ceiling.py` from the repository root to regenerate the 31 workload floors and timing comparisons.

## Authorship

Codex wrote the Glance encoder, reference decoder, dispatch, loader, probes, tests, runner and reports, adapted the upstream Apache-2.0 CUDA kernel, implemented the direct split-K and sparse-epilogue experiments, and ran the measurements. The reviewer supplied the configured engopt VM, checkpoint, datasets and prior clean baselines. Worktree base is `c4433de`.
