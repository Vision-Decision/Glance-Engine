# Frame throughput and cached follow-ups on the G4

Across 64 clients issuing 32 new frames each, Glance finished in **9,695.5 ms**, down from 9,917.9 ms. At Tv=256 with 256 fresh frames it fell from **2,493.8 to 2,420.5 ms**. A 64-frame video call fell from **773.6 to 587.7 ms**, versus 695.7 ms for vLLM FP8 + MTP. Glance finished ahead of both BF16 comparison servers in 8/8 fresh and 24/24 scheduler cells. The large fresh frame burst still has 2,542.0 ms of modeled headroom. [Fresh timings](raw/item7_final_fresh.json) · [Video timings](raw/item7_checked_video.json) · [Scheduler timings](raw/item7_final_scheduled.json)

## Measurements

All latencies are warm medians in milliseconds on the RTX PRO 6000 Blackwell Server Edition. The timers start with resized host images and stop when the requested answers reach the host. A fresh scan applies one yes/no question to each frame. The video workload uses an image-specific box question at each frame's original resized aspect ratio. A scheduler burst ends when the last client has all its answers. Glance uses PyTorch 2.14.1+cu130, the v1 checkpoint, BF16 weights and the optional SM120 FlashQLA stack. The comparison servers are vLLM 0.34.0 and SGLang 0.5.23. The frame benchmarks use ten fresh calls and twelve video calls. Scheduler cells use five simultaneous bursts after two warm-ups. Each detached process acquired both GPU locks and checked for an empty GPU before timing. [Reproduction runner](../../../scan/vm/frame_throughput.sh) · [Checkpoint and serving protocols](../CEILING.md#benchmark-boundaries)

### Fresh scan and video

The columns for vision and LM show measured time / floor. Vision includes H2D. On the video route it also includes packing the resized images into pinned RGB storage. CPU preparation has a zero modeled floor. The last percentage is the total floor divided by measured call latency. Medians of individual stages can sum to a different total.

| Workload | Before | After | CPU | Vision / floor | LM / floor | Total floor | Floor / after | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tv=128, N=32 | 146.3 | 142.9 | 2.5 | 41.6 / 25.4 | 98.9 / 85.9 | 111.2 | 77.8% | 177.9 | 259.3 |
| Tv=128, N=64 | 295.7 | 287.5 | 4.0 | 83.1 / 50.7 | 200.5 / 171.7 | 222.5 | 77.4% | 332.1 | 431.4 |
| Tv=128, N=128 | 598.8 | 580.8 | 7.9 | 168.7 / 101.4 | 403.6 / 343.5 | 444.9 | 76.6% | 637.8 | 815.3 |
| Tv=128, N=256 | 1,206.7 | 1,165.3 | 15.8 | 338.5 / 202.8 | 810.9 / 687.0 | 889.8 | 76.4% | 1,252.8 | 1,572.9 |
| Tv=256, N=32 | 311.3 | 301.7 | 4.6 | 102.9 / 58.2 | 194.4 / 165.0 | 223.3 | 74.0% | 322.8 | 461.0 |
| Tv=256, N=64 | 621.2 | 602.1 | 8.1 | 203.9 / 116.5 | 390.0 / 330.0 | 446.5 | 74.2% | 629.4 | 848.4 |
| Tv=256, N=128 | 1,244.4 | 1,207.5 | 16.4 | 409.4 / 232.9 | 782.0 / 660.1 | 893.0 | 74.0% | 1,244.7 | 1,567.6 |
| Tv=256, N=256 | 2,493.8 | 2,420.5 | 32.5 | 818.9 / 465.9 | 1,569.1 / 1,320.2 | 1,786.1 | 73.8% | 2,495.9 | 2,997.1 |
| 32 frames | 382.7 | 290.1 | 4.1 | 103.4 / 57.7 | 182.5 / 157.0 | 214.7 | 74.0% | 549.4 | 666.3 |
| 64 frames | 773.6 | 587.7 | 8.0 | 207.5 / 115.4 | 372.1 / 314.3 | 429.7 | 73.1% | 928.1 | 1,052.0 |

[Fresh before](raw/item7_before_fresh.json) · [Fresh after](raw/item7_final_fresh.json) · [Video before](raw/item7_before_video.json) · [Video after](raw/item7_checked_video.json) · [vLLM fresh](raw/item7_final_fresh_vllm.json) · [SGLang fresh](raw/item7_final_fresh_sglang.json) · [vLLM video](raw/item7_matched_video_vllm.json) · [SGLang video](raw/item7_final_video_sglang.json) · [vLLM FP8 + MTP](raw/item7_matched_video_tuned.json) · [Compact video floors](raw/item7_video_floors.json)

At Tv=256 and N=256, the earlier ceiling map had Glance at 2,497.5 ms and vLLM at 2,482.4 ms. The current 2,420.5 ms result is below both the archived vLLM median and the 2,495.9 ms rerun. [Earlier Glance sweep](../03_lm_prefill/raw/item3_accel_production.json) · [Earlier vLLM sweep](../02_vision_prep/raw/item2_cold_vllm.json)

Both fresh servers reported zero cached prompt tokens in all eight cells. Fresh vLLM resets the prefix, media and encoder caches. Fresh SGLang flushes its cache and receives content-unique images. The video comparison reproduces the sprint protocol, including its media resets and 24-token limit for a full box. The historical BF16 one-token box proxy is listed separately in the ceiling map. [Fresh vLLM protocol](../../../scan/bench_vllm.py) · [Fresh SGLang protocol](../../../scan/bench_sglang.py) · [Video serving protocol](../../../speed/bench_serve.py)

### Simultaneous clients

The before column uses the item 6 production scheduler on the same GPU and workload. The table resends pixels on cache hits. The two servers process the complete requested outputs. Each row gives the median last-completion time, with separate client latency samples and stage samples in the raw files.

| Workload | Clients | Before | After | CPU | Vision / floor | LM / floor | Total floor | Floor / after | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold box | 1 | 32.2 | 32.2 | 0.4 | 9.7 / 5.4 | 20.9 / 11.8 | 17.2 | 53.4% | 166.2 | 188.1 |
| Cold box | 4 | 103.6 | 104.0 | 1.2 | 36.1 / 21.3 | 64.3 / 46.5 | 67.8 | 65.2% | 264.9 | 322.8 |
| Cold box | 16 | 368.7 | 362.7 | 2.8 | 133.5 / 85.9 | 228.6 / 187.0 | 272.8 | 75.2% | 578.6 | 715.4 |
| Cold box | 64 | 1,439.4 | 1,424.3 | 10.8 | 529.3 / 341.3 | 882.1 / 742.9 | 1,084.2 | 76.1% | 1,828.0 | 2,330.0 |
| Cold yes/no | 1 | 32.2 | 32.2 | 0.4 | 9.7 / 5.4 | 20.9 / 12.0 | 17.4 | 54.2% | 39.5 | 53.8 |
| Cold yes/no | 4 | 105.6 | 104.3 | 1.0 | 35.5 / 21.3 | 66.4 / 47.5 | 68.8 | 65.9% | 111.1 | 157.2 |
| Cold yes/no | 16 | 374.8 | 370.3 | 3.4 | 134.6 / 85.9 | 234.2 / 190.7 | 276.6 | 74.7% | 381.7 | 508.6 |
| Cold yes/no | 64 | 1,471.5 | 1,456.3 | 11.7 | 531.4 / 341.3 | 906.5 / 757.9 | 1,099.2 | 75.5% | 1,505.3 | 1,946.8 |
| Eight mixed questions | 1 | 45.6 | 45.3 | 1.0 | 9.7 / 5.4 | 33.3 / 17.1 | 22.5 | 49.6% | 199.6 | 378.3 |
| Eight mixed questions | 4 | 142.4 | 141.1 | 3.0 | 35.5 / 21.3 | 100.3 / 68.1 | 89.5 | 63.4% | 351.1 | 888.8 |
| Eight mixed questions | 16 | 538.4 | 534.0 | 10.7 | 135.8 / 85.9 | 387.2 / 272.1 | 358.0 | 67.0% | 979.4 | 2,734.1 |
| Eight mixed questions | 64 | 2,161.0 | 2,144.3 | 42.5 | 543.6 / 341.3 | 1,557.1 / 1,084.6 | 1,425.9 | 66.5% | 3,499.5 | 12,305.1 |
| Cached box, resent image | 1 | 11.7 | 8.7 | 0.5 | 0.0 / 0.0 | 8.0 / 5.0 | 5.0 | 57.2% | 140.3 | 167.1 |
| Cached box, resent image | 4 | 17.4 | 17.3 | 5.8 | 0.0 / 0.0 | 10.1 / 5.2 | 5.2 | 30.1% | 182.0 | 248.2 |
| Cached box, resent image | 16 | 48.6 | 47.4 | 22.3 | 0.0 / 0.0 | 24.5 / 6.2 | 6.2 | 13.1% | 253.3 | 452.8 |
| Cached box, resent image | 64 | 175.2 | 173.1 | 87.8 | 0.0 / 0.0 | 82.9 / 19.8 | 19.8 | 11.5% | 1,019.0 | 1,801.8 |
| GUI box, ≤2,048 tokens | 1 | 98.6 | 98.3 | 0.5 | 42.8 / 29.0 | 53.7 / 38.2 | 67.2 | 68.4% | 233.7 | 297.6 |
| GUI box, ≤2,048 tokens | 4 | 358.6 | 354.1 | 1.5 | 161.0 / 115.2 | 190.3 / 152.0 | 267.2 | 75.5% | 521.2 | 705.5 |
| GUI box, ≤2,048 tokens | 16 | 1,409.5 | 1,384.3 | 4.4 | 642.6 / 461.2 | 735.9 / 608.5 | 1,069.7 | 77.3% | 1,607.2 | 2,221.4 |
| GUI box, ≤2,048 tokens | 64 | 5,621.2 | 5,515.6 | 17.3 | 2,546.8 / 1,838.7 | 2,945.9 / 2,428.7 | 4,267.5 | 77.4% | 5,983.4 | 8,640.1 |
| 32 fresh frames, yes/no | 1 | 160.3 | 157.5 | 0.6 | 48.3 / 25.4 | 107.3 / 87.3 | 112.7 | 71.5% | 181.1 | 248.9 |
| 32 fresh frames, yes/no | 4 | 622.4 | 609.3 | 1.7 | 191.3 / 101.4 | 414.6 / 346.4 | 447.8 | 73.5% | 635.2 | 809.8 |
| 32 fresh frames, yes/no | 16 | 2,479.1 | 2,422.7 | 6.1 | 763.9 / 405.7 | 1,646.9 / 1,382.7 | 1,788.4 | 73.8% | 2,552.3 | 3,069.3 |
| 32 fresh frames, yes/no | 64 | 9,917.9 | 9,695.5 | 24.3 | 3,053.9 / 1,622.8 | 6,592.8 / 5,530.8 | 7,153.5 | 73.8% | 10,603.3 | 12,724.0 |

[Before scheduler](../06_scheduler/raw/item6_production.json) · [After scheduler and percentiles](raw/item7_final_scheduled.json) · [Burst floors](../06_scheduler/raw/item6_floors.json) · [vLLM bursts](raw/item7_final_scheduler_vllm.json) · [SGLang bursts](raw/item7_final_scheduler_sglang.json)

The singleton median was **8.64 ms** in its dedicated run and **8.71 ms** in the full matrix. The same-G4 pre-item scheduled and serial medians were 11.63 and 8.77 ms. The serial harness already knows the prefix by object identity. The scheduler still verifies resent pixel content. An immediately available cached request can bypass the gathering delay. A bounded snapshot of the most recently hashed RGB input permits byte-for-byte confirmation, including dimensions. Padded prefix state and its spatial metadata can be reused by the one-question graph. On a miss or on pixel mutation the usual content digest determines the cache key. [Before scheduler](raw/item7_before_followup.json) · [Serial control](raw/item7_before_serial.json) · [Dedicated retest](raw/item7_final_singleton.json) · [Mutable-image regression](../../../speed/test_server.py) The final-source repeat measured 8.73 ms scheduled and 8.84 ms serial. [Repeat scheduler](raw/item7_patch_singleton.json) · [Repeat serial](raw/item7_patch_serial.json)

With 16,384 tokens available per scheduler batch, the 16-client and 64-client fresh bursts measured 2,398.2 ms (client median 1,353.4 ms), 9,658.5 ms (client median 4,994.5 ms). The 8,192-token production sweep measured 2,422.7 ms (client median 1,292.6 ms), 9,695.5 ms (client median 4,928.5 ms). The larger budget saved 0.4% on the 64-client burst and delayed its median client by 66 ms. The default stays at 8,192 tokens. [Larger scheduler batches](raw/item7_budget_larger_bursts.json)

The device LRU charges the padded prefix and spatial metadata at insertion. Relative to compact storage, up to 63 additional prefix positions cost at most 1.97 MiB of KV per singleton, and up to 63 visual rows cost at most 0.31 MiB. The byte snapshot uses at most 8 MiB of host memory across the worker. A sequence of distinct large image resends still pays for hashing each image. [Cache accounting](../../../speed/decision_worker.py) · [Per-token state geometry](../ROOFLINE.md)

## Kernel profile and design choices

The fresh tower runs 24 blocks. Tv=128 and Tv=256 in these traces use 468 and 1,008 raw patches per image, respectively. The CUPTI measurements below use 32 and 64 images. Each time covers all invocations of one operation in the tower. Dense lower bounds divide useful FLOPs by the measured 401.73 BF16 TFLOP/s. Norm, rotary and residual bounds count minimum useful vector traffic at 1,463.2 GB/s. The attention and end-to-end floors use the existing per-stage maximum of arithmetic and bandwidth time. Individual traffic floors describe unfused operators and cannot be added to the end-to-end model. Cache reuse can bring an individual kernel below a device-memory estimate. The measured dense peak also varies by matrix shape. Kernel sums across attention streams include overlap. The joined timing measures elapsed time on the calling stream. [Hardware calibration and equations](../ROOFLINE.md) · [Before CUPTI kernel names and shapes](raw/item7_kernels.json) · [After CUPTI trace](raw/item7_final_kernels.json) · [Before caller-stream profile](raw/item7_before_profile.json) · [After caller-stream profile](raw/item7_matched_profile.json)

| ViT operation | Tv128 before | After | Floor | Tv256 before | After | Floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch convolution, layout and bias | 4.29 | 4.28 | 0.12 | 18.46 | 18.47 | 0.51 |
| QKV projection | 6.29 | 6.28 | 5.63 | 26.54 | 26.52 | 24.25 |
| Attention output projection | 2.46 | 2.47 | 1.88 | 9.02 | 9.00 | 8.08 |
| FC1 + GELU | 10.99 | 7.91 | 7.51 | 51.84 | 33.63 | 32.33 |
| FC2 | 8.55 | 8.29 | 7.51 | 31.79 | 31.60 | 32.33 |
| Merger | 0.58 | 0.58 | 0.51 | 2.32 | 2.32 | 2.19 |
| LayerNorm (both block norms and merger) | 1.33 | 1.33 | 2.05 | 8.05 | 8.01 | 8.85 |
| Residual and position adds | 1.52 | 1.52 | 3.08 | 12.76 | 12.75 | 13.27 |
| Rotary | 1.30 | 1.28 | 2.01 | 8.79 | 8.79 | 8.67 |
| Attention, kernel sum | 11.70 | 11.60 | — | 59.66 | 59.87 | — |
| Attention, joined wall | 8.80 | 8.57 | 2.01 | 54.89 | 54.85 | 15.91 |
| Whole tower, CUDA event | 45.25 | 41.90 | 25.15 | 224.37 | 206.06 | 115.60 |

| Tv / frames | Pinned H2D before / after / floor | Device patchify before / after | Host RGB pack before / after |
| --- | ---: | ---: | ---: |
| 128 / 32 | 0.22 / 0.22 / 0.20 | 0.19 / 0.03 | 2.08 / 2.05 |
| 256 / 64 | 0.91 / 0.91 / 0.87 | 1.43 / 0.15 | 8.14 / 8.07 |

**Vision MLP.** A separate BF16 GELU makes a round trip through device memory for every element of the four-times-expanded hidden vector. The SM120 Triton projection uses TMA, accumulates in FP32, adds bias and rounds to BF16 at the same boundary as the existing linear layer. It indexes a 128 KiB table containing the model's actual GELU for every BF16 bit pattern and writes the activated output directly. With 14,976 and 64,512 patch rows, the first block's projection plus activation took 0.368 and 1.444 ms, down from 0.491 and 2.184 ms. The winning tile was 128×128×32 with four warps. All five tested tiles reproduced the reference BF16 tensors at both shapes. Dispatch requires the tested BF16 weight geometry, activation family, SM120 and at least 4,096 rows in a multi-image tower. [Tile sweep](raw/item7_candidate_mlp.json) · [Implementation](../../../speed/vision_mlp.py)

**Attention.** A second design reused the exact PyTorch FlashAttention split-KV reduction and combined up to 16 images per launch. On 32 images, it matched every output element. On 468-patch images the eager first-block measurement fell from 0.923 to 0.357 ms, while on 1,008-patch images it rose from 1.309 to 1.508 ms. In the production fresh sweep, enabling the candidate at Tv=128, N=128 moved vision from 186.74 to 190.56 ms. The shipped implementation retains per-image SDPA on concurrent streams. A separate paged-varlen sweep tested padding, split count and reported key length. Its numerical differences are recorded per image. [Exact attention microbenchmark](raw/item7_verify_split.json) · [Production control](raw/item7_rgb_fresh.json) · [Batched attention trial](raw/item7_dispatch_fresh.json) · [Paged attention and patch probes](raw/item7_probe.json)

### Cost of matching the HF patch embedding

The fixed-grid tower uses the HF cuDNN 3D convolution. Ninety flattened reduction and padding layouts were tested per grid against the real projection. The best layout still disagreed on 3 and 13 entries of the 256×256 controls at the two grids. [Reduction orders](raw/item7_patch_order.json)

| Patch control | Tv | Frames | Exact tower ms | Control tower ms | Decision matches | Max typed probability difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flattened GEMM + separate bias | 128 | 32 | 41.07 | 37.03 | 128/128 | 0.01706 |
| Flattened GEMM + separate bias | 256 | 64 | 202.44 | 184.81 | 255/256 | 0.02140 |
| 2D convolution, channels last | 128 | 32 | 41.19 | 37.95 | 128/128 | 0.02860 |
| 2D convolution, channels last | 256 | 64 | 203.00 | 189.15 | 256/256 | 0.10883 |

Each row covers a box, two yes/no questions and a three-way choice. The linear projection changed one three-way choice. The 2D projection preserved the choices but crossed the existing 0.02335 limit on typed probability at both grids. The default retains the 3D convolution. [Linear control](raw/item7_checked_patch_control.json) · [2D control](raw/item7_twod_conv_control.json)

The 2D patch operator alone took 1.004 and 4.520 ms, against 4.291 and 18.442 ms for 3D. It changed 3,784 of 15,335,424 and 15,295 of 66,060,288 BF16 patch components. The channels-last 3D variant reproduced every element and gave no consistent speedup. [Convolution shape and layout controls](raw/item7_patch_patch_conv.json)

The binary-only calibration used 256 images per grid and the ten questions from the fresh sweep. Both speedups were rejected for the opt-in path under the existing continuous-output gate. [Linear binary calibration](raw/item7_binary_binary_control.json) · [2D binary calibration](raw/item7_twod_conv_binary.json)

| Patch control | Tv | Binary decisions matched | Probability outliers | Max difference |
| --- | ---: | ---: | ---: | ---: |
| Flattened GEMM | 128 | 2,560/2,560 | 16 | 0.07328 |
| Flattened GEMM | 256 | 2,560/2,560 | 2 | 0.03184 |
| 2D convolution | 128 | 2,558/2,560 | 12 | 0.04883 |
| 2D convolution | 256 | 2,560/2,560 | 5 | 0.04644 |

**Residual add and LayerNorm.** For a residual add followed by LayerNorm, reference / fused times were 0.044 / 0.094 ms, 0.449 / 0.372 ms. Both tensors matched the first-block reference exactly. The large-grid operator saves 0.077 ms per pair and the shorter grid regresses. Production retains the existing block norms. [Operator control](raw/item7_checked_norm.json)

**Pixel upload and mixed grids.** The processor's own normalization code supplies exact values for each channel and possible RGB byte. A single upload carries the packed RGB pixels. A Triton kernel uses each image's width and offset to write the processor's spatial merge order and duplicate temporal slice. It can write BF16 directly when the consuming tower will cast to BF16. The video tower keeps its existing flattened patch projection and attention convention. Fresh scan keeps HF's patch convolution and per-image attention. The same mixed-grid packer is used for eligible scheduler batches. [Preparation](../../../scan/vision_prep.py) · [Pixel kernel](../../../speed/frame_pixels.py) · [Video feature and decision comparisons](raw/item7_checked_video_checks.json)

### LM prefill and GDN

The stage profile uses one actual packed question per image, totaling 4,672 and 17,984 LM tokens. Its floors use those unpadded lengths. The fresh sweep above rotates questions of different lengths. Dense time includes unassigned work inside the packed forward.

| Tv / frames | LM stage | Before | After | Floor | Floor / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 32 | LM dense | 87.9 | 87.7 | 83.0 | 94.6% |
| 128 / 32 | GDN | 8.2 | 8.1 | 1.9 | 23.3% |
| 128 / 32 | LM attention | 1.8 | 1.8 | 0.5 | 29.2% |
| 128 / 32 | heads | 0.7 | 0.7 | 0.2 | 23.8% |
| 256 / 64 | LM dense | 346.9 | 347.1 | 319.5 | 92.1% |
| 256 / 64 | GDN | 33.2 | 33.1 | 7.2 | 21.9% |
| 256 / 64 | LM attention | 6.7 | 6.7 | 2.0 | 30.1% |
| 256 / 64 | heads | 2.0 | 2.0 | 0.6 | 33.2% |

The optional GDN kernel with a 128-wide value tile requested 112,640 bytes of dynamic shared memory per block. The installed CUDA launcher rejected it on this GPU at both sequence lengths. [Wider-tile control](raw/item7_tile_wide_gdn.json)

For the GDN core at 128 sequences of width 146, fixed-length layout reduced the one-layer median from 0.935 to 0.881 ms with a value tile of 64. A tile of 32 took 1.352 ms even in the fixed layout. For 64 sequences of width 281, the existing packed tile-64 path took 0.732 ms, versus 0.736 ms fixed and 1.102 ms packed with tile 32. Every layout and tile matched the packed output bit for bit on both shapes. The dispatcher selects fixed layout for uniform short groups of at least 32 sequences. The Tv=256 result retains packed GDN. [Layout and value-tile sweep](raw/item7_gdn.json) · [Dispatcher](../../../speed/flash_gdn.py)

## Remaining headroom

Fresh Tv=256, N=256 has 353.1 ms of vision headroom and 249.0 ms in LM/readout. 64-frame video has 92.1 ms of vision headroom and 57.8 ms in LM/readout. The 64-client fresh burst has 1,431.2 ms in vision and 1,062.1 ms in LM/readout. Absolute differences use the stage medians. [Full ceiling map](../CEILING.md)

## Equivalence and scope

- The tiny CPU suite and the calibrated real-weight GPU suite completed with `SCAN_EQUIVALENCE_OK`. On real weights, all 234 decision comparisons agreed. [CPU log](raw/item7_cpu.log) · [Real-weight log](raw/item7_final_real.log)
- The scan benchmark compares cached, fresh, separate-engine and reference outputs across 20 rows with 24 examples per row. Every decision agreed. [Per-row scan results](raw/item7_final_equiv.json)
- Against shipped BF16 LoRA, each of the five BF16 engine variants in the 930-row speed diagnostic agreed on all 930 decisions. The FP32 control and the separately labeled FP8 experiment have their own results in the diagnostic. [Diagnostic](raw/item7_checked_diag.json)
- The randomized scheduler matched isolated execution on 254/254 decisions, shipped BF16 on 254/254 and the isolated-to-shipped control on 127/127. Every continuous metric stayed inside the existing calibrated limit. Seven model-free API and scheduler tests passed, including changed pixels at an identical object address and identical bytes with different dimensions. [Interleaving comparison and limits](raw/item7_final_interleaving.json) · [API tests](raw/item7_api.log)
- Fresh normalized pixels, question layouts and full tower outputs matched the batched HF reference at Tv=128 and 256 with 1, 8, 32, 64 and 128 frames. These checks compare tensor values directly. [Ten fixed-grid checks](raw/item7_final_features.json)
- Direct BF16 pixel writes matched the reference processor followed by its BF16 cast at all ten grid and batch shapes. Replaying a single graph with the image order reversed also matched HF, at Tv=128/N=128 and Tv=256/N=64. [Device pixels and tower](raw/item7_graph_features.json) · [Forward and reversed replay](raw/item7_graph_graph_features.json)
- The public video benchmark entry point was also run with the pre-item and optimized engines on the same eight windows. [Direct end-to-end comparison](raw/item7_entry_video_entry.json)

The 32- and 64-frame video checks covered eight windows and 384 box decisions. The compact upload reproduced all FP32 processor pixels. The optimized video tower reproduced the pre-item tower features bit for bit, with zero differences in boxes, presence and IoU probabilities. Substituting the HF batched tower for the established video path changed three decisions on the same windows. The fresh and video entry points therefore retain their respective pre-item numerical contracts. [Video control](raw/item7_checked_video_checks.json) · [HF substitution trial](raw/item7_video_checks.json)

Set `GLANCE_FRAME_OPT=0` before constructing the engines to reproduce the prior frame dispatch, preparation and singleton path. The runs share the model architecture, checkpoint and calibrated tolerances. The hardware-dependent results above cover the measured G4, software build, image shapes and question protocols.

## Reproducing the measurements

Run the sync from this worktree. A new label produces independent raw files on the VM. The runner checks for GPU owners before each stage, serializes with other item chains and uploads each completed stage to the project dataset.

```sh
MONO=~/Developer/decision-vison/.claude/worktrees/engine-opt bash scan/vm/sync.sh engopt
bash scan/vm/bg.sh engopt item7_recheck 'ITEM7_LABEL=recheck bash scan/vm/frame_throughput.sh features graph_features video_entry fresh video scheduled real equiv diag interleaving'
```

The serving stage names are `fresh_vllm`, `fresh_sglang`, `video_vllm`, `video_tuned`, `video_sglang`, `scheduler_vllm` and `scheduler_sglang`. [Detached runner](../../../scan/vm/frame_throughput.sh)

## Authorship

Codex wrote the engine changes, kernels, probes, regression check, detached runner and reports, and ran the measurements and equivalence suites on the supplied engopt VM. The reviewer supplied the existing environment and earlier baselines. The attention probe uses the installed PyTorch FA2 launcher and its revision-matched parameter declaration with the upstream license. The production path has no dependency on that private launcher. [Probe source and license](../../../scan/flash_vision)
