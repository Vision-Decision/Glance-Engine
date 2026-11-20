# G4 inference roofline

The latency model sums the time for host-to-device transfer, vision dense projections, vision attention, LM dense projections, the gated delta rule, LM attention, FP32 readout and one GPU dispatch. Within each compute stage it takes the larger of useful FLOPs / measured arithmetic rate and compulsory bytes / measured device-memory bandwidth. Rates use decimal GB and TFLOP. Each multiply-add counts as two FLOPs. Percent of ceiling is 100 × modeled minimum latency / measured latency. [Calculation](../../speed/ceiling_model.py) · [Per-call shapes](00_ceiling/raw/c0_shapes.json) · [Stage floors](00_ceiling/raw/c0_floors.json)

## Achievable hardware rates

RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.82.07, PyTorch 2.14.1+cu130. Rates below are medians of nine series after warm-up on the idle GPU. GEMMs use CUDA events and rotate at least 256 MiB of weights, including for skinny shapes. No operand quantization is inside the FP8 timing. [Raw samples and model config](00_ceiling/raw/c0_peaks.json) · [Invocation and device check](00_ceiling/raw/c0_peaks.log) · [Microbenchmark](../../speed/roofline.py)

| GEMM M × K × N | BF16 TFLOP/s | E4M3 → BF16 TFLOP/s | FP32 TFLOP/s |
|---|---:|---:|---:|
| 1 × 2,560 × 9,216 | 1.26 | — | — |
| 16 × 2,560 × 9,216 | 20.08 | 37.62 | — |
| 32 × 2,560 × 9,216 | 40.31 | 74.58 | — |
| 64 × 2,560 × 9,216 | 80.52 | 142.68 | — |
| 256 × 2,560 × 9,216 | 263.12 | 438.3 | — |
| 1,024 × 2,560 × 9,216 | 303.83 | 554.94 | 49.0 |
| 4,096 × 2,560 × 9,216 | 339.88 | 621.38 | — |
| 8,192 × 8,192 × 8,192 | 401.73 | 831.88 | 67.4 |

The square BF16 result gives a compute/bandwidth crossover of **275 FLOPs/byte**. A one-row BF16 projection streams roughly one byte per FLOP, which explains the 1.26 TFLOP/s skinny result. The 4,096-row MLP-shaped GEMM reaches 85% of the measured square peak.

| Operation | Achieved | Timing boundary |
|---|---:|---|
| Device memory copy | 1,463.2 GB/s | 256 MiB read + 256 MiB written per iteration |
| Pinned H2D | 56.9 GB/s | 256 MiB payload, host submission through completion |
| Pageable H2D | 27.8 GB/s | Same payload and boundary |
| One eager one-element add | 3.24 µs | 2,000 queued submissions, amortized wall time |
| One single-kernel graph replay | 2.23 µs | 2,000 queued submissions, amortized wall time |
| Eager add with completion fence | 13.08 µs | One submission per sample, stream-fenced at the end |
| Graph replay with completion fence | 11.54 µs | One submission per sample, stream-fenced at the end |

The 1,463 GB/s figure measures device DRAM copy traffic, counting reads and writes. The transfer numbers count host payload once. A single launch floor is small next to one 7.14 GB weight pass (4.88 ms at the copy rate). Hundreds of launches and synchronizations can still matter.

## Language model

Let P be the number of valid prefix tokens for one image and q the valid length of an independent question block. A cold call evaluates P + Σq tokens per image. A warm call evaluates Σq. Different images have independent states. The BF16 backbone has width 2,560 and 32 layers. The dense weight counts below exclude the embedding vocabulary, normalization, biases and the depthwise convolution. The engine gathers only required embedding rows and returns decision heads directly. [Model dimensions](00_ceiling/raw/c0_peaks.json) · [Reference backbone](../../dv/model/backbone.py)

| Component | Weights across all layers | Dense FLOPs per evaluated token |
|---|---:|---:|
| SwiGLU gate, up and down, 32 × 3 × 2,560 × 9,216 | 2,264,924,160 | 4,529,848,320 |
| GDN projections, 24 × 2,560 × (2×2,048 + 3×4,096 + 64) | 1,010,565,120 | 2,021,130,240 |
| Gated GQA projections, 8 × 2,560 × (3×4,096 + 2×1,024) | 293,601,280 | 587,202,560 |
| Total | **3,569,090,560** | **7,138,181,120** |

One BF16 pass over those dense weights costs 7.138 GB per call, even for a one-question follow-up. The dense-stage byte model adds 2 × 32 × 2,560 BF16 elements per evaluated token for layer inputs and outputs. Intermediate projections, gating, permutations, normalization, padding, re-reading weights and graph capture can increase traffic or runtime. The model allows weights to be shared across every row in a burst.

**Gated delta rule.** Each GDN layer uses 16 key heads, 32 value heads, key/value dimension 128, 8,192 depthwise convolution channels and four taps. For T tokens the three state contractions (predict from key, rank-one update, read with query) cost approximately 6T × 32 × 128² FLOPs per layer, plus 2T × 4 × 8,192 for convolution. Gating, normalization and pointwise arithmetic are omitted from this optimistic contraction count. Per-token mixer traffic counts BF16 Q, K, V and output once, or 2 × (2×2,048 + 2×4,096) bytes per layer. A cold packed call discards the final state. A warm call must read at least one copy per distinct prefix of the FP32 recurrent state and BF16 convolution tail. Across 24 layers these occupy **48 MiB + 1.125 MiB** per prefix. Several questions on one image can share that read. Independent images cannot. [Branch implementation](../../scan/engine.py) · [Packed implementation](../../scan/varlen.py)

For a chunk of c valid tokens, per value head, the state contractions cost about 6c·k·v, where k=v=128. Building the key/key and query/key products costs up to 4c²k dense FLOPs. Solving a unit lower triangular system against k+v columns costs approximately c²(k+v), and applying intra-chunk attention to V costs up to 2c²v. An implementation that explicitly forms the triangular inverse can also spend O(c³) work per chunk. Sum over chunks of each sequence separately. With bounded chunk width this work is linear in total sequence length. Writing a chunk boundary state to device memory and reading it back costs an additional 2 × 4 × 32 × k × v bytes per boundary per layer for an FP32 state. Materialized BF16 chunk transforms add O(32c²) elements per chunk, plus O(32c(k+v)) for transformed vectors. Actual FLA traffic depends on tiling, intermediate dtype, recomputation and cache residency. The map uses the smaller contraction workload attainable by a fused implementation and does not charge these implementation-specific intermediates.

**Gated grouped-query attention.** Each of eight layers has 16 query heads and four KV heads of width 256. Per layer, score and value products cost 4 × 4,096 × A FLOPs. For a block-bidirectional question on a causal prefix, A = P(P+1)/2 + Σ q(P+q). On a cache hit omit the prefix triangle. The minimum Q/K/V/output stream costs 2 × (2×4,096 + 2×1,024) bytes per new token per layer. Reading stored BF16 KV costs another 8 × 2 × 2 × 1,024 = **32 KiB per cached prefix token** across the model. This assumes one KV read per distinct prefix. The score matrix can stay on chip. [Attention masks](../../dv/model/backbone.py) · [Varlen attention](../../scan/varlen.py)

## Vision, transfer and heads

A spatially merged visual token represents four raw patches. Each raw patch contains 2 × 16 × 16 × 3 = 1,536 pixel components. The C0 host processors supply FP32 tensors, so that H2D boundary costs 6,144 bytes per patch. The floor uses pinned bandwidth and allows faster host collation. Host JPEG decode and resize are outside all timings. The host prep time inside the existing benchmark counts toward the measured gap.

Item 2 changes the fixed-grid fresh RGB input to 3 × 16 × 16 = 768 host bytes per patch. GPU preprocessing expands the repeated temporal slice. At 128 frames and Tv=128 this reduces the pinned transfer floor from 6.47 to 0.81 ms and the serial call floor from 450.6 to 444.9 ms. The single-image speed path retains the original transfer representation. [Item-2 floors](02_vision_prep/raw/item2_floors.json) · [Fresh-frame measurements](02_vision_prep/summary.md)

For L raw patches in one spatial group the 24-block vision tower performs **620,756,992 dense FLOPs per patch**, including patch embedding and one quarter of the merger per patch. The dense coefficient is 2 × [1,536×1,024 + 24(4×1,024² + 2×1,024×4,096) + (4,096² + 4,096×2,560)/4]. Full spatial attention adds 24 × 4 × 1,024 × L² FLOPs per group. Sum L² over images or temporal groups. Attention traffic counts four BF16 vectors of width 1,024 per patch per layer. Dense traffic includes one BF16 weight pass plus one read and one write of the hidden vector per patch per layer. Four-times-longer attention on one image has sixteen times the score/value arithmetic. [FastVision](../../speed/engine.py)

The FP32 heads use D=2,560, hidden width h=512 and 64 bins per box edge. Per question, presence, query-to-center, edge, IoU and typed projections touch R = (D·h+h) + D·h + (2D+64)·3h + 2h·256 + h + D·(30+28) weights. The two typed terms are the copied vocabulary rows and learned group residual. Their cost is 2R FLOPs per question plus 2h per scored visual token. Computing center keys adds 2D·h FLOPs per visual token once per image on the speed and fresh paths. Cached scan already owns those keys and reads 4h bytes per visual token. The model charges a single FP32 pass over the head weights and the required visual features or cached keys. A clip bank has one masked spatial slot. Both head families execute in these entry points. [Box heads](../../dv/model/heads.py) · [Typed head](../../dv/model/typed.py) · [Cached branch heads](../../scan/engine.py)

## Interpreting the gap

The model uses unpadded, tokenized examples and per-example image sizes. It calculates each call before taking medians. It assumes ideal batching and data reuse across a client burst, and uses one minimum replay dispatch per call. Measured stage medians can therefore differ from their sum. Current-padding overhead, graph traffic, additional kernel launches, CPU work and implementation-specific materialization remain visible as headroom. Small-M GEMM rates characterize how much of the memory-limited region a real operator can reach. The map uses the maximum measured dense rate and the separate byte bound for every shape. These are empirical engineering targets for the current numerical format and stage boundaries. Kernel fusion, a different host representation or overlapping stages can change the model.

At the 128-frame, Tv=128 point the C0 model assigns 333.2 ms to LM dense work, 7.6 ms to GDN, 2.1 ms to LM attention and 0.6 ms to FP32 heads. Vision contributes 92.6 ms dense and 8.1 ms attention, with 6.5 ms for pixel transfer. These sum to the 450.6 ms call floor. At 4k GUI resolution and one question, full vision attention alone accounts for 63.1 of 165.8 ms. The fresh-frame and screenshot attention terms depend on the sum of squared lengths within each image. [Per-stage floors](00_ceiling/raw/c0_floors.json)

**C0 sensitivity to GEMM shape.** Replacing 401.73 TFLOP/s with the measured 339.88 TFLOP/s MLP-shaped rate for all BF16 arithmetic raises the fresh Tv=128, N=128 floor from 450.6 to 528.1 ms and the Tv=256, N=256 floor from 1,810.4 to 2,127.6 ms. The single cached-image follow-up stays at 4.98 ms because bandwidth dominates. These are alternate model evaluations of the same calls. [Measured rates](00_ceiling/raw/c0_peaks.json) · [Workload shapes](00_ceiling/raw/c0_shapes.json)

Item 7 extends compact RGB transfer to the mixed-grid, one-box-per-frame video workload. For the 64-frame median, H2D falls from 6.916 to 0.864 ms at the existing measured host-link rate, including a 12-byte layout descriptor per image, giving a 429.731 ms total floor. The corresponding 32-frame floor is 214.7 ms. The fresh and concurrent workload shapes already used compact RGB in the model. Per-operator floors and CUPTI traces for the fresh tower are in the frame report. [Mixed-grid floor inputs](00_ceiling/raw/c0_floors.json) · [Frame kernel profile](07_frames/summary.md#kernel-profile-and-design-choices)

## FP64 center selection

Item 9 measured 1.58 FP64 TFLOP/s on a 4,096³ GEMM. The BF16-backbone engine still uses FP32 weights and projections in its box head. For a dot product of width 512, charge 1,024 FP64 FLOPs and four bytes per distinct projected key element. Scoring 256 independent 640-patch images has a 0.106 ms arithmetic floor and a 0.229 ms key-read floor at the measured 1,463.2 GB/s. Reuse across questions reduces the byte term for a shared image. The FP64 probe used a warm pair of FP64 matrices and 12 CUDA event samples. [FP64 measurements and score-kernel controls](09_numerics/raw/item9_tiled_micro.json)
