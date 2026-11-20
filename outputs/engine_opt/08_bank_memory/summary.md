# Memory and query cost of cached image and video prefixes

The default bank packs valid BF16 K/V and spatial records, shares byte-identical leading K/V across inputs and preserves FP32 recurrent state. On the 32-prefix inventory, a Tv640 image went from 75.49 to 72.87 MiB and a four-frame v96 clip from 55.29 to 54.95 MiB. Across 14 paired GPU workloads, the median query-time change ranged from 1.74% faster to 1.37% slower. The calibrated GPU suite retained all 234 decisions. Exact exponent packing reduces the four-frame bank to 49.96 MiB/clip. This is the host-tier default and a GPU option. [Bytes](raw/item8_measured_after_bytes.json) · [Before](raw/item8_before.json) · [After](raw/item8_release_after.json) · [Real-weight suite](raw/item8_release_real.log)

## Storage and reads

The scan branch reads a recurrent state and a three-tap convolution tail at each of 24 GDN layers, and post-RoPE keys and values at each of eight attention layers. The full query API returns every head, including for a yes/no question. Box localization scores all valid FP32 center keys, then reads the winning visual vector and its center for edges and IoU. The typed head uses only the final question state and label indices. A typed-only bank omits all spatial tensors. The video service can use that contract. The mask and starting position supply attention visibility and question RoPE. The grid dimensions remain available to callers. Fresh inference rereads pixels without accessing a bank. [Branch and readouts](../../../scan/engine.py) · [Video service](../../../speed/server.py)

Each input below contributes exactly **48 MiB FP32 recurrent state and 1.125 MiB BF16 convolution tails** before state compression. Totals include metadata. The accounting uses 32 real model prefixes per format. Tv640 preserves aspect ratio. The other image formats use fixed 4:3 frames. Earlier phase-1 totals used different padding. [Original byte inventory](raw/item8_before_bytes.json) · [Packed KV and spatial inventory](raw/item8_measured_after_bytes.json)

| Prefix | Valid tokens | KV MiB before → compact | Visual MiB before → compact | Center keys MiB before → compact | Legacy MiB | Compact MiB | Exact state-packed MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tv640 | 605–643 | 22.000 → 19.496 | 3.115 → 3.027 | 1.246 → 1.211 | 75.492 | 72.869 | 67.793 |
| Tv256 | 257–257 | 8.031 → 7.911 | 1.230 → 1.230 | 0.492 → 0.492 | 58.881 | 58.762 | 53.701 |
| Tv64 | 68–68 | 2.125 → 2.005 | 0.308 → 0.308 | 0.123 → 0.123 | 51.681 | 51.561 | 46.511 |
| f4_v96 | 197–197 | 6.156 → 5.824 | 0.005 → 0.000 | 0.002 → 0.000 | 55.288 | 54.950 | 49.961 |
| f4_v48 | 117–117 | 3.656 → 3.324 | 0.005 → 0.000 | 0.002 → 0.000 | 52.788 | 52.450 | 47.464 |
| f2_v96 | 101–101 | 3.156 → 2.824 | 0.005 → 0.000 | 0.002 → 0.000 | 52.288 | 51.950 | 46.960 |

The clip bank keeps one shared, masked spatial slot for full-readout compatibility. Its per-clip contribution rounds below 0.001 MiB. Typed-only banks have no spatial slot. At one four-frame v96 clip every two seconds, 1,800 compact GPU prefixes take about 103.7 GB per video hour, versus 94.3 GB with exact state packing. Overlapping clips scale with their actual count. The compact position maps, masks, center coordinates, grid and question-start metadata together cost 11.42, 3.49 and 0.91 KiB per image at Tv640/256/64, and 0.99, 0.60 and 0.52 KiB per clip at f4_v96/f4_v48/f2_v96. Bank totals count tensor payloads and position maps. CUDA graph pools, allocator reservation and model weights are separate. A prefix view can retain its parent allocation.

The compact representation stores original BF16 K/V records and visual hidden states at valid token positions, FP32 center keys, and integer maps back to the original padded positions. Sharing a leading K/V position requires identical stored bytes in both tensors at every attention layer. The encoder examines the first 16 valid positions. In these banks it could share 11 leading clip tokens, four fixed-grid image tokens and one mixed-grid image token. The attention reduction keeps the original order and masking. Directly indexing the compact records inside image attention avoids rebuilding dense KV on every layer. Native-video attention retains the split-BMM calculation used to calibrate the original readout. [Storage and sharing](../../../scan/bank.py) · [Image attention](../../../speed/gdn_branch.py)

## Recurrent-state structure

Across 24 layers of a sampled clip and 24 layers of a sampled Tv640 image, each tested 128 × 128 first-head matrix had numerical rank 128 under an FP64 SVD. On average only 0.0042% and 0.0050% of state elements respectively survived a BF16 round trip exactly. Level-1 zlib kept 94.19% and 94.21% of the raw bytes. A sequential CPU decode plus equality check over the 24 layer buffers took about 164 ms for either prefix. This motivated on-device exponent packing. [Per-layer measurements](raw/item8_probe.json) · [Probe](../../../scan/bench_bank_memory.py)

A second exact representation stores the original sign and 23 mantissa bits. Each 128-value row stores a four-bit exponent offset when its exponent span fits 16 consecutive values. Wider rows carry a lossless exponent escape. Integer reconstruction preserves signed zeros, subnormals, infinities and NaN payloads. Across the six configurations, the recurrent state fell from 48 MiB to approximately 43 MiB per prefix. [Plane storage](../../../scan/bank_codec.py) · [Special-value and row-selection tests](../../../scan/test_bank.py) · [Exact byte inventory](raw/item8_planes_after_bytes.json)

Decoding the planes inside the image WY kernel produced correct FP32 bits on an isolated tile read, but changed some downstream results. The control found a maximum box-coordinate difference of 0.01225 and a maximum IoU-logit difference of 0.48058. Decoding the same state into a dense tensor before the original kernel gave zero differences in every output field. The implemented variant decodes one layer ahead on a CUDA stream and uses two reusable FP32 buffers. [Fused versus staged control](raw/item8_exact_diag.json) · [Decoder](../../../scan/bank_kernels.py)

## Warm GPU queries

The before and after runs used the same G4, checkpoint, 16-question warm-up and 15 timed calls per row. Times include the host answer. The serving columns use the measured BF16 full-box or one-token yes/no protocol with a warm prefix. The f4_v48 and f2_v96 serving reruns used eight timed calls after warm-up. In those four rows vLLM reported zero cached prompt tokens. SGLang reported 64–240 cached tokens per request. The two serving preprocessors also reported different prompt lengths for the same clips. [Before](raw/item8_before.json) · [Default](raw/item8_release_after.json) · [Earlier vLLM](../../scan/raw/scan-a/vllm_bf16.json) · [Corrected SGLang](../00_ceiling/raw/c0_sglang_bank.json) · [Additional vLLM](raw/item8_test_vllm.json) · [Additional SGLang](raw/item8_test_sglang.json)

| Cached workload | Before ms | After ms | Change | Floor ms | Floor / after | vLLM ms | SGLang ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tv640_N64_box | 23.65 | 23.65 | +0.0% | 17.27 | 73.0% | 899.1 | 1,556.6 |
| Tv640_N64_noul | 45.50 | 45.37 | -0.3% | 32.49 | 71.6% | 680.6 | 1,287.3 |
| Tv640_N256_box | 96.23 | 95.23 | -1.0% | 69.02 | 72.5% | 3,151.6 | 7,732.9 |
| Tv640_N256_noul | 179.83 | 176.70 | -1.7% | 129.86 | 73.5% | 2,679.9 | 5,670.7 |
| Tv256_N64_box | 22.47 | 22.49 | +0.1% | 16.72 | 74.3% | 653.1 | 592.2 |
| Tv256_N256_box | 84.55 | 84.49 | -0.1% | 66.78 | 79.0% | 2,950.1 | 2,981.4 |
| Tv64_N64_box | 21.99 | 22.02 | +0.2% | 16.43 | 74.6% | 401.3 | 446.1 |
| Tv64_N256_box | 82.51 | 82.62 | +0.1% | 65.66 | 79.5% | 1,112.6 | 1,440.6 |
| f4_v96_N64_noul | 54.98 | 55.74 | +1.4% | 31.82 | 57.1% | 342.7 | 620.7 |
| f4_v96_N256_noul | 238.68 | 241.40 | +1.1% | 127.24 | 52.7% | 1,355.1 | 2,203.7 |
| f4_v48_N64_noul | 54.46 | 55.02 | +1.0% | 31.71 | 57.6% | 236.0 | 556.5 |
| f4_v48_N256_noul | 233.26 | 234.88 | +0.7% | 126.78 | 54.0% | 907.7 | 2,002.2 |
| f2_v96_N64_noul | 54.53 | 54.86 | +0.6% | 31.68 | 57.8% | 214.2 | 479.1 |
| f2_v96_N256_noul | 233.04 | 234.28 | +0.5% | 126.69 | 54.1% | 823.3 | 1,784.3 |

### Where the query time goes

These fixed-question profiles time disjoint groups of CUDA operations on graph replay. Each pair gives measured / modeled milliseconds. The peak rates were 401.7 BF16 TFLOP/s, 67.4 FP32 TFLOP/s and 1,463.2 GB/s device copy bandwidth. The model charges dense projection FLOPs, valid prefix K/V reads and a 48 MiB state plus its convolution tails per prefix. It assumes independent reads of shared initial K/V. Clip gathers reconstruct the dense BMM operands. Their access time is measured separately, and the useful-work floor charges the required KV read once in attention. [Stage events](raw/item8_final_stages.json) · [Per-question and rotating-call floors](raw/item8_floors.json) · [Floor calculation](ceiling.py) · [G4 peak measurements](../ROOFLINE.md)

| Fixed question | Dense ms | GDN ms | Attention ms | Heads ms | KV gather and storage access ms |
|---|---:|---:|---:|---:|---:|
| Tv640_N64_box | 17.21 / 12.51 | 3.36 / 2.54 | 1.82 / 0.97 | 0.41 / 0.09 | 0.01 |
| Tv640_N64_noul | 31.56 / 27.29 | 6.69 / 2.87 | 3.49 / 1.06 | 0.43 / 0.09 | 0.01 |
| Tv640_N256_box | 55.38 / 50.04 | 16.06 / 10.15 | 14.07 / 3.90 | 1.17 / 0.25 | 0.01 |
| Tv640_N256_noul | 111.94 / 109.17 | 27.37 / 11.49 | 27.51 / 4.27 | 1.21 / 0.25 | 0.01 |
| Tv256_N64_box | 17.22 / 12.51 | 3.36 / 2.54 | 0.80 / 0.45 | 0.34 / 0.05 | 0.01 |
| Tv256_N256_box | 55.42 / 50.04 | 16.06 / 10.15 | 3.02 / 1.79 | 0.58 / 0.12 | 0.01 |
| Tv64_N64_box | 17.22 / 12.51 | 3.36 / 2.54 | 0.33 / 0.18 | 0.32 / 0.04 | 0.01 |
| Tv64_N256_box | 55.41 / 50.04 | 16.07 / 10.15 | 1.18 / 0.71 | 0.52 / 0.08 | 0.01 |
| f4_v96_N64_noul | 30.50 / 27.29 | 14.86 / 2.87 | 2.75 / 0.45 | 0.31 / 0.03 | 0.44 |
| f4_v96_N256_noul | 112.09 / 109.17 | 77.47 / 11.49 | 19.54 / 1.82 | 0.49 / 0.08 | 2.13 |
| f4_v48_N64_noul | 30.51 / 27.29 | 14.86 / 2.87 | 2.22 / 0.34 | 0.31 / 0.03 | 0.26 |
| f4_v48_N256_noul | 112.19 / 109.17 | 77.56 / 11.49 | 13.71 / 1.36 | 0.49 / 0.08 | 1.20 |
| f2_v96_N64_noul | 30.50 / 27.29 | 14.85 / 2.87 | 2.14 / 0.32 | 0.31 / 0.03 | 0.22 |
| f2_v96_N256_noul | 112.23 / 109.17 | 77.52 / 11.49 | 13.04 / 1.27 | 0.50 / 0.08 | 1.02 |

At f4_v96 with 256 clips, GDN continuation took 77.47 ms against an 11.49 ms floor, the largest stage gap. Keeping the calibrated split-BMM path costs a separate KV gather. Tv640 image attention reads the compact table in the kernel. This is also why the custom clip gather restores the original BMM operand order. [Initial profiling](../04_gdn_branch/summary.md) · [Gather and unpack kernels](../../../scan/bank_kernels.py)

## Storage precision and accuracy

The following control builds each format from the same dense prefix batch, holds the question fixed and times nine warmed GPU calls. MiB measures stored payload per prefix. A 64-image typed-only compact bank used 68.52 MiB/image. The full-readout compact bank includes the data for spatial localization. [Format timings, decoded bits and subset checks](raw/item8_final_formats.json)

| Format | Tv640 MiB/image | 64-image box ms | Clip MiB/clip | 64-clip yes/no ms |
|---|---:|---:|---:|---:|
| legacy | 75.51 | 23.31 | 55.29 | 52.86 |
| compact | 72.75 | 23.25 | 54.94 | 53.49 |
| state_packed | 67.67 | 26.87 | 49.95 | 57.48 |
| state_bf16 | 48.75 | 26.32 | 30.94 | 56.64 |
| state_fp8 | 37.12 | 29.92 | 19.32 | 60.61 |
| state_int8 | 37.12 | 29.77 | 19.32 | 60.51 |
| kv_fp8 | 63.20 | 31.49 | 52.08 | 54.75 |
| state_kv_fp8 | 27.58 | 38.71 | 16.46 | 61.99 |

The exact exponent-packed format also passed the calibrated suite on 234/234 decisions and the 480 scan decision comparisons. In the full 14-workload rotating sweep it added 5.4–17.6% to legacy GPU query time. Compact storage met the 5% latency target on every row, which determined the default. [Full exact-packed timing sweep](raw/item8_exact_after.json) · [Calibrated comparisons](raw/item8_exact_real.log) · [Scan comparisons](raw/item8_exact_equiv.json)

The BF16 state format stores unscaled BF16 and restores FP32 before the GDN calculation. The FP8 and INT8 state formats use one FP32 scale per 128-value row. The FP8 K/V format uses one FP32 scale per token, head and vector. All five approximate formats are opt-in. Recurrent-state quantization gives the largest byte saving because the initial FP32 state accounts for 48 MiB per prefix. [Storage implementation](../../../scan/bank.py)

The compact and exponent-packed banks reproduced the dense bank’s probabilities on the labeled tests. The image panel uses 60 gRefCOCO images and 455 expressions, including 146 positives. The video panel uses 64 query IDs split between calibration and evaluation, 64 positive and 128 negative scans, 167 distinct Charades videos and 5,419 scored clips from the E4 manifest. The short-readout checkpoint is the E4 configuration. Absolute probability deltas and flips use paired scores and the 0.5 threshold. The N=16 false-alarm rate is the mean over 50 query-split MaxStat fits at nominal α=0.10. The last column shows calibration refitted for each format and calibration frozen at the legacy format. [Image metrics](raw/item8_final_presence.json) · [Clip AUROC, delta quantiles and N=8/16/32 set metrics](raw/item8_retry_video.json) · [Scoring protocol](../../../scan/bench_bank_precision.py)

| Format | Image AUROC | Image flips | Image ΔP p95 / max | Clip AUROC | Clip flips | Clip ΔP p95 / max | N16 set FA refit / fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 0.96972 | 0/455 | 0.00000 / 0.00000 | 0.82024 | 0/5419 | 0.00000 / 0.00000 | 8.72% / 8.72% |
| compact | 0.96972 | 0/455 | 0.00000 / 0.00000 | 0.82024 | 0/5419 | 0.00000 / 0.00000 | 8.72% / 8.72% |
| state_packed | 0.96972 | 0/455 | 0.00000 / 0.00000 | 0.82024 | 0/5419 | 0.00000 / 0.00000 | 8.72% / 8.72% |
| state_bf16 | 0.96941 | 0/455 | 0.00556 / 0.01755 | 0.82021 | 3/5419 | 0.00203 / 0.00706 | 8.72% / 8.72% |
| state_fp8 | 0.96941 | 0/455 | 0.00727 / 0.01569 | 0.82035 | 2/5419 | 0.00242 / 0.00692 | 8.72% / 8.72% |
| state_int8 | 0.96868 | 1/455 | 0.02659 / 0.06804 | 0.81992 | 5/5419 | 0.00632 / 0.02480 | 8.72% / 8.75% |
| kv_fp8 | 0.96948 | 0/455 | 0.00754 / 0.01448 | 0.82016 | 1/5419 | 0.00328 / 0.01223 | 8.68% / 8.72% |
| state_kv_fp8 | 0.96954 | 0/455 | 0.00789 / 0.02466 | 0.82023 | 1/5419 | 0.00365 / 0.01449 | 8.72% / 8.72% |

On the real-weight equivalence fixtures, two box expressions and one typed question were evaluated on each group of 24 images, and one typed question on 24 native clips. Each probability column gives the largest absolute probability difference from the dense bank in that group. For compact and exact exponent-packed storage all five output tensors were also bit-identical on every one of these fixtures. The raw file includes each question’s mean, median, p95, p99, maximum, and agreement with both the dense bank and the shipped single-input model. [Paired fixture distributions](raw/item8_final_fixtures.json)

| Format | Tv640 max | Tv256 max | Clip max | Agree with dense | Agree with shipped |
|---|---:|---:|---:|---:|---:|
| compact | 0.00000 | 0.00000 | 0.00000 | 168/168 | 168/168 |
| state_packed | 0.00000 | 0.00000 | 0.00000 | 168/168 | 168/168 |
| state_bf16 | 0.00851 | 0.01030 | 0.00414 | 168/168 | 168/168 |
| state_fp8 | 0.00972 | 0.01118 | 0.00469 | 168/168 | 168/168 |
| state_int8 | 0.02095 | 0.02324 | 0.02043 | 167/168 | 167/168 |
| kv_fp8 | 0.01700 | 0.01128 | 0.00921 | 168/168 | 168/168 |
| state_kv_fp8 | 0.01340 | 0.01265 | 0.00778 | 168/168 | 168/168 |

## Banks in pinned host memory

The host tier writes one contiguous pinned allocation per encoded chunk. On a query, one copy stream stages the next chunk while the compute stream reads the previous one. Events protect each of two reusable device slots. The last partial chunk retains its own layout. The capacity benchmark built each 64-clip chunk with two 32-prefix prefills and varied the four-frame tuple for every physically stored clip. Every reported size passed exact equality on all five output tensors against queries on the same resident GPU chunks. [Host bank implementation](../../../scan/host_bank.py) · [Capacity and transfer benchmark](../../../scan/bench_bank_tier.py) · [Public default and explicit host formats](raw/item8_shipping_entry.json)

Each pair gives compact / exact exponent-packed. The device-resident column uses a single compact bank and one GPU query. PCIe floors use the measured pinned H2D rate of 56.87 GB/s. [Final compact transfers](raw/item8_coalesced_tier.json) · [Final exponent-packed transfers](raw/item8_coalesced_tier_packed.json) · [GPU-resident comparison](raw/item8_exact_resident.json)

| Clips | Transferred GB | PCIe floor ms | Copy only ms | Host query ms | Floor / query | GPU resident ms |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 3.69 / 3.35 | 64.8 / 59.0 | 64.8 / 59.1 | 118.2 / 116.3 | 54.8% / 50.7% | 53.4 |
| 256 | 14.75 / 13.41 | 259.4 / 235.8 | 259.2 / 236.8 | 312.6 / 296.3 | 83.0% / 79.6% | 234.5 |
| 1,024 | 59.00 / 53.64 | 1,037.4 / 943.2 | 1,036.9 / 946.6 | 1,090.4 / 1,010.5 | 95.1% / 93.3% | 963.0 |
| 1,920 | 110.62 / 100.57 | 1,945.2 / 1,768.5 | 1,944.2 / 1,767.7 | 1,997.7 / 1,845.3 | 97.4% / 95.8% | — |
| 10,000 | 576.13 / 523.81 | 10,131.3 / 9,211.2 | 10,125.7 / 9,206.1 | 10,181.9 / 9,333.2 | 99.5% / 98.7% | — |

The 1,920 distinct records occupied 110.62 GB in compact form, exceeding the device's 101.97 GB of physical memory. Exact state packing reduced that backing to 100.57 GB. The two GPU staging buffers occupied 7.37 GB compact or 6.71 GB exact-packed. A 10,000-entry query traversed the 1,920-clip backing repeatedly and encoded a separate partial tail. The transfer and time are measured for the entire scan. Physical capacity was demonstrated for 1,920 distinct clips.

With the exact-packed host default, 1,024 clips took 1,010.5 ms against 963.0 ms for one compact resident query, a 4.9% difference. The additional eager resident control kept 1,024 distinct clips on device and queried 64 at a time in 902.9 ms. At 10,000 logical entries, exact-packed host execution reached 98.7% of the measured PCIe ceiling. [Resident batch](raw/item8_exact_resident.json) · [Resident eager chunks](raw/item8_final_resident_chunks.json)

The original variable-table implementation submitted a separate H2D copy for each tensor whenever exception counts differed. At 1,920 exact-packed clips it took 3,441.4 ms. Moving the sparse exception buffers after the fixed fields and coalescing adjacent byte ranges reduced that to 1,845.3 ms. Copy-only times were 1,767.6 and 1,767.7 ms. This measured overlap, together with the lower payload, motivated exact state packing as the host default. The long-scan timing covers native clips. The typed host API and changing image layouts were checked separately for exactness. [Original packed host control](raw/item8_exact_tier_packed.json) · [Coalesced packed host scan](raw/item8_coalesced_tier_packed.json) · [Initial compact host control](raw/item8_stream_tier.json)

## How many questions repay bank construction

For a warmed branch, the crossing is E/(F−C), where E is the one-time encode and F and C are the fresh and cached cost per question. Each experiment queries 64 fixed inputs. The capture-inclusive experiment starts a new bank, issues four changing questions for f4_v96, Tv640 and Tv256 and a full 16-question cycle for Tv64, and times the whole sequence against fresh calls on the same inputs. It warms the engine before each recorded series. The reported first win uses medians of cumulative times from three series. It includes bank-local graph capture. [Warm cost samples](raw/item8_stream_breakeven.json) · [Changing questions on a new bank](raw/item8_release_lifecycle.json) · [Extended cycle on small frames](raw/item8_extended_lifecycle.json)

| 64 prefixes | Encode ms | Warm cached ms | Fresh ms | Warm ratio E/(F−C) | First win with capture | 2 questions cached / fresh ms |
|---|---:|---:|---:|---:|---:|---:|
| f4_v96 | 894.9 | 53.7 | 782.5 | 1.23 | 2 | 1,271.5 / 1,572.3 |
| Tv256 | 766.0 | 41.0 | 618.8 | 1.33 | 2 | 1,065.6 / 1,248.9 |
| Tv640 | 2034.0 | 43.2 | 1906.1 | 1.09 | 2 | 2,372.8 / 3,805.3 |
| Tv64 | 256.4 | 40.3 | 175.9 | 1.89 | 5 | 538.9 / 358.8 |

Tv64 reached the crossing on question five at 880.9 ms cumulative with the bank and 907.5 ms fresh. Through 16 questions the totals were 1,640.4 and 2,892.1 ms. [Full question sequence](raw/item8_extended_lifecycle.json)

## Equivalence and reproduction

The tiny FP32 CPU suite passed, including the HTTP clip-scoring check. The model-free tests cover ragged storage, repeated selection, and lossless reconstruction of arbitrary FP32 bits. Five service tests passed. The calibrated real-weight suite agreed on 234/234 decisions, and all 20 comparisons in the 24-input scan benchmark had 100% decision agreement. The scheduler agreed with the shipped path on 254/254 interleaved decisions and with isolated execution on 254/254. The isolated control agreed on 127/127. The BF16 diagnostic’s plain, taps, vision, full and fused engine paths each agreed with shipped BF16 on 930/930 rows. Its independent FP32 control agreed on 928/930. The existing tolerances were used throughout. [Tiny CPU](raw/item8_shipping_cpu.log) · [Storage tests](raw/item8_codec_cpu.log) · [Service tests](raw/item8_server_cpu.log) · [Real-weight suite](raw/item8_shipping_real.log) · [Scan equivalence](raw/item8_shipping_equiv.json) · [Interleaving](raw/item8_release_interleaving.json) · [930-row diagnostic](raw/item8_release_diag.json)

The explicit GPU reference is `eng.encode(units, bank_format="legacy")`. A per-call `bank_format` overrides `GLANCE_BANK_FORMAT`. The default `eng.encode(units)` supports all output heads. Use `readout="typed"` for a bank that returns only typed logits, and `tier="cpu", chunk=64` to stream from pinned memory. The CPU tier defaults to exact exponent-packed state. Pass `bank_format="compact"` to retain the uncompressed FP32 state on the host. `eng.query(bank, question)` works for both tiers. [Public engine API](../../../scan/engine.py)

On engopt, run the detached driver with `before_bytes after_bytes probe formats entry real equiv interleaving diag stages before after breakeven lifecycle fixtures presence video resident tier tier_packed vllm sglang` as stage arguments. The driver serializes runs under the shared GPU lock and checks for an empty compute-process list before launch. The logs contain commands and source hashes. To reproduce the historical baseline, run this driver and its accounting script with the starting engine at `a5d7087` for `before` and `before_bytes`. The raw directory also preserves superseded prototypes, including the rejected in-kernel decoder. The first video run used an incorrect manifest repository and ended before scoring. The completed rerun used the E4 manifest in the scan-data dataset. [GPU driver](../../../scan/vm/bank_memory.sh) · [Raw results](raw/)

## Authorship

Codex wrote the bank representations, kernels, host tier, engine integration, tests, measurement drivers and this report. The reviewer provided the prepared engopt VM and prior benchmark infrastructure.
