# Inference engine research log

## Strongest findings

1. **A hybrid prefix has two different reuse costs.** Thirty-two questions on one image can share the same recurrent state and attention KV. At 704 prefix slots, the reference implementation created 1536 MiB of recurrent-state copies and 704 MiB of KV gather destinations across layers. A batch of 256 independent images still needs 12 GiB of recurrent state. Reading each bank entry directly and packing the question work took a 256-image yes/no query from 300.69 to 177.43 ms. [Finding on repeated and independent prefixes](#forking-one-image-for-many-questions-and-querying-many-independent-images) · [Copy measurements](04_gdn_branch/raw/item4_staged_micro.json) · [Reference timing](04_gdn_branch/raw/item4_reference_cached.json) · [Optimized timing](04_gdn_branch/raw/item4_checked_cached.json)
2. **The number of questions determines when a bank repays its construction cost.** On a newly encoded bank of 64 inputs, with changing questions and graph capture charged, cumulative cached time first beat repeated fresh scans on question two for four-frame v96 clips, Tv640 images and Tv256 images. For the smaller Tv64 images the first saving came on question five. Warm query medians alone predicted question two for all four shapes. [Finding on bank construction](#cost-of-the-first-questions-on-a-new-bank) · [Warm timings](08_bank_memory/raw/item8_stream_breakeven.json) · [New banks](08_bank_memory/raw/item8_release_lifecycle.json) · [Tv64 question sequence](08_bank_memory/raw/item8_extended_lifecycle.json)
3. **Batch size changes when a client receives its decision.** One forward completes every admitted answer. For 64 simultaneous cold box requests on the standard BF16 stack, doubling the admission budget from 8192 to 16384 tokens kept throughput at 41.4 requests/s. Median client latency rose from 883.6 to 1207.7 ms while the last completion moved from 1550.1 to 1545.3 ms. [Finding on decision batches](#a-decision-batch-finishes-its-answers-together) · [8192-token run](06_scheduler/raw/item6_scheduled.json) · [16384-token run](06_scheduler/raw/item6_tune16384.json)
4. **Exactness needs separate checks for the label, selected patch and regressed box.** The fixed-input head reproduced all 210 spatial readouts bit for bit. On the 60-image end-to-end diagnostic, fused BF16 agreed with shipped BF16 on 930/930 presence or typed decisions and 472/570 patch centers. Injecting a 0.04 coordinate error left all 234 decisions in the real-weight suite intact and tripped the calibrated 0.03632 same-center gate. On a separate fresh Tv640 panel, both legacy and production heads crossed the presence threshold on the same one of 192 comparisons with the singleton reference. [Finding on layout and box readout](#where-layout-changes-reach-the-box-readout) · [Frozen-state replay](09_numerics/raw/item9_strict_replay.json) · [Decision and center tallies](09_numerics/raw/item9_release_diag.json) · [Injected coordinate fault](09_numerics/raw/item9_gates_small_fault.log) · [Paired threshold crossing](09_numerics/raw/item9_release_presence_audit.json)

## Planned ablations

- **Decision head versus token readout.** Hold visual inputs, backbone and prefix policy fixed. Compare a direct label or box with a token readout that returns the same answer type. Report latency to the complete answer alongside label agreement and box accuracy.
- **State reuse.** With the same head, compare fresh prefill, a retained recurrent-state and KV bank with reference continuation, and the same bank with branch kernels. Vary questions per prefix and independent prefixes per batch. Charge encoding, storage and first-use capture when reporting the break-even point.
- **Runtime implementation.** Hold the head and bank policy fixed. Toggle fused kernels, graph replay and cross-request scheduling separately. Report stage time, burst throughput, per-client latency and the same discrete and continuous equivalence gates for each toggle.
- **Public-checkpoint replication.** Port this comparison to a public hybrid checkpoint, for example a Qwen3-Next model, and repeat the reuse crossover and readout gates. The measurements below use the project's Glance checkpoint.

## Known limit: the bf16 backbone, not the readout, decides near-tie box centres

**Claim**. After making the box readout deterministic, the remaining layout dependence of box outputs comes entirely from bf16 rounding in the backbone's hidden states; an fp32 backbone removes it at a large cost.

**Numbers**. Fixed hidden states: centre scores, selected patches, boxes and IoU logits bit-identical on 210 real-weight replays. End to end in bf16: 28/420 centres switch across five equivalent layouts, coordinate spread 0.21247. fp32 backbone control: 0/420 switches, spread 0.000357, 3.12x latency. Presence/typed decisions 930/930. [Item 9 summary](09_numerics/summary.md)

**Generality**. Any argmax-over-patches localisation head on a bf16 transformer with near-tie patch logits; margins here are <= ~0.06 logit.

## Fresh prefill at 8k and 16k nominal visual tokens

**Claim**. Under the scan benchmark's CUDA graph policy, a change in the fraction of timed calls that capture a graph accounts for the jump in median LM latency.

**Numbers**. At Tv=128 the original LM median rose from 334 to 2,212 ms for 64 to 128 frames. Isolated eager execution rose from 330 to 676 ms. Four of ten timed 64-frame calls captured a graph. At 128 frames seven of ten captured one, spending 1,534–1,805 ms inside the capture helper, including its two warm-ups. The other three took 671, 682 and 671 ms for the whole LM stage. The number of live LM graphs fell from three to one twice when the next shape triggered eviction. [Reference benchmark](01_varlen_prefill/raw/item1_before_initial.json) · [Capture timeline and eager samples](01_varlen_prefill/raw/item1_profile_reference.json)

**Generality**. The median changes abruptly when more than half the requests pay a setup cost. The crossover depends on GPU capacity, question-length reuse, graph outputs and eviction policy. It was measured on one G4 with ten rotating questions.

## Prefix state that outlives a single decision

**Claim**. Returning a reusable prefix state from a one-question fresh call keeps considerably more storage live than the readout needs.

**Numbers**. At 4k, 8k, 16k and 32k nominal visual tokens, unique backing storages referenced by the old cache totalled 4.85, 9.70, 19.41 and 38.82 GiB. Convolution tails and attention V are views into larger projection buffers. At 16k, omitting the bank and continuing the question in its image's packed sequence reduced isolated peak allocation from 32.3 to 11.9 GiB. Kernel time classified as data movement fell from 147.2 to 74.3 ms, while GEMM time was 328.1 and 328.9 ms. [Reference storage and kernels](01_varlen_prefill/raw/item1_profile_reference.json) · [Packed storage and kernels](01_varlen_prefill/raw/item1_profile_varlen.json)

**Generality**. These measurements apply to the Qwen3.5-4B hybrid GDN/attention implementation with fused projections. Repeated questions can reuse a bank. Here each image receives one new question and the answer is returned immediately.

## End-to-end headroom after packing

**Claim**. Removing repeated graph capture recovers the original 4k-chunk throughput at large N. A separate gap to the serving engines persists.

**Numbers**. For 128 frames at Tv=128, the clean initial packed run took 944 ms against 638 ms for vLLM and 823 ms for SGLang. The coordinator's best legacy 4k chunks took 951 ms. Of the packed 944 ms, host preparation took 93 ms, vision 264 ms and LM/readout 587 ms. An isolated LM trace assigned 329 ms to GEMMs. Holding preparation, vision and GEMM time fixed would use about 686 ms before accounting for the rest of the LM. A second packed run reached 931 ms, with 81 ms in prep, 264 ms in vision and 586 ms in LM/readout. [Initial packed timings](01_varlen_prefill/raw/item1_after_initial.json) · [Packed rerun](01_varlen_prefill/raw/item1_after.json) · [vLLM](01_varlen_prefill/raw/item1_vllm.json) · [SGLang](01_varlen_prefill/raw/item1_sglang.json) · [Packed trace](01_varlen_prefill/raw/item1_profile_varlen.json)

**Generality**. The comparison includes image processing and the vision tower on one G4. The serving runs expose only end-to-end latency. Their individual stage gaps cannot be inferred from these data.

## Using identical vision features to assess LM equivalence

**Claim**. Vision batch shape can dominate measured probability drift between two BF16 execution layouts even when their decision and weights agree.

**Numbers**. On 24 COCO frames at Tv=256, the largest packed-versus-separate ΔP for “the dog on the left” was 0.08213 end-to-end. Passing both LMs the same batched vision features reduced it to 0.00140. Packed and legacy batched LMs differed by 0.00014 on those features. Across 144 decisions, the largest fixed-feature packed-versus-separate ΔP was 0.00558, with 100% agreement. [Fixed-feature control](01_varlen_prefill/raw/item1_fixed_vision.json) · [Source](01_varlen_prefill/raw/fixed_vision.py)

**Generality**. This control separates the vision and LM contributions for two resolutions and three questions on the deployed checkpoint. It does not estimate a population-wide error bound.

## A BF16 reference can cross its localization tolerance under padding alone

**Claim**. The discrete patch-selection step makes a raw-coordinate tolerance sensitive to BF16 layout changes on nearly tied centers.

**Numbers**. Running the shipped per-image reference twice with exactly the same images, model and causal question, and changing only the left-padding bucket from 64 to 1, switched the fourth fixture's center from patch 122 to 161. Its reference top-two score margin was 0.00139. Across six fixtures, the reference-to-reference normalized box-coordinate difference was 0.0545 absolute and 0.0670 relative. The relative tolerance in the scan suite is 0.02. [Coordinates, predicted patch and margin](01_varlen_prefill/raw/item1_reference_layout.json) · [Control script](01_varlen_prefill/raw/reference_layout.py)

**Generality**. This was observed on the scan suite's seeded noise images with a trained hard-argmax box head. The frequency on natural images remains unmeasured by this control.

## Useful work in the fresh 128-frame scan

**Claim**. At 128 frames and a visual budget of 128, improving only the LM or only the vision implementation to its modeled floor still leaves the complete scan above the clean vLLM time.

**Numbers**. Glance took 931.1 ms and vLLM took 638.3 ms. The Glance medians were 81.0 ms host prep, 264.0 ms vision including transfer, and 586.3 ms LM/readout. The full stage floor was 450.6 ms. Even an LM/readout at its 343.5 ms floor leaves approximately 689 ms with the other stages fixed. Removing all 264 ms of measured vision still leaves approximately 667 ms. The latter calculation deliberately grants a zero-cost vision tower. [Item 1 timings](01_varlen_prefill/raw/item1_after.json) · [vLLM](01_varlen_prefill/raw/item1_vllm.json) · [C0 stage floors](00_ceiling/raw/c0_floors.json)

**Generality**. These counterfactuals add independent medians and assume serial stages. Host/GPU overlap, input representation and precision can change the attainable boundary.

## Cached image queries retain a large fixed state cost

**Claim**. A tenfold decrease in the visual-token budget produces a modest latency reduction for a cached box query. Every cached prefix still supplies a 48 MiB FP32 GDN state and a 1.125 MiB convolution tail across 24 layers.

**Numbers**. At N=256, decreasing the budget from 640 to 64 cut resident bank storage from 75.50 to 51.68 MiB per image and latency from 139.9 to 120.4 ms. The recurrent states alone occupy 12 GiB for 256 images. For one fixed yes/no question at Tv=640 and N=256, the warmed GPU trace attributed 122.5 ms to GEMM kernels, 67.1 ms to copy/concatenation kernels and 38.6 ms to the fused recurrent kernel. The corresponding box trace gave 65.5, 27.8 and 20.7 ms. [Bank and latency](00_ceiling/raw/c0_cached.json) · [Kernel names and timings](00_ceiling/raw/c0_profile_bank.json)

**Generality**. State size follows the hybrid model's fixed GDN head geometry. Latencies depend on question length, kernel layout and hardware. Reducing the number of pixels can reduce KV storage and vision work while leaving recurrent state geometry intact.

## A cold serving burst can look faster than its modeled floor

**Claim**. A failed asynchronous cache reset changed the ranking of the two generative servers on the 64-client cold-image workload.

**Numbers**. The SGLang wrapper caught an exception from a synchronous reset inside its running event loop. The original 64-client burst reused 663.0 prompt tokens per request and took 953.6 ms. Awaiting the reset and checking its result yielded zero cached tokens and 2,301.9 ms. vLLM reported zero cached tokens and 1,835.9 ms. Glance's serial worker took 2,698.7 ms. [Original log](00_ceiling/raw/c0_clients_sglang.log) · [Corrected SGLang](00_ceiling/raw/c0_sglang_clean.json) · [vLLM](00_ceiling/raw/c0_clients_vllm.json) · [Glance](00_ceiling/raw/c0_clients.json)

**Generality**. The observed 2.41× difference depends on the repeated image pool, the particular cache implementation and the timing boundary. Distinct requests containing the same image within one batch can still share a prefix after a successful reset.

## Vision rotary embedding and BF16 decision equivalence

**Claim**. At a fixed visual-token budget, the eager Qwen vision rotary operations can cost more than the attention score and value products. Fusing the two rotations can preserve their FP32 intermediate rounding.

**Numbers**. At Tv=128 over 128 frames, a warmed eager tower spent 106.5 ms in rotary operations and 10.5 ms in batched SDPA. Fused Q/K rotation took 8.2 ms on the same workload. The first-block Q and K tensors were bitwise equal to the HF operations at both tested budgets. [Before](02_vision_prep/raw/item2_profile_before.json) · [Fused profile](02_vision_prep/raw/item2_profile_fast.json) · [Direct Q/K comparison](02_vision_prep/raw/item2_dispatch.json)

**Generality**. These timings apply to 24 blocks, 16 heads of width 64 and 468 patches per frame on the measured G4. Attention arithmetic grows quadratically with patches per image while rotary arithmetic grows linearly.

## The vision tower can change decisions before LM prefill

**Claim**. A BF16 patch projection and a shape-dependent attention reduction each changed the output of the vision tower on this G4.

**Numbers**. For 32 frames at Tv=128, using a flattened biased GEMM in place of the HF patch convolution changed patch outputs by a mean absolute 0.000344 and a maximum 0.03125. The single-image speed diagnostic moved from 928/930 to 930/930 fused BF16 decisions after restoring the convolution. With that convolution in both paths, the eight-frame first-block batched SDPA still differed from per-image SDPA by up to 0.015625. The trace identified a split-KV reduction and combine kernel on the singleton. Preserving the per-image reduction brought all ten full-tower comparisons to bitwise equality against matching HF batches. [Patch trials](02_vision_prep/raw/item2_patch.json) · [C0 diagnostic](00_ceiling/raw/c0_diag.json) · [Corrected diagnostic](02_vision_prep/raw/item2_diag_fused.json) · [Attention trace](02_vision_prep/raw/item2_dispatch.json) · [Full features](02_vision_prep/raw/item2_features_fast.json)

**Generality**. The observation concerns these BF16 kernels, grids, hardware and checkpoint. The HF tower itself changes numerically with dense-projection batch size when compared against isolated full-image forwards.

## Encoder-cache reuse in fresh multimodal serving

**Claim**. A zero cached-prompt-token count can coincide with substantially faster requests when vLLM retains visual embeddings across calls.

**Numbers**. At Tv=256, N=32 with a rotating 64-image pool, vLLM 0.34 measured 237.2 ms after clearing prefix and multimodal processor caches. Explicitly clearing the encoder cache as well raised the median to 327.5 ms. Both runs reported zero cached prompt tokens. In the 256-image pool, the full reset measured 324.4 ms, and changing one pixel in each request as an additional control gave 326.2 ms. [Original small pool](02_vision_prep/raw/item2_small_vllm.json) · [Small pool with encoder reset](02_vision_prep/raw/item2_small_cold_vllm.json) · [Full pool with encoder reset](02_vision_prep/raw/item2_cold_vllm.json) · [Unique content](02_vision_prep/raw/item2_unique_vllm.json)

**Generality**. The measured effect depends on repeat distance, image budget and the serving engine's cache implementation. At Tv=128, N=128 with 256 images, the two full-pool reset protocols yielded 635.1 and 635.0 ms. [Historical reset](02_vision_prep/raw/item2_serving_vllm.json) · [Full reset](02_vision_prep/raw/item2_cold_vllm.json)

## A native SwiGLU epilogue saves more than changing the GEMM backend

**Claim**. The kernel architecture in a cuBLAS name was a poor predictor of the opportunity in this BF16 prefill. Keeping the gate and up projection results on chip saved time even when the native plain GEMM was slower.

**Numbers**. At 18,688 rows, the 2,560 × 18,432 projection took 4.369 ms with cuBLAS, 4.381 ms with cuBLASLt and 4.594 ms for the fastest of three tested plain native Triton tiles. The cuBLAS projection plus SwiGLU took 5.197 ms. A native fused tile took 4.555 ms, and a separate tensor-memory-accelerator tile sweep reached 4.255 ms. Absorbing the gate eliminates a 657 MiB BF16 projection write and its subsequent read per MLP layer at this shape. [Projection timings and compiled targets](03_lm_prefill/raw/item3_dispatch_final.json) · [Tensor-descriptor sweep](03_lm_prefill/raw/item3_tma.json)

**Generality**. Measurements cover the Qwen3.5-4B weight layout, this SM120 card and its installed BLAS stack. The activation footprint scales with token count and intermediate width.

## The gated delta rule pays for materialized intermediate states

**Claim**. Several FLA chunk kernels already approach the card’s measured device-memory bandwidth. A floor that counts only the recurrence’s useful contractions and compulsory inputs understates the cost of the implemented algorithm.

**Numbers**. On 128 independent 146-token sequences, the packed 64-token GDN recurrence took 1.946 ms per layer. Nsight Compute measured 1,427 GB/s for the transformed key/value kernel, 1,440 GB/s for the state update and 1,477 GB/s for the output contraction. The calibrated copy peak was 1,463 GB/s. Across 24 layers the core alone took about 46.7 ms, while the model assigns 7.56 ms to the ideal GDN stage including convolution. [Chunk sweep](03_lm_prefill/raw/item3_gdn.json) · [Hardware counters](03_lm_prefill/raw/item3_dram_rates.tsv) · [Floor equations](ROOFLINE.md)

A fused SM120 kernel reduced the intermediate traffic by splitting the value dimension into 64-wide tiles. Its 128-wide default requested 112,640 bytes of shared memory per block, above the G4’s 101,376-byte opt-in limit. The smaller tile completed the same GDN core in 0.906 ms per layer at 128 × 146 tokens. In one representative launch of each core kernel, the five FLA stages transferred 2.548 GB over DRAM. The three accelerated stages transferred 0.634 GB, a 75.1% reduction. The full fixed-feature LM and readout fell from 426.8 to 402.3 ms in the corresponding isolated runs. [Tile sweep](03_lm_prefill/raw/item3_gdn_tile64.json) · [Accelerated counters](03_lm_prefill/raw/item3_accel_ncu_production_128.csv) · [Standard stack](03_lm_prefill/raw/item3_profile_production.json) · [Accelerated stack](03_lm_prefill/raw/item3_accel_profile_production.json)

**Generality**. The measured rates cover one warmed launch per kernel configuration on this G4. Chunk size, retained state, head dimensions, batch lengths and cache residency affect the intermediate traffic.

## Bit-exact vision costs about 2% of fresh latency (optional non-exact mode)

**Claim**. Keeping the vision tower bit-identical to HF (one SDPA per image, HF's reduction order) costs little against a batched flash-attention variant whose split-KV reduction order differs.

**Numbers**. Fresh Tv=128, N=128 on engopt (item 2): bit-exact concurrent per-image SDPA 774.2 ms end to end (vision 187.3 ms) vs batched flash 762.0 ms (vision 171.5 ms); Tv=256, N=256: 3,173.8 vs 3,055.5 ms (vision 897.1 vs 774.2 ms). [Bit-exact run](02_vision_prep/raw/item2_fast.json) · [Batched-flash run](02_vision_prep/raw/item2_after.json)

**Generality**. Specific to this tower (equal grids, 512-1,024 patches per image) and the G4; the gap grows with patches per image. The batched variant is kept only as an optional non-exact mode; the production path stays bit-identical.

## Forking one image for many questions and querying many independent images

**Claim**. Repeated questions on one image can share both its fixed-size GDN state and its length-dependent attention KV. Their copy costs depend on different dimensions. Independent images still require a distinct GDN state read per image.

**Numbers**. With 704 prefix slots and 32 questions, the reference state gather created 1,536 MiB of destinations across 24 layers. The two KV gathers created 704 MiB across eight layers before concatenation. Replaying these operations without downstream consumers took 0.559 ms for state and 0.606 ms for KV gather plus concatenation. At one question the times were 0.065 and 0.071 ms. These byte counts describe logical destinations, and the graph can reuse scratch memory between layers. With 256 independent Tv640 images, the recurrent bank alone is 12 GiB. The output-only image branch and packed attention reduced yes/no query latency from 300.69 to 177.43 ms on the same G4. [Isolated copy graphs](04_gdn_branch/raw/item4_staged_micro.json) · [Reference sweep](04_gdn_branch/raw/item4_reference_cached.json) · [Production sweep](04_gdn_branch/raw/item4_checked_cached.json)

**Generality**. GDN state size scales with value heads and the product of key and value widths. KV grows with cached sequence length. The measured latencies depend on this GPU, layer layout and allocation reuse.

## The reference rounding points differ between recurrent and chunked branches

**Claim**. A more accurate state contraction can move a decision away from the BF16 serving reference when the serving reference rounds the boundary state inside its chunk algorithm.

**Numbers**. Three BF16 residual parts exactly reconstructed 6,291,456 sampled FP32 state elements from three real layers. For 393,216 seed projections, 393,213 landed within one BF16 ULP of the FP32 product. The worst distance was 29 ULPs and the largest absolute difference was 0.00000573. Applying the high-precision recurrence to the shared-question path changed one decision in 618 paired full and follow-up answers. Matching FLA's normalized-Q/K, boundary-state and intermediate-value rounding restored 618/618 agreement on the same inputs. [State split](04_gdn_branch/raw/item4_staged_micro.json) · [WY comparison](04_gdn_branch/raw/item4_staged_questions.json) · [Chunk-compatible comparison](04_gdn_branch/raw/item4_checked_questions.json)

**Generality**. Bitwise reconstruction of a stored state establishes a storage property. The projection reductions, triangular solve and output rounding determine how closely the continuation matches a chosen implementation.

## A 16-binade window captures nearly all weights after LoRA merge

**Claim**. A 4-bit exponent offset per weight, with one window per output channel and a sparse list of exceptions, preserves every BF16 bit in the six projection families of the merged Glance language model.

**Numbers**. Of 3,569,090,560 weights in 128 fused projections, 281,134 fell outside their channel window (0.007877%). The packed tensors occupied 5,366,326,968 bytes, or 12.02845 bits/weight including metadata and tile padding. GDN input had the highest cost at 12.08014 bits/weight because its 12,352 output channels pad to 12,416. MLP down used 12.00728 bits/weight. The reference and native decoders reproduced the original weights. [Per-matrix export](05_w12/raw/item5_encode.json) · [Codec and GPU validation](../../speed/w12.py)

**Generality**. The escape counts describe one LoRA-merged Qwen3.5-4B checkpoint. The same integer codec can represent arbitrary BF16 patterns through escapes. The effective storage rate depends on the channelwise exponent distribution and matrix dimensions.

## Short cached queries can spend half their dense time fetching weights

**Claim**. On the G4, fetching different BF16 weights accounts for about half the dense projection time at 16–64 rows. Reusing the weights has little effect once a projection processes several hundred rows. W12 therefore helps a short box query over one or eight cached images.

**Numbers**. Rotating real weights through a working set larger than the 128 MiB L2 took 5.39 ms across 128 BF16 projections at 16 rows. Repeating the same matrix at each shape took 2.61 ms. The excess was 51.6% of the rotating time at 16 rows, 54.7% at 64, 35.0% at 128, 6.1% at 256 and 0.9% at 1,024. The one-image scan-bank box query fell from 7.32 to 5.55 ms with W12 and compact graph buckets. At eight images it fell from 8.40 to 7.02 ms. At 32 images the median box query contained 368 useful rows and ran entirely in BF16. [Cold versus reused weight timings](05_w12/raw/item5_weight_traffic.json) · [BF16 bank sweep](05_w12/raw/item5_scan_bf16.json) · [W12 bank sweep](05_w12/raw/item5_verified_scan_w12.json) · [Bucket-matched dense graph timings](05_w12/raw/item5_verified_stages.json)

**Generality**. Prefix caching removes visual-prefix tokens from the query GEMMs. Physical row count then depends on question length, number of independent prefixes and graph padding. The measured cutoffs depend on matrix width, kernel, GPU bandwidth and L2 capacity.

## A decision batch finishes its answers together

**Claim**. Increasing the prefill token budget can delay the median client's answer even when the last completion barely changes. Each Glance batch produces all of its decisions in one forward. Clients assigned to the next batch wait for the preceding batch's complete readout.

**Numbers**. For 64 simultaneous cold box requests on the G4, increasing the standard BF16 scheduler's budget from 8,192 to 16,384 tokens changed the last-completion median from 1,550.1 to 1,545.3 ms. Median time to answer rose from 883.6 to 1,207.7 ms. Both runs delivered 41.4 requests/s. A 4,096-token budget delivered 41.1 requests/s with a median client latency of 876.3 ms. [8,192 tokens](06_scheduler/raw/item6_scheduled.json) · [16,384 tokens](06_scheduler/raw/item6_tune16384.json) · [4,096 tokens](06_scheduler/raw/item6_tune4096.json)

**Generality**. These are synchronized arrivals with one independent image per client. A generative scheduler also manages unfinished token sequences across steps. Here only the next prefill admission and any reusable prefix state carry forward. The optimum depends on arrival spacing and image size.

## The input protocol changes the cost of a cached follow-up

**Claim**. Sending an image identifier can save a substantial fraction of follow-up time even when both protocols hit the same GPU prefix cache.

**Numbers**. For 64 box follow-ups on independent images, resubmitting decoded RGB pixels took 175.6 ms. Supplying the retained prefix IDs took 65.8 ms. The pixel path hashes the resized pixels to look up each prefix. The ID path also admits requests by their question-token cost, allowing more clients into a single query batch. The difference includes both effects. [Pixel requests](06_scheduler/raw/item6_final.json) · [ID requests and stage times](06_scheduler/raw/item6_by_id.json)

**Generality**. Content-addressed lookup requires reading the resent media. An ID issued at encode time avoids that read while the prefix remains resident. Both paths still read the hybrid model's GDN states and attention KV.

## Vision batch composition affects the typed BF16 readout

**Claim**. On the tested hybrid model, the layout used for vision before packed LM prefill can move a downstream label probability beyond the existing BF16 limit, including for an isolated logical request that contains multiple images.

**Numbers**. With 64-token GDN chunk alignment and singleton cache insertion, the mixed-grid run had one outlying typed probability against shipped BF16, at an absolute difference of 0.02743. Grouping images by spatial grid and restoring their original LM order reduced the maximum to 0.01181 against shipped BF16 and 0.01881 against isolated requests. Both were below the existing 0.02335 limit. All 254 interleaved decisions agreed in the grouped run. [Mixed-grid control](06_scheduler/raw/item6_canonical_equiv.json) · [Grouped-grid standard-stack run](06_scheduler/raw/item6_validation_standard_check.json)

**Generality**. This tests one checkpoint and 32 logical requests in two shuffled waves. The effect includes any change in vision kernel dispatch and BF16 rounding that follows from changing the grid composition. The experiment does not identify an individual arithmetic instruction.

## The BF16 rounding boundary can be retained inside a vision MLP epilogue

**Claim**. A fused BF16 projection can reproduce the checkpoint's GELU exactly on tested tower inputs while avoiding the separate expanded preactivation buffer. Rounding the biased FP32 accumulator before indexing the activation table preserves the intermediate seen by the reference GELU.

**Numbers**. On the G4, the first vision block's 1,024→4,096 projection plus GELU fell from 0.491 to 0.368 ms at 14,976 patch rows and from 2.184 to 1.444 ms at 64,512 rows. All five tested tiles produced zero differing elements at both shapes. The complete fixed-grid tower matched HF at two grids and five batch sizes each. [Projection tile sweep](07_frames/raw/item7_candidate_mlp.json) · [Full tower feature checks](07_frames/raw/item7_final_features.json)

**Generality**. Enumerating all 65,536 BF16 inputs costs 128 KiB for a BF16 scalar activation. Reproducing the entire fused operator also depends on the projection's accumulation, bias and rounding behavior on the target hardware.

## A faster isolated attention launch can lose inside the full frame pipeline

**Claim**. Matching the per-image FlashAttention reduction made a batched implementation bit-exact, but its isolated speedup at the shorter grid did not predict a production tower speedup.

**Numbers**. Across 32 images with 468 patches each, the batched split-KV call took 0.357 ms against 0.923 ms with the existing per-image stream path. At 1,008 patches it took 1.508 against 1.309 ms. Both outputs were bit-identical. At Tv=128 and 128 frames in the customer-facing scan, the candidate changed vision time from 186.74 to 190.56 ms. [First-block timing](07_frames/raw/item7_verify_split.json) · [Fresh control](07_frames/raw/item7_rgb_fresh.json) · [Full scan with batched attention](07_frames/raw/item7_dispatch_fresh.json)

**Generality**. These results include the launch, workspace and stream behavior of the installed PyTorch build on one G4. They do not establish an optimal reduction schedule for other GPUs or sequence lengths.

## Fixed-grid and mixed-grid video use different BF16 reference paths

**Claim**. For this checkpoint, making the mixed-grid video tower agree with the batched HF tower can change decisions already served by the video engine. The fixed-grid scan and mixed-grid video interfaces need separate reference comparisons.

**Numbers**. The fresh tower matched HF at all ten grid and batch-size combinations. Across eight 32- or 64-frame mixed-grid windows, HF and the pre-item video tower agreed on 381 of 384 box-presence decisions. Retaining the video tower's patch projection and attention behavior while fusing its projection and GELU reproduced all 384 decisions with zero maximum difference in features, boxes, presence probability and IoU probability. [Fixed-grid features](07_frames/raw/item7_final_features.json) · [Mixed-grid references and output deltas](07_frames/raw/item7_final_video_checks.json)

**Generality**. The video workload uses resized RefCOCO images, including repeated images across windows, with a distinct box question per image. The intervention compares complete tower implementations. It does not isolate the share of the difference due to patch embedding versus attention.

## Rare patch-rounding differences can change a calibrated readout

**Claim**. Folding a BF16 patch convolution into two spatial dimensions preserved most patch components and substantially accelerated the operator. Small changes at that boundary still reached the final labels and probabilities.

**Numbers**. At 64 images of 1,008 raw patches, the 2D convolution took 4.52 ms against 18.44 ms for 3D. Of 66,060,288 output components, 15,295 differed (0.0232%). Across 5,120 binary comparisons at two grids the 2D path changed two decisions and exceeded the existing 0.02335 typed-probability limit on 17 examples. A flattened GEMM kept all 5,120 binary labels but exceeded the same limit on 18 examples and changed one three-way choice in the separate mixed-question panel. [Patch components and times](07_frames/raw/item7_patch_patch_conv.json) · [2D binary comparisons](07_frames/raw/item7_twod_conv_binary.json) · [GEMM binary comparisons](07_frames/raw/item7_binary_binary_control.json) · [Mixed questions](07_frames/raw/item7_checked_patch_control.json)

**Generality**. These figures cover one BF16 checkpoint, two resized grids and a fixed ten-question sweep. Patch disagreement frequency alone does not predict which queries cross a decision boundary.

## Exact structure in a cached recurrent state

**Claim**. The sampled GDN states retain mantissa information beyond BF16 throughout the model, while identical initial K/V tokens can be shared across independent inputs.

**Numbers**. All 24 sampled first-head matrices from one four-frame clip and all 24 from one Tv640 image had numerical rank 128 of 128. On average, only 0.0042% and 0.0050% of their full layer-state elements survived an exact BF16 round trip. Level-1 zlib retained 94.19% and 94.21% of the bytes. Across 32 prefixes the KV bit comparison shared 11 leading positions for each native clip format, four for equal-grid images and one for mixed-grid Tv640. [State probe](08_bank_memory/raw/item8_probe.json) · [Shared positions and bytes](08_bank_memory/raw/item8_measured_after_bytes.json)

**Generality**. The rank observation covers one head per layer and two inputs of this checkpoint. The number of shareable prefix tokens depends on the tokenizer, input layout and numerical behavior of the prefix encoder.

## Quantizing GDN state and attention KV changes different scan decisions

**Claim**. Similar AUROC and calibrated set-level false-alarm rates can coexist with different clip-level decisions after bank quantization.

**Numbers**. On 5,419 Charades clip decisions, storing BF16 recurrent state flipped three, row-scaled FP8 state flipped two, row-scaled INT8 state flipped five, and FP8 KV alone flipped one. Maximum changes in the yes probability were 0.00706, 0.00692, 0.02480 and 0.01223 respectively. The reference N=16 MaxStat false-alarm rate was 8.72%. Holding its calibration fixed gave 8.75% for INT8 state and 8.72% for FP8 KV. On the 455-question gRefCOCO presence check, INT8 state flipped one decision, with a maximum probability change of 0.06804. [Paired Charades scores and set metrics](08_bank_memory/raw/item8_retry_video.json) · [Image presence metrics](08_bank_memory/raw/item8_final_presence.json)

**Generality**. The state and KV encoders use one FP32 scale per last-dimension vector. The video check used the short-readout checkpoint, 64 query IDs and 167 distinct videos from the E4 manifest. These samples do not establish a universal ranking of formats.

## Cost of the first questions on a new bank

**Claim**. Warm branch latency predicts a saving at two questions for all four measured shapes. Graph capture on a newly encoded bank moves the observed crossing for small frames.

**Numbers**. Across 64 prefixes the warm encode / cached query / fresh query times in ms were 894.9 / 53.7 / 782.5 for four-frame v96 clips, 2,034.0 / 43.2 / 1,906.1 for Tv640, 766.0 / 41.0 / 618.8 for Tv256 and 256.4 / 40.3 / 175.9 for Tv64. With new-bank capture and changing text, the first three configurations first won on question two. Tv64 still took 842 ms through question four versus 729 ms fresh in the initial sweep. Its first win in a full 16-question cycle was question 5. [Warm amortization](08_bank_memory/raw/item8_stream_breakeven.json) · [Four changing questions](08_bank_memory/raw/item8_release_lifecycle.json) · [Extended small-frame sweep](08_bank_memory/raw/item8_extended_lifecycle.json)

**Generality**. These crossings apply to a warmed G4 engine and reuse of the same 64 inputs. The order and lengths of the questions determine which bank-local graphs require capture. The storage footprint persists for the lifetime of the bank.

## PCIe throughput for a bank larger than device memory

**Claim**. Chunked host prefetch overlaps most of a full-readout query with transfer at large bank sizes. Short scans pay the latency to fill and drain the pipeline.

**Numbers**. The experiment pinned 110.62 GB of distinct compact clip records for 1,920 clips on a G4 with 101.97 GB of device memory. The 1,024-clip host scan took 1,090.4 ms against 963.0 ms resident. At 1,920, it took 1,997.7 ms with a 1,945.2 ms PCIe floor. The coalesced exact state-packed host default brought the 1,024-clip query to 1,010.5 ms, and the 10,000-entry traversal to 9.33 s. Its PCIe floor was 9.21 s. The compact 10,000-entry scan transferred 576.13 GB in 10.18 s, using repeated visits to the physical backing. [Exact-packed transfers](08_bank_memory/raw/item8_coalesced_tier_packed.json) · [Compact query and copy times](08_bank_memory/raw/item8_coalesced_tier.json) · [Resident controls](08_bank_memory/raw/item8_exact_resident.json) · [Measured H2D rate](ROOFLINE.md)

**Generality**. Each chunk held 64 native clips, and each query returned the same five tensors as a GPU-resident chunk. The result depends on payload size, device compute time, link bandwidth and the available pinned host RAM. The benchmark physically stored 1,920 distinct frame tuples.

## Where layout changes reach the box readout

**Claim**. In the five-layout BF16 reference calibration, changes in the incoming question and visual states dominate arithmetic roundoff in center selection. A different winning patch can move the regressed box and the IoU estimate together.

**Numbers**. Of 420 pairwise box comparisons, 28 changed center with the shipped dot product, with an FP64 dot product, and with FP64 normalization, projections and dot product. Changing only the projected question switched 13 centers. Changing only the projected visual keys switched 28. The largest score difference between layouts was 0.9214 logit. On identical inputs the maximum score change from promoting the final dot to FP64 was 0.0000030. Across layouts the maximum raw normalized-coordinate difference was 0.21247. Among matching centers it was 0.01816. Choosing the first patch within 0.06 logit of the winner increased the number of layout switches to 46. The fixed-row head reproduced every score and output with zero difference in 210 frozen-hidden-state replays. A whole-backbone FP32 control retained 420/420 centers across equivalently arranged inputs and reduced the raw maximum coordinate spread to 0.000357. The shipped FP32 forward took 300.68 ms versus 96.37 ms in BF16 for 32 box questions on one image. [Fixed-state replays](09_numerics/raw/item9_strict_replay.json) · [FP32 layouts](09_numerics/raw/item9_precise_fp32_probe.json) · [Paired precision timing](09_numerics/raw/item9_controls_precision_latency.json) [Paired operands, winners, margins and edge controls](09_numerics/raw/item9_calibrated_calibration.json)

**Generality**. These are pairwise comparisons over both attention masks, with repeated observations of the same v1 inputs. On a separate 48-image COCO panel at Tv256, the largest singleton-reference margin at a cached-path disagreement was 0.1574 logit. The speed diagnostic also distinguishes 930 presence or typed decisions from its 570 box-center choices. [COCO margins](09_numerics/raw/item9_blocks_traced_margins.json) · [Speed diagnostic](09_numerics/raw/item9_release_diag.json)
