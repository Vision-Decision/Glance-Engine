# Fresh-image variable-length prefill

The packed LM restores approximately linear scaling. Doubling the nominal visual tokens from 8k to 16k took 2.02× the LM time, compared with 6.63× on the legacy 16k-chunk run. The final clean end-to-end result at Tv=128, N=128 was **931 ms**, against **2,586 ms** for legacy 16k chunks, **951 ms** for the coordinator's tuned 4k chunks, **638 ms** for this vLLM rerun and **823 ms** for SGLang. The <629 ms end-to-end target remains open. The real-weight scan test also remains red under its original numerical gates.

## Clean G4 timings

One yes/no question over N new, pre-resized COCO frames. Timings in milliseconds include image processing through answers on the host. Glance used four warm-ups and ten measured calls per cell. The serving scripts ran eight measured calls. Each stage checked for an empty GPU process list. The device was an RTX PRO 6000 Blackwell Server Edition with 97,887 MiB, driver 580.82.07. Glance used Torch 2.14.1+cu130 and BF16. All stages used the existing expandable-segments allocator setting. The two baselines were vLLM 0.34.0 and SGLang 0.5.23. Both recorded zero cached prompt tokens and 146 mean prompt tokens per request at Tv=128, N=128. [Glance invocation](raw/item1_after.log) · [vLLM run](raw/item1_vllm.log) · [SGLang run](raw/item1_sglang.log)

| Visual budget | Frames | Legacy prep / vision / LM | Legacy total | Packed prep / vision / LM | Packed total | Packed chunks | vLLM total | SGLang total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 28.6 / 60.1 / 153.1 | 244.0 | 22.7 / 60.1 / 137.4 | 220.7 | 1 | 176.8 | 257.1 |
| 128 | 64 | 50.9 / 128.4 / 335.3 | 514.9 | 42.3 / 128.2 / 289.8 | 460.6 | 1 | 330.4 | 442.1 |
| 128 | 128 | 96.5 / 265.2 / 2,223.6 | 2,586.2 | 81.0 / 264.0 / 586.3 | 931.1 | 1 | 638.3 | 823.3 |
| 128 | 256 | 206.1 / 530.3 / 2,924.2 | 3,650.0 | 164.7 / 529.0 / 1,175.9 | 1,874.4 | 2 | 1,249.5 | 1,593.8 |
| 256 | 32 | 46.0 / 145.8 / 313.0 | 504.7 | 31.5 / 145.6 / 283.4 | 460.6 | 1 | 326.0 | 456.4 |
| 256 | 64 | 88.6 / 293.8 / 1,363.4 | 1,745.9 | 61.1 / 293.4 / 569.6 | 924.0 | 1 | 622.3 | 838.9 |
| 256 | 128 | 177.9 / 587.6 / 2,005.8 | 2,770.9 | 121.6 / 587.1 / 1,136.9 | 1,845.3 | 2 | 1,244.2 | 1,569.4 |
| 256 | 256 | 354.2 / 1,175.3 / 3,288.8 | 4,820.3 | 243.1 / 1,174.7 / 2,274.6 | 3,693.7 | 4 | 2,487.7 | 3,042.2 |

The table uses independent stage and end-to-end medians. The final legacy rerun used the pre-item benchmark and inference sources from `2a51eed`, with the original 16k visual-token chunk setting. After the graph repair the packed run took 931 and 1,874 ms at Tv=128, N=128/256, leaving 293 and 625 ms to vLLM. At Tv=256 the corresponding gaps were 601 and 1,206 ms. At the headline point the packed stage medians were 81 ms preparation, 264 ms vision and 586 ms LM/readout. The serving engines expose only end-to-end timings. [Original-source rerun](raw/item1_before.json) · [Packed rerun](raw/item1_after.json) · [vLLM](raw/item1_vllm.json) · [SGLang](raw/item1_sglang.json)

An independent first run used the worktree's legacy flag. At Tv=128, N=128 it measured 2,596 ms legacy and 944 ms packed. The coordinator's clean, tuned 4k legacy chunks took 951/1,907 ms at N=128/256 for Tv=128 and 1,883/3,765 ms for Tv=256. Packing largely recovers that tuned performance automatically. Across the initial full timing series, peak PyTorch allocation fell from 81.44 to 12.88 GiB. An attempted timing invocation on the older `origin/main` lacked the E1b CLI options and exited before loading a model. The table uses the successful matching-workload rerun. [Initial before](raw/item1_before_initial.json) · [Initial after](raw/item1_after_initial.json) · [CLI error](raw/item1_before_origin_cli.log) · [Rerun log](raw/item1_before.log)

## Why the original median rose

The rotating questions create multiple LM graph keys. Each legacy graph keeps its returned prefix cache alive. A convolution tail can pin an entire fused GDN input projection, and an attention V view can pin the fused QKV allocation. At 16k nominal visual tokens the cache references **19.41 GiB** of unique backing storage per profiled call. Reserved memory reached 78.0 and 78.3 GiB just before the two timed evictions. The allocator recorded zero allocation retries. When GPU memory reserved by PyTorch exceeds 80% of capacity, a capture on a missing key evicts that owner's existing graphs. During the ten measured calls at 128 frames the graph count fell from three to one twice. Seven calls spent **1,534–1,805 ms** inside the capture helper, including its two warm-ups. The three replay-only LM calls took **671, 682 and 671 ms**. At 64 frames only four calls captured, so replay still determined the median.

| Nominal visual tokens | Actual prefix + question tokens¹ | Eager LM before / after, ms | Returned cache before / after, GiB | Peak allocated before / after, GiB | Legacy timed calls capturing |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,672 | 157.8 / 133.9 | 4.85 / 0 | 15.8 / 9.8 | 0 / 10 |
| 8,192 | 9,344 | 330.4 / 284.0 | 9.70 / 0 | 21.3 / 10.5 | 4 / 10 |
| 16,384 | 18,688 | 675.7 / 576.6 | 19.41 / 0 | 32.3 / 11.9 | 7 / 10 |
| 32,768 | 37,376 | 1,374.1 / 1,160.4 | 38.82 / 0 | 54.3 / 15.2 | 0 / 10 |

¹ Counts are for the first question. The isolated CUDA-event timings use one unchunked call with fixed vision features. At 32k the original unchunked profile exceeds its 32,768-LM-token capture cutoff and runs eager. The end-to-end legacy benchmark instead splits this cell into two nominal 16k chunks. [Reference timeline, kernel names and allocator statistics](raw/item1_profile_reference.json) · [Packed profile](raw/item1_profile_varlen.json)

At 16k nominal tokens, the profiler classified 328.1 / 328.9 ms as GEMMs, 147.2 / 74.3 ms as data movement, 57.8 / 39.0 ms as gated-delta kernels, and 3.4 / 2.7 ms as attention (legacy / packed). It identified 9.2 ms of packed convolution. The residual category was 134.3 / 120.4 ms and includes legacy operations whose compiled names do not identify their role. Both eager LM timings roughly doubled from 8k to 16k. Equal-length legacy prefix attention already uses causal SDPA, and its question masks operate within individual images. The observed extra second and a half comes from repeated graph construction.

## Implementation and graph lifetime

The packed sequence contains the valid prefix followed immediately by that image's question. Three cumulative-length arrays describe full sequences, prefixes, and questions. FLA's causal depthwise convolution and chunked gated delta rule reset only at image boundaries. Both carry the prefix through to the question. The GDN returns no final state. Full attention uses native PyTorch varlen attention. Bidirectional questions use causal prefix self-attention followed by question-to-own-sequence attention. Positions and readout indices follow the original examples. Host collation and the head's visual axis can still be rectangular. There is no batch-wide quadratic LM mask. [Packed LM](../../../scan/varlen.py)

The chunk planner counts valid tokens. Its per-layer estimate uses model widths and a workspace budget capped at 2 GiB or one quarter of free device memory, whichever is smaller. Uniform lengths use stable power-of-two image counts when splitting. Oversize single examples run alone. The legacy implementation is available through `fresh_varlen=False`, `--fresh_impl reference` and `keep_bank=True`. Packed LM prefill executes eagerly, with the existing eligible vision and bank-query graphs enabled. The isolated eager-to-eager numbers above separate the kernel improvement from the cost of legacy graph capture. [Engine](../../../scan/engine.py)

The original benchmark crashed at `capture_begin` when it moved between image banks. Cached branch graphs belonged to the bank, while their CUDA pool handle belonged to the longer-lived engine. Releasing the bank could release the last graph using that pool and leave the engine with a stale handle. Each bank now owns its graph cache and pool together. Capture refreshes the handle when its owning graph cache is empty. A model-free reproducer hit the identical allocator assertion only when an output allocation survived its original graph and the released handle was reused. Recapturing into a fresh pool passed with that output still live. Both outcomes reproduced with expandable segments enabled and disabled. [Controlled reproducer](raw/graph_pool_lifetime.py) · [Allocator matrix](raw/item1_pool_matrix.log)

The same graph-enabled benchmark now reaches its final clip bank and exits zero with `expandable_segments:True`. [Original exception](raw/item1_equiv_initial.log) · [Repaired benchmark](raw/item1_equiv.log)

## Equivalence

| Check | Result |
|---|---|
| Tiny CPU fp32 | Both suites passed. All 18 mixed-length decision comparisons agreed, maximum ΔP 1.79 × 10⁻⁷. Forced chunking, both question masks, clips and the API checks passed. The separate fused CPU run gave 1.20 × 10⁻⁷. |
| BF16 mixed-length scan, real weights | All 18 comparisons against the shipped reference, separate engine and legacy batch agreed on decisions. Maximum absolute ΔP was 0.007919 against a 0.02 limit. Some raw box-output comparisons exceeded the existing relative-error limit of 0.02, reaching 0.03715 against the separate engine and 0.03576 with forced chunks. |
| BF16 graph-enabled COCO scan | All 20 comparison rows agreed on all 24 decisions per row, including images and four-frame clips. Fresh versus shipped reference maximum ΔP was 0.01835 on native-aspect images, 0.07797 on fixed-size frames and 0.02217 on clips. The largest fresh normalized box-coordinate difference was 0.15584. |
| Speed diagnostic | Completed on 60 images with 930 decisions and 450 box ground truths. The original and revised diagnostic JSONs matched byte-for-byte. Against the shipped BF16 reference, plain engine and fused-convolution-only engine agreed on all 930 decisions. FastVision + graphs agreed on 926/930, fused BF16 on 928/930. This diagnostic reports discrepancies and exits zero. |
| Full BF16 real-weight scan | The direct HF invocation now completes. The suite exits 1 on the unchanged numerical thresholds. The origin control also prints `SCAN_EQUIVALENCE_FAIL` after the same invocation fix. |

[CPU before](raw/cpu_before.log) · [CPU after](raw/cpu_after.log) · [Fused CPU](raw/cpu_fused.log) · [Full real-weight scan](raw/item1_scan_real.log) · [COCO equivalence JSON](raw/item1_equiv.json) · [Speed diagnostic](raw/item1_diag.json) · [Origin speed diagnostic](raw/item1_diag_origin.json)

On unmodified `origin/main` at `30e7d19` the direct HF reference crashed in PEFT with a BF16 input and FP32 adapter weights. The model builder deliberately keeps FP32 master adapters and relies on the caller's autocast. The test now supplies the same BF16 autocast used by the shipped LM and evaluates the readout outside that context in FP32. All test cutoffs are unchanged. [Unmodified origin exception](raw/item1_scan_origin.log) · [Initial worktree exception](raw/item1_scan_real_initial.log)

With that invocation correction alone, origin and current both gave cached-box relative errors of 0.2483 (causal) and 0.6952 (bidirectional) against the per-image reference. Legacy fresh causal box error was 0.4023. The API probability differences on origin were 0.0006 for images and 0.0011 for clips, above its 0.0001 threshold. The corresponding packed values were 0.0023 and 0.0013. All ten failing original checks fail on the same named checks in the current run. Four additional raw box checks belong to the new mixed-length fixture. The original script printed a failure while returning process status zero. [Origin with HF autocast only](raw/item1_scan_origin_autocast.log) · [Current full run](raw/item1_scan_real.log)

The per-image BF16 reference also crosses the 0.02 box cutoff **against itself** when only its padding bucket changes from 64 to 1. On the fourth causal fixture, the top two patch-center logits were 0.00139 apart with bucket 64. The winning patch changed from 122 to 161 with bucket 1. Across the six images this changed box coordinates by up to 0.0545 absolute (0.0670 relative), IoU logits by 0.0459 relative and presence logits by 0.00763 relative. The images, weights, vision implementation and per-image reference were identical. The box head selects a patch by hard argmax before predicting its edges and IoU. This discontinuity explains how a small numerical change can move a box appreciably. [Per-patch margins and coordinates](raw/item1_reference_layout.json) · [Reference-only control](raw/reference_layout.py)

Two additional full-suite runs tested BF16 GEMM accumulation as a possible repair. With reduced-precision reductions disabled, the cached bidirectional box error was 0.6695 with cuBLAS and 0.6710 with cuBLASLt and split-K disabled. Both runs completed and failed the original gates. The timing runs retain the deployed defaults. [cuBLAS control](raw/item1_reduction_default.log) · [cuBLASLt control](raw/item1_reduction_cublaslt.log)

### Isolating the LM from the vision tower

A separate control used the **same vision embedding tensor** for the packed LM, original batched LM, and original LM run one image at a time. It covered 24 COCO images at each of two resolutions and three questions. All 144 decisions matched in every comparison. Over all six workloads, the maximum probability difference was **0.01038** against the legacy batched LM and **0.00558** against the separate LM. Both are below 0.02.

For the four newly added mixed-length raw box checks, holding vision features fixed also brings every measured relative difference under the existing 0.02 cutoff. Against the separate LM, maxima over presence logit, IoU logit and coordinates were 0.00460 (causal) and 0.01976 (bidirectional). Against the packed LM run in singleton chunks they were 0.00896 and 0.01888. The original end-to-end checks remain active. [Per-head errors](raw/item1_mixed_box.json) · [Mixed-fixture control](raw/mixed_box.py)

The largest end-to-end difference came from the Tv=256 question “the dog on the left.” Packed versus separate had maximum ΔP **0.08213** with each path running its own vision batch. Feeding those same two LM paths identical features reduced it to **0.00140**. Against the original batched LM on those features it was **0.00014**. This isolates a discrepancy arising before the LM in this case. All three typed yes/no workloads in the graph-enabled COCO scan (images, frames and clips) kept 100% decision agreement. [Fixed-feature control results](raw/item1_fixed_vision.json) · [Reproduction script](raw/fixed_vision.py)

## Reproduction

The VM stage runner accepts `before`, `after`, `vllm`, `sglang`, `profile_reference`, `profile_varlen`, `scan_real`, `equiv` and `diag`. Run each serially with `bash scan/vm/e1b_varlen.sh STAGE` from the configured VM. Use `ITEM1_BEFORE_DIR` to point the before stage at the unmodified checkout. The checkpoint was `/content/ckpt/v1/ckpt_final.pt`. [Runner](../../../scan/vm/e1b_varlen.sh) · [Initial chain](raw/item1_chain.log) · [Repair chain](raw/item1_repair_chain.log)

## Authorship

Codex wrote the packed prefill, chunk selection, shared projection changes, reference flag, regressions, profiler, stage runner, graph-pool lifetime repair, HF invocation correction and analysis. The reviewer provisioned engopt, prepared the serving environments, synced and launched the initial chain, reviewed the first implementation and committed it. The reviewer made no inference-engine changes. Codex ran the repair and control workloads and left the follow-up changes for review.
