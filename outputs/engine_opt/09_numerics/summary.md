# Item 9 · Box readout numerics on G4

The box head can return identical results for identical hidden states across batch and padding layouts. The BF16 backbone supplies different hidden states on those layouts. Over 420 box comparisons, the five-layout reference calibration still had 28 patch switches and a maximum normalized-coordinate difference of **0.212472**. Raising precision throughout the backbone kept all 420 centers fixed and reduced that difference to **0.000357**. [Production-head BF16 calibration](raw/item9_strict_replay.json) · [Whole-backbone FP32 control](raw/item9_precise_fp32_probe.json)

## Readout

The default engine sets `GLANCE_BOX_REDUCTION=blocks`. It evaluates the trained FP32 linear layers on fixed contiguous row blocks, padding the final block with zeros. The visual key projection uses 512 rows per block. The question, presence, edge and IoU projections use 16. A CUDA kernel uses a fixed FP64 reduction across the projected channels of each patch. The hard argmax selects the lowest index on an exact tie. The edge head and IoU head share the features of the selected patch. Cached scan uses the same projection when building the bank, and scores its compact key table directly. The reference mode is `GLANCE_BOX_REDUCTION=legacy`. [Engine readout](../../../speed/box_readout.py) · [CUDA score reduction](../../../speed/box_scores_kernel.py) · [Cached bank and query](../../../scan/engine.py)

The FP64 prototype with ordinary FP32 linear layers produced different scores in all 189 valid slots of a frozen-input padding test. Its largest discrepancy was 3.82×10⁻⁷. Fixed projection tiles removed the discrepancy. A custom fixed-order FP32 projection also passed the identical-input control, but took 9.40 ms on a one-question box follow-up, versus 8.83 ms with fixed-size library GEMMs in the design sweep. At N=256 and 640 patches, an eager FP64 multiply-and-sum took 2.10 ms for selection, including argmax. The CUDA score kernel took 0.49 ms. It reads the original FP32 keys and accumulates in registers, avoiding the full FP64 product tensor. [Frozen-input and kernel controls](raw/item9_tiled_micro.json) · [Custom projection timing](raw/item9_full_speed.json) · [Fixed-size GEMM timing](raw/item9_blocks_speed.json)

For each of the **210 real-weight box observations**, the replay took the hidden inputs from that layout, removed its batch peers and visual padding, and ran the production head again. Every valid center logit, selected index, presence logit, box coordinate and IoU logit was bit-identical. Synthetic cases additionally cover the first-index rule on an exact tie, a difference rounded away in FP32, fully masked inputs, a compact key table and an outer BF16 autocast context. [All real-weight replay differences](raw/item9_strict_replay.json) · [Kernel controls](raw/item9_release_micro.json)

## Where the differences enter

The calibration uses the same 24 fixtures, five reference layouts and ten layout pairs per fixture as C0. A second production run reproduced the complete calibration JSON byte for byte. The switches were reproducible changes between layouts. [Independent repeat](raw/item9_shipping_replay.json) Each box call records the last question hidden state, valid visual states, projected queries and keys, winning index, score margin and regressed edges. The independent reference uses a PyTorch FP64 sum over the FP32 projections. The scores produced by the shipped head are retained for comparison. The two source operands can be exchanged between layouts independently. The production calibration additionally runs the fixed-size projection head on those same shipped-backbone states. The normal engine tests use the independent oracle. [Calibration procedure](../../../scan/test_scan_equivalence.py) · [Operand and edge controls](../../../scan/readout_probe.py) · [C0 calibration](../00_ceiling/summary.md#real-weight-test)

| Controlled quantity | Maximum absolute difference or count |
|---|---:|
| Incoming BF16 question hidden component | 1.50 |
| Incoming BF16 visual hidden component | 5.50 |
| FP32 query / key projection component | 0.10834 / 0.58495 |
| Center score between layouts | 0.92137 logit |
| Score change from promoting the shipped dot to FP64 on identical projections | 0.00000299 logit |
| Score change from also promoting normalization and projections to FP64 on identical hidden inputs | 0.00000754 logit |
| Patch switches when exchanging only query / only keys | 13 / 28 of 420 |
| Patch switches, shipped dot / FP64 dot / FP64 projections and dot | 28 / 28 / 28 of 420 |
| Edge logit between layouts, actual winner / held to the first layout's winner | 5.2552 / 0.2255 |
| Normalized-coordinate change from FP64 expectation on identical edge logits and patch | 0.000000094 |

The query-only and key-only switch counts share examples. Of 210 box observations across all five layouts, 37 had a top-two margin at most 0.06 logit. The minimum was 0.000231 and the median was 0.1754. Choosing the first patch within 0.06 of the winner produced **46** switches across the same 420 comparisons. That rule also changes the trained argmax. Keeping the final RMSNorm output in FP32 gave a raw coordinate spread of 0.212579 and IoU-probability spread of 0.203007. [Per-layout margins and tie-band experiment](raw/item9_calibrated_calibration.json) · [Final-norm control](raw/item9_calibrated_norm_probe.json)

Keeping the LM residual streams in FP32 with BF16 projections gave a raw coordinate spread of 0.20253 and an IoU-probability spread of 0.20641. Promoting the entire backbone gave 0.000357 and 0.000140, respectively. On the same 50 one-image, 32-box-question calls through the shipped forward, BF16 took 96.37 ms and FP32 took 300.68 ms end to end (3.12×). This experiment compares the same reference implementation in both precisions. [FP32 residuals](raw/item9_controls_residual_probe.json) · [Whole-backbone calibration](raw/item9_precise_fp32_probe.json) · [Paired precision timing](raw/item9_controls_precision_latency.json)

On 48 COCO images with four questions per image, the legacy and production runs recorded identical center-agreement counts and disagreement margins. At Tv640, cached and fresh agreed with the singleton reference on 173/192 and 170/192 centers. At Tv256, the counts were 174/192 and 171/192. The largest margin at a cached Tv256 disagreement was **0.1574** logit. Both readouts agreed on 192/192 presence decisions for cached Tv640 and both Tv256 paths. Fresh Tv640 agreed on 191/192 with each head. In a paired run, both fresh-scan heads returned exactly 0.508259 on the same image (zero-based index 40) for “man in a blue shirt holding a phone.” The shipped singleton returned 0.490466. Across all 192 cases, switching the head changed the fresh-scan presence logit by at most 9.54×10⁻⁷. Both six-image random panels agreed on all presence decisions. [Paired threshold crossing](raw/item9_release_presence_audit.json) [Initial margin characterization](raw/item9_before_margins.json) · [Traced legacy](raw/item9_legacy_traced_margins.json) · [Traced production](raw/item9_blocks_traced_margins.json)

## Calibrated gates

All gates below use twice the maximum reference-only spread. The public presence and typed probability limits are 0.031704 and 0.023353. The original mixed-prefix raw coordinate and IoU limits remain 0.006686 and 0.013392. On that fixture all 60 pairwise centers agreed. [Frozen limits](../../../scan/equivalence_limits.py) · [Per-fixture production calibration](raw/item9_strict_replay.json)

| Quantity | Measured spread | Acceptance limit |
|---|---:|---:|
| Raw box coordinate, any center | 0.212472 | 0.424945 |
| Raw IoU probability, any center | 0.203709 | 0.407418 |
| Box coordinate, matching centers | 0.018160 | **0.036320** |
| IoU probability, matching centers | 0.024313 | **0.048626** |
| Box coordinate at the engine's selected patch, including switches | 0.046973 | **0.093947** |
| IoU probability at the engine's selected patch, including switches | 0.033918 | **0.067836** |
| Reference score deficit of the selected patch | 0.487016 | 0.974032 |

Comparisons against the instrumented shipped forward retain the raw coordinate and IoU tests and score the engine's chosen patch. On a switch, the test recomputes the independent reference edge and IoU outputs at that patch. At the same center the localization and IoU gates are 11.7× and 8.4× tighter than the raw-output limits. The largest reference score deficit (0.487016) occurred for a causal image box query between the singleton and six-image layouts at padding 64. The raw BF16 limits themselves could not be tightened by this calibration. These are empirical ceilings for the fixtures and software tested.

## Decision and center agreement

The diagnostic has 930 presence or typed decisions and 570 spatial choices. The shipped BF16 forward and the whole-backbone FP32 forward are both used as references.

| 60-image RefCOCO diagnostic | Decisions vs BF16 | Centers vs BF16 | Decisions vs FP32 | Centers vs FP32 |
|---|---:|---:|---:|---:|
| Shipped BF16 round trip | 930 | 570 | 928 | 459 |
| Merged BF16 reference | 930 | 486 | 928 | 465 |
| BF16 engine, plain / taps / fast vision / full graph (each) | 930 | 467 | 928 | 454 |
| Fused BF16 engine | **930** | **472** | 928 | 470 |
| Whole-backbone FP32 reference | 928 | 459 | 930 (self) | 570 (self) |
| FP8 language model | 926 | 414 | 928 | 415 |
| FP8 language and vision models | 923 | 351 | 925 | 356 |

The shipped BF16 margin percentiles across the 570 box questions were 0.00494, 0.03271 and 0.11271 logit at p10, p50 and p90. There were zero exact ties. Of 98 fused-engine center switches, 97 had singleton-reference margins at or below 0.06. The median switch margin was 0.00817 and the largest was 0.10067. The two decisions on which the BF16 engines differed from FP32 were typed scores. The shipped BF16 forward disagreed on the same two. The FP8 model's four disagreements were typed scores, and the FP8 language-plus-vision model's seven were typed scores, choice or yes/no. Those decisions come from the typed head. The diagnostic records the mismatching row, question type and probabilities for both reference precisions. [Decision tallies, margin percentiles and disagreement rows](raw/item9_release_diag.json)

## End-to-end latency and hardware floors

All values are warm medians in milliseconds on the same RTX PRO 6000 Blackwell. The speed rows use 50 calls per configuration. Cached scan uses 15 changing questions and excludes encoding. Fresh uses 10 calls, one yes/no question over 128 new frames and a 16,384-token prefill. The speed and follow-up controls use the same optional QLA environment and legacy flag. Each stage held the experiment lock and checked for an empty device compute-process list. The vLLM and SGLang columns reuse the clean BF16 full-output baselines documented in the ceiling map. [Legacy speed](raw/item9_matched_speed.json) · [Production speed](raw/item9_release_speed.json) · [Legacy mixed](raw/item9_matched_mixed.json) · [Production mixed](raw/item9_release_mixed.json) · [Legacy cache](raw/item9_before_cached.json) · [Production cache](raw/item9_release_cached.json) · [Legacy fresh](raw/item9_before_fresh.json) · [Production fresh](raw/item9_release_fresh.json) · [Serving protocols and source runs](../CEILING.md#benchmark-boundaries)

| Workload | Legacy ms | Production ms | Change | Floor ms | Floor / production | vLLM ms | SGLang ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box Q=1 | 37.43 | 37.72 | +0.29 (+0.77%) | 17.48 | 46.3% | 166.5 | 187.3 |
| Box Q=8 | 38.28 | 38.54 | +0.26 (+0.68%) | 19.12 | 49.6% | 221.0 | 395.8 |
| Box Q=32 | 48.49 | 48.89 | +0.40 (+0.82%) | 24.76 | 50.7% | 334.0 | 658.0 |
| Mixed Q=8 | 45.80 | 46.24 | +0.44 (+0.97%) | 22.89 | 49.5% | 199.2 | 375.3 |
| Mixed Q=32 | 68.77 | 69.21 | +0.45 (+0.65%) | 39.56 | 57.2% | 278.2 | 655.4 |
| Cached box N=256 | 95.13 | 96.71 | +1.58 (+1.67%) | 69.02 | 71.4% | 3,151.6 | 7,732.9 |
| Cached yes/no N=256 | 175.76 | 175.39 | -0.37 (-0.21%) | 129.86 | 74.0% | 2,679.9 | 5,670.7 |
| Fresh yes/no N=128 | 580.77 | 581.47 | +0.70 (+0.12%) | 444.90 | 76.5% | 637.8 | 815.3 |
| Follow-up box Q=1 | 8.68 | 8.83 | +0.16 (+1.83%) | 4.98 | 56.3% | 145.2 | 162.0 |
| Follow-up mixed Q=1 | 8.65 | 8.84 | +0.19 (+2.19%) | 4.98 | 56.3% | 145.1 | 161.7 |
| Follow-up box Q=8 | 10.15 | 10.31 | +0.17 (+1.63%) | 5.04 | 48.9% | 196.6 | 346.1 |
| Follow-up mixed Q=8 | 14.84 | 15.00 | +0.17 (+1.12%) | 5.65 | 37.7% | 175.3 | 342.2 |
| Follow-up box Q=32 | 19.30 | 19.62 | +0.33 (+1.69%) | 7.62 | 38.8% | 316.9 | 606.5 |
| Follow-up mixed Q=32 | 39.61 | 39.87 | +0.26 (+0.67%) | 22.32 | 56.0% | 254.9 | 561.8 |

The largest absolute increase was 1.58 ms on the cached box scan (1.67%). The largest percentage increase was 2.19% on the shortest mixed follow-up, or 0.19 ms. These are differences of warm medians.

Per-stage pairs below are measured / modeled milliseconds. The C0 useful-work model uses unpadded shapes, 401.7 BF16 TFLOP/s, 67.4 FP32 TFLOP/s, 1.58 FP64 TFLOP/s, 1,463 GB/s device bandwidth and 56.9 GB/s pinned transfer. Each head stage uses the maximum of FP32 projection time, FP64 score time and device traffic. The full 256 × 640 × 512 score control reads 335.54 MB of FP32 keys. Its key-read floor is 0.229 ms. The release kernel measured 0.519 ms including argmax, against 0.239 ms for the legacy FP32 selection and 2.099 ms for eager FP64. Per-stage medians can sum to a different value than the end-to-end median. [Floor calculation](ceiling.py) · [Stage floors](raw/item9_floors.json) · [Measured G4 rates](../ROOFLINE.md#fp64-center-selection) · [Kernel release run](raw/item9_release_micro.json)

| Workload | CPU prep | Vision and H2D | LM and readout |
|---|---:|---:|---:|
| Box Q=1 | 2.20 / 0 | 9.9 / 5.7 | 25.5 / 11.8 |
| Box Q=8 | 2.32 / 0 | 9.9 / 5.7 | 26.4 / 13.4 |
| Box Q=32 | 3.31 / 0 | 9.9 / 5.7 | 35.7 / 19.1 |
| Mixed Q=8 | 2.74 / 0 | 9.9 / 5.7 | 33.4 / 17.1 |
| Mixed Q=32 | 4.05 / 0 | 9.9 / 5.7 | 55.3 / 33.9 |
| Cached box N=256 | 0.13 / 0 | — | 96.6 / 69.0 |
| Cached yes/no N=256 | 0.16 / 0 | — | 175.2 / 129.9 |
| Fresh yes/no N=128 | 8.48 / 0 | 168.6 / 101.4 | 404.5 / 343.5 |
| Follow-up box Q=1 | 0.32 / 0 | — | 8.5 / 5.0 |
| Follow-up mixed Q=1 | 0.33 / 0 | — | 8.5 / 5.0 |
| Follow-up box Q=8 | 0.63 / 0 | — | 9.7 / 5.0 |
| Follow-up mixed Q=8 | 0.86 / 0 | — | 14.1 / 5.6 |
| Follow-up box Q=32 | 1.52 / 0 | — | 18.1 / 7.6 |
| Follow-up mixed Q=32 | 2.29 / 0 | — | 37.6 / 22.3 |

## Verification

- The five-layout calibration agreed on 1,260/1,260 presence or typed decisions. Its box subset agreed on 392/420 centers. The real-weight engine checks passed with both masks and both prefill implementations. Each agreed on 234/234 decisions. Center agreements were 61/66 for standard prefill and 62/66 for fused prefill. The tiny CPU suite also passed, including the loopback scan-server test. [Reference calibration](raw/item9_strict_replay.json) · [Standard prefill](raw/item9_gates_real.log) · [Fused prefill](raw/item9_gates_fused.log) · [CPU and HTTP](raw/item9_default_cpu.log)
- Both negative controls exited 1 with a numerical failure. The one-token shift violated the causal mixed-prefix raw box gate (0.02424 against 0.00669), among others. Adding 0.04 to a single normalized x-coordinate produced 0.03999 error and failed the 0.03632 same-center gate with all 234 decision thresholds intact. The original raw box limit would have admitted that corruption. [One-token shift](raw/item9_gates_fault.log) · [Single-coordinate fault](raw/item9_gates_small_fault.log)
- The scan benchmark agreed on every decision across 20 comparisons of 24 inputs including fresh, cached, separate-engine and native-clip paths. In the scheduler test, interleaved results agreed with isolated and shipped outputs in 254/254 decisions each. The isolated versus shipped control agreed in 127/127. [Scan benchmark](raw/item9_release_equiv.json) · [Scheduler interleaving](raw/item9_release_interleaving.json)
- The 60-image speed diagnostic reached 930/930 against shipped BF16 for every BF16 engine variant. The FP32 reference and optional FP8 variants had typed disagreements. The fused engine matched 472/570 spatial argmaxes. [Speed diagnostic](raw/item9_release_diag.json)

## Reproduction

Sync the Glance worktree and its matching core checkout using the existing engopt sync helper. On engopt, run `ITEM9_LABEL=rerun bash scan/vm/numerics.sh micro replay real fused fault small_fault cached fresh speed mixed equiv diag interleaving`. The runner serializes complete chains, checks the compute-process list before each stage and records the selected mode, command and exit status in the stage log. Use `GLANCE_BOX_REDUCTION=legacy` for the paired timing controls and `GLANCE_TRACE_CENTERS=1` with the `margins` stage for fresh and cached COCO center counts. The `norm_probe`, `residual_probe`, `fp32_probe` and `precision_latency` stages reproduce the separate backbone controls. The `presence_audit` stage records individual threshold crossings for the Tv640 COCO fresh-scan panel. [GPU runner](../../../scan/vm/numerics.sh) · [Raw runs and per-fixture data](raw/)

## Authorship

Codex wrote the readout and GPU kernel, integrated the fresh and cached paths, added the calibration, negative controls and diagnostics, ran the experiments, and wrote the report. The reviewer supplied the prepared engopt VM and the existing sync and benchmark infrastructure.
