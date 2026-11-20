# Item 4 · GDN continuations from a cached prefix

All 21 G4 workloads in the item’s ceiling table improved. A yes/no query over 256 cached images at up to 640 visual tokens fell from 300.69 to 177.43 ms (1.69×), reaching 71.9% of its useful-work ceiling. On one image, 32 mixed follow-ups fell from 56.12 to 39.87 ms (1.41×). The 256-clip query reached 53.0% of its ceiling and has the largest remaining cached-scan gap. [Paired timings and verification](raw/item4_verification.json)

## Workload and design

A Glance prefix contains 24 FP32 recurrent states. Across those layers one image owns 48 MiB of state and 1.125 MiB of BF16 convolution history. The eight attention layers add 32 KiB of BF16 KV per prefix token. A second question about the same image can reuse all three. A scan over different images has to read each image's state. On the G4, using 1,463.2 GB/s device bandwidth and 401.73 BF16 TFLOP/s, the compulsory state plus tail costs 2.25 ms at 64 images and 9.01 ms at 256. The cached Tv640, N256 yes/no stage floors are 111.44 ms for dense projections, 11.54 ms for GDN including its token streams, 4.28 ms for attention and 0.25 ms for heads. A single box follow-up has a 4.88 ms dense-weight floor and a 0.04 ms GDN floor. [Measured rates and equations](../ROOFLINE.md) · [Per-workload floors](../00_ceiling/raw/c0_floors.json)

The reference scan already passed bank states directly to FLA's output-only recurrent kernel. Its token scatter, history concatenation, transposes, split softmax and per-layer gathers still moved data for padded question blocks. On one Tv640 query over 256 images the reference trace recorded 119.79 ms in the GEMM category, 66.72 ms in copy/concatenation and 38.60 ms in FLA recurrence. The reference multi-question engine instead gathered one FP32 state per question and called a chunk implementation that also returned the final states. Its 32 mixed follow-ups on the profiled image recorded 13.31 ms in copy/concatenation and 9.27 ms in FLA kernels. These are individual warmed traces. The latency table uses independent multi-call medians. [Complete reference trace](raw/item4_reference_profile.json)

The new SM120 BF16 path convolves the actual question rows directly against a strided prefix tail and rounds convolution and SiLU at the existing BF16 boundaries. The cached scan computes beta and log-decay while it processes the value channels. The WY kernel normalizes Q/K, uses tensor cores for the state projections and solves the strictly lower triangular delta-rule system by repeated squaring. Each value tile reads its initial state and writes the answer rows. The first 64-token WY tile required 123,392 shared-memory bytes per block on a device permitting 101,376. The production tile processes 32 tokens and carries the FP32 update on chip to the next tile when required. The initial state is read-only. A device-side question order and offsets let an image with several questions reuse its tile through graph replay. [Initial tile compilation](raw/item4_fused_merged.log) · [Branch kernels](../../../speed/gdn_branch.py) · [Packed collation and multi-question dispatch](../../../speed/engine.py)

SpecPilot’s Apache-2.0-licensed tree kernel decomposes the FP32 boundary state into three BF16 parts for tensor-core projection. It uses ancestor masks and saves per-node inputs for committing an accepted continuation. Glance's blocks are independent linear sequences. Glance needs the outputs of every block and has no commit operation. The implementation here was written for Glance's packed rows, grouped-query heads, prefix layout and readout. It uses the three BF16 residual parts of a read-once FP32 state for tensor-core projection. The timing control also implements a scalar FP32 recurrent walk over the same packed convolution. [SpecPilot kernel at inspected revision](https://github.com/StargazerX0/SpecPilot/blob/1058dc29481150e66ae11b8e8cf55ed60caf254a/settings/pro6000_c1_qwen38-27b_dflash2/files/python/sglang/srt/speculative/specpilot_tree_gdn.py) · [License](https://github.com/StargazerX0/SpecPilot/blob/1058dc29481150e66ae11b8e8cf55ed60caf254a/LICENSE)

The attention kernel reads prefix KV from the bank and question KV from the packed projection. A first pass finds the common softmax maximum. A second accumulates the two value products, honoring prefix padding and the causal or block-bidirectional question mask. Scan and multi-question entry points have different reference rounding boundaries. The scan also reuses the existing large-row MLP epilogue and residual normalization. The PyTorch/FLA path remains selectable with `GLANCE_BRANCH_MODE=reference`. [Scan dispatch](../../../scan/engine.py)

## Reference arithmetic and dispatch

The cached image kernel continues an FP32 recurrence from each image's state. On the Glance multi-question path, FLA's reference chunk implementation rounds Q/K after normalization, the boundary state before its tensor-core products, and the updated values before computing outputs. Substituting the three-part recurrence for this reference changed one typed decision among 618 paired cold and follow-up outputs across both masks. The two implementations can round to different answers on the same weights even when they express the same real-valued recurrence. [Paired comparison](raw/item4_release_questions.json) · [FP32 and FLA mixer controls](raw/item4_release_micro.json)

The production multi-question kernel keeps FLA's 64-token within-chunk transform. It loads each image's FP32 state tile once, forks it in registers, computes the boundary and within-chunk products at FLA's rounding points, and writes packed answers directly. It can advance to a second chunk without writing a boundary state to DRAM. The bank remains in FP32. A separate experiment precomputed Q/K products against one shared state and parallelized the remaining solves over questions. Its shared-prefix timing and decision comparisons are retained with the prototype results. [Staged contraction timing](raw/item4_staged_micro.json) · [Decision comparison](raw/item4_staged_questions.json)

A fixed 24-clip bank distinguished the numerical effects of GDN and attention. The original maximum typed-probability difference from the shipped model was 0.023087. WY alone gave 0.023545, fused attention and layer transitions gave 0.024261, and the combination gave 0.027082. Native-clip banks therefore retain the reference FLA recurrence and split attention. Their projection rows are already in image order, so the production engine can pass views of the valid rows, skip four input scatters, and omit the padded recurrent steps. The final clip check returned to 0.023087 against the existing 0.023353 limit. [Clip controls](raw/item4_checked_clip.json) · [Final scan equivalence](raw/item4_checked_equiv.json)

## Idle-G4 wall time

Every entry is a warm median in milliseconds through answers on the host. The paired Glance sweeps used the same checkpoint and settings, 30 calls per cached row and 50 per speed row. Cached scans exclude encoding the bank. Follow-ups start with a retained prefix. The serving columns reuse the existing matched BF16 cache-protocol runs. “Ceiling” is useful-work floor divided by observed latency. Host preparation has a zero modeled floor. [Glance before, banks](raw/item4_reference_cached.json) · [Glance after, banks](raw/item4_checked_cached.json) · [Glance before, single and follow-up](raw/item4_reference_speed.json) · [Glance after, single and follow-up](raw/item4_checked_speed.json) · [Serving sources and workload definitions](../CEILING.md)

| Workload | Before | After | Speedup | Floor | Ceiling | Host prep | Vision / floor | LM and heads / floor | vLLM | SGLang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single, box Q=1 | 39.4 | 37.5 | 1.05× | 17.5 | 46.6% | 2.3 | 9.8 / 5.7 | 25.4 / 11.8 | 166.5 | 187.3 |
| Single, box Q=8 | 41.8 | 38.4 | 1.09× | 19.1 | 49.8% | 2.5 | 9.8 / 5.7 | 26.2 / 13.4 | 221.0 | 395.8 |
| Single, box Q=32 | 57.6 | 48.8 | 1.18× | 24.8 | 50.7% | 3.6 | 9.8 / 5.7 | 35.5 / 19.1 | 334.0 | 658.0 |
| Single, mixed Q=8 | 50.4 | 46.2 | 1.09× | 22.9 | 49.6% | 3.0 | 9.8 / 5.7 | 33.3 / 17.1 | 199.2 | 375.3 |
| Single, mixed Q=32 | 86.7 | 69.6 | 1.25× | 39.6 | 56.8% | 4.8 | 9.8 / 5.7 | 55.1 / 33.9 | 278.2 | 655.4 |
| Tv=640, N=64, box | 34.5 | 23.5 | 1.47× | 16.7 | 71.1% | 0.1 | — | 23.4 / 16.7 | 899.1 | 1,556.6 |
| Tv=640, N=64, yes/no | 65.8 | 44.4 | 1.48× | 31.9 | 71.9% | 0.1 | — | 44.2 / 31.9 | 680.6 | 1,287.3 |
| Tv=640, N=256, box | 139.8 | 92.5 | 1.51× | 66.7 | 72.1% | 0.1 | — | 92.4 / 66.7 | 3,151.6 | 7,732.9 |
| Tv=640, N=256, yes/no | 300.7 | 177.4 | 1.69× | 127.5 | 71.9% | 0.2 | — | 177.3 / 127.5 | 2,679.9 | 5,670.7 |
| Tv=256, N=64, box | 33.0 | 22.3 | 1.48× | 16.1 | 72.4% | 0.1 | — | 22.2 / 16.1 | 653.1 | 592.2 |
| Tv=256, N=256, box | 124.1 | 81.0 | 1.53× | 64.4 | 79.6% | 0.1 | — | 80.9 / 64.4 | 2,950.1 | 2,981.4 |
| Tv=64, N=64, box | 32.2 | 21.8 | 1.48× | 15.8 | 72.7% | 0.1 | — | 21.7 / 15.8 | 401.3 | 446.1 |
| Tv=64, N=256, box | 120.4 | 78.9 | 1.53× | 63.3 | 80.2% | 0.1 | — | 78.8 / 63.3 | 1,112.6 | 1,440.6 |
| 4-frame clip, N=64, yes/no | 60.7 | 53.9 | 1.13× | 31.2 | 57.9% | 0.1 | — | 53.8 / 31.2 | 342.7 | 620.7 |
| 4-frame clip, N=256, yes/no | 273.0 | 235.8 | 1.16× | 124.9 | 53.0% | 0.2 | — | 235.7 / 124.9 | 1,355.1 | 2,203.7 |
| Follow-up, box Q=1 | 10.5 | 8.7 | 1.21× | 5.0 | 57.1% | 0.3 | — | 8.4 / 5.0 | 145.2 | 162.0 |
| Follow-up, mixed Q=1 | 10.5 | 8.7 | 1.21× | 5.0 | 57.2% | 0.3 | — | 8.4 / 5.0 | 145.1 | 161.7 |
| Follow-up, box Q=8 | 13.2 | 10.2 | 1.30× | 5.0 | 49.4% | 0.7 | — | 9.5 / 5.0 | 196.6 | 346.1 |
| Follow-up, mixed Q=8 | 19.1 | 14.9 | 1.29× | 5.6 | 38.0% | 1.0 | — | 13.8 / 5.6 | 175.3 | 342.2 |
| Follow-up, box Q=32 | 27.3 | 19.3 | 1.42× | 7.6 | 39.6% | 1.5 | — | 17.7 / 7.6 | 316.9 | 606.5 |
| Follow-up, mixed Q=32 | 56.1 | 39.9 | 1.41× | 22.3 | 56.0% | 2.6 | — | 37.3 / 22.3 | 254.9 | 561.8 |

## State split and mixer controls

Four image prefixes at layers 0, 14 and 30 supplied 6,291,456 FP32 state elements. The BF16 high part, rounded residual and second rounded residual reconstructed all of them exactly. The independent seed-product probe used BF16 vectors, BF16 tensor-core products with FP32 output, and a full FP32 matrix product with TF32 disabled. Of 393,216 rounded outputs, 393,213 were within one BF16 ULP and 393,149 were equal. Maximum BF16 distances by layer were 4, 29 and 1 ULP. The maximum absolute FP32 difference across the three layers was 0.00000573. The largest absolute FP32 difference using only the high part was 0.04752. [Per-layer reconstructions and projections](raw/item4_staged_micro.json)

For 256 independent 25-token branches on real layer-0 state and checkpoint projections of synthetic activations, the warmed mixer took 3.431 ms with the original FLA path, 3.006 ms with the packed FP32 recurrent walk and 1.198 ms with the packed WY kernel. For 64 independent 13-token branches the times were 0.414, 0.417 and 0.146 ms. For 32 shared 64-token branches the inline WY solve took 2.288 ms and staging state products took 0.922 ms. Original FLA chunk took 0.883 ms on that shared shape. Those controls motivated preserving its within-chunk transform and fusing the fork, update and readout of the branch state. The isolated mixer times exclude the input/output linear projections. The compiled WY, staged projection, chunk output and attention kernels target SM120a and use BF16-input, FP32-accumulating `mma.sync.aligned.m16n8k16`. [Isolated controls](raw/item4_staged_micro.json) · [Compiled instruction evidence](raw/item4_ptx.json)

On a warm graph with 704 cached prefix slots, Q=32 caused 1,536 MiB of logical destination writes in the old state gather and 704 MiB in the two KV gathers before concatenation. The isolated 24-layer state gather took 0.559 ms. The eight-layer KV gather-plus-concatenation took 0.606 ms. At Q=1 the same controls took 0.065 and 0.071 ms. The controls permit scratch reuse between layers and exclude downstream consumers. End-to-end improvements also reflect recurrent steps, final-state materialization, padded token operations and attention. [Fork controls](raw/item4_staged_micro.json)

On the final repeat, the production output-only chunk mixer took 0.302 ms versus 0.880 ms for FLA on 32 shared 64-token questions (2.91×). On this synthetic-activation probe, 8,383,643 of 8,388,608 post-normalization outputs matched the FLA BF16 bits and 8,386,796 were within one ULP. The maximum absolute difference was 0.125. Full-model decision and probability comparisons appear below. [Production mixer repeat](raw/item4_checked_micro.json)

## Where the time went

On the 256-image Tv640 yes/no trace, copy/concatenation fell from 66.72 to 2.62 ms. The original recurrence alone used 38.60 ms. The new WY kernel used 21.05 ms, packed convolution and gates 4.13 ms, and gated output normalization 1.83 ms. The packed attention kernel accounted for another 30.75 ms. The 256-clip trace retained 44.20 ms of copy/concatenation and 29.52 ms of FLA recurrence. On 32 mixed follow-ups, copy/concatenation fell from 13.31 to 1.17 ms. The output-only chunk kernel used 7.04 ms and packed attention used 4.02 ms. These are sums of matching kernels during representative replays. [Reference traces](raw/item4_reference_profile.json) · [Production traces](raw/item4_checked_profile.json)

### LM stage measurements against the roofline

Each cell gives measured GPU milliseconds / useful-work floor for the first query and image set in the corresponding workload. External CUDA events time disjoint stage ranges inside the graph. The measurement is the median of seven warmed replays. Dense covers the projections and MLP, GDN includes the mixer and its output gate, and attention covers the attention core. Rotary preparation, inter-layer residual operations, host work and vision are outside these LM ranges. The mixed Q=1 follow-up is the same box query as the Q=1 box row. The latency sweep above used changing questions and its own uninstrumented wall clock. [Stage records and ceiling percentages](raw/item4_stage_floors.json) · [Stage timings](raw/item4_checked_stages.json) · [Instrumentation](../../../scan/profile_branch.py)

| Workload | Dense projections and MLP | GDN | Attention | Heads |
|---|---:|---:|---:|---:|
| Tv=640, N=64, box | 17.18 / 12.51 | 3.36 / 2.54 | 1.89 / 0.97 | 0.39 / 0.09 |
| Tv=640, N=64, yes/no | 31.51 / 27.29 | 6.68 / 2.87 | 3.80 / 1.06 | 0.40 / 0.09 |
| Tv=640, N=256, box | 55.28 / 50.04 | 16.03 / 10.15 | 15.08 / 3.90 | 0.74 / 0.25 |
| Tv=640, N=256, yes/no | 111.13 / 109.17 | 27.32 / 11.49 | 30.52 / 4.27 | 0.75 / 0.25 |
| Tv=256, N=64, box | 17.20 / 12.51 | 3.36 / 2.54 | 0.74 / 0.45 | 0.34 / 0.05 |
| Tv=256, N=256, box | 55.36 / 50.04 | 16.03 / 10.15 | 3.08 / 1.79 | 0.58 / 0.12 |
| Tv=64, N=64, box | 17.21 / 12.51 | 3.36 / 2.54 | 0.30 / 0.18 | 0.32 / 0.04 |
| Tv=64, N=256, box | 55.37 / 50.04 | 16.06 / 10.15 | 1.17 / 0.71 | 0.52 / 0.08 |
| 4-frame clip, N=64, yes/no | 30.49 / 27.29 | 14.85 / 2.87 | 2.80 / 0.45 | 0.31 / 0.03 |
| 4-frame clip, N=256, yes/no | 111.97 / 109.17 | 77.57 / 11.49 | 19.55 / 1.82 | 0.49 / 0.08 |
| Single, box Q=1 | 17.55 / 11.62 | 5.30 / 0.26 | 1.36 / 0.07 | 0.28 / 0.04 |
| Follow-up, box Q=1 | 6.18 / 4.88 | 0.63 / 0.04 | 0.60 / 0.02 | 0.27 / 0.04 |
| Single, box Q=8 | 19.22 / 13.34 | 5.80 / 0.30 | 1.49 / 0.09 | 0.32 / 0.04 |
| Follow-up, box Q=8 | 6.78 / 4.90 | 1.11 / 0.08 | 0.62 / 0.03 | 0.31 / 0.04 |
| Single, mixed Q=8 | 23.41 / 17.02 | 7.64 / 0.39 | 2.12 / 0.14 | 0.32 / 0.04 |
| Follow-up, mixed Q=8 | 9.96 / 5.60 | 2.93 / 0.16 | 1.26 / 0.07 | 0.31 / 0.04 |
| Single, box Q=32 | 24.24 / 19.10 | 7.50 / 0.43 | 2.58 / 0.16 | 0.38 / 0.04 |
| Follow-up, box Q=32 | 12.81 / 7.68 | 2.80 / 0.21 | 1.75 / 0.09 | 0.36 / 0.04 |
| Single, mixed Q=32 | 42.52 / 33.81 | 14.56 / 0.77 | 4.97 / 0.35 | 0.38 / 0.04 |
| Follow-up, mixed Q=32 | 26.90 / 22.39 | 9.69 / 0.54 | 4.08 / 0.29 | 0.37 / 0.04 |

For the fixed 256-image yes/no question, the ratios of floor to measured time were 98.2% for dense, 42.0% for GDN and 14.0% for attention. On the 256-clip case, the GDN range took 77.57 ms against 11.49 ms of useful work. On 32 mixed follow-ups the GDN range took 9.69 ms against 0.54 ms. The image attention scan traverses prefix KV twice. The native clip path retains the split-attention intermediates and the original recurrent arithmetic that met the clip probability gate. [Stage floors and measured times](raw/item4_stage_floors.json)

## Decision agreement

| Check | Result on production path |
|---|---|
| Tiny FP32 scan, CPU | Both masks passed, including cached versus fresh, clip API, repeatability and image isolation. [CPU log](raw/item4_cpu.log) |
| Real-weight scan, G4 | 234/234 decisions and all calibrated continuous gates passed in each of the standard and optimized merged runs. [Standard](raw/item4_checked_real.log) · [Merged](raw/item4_checked_merged.log) |
| Scan benchmark equivalence | 480/480 decisions across 20 groups of 24. Maximum absolute differences were 0.023087 for typed probability, 0.024397 for presence, 0.155096 for normalized box coordinates and 0.083784 for predicted IoU. [Scan sweep](raw/item4_checked_equiv.json) |
| Original versus production question path | 618/618 decisions on cold calls and retained-prefix follow-ups, covering both masks, Q=1/8/32, changing question order and unequal block counts across three images. Maxima were 0.013973 for typed probability, 0.008826 for presence and 0.007565 for normalized coordinates. [Paired records](raw/item4_checked_questions.json) |
| Full speed diagnostic | All five BF16 engine variants matched 930/930 shipped unmerged BF16 decisions on 60 images. Both the shipped reference and production BF16 matched 928/930 FP32 decisions. Separately, the existing language-only FP8 and language-plus-vision FP8 modes matched 926 and 923 of the shipped BF16 decisions. [Diagnostic](raw/item4_checked_diag.json) |

The reported improvements use BF16 and the existing calibrated tolerances. The optimized kernels target supported SM120 BF16 layers and short padded questions of up to 128 tokens. Native-video banks use the packed FLA path. The G4 results cover this checkpoint and the image, frame and clip distributions above. [Validation totals](raw/item4_verification.json) · [Production clip controls](raw/item4_checked_clip.json)

An independent run of the final dispatcher reproduced every scan-equivalence metric, and the merged suite again passed 234/234 decisions and its continuous gates. All ten cached timing medians reproduced within 0.94%. The 256-image yes/no query took 177.65 ms on the repeat and the 256-clip query took 235.93 ms. [Repeat sweep](raw/item4_final_equiv.json) · [Repeat merged run](raw/item4_final_merged.log) · [Repeat timings](raw/item4_final_cached.json)

## Reproduction

From the worktree root, run the sync and then the detached benchmark chain. The runner takes an exclusive lock, waits for an empty GPU process list before each invocation, and archives each result and log. The reference and production sweeps select their dispatch through `GLANCE_BRANCH_MODE`. [Runner](../../../scan/vm/gdn_branch.sh) · [Raw result archive](https://huggingface.co/datasets/Vision-Decision/collection-scan-results/tree/main/engine_opt/04_gdn_branch)

```sh
MONO=~/Developer/decision-vison/.claude/worktrees/engine-opt bash scan/vm/sync.sh engopt
bash scan/vm/bg.sh engopt item4_reproduce 'bash scan/vm/gdn_branch.sh reference_cached reference_speed reference_profile checked_real checked_merged checked_equiv checked_questions checked_diag checked_cached checked_speed checked_profile checked_micro checked_clip checked_stages'
```

## Authorship

Codex wrote the engine changes, Triton kernels, benchmark runner, profiling and equivalence probes, and reports, and ran the tests and measurements. The reviewer provided the configured engopt VM, checkpoint, data and prior serving baselines. The changes are in the worktree at base commit `865cc67`.
