# C0 · G4 ceiling and equivalence

Glance has about **1.88 s of modeled headroom** on the largest fresh scan and **1.60 s** on a 64-client burst. The fresh 128-frame headline remains 931.1 ms against vLLM at 638.3 ms. The modeled floor is 450.6 ms. Closing the gap requires work on both the LM/readout and the input/vision stages. [Complete workload map and attack order](../CEILING.md) · [Hardware measurements and formulas](../ROOFLINE.md)

## What changed

C0 measures the item-1 checkout, `73519e8`, without modifying inference code or weights. The benchmark additions cover reproducible hardware rates and useful-token arithmetic, independent cold client bursts, cached SGLang box output and a warmed cached-branch kernel trace. Each GPU process ran in a serialized detached chain with an empty compute-process list before launch. COCO and ScreenSpot-Pro use the existing resize rules. The speed sweep used 50 measured calls, GUI used 30, video and client bursts used 12, cached Glance used 30, and packed fresh timings are the existing 10-call item-1 series. [Runner](../../../scan/vm/ceiling.sh) · [Shape collection](../../../speed/ceiling_shapes.py) · [Floor calculation](../../../speed/ceiling_model.py)

The G4 achieved **401.73 BF16 and 831.88 FP8 TFLOP/s** at 8,192³, **339.88 BF16 TFLOP/s** at the 4,096-row MLP shape, and **1,463.21 GB/s** device-memory copy. Pinned/pageable transfers reached 56.87/27.80 GB/s. One warmed single-kernel graph replay cost 2.23 µs amortized. The current LM streams 7.138 GB of dense BF16 weights per pass, a 4.88 ms weight-read floor. [Raw microbenchmark samples](raw/c0_peaks.json)

## Selected latency measurements

All values are milliseconds. The prior column gives the archived BF16 fused speed/scan run or the item-1 legacy 16k-chunk control. The fresh rows reuse the item-1 measurements. Differences in the other rows are between separate benchmark runs. [Prior speed](../../sprint_speed/raw/glance5fused_g4.json) · [Prior cached scan](../../scan/raw/scan-a/bench_g4.json) · [Item-1 before](../01_varlen_prefill/raw/item1_before.json) · [41 current workload rows and paired stage times](../CEILING.md)

| Workload | Prior Glance | Current Glance | Floor | % of ceiling | vLLM BF16 | SGLang BF16 |
|---|---:|---:|---:|---:|---:|---:|
| Image, Q=1 | 43.0 | 42.5 | 17.5 | 41.2% | 166.5 | 187.3 |
| Image, Q=8 | 44.9 | 45.0 | 19.1 | 42.5% | 221.0 | 395.8 |
| Image, Q=32 | 61.0 | 61.0 | 24.8 | 40.6% | 334.0 | 658.0 |
| 4k screenshot, Q=1 | 298.7 | 306.0 | 165.8 | 54.2% | 353.7 | 489.0 |
| 64 frames, one box each | 1,160.7 | 1,157.0 | 435.8 | 37.7% | 933.1 | 1,046.3 |
| 128 new frames, Tv=128 | 2,586.2 | 931.1 | 450.6 | 48.4% | 638.3 | 823.3 |
| 256 new frames, Tv=256 | 4,820.3 | 3,693.7 | 1,810.4 | 49.0% | 2,487.7 | 3,042.2 |
| 256 cached images, box | 163.7 | 139.9 | 66.7 | 47.6% | 3,151.6 | 7,732.9 |
| 256 cached four-frame clips, yes/no | 288.9 | 272.9 | 124.9 | 45.8% | 1,355.1 | 2,203.7 |
| One cached-image follow-up | 10.8 | 10.8 | 5.0 | 45.9% | 145.2 | 162.0 |
| 64 independent cold clients | — | 2,698.7 | 1,100.9 | 40.8% | 1,835.9 | 2,301.9 |

The original SGLang cold 64-client burst reused 663.0 prompt tokens per request and took 953.6 ms. Awaiting and checking the cache flush brought reuse to 0.0 tokens and gave 2,301.9 ms. The comparison table uses the corrected run. [Original](raw/c0_clients_sglang.json) · [Corrected](raw/c0_sglang_clean.json)

## What to optimize

1. **Fresh scan.** At Tv=256, N=256 the existing 3,693.7 ms call spends 243.1 ms in prep, 1,174.7 ms in vision/H2D and 2,274.6 ms in LM/readout. Their modeled floors are 0, 490.3 and 1,320.2 ms. Item 2 can address prep and vision. The larger remaining stage gap belongs to packed LM prefill (item 5). At the headline point even a zero-cost host prep and a 107.1 ms vision/H2D stage leave 693.4 ms with the current LM. [Packed measurements](../01_varlen_prefill/raw/item1_after.json) · [Floor breakdown](raw/c0_floors.json)
2. **Client scheduling.** Glance processes 64 cold images serially. Batch scheduling can amortize weight movement and dispatch, while prefill arithmetic still grows with the number and size of images. The ideal batched floor is 1,100.9 ms. Item 6 precedes an HTTP server experiment in item 9. [Glance bursts](raw/c0_clients.json)
3. **Video and long GUI prefill.** The existing speed engine needs 1,157.0 ms for 64 separate frame/box requests, versus a 435.8 ms floor. For a single 4k screenshot it needs 306.0 ms versus 165.8 ms. Vision attention alone accounts for 63.1 ms of that screenshot floor. [Video sweep](raw/c0_video.json) · [GUI sweep](raw/c0_gui4096.json)
4. **Cached branch movement.** At N=256, Tv=640, the fixed yes/no profile spends 122.5 ms in GEMMs, 67.1 ms in copy/concatenation kernels and 38.6 ms in the FLA recurrent kernel. The recurrent kernel is already fused. A candidate experiment is to fuse the scatter, gather and adjacent projection layouts around that kernel, profiling the residual category first. Item 7 also addresses the 48 MiB FP32 recurrent state per prefix. [Kernel trace](raw/c0_profile_bank.json) · [Bank sizes and rotating-query medians](raw/c0_cached.json)

## Real-weight test

The previous GPU test failed its FP32-style relative logit gates at item 1. The calibration run exercised only the shipped BF16 forward, using groups of one, two and the entire fixture, with padding multiples of 1 or 64. It covered both attention masks, plain and query-first images, repeated unequal prefixes, clips, and the API prompts. Ten layout pairs on each of 24 fixtures gave **1,260/1,260 identical presence/typed decisions**. Raw reference-only logit spreads reached 0.17066 for presence and 0.92095 for IoU. The test freezes twice the measured per-output spread. It evaluates thresholded presence and valid-label argmax separately from the continuous gates. [Calibration JSON](raw/c0_reference_noise.json) · [Reference-only invocation](raw/c0_reference_noise.log) · [Previous failure](../01_varlen_prefill/raw/item1_scan_real.log) · [Test and frozen limits](../../../scan/test_scan_equivalence.py)

| Output | Max reference-to-reference spread | Acceptance limit | Max observed in real suite |
|---|---:|---:|---:|
| Presence probability | 0.015852 | 0.031704 | 0.01032 |
| Normalized box coordinate | 0.212472 | 0.424945 | 0.2018 |
| IoU probability | 0.203709 | 0.407418 | 0.1440 |
| Typed-label probability | 0.011677 | 0.023353 | 0.007088 |
| Box coordinate, repeated unequal-prefix fixture | 0.003343 | 0.006686 | 0.006256 |
| IoU probability, repeated unequal-prefix fixture | 0.006696 | 0.013392 | 0.006576 |

The largest coordinate spread occurred on a query-first bidirectional image. The box head selects a patch by argmax before regressing its edges. Previous per-patch traces on these fixtures showed that nearly tied centers can exchange rank when reference padding changes. The tighter unequal-prefix limits keep that fixture sensitive to small localization errors. The envelopes describe these particular inputs and kernels. [Per-patch layout control](../01_varlen_prefill/raw/item1_reference_layout.json)

The normal real-weight test passed with **234/234** decisions across both masks and the API checks. The tiny FP32 suite also passed. For the negative control, the VM stage runner moved the packed GPU readout one token early. All 234 binary/typed decisions still matched, but 12 coordinate or IoU probability comparisons failed. On the causal unequal-prefix fixture the coordinate error was **0.02424** against a **0.00669** limit, and the IoU probability error was **0.08321** against **0.01339**. The full fault run reached its numerical gates and exited 1. This shows what the probability and localization checks catch even when classification decisions agree. [CPU](raw/c0_cpu.log) · [Normal real test](raw/c0_real.log) · [Injected error](raw/c0_fault.log)

Reproduce on engopt with `bash scan/vm/ceiling.sh reference_noise real fault photo_equiv diag`. The fault applies only in its own Python process. An earlier injection also modified the CPU index-layout probe and tripped its structural assertion immediately. The retained negative-control measurement restricts the injection to GPU execution. [Earlier structural rejection](raw/c0_fault_structural.log) · [Stage runner](../../../scan/vm/ceiling.sh)

The independent graph-enabled COCO/clip scan checked 24 units in each of 20 path/question combinations. All 480 decision comparisons agreed. Its largest presence-probability delta was 0.07797, on the 256-token fresh-frame dog query. Separately, the 60-image speed diagnostic completed and reported **928/930 fused BF16 decisions** against the shipped reference. The same two-decision shortfall was recorded at item 1. The unfused plain engine agreed on 930/930. The broader fused-engine 100% target remains open. [Graph-enabled scan](raw/c0_photo_equiv.json) · [Speed diagnostic](raw/c0_diag.json) · [Item-1 diagnostic](../01_varlen_prefill/raw/item1_diag.json)

## Reproduction and authorship

Sync from the worktree with `bash scan/vm/sync.sh engopt`. On the configured VM, run `bash scan/vm/ceiling.sh peaks speed gui2048 gui4096 video cached clients clients_vllm sglang_clean sglang_bank shapes profile_bank`. For the local floor calculation run `python -m speed.ceiling_model outputs/engine_opt/00_ceiling/raw`. The frozen input manifests are encoded as multiplicities, merged visual-token counts, prefix lengths and per-question token lengths in the shape JSON. The prior fresh measurements and published sprint vLLM results remain linked at their original paths. [Call shapes](raw/c0_shapes.json)

All 49 raw JSON/log files are present locally. The configured result archive holds 46 in the primary dataset and three logs in its private fallback (cached Glance and the original vLLM/SGLang client runs). [Local evidence](raw)

### Authorship

Codex wrote the measurement harnesses, cache-reset repair, floor model, reports and test changes, and ran the measurements. The reviewer supplied the existing engopt VM, serving environments, checkpoint and item-1 checkout. Codex added the missing PyAV 19.0.2 wheel to the existing SGLang environment to encode the four-frame clip inputs after the initial clip launch failed at import. [Initial launch](raw/c0_sglang_bank_missing_av.log) · [Repair log](raw/c0_bank_repair.log)
