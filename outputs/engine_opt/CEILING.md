# G4 ceiling map

## Final state (frozen at 8bc7d58)

**vLLM FP8+MTP (lossy) still wins one of the 24 full-output cells with a matched FP8+MTP run.** On a 4096-token ScreenSpot-Pro screenshot with one box question, vLLM takes **262.4 ms**, 15.5 ms faster than BF16 Glance at **277.9 ms**. Glance leads both BF16 servers in all 65 workload cells. The GUI value is the last committed measurement of that workload, from item 3. [Glance GUI run](03_lm_prefill/raw/item3_gui4096.json) · [vLLM full-box sweep](../sprint_speed/raw/serve_vllm_fp8mtp_g4.json)

Item 7 (`a5d7087`) changed the outcome on **64 frames × one box per frame**. Glance went from 773.6 to **587.7 ms**, ahead of FP8+MTP at **695.7 ms**. On **fresh Tv256, N=256**, the earlier deficit was against **BF16** vLLM (2497.5 vs 2482.4 ms). Item 7 measured **2420.5 vs 2495.9 ms** with full encoder-cache resets. The committed FP8+MTP scan sweep covers cached images only. [Video before](07_frames/raw/item7_before_video.json) · [Video after][video7] · [Tuned video][video-tuned] · [Earlier fresh Glance](03_lm_prefill/raw/item3_accel_production.json) · [Earlier fresh vLLM](02_vision_prep/raw/item2_cold_vllm.json) · [Final fresh Glance][fresh7] · [Final fresh vLLM][fresh-v] · [FP8+MTP scan coverage][bank-tuned]

Times are warm median milliseconds on one RTX PRO 6000 Blackwell. Read each series in the order listed at left. Q counts questions per image, N counts images or clips, Tv is the visual-token budget, and C counts simultaneous clients. The family names link to the per-cell tables below. The percentage is 100 × modeled latency floor / measured Glance latency, calculated before rounding. “Final” uses the most recent production measurement for each workload. The frame and burst sweeps enabled the optional SM120 BF16 kernel for eligible packed prefills. Glance uses BF16 weights throughout. The vLLM and SGLang timings cover the same requested task on the generative Qwen3.5-4B model. The equivalence tests compare Glance with its own shipped checkpoint. [Hardware measurements](00_ceiling/raw/c0_peaks.json) · [Floor model](ROOFLINE.md)

| Workload and order | Original Glance | Final Glance | Floor ms (% of ceiling) | vLLM BF16 | SGLang BF16 | Best measured full-output vLLM | Verdict |
|---|---|---|---|---|---|---|---|
| **[One image, ≤640 visual tokens](#single-image-640-visual-tokens)** · box Q=1/8/32<br>Mixed Q=8/32 | [43.0 / 44.9 / 61.0][sprint]<br>[53.5 / 90.2][sprint] | [37.7 / 38.5 / 48.9][box9]<br>[46.2 / 69.2][mixed9] | [17.5 (46.3%) / 19.1 (49.6%) / 24.8 (50.7%)][floor9]<br>[22.9 (49.5%) / 39.6 (57.2%)][floor9] | [166.5 / 221.0 / 334.0][v-bf16]<br>[199.2 / 278.2][v-bf16] | [187.3 / 395.8 / 658.0][sg-speed]<br>[375.3 / 655.4][sg-speed] | [99.2 / 134.0 / 229.2][v-fp8] FP8+MTP<br>[122.1 / 177.0][v-fp8] FP8+MTP | Glance leads in all five. |
| **[GUI, box Q=1/8/32](#screenspot-pro-screenshots)** · 2048 tokens<br>4096 tokens | [127.7 / 130.3 / 154.6][sprint]<br>[298.7 / 303.7 / 331.2][sprint] | [119.2 / 122.6 / 146.1][gui2k]<br>[277.9 / 282.9 / 311.9][gui4k] | [67.9 (57.0%) / 69.5 (56.7%) / 75.5 (51.7%)][c0-floor]<br>[165.8 (59.7%) / 167.9 (59.3%) / 174.9 (56.1%)][c0-floor] | [226.2 / 311.7 / 572.2][v-bf16]<br>[353.7 / 418.4 / 566.4][v-bf16] | [301.7 / 566.9 / 1406.2][sg-speed]<br>[489.0 / 1305.1 / 2398.4][sg-speed] | [149.5 / 208.0 / 418.6][v-fp8] FP8+MTP<br>[262.4 / 299.5 / 435.7][v-fp8] FP8+MTP | Glance leads both BF16 servers in all six. FP8+MTP leads at 4096, Q=1. |
| **[Prefill-heavy video](#video-style-speed-workload-one-box-question-per-frame)** · 32/64 frames × 1 box each | [575.6 / 1160.7][sprint] | [290.1 / 587.7][video7] | [214.7 (74.0%) / 429.7 (73.1%)][video-floor] | [549.4 / 928.1][video-v] | [666.3 / 1052.0][video-sg] | [387.1 / 695.7][video-tuned] FP8+MTP | Glance leads on both frame counts. |
| **[Cached scan, N=64/256](#cached-scan-one-new-question-over-n-independent-prefixes)** · Tv640 box<br>Tv640 yes/no<br>Tv256 box<br>Tv64 box<br>4-frame v96 yes/no<br>4-frame v48 yes/no<br>2-frame v96 yes/no | [39.9 / 163.7][bank0]<br>[68.5 / 313.3][bank0]<br>[38.4 / 149.8][bank0]<br>[37.7 / 145.2][bank0]<br>[63.6 / 288.9][bank0]<br>[63.0 / 280.8][bank0]<br>[63.0 / 280.7][bank0] | [23.7][bank8] / [96.7][bank9]<br>[45.4][bank8] / [175.4][bank9]<br>[22.5 / 84.5][bank8]<br>[22.0 / 82.6][bank8]<br>[55.7 / 241.4][bank8]<br>[55.0 / 234.9][bank8]<br>[54.9 / 234.3][bank8] | [17.3 (73.0%)][bank-floor] / [69.0 (71.4%)][floor9]<br>[32.5 (71.6%)][bank-floor] / [129.9 (74.0%)][floor9]<br>[16.7 (74.3%) / 66.8 (79.0%)][bank-floor]<br>[16.4 (74.6%) / 65.7 (79.5%)][bank-floor]<br>[31.8 (57.1%) / 127.2 (52.7%)][bank-floor]<br>[31.7 (57.6%) / 126.8 (54.0%)][bank-floor]<br>[31.7 (57.8%) / 126.7 (54.1%)][bank-floor] | [899.1 / 3151.6][bank-v]<br>[680.6 / 2679.9][bank-v]<br>[653.1 / 2950.1][bank-v]<br>[401.3 / 1112.6][bank-v]<br>[342.7 / 1355.1][bank-v]<br>[236.0 / 907.7][clip-v]<br>[214.2 / 823.3][clip-v] | [1556.6 / 7732.9][bank-sg]<br>[1287.3 / 5670.7][bank-sg]<br>[592.2 / 2981.4][bank-sg]<br>[446.1 / 1440.6][bank-sg]<br>[620.7 / 2203.7][bank-sg]<br>[556.5 / 2002.2][clip-sg]<br>[479.1 / 1784.3][clip-sg] | [720.7 / 2672.2][bank-tuned] FP8+MTP<br>[582.8 / 2292.3][bank-tuned] FP8+MTP<br>[479.4][bank-tuned] FP8+MTP / [2950.1][bank-v] BF16<br>[401.3 / 1112.6][bank-v] BF16<br>[339.6][clip-mb16] BF16 block 16 / [1355.1][bank-v] BF16<br>[236.0 / 907.7][clip-v] BF16<br>[214.2 / 823.3][clip-v] BF16 | Glance leads on all 14 warm queries. |
| **[Fresh scan, N=32/64/128/256](#fresh-scan-one-yesno-question-over-n-new-frames)** · Tv128<br>Tv256 | [241.8 / 473.1 / 951.2 / 1907.3][fresh0]<br>[468.4 / 941.9 / 1883.4 / 3764.7][fresh0] | [142.9 / 287.5][fresh7] / [581.5][fresh9] / [1165.3][fresh7]<br>[301.7 / 602.1 / 1207.5 / 2420.5][fresh7] | [111.2 (77.8%) / 222.5 (77.4%)][fresh-floor] / [444.9 (76.5%)][floor9] / [889.8 (76.4%)][fresh-floor]<br>[223.3 (74.0%) / 446.5 (74.2%) / 893.0 (74.0%) / 1786.1 (73.8%)][fresh-floor] | [177.9 / 332.1 / 637.8 / 1252.8][fresh-v]<br>[322.8 / 629.4 / 1244.7 / 2495.9][fresh-v] | [259.3 / 431.4 / 815.3 / 1572.9][fresh-sg]<br>[461.0 / 848.4 / 1567.6 / 2997.1][fresh-sg] | BF16 column at both budgets | Glance leads both BF16 servers in all eight. |
| **[One cached image, Q=1/8/32](#follow-up-questions-on-one-cached-image)** · Box<br>Mixed | [10.8 / 13.6 / 27.9][sprint]<br>[10.8 / 19.6 / 57.0][sprint] | [8.8 / 10.3 / 19.6][box9]<br>[8.8 / 15.0 / 39.9][box9] | [5.0 (56.3%) / 5.0 (48.9%) / 7.6 (38.8%)][floor9]<br>[5.0 (56.3%) / 5.6 (37.7%) / 22.3 (56.0%)][floor9] | [145.2 / 196.6 / 316.9][v-bf16]<br>[145.1 / 175.3 / 254.9][v-bf16] | [162.0 / 346.1 / 606.5][sg-speed]<br>[161.7 / 342.2 / 561.8][sg-speed] | [78.3 / 112.9 / 211.8][v-fp8] FP8+MTP<br>[78.5 / 96.6 / 157.4][v-fp8] FP8+MTP | Glance leads on all six follow-ups. |
| **[Concurrent clients, C=1/4/16/64](#concurrent-clients)** · Cold box<br>Cold yes/no<br>Eight mixed questions<br>Cached box, resent pixels<br>GUI box, ≤2048 tokens<br>32 fresh frames, yes/no | [42.6 / 170.2 / 677.9 / 2698.7][c0-burst]<br>—<br>—<br>—<br>—<br>— | [32.2 / 104.0 / 362.7 / 1424.3][burst7]<br>[32.2 / 104.3 / 370.3 / 1456.3][burst7]<br>[45.3 / 141.1 / 534.0 / 2144.3][burst7]<br>[8.7 / 17.3 / 47.4 / 173.1][burst7]<br>[98.3 / 354.1 / 1384.3 / 5515.6][burst7]<br>[157.5 / 609.3 / 2422.7 / 9695.5][burst7] | [17.2 (53.4%) / 67.8 (65.2%) / 272.8 (75.2%) / 1084.2 (76.1%)][burst-floor]<br>[17.4 (54.2%) / 68.8 (65.9%) / 276.6 (74.7%) / 1099.2 (75.5%)][burst-floor]<br>[22.5 (49.6%) / 89.5 (63.4%) / 358.0 (67.0%) / 1425.9 (66.5%)][burst-floor]<br>[5.0 (57.2%) / 5.2 (30.1%) / 6.2 (13.1%) / 19.8 (11.5%)][burst-floor]<br>[67.2 (68.4%) / 267.2 (75.5%) / 1069.7 (77.3%) / 4267.5 (77.4%)][burst-floor]<br>[112.7 (71.5%) / 447.8 (73.5%) / 1788.4 (73.8%) / 7153.5 (73.8%)][burst-floor] | [166.2 / 264.9 / 578.6 / 1828.0][burst-v]<br>[39.5 / 111.1 / 381.7 / 1505.3][burst-v]<br>[199.6 / 351.1 / 979.4 / 3499.5][burst-v]<br>[140.3 / 182.0 / 253.3 / 1019.0][burst-v]<br>[233.7 / 521.2 / 1607.2 / 5983.4][burst-v]<br>[181.1 / 635.2 / 2552.3 / 10603.3][burst-v] | [188.1 / 322.8 / 715.4 / 2330.0][burst-sg]<br>[53.8 / 157.2 / 508.6 / 1946.8][burst-sg]<br>[378.3 / 888.8 / 2734.1 / 12305.1][burst-sg]<br>[167.1 / 248.2 / 452.8 / 1801.8][burst-sg]<br>[297.6 / 705.5 / 2221.4 / 8640.1][burst-sg]<br>[248.9 / 809.8 / 3069.3 / 12724.0][burst-sg] | BF16 column for all bursts | Glance leads both BF16 servers in all 24 bursts. |

Original Glance uses the pre-item-1 fused sprint, scan bank and tuned 4k-chunk fresh runs. The older 16k-chunk fresh control took 2586.2 ms at Tv128/N128 and 4820.3 ms at Tv256/N256. C0 measured only cold-box concurrency. Dashes denote unavailable C0 timings. Item 6 introduced those five burst workloads and measured a serial control for each. [Sprint][sprint] · [Scan bank][bank0] · [Tuned fresh][fresh0] · [16k legacy control](01_varlen_prefill/raw/item1_before.json) · [C0 bursts][c0-burst] · [Item 6 serial controls](06_scheduler/raw/item6_warm_serial.json)

The best-vLLM column compares configurations with the same requested output and cache protocol. A box allows up to 24 generated tokens. A yes/no scan returns one token with logprobs. FP8+MTP combines quantized inference with speculative decoding. “BF16 column” identifies the only configuration in the matched fresh and burst sweeps. The clip comparison includes the tested Mamba block sizes. Cache-hit scans exclude encoding, and client bursts end at the last complete answer. [BF16 MTP sprint](../sprint_speed/raw/serve_vllm_mtp_g4.json) · [FP8+MTP sprint][v-fp8] · [Cached FP8+MTP][bank-tuned] · [Block 16 clip sweep][clip-mb16] · [Block 64 clip sweep](../scan/raw/scan-e/e1_vllm_mb64.json)

**Known limit: bf16 box-centre spread.** Item 9 made the box readout itself deterministic (bit-identical centre scores, patches, boxes and IoU on 210 replays with fixed hidden states), but the bf16 backbone still perturbs the hidden states enough that 28 of 420 box centres switch across five equivalent batch/padding layouts of the shipped reference (normalized-coordinate spread 0.212). Running the whole backbone in fp32 holds all 420 centres (spread 0.000357) at 3.12x the latency of the bf16 reference forward. The fp32 backbone is a measured control today, not an engine mode; it is the candidate opt-in exact mode. Presence and typed decisions are unaffected (930/930). [Item 9](09_numerics/summary.md)

**Largest remaining gaps**

- **Absolute, 64 clients × 32 fresh frames.** Glance takes 9695.5 ms against a 7153.5 ms floor, a 2542.0 ms gap. Vision and upload account for 1431.2 ms above their floor and packed LM/readout for 1062.1 ms. The tower profile identifies per-image attention and patch convolution among the remaining costs. [Burst stages][burst7] · [Burst floors][burst-floor] · [Tower profile](07_frames/raw/item7_final_kernels.json)
- **Single fresh scan, Tv256/N256.** The 634.4 ms gap includes 353.1 ms in vision/upload and 249.0 ms in LM/readout. Faster patch projections changed outputs on the numerical checks, so this path retains the reference patch convolution. [Fresh stages][fresh7] · [Stage floors][fresh-floor] · [Patch controls](07_frames/raw/item7_twod_conv_binary.json)
- **Relative, 64 cached-box clients resending pixels.** A 19.8 ms floor against 173.1 ms gives 11.5% of ceiling. The CPU spends 87.8 ms preparing and hashing resent images, and LM/readout takes 82.9 ms. The model assigns zero time to host preparation. [Pixel burst][burst7] · [Burst floors][burst-floor] · [Retained-ID control](06_scheduler/raw/item6_production_ids.json)
- **Cached clips, 256 four-frame v96 prefixes.** The query takes 241.4 ms against a 127.2 ms floor (52.7%). On a fixed question the reference-compatible GDN continuation takes 77.5 ms against 11.5 ms, the largest profiled stage gap. [Changing questions][bank8] · [Per-call floors][bank-floor] · [Fixed-question stages](08_bank_memory/raw/item8_final_stages.json)

[sprint]: ../sprint_speed/raw/glance5fused_g4.json
[v-bf16]: ../sprint_speed/raw/serve_vllm_async_g4.json
[v-fp8]: ../sprint_speed/raw/serve_vllm_fp8mtp_g4.json
[sg-speed]: 00_ceiling/raw/c0_sglang_clean.json
[c0-floor]: 00_ceiling/raw/c0_floors.json
[c0-burst]: 00_ceiling/raw/c0_clients.json
[box9]: 09_numerics/raw/item9_release_speed.json
[mixed9]: 09_numerics/raw/item9_release_mixed.json
[floor9]: 09_numerics/raw/item9_floors.json
[gui2k]: 03_lm_prefill/raw/item3_gui2048.json
[gui4k]: 03_lm_prefill/raw/item3_gui4096.json
[video7]: 07_frames/raw/item7_checked_video.json
[video-floor]: 07_frames/raw/item7_video_floors.json
[video-v]: 07_frames/raw/item7_matched_video_vllm.json
[video-sg]: 07_frames/raw/item7_final_video_sglang.json
[video-tuned]: 07_frames/raw/item7_matched_video_tuned.json
[bank0]: ../scan/raw/scan-a/bench_g4.json
[bank8]: 08_bank_memory/raw/item8_release_after.json
[bank9]: 09_numerics/raw/item9_release_cached.json
[bank-floor]: 08_bank_memory/raw/item8_floors.json
[bank-v]: ../scan/raw/scan-a/vllm_bf16.json
[bank-sg]: 00_ceiling/raw/c0_sglang_bank.json
[bank-tuned]: ../scan/raw/scan-a/vllm_fp8mtp.json
[clip-v]: 08_bank_memory/raw/item8_test_vllm.json
[clip-sg]: 08_bank_memory/raw/item8_test_sglang.json
[clip-mb16]: ../scan/raw/scan-e/e1_vllm_mb16.json
[fresh0]: ../scan/raw/scan-e/e1b_ours_fresh_c4096.json
[fresh7]: 07_frames/raw/item7_final_fresh.json
[fresh9]: 09_numerics/raw/item9_release_fresh.json
[fresh-floor]: 02_vision_prep/raw/item2_floors.json
[fresh-v]: 07_frames/raw/item7_final_fresh_vllm.json
[fresh-sg]: 07_frames/raw/item7_final_fresh_sglang.json
[burst7]: 07_frames/raw/item7_final_scheduled.json
[burst-floor]: 06_scheduler/raw/item6_floors.json
[burst-v]: 07_frames/raw/item7_final_scheduler_vllm.json
[burst-sg]: 07_frames/raw/item7_final_scheduler_sglang.json

## Measurement history

C0 measured the item 1 checkout (monorepo `73519e8`). Its timing map and ranking are historical. The tables below carry forward the last production run for each workload. Item 3 supplies the GUI rows, item 7 the video, burst and fresh rows, and item 8 the compact-bank rows. Item 9 supersedes the earlier BF16 rows for the ≤640-token single image, follow-ups, Tv640/N256 bank queries and Tv128/N128 fresh scan. Its floor includes the fixed-size FP32 projections and FP64 center reduction. The resource in parentheses contributes the most time to the modeled stage sum. [C0 ranking](00_ceiling/raw/c0_ranked_gaps.json) · [Item 9 paired measurements](09_numerics/summary.md) · [Item 9 floor][floor9]

## Single image, ≤640 visual tokens

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---|
| box Q=1 | 17.5 (bf16) | 37.7 | 46.3% | 166.5 | 187.3 | LM/readout +13.7 |
| box Q=8 | 19.1 (bf16) | 38.5 | 49.6% | 221.0 | 395.8 | LM/readout +13.0 |
| box Q=32 | 24.8 (bf16) | 48.9 | 50.7% | 334.0 | 658.0 | LM/readout +16.6 |
| mixed Q=8 | 22.9 (bf16) | 46.2 | 49.5% | 199.2 | 375.3 | LM/readout +16.2 |
| mixed Q=32 | 39.6 (bf16) | 69.2 | 57.2% | 278.2 | 655.4 | LM/readout +21.4 |

## ScreenSpot-Pro screenshots

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---|
| 2048 tokens, Q=1 | 67.9 (bf16) | 119.2 | 57.0% | 226.2 | 301.7 | LM/readout +30.8 |
| 2048 tokens, Q=8 | 69.5 (bf16) | 122.6 | 56.7% | 311.7 | 566.9 | LM/readout +31.8 |
| 2048 tokens, Q=32 | 75.5 (bf16) | 146.1 | 51.7% | 572.2 | 1,406.2 | LM/readout +48.8 |
| 4096 tokens, Q=1 | 165.8 (bf16) | 277.9 | 59.7% | 353.7 | 489.0 | LM/readout +66.9 |
| 4096 tokens, Q=8 | 167.9 (bf16) | 282.9 | 59.3% | 418.4 | 1,305.1 | LM/readout +69.1 |
| 4096 tokens, Q=32 | 174.9 (bf16) | 311.9 | 56.1% | 566.4 | 2,398.4 | LM/readout +88.7 |

## Video-style speed workload, one box question per frame

Item 7 packs the resized RGB bytes into one transfer. Both rows use the compact-input floor. [Video stages and floors](07_frames/raw/item7_video_floors.json)

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---|
| 32 frames | 214.7 (bf16) | 290.1 | 74.0% | 549.4 | 666.3 | Vision/H2D +45.7 |
| 64 frames | 429.7 (bf16) | 587.7 | 73.1% | 928.1 | 1,052.0 | Vision/H2D +92.1 |

## Fresh scan, one yes/no question over N new frames

Item 2 uploads uint8 RGB. Its pinned H2D floor is one eighth of the C0 FP32 patch transfer. The compute and readout floors use the same model. Large packed calls use the optional SM120 BF16 GDN kernel. The item 3 standard FLA stack measured 619.6 ms at Tv=128, N=128. [Standard stack](03_lm_prefill/raw/item3_production.json) · [Compact-input stage floors](02_vision_prep/raw/item2_floors.json) · [Item-3 measurements and exactness](03_lm_prefill/summary.md)

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---|
| Tv=128, N=32 | 111.2 (bf16) | 142.9 | 77.8% | 177.9 | 259.3 | Vision/H2D +16.3 |
| Tv=128, N=64 | 222.5 (bf16) | 287.5 | 77.4% | 332.1 | 431.4 | Vision/H2D +32.4 |
| Tv=128, N=128 | 444.9 (bf16) | 581.5 | 76.5% | 637.8 | 815.3 | Vision/H2D +67.2 |
| Tv=128, N=256 | 889.8 (bf16) | 1,165.3 | 76.4% | 1,252.8 | 1,572.9 | Vision/H2D +135.7 |
| Tv=256, N=32 | 223.3 (bf16) | 301.7 | 74.0% | 322.8 | 461.0 | Vision/H2D +44.7 |
| Tv=256, N=64 | 446.5 (bf16) | 602.1 | 74.2% | 629.4 | 848.4 | Vision/H2D +87.5 |
| Tv=256, N=128 | 893.0 (bf16) | 1,207.5 | 74.0% | 1,244.7 | 1,567.6 | Vision/H2D +176.4 |
| Tv=256, N=256 | 1,786.1 (bf16) | 2,420.5 | 73.8% | 2,495.9 | 2,997.1 | Vision/H2D +353.1 |

## Cached scan, one new question over N independent prefixes

Item 8 uses compact lossless K/V and spatial storage with full FP32 recurrent state. Floors use the first 15 changing questions in the paired before/after run. The report includes fixed-question per-stage profiles and the measured PCIe ceiling for banks in pinned host memory. [Bank timings and stages](08_bank_memory/summary.md) · [Per-call floors](08_bank_memory/raw/item8_floors.json) · [vLLM clip extensions](08_bank_memory/raw/item8_test_vllm.json) · [SGLang clip extensions](08_bank_memory/raw/item8_test_sglang.json)

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---:|
| Tv=640, N=64, box | 17.3 (bf16) | 23.7 | 73.0% | 899.1 | 1,556.6 | LM/readout +6.3 |
| Tv=640, N=64, yes/no | 32.5 (bf16) | 45.4 | 71.6% | 680.6 | 1,287.3 | LM/readout +12.8 |
| Tv=640, N=256, box | 69.0 (bf16) | 96.7 | 71.4% | 3,151.6 | 7,732.9 | LM/readout +27.6 |
| Tv=640, N=256, yes/no | 129.9 (bf16) | 175.4 | 74.0% | 2,679.9 | 5,670.7 | LM/readout +45.4 |
| Tv=256, N=64, box | 16.7 (bf16) | 22.5 | 74.3% | 653.1 | 592.2 | LM/readout +5.7 |
| Tv=256, N=256, box | 66.8 (bf16) | 84.5 | 79.0% | 2,950.1 | 2,981.4 | LM/readout +17.6 |
| Tv=64, N=64, box | 16.4 (bf16) | 22.0 | 74.6% | 401.3 | 446.1 | LM/readout +5.5 |
| Tv=64, N=256, box | 65.7 (bf16) | 82.6 | 79.5% | 1,112.6 | 1,440.6 | LM/readout +16.8 |
| f4_v96 clip, N=64, yes/no | 31.8 (bf16) | 55.7 | 57.1% | 342.7 | 620.7 | LM/readout +23.8 |
| f4_v96 clip, N=256, yes/no | 127.2 (bf16) | 241.4 | 52.7% | 1,355.1 | 2,203.7 | LM/readout +114.0 |
| f4_v48 clip, N=64, yes/no | 31.7 (bf16) | 55.0 | 57.6% | 236.0 | 556.5 | LM/readout +23.2 |
| f4_v48 clip, N=256, yes/no | 126.8 (bf16) | 234.9 | 54.0% | 907.7 | 2,002.2 | LM/readout +107.9 |
| f2_v96 clip, N=64, yes/no | 31.7 (bf16) | 54.9 | 57.8% | 214.2 | 479.1 | LM/readout +23.1 |
| f2_v96 clip, N=256, yes/no | 126.7 (bf16) | 234.3 | 54.1% | 823.3 | 1,784.3 | LM/readout +107.4 |

At 1,024 f4_v96 clips, a single GPU-resident query took 963.0 ms. The exact state-packed pinned tier took 1,010.5 ms, including 53.64 GB over PCIe. A 10,000-entry traversal transferred 523.81 GB in 9.333 s against a 9.211 s PCIe floor. It cycled a physical 1,920-clip backing. [GPU-resident control](08_bank_memory/raw/item8_exact_resident.json) · [Pinned-tier timings and bytes](08_bank_memory/raw/item8_coalesced_tier_packed.json)

## Follow-up questions on one cached image

| Workload | Floor ms (resource) | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---|
| box Q=1 | 5.0 (device BW) | 8.8 | 56.3% | 145.2 | 162.0 | LM/readout +3.5 |
| mixed Q=1 | 5.0 (device BW) | 8.8 | 56.3% | 145.1 | 161.7 | LM/readout +3.5 |
| box Q=8 | 5.0 (device BW) | 10.3 | 48.9% | 196.6 | 346.1 | LM/readout +4.6 |
| mixed Q=8 | 5.6 (bf16) | 15.0 | 37.7% | 175.3 | 342.2 | LM/readout +8.5 |
| box Q=32 | 7.6 (bf16) | 19.6 | 38.8% | 316.9 | 606.5 | LM/readout +10.5 |
| mixed Q=32 | 22.3 (bf16) | 39.9 | 56.0% | 254.9 | 561.8 | LM/readout +15.3 |

## Optional W12 for short cached queries

**Historical item 5 comparison.** These paired rows show the effect of optional lossless weight packing and shorter question buckets on the item 5 engine. The frozen BF16 follow-up and bank medians appear above. The floor charges packed bytes only for projections that dispatch W12. Both serving columns use BF16. Rows with more than 128 physical question rows use BF16 for all projections. The full item report also covers fresh scan, cold multi-question calls, video, the separate lossy FP8 mode and stage-level ceilings. [31-workload comparison and methods](05_w12/summary.md) · [Per-workload floor breakdown](05_w12/raw/item5_workload_floors.json)

| Workload (≤640 visual tokens) | Default BF16 ms | With W12 ms | W12 floor ms | Floor / W12 | vLLM ms | SGLang ms |
|---|---:|---:|---:|---:|---:|---:|
| One cached image, 1 box question | 8.67 | 7.04 | 3.77 | 53.5% | 145.2 | 162.0 |
| One cached image, 8 box questions | 10.11 | 9.84 | 4.53 | 46.1% | 196.6 | 346.1 |
| One cached image, 8 mixed questions | 14.65 | 14.66 | 5.63 | 38.4% | 175.3 | 342.2 |
| One cached image, 32 box questions | 19.20 | 19.20 | 7.61 | 39.6% | 316.9 | 606.5 |
| Bank, 1 image, box | 7.32 | 5.55 | 3.76 | 67.8% | 26.7 | 56.1 |
| Bank, 1 image, yes/no | 7.44 | 6.04 | 3.89 | 64.3% | 14.2 | 45.5 |
| Bank, 8 images, box | 8.40 | 7.02 | 4.54 | 64.7% | 175.7 | 283.3 |
| Bank, 8 images, yes/no | 9.79 | 9.81 | 5.46 | 55.6% | 34.9 | 147.0 |
| Bank, 32 images, box | 13.50 | 13.50 | 8.36 | 61.9% | 401.3 | 578.1 |
| Bank, 64 images, box | 23.48 | 23.50 | 16.69 | 71.0% | 899.1 | 1556.6 |

With one question, ideal dense weight-read time falls from 4.88 to 3.67 ms when all projections use the packed representation. The G4 microbenchmark attributed 51.6% of the 16-row BF16 dense time to bringing different weights into cache, 6.1% at 256 rows and 0.9% at 1,024. At the nominal 16k fresh prefill, the 18,688-row gate/up control delivered 382.5 TFLOP/s against a measured 401.7 TFLOP/s peak. [Weight-residency experiment](05_w12/raw/item5_weight_traffic.json) · [Speed timings](05_w12/raw/item5_verified_w12.json) · [Bank timings](05_w12/raw/item5_verified_scan_w12.json)

## Concurrent clients

The item 7 run times each simultaneous burst through the last complete answer. Glance uses cross-request BF16 prefill with a 0.5 ms gathering window, an 8,192-token budget and the optional native SM120 GDN kernel. The floor allows ideal batching over the same useful shapes and charges uint8 RGB upload. In item 6 the standard FLA stack beat both serving baselines in 21 of these 24 cells and took 1,486.8 ms on the 64-client cold box burst. Per-client distributions and all stage measurements are in the item report. [All 24 latency distributions and stage floors](06_scheduler/summary.md) · [Current scheduler measurements](07_frames/raw/item7_final_scheduled.json) · [Frame comparisons](07_frames/summary.md) · [Burst floors](06_scheduler/raw/item6_floors.json)

| Workload | Clients | Floor ms | Glance ms | % of ceiling | vLLM ms | SGLang ms | Largest measured stage gap, ms |
|---|---:|---:|---:|---:|---:|---:|---|
| Cold box | 1 | 17.2 | 32.2 | 53.4% | 166.2 | 188.1 | LM/readout +9.1 |
| Cold box | 4 | 67.8 | 104.0 | 65.2% | 264.9 | 322.8 | LM/readout +17.8 |
| Cold box | 16 | 272.8 | 362.7 | 75.2% | 578.6 | 715.4 | Vision/H2D +47.6 |
| Cold box | 64 | 1,084.2 | 1,424.3 | 76.1% | 1,828.0 | 2,330.0 | Vision/H2D +188.1 |
| Cold yes/no | 1 | 17.4 | 32.2 | 54.2% | 39.5 | 53.8 | LM/readout +8.9 |
| Cold yes/no | 4 | 68.8 | 104.3 | 65.9% | 111.1 | 157.2 | LM/readout +18.9 |
| Cold yes/no | 16 | 276.6 | 370.3 | 74.7% | 381.7 | 508.6 | Vision/H2D +48.8 |
| Cold yes/no | 64 | 1,099.2 | 1,456.3 | 75.5% | 1,505.3 | 1,946.8 | Vision/H2D +190.2 |
| Eight mixed questions | 1 | 22.5 | 45.3 | 49.6% | 199.6 | 378.3 | LM/readout +16.3 |
| Eight mixed questions | 4 | 89.5 | 141.1 | 63.4% | 351.1 | 888.8 | LM/readout +32.2 |
| Eight mixed questions | 16 | 358.0 | 534.0 | 67.0% | 979.4 | 2,734.1 | LM/readout +115.1 |
| Eight mixed questions | 64 | 1,425.9 | 2,144.3 | 66.5% | 3,499.5 | 12,305.1 | LM/readout +472.4 |
| Cached box, resent image | 1 | 5.0 | 8.7 | 57.2% | 140.3 | 167.1 | LM/readout +3.1 |
| Cached box, resent image | 4 | 5.2 | 17.3 | 30.1% | 182.0 | 248.2 | CPU prep +5.8 |
| Cached box, resent image | 16 | 6.2 | 47.4 | 13.1% | 253.3 | 452.8 | CPU prep +22.3 |
| Cached box, resent image | 64 | 19.8 | 173.1 | 11.5% | 1,019.0 | 1,801.8 | CPU prep +87.8 |
| GUI box, ≤2,048 tokens | 1 | 67.2 | 98.3 | 68.4% | 233.7 | 297.6 | LM/readout +15.5 |
| GUI box, ≤2,048 tokens | 4 | 267.2 | 354.1 | 75.5% | 521.2 | 705.5 | Vision/H2D +45.8 |
| GUI box, ≤2,048 tokens | 16 | 1,069.7 | 1,384.3 | 77.3% | 1,607.2 | 2,221.4 | Vision/H2D +181.3 |
| GUI box, ≤2,048 tokens | 64 | 4,267.5 | 5,515.6 | 77.4% | 5,983.4 | 8,640.1 | Vision/H2D +708.0 |
| 32 fresh frames, yes/no | 1 | 112.7 | 157.5 | 71.5% | 181.1 | 248.9 | Vision/H2D +22.9 |
| 32 fresh frames, yes/no | 4 | 447.8 | 609.3 | 73.5% | 635.2 | 809.8 | Vision/H2D +89.9 |
| 32 fresh frames, yes/no | 16 | 1,788.4 | 2,422.7 | 73.8% | 2,552.3 | 3,069.3 | Vision/H2D +358.2 |
| 32 fresh frames, yes/no | 64 | 7,153.5 | 9,695.5 | 73.8% | 10,603.3 | 12,724.0 | Vision/H2D +1,431.2 |

The 64-client cold box burst took 1,424.3 ms against the item 6 serial control of 2,384.6 ms, leaving a 340.1 ms gap to the hardware floor. Cached box follow-ups using returned image IDs took 66.3 ms at 64 clients, including 55.4 ms in LM/readout. The matched cross-engine table resends the pixels. [Serial controls](06_scheduler/raw/item6_warm_serial.json) · [Image-ID timings](06_scheduler/raw/item6_production_ids.json)

## Benchmark boundaries

Speed and GUI rows time the fused Glance engine. The video-style row uses separately resized COCO images at ≤256 tokens, one image-specific box request per frame. Fresh scan uses fixed 4:3 frames and broadcasts one changing yes/no question. The cached image/clip rows exclude bank construction. Follow-ups use the benchmark’s retained prefix. Its box questions cycle the image’s expressions, so the warm-up and first box request can repeat the same text. Client rows measure the last completion in a simultaneous in-process burst. One Glance device worker batches across clients. vLLM and SGLang can independently schedule each image/question pair. Each logical client waits for all its answers. The burst floor allows ideal batching. [Single-image boxes and follow-ups][box9] · [Single-image mixed questions][mixed9] · [GUI 2k](03_lm_prefill/raw/item3_gui2048.json) · [GUI 4k](03_lm_prefill/raw/item3_gui4096.json) · [Video](07_frames/raw/item7_checked_video.json) · [Fresh][fresh7] · [Fresh Tv128/N128][fresh9] · [Cached][bank8] · [Cached Tv640/N256][bank9] · [Clients](07_frames/raw/item7_final_scheduled.json)

Serving sources use the same task, dimensions, output kind and cache protocol in each row. Speed box baselines generate up to 24 tokens. Fresh and cached yes/no baselines use one token with logprobs. The sprint also published one-token box proxies, which are ineligible for the full-box comparison. [Sprint vLLM](../sprint_speed/raw/serve_vllm_async_g4.json) · [Corrected SGLang sprint rerun](00_ceiling/raw/c0_sglang_clean.json) · [Fresh vLLM, full resets](07_frames/raw/item7_final_fresh_vllm.json) · [Fresh SGLang](07_frames/raw/item7_final_fresh_sglang.json) · [Cached vLLM](../scan/raw/scan-a/vllm_bf16.json) · [Corrected SGLang bank rerun](00_ceiling/raw/c0_sglang_bank.json) · [Burst vLLM](07_frames/raw/item7_final_scheduler_vllm.json) · [Burst SGLang](07_frames/raw/item7_final_scheduler_sglang.json)

The earlier SGLang harness caught an event-loop exception while flushing its cache. Its uncorrected 64-client burst reused 663.0 prompt tokens per request. The corrected harness awaits the tokenizer manager’s flush and checks success. This map uses the corrected speed, GUI, follow-up and bank runs, and the item 7 concurrent-client reruns. It excludes the historical SGLang MTP sweep from the fastest-configuration comparisons. Item 2 also checks vLLM visual-embedding reuse separately from cached prompt tokens. Its fresh columns use explicit encoder-cache reset in vLLM and content-unique requests in SGLang. [Uncorrected burst](00_ceiling/raw/c0_clients_sglang.json) · [Reset warning](00_ceiling/raw/c0_clients_sglang.log) · [Corrected benchmark](../../speed/bench_serve.py)

## Single-question prefill against tuned serving

The tuned full-output comparison uses the item 7 video rerun and the historical GUI measurements. The first baseline here returns the box. The last column is a decode-free timing proxy that stops vLLM after one generated token. [Tuned vLLM full outputs](../sprint_speed/raw/serve_vllm_fp8mtp_g4.json) · [Video tuned rerun](07_frames/raw/item7_matched_video_tuned.json) · [BF16 one-token timing](../sprint_speed/raw/serve_vllm_1tok_g4.json)

| Workload | Glance BF16 ms | Fastest eligible full box, ms | Glance − fastest, ms | vLLM BF16 one token, ms |
|---|---:|---|---:|---:|
| 2048 tokens, Q=1 | 119.2 | vLLM FP8 + MTP 149.5 | -30.3 | 107.6 |
| 4096 tokens, Q=1 | 277.9 | vLLM FP8 + MTP 262.4 | +15.5 | 236.6 |
| 32 frames | 290.1 | vLLM FP8 + MTP 387.1 | -96.9 | 353.0 |
| 64 frames | 587.7 | vLLM FP8 + MTP 695.7 | -108.0 | 681.9 |

## Attack order

The original optimization assignments in the last column are historical. Each row selects the largest absolute hardware gap within its family, so the GUI row here is Q=32. The tuned vLLM win at Q=1 appears in the preceding table.

Ranked by the largest absolute gap within each workload family. For the three frame rows, the signed difference subtracts the fastest eligible configuration in the matched item 7 reruns. Other families use the serving measurements from their respective items. A positive value is a Glance deficit. The sprint configurations include speculative decoding and separately labeled FP8. [Sprint measurement series](../sprint_speed/raw)

| Rank | Largest-gap workload in family | Glance − floor, ms | Fastest serving configuration, ms | Glance − fastest, ms | Optimization items |
|---:|---|---:|---|---:|---|
| 1 | 64 clients × 32 fresh frames | 2,542.0 | vLLM 10,603.3 | -907.7 | 7 remaining fixed-grid vision and packed LM |
| 2 | fresh_Tv256_N256 | 634.4 | vLLM 2,495.9 | -75.5 | 7 exact patch embedding, vision attention and GDN |
| 3 | video_f64_N1 | 157.9 | vLLM FP8 + MTP 695.7 | -108.0 | 7 mixed-grid vision and packed LM |
| 4 | gui4096_box_N32 | 137.0 | vLLM FP8 + MTP 435.7 | -123.8 | 5 long prefill, 2 vision/prep |
| 5 | f4_v96_N256_noul | 114.2 | vLLM 1,355.1 | -1,113.7 | 7 native clip branch traffic, 8 bank storage |
| 6 | bs1_mixed_N32 | 29.7 | vLLM FP8 + MTP 177.0 | -107.8 | 3 small-N replay / dispatch, 4 lossless weights |
| 7 | followup_mixed_N32 | 17.6 | vLLM FP8 + MTP 157.4 | -117.5 | 3 branch graphs, 4 weight traffic |

Item 9 recalibrated the numerical gates on five shipped-backbone layouts and added same-center and selected-patch checks. [Calibration and fault injection](09_numerics/summary.md#calibrated-gates) Item 5 records W12 reconstruction and optional FP8 accuracy. The frame report accounts for padded singleton state and the bounded host image snapshot. Item 6 includes the loopback HTTP load test and fault checks. [Weight modes and accuracy](05_w12/summary.md#accuracy) · [HTTP measurements](06_scheduler/raw/item6_production_http.json) For the cached 256-image yes/no case a separate replay recorded 122.5 ms in GEMMs, 67.1 ms in copy/concatenation kernels and 38.6 ms in the fused recurrent kernel. Item 4 measures the output-only branch kernel and the packed attention path against this baseline. [Branch measurements](04_gdn_branch/summary.md) · [Cached branch traces](00_ceiling/raw/c0_profile_bank.json) · [Historical C0 gaps](00_ceiling/raw/c0_ranked_gaps.json)

## Stage timings and floors

Each pair is measured / modeled milliseconds. Vision includes pixel H2D. LM includes GDN, GQA, the readout heads and return of answers. The cached benchmarks expose one model timer. CPU prep has a zero lower bound here. Burst latency includes admission, batched dispatch and delivery of the last client’s result. Per-stage medians need not sum to the median of the end-to-end call. The item 7 report gives the corresponding stages for all 24 client bursts and the per-kernel fresh tower profile. [Frame timings and floors](07_frames/summary.md) The item 8 profile separates compact-bank access, dense projections, GDN continuation, attention and readout for all six bank formats. [Cached bank stage measurements](08_bank_memory/summary.md#where-the-query-time-goes) The item 4 breakdown times dense projections, GDN, attention and heads separately across 20 distinct branch workloads. [Branch stage timings and floors](04_gdn_branch/summary.md#lm-stage-measurements-against-the-roofline)

| Workload | CPU prep | Vision + H2D | LM + readout |
|---|---:|---:|---:|
| bs1_box_N1 | 2.2 / 0.0 | 9.9 / 5.7 | 25.5 / 11.8 |
| bs1_box_N8 | 2.3 / 0.0 | 9.9 / 5.7 | 26.4 / 13.4 |
| bs1_box_N32 | 3.3 / 0.0 | 9.9 / 5.7 | 35.7 / 19.1 |
| bs1_mixed_N8 | 2.7 / 0.0 | 9.9 / 5.7 | 33.4 / 17.1 |
| bs1_mixed_N32 | 4.0 / 0.0 | 9.9 / 5.7 | 55.3 / 33.9 |
| gui2048_box_N1 | 7.9 / 0.0 | 42.4 / 29.8 | 69.0 / 38.1 |
| gui2048_box_N8 | 8.5 / 0.0 | 42.4 / 29.8 | 71.5 / 39.7 |
| gui2048_box_N32 | 9.3 / 0.0 | 42.5 / 29.8 | 94.5 / 45.7 |
| gui4096_box_N1 | 22.9 / 0.0 | 112.1 / 89.7 | 143.0 / 76.1 |
| gui4096_box_N8 | 24.8 / 0.0 | 112.5 / 89.7 | 147.3 / 78.2 |
| gui4096_box_N32 | 26.0 / 0.0 | 112.5 / 89.7 | 172.1 / 83.4 |
| video_f32_N1 | 4.1 / 0.0 | 103.4 / 57.7 | 182.5 / 157.0 |
| video_f64_N1 | 8.0 / 0.0 | 207.5 / 115.4 | 372.1 / 314.3 |
| fresh_Tv128_N32 | 2.5 / 0.0 | 41.6 / 25.4 | 98.9 / 85.9 |
| fresh_Tv128_N64 | 4.0 / 0.0 | 83.1 / 50.7 | 200.5 / 171.7 |
| fresh_Tv128_N128 | 8.5 / 0.0 | 168.6 / 101.4 | 404.5 / 343.5 |
| fresh_Tv128_N256 | 15.8 / 0.0 | 338.5 / 202.8 | 810.9 / 687.0 |
| fresh_Tv256_N32 | 4.6 / 0.0 | 102.9 / 58.2 | 194.4 / 165.0 |
| fresh_Tv256_N64 | 8.1 / 0.0 | 203.9 / 116.5 | 390.0 / 330.0 |
| fresh_Tv256_N128 | 16.4 / 0.0 | 409.4 / 232.9 | 782.0 / 660.1 |
| fresh_Tv256_N256 | 32.5 / 0.0 | 818.9 / 465.9 | 1,569.1 / 1,320.2 |
| Tv640_N64_box | 0.1 / 0.0 | — | 23.5 / 17.3 |
| Tv640_N64_noul | 0.1 / 0.0 | — | 45.3 / 32.5 |
| Tv640_N256_box | 0.1 / 0.0 | — | 96.6 / 69.0 |
| Tv640_N256_noul | 0.2 / 0.0 | — | 175.2 / 129.9 |
| Tv256_N64_box | 0.1 / 0.0 | — | 22.4 / 16.7 |
| Tv256_N256_box | 0.1 / 0.0 | — | 84.3 / 66.8 |
| Tv64_N64_box | 0.1 / 0.0 | — | 21.9 / 16.4 |
| Tv64_N256_box | 0.1 / 0.0 | — | 82.5 / 65.7 |
| f4_v96_N64_noul | 0.1 / 0.0 | — | 55.6 / 31.8 |
| f4_v96_N256_noul | 0.2 / 0.0 | — | 241.2 / 127.2 |
| f4_v48_N64_noul | 0.1 / 0.0 | — | 54.9 / 31.7 |
| f4_v48_N256_noul | 0.2 / 0.0 | — | 234.7 / 126.8 |
| f2_v96_N64_noul | 0.1 / 0.0 | — | 54.7 / 31.7 |
| f2_v96_N256_noul | 0.2 / 0.0 | — | 234.1 / 126.7 |
| followup_box_N1 | 0.3 / 0.0 | — | 8.5 / 5.0 |
| followup_mixed_N1 | 0.3 / 0.0 | — | 8.5 / 5.0 |
| followup_box_N8 | 0.6 / 0.0 | — | 9.7 / 5.0 |
| followup_mixed_N8 | 0.9 / 0.0 | — | 14.1 / 5.6 |
| followup_box_N32 | 1.5 / 0.0 | — | 18.1 / 7.6 |
| followup_mixed_N32 | 2.3 / 0.0 | — | 37.6 / 22.3 |
| clients_1 | 0.4 / 0.0 | 9.7 / 5.4 | 20.9 / 11.8 |
| clients_4 | 1.2 / 0.0 | 36.1 / 21.3 | 64.3 / 46.5 |
| clients_16 | 2.8 / 0.0 | 133.5 / 85.9 | 228.6 / 187.0 |
| clients_64 | 10.8 / 0.0 | 529.3 / 341.3 | 882.1 / 742.9 |
