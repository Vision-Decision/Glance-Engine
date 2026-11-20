# Item 6 — concurrent decision serving

On an idle RTX PRO 6000 Blackwell, 64 independent cold box clients finished in **1,439.4 ms** with the BF16 scheduler, down **39.6%** from **2,384.6 ms** with serial Glance. vLLM took **1,826.8 ms** and SGLang **2,328.4 ms**. The scheduled run delivered **44.3 requests/s**, with client p50/p90/p99 of **820.5 / 1,358.4 / 1,453.9 ms**. The measured median burst was shorter than both serving baselines in all 24 workload/concurrency combinations. All 254 interleaved decisions agreed with both isolated requests and shipped BF16. [Scheduled run](raw/item6_production.json) · [Independent BF16 comparisons](raw/item6_production_check.json)

The smallest margins deserve separate interpretation. At 16 cold yes/no clients Glance led vLLM by 6.0 ms (1.6%). At 64, vLLM returned the median yes/no client earlier, even though Glance finished the full burst earlier. One cached box follow-up took 11.7 ms against 8.8 ms serially. One fresh 32-frame scan took 160.3 ms against 160.1 ms. Each measured cell contains five bursts. [Client observations](raw/item6_production.json) · [Serial observations](raw/item6_warm_serial.json) · [vLLM observations](raw/item6_vllm.json)

## Implementation

- The asyncio FIFO gathers requests for up to 0.5 ms from the oldest arrival, with an 8,192-token admission budget and a 64-request cap. A single executor thread owns the GPU. The worker checks exact tokenized lengths, splits at image and question boundaries, and routes answers by original request, image and question. Each forward produces decisions without an autoregressive decode loop. [Scheduler](../../../speed/scheduler.py) · [Device worker](../../../speed/decision_worker.py)
- Newly submitted single-question images share the item 1/3 variable-length LM prefill. Multi-question images share the dense projections across packed prefixes and branches, using the item 4 GDN question kernel. Each GDN prefix retains its isolated 64-token chunk alignment. Vision uploads uint8 RGB, batches matching spatial grids, and restores client order before LM prefill. Small eligible shapes use bounded CUDA graph caches. Larger shapes run eagerly. The existing serial engine remains available through the benchmark's `--backend serial`, and the server exposes the padded multi-question control through `--shared-prefill padded`. [Packed prefix mixers](../../../scan/varlen.py) · [Engine dispatch](../../../speed/engine.py)
- An 8 GiB default LRU owns compact per-image GDN states, convolution tails, attention KV and visual features, plus named video banks. It accounts for the retained tensor bytes. Inserting an image builds its prefix in a singleton layout so future clients reuse a stable BF16 state. Ordinary cold requests and warm queries still batch across clients. A caller can resend the image or pass its returned SHA-256 ID. An expired ID yields HTTP 404. Prefix creation and process-local cache lifetime matter when using these results to budget a deployment. [Cache and routing](../../../speed/decision_worker.py)
- The HTTP application exposes typed decisions, a multimodal chat-completions adapter, the three legacy video routes, health, readiness and metrics. Queue saturation returns 429, timeouts return 504 and an unsplittable device OOM returns 503. JSON decode and image preparation run off the event loop. Cancellation abandons pending work and shutdown drains accepted requests. The GPU worker splits a failed batch and retries smaller groups. [Server](../../../speed/server.py) · [API and startup](../../../README.md#async-decision-server)

## Measurement boundary

All four engines received the same decoded, resized pixels, question content and logical request boundaries on engopt. The clock starts with simultaneous in-process submission and stops when the last answer for that client returns. Glance includes host preparation, H2D, vision, LM, heads and host results. The separate HTTP test includes JPEG/base64, transport and server parsing. GPU access was exclusive, with an empty compute-process check before each timed run. [Harness](../../../speed/bench_scheduler.py) · [Run log and source hashes](raw/item6_production.log)

The six workloads are a RefCOCO box and yes/no at ≤640 visual tokens, eight mixed box/yes-no/choice/score questions on one image, a new box question on a cached image, a ScreenSpot-Pro box at ≤2,048 tokens, and a yes/no question over 32 fresh ≤128-token frames per client. With 64 fresh clients this covers 2,048 frames. The 100 COCO and 40 GUI sources cycle with a unique pixel marker in every cold input. The follow-up workload primes each client’s image with a different yes/no question before timing. The main table resends the image so all four engines receive the same media. [Inputs and prompts](../../../speed/workload.py) · [Burst construction and cache resets](../../../speed/bench_scheduler.py)

Each cell has two warm-up bursts and five measured bursts. Serial and scheduled Glance also warm the timed shapes on distinct cold pixels before the two warm-ups. vLLM 0.34.0 and SGLang 0.5.23 use the existing serving configurations with BF16 prefix caching and continuous batching. The harness submits each generative image/question pair independently and waits for every constituent answer. A full box allows up to 24 output tokens. Typed requests allow one label token. These baseline times measure task service, with no cross-model accuracy or probability comparison. Primary scheduled times use the optional item 3 native SM120 BF16 GDN kernel. [vLLM configuration](raw/item6_vllm.json) · [SGLang configuration](raw/item6_sglang.json) · [Glance configuration](raw/item6_production.json)

The initial ceiling check used the existing G4 calibration of 401.73 BF16 TFLOP/s, 67.4 FP32 TFLOP/s, 1,463.2 GB/s device copy and 56.9 GB/s pinned H2D. The stage model charges ideal batching for the useful shapes in each measured burst and compact RGB upload. For 64 cold box clients, the floor is 1,084.2 ms, versus 1,100.9 ms in the earlier C0 workload. The original C0 and item 3 serial bursts were 2,699 and 2,527 ms. The initial same-session legacy harness gave 2,390.0 ms. The before/after tables use the independently rerun, input-matched serial harness. [Hardware measurements](../ROOFLINE.md) · [Workload floor calculation](../../../speed/scheduler_floors.py) · [Legacy harness rerun](raw/item6_serial_box.json) · [Item 3 serial burst](../03_lm_prefill/raw/item3_clients.json)

## Burst completion and hardware floor

Milliseconds from simultaneous submission to the last client’s complete answer. Floor / achieved is the fraction of the empirical G4 ceiling. Each serving backend used the same resized pixels, question strings and logical request boundaries. [Serial Glance](raw/item6_warm_serial.json) · [vLLM](raw/item6_vllm.json) · [SGLang](raw/item6_sglang.json) · [Ideal-batching stage floors](raw/item6_floors.json)

| Workload | Clients | Serial ms | Scheduled ms | vLLM ms | SGLang ms | Floor ms | Floor / achieved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold box | 1 | 37.7 | 32.2 | 168.9 | 188.1 | 17.2 | 53.4% |
| Cold box | 4 | 150.9 | 103.6 | 265.0 | 318.2 | 67.8 | 65.5% |
| Cold box | 16 | 599.4 | 368.7 | 578.4 | 715.9 | 272.8 | 74.0% |
| Cold box | 64 | 2,384.6 | 1,439.4 | 1,826.8 | 2,328.4 | 1,084.2 | 75.3% |
| Cold yes/no | 1 | 37.6 | 32.2 | 39.3 | 53.8 | 17.4 | 54.2% |
| Cold yes/no | 4 | 151.3 | 105.6 | 110.8 | 158.0 | 68.8 | 65.1% |
| Cold yes/no | 16 | 600.8 | 374.8 | 380.8 | 509.7 | 276.6 | 73.8% |
| Cold yes/no | 64 | 2,401.4 | 1,471.5 | 1,504.3 | 1,964.7 | 1,099.2 | 74.7% |
| Eight mixed questions | 1 | 45.8 | 45.6 | 199.1 | 372.6 | 22.5 | 49.3% |
| Eight mixed questions | 4 | 184.5 | 142.4 | 351.1 | 866.0 | 89.5 | 62.9% |
| Eight mixed questions | 16 | 734.2 | 538.4 | 979.3 | 2,840.5 | 358.0 | 66.5% |
| Eight mixed questions | 64 | 2,930.2 | 2,161.0 | 3,499.7 | 12,345.1 | 1,425.9 | 66.0% |
| Follow-up, resent image | 1 | 8.8 | 11.7 | 140.3 | 164.0 | 5.0 | 42.7% |
| Follow-up, resent image | 4 | 34.7 | 17.4 | 182.0 | 242.9 | 5.2 | 30.0% |
| Follow-up, resent image | 16 | 138.1 | 48.6 | 251.8 | 423.7 | 6.2 | 12.8% |
| Follow-up, resent image | 64 | 549.6 | 175.2 | 1,018.6 | 1,786.7 | 19.8 | 11.3% |
| GUI box, ≤2,048 tokens | 1 | 114.8 | 98.6 | 233.8 | 296.1 | 67.2 | 68.2% |
| GUI box, ≤2,048 tokens | 4 | 459.6 | 358.6 | 521.4 | 707.5 | 267.2 | 74.5% |
| GUI box, ≤2,048 tokens | 16 | 1,838.1 | 1,409.5 | 1,621.3 | 2,216.5 | 1,069.7 | 75.9% |
| GUI box, ≤2,048 tokens | 64 | 7,337.9 | 5,621.2 | 5,991.3 | 8,578.8 | 4,267.5 | 75.9% |
| 32 fresh frames, yes/no | 1 | 160.1 | 160.3 | 175.0 | 263.0 | 112.7 | 70.3% |
| 32 fresh frames, yes/no | 4 | 625.7 | 622.4 | 635.4 | 806.8 | 447.8 | 71.9% |
| 32 fresh frames, yes/no | 16 | 2,498.3 | 2,479.1 | 2,557.3 | 3,051.9 | 1,788.4 | 72.1% |
| 32 fresh frames, yes/no | 64 | 10,004.1 | 9,917.9 | 10,246.1 | 12,984.6 | 7,153.5 | 72.1% |

## Time to answer per client and throughput

Each entry gives p50 / p90 / p99 milliseconds, followed by completed logical requests per second in parentheses. Percentiles pool individual client completions over five timed bursts. Throughput divides all completions by the sum of burst durations. At one client the percentiles describe only five observations.

| Workload | Clients | Serial Glance | Scheduled Glance | vLLM | SGLang |
|---|---:|---:|---:|---:|---:|
| Cold box | 1 | 37.7 / 38.9 / 39.2 (26.3) | 32.2 / 32.3 / 32.3 (31.8) | 168.9 / 182.5 / 182.8 (5.8) | 188.1 / 201.6 / 202.1 (5.2) |
| Cold box | 4 | 93.7 / 151.0 / 151.7 (26.5) | 103.6 / 106.1 / 106.1 (38.4) | 249.3 / 265.5 / 270.5 (15.0) | 303.1 / 318.5 / 330.0 (12.5) |
| Cold box | 16 | 312.3 / 561.2 / 600.7 (26.8) | 271.0 / 369.1 / 374.3 (43.3) | 547.8 / 569.3 / 580.2 (27.9) | 694.5 / 716.4 / 725.6 (22.4) |
| Cold box | 64 | 1212.9 / 2174.9 / 2381.5 (26.8) | 820.5 / 1358.4 / 1453.9 (44.3) | 1786.6 / 1821.2 / 1837.2 (34.9) | 2296.1 / 2673.4 / 2707.2 (26.7) |
| Cold yes/no | 1 | 37.6 / 38.2 / 38.3 (26.5) | 32.2 / 32.2 / 32.2 (32.1) | 39.3 / 39.6 / 39.8 (26.0) | 53.8 / 54.2 / 54.5 (18.8) |
| Cold yes/no | 4 | 93.9 / 151.3 / 152.1 (26.5) | 105.6 / 107.2 / 107.2 (37.8) | 110.7 / 113.2 / 113.2 (36.0) | 157.5 / 160.5 / 160.5 (25.2) |
| Cold yes/no | 16 | 312.2 / 562.7 / 601.2 (26.7) | 274.1 / 375.1 / 379.3 (42.6) | 268.5 / 381.7 / 383.1 (42.2) | 500.6 / 510.8 / 512.1 (31.5) |
| Cold yes/no | 64 | 1221.7 / 2174.8 / 2396.1 (26.7) | 825.0 / 1372.0 / 1477.4 (43.5) | 792.0 / 1504.3 / 1519.4 (42.4) | 1452.7 / 1920.0 / 1989.6 (32.5) |
| Eight mixed questions | 1 | 45.8 / 46.5 / 46.5 (21.8) | 45.6 / 46.4 / 46.4 (21.8) | 199.1 / 204.0 / 204.5 (5.0) | 372.6 / 383.0 / 386.3 (2.7) |
| Eight mixed questions | 4 | 114.5 / 184.5 / 185.3 (21.7) | 142.4 / 145.5 / 145.5 (28.1) | 340.3 / 352.4 / 368.1 (11.3) | 864.6 / 889.6 / 902.7 (4.6) |
| Eight mixed questions | 16 | 384.9 / 687.3 / 736.6 (21.8) | 308.0 / 541.9 / 550.2 (29.6) | 918.6 / 982.8 / 1022.2 (16.3) | 2834.0 / 2921.6 / 2928.2 (5.7) |
| Eight mixed questions | 64 | 1490.1 / 2666.0 / 2926.5 (21.8) | 1121.5 / 2144.9 / 2167.1 (29.6) | 2698.5 / 3435.8 / 3500.3 (18.3) | 12330.0 / 12777.5 / 12789.8 (5.2) |
| Follow-up, resent image | 1 | 8.8 / 8.8 / 8.8 (113.8) | 11.7 / 11.7 / 11.7 (86.0) | 140.3 / 153.9 / 153.9 (6.9) | 164.0 / 183.7 / 187.7 (5.9) |
| Follow-up, resent image | 4 | 21.9 / 34.7 / 35.0 (115.0) | 17.4 / 17.6 / 17.6 (230.2) | 162.1 / 182.3 / 186.7 (21.9) | 228.8 / 243.0 / 244.7 (16.5) |
| Follow-up, resent image | 16 | 73.6 / 129.3 / 138.1 (116.0) | 32.4 / 48.6 / 48.8 (329.3) | 233.6 / 249.9 / 259.0 (63.2) | 407.4 / 425.0 / 437.2 (37.5) |
| Follow-up, resent image | 64 | 279.6 / 498.4 / 549.3 (116.4) | 97.0 / 160.2 / 175.7 (365.5) | 971.9 / 1001.0 / 1015.7 (62.9) | 1748.8 / 1799.8 / 1811.1 (35.9) |
| GUI box, ≤2,048 tokens | 1 | 114.8 / 114.8 / 114.8 (8.7) | 98.6 / 98.6 / 98.6 (10.2) | 233.8 / 247.8 / 247.8 (4.8) | 296.1 / 310.5 / 311.0 (3.7) |
| GUI box, ≤2,048 tokens | 4 | 287.4 / 459.6 / 460.1 (8.7) | 264.1 / 358.6 / 361.2 (11.2) | 500.6 / 521.4 / 522.8 (7.7) | 692.8 / 709.1 / 710.8 (5.7) |
| GUI box, ≤2,048 tokens | 16 | 972.7 / 1720.2 / 1839.5 (8.7) | 800.0 / 1391.3 / 1417.6 (11.4) | 1588.2 / 1621.3 / 1643.8 (9.9) | 2182.1 / 2221.7 / 2222.0 (7.3) |
| GUI box, ≤2,048 tokens | 64 | 3728.4 / 6647.2 / 7330.6 (8.7) | 2917.2 / 5196.4 / 5621.2 (11.4) | 5899.3 / 5974.2 / 5992.8 (10.7) | 8480.1 / 8589.6 / 8590.4 (7.5) |
| 32 fresh frames, yes/no | 1 | 160.1 / 160.5 / 160.5 (6.4) | 160.3 / 160.8 / 160.9 (6.3) | 175.0 / 175.9 / 176.3 (5.7) | 263.0 / 275.4 / 278.6 (3.8) |
| 32 fresh frames, yes/no | 4 | 392.8 / 625.8 / 628.1 (6.4) | 391.0 / 622.5 / 623.3 (6.4) | 471.1 / 635.5 / 637.2 (6.3) | 708.9 / 807.1 / 816.5 (5.0) |
| 32 fresh frames, yes/no | 16 | 1331.5 / 2340.4 / 2499.3 (6.4) | 1321.5 / 2320.6 / 2482.5 (6.5) | 1532.7 / 2549.8 / 2558.2 (6.3) | 2343.4 / 3059.9 / 3473.7 (5.1) |
| 32 fresh frames, yes/no | 64 | 5080.4 / 9073.7 / 10000.7 (6.4) | 5040.6 / 9002.4 / 9915.8 (6.5) | 5851.3 / 9600.6 / 10663.0 (6.2) | 7921.3 / 12242.5 / 13990.2 (4.9) |

## Where scheduled time went

Stage entries are measured / modeled milliseconds per burst. Vision includes pixel upload and synchronization. LM includes the question blocks, readout, transfer to host and conversion to typed answers. CPU preparation has a zero compute-floor entry. Admission and event-loop delivery are included in end-to-end time. Independent medians can differ from the median total.

| Workload | Clients | CPU prep | Vision + H2D | LM + readout |
|---|---:|---:|---:|---:|
| Cold box | 1 | 0.4 / 0.0 | 9.8 / 5.4 | 20.9 / 11.8 |
| Cold box | 4 | 0.9 / 0.0 | 37.2 / 21.3 | 64.0 / 46.5 |
| Cold box | 16 | 3.1 / 0.0 | 138.3 / 85.9 | 228.2 / 187.0 |
| Cold box | 64 | 10.9 / 0.0 | 549.3 / 341.3 | 878.0 / 742.9 |
| Cold yes/no | 1 | 0.4 / 0.0 | 9.7 / 5.4 | 20.9 / 12.0 |
| Cold yes/no | 4 | 1.3 / 0.0 | 36.3 / 21.3 | 66.7 / 47.5 |
| Cold yes/no | 16 | 3.3 / 0.0 | 138.6 / 85.9 | 233.9 / 190.7 |
| Cold yes/no | 64 | 11.5 / 0.0 | 551.2 / 341.3 | 902.6 / 757.9 |
| Eight mixed questions | 1 | 1.2 / 0.0 | 9.8 / 5.4 | 33.3 / 17.1 |
| Eight mixed questions | 4 | 3.0 / 0.0 | 36.3 / 21.3 | 100.5 / 68.1 |
| Eight mixed questions | 16 | 11.0 / 0.0 | 139.7 / 85.9 | 387.2 / 272.1 |
| Eight mixed questions | 64 | 42.5 / 0.0 | 560.8 / 341.3 | 1,554.9 / 1,084.6 |
| Follow-up, resent image | 1 | 1.7 / 0.0 | 0.0 / 0.0 | 8.7 / 5.0 |
| Follow-up, resent image | 4 | 5.7 / 0.0 | 0.0 / 0.0 | 10.3 / 5.2 |
| Follow-up, resent image | 16 | 22.6 / 0.0 | 0.0 / 0.0 | 25.5 / 6.2 |
| Follow-up, resent image | 64 | 87.1 / 0.0 | 0.0 / 0.0 | 85.8 / 19.8 |
| GUI box, ≤2,048 tokens | 1 | 0.5 / 0.0 | 42.8 / 29.0 | 53.8 / 38.2 |
| GUI box, ≤2,048 tokens | 4 | 1.5 / 0.0 | 165.6 / 115.2 | 189.9 / 152.0 |
| GUI box, ≤2,048 tokens | 16 | 4.6 / 0.0 | 661.6 / 461.2 | 733.9 / 608.5 |
| GUI box, ≤2,048 tokens | 64 | 17.6 / 0.0 | 2,658.4 / 1,838.7 | 2,940.9 / 2,428.7 |
| 32 fresh frames, yes/no | 1 | 0.5 / 0.0 | 51.3 / 25.4 | 107.1 / 87.3 |
| 32 fresh frames, yes/no | 4 | 1.8 / 0.0 | 204.8 / 101.4 | 414.2 / 346.4 |
| 32 fresh frames, yes/no | 16 | 6.4 / 0.0 | 816.8 / 405.7 | 1,650.1 / 1,382.7 |
| 32 fresh frames, yes/no | 64 | 25.4 / 0.0 | 3,264.7 / 1,622.8 | 6,602.5 / 5,530.8 |

## Standard FLA and optional SM120 prefill

The standard FLA stack ran the same 24 comparisons with the shipping scheduler settings. Its median burst time beat both serving baselines in 21 of 24 cells. These are separate five-burst medians. The native BF16 kernel is eligible on the one-pass packed prefill path. Mixed-question and cache-hit timings also reflect run-to-run variation. [Full standard-stack samples](raw/item6_validation_standard.json) · [Full SM120 samples](raw/item6_production.json)

| 64 clients | Standard FLA ms | Optional SM120 ms |
|---|---:|---:|
| Cold box | 1,486.8 | 1,439.4 |
| Cold yes/no | 1,515.3 | 1,471.5 |
| Eight mixed questions | 2,163.9 | 2,161.0 |
| Cached box, resent image | 175.2 | 175.2 |
| GUI box, ≤2,048 tokens | 5,770.8 | 5,621.2 |
| 32 fresh frames, yes/no | 10,238.4 | 9,917.9 |

Standard-stack losses against the fastest serving baseline occurred on cold yes/no at 16 clients (383.9 ms vs 380.8 ms), cold yes/no at 64 clients (1,515.3 ms vs 1,504.3 ms), 32 fresh frames, yes/no at 4 clients (640.3 ms vs 635.4 ms).

## Scheduling and remaining headroom

The first cross-request scheduler cut the 64-client cold box burst to 1,550.1 ms. Increasing its token budget from 8,192 to 16,384 kept throughput at 41.4 requests/s while raising p50 from 883.6 to 1,207.7 ms. A 4,096-token budget reached 41.1 requests/s. In the wait-window sweep, the one-client box time was about 31 ms at zero wait and 33 ms at 2 ms. The 0.5 ms default is a tradeoff for these synchronized bursts. Arrival-rate sweeps and long-running queue saturation are outside this dataset. [First scheduler](raw/item6_scheduled.json) · [4k](raw/item6_tune4096.json) · [16k](raw/item6_tune16384.json) · [Zero wait](raw/item6_wait0.json) · [2 ms wait](raw/item6_wait2.json)

The second implementation packed shared prefixes and question blocks, reused spatial templates and grouped vision by grid. On 64 cold box clients the final 8k run used six GPU steps. Vision and H2D took 549.3 ms against a 341.3 ms floor, LM and readout took 878.0 ms against about 743 ms, and CPU preparation took 10.9 ms. This reaches 75.3% of the modeled ceiling. Fresh 32-frame scans already batch the frames of each client, and at 64 clients the scheduler still needed 64 steps. They gained 0.9% over serial Glance. [Stage samples](raw/item6_production.json) · [Stage floors](raw/item6_floors.json)

Cached images show the largest relative headroom. At 64 clients, resending pixels took 175.2 ms against a 19.8 ms ideal device floor. Lookup, host preparation and six query steps contributed to the gap. Passing returned image IDs took **66.3 ms** and **964.7 requests/s** with one query step, including 9.5 ms of preparation and 55.4 ms of LM/readout. The ID burst times at 1/4/16/64 clients were 10.4 / 12.5 / 22.1 / 66.3 ms. These timings exclude the initial retained-prefix construction. [Image follow-ups](raw/item6_production.json) · [ID follow-ups](raw/item6_production_ids.json)

## Equivalence and HTTP checks

The randomized test ran 32 logical requests in isolation, then in two shuffled waves covering single and multiple images, mixed question types, cache insertion and cache reuse. The two waves produced 254/254 matching decisions against each reference, with no continuous-output tolerance violations. Isolated requests also matched shipped BF16 on 127/127 decisions. Standard FLA and the optional native kernel passed. [Final native-kernel check](raw/item6_compatibility_production_check.json) · [Final standard-stack check](raw/item6_validation_standard_check.json) · [Test and references](../../../speed/test_scheduler_gpu.py)

| Maximum absolute difference | Concurrent vs isolated | Concurrent vs shipped BF16 | Existing limit |
|---|---:|---:|---:|
| Presence probability | 0.014418 | 0.013550 | 0.031704 |
| IoU probability | 0.051116 | 0.050738 | 0.407418 |
| Normalized box coordinate | 0.039069 | 0.040033 | 0.424945 |
| Typed probability | 0.018807 | 0.011805 | 0.023353 |

Early layouts passed the decision check but exceeded the typed-probability gate. Keeping isolated GDN chunk boundaries, producing a canonical retained state at insertion and grouping equal vision grids cleared the gate. The limits and model weights are unchanged. The original scan suite passed 234/234 real-weight decisions on engopt. The tiny CPU FP32 suite passed both masks and the legacy clip endpoint check on the Mac. The CUDA-hidden VM attempt encountered a GPU-only FLA operator. All five BF16 engine variants in the 60-image diagnostic matched shipped BF16 on 930/930 decisions each. [Intermediate outliers](raw/item6_release_equiv.json) · [Grid control](raw/item6_grouped_equiv.json) · [Real scan suite](raw/item6_release_scan.log) · [Tiny CPU suite](raw/item6_cpu_release.log) · [60-image diagnostic](raw/item6_release_diag.json)

The loopback HTTP load test used 16 clients, one warm-up and five bursts of base64 JPEGs with real weights. Three independent runs measured 57.7–58.2 requests/s. The final run measured 58.1 requests/s and p50/p90/p99 of 252.2 / 337.5 / 338.4 ms. It exercised typed and chat decisions, ID hit/miss, malformed input, clip scoring, video encode/scan, readiness, health and metrics. The two failed requests in the metrics are the intentional missing image and video keys. Model-free transport and scheduler tests cover saturation, cancellation, timeouts, graceful drain, LRU eviction, in-order oversized requests and injected OOM split/retry. The final legacy smoke also exercised trailing slashes, an empty question list with a scalar question, and default clip timestamps. Physical device exhaustion, process restart behind the external reverse tunnel and WAN latency remain unmeasured. [HTTP observations](raw/item6_production_http.json) · [Repeat HTTP run](raw/item6_validation_production_http.json) · [Fault tests](raw/item6_transport_final.log) · [Final legacy smoke](raw/item6_compatibility_production_http.json) · [Engopt fault tests](raw/item6_compatibility_api_tests.log) · [HTTP client](../../../speed/load_server.py)

## Reproduction

From the worktree root, use a fresh run label to preserve independent samples. The runner holds both GPU locks throughout the chain and archives each stage's JSON and log. For an interrupted run, reuse its label to resume completed benchmark cells.

```bash
MONO=~/Developer/decision-vison/.claude/worktrees/engine-opt bash scan/vm/sync.sh engopt
bash scan/vm/bg.sh engopt item6_review 'ITEM6_RUN_LABEL=review bash scan/vm/scheduler_bench.sh floors warm_serial vllm sglang standard_check production_check production production_ids production_http standard release_scan release_diag api_tests'
```

## Authorship

Codex wrote the scheduler, worker, packed multi-question prefix path, HTTP service, compatibility adapter, benchmark harness, tests and report, and ran the measurements. The reviewer supplied the prepared engopt VM and earlier comparison data. The worktree contains the uncommitted implementation for review.
