# Sprint: making Glance fast, with a fair efficiency comparison
Medians over 50 calls (20 on L4) after an untimed warm-up pass. Full tables: `tables.md`. Figure: `cost_vs_q_g4.{png,pdf}`.

**Headline.** On an RTX PRO 6000 (Blackwell), the Glance engine answers 1 / 8 / 32 box questions about a 640-token image in **43 / 45 / 61 ms**. That is bf16 and numerically equivalent to the reference forward. The strongest generative baseline we could configure needs **99 / 134 / 229 ms**: Qwen3.5-4B in vLLM 0.34 with prefix caching, CUDA graphs, async scheduling, MTP speculative decoding and FP8 weights. That makes Glance **2.3× / 3.0× / 3.8×** faster, and 2.7× / 3.6× / 4.6× faster than the same baseline in bf16. Follow-up questions on an already-seen screen take 11–14 ms vs 78–113 ms (**7–8×**). On an L4 the gap is 2.6× / 3.2× / 5.4× and 10–12× for follow-ups. Scope: when a call is prefill-bound (a 4k-token screenshot or 32–64 frames, with one question each), Glance is at parity or slower (0.5–1.0×).

**Settings.**
- Hardware: G4 = RTX PRO 6000 (1.46 TB/s; bf16 GEMM 277–343 TFLOP/s; FP8 473–663 TFLOP/s) and L4.
- Software: torch 2.14.1/cu130, transformers 5.23, fla 0.6, vLLM 0.34.0, SGLang 0.5.23.
- Weights: Glance-4B-preview (`v1/ckpt_final.pt`). Baseline: Qwen3.5-4B, thinking off, greedy.
- Workload: the first 100 RefCOCO-val images at ≤640 visual tokens; questions are the image's own expressions, cycled.
  - Mixed: box, yes/no, choice, score at 1:1:1:1.
  - GUI: 40 ScreenSpot-Pro screenshots at ≤2048 / ≤4096 tokens.
  - Video: frames at ≤256 tokens.
- What a call includes, on both sides: from the resized PIL image to all answers on the host, including the image processor; JPEG decode and resize excluded. All requests are submitted at once and the prefix cache is reset per call. The follow-up regime is the exception: warm prefix and encoder caches.
- Generated answers: box answers cap at 24 tokens (19.3 on average); typed answers are 1 token.
- vLLM settings: `max_num_seqs` 256, `max_num_batched_tokens` 16384, `gpu_memory_utilization` 0.85.

**Optimization steps (N = 1 / 8 / 32):**

| Step | ms |
|---|---|
| Shipped path | 87 / 87 / 96 |
| + LoRA merged | 66 / 66 / 83 |
| + rewritten engine, eager | 62 / 62 / 69 |
| + CUDA graphs | 46 / 49 / 64 |
| + fused ops | 43 / 45 / 61 |
| FP8 language model | 36 / 38 / 51 |
| Follow-up questions, cached prefix | 11 / 14 / 28 |

At N=1 the engine's GPU time is 42 ms, of which GEMMs take 27 ms, roughly the bf16 limit.

**Equivalence vs fp32** (60 images, 930 answers, 450 ground-truth boxes):

| Variant | mean ΔP(present) | decision agreement | Acc@0.5 |
|---|---|---|---|
| Shipped bf16 path | 0.00086 | 99.78% | 94.0 |
| bf16 engine (fused) | 0.00083 | 99.78% | 94.0 |
| FP8 language model | 0.00230 | 99.14% | 93.6 |
| FP8 language model + vision | 0.00336 | 99.46% | 94.0 |

The 4-tap conv is bit-identical to `F.conv1d`.

**G4 latency (ms), N = 1 / 2 / 4 / 8 / 16 / 32 / 64:**

| System | Latency |
|---|---|
| Glance bf16 | 43 / 43 / 43 / 45 / 50 / 61 / 85 |
| Glance FP8 | 36 / 36 / 37 / 38 / 43 / 51 / 74 |
| vLLM bf16 | 167 / 186 / 202 / 221 / 259 / 334 / 481 |
| vLLM bf16 + MTP | 118 / 134 / 144 / 164 / 200 / 283 / 449 |
| vLLM FP8 + MTP | 99 / 112 / 117 / 134 / 166 / 229 / 361 |
| SGLang (best) | 157 / 184 / 202 / 272 / 377 / 589 / 965 |
| AR floor without speculative decoding | 154 / … / 197 |
| vLLM, 1 token per question (engine-matched bound) | 39 / 50 / 53 / 65 / 87 / 141 / 234 |

**Other G4 regimes:**

| Regime | Glance (bf16) | Strongest baseline | Speed-up |
|---|---|---|---|
| Mixed typed, N = 4 / 8 / 16 / 32 / 64 | 46 / 54 / 64 / 90 / 157 | 102 / 122 / 139 / 177 / 258 | 2.2× → 1.6× |
| Follow-up: 1 box / 8 box / 8 mixed / 32 box / 32 mixed | 11 / 14 / 20 / 28 / 57 | 78 / 113 / 97 / 212 / 157 | 7.2× / 8.3× / 4.9× / 7.6× / 2.8× |
| GUI 2048 tokens, N = 1 / 8 / 32 | 128 / 130 / 155 | 150 / 208 / 419 | 1.2× / 1.6× / 2.7× |
| GUI 4096 tokens, N = 1 / 8 / 32 | 299 / 304 / 331 | 262 / 299 / 436 | 0.88× / 0.99× / 1.3× |
| Video, frames × questions: 8×1 / 8×4 / 32×1 / 32×4 / 64×1 / 64×4 | 133 / 185 / 576 / 833 / 1161 / 1678 | 158 / 280 / 392 / 877 / 609 / 1530 | 1.2× / 1.5× / 0.68× / 1.05× / 0.52× / 0.91× |

**Saturation throughput (questions/s) and cost, by questions per image:**

| Questions per image | Glance bf16 | Glance FP8 | vLLM FP8+MTP | SGLang |
|---|---|---|---|---|
| 1 | 27 | 28 | 43 | 53 |
| 8 | 191 | 207 | 136 | 55 |
| 32 | 519 | 618 | 192 | 53 |
| 64 | 738 | 842 | 207 | 66 |

GPU-seconds per 1k questions: Glance 37.5 / 5.2 / 1.9 / 1.4 vs vLLM 23.4 / 7.4 / 5.2 / 4.8.

**L4 (ms), Glance bf16 vs the best vLLM setting (bf16 + MTP):**

| Regime | Glance | vLLM | Speed-up |
|---|---|---|---|
| N=1 | 234 | 615 | 2.6× |
| N=8 | 254 | 822 | 3.2× |
| N=32 | 354 | 1909 | 5.4× |
| Follow-up, 1 question | 48 | 494 | 10.3× |
| Follow-up, 8 questions | 63 | 753 | 12× |
| Mixed, N=8 | 302 | 703 | 2.3× |
| Mixed, N=32 | 578 | 1329 | 2.3× |

**Engine-matched bound.** Within vLLM, a 1-token readout instead of full decoding gives 167→39, 221→65 and 334→141 ms (4.3× / 3.4× / 2.4× at N = 1 / 8 / 32). In the prefill-bound regimes it still gives 1.4–1.5× (GUI 4k ×1: 353→237; 64 frames ×1: 933→682). The decode-free readout is the source of the advantage. Where Glance loses, the cause is our research engine's batched and long prefill, not the readout.

**AR head.** The in-engine incremental version measures 446–603 ms. The no-speculation floor is 154 ms (N=1), and vLLM sits within about 10% of it, so vLLM is the AR reference.

**Rejected.** FP8 on the vision tower (box quality); FP8 on the L4 (slower); batching images (no gain once the GPU is saturated); CUDA graphs for large calls (memory thrash, no gain).
