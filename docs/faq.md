# FAQ

**Why not serve Glance with vLLM or SGLang?**
Both are built around autoregressive decoding: a request is a prompt plus generated tokens, and scheduling, caching and kernels are tuned for that loop. Glance has no loop. Its unit of work is a prefix plus a set of short question blocks read out by heads, and its expensive object is a reusable hybrid prefix state (GDN recurrent state plus attention K/V). Serving that well needs branch kernels that read one state for many questions, banks that hold many states for one question, and a scheduler that batches decisions rather than decode steps. The [performance page](performance.md) shows what that buys and where it does not help (prefill-bound fresh scans).

**Is the comparison with vLLM fair? It runs a different model.**
The baselines run the generative Qwen3.5-4B, which Glance is fine-tuned from, on the same resized pixels, question text and request boundaries, and must produce the same kind of answer (a box as up to 24 tokens, or one yes/no token with logprobs). They get prefix caching, CUDA graphs, continuous batching, and for vLLM also FP8 weights with MTP speculative decoding. The timings compare the cost of getting a decision with each serving approach. They say nothing about which model answers more accurately; Glance's accuracy is reported with the model in [glance-core](https://github.com/Vision-Decision/glance-core).

**Does the engine change Glance's answers?**
Not beyond BF16 noise. On CPU in FP32 the engine matches the reference forward to about 2×10⁻⁶. On GPU in BF16 every presence and typed decision matched the shipped model in the 930-decision diagnostic and the 234-decision suite. Box centres can switch between near-tied patches, as they do when the shipped model itself runs with a different batch layout. See [exactness](exactness.md).

**What GPU do I need?**
Real checkpoints run in BF16 on CUDA. All numbers were measured on an RTX PRO 6000 Blackwell (SM120); several custom kernels are SM120-only and fall back to FLA or PyTorch elsewhere, so other GPUs work but are slower and unmeasured. The CPU path (`Glance.tiny()`) exists for tests.

**How much memory does a cached image cost?**
A scan bank holds 72.87 MiB per ≤640-token image in the default lossless compact format (75.49 MiB before item 8). Most of it is fixed recurrent state: 48 MiB of FP32 GDN state per prefix regardless of image size, so smaller images save less than you might expect. The server's image-prefix cache keeps the same kinds of tensors and is an LRU bounded by `--cache-gib` (8 GiB by default). Banks too large for the GPU can live in pinned host memory (`tier="cpu"`) and stream at close to PCIe speed. [Item 8](../outputs/engine_opt/08_bank_memory/summary.md)

**When does building a bank pay off?**
From the second question on most shapes: on a 64-input bank with changing questions, cumulative cached time beat repeated fresh scans on question two for Tv640/Tv256 images and four-frame clips, and on question five for Tv64 images. [Finding](../outputs/engine_opt/FINDINGS.md#strongest-findings)

**What does "calibrated" mean here?**
Two things. Per answer: release checkpoints ship temperatures fitted on held-out data, and `Glance.ask` applies them, so a probability of 0.8 is meant to be right about 80% of the time on in-distribution data. Per set: after `Scanner.calibrate(...)` on scans known not to contain the target, the "not in any of these" verdict has a false-alarm rate of at most α on new target-free scans, on average over calibration draws, by a split-conformal max-statistic rule. It bounds false alarms, not misses.

**Why is the 4,096-token GUI case slower than vLLM FP8+MTP?**
With one question, a 4k-token screenshot is almost all prefill, and vLLM is running FP8 weights while Glance runs BF16. Glance's optional FP8 language model exists but is a separate, lossy mode. From eight questions on the same screenshot Glance leads again (282.9 vs 299.5 ms).

**Why do `speed/` and `scan/` still hold the engine code?**
Work continues on those paths, so the release keeps file locations stable and adds `glance_engine` as the public API and stable import surface. See [architecture](architecture.md#repository-map).

**Why is the first call slow?**
The first call of each new shape compiles Triton kernels and captures CUDA graphs, which takes seconds (27.5 s for the very first image on a fresh G4 process). Later calls of that shape replay the graph. Warm up the shapes you serve before timing; see the [G4 smoke test](smoke_g4.md).

**Can I use the HTTP server from an OpenAI client?**
`POST /v1/chat/completions` accepts one user message with image data URLs and a yes/no question and returns the typed results as JSON text in the assistant message. For anything else use `/v1/decisions`.
