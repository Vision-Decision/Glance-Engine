# Sprint speed: generated tables

## G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition)

### Latency per call, 1 image (<=640 visual tokens), N box questions: median ms

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 87 | 87 | 87 | 87 | 87 | 96 | 124 |
| Glance, LoRA merged (eager) | 66 | 66 | 66 | 66 | 72 | 83 | 110 |
| Glance engine (eager) | 62 | 62 | 62 | 62 | 63 | 69 | 95 |
| Glance engine (CUDA graphs, bf16) | 46 | 47 | 47 | 49 | 53 | 64 | 90 |
| Glance engine (CUDA graphs + fused ops, bf16) | 43 | 43 | 43 | 45 | 50 | 61 | 85 |
| Glance engine (CUDA graphs, FP8 W8A8) | 36 | 36 | 37 | 38 | 43 | 51 | 74 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 187 | 215 | 242 | 303 | 390 | 589 | 965 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 157 | 184 | 202 | 272 | 377 | 764 | 1259 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 39 | 50 | 53 | 65 | 87 | 141 | 234 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 167 | 186 | 202 | 221 | 259 | 334 | 481 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 35 | 42 | 46 | 53 | 72 | 110 | 178 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 99 | 112 | 117 | 134 | 166 | 229 | 361 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 169 | 185 | 203 | 222 | 259 | 334 | 482 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 118 | 134 | 144 | 164 | 200 | 283 | 449 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | 154 | 154 | 155 | 157 | 161 | 171 | 197 |

p90:

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 87 | 88 | 88 | 88 | 90 | 108 | 154 |
| Glance, LoRA merged (eager) | 67 | 67 | 67 | 69 | 75 | 94 | 136 |
| Glance engine (eager) | 62 | 63 | 63 | 62 | 63 | 72 | 104 |
| Glance engine (CUDA graphs, bf16) | 48 | 48 | 48 | 50 | 56 | 66 | 98 |
| Glance engine (CUDA graphs + fused ops, bf16) | 44 | 44 | 44 | 47 | 54 | 64 | 93 |
| Glance engine (CUDA graphs, FP8 W8A8) | 37 | 37 | 38 | 40 | 44 | 54 | 81 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 203 | 234 | 252 | 311 | 403 | 606 | 993 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 170 | 197 | 224 | 286 | 394 | 795 | 1330 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 40 | 50 | 56 | 70 | 94 | 150 | 260 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 181 | 197 | 208 | 228 | 267 | 349 | 508 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 35 | 44 | 47 | 55 | 74 | 117 | 190 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 100 | 120 | 126 | 145 | 174 | 244 | 385 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 184 | 198 | 209 | 229 | 267 | 348 | 509 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 123 | 136 | 154 | 175 | 213 | 304 | 479 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – | – |

Time to first answer (ms; Glance: all answers arrive together):

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 87 | 87 | 87 | 87 | 87 | 96 | 124 |
| Glance, LoRA merged (eager) | 66 | 66 | 66 | 66 | 72 | 83 | 110 |
| Glance engine (eager) | 62 | 62 | 62 | 62 | 63 | 69 | 95 |
| Glance engine (CUDA graphs, bf16) | 46 | 47 | 47 | 49 | 53 | 64 | 90 |
| Glance engine (CUDA graphs + fused ops, bf16) | 43 | 43 | 43 | 45 | 50 | 61 | 85 |
| Glance engine (CUDA graphs, FP8 W8A8) | 36 | 36 | 37 | 38 | 43 | 51 | 74 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 187 | 211 | 219 | 277 | 362 | 548 | 905 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 157 | 179 | 189 | 247 | 337 | 719 | 1179 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 39 | 39 | 40 | 41 | 44 | 48 | 53 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 167 | 174 | 178 | 195 | 229 | 292 | 422 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 35 | 40 | 39 | 50 | 55 | 60 | 63 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 99 | 104 | 103 | 114 | 142 | 198 | 298 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 169 | 174 | 177 | 195 | 228 | 291 | 427 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 118 | 120 | 127 | 142 | 175 | 241 | 372 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – | – |

### Mixed typed questions (box : yes/no : choice : score = 1:1:1:1), 1 image: median ms

| system | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 87 | 88 | 105 | 149 | 246 |
| Glance, LoRA merged (eager) | 67 | 74 | 91 | 130 | 212 |
| Glance engine (eager) | 62 | 63 | 72 | 100 | 163 |
| Glance engine (CUDA graphs, bf16) | 50 | 56 | 66 | 95 | 164 |
| Glance engine (CUDA graphs + fused ops, bf16) | 46 | 54 | 64 | 90 | 157 |
| Glance engine (CUDA graphs, FP8 W8A8) | 39 | 44 | 56 | 76 | 134 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 212 | 284 | 382 | 554 | 902 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 197 | 257 | 356 | 703 | 1112 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 53 | 64 | 82 | 129 | 213 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 173 | 199 | 224 | 278 | 380 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 45 | 52 | 67 | 100 | 158 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 102 | 122 | 139 | 177 | 258 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 173 | 198 | 225 | 278 | 379 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 124 | 143 | 170 | 228 | 334 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – |

### Follow-up questions on an already-seen image (warm prefix cache), median ms

| system | box_N1 | mixed_N1 | box_N8 | mixed_N8 | box_N32 | mixed_N32 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | – | – | – | – | – | – |
| Glance, LoRA merged (eager) | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 13 | 13 | 16 | 21 | 30 | 60 |
| Glance engine (CUDA graphs + fused ops, bf16) | 11 | 11 | 14 | 20 | 28 | 57 |
| Glance engine (CUDA graphs, FP8 W8A8) | 9 | 9 | 12 | 16 | 23 | 49 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 162 | 161 | 292 | 281 | 588 | 557 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 132 | 131 | 275 | 255 | 625 | 551 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 14 | 14 | 39 | 37 | 120 | 105 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 145 | 145 | 197 | 175 | 317 | 255 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 12 | 12 | 31 | 30 | 88 | 79 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 78 | 78 | 113 | 97 | 212 | 157 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 145 | 145 | 197 | 175 | 316 | 255 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 101 | 101 | 136 | 117 | 269 | 203 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – |

### Batched images (throughput regime): questions/s

| system | bs2_N1 | bs2_N8 | bs4_N1 | bs4_N8 | bs8_N1 | bs8_N8 | bs16_N1 | bs16_N8 | bs32_N1 | bs32_N8 | bs4_N32 | bs8_N32 | bs16_N32 | bs4_N64 | bs8_N64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 18 | 131 | 20 | 138 | 20 | 133 | 19 | 126 | 19 | 122 | – | – | – | – | – |
| Glance, LoRA merged (eager) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 24 | 176 | 26 | 175 | 24 | 166 | 23 | 157 | 23 | 156 | – | – | – | – | – |
| Glance engine (CUDA graphs + fused ops, bf16) | 26 | 191 | 27 | 186 | 25 | 176 | 25 | 171 | 25 | 170 | 506 | 473 | 449 | 670 | 644 |
| Glance engine (CUDA graphs, FP8 W8A8) | 30 | 209 | 31 | 212 | 29 | 199 | 28 | 191 | 27 | 187 | 570 | 527 | 501 | 750 | 717 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 10 | 41 | 17 | 55 | 21 | 53 | 40 | 52 | 53 | 49 | – | – | – | – | – |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 11 | 36 | 21 | 44 | 22 | 34 | 35 | 34 | 52 | 33 | – | – | – | – | – |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 30 | 165 | 41 | 157 | 42 | 162 | 40 | 152 | 41 | 160 | 260 | 253 | 256 | 293 | 285 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 10 | 59 | 16 | 79 | 23 | 98 | 28 | 102 | 32 | 109 | – | – | – | – | – |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 35 | 214 | 55 | 207 | 55 | 216 | 54 | 201 | 55 | 216 | 361 | 351 | 358 | 387 | 398 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 16 | 92 | 27 | 108 | 35 | 124 | 39 | 130 | 43 | 136 | 186 | 188 | 192 | 207 | 207 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 10 | 59 | 16 | 80 | 23 | 98 | 28 | 102 | 32 | 109 | – | – | – | – | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 13 | 75 | 21 | 88 | 26 | 99 | 30 | 103 | 33 | 107 | 151 | 151 | 153 | 163 | 164 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |

### Video-style: F frames (<=256 visual tokens each) x N questions per frame, one call: median ms

| system | f8_N1 | f8_N4 | f32_N1 | f32_N4 | f64_N1 | f64_N4 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 180 | 260 | 774 | 1252 | 1561 | 2549 |
| Glance, LoRA merged (eager) | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 140 | 197 | 642 | 918 | 1282 | 1842 |
| Glance engine (CUDA graphs + fused ops, bf16) | 133 | 185 | 576 | 833 | 1161 | 1678 |
| Glance engine (CUDA graphs, FP8 W8A8) | 113 | 160 | 509 | 739 | 1038 | 1495 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 304 | 475 | 412 | 877 | 609 | 1530 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 281 | 535 | 413 | 1389 | 663 | 2850 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 106 | 263 | 353 | 1022 | 682 | 2022 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 259 | 419 | 549 | 1211 | 933 | 2269 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 85 | 189 | 249 | 707 | 482 | 1407 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 158 | 280 | 392 | 907 | 696 | 1739 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 259 | 419 | 553 | 1209 | 933 | 2270 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 211 | 377 | 506 | 1241 | 920 | 2390 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – |

(questions/s)

| system | f8_N1 | f8_N4 | f32_N1 | f32_N4 | f64_N1 | f64_N4 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 45 | 123 | 41 | 102 | 41 | 100 |
| Glance, LoRA merged (eager) | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 57 | 163 | 50 | 139 | 50 | 139 |
| Glance engine (CUDA graphs + fused ops, bf16) | 60 | 173 | 56 | 154 | 55 | 153 |
| Glance engine (CUDA graphs, FP8 W8A8) | 71 | 200 | 63 | 173 | 62 | 171 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 27 | 71 | 78 | 146 | 96 | 163 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 30 | 64 | 75 | 90 | 96 | 89 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 80 | 124 | 91 | 125 | 94 | 127 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 31 | 77 | 58 | 106 | 69 | 113 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 101 | 175 | 128 | 181 | 133 | 178 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 51 | 115 | 82 | 136 | 92 | 147 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 31 | 77 | 58 | 106 | 69 | 111 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 38 | 85 | 63 | 103 | 69 | 106 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – |

### GUI screenshots (ScreenSpot-Pro, <=2048 / <=4096 visual tokens), 1 image: median ms

| system | gui2048_N1 | gui2048_N8 | gui2048_N32 | gui4096_N1 | gui4096_N8 | gui4096_N32 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 174 | 176 | 197 | 364 | 368 | 403 |
| Glance, LoRA merged (eager) | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 132 | 135 | 158 | 310 | 314 | 340 |
| Glance engine (CUDA graphs + fused ops, bf16) | 128 | 130 | 155 | 299 | 304 | 331 |
| Glance engine (CUDA graphs, FP8 W8A8) | 112 | 115 | 136 | 276 | 281 | 308 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 288 | 464 | 1297 | 483 | 709 | 2241 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 279 | 479 | 1373 | 461 | 687 | 2273 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 108 | 175 | 392 | 237 | 277 | 374 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 226 | 312 | 572 | 354 | 418 | 566 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 85 | 129 | 276 | 201 | 229 | 291 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 150 | 208 | 419 | 262 | 299 | 436 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 227 | 313 | 571 | 353 | 419 | 567 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 182 | 267 | 552 | 314 | 386 | 538 |
| AR lower bound w/o speculative decoding (Glance prefill + 19.4 tok x 5.5 ms weight read) | – | – | – | – | – | – |

### Speed-up of the Glance engine (best bf16 variant; numerically equivalent) over the strongest generative baseline per regime

| regime | Glance ms | best baseline ms | best baseline | speed-up |
|---|---|---|---|---|
| bs1_box_N1 | 43.0 | 99.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.31x |
| bs1_box_N2 | 42.8 | 111.9 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.61x |
| bs1_box_N4 | 43.3 | 116.7 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.70x |
| bs1_box_N8 | 44.9 | 134.0 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.98x |
| bs1_box_N16 | 50.1 | 165.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 3.31x |
| bs1_box_N32 | 61.0 | 229.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 3.76x |
| bs1_box_N64 | 84.6 | 361.0 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 4.27x |
| bs1_mixed_N4 | 46.5 | 102.4 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.20x |
| bs1_mixed_N8 | 53.5 | 122.1 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.28x |
| bs1_mixed_N16 | 63.8 | 139.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.18x |
| bs1_mixed_N32 | 90.2 | 177.0 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.96x |
| bs1_mixed_N64 | 157.1 | 257.7 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.64x |
| followup_box_N1 | 10.8 | 78.3 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 7.22x |
| followup_mixed_N1 | 10.8 | 78.5 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 7.25x |
| followup_box_N8 | 13.6 | 112.9 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 8.32x |
| followup_mixed_N8 | 19.6 | 96.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 4.94x |
| followup_box_N32 | 27.9 | 211.8 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 7.59x |
| followup_mixed_N32 | 57.0 | 157.4 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.76x |
| gui2048_box_N1 | 127.7 | 149.5 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.17x |
| gui2048_box_N8 | 130.3 | 208.0 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.60x |
| gui2048_box_N32 | 154.6 | 418.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 2.71x |
| gui4096_box_N1 | 298.7 | 262.4 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 0.88x |
| gui4096_box_N8 | 303.7 | 299.5 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 0.99x |
| gui4096_box_N32 | 331.2 | 435.7 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.32x |
| video_f8_N1 | 133.0 | 157.9 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.19x |
| video_f8_N4 | 185.3 | 280.1 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1.51x |
| video_f32_N1 | 575.6 | 391.8 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 0.68x |
| video_f32_N4 | 832.6 | 876.8 | SGLang 0.5.23 + prefix cache | 1.05x |
| video_f64_N1 | 1160.7 | 609.3 | SGLang 0.5.23 + prefix cache | 0.52x |
| video_f64_N4 | 1677.8 | 1530.4 | SGLang 0.5.23 + prefix cache | 0.91x |

### Saturation throughput and cost per 1k questions (assumed $2.00/GPU-hour)

| system | questions/image | best q/s (config) | GPU-s per 1k q | $ per 1k q |
|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 1 | 20 (bs4_box_N1) | 49.89 | 0.0277 |
| Glance, shipped path (PEFT LoRA, eager) | 8 | 138 (bs4_box_N8) | 7.24 | 0.0040 |
| Glance, shipped path (PEFT LoRA, eager) | 32 | 324 (bs1_box_N32) | 3.08 | 0.0017 |
| Glance, shipped path (PEFT LoRA, eager) | 64 | 478 (bs1_box_N64) | 2.09 | 0.0012 |
| Glance, LoRA merged (eager) | 1 | 15 (bs1_box_N1) | 65.93 | 0.0366 |
| Glance, LoRA merged (eager) | 8 | 119 (bs1_box_N8) | 8.41 | 0.0047 |
| Glance, LoRA merged (eager) | 32 | 372 (bs1_box_N32) | 2.69 | 0.0015 |
| Glance, LoRA merged (eager) | 64 | 541 (bs1_box_N64) | 1.85 | 0.0010 |
| Glance engine (eager) | 1 | 16 (bs1_box_N1) | 61.68 | 0.0343 |
| Glance engine (eager) | 8 | 129 (bs1_box_N8) | 7.73 | 0.0043 |
| Glance engine (eager) | 32 | 457 (bs1_box_N32) | 2.19 | 0.0012 |
| Glance engine (eager) | 64 | 659 (bs1_box_N64) | 1.52 | 0.0008 |
| Glance engine (CUDA graphs, bf16) | 1 | 26 (bs4_box_N1) | 39.02 | 0.0217 |
| Glance engine (CUDA graphs, bf16) | 8 | 176 (bs2_box_N8) | 5.68 | 0.0032 |
| Glance engine (CUDA graphs, bf16) | 32 | 499 (bs1_box_N32) | 2.01 | 0.0011 |
| Glance engine (CUDA graphs, bf16) | 64 | 698 (bs1_box_N64) | 1.43 | 0.0008 |
| Glance engine (CUDA graphs + fused ops, bf16) | 1 | 27 (bs4_box_N1) | 37.51 | 0.0208 |
| Glance engine (CUDA graphs + fused ops, bf16) | 8 | 191 (bs2_box_N8) | 5.25 | 0.0029 |
| Glance engine (CUDA graphs + fused ops, bf16) | 32 | 519 (bs1_box_N32) | 1.93 | 0.0011 |
| Glance engine (CUDA graphs + fused ops, bf16) | 64 | 738 (bs1_box_N64) | 1.36 | 0.0008 |
| Glance engine (CUDA graphs, FP8 W8A8) | 1 | 31 (bs4_box_N1) | 32.31 | 0.0180 |
| Glance engine (CUDA graphs, FP8 W8A8) | 8 | 212 (bs4_box_N8) | 4.72 | 0.0026 |
| Glance engine (CUDA graphs, FP8 W8A8) | 32 | 618 (bs1_box_N32) | 1.62 | 0.0009 |
| Glance engine (CUDA graphs, FP8 W8A8) | 64 | 842 (bs1_box_N64) | 1.19 | 0.0007 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 1 | 53 (bs32_box_N1) | 18.83 | 0.0105 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 8 | 55 (bs4_box_N8) | 18.30 | 0.0102 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 32 | 53 (bs1_box_N32) | 18.71 | 0.0104 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache | 64 | 66 (bs1_box_N64) | 15.11 | 0.0084 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 1 | 52 (bs32_box_N1) | 19.29 | 0.0107 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 8 | 44 (bs4_box_N8) | 22.47 | 0.0125 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 32 | 42 (bs1_box_N32) | 23.58 | 0.0131 |
| Qwen3.5-4B generative, SGLang 0.5.23 + prefix cache, MTP spec. decoding k=2 | 64 | 51 (bs1_box_N64) | 19.63 | 0.0109 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 1 | 42 (bs8_box_N1) | 24.02 | 0.0133 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 8 | 165 (bs2_box_N8) | 6.07 | 0.0034 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 32 | 260 (bs4_box_N32) | 3.84 | 0.0021 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question (no box output) | 64 | 293 (bs4_box_N64) | 3.42 | 0.0019 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 1 | 32 (bs32_box_N1) | 31.14 | 0.0173 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 8 | 109 (bs32_box_N8) | 9.15 | 0.0051 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 32 | 96 (bs1_box_N32) | 10.39 | 0.0058 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 64 | 134 (bs1_box_N64) | 7.49 | 0.0042 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 1 | 55 (bs4_box_N1) | 18.04 | 0.0100 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 8 | 216 (bs32_box_N8) | 4.62 | 0.0026 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 32 | 361 (bs4_box_N32) | 2.77 | 0.0015 |
| Engine-matched decode-free bound: vLLM 0.34.0 prefill + 1 token per question, FP8 (no box output) | 64 | 398 (bs8_box_N64) | 2.51 | 0.0014 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 1 | 43 (bs32_box_N1) | 23.35 | 0.0130 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 8 | 136 (bs32_box_N8) | 7.36 | 0.0041 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 32 | 192 (bs16_box_N32) | 5.20 | 0.0029 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling, FP8 | 64 | 207 (bs8_box_N64) | 4.82 | 0.0027 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 1 | 32 (bs32_box_N1) | 31.11 | 0.0173 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 8 | 109 (bs32_box_N8) | 9.15 | 0.0051 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 32 | 96 (bs1_box_N32) | 10.39 | 0.0058 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache | 64 | 133 (bs1_box_N64) | 7.50 | 0.0042 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 1 | 33 (bs32_box_N1) | 30.42 | 0.0169 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 8 | 107 (bs32_box_N8) | 9.35 | 0.0052 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 32 | 153 (bs16_box_N32) | 6.52 | 0.0036 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 64 | 164 (bs8_box_N64) | 6.09 | 0.0034 |

## L4 (NVIDIA L4)

### Latency per call, 1 image (<=640 visual tokens), N box questions: median ms

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 315 | 315 | 315 | 327 | 366 | 444 | 611 |
| Glance engine (eager) | 226 | 228 | 237 | 257 | 296 | 354 | 522 |
| Glance engine (CUDA graphs, bf16) | 234 | 242 | 245 | 254 | 286 | 354 | 1535 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 871 | 858 | 971 | 1100 | 1314 | 1909 | 2971 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 615 | 610 | 679 | 822 | 1085 | 2048 | 3543 |

p90:

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 322 | 322 | 320 | 334 | 412 | 558 | 869 |
| Glance engine (eager) | 232 | 237 | 243 | 271 | 316 | 387 | 580 |
| Glance engine (CUDA graphs, bf16) | 243 | 250 | 259 | 273 | 315 | 380 | 1752 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 942 | 932 | 1003 | 1140 | 1378 | 2001 | 3168 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 647 | 674 | 730 | 885 | 1149 | 2181 | 3733 |

Time to first answer (ms; Glance: all answers arrive together):

| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 315 | 315 | 315 | 327 | 366 | 444 | 611 |
| Glance engine (eager) | 226 | 228 | 237 | 257 | 296 | 354 | 522 |
| Glance engine (CUDA graphs, bf16) | 234 | 242 | 245 | 254 | 286 | 354 | 1535 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 871 | 844 | 877 | 961 | 1168 | 1712 | 2665 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 615 | 607 | 617 | 721 | 961 | 1308 | 1310 |

### Mixed typed questions (box : yes/no : choice : score = 1:1:1:1), 1 image: median ms

| system | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 343 | 404 | 538 | 838 | 1537 |
| Glance engine (eager) | 269 | 298 | 400 | 578 | 1000 |
| Glance engine (CUDA graphs, bf16) | 270 | 302 | 396 | 578 | 1067 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 860 | 1003 | 1138 | 1482 | 2207 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 590 | 703 | 865 | 1329 | 2212 |

### Follow-up questions on an already-seen image (warm prefix cache), median ms

| system | box_N1 | mixed_N1 | box_N8 | mixed_N8 | box_N32 | mixed_N32 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | – | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 48 | 48 | 63 | 97 | 158 | 386 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 716 | 717 | 994 | 899 | 1799 | 1407 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 494 | 494 | 753 | 624 | 1942 | 1219 |

### Batched images (throughput regime): questions/s

| system | bs2_N1 | bs2_N8 | bs4_N1 | bs4_N8 | bs8_N1 | bs8_N8 | bs16_N1 | bs16_N8 | bs32_N1 | bs32_N8 | bs4_N32 | bs8_N32 | bs16_N32 | bs4_N64 | bs8_N64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 3 | 24 | 3 | 23 | 3 | 21 | 3 | 20 | 3 | 20 | – | – | – | – | – |
| Glance engine (eager) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | 4 | 25 | 4 | 27 | 4 | 27 | – | – | – | – | 75 | 72 | – | 105 | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 2 | 11 | 3 | 14 | 4 | 17 | 4 | 16 | 5 | 17 | – | – | – | – | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 3 | 13 | 4 | 14 | 4 | 14 | 4 | 14 | 4 | 14 | – | – | – | – | – |

### Video-style: F frames (<=256 visual tokens each) x N questions per frame, one call: median ms

| system | f8_N1 | f8_N4 | f32_N1 | f32_N4 | f64_N1 | f64_N4 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 1059 | 1647 | 4528 | 7549 | 9109 | 15132 |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | – | – | – | – | – | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 1347 | 2575 | 3572 | 8646 | 6420 | 16480 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 1122 | 2366 | 3775 | 8889 | 6993 | 17244 |

(questions/s)

| system | f8_N1 | f8_N4 | f32_N1 | f32_N4 | f64_N1 | f64_N4 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 8 | 19 | 7 | 17 | 7 | 17 |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | – | – | – | – | – | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 6 | 12 | 9 | 15 | 10 | 16 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 7 | 14 | 8 | 14 | 9 | 15 |

### GUI screenshots (ScreenSpot-Pro, <=2048 / <=4096 visual tokens), 1 image: median ms

| system | gui2048_N1 | gui2048_N8 | gui2048_N32 | gui4096_N1 | gui4096_N8 | gui4096_N32 |
|---|---|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 965 | 999 | 1140 | 2303 | 2342 | 2517 |
| Glance engine (eager) | – | – | – | – | – | – |
| Glance engine (CUDA graphs, bf16) | – | – | – | – | – | – |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 1279 | 1796 | 3820 | 2075 | 2391 | 3428 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 1070 | 1604 | 4064 | 1789 | 2161 | 3764 |

### Speed-up of the Glance engine (best bf16 variant; numerically equivalent) over the strongest generative baseline per regime

| regime | Glance ms | best baseline ms | best baseline | speed-up |
|---|---|---|---|---|
| bs1_box_N1 | 226.0 | 615.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.72x |
| bs1_box_N2 | 228.2 | 610.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.67x |
| bs1_box_N4 | 237.3 | 679.5 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.86x |
| bs1_box_N8 | 254.0 | 822.5 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 3.24x |
| bs1_box_N16 | 285.9 | 1084.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 3.79x |
| bs1_box_N32 | 353.6 | 1908.5 | vLLM 0.34.0 + prefix cache, async scheduling | 5.40x |
| bs1_box_N64 | 522.2 | 2971.4 | vLLM 0.34.0 + prefix cache, async scheduling | 5.69x |
| bs1_mixed_N4 | 268.5 | 589.8 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.20x |
| bs1_mixed_N8 | 298.5 | 702.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.35x |
| bs1_mixed_N16 | 395.7 | 865.2 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.19x |
| bs1_mixed_N32 | 578.0 | 1328.6 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 2.30x |
| bs1_mixed_N64 | 1000.1 | 2206.7 | vLLM 0.34.0 + prefix cache, async scheduling | 2.21x |
| followup_box_N1 | 47.8 | 494.1 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 10.34x |
| followup_mixed_N1 | 47.7 | 494.4 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 10.36x |
| followup_box_N8 | 63.0 | 753.4 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 11.95x |
| followup_mixed_N8 | 97.1 | 623.7 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 6.42x |
| followup_box_N32 | 158.2 | 1798.6 | vLLM 0.34.0 + prefix cache, async scheduling | 11.37x |
| followup_mixed_N32 | 385.8 | 1219.1 | vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 3.16x |

### Saturation throughput and cost per 1k questions (assumed $0.70/GPU-hour)

| system | questions/image | best q/s (config) | GPU-s per 1k q | $ per 1k q |
|---|---|---|---|---|
| Glance, shipped path (PEFT LoRA, eager) | 1 | 3 (bs2_box_N1) | 287.12 | 0.0558 |
| Glance, shipped path (PEFT LoRA, eager) | 8 | 25 (bs1_box_N8) | 40.81 | 0.0079 |
| Glance, shipped path (PEFT LoRA, eager) | 32 | 69 (bs1_box_N32) | 14.59 | 0.0028 |
| Glance, shipped path (PEFT LoRA, eager) | 64 | 96 (bs1_box_N64) | 10.47 | 0.0020 |
| Glance engine (eager) | 1 | 4 (bs1_box_N1) | 225.73 | 0.0439 |
| Glance engine (eager) | 8 | 31 (bs1_box_N8) | 32.26 | 0.0063 |
| Glance engine (eager) | 32 | 89 (bs1_box_N32) | 11.26 | 0.0022 |
| Glance engine (eager) | 64 | 121 (bs1_box_N64) | 8.25 | 0.0016 |
| Glance engine (CUDA graphs, bf16) | 1 | 4 (bs1_box_N1) | 232.83 | 0.0453 |
| Glance engine (CUDA graphs, bf16) | 8 | 31 (bs1_box_N8) | 32.30 | 0.0063 |
| Glance engine (CUDA graphs, bf16) | 32 | 90 (bs1_box_N32) | 11.11 | 0.0022 |
| Glance engine (CUDA graphs, bf16) | 64 | 105 (bs4_box_N64) | 9.53 | 0.0019 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 1 | 5 (bs32_box_N1) | 216.45 | 0.0421 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 8 | 17 (bs32_box_N8) | 59.79 | 0.0116 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 32 | 17 (bs1_box_N32) | 58.80 | 0.0114 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, async scheduling | 64 | 22 (bs1_box_N64) | 45.97 | 0.0089 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 1 | 4 (bs32_box_N1) | 226.60 | 0.0441 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 8 | 14 (bs8_box_N8) | 70.22 | 0.0137 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 32 | 16 (bs1_box_N32) | 64.09 | 0.0125 |
| Qwen3.5-4B generative, vLLM 0.34.0 + prefix cache, MTP spec. decoding k=2, async scheduling | 64 | 18 (bs1_box_N64) | 55.27 | 0.0107 |
