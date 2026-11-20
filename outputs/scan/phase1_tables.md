GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; Glance v1 (bf16, merged LoRA, fused ops, CUDA graphs); vLLM 0.34.0 (prefix caching, async scheduling, CUDA graphs). Medians, ms, from the query to all N answers on the host.

## (A) cached clips: the video's clips were encoded once; one new yes/no question over N clips

Clip = 4 frames (2 temporal groups x <= 96 visual tokens, 197 prefix tokens). vLLM: same clips as native video, every clip already asked one question (warm cache); max_tokens=1 with logprobs.

| N clips | Glance | vLLM bf16 | vLLM FP8 | Glance speed-up vs best vLLM | vLLM cached tokens |
|---|---|---|---|---|---|
| 1 | 10.0 | 15.5 | — | 1.5× | 0.0 |
| 8 | 14.4 | 55.4 | — | 3.8× | 0.0 |
| 16 | 20.9 | 95.5 | — | 4.6× | 0.0 |
| 32 | 33.3 | 178 | — | 5.3× | 0.0 |
| 64 | 63.6 | 343 | — | 5.4× | 0.0 |
| 128 | 131 | 676 | — | 5.2× | 0.0 |
| 256 | 289 | 1355 | — | 4.7× | 0.0 |

## (A) cached images: one new question over N cached images

| Tv | N | Glance box (box + presence + IoU) | Glance yes/no | vLLM bf16 yes/no | vLLM bf16 box | vLLM FP8+MTP yes/no | vLLM FP8+MTP box | vLLM cached tokens |
|---|---|---|---|---|---|---|---|---|
| 640 | 1 | 9.6 | 9.8 | 14.2 | 26.7 | — | — | 576 |
| 640 | 8 | 12.2 | 14.3 | 34.9 | 176 | 31.2 | 98.6 | 576 |
| 640 | 16 | 14.8 | 21.1 | 54.0 | 197 | — | — | 576 |
| 640 | 32 | 22.4 | 34.7 | 230 | 401 | — | — | 576 |
| 640 | 64 | 39.9 | 68.5 | 681 | 899 | 583 | 721 | 567 |
| 640 | 128 | 77.6 | 146 | 1352 | 1643 | — | — | 562 |
| 640 | 256 | 164 | 313 | 2680 | 3152 | 2292 | 2672 | 569 |
| 256 | 1 | 9.6 | — | 17.3 | 30.5 | — | — | 0.0 |
| 256 | 8 | 12.0 | — | 68.4 | 191 | 50.4 | 116 | 0.0 |
| 256 | 16 | 14.4 | — | 122 | 258 | — | — | 0.0 |
| 256 | 32 | 21.7 | — | 226 | 398 | — | — | 0.0 |
| 256 | 64 | 38.4 | — | 433 | 653 | 308 | 479 | 0.0 |
| 256 | 128 | 72.7 | — | 1230 | 1536 | — | — | 0.0 |
| 256 | 256 | 150 | — | 2465 | 2950 | — | — | 0.0 |
| 64 | 1 | 9.6 | — | 13.1 | 26.0 | — | — | 0.0 |
| 64 | 8 | 11.9 | — | 35.1 | 159 | — | — | 0.0 |
| 64 | 16 | 14.2 | — | 52.6 | 207 | — | — | 0.0 |
| 64 | 32 | 21.4 | — | 91.6 | 278 | — | — | 0.0 |
| 64 | 64 | 37.7 | — | 164 | 401 | — | — | 0.0 |
| 64 | 128 | 71.2 | — | 313 | 640 | — | — | 0.0 |
| 64 | 256 | 145 | — | 622 | 1113 | — | — | 0.0 |

## (B) fresh frames: N new frames + 1 question, end to end (Glance: box readout, unpadded prefill)

Cells: Glance / vLLM bf16 yes-no (1 token) / vLLM bf16 box. Bold = Glance < 100 ms.

| Tv \ N | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 64 | **16.5** / 19.9 / 21.9 | **19.4** / 33.5 / 89.3 | **25.9** / 36.4 / 91.6 | **40.3** / 41.3 / 166 | **66.6** / 65.3 / 221 | 138 / 111 / 289 |
| 128 | **18.2** / 20.9 / 22.6 | **23.5** / 34.8 / 91.1 | **35.0** / 42.5 / 42.3 | **56.9** / 50.6 / 113 | 109 / 81.8 / 237 | 226 / 143 / 318 |
| 256 | **25.0** / 25.0 / 27.4 | **35.9** / 43.9 / 43.0 | **57.7** / 59.2 / 56.0 | 109 / 71.7 / 97.7 | 225 / 134 / 280 | 482 / 236 / 400 |

(Tv=640 fresh, padded run: N=1 43 ms, N=2 76 ms, N=4 145 ms.)

## Clip encode cost and bank size (Glance)

- f4_v96: 197 prefix tokens, encode 23.1 ms/clip (eager, batched), bank 57.1 MB/clip; N=64 query 63.6 ms, N=256 289 ms
- f4_v48: 117 prefix tokens, encode 11.5 ms/clip (eager, batched), bank 53.1 MB/clip; N=64 query 63.0 ms, N=256 281 ms
- f2_v96: 101 prefix tokens, encode 9.5 ms/clip (eager, batched), bank 53.1 MB/clip; N=64 query 63.0 ms, N=256 281 ms

## Exactness on GPU (bf16, real weights, real images) vs the shipped reference path, each image alone

| set / query / path | max ΔP | mean ΔP | decision agreement |
|---|---|---|---|
| img640/q0/cached | 0.0234 | 0.0024 | 1.000 |
| img640/q0/fresh | 0.0223 | 0.0021 | 1.000 |
| img640/q0/separate_engine | 0.0271 | 0.0037 | 1.000 |
| img640/q1/cached | 0.0030 | 0.0001 | 1.000 |
| img640/q1/fresh | 0.0027 | 0.0001 | 1.000 |
| img640/q1/separate_engine | 0.0016 | 0.0001 | 1.000 |
| img640/q2/cached | 0.0053 | — | 1.000 |
| img640/q2/fresh | 0.0061 | — | 1.000 |
| img640/q2/separate_engine | 0.0035 | — | 1.000 |
| frame256/q0/cached | 0.0116 | 0.0021 | 1.000 |
| frame256/q0/fresh | 0.0156 | 0.0025 | 1.000 |
| frame256/q0/separate_engine | 0.0195 | 0.0029 | 1.000 |
| frame256/q1/cached | 0.0744 | 0.0031 | 1.000 |
| frame256/q1/fresh | 0.0790 | 0.0033 | 1.000 |
| frame256/q1/separate_engine | 0.0083 | 0.0004 | 1.000 |
| frame256/q2/cached | 0.0038 | — | 1.000 |
| frame256/q2/fresh | 0.0075 | — | 1.000 |
| frame256/q2/separate_engine | 0.0080 | — | 1.000 |
| clip4x96/noul/cached | 0.0215 | — | 1.000 |
| clip4x96/noul/fresh | 0.0217 | — | 1.000 |
