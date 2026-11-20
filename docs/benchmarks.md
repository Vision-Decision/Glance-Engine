# Reproducing the benchmarks

Every number in [`outputs/engine_opt/CEILING.md`](../outputs/engine_opt/CEILING.md) comes from one command below. The exact command line of each run is recorded at the top of the `.log` file next to its raw JSON, together with the GPU name, driver and the hashes of the engine sources.

## Hardware and software

| | Used for every CEILING row |
|---|---|
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), driver 580.82.07, Colab "G4" VM |
| Glance stack | torch 2.14.1+cu130, transformers 5.23.0, flash-linear-attention 0.6.0; flash-qla 0.1.3 + tilelang 0.1.15 with `FLA_FLASH_QLA=0` for the item 7 and 9 runs ([`constraints/g4-benchmarks.txt`](../constraints/g4-benchmarks.txt)) |
| vLLM | 0.34.0 in its own Python 3.12 venv |
| SGLang | 0.5.23 in its own Python 3.12 venv |
| Model | Glance v1 training checkpoint `Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt` (the weights released as `Vision-Decision/Glance-4B-preview`); baselines run `Qwen/Qwen3.5-4B` |
| Data | First 200 RefCOCO-val COCO images, 40 ScreenSpot-Pro screenshots, fetched by the setup script |

Only one process uses the GPU during a measurement; the runners in `scan/vm/` wait for an empty compute-process list before every stage.

## Setup (on the GPU VM)

The benchmark scripts expect this repository and a [glance-core](https://github.com/Vision-Decision/glance-core) checkout merged at `/content/dv` (glance-core provides `data/prep/eval_refcoco_val.jsonl`), as `scan/vm/sync.sh` arranges.

```bash
cd /content/dv
bash speed/setup_vm.sh --serve      # transformers 5.23 + FLA 0.6, Qwen3.5-4B, the v1 checkpoint, COCO + ScreenSpot-Pro,
                                    # and separate venvs /content/vllm_env (vLLM) and /content/sglang_env (SGLang)
pip install -e '.[gpu,serve,sm120-prefill]'
export CKPT=/content/ckpt/v1/ckpt_final.pt FLA_FLASH_QLA=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Baseline configurations

| Engine | Settings (recorded as `engine_args` in each raw JSON) |
|---|---|
| vLLM BF16 | prefix caching on, CUDA graphs (default), async scheduling, `max_num_seqs=256`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.85`, seed 0 |
| vLLM FP8+MTP | the above plus `quantization=fp8` and `speculative_config={method: mtp, num_speculative_tokens: 2}` |
| SGLang BF16 | radix prefix cache, CUDA graphs (default), `max_running_requests=256`, `chunked_prefill_size=16384`, `mem_fraction_static` 0.8–0.85, seed 0 |

Boxes are generated with greedy decoding and up to 24 new tokens; yes/no scans generate one token with logprobs. The harnesses reset prefix and encoder caches where a row is "fresh" and await SGLang's cache flush (an earlier harness that swallowed the flush error is documented in CEILING.md).

## One command per family

Glance commands run in the VM's main Python; baseline commands run with `/content/vllm_env/bin/python` or `/content/sglang_env/bin/python` (shown as `VLLM_PY` and `SGLANG_PY`).

### One image (≤640 visual tokens) and follow-ups on a cached image

```bash
python -m speed.bench_glance --modes eng_fused --parts lat,followup --ckpt $CKPT --out speed.json    # box Q=1/8/32 + follow-ups
python -m speed.bench_glance --modes eng_fused --parts mixed --ckpt $CKPT --out mixed.json           # mixed Q=8/32
$VLLM_PY -m speed.bench_serve --backend vllm --async_sched --parts lat,mixed,followup,tput,video,gui --out vllm_bf16.json
$VLLM_PY -m speed.bench_serve --backend vllm --async_sched --spec mtp --spec_k 2 --quant fp8 \
         --parts lat,mixed,followup,tput,video,gui --out vllm_fp8mtp.json
$SGLANG_PY -m speed.bench_serve --backend sglang --parts clients,ceiling,followup,gui --out sglang.json
```

Results: [Glance](../outputs/engine_opt/09_numerics/raw/item9_release_speed.json), [mixed](../outputs/engine_opt/09_numerics/raw/item9_release_mixed.json) · [vLLM BF16](../outputs/sprint_speed/raw/serve_vllm_async_g4.json) · [vLLM FP8+MTP](../outputs/sprint_speed/raw/serve_vllm_fp8mtp_g4.json) · [SGLang](../outputs/engine_opt/00_ceiling/raw/c0_sglang_clean.json)

### GUI screenshots (ScreenSpot-Pro, ≤2,048 / ≤4,096 visual tokens)

```bash
python -m speed.bench_glance --modes eng_fused --parts gui --gui_nv 2048 --ckpt $CKPT --out gui2048.json
python -m speed.bench_glance --modes eng_fused --parts gui --gui_nv 4096 --ckpt $CKPT --out gui4096.json
```

Baselines: the `gui` part of the three serving runs above. These two Glance runs used the standard FLA stack (no `[sm120-prefill]`). Results: [2k](../outputs/engine_opt/03_lm_prefill/raw/item3_gui2048.json) · [4k](../outputs/engine_opt/03_lm_prefill/raw/item3_gui4096.json)

### Video (32 / 64 frames, one box question per frame)

```bash
python -m speed.bench_glance --modes eng_fused --parts video1 --ckpt $CKPT --out video.json
$VLLM_PY -m speed.bench_serve --backend vllm --parts video1 --async_sched --out video_vllm.json
$VLLM_PY -m speed.bench_serve --backend vllm --parts video1 --async_sched --spec mtp --quant fp8 --out video_fp8mtp.json
$SGLANG_PY -m speed.bench_serve --backend sglang --parts video1 --out video_sglang.json
```

Results: [Glance](../outputs/engine_opt/07_frames/raw/item7_checked_video.json) · [vLLM](../outputs/engine_opt/07_frames/raw/item7_matched_video_vllm.json) · [FP8+MTP](../outputs/engine_opt/07_frames/raw/item7_matched_video_tuned.json) · [SGLang](../outputs/engine_opt/07_frames/raw/item7_final_video_sglang.json)

### Cached scan (one new question over N retained prefixes)

```bash
python -m scan.bench_scan --parts clips,cached --Ns 64,256 --n_calls 15 --clip_fresh 0 --ckpt $CKPT --out cached.json
python -m scan.bench_scan --parts cached --cached_tvs 640 --Ns 256 --n_calls 15 --ckpt $CKPT --out cached_tv640.json
$VLLM_PY -m scan.bench_vllm --parts clips,cached,fresh --kinds yesno,box --Ns 1,8,16,32,64,128,256 --Tvs 64,128,256 \
         --n_calls 12 --out scan_vllm_bf16.json
$VLLM_PY -m scan.bench_vllm --parts clips,cached --kinds yesno,box --Ns 8,64,256 --Tvs 64,128,256,640 \
         --n_calls 12 --quant fp8 --spec mtp --out scan_vllm_fp8mtp.json
$VLLM_PY -m scan.bench_vllm --parts cached_clips --clip_cfgs 4x48,2x96 --Ns 64,256 --n_calls 8 --out clips_vllm.json
$SGLANG_PY -m scan.bench_sglang --parts clips,cached,cached_box --Ns 64,256 --n_calls 8 --out scan_sglang.json
$SGLANG_PY -m scan.bench_sglang --parts cached_clips --clip_cfgs 4x48,2x96 --Ns 64,256 --n_calls 8 --out clips_sglang.json
```

Results: [Glance](../outputs/engine_opt/08_bank_memory/raw/item8_release_after.json), [Tv640 N=256](../outputs/engine_opt/09_numerics/raw/item9_release_cached.json) · [vLLM BF16](../outputs/scan/raw/scan-a/vllm_bf16.json) · [FP8+MTP](../outputs/scan/raw/scan-a/vllm_fp8mtp.json) · [vLLM clips](../outputs/engine_opt/08_bank_memory/raw/item8_test_vllm.json) · [SGLang](../outputs/engine_opt/00_ceiling/raw/c0_sglang_bank.json) · [SGLang clips](../outputs/engine_opt/08_bank_memory/raw/item8_test_sglang.json)

### Fresh scan (one yes/no question over N new frames)

```bash
python -m scan.bench_scan --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno \
       --fresh_chunked 1 --max_fresh_tokens 16384 --ckpt $CKPT --out fresh.json
$VLLM_PY -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --kinds yesno --n_calls 8 \
         --max_fresh_tokens 70000 --out fresh_vllm.json
$SGLANG_PY -m scan.bench_sglang --parts fresh --Tvs 128,256 --Ns 32,64,128,256 --n_calls 8 --out fresh_sglang.json
```

Results: [Glance](../outputs/engine_opt/07_frames/raw/item7_final_fresh.json) (Tv128/N128 from [item 9](../outputs/engine_opt/09_numerics/raw/item9_release_fresh.json)) · [vLLM](../outputs/engine_opt/07_frames/raw/item7_final_fresh_vllm.json) · [SGLang](../outputs/engine_opt/07_frames/raw/item7_final_fresh_sglang.json)

### Concurrent clients (C = 1, 4, 16, 64)

```bash
python -m speed.bench_scheduler --backend scheduled --warm-shapes --out clients.json
$VLLM_PY -m speed.bench_scheduler --backend vllm --model <local Qwen3.5-4B snapshot> --out clients_vllm.json
$SGLANG_PY -m speed.bench_scheduler --backend sglang --model <local Qwen3.5-4B snapshot> --out clients_sglang.json
python -m speed.bench_scheduler --backend serial --warm-shapes --out clients_serial.json     # the item 6 control
```

Results: [Glance](../outputs/engine_opt/07_frames/raw/item7_final_scheduled.json) · [vLLM](../outputs/engine_opt/07_frames/raw/item7_final_scheduler_vllm.json) · [SGLang](../outputs/engine_opt/07_frames/raw/item7_final_scheduler_sglang.json). An HTTP load test against a running server: `python -m speed.load_server --url http://127.0.0.1:8002 --clients 16 --out load.json`.

### Hardware floors

```bash
bash scan/vm/ceiling.sh peaks                                   # GEMM, copy, H2D and launch rates on this GPU
python -m speed.ceiling_model outputs/engine_opt/00_ceiling/raw # stage floors from the measured rates and call shapes
python -m speed.scheduler_floors --out floors.json              # ideal-batching floors for the client bursts
```

## Stage runners

Each optimization item has a runner that executes its stages serially under a GPU lock, logs the command, GPU and source hashes, and uploads results. They are the most faithful way to rerun a whole item:

| Item | Runner | Example |
|---|---|---|
| 3 (GUI, LM prefill) | [`scan/vm/lm_prefill.sh`](../scan/vm/lm_prefill.sh) | `bash scan/vm/lm_prefill.sh gui2048 gui4096` |
| 6 (scheduler) | [`scan/vm/scheduler_bench.sh`](../scan/vm/scheduler_bench.sh) | see [item 6](../outputs/engine_opt/06_scheduler/summary.md#reproduction) |
| 7 (frames, fresh, bursts) | [`scan/vm/frame_throughput.sh`](../scan/vm/frame_throughput.sh) | `ITEM7_LABEL=rerun bash scan/vm/frame_throughput.sh video fresh scheduled` |
| 8 (banks) | [`scan/vm/bank_memory.sh`](../scan/vm/bank_memory.sh) | see [item 8](../outputs/engine_opt/08_bank_memory/summary.md#equivalence-and-reproduction) |
| 9 (numerics) | [`scan/vm/numerics.sh`](../scan/vm/numerics.sh) | `ITEM9_LABEL=rerun bash scan/vm/numerics.sh speed mixed cached fresh real fused diag` |

The runners assume the `/content/dv` layout and the paths created by `speed/setup_vm.sh`.
