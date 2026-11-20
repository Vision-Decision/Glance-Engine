#!/usr/bin/env bash
# VM chain after the main bench: exactness + unpadded fresh grid, vLLM baselines, Charades frame pilot (v1).
#   bash scan/vm/chain1.sh   (run detached via scan/vm/bg.sh; each step logs to /content/out/scan/<step>.log)
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[chain] $1 start $(date +%H:%M:%S)"; }
step eqfresh
timeout 3600 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/bench_eqfresh.json \
  --parts equiv,fresh --Tvs 64,128,256 --Ns_fresh 1,2,4,8,16,32 --n_calls 20 > $O/eqfresh.log 2>&1
step vllm_bf16
timeout 5400 /content/vllm_env/bin/python -m scan.bench_vllm --out $O/vllm_bf16.json --parts clips,cached,fresh \
  --Tvs 64,128,256 --Ns_fresh 1,2,4,8,16,32 --n_calls 12 > $O/vllm_bf16.log 2>&1
step vllm_fp8mtp
timeout 3600 /content/vllm_env/bin/python -m scan.bench_vllm --out $O/vllm_fp8mtp.json --parts clips,cached \
  --quant fp8 --spec mtp --Ns 8,64,256 --n_calls 12 > $O/vllm_fp8mtp.log 2>&1
step pilot_frames
timeout 7200 python -m scan.pilot_charades --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/charades_pilot \
  --out $O/pilot_frames_v1.json.gz --tv 256 --max_videos 700 > $O/pilot_frames.log 2>&1
echo "[chain] ALL DONE $(date +%H:%M:%S)"
