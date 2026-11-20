#!/usr/bin/env bash
# E1b: no-cache comparison. One question over N NEW frames (Tv 128 / 256), end to end including the vision tower.
# (a) ours, fresh path, yes/no readout (chunked above 16k prompt tokens); (b) SGLang 0.5.23, N requests, 1 new token;
# (c) vLLM 0.34 bf16, N requests, 1 new token. Images/s = N / median latency.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # sglang needs its venv on PATH (ninja): set only for it
step() { echo "[e1b] $1 start $(date +%H:%M:%S)"; }
step ours
timeout 3600 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e1b_ours_fresh.json --parts fresh \
  --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 > $O/e1b_ours.log 2>&1
step sglang
PATH=/content/sglang_env/bin:$PATH timeout 3600 /content/sglang_env/bin/python -m scan.bench_sglang --parts fresh --Tvs 128,256 --Ns 32,64,128,256 --n_calls 8 \
  --out $O/e1b_sglang_fresh.json > $O/e1b_sglang.log 2>&1
step vllm
timeout 3600 /content/vllm_env/bin/python -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --kinds yesno \
  --n_calls 8 --max_fresh_tokens 70000 --out $O/e1b_vllm_fresh.json > $O/e1b_vllm.log 2>&1
python scan/vm/push.py e1b $O/e1b_ours_fresh.json $O/e1b_sglang_fresh.json $O/e1b_vllm_fresh.json $O/e1b_*.log
echo "[e1b] ALL DONE $(date +%H:%M:%S)"
