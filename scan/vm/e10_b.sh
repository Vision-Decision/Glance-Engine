#!/usr/bin/env bash
# VM B after E1b (SGLang/vLLM) and E9: E1b ours (fresh, yes/no, chunked), then E10 step 1 only (peak + profile);
# the engineering half of E10 is postponed (coordinator, 2026-11-19); the algorithmic half waits for E2-E6.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[e10b] $1 start $(date +%H:%M:%S)"; }
while pgrep -f "e1b.s[h]|e9_b.s[h]" >/dev/null; do sleep 20; done
step e1b_ours
timeout 3600 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e1b_ours_fresh.json --parts fresh \
  --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 > $O/e1b_ours.log 2>&1
python scan/vm/push.py e1b $O/e1b_ours_fresh.json $O/e1b_ours.log
step e10
timeout 10800 python -m scan.e10_fresh --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e10_fresh.json --parts peak,profile --no_fp8 > $O/e10_fresh.log 2>&1
python scan/vm/push.py e10 $O/e10_fresh.json $O/e10_fresh.log
echo "[e10b] ALL DONE $(date +%H:%M:%S)"
