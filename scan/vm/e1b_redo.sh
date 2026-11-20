#!/usr/bin/env bash
# E1b clean rerun (GPU must be idle): ours fresh yes/no (chunked) and SGLang fresh with unique images per call.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 5; done
echo "[e1bredo] ours start $(date +%H:%M:%S)"
timeout 3600 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e1b_ours_fresh.json --parts fresh \
  --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 > $O/e1b_ours.log 2>&1
echo "[e1bredo] sglang start $(date +%H:%M:%S)"
PATH=/content/sglang_env/bin:$PATH timeout 3600 /content/sglang_env/bin/python -m scan.bench_sglang --parts fresh --Tvs 128,256 \
  --Ns 32,64,128,256 --n_calls 8 --out $O/e1b_sglang_fresh2.json > $O/e1b_sglang2.log 2>&1
python scan/vm/push.py e1b $O/e1b_ours_fresh.json $O/e1b_ours.log $O/e1b_sglang_fresh2.json $O/e1b_sglang2.log
echo "[e1bredo] DONE $(date +%H:%M:%S)"
