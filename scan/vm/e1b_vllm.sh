#!/usr/bin/env bash
# vLLM fresh rerun on an idle GPU (the first run may have overlapped another job).
cd /content/dv
O=/content/out/scan
while pgrep -f "e1b_redo[.]sh" >/dev/null; do sleep 10; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 5; done
echo "[e1bvllm] start $(date +%H:%M:%S)"
timeout 3600 /content/vllm_env/bin/python -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --kinds yesno \
  --n_calls 8 --max_fresh_tokens 70000 --out $O/e1b_vllm_fresh2.json > $O/e1b_vllm2.log 2>&1
python scan/vm/push.py e1b $O/e1b_vllm_fresh2.json $O/e1b_vllm2.log
echo "[e1bvllm] DONE $(date +%H:%M:%S)"
