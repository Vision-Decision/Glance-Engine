#!/usr/bin/env bash
# E10 algorithmic items on VM C after E4: fewer visual tokens (latency sweep + quality vs the Tv256 reference) and
# the early-exit probe. Engine untouched (engine work belongs to engine-opt).
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^bash scan/vm/e4_c.sh" >/dev/null; do sleep 30; done
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/collection-scan-results', 'e1b/e1b_ours_fresh_c4096.json', repo_type='dataset', local_dir='/content/dl'))" && cp /content/dl/e1b/e1b_ours_fresh_c4096.json $O/ 2>/dev/null
echo "[e10c] tokens start $(date +%H:%M:%S)"
timeout 7200 python -m scan.e10_fresh --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e10_tokens.json --parts sweep,quality --no_fp8 \
  --sweep_chunks 4096 --n_img 200 > $O/e10_tokens.log 2>&1
python scan/vm/push.py e10 $O/e10_tokens.json $O/e10_tokens.log
echo "[e10c] probe start $(date +%H:%M:%S)"
timeout 7200 python -m scan.e10_probe --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e10_probe.json --n_img 400 --tv 128 > $O/e10_probe.log 2>&1
python scan/vm/push.py e10 $O/e10_probe.json $O/e10_probe.log
echo "[e10c] DONE $(date +%H:%M:%S)"
