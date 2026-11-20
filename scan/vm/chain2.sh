#!/usr/bin/env bash
# Branch-kernel A/B (chunk vs fused recurrent, no final state) on cached clips + cached frames, then the Charades pilots.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[chain2] $1 start $(date +%H:%M:%S)"; }
for K in recurrent chunk; do
  step kernel_$K
  SCAN_BRANCH_GDR=$K timeout 1800 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/kernel_$K.json \
    --parts clips,cached --clip_cfgs 4x96 --clip_fresh 0 --cached_tvs 64 --Ns 64,128,256 --n_calls 20 > $O/kernel_$K.log 2>&1
done
step pilot_frames
timeout 7200 python -m scan.pilot_charades --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/charades_pilot \
  --out $O/pilot_frames_v1.json.gz --tv 256 --max_videos 700 > $O/pilot_frames.log 2>&1
echo "[chain2] ALL DONE $(date +%H:%M:%S)"
