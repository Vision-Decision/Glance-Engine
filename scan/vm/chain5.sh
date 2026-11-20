#!/usr/bin/env bash
# After chain4: v1 (long block) on the same Charades-STA test clips, the reference for the short-readout parity.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "chain4.s[h]" >/dev/null; do sleep 20; done
echo "[chain5] pilot_clips_v1 start $(date +%H:%M:%S)"
timeout 5400 python -m scan.pilot_charades --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/charades_pilot2 \
  --out $O/pilot_clips_v1.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes > $O/pilot_clips_v1.log 2>&1
echo "[chain5] ALL DONE $(date +%H:%M:%S)"
