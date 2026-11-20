#!/usr/bin/env bash
# After chain7: Visual Haystacks N=50/100, then (when it lands on the org repo) the Glance-Video FINAL checkpoint on the
# Charades clips. Exits without the final if it has not appeared by DEADLINE_UTC (HH:MM, default 20:30).
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[chain8] $1 start $(date +%H:%M:%S)"; }
while pgrep -f "chain7.s[h]" >/dev/null; do sleep 20; done
step vhs_rest
timeout 3600 python -m scan.pilot_vhs --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/vhs_v1_b.json --base --Ns 50,100 > $O/vhs_b.log 2>&1
step wait_gv_final
DL=${DEADLINE_UTC:-20:30}
until python -c "
from huggingface_hub import HfApi; import sys
sys.exit(0 if 'glance-video-v0/ckpt_final.pt' in HfApi().list_repo_files('Vision-Decision/glance-train-ckpts') else 1)"; do
  [ "$(date -u +%H:%M)" \> "$DL" ] && { echo "[chain8] no gv final by $DL"; exit 0; }
  sleep 120
done
GV=glance-video-v0/ckpt_final.pt TAG=gvfinal bash scan/vm/chain6.sh
echo "[chain8] ALL DONE $(date +%H:%M:%S)"
