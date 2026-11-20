#!/usr/bin/env bash
# After chain5: Glance-Video (clip-level model) on the same Charades-STA test clips. GV=<path in the org repo>.
cd /content/dv
O=/content/out/scan
GV=${GV:-glance-video-v0/ckpt_last.pt}
TAG=${TAG:-gvlast}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "chain5.s[h]" >/dev/null; do sleep 20; done
if [ ! -d /content/charades_pilot2 ]; then      # fresh VM: 2-fps Charades-STA test frames from the org dataset
  python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/scan-pilot-data', 'charades_sta_test_2fps.tar', repo_type='dataset', local_dir='/content/dl'))"
  tar xf /content/dl/charades_sta_test_2fps.tar -C /content
fi
echo "[chain6] download $GV $(date +%H:%M:%S)"
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/glance-train-ckpts', '$GV', local_dir='/content/ckpt'))"
echo "[chain6] pilot_clips_$TAG start $(date +%H:%M:%S)"
timeout 5400 python -m scan.pilot_charades --ckpt /content/ckpt/$GV --data /content/charades_pilot2 \
  --out $O/pilot_clips_$TAG.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes > $O/pilot_clips_$TAG.log 2>&1
echo "[chain6] ALL DONE $(date +%H:%M:%S)"
