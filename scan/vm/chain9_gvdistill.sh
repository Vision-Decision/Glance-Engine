#!/usr/bin/env bash
# After chain6 (gvfinal clip pilot): short-readout self-distillation on the FINAL Glance-Video checkpoint
# (teacher = glance-video-v0/ckpt_final.pt on its own long yes/no block, Charades TRAIN clips), then the clip pilot of
# the distilled model with both readouts (g_yes long, g_short) on the same Charades-STA test clips, a scan-vs-core_gv
# parity test, and pushes to Vision-Decision/glance-train-ckpts (scan/short_gv0/, results/gv0/scan/).
cd /content/dv
O=/content/out/scan; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/content/dv
step() { echo "[chain9] $1 start $(date +%H:%M:%S)"; }
GV=/content/ckpt/glance-video-v0/ckpt_final.pt
while pgrep -f "chain6.s[h]" >/dev/null; do sleep 20; done
[ -f $GV ] || python -c "from huggingface_hub import hf_hub_download as h; h('Vision-Decision/glance-train-ckpts', 'glance-video-v0/ckpt_final.pt', local_dir='/content/ckpt')"
step data
if [ ! -f /content/distill/records.jsonl ]; then
  python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/scan-pilot-data', 'charades_train_clips_2fps4.tar', repo_type='dataset', local_dir='/content/dl'))"
  mkdir -p /content/distill_x && tar xf /content/dl/charades_train_clips_2fps4.tar -C /content/distill_x
  R=$(dirname $(find /content/distill_x -name records.jsonl | head -1))
  mkdir -p /content/distill && cp -r $R/. /content/distill/
fi
ls /content/distill | head; wc -l /content/distill/records.jsonl
step parity_scan_vs_core
timeout 2400 python scripts/video/test_scan_vs_core.py --ckpt $GV --n_videos 6 --out $O/gv_scan_vs_core.json > $O/gv_scan_vs_core.log 2>&1
step teacher
timeout 3600 python -m scan.distill_teacher --ckpt $GV --data /content/distill --out /content/distill/train_gv.jsonl > $O/teacher_gv.log 2>&1
step train
timeout 5400 python train.py --cfg configs/scan_short_v1.json --set train_file=/content/distill/train_gv.jsonl \
  init_ckpt=$GV out_dir=/content/out/distill_gv > $O/train_short_gv.log 2>&1
C=/content/out/distill_gv/ckpt_final.pt
[ -f $C ] || { echo "[chain9] no checkpoint"; exit 1; }
step push_ckpt
python - <<'PY' > $O/push_gv.log 2>&1
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/content/out/distill_gv/ckpt_final.pt", path_in_repo="scan/short_gv0/ckpt_final.pt",
                    repo_id="Vision-Decision/glance-train-ckpts", repo_type="model")
print("PUSHED")
PY
step pilot_clips_shortgv
timeout 7200 python -m scan.pilot_charades --ckpt $C --data /content/charades_pilot2 \
  --out $O/pilot_clips_shortgv.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes,g_short > $O/pilot_clips_shortgv.log 2>&1
step push_results
python - <<'PY' >> $O/push_gv.log 2>&1
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="/content/out/scan", path_in_repo="results/gv0/scan", repo_id="Vision-Decision/glance-train-ckpts",
                  repo_type="model", allow_patterns=["*gvfinal*", "*shortgv*", "*_gv*", "gv_*", "chain*.log"])
api.upload_file(path_or_fileobj=b"done\n", path_in_repo="results/gv0/scan/DONE", repo_id="Vision-Decision/glance-train-ckpts", repo_type="model")
print("RESULTS PUSHED")
PY
echo "[chain9] ALL DONE $(date +%H:%M:%S)"
