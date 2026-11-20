#!/usr/bin/env bash
# Short-readout self-distillation from v1 (teacher labels already in /content/distill/train.jsonl), then: clip pilot
# with the distilled model (long vs short block, same clips as the v1 clip pilot), image-presence parity vs v1,
# latency of the short block (bf16, FP8), and the checkpoint pushed to the org repo.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[chain4] $1 start $(date +%H:%M:%S)"; }
while pgrep -f "chain3.s[h]" >/dev/null; do sleep 20; done
step train
timeout 5400 python train.py --cfg configs/scan_short_v1.json > $O/train_short.log 2>&1
C=/content/out/distill_v1/ckpt_final.pt
[ -f $C ] || { echo "[chain4] no checkpoint"; exit 1; }
step push
python - <<'PY' > $O/push.log 2>&1
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/content/out/distill_v1/ckpt_final.pt", path_in_repo="scan/short_v1/ckpt_final.pt",
                    repo_id="Vision-Decision/glance-train-ckpts", repo_type="model")
print("PUSHED")
PY
step pilot_clips_short
timeout 7200 python -m scan.pilot_charades --ckpt $C --data /content/charades_pilot2 \
  --out $O/pilot_clips_short.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes,g_short,base_yes,siglip > $O/pilot_clips_short.log 2>&1
step parity_images
timeout 3600 python -m scan.bench_scan --ckpt $C --out $O/parity_short.json --parts acc --Tvs 256 --n_acc 60 > $O/parity_short.log 2>&1
step latency_short
timeout 1800 python -m scan.bench_scan --ckpt $C --out $O/latency_short.json \
  --parts clips --clip_cfgs 4x96 --clip_fresh 0 --Ns 64,128,256 --n_calls 20 --short 1 > $O/latency_short.log 2>&1
step latency_short_fp8
timeout 1800 python -m scan.bench_scan --ckpt $C --out $O/latency_short_fp8.json \
  --parts clips --clip_cfgs 4x96 --clip_fresh 0 --Ns 64,128,256 --n_calls 20 --short 1 --fp8 1 > $O/latency_short_fp8.log 2>&1
echo "[chain4] ALL DONE $(date +%H:%M:%S)"
