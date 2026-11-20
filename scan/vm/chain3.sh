#!/usr/bin/env bash
# After the Charades data is on the VM: teacher labels -> short-readout self-distillation from v1 -> clip pilot
# (v1 long block vs distilled short block, same Charades-STA test clips) -> latency of the short block.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[chain3] $1 start $(date +%H:%M:%S)"; }
pip install -q av
until grep -q "DONE clips" $O/distill_prep.log 2>/dev/null; do sleep 30; done      # train clips ready
while pgrep -f "scan.pilot_vh[s]" >/dev/null; do sleep 20; done
step teacher
timeout 3600 python -m scan.distill_teacher --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/distill \
  --out /content/distill/train.jsonl > $O/teacher.log 2>&1
step train
timeout 5400 python train.py --cfg configs/scan_short_v1.json > $O/train_short.log 2>&1
step pilot_clips
until grep -q CHARADES_VM_DONE $O/charades_vm.log 2>/dev/null; do sleep 30; done  # 2-fps test frames ready
timeout 7200 python -m scan.pilot_charades --ckpt /content/out/distill_v1/ckpt_final.pt --data /content/charades_pilot2 \
  --out $O/pilot_clips_short.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes,g_short,base_yes,siglip > $O/pilot_clips_short.log 2>&1
step pilot_clips_v1
timeout 5400 python -m scan.pilot_charades --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/charades_pilot2 \
  --out $O/pilot_clips_v1.json.gz --unit clip --K 4 --S 2 --clip_vis 96 --max_videos 700 \
  --scorers g_yes > $O/pilot_clips_v1.log 2>&1
step latency_short
timeout 1800 python -m scan.bench_scan --ckpt /content/out/distill_v1/ckpt_final.pt --out $O/latency_short.json \
  --parts clips --clip_cfgs 4x96 --clip_fresh 0 --Ns 64,128,256 --n_calls 20 --short 1 > $O/latency_short.log 2>&1
step latency_short_fp8
timeout 1800 python -m scan.bench_scan --ckpt /content/out/distill_v1/ckpt_final.pt --out $O/latency_short_fp8.json \
  --parts clips --clip_cfgs 4x96 --clip_fresh 0 --Ns 64,128,256 --n_calls 20 --short 1 --fp8 1 > $O/latency_short_fp8.log 2>&1
echo "[chain3] ALL DONE $(date +%H:%M:%S)"
