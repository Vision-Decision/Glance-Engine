#!/usr/bin/env bash
# E5 (memory tiers), E7 (ablations: clip config, FP8 accuracy), E8 (Visual Haystacks N=100/1000) on VM A after E2.
cd /content/dv
O=/content/out/scan
step() { echo "[e578] $1 start $(date +%H:%M:%S)"; }
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/glance-train-ckpts', 'scan/short_gv0/ckpt_final.pt', local_dir='/content/ckpt'))"
C=/content/ckpt/scan/short_gv0/ckpt_final.pt
step e5
timeout 5400 python -m scan.e5_memory --ckpt $C --data /content/charades_pilot2 --out $O/e5_memory.json --n_long 1800 > $O/e5_memory.log 2>&1
python scan/vm/push.py e5 $O/e5_memory.json $O/e5_memory.log
for cfg in "v48 4 2 48" "k2 2 1 96" "fp8 4 2 96"; do
  set -- $cfg
  step "e7_$1"
  X=""; [ "$1" = fp8 ] && X="--fp8"
  timeout 7200 python -m scan.pilot_charades --ckpt $C --data /content/charades_pilot2 --out $O/e7_pilot_$1.json.gz \
    --unit clip --K $2 --S $3 --clip_vis $4 --max_videos 700 --scorers g_short $X > $O/e7_pilot_$1.log 2>&1
  python scan/vm/push.py e7 $O/e7_pilot_$1.json.gz $O/e7_pilot_$1.log
done
step e8
timeout 10800 python -m scan.pilot_vhs --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e8_vhs.json --base --Ns 100,1000 --max_q 100 > $O/e8_vhs.log 2>&1
python scan/vm/push.py e8 $O/e8_vhs.json $O/e8_vhs.log
echo "[e578] ALL DONE $(date +%H:%M:%S)"
