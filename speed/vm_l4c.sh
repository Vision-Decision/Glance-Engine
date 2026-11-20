#!/usr/bin/env bash
# VM side, L4 (22 GB): bf16 engine pass with at most 8 images per call; saves after every regime.
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TO=7200 bash speed/run_dev.sh glance5graph_l4 --n_calls 20 --max_batch 8 --gui_nv 2048 --modes eng_graph \
  --parts lat,mixed,followup,tput,tput2,video,gui
echo "[l4c] ALL DONE" >> /content/out/spd/main_l4.log
