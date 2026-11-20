#!/usr/bin/env bash
# VM side, L4: engine-mode Glance passes, one fresh process per mode (after the serving baselines).  bash speed/vm_l4b.sh
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for M in eng_graph eng_fused eng_fp8; do
  TO=5400 bash speed/run_dev.sh glance5${M#eng_}_l4 --n_calls 20 --modes $M --parts lat,mixed,followup,tput,video,gui
done
echo "[l4b] ALL DONE" >> /content/out/spd/main_l4.log
