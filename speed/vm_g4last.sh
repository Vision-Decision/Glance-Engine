#!/usr/bin/env bash
# VM side, G4: the remaining engine modes, each in its own process (all regimes).
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for M in eng_fused eng_fp8; do
  TO=5400 bash speed/run_dev.sh glance5${M#eng_}_g4 --n_calls 50 --modes $M --parts lat,mixed,followup,tput,video,gui
done
echo "[g4last] ALL DONE" >> /content/out/spd/main_g4.log
