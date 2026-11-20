#!/usr/bin/env bash
# VM side, main measurement pass on one GPU: Glance (all modes, all regimes), then every generative baseline.
#   bash speed/vm_main.sh GPU_TAG [glance|serve|all]
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=${1:?gpu tag}; W=${2:-all}
O=/content/out/spd; mkdir -p $O
if [ "$W" = "glance" ] || [ "$W" = "all" ]; then
  echo "[main] glance start $(date +%H:%M:%S)" >> $O/main_$G.log
  TO=7200 bash speed/run_dev.sh glance_$G --fp8_vision --n_equiv ${NEQ:-100} --n_calls ${NCALLS:-50} \
    --modes ${MODES:-ref_lora,ref_merged,eng_eager,eng_graph,eng_fp8}
  echo "[main] glance end $(date +%H:%M:%S) $(tail -1 $O/glance_$G.log)" >> $O/main_$G.log
  [ -z "${SKIP_EQUIV:-}" ] && timeout 2400 python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --n_img 60 --out $O/equiv_$G.json > $O/equiv_$G.log 2>&1
  echo "[main] equiv end $(date +%H:%M:%S)" >> $O/main_$G.log
fi
if [ "$W" = "serve" ] || [ "$W" = "all" ]; then
  bash speed/vm_serve.sh $G
  echo "[main] serve end $(date +%H:%M:%S)" >> $O/main_$G.log
fi
echo "[main] ALL DONE" >> $O/main_$G.log
