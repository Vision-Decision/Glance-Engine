#!/usr/bin/env bash
# VM side: engine-mode Glance pass (all regimes) after the serving queue, plus the light ablation modes (latency only).
#   bash speed/vm_glance2.sh GPU_TAG     -> glance2_<tag>.json (engine modes), glance3_<tag>.json (ref_merged, eng_eager)
# The fused-ops mode is smoke-tested first and dropped (with a note in main_<tag>.log) if it fails.
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=${1:?gpu tag}
L=/content/out/spd/main_$G.log
MODES=eng_graph,eng_fused,eng_fp8
TO=1200 bash speed/run_dev.sh smoke_fused_$G --quick --modes eng_fused,eng_fp8 --parts lat
if ! tail -1 /content/out/spd/smoke_fused_$G.log | grep -q "exit 0"; then
  MODES=eng_graph,eng_fp8; echo "[glance2] fused-ops smoke FAILED; running $MODES" >> $L
  export SPD_NO_FUSED=1
fi
timeout 2400 python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --n_img 60 --out /content/out/spd/equiv2_$G.json \
  > /content/out/spd/equiv2_$G.log 2>&1
TO=7200 bash speed/run_dev.sh glance2_$G --n_calls ${NCALLS:-50} --modes $MODES --parts lat,mixed,followup,tput,video,gui
TO=3600 bash speed/run_dev.sh glance3_$G --n_calls ${NCALLS:-50} --modes ref_merged,eng_eager --parts lat,mixed
echo "[glance2] ALL DONE" >> $L
