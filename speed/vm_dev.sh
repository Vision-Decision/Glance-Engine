#!/usr/bin/env bash
# VM side, dev pass on one GPU: hardware roofline numbers, profile, quick engine smoke.  bash speed/vm_dev.sh GPU_TAG
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=${1:?gpu tag}
O=/content/out/spd; mkdir -p $O
timeout 600 python -m speed.hwbench --out $O/hw_$G.json > $O/hw_$G.log 2>&1
timeout 1800 python -m speed.profile_glance --ckpt /content/ckpt/v1/ckpt_final.pt --modes ref_lora,ref_merged,eng_eager,eng_graph \
  --out $O/prof_$G.json > $O/prof_$G.log 2>&1
TO=2400 bash speed/run_dev.sh dev_$G --quick --modes ref_lora,eng_eager,eng_graph,eng_fp8 --parts equiv,lat,followup
echo VM_DEV_DONE >> $O/prof_$G.log
