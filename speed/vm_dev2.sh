#!/usr/bin/env bash
# VM side: quick check of the packed engine + FP8, then the fp32-referenced equivalence study.  bash speed/vm_dev2.sh GPU_TAG
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=${1:?gpu tag}
O=/content/out/spd; mkdir -p $O
TO=2400 bash speed/run_dev.sh dev2_$G --quick --fp8_vision --modes ref_lora,eng_graph,eng_fp8 --parts equiv,lat,mixed,followup
timeout 2400 python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --n_img 60 --out $O/equiv_$G.json > $O/equiv_$G.log 2>&1
echo VM_DEV2_DONE >> $O/equiv_$G.log
