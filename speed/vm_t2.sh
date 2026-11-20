#!/usr/bin/env bash
# VM side: saturation with many questions per image (tput2) for Glance engine modes and the strongest baselines.
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=${1:?gpu tag}; NC=${NCALLS:-50}
TO=3600 bash speed/run_dev.sh glance4_$G --n_calls $NC --modes eng_fused,eng_fp8 --parts tput2
bash speed/run_serve.sh vllm vllm_mtp_t2_$G --n_calls $NC --parts tput2 --async_sched --spec mtp --spec_k 2
bash speed/run_serve.sh vllm vllm_fp8mtp_t2_$G --n_calls $NC --parts tput2 --async_sched --spec mtp --spec_k 2 --quant fp8
echo "[t2] ALL DONE" >> /content/out/spd/main_$G.log
