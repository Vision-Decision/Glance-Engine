#!/usr/bin/env bash
# VM side: engine-matched decode-free bound (vLLM prefill + 1 token per question), bf16 and FP8, all regimes.
cd /content/dv
G=${1:?gpu tag}; NC=${NCALLS:-50}
bash speed/run_serve.sh vllm vllm_1tok_$G --n_calls $NC --async_sched --one_token --parts lat,mixed,followup,tput,tput2,video,gui
bash speed/run_serve.sh vllm vllm_fp8_1tok_$G --n_calls $NC --async_sched --one_token --quant fp8 --parts lat,mixed,followup,tput,tput2,video,gui
echo "[onetok] ALL DONE" >> /content/out/spd/main_$G.log
