#!/usr/bin/env bash
# VM side, reduced main pass for the cheaper GPU (L4): fewer calls, the main Glance modes, the two strongest vLLM settings.
export NCALLS=20 NEQ=40 SKIP_EQUIV=1 MODES=ref_lora,eng_graph,eng_fp8 SERVE_SET=vllm_async,vllm_mtp
bash /content/dv/speed/vm_main.sh ${1:-l4} all
