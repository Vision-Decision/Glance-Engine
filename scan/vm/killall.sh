#!/usr/bin/env bash
# Kill every experiment process on this VM (anchored patterns: never matches the shell running this script).
for pat in '^bash -lc export PYTHONUNBUFFERED' '^bash scan/vm/' '^timeout ' '^python3? -m scan' '^/content/sglang_env/bin/python' \
           '^/content/vllm_env/bin/python' '^sglang::' '^VLLM' ; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 5
ps -eo pid,args | grep -E "scan|sglang|vllm" | grep -v -E "grep|killall" | cut -c1-120
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo KILLALL_DONE
