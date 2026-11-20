#!/usr/bin/env bash
# VM side: (re)run only the SGLang configurations.  bash speed/vm_sglang.sh GPU_TAG
export SERVE_SET=sglang,sglang_mtp
bash /content/dv/speed/vm_serve.sh ${1:?gpu tag}
echo "[sglang] ALL DONE" >> /content/out/spd/main_$1.log
