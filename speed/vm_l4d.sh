#!/usr/bin/env bash
# VM side, L4: eager engine latency/mixed sweep (no graph capture; L4 is GPU-bound so graphs gain little, and on 22 GB
# large-N graphs thrash).  -> glance6eager_l4.json
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TO=3600 bash speed/run_dev.sh glance6eager_l4 --n_calls 20 --max_batch 8 --modes eng_eager --parts lat,mixed
echo "[l4d] ALL DONE" >> /content/out/spd/main_l4.log
