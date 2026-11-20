#!/usr/bin/env bash
# VM side: dev loop. speed/run_dev.sh TAG [bench_glance args...]  -> /content/out/spd/TAG.{json,log}
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TAG=${1:?tag}; shift
mkdir -p /content/out/spd
timeout ${TO:-3000} python -m speed.bench_glance --ckpt /content/ckpt/v1/ckpt_final.pt --out /content/out/spd/$TAG.json "$@" > /content/out/spd/$TAG.log 2>&1
echo "exit $?" >> /content/out/spd/$TAG.log
