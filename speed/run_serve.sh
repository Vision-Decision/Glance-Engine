#!/usr/bin/env bash
# VM side: one generative-baseline run.  bash speed/run_serve.sh BACKEND TAG [bench_serve args...]  -> /content/out/spd/serve_TAG.{json,log}
cd /content/dv
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
BE=${1:?backend}; TAG=${2:?tag}; shift 2
PY=/content/${BE}_env/bin/python
export PATH=/content/${BE}_env/bin:$PATH        # SGLang JIT kernels need the venv's ninja
mkdir -p /content/out/spd
timeout ${TO:-5400} $PY -m speed.bench_serve --backend $BE --out /content/out/spd/serve_$TAG.json "$@" > /content/out/spd/serve_$TAG.log 2>&1
echo "exit $?" >> /content/out/spd/serve_$TAG.log
