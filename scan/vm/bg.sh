#!/usr/bin/env bash
# Start a detached job on a pooled VM (own session, no inherited pipes):  scan/vm/bg.sh SESSION LOGNAME 'command'
# log -> /content/out/scan/LOGNAME.log ; cwd /content/dv ; PYTHONPATH=/content/dv
S=${1:?session}; L=${2:?log}; C=${3:?cmd}
CJ=$(python3 -c 'import json,sys;print(json.dumps("export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/content/dv HF_HUB_DISABLE_XET=1; "+sys.argv[1]))' "$C")
printf 'import subprocess,os\nos.makedirs("/content/out/scan",exist_ok=True)\np=subprocess.Popen(["bash","-lc",%s],stdin=subprocess.DEVNULL,stdout=open("/content/out/scan/%s.log","a"),stderr=subprocess.STDOUT,start_new_session=True,cwd="/content/dv")\nprint("started pid",p.pid)\n' "$CJ" "$L" \
  | timeout 900 colab-pool exec "$S" --timeout 120 | grep -v '^\[pool\]'
