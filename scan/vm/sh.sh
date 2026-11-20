#!/usr/bin/env bash
# Run a shell command on a pooled VM and print its output (tail).   scan/vm/sh.sh SESSION 'command' [timeout_s]
# Detached jobs: scan/vm/sh.sh S 'cd /content/dv && nohup bash -c "..." > /content/out/scan/x.log 2>&1 &'
S=${1:?session}; C=${2:?cmd}; T=${3:-240}
CJ=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$C")
printf 'import subprocess\nr=subprocess.run(["bash","-lc",%s],capture_output=True,text=True,timeout=%s)\nprint(r.stdout[-8000:]); print(r.stderr[-3000:])\n' "$CJ" "$((T-20))" \
  | timeout $((T+90)) colab-pool exec "$S" --timeout "$T" | grep -v '^\[pool\]'
