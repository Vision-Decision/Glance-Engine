#!/usr/bin/env bash
# Launch a command detached on a pooled VM (survives exec disconnects):  speed/rbg.sh SESSION LOGFILE "command"
set -euo pipefail
S=${1:?session}; LOG=${2:?log}; shift 2
cd "$(dirname "$0")/.."
export CMD="$*" LOG
python3 scripts/remote_bg.py | timeout 300 colab-pool exec "$S" --timeout 200 2>&1 | grep -v '^\[pool\]'
