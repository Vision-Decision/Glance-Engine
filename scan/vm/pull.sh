#!/usr/bin/env bash
# Pull small result files (json + logs) from /content/out/scan on a VM into outputs/scan/raw/<SESSION>/.
#   scan/vm/pull.sh SESSION
set -euo pipefail
S=${1:?session}
cd "$(dirname "$0")/../.."
D=outputs/scan/raw/$S; mkdir -p "$D"
bash scan/vm/sh.sh "$S" 'cd /content/out/scan && tar czf /content/scan_results.tgz $(ls *.json *.log 2>/dev/null)' 120 >/dev/null
timeout 300 colab-pool download "$S" /content/scan_results.tgz "$D/results.tgz" > /dev/null
tar xzf "$D/results.tgz" -C "$D" && rm "$D/results.tgz"
ls "$D"
