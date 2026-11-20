#!/usr/bin/env bash
# Pull small result files (json + logs) of the sprint from a VM into outputs/sprint_speed/vm_<SESSION>/.
# speed/pull.sh SESSION
set -euo pipefail
S=${1:?session}
cd "$(dirname "$0")/.."
D=outputs/sprint_speed/vm_$S; mkdir -p "$D"
echo 'import subprocess;print(subprocess.run("cd /content/out/spd && tar czf /content/spd_results.tgz *.json *.log 2>/dev/null; ls -la /content/spd_results.tgz",shell=True,capture_output=True,text=True).stdout)' \
  | timeout 300 colab-pool exec "$S" --timeout 200 | grep -v '^\[pool\]'
timeout 300 colab-pool download "$S" /content/spd_results.tgz "$D/results.tgz" > /dev/null
tar xzf "$D/results.tgz" -C "$D" && rm "$D/results.tgz"
ls "$D"
