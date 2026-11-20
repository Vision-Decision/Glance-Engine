#!/usr/bin/env bash
# Local keep-alive for sprint VMs: ping each session every 12 min (hard timeout per ping) until the stop file exists
# or MAX_H hours pass.  speed/keepalive.sh SESSION [SESSION...]   stop: touch /tmp/spd_keepalive_stop
cd "$(dirname "$0")/.."
end=$(( $(date +%s) + ${MAX_H:-10} * 3600 ))
while [ ! -f /tmp/spd_keepalive_stop ] && [ "$(date +%s)" -lt "$end" ]; do
  for S in "$@"; do
    o=$(RSH_WALL=300 RSH_TIMEOUT=60 scripts/rsh.sh "$S" "echo ok; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>&1 | tr '\n' ' ')
    echo "[keepalive $(date +%H:%M)] $S: ${o:0:120}"
  done
  sleep 720
done
echo "[keepalive] stopped"
