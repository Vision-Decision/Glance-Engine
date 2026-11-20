#!/usr/bin/env bash
# Wait until a VM log matches a regex (or timeout), then print its tail.
#   scan/vm/wait.sh SESSION LOGNAME 'REGEX' [max_seconds] [tail_lines]
S=${1:?session}; L=${2:?log}; R=${3:?regex}; T=${4:-170}; N=${5:-30}
D="$(dirname "$0")"
end=$((SECONDS + T))
while [ $SECONDS -lt $end ]; do
  if bash "$D/sh.sh" "$S" "grep -cE '$R' /content/out/scan/$L.log" 40 2>/dev/null | grep -qE '^[1-9]'; then break; fi
  sleep 20
done
bash "$D/sh.sh" "$S" "grep -vE 'Warning|warn|it/s|s/it' /content/out/scan/$L.log | tail -$N" 60
