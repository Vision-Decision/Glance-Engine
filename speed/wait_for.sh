#!/usr/bin/env bash
# Local helper: wait until FILE is non-empty (or contains PATTERN), up to SECONDS.  speed/wait_for.sh FILE SECONDS [PATTERN]
F=${1:?file}; T=${2:-600}; P=${3:-}
end=$(( $(date +%s) + T ))
while [ "$(date +%s)" -lt "$end" ]; do
  if [ -s "$F" ] && { [ -z "$P" ] || grep -q -- "$P" "$F"; }; then exit 0; fi
  sleep 5
done
exit 1
