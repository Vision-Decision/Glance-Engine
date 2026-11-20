#!/usr/bin/env bash
# Retry creating a G4 session NAME on acct-0, then acct-7, then acct-8, every 4 min, for up to N tries.
# speed/get_g4.sh NAME [TRIES]
NAME=${1:?name}; TRIES=${2:-20}
for i in $(seq 1 "$TRIES"); do
  for acc in acct-0 acct-7 acct-8; do
    if timeout 900 colab-pool new --if-missing "$NAME" --gpu g4 --account "$acc" --allow-other-account > /tmp/get_g4_$NAME.log 2>&1; then
      echo "GOT $NAME on $acc (try $i) $(date)"; exit 0
    fi
  done
  echo "try $i failed $(date +%H:%M)"; sleep 240
done
exit 1
