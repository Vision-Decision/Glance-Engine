#!/usr/bin/env bash
# Local: copy the final result JSONs of each VM (outputs/sprint_speed/vm_*/) into outputs/sprint_speed/raw/, then build
# the report.  Dev/smoke runs are left out.   speed/collect.sh
set -euo pipefail
cd "$(dirname "$0")/.."
O=outputs/sprint_speed; mkdir -p $O/raw
for d in $O/vm_*; do
  for f in "$d"/glance*_*.json "$d"/serve_*_*.json "$d"/hw_*.json "$d"/equiv*_*.json "$d"/prof_*.json; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    case "$b" in serve_*quick*|dev*|smoke*) continue ;; esac
    cp "$f" $O/raw/
  done
done
ls $O/raw
uv run -q --with matplotlib python speed/report.py --raw $O/raw --out $O > /dev/null
echo "report: $O/tables.md"
