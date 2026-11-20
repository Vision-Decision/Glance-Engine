#!/usr/bin/env bash
# VM side: wait until FILE contains PATTERN, then run a command.  bash speed/vm_then.sh FILE PATTERN CMD...
F=$1; P=$2; shift 2
until grep -q -- "$P" "$F" 2>/dev/null; do sleep 30; done
"$@"
