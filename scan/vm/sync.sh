#!/usr/bin/env bash
# Push code + HF token to /content/dv on a pooled VM.   scan/vm/sync.sh SESSION
# speed/, scan/ and configs come from this repo (glance-engine). The model code (dv/, train.py, eval.py, small preps,
# video/vdm/gv) lives in glance-core / glance-video: point MONO at a checkout that has them (default: ../decision-vison,
# the old monorepo, read-only). Paths missing in this repo are taken from MONO.
set -euo pipefail
S=${1:?session}
cd "$(dirname "$0")/../.."
HERE=$(pwd)
MONO=${MONO:-~/Developer/decision-vison}
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
LOCAL=(speed scan configs)
EXT=(dv video/vdm/gv train.py eval.py data/prep/eval_refcoco_val.jsonl data/prep/eval_grefcoco_val.jsonl)
mkdir -p "$TMP/stage"
tar cf - --exclude=__pycache__ "${LOCAL[@]}" ${SYNC_EXTRA:-} | tar xf - -C "$TMP/stage"
for p in "${EXT[@]}"; do
  if [ -e "$HERE/$p" ]; then (cd "$HERE" && tar cf - --exclude=__pycache__ "$p") | tar xf - -C "$TMP/stage"
  else (cd "$MONO" && git show HEAD:"$p" >/dev/null 2>&1 || [ -e "$p" ]) && (cd "$MONO" && tar cf - --exclude=__pycache__ "$p") | tar xf - -C "$TMP/stage"; fi
done
(cd "$TMP/stage" && tar czf "$TMP/c.tgz" .)
cp ~/.cache/huggingface/token "$TMP/hf_token"
timeout 300 colab-pool upload "$S" "$TMP/hf_token" /content/hf_token >/dev/null
timeout 600 colab-pool upload "$S" "$TMP/c.tgz" /content/c.tgz >/dev/null
bash scan/vm/sh.sh "$S" 'mkdir -p /content/dv /root/.cache/huggingface && cp /content/hf_token /root/.cache/huggingface/token && tar xzf /content/c.tgz -C /content/dv && ls /content/dv' 200
rm -rf "$TMP"
