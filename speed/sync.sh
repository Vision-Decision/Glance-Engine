#!/usr/bin/env bash
# Push repo code (dv/, speed/, ablation/, train.py, eval.py, small manifest) + HF token to /content/dv on a pooled VM.
# speed/sync.sh SESSION
set -euo pipefail
S=${1:?session}
cd "$(dirname "$0")/.."
export COPYFILE_DISABLE=1
TMP=$(mktemp -d)
tar czf "$TMP/c.tgz" --exclude='__pycache__' dv speed ablation train.py eval.py data/prep/eval_refcoco_val.jsonl
cp ~/.cache/huggingface/token "$TMP/hf_token"
timeout 300 colab-pool upload "$S" "$TMP/hf_token" /content/hf_token >/dev/null
timeout 600 colab-pool upload "$S" "$TMP/c.tgz" /content/c.tgz >/dev/null
echo 'import subprocess;print(subprocess.run("mkdir -p /content/dv /root/.cache/huggingface && cp /content/hf_token /root/.cache/huggingface/token && tar xzf /content/c.tgz -C /content/dv && ls /content/dv",shell=True,capture_output=True,text=True).stdout)' \
  | timeout 300 colab-pool exec "$S" --timeout 240 | grep -v '^\[pool\]'
rm -rf "$TMP"
