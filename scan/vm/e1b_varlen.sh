#!/usr/bin/env bash
# Run one item-1 stage on engopt from /content/dv. Files for each stage can be pulled independently.
set -euo pipefail
cd /content/dv
runner=$PWD
stage=${1:?expected before, after, profile_reference, profile_varlen, equiv, scan_real, diag, vllm or sglang}
out=/content/out/scan
mkdir -p "$out"
file="$out/item1_$stage"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ckpt=/content/ckpt/v1/ckpt_final.pt
case "$stage" in
  before|after)
    mode=(--fresh_impl varlen)
    if [[ $stage = before ]]; then
      mode=(--fresh_impl reference)
      if [[ -n ${ITEM1_BEFORE_DIR:-} ]]; then
        cd "$ITEM1_BEFORE_DIR"
        export PYTHONPATH="$PWD"
        mode=()  # unmodified checkout, before --fresh_impl existed
      fi
    fi
    cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256
         --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 "${mode[@]}" --out "$file.json");;
  profile_reference|profile_varlen)
    cmd=(python -m scan.profile_fresh --ckpt "$ckpt" --fresh_impl "${stage#profile_}" --out "$file.json");;
  scan_real) cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt");;
  equiv) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts equiv --out "$file.json");;
  diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt" --out "$file.json");;
  vllm)
    cmd=(/content/vllm_env/bin/python -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256
         --kinds yesno --n_calls 8 --max_fresh_tokens 70000 --out "$file.json");;
  sglang)
    export PATH=/content/sglang_env/bin:$PATH
    cmd=(/content/sglang_env/bin/python -m scan.bench_sglang --parts fresh --Tvs 128,256 --Ns 32,64,128,256
         --n_calls 8 --out "$file.json");;
  *) echo "Unknown stage $stage" >&2; exit 2;;
esac
while true; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [[ -z $pids ]] && break
  sleep 10
done
{
  date -u
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
  echo "GPU compute process list empty before $stage in $PWD"
  printf '%q ' "${cmd[@]}"; echo
} > "$file.log"
rc=0
"${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
echo "stage=$stage exit=$rc" | tee -a "$file.log"
python "$runner/scan/vm/push.py" engine_opt/01_varlen_prefill "$file.log" "$file.json"
exit "$rc"
