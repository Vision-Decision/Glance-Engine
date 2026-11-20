#!/usr/bin/env bash
# Item 4 serialized runs. Each process starts on an idle device.
set -eu -o pipefail
cd /content/dv
exec 9>/tmp/engopt-vision-prep.lock
flock 9
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ckpt=/content/ckpt/v1/ckpt_final.pt
out=/content/out/scan
failed=0
for name in "$@"; do
    file=$out/item4_$name
    export GLANCE_BRANCH_MODE=${name%%_*}
    case $GLANCE_BRANCH_MODE in release|final|staged|checked) export GLANCE_BRANCH_MODE=fused;; esac
    stage=${name#*_}
    case "$stage" in
        cached) cmd=(python -m scan.bench_scan --parts clips,cached --Ns 64,256 --clip_cfgs 4x96 --clip_fresh 0 --n_calls 30);;
        speed) cmd=(python -m speed.bench_glance --modes eng_fused --parts lat,mixed,followup);;
        profile) cmd=(python -m scan.profile_branch --stage profile);;
        micro|questions|clip|stages) cmd=(python -m scan.profile_branch --stage "$stage");;
        real) cmd=(python -m scan.test_scan_equivalence --real);;
        merged) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized);;
        equiv) cmd=(python -m scan.bench_scan --parts equiv);;
        diag) cmd=(python -m speed.diag_equiv);;
        *) exit 2;;
    esac
    cmd+=(--ckpt "$ckpt")
    [[ $stage == real || $stage == merged ]] || cmd+=(--out "$file.json")
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 2
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv
      echo 'GPU compute process list empty'
      sha256sum speed/engine.py scan/engine.py scan/profile_branch.py scan/branch_micro.py
      [[ ! -f speed/gdn_branch.py ]] || sha256sum speed/gdn_branch.py
      echo "GLANCE_BRANCH_MODE=$GLANCE_BRANCH_MODE"
      printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$name exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/04_gdn_branch "$file.json" "$file.log"
    if [[ $rc != 0 ]]; then failed=1; fi
done
exit "$failed"
