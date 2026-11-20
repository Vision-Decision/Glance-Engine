#!/usr/bin/env bash
# Run on engopt after sync. All item 9 chains share this lock, including the initial margin job.
set -eu -o pipefail
cd /content/dv
exec 9>/tmp/engopt-item9.lock
flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:/content/item3-qla-deps:/content/item3-extras" TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/content/item3-qla-deps/z3/lib:${LD_LIBRARY_PATH:-}" FLA_FLASH_QLA=0
ckpt=/content/ckpt/v1/ckpt_final.pt
failed=0
for stage in "$@"; do
    file=/content/out/scan/item9_${ITEM9_LABEL:+${ITEM9_LABEL}_}$stage
    case $stage in
        presence_audit) cmd=(python -m scan.bench_box_readout --presence-audit "$ckpt" --out "$file.json");;
        precision_latency) cmd=(python -m scan.bench_box_readout --precision-latency "$ckpt" --out "$file.json");;
        micro) cmd=(python -m scan.bench_box_readout --out "$file.json");;
        margins) cmd=(python -m scan.e0_box_equiv --ckpt "$ckpt" --out "$file.json");;
        calibration|replay|norm_probe|fp32_probe|residual_probe)
            cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt" --measure-reference "$file.json")
            if [[ $stage == *probe ]]; then cmd+=(--calibration-head oracle); fi
            if [[ $stage == replay ]]; then cmd+=(--verify-head-layout "${GLANCE_BOX_REDUCTION:-blocks}"); fi
            if [[ $stage == norm_probe ]]; then cmd+=(--probe-final-norm); fi
            if [[ $stage == residual_probe ]]; then cmd+=(--probe-residual); fi
            if [[ $stage == fp32_probe ]]; then cmd+=(--probe-fp32); fi;;
        cached) cmd=(python -m scan.bench_scan --parts cached --cached_tvs 640 --Ns 256 --n_calls 15 --ckpt "$ckpt" --out "$file.json");;
        fresh) cmd=(python -m scan.bench_scan --parts fresh --Tvs 128 --Ns_fresh 128 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --ckpt "$ckpt" --out "$file.json");;
        mixed) cmd=(python -m speed.bench_glance --modes eng_fused --parts mixed --ckpt "$ckpt" --out "$file.json");;
        speed) cmd=(python -m speed.bench_glance --modes eng_fused --parts lat,followup --ckpt "$ckpt" --out "$file.json");;
        real|fused|fault|small_fault)
            cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt")
            if [[ $stage == fused ]]; then cmd+=(--fused-prefill optimized); fi
            if [[ $stage == fault || $stage == small_fault ]]; then cmd+=(--inject "$stage"); fi;;
        equiv) cmd=(python -m scan.bench_scan --parts equiv --ckpt "$ckpt" --out "$file.json");;
        interleaving) cmd=(python -m speed.test_scheduler_gpu --ckpt "$ckpt" --out "$file.json");;
        diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt" --out "$file.json");;
        cpu) cmd=(env CUDA_VISIBLE_DEVICES= python -m scan.test_scan_equivalence);;
        *) exit 2;;
    esac
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 2
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv
      echo 'GPU compute process list empty'; echo "GLANCE_BOX_REDUCTION=${GLANCE_BOX_REDUCTION:-blocks}"
      printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$stage exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/09_numerics "$file.log" "$file.json"
    if [[ $stage == *fault ]]; then
        if [[ $rc != 1 ]] || ! grep -qx SCAN_EQUIVALENCE_FAIL "$file.log"; then failed=1; fi
    elif [[ $rc != 0 ]]; then failed=1
    fi
done
exit "$failed"
