#!/usr/bin/env bash
# Serialized item 5 measurements on the synced engopt checkout.
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
    file=$out/item5_$name
    stage=${name#final_}
    stage=${stage#release_}
    stage=${stage#verified_}
    case "$stage" in
        bf16|fp8|w12)
            mode=eng_$stage
            [[ $stage != bf16 ]] || mode=eng_fused
            cmd=(python -m speed.bench_glance --modes "$mode" --parts lat,mixed,followup,video1 --n_calls 30)
            [[ $stage != w12 ]] || cmd+=(--w12 /content/item5-w12-v1);;
        scan_bf16|scan_fp8|scan_w12)
            cmd=(python -m scan.bench_scan --parts cached,fresh --Ns 1,8,32,64,256 --cached_tvs 640
                 --Ns_fresh 1,2,4,8 --Tvs 128,256 --fresh_kind yesno --fresh_chunked 1 --n_calls 10)
            [[ $stage != scan_fp8 ]] || cmd+=(--fp8 1)
            [[ $stage != scan_w12 ]] || cmd+=(--w12 /content/item5-w12-v1);;
        sglang) cmd=(env HF_HUB_OFFLINE=1 /content/sglang_env/bin/python -m scan.bench_sglang
                     --parts cached,cached_box --Ns 1,8,32 --n_calls 8);;
        equiv) cmd=(python -m scan.bench_scan --parts equiv --w12 /content/item5-w12-v1);;
        counters) cmd=(ncu --profile-from-start off --clock-control none --cache-control none
                       --metrics dram__bytes.sum,gpu__time_duration.sum
                       --page raw --csv --print-units base --log-file "$file.csv" python -m speed.bench_w12 --stage counters);;
        diag) cmd=(python -m speed.diag_equiv);;
        real) cmd=(python -m scan.test_scan_equivalence --real);;
        w12_real) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized --w12 /content/item5-w12-v1);;
        merged) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized);;
        *) cmd=(python -m speed.bench_w12 --stage "$stage");;
    esac
    [[ $stage == sglang ]] || cmd+=(--ckpt "$ckpt")
    [[ $stage == real || $stage == merged || $stage == w12_real ]] || cmd+=(--out "$file.json")
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 2
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv
      echo 'GPU compute process list empty'
      sha256sum speed/engine.py speed/bench_glance.py
      for src in speed/*w12*.py speed/specpilot/w12.cu scan/engine.py scan/bench_scan.py; do [[ ! -f $src ]] || sha256sum "$src"; done
      printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$name exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/05_w12 "$file.json" "$file.log" "$file.csv"
    if [[ $rc != 0 ]]; then failed=1; fi
done
exit "$failed"
