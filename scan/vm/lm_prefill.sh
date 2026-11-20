#!/usr/bin/env bash
# Run from the synced checkout. Serialize all Item 3 chains on engopt.
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
    stage=${name#accel_}
    file="$out/item3_$name"
    environment=()
    if [[ $name == accel_* ]]; then
        environment=(env PYTHONPATH="$PWD:/content/item3-qla-deps:/content/item3-extras"
                     LD_LIBRARY_PATH="/content/item3-qla-deps/z3/lib:${LD_LIBRARY_PATH:-}" FLA_FLASH_QLA=0)
    fi
    case "$stage" in
        headline) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts fresh --Tvs 128 --Ns_fresh 128,256
                       --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 --out "$file.json");;
        before|after|final|production)
            cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts fresh --Tvs 128,256 --Ns_fresh 1,2,4,8,16,32,64,128,256
                 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384 --out "$file.json")
            case "$stage" in
                before) cmd+=(--lm_prefill reference);;
                after) cmd+=(--lm_prefill pointwise);;
                final) cmd+=(--lm_prefill direct);;
            esac;;
        stages_*|profile_before|profile_after|profile_final|profile_production|dispatch*|graphs*|gdn|fixed_features|tma|video_equiv)
            cmd=(python -m scan.profile_lm --stage "$stage" --ckpt "$ckpt" --out "$file.json");;
        gdn_optional|gdn_single_card|gdn_tile64|qla_model|qla_fresh)
            cmd=(env PYTHONPATH="$PWD:/content/item3-qla-deps:/content/item3-extras" LD_LIBRARY_PATH="/content/item3-qla-deps/z3/lib:${LD_LIBRARY_PATH:-}"
                 FLA_FLASH_QLA=0 python -m scan.profile_lm --stage "$stage" --ckpt "$ckpt" --out "$file.json");;
        real_fused|real_tower|real_release) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized --ckpt "$ckpt");;
        real_control) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill reference --ckpt "$ckpt");;
        ncu_*)
            pass=${stage%_*}
            n=${stage##*_}
            cmd=(ncu --profile-from-start off --clock-control none --cache-control none --filter-mode per-launch-config
                 --launch-count 1 --metrics dram__bytes.sum,gpu__time_duration.sum --page raw --csv --print-units base
                 --log-file "$file.csv" python -m scan.profile_lm --stage "$pass" --n "$n" --ckpt "$ckpt" --out "$file.json");;
        real) cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt");;

        cached) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts clips,cached --Ns 64,256 --clip_cfgs 4x96
                     --clip_fresh 0 --n_calls 30 --out "$file.json");;
        equiv|equiv_tower) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts equiv --out "$file.json");;
        heuristic)
            nvcc -O2 -arch=sm_120 scan/lm_blaslt.cu -lcublasLt -o /tmp/lm_blaslt
            cmd=(/tmp/lm_blaslt);;
        counters) cmd=(ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum --launch-count 1 python -c 'import torch; x=torch.ones(1024,device="cuda"); x.add_(1); torch.cuda.synchronize()');;
        diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt" --out "$file.json");;
        followup) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts followup --out "$file.json");;
        clients) cmd=(python -m speed.bench_clients --ckpt "$ckpt" --out "$file.json");;
        single) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts lat,mixed --out "$file.json");;
        gui2048|gui4096) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts gui --gui_nv "${stage#gui}" --out "$file.json");;
        video|video_packed) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts video1 --out "$file.json");;
        *) exit 2;;
    esac
    cmd=("${environment[@]}" "${cmd[@]}")
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 2
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; echo 'GPU compute process list empty'; sha256sum speed/lm_prefill.py speed/flash_gdn.py speed/engine.py scan/varlen.py scan/engine.py; printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$name exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/03_lm_prefill "$file.json" "$file.log" "$file.csv"
    if [[ $rc != 0 ]]; then failed=1; fi
done
exit "$failed"
