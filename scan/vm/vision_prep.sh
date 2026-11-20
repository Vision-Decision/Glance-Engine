#!/usr/bin/env bash
# Run from the synced checkout on engopt. Multiple detached chains share this lock.
set -euo pipefail
cd /content/dv
exec 9>/tmp/engopt-vision-prep.lock
flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
out=/content/out/scan
ckpt=/content/ckpt/v1/ckpt_final.pt
failed=0
for stage in "$@"; do
    file="$out/item2_$stage"
    case "$stage" in
        before|prep_only|fast)
            cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts fresh --Tvs 128,256
                --Ns_fresh 1,2,4,8,16,32,64,128,256 --n_calls 10 --fresh_kind yesno
                --fresh_chunked 1 --max_fresh_tokens 16384 --fresh_impl varlen --out "$file.json")
            case "$stage" in
                before) cmd+=(--vision_prep reference);;
                prep_only) cmd+=(--vision_prep rgb);;
                *) cmd+=(--vision_prep fused);;
            esac;;
        profile_before|profile_fast|features_fast|features_verify|graph_features|tune)
            cmd=(python -m scan.profile_vision --stage "$stage" --ckpt "$ckpt" --out "$file.json");;
        small_vllm|serving_vllm|unique_vllm|cold_vllm|small_cold_vllm)
            sizes=1,2,4,8,16,32
            if [[ $stage != small* ]]; then sizes+=,64,128,256; fi
            cmd=(/content/vllm_env/bin/python -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh "$sizes"
                --kinds yesno --n_calls 8 --max_fresh_tokens 70000 --out "$file.json")
            if [[ $stage == small_vllm || $stage == serving_vllm ]]; then cmd+=(--retain_encoder_cache); fi
            if [[ $stage == unique_vllm ]]; then cmd+=(--unique_fresh); fi;;
        small_sglang|serving_sglang)
            sizes=1,2,4,8,16,32
            if [[ $stage == serving_sglang ]]; then sizes+=,64,128,256; fi
            export PATH=/content/sglang_env/bin:$PATH
            cmd=(/content/sglang_env/bin/python -m scan.bench_sglang --parts fresh --Tvs 128,256 --Ns "$sizes"
                --n_calls 8 --out "$file.json");;
        single_fused) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts lat,mixed --out "$file.json");;
        fused_gui2048|fused_gui4096) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts gui --gui_nv "${stage##*gui}" --out "$file.json");;
        clients_fused) cmd=(python -m speed.bench_clients --ckpt "$ckpt" --out "$file.json");;
        streams) cmd=(python -m scan.vision_streams --out "$file.json");;
        paged2) cmd=(python -m scan.vision_dispatch --attention-only --ckpt "$ckpt" --out "$file.json");;
        patch) cmd=(python -m scan.vision_dispatch --patch-only --ckpt "$ckpt" --out "$file.json");;
        dispatch) cmd=(python -m scan.vision_dispatch --ckpt "$ckpt" --out "$file.json");;
        real_fast) cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt");;
        cpu) cmd=(env CUDA_VISIBLE_DEVICES= python -m scan.test_scan_equivalence);;
        equiv_fast) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts equiv --out "$file.json");;
        diag_fused) cmd=(python -m speed.diag_equiv --ckpt "$ckpt" --out "$file.json");;
        *) exit 2;;
    esac
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 3
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; echo 'GPU compute process list empty'; printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$stage exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/02_vision_prep "$file.json" "$file.log"
    if [[ $rc != 0 ]]; then failed=1; fi
done
exit "$failed"
