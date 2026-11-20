#!/usr/bin/env bash
# Run on the dedicated engopt VM. Queue independent chains behind the existing locks.
set -eu -o pipefail
cd /content/dv
exec 9>/content/item6.gpu.lock
flock 9
exec 8>/tmp/engopt-vision-prep.lock
flock 8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:/content/item3-qla-deps:/content/item3-extras" FLA_FLASH_QLA=0
export LD_LIBRARY_PATH="/content/item3-qla-deps/z3/lib:${LD_LIBRARY_PATH:-}"
ckpt=/content/ckpt/v1/ckpt_final.pt
failed=0
for stage in "$@"; do
    file="/content/out/scan/item7_${ITEM7_LABEL:+${ITEM7_LABEL}_}${stage}"
    if [[ -f "$file.log" ]] && grep -qx "$stage exit=0" "$file.log"; then
        echo "Completed $stage under this run label"
        continue
    fi
    case "$stage" in
        kernels|probe|split|patch_order|gdn|wide_gdn|video_checks|video_entry|mlp|norm|patch_control|patch_conv|binary_control|conv_control|conv_binary) cmd=(python -m scan.profile_frames --stage "$stage" --ckpt "$ckpt");;
        profile) cmd=(python -m scan.profile_vision --stage profile_final --ckpt "$ckpt");;
        fresh) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens 16384);;
        video) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts video1);;
        singleton) cmd=(python -m speed.bench_scheduler --backend scheduled --warm-shapes --parts followup --clients 1);;
        larger_bursts) cmd=(python -m speed.bench_scheduler --backend scheduled --warm-shapes --parts fresh --clients 16,64 --tokens 16384);;
        scheduled|followup|serial)
            cmd=(python -m speed.bench_scheduler --backend scheduled --warm-shapes)
            if [[ $stage == followup ]]; then cmd+=(--parts followup); fi
            if [[ $stage == serial ]]; then cmd=(python -m speed.bench_scheduler --backend serial --warm-shapes --parts followup); fi;;
        features) cmd=(python -m scan.profile_vision --stage features_verify --ckpt "$ckpt");;
        graph_features) cmd=(python -m scan.profile_vision --stage graph_features --ckpt "$ckpt");;
        interleaving) cmd=(python -m speed.test_scheduler_gpu);;
        equiv) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts equiv);;
        real) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized --ckpt "$ckpt");;
        diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt");;
        fresh_vllm|fresh_sglang)
            backend=${stage#fresh_}
            cmd=(env HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" PATH="/content/${backend}_env/bin:$PATH"
                 "/content/${backend}_env/bin/python" -m "scan.bench_${backend}" --parts fresh --Tvs 128,256 --n_calls 8)
            if [[ $backend == vllm ]]; then
                cmd+=(--Ns_fresh 32,64,128,256 --kinds yesno --max_fresh_tokens 70000)
            else
                cmd+=(--Ns 32,64,128,256)
            fi;;
        video_vllm|video_sglang|video_tuned)
            backend=${stage#video_}
            if [[ $backend == tuned ]]; then backend=vllm; fi
            cmd=(env HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" PATH="/content/${backend}_env/bin:$PATH"
                 "/content/${backend}_env/bin/python" -m speed.bench_serve --backend "$backend" --parts video1)
            if [[ $backend == vllm ]]; then cmd+=(--async_sched); fi
            if [[ $stage == video_tuned ]]; then cmd+=(--spec mtp --quant fp8); fi;;
        scheduler_vllm|scheduler_sglang)
            backend=${stage#scheduler_}
            snapshots=(/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*)
            cmd=(env PYTHONPATH="$PWD" PATH="/content/${backend}_env/bin:$PATH" "/content/${backend}_env/bin/python"
                 -m speed.bench_scheduler --model "${snapshots[0]}" --backend "$backend");;
        *) echo "Unknown stage $stage" >&2; exit 2;;
    esac
    if [[ $stage != real ]]; then cmd+=(--out "$file.json"); fi
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 3
    done
    { date -u; nvidia-smi --query-gpu=name,memory.used --format=csv; echo GPU_IDLE; sha256sum speed/engine.py speed/decision_worker.py speed/vision_mlp.py scan/engine.py; printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$stage exit=$rc" | tee -a "$file.log"
    [[ $rc == 0 ]] || failed=1
    python scan/vm/push.py engine_opt/07_frames "$file.log" "$file.json"
done
exit "$failed"
