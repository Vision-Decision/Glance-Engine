#!/usr/bin/env bash
# Run through scan/vm/bg.sh on engopt. A fresh ITEM6_RUN_LABEL starts a new series.
set -eu -o pipefail
exec 9>/content/item6.gpu.lock
flock 9
exec 8>/tmp/engopt-vision-prep.lock
flock 8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=/content/out/scan
native=(env PYTHONPATH="$PWD:/content/item3-qla-deps:/content/item3-extras"
    LD_LIBRARY_PATH="/content/item3-qla-deps/z3/lib:${LD_LIBRARY_PATH:-}" FLA_FLASH_QLA=0)
failed=0
for mode in "$@"; do
    file="$out/item6_${ITEM6_RUN_LABEL:+${ITEM6_RUN_LABEL}_}$mode"
    cmd=(python -m speed.bench_scheduler --out "$file.json")
    case "$mode" in
        vllm|sglang)
            snapshots=(/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*)
            cmd=(env PATH="/content/${mode}_env/bin:$PATH" "/content/${mode}_env/bin/python"
                "${cmd[@]:1}" --model "${snapshots[0]}" --backend "$mode");;
        warm_serial) cmd+=(--backend serial --warm-shapes);;
        production|production_ids|standard)
            cmd+=(--backend scheduled --warm-shapes)
            if [[ "$mode" != standard ]]; then cmd=("${native[@]}" "${cmd[@]}"); fi
            if [[ "$mode" == production_ids ]]; then cmd+=(--by-id --parts followup); fi;;
        standard_check|production_check)
            cmd=(python -m speed.test_scheduler_gpu --out "$file.json")
            if [[ "$mode" == production_check ]]; then cmd=("${native[@]}" "${cmd[@]}"); fi;;
        production_http) cmd=("${native[@]}" python -m speed.load_server --self-test --out "$file.json");;
        release_diag) cmd=(python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --out "$file.json");;
        release_scan) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized --ckpt /content/ckpt/v1/ckpt_final.pt);;
        api_tests) cmd=(python -m unittest speed.test_scheduler speed.test_server);;
        floors) cmd=(python -m speed.scheduler_floors --out "$file.json");;
        *) echo "Unknown benchmark stage: $mode" >&2; exit 2;;
    esac
    while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 3; done
    {
        date -u
        nvidia-smi --query-gpu=name,memory.used --format=csv
        echo GPU_IDLE
        sha256sum speed/decision_worker.py speed/scheduler.py speed/engine.py scan/varlen.py
        printf '%q ' "${cmd[@]}"
        echo
    } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "exit=$rc" | tee -a "$file.log"
    [[ $rc == 0 ]] || failed=1
    python scan/vm/push.py engine_opt/06_scheduler "$file.log" "$file.json"
done
exit "$failed"
