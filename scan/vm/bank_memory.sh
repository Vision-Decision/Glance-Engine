#!/usr/bin/env bash
set -eu -o pipefail
cd /content/dv
exec 9>/tmp/engopt-vision-prep.lock
flock 9
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=/content/out/scan
ckpt=/content/ckpt/v1/ckpt_final.pt
failed=0
for stage in "$@"; do
    file=$out/item8_${ITEM8_LABEL:+${ITEM8_LABEL}_}$stage
    case "$stage" in
        before_bytes|after_bytes) cmd=(python -m scan.bench_bank_memory --stage accounting);;
        before|after) cmd=(python -m scan.bench_scan --parts clips,cached --Ns 64,256 --n_calls 15 --clip_fresh 0 --ckpt "$ckpt");;
        real) cmd=(python -m scan.test_scan_equivalence --real --fused-prefill optimized --ckpt "$ckpt");;
        equiv) cmd=(python -m scan.bench_scan --parts equiv --ckpt "$ckpt");;
        interleaving) cmd=(python -m speed.test_scheduler_gpu);;
        diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt");;
        tier_packed) cmd=(python -m scan.bench_bank_tier --bank-format state_packed);;
        vllm|sglang)
            cmd=(env HF_HUB_OFFLINE=1 PATH="/content/${stage}_env/bin:$PATH" "/content/${stage}_env/bin/python"
                 -m "scan.bench_${stage}" --parts cached_clips --clip_cfgs 4x48,2x96 --Ns 64,256 --n_calls 8);;
        resident_chunks) cmd=(python -m scan.bench_bank_tier --resident-chunks);;
        resident) cmd=(python -m scan.bench_bank_tier --resident);;
        tier) cmd=(python -m scan.bench_bank_tier);;
        fixtures|presence|video)
            checkpoint=$ckpt
            if [[ $stage == video ]]; then checkpoint=/content/ckpt/scan/short_gv0/ckpt_final.pt; fi
            cmd=(python -m scan.bench_bank_precision --stage "$stage" --ckpt "$checkpoint");;
        stages) cmd=(python -m scan.profile_branch --stage stages --banks-only --ckpt "$ckpt");;
        probe|breakeven|formats|entry|lifecycle) cmd=(python -m scan.bench_bank_memory --stage "$stage");;
        *) exit 2;;
    esac
    if [[ $stage == before* || $stage == probe ]]; then cmd=(env GLANCE_BANK_FORMAT=legacy "${cmd[@]}"); fi
    if [[ $stage != real ]]; then cmd+=(--out "$file.json"); fi
    while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 2; done
    { date -u; nvidia-smi --query-gpu=name,memory.used --format=csv
      echo 'GPU compute process list empty'; sha256sum scan/engine.py scan/bench_bank*.py scan/bank*.py scan/host_bank.py speed/gdn_branch.py
      printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$stage exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/08_bank_memory "$file.json" "$file.log"
    if [[ $rc != 0 ]]; then failed=1; fi
done
exit "$failed"
