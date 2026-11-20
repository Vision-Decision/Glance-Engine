#!/usr/bin/env bash
# Run inside the existing engopt VM. Only one C0 chain can own the GPU at a time.
set -eu -o pipefail
cd /content/dv
exec 9>/tmp/engopt-c0.lock
flock 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
out=/content/out/scan
ckpt=/content/ckpt/v1/ckpt_final.pt
mkdir -p "$out"
failed=0
for stage in "$@"; do
    file="$out/c0_$stage"
    if [[ ${C0_RESUME:-0} == 1 ]] && grep -qx "$stage exit=0" "$file.log" 2>/dev/null; then continue; fi
    case "$stage" in
        cached) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts clips,cached --Ns 64,256 --clip_cfgs 4x96 --clip_fresh 0 --n_calls 30 --out "$file.json");;
        speed) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts lat,mixed,followup --out "$file.json");;
        gui2048|gui4096) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts gui --gui_nv "${stage#gui}" --out "$file.json");;
        floors) cmd=(python -m speed.ceiling_model "$out");;
        peaks) cmd=(python -m speed.roofline --out "$file.json");;
        reference_noise) cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt" --measure-reference "$file.json");;
        photo_equiv) cmd=(python -m scan.bench_scan --ckpt "$ckpt" --parts equiv --out "$file.json");;
        fault) cmd=(python -c 'import runpy, sys; from scan.varlen import FreshPack; init = FreshPack.__init__
def off_by_one(self, *args):
    init(self, *args)
    if self.ids.is_cuda:
        self.heads["q_last_packed"] -= 1
FreshPack.__init__ = off_by_one
runpy.run_module("scan.test_scan_equivalence", run_name="__main__")' --real --ckpt "$ckpt");;
        real) cmd=(python -m scan.test_scan_equivalence --real --ckpt "$ckpt");;
        diag) cmd=(python -m speed.diag_equiv --ckpt "$ckpt" --out "$file.json");;
        sglang_clean) cmd=(/content/sglang_env/bin/python -m speed.bench_serve --backend sglang --parts clients,ceiling,followup,gui --out "$file.json");;
        sglang_bank) cmd=(/content/sglang_env/bin/python -m scan.bench_sglang --parts clips,cached,cached_box --Ns 64,256 --n_calls 8 --out "$file.json");;
        sglang_box) cmd=(/content/sglang_env/bin/python -m scan.bench_sglang --parts cached_box --Ns 64,256 --n_calls 8 --out "$file.json");;
        video) cmd=(python -m speed.bench_glance --ckpt "$ckpt" --modes eng_fused --parts video1 --out "$file.json");;
        profile_bank) cmd=(python -m speed.profile_cached_scan --ckpt "$ckpt" --out "$file.json");;
        shapes) cmd=(python -m speed.ceiling_shapes --out "$file.json");;
        clients) cmd=(python -m speed.bench_clients --ckpt "$ckpt" --out "$file.json");;
        clients_vllm|clients_sglang) backend=${stage#clients_}; cmd=("/content/${backend}_env/bin/python" -m speed.bench_serve --backend "$backend" --parts clients --async_sched --out "$file.json");;
        *) echo "Unknown stage $stage" >&2; exit 2;;
    esac
    while true; do
        owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $owners ]] && break
        sleep 3
    done
    { date -u; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; echo 'GPU compute process list empty'; printf '%q ' "${cmd[@]}"; echo; } > "$file.log"
    rc=0
    PATH="$(dirname "${cmd[0]}"):$PATH" "${cmd[@]}" >> "$file.log" 2>&1 || rc=$?
    echo "$stage exit=$rc" | tee -a "$file.log"
    python scan/vm/push.py engine_opt/00_ceiling "$file.json" "$file.log"
    if [[ $stage = fault ]]; then
        if [[ $rc != 1 ]] || ! grep -qx SCAN_EQUIVALENCE_FAIL "$file.log"; then failed=1; fi
    elif [[ $rc != 0 ]]; then failed=1
    fi
done
exit "$failed"
