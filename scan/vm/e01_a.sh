#!/usr/bin/env bash
# E0 (exactness) + our side of E1 (latency on this VM), run detached on VM A after setup.sh.
cd /content/dv
O=/content/out/scan; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[e01a] $1 start $(date +%H:%M:%S)"; }
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("Vision-Decision/glance-train-ckpts", "scan/short_v1/ckpt_final.pt", local_dir="/content/ckpt"))
PY
step cpu_equiv
(CUDA_VISIBLE_DEVICES= timeout 3600 python -m scan.test_scan_equivalence > $O/e0_cpu_equiv.log 2>&1; echo "[e01a] cpu_equiv exit $?") &
step real_equiv_bidir
timeout 1800 python -m scan.test_scan_equivalence --real --ckpt /content/ckpt/v1/ckpt_final.pt --mask block_bidir > $O/e0_real_bidir.log 2>&1
step box_diag_bf16
timeout 3600 python -m scan.e0_box_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e0_box_bf16.json > $O/e0_box_bf16.log 2>&1
step box_diag_fp32
timeout 3600 python -m scan.e0_box_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e0_box_fp32.json --fp32 --n_img 24 > $O/e0_box_fp32.log 2>&1
wait   # CPU test must not overlap the latency run
step latency_short
timeout 1800 python -m scan.bench_scan --ckpt /content/ckpt/scan/short_v1/ckpt_final.pt --out $O/e1_ours_latency_short.json \
  --parts clips --clip_cfgs 4x96 --clip_fresh 0 --Ns 8,32,64,128,256 --n_calls 20 --short 1 > $O/e1_ours_latency_short.log 2>&1
echo "[e01a] ALL DONE $(date +%H:%M:%S)"
