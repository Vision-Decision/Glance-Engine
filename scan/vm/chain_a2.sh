#!/usr/bin/env bash
# VM A after E2 (p2-b was released): E1b leftovers -> E6 agent -> E9 Glance-X -> E10 profile -> E5/E7/E8.
cd /content/dv
O=/content/out/scan
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "[a2] $1 start $(date +%H:%M:%S)"; }
while pgrep -f "e2_a[.]sh" >/dev/null; do sleep 30; done
step sglang_install
/content/sglang_env/bin/python -c "import sglang" 2>/dev/null || (uv venv -q --allow-existing -p 3.12 /content/sglang_env && \
  uv pip install -q --python /content/sglang_env/bin/python "sglang[all]" av pillow ninja) > $O/e1_sglang_install_a.log 2>&1
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/glance-train-ckpts', 'v1/ckpt_final.pt', local_dir='/content/ckpt'))" 2>/dev/null || \
python -c "from huggingface_hub import hf_hub_download as h; print(h('Vision-Decision/glance-train-ckpts-legacy', 'v1/ckpt_final.pt', local_dir='/content/ckpt'))"
step e1b_vllm
timeout 3600 /content/vllm_env/bin/python -m scan.bench_vllm --parts fresh --Tvs 128,256 --Ns_fresh 32,64,128,256 --kinds yesno \
  --n_calls 8 --max_fresh_tokens 70000 --out $O/e1b_vllm_fresh2.json > $O/e1b_vllm2.log 2>&1
step e1b_sglang
PATH=/content/sglang_env/bin:$PATH timeout 3600 /content/sglang_env/bin/python -m scan.bench_sglang --parts fresh --Tvs 128,256 \
  --Ns 32,64,128,256 --n_calls 8 --out $O/e1b_sglang_fresh3.json > $O/e1b_sglang3.log 2>&1
for C in 16384 8192 4096; do
  step e1b_ours_c$C
  timeout 3600 python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out $O/e1b_ours_fresh_c$C.json --parts fresh \
    --Tvs 128,256 --Ns_fresh 32,64,128,256 --n_calls 10 --fresh_kind yesno --fresh_chunked 1 --max_fresh_tokens $C > $O/e1b_ours_c$C.log 2>&1
done
python scan/vm/push.py e1b $O/e1b_vllm_fresh2.json $O/e1b_vllm2.log $O/e1b_sglang_fresh3.json $O/e1b_sglang3.log $O/e1b_ours_fresh_c*.json $O/e1b_ours_c*.log
bash scan/vm/e6_b.sh                     # E6 (data, smoke, full), then E9 and the E10 profile
bash scan/vm/e578_a.sh                   # E5 / E7 / E8
echo "[a2] ALL DONE $(date +%H:%M:%S)"
