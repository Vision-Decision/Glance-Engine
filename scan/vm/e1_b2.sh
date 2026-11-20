#!/usr/bin/env bash
# E1 sweep on VM B: SGLang (default + mamba radix strategies, token-only bound) and vLLM (default + smaller mamba blocks,
# token-only bound). Every run: same 4-frame clips (or same-length token prefixes), warm cache, one new yes/no query.
cd /content/dv
O=/content/out/scan
export PATH=/content/sglang_env/bin:$PATH
S="/content/sglang_env/bin/python -m scan.bench_sglang --Ns 8,64,256 --n_calls 8"
V="/content/vllm_env/bin/python -m scan.bench_vllm --Ns 8,64,256 --n_calls 8 --kinds yesno"
run() { echo "[e1b2] $1 start $(date +%H:%M:%S)"; timeout 2400 bash -c "$2" > $O/e1_$1.log 2>&1; echo "[e1b2] $1 exit $? $(date +%H:%M:%S)"; }
run sgl_tokens        "$S --parts tokens --out $O/e1_sglang_tokens.json"
run sgl_xbuf_clips    "$S --parts clips,tokens --server_kw '{\"mamba_radix_cache_strategy\": \"extra_buffer\"}' --out $O/e1_sglang_xbuf.json"
run sgl_cached_img    "$S --parts cached --out $O/e1_sglang_img640.json"
run vllm_tokens       "$V --parts tokens --out $O/e1_vllm_tokens.json"
for B in 64 16; do
  run vllm_mb$B "$V --parts clips,tokens --engine_kw '{\"mamba_block_size\": $B, \"block_size\": $B}' --out $O/e1_vllm_mb$B.json"
done
echo "[e1b2] ALL DONE $(date +%H:%M:%S)"
