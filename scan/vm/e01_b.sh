#!/usr/bin/env bash
# E1 baselines on VM B after setup.sh --vllm: SGLang venv + bench; vLLM hybrid-cache options probe.
cd /content/dv
O=/content/out/scan; mkdir -p $O
step() { echo "[e01b] $1 start $(date +%H:%M:%S)"; }
step sglang_install
/content/sglang_env/bin/python -c "import sglang" 2>/dev/null || (uv venv -q --allow-existing -p 3.12 /content/sglang_env && \
  uv pip install -q --python /content/sglang_env/bin/python "sglang[all]" av pillow) > $O/e1_sglang_install.log 2>&1
/content/sglang_env/bin/python -c "import sglang,torch;print('sglang',sglang.__version__,'torch',torch.__version__)" >> $O/e1_sglang_install.log 2>&1
step probe
/content/sglang_env/bin/python - > $O/e1_probe_sglang.log 2>&1 <<'PY'
import dataclasses
from sglang.srt.server_args import ServerArgs
for f in dataclasses.fields(ServerArgs):
    n = f.name.lower()
    if any(k in n for k in ("mamba", "radix", "hybrid", "cache", "linear", "page")):
        print(f.name, "=", f.default)
PY
/content/vllm_env/bin/python - > $O/e1_probe_vllm.log 2>&1 <<'PY'
import dataclasses
from vllm.config import CacheConfig
for f in dataclasses.fields(CacheConfig):
    n = f.name.lower()
    if any(k in n for k in ("mamba", "block", "prefix", "mode", "align", "hybrid")):
        print(f.name, "=", f.default)
PY
echo "[e01b] PROBE DONE $(date +%H:%M:%S)"
