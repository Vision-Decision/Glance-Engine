#!/usr/bin/env bash
# Scan VM setup (idempotent, run ON the VM, detached):  bash /content/dv/scan/vm/setup.sh [--vllm]
#   base   : transformers 5.23 + fla 0.6 + peft (as speed/ and training), Qwen3.5-4B, v1 and v1.1 checkpoints
#   images : COCO train2014 images of the first 400 RefCOCO-val and first 200 gRefCOCO-val records (fetched one by one)
#   --vllm : vLLM 0.34 in its own venv (baseline)
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_XET=1
cd /content/dv
echo "[setup] $(date) start"; nvidia-smi --query-gpu=name,memory.total --format=csv
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate av
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]))
for f in ("v1/ckpt_final.pt", "v1.1/ckpt_final.pt"):
    print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", f, local_dir="/content/ckpt"))
PY
mkdir -p /content/data/coco/train2014
python - <<'PY'
import json, os, urllib.request, concurrent.futures as cf
fs = []
for p, n in (("data/prep/eval_refcoco_val.jsonl", 400), ("data/prep/eval_grefcoco_val.jsonl", 200)):
    fs += [json.loads(l)["file"] for l in list(open(p))[:n]]
fs = sorted(set(fs))
def get(f):
    p = f"/content/data/coco/train2014/{f}"
    if not os.path.exists(p):
        for _ in range(3):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/train2014/{f}", p); return
            except Exception:
                pass
with cf.ThreadPoolExecutor(32) as ex:
    list(ex.map(get, fs))
print("coco images", len(os.listdir("/content/data/coco/train2014")), "wanted", len(fs))
PY
if [ "${1:-}" = "--vllm" ]; then
  pip install -q uv
  /content/vllm_env/bin/python -c "import vllm" 2>/dev/null || (uv venv -q --allow-existing -p 3.12 /content/vllm_env && uv pip install -q --python /content/vllm_env/bin/python "vllm==0.34.0" pillow)
  /content/vllm_env/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
fi
echo "[setup] $(date) DONE"
