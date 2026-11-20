#!/usr/bin/env bash
# Sprint-speed VM setup (idempotent). On the VM:  bash /content/dv/speed/setup_vm.sh [--serve] > /content/setup.log 2>&1
#   base   : transformers 5.23 + fla 0.6 + peft (Glance engine, same versions as training/eval)
#   --serve: also separate venvs for vLLM and SGLang (generative baselines)
# Data (small): the first 200 RefCOCO-val COCO images fetched one by one; 40 ScreenSpot-Pro screenshots.
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd /content/dv
echo "[setup] $(date) start"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate
python -c "import transformers, fla; print('transformers', transformers.__version__, 'fla', fla.__version__)"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]))
print(hf_hub_download("Vision-Decision/glance-train-ckpts-legacy", "v1/ckpt_final.pt", local_dir="/content/ckpt"))
PY
# ---- COCO images of the first 200 RefCOCO-val records
mkdir -p /content/data/coco/train2014
python - <<'PY'
import json, os, urllib.request, concurrent.futures as cf
recs = [json.loads(l) for l in open("data/prep/eval_refcoco_val.jsonl")][:200]
def get(f):
    p = f"/content/data/coco/train2014/{f}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"http://images.cocodataset.org/train2014/{f}", p)
with cf.ThreadPoolExecutor(16) as ex:
    list(ex.map(get, [r["file"] for r in recs]))
print("coco images", len(os.listdir("/content/data/coco/train2014")))
PY
# ---- ScreenSpot-Pro: up to 40 screenshots (<=3 per application) + their instructions -> /content/gui/sspro40.jsonl
python - <<'PY'
import json, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); rid = "likaixin/ScreenSpot-Pro"
files = api.list_repo_files(rid, repo_type="dataset")
anns = sorted(f for f in files if f.startswith("annotations/") and f.endswith(".json"))
recs = []
os.makedirs("/content/gui", exist_ok=True)
for a in anns:
    items = json.load(open(hf_hub_download(rid, a, repo_type="dataset")))
    fns = []
    for it in items:
        if it["img_filename"] not in fns:
            fns.append(it["img_filename"])
    for fn in fns[:3]:
        p = hf_hub_download(rid, f"images/{fn}", repo_type="dataset")
        qs = [dict(text=x["instruction"], present=True, box=x["bbox"]) for x in items if x["img_filename"] == fn]
        recs.append(dict(path=p, wh=[x for x in items if x["img_filename"] == fn][0]["img_size"], questions=qs))
    if len(recs) >= 40:
        break
with open("/content/gui/sspro40.jsonl", "w") as f:
    for r in recs[:40]:
        f.write(json.dumps(r) + "\n")
print("gui screenshots", len(recs[:40]))
PY
if [ "${1:-}" = "--serve" ]; then
  pip install -q uv
  /content/vllm_env/bin/python -c "import vllm" 2>/dev/null || (uv venv -q --allow-existing -p 3.12 /content/vllm_env && uv pip install -q --python /content/vllm_env/bin/python vllm)
  /content/vllm_env/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  /content/sglang_env/bin/python -c "import sglang" 2>/dev/null || (uv venv -q --allow-existing -p 3.12 /content/sglang_env && uv pip install -q --python /content/sglang_env/bin/python sglang)
  /content/sglang_env/bin/python -c "import sglang,torch;print('sglang',sglang.__version__,'torch',torch.__version__)"
fi
echo "[setup] $(date) DONE"
