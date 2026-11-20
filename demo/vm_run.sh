#!/usr/bin/env bash
# All GPU measurements for the README GIFs, serialized on one G4 (run ON the VM from /content/dv after
# scan/vm/setup.sh --vllm):   bash demo/vm_run.sh STAGE [STAGE ...]
#   pick    choose redistributable COCO val2017 images (needs /content/coco/annotations/instances_val2017.json)
#   fetch   download the picked images to /content/demo_imgs
#   race    32-question race: Glance bf16 engine, vLLM FP8+MTP, vLLM bf16
#   search  256-image search: Glance cached scan + calibration, vLLM bf16 and FP8+MTP (cached regime)
set -uo pipefail
cd /content/dv
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ckpt=/content/ckpt/v1/ckpt_final.pt
out=demo/data
vpy=/content/vllm_env/bin/python
mkdir -p $out /content/demo_imgs
img=/content/demo_imgs/$(python -c "import json;print(json.load(open('demo/race_questions.json'))['file'])" 2>/dev/null)
wait_idle() { while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 2; done; }
for st in "$@"; do
  echo "=== $st $(date -u)"; wait_idle
  case $st in
    pick) python demo/pick_coco.py --ann /content/coco/annotations/instances_val2017.json --out $out/coco_pick.json;;
    fetch) python - <<'PY'
import json, os, urllib.request, concurrent.futures as cf
p = json.load(open("demo/data/coco_pick.json"))
recs = p["race"] + p["search"]["bank"]
def get(r):
    f = f"/content/demo_imgs/{r['file']}"
    if not os.path.exists(f):
        urllib.request.urlretrieve(r["coco_url"], f)
with cf.ThreadPoolExecutor(32) as ex:
    list(ex.map(get, recs))
print("images", len(os.listdir("/content/demo_imgs")))
PY
    ;;
    race)
      python demo/race_glance.py --ckpt $ckpt --image "$img" --questions demo/race_questions.json --out $out/race_glance.json; wait_idle
      $vpy demo/race_vllm.py --image "$img" --questions demo/race_questions.json --config fp8mtp --out $out/race_vllm_fp8mtp.json; wait_idle
      $vpy demo/race_vllm.py --image "$img" --questions demo/race_questions.json --config bf16 --n_calls 30 --out $out/race_vllm_bf16.json;;
    search)
      python demo/search_glance.py --ckpt $ckpt --pick $out/coco_pick.json --image_dir /content/demo_imgs --out $out/search_glance.json; wait_idle
      $vpy demo/search_vllm.py --pick $out/coco_pick.json --image_dir /content/demo_imgs --config bf16 --out $out/search_vllm_bf16.json; wait_idle
      $vpy demo/search_vllm.py --pick $out/coco_pick.json --image_dir /content/demo_imgs --config fp8mtp --out $out/search_vllm_fp8mtp.json;;
    *) echo "unknown stage $st"; exit 2;;
  esac
  echo "=== $st exit=$? $(date -u)"
done
