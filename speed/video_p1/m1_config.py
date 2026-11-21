import json, sys
from huggingface_hub import hf_hub_download, get_safetensors_metadata
out = {}
for m in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]:
    d = {}
    for f in ["config.json", "preprocessor_config.json", "video_preprocessor_config.json"]:
        try: d[f] = json.load(open(hf_hub_download(m, f)))
        except Exception as e: d[f] = f"ERR {type(e).__name__}"
    md = get_safetensors_metadata(m)
    cnt = {}
    for fn, fm in md.files_metadata.items():
        for name, ti in fm.tensors.items():
            n = 1
            for s in ti.shape: n *= s
            key = "visual" if ".visual." in name or name.startswith("visual") else ("mtp" if "mtp" in name else ("embed" if "embed_tokens" in name else ("lm_head" if "lm_head" in name else "lm")))
            cnt[key] = cnt.get(key, 0) + n
            if key == "visual" and ("merger" in name):
                cnt["visual_merger"] = cnt.get("visual_merger", 0) + n
    d["params"] = cnt
    out[m] = d
json.dump(out, open("m1_raw.json", "w"), indent=1)
print(json.dumps({m: v["params"] for m, v in out.items()}, indent=1))
