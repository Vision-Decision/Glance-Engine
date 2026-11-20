# Print a compact view of a bench JSON on the VM:  colab-pool exec S -f scan/vm/show.py   (edit PATH / KEYS below)
import json, os
PATH = os.environ.get("SHOW", "/content/out/scan/bench_g4.json")
d = json.load(open(PATH))


def r(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: r(x) for k, x in v.items() if not isinstance(x, list)}
    return v


for top, val in d.items():
    if top in ("args",):
        continue
    if isinstance(val, dict):
        for k, v in val.items():
            print(top, k, json.dumps(r(v)))
    else:
        print(top, val)
