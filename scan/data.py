"""Shared, model-free workload helpers for the scan benchmarks (importable from the vLLM venv)."""
import json, os
from PIL import Image

from dv.data.dataset import resize_dims


def load_images(prep, root, n, max_vis, frame=False):
    recs = [json.loads(l) for l in open(prep)]
    out = []
    for r in recs:
        p = f"{root}/{r['file']}"
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size
        w, h = resize_dims(640, 480, max_vis, 2.0) if frame else resize_dims(W, H, max_vis, 2.0)
        out.append(dict(img=img.resize((w, h), Image.BICUBIC), wh=(W, H), qs=r["questions"]))
        if len(out) >= n:
            break
    return out


QUERIES = ["person standing", "a red car", "the dog on the left", "man in a blue shirt holding a phone", "white plate",
           "the umbrella", "a cat sleeping on the sofa", "bus", "woman with glasses near the window", "green bottle",
           "the clock on the wall", "a child riding a bike", "laptop", "horse", "the tallest giraffe", "pizza slice"]


def serving_clips(prep, root, count, configs):
    """Native-video payloads for the same cached clip sweep in both serving harnesses."""
    import numpy as np
    frames = [r["img"] for r in load_images(prep, root, 64, 640, frame=True)]
    for config in configs.split(","):
        length, budget = map(int, config.split("x"))
        size = resize_dims(640, 480, budget, 2.0)
        resized = [np.asarray(im.resize(size)) for im in frames]
        units = [np.stack([resized[(c*3+k) % len(resized)] for k in range(length)]) for c in range(count)]
        yield f"f{length}_v{budget}", units
