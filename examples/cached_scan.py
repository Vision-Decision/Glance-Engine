"""One question over many images: encode once, then every new question scans the whole bank.

    python examples/cached_scan.py                   # Glance-4B-preview, needs CUDA
    python examples/cached_scan.py --model tiny      # random tiny model on CPU (meaningless answers)
"""

import argparse
import glob

from PIL import Image

from glance_engine import Glance

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview")
ap.add_argument("--images", default="demo/data/search_large/*.jpg")  # COCO val2017, CC BY 2.0
a = ap.parse_args()

glance = Glance.tiny() if a.model == "tiny" else Glance.from_pretrained(a.model)
images = [Image.open(p).convert("RGB") for p in sorted(glob.glob(a.images))]
scanner = glance.scanner(max_vis=64 if a.model == "tiny" else 640)

# The first call encodes the bank; later calls only run the query. The first query of each new
# shape also captures CUDA graphs on GPU, so the second round shows steady-state latency.
for query in ("a giraffe", "a toaster", "a giraffe", "a toaster"):
    r = scanner.scan(query, images)
    best = r["images"][r["set"]["argmax"]]
    print(f"{query!r}: any={r['set']['any']} best={best} latency_ms={r['latency_ms']}")
