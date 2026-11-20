"""Call a running glance-serve: ask about an image, then follow up by image id without resending pixels.

    glance-serve --tiny --port 8002 &        # or: glance-serve  (Glance-4B-preview on CUDA)
    python examples/http_client.py --url http://127.0.0.1:8002
"""

import argparse
import base64
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8002")
ap.add_argument("--image", default="demo/data/search_large/211069.jpg")  # COCO val2017, CC BY 2.0
a = ap.parse_args()


def post(body):
    req = urllib.request.Request(a.url + "/v1/decisions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))


with open(a.image, "rb") as f:
    image = base64.b64encode(f.read()).decode()
first = post({"image": image, "cache": True, "questions": [
    {"type": "box", "text": "the giraffe"},
    {"type": "noul", "instr": "Is this photo taken outdoors?"},
]})
print(json.dumps(first["results"][0], indent=1))
again = post({"image_ids": first["image_ids"], "questions": [
    {"type": "choice", "instr": "How many giraffes are there?", "options": ["one", "two", "three or more"]},
]})
print(json.dumps(again["results"][0], indent=1), f"({again['sec'] * 1e3:.1f} ms, cached prefix)")
