"""Ask several typed questions about one image with one prefill, then follow up from the cache.

    python examples/quickstart.py                    # Glance-4B-preview, needs CUDA
    python examples/quickstart.py --model tiny       # random tiny model on CPU (meaningless answers)
"""

import argparse
import time

from glance_engine import Box, Choice, Glance, YesNo

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview")
ap.add_argument("--image", default="demo/data/search_large/211069.jpg")  # COCO val2017, CC BY 2.0
a = ap.parse_args()

glance = Glance.tiny() if a.model == "tiny" else Glance.from_pretrained(a.model)
image_id = glance.add_image(a.image)  # vision tower + prefill, once

questions = [
    Box("the giraffe"),
    YesNo("Is this photo taken outdoors?"),
    Choice("How many giraffes are there?", ["one", "two", "three or more"]),
]
glance.ask(image_id, questions)  # first call of a new shape captures CUDA graphs (seconds on GPU)
t0 = time.perf_counter()
answers = glance.ask(image_id, questions)
ms = (time.perf_counter() - t0) * 1e3
for ans in answers:
    print(ans)
print(f"{len(answers)} follow-up answers from the cached prefix in {ms:.1f} ms")
