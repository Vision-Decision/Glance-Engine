"""Workloads shared by the Glance engine benchmark and the generative (vLLM / SGLang) baselines.

Images: the first records of RefCOCO val (COCO train2014), resized to <= max_vis visual tokens (sides multiples of
32, as dv/data/dataset.resize_dims), or ScreenSpot-Pro screenshots for the GUI regime.
Questions per image:
  box    the record's referring expressions, cycled to N
  mixed  typed questions in the ratio box : yes/no : choice : score = 1 : 1 : 1 : 1 (cycled), built from the record:
         yes/no  'Is there {expr} in the image?'
         choice  'Which of these is in the image?' over the image's expression + 3 expressions of other images
         score   'How cluttered is the scene?' over 5 levels
Generative prompts (baselines) ask for the same answers: a box '[x1, y1, x2, y2]' (0-1000) or 'none', 'yes'/'no',
an option letter, a level letter.
"""
import json, math, random
from PIL import Image

from dv.data.dataset import resize_dims

LEVELS = ["very low", "low", "medium", "high", "very high"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

BOX_PROMPT = ('Locate the object described as "{expr}" in the image. Output its bounding box as [x1, y1, x2, y2] '
              'with coordinates normalized to 0-1000. If no such object is in the image, answer exactly "none".')


def load_coco(prep, image_root, n, max_vis=640, max_upscale=2.0):
    recs = [json.loads(l) for l in open(prep)][:n]
    out = []
    for r in recs:
        img = Image.open(f"{image_root}/{r['file']}").convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis, max_upscale)
        out.append(dict(img=img.resize((w, h), Image.BICUBIC), wh=(W, H), exprs=[q["text"] for q in r["questions"]],
                        gts=[q["box"] for q in r["questions"]]))
    return out


def load_gui(path, n, max_vis=2048, max_upscale=1.0):
    out = []
    for l in list(open(path))[:n]:
        r = json.loads(l)
        img = Image.open(r["path"]).convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis, max_upscale)
        out.append(dict(img=img.resize((w, h), Image.BICUBIC), wh=(W, H), exprs=[q["text"] for q in r["questions"]]))
    return out


def questions(images, i, n, kind="box", seed=0):
    """n typed questions (dicts in dv/data/mixed.py record format) for image i."""
    ex = images[i]["exprs"]
    rng = random.Random(seed * 100003 + i)
    qs = []
    for j in range(n):
        t = ("box", "noul", "choice", "score")[j % 4] if kind == "mixed" else kind
        e = ex[j % len(ex)]
        if t == "box":
            qs.append(dict(type="box", text=e, present=True, box=None))
        elif t == "noul":
            qs.append(dict(type="noul", instr=f"Is there {e} in the image?", target=0.5))
        elif t == "choice":
            others = [images[(i + k) % len(images)]["exprs"][0] for k in (1, 2, 3)]
            opts = [e] + others
            rng.shuffle(opts)
            qs.append(dict(type="choice", instr="Which of these is in the image?", options=opts, target=[1] + [0] * 3))
        else:
            qs.append(dict(type="score", instr="How cluttered is the scene?", options=LEVELS, target=[0.2] * 5))
    return qs


def gen_prompt(q):
    """Text prompt for the generative baseline answering the same typed question; max new tokens it needs."""
    if q["type"] == "box":
        return BOX_PROMPT.format(expr=q["text"]), 24
    if q["type"] == "noul":
        return q["instr"] + " Answer yes or no.", 2
    opts = "".join(f"{LETTERS[k]}. {o}\n" for k, o in enumerate(q["options"]))
    head = "Options:" if q["type"] == "choice" else "Scale (lowest to highest):"
    return f"{q['instr']}\n\n{head}\n{opts}Answer with the letter only.", 2
