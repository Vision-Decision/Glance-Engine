"""Per-stage ideal batching floors using the same calls as bench_scheduler (no GPU work)."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from dv.config import Config
from dv.data.mixed import ChatTok
from transformers import AutoProcessor

from scan.data import load_images
from scan.engine import question_block
from speed import workload as W
from speed.bench_scheduler import burst
from speed.ceiling_model import call_floor


def main(out):
    # Rates measured by speed/roofline.py, outputs/engine_opt/ROOFLINE.md
    rates = {"bf16": 401.73, "fp32": 67.4, "copy": 1463.2, "h2d": 56.9, "launch_ms": 0.002233}
    snapshots = list(Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots").iterdir())
    tok = ChatTok(AutoProcessor.from_pretrained(snapshots[0]), Config())
    prep, root = "data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014"
    res = {}
    for work in ("box", "yesno", "multi", "followup", "gui", "fresh"):
        if work == "fresh":
            pool = load_images(prep, root, 100, 128, frame=True)
        elif work == "gui":
            pool = W.load_gui("/content/gui/sspro40.jsonl", 40)
        else:
            pool = W.load_coco(prep, root, 100)
        for c in (1, 4, 16, 64):
            stages = []
            for it in range(2, 7):
                calls, _ = burst(pool, work, c, it, 0, mark=False)
                shapes = Counter()
                for ims, qs in calls:
                    lens = tuple(len(question_block(tok, q)[0]) for q in qs)
                    for im in ims:
                        v = im.width * im.height // 1024
                        shapes[v, v + len(tok.user_open) + 2, lens] += 1
                row, _ = call_floor(work, [(n, v, p, l) for (v, p, l), n in shapes.items()], rates)
                if "H2D" in row:
                    row["H2D"] /= 8  # uint8 RGB vs duplicated FP32 temporal patches
                stages.append(row)
            res[f"{work}_{c}"] = {
                "stages": {k: median(r[k] for r in stages) for k in stages[0]},
                "floor_ms": median(sum(r.values()) for r in stages),
            }
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/content/out/scan/item6_floors.json"))
    main(parser.parse_args().out)
