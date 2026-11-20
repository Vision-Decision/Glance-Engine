"""Real-weight randomized concurrent requests against isolated BF16 requests."""

import argparse
import asyncio
import json
import math
import random
from collections import Counter
from pathlib import Path

from scan.engine import ScanEngine
from scan.equivalence_limits import BF16_LIMIT
from speed import workload as W
from speed.bench_glance import Runner
from speed.decision_worker import Decision, DecisionWorker
from speed.engine import fuse_weights_
from speed.scheduler import Scheduler


def reference(runner, req):
    rows = []
    for im in req.images:
        out, _ = runner.call([im], [req.questions])
        row = []
        for i, q in enumerate(req.questions):
            if q["type"] == "box":
                row.append(
                    {
                        "box": out["box_pred"][i].tolist(),
                        "p_present": out["presence_logit"][i].sigmoid().item(),
                        "p_iou50": out["iou_logit"][i].sigmoid().item(),
                    }
                )
            else:
                logits = out["typed_logits"][i]
                row.append({"probs": logits[logits.isfinite()].softmax(-1).tolist()})
        rows.append(row)
    return rows


def compare(actual, expected, metrics, sample):
    counts, errors, failures = metrics["counts"], metrics["max_abs"], metrics["outliers"]
    assert len(actual) == len(expected)
    for j, (row, ref) in enumerate(zip(actual, expected)):
        assert len(row) == len(ref)
        for k, (pred, truth) in enumerate(zip(row, ref)):
            if "box" in truth:
                ok = (pred["p_present"] >= 0.5) == (truth["p_present"] >= 0.5)
                delta = {
                    "presence_prob": abs(pred["p_present"] - truth["p_present"]),
                    "iou_prob": abs(pred["p_iou50"] - truth["p_iou50"]),
                    "box_pred": max(abs(x - y) for x, y in zip(pred["box"], truth["box"])),
                }
            else:
                p, r = pred["probs"], truth["probs"]
                assert len(p) == len(r) and len(p) > 0
                ok = p.index(max(p)) == r.index(max(r))
                delta = {"typed_prob": max(abs(x - y) for x, y in zip(p, r))}
            counts.update(total=1, agreed=int(ok))
            assert all(math.isfinite(v) for v in delta.values()), (sample, j, k, delta)
            for field, v in delta.items():
                errors[field] = max(errors[field], v)
            if not ok or any(v > BF16_LIMIT[f] for f, v in delta.items()):
                failures.append(
                    {
                        "sample": sample,
                        "image": j,
                        "question": k,
                        "actual": pred,
                        "expected": truth,
                        "delta": delta,
                    }
                )


async def check(a):
    rng = random.Random(1701)
    runner = Runner(a)
    pool = W.load_coco(a.prep, a.image_root, 32)
    jobs, refs = [], []
    for i, rec in enumerate(pool):
        qs = W.questions(pool, i, (1, 1, 4, 8)[i % 4], ("box", "noul", "mixed", "mixed")[i % 4])
        ims = [rec["img"]]
        if i % 7 == 0:
            ims.append(pool[(i + 9) % len(pool)]["img"])
        req = Decision(ims, qs, cache=i % 3 == 0)
        jobs.append(req)
        refs.append(reference(runner, req))
    runner.set_mode("ref_merged")
    fuse_weights_(runner.net.lm, runner.net.layer_types)
    engine = ScanEngine(runner.net, runner.heads, runner.tok, runner.proc, fast_ops=True)
    if a.single_vision:
        import torch

        vision = engine.fresh_vision

        def per_image(pixels, grid):
            sizes = grid.prod(-1).tolist()
            return torch.cat([vision(p, g[None]).clone() for p, g in zip(pixels.split(sizes), grid)])

        engine.fresh_vision = per_image
    worker = DecisionWorker(engine, packed_shared=not a.padded_shared, group_vision=a.group_vision)
    scheduler = Scheduler(worker)
    scheduler.start()
    metrics = {
        k: {"counts": Counter(), "max_abs": Counter(), "outliers": []} for k in ("vs_isolated", "vs_shipped")
    }
    try:
        isolated = [await scheduler.submit(req) for req in jobs]
        isolated_reference = {"counts": Counter(), "max_abs": Counter(), "outliers": []}
        for i, prediction in enumerate(isolated):
            compare(prediction, refs[i], isolated_reference, [i])
        worker.cache.clear()
        for wave in range(2):
            indices = list(range(len(jobs)))
            rng.shuffle(indices)
            outs = await asyncio.gather(*(scheduler.submit(jobs[i], timeout=600) for i in indices))
            for i, out in zip(indices, outs):
                compare(out, refs[i], metrics["vs_shipped"], [wave, i])
                compare(out, isolated[i], metrics["vs_isolated"], [wave, i])
    finally:
        await scheduler.close()
    result = dict(
        **metrics, isolated_vs_shipped=isolated_reference, limits=BF16_LIMIT, metrics=scheduler.metrics()
    )
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert all(not m["outliers"] for m in (*metrics.values(), isolated_reference)), (
        "see per-request outliers in result JSON"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/content/ckpt/v1/ckpt_final.pt")
    p.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    p.add_argument("--image-root", default="/content/data/coco/train2014")
    p.add_argument("--group-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--single-vision", action="store_true")
    p.add_argument("--padded-shared", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(check(p.parse_args()))
