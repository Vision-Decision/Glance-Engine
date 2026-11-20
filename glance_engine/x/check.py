"""GPU gates for the Glance-X backend: agreement with glancex's reference, and latency.

    python -m glance_engine.x.check --model Vision-Decision/Glance-X-Lite --out x_lite.json
    python -m glance_engine.x.check --model Vision-Decision/Glance-X-4B --images 'coco/*.jpg' --out x_4b.json

1. **exact**: engine with unmerged LoRA (``exact=True``) against ``glancex.GlanceX(fast=False)``, every trained
   direction, the same images and requests. Same weights and arithmetic order; BF16 kernels may still differ in the
   last bits, so the gate is on P(present) and box pixels.
2. **fast**: the default engine (merged LoRA) against the same reference.
3. **cache**: follow-ups on cached image ids against fresh answers, and the cached scan against fresh.
4. **latency**: warm medians for one image with Q requests, a cached follow-up, and one request over N cached images,
   for the engine and for glancex's fast path (merged LoRA, CUDA graphs) on the same inputs.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import time

import torch
from PIL import Image

REQUESTS = [
    "the giraffe on the right", "the animal on the left", "a person", "the tallest giraffe", "a tree", "a car",
    "the giraffe in the middle", "左边的长颈鹿", "the sky", "a giraffe eating leaves", "a dog", "the fence",
    "grass", "a building", "the smallest animal", "a red umbrella",
]


def sync():
    torch.cuda.synchronize()


def timed(fn, warm=3, reps=20):
    for _ in range(warm):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(out), 2)


def compare(engine_rows, ref_rows, sizes):
    """Max |dP(present)|, max box-corner error in pixels, present-decision agreement, mean |dP(IoU)|."""
    dp, dpx, agree, diou = [], [], 0, []
    for (a, r), (w, h) in zip(zip(engine_rows, ref_rows), sizes):
        dp.append(abs(a.p_present - r.present_prob))
        diou.append(abs(a.p_iou50 - r.iou_prob))
        dpx.append(max(abs(x - y) for x, y in zip(a.box_px, r.box)))
        agree += (a.p_present >= 0.5) == (r.present_prob >= 0.5)
    n = len(dp)
    return dict(n=n, max_dp=round(max(dp), 5), mean_dp=round(sum(dp) / n, 6), max_box_px=round(max(dpx), 2),
                median_box_px=round(statistics.median(dpx), 3), present_agree=f"{agree}/{n}",
                mean_diou=round(sum(diou) / n, 6))


def run_reference(ref, images, requests, direction):
    rows = []
    for im in images:
        rows += ref.ask(im, requests, mode=direction)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Vision-Decision/Glance-X-Lite")
    ap.add_argument("--images", default="demo/data/search_large/*.jpg")
    ap.add_argument("--scan_n", type=int, default=256)
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip_latency", action="store_true")
    a = ap.parse_args(argv)

    from glancex import GlanceX

    from glance_engine import Box, Glance

    paths = sorted(glob.glob(a.images))
    images = [Image.open(p).convert("RGB") for p in paths]
    requests = REQUESTS[: a.q]
    report = dict(model=a.model, images=len(images), requests=len(requests), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__)

    ref = GlanceX.from_pretrained(a.model, fast=False)
    exact = Glance.from_pretrained(a.model, exact=True)
    sizes = [im.size for im in images for _ in requests]
    report["directions"] = list(exact.engine.directions)
    report["bidir_k"] = exact.engine.K
    for d in exact.engine.directions:
        ref_rows = run_reference(ref, images, requests, d)
        rows = [x for im in images for x in exact.ask(im, [Box(t) for t in requests], cache=False, direction=d)]
        report[f"exact_{d}"] = compare(rows, ref_rows, sizes)
        print(d, "exact", report[f"exact_{d}"], flush=True)
    del exact
    torch.cuda.empty_cache()

    fast = Glance.from_pretrained(a.model)
    for d in fast.engine.directions:
        ref_rows = run_reference(ref, images, requests, d)
        rows = [x for im in images for x in fast.ask(im, [Box(t) for t in requests], cache=False, direction=d)]
        report[f"fast_{d}"] = compare(rows, ref_rows, sizes)
        print(d, "fast", report[f"fast_{d}"], flush=True)
    del ref
    torch.cuda.empty_cache()

    # graphs: the CUDA-graph replay (bucketed shapes) against the eager engine on the same weights
    fast.engine.graphs = False
    eager = [x for im in images for x in fast.ask(im, [Box(t) for t in requests], cache=False)]
    fast.engine.graphs = True
    replay = [x for im in images for x in fast.ask(im, [Box(t) for t in requests], cache=False)]
    report["graphs_vs_eager_max_dp"] = round(max(abs(x.p_present - y.p_present) for x, y in zip(eager, replay)), 6)
    report["graphs_vs_eager_max_box_px"] = round(max(max(abs(u - v) for u, v in zip(x.box_px, y.box_px))
                                                     for x, y in zip(eager, replay)), 3)
    print("graphs", report["graphs_vs_eager_max_dp"], report["graphs_vs_eager_max_box_px"], flush=True)

    # cache: cached follow-ups and cached scans equal fresh answers
    qs = [Box(t) for t in requests]
    ids = [fast.add_image(im) for im in images]
    cached = [x for i in ids for x in fast.ask(i, qs)]
    fresh = [x for im in images for x in fast.ask(im, qs, cache=False)]
    report["cache_followup_max_dp"] = round(max(abs(c.p_present - f.p_present) for c, f in zip(cached, fresh)), 6)
    scanner = fast.scanner(max_vis=fast.engine.max_vis)
    c1 = scanner.scan(requests[0], images)
    c2 = scanner.scan(requests[0], images, mode="fresh")
    report["cache_scan_max_dp"] = round(max(abs(x["p_present"] - y["p_present"])
                                            for x, y in zip(c1["images"], c2["images"])), 6)
    print("cache", report["cache_followup_max_dp"], report["cache_scan_max_dp"], flush=True)

    if not a.skip_latency:
        im = images[0]
        lat = {}
        q32 = [Box(REQUESTS[i % len(REQUESTS)] + ("" if i < len(REQUESTS) else f" {i}")) for i in range(32)]
        lat["engine_one_image_q1"] = timed(lambda: fast.ask(im, q32[:1], cache=False))
        lat["engine_one_image_q32"] = timed(lambda: fast.ask(im, q32, cache=False))
        image_id = fast.add_image(im)
        lat["engine_followup_q32"] = timed(lambda: fast.ask(image_id, q32))
        bank_images = [images[i % len(images)] for i in range(a.scan_n)]
        scanner = fast.scanner(max_vis=fast.engine.max_vis)
        t = time.perf_counter()
        scanner.scan(requests[0], bank_images)
        sync()
        lat[f"engine_encode_and_first_scan_{a.scan_n}"] = round((time.perf_counter() - t) * 1e3, 1)
        lat[f"engine_cached_scan_{a.scan_n}"] = timed(lambda: scanner.scan(requests[1], bank_images), reps=10)
        if "text_first" in fast.engine.directions:
            few = bank_images[:64]
            lat["engine_text_first_64"] = timed(lambda: scanner.scan(requests[1], few, mode="text_first"), reps=5)
        del fast
        torch.cuda.empty_cache()
        gx = GlanceX.from_pretrained(a.model)  # glancex fast path: merged LoRA + CUDA graphs
        texts = [q.text for q in q32]
        lat["glancex_one_image_q1"] = timed(lambda: gx.ask(im, texts[:1]))
        lat["glancex_one_image_q32"] = timed(lambda: gx.ask(im, texts))
        bank = gx.encode(bank_images)
        lat[f"glancex_cached_scan_{a.scan_n}"] = timed(lambda: gx.search(requests[1], bank), reps=10)
        if float(gx.cfg.mode_probs.split(",")[1]) > 0:
            lat["glancex_text_first_64"] = timed(lambda: gx.search(requests[1], bank_images[:64]), reps=5)
        report["latency_ms"] = lat
        print(lat, flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
