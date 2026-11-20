"""Glance side of the 256-image search: encode a bank of 256 COCO images once, then answer box queries over all of
them in one cached-scan call, with a calibrated set-level "none of these" rule.

PYTHONPATH=. python demo/search_glance.py --ckpt /content/ckpt/v1/ckpt_final.pt --pick demo/data/coco_pick.json \
    --image_dir /content/demo_imgs --out OUT.json

Engine: scan.api.Scanner (merged LoRA, fused ops, CUDA graphs, bf16) at <= 640 visual tokens per image, the
CEILING.md "Tv=640, N=256, box" workload. Timing (as scan/bench_scan.py time_cached): from the query text to all 256
answers on the host; the bank is built beforehand and its build time is reported separately.
Calibration: the absent categories of the pick (no bank image contains them) are split; the first n_cal give the
negative scans for Scanner.calibrate (max-statistic "none" rule at alpha, BH selection at q); the rest are held-out
absent test queries, never seen by calibration.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image

import speed.engine as SE
from scan.api import Scanner
from scan.engine import _sync

SE.GRAPH_MAX_TOKENS = 1 << 15          # as scan/bench_scan.py: graph every scan shape on the 96 GB G4
SE.GRAPH_MAX_PATCHES = 1 << 17


def article(c):
    return ("an " if c[0] in "aeiou" else "a ") + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cal", type=int, default=34)
    ap.add_argument("--n_calls", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    pick = json.load(open(a.pick))["search"]
    bank_recs = pick["bank"]
    images = [Image.open(os.path.join(a.image_dir, r["file"])).convert("RGB") for r in bank_recs]
    sc = Scanner.load(a.ckpt, max_vis=640, alpha=a.alpha, q=0.1)
    eng = sc.eng
    prepped = sc._prep(images)
    _sync(); t0 = time.perf_counter()
    bank = eng.encode(prepped, chunk=16)
    _sync()
    enc_ms = (time.perf_counter() - t0) * 1e3

    held = pick["held"]
    cal_q = [article(c) for c in held[:a.n_cal]]
    test_absent = [article(c) for c in held[a.n_cal:]]
    target = article(pick["target"])
    all_q = [target] + test_absent + cal_q
    Q = {t: dict(type="box", text=t) for t in all_q}

    def run(t, T=None):
        return eng.query(bank, Q[t], timings=T if T is not None else {})

    for t in all_q:                     # warm-up: capture one graph per question length
        run(t); run(t)
    neg = [run(t)["presence_logit"].numpy().astype(np.float64) for t in cal_q]
    sc.calibrate(neg)
    thr = getattr(sc.none_rule, "thr", None) or getattr(sc.none_rule, "tau", None)

    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, n_images=len(images), max_vis=640,
               bank_encode_ms_total=enc_ms, bank_encode_ms_per_image=enc_ms / len(images),
               bank_MB_per_image=bank.nbytes() / len(images) / 2**20,
               calibration=dict(alpha=a.alpha, bh_q=0.1, negative_queries=cal_q,
                                none_rule=repr(sc.none_rule), threshold=None if thr is None else float(thr)),
               queries={})
    gt = {pick["target"]: [any(o["cat"] == pick["target"] for o in r["objects"]) for r in bank_recs]}
    for t in [target] + test_absent:
        calls = []
        for _ in range(a.n_calls):
            T = {}
            _sync(); s0 = time.perf_counter()
            out = run(t, T)
            calls.append(dict(T, wall=(time.perf_counter() - s0) * 1e3))
        s = out["presence_logit"].numpy().astype(np.float64)
        p = 1 / (1 + np.exp(-s))
        sel = [int(i) for i in sc.select_rule(s)]
        cat = t.split(" ", 1)[1]
        truth = gt.get(cat, [False] * len(images))
        res["queries"][t] = dict(
            total_ms_median=float(np.median([c["total"] for c in calls])),
            total_ms_p10=float(np.percentile([c["total"] for c in calls], 10)),
            total_ms_p90=float(np.percentile([c["total"] for c in calls], 90)), calls=calls,
            none_conformal=bool(not sc.none_rule.alarm(s)), selected=sel, naive_any=bool((p > .5).any()),
            naive_count=int((p > .5).sum()), truth_present=[i for i, x in enumerate(truth) if x],
            presence_logit=[round(float(x), 4) for x in s],
            p_present=[round(float(x), 5) for x in p],
            box=[[round(float(v), 5) for v in b] for b in out["box_pred"]])
        q = res["queries"][t]
        print(t, round(q["total_ms_median"], 2), "none" if q["none_conformal"] else f"found {sel}",
              "truth", q["truth_present"], "naive", q["naive_count"], flush=True)

    # back-to-back: every query once in a row, wall-clock completion times (what a user firing queries would see)
    seq, _ = [], _sync()
    s0 = time.perf_counter()
    for t in [target] + test_absent + cal_q:
        run(t)
        seq.append(dict(query=t, done_ms=(time.perf_counter() - s0) * 1e3))
    res["back_to_back"] = seq
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
