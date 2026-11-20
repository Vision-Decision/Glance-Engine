"""Storage-only perturbations on the cached scan fixtures and labeled task data."""

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scan.bank import FORMATS, StoredBank
from scan.bench_scan import Bench, make_clips
from scan.data import load_images
from scan.pilot_analyze import none_of_these


def metrics(ref, arr):
    ref, arr = np.asarray(ref), np.asarray(arr)
    delta = np.abs(arr - ref)
    return {
        "n": ref.size,
        "flips": int(np.sum((ref >= 0.5) != (arr >= 0.5))),
        "mean": float(delta.mean()),
        "p50": float(np.median(delta)),
        "p95": float(np.quantile(delta, 0.95)),
        "p99": float(np.quantile(delta, 0.99)),
        "max": float(delta.max()),
    }


def area(pos, neg):
    # Ranking with half credit for ties.
    neg = np.sort(neg)
    return float(
        ((np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")) / (2 * len(neg))).mean()
    )


def pack(bank, mode):
    return bank if mode == "legacy" else StoredBank.pack(bank, mode)


def fixtures(bench, a):
    imgs = lambda tv: [x["img"] for x in load_images(a.prep, a.image_root, 24, tv, frame=tv != 640)]
    qs = [
        {"type": "box", "text": "person standing"},
        {"type": "box", "text": "the dog on the left"},
        {"type": "noul", "instr": "Is there a person in the image?"},
    ]
    sets = [
        ("img640", imgs(640), qs),
        ("frame256", imgs(256), qs),
        (
            "clip",
            make_clips(a, 24),
            [{"type": "noul", "instr": "Is a person walking in this clip?"}],
        ),
    ]

    def prob(out, q):
        if q["type"] == "box":
            return out["presence_logit"].sigmoid().numpy()
        return out["typed_logits"].softmax(-1)[:, 1].numpy()

    shipped = {}
    for shape, ims, questions in sets:
        for j, q in enumerate(questions):
            out = bench.reference(ims, q, clip_kw={"dt": 0.5, "max_vis": 96} if shape == "clip" else None)
            shipped[shape, j] = prob(out, q)
    eng = bench.to_engine()
    res = {}
    for shape, ims, questions in sets:
        base = eng.encode(ims, chunk=16, bank_format="legacy")
        refs = [prob(eng.query(base, q), q) for q in questions]
        for mode in FORMATS[1:]:
            bank = pack(base, mode)
            all_equal = []
            rows = []
            for j, q in enumerate(questions):
                out = eng.query(bank, q)
                refout = eng.query(base, q)
                all_equal.append(all(torch.equal(out[k], refout[k]) for k in out))
                val = prob(out, q)
                rows.append(
                    {
                        "stored": metrics(refs[j], val),
                        "shipped": metrics(shipped[shape, j], val),
                    }
                )
            res[shape + "/" + mode] = {
                "questions": rows,
                "bit_equal": all_equal,
                "bytes": bank.components(),
                "n": bank.N,
            }
            del bank
        del base
        save(a, res)
    return res


def presence(eng, a):
    items = load_images("data/prep/eval_grefcoco_val.jsonl", a.image_root, 60, 640)
    labels, scores = [], defaultdict(list)
    eng.graphs = False
    for row in items:
        base = eng.encode([row["img"]], bank_format="legacy")
        qs = row["qs"][:8]
        labels.extend(bool(q["present"]) for q in qs)
        for mode in FORMATS:
            bank = pack(base, mode)
            scores[mode].extend(
                float(eng.query(bank, {"type": "box", "text": q["text"]})["presence_logit"].sigmoid()[0])
                for q in qs
            )
            del bank
    y = np.asarray(labels)
    return {
        "n_images": len(items),
        "positive": int(y.sum()),
        "negative": int((~y).sum()),
        "formats": {
            mode: dict(
                auroc=area(np.array(p)[y], np.array(p)[~y]),
                **metrics(scores["legacy"], p),
            )
            for mode, p in scores.items()
        },
    }


def video(eng, a):
    from huggingface_hub import hf_hub_download

    from scan.pilot_analyze import build_sets
    from scan.setlevel import MaxStat

    manifest = hf_hub_download(
        "Vision-Decision/collection-scan-data",
        "e4/e4_charades_hard.json.gz",
        repo_type="dataset",
    )
    with gzip.open(manifest, "rt") as f:
        scans = json.load(f)["scans"]
    groups = defaultdict(list)
    for s in scans:
        groups[s["qid"]].append(s)
    rng = random.Random(8)
    eligible = [ss for ss in groups.values() if any(s["pos"] for s in ss) and sum(not s["pos"] for s in ss) >= 2]
    selected = []
    for ss in rng.sample(eligible, min(64, len(eligible))):
        selected += [next(s for s in ss if s["pos"])] + rng.sample([s for s in ss if not s["pos"]], 2)
    by_vid = defaultdict(list)
    for s in selected:
        s["scores"] = {}
        by_vid[s["vid"]].append(s)
    eng.graphs = False
    root = Path("/content/charades_pilot2/frames")
    for i, (vid, subset) in enumerate(by_vid.items()):
        frames = [Image.open(p).convert("RGB") for p in sorted((root / vid).glob("*.jpg"))]
        clips = [frames[j : j + 4] for j in range(0, len(frames) - 3, 2)]
        base = eng.encode(clips, chunk=64, bank_format="legacy")
        for mode in FORMATS:
            bank = pack(base, mode)
            for s in subset:
                logits = eng.query(bank, {"type": "noul", "instr": s["text"], "block": "short"})["typed_logits"]
                s["scores"][mode] = (logits[:, 1] - logits[:, 0]).tolist()
            del bank
        if i % 20 == 0:
            print("Charades videos", i + 1, "/", len(by_vid), flush=True)
    res = {}
    for mode in FORMATS:

        def probs(m):
            logits = np.concatenate([s["scores"][m] for s in selected])
            return 1 / (1 + np.exp(-logits))

        neg = np.concatenate([s["scores"][mode] for s in selected if not s["pos"]])
        pos = np.concatenate([np.array(s["scores"][mode])[s["labels"]] for s in selected if s["pos"]])
        res[mode] = {
            "clip_auroc": area(pos, neg),
            "delta": metrics(probs("legacy"), probs(mode)),
            "setlevel": none_of_these(selected, mode, [8, 16, 32], 0.1, 50),
        }
        # A deployment might retain the reference calibration after changing bank precision.
        frozen = {}
        for n in (8, 16, 32):
            ref = build_sets(selected, "legacy", n, np.random.default_rng(n))
            cur = build_sets(selected, mode, n, np.random.default_rng(n))
            rates = []
            for sp in range(50):
                ids = np.random.default_rng(1000 + sp).permutation(sorted(ref))
                cal = [x for q in ids[: len(ids) // 2] for x in ref[q][0]]
                test = [x for q in ids[len(ids) // 2 :] for x in cur[q][0]]
                if len(cal) >= 20 and test:
                    rule = MaxStat(0.1).fit(cal)
                    rates.append(np.mean([rule.alarm(x) for x in test]))
            frozen[n] = float(np.mean(rates)) if rates else None
        res[mode]["frozen_maxstat_fa"] = frozen
    return {
        "n_videos": len(by_vid),
        "n_scans": len(selected),
        "formats": res,
        "manifest": [(s["vid"], s["qid"], s["pos"]) for s in selected],
    }


def save(a, result):
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    p.add_argument("--image_root", default="/content/data/coco/train2014")
    args = p.parse_args()
    with torch.inference_mode():
        b = Bench(args)
        if args.stage == "fixtures":
            save(args, fixtures(b, args))
        else:
            save(
                args,
                {"presence": presence, "video": video}[args.stage](b.to_engine(), args),
            )
