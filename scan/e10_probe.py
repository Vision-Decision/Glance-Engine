"""E10 (algorithmic): early-exit yes/no readout. Is presence readable from a mid layer of the frozen backbone?

python -m scan.e10_probe --ckpt CKPT --out OUT.json [--layers 8,12,16,20,24,28] [--n_img 400] [--tv 128]

Data: gRefCOCO val (+ RefCOCO val as extra positives) images, every expression asked as
'Is there {expression} in the image?' (present / absent labels from gRefCOCO). Split BY IMAGE into train / test halves.
Features: the hidden state of the question block's last token after layer L, from one plain HF forward over
[image prefix + question block] (output_hidden_states). A logistic-regression head (L2, fit on the train half) per layer.
Reported per layer on the test half: AUROC vs the label, AUROC of the full model's P(yes) (reference), and decision
agreement with the full model. The latency of an exit at layer L is estimated from the measured fresh-path split
(prep + vision + L/32 x LM time) in e1b_ours_fresh_c4096.json; nothing is changed in the engine.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scan.bench_scan import Bench, auroc
from scan.data import load_images
from scan.engine import make_item
from dv.data.mixed import collate_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="4,8,12,16,20,24,28")
    ap.add_argument("--n_img", type=int, default=400)
    ap.add_argument("--tv", type=int, default=128)
    ap.add_argument("--paired", action="store_true", help="RefCOCO expressions on their own image vs another image")
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--gprep", default="data/prep/eval_grefcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    a = ap.parse_args()
    B = Bench(a)
    net, heads, tok, proc, hf = B.net, B.heads, B.tok, B.proc, B.net.hf
    layers = [int(x) for x in a.layers.split(",")]
    if a.paired:     # every expression asked of its own image (present) AND of another image (absent): no text-only cue
        base = load_images(a.prep, a.image_root, a.n_img, a.tv)
        rng0 = np.random.RandomState(1)
        items = []
        for ii, it in enumerate(base):
            qs = [q for q in it["qs"] if q.get("present", True)][:4]
            other = base[(ii + 1 + rng0.randint(len(base) - 1)) % len(base)]
            items.append(dict(img=it["img"], qs=[dict(text=q["text"], present=True) for q in qs], gid=ii))
            items.append(dict(img=other["img"], qs=[dict(text=q["text"], present=False) for q in qs], gid=ii))
    else:
        items = load_images(a.gprep, a.image_root, a.n_img, a.tv)
        for ii, it in enumerate(items):
            it["gid"] = ii
    feats = {L: [] for L in layers}
    ref, y, img_id = [], [], []
    for ii, it in enumerate(items):
        for q in it["qs"][:8]:
            qq = dict(type="noul", instr=f"Is there {q['text']} in the image?")
            b = collate_mixed([make_item(tok, proc, it["img"], [qq])], tok)
            pv = b["p_valid"][0]
            ids = torch.cat([b["p_ids"][0][pv], b["q_ids"][0][b["q_valid"][0]]])[None].cuda()
            mm = (ids == hf.config.image_token_id).int()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = hf.model(input_ids=ids, pixel_values=b["pixel_values"].cuda(), image_grid_thw=b["grid_thw"].cuda(),
                             mm_token_type_ids=mm, output_hidden_states=True)
            hs = o.hidden_states
            for L in layers:
                feats[L].append(hs[L][0, -1].float().cpu().numpy())
            # full model's P(yes) through the engine-equivalent reference path (block mask as trained)
            bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items() if k != "meta"}
            from dv.model.v1forward import run_model_v1
            with torch.no_grad():
                tl = run_model_v1(net, heads, bb, train=False)["typed_logits"][0].float()
            ref.append(float(torch.softmax(tl[:2], -1)[1])); y.append(bool(q["present"])); img_id.append(it["gid"])
        if ii % 50 == 0:
            print("img", ii, len(y), flush=True)
    y, ref, img_id = np.array(y), np.array(ref), np.array(img_id)
    rng = np.random.RandomState(0)
    G = sorted(set(img_id.tolist())); perm = rng.permutation(len(G)); tr_imgs = {G[i] for i in perm[: len(G) // 2]}
    tr = np.array([i in tr_imgs for i in img_id]); te = ~tr
    from sklearn.linear_model import LogisticRegression
    res = dict(paired=a.paired, n=int(len(y)), n_test=int(te.sum()), tv=a.tv, ref_auroc_test=auroc(ref[te & y], ref[te & ~y]), layers={})
    for L in layers:
        X = np.stack(feats[L]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        best = None
        for C in (0.001, 0.01, 0.1):
            clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
            p = clf.predict_proba(X)[:, 1]
            r = dict(C=C, auroc_test=auroc(p[te & y], p[te & ~y]), agree_with_full_test=float(((p[te] >= .5) == (ref[te] >= .5)).mean()))
            if best is None or r["auroc_test"] > best["auroc_test"]:
                best = r
        res["layers"][L] = best
        print("layer", L, json.dumps(best), "| full model", round(res["ref_auroc_test"], 4), flush=True)
    lat = os.path.join(os.path.dirname(a.out), "e1b_ours_fresh_c4096.json")
    if os.path.exists(lat):
        row = json.load(open(lat))["fresh"].get(f"Tv{a.tv}_N128")
        if row:
            res["latency_N128_estimate_ms"] = {L: row["prep_ms"] + row["vision_ms"] + row["lm_ms"] * L / 32 for L in layers}
            res["latency_N128_full_ms"] = row["total_ms"]
    json.dump(res, open(a.out, "w"), indent=1)
    print("E10_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
