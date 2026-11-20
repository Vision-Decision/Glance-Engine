"""E0: diagnose the GPU box-readout equivalence failure of scan/test_scan_equivalence.py --real
(block_bidir/box/cached_vs_alone rel. err 0.685).

Hypothesis: the box readout picks a patch by argmax over centre logits; in bf16 a near-tie can flip the argmax between
two runs, and the box (relative to the chosen patch centre) jumps. Presence / IoU logits and the per-patch logits are
then still within bf16 noise. The test's metric (max abs diff / max |ref| over presence, box and IoU together) turns a
single flip into a large relative error.

For each (image, box query): cached scan, fresh scan and the reference (run_model_v1 on that image alone) ->
presence / IoU abs diffs, argmax agreement, box L1 split by argmax agreement, and the reference's top-1 minus top-2
centre-logit margin at every disagreement. --fp32 runs the same with fp32 weights and autocast disabled (if the flips
disappear the difference is numeric, not a logic bug).

python -m scan.e0_box_equiv --ckpt CKPT --out OUT.json [--fp32] [--n_img 48] [--Tvs 640,256]
"""
import argparse, contextlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--n_img", type=int, default=48)
ap.add_argument("--Tvs", default="640,256")
ap.add_argument("--n_random", type=int, default=6)
ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
ap.add_argument("--image_root", default="/content/data/coco/train2014")
a = ap.parse_args()

if a.fp32:      # disable every bf16 autocast region (scan engine, speed engine, reference forward)
    _orig = torch.autocast

    class _NoAC(_orig):
        def __init__(self, *args, **kw):
            kw["enabled"] = False
            super().__init__(*args, **kw)
    torch.autocast = _NoAC

from transformers import AutoProcessor
from dv.config import Config
from dv.data.mixed import ChatTok, collate_mixed
from dv.data.dataset import resize_dims
from dv.model.backbone import SharedPrefixLM
from dv.model.build import build, load_ckpt
from dv.model.v1forward import run_model_v1
from scan.engine import ScanEngine, make_item
from scan.data import load_images, QUERIES

dev = "cuda"
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg0 = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
net0, heads, _ = build(cfg0, dev)
load_ckpt(a.ckpt, net0, heads)
hf = net0.hf.eval()
if a.fp32:
    hf.float(); heads.float()
print("ckpt attn_mask:", cfg0.attn_mask, "fp32:", a.fp32, flush=True)

res = dict(ckpt=a.ckpt, fp32=a.fp32, attn_mask=cfg0.attn_mask, sets={})
rng = np.random.RandomState(0)
for Tv in [int(x) for x in a.Tvs.split(",")]:
    cfg = Config().update(dict(task="v1", attn_mask=cfg0.attn_mask, max_vis_tokens=Tv, grad_ckpt=False))
    net = SharedPrefixLM(hf, cfg).eval()
    tok = ChatTok(proc, cfg)
    eng = ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=False)
    real = [x["img"] for x in load_images(a.prep, a.image_root, a.n_img, Tv)]
    sizes = [(320, 240), (200, 300), (256, 256), (320, 240), (160, 480), (288, 224)][: a.n_random]
    rand = [Image.fromarray((rng.rand(H, W, 3) * 255).astype("uint8")) for W, H in sizes]
    rand = [im.resize(resize_dims(*im.size, Tv, 2.0), Image.BICUBIC) for im in rand]
    for kind, ims in (("real", real), ("random", rand)):
        t0 = time.time()
        bank = eng.encode(ims, chunk=8)
        rows = []
        for qi, qt in enumerate(QUERIES[:4]):
            q = dict(type="box", text=qt)
            oc, of = eng.query(bank, q), eng.fresh(ims, q)
            for i, im in enumerate(ims):
                b = collate_mixed([make_item(tok, proc, im, [q])], tok); b.pop("meta", None)
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.no_grad():
                    r = run_model_v1(net, heads, b, train=False)["box"]
                cl = r["center_logits"][0].float()
                top2 = torch.topk(cl, 2).values
                ref = dict(p=float(r["presence_logit"][0]), iou=float(r["iou_logit"][0]), idx=int(r["pred_idx"][0]),
                           box=r["box_pred"][0].float().cpu(), margin=float(top2[0] - top2[1]))
                for path, o in (("cached", oc), ("fresh", of)):
                    idx = int(o["center_idx"][i]) if "center_idx" in o else -1
                    rows.append(dict(set=kind, q=qi, i=i, path=path,
                                     dp=abs(float(o["presence_logit"][i]) - ref["p"]),
                                     dprob=abs(float(torch.sigmoid(o["presence_logit"][i])) - float(torch.sigmoid(torch.tensor(ref["p"])))),
                                     diou=abs(float(o["iou_logit"][i]) - ref["iou"]),
                                     agree=(idx == ref["idx"]) if idx >= 0 else None, idx_known=idx >= 0, margin=ref["margin"],
                                     dbox=float((o["box_pred"][i] - ref["box"]).abs().max()),
                                     dec_agree=(float(o["presence_logit"][i]) > 0) == (ref["p"] > 0)))
        summ = {}
        for path in ("cached", "fresh"):
            R = [x for x in rows if x["path"] == path]
            ag = [x for x in R if x["agree"]]
            dis = [x for x in R if x["agree"] is False]
            summ[path] = dict(
                n=len(R), presence_logit_absdiff_max=max(x["dp"] for x in R), presence_prob_absdiff_max=max(x["dprob"] for x in R),
                presence_prob_absdiff_mean=float(np.mean([x["dprob"] for x in R])), iou_logit_absdiff_max=max(x["diou"] for x in R),
                presence_decision_agree=float(np.mean([x["dec_agree"] for x in R])),
                argmax_agree=(float(np.mean([x["agree"] for x in R])) if R[0]["idx_known"] else None), idx_known=all(x["idx_known"] for x in R),
                box_absdiff_max_when_argmax_agrees=max([x["dbox"] for x in ag], default=None),
                box_absdiff_max_when_argmax_differs=max([x["dbox"] for x in dis], default=None),
                ref_margin_at_disagreements=sorted(round(x["margin"], 4) for x in dis),
                ref_margin_median_all=float(np.median([x["margin"] for x in R])))
        res["sets"][f"Tv{Tv}_{kind}"] = summ
        print(f"Tv{Tv}_{kind} ({time.time() - t0:.0f}s)", json.dumps(summ), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
print("E0_DONE", flush=True)
