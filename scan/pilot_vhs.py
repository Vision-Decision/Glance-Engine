"""Phase-2 pilot (P1): Visual Haystacks single-needle with the current checkpoint, cached scan mode, no training.

python -m scan.pilot_vhs --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json [--Ns 2,3,5,10,20,50,100] [--tv 256]

Question: 'For the image with {anchor}, is there {target}?'  N images, exactly one contains the anchor.
Every haystack is encoded once (bank); two queries run over all N images in one branch forward each:
  P_a(i) = presence of the anchor,  P_t(i) = presence of the target  (Glance box head; same-base yes/no as baseline)
Answer: P_t at argmax_i P_a  (hard), and sum_i softmax-weight(P_a) P_t (soft).
Reported vs N: accuracy, anchor retrieval error (argmax is not the needle image), and the accuracy given the right
anchor -- the split separates the multiple-testing failure (wrong anchor as N grows) from per-image perception.
Images: COCO train2017 fetched on the VM. Caveat: Glance v1 was trained on RefCOCO (COCO train2014 images, which
overlap train2017), so the images may have been seen in training (not these questions).
"""
import argparse, json, os, urllib.request, concurrent.futures as cf
import numpy as np
import torch
from PIL import Image

from dv.data.dataset import resize_dims
from scan.pilot_charades import load_glance, load_base


def fetch(paths, root):
    def get(p):
        d = f"{root}/{p}"
        if not os.path.exists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            for _ in range(3):
                try:
                    urllib.request.urlretrieve(f"http://images.cocodataset.org/{p}", d); return
                except Exception:
                    pass
    with cf.ThreadPoolExecutor(32) as ex:
        list(ex.map(get, sorted(set(paths))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ns", default="2,3,5,10,20,50,100")
    ap.add_argument("--tv", type=int, default=256)
    ap.add_argument("--root", default="/content/data/coco2017")
    ap.add_argument("--base", action="store_true", help="also the same-base yes/no scanner")
    ap.add_argument("--chunk", type=int, default=64, help="images per bank (memory bound for large N)")
    ap.add_argument("--max_q", type=int, default=0, help="subsample questions per N")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    G, cfg = load_glance(a.ckpt)
    Bm = load_base(cfg) if a.base else None
    res = dict(args=vars(a), per_N={}, rows={})
    for N in [int(x) for x in a.Ns.split(",")]:
        d = json.load(open(hf_hub_download("tsunghanwu/visual_haystacks", f"single_needle/VHs_small/visual_haystack_{N}.json",
                                           repo_type="dataset")))
        if a.max_q and len(d) > a.max_q:
            d = [d[i] for i in sorted(np.random.RandomState(0).choice(len(d), a.max_q, replace=False))]
        fetch([p for x in d for p in x["pos_image"] + x["neg_image"]], a.root)
        rows = []
        for x in d:
            paths = x["pos_image"] + x["neg_image"]          # index 0 = needle image
            ims = []
            for p in paths:
                im = Image.open(f"{a.root}/{p}").convert("RGB")
                ims.append(im.resize(resize_dims(*im.size, a.tv, 2.0), Image.BICUBIC))
            gt = x["conversations"][1]["value"].strip().lower().startswith("yes")
            r = dict(id=x["id"], gt=gt)
            for name, eng in (("glance", G),) + ((("base", Bm),) if Bm else ()):
                PA, PT = [], []
                for c0 in range(0, len(ims), a.chunk):     # banks per chunk of images (branches are independent: exact)
                    bank = eng.encode(ims[c0:c0 + a.chunk], chunk=32)
                    if name == "glance":
                        PA.append(eng.query(bank, dict(type="box", text=x["needle"]))["presence_logit"].numpy())
                        PT.append(eng.query(bank, dict(type="box", text=x["target"]))["presence_logit"].numpy())
                    else:
                        f = lambda t: eng.query(bank, dict(type="noul", instr=f"Is there {t} in the image?"))["typed_logits"]
                        la, lt = f(x["needle"]), f(x["target"])
                        PA.append((la[:, 1] - la[:, 0]).numpy()); PT.append((lt[:, 1] - lt[:, 0]).numpy())
                    del bank
                    eng.free_graphs()
                torch.cuda.empty_cache()
                pa, pt = np.concatenate(PA), np.concatenate(PT)
                i = int(np.argmax(pa))
                w = np.exp(pa - pa.max()); w /= w.sum()
                r[name] = dict(anchor_ok=i == 0, hard=bool(pt[i] > 0), soft=bool((w * (1 / (1 + np.exp(-pt)))).sum() > 0.5),
                               pa=pa.round(3).tolist(), pt=pt.round(3).tolist())
            rows.append(r)
        summ = {}
        for name in ("glance", "base"):
            rr = [r for r in rows if name in r]
            if not rr:
                continue
            ok = [r for r in rr if r[name]["anchor_ok"]]
            summ[name] = dict(acc_hard=float(np.mean([r[name]["hard"] == r["gt"] for r in rr])),
                              acc_soft=float(np.mean([r[name]["soft"] == r["gt"] for r in rr])),
                              anchor_error=float(1 - len(ok) / len(rr)),
                              acc_given_right_anchor=float(np.mean([r[name]["hard"] == r["gt"] for r in ok])) if ok else None,
                              n=len(rr))
        res["per_N"][N], res["rows"][N] = summ, rows
        print(N, json.dumps(summ), flush=True)
        json.dump(res, open(a.out, "w"))
    print("VHS_DONE", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
