"""Scan benchmark on one GPU: exactness vs per-image runs, cached / fresh / query-first latency, accuracy vs tokens/image.

python -m scan.bench_scan --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json [--parts equiv,cached,fresh,qfirst,acc,fp8]

Timing per call (median over n_calls after an untimed warm-up that captures CUDA graphs):
  cached  from the query text to all N answers on the host (tokenise, H2D, branch forward, heads, D2H). The bank (each
          image's prefix state) is built beforehand; its build time per image is reported separately.
  fresh   from N resized PIL frames + the query to all N answers on the host (image processor, vision tower, packed
          prefill, branches, heads, D2H). JPEG decode / resize excluded (identical for every system).
Workloads: RefCOCO-val COCO images (as speed/), either at their own aspect (cached @640, accuracy) or all resized to
one 4:3 frame size per token budget ("frames": what a video scan sees; lets the vision tower batch and graph).
Exactness: bf16 engine outputs vs the shipped reference path (dv.model, PEFT LoRA unmerged, each image alone).
"""
import argparse, gc, json, math, os, time
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.data.dataset import resize_dims
from dv.data.mixed import ChatTok, collate_mixed
from dv.model.build import build, load_ckpt
from dv.model.v1forward import run_model_v1
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import speed.engine as SE
from speed.engine import merge_lora_, fuse_weights_, fp8_language_model_
from scan.engine import ScanEngine, make_item, make_clip_item
from scan.data import load_images, QUERIES

SE.GRAPH_MAX_TOKENS = 1 << 15           # this GPU has 96 GB: graph every scan shape (large calls were eager in speed/)
SE.GRAPH_MAX_PATCHES = 1 << 17


def mem(tag):
    print(f"[mem] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB", flush=True)


def med(xs):
    return float(np.median(xs))


class Bench:
    def __init__(self, a):
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
        self.cfg = cfg
        self.w12 = getattr(a, "w12", None)
        self.fresh_varlen = getattr(a, "fresh_impl", "varlen") == "varlen"
        self.lm_prefill = getattr(a, "lm_prefill", "fused")
        self.vision_prep = getattr(a, "vision_prep", "fused")
        net, heads, proc = build(cfg, "cuda")
        load_ckpt(a.ckpt, net, heads)
        net.eval(); heads.eval()
        self.net, self.heads, self.proc, self.tok = net, heads, proc, ChatTok(proc, cfg)
        self.eng = None

    def reference(self, images, q, clip_kw=None):
        """shipped path, each (image or clip, question) alone"""
        rows = []
        for im in images:
            it = make_clip_item(self.tok, self.proc, im, [q], **clip_kw) if clip_kw else make_item(self.tok, self.proc, im, [q])
            b = collate_mixed([it], self.tok); b.pop("meta", None)
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad():
                r = run_model_v1(self.net, self.heads, b, train=False)
            if q["type"] == "box":
                rows.append(dict(presence_logit=r["box"]["presence_logit"].float().cpu(), box_pred=r["box"]["box_pred"].float().cpu(),
                                 iou_logit=r["box"]["iou_logit"].float().cpu()))
            else:
                rows.append(dict(typed_logits=r["typed_logits"].float().cpu()))
        return {k: torch.cat([r[k] for r in rows]) for k in rows[0]}

    def to_engine(self, fp8=False):
        if self.eng is None:
            self.n_merged = merge_lora_(self.net.hf)
            fuse_weights_(self.net.lm, self.net.layer_types, keep_ab_separate=fp8)
            if self.w12:
                from speed.w12 import install
                install(self.net.lm, self.w12)
        if fp8:
            self.n_fp8 = fp8_language_model_(self.net.lm)
        self.eng = None; gc.collect(); torch.cuda.empty_cache()
        self.eng = ScanEngine(self.net, self.heads, self.tok, self.proc, graphs=True, fast_ops=True, fresh_varlen=self.fresh_varlen,
                              vision_prep=self.vision_prep, lm_prefill=self.lm_prefill != "reference")
        if self.lm_prefill != "fused":
            self.eng.flash_gdn = None
        if self.lm_prefill == "pointwise":
            self.eng.native_gate_up = None
        elif self.lm_prefill == "direct" and self.eng.native_gate_up is not None:
            from speed.lm_prefill import gate_up_direct
            self.eng.native_gate_up = gate_up_direct
        return self.eng


def diff(out, ref):
    d = {}
    if "presence_logit" in ref:
        p, pr = torch.sigmoid(out["presence_logit"]), torch.sigmoid(ref["presence_logit"])
        d.update(p_absdiff_max=float((p - pr).abs().max()), p_absdiff_mean=float((p - pr).abs().mean()),
                 decision_agree=float(((p >= .5) == (pr >= .5)).float().mean()),
                 box_absdiff_max=float((out["box_pred"] - ref["box_pred"]).abs().max()),
                 box_absdiff_mean=float((out["box_pred"] - ref["box_pred"]).abs().mean()),
                 piou_absdiff_max=float((torch.sigmoid(out["iou_logit"]) - torch.sigmoid(ref["iou_logit"])).abs().max()))
    else:
        tl, tr = out["typed_logits"][:, :ref["typed_logits"].shape[1]], ref["typed_logits"]
        p, pr = tl.softmax(-1), tr.softmax(-1)
        d.update(prob_absdiff_max=float((p - pr).abs().max()), decision_agree=float((p.argmax(-1) == pr.argmax(-1)).float().mean()))
    return d


def make_clips(a, n, frames_per_clip=4, max_vis=96):
    """Clips of `frames_per_clip` frames (4 frames = 2 temporal groups, as vdm1's 2-s clips at 1 s spacing would be
    at dt=0.5) built from 640x480-resized COCO frames; content does not matter for latency."""
    fr = [x["img"] for x in load_images(a.prep, a.image_root, max(64, frames_per_clip * 8), 640, frame=True)]
    return [[fr[(c * 3 + k) % len(fr)] for k in range(frames_per_clip)] for c in range(n)]


def part_equiv(B, a, res):
    clips = make_clips(a, a.n_equiv)
    cq = dict(type="noul", instr="Is a person walking in this clip?")
    ckw = dict(dt=0.5, max_vis=96)
    ref_clip = B.reference(clips, cq, clip_kw=ckw)
    imgs = [x["img"] for x in load_images(a.prep, a.image_root, a.n_equiv, 640)]
    frames = [x["img"] for x in load_images(a.prep, a.image_root, a.n_equiv, 256, frame=True)]
    qs = [dict(type="box", text="person standing"), dict(type="box", text="the dog on the left"),
          dict(type="noul", instr="Is there a person in the image?")]
    refs = {(s, i): B.reference(ims, q) for s, ims in (("img640", imgs), ("frame256", frames)) for i, q in enumerate(qs)}
    eng = B.to_engine()
    out = {}
    for s, ims in (("img640", imgs), ("frame256", frames)):
        bank = eng.encode(ims, chunk=16)
        for i, q in enumerate(qs):
            out[f"{s}/q{i}/cached"] = diff(eng.query(bank, q), refs[(s, i)])
            out[f"{s}/q{i}/fresh"] = diff(eng.fresh(ims, q), refs[(s, i)])
            out[f"{s}/q{i}/separate_engine"] = diff(eng.separate(ims, q), refs[(s, i)])
        del bank
    eng.dt, eng.clip_max_vis = 0.5, 96
    cb = eng.encode(clips, chunk=16)
    out["clip4x96/noul/cached"] = diff(eng.query(cb, cq), ref_clip)
    out["clip4x96/noul/fresh"] = diff(eng.fresh(clips, cq), ref_clip)
    del cb
    eng.free_graphs()
    res["equiv"] = out
    json.dump(res, open(a.out, "w"), indent=1)
    mem("after equiv")
    print("equiv", json.dumps(out, indent=0), flush=True)


def time_cached(eng, bank, n_calls, kind="box"):
    mk = {"box": lambda t: dict(type="box", text=t),
          "noul": lambda t: dict(type="noul", instr=f"Is there {t} in the image?"),
          "short": lambda t: dict(type="noul", instr=t, block="short")}[kind]
    qs = [mk(t) for t in QUERIES]
    for q in qs:                       # warm-up: capture one graph per block length
        eng.query(bank, q); eng.query(bank, q)
    Ts = []
    for c in range(n_calls):
        T = {}
        eng.query(bank, qs[c % len(qs)], timings=T)
        Ts.append(T)
    return {f"{k}_ms": med([t[k] for t in Ts]) for k in Ts[0]} | dict(total_ms_p90=float(np.percentile([t["total"] for t in Ts], 90)))


def part_cached(B, a, res, tag="cached"):
    eng = B.eng
    out = res.setdefault(tag, {})
    for Tv, frame in [(int(t), int(t) != 640) for t in a.cached_tvs.split(",")]:
        pool = load_images(a.prep, a.image_root, max(a.Ns), Tv, frame=frame)
        ims = [x["img"] for x in pool]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        big = eng.encode(ims, chunk=16 if Tv >= 256 else 64)
        mem(f"image bank Tv{Tv}")
        torch.cuda.synchronize()
        enc_ms_per_img = (time.perf_counter() - t0) * 1e3 / len(ims)
        out[f"Tv{Tv}_encode_ms_per_image"] = enc_ms_per_img
        out[f"Tv{Tv}_bank_MB_per_image"] = big.nbytes() / len(ims) / 2**20
        for N in a.Ns:
            if N > len(ims):
                continue
            bank = big.head(N)
            for kind in (("box", "noul") if Tv == 640 else ("box",)):
                r = time_cached(eng, bank, a.n_calls, kind)
                out[f"Tv{Tv}_N{N}_{kind}"] = r
                print(tag, Tv, N, kind, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
            eng.free_graphs(bank)
            del bank
            json.dump(res, open(a.out, "w"), indent=1)
        del big
        eng.free_graphs()


def part_clips(B, a, res, tag="clips"):
    """(A) headline for video: each clip of the video encoded once, then every agent query scans all clips in one
    branch forward (yes/no readout). Also (B) fresh clips for reference."""
    eng = B.eng
    out = res.setdefault(tag, {})
    for fpc, mv in [tuple(map(int, c.split("x"))) for c in a.clip_cfgs.split(",")]:
        eng.dt, eng.clip_max_vis = 0.5, mv
        clips = make_clips(a, max(a.Ns), fpc, mv)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        big = eng.encode(clips, chunk=32)
        torch.cuda.synchronize()
        mem(f"clips bank f{fpc} v{mv}")
        key = f"f{fpc}_v{mv}"
        out[f"{key}_encode_ms_per_clip"] = (time.perf_counter() - t0) * 1e3 / len(clips)
        out[f"{key}_bank_MB_per_clip"] = big.nbytes() / len(clips) / 2**20
        out[f"{key}_prefix_tokens"] = int(big.p_valid[0].sum())
        for N in a.Ns:
            if N > big.N:
                continue
            bank = big.head(N)
            kind = "short" if a.short else "noul"
            r = time_cached(eng, bank, a.n_calls, kind)
            out[f"{key}_N{N}_{kind}"] = r
            print(tag, key, N, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
            eng.free_graphs(bank)
            del bank
            json.dump(res, open(a.out, "w"), indent=1)
        del big
        eng.free_graphs()
        for N in ((8, 16, 32, 64) if a.clip_fresh else ()):
            q = dict(type="noul", instr="Is a person walking in this clip?")
            cs = clips[:N]
            for _ in range(3):
                eng.fresh(cs, q)
            Ts = []
            for c in range(a.n_calls):
                T = {}
                eng.fresh(cs, dict(type="noul", instr=f"Is there {QUERIES[c % len(QUERIES)]} in this clip?"), timings=T)
                Ts.append(T)
            r = {f"{k}_ms": med([t[k] for t in Ts]) for k in Ts[0]}
            out[f"fresh_{key}_N{N}"] = r
            print(tag, "fresh", key, N, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        eng.free_graphs()


def part_fresh(B, a, res, tag="fresh", query_first=False):
    eng = B.eng
    out = res.setdefault(tag, {})
    for Tv in a.Tvs:
        pool = [x["img"] for x in load_images(a.prep, a.image_root, max(a.Ns_fresh), Tv, frame=True)]
        for N in a.Ns_fresh:
            reference_chunk = max(1, a.max_fresh_tokens // Tv)
            chunk = N if eng.fresh_varlen else reference_chunk
            if N > reference_chunk and not a.fresh_chunked:
                continue
            mk = (lambda t: dict(type="noul", instr=f"Is there {t} in the image?")) if a.fresh_kind == "yesno" else \
                 (lambda t: dict(type="box", text=t))
            calls = [([pool[(c * N + j) % len(pool)] for j in range(N)], mk(QUERIES[c % len(QUERIES)]))
                     for c in range(a.n_calls)]

            def run(ims, q, T=None):
                tot = {}
                chunks = 0
                for c0 in range(0, len(ims), chunk):
                    t = {}
                    eng.fresh(ims[c0:c0 + chunk], q, query_first=query_first, timings=t)
                    chunks += eng.last_fresh_chunks
                    for k, v in t.items():
                        tot[k] = tot.get(k, 0.0) + v
                if T is not None:
                    T.update(tot)
                return chunks
            try:
                for ims, q in calls[:4]:
                    run(ims, q)
                Ts, chunks = [], []
                for ims, q in calls:
                    T = {}
                    chunks.append(run(ims, q, T))
                    Ts.append(T)
            except torch.cuda.OutOfMemoryError:
                print("OOM", Tv, N, flush=True); eng._graphs.clear(); torch.cuda.empty_cache(); continue
            r = {f"{k}_ms": med([t[k] for t in Ts]) for k in Ts[0]}
            r["total_ms_p90"] = float(np.percentile([t["total"] for t in Ts], 90))
            r["tokens"] = N * Tv
            r["chunks"] = chunks[0] if len(set(chunks)) == 1 else med(chunks)
            r["chunks_per_call"] = chunks
            r["images_per_s"] = N / (r["total_ms"] / 1e3)
            r["readout"] = a.fresh_kind
            out[f"Tv{Tv}_N{N}"] = r
            print(tag, Tv, N, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        eng.free_graphs()


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ar = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    return inter / max(1e-9, ar(a) + ar(b) - inter)


@torch.no_grad()
def eval_set(eng, items, query_first=False, max_q=8):
    """items: load_images rows. Every (image, question) is its own block (exact = asked alone); eager calls.
    Image-first: all questions of an image share one prefix call. Query-first: the prefix holds the query, one call each."""
    rows = []
    for it in items:
        qs = [dict(type="box", text=q["text"]) for q in it["qs"][:max_q]]
        groups = [[q] for q in qs] if query_first else [qs]
        outs = []
        for g in groups:
            batch, e, host = eng.build([it["img"]], [g], query_first=g[0]["text"] if query_first else None)
            feats = eng.vision(batch["pixel_values"].cuda(), batch["grid_thw"])
            o, _, _ = eng._core(eng.to_device(e, host, feats), host)
            outs += [{k: v[j].float().cpu() for k, v in o.items()} for j in range(len(g))]
        W, H = it["wh"]
        for q, o in zip(it["qs"][:max_q], outs):
            bx = (o["box_pred"] * torch.tensor([W, H, W, H])).tolist()
            gt = (q.get("box") or (q.get("boxes") or [None])[0]) if q["present"] else None
            rows.append(dict(p=float(torch.sigmoid(o["presence_logit"])), present=bool(q["present"]),
                             iou=iou(bx, gt) if gt else None, single=bool(q["present"]) and q.get("n_targets", 1) == 1))
    return rows


def summarize_acc(rows):
    pos = np.array([r["p"] for r in rows if r["present"]]); neg = np.array([r["p"] for r in rows if not r["present"]])
    single = [r for r in rows if r["single"] and r["iou"] is not None]
    d = dict(n_pos=len(pos), n_neg=len(neg), acc50=float(np.mean([r["iou"] >= .5 for r in single])) if single else None)
    if len(neg):
        d.update(auroc=auroc(pos, neg), false_alarm_at_05=float((neg >= .5).mean()), miss_at_05=float((pos < .5).mean()))
        p = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        bins = np.minimum((p * 15).astype(int), 14)                 # ECE, 15 equal-width bins, presence probability
        d["ece15"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(15) if (bins == b).any()))
    return d


def part_acc(B, a, res):
    eng = B.eng
    out = res.setdefault("acc_vs_tokens", {})
    for Tv in a.Tvs + ([640] if 640 not in a.Tvs else []):
        for qf in ((False, True) if Tv in (256, 640) else (False,)):
            r = {}
            for name, prep, n in (("refcoco_val", a.prep, a.n_acc), ("grefcoco_val", a.gprep, a.n_acc)):
                items = load_images(prep, a.image_root, n, Tv)
                rows = eval_set(eng, items, query_first=qf)
                r[name] = summarize_acc(rows)
            out[f"Tv{Tv}{'_qfirst' if qf else ''}"] = r
            print("acc", Tv, qf, json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--gprep", default="data/prep/eval_grefcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--parts", default="equiv,clips,cached,fresh,qfirst,acc,fp8")
    ap.add_argument("--Ns", default="1,8,16,32,64,128,256")
    ap.add_argument("--Ns_fresh", default="1,2,4,8,16,32,64")
    ap.add_argument("--Tvs", default="64,128,256,640")
    ap.add_argument("--max_fresh_tokens", type=int, default=16384)
    ap.add_argument("--fresh_impl", choices=["reference", "varlen"], default="varlen")
    ap.add_argument("--lm_prefill", choices=["reference", "pointwise", "direct", "fla", "fused"], default="fused")
    ap.add_argument("--vision_prep", choices=["reference", "rgb", "fused"], default="fused",
                    help="HF preprocessing, compact RGB with the legacy tower, or reference-compatible fused vision")
    ap.add_argument("--fresh_kind", default="box", choices=["box", "yesno"], help="readout block for the fresh part")
    ap.add_argument("--fresh_chunked", type=int, default=0, help="E1b: run N above max_fresh_tokens in chunks")
    ap.add_argument("--n_equiv", type=int, default=24)
    ap.add_argument("--clip_cfgs", default="4x96,4x48,2x96", help="frames-per-clip x visual tokens per 2-frame group")
    ap.add_argument("--clip_fresh", type=int, default=1)
    ap.add_argument("--cached_tvs", default="640,256,64")
    ap.add_argument("--short", type=int, default=0, help="clips part: time the short scan readout (event phrase)")
    ap.add_argument("--w12", help="offline packed weight directory for SM120")
    ap.add_argument("--fp8", type=int, default=0, help="FP8 language model from the start (numerics change)")
    ap.add_argument("--n_acc", type=int, default=60)
    ap.add_argument("--n_calls", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.w12 and (a.fp8 or "fp8" in a.parts.split(",")):
        ap.error("select one LM weight format per scan sweep")
    a.Ns = [int(x) for x in a.Ns.split(",")]
    a.Ns_fresh = [int(x) for x in a.Ns_fresh.split(",")]
    a.Tvs = [int(x) for x in a.Tvs.split(",")]
    parts = a.parts.split(",")
    B = Bench(a)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, ckpt=a.ckpt, args=vars(a))
    if "equiv" in parts:
        part_equiv(B, a, res)
    if B.eng is None:
        B.to_engine(fp8=bool(a.fp8))
    res["n_merged"] = B.n_merged
    if "clips" in parts:
        part_clips(B, a, res)
    if "cached" in parts:
        part_cached(B, a, res)
    if "fresh" in parts:
        part_fresh(B, a, res)
    if "qfirst" in parts:
        part_fresh(B, a, res, tag="fresh_qfirst", query_first=True)
    if "acc" in parts:
        part_acc(B, a, res)
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(res, open(a.out, "w"), indent=1)
    if "fp8" in parts:
        B.to_engine(fp8=True)
        res["n_fp8"] = B.n_fp8
        if "clips" in parts:
            part_clips(B, a, res, tag="clips_fp8")
        part_cached(B, a, res, tag="cached_fp8")
        part_fresh(B, a, res, tag="fresh_fp8")
        if "acc" in parts:
            out = res.setdefault("acc_fp8", {})
            for Tv in (128, 256):
                out[f"Tv{Tv}"] = {n: summarize_acc(eval_set(B.eng, load_images(p, a.image_root, a.n_acc, Tv)))
                                  for n, p in (("refcoco_val", a.prep), ("grefcoco_val", a.gprep))}
                print("acc_fp8", Tv, json.dumps(out[f"Tv{Tv}"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
