"""Glance speed benchmark: shipped path vs the fast engine, equivalence + latency / throughput sweeps.

python -m speed.bench_glance --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json [--quick] [--parts equiv,lat,...]

Modes (one model load; LoRA merging and FP8 conversion are applied in place, in this order):
  ref_lora    the shipped inference path (dv.model.v1forward.run_model_v1, PEFT LoRA unmerged, eager)  = the research forward's 85 ms
  ref_merged  same code, LoRA merged into the base weights
  eng_eager   speed/engine.py, eager
  eng_graph   speed/engine.py, CUDA graphs per shape bucket
  eng_fused   eng_graph + concatenated projections (gate|up, q|k|v, GDN in-projections) + compiled fused
              RMSNorm/residual/SiLU-mul/GDN-gate kernels
  eng_w12     eng_fused with lossless packed LM weights for measured skinny GEMMs (requires --w12 DIR)
  eng_fp8     eng_fused with W8A8 FP8 language-model (and with --fp8_vision vision-tower) linears
              (not numerically equivalent; accuracy reported)
Timing per call (batch of images, each with its questions): host preprocessing (image processor on the already
resized image, tokenisation, collation) + H2D + vision tower + language model + heads + D2H of all answers.
JPEG decode and resize are excluded (identical for every system, also for the generative baselines).
Reported: ms median / p90 per call, model-only ms (H2D..D2H), questions/s, plus per-stage medians.
"""
import argparse, json, math, os, statistics, time
import numpy as np
import torch

from dv.config import Config
from dv.data.mixed import ChatTok, QTYPES, collate_mixed
from dv.model.build import build, load_ckpt
from dv.model.v1forward import run_model_v1
from speed import workload as W
from speed.engine import GlanceEngine, merge_lora_, fp8_language_model_, fuse_weights_, prepare


def make_item(tok, proc, img, qs):
    pv = proc.image_processor(images=[img], do_resize=False, return_tensors="pt")
    segs = [("ids", tok.user_open), ("ids", [tok.vs]), ("img", None), ("ids", [tok.ve])]
    q_ids, qtype, labs, tgts = [], [], [], []
    for q in qs:
        t = q["type"]
        qtype.append(QTYPES[t])
        if t == "box":
            q_ids.append(tok.box_block(q["text"])); labs.append([]); tgts.append([])
        else:
            ids, lab = tok.typed_block(q)
            q_ids.append(ids); labs.append(lab); tgts.append([0.0] * len(lab))
    n = len(qs)
    return dict(rec={}, segs=segs, pixel_values=pv["pixel_values"], grid_thw=pv["image_grid_thw"][0], q_ids=q_ids,
                qtype=qtype, labs=labs, tgts=tgts, present=[False] * n, has_box=[False] * n, box=[[0, 0, 0, 0]] * n,
                points=[[] for _ in range(n)], qs=qs, orig_wh=img.size)


class Runner:
    def __init__(self, a):
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
        self.cfg = cfg
        net, heads, proc = build(cfg, "cuda")
        load_ckpt(a.ckpt, net, heads)
        net.eval(); heads.eval()
        self.net, self.heads, self.proc = net, heads, proc
        self.tok = ChatTok(proc, cfg)
        self.eng = None
        self.mode = "ref_lora"
        self.fp8_vision = getattr(a, "fp8_vision", False)
        self.w12 = getattr(a, "w12", None)
        self.lm_prefill = getattr(a, "lm_prefill", "fused") == "fused"
        self.no_fused = bool(os.environ.get("SPD_NO_FUSED"))

    def set_mode(self, mode):
        """ref_lora -> ref_merged -> eng_eager -> eng_graph -> eng_fused -> eng_fp8 (in this order; in-place changes)."""
        if mode != "ref_lora" and self.mode == "ref_lora":
            self.n_merged = merge_lora_(self.net.hf)
        fast = mode in ("eng_fused", "eng_fp8", "eng_w12") and not getattr(self, "no_fused", False)
        if fast:
            fuse_weights_(self.net.lm, self.net.layer_types, keep_ab_separate=(mode == "eng_fp8"))
        if mode == "eng_w12":
            from speed.w12 import install
            install(self.net.lm, self.w12)
        if mode == "eng_fp8":
            self.n_fp8 = fp8_language_model_(self.net.lm)
            if getattr(self, "fp8_vision", False):
                self.n_fp8 += fp8_language_model_(self.net.visual, skip=("patch_embed",))
        self.eng = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if mode.startswith("eng"):
            self.eng = GlanceEngine(self.net, self.heads, graphs=(mode != "eng_eager"), fast_ops=fast, lm_prefill=self.lm_prefill)
        self.frames = None
        if mode == "eng_fused" and self.eng.lm_prefill:
            from scan.vision_prep import FramePrep
            self.frames = FramePrep(self, device=self.eng.dev)
        self.mode = mode

    @torch.no_grad()
    def call(self, imgs, qss, Nv=None, keep_prefix=False):
        """One call. imgs: list of PIL (resized); qss: list of question lists. Returns (answers, timings dict)."""
        T = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rgb_batch = (getattr(self, "frames", None) is not None and self.frames.fused and not keep_prefix and len(imgs) > 1
                     and all(len(qs) == 1 for qs in qss) and all(self.frames.supports([im]) for im in imgs))
        if rgb_batch:
            from scan.engine import image_item
            items = [image_item(self.tok, torch.tensor([1, im.height // 16, im.width // 16]), im.size, qs)
                     for im, qs in zip(imgs, qss)]
        else:
            items = [make_item(self.tok, self.proc, im, qs) for im, qs in zip(imgs, qss)]
        batch = collate_mixed(items, self.tok)
        batch.pop("meta")
        if self.mode.startswith("ref"):
            t1 = time.perf_counter()
            b = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            r = run_model_v1(self.net, self.heads, b, train=False)
            out = dict(qtype=batch["qtype"])
            if "box" in r:
                bi = r["box_idx"].cpu()
                if getattr(self, "trace_centers", False):
                    index = torch.zeros(len(batch["qtype"]), dtype=torch.long)
                    index[bi] = r["box"]["pred_idx"].cpu()
                    out["center_idx"] = index
                    margins = torch.full((len(index),), float("nan"))
                    scores = r["box"]["center_logits"]
                    if scores.shape[-1] >= 2:
                        best = scores.topk(2).values
                        margins[bi] = (best[:, 0] - best[:, 1]).float().cpu()
                    out["center_margin"] = margins
                for k in ("presence_logit", "box_pred", "iou_logit"):
                    v = r["box"][k].float().cpu()
                    full = torch.zeros((len(batch["qtype"]),) + v.shape[1:])
                    full[bi] = v
                    out[k] = full
            if "typed_logits" in r:
                ti = r["typed_mask"].nonzero()[:, 0].cpu()
                tl = r["typed_logits"].float().cpu()
                full = torch.full((len(batch["qtype"]), 26), -float("inf"))
                full[ti, :tl.shape[1]] = tl
                out["typed_logits"] = full
            torch.cuda.synchronize(); t2 = time.perf_counter()
            T.update(prep=(t1 - t0) * 1e3, model=(t2 - t1) * 1e3, total=(t2 - t0) * 1e3)
            return out, T
        independent = (self.eng.lm_prefill and not keep_prefix and len(imgs) > 1 and
                       all(len(qs) == 1 for qs in qss) and batch["n_tokens"] >= 4096)
        if not independent:
            eng, host = prepare(batch, Nv=Nv)
        t1 = time.perf_counter()
        if rgb_batch:
            pv = self.frames.mixed_pixels(imgs, dtype=self.eng.fv.W.dtype)
        else:
            pv = batch["pixel_values"].to("cuda", non_blocking=True)
        feats = self.eng.vision(pv, batch["grid_thw"])
        torch.cuda.synchronize(); tv = time.perf_counter()
        if independent:
            res = self.eng.answer_independent(batch, feats)
            M0 = len(qss)
        else:
            E = self.eng.to_device(eng, host, feats)
            res, cache, vis = self.eng.run(E, host)
            M0 = host["M0"]
        out = {k: v[:M0].cpu() for k, v in res.items()}
        out["qtype"] = batch["qtype"]
        t2 = time.perf_counter()
        T.update(prep=(t1 - t0) * 1e3, vision=(tv - t1) * 1e3, lm=(t2 - tv) * 1e3, model=(t2 - t1) * 1e3, total=(t2 - t0) * 1e3)
        if keep_prefix:
            self._prefix = ([(a.clone(), b.clone()) for a, b in cache], vis.clone(), eng, host)
        return out, T

    @torch.no_grad()
    def followup(self, qss):
        """Questions about the image of the last call(keep_prefix=True): prefix cache hit, no vision / prefill."""
        cache, vis, peng, phost = self._prefix
        torch.cuda.synchronize(); t0 = time.perf_counter()
        items = [dict(rec={}, segs=[], q_ids=[], qtype=[], labs=[], tgts=[], present=[], has_box=[], box=[], points=[],
                      qs=[], orig_wh=None)]
        it = items[0]
        for q in qss[0]:
            t = q["type"]; it["qtype"].append(QTYPES[t])
            if t == "box":
                it["q_ids"].append(self.tok.box_block(q["text"])); it["labs"].append([]); it["tgts"].append([])
            else:
                ids, lab = self.tok.typed_block(q); it["q_ids"].append(ids); it["labs"].append(lab); it["tgts"].append([0.0] * len(lab))
            it["present"].append(False); it["has_box"].append(False); it["box"].append([0, 0, 0, 0]); it["points"].append([])
        qb = collate_mixed(items, self.tok)
        # question positions continue from the cached prefix
        q_start = int(peng["q_pos"][0, 0, 0])
        Tq = qb["q_ids"].shape[1]
        qb["q_pos"][:] = q_start + torch.arange(Tq)
        qb.update(p_ids=peng["p_ids"], p_valid=peng["p_valid"], p_img_mask=torch.zeros_like(peng["p_valid"]),
                  p_pos=peng["p_pos"], vis_valid=peng["vis_valid"], centers=peng["centers"])
        eng, host = prepare(qb, Nv=phost["Nv"], compact_questions=self.eng.compact_questions)
        host["nopad"] = phost["nopad"]
        t1 = time.perf_counter()
        E = {k: v.to("cuda", non_blocking=True) for k, v in eng.items() if k not in ("img_flat",)}
        res = self.eng.run_q(E, host, cache, vis)
        out = {k: v[:host["M0"]].cpu() for k, v in res.items()}
        out["qtype"] = qb["qtype"]
        t2 = time.perf_counter()
        return out, dict(prep=(t1 - t0) * 1e3, model=(t2 - t1) * 1e3, total=(t2 - t0) * 1e3)


MAX_BATCH = 10 ** 9   # --max_batch: skip regimes with more images per call (small GPUs)
SAVE = None            # (results dict, path) for incremental saving


def summarize_times(Ts, nq):
    d = {}
    for k in Ts[0]:
        v = np.array([t[k] for t in Ts])
        d[f"{k}_ms_median"] = float(np.median(v))
        if k in ("total", "model"):
            d[f"{k}_ms_p90"] = float(np.percentile(v, 90))
    tot = np.array([t["total"] for t in Ts])
    d["ttfa_ms_median"] = d["total_ms_median"]          # every answer of a call arrives at once
    d.update(q_per_s=float(nq * len(tot) / (tot.sum() / 1e3)), n_calls=len(Ts), questions_per_call=nq)
    return d


def sweep(R, images, cfgs, n_calls, warm=True, tag=""):
    """cfgs: list of (name, ipb, nq, kind, Nv). Every config: one untimed pass over its calls (graph capture,
    autotune), then the timed pass."""
    res = {}
    for name, ipb, nq, kind, Nv in cfgs:
        if ipb > MAX_BATCH:
            continue
        calls = []
        for c in range(n_calls):
            idx = [(c * ipb + j) % len(images) for j in range(ipb)]
            calls.append(([images[i]["img"] for i in idx], [W.questions(images, i, nq, kind) for i in idx]))
        if warm:
            for im, qs in calls[: min(len(calls), 12)]:
                R.call(im, qs, Nv=Nv)
            for im, qs in calls:
                R.call(im, qs, Nv=Nv)
        Ts = [R.call(im, qs, Nv=Nv)[1] for im, qs in calls]
        res[name] = summarize_times(Ts, ipb * nq)
        print(R.mode, tag, name, json.dumps({k: round(v, 2) for k, v in res[name].items()}), flush=True)
        if SAVE is not None:            # incremental save (a later OOM must not lose finished regimes)
            SAVE[0]["modes"][R.mode].update(res)
            json.dump(SAVE[0], open(SAVE[1], "w"), indent=1)
    return res


def followup_sweep(R, images, nqs, n_calls):
    res = {}
    for nq in nqs:
        for kind in ("box", "mixed"):
            Ts = []
            for c in range(n_calls + 3):
                i = c % len(images)
                R.call([images[i]["img"]], [W.questions(images, i, 1, "box")], Nv=640, keep_prefix=True)
                _, T = R.followup([W.questions(images, i, nq, kind, seed=1)])
                if c >= 3:
                    Ts.append(T)
            # second pass after capture
            Ts = []
            for c in range(n_calls):
                i = c % len(images)
                R.call([images[i]["img"]], [W.questions(images, i, 1, "box")], Nv=640, keep_prefix=True)
                Ts.append(R.followup([W.questions(images, i, nq, kind, seed=1)])[1])
            name = f"followup_{kind}_N{nq}"
            res[name] = summarize_times(Ts, nq)
            print(R.mode, name, json.dumps({k: round(v, 2) for k, v in res[name].items()}), flush=True)
    return res


def box_px(out, k, wh):
    W_, H_ = wh
    return (out["box_pred"][k] * torch.tensor([W_, H_, W_, H_])).tolist()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ar = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    return inter / max(1e-9, ar(a) + ar(b) - inter)


def equiv_outputs(R, images, n_img, nq=8):
    """Per mode: outputs for n_img images x nq mixed questions + all box questions of the image (Acc@0.5)."""
    rows = []
    for i in range(n_img):
        im = images[i]
        qs = W.questions(images, i, nq, "mixed") + [dict(type="box", text=e, present=True, box=None) for e in im["exprs"]]
        out, _ = R.call([im["img"]], [qs], Nv=640)
        for k, q in enumerate(qs):
            if q["type"] == "box":
                r = dict(t="box", p=float(torch.sigmoid(out["presence_logit"][k])), box=box_px(out, k, im["wh"]),
                         piou=float(torch.sigmoid(out["iou_logit"][k])))
                if "center_idx" in out:
                    r["center_idx"] = int(out["center_idx"][k])
                if "center_margin" in out:
                    r["center_margin"] = float(out["center_margin"][k])
                if k >= nq:
                    r["gt"] = im["gts"][k - nq]
            else:
                lg = out["typed_logits"][k]
                r = dict(t=q["type"], probs=torch.softmax(lg[torch.isfinite(lg)], -1).tolist())
            rows.append(r)
    return rows


def compare(rows, ref):
    dp, db, di, dt, agree, ious = [], [], [], [], [], []
    for a, b in zip(rows, ref):
        if a["t"] == "box":
            dp.append(abs(a["p"] - b["p"])); di.append(abs(a["piou"] - b["piou"]))
            db.append(max(abs(x - y) for x, y in zip(a["box"], b["box"]))); ious.append(iou(a["box"], b["box"]))
            agree.append((a["p"] >= 0.5) == (b["p"] >= 0.5))
        else:
            dt.append(max(abs(x - y) for x, y in zip(a["probs"], b["probs"])))
            agree.append(int(np.argmax(a["probs"])) == int(np.argmax(b["probs"])))
    acc = lambda rs: float(np.mean([iou(r["box"], r["gt"]) >= 0.5 for r in rs if "gt" in r]))
    return dict(presence_absdiff_max=float(max(dp)), presence_absdiff_mean=float(np.mean(dp)),
                piou_absdiff_max=float(max(di)), box_px_maxdiff_max=float(max(db)), box_px_maxdiff_mean=float(np.mean(db)),
                box_iou_vs_ref_min=float(min(ious)), box_iou_vs_ref_mean=float(np.mean(ious)),
                typed_prob_absdiff_max=float(max(dt)) if dt else None, decision_agreement=float(np.mean(agree)),
                acc50=acc(rows), acc50_ref=acc(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--gui", default="/content/gui/sspro40.jsonl")
    ap.add_argument("--modes", default="ref_lora,ref_merged,eng_eager,eng_graph,eng_fp8")
    ap.add_argument("--parts", default="equiv,lat,mixed,followup,tput,video,gui")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--n_equiv", type=int, default=100)
    ap.add_argument("--n_calls", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lm_prefill", choices=["reference","fused"], default="fused")
    ap.add_argument("--w12", help="offline packed weight directory for eng_w12")
    ap.add_argument("--fp8_vision", action="store_true", help="eng_fp8 also quantises the vision tower linears")
    ap.add_argument("--max_batch", type=int, default=10 ** 9, help="skip regimes with more images per call")
    ap.add_argument("--gui_nv", default="2048,4096")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "eng_w12" in a.modes.split(",") and (not a.w12 or a.modes.split(",")[-1] != "eng_w12" or "eng_fp8" in a.modes):
        ap.error("eng_w12 requires --w12 and must be the last mode in a BF16 sweep")
    if a.quick:
        a.n_images, a.n_equiv, a.n_calls = 20, 10, 10
    parts = a.parts.split(",")
    R = Runner(a)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, ckpt=a.ckpt, modes={}, equiv={})
    global MAX_BATCH, SAVE
    MAX_BATCH, SAVE = a.max_batch, (res, a.out)
    imgs = W.load_coco(a.prep, a.image_root, max(a.n_images, a.n_equiv))
    frames = W.load_coco(a.prep, a.image_root, a.n_images, max_vis=256)
    gui = {}
    if "gui" in parts:
        import os
        if os.path.exists(a.gui):
            gui = {nv: W.load_gui(a.gui, 40, max_vis=nv) for nv in map(int, a.gui_nv.split(","))}
    QS = [1, 2, 4, 8, 16, 32, 64]
    ref_rows = {}
    for mode in a.modes.split(","):
        R.set_mode(mode)
        res["modes"][mode] = r = {}
        torch.cuda.reset_peak_memory_stats()
        if "equiv" in parts:
            rows = equiv_outputs(R, imgs, a.n_equiv)
            ref_rows[mode] = rows
            res["equiv"][mode] = dict(vs_ref_lora=compare(rows, ref_rows["ref_lora"]) if "ref_lora" in ref_rows else None,
                                      vs_ref_merged=compare(rows, ref_rows["ref_merged"]) if "ref_merged" in ref_rows else None)
            print("equiv", mode, json.dumps(res["equiv"][mode]), flush=True)
        if "lat" in parts:
            r.update(sweep(R, imgs, [(f"bs1_box_N{n}", 1, n, "box", 640) for n in QS], a.n_calls))
        if "mixed" in parts:
            r.update(sweep(R, imgs, [(f"bs1_mixed_N{n}", 1, n, "mixed", 640) for n in (4, 8, 16, 32, 64)], a.n_calls))
        if "followup" in parts and mode.startswith("eng"):
            r.update(followup_sweep(R, imgs, (1, 8, 32), a.n_calls))
        if "tput" in parts:
            cf = [(f"bs{b}_box_N{n}", b, n, "box", 640) for b in (2, 4, 8, 16, 32) for n in (1, 8)]
            r.update(sweep(R, imgs, cf, max(6, a.n_calls // 4)))
        if "tput2" in parts:      # saturation with many questions per image
            cf = [(f"bs{b}_box_N{n}", b, n, "box", 640) for b, n in ((4, 32), (8, 32), (16, 32), (4, 64), (8, 64))]
            r.update(sweep(R, imgs, cf, max(6, a.n_calls // 4)))
        if "video1" in parts:
            cf = [(f"video_f{b}_N1", b, 1, "box", 256) for b in (32, 64)]
            r.update(sweep(R, frames, cf, max(6, a.n_calls // 4)))
        if "video" in parts:
            cf = [(f"video_f{b}_N{n}", b, n, "mixed" if n > 1 else "box", 256) for b in (8, 32, 64) for n in (1, 4)]
            r.update(sweep(R, frames, cf, max(6, a.n_calls // 4)))
        if gui:
            for nv, g in gui.items():
                cf = [(f"gui{nv}_box_N{n}", 1, n, "box", nv) for n in (1, 8, 32)]
                r.update(sweep(R, g, cf, min(a.n_calls, 30)))
        r["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        res["n_merged"] = getattr(R, "n_merged", None); res["n_fp8"] = getattr(R, "n_fp8", None)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
