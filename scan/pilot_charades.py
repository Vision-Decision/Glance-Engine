"""Phase-2 pilot: frame scanning on Charades-STA test with the CURRENT checkpoints (no training).

python -m scan.pilot_charades --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/charades_pilot --out OUT.json.gz
       [--tv 256] [--k_neg 3] [--max_videos 0] [--scorers g_box,g_yes,base_yes,siglip]

Every Charades test video is encoded ONCE per model into an ImageBank (cached mode); then every query that touches the
video is asked against all its frames in one branch forward:
  positive scan  an STA query on its own video; frame labels = inside any instance of the query's action class
  negative scan  the same query on k_neg other test videos that contain neither the query's class nor any class with
                 the same object (Charades class -> object mapping), i.e. clean "did not happen" scans
Per-frame scores (one row per scorer, logit scale):
  g_box     Glance box block with the sentence: presence logit (also stores box + P(IoU>=.5) of the top frame)
  g_yes     Glance typed yes/no head: 'Does this image show: {sentence}?'  logit(yes) - logit(no)
  g_short   the short scan block '\n<ref>{sentence}</ref><im_end>' read by the same yes/no labels (distilled model)
  base_yes  same-base generative readout: Qwen3.5-4B without the LoRA, causal block, LM-head next-token logits of
            yes/Yes vs no/No after the same prompt (= the vdm 'qwen' backend; FRAG / SeViLA-style scanner)
  siglip    embedding retriever: SigLIP2 so400m image-text logit (scale * cos + bias)
Output: scans with per-frame scores, labels, timings; analysis in scan/pilot_analyze.py (CPU).
"""
import argparse, gzip, json, os, random, time
import numpy as np
import torch
from PIL import Image

from dv.config import Config
from dv.data.dataset import resize_dims
from dv.data.mixed import ChatTok
from dv.model.build import build, load_ckpt, V1Heads
from speed.engine import merge_lora_, fuse_weights_, fp8_language_model_
from scan.engine import ScanEngine


def load_glance(ckpt, fp8=False, w12=None):
    if w12 and fp8:
        raise ValueError("choose one LM weight format")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(ckpt, net, heads)
    net.eval(); heads.eval()
    merge_lora_(net.hf)
    fuse_weights_(net.lm, net.layer_types, keep_ab_separate=fp8)
    if fp8:                                   # E7: W8A8 FP8 language model (as bench_scan --fp8 1)
        fp8_language_model_(net.lm)
    if w12:
        from speed.w12 import install
        install(net.lm, w12)
    tok = ChatTok(proc, cfg)
    return ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=True), cfg


def load_base(cfg_g):
    cfg = Config().update(dict(cfg_g.__dict__, attn_mask="causal", lora_r=0))
    net, heads, proc = build(cfg, "cuda", lora=False)
    net.eval()
    tok = ChatTok(proc, cfg)
    heads = V1Heads(net.hidden, cfg, net.hf.get_output_embeddings().weight, tok.label_token_groups()).cuda().eval()
    fuse_weights_(net.lm, net.layer_types)
    return ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=True)


class SigLIP:
    def __init__(self, mid="google/siglip2-so400m-patch14-384"):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(mid, dtype=torch.float16).cuda().eval()
        self.p = AutoProcessor.from_pretrained(mid)

    @torch.no_grad()
    def images(self, ims):
        x = self.p(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        e = self.m.get_image_features(pixel_values=x)
        e = getattr(e, "pooler_output", e)
        return torch.nn.functional.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def text(self, t):
        x = self.p(text=[t], padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        e = self.m.get_text_features(input_ids=x["input_ids"].cuda())
        e = getattr(e, "pooler_output", e)
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]

    @torch.no_grad()
    def logits(self, img_emb, txt_emb):
        return (img_emb @ txt_emb * self.m.logit_scale.exp().float() + self.m.logit_bias.float()).cpu().numpy()


def unit_times(n, fps, unit, K, S):
    """centre time of each scan unit (frame k at k/fps; clip = frames [i, i+K) starting every S frames)"""
    if unit == "frame":
        return np.arange(n) / fps
    starts = np.arange(0, max(1, n - K + 1), S)
    return (starts + (K - 1) / 2) / fps


def frame_labels(n, fps, instances, unit="frame", K=4, S=2):
    t = unit_times(n, fps, unit, K, S)
    y = np.zeros(len(t), dtype=bool)
    for s, e in instances:
        y |= (t >= s) & (t <= e)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tv", type=int, default=256)
    ap.add_argument("--unit", default="frame", choices=["frame", "clip"],
                    help="clip: K frames (at the data fps) per unit, stride S frames, as the vdm1 client (2 fps, 4, 2)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--clip_vis", type=int, default=96)
    ap.add_argument("--gv_ckpt", default="", help="also score with a Glance-Video checkpoint (scorer gv_yes)")
    ap.add_argument("--k_neg", type=int, default=8)
    ap.add_argument("--k_hard", type=int, default=0, help="E4: also k hard negatives (same object, other action)")
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--scorers", default="g_box,g_yes,base_yes,siglip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp8", action="store_true", help="E7: FP8 language model for the Glance scorer")
    a = ap.parse_args()
    sc = a.scorers.split(",")
    meta = json.load(open(f"{a.data}/meta.json"))
    V, Q, fps = meta["videos"], meta["queries"], meta["fps"]
    obj = meta["class_obj"]
    rng = random.Random(a.seed)
    vids = sorted(V)
    # ---- scan list: positives + clean negatives
    scans = []
    for q in Q:
        if q["vid"] not in V:
            continue
        scans.append(dict(qid=q["qid"], vid=q["vid"], pos=True, text=q["text"], cls=q["cls"],
                          labels=(lab := frame_labels(V[q["vid"]]["n_frames"], fps, q["instances"], a.unit, a.K, a.S)).tolist(),
                          onset_frame=int(np.argmax(lab)) if lab.any() else None))
        src = V[q["vid"]].get("source")          # QVHighlights: negatives from another YouTube source
        cand = [v for v in vids if v != q["vid"] and q["cls"] not in V[v]["classes"]
                and obj.get(q["cls"]) not in {obj.get(c) for c in V[v]["classes"]}
                and (src is None or V[v].get("source") != src)]
        for v in rng.sample(cand, min(a.k_neg, len(cand))):
            scans.append(dict(qid=q["qid"], vid=v, pos=False, text=q["text"], cls=q["cls"], labels=None, onset_frame=None))
        if a.k_hard and obj.get(q["cls"]):       # E4 hard negatives: same object, different action, query class absent
            hard = [v for v in vids if v != q["vid"] and q["cls"] not in V[v]["classes"]
                    and obj.get(q["cls"]) in {obj.get(c) for c in V[v]["classes"]}]
            for v in rng.sample(hard, min(a.k_hard, len(hard))):
                scans.append(dict(qid=q["qid"], vid=v, pos=False, hard=True, text=q["text"], cls=q["cls"], labels=None,
                                  onset_frame=None))
    by_vid = {}
    for i, s in enumerate(scans):
        by_vid.setdefault(s["vid"], []).append(i)
    order = sorted(by_vid)
    if a.max_videos:
        keep = set(rng.sample(order, a.max_videos))
        order = [v for v in order if v in keep]
        scans_keep = [i for v in order for i in by_vid[v]]
    print(f"videos {len(order)} scans {sum(len(by_vid[v]) for v in order)}", flush=True)
    G, cfg = load_glance(a.ckpt, fp8=a.fp8)
    GV = load_glance(a.gv_ckpt)[0] if a.gv_ckpt else None
    for e in (G, GV):
        if e is not None:
            e.dt, e.clip_max_vis = 1.0 / fps, a.clip_vis
    Bm = load_base(cfg) if "base_yes" in sc else None
    S = SigLIP() if "siglip" in sc else None
    T = dict(encode_glance=0.0, encode_base=0.0, query_glance=0.0, query_base=0.0, n_frames=0, n_queries=0)
    t_start = time.time()
    for vi, v in enumerate(order):
        n = V[v]["n_frames"]
        raw = [Image.open(f"{a.data}/frames/{v}/{k:03d}.jpg").convert("RGB") for k in range(n)]
        w, h = resize_dims(*raw[0].size, a.tv, 2.0)
        ims = [im.resize((w, h), Image.BICUBIC) for im in raw]
        if a.unit == "clip":     # clips keep the raw frames; the video processor resizes to clip_vis per 2-frame group
            starts = range(0, max(1, n - a.K + 1), a.S)
            ims = [raw[i:i + a.K] if i + a.K <= n else (raw[i:] + [raw[-1]] * (i + a.K - n)) for i in starts]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        bank = G.encode(ims, chunk=64)
        torch.cuda.synchronize(); T["encode_glance"] += time.perf_counter() - t0
        if Bm is not None:
            t0 = time.perf_counter(); bbank = Bm.encode(ims, chunk=64); torch.cuda.synchronize()
            T["encode_base"] += time.perf_counter() - t0
        if GV is not None:
            gvbank = GV.encode(ims, chunk=32)
        semb = S.images(raw) if S is not None else None
        if semb is not None and a.unit == "clip":       # clip embedding = mean of its frames' embeddings
            semb = torch.nn.functional.normalize(torch.stack([semb[i:i + a.K].mean(0) for i in range(0, max(1, n - a.K + 1), a.S)]), dim=-1)
        T["n_frames"] += n
        for i in by_vid[v]:
            s = scans[i]
            yq = dict(type="noul", instr=f"Does this {'clip' if a.unit == 'clip' else 'image'} show: {s['text']}?")
            sco = {}
            t0 = time.perf_counter()
            if GV is not None:
                tl = GV.query(gvbank, yq)["typed_logits"]
                sco["gv_yes"] = (tl[:, 1] - tl[:, 0]).numpy().round(4).tolist()
            if "g_box" in sc and a.unit == "frame":
                o = G.query(bank, dict(type="box", text=s["text"]))
                sco["g_box"] = o["presence_logit"].numpy().round(4).tolist()
                top = int(o["presence_logit"].argmax())
                s["g_box_top"] = dict(frame=top, box=o["box_pred"][top].tolist(), iou_logit=float(o["iou_logit"][top]))
            if "g_yes" in sc:
                tl = G.query(bank, yq)["typed_logits"]
                sco["g_yes"] = (tl[:, 1] - tl[:, 0]).numpy().round(4).tolist()
            if "g_short" in sc:       # short scan readout (event phrase, ~10 tokens; trained by scan/distill_*)
                tl = G.query(bank, dict(type="noul", instr=s["text"], block="short"))["typed_logits"]
                sco["g_short"] = (tl[:, 1] - tl[:, 0]).numpy().round(4).tolist()
            T["query_glance"] += time.perf_counter() - t0
            if Bm is not None:
                t0 = time.perf_counter()
                tl = Bm.query(bbank, yq)["typed_logits"]
                sco["base_yes"] = (tl[:, 1] - tl[:, 0]).numpy().round(4).tolist()
                T["query_base"] += time.perf_counter() - t0
            if S is not None:
                sco["siglip"] = S.logits(semb, S.text(s["text"])).round(4).tolist()
            s["scores"] = sco
            T["n_queries"] += 1
        del bank
        if GV is not None:
            del gvbank
        if Bm is not None:
            del bbank
        if vi % 50 == 0:
            el = time.time() - t_start
            print(f"video {vi}/{len(order)} {el:.0f}s eta {el / (vi + 1) * (len(order) - vi - 1):.0f}s", flush=True)
        if vi % 200 == 0 or vi == len(order) - 1:
            done = [scans[i] for u in order[:vi + 1] for i in by_vid[u]]
            with gzip.open(a.out, "wt") as f:
                json.dump(dict(args=vars(a), timing=T, fps=fps, scans=done), f)
    print("PILOT_DONE", json.dumps(T), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
