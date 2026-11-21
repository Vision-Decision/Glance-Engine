"""P1 video-latency microbenchmarks (M2, M3, M4, M9 + correctness check). Run on the G4:
  python -m speed.video_p1.bench --model Qwen/Qwen3.5-4B --tests check,m2,m3,m4,m9 --out outputs/video_p1/raw/4b.json
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from speed.video_p1.stream import StreamLM, Readout, Pipeline, patchify

ARGS = None


def stats(samples):
    a = np.asarray(samples)
    return dict(p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                mean=float(a.mean()), min=float(a.min()), n=len(a))


def timeit(fn, n=50, warm=10):
    """Wall-clock per call between device syncs (includes host dispatch, which is the point for eager mode)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return stats(out)


def load(model_id):
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return m


# ---------------------------------------------------------------------------------------------- check
@torch.no_grad()
def check(hf, res):
    """Incremental streaming step == HF full forward (frame 1, frame 2, then one question block)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Q  # noqa: F401
    slm = StreamLM(hf)
    slm.q_causal = True                                                # match HF's causal mask inside the block
    T, Tq, M = 64, 20, 2
    dev = slm.dev
    torch.manual_seed(0)
    emb = slm.lm.embed_tokens.weight
    ids = torch.randint(0, 200000, (2 * T + M * Tq,), device=dev)
    x_all = emb[ids]                                                   # token embeddings as stand-in frame features
    slm.alloc(1, T, T, M, Tq)
    slm.win_valid.zero_()
    slm._build_mask()
    slm.pos.zero_()
    for d in list(slm.S.values()):
        d.zero_()
    for d in list(slm.tail.values()):
        d.zero_()
    slm.attn_impl = "sdpa_mask"                                        # frame 1: empty window needs the mask
    h1 = slm.step(torch.cat([x_all[:T], x_all[2 * T:]]))
    slm.win_valid.fill_(True)
    slm._build_mask()
    snap = ({k: v.clone() for k, v in slm.S.items()}, {k: v.clone() for k, v in slm.tail.items()},
            {k: v.clone() for k, v in slm.K.items()}, {k: v.clone() for k, v in slm.V.items()}, slm.pos.clone(), slm.wptr.clone())
    slm.attn_impl = "sdpa_mask"
    h2m = slm.step(torch.cat([x_all[T:2 * T], x_all[2 * T:]]))
    for d, s_ in zip((slm.S, slm.tail, slm.K, slm.V), snap[:4]):
        for k in d:
            d[k].copy_(s_[k])
    slm.pos.copy_(snap[4]); slm.wptr.copy_(snap[5])
    slm.attn_impl = "split"                                            # frame 2: production path (full window)
    h2 = slm.step(torch.cat([x_all[T:2 * T], x_all[2 * T:]]))
    # reference: HF text model over [f1, f2, block j] for each block j
    errs = {}
    lm = slm.lm
    for j in range(M):
        seq = torch.cat([x_all[:2 * T], x_all[2 * T + j * Tq: 2 * T + (j + 1) * Tq]])[None]
        L = seq.shape[1]
        pos = torch.arange(L, device=dev)[None, None].expand(3, 1, L)
        ref = lm(inputs_embeds=seq, position_ids=pos).last_hidden_state[0]
        mine_q = h2[T:].view(M, Tq, -1)[j]
        rq = ref[2 * T:].float()
        errs[f"block{j}_rel"] = float((mine_q.float() - rq).norm() / rq.norm())
        errs[f"block{j}_cos_min"] = float(F.cosine_similarity(mine_q.float(), rq, -1).min())
    ref2 = lm(inputs_embeds=x_all[None, :2 * T], position_ids=torch.arange(2 * T, device=dev)[None, None].expand(3, 1, -1)).last_hidden_state[0]
    errs["split_vs_mask_rel"] = float((h2.float() - h2m.float()).norm() / h2m.float().norm())
    errs["frame2_rel"] = float((h2[:T].float() - ref2[T:].float()).norm() / ref2[T:].float().norm())
    errs["frame1_rel"] = float((h1[:T].float() - ref2[:T].float()).norm() / ref2[:T].float().norm())
    errs["frame2_cos_min"] = float(F.cosine_similarity(h2[:T].float(), ref2[T:].float(), -1).min())
    # bf16 noise floor: HF vs itself with the same inputs in a different batch shape
    ref2b = lm(inputs_embeds=torch.cat([x_all[None, :2 * T]] * 2), position_ids=torch.arange(2 * T, device=dev)[None, None].expand(3, 2, -1)).last_hidden_state[1]
    errs["hf_self_rel"] = float((ref2b.float() - ref2.float()).norm() / ref2.float().norm())
    print("check", errs, flush=True)
    res["check"] = errs


# ---------------------------------------------------------------------------------------------- M2 vision
@torch.no_grad()
def m2(hf, res):
    from speed.engine import FastVision, fp8_language_model_
    import copy
    visual = hf.model.visual
    dev = next(visual.parameters()).device
    out = []
    variants = [("bf16", visual)]
    try:
        v8 = copy.deepcopy(visual)
        n = fp8_language_model_(v8, skip=())
        variants.append(("fp8", v8))
        print("fp8 vision linears", n, flush=True)
    except Exception as e:
        res["m2_fp8_error"] = repr(e)[:300]
    for prec, vis in variants:
        for graphs in (True, False):
            fv = FastVision(vis, graphs=graphs, optimized_singleton=True, fused_rotary=True)
            for side in (224, 256, 448):
                g = side // 16
                grid = torch.tensor([[1, g, g]])
                f_prev = torch.rand(1, 3, side, side, device=dev) * 2 - 1
                f_cur = torch.rand(1, 3, side, side, device=dev) * 2 - 1
                single = patchify(torch.stack([f_cur, f_cur], 1)).to(torch.bfloat16)        # image mode: frame x2
                sliding = patchify(torch.stack([f_prev, f_cur], 1)).to(torch.bfloat16)      # (t-1, t) tubelet
                for mode, px in (("single", single), ("sliding_pair", sliding)):
                    try:
                        st = timeit(lambda: fv(px, grid), n=100, warm=10)
                        tok = fv(px, grid).shape[0]
                    except Exception as e:
                        st, tok = dict(error=repr(e)[:300]), None
                    row = dict(prec=prec, graphs=graphs, side=side, mode=mode, tokens=tok, patches=g * g, **st)
                    print("m2", row, flush=True)
                    out.append(row)
            # HF eager reference (no engine fusions)
            if graphs:
                for side in (256,):
                    g = side // 16
                    px = torch.randn(g * g, 1536, device=dev, dtype=torch.bfloat16)
                    grid = torch.tensor([[1, g, g]], device=dev)
                    st = timeit(lambda: vis(px, grid_thw=grid), n=50)
                    out.append(dict(prec=prec, graphs=False, side=side, mode="hf_eager_reference", **st))
                    print("m2", out[-1], flush=True)
    res["m2"] = out


# ---------------------------------------------------------------------------------------------- M3 / M4
@torch.no_grad()
def m3(hf, res, fp8_too=True):
    D = hf.config.text_config.hidden_size
    heads_embed = hf.model.language_model.embed_tokens.weight
    out = []
    precs = ["bf16", "fp8"] if fp8_too else ["bf16"]
    for prec in precs:
        slm = StreamLM(hf, fp8=(prec == "fp8")) if prec == "bf16" else None
        if prec == "fp8":
            import copy
            slm = StreamLM(copy.deepcopy(hf), fp8=True)
        heads = Readout(D, heads_embed, slm.dev)
        for T in (64, 128, 256):
            for M in (0, 8, 32):
                for graphs in (True, False):
                    pl = Pipeline(slm, heads, 1, T, M, 20, ARGS.window, graphs=graphs)
                    try:
                        st = timeit(lambda: pl(), n=60 if graphs else 30)
                    except Exception as e:
                        st = dict(error=repr(e)[:400])
                    row = dict(prec=prec, T=T, M=M, Tq=20, W=ARGS.window, graphs=graphs, tokens=T + 20 * M, **st)
                    print("m3", row, flush=True)
                    out.append(row)
                    pl.release()
        if prec == "fp8":
            del slm
            torch.cuda.empty_cache()
        if prec == "bf16":
            slm.alloc(1, 1, 64, 0, 20)
            sb = slm.state_bytes()
            res["state_bytes_per_stream"] = dict(gdn_state=sb["gdn_state"], conv_tail=sb["conv_tail"],
                                                 kv_per_token=sb["kv_window"])
    res["m3"] = out


@torch.no_grad()
def profile_breakdown(hf, res, T=64, M=8, B=1):
    """Kernel-time breakdown of one eager step (CUDA time per category)."""
    from torch.profiler import profile, ProfilerActivity
    D = hf.config.text_config.hidden_size
    slm = StreamLM(hf)
    heads = Readout(D, hf.model.language_model.embed_tokens.weight, slm.dev)
    pl = Pipeline(slm, heads, B, T, M, 20, ARGS.window, graphs=False)
    for _ in range(5):
        pl()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            pl()
        torch.cuda.synchronize()
    cats = {}
    n_k = 0
    for e in prof.key_averages():
        if e.device_type is None or getattr(e, "device_time_total", 0) == 0:
            continue
        name = e.key.lower()
        if any(s in name for s in ("fmha", "attention", "flash_fwd", "efficient")):
            c = "attention"
        elif any(s in name for s in ("gemm", "cutlass", "sm90", "sm100", "sm120", "nvjet", "cublas", "matmul", "_mm")):
            c = "gemm"
        elif any(s in name for s in ("chunk", "delta", "fwd_kernel", "recompute", "wy", "solve_tril", "kkt", "gdn", "l2norm")):
            c = "gdn"
        else:
            c = "elementwise/other"
        cats.setdefault(c, 0.0)
        cats[c] += e.device_time_total / 1e3 / 10
        n_k += e.count / 10
    top = sorted(((e.key, e.device_time_total / 1e3 / 10) for e in prof.key_averages() if getattr(e, "device_time_total", 0)),
                 key=lambda t: -t[1])[:25]
    key = f"profile_T{T}_M{M}" + (f"_B{B}" if B > 1 else "")
    res[key] = dict(ms_per_step=cats, kernels_per_step=n_k, top=top)
    print("profile", key, cats, n_k, flush=True)
    pl.release()


@torch.no_grad()
def m4(hf, res):
    """Attention time per frame vs KV window. (a) isolated masked SDPA of one layer x n_attn layers,
    (b) the no-mask flash lower bound, (c) whole graph step at that window."""
    cfg = hf.config.text_config
    D = cfg.hidden_size
    slm = StreamLM(hf)
    heads = Readout(D, hf.model.language_model.embed_tokens.weight, slm.dev)
    H, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = H // Hkv
    T, M, Tq = 64, 8, 20
    R = T + M * Tq
    out = []
    for W in (8192, 15360, 30720, 115712):
        k = torch.randn(1, Hkv, W + R, hd, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(1, Hkv, G * R, hd, device="cuda", dtype=torch.bfloat16)
        slm.alloc(1, W, T, M, Tq)
        mask = slm.mask
        a = timeit(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask), n=100)
        b = timeit(lambda: F.scaled_dot_product_attention(q, k, v), n=100)
        # frame only (N=0): T rows
        qf = q[:, :, :G * T].contiguous()
        slm.alloc(1, W, T, 0, Tq)
        mf = slm.mask
        kf, vf = k[:, :, :W + T].contiguous(), v[:, :, :W + T].contiguous()
        c = timeit(lambda: F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mf), n=100)
        from speed.video_p1.stream import window_attention
        kw, vw = k[:, :, :W].contiguous(), v[:, :, :W].contiguous()
        kl, vl = k[:, :, W:].contiguous(), v[:, :, W:].contiguous()
        slm.alloc(1, W, T, M, Tq)
        ml = slm.mask[..., W:]

        def split():
            o1, l1 = window_attention(q, kw, vw, 0.0625)
            s2 = torch.matmul(q.float(), kl.float().transpose(-1, -2)).masked_fill(~ml, float("-inf"))
            l2 = torch.logsumexp(s2, -1)
            o2 = torch.matmul(torch.exp(s2 - l2[..., None]), vl.float())
            lt = torch.logaddexp(l1, l2)
            return (o1.float() * torch.exp(l1 - lt)[..., None] + o2 * torch.exp(l2 - lt)[..., None]).to(q.dtype)
        sp = timeit(split, n=100)
        wo = timeit(lambda: window_attention(q, kw, vw, 0.0625), n=100)
        del kw, vw
        del k, v
        pl = Pipeline(slm, heads, 1, T, M, Tq, W, graphs=True)
        st = timeit(lambda: pl(), n=60)
        pl.release()
        slm.attn_impl = "sdpa_mask"
        pl = Pipeline(slm, heads, 1, T, M, Tq, W, graphs=True)
        stm = timeit(lambda: pl(), n=40)
        pl.release()
        slm.attn_impl = "split"
        row = dict(W=W, step_graph_masked_p50=stm["p50"], n_attn_layers=slm.n_attn, rows=R,
                   attn_masked_layer_ms=a["p50"], attn_masked_per_frame_ms=a["p50"] * slm.n_attn,
                   attn_nomask_layer_ms=b["p50"], attn_nomask_per_frame_ms=b["p50"] * slm.n_attn,
                   attn_frame_only_per_frame_ms=c["p50"] * slm.n_attn,
                   attn_split_layer_ms=sp["p50"], attn_split_per_frame_ms=sp["p50"] * slm.n_attn,
                   attn_window_kernel_per_frame_ms=wo["p50"] * slm.n_attn,
                   step_graph_p50=st["p50"], step_graph_p99=st["p99"],
                   kv_bytes_per_stream=W * slm.n_attn * 2 * Hkv * hd * 2)
        print("m4", row, flush=True)
        out.append(row)
    res["m4"] = out


# ---------------------------------------------------------------------------------------------- M9
class Frontend:
    """NVDEC decode (torchcodec) + GPU preprocess (resize, normalise, sliding pair, patchify) for B streams."""

    def __init__(self, path, B, side_hw, dev="cuda"):
        from torchcodec.decoders import VideoDecoder
        self.decs = [VideoDecoder(path, device="cuda", seek_mode="approximate") for _ in range(B)]
        self.n = self.decs[0].metadata.num_frames
        self.B, self.hw = B, side_hw
        self.prev = None
        self.i = 0
        self.mean = 0.5
        self.std = 0.5

    def decode(self):
        fr = [d[(self.i + 7 * j) % self.n] for j, d in enumerate(self.decs)]   # sequential index per stream
        self.i += 1
        return torch.stack(fr)                                                 # [B, 3, 1080, 1920] uint8 cuda

    def prep(self, fr):
        x = F.interpolate(fr.float(), size=self.hw, mode="bilinear", antialias=True, align_corners=False)
        x = (x / 255.0 - self.mean) / self.std
        prev = x if self.prev is None else self.prev
        self.prev = x
        return patchify(torch.stack([prev, x], 1)).to(torch.bfloat16)


@torch.no_grad()
def m9(hf, res, video, configs):
    from speed.engine import FastVision
    D = hf.config.text_config.hidden_size
    out = []
    for cfgname, fp8, T, hw, M, W, Bs in configs:
        if fp8:
            import copy
            hfm = copy.deepcopy(hf)
        else:
            hfm = hf
        slm = StreamLM(hfm, fp8=fp8)
        heads = Readout(D, hfm.model.language_model.embed_tokens.weight, slm.dev)
        fv = FastVision(hfm.model.visual, graphs=True, optimized_singleton=True, fused_rotary=True)
        for B in Bs:
            try:
                row = run_stream(slm, heads, fv, video, B, T, hw, M, W)
            except Exception as e:
                import traceback
                traceback.print_exc()
                row = dict(error=repr(e)[:400])
            row.update(config=cfgname, fp8=fp8, B=B, T=T, hw=hw, M=M, W=W)
            print("m9", {k: v for k, v in row.items() if k != "trace"}, flush=True)
            out.append(row)
            torch.cuda.empty_cache()
        del slm, heads, fv
        if fp8:
            del hfm
        torch.cuda.empty_cache()
    res.setdefault("m9", []).extend(out)


def run_stream(slm, heads, fv, video, B, T, hw, M, W, n_frames=900, fps=30.0):
    pl = Pipeline(slm, heads, B, T, M, 20, W, graphs=True)
    fe = Frontend(video, B, hw)
    gh, gw = hw[0] // 16, hw[1] // 16
    grid = torch.tensor([[1, gh, gw]] * B)

    def one():
        e = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        e[0].record()
        fr = fe.decode()
        e[1].record()
        px = fe.prep(fr)
        e[2].record()
        feats = fv(px, grid)
        e[3].record()
        o = pl(feats)
        e[4].record()
        host = o.cpu()                                                         # answers reach the host
        return e, host

    for _ in range(30):
        one()
    torch.cuda.synchronize()
    lat, stage = [], {k: [] for k in ("decode", "prep", "vit", "lm_heads", "d2h_sync")}
    period = 1.0 / fps
    t0 = time.perf_counter() + 0.05
    backlog = 0
    for i in range(n_frames):
        arrive = t0 + i * period
        while time.perf_counter() < arrive:
            pass
        e, _ = one()
        done = time.perf_counter()
        e[4].synchronize()
        lat.append((done - arrive) * 1e3)
        stage["decode"].append(e[0].elapsed_time(e[1]))
        stage["prep"].append(e[1].elapsed_time(e[2]))
        stage["vit"].append(e[2].elapsed_time(e[3]))
        stage["lm_heads"].append(e[3].elapsed_time(e[4]))
        backlog = max(backlog, (done - arrive) / period)
    # unpaced: back-to-back service time (what one frame costs when nothing waits)
    svc = []
    for _ in range(200):
        t = time.perf_counter()
        one()
        svc.append((time.perf_counter() - t) * 1e3)
    pl.release()
    return dict(e2e=stats(lat), service=stats(svc), stages={k: stats(v) for k, v in stage.items() if v},
                keeps_up=bool(np.percentile(lat, 99) < 1000 * period * 2 and lat[-1] < 1000 * period * 3),
                max_backlog_frames=float(backlog))


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tests", default="check,m2,m3,profile,m4")
    p.add_argument("--window", type=int, default=15360)
    p.add_argument("--video", default="/content/v1080.mp4")
    p.add_argument("--m9", default="A")
    p.add_argument("--streams", default="1,4,8,16")
    p.add_argument("--no-fp8", action="store_true")
    p.add_argument("--out", required=True)
    ARGS = p.parse_args()
    res = dict(model=ARGS.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               python=platform.python_version(), window=ARGS.window)
    try:
        import transformers, fla
        res["transformers"], res["fla"] = transformers.__version__, fla.__version__
    except Exception:
        pass
    if os.path.exists(ARGS.out):
        res = json.load(open(ARGS.out)) | {k: v for k, v in res.items()}
    hf = load(ARGS.model)
    tests = ARGS.tests.split(",")
    Bs = [int(b) for b in ARGS.streams.split(",")]

    def save():
        os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
        json.dump(res, open(ARGS.out, "w"), indent=1)

    for t in tests:
        t0 = time.time()
        if t == "check":
            check(hf, res)
        elif t == "m2":
            m2(hf, res)
        elif t == "m3":
            m3(hf, res, fp8_too=not ARGS.no_fp8)
        elif t == "profile":
            for T_, M_, B_ in ((64, 0, 1), (64, 8, 1), (64, 32, 1), (128, 6, 1), (64, 8, 4)):
                profile_breakdown(hf, res, T_, M_, B_)
        elif t == "m4":
            m4(hf, res)
        elif t == "m9":
            cfgs = []
            for c in ARGS.m9.split(","):
                if c == "A":
                    cfgs.append(("A_bf16", False, 64, (256, 256), 8, 15360, Bs))
                elif c == "A8":
                    cfgs.append(("A_fp8", True, 64, (256, 256), 8, 15360, Bs))
                elif c == "B":
                    cfgs.append(("B_bf16", False, 128, (256, 512), 6, 29696, Bs))
                elif c == "B8":
                    cfgs.append(("B_fp8", True, 128, (256, 512), 6, 29696, Bs))
            m9(hf, res, ARGS.video, cfgs)
        res.setdefault("elapsed_s", {})[t] = time.time() - t0
        save()
    print("done", ARGS.out, flush=True)


if __name__ == "__main__":
    main()
