"""E10: fresh mode (one question over N NEW images, nothing cached): where the time goes and what can be cut.

python -m scan.e10_fresh --ckpt CKPT --out OUT.json [--parts peak,profile,sweep,quality]

peak     measured dense GEMM peak on this GPU (bf16 and FP8 e4m3 via torch._scaled_mm), 8192^3
profile  N=128, Tv=128, yes/no readout: engine timings (host prep / vision tower / LM), torch.profiler CUDA time by kernel
         family (GEMM, attention, GDN / Triton, other) vs wall time (= launch / Python overhead), and the FLOPs floor:
         2 * params * tokens for the LM linears + 2 * params * patches for the vision tower, at the measured peak.
sweep    latency and images/s for N in 32/64/128/256 over configurations:
           Tv 32/64/128/256 x {bf16, FP8 LM} x {image-first, query-first (Tv128 only)} x prefill chunk (16k / 32k tokens)
quality  per configuration (Tv, FP8, query-first): gRefCOCO val yes/no 'Is there {expression} in the image?' on up to
         `n_img` images (all their expressions): AUROC of P(yes) for present vs absent, and decision agreement with the
         reference configuration (Tv256, bf16, image-first). Same images and questions for every configuration.
"""
import argparse, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import speed.engine as SE
from scan.bench_scan import Bench, auroc
from scan.data import load_images, QUERIES


def sync():
    torch.cuda.synchronize()


def peak():
    out = {}
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    sync(); t0 = time.perf_counter()
    for _ in range(20):
        a @ b
    sync(); out["bf16_tflops"] = 20 * 2 * n ** 3 / (time.perf_counter() - t0) / 1e12
    try:
        a8 = a.to(torch.float8_e4m3fn); b8 = b.to(torch.float8_e4m3fn).t().contiguous().t()
        one = torch.ones((), device="cuda")
        for _ in range(3):
            torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        sync(); t0 = time.perf_counter()
        for _ in range(20):
            torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        sync(); out["fp8_tflops"] = 20 * 2 * n ** 3 / (time.perf_counter() - t0) / 1e12
    except Exception as e:
        out["fp8_error"] = str(e)[:200]
    return out


def yq(t):
    return dict(type="noul", instr=f"Is there {t} in the image?")


def run_fresh(eng, ims, q, chunk, query_first=False):
    T = {}
    for c0 in range(0, len(ims), chunk):
        t = {}
        eng.fresh(ims[c0:c0 + chunk], q, query_first=query_first, timings=t)
        for k, v in t.items():
            T[k] = T.get(k, 0.0) + v
    return T


def time_cfg(eng, pool, N, Tv, chunk_tokens, qf, n_calls):
    chunk = max(1, chunk_tokens // Tv)
    calls = [([pool[(c * N + j) % len(pool)] for j in range(N)], yq(QUERIES[c % len(QUERIES)])) for c in range(n_calls + 3)]
    Ts = []
    for i, (ims, q) in enumerate(calls):
        sync(); t0 = time.perf_counter()
        T = run_fresh(eng, ims, q, chunk, qf)
        sync(); T["wall"] = (time.perf_counter() - t0) * 1e3
        if i >= 3:
            Ts.append(T)
    r = {f"{k}_ms": float(np.median([t[k] for t in Ts])) for k in Ts[0]}
    r.update(images_per_s=N / (r["wall_ms"] / 1e3), chunks=-(-N // chunk))
    return r


def lm_params(net):
    return sum(p.numel() for n, p in net.lm.named_parameters() if "embed" not in n)


def vit_params(net):
    return sum(p.numel() for p in net.visual.parameters())


def profile(B, a, res):
    eng = B.eng
    N, Tv = 128, 128
    pool = [x["img"] for x in load_images(a.prep, a.image_root, 256, Tv, frame=True)]
    r = time_cfg(eng, pool, N, Tv, a.chunk_tokens, False, 5)
    from torch.profiler import profile as prof, ProfilerActivity
    ims = pool[:N]
    with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        sync(); t0 = time.perf_counter()
        run_fresh(eng, ims, yq(QUERIES[0]), max(1, a.chunk_tokens // Tv))
        sync(); wall = (time.perf_counter() - t0) * 1e3
    fam = {"gemm": 0.0, "attention": 0.0, "gdn_triton": 0.0, "other": 0.0}
    top = []
    for e in p.key_averages():
        ct = getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3
        if ct <= 0 or e.key.startswith("ProfilerStep") or "aten::" in e.key or e.key.startswith("cuda"):
            continue
        k = e.key.lower()
        f = ("gemm" if any(s in k for s in ("gemm", "cutlass", "sm90", "sm100", "sm120", "nvjet", "matmul", "scaled_mm"))
             else "attention" if any(s in k for s in ("flash", "fmha", "attention", "sdpa"))
             else "gdn_triton" if any(s in k for s in ("chunk", "gated", "delta", "triton", "fused_recurrent", "l2norm"))
             else "other")
        fam[f] += ct
        top.append((ct, e.key[:90]))
    top.sort(reverse=True)
    gpu_ms = sum(fam.values())
    toks = N * (Tv + 30)
    patches = N * Tv * 4
    pk = res.get("peak") or peak()
    flops_lm = 2 * lm_params(B.net) * toks
    flops_vit = 2 * vit_params(B.net) * patches
    res["profile"] = dict(N=N, Tv=Tv, timings=r, profiled_wall_ms=wall, gpu_kernel_ms=gpu_ms, kernel_ms_by_family=fam,
                          idle_or_overhead_ms=wall - gpu_ms, top_kernels=[dict(ms=round(c, 2), name=n) for c, n in top[:15]],
                          lm_params=lm_params(B.net), vit_params=vit_params(B.net), lm_tokens=toks, vit_patches=patches,
                          flops_floor_ms_bf16=(flops_lm + flops_vit) / (pk["bf16_tflops"] * 1e12) * 1e3,
                          flops_floor_ms_fp8_lm=(flops_lm / (pk.get("fp8_tflops", pk["bf16_tflops"]) * 1e12) +
                                                 flops_vit / (pk["bf16_tflops"] * 1e12)) * 1e3)
    print("profile", json.dumps(res["profile"])[:3000], flush=True)


def sweep(B, a, res, fp8):
    eng = B.eng
    out = res.setdefault("sweep", {})
    for Tv in (32, 64, 128, 256):
        pool = [x["img"] for x in load_images(a.prep, a.image_root, 256, Tv, frame=True)]
        for qf in ((False, True) if Tv == 128 else (False,)):
            for ct in [int(c) for c in a.sweep_chunks.split(",")]:
                for N in (32, 64, 128, 256):
                    key = f"Tv{Tv}_N{N}_{'fp8' if fp8 else 'bf16'}{'_qfirst' if qf else ''}_chunk{ct // 1024}k"
                    try:
                        out[key] = time_cfg(eng, pool, N, Tv, ct, qf, a.n_calls)
                    except torch.cuda.OutOfMemoryError:
                        out[key] = "OOM"; eng._graphs.clear(); torch.cuda.empty_cache()
                    print("sweep", key, json.dumps(out[key]) if isinstance(out[key], dict) else out[key], flush=True)
                    json.dump(res, open(a.out, "w"), indent=1)
            eng.free_graphs()


def quality(B, a, res, fp8, ref_store):
    eng = B.eng
    out = res.setdefault("quality", {})
    for Tv in (32, 64, 128, 256):
        items = load_images(a.gprep, a.image_root, a.n_img, Tv)
        for qf in ((False, True) if Tv in (128, 256) else (False,)):
            key = f"Tv{Tv}_{'fp8' if fp8 else 'bf16'}{'_qfirst' if qf else ''}"
            ps, ys = [], []
            for it in items:
                for q in it["qs"][:8]:
                    tl = eng.fresh([it["img"]], yq(q["text"]), query_first=qf)["typed_logits"][0]
                    ps.append(float(torch.softmax(tl[:2], -1)[1])); ys.append(bool(q["present"]))
            ps, ys = np.array(ps), np.array(ys)
            if key == "Tv256_bf16":
                ref_store["p"] = ps
            r = dict(n=len(ps), auroc=auroc(ps[ys], ps[~ys]), false_alarm_at_05=float((ps[~ys] >= .5).mean()),
                     miss_at_05=float((ps[ys] < .5).mean()))
            if "p" in ref_store:
                r.update(agree_vs_ref=float(((ps >= .5) == (ref_store["p"] >= .5)).mean()),
                         p_absdiff_mean_vs_ref=float(np.abs(ps - ref_store["p"]).mean()))
            out[key] = r
            print("quality", key, json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="peak,profile,sweep,quality")
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--gprep", default="data/prep/eval_grefcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--chunk_tokens", type=int, default=16384)
    ap.add_argument("--n_calls", type=int, default=8)
    ap.add_argument("--n_img", type=int, default=200)
    ap.add_argument("--no_fp8", action="store_true")
    ap.add_argument("--sweep_chunks", default="4096", help="prefill chunk sizes (tokens) for the sweep")
    a = ap.parse_args()
    SE.GRAPH_MAX_TOKENS = 1 << 16
    SE.GRAPH_MAX_PATCHES = 1 << 18
    parts = a.parts.split(",")
    B = Bench(a)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, ckpt=a.ckpt)
    if "peak" in parts:
        res["peak"] = peak(); print("peak", res["peak"], flush=True)
    ref = {}
    for fp8 in ((False,) if a.no_fp8 else (False, True)):
        B.to_engine(fp8=fp8)
        if "profile" in parts:
            profile(B, a, res) if not fp8 else None
        if "sweep" in parts:
            sweep(B, a, res, fp8)
        if "quality" in parts:
            quality(B, a, res, fp8, ref)
        json.dump(res, open(a.out, "w"), indent=1)
    print("E10_DONE", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
