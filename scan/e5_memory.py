"""E5: memory tiers, long videos and bank precision for the clip scan (Glance-Video short readout).

python -m scan.e5_memory --ckpt CKPT --data /content/charades_pilot2 --out OUT.json [--n_long 1800]

1. Bank size per clip, by component (GDN recurrent state, conv tail, attention K/V, visual hidden states, box keys).
2. One query over N cached clips, GPU-resident, N = 64 / 256 / 1024 (if it fits).
3. Host tier: the bank lives in pinned host memory; each query streams it to the GPU in chunks of `chunk` clips
   (copy of chunk i+1 overlaps the query of chunk i on a second stream) -> latency for N up to n_long (a one-hour
   video at 2 s clips is ~1,800 clips).
4. Precision: GDN state stored in bf16 (half the bank) or fp8 e4m3 + per-head scale, upcast for the query; scores vs the
   fp32-state bank on real Charades clips and several queries (max / mean |dP|, decision agreement).
5. Throughput: Q sessions (different videos) interleaved, queries per second.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

from scan.engine import ImageBank

QUERIES = ["the person opens the door", "someone is drinking from a cup", "a person sits down on a chair",
           "the person turns on the light", "someone is holding a phone", "a person is laughing"]


def sync():
    torch.cuda.synchronize()


def bank_map(b, fn):
    cache = [tuple(fn(x) for x in c) for c in b.cache]
    return ImageBank(cache, b.types, fn(b.p_valid), fn(b.q_start), fn(b.vis), fn(b.vis_valid), fn(b.centers), fn(b.ck),
                     fn(b.grid_hw))


def components(b):
    out = {}
    for li, lt in enumerate(b.types):
        for j, x in enumerate(b.cache[li]):
            k = (f"gdn_{'conv' if x.dim() == 3 and x.shape[-1] <= 8 else 'state'}" if lt == "linear_attention"
                 else ("attn_k" if j == 0 else "attn_v"))
            out[k] = out.get(k, 0) + x.numel() * x.element_size()
    out["vis_hidden"] = b.vis.numel() * b.vis.element_size()
    out["box_keys"] = b.ck.numel() * b.ck.element_size()
    return {k: v / b.N / 2 ** 20 for k, v in out.items()}


def time_query(eng, bank, q, n=10):
    eng.query(bank, q); sync()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); eng.query(bank, q); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def scores(eng, bank, q):
    tl = eng.query(bank, q)["typed_logits"]
    return torch.softmax(tl[:, :2], -1)[:, 1].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_long", type=int, default=1800)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    from scan.pilot_charades import load_glance
    eng, _ = load_glance(a.ckpt)
    eng.dt, eng.clip_max_vis = 0.5, 96
    meta = json.load(open(f"{a.data}/meta.json"))
    vids = sorted(meta["videos"])
    clips = []
    for v in vids:                                     # real clips, 4 frames each, stride 2 (2 fps)
        n = meta["videos"][v]["n_frames"]
        fr = [f"{a.data}/frames/{v}/{k:03d}.jpg" for k in range(n)]
        clips += [fr[i:i + 4] for i in range(0, n - 3, 2)]
        if len(clips) >= a.n_long:
            break
    clips = clips[: a.n_long]
    load = lambda cs: [[Image.open(p).convert("RGB") for p in c] for c in cs]
    q = lambda t: dict(type="noul", instr=t, block="short")
    res = dict(ckpt=a.ckpt, gpu=torch.cuda.get_device_name(), n_long=len(clips))
    # 1-2: GPU-resident banks
    sync(); t0 = time.perf_counter()
    b256 = eng.encode(load(clips[:256]), chunk=64); sync()
    res["encode_ms_per_clip"] = (time.perf_counter() - t0) * 1e3 / 256
    res["bank_MB_per_clip"] = components(b256)
    res["bank_MB_per_clip_total"] = sum(res["bank_MB_per_clip"].values())
    res["gpu_resident_ms"] = {str(n): time_query(eng, b256.head(n), q(QUERIES[0])) for n in (64, 256)}
    print(json.dumps(res), flush=True)
    # 3: host tier, streamed in chunks
    host = [bank_map(b256, lambda x: x.to("cpu").pin_memory())]
    del b256; torch.cuda.empty_cache()
    for s in range(256, len(clips), 256):
        b = eng.encode(load(clips[s:s + 256]), chunk=64)
        host.append(bank_map(b, lambda x: x.to("cpu").pin_memory())); del b
    torch.cuda.empty_cache()
    copy_stream = torch.cuda.Stream()

    def host_query(n_chunks, text):
        out, nxt = [], None
        with torch.cuda.stream(copy_stream):
            nxt = bank_map(host[0], lambda x: x.to("cuda", non_blocking=True))
        for i in range(n_chunks):
            torch.cuda.current_stream().wait_stream(copy_stream)
            cur = nxt
            if i + 1 < n_chunks:
                with torch.cuda.stream(copy_stream):
                    nxt = bank_map(host[i + 1], lambda x: x.to("cuda", non_blocking=True))
            out.append(scores(eng, cur, q(text)))
        return np.concatenate(out)
    res["host_tier_ms"] = {}
    for nc in sorted({1, 4, len(host)}):
        host_query(nc, QUERIES[1]); sync()
        ts = []
        for r in range(5):
            sync(); t0 = time.perf_counter(); host_query(nc, QUERIES[r % len(QUERIES)]); sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        res["host_tier_ms"][str(sum(h.N for h in host[:nc]))] = float(np.median(ts))
        print("host", nc, res["host_tier_ms"], flush=True)
    res["host_bank_GB"] = sum(h.nbytes() for h in host) / 2 ** 30
    # 4: precision of the stored GDN state (bf16, fp8 + per-head scale) on the first chunk
    ref_bank = bank_map(host[0], lambda x: x.to("cuda"))

    def quant(b, kind):
        def f(li, x):
            if b.types[li] != "linear_attention" or x.dtype != torch.float32 or x.dim() != 4:
                return x
            if kind == "bf16":
                return x.to(torch.bfloat16).float()
            s = x.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8) / 448.0
            return (x / s).to(torch.float8_e4m3fn).float() * s
        cache = [tuple(f(li, x) for x in c) for li, c in enumerate(b.cache)]
        return ImageBank(cache, b.types, b.p_valid, b.q_start, b.vis, b.vis_valid, b.centers, b.ck, b.grid_hw)
    res["state_precision"] = {}
    for kind in ("bf16", "fp8"):
        qb = quant(ref_bank, kind)
        d, agree = [], []
        for t in QUERIES:
            p0, p1 = scores(eng, ref_bank, q(t)), scores(eng, qb, q(t))
            d.append(np.abs(p0 - p1)); agree.append(((p0 > 0.5) == (p1 > 0.5)).mean())
        d = np.concatenate(d)
        res["state_precision"][kind] = dict(max_dP=float(d.max()), mean_dP=float(d.mean()), decision_agree=float(np.mean(agree)))
        print(kind, res["state_precision"][kind], flush=True)
    # 5: throughput, 4 sessions x 256 clips, interleaved queries (GPU-resident)
    sess = [ref_bank] + [bank_map(host[i % len(host)], lambda x: x.to("cuda")) for i in range(1, 4)]
    sync(); t0 = time.perf_counter(); nq = 0
    for r in range(10):
        for b in sess:
            eng.query(b, q(QUERIES[r % len(QUERIES)])); nq += 1
    sync()
    res["throughput_qps_4x256"] = nq / (time.perf_counter() - t0)
    json.dump(res, open(a.out, "w"), indent=1)
    print("E5_DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
