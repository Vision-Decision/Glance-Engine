"""Profile E1b graph behaviour and isolated eager LM kernels at 4k..32k nominal visual tokens.

Run on an idle GPU: python -m scan.profile_fresh --ckpt CKPT --out FILE --fresh_impl reference|varlen
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from dv.data.mixed import collate_mixed
from scan.bench_scan import Bench
from scan.data import QUERIES, load_images
from scan.varlen import FreshPack, packed_forward


def kernel_group(name):
    n = name.lower()
    if "conv" in n:
        return "conv"
    if any(x in n for x in ("gated_delta", "chunk_", "cumsum", "wy_fast", "solve_tril", "l2norm")):
        return "delta_rule"
    if any(x in n for x in ("flash", "fmha", "attention", "cudnn_generated")):
        return "attention"
    if any(x in n for x in ("gemm", "cutlass", "cublas", "matmul", "xmma", "nvjet", "gemv")):
        return "gemm"
    if any(x in n for x in ("index", "scatter", "gather", "copy", "memcpy", "memset", "catarray")):
        return "data_movement"
    return "other"


def memory():
    stats = torch.cuda.memory_stats()
    return {k: stats[k] for k in ("allocated_bytes.all.current", "reserved_bytes.all.current",
                                  "allocated_bytes.all.peak", "num_alloc_retries", "num_device_alloc")}


@torch.no_grad()
def profile_lm(eng, ims, q):
    eng.free_graphs()
    eng.fv.graphs = False  # vision is outside the LM trace and need not pin a graph pool
    if eng.fresh_varlen:
        batch = collate_mixed(eng._items(ims, [[q]] * len(ims), None), eng.tok, pad_to=1)
        feats = eng.feats(batch)
        pack = FreshPack(batch, eng.dev)
        def run():
            return packed_forward(eng, pack, feats)
    else:
        batch, e, host = eng.build(ims, [[q]] * len(ims))
        inp = eng.to_device(e, host, eng.feats(batch))
        def run():
            return eng._core(inp, host)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        if len(samples) < 5:
            del result
    after = memory()
    # Count unique backing storages. A small tail or V view can pin an entire fused projection.
    retained = 0
    if not eng.fresh_varlen:
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for c in result[1] for t in c}
        retained = sum(storage.values())
    del result
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        run()
        torch.cuda.synchronize()
    kernels, groups = defaultdict(float), defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            duration = ev.time_range.elapsed_us() / 1000
            kernels[ev.name] += duration
            groups[kernel_group(ev.name)] += duration
    cpu_ops = [ev for ev in prof.key_averages() if ev.device_type == torch.autograd.DeviceType.CPU]
    alloc = {ev.key: ev.self_device_memory_usage for ev in cpu_ops if ev.self_device_memory_usage > 0}
    op_times = {ev.key: dict(calls=ev.count, self_cuda_ms=ev.self_device_time_total / 1000,
                             self_cpu_ms=ev.self_cpu_time_total / 1000) for ev in cpu_ops}
    return dict(eager_ms=median(samples), eager_samples_ms=samples, before=before, after=after, retained_cache_bytes=retained,
                prefix_tokens=int(batch["p_valid"].sum()), question_tokens=int(batch["q_valid"].sum()),
                kernel_ms=dict(sorted(kernels.items(), key=lambda it: -it[1])), group_ms=dict(groups),
                positive_net_allocation_bytes_by_op=alloc, operator_times=op_times)


def timeline(eng, pool, n):
    """Same question rotation and graph-cache lifetime as E1b. Count LM captures inside each timed call."""
    captures = []
    orig = eng._capture
    def capture(*args):
        t = time.perf_counter()
        result = orig(*args)
        captures.append((time.perf_counter() - t) * 1000)
        return result

    samples = []
    with patch.object(eng, "_capture", capture):
        for warm, i in [(True, j) for j in range(4)] + [(False, j) for j in range(10)]:
            ims = [pool[(i * n + j) % len(pool)] for j in range(n)]
            q = dict(type="noul", instr=f"Is there {QUERIES[i]} in the image?")
            captures.clear()
            torch.cuda.reset_peak_memory_stats()
            t = {}
            eng.fresh(ims, q, timings=t)
            samples.append(dict(warmup=warm, query=i, **t, lm_capture_ms=sum(captures), captures=len(captures),
                                live_graphs=len(eng._graphs), chunks=eng.last_fresh_chunks, memory=memory()))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh_impl", choices=["reference", "varlen"], required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    a = ap.parse_args()
    eng = Bench(a).to_engine()
    pool = [r["img"] for r in load_images(a.prep, a.image_root, 256, 128, frame=True)]
    if len(pool) != 256:
        raise ValueError(f"Expected 256 E1b images, got {len(pool)}")
    res = dict(impl=a.fresh_impl, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    q = dict(type="noul", instr=f"Is there {QUERIES[0]} in the image?")
    for part in ("timeline", "eager"):
        res[part] = {}
        for n in (32, 64, 128, 256):
            print(part, n, flush=True)
            try:
                row = timeline(eng, pool, n) if part == "timeline" else profile_lm(eng, pool[:n], q)
            except torch.cuda.OutOfMemoryError as exc:
                row = {"oom": str(exc), "memory": memory()}
            res[part][n] = row
            a.out.write_text(json.dumps(res, indent=1))
            gc.collect()
            if "oom" in row:
                eng.free_graphs()


if __name__ == "__main__":
    main()
