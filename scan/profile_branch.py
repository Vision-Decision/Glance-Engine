"""Kernel attribution for warmed, graph-replayed item 4 workloads on engopt."""

import argparse
from collections import Counter
from functools import partial
import json

import torch
from scan.bench_scan import Bench, make_clips
from scan.data import load_images, QUERIES
from speed import workload as W
from speed.bench_glance import Runner
from speed.profile_glance import cat


def profile_calls(args):
    result = {}

    def sample(name, call):
        for _ in range(3):
            call()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            call()
        categories, kernels = Counter(), Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                ms = e.time_range.elapsed_us() / 1000
                categories[cat(e.name)] += ms
                kernels[e.name] += ms
        result[name] = dict(categories_ms=dict(categories.most_common()), kernels_ms=kernels.most_common())
        print(name, result[name]["categories_ms"], flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    eng = Bench(args).to_engine()
    for tv in (640, 256, 64, "clip"):
        if tv == "clip":
            items = make_clips(args, 256)
        else:
            items = [x["img"] for x in load_images(args.prep, args.image_root, 256, tv, frame=tv != 640)]
        bank = eng.encode(items, chunk=16 if tv in (640, 256) else 32)
        for n in (64, 256):
            sub = bank.head(n)
            for kind in ("box", "noul") if tv == 640 else ("noul",) if tv == "clip" else ("box",):
                q = (
                    dict(type="box", text=QUERIES[0])
                    if kind == "box"
                    else dict(type="noul", instr=f"Is there {QUERIES[0]} in the image?")
                )
                sample(f"{tv}_{n}_{kind}", partial(eng.query, sub, q))
            eng.free_graphs(sub)
            del sub
        del bank
        eng.free_graphs()
    del eng
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    r = Runner(args)
    r.set_mode("eng_fused")
    imgs = W.load_coco(args.prep, args.image_root, 12)
    for n in (1, 8, 32):
        for kind in ("box",) if n == 1 else ("box", "mixed"):
            qs = [W.questions(imgs, 0, n, kind)]
            sample(f"single_{n}_{kind}", lambda: r.call([imgs[0]["img"]], qs, Nv=640))
            r.call([imgs[0]["img"]], [W.questions(imgs, 0, 1, "box")], Nv=640, keep_prefix=True)
            sample(f"followup_{n}_{kind}", lambda: r.followup(qs))


def measure_stages(a):
    """Time disjoint GPU stage ranges on graph replay. Keep the latency sweep uninstrumented."""
    from contextlib import ExitStack
    from statistics import median
    from unittest.mock import patch

    result = {}

    def sample(name, eng, run, clear):
        run()
        clear()
        events = []
        inside = False

        def instrument(obj, attr, label, stack):
            original = getattr(obj, attr, None)
            if original is None:
                return

            def call(*args, **kwargs):
                nonlocal inside
                if inside or not torch.cuda.is_current_stream_capturing():
                    return original(*args, **kwargs)
                start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                inside = True
                try:
                    start.record()
                    val = original(*args, **kwargs)
                    end.record()
                finally:
                    inside = False
                events.append((label, start, end))
                return val

            stack.enter_context(patch.object(obj, attr, call))

        with ExitStack() as stack:
            if getattr(a, "banks_only", False):
                from scan.bank import StoredBank
                for attr in ("layer", "attention"):
                    instrument(StoredBank, attr, "bank decode", stack)
            for attr in ("_mix_gdn_prefix", "_mix_gdn_q", "_mix_gdn_branch", "branch_output", "question_branch"):
                instrument(eng, attr, "GDN", stack)
            for attr in ("branch_attention", "_split_attn"):
                instrument(eng, attr, "LM attention", stack)
            instrument(torch.nn.functional, "scaled_dot_product_attention", "LM attention", stack)
            for attr in ("_branch_heads", "heads_forward"):
                instrument(eng, attr, "heads", stack)
            instrument(eng, "_mlp_update", "LM dense", stack)
            for layer, kind in zip(eng.layers, eng.types):
                mods = (
                    (layer.linear_attn._in, layer.linear_attn.out_proj)
                    if kind == "linear_attention"
                    else (layer.self_attn._qkv, layer.self_attn.o_proj)
                )
                for mod in (*mods, layer._gu, layer.mlp.down_proj):
                    instrument(mod, "forward", "LM dense", stack)
            run()
        assert events, name
        samples = []
        for _ in range(7):
            run()
            torch.cuda.synchronize()
            times = Counter()
            for stage, t0, t1 in events:
                times[stage] += t0.elapsed_time(t1)
            samples.append(dict(times))
        result[name] = {k: median(s[k] for s in samples) for k in samples[0]}
        print(name, result[name], flush=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
        # A captured external event must outlive every replay of its graph.
        # The next workload can warm up using this same shape.
        clear()

    def clear(owner):
        owner._graphs.clear()
        owner._pool = None

    eng = Bench(a).to_engine()
    clip_shapes = [(4, 96), (4, 48), (2, 96)] if a.banks_only else [(4, 96)]
    for tv in [640, 256, 64, *clip_shapes]:
        if isinstance(tv, tuple):
            frames, eng.clip_max_vis = tv
            items = make_clips(a, 256, frames, eng.clip_max_vis)
            prefix, kinds = f"f{frames}_v{eng.clip_max_vis}", ("noul",)
        else:
            items = [r["img"] for r in load_images(a.prep, a.image_root, 256, tv, frame=tv != 640)]
            prefix, kinds = f"Tv{tv}", ("box", "noul") if tv == 640 else ("box",)
        bank = eng.encode(items, chunk=16 if tv in (640, 256) else 32)
        for n in (64, 256):
            sub = bank.head(n)
            for kind in kinds:
                q = (
                    dict(type="box", text=QUERIES[0])
                    if kind == "box"
                    else dict(type="noul", instr=f"Is there {QUERIES[0]} in the image?")
                )
                sample(f"{prefix}_N{n}_{kind}", eng, partial(eng.query, sub, q), partial(clear, sub))
            eng.free_graphs(sub)
        del sub, bank
        eng.free_graphs()
    if getattr(a, "banks_only", False):
        return
    del eng
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    r = Runner(a)
    r.set_mode("eng_fused")
    imgs = W.load_coco(a.prep, a.image_root, 12)
    for n in (1, 8, 32):
        for kind in ("box",) if n == 1 else ("box", "mixed"):
            qs = [W.questions(imgs, 0, n, kind)]
            sample(f"bs1_{kind}_N{n}", r.eng, partial(r.call, [imgs[0]["img"]], qs, Nv=640), partial(clear, r.eng))
            r.call([imgs[0]["img"]], [W.questions(imgs, 0, 1, "box")], Nv=640, keep_prefix=True)
            sample(f"followup_{kind}_N{n}", r.eng, partial(r.followup, qs), partial(clear, r.eng))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--banks-only", action="store_true", help="time cached scan stages including storage decode")
    p.add_argument("--stage", choices=("profile", "micro", "questions", "clip", "stages"), required=True)
    p.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    p.add_argument("--image_root", default="/content/data/coco/train2014")
    a = p.parse_args()
    if a.stage == "stages":
        measure_stages(a)
    elif a.stage == "profile":
        profile_calls(a)
    elif a.stage == "clip":
        from scan.branch_micro import clip_controls

        clip_controls(a)
    elif a.stage == "questions":
        from scan.branch_micro import compare_questions

        compare_questions(a)
    else:
        from scan.branch_micro import micro

        micro(a)
