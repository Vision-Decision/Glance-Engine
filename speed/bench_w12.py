"""Offline W12 export, cold-weight CUDA graph microbenchmarks and paired engine probes."""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from statistics import median

import torch
from torch.nn import functional as F

from speed.bench_glance import Runner
from speed.w12 import PackedWeight, decode_record, encode, projections


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


@torch.no_grad()
def export(args):
    runner = Runner(args)
    runner.set_mode("eng_fused")
    dest = Path(args.weights)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.is_dir()
    manifest = json.loads((dest / "manifest.json").read_text()) if existing else None
    if existing and manifest["version"] != 1:
        raise ValueError("unknown W12 artifact version")
    stats = {}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        folder = Path(tmp) / "encoded"
        folder.mkdir()
        for name, linear in projections(runner.net.lm):
            w = linear.weight
            if existing:
                record = torch.load(dest / f"{name}.pt", map_location=w.device, weights_only=True)
            else:
                record = encode(w)
            exact = torch.equal(decode_record(record).view(torch.int16), w.view(torch.int16))
            if not exact:
                raise ValueError(f"offline decode differs at {name}")
            nbytes = sum(v.nbytes for v in record.values() if torch.is_tensor(v))
            escapes = len(record["values"])
            stats[name] = {
                "shape": list(w.shape),
                "weights": w.numel(),
                "bytes": nbytes,
                "bits_per_weight": 8 * nbytes / w.numel(),
                "escapes": escapes,
                "escape_fraction": escapes / w.numel(),
                "bit_exact": exact,
            }
            if not existing:
                torch.save(
                    {k: v.cpu() if torch.is_tensor(v) else v for k, v in record.items()}, folder / f"{name}.pt"
                )
            if len(stats) % 4 == 0:
                print(name, len(stats), flush=True)
        if existing:
            if stats != manifest["matrices"]:
                raise ValueError("stored W12 statistics differ from the validated records")
        else:
            save(folder / "manifest.json", {"version": 1, "matrices": stats})
            os.rename(folder, dest)
    counts = {k: sum(row[k] for row in stats.values()) for k in ("weights", "bytes", "escapes")}
    save(
        args.out,
        dict(
            matrices=stats,
            **counts,
            bits_per_weight=8 * counts["bytes"] / counts["weights"],
            escape_fraction=counts["escapes"] / counts["weights"],
        ),
    )
    print(counts)


def graph_time(call, repetitions=15):
    # Each replay rotates through a group of distinct layer weights larger than L2.
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(repetitions):
        first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        first.record()
        graph.replay()
        last.record()
        last.synchronize()
        samples.append(first.elapsed_time(last))
    return median(samples)


def records(args, modules):
    for name, mod in modules:
        record = torch.load(Path(args.weights) / f"{name}.pt", map_location=mod.weight.device, weights_only=True)
        packed = PackedWeight(record)
        if not torch.equal(packed.decode().view(torch.int16), mod.weight.view(torch.int16)):
            raise ValueError(f"W12 artifact differs from merged weights at {name}")
        yield name, mod, packed


@torch.no_grad()
def micro(args):
    runner = Runner(args)
    runner.set_mode("eng_fused")
    groups = {}
    for name, mod in projections(runner.net.lm):
        groups.setdefault(tuple(mod.weight.shape), []).append((name, mod))
    result = {}
    torch.manual_seed(11)
    for shape, members in groups.items():
        family = "x".join(map(str, shape))
        # Rotate at least 256 MiB of packed records on this 128 MiB L2 device.
        per_layer = shape[0] * shape[1] * 3 // 2
        count = (256 * 2**20 + per_layer - 1) // per_layer
        group = list(records(args, members[:count]))
        w = group[0][1].weight
        n, k = w.shape
        row = result[family] = {"shape": [n, k], "layers": len(group), "samples": {}}
        for m in (1, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256, 512, 1024):
            x = torch.randn((m, k), dtype=w.dtype, device=w.device)

            def run(method, group=group, x=x):
                for _, mod, packed in group:
                    if method == "bf16":
                        mod(x)
                    elif method == "decode_blas":
                        F.linear(x, packed.decode())
                    elif method.startswith("direct"):
                        packed.multiply(x, split_stages=int(method[6:]))
                    elif method == "chunk64":
                        torch.cat([packed.multiply(part) for part in x.split(64)])
                    else:
                        packed.multiply(x, whole=method == "whole", epilogue=method == "epilogue")

            times = {}
            methods = ["bf16"]
            if m <= 64:
                methods += ["segment", "whole"]
            if args.stage == "micro_direct":
                if m <= 64:
                    methods += ["direct4", "direct8", "direct16"]
                if m in (96, 128):
                    methods += ["segment", "whole", "chunk64"]
            if args.stage == "micro_epilogue" and m <= 128:
                methods = ["bf16", "segment", "epilogue"]
            if m in (8, 64):
                methods += ["decode_blas"]
            for fn in methods:
                times[fn] = graph_time(lambda fn=fn: run(fn)) * 1000 / len(group)
            if args.stage == "micro_direct":

                def reuse_weight(group=group, x=x):
                    for _ in group:
                        group[0][1](x)

                times["bf16_hot"] = graph_time(reuse_weight) * 1000 / len(group)
            if m <= 64:
                baseline = F.linear(x, w)
                diff = (baseline.float() - group[0][2].multiply(x).float()).abs()
                if args.stage == "micro_direct":
                    direct = group[0][2].multiply(x, split_stages=8)
                    direct_diff = (baseline.float() - direct.float()).abs()
                    times["direct_max_error"] = direct_diff.max().item()
                    times["direct_bit_agreement"] = (direct_diff == 0).float().mean().item()
                times.update(max_error=diff.max().item(), bit_agreement=(diff == 0).float().mean().item())
            if args.stage == "micro_epilogue" and m <= 128:
                ref = F.linear(x, w)
                fast = group[0][2].multiply(x, epilogue=True)
                times["epilogue_max_error"] = (ref.float() - fast.float()).abs().max().item()
                times["epilogue_repeatable"] = torch.equal(fast, group[0][2].multiply(x, epilogue=True))
            row["samples"][m] = times
            print(family, m, times, flush=True)
            save(args.out, result)
        del group
        gc.collect()


@torch.no_grad()
def accuracy(args):
    from scan.data import load_images
    from speed import workload as W
    from speed.bench_glance import compare, equiv_outputs

    pool = W.load_coco("data/prep/eval_refcoco_val.jsonl", args.image_root, 60)
    gref = load_images("data/prep/eval_grefcoco_val.jsonl", args.image_root, 60, 640)
    results = {}
    reference = None
    for mode in ("eng_fused", "eng_w12", "eng_fp8"):
        r = Runner(args)
        r.set_mode(mode)
        cold = equiv_outputs(r, pool, len(pool))
        follow, presence = [], []
        for i, image in enumerate(pool):
            img, qs = [image["img"]], [W.questions(pool, i, 1)]
            r.call(img, qs, Nv=640, keep_prefix=True)
            for n in (1, 8, 32):
                # Rotate the first question across all four readouts, including Q=1.
                qset = W.questions(pool, i, n + 3, "mixed", seed=3)[i % 4 : i % 4 + n]
                out = r.followup([qset])[0]
                for j, q in enumerate(qset):
                    if q["type"] == "box":
                        follow.append(
                            {
                                "t": "box",
                                "p": out["presence_logit"][j].sigmoid().item(),
                                "box": out["box_pred"][j].tolist(),
                                "piou": out["iou_logit"][j].sigmoid().item(),
                            }
                        )
                    else:
                        logits = out["typed_logits"][j]
                        follow.append({"t": q["type"], "probs": logits[logits.isfinite()].softmax(-1).tolist()})
        for item in gref:
            qs = [{"type": "box", "text": q["text"]} for q in item["qs"][:8]]
            r.call([item["img"]], [[qs[0]]], Nv=640, keep_prefix=True)
            for q, label in zip(qs, item["qs"]):
                out = r.followup([[q]])[0]
                presence.append((bool(label["present"]), out["presence_logit"][0].sigmoid().item()))
        if reference is None:
            reference = (cold, follow, presence)
        import numpy as np

        pos = np.array([p for present, p in presence if present])
        neg = np.array([p for present, p in presence if not present])
        diff = pos[:, None] - neg
        roc = float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
        groups = {}
        for k, rows, ref in (("cold", cold, reference[0]), ("followup", follow, reference[1])):
            stats = compare(rows, ref) if k == "cold" else compare_follow(rows, ref)
            groups[k] = dict(n=len(rows), **stats)
        groups["gref_presence"] = {
            "auroc": roc,
            "positive": len(pos),
            "negative": len(neg),
            "agreed": sum((p >= 0.5) == (b >= 0.5) for (_, p), (_, b) in zip(presence, reference[2])),
            "max_abs": max(abs(p - b) for (_, p), (_, b) in zip(presence, reference[2])),
        }
        groups["n_box_gt"] = sum("gt" in row for row in cold)
        results[mode] = groups
        print(mode, groups, flush=True)
        save(args.out, results)
        del r
        gc.collect()
        torch.cuda.empty_cache()

    from scan.equivalence_limits import BF16_LIMIT

    mode = results["eng_w12"]
    follow, gref = mode["followup"], mode["gref_presence"]
    if (
        mode["cold"]["decision_agreement"] != 1
        or follow["agreed"] != follow["n"]
        or gref["agreed"] != gref["positive"] + gref["negative"]
        or max(follow["presence_abs"], gref["max_abs"]) > BF16_LIMIT["presence_prob"]
        or follow["typed_abs"] > BF16_LIMIT["typed_prob"]
        or follow["box_abs"] > BF16_LIMIT["box_pred"]
        or follow["iou_abs"] > BF16_LIMIT["iou_prob"]
    ):
        raise AssertionError("W12 failed its paired BF16 decision or continuous-output gate")


def compare_follow(rows, ref):
    import numpy as np

    boxes = [(a, b) for a, b in zip(rows, ref) if a["t"] == "box"]
    typed = [(np.array(a["probs"]), np.array(b["probs"])) for a, b in zip(rows, ref) if a["t"] != "box"]
    agree = sum((a["p"] >= 0.5) == (b["p"] >= 0.5) for a, b in boxes)
    agree += sum(a.argmax() == b.argmax() for a, b in typed)
    return {
        "agreed": int(agree),
        "presence_abs": max(abs(a["p"] - b["p"]) for a, b in boxes),
        "iou_abs": max(abs(a["piou"] - b["piou"]) for a, b in boxes),
        "box_abs": max(abs(x - y) for a, b in boxes for x, y in zip(a["box"], b["box"])),
        "typed_abs": max(abs(a - b).max().item() for a, b in typed),
    }


@torch.no_grad()
def counters(args):
    """Bracket one cold projection at a time. The NCU CSV follows the JSON order."""
    r = Runner(args)
    r.set_mode("eng_fused")
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")
    log = []
    seen = set()
    for name, mod in projections(r.net.lm):
        n, k = mod.weight.shape
        if (n, k) in seen:
            continue
        seen.add((n, k))
        for m in (16, 64, 128, 256, 704, 2816, 18688):
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            mod(x)
            flush.zero_()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            mod(x)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            log.append({"name": name, "m": m, "k": k, "n": n, "weight_bytes": mod.weight.nbytes})
    save(args.out, log)


@torch.no_grad()
def stages(args):
    """Bracket all 128 LM linears on warmed query graphs; compare the same question/feature inputs."""
    from contextlib import ExitStack
    from functools import partial
    from unittest.mock import patch

    from scan.bench_scan import Bench
    from scan.data import QUERIES, load_images
    from speed import workload as W
    from speed.w12 import W12Linear

    data = {}

    def sample(name, engine, owner, call):
        # Separate graph and graph-pool lifetime for external timing events.
        def clear():
            owner._graphs.clear()
            owner._pool = None

        clear()
        call()
        clear()
        events = []

        def wrap(original, mod=None):
            def forward(x, *args):
                if not torch.cuda.is_current_stream_capturing():
                    return original(x, *args)
                weight = mod.weight if mod is not None else args[0]
                m = x.numel() // weight.shape[1]
                compressed = isinstance(mod, W12Linear) and m in mod.rows
                a, b = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                a.record()
                y = original(x, *args)
                b.record()
                events.append((a, b, m, compressed, weight.numel()))
                return y

            return forward

        with ExitStack() as stack:
            for _, mod in projections(engine.lm):
                stack.enter_context(patch.object(mod, "forward", wrap(mod.forward, mod)))
            if engine.native_gate_up is not None:
                stack.enter_context(patch.object(engine, "native_gate_up", wrap(engine.native_gate_up)))
            call()
        if len(events) != 128:
            raise RuntimeError(f"{name} captured {len(events)} linears")
        dense, w12 = [], []
        for _ in range(11):
            call()
            torch.cuda.synchronize()
            elapsed = [a.elapsed_time(b) for a, b, *_ in events]
            dense.append(sum(elapsed))
            w12.append(sum(v for v, e in zip(elapsed, events) if e[3]))
        data[name] = {
            "m": events[0][2],
            "dense_ms": median(dense),
            "w12_ms": median(w12),
            "n_w12": sum(e[3] for e in events),
            "weights_w12": sum(e[4] for e in events if e[3]),
        }
        save(args.out, data)
        print(name, data[name], flush=True)
        clear()

    pool = W.load_coco("data/prep/eval_refcoco_val.jsonl", args.image_root, 12)
    folder = args.weights
    for mode in ("bf16", "compact_bf16", "w12"):
        args.w12 = folder if mode == "w12" else None
        r = Runner(args)
        r.set_mode("eng_w12" if args.w12 else "eng_fused")
        r.eng.compact_questions = mode != "bf16"
        r.call([pool[0]["img"]], [W.questions(pool, 0, 1)], Nv=640, keep_prefix=True)
        for q in (1, 8, 32):
            for kind in ("box",) if q == 1 else ("box", "mixed"):
                qs = [W.questions(pool, 0, q, kind)]
                sample(f"{mode}_followup_{kind}_N{q}", r.eng, r.eng, partial(r.followup, qs))
        del r
        gc.collect()
        torch.cuda.empty_cache()
        bench = Bench(args)
        eng = bench.to_engine()
        eng.compact_questions = mode != "bf16"
        imgs = load_images("data/prep/eval_refcoco_val.jsonl", args.image_root, 256, 640)
        bank = eng.encode([row["img"] for row in imgs], chunk=16)
        for n in (1, 8, 32, 64, 256):
            sub = bank.head(n)
            for kind in ("box", "noul"):
                query = (
                    {"type": kind, "text": QUERIES[0]}
                    if kind == "box"
                    else {"type": kind, "instr": f"Is there {QUERIES[0]} in the image?"}
                )
                sample(f"{mode}_Tv640_N{n}_{kind}", eng, sub, partial(eng.query, sub, query))
            eng.free_graphs(sub)
        del bank, sub, eng, bench
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def api_check(args):
    from scan.api import Scanner
    from scan.data import load_images

    items = load_images("data/prep/eval_refcoco_val.jsonl", args.image_root, 8, 640)
    outputs = {}
    for mode, kwargs in (("bf16", {}), ("w12", {"w12": args.weights}), ("fp8", {"fp8": True})):
        scanner = Scanner.load(args.ckpt, max_vis=640, **kwargs)
        results = {}
        for n in (1, 8):
            images = [row["img"] for row in items[:n]]
            for readout, q in (("box", "a dog"), ("yes", "Is there a person?")):
                for path in ("cached", "fresh"):
                    reply = scanner.scan(q, images, mode=path, readout=readout)
                    results[f"{n}_{readout}_{path}"] = reply["set"]["curve"]
        outputs[mode] = results
        del scanner
        gc.collect()
        torch.cuda.empty_cache()
    save(args.out, outputs)
    for key, expected in outputs["bf16"].items():
        assert [(v > 0.5) for v in outputs["w12"][key]] == [(v > 0.5) for v in expected], key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="/content/item5-w12-v1")
    parser.add_argument("--image_root", default="/content/data/coco/train2014")
    args = parser.parse_args()
    args.w12 = args.weights
    {
        "encode": export,
        "micro": micro,
        "micro_direct": micro,
        "micro_epilogue": micro,
        "accuracy": accuracy,
        "api": api_check,
        "counters": counters,
        "stages": stages,
    }[args.stage](args)


if __name__ == "__main__":
    main()
