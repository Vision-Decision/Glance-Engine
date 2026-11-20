"""Item 8 measurements on real image and native video prefixes."""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import torch

from scan.bench_scan import Bench, make_clips
from scan.data import load_images


def inventory(bank):
    if hasattr(bank, "components"):
        return bank.components()
    groups = {"gdn_state": [], "conv_tail": [], "key": [], "value": []}
    for kind, (a, b) in zip(bank.types, bank.cache):
        left, right = ("gdn_state", "conv_tail") if kind == "linear_attention" else ("key", "value")
        groups[left].append(a)
        groups[right].append(b)
    for name in ("vis", "ck", "centers", "grid_hw", "vis_valid", "q_start", "p_valid"):
        groups[name] = [getattr(bank, name)]
    return {name: sum(t.untyped_storage().nbytes() for t in ts) for name, ts in groups.items()}


def accounting(eng, a, res):
    for shape in ("f4_v96", "f4_v48", "f2_v96", "Tv640", "Tv256", "Tv64"):
        if shape.startswith("f"):
            frames, vis = map(int, shape[1:].split("_v"))
            eng.clip_max_vis = vis
            units = make_clips(a, a.count, frames, vis)
        else:
            vis = int(shape[2:])
            units = [x["img"] for x in load_images(a.prep, a.image_root, a.count, vis, frame=vis != 640)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        bank = eng.encode(units, chunk=16)
        torch.cuda.synchronize()
        res[shape] = {
            "n": bank.N,
            "encode_ms": 1000 * (time.perf_counter() - start),
            "bytes": inventory(bank),
            "valid_tokens": bank.p_valid.sum(1).tolist(),
            "prefix_slots": bank.Lp,
            "visual_slots": bank.vis_valid.shape[1],
            "shared": getattr(bank, "shared", 0),
        }
        print(shape, res[shape], flush=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        del bank
        eng.free_graphs()


def probe(eng, a, res):
    import zlib

    for name, units in (
        ("clip", make_clips(a, 4)),
        ("image", [x["img"] for x in load_images(a.prep, a.image_root, 4, 640)]),
    ):
        bank = eng.encode(units)
        rows = []
        for kind, (state, _) in zip(bank.types, bank.cache):
            if kind != "linear_attention":
                continue
            x = state[0].float().contiguous()
            a32 = x.cpu().numpy()
            wire = a32.tobytes()
            coded = zlib.compress(wire, 1)
            start = time.perf_counter()
            assert zlib.decompress(coded) == wire
            decode_ms = (time.perf_counter() - start) * 1000
            rows.append(
                {
                    "bytes": len(wire),
                    "compressed": len(coded),
                    "decode_ms": decode_ms,
                    "exact_bf16": float((x == x.bfloat16().float()).float().mean()),
                    "zero": float((x == 0).float().mean()),
                    "rank": int(torch.linalg.matrix_rank(x[0].double()).item()),
                }
            )
        res[name] = rows
        del bank
    Path(a.out).write_text(json.dumps(res, indent=2))


def breakeven(eng, a, res):
    import numpy as np

    for shape in ("f4_v96", "Tv256", "Tv640", "Tv64"):
        units = (
            make_clips(a, 64)
            if shape.startswith("f")
            else [x["img"] for x in load_images(a.prep, a.image_root, 64, int(shape[2:]), frame=shape != "Tv640")]
        )
        q = {"type": "noul", "instr": "Is there a person in the image?"}
        if shape.startswith("f"):
            q = {"type": "noul", "instr": "Is a person walking in this clip?"}

        def measure(fn, reps):
            vals = []
            for _ in range(reps + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                obj = fn()
                torch.cuda.synchronize()
                vals.append((time.perf_counter() - start) * 1000)
                del obj
            return vals[1:]

        chunk = 32 if shape.startswith("f") else 16
        encode = partial(eng.encode, units, chunk=chunk)
        enc = measure(encode, 3)
        bank = encode()
        cached = measure(partial(eng.query, bank, q), 7)
        del bank
        eng.free_graphs()
        fresh = measure(partial(eng.fresh, units, q), 5)
        res[shape] = {
            "n": len(units),
            "encode_ms": enc,
            "cached_ms": cached,
            "fresh_ms": fresh,
        }
        print(shape, {k: np.median(v) for k, v in res[shape].items()}, flush=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        eng.free_graphs()


def formats(eng, a, res):
    from scan.bank import FORMATS, StoredBank

    for shape, units in (
        ("clip", make_clips(a, 64)),
        ("Tv640", [r["img"] for r in load_images(a.prep, a.image_root, 64, 640)]),
    ):
        base = eng.encode(units, chunk=16, bank_format="legacy")
        q = (
            {"type": "noul", "instr": "Is a person walking in this clip?"}
            if shape == "clip"
            else {"type": "box", "text": "person standing"}
        )
        ref = eng.query(base, q)
        for mode in FORMATS:
            bank = base if mode == "legacy" else StoredBank.pack(base, mode)
            ok = None
            if mode == "state_packed":
                ok = all(
                    torch.equal(bank.layer(i)[0].view(torch.int32), c[0].view(torch.int32))
                    for i, c in enumerate(base.cache)
                    if base.types[i] == "linear_attention"
                )
                assert ok
            result = eng.query(bank, q)
            times = []
            for _ in range(9):
                timer = {}
                eng.query(bank, q, timings=timer)
                times.append(timer["total"])
            equal = all(torch.equal(ref[k], result[k]) for k in ref)
            if mode in ("compact", "state_packed"):
                assert equal
            res[shape + "/" + mode] = {
                "bytes": bank.nbytes(),
                "timing_ms": times,
                "bit_equal": equal,
                "state_bits_equal": ok,
                "n": bank.N,
            }
            eng.free_graphs(bank)
            del bank
            Path(a.out).write_text(json.dumps(res, indent=2))
        # Typed-only layout, head/select order, changing question length and a partial host tail.
        from scan.host_bank import HostBank

        typed = {"type": "noul", "instr": "Is there a person?"}
        ids = [3, 1, 3, 7, 0]
        small = base.select(ids)
        exact = eng.query(small, typed)["typed_logits"]
        qs = [typed, {"type": "noul", "instr": "Can you see a person in the image?"}]
        for mode in ("compact", "state_packed", "state_fp8", "kv_fp8"):
            bank = StoredBank.pack(base, mode, "typed")
            subset = bank.select(ids)
            host = HostBank.from_chunks([subset.head(3), subset.select([3, 4])])
            for question in qs:
                out = eng.query(host, question)
                expected = torch.cat(
                    [
                        eng.query(subset.head(3), question)["typed_logits"],
                        eng.query(subset.select([3, 4]), question)["typed_logits"],
                    ]
                )
                assert set(out) == {"typed_logits"} and torch.equal(out["typed_logits"], expected)
            if mode in ("compact", "state_packed"):
                assert torch.equal(eng.query(subset, typed)["typed_logits"], exact)
            eng.free_graphs(host)
            res[shape + "/" + mode]["typed_only_bytes"] = bank.nbytes()
            res[shape + "/" + mode]["subset_and_prefetch"] = True
        Path(a.out).write_text(json.dumps(res, indent=2))
        del base
        eng.free_graphs()


def lifecycle(eng, a, res):
    from scan.data import QUERIES

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        val = fn()
        torch.cuda.synchronize()
        return val, 1000 * (time.perf_counter() - start)

    for shape in ("f4_v96", "Tv640", "Tv256", "Tv64"):
        clip = shape.startswith("f")
        units = (
            make_clips(a, 64)
            if clip
            else [r["img"] for r in load_images(a.prep, a.image_root, 64, int(shape[2:]), frame=shape != "Tv640")]
        )
        media = "clip" if clip else "image"
        # Small frames have the lowest fresh cost. Follow a full question cycle so
        # the observed crossing includes the captures for new question lengths.
        texts = QUERIES if shape == "Tv64" else QUERIES[:4]
        qs = [{"type": "noul", "instr": f"Is there {text} in the {media}?"} for text in texts]
        fresh = []
        for _ in range(4):
            fresh.append([timed(partial(eng.fresh, units, q))[1] for q in qs])
        encodes, cached = [], []
        for _ in range(4):
            # Capture belongs to each new bank. Preserve the engine's warmed vision graphs.
            bank, ms = timed(partial(eng.encode, units, chunk=32 if clip else 16))
            encodes.append(ms)
            cached.append([timed(partial(eng.query, bank, q))[1] for q in qs])
            del bank
        res[shape] = {
            "n": len(units),
            "questions": qs,
            "encode_ms": encodes[1:],
            "fresh_ms": fresh[1:],
            "cached_ms": cached[1:],
        }
        Path(a.out).write_text(json.dumps(res, indent=2))
        eng.free_graphs()


def entry(eng, a, res):
    units = [x["img"] for x in load_images(a.prep, a.image_root, 9, 640)]
    for mode in (None, "compact", "state_packed", "state_fp8"):
        bank = eng.encode(units, chunk=2, bank_format=mode, readout="typed", tier="cpu")
        stored_mode = mode or os.getenv("GLANCE_BANK_FORMAT", "state_packed")
        assert all(c.bank.mode == stored_mode for c in bank.chunks)
        reference = [eng.encode(units[i : i + 2], chunk=2, bank_format=mode) for i in range(0, len(units), 2)]
        rows = []
        for text in ("Is there a dog?", "Can you see a person walking in the image?", "Is there a dog?"):
            q = {"type": "noul", "instr": text}
            val = eng.query(bank, q)
            target = torch.cat([eng.query(b, q)["typed_logits"] for b in reference])
            rows.append(torch.equal(val["typed_logits"], target))
        try:
            eng.query(bank, {"type": "box", "text": "a bicycle"})
        except ValueError:
            pass
        else:
            raise AssertionError("typed bank accepted a spatial query")
        assert all(rows), (mode, rows)
        res[mode or "default"] = {
            "format": stored_mode, "chunks": len(bank.chunks), "exact": rows,
            "keys": list(val), "bytes": bank.nbytes(),
        }
        eng.free_graphs(bank, *reference)
        Path(a.out).write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="accounting")
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--ckpt", default="/content/ckpt/v1/ckpt_final.pt")
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bench = Bench(a)
    {
        "accounting": accounting,
        "probe": probe,
        "breakeven": breakeven,
        "formats": formats,
        "entry": entry,
        "lifecycle": lifecycle,
    }[a.stage](bench.to_engine(), a, {})


if __name__ == "__main__":
    with torch.inference_mode():
        main()
