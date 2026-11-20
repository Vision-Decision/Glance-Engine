"""Physical pinned capacity and sustained transfers over a longer logical scan."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

from scan.bench_scan import Bench
from scan.data import load_images
from scan.host_bank import HostBank, HostChunk


def samples(fn, reps):
    fn()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(1000 * (time.perf_counter() - start))
    return out


def run(eng, args):
    frames = [x["img"] for x in load_images(args.prep, args.image_root, 64, 640, frame=True)]

    # Distinct frame tuples avoid overstating savings through duplicate clip storage.
    def clip(i):
        a, b = divmod(i, len(frames))
        return [
            frames[b],
            frames[(b + a + 1) % len(frames)],
            frames[(2 * b + a + 7) % len(frames)],
            frames[(3 * b + a + 11) % len(frames)],
        ]

    q = {"type": "noul", "instr": "Is a person walking in this clip?"}
    if args.resident_chunks:
        eng.graphs = False
        banks = [
            eng.encode([clip(i) for i in range(s, s + args.chunk)], chunk=32, bank_format=args.bank_format)
            for s in range(0, 1024, args.chunk)
        ]
        res = {}
        for n in (64, 256, 1024):

            def scan(banks):
                parts = [eng.query(b, q) for b in banks]
                return {k: torch.cat([r[k] for r in parts]) for k in parts[0]}

            subset = banks[: n // args.chunk]
            res[n] = {
                "ms": samples(partial(scan, subset), 5),
                "bytes": sum(b.nbytes() for b in subset),
                "chunk": args.chunk,
                "graphs": False,
            }
            Path(args.out).write_text(json.dumps(res, indent=2))
        return
    if args.resident:
        res = {}
        for n in (64, 256, 1024):
            bank = eng.encode([clip(i) for i in range(n)], chunk=32, bank_format=args.bank_format)
            try:
                ms = samples(partial(eng.query, bank, q), 5)
                res[n] = {
                    "ms": ms,
                    "resident_bytes": bank.nbytes(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                }
            except torch.cuda.OutOfMemoryError:
                res[n] = {"oom": True, "resident_bytes": bank.nbytes()}
            eng.free_graphs(bank)
            del bank
            Path(args.out).write_text(json.dumps(res, indent=2))
        return
    chunks, answers = [], []
    res = {
        "chunk_size": args.chunk,
        "format": args.bank_format,
        "physical_clips": args.physical,
        "rows": {},
    }

    def save():
        Path(args.out).write_text(json.dumps(res, indent=2))

    for start in range(0, args.physical, args.chunk):
        bank = eng.encode(
            [clip(i) for i in range(start, start + args.chunk)],
            chunk=32,
            bank_format=args.bank_format,
        )
        if not start:
            res["resident_chunk_ms"] = samples(partial(eng.query, bank, q), 7)
        expected = eng.query(bank, q)
        answers.append(expected)
        chunks.append(HostChunk(bank))
        eng.free_graphs(bank)
        del bank
        if len(chunks) % 4 == 0:
            print("pinned", start + args.chunk, flush=True)
    res["physical_pinned_bytes"] = sum(c.slab.nbytes for c in chunks)
    for n in [args.chunk, 256, 1024, args.physical, 10000]:
        # Repeated traversal measures the actual traffic for 10k when RAM cannot hold 10k distinct records.
        count, tail = divmod(n, args.chunk)
        source = [chunks[i % len(chunks)] for i in range(count)]
        expected = [answers[i % len(chunks)] for i in range(count)]
        if tail:
            r = eng.encode([clip(i) for i in range(tail)], chunk=32, bank_format=args.bank_format)
            source.append(HostChunk(r))
            expected.append(eng.query(r, q))
            eng.free_graphs(r)
            del r
        hb = HostBank(source)
        got = eng.query(hb, q)
        diffs = {}
        for k in got:
            ref = torch.cat([o[k] for o in expected])
            diffs[k] = bool(torch.equal(got[k], ref))
        assert all(diffs.values()), diffs
        times = samples(partial(eng.query, hb, q), 3 if n > 1000 else 5)
        stage = torch.empty(max(c.slab.numel() for c in source), dtype=torch.uint8, device=eng.dev)

        def copy_only(source, stage):
            for c in source:
                stage[: len(c.slab)].copy_(c.slab, non_blocking=True)

        copies = samples(partial(copy_only, source, stage), 3)
        res["rows"][n] = {
            "bytes": hb.nbytes(),
            "ms": times,
            "copy_ms": copies,
            "exact": diffs,
            "staging_bytes": sum(s["slab"].nbytes for s in hb._slots if s),
        }
        print(n, res["rows"][n], flush=True)
        eng.free_graphs(hb)
        del stage, hb
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/content/ckpt/v1/ckpt_final.pt")
    p.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    p.add_argument("--image_root", default="/content/data/coco/train2014")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--resident-chunks", action="store_true")
    p.add_argument("--resident", action="store_true")
    p.add_argument("--physical", type=int, default=1920)
    p.add_argument("--bank-format", default="compact")
    args = p.parse_args()
    with torch.inference_mode():
        run(Bench(args).to_engine(), args)
