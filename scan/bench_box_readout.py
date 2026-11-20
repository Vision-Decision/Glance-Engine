"""Fixed-input center controls, measured independently of the BF16 backbone."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from speed.box_readout import box_forward, center_scores, project


def milliseconds(fn, repeats=12):
    for _ in range(3):
        fn()
    events = []
    for _ in range(repeats):
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        stop.record()
        events.append((start, stop))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def layout_controls():
    # Same hidden rows and parameters, presented alone and in different padded batches.
    norm = torch.nn.LayerNorm(2560, device="cuda")
    cq = torch.nn.Linear(2560, 512, device="cuda")
    ck = torch.nn.Linear(2560, 512, device="cuda")
    h = torch.randn(3, 2560, device="cuda").bfloat16()
    v = torch.randn(3, 63, 2560, device="cuda").bfloat16()
    rows = {}

    def run(hidden, visual, mode):
        q = project(cq, norm(hidden.float()), mode)
        k = project(ck, norm(visual.float()), mode, 512)
        valid = torch.ones(k.shape[:2], dtype=torch.bool, device=k.device)
        return center_scores(q, k, valid)

    for mode in ("fixed", "tiled", "blocks"):
        single = torch.cat([run(h[i : i + 1], v[i : i + 1], mode) for i in range(3)])
        padded = run(
            h.repeat(3, 1),
            torch.nn.functional.pad(v.repeat(3, 1, 1), (0, 0, 0, 35)),
            mode,
        )[:3, :63]
        rows[mode] = {
            "max_score": float((single - padded).abs().max()),
            "different_scores": int((single != padded).sum()),
            "different_centers": int((single.argmax(1) != padded.argmax(1)).sum()),
        }
        if mode != "fixed":
            assert torch.equal(single, padded)
    from dv.config import Config
    from dv.model.heads import BoxHeads

    head = BoxHeads(2560, Config()).cuda().eval()
    xy = torch.rand(3, 63, 2, device="cuda")

    def box(hidden, visual, centers, valid, mode):
        images = torch.arange(len(hidden), device="cuda")
        return box_forward(head, hidden, visual, valid, centers, images, mode)

    mask = torch.ones(3, 63, dtype=torch.bool, device="cuda")
    for mode in ("blocks", "tiled"):
        single = [box(h[i : i + 1], v[i : i + 1], xy[i : i + 1], mask[i : i + 1], mode) for i in range(3)]
        pad = torch.nn.functional.pad
        batched = box(
            h.repeat(3, 1),
            pad(v.repeat(3, 1, 1), (0, 0, 0, 35)),
            pad(xy.repeat(3, 1, 1), (0, 0, 0, 35)),
            pad(mask.repeat(3, 1), (0, 35)),
            mode,
        )
        errors = {}
        for key in ("presence_logit", "pred_idx", "box_pred", "iou_logit"):
            delta = torch.cat([item[key] for item in single]) - batched[key][:3]
            errors[key] = float(delta.abs().max())
        rows[mode + "_box_outputs"] = errors
        assert not any(errors.values()), errors
    expected = box(h, v, xy, mask, "blocks")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        under_autocast = box(h, v, xy, mask, "blocks")
    assert all(torch.equal(value, under_autocast[key]) for key, value in expected.items())
    # A real tie, a masked high score, a difference lost to FP32 rounding, and an empty mask.
    query = torch.ones(4, 2, device="cuda")
    keys = torch.tensor(
        [
            [[2**24, 0.0], [2**24, 0.25], [2**25, 0.0]],
            [[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]],
        ],
        device="cuda",
    )
    valid = torch.tensor([[True, True, False], [True, True, False]], device="cuda")
    images = torch.tensor([0, 1, 1, 0], device="cuda")
    expected = torch.tensor([1, 0, 0, 1], device="cuda")
    dense = center_scores(query, keys, valid, images)
    assert torch.equal(dense.argmax(1), expected)
    slots = torch.arange(6, device="cuda").view(2, 3)
    compact = center_scores(query, keys.flatten(0, 1), valid, images, slots)
    assert torch.equal(dense, compact)
    valid.zero_()
    assert torch.equal(center_scores(query, keys, valid, images).argmax(1), torch.zeros_like(expected))
    return rows


@torch.no_grad()
def precision_latency(ckpt):
    from types import SimpleNamespace

    from speed import workload
    from speed.bench_glance import Runner, summarize_times
    from speed.diag_equiv import fp32_call_factory, set_dtype

    runner = Runner(SimpleNamespace(ckpt=ckpt))
    pool = workload.load_coco("data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014", 50)
    calls = [([im["img"]], [workload.questions(pool, i, 32, "box")]) for i, im in enumerate(pool)]
    result = {}
    for dtype in ("bf16", "fp32"):
        if dtype == "fp32":
            set_dtype(runner.net.hf, torch.float32)
            runner.call = fp32_call_factory(runner)
        for images, questions in calls:
            runner.call(images, questions)
        totals = [runner.call(images, questions)[1] for images, questions in calls]
        result[dtype] = summarize_times(totals, 32)
    return result


@torch.no_grad()
def presence_audit(ckpt):
    """Compare both readouts on the fresh 48-image COCO margin panel against each shipped singleton."""
    from types import SimpleNamespace

    from dv.config import Config
    from dv.data.mixed import ChatTok
    from dv.model.backbone import SharedPrefixLM

    from scan.bench_scan import Bench
    from scan.data import QUERIES, load_images
    from scan.engine import ScanEngine

    bench = Bench(SimpleNamespace(ckpt=ckpt))
    cfg = Config().update(
        {
            "task": "v1",
            "attn_mask": bench.cfg.attn_mask,
            "max_vis_tokens": 640,
            "grad_ckpt": False,
        }
    )
    bench.net = SharedPrefixLM(bench.net.hf, cfg).eval()
    bench.tok = ChatTok(bench.proc, cfg)
    engine = ScanEngine(bench.net, bench.heads, bench.tok, bench.proc, graphs=False, fast_ops=False)
    panel = load_images("data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014", 48, 640)
    images = [row["img"] for row in panel]
    cases, totals = [], {"legacy": 0, "blocks": 0}
    max_change = 0.0
    for text in QUERIES[:4]:
        question = {"type": "box", "text": text}
        reference = bench.reference(images, question)["presence_logit"]
        logits = {}
        for mode in totals:
            engine.box_reduction = mode
            logits[mode] = engine.fresh(images, question)["presence_logit"]
            totals[mode] += int(((logits[mode] > 0) == (reference > 0)).sum())
        max_change = max(max_change, float((logits["blocks"] - logits["legacy"]).abs().max()))
        for i in range(len(images)):
            if any((x[i] > 0) != (reference[i] > 0) for x in logits.values()):
                probs = {mode: float(value[i].sigmoid()) for mode, value in logits.items()}
                cases.append(
                    dict(
                        image_index=i,
                        question=text,
                        reference=float(reference[i].sigmoid()),
                        **probs,
                    )
                )
    return {
        "count": len(images) * 4,
        "agreed": totals,
        "max_head_logit_change": max_change,
        "disagreements": cases,
    }


def score_timings(images, questions, patches):
    k = torch.randn(images, patches, 512, device="cuda")
    q = torch.randn(questions, 512, device="cuda")
    mask = torch.ones(images, patches, device="cuda", dtype=torch.bool)
    owner = torch.zeros(questions, device="cuda", dtype=torch.long) if images != questions else None

    def legacy():
        keys = k if owner is None else k[owner]
        valid = mask if owner is None else mask[owner]
        return (torch.einsum("md,mnd->mn", q, keys) / (512**0.5)).masked_fill(~valid, -1.0e4).argmax(-1)

    def fixed():
        return center_scores(q, k, mask, owner).argmax(-1)

    def eager64():
        keys = k if owner is None else k[owner]
        valid = mask if owner is None else mask[owner]
        return ((q.double()[:, None] * keys.double()).sum(-1) / (512**0.5)).masked_fill(~valid, -1.0e4).argmax(-1)

    assert torch.equal(fixed(), eager64())
    return {
        "legacy_ms": milliseconds(legacy),
        "eager_fp64_ms": milliseconds(eager64),
        "fixed_ms": milliseconds(fixed),
        "dots": questions * patches,
        "width": 512,
        "key_bytes": k.numel() * k.element_size(),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    control = ap.add_mutually_exclusive_group()
    control.add_argument("--precision-latency", metavar="CKPT")
    control.add_argument("--presence-audit", metavar="CKPT")
    args = ap.parse_args()
    if args.precision_latency or args.presence_audit:
        result = (
            precision_latency(args.precision_latency) if args.precision_latency else presence_audit(args.presence_audit)
        )
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    torch.manual_seed(9)
    controls = layout_controls()
    n = 4096
    a, b = [torch.randn(n, n, device="cuda", dtype=torch.float64) for _ in range(2)]
    c = torch.empty_like(a)
    ms = milliseconds(lambda: torch.mm(a, b, out=c))
    result = {
        "controls": controls,
        "fp64_gemm": {"n": n, "ms": ms, "tflops": 2 * n**3 / ms / 1.0e9},
        "workloads": {},
    }
    for label, images, questions, patches in (
        ("followup_1", 1, 1, 640),
        ("image_32", 1, 32, 640),
        ("cached_256", 256, 256, 640),
        ("fresh_128", 128, 128, 117),
    ):
        result["workloads"][label] = score_timings(images, questions, patches)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
