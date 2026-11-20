"""Real-weight kernel profiles and reduction-order probes for the frame workloads."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from scan.bench_scan import Bench
from scan.data import load_images
from scan.profile_lm import check_outputs, stage_breakdown, trace
from scan.profile_vision import elapsed, errors
from scan.varlen import FreshPack
from scan.vision_prep import frame_layout, upload_frames
from speed.engine import image_attention


def attention(q, k, v):
    from torch.nn.attention.varlen import varlen_attn

    b, length, heads, dim = q.shape
    one = [torch.cuda.current_stream()]
    ref = image_attention(q, k, v, one)
    pages = (length + 255) // 256
    padded = pages * 256
    seq = torch.arange(b + 1, dtype=torch.int32, device=q.device)
    table = torch.arange(b * pages, dtype=torch.int32, device=q.device).reshape(b, pages)
    lengths = torch.full((b,), length, dtype=torch.int32, device=q.device)

    def pad(x):
        return F.pad(x, (0, 0, 0, 0, 0, padded - length))

    kp, vp = [pad(x).reshape(-1, 256, heads, dim) for x in (k, v)]
    row = {}
    for qlen in (length, padded):
        qp = (pad(q) if qlen != length else q).flatten(0, 1)
        for splits in (1, 2, 3, 4):
            for max_k in (length, padded):

                def run(qp=qp, qlen=qlen, max_k=max_k, splits=splits):
                    out = varlen_attn(
                        qp,
                        kp,
                        vp,
                        seq * qlen,
                        seq * length,
                        qlen,
                        max_k,
                        seqused_k=lengths,
                        block_table=table,
                        num_splits=splits,
                    )
                    return out.view(b, qlen, heads, dim)[:, :length]

                val = run()
                row[f"{qlen}/{splits}/{max_k}"] = dict(
                    ms=elapsed(run),
                    **errors(val, ref),
                    per_image=[(a != r).sum().item() for a, r in zip(val, ref)],
                )
    return row


def reduction_orders(mod, px):
    import itertools

    # Compare a slice of the actual convolution output. Keep the full input
    # shape for the reference so cuDNN dispatch is the production one.
    x, w = px.to(mod.proj.weight.dtype), mod.proj.weight
    ref = mod(x)[:256, :256]
    x, w = x[:256].reshape(-1, *w.shape[1:]), w[:256]
    rows = []
    for spatial in itertools.permutations((2, 3, 4)):
        for reversed_axes in ((), spatial, (spatial[-1],)):
            for pad in (3, 4, 8, 16, 32):

                def layout(t, reversed_axes=reversed_axes, spatial=spatial, pad=pad):
                    t = t.flip(reversed_axes) if reversed_axes else t
                    a = t.permute(0, *spatial, 1)
                    return F.pad(a, (0, pad - 3)).reshape(len(t), -1).contiguous()

                y = F.linear(layout(x), layout(w)) + mod.proj.bias[:256]
                rows.append(
                    {
                        "order": spatial,
                        "reverse": reversed_axes,
                        "pad": pad,
                        "differences": int((y != ref).sum()),
                    }
                )
    return sorted(rows, key=lambda r: r["differences"])


def patch_probe(mod, px):
    x = px.to(mod.proj.weight.dtype)
    weight = mod.proj.weight
    ref = mod(x)
    row = {"reference": {"ms": elapsed(lambda: mod(x)), "kernels": trace(lambda: mod(x))}}
    for label, axes in (
        ("cthw", (0, 1, 2, 3, 4)),
        ("thwc", (0, 2, 3, 4, 1)),
        ("whct", (0, 4, 3, 1, 2)),
    ):
        w = weight.permute(axes).reshape(len(weight), -1).contiguous()
        p = x.view(len(x), *weight.shape[1:]).permute(axes).reshape(len(x), -1).contiguous()

        def run(p=p, w=w):
            return F.linear(p, w) + mod.proj.bias

        row[label] = dict(ms=elapsed(run), **errors(run(), ref))
    return row


def patch_convolutions(mod, pixels):
    """Same convolution, including input-layout cost, across equivalent spatial views."""
    layer = mod.proj
    x = pixels.to(layer.weight.dtype)
    reference = mod(x)
    rows = {"reference": {"ms": elapsed(lambda: mod(x), reps=9)}}
    for shape in ((2, 16, 16), (32, 16), (2, 256), (1, 512)):
        conv = F.conv3d if len(shape) == 3 else F.conv2d
        nhwc = torch.channels_last_3d if len(shape) == 3 else torch.channels_last
        for order in (torch.contiguous_format, nhwc):
            weight = layer.weight.reshape(-1, 3, *shape).contiguous(memory_format=order)

            def run(shape=shape, order=order, weight=weight, conv=conv):
                image = x.reshape(-1, 3, *shape).contiguous(memory_format=order)
                return conv(image, weight, layer.bias, stride=shape).flatten(1)

            output = run()
            rows[f"{shape}/{order}"] = dict(
                ms=elapsed(run, reps=9),
                **errors(output, reference),
                differing_elements=(output != reference).sum().item(),
            )
    return rows


def video_windows():
    from speed import workload as W

    pool = W.load_coco(
        "data/prep/eval_refcoco_val.jsonl",
        "/content/data/coco/train2014",
        100,
        max_vis=256,
    )
    for n in (32, 64):
        for call in range(4):
            indices = [(call * n + j) % len(pool) for j in range(n)]
            images = [pool[i]["img"] for i in indices]
            questions = [W.questions(pool, i, 1, "box") for i in indices]
            yield n, call, images, questions


def video_entry_checks(args):
    """Exercise preprocessing, model dispatch, and CPU readout through Runner.call."""
    import os
    from unittest.mock import patch

    from scan.bench_scan import diff
    from speed.bench_glance import Runner
    from speed.engine import GlanceEngine

    runner = Runner(args)
    runner.set_mode("eng_fused")
    with patch.dict(os.environ, {"GLANCE_FRAME_OPT": "0"}):
        original = GlanceEngine(runner.net, runner.heads, fast_ops=True)
    rows = []
    for n, call, images, questions in video_windows():
        with (
            patch.object(runner, "eng", original),
            patch.object(runner, "frames", None),
        ):
            reference, _ = runner.call(images, questions)
        candidate, _ = runner.call(images, questions)
        values = {k: reference[k] for k in ("presence_logit", "iou_logit", "box_pred")}
        rows.append(dict(n=n, call=call, **diff(candidate, values)))
    return rows


def video_checks(eng):
    from dv.data.mixed import collate_mixed

    from scan.bench_scan import diff
    from speed.bench_glance import make_item
    from speed.engine import FastVision

    video_tower = FastVision(eng.net.visual, graphs=False, fused_rotary=True)
    results = []
    for n, it, imgs, qs in video_windows():
        batch = collate_mixed(
            [make_item(eng.tok, eng.proc, im, q) for im, q in zip(imgs, qs)],
            eng.tok,
        )
        px = eng.frame_prep.mixed_pixels(imgs)
        grid = batch["grid_thw"]
        assert torch.equal(px, batch["pixel_values"].cuda())
        feats = video_tower(px, grid)
        hf = eng.net.visual(px, grid.cuda()).pooler_output
        old = eng.vision(px, grid)

        score = partial(eng.answer_independent, batch)

        candidate = score(feats)
        fields = ("presence_logit", "iou_logit", "box_pred")
        compared = {}
        for key, other in (("hf", hf), ("before", old)):
            ref = score(other)
            compared[key] = diff(candidate, {k: ref[k] for k in fields})
        results.append(
            dict(
                n=n,
                call=it,
                pixels_equal=True,
                features_vs_hf=errors(feats, hf),
                features_vs_before=errors(feats, old),
                **compared,
            )
        )
    return results


def patch_alternative(tower, use_2d):
    if use_2d:
        weight = tower.v.patch_embed.proj.weight.reshape(-1, 3, 32, 16).contiguous(
            memory_format=torch.channels_last
        )

        def conv2d(pixels):
            image = (
                pixels.to(weight.dtype).reshape(-1, 3, 32, 16).contiguous(memory_format=torch.channels_last)
            )
            return F.conv2d(image, weight, tower.b, stride=(32, 16)).flatten(1)

        return conv2d

    def linear(pixels):
        return F.linear(pixels.to(tower.W.dtype), tower.W) + tower.b

    return linear


def patch_control(eng, tower, px, grid, frames, use_2d):
    prep = eng.frame_prep
    count = len(frames)
    from unittest.mock import patch

    from scan.bench_scan import diff
    from scan.varlen import packed_forward

    tower.graphs = True

    def exact():
        return tower(px, grid)

    correct = exact().clone()
    exact_ms = elapsed(exact)
    tower._graphs.clear()
    tower._pool = None

    alternative = patch_alternative(tower, use_2d)

    with patch.object(tower.v.patch_embed, "forward", alternative):
        candidate = tower(px, grid).clone()
        candidate_ms = elapsed(lambda: tower(px, grid))
    rows = []
    for question in (
        {"type": "box", "text": "the person"},
        {"type": "noul", "instr": "Is there a person in the image?"},
        {"type": "noul", "instr": "Is there a clock in the image?"},
        {"type": "choice", "instr": "What color?", "options": ["red", "green", "blue"]},
    ):
        b = frame_layout(prep.template(frames[0].size, question, None), count)
        pack = FreshPack(b, eng.dev)
        fields = (
            ("presence_logit", "iou_logit", "box_pred") if question["type"] == "box" else ("typed_logits",)
        )
        ref = packed_forward(eng, pack, correct)
        cand = packed_forward(eng, pack, candidate)
        rows.append(diff(cand, {k: ref[k] for k in fields}))
    row = {
        "count": count,
        "exact_ms": exact_ms,
        "candidate_ms": candidate_ms,
        "features": errors(candidate, correct),
        "decisions": rows,
    }
    return row


def binary_patch_control(eng, tv, use_2d):
    """Compare approximate patch embeddings on every question in the fresh sweep."""
    from unittest.mock import patch

    from scan.data import QUERIES
    from scan.equivalence_limits import BF16_LIMIT
    from scan.varlen import packed_forward

    frames = [
        r["img"]
        for r in load_images(
            "data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014", 256, tv, frame=True
        )
    ]
    tower = eng.frame_tower

    # Separate eager forwards on the same real batches avoid sharing static graph outputs.
    alternative = patch_alternative(tower, use_2d)

    prep = eng.frame_prep
    size = 32 if tv == 128 else 64
    records = []
    for lo in range(0, len(frames), size):
        images = frames[lo : lo + size]
        pixels = prep.pixels(upload_frames(images), dtype=eng.dtype)
        grid = prep.template(images[0].size, {"type": "box", "text": "person"}, None)["grid_thw"].expand(
            size, -1
        )
        exact = tower(pixels, grid)
        with patch.object(tower.v.patch_embed, "forward", alternative):
            candidate = tower(pixels, grid)
        for prompt in QUERIES[:10]:
            question = {"type": "noul", "instr": f"Is there {prompt} in the image?"}
            batch = frame_layout(prep.template(images[0].size, question, None), size)
            pack = FreshPack(batch, eng.dev)
            ref = packed_forward(eng, pack, exact)["typed_logits"].float().softmax(-1)
            out = packed_forward(eng, pack, candidate)["typed_logits"].float().softmax(-1)
            delta = (out - ref).abs().amax(-1)
            top = out.topk(2, dim=-1).values
            ambiguous = (top[:, 0] - top[:, 1]) <= 2 * BF16_LIMIT["typed_prob"]
            mismatch = out.argmax(-1) != ref.argmax(-1)
            records.append(
                {
                    "offset": lo,
                    "query": prompt,
                    "n": size,
                    "changed": int(mismatch.sum()),
                    "max_abs": delta.max().item(),
                    "outside_limit": int((delta > BF16_LIMIT["typed_prob"]).sum()),
                    "within_margin": int(ambiguous.sum()),
                    "changed_outside_margin": int((mismatch & ~ambiguous).sum()),
                }
            )
    return records


def norm_probe(tower, px, grid):
    from scan.vision_norm_probe import add_norm

    x = tower.v.patch_embed(px)
    _, (pos, _, _, _) = tower._consts(grid, px.device)
    update = pos.to(x.dtype).contiguous()
    norm = tower.v.blocks[0].norm1

    def eager():
        residual = x + update
        return residual, norm(residual)

    def fused():
        return add_norm(x, update, norm)

    r, y = eager()
    rf, yf = fused()
    row = {
        "residual": errors(rf, r),
        "norm": errors(yf, y),
        "eager_ms": elapsed(eager, reps=11),
        "fused_ms": elapsed(fused, reps=11),
    }
    return row


def mlp_probe(tower, px):
    from speed.vision_mlp import activation_table, fc1

    blk = tower.v.blocks[0]
    mlp = blk.mlp
    x = blk.norm1(tower.v.patch_embed(px))
    table = activation_table(mlp)

    def eager():
        return mlp.act_fn(mlp.linear_fc1(x))

    ref = eager()
    row = {"reference": elapsed(eager, reps=15)}
    for config in (
        (128, 128, 32, 8),
        (128, 128, 32, 4),
        (128, 256, 32, 8),
        (64, 128, 64, 4),
        (128, 64, 64, 4),
    ):
        run = partial(fc1, x, mlp.linear_fc1, table, config)

        row[str(config)] = dict(ms=elapsed(run, reps=15), **errors(run(), ref))
    return row


def gdn_probe(eng, tv, tiles):
    from scan.varlen import Segments
    from speed.lm_prefill import conv_qkv

    width, n = (146, 128) if tv == 128 else (281, 64)
    seq = Segments([width] * n, eng.dev)
    layer = eng.layers[0]
    m = layer.linear_attn
    torch.manual_seed(19)
    hidden = torch.randn((n * width, eng.net.hidden), device=eng.dev, dtype=torch.bfloat16)
    mixed, _, beta, decay = eng._gdn_projections(m, layer.input_layernorm(hidden))
    args = (
        *conv_qkv(m, mixed, seq.token_starts()),
        decay[None],
        beta[None],
        seq,
    )
    ref = eng.flash_gdn(*args, dense=False)
    row = {}
    for dense in (False, True):
        for tile in tiles:
            name = f"{'dense' if dense else 'packed'}_{tile}"

            run = partial(eng.flash_gdn, *args, dense=dense, value_tile=tile)

            try:
                row[name] = dict(ms=elapsed(run, reps=21), **errors(run(), ref))
            except Exception as exc:
                # TileLang's CUDA launcher rejects a tile exceeding the SM's per-block workspace.
                message = str(exc)
                if tile != 128 or "allowed dynamic shared memory size" not in message:
                    raise
                row[name] = {"supported": False, "error": message.splitlines()[-1]}
    return row


def kernel_profile(eng, tower, px, grid, batch, feats):
    row = {
        "vision": trace(lambda: tower(px, grid)),
        "tower_ms": elapsed(lambda: tower(px, grid)),
    }
    pack = FreshPack(batch, eng.dev)
    row["lm_tokens"] = len(pack.ids)
    row["lm_stages"] = stage_breakdown(eng, pack, feats)
    return row


def attention_probe(stage, tower, px, grid, count):
    _, (pos, cos, sin, lens) = tower._consts(grid, px.device)
    blk = tower.v.blocks[0]
    x = tower.v.patch_embed(px)
    z = blk.attn.qkv(blk.norm1(x + pos.to(x.dtype))).reshape(len(x), 3, tower.H, -1)
    from speed.engine import vision_rotary

    qr, kr = vision_rotary(z[:, 0], z[:, 1], cos, sin)
    shape = (count, lens[0], tower.H, -1)
    if stage == "patch_order":
        row = reduction_orders(tower.v.patch_embed, px)
    elif stage == "split":
        from scan.splitkv_probe import batched, load

        op = load()
        assert op is not None
        args = tuple(t.reshape(shape) for t in (qr, kr, z[:, 2]))
        streams = tower.attention_streams[: tower._worker_count(lens)]
        ref = image_attention(*args, streams)

        def run():
            return batched(op, *args, blk.attn.scaling)

        row = dict(
            **errors(run(), ref),
            ms=elapsed(run),
            stream_ms=elapsed(lambda: image_attention(*args, streams)),
            sms=torch.cuda.get_device_properties(px.device).multi_processor_count,
        )
    else:
        row = {
            "patch": patch_probe(tower.v.patch_embed, px),
            "attention": attention(*(t.reshape(shape) for t in (qr, kr, z[:, 2]))),
        }
    return row


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--stage",
        choices=(
            "kernels",
            "probe",
            "split",
            "patch_order",
            "gdn",
            "video_checks",
            "video_entry",
            "wide_gdn",
            "mlp",
            "norm",
            "patch_control",
            "patch_conv",
            "binary_control",
            "conv_control",
            "conv_binary",
        ),
        required=True,
    )
    a = ap.parse_args()
    if a.stage == "video_entry":
        rows = video_entry_checks(a)
        a.out.write_text(json.dumps(rows, indent=1))
        check_outputs(rows)
        return
    eng = Bench(a).to_engine()
    eng.frame_tower.graphs = False
    if a.stage == "video_checks":
        rows = video_checks(eng)
        a.out.write_text(json.dumps(rows, indent=1))
        assert all(r["features_vs_before"]["equal"] for r in rows)
        check_outputs([r["before"] for r in rows])
        return
    if a.stage in ("binary_control", "conv_binary"):
        results = {}
        for tv in (128, 256):
            results[str(tv)] = binary_patch_control(eng, tv, a.stage == "conv_binary")
            a.out.write_text(json.dumps(results, indent=1))
        return
    q = {"type": "noul", "instr": "Is there a person in the image?"}
    res = {}
    for tv in (128, 256):
        count = (
            (64 if tv == 256 else 32)
            if a.stage in ("kernels", "mlp", "norm", "patch_control", "patch_conv", "conv_control")
            else (32 if a.stage == "split" else 8)
        )
        frames = [
            r["img"]
            for r in load_images(
                "data/prep/eval_refcoco_val.jsonl",
                "/content/data/coco/train2014",
                count,
                tv,
                frame=True,
            )
        ]
        prep = eng.frame_prep
        batch = frame_layout(prep.template(frames[0].size, q, None), count)
        px = prep.pixels(upload_frames(frames))
        grid = batch["grid_thw"]
        tower = eng.frame_tower
        feats = tower(px, grid)
        if a.stage == "patch_conv":
            row = patch_convolutions(tower.v.patch_embed, px)
        elif a.stage in ("patch_control", "conv_control"):
            row = patch_control(eng, tower, px, grid, frames, a.stage == "conv_control")
        elif a.stage == "norm":
            row = norm_probe(tower, px, grid)
        elif a.stage == "mlp":
            row = mlp_probe(tower, px)
        elif a.stage in ("gdn", "wide_gdn"):
            row = gdn_probe(eng, tv, (64, 128) if a.stage == "wide_gdn" else (64, 32))
        elif a.stage == "kernels":
            row = kernel_profile(eng, tower, px, grid, batch, feats)
        else:
            row = attention_probe(a.stage, tower, px, grid, count)
        res[str(tv)] = row
        a.out.write_text(json.dumps(res, indent=1))
        print(a.stage, tv, flush=True)


if __name__ == "__main__":
    main()
