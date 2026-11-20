"""Branch state split, output accuracy, and warmed isolated mixer timings."""

import json
import statistics

import torch
import torch.nn.functional as F
from scan.bench_scan import Bench
from scan.data import load_images
from speed.gdn_branch import branch_core, branch_output, load_chunk_branch


def ulps(actual, expected):
    def ordered(x):
        i = x.to(torch.bfloat16).contiguous().view(torch.int16).int()
        return torch.where(i < 0, -32768 - i, i)

    dist = (ordered(actual) - ordered(expected)).abs()
    return dict(
        max_abs=(actual.float() - expected.float()).abs().max().item(),
        max_bf16_ulp=dist.max().item(),
        within_one=int((dist <= 1).sum()),
        exact=int((dist == 0).sum()),
        count=dist.numel(),
    )


def timed(fn):
    for _ in range(5):
        fn()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(22)]
    for i in range(0, len(events), 2):
        events[i].record()
        fn()
        events[i + 1].record()
    torch.cuda.synchronize()
    return statistics.median(events[i].elapsed_time(events[i + 1]) for i in range(0, len(events), 2))


def fork_copies(bank):
    """Replay the two removed gather paths over all actual layers on the same stream."""
    states = [c[0][:1] for kind, c in zip(bank.types, bank.cache) if kind == "linear_attention"]
    kv = [(k[:1], v[:1]) for kind, (k, v) in zip(bank.types, bank.cache) if kind != "linear_attention"]
    report = {}
    for questions in (1, 8, 32):
        img = torch.zeros(questions, device=states[0].device, dtype=torch.long)
        spare = kv[0][0].new_zeros(questions, kv[0][0].shape[1], 16, kv[0][0].shape[-1])

        def state_gather():
            for state in states:
                state[img].contiguous()

        def kv_gather_cat():
            for k, v in kv:
                torch.cat((k[img], spare), dim=2)
                torch.cat((v[img], spare), dim=2)

        times = {}
        for name, fn in (("state_gather", state_gather), ("kv_gather_cat", kv_gather_cat)):
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            times[name + "_ms"] = timed(graph.replay)
            del graph
        times.update(
            state_gather_dst_bytes=questions * sum(t.nbytes for t in states),
            kv_gather_dst_bytes=questions * sum(k.nbytes + v.nbytes for k, v in kv),
            prefix_slots=kv[0][0].shape[2],
            block_slots=16,
        )
        report[str(questions)] = times
    return report


@torch.no_grad()
def micro(a):
    torch.manual_seed(41)
    eng = Bench(a).to_engine()
    eng.branch_output = eng.question_branch = None
    chunk_branch = load_chunk_branch()
    bank = eng.encode([x["img"] for x in load_images(a.prep, a.image_root, 4, 640)])
    result = {"fork_copies": fork_copies(bank)}
    for li in (0, 14, 30):
        m = eng.layers[li].linear_attn
        s, tail = bank.cache[li]
        hi = s.bfloat16().float()
        mid = (s - hi).bfloat16().float()
        lo = (s - hi - mid).bfloat16().float()
        result[f"state_{li}"] = dict(
            count=s.numel(),
            reconstructed_exact=int((hi + mid + lo == s).sum()),
            max_fp32_abs=(hi + mid + lo - s).abs().max().item(),
        )
        # Isolate the state contraction from the triangular solve and its reduction order.
        probe = torch.randn((s.shape[1], 32, 128), device=s.device, dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = False
        oracle = torch.bmm(probe.float(), s[0])

        def product(part):
            return torch.bmm(probe, part[0].bfloat16(), out_dtype=torch.float32)

        high = product(hi)
        triple = high + (product(mid) + product(lo))
        result[f"state_{li}"].update(high_only=ulps(high, oracle), split_product=ulps(triple, oracle))
        for b, t, shared in [(1, 13, True), (8, 23, True), (32, 64, True), (64, 13, False), (256, 25, False)]:
            if li != 0 and b not in (1, 8):
                continue
            n = b * t
            rows = torch.randn(n, eng.net.hidden, device="cuda", dtype=torch.bfloat16)
            x, z, beta, g = eng._gdn_projections(m, rows)
            idx = torch.arange(b, device="cuda") % len(s)
            state, tails = (s[:1], tail[:1]) if shared else (s[idx], tail[idx])
            img = torch.zeros_like(idx) if shared else idx
            offsets = torch.arange(b + 1, device="cuda", dtype=torch.int32) * t if shared else None
            args = (m, x, z, beta, g, state, tails, b, t, offsets, img if shared else None)
            old = (
                (
                    lambda: eng._mix_gdn_q(
                        m,
                        x.view(b, t, -1),
                        z.view(b, t, -1),
                        beta.view(b, t, -1),
                        g.view(b, t, -1),
                        b,
                        t,
                        state,
                        tails,
                        img,
                    )
                )
                if shared
                else (
                    lambda: eng._mix_gdn_branch(
                        m, x.view(b, t, -1), z.view(b, t, -1), beta.view(b, t, -1), g.view(b, t, -1), b, t, state, tails
                    )
                )
            )
            d = dict(
                reference_ms=timed(old),
                wy_ms=timed(lambda: branch_output(*args)),
                recurrent_ms=timed(lambda: branch_output(*args, recurrent=True)),
            )
            if shared and b >= 8:
                d["inline_wy_ms"] = timed(lambda: branch_output(*args, staged=False))
                d["staged_wy_ms"] = timed(lambda: branch_output(*args, staged=True))
            ref = old().reshape(n, -1)
            if shared and chunk_branch is not None:

                def chunk():
                    return chunk_branch(*args, groups=(None, None))

                d["chunk_ms"] = timed(chunk)
                d["chunk_vs_fla"] = ulps(chunk(), ref)
            d["post_norm_vs_fla"] = ulps(branch_output(*args), ref)
            if b <= 8:
                # Explicit sequential FP32 recurrence on the same rounded convolution.
                xpad = torch.cat((tails[img], x.view(b, t, -1).transpose(1, 2)), -1)
                from speed.engine import conv_taps

                q, k, v = (
                    F.silu(conv_taps(xpad, m.conv1d.weight, m.conv1d.bias))
                    .transpose(1, 2)
                    .split([m.key_dim, m.key_dim, m.value_dim], -1)
                )

                def norm(v):
                    v = v.reshape(b, t, -1, 128).float().repeat_interleave(2, 2)
                    return v * torch.rsqrt(v.square().sum(-1, keepdim=True) + 1.0e-6)

                q, k, v = norm(q) * 128**-0.5, norm(k), v.reshape(b, t, -1, 128).float()
                h = state[img].float().clone()
                ys = []
                for i in range(t):
                    h *= g.view(b, t, -1)[:, i, :, None, None].exp()
                    delta = (v[:, i] - (h * k[:, i, :, :, None]).sum(-2)) * beta.view(b, t, -1)[:, i, :, None]
                    h += k[:, i, :, :, None] * delta.unsqueeze(-2)
                    ys.append((h * q[:, i, :, :, None]).sum(-2))
                expected = torch.stack(ys, 1).reshape(n, -1, 128)
                actual = branch_core(m, x, beta, g, state, tails, b, t, offsets, img)
                d["core_vs_fp32"] = ulps(actual, expected)
            result[f"L{li}_B{b}_T{t}_shared{int(shared)}"] = d
            print(li, b, t, d, flush=True)
            with open(a.out, "w") as f:
                json.dump(result, f, indent=1)


@torch.no_grad()
def compare_questions(a):
    """Pair the original and production engines on the same checkpoint and question order."""
    from speed.bench_glance import Runner
    from speed.engine import GlanceEngine
    from speed import workload as W
    from dv.data.mixed import QTYPES

    r = Runner(a)
    r.set_mode("eng_fused")
    current = r.eng
    reference = GlanceEngine(r.net, r.heads, graphs=True, fast_ops=True)
    reference.branch_attention = reference.branch_output = reference.question_branch = None
    pool = W.load_coco(a.prep, a.image_root, 12)
    results = {}
    for mask in ("block_bidir", "causal"):
        for e in (reference, current):
            e.mask_mode = mask
            e._graphs.clear()
            e._pool = None
        for batch, nq in [(1, 1), (1, 8), (1, 32), (3, 8)]:
            for seed in (0, 1, 2):
                items = [(seed + j) % len(pool) for j in range(batch)]
                qs = [W.questions(pool, i, nq - j, "mixed", seed=seed) for j, i in enumerate(items)]
                imgs = [pool[i]["img"] for i in items]
                pair = []
                for e in (reference, current):
                    r.eng = e
                    ans = r.call(imgs, qs, Nv=640, keep_prefix=batch == 1)[0]
                    pair.append([ans])
                    if batch == 1:
                        pair[-1].append(r.followup([W.questions(pool, items[0], nq, "mixed", seed=seed + 1)])[0])
                for label, x, y in zip(("full", "followup"), *pair):
                    box = x["qtype"] == QTYPES["box"]
                    typed = ~box
                    present = [v["presence_logit"][box].sigmoid() for v in (x, y)]
                    tp = [v["typed_logits"][typed].softmax(-1) for v in (x, y)]
                    ok = ((present[0] >= 0.5) == (present[1] >= 0.5)).sum().item()
                    if typed.any():
                        ok += (tp[0].argmax(-1) == tp[1].argmax(-1)).sum().item()
                    results[f"{mask}_{batch}_{nq}_{seed}_{label}"] = dict(
                        agreed=ok,
                        n=len(box),
                        presence_abs=(present[0] - present[1]).abs().max().item(),
                        typed_abs=(tp[0] - tp[1]).abs().max().item() if typed.any() else 0,
                        box_abs=(x["box_pred"][box] - y["box_pred"][box]).abs().max().item(),
                    )
        print(mask, sum(v["agreed"] for v in results.values()), sum(v["n"] for v in results.values()), flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    assert all(v["agreed"] == v["n"] for v in results.values())


@torch.no_grad()
def clip_controls(a):
    """Isolate recurrence and attention arithmetic on one fixed native clip bank."""
    from functools import partial
    from scan.bench_scan import diff, make_clips
    from speed.gdn_branch import branch_attention

    bench = Bench(a)
    clips = make_clips(a, 24)
    q = dict(type="noul", instr="Is a person walking in this clip?")
    ref = bench.reference(clips, q, dict(dt=0.5, max_vis=96))
    eng = bench.to_engine()
    bank = eng.encode(clips, chunk=16)
    result = {}
    for label, mixer, attn in [
        ("reference", None, None),
        ("packed_fla", branch_output, branch_attention),
        ("wy", branch_output, None),
        ("attention", None, branch_attention),
        ("both", branch_output, branch_attention),
        ("serial", partial(branch_output, recurrent=True), None),
        ("serial_attention", partial(branch_output, recurrent=True), branch_attention),
    ]:
        # Exercise each arithmetic path on identical video features.
        bank.has_clips = label == "packed_fla"
        eng.branch_output, eng.branch_attention = mixer, attn
        eng.free_graphs(bank)
        out = eng.query(bank, q)
        result[label] = diff(out, ref)
        print(label, result[label], flush=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
