"""Profile one unchunked packed LM call or sweep its real BF16 projection shapes on engopt."""

import argparse
import json
from collections import defaultdict
from contextlib import contextmanager
from importlib.metadata import version
from math import prod
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import torch

from scan.bench_scan import Bench, diff
from scan.equivalence_limits import BF16_LIMIT
from scan.data import QUERIES, load_images
from scan.profile_fresh import kernel_group
from scan.varlen import FreshPack, packed_forward
from dv.data.mixed import collate_mixed


def elapsed(fn, count=5):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(count):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


def trace(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        fn()
        torch.cuda.synchronize()
    with TemporaryDirectory() as td:
        path = Path(td) / "trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    ops = {
        e["args"]["External id"]: e
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    rows = defaultdict(lambda: dict(calls=0, ms=0.0))
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = e.get("args", {})
        op = ops.get(args.get("External id"), {})
        inputs = op.get("args", {})
        key = (
            e["name"],
            op.get("name", ""),
            json.dumps(inputs.get("Input Dims", [])),
            json.dumps(inputs.get("Input type", [])),
        )
        row = rows[key]
        row["calls"] += 1
        row["ms"] += e["dur"] / 1000
        row["transfer_bytes"] = row.get("transfer_bytes", 0) + args.get("bytes", 0)
    result = []
    sizes = dict(float=4, double=8, long=8, int=4, short=2, bool=1, **{"c10::BFloat16": 2, "c10::Half": 2})
    for (name, op, dims, types), row in rows.items():
        dims, types = json.loads(dims), json.loads(types)
        row.update(kernel=name, op=op, dims=dims, types=types, group=kernel_group(name))
        if (
            op in ("aten::mm", "aten::bmm", "aten::addmm")
            and "reduce" not in name.lower()
            and "epilogue" not in name.lower()
        ):
            lhs, rhs = dims[-2:] if op != "aten::addmm" else dims[1:3]
            work = 2 * prod(lhs) * rhs[-1] * row["calls"]
            row["tflops"] = work / row["ms"] / 1e9
        # Logical tensor traffic for direct ATen kernels. Never attribute an enclosing operator's
        # full input traffic to each subkernel of a multi-kernel algorithm.
        if (
            op
            in (
                "aten::copy_",
                "aten::mul",
                "aten::add",
                "aten::silu",
                "aten::sigmoid",
                "aten::pow",
                "aten::neg",
                "aten::rsqrt",
                "aten::cos",
                "aten::sin",
                "aten::exp",
            )
            and dims
            and dims[0]
        ):
            operands = [prod(d) * sizes.get(t, 0) for d, t in zip(dims, types) if d]
            if operands and operands[0]:
                total = sum(operands) + (0 if op == "aten::copy_" else max(operands))
                row["logical_GBs"] = total * row["calls"] / row["ms"] / 1e6
        if row["transfer_bytes"]:
            row["transfer_GBs"] = row["transfer_bytes"] / row["ms"] / 1e6
        result.append(row)
    return sorted(result, key=lambda r: -r["ms"])


def dispatch(eng, result, save):
    from scan.lm_gemm_probe import sweep

    layer = eng.layers[0]
    att = next(l.self_attn for l, kind in zip(eng.layers, eng.types) if kind != "linear_attention")
    modules = dict(
        gdn_in=layer.linear_attn._in,
        gdn_out=layer.linear_attn.out_proj,
        qkv=att._qkv,
        attn_out=att.o_proj,
        gate_up=layer._gu,
        down=layer.mlp.down_proj,
    )
    for m in (4672, 9344, 18688, 37376):
        for key, mod in modules.items():
            print("dispatch", m, key, flush=True)
            result[f"{m}/{key}"] = sweep(m, mod.weight, elapsed, trace)
            save()


def stage_breakdown(eng, pack, feats):
    from contextlib import ExitStack
    from functools import wraps
    from unittest.mock import patch
    from scan import varlen

    records, parents = [], []

    def timed(name, fn):
        @wraps(fn)
        def call(*args, **kwargs):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            index = len(records)
            records.append((name, parents[-1] if parents else None, start, end))
            parents.append(index)
            start.record()
            try:
                return fn(*args, **kwargs)
            finally:
                end.record()
                parents.pop()

        return call

    packed_forward(eng, pack, feats)
    with ExitStack() as stack:

        def measure(obj, attr, name):
            stack.enter_context(patch.object(obj, attr, timed(name, getattr(obj, attr))))

        measure(varlen, "_gdn_layer", "GDN")
        measure(varlen, "_attention_layer", "LM attention")
        measure(eng, "_gdn_projections", "LM dense")
        measure(eng, "_attention_projections", "LM dense")
        measure(eng, "heads_forward", "heads")
        for layer, kind in zip(eng.layers, eng.types):
            mod = layer.linear_attn.out_proj if kind == "linear_attention" else layer.self_attn.o_proj
            measure(mod, "forward", "LM dense")
        samples = []
        for _ in range(3):
            records.clear()
            timed("LM dense", packed_forward)(eng, pack, feats)
            torch.cuda.synchronize()
            result = defaultdict(float)
            for name, parent, start, end in records:
                ms = start.elapsed_time(end)
                result[name] += ms
                if parent is not None:
                    result[records[parent][0]] -= ms
            samples.append(dict(result))
    return dict(samples=samples, median_ms={k: median(r[k] for r in samples) for k in samples[0]})


def graph_checks(eng, frames):
    from scan import vision_prep as prep
    from scan.varlen import packed_forward

    results = []
    # Also measure beyond the default crossover to quantify the cost of padding.
    eng.packed_graphs.max_tokens = 2048
    for mode in ("causal", "block_bidir"):
        eng.mask_mode = mode
        eng.packed_graphs.clear()
        for n in (1, 4, 8, 1):
            for i, text in enumerate(("a dog", "a red car parked on the left side", "a cat", "a dog")):
                units = frames[:n] if i % 2 == 0 else frames[:n][::-1]
                q = dict(type="noul", instr=f"Is there {text} in the image?")
                batch = prep.frame_layout(eng.frame_prep.template(units[0].size, q, None), n)
                pack = FreshPack(batch, eng.dev)
                feats = eng.fresh_vision(eng.frame_prep.pixels(prep.upload_frames(units)), batch["grid_thw"])
                eng.lm_prefill = False
                ref = packed_forward(eng, pack, feats)
                eng.lm_prefill = True
                eager = packed_forward(eng, pack, feats)
                before = len(eng.packed_graphs._graphs)
                g = lambda: packed_forward(eng, pack, feats, use_graph=True)
                actual = g()
                after = len(eng.packed_graphs._graphs)
                row = dict(
                    mode=mode,
                    n=n,
                    query=i,
                    tokens=len(pack.ids),
                    captures=after - before,
                    live_graphs=after,
                    reserved_GiB=torch.cuda.memory_reserved() / 2**30,
                    eager_ms=elapsed(lambda: packed_forward(eng, pack, feats)),
                    graph_ms=elapsed(g),
                    vs_eager=diff(actual, {"typed_logits": eager["typed_logits"]}),
                    vs_original=diff(actual, {"typed_logits": ref["typed_logits"]}),
                )
                results.append(row)
                print(mode, n, i, before, after, row["graph_ms"], row["vs_original"], flush=True)
                if after - before != int(i == 0 and before < 3) or row["vs_original"]["decision_agree"] != 1:
                    raise RuntimeError(f"graph check failed: {row}")
    return results


def graph_limits(eng, frames):
    from scan import vision_prep as prep

    cache = eng.packed_graphs
    cache.max_graphs = 1
    results = []
    short = "Is there a person?"
    long = "Is there a person " + "standing beside a bench " * 25 + "in the image?"
    for name, n, text in (
        ("capture", 1, short),
        ("question_overflow", 1, long),
        ("cache_full", 2, short),
        ("return", 1, short),
    ):
        q = dict(type="noul", instr=text)
        batch = prep.frame_layout(eng.frame_prep.template(frames[0].size, q, None), n)
        pack = FreshPack(batch, eng.dev)
        feats = eng.fresh_vision(eng.frame_prep.pixels(prep.upload_frames(frames[:n])), batch["grid_thw"])
        eager = packed_forward(eng, pack, feats)
        before = len(cache._graphs)
        actual = packed_forward(eng, pack, feats, use_graph=True)
        metrics = diff(actual, {"typed_logits": eager["typed_logits"]})
        if name in ("question_overflow", "cache_full"):
            assert torch.equal(actual["typed_logits"], eager["typed_logits"]), name
        results.append(
            dict(
                case=name,
                question_tokens=pack.question.maximum,
                captures=len(cache._graphs) - before,
                live_graphs=len(cache._graphs),
                **metrics,
            )
        )
        assert len(cache._graphs) == 1 and cache.max_graphs == 1
    check_outputs(results)
    return results


def fixed_features(eng, frames):
    from scan import vision_prep as prep

    results = []
    questions = [
        dict(type="box", text="the dog on the left"),
        dict(type="noul", instr="Is there a person in the image?"),
    ]
    for n in (32, 64, 128, 256):
        for q in questions:
            batch = prep.frame_layout(eng.frame_prep.template(frames[0].size, q, None), n)
            pack = FreshPack(batch, eng.dev)
            feats = eng.fresh_vision(eng.frame_prep.pixels(prep.upload_frames(frames[:n])), batch["grid_thw"])
            eng.lm_prefill = False
            reference = packed_forward(eng, pack, feats)
            eng.lm_prefill = True
            candidate = packed_forward(eng, pack, feats)
            if q["type"] == "box":
                fields = ("presence_logit", "iou_logit", "box_pred")
            else:
                fields = ("typed_logits",)
            stats = diff(candidate, {k: reference[k] for k in fields})
            row = dict(n=n, tokens=len(pack.ids), kind=q["type"], **stats)
            results.append(row)
            print(row, flush=True)
    return results


def flash_qla_single_card(q, k, v, g, beta, cu_seqlens=None, **unused):
    """FlashQLA 0.1.3 SM120 forward with no context partitioning or final state."""
    from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120 import kkt_solve, fused_gdr_fwd
    from flash_qla.ops.utils import chunk_local_cumsum

    g = chunk_local_cumsum(g=g, cu_seqlens=cu_seqlens, chunk_size=32)
    a = kkt_solve(k=k, b=beta, cu_seqlens=cu_seqlens, chunk_size=32)
    out, _, _ = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        b=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
        output_h=False,
        output_o=True,
    )
    return out, None


@contextmanager
def narrow_flash_qla():
    from importlib import import_module
    from unittest.mock import patch

    backend = import_module("flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_fwd")
    compile_tile = backend.tilelang_fused_chunk_gdr_fwd

    def compile_narrow(*args, **kwargs):
        return compile_tile(*args, **(kwargs | {"block_DV": 64}))

    with patch.object(backend, "tilelang_fused_chunk_gdr_fwd", compile_narrow):
        yield


@contextmanager
def large_packed_flash(eng):
    from unittest.mock import patch
    from dv.model import backbone

    chunk = backbone.fla_chunk_gdr

    def run(q, k, v, **kwargs):
        if (
            eng.lm_prefill
            and q.shape[0] == 1
            and q.shape[1] >= 4096
            and kwargs.get("cu_seqlens") is not None
            and not kwargs.get("use_qk_l2norm_in_kernel", True)
            and not kwargs.get("output_final_state")
            and kwargs.get("initial_state") is None
        ):
            return flash_qla_single_card(q, k, v, **kwargs)
        return chunk(q, k, v, **kwargs)

    with narrow_flash_qla(), patch.object(eng, "flash_gdn", None), patch.object(backbone, "fla_chunk_gdr", run):
        yield


def delta_sweep(eng, frames, flash_kernel=None):
    from dv.model.backbone import fla_chunk_gdr
    from speed.lm_prefill import conv_qkv
    from scan.varlen import Segments

    layer = eng.layers[0]
    m = layer.linear_attn
    rows = []
    for n in (32, 128):
        for width in (146, 281):
            seq = Segments([width] * n, eng.dev)
            # Normalize synthetic hidden rows and project them with the checkpoint weights.
            hidden = torch.randn((n * width, eng.net.hidden), device=eng.dev, dtype=torch.bfloat16)
            mixed, z, beta, g = eng._gdn_projections(m, layer.input_layernorm(hidden))
            q, k, v = conv_qkv(m, mixed, seq.token_starts())
            for dense in (False, True):

                def run(size):
                    shape = (n, width) if dense else (1, n * width)
                    return fla_chunk_gdr(
                        *(a.reshape(*shape, *a.shape[2:]) for a in (q, k, v)),
                        g=g.reshape(*shape, -1),
                        beta=beta.reshape(*shape, -1),
                        cu_seqlens=None if dense else seq.cu,
                        cu_seqlens_cpu=None if dense else seq.cpu,
                        chunk_size=size,
                        use_qk_l2norm_in_kernel=False,
                    )[0].reshape(n, width, -1)

                if not dense:
                    ref = run(64)
                if flash_kernel is not None:
                    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

                    for name, fn in [
                        ("persistent_recurrent", fused_recurrent_gated_delta_rule),
                        ("flash_qla", flash_kernel),
                    ]:

                        def candidate():
                            layout = (n, width) if dense else (1, n * width)
                            extra = dict(auto_cp=False) if name == "flash_qla" else {}
                            return fn(
                                *(a.reshape(*layout, *a.shape[2:]) for a in (q, k, v)),
                                g=g.reshape(*layout, -1),
                                beta=beta.reshape(*layout, -1),
                                cu_seqlens=None if dense else seq.cu,
                                use_qk_l2norm_in_kernel=False,
                                **extra,
                            )[0].reshape(n, width, -1)

                        try:
                            out = candidate()
                            rows.append(
                                dict(
                                    n=n,
                                    width=width,
                                    dense=dense,
                                    backend=name,
                                    ms=elapsed(candidate),
                                    max_abs=(out - ref).abs().max().item(),
                                    mean_abs=(out - ref).abs().float().mean().item(),
                                )
                            )
                        except (RuntimeError, ValueError, AssertionError, NotImplementedError) as exc:
                            rows.append(dict(n=n, width=width, dense=dense, backend=name, error=str(exc)[-800:]))
                        print(rows[-1], flush=True)
                    continue
                for size in (64, 32, 16):
                    out = run(size)
                    rows.append(
                        dict(
                            n=n,
                            width=width,
                            dense=dense,
                            chunk_size=size,
                            ms=elapsed(lambda: run(size)),
                            max_abs=(out - ref).abs().max().item(),
                            mean_abs=(out - ref).abs().float().mean().item(),
                        )
                    )
                    print(rows[-1], flush=True)
    return rows


def video_equivalence(eng):
    from speed import workload as W
    from speed.bench_glance import make_item
    from speed.engine import prepare

    frames = W.load_coco("data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014", 100, max_vis=256)
    results = []
    for n in (32, 64):
        for c in range(4):
            ids = [(c * n + j) % len(frames) for j in range(n)]
            batch = collate_mixed(
                [make_item(eng.tok, eng.proc, frames[i]["img"], W.questions(frames, i, 1, "box")) for i in ids],
                eng.tok,
            )
            feats = eng.feats(batch)
            inputs, host = prepare(batch, Nv=256)
            inputs = eng.to_device(inputs, host, feats)
            eng.lm_prefill = False
            original = eng._core(inputs, host)[0]
            eng.lm_prefill = True
            cand = eng.answer_independent(batch, feats)
            fields = ("box_pred", "presence_logit", "iou_logit")
            metrics = diff(cand, {k: original[k][:n] for k in fields})
            print(n, c, metrics, flush=True)
            results.append(dict(n=n, call=c, **metrics))
    return results


def check_outputs(rows):
    fields = {
        "prob_absdiff_max": "typed_prob",
        "p_absdiff_max": "presence_prob",
        "box_absdiff_max": "box_pred",
        "piou_absdiff_max": "iou_prob",
    }
    for row in rows:
        assert row["decision_agree"] == 1.0, row
        for measured, gate in fields.items():
            if measured in row:
                assert row[measured] <= BF16_LIMIT[gate], (gate, row)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.lm_prefill = {
        "profile_before": "reference",
        "stages_before": "reference",
        "ncu_before": "reference",
        "profile_after": "pointwise",
        "profile_final": "direct",
        "ncu_final": "direct",
    }.get(a.stage, "fused")
    bench = Bench(a)
    eng = bench.to_engine()
    res = dict(
        cublas=version("nvidia-cublas"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        triton=version("triton"),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_gdn_available=eng.flash_gdn is not None,
        results={},
    )

    def save():
        a.out.write_text(json.dumps(res, indent=1))

    if a.stage == "tma":
        from scan.lm_gemm_probe import tma_sweep

        for m in (4672, 9344, 18688, 37376):
            res["results"][m] = tma_sweep(m, eng.layers[0]._gu.weight, elapsed)
            print(m, res["results"][m], flush=True)
            save()
        return
    if a.stage.startswith("dispatch"):
        dispatch(eng, res["results"], save)
        return
    if a.stage == "qla_fresh":
        from scan.bench_scan import part_fresh

        a.prep, a.image_root = "data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014"
        a.Tvs, a.Ns_fresh = [128, 256], [32, 64, 128, 256]
        a.n_calls, a.fresh_kind, a.fresh_chunked, a.max_fresh_tokens = 10, "yesno", 1, 16384
        with large_packed_flash(eng):
            part_fresh(bench, a, res)
        save()
        return
    if a.stage == "video_equiv":
        res["results"] = video_equivalence(eng)
        save()
        check_outputs(res["results"])
        return
    frames = [
        r["img"]
        for r in load_images(
            "data/prep/eval_refcoco_val.jsonl", "/content/data/coco/train2014", 256, 128, frame=True
        )
    ]
    if a.stage == "qla_model":
        with large_packed_flash(eng):
            res["results"]["fixed_features"] = fixed_features(eng, frames)
            save()
            res["results"]["video"] = video_equivalence(eng)
            save()
            q = dict(type="noul", instr=f"Is there {QUERIES[0]} in the image?")
            timed = {}
            for n in (32, 64, 128, 256):
                batch = collate_mixed(eng._items(frames[:n], [[q]] * n, None), eng.tok, pad_to=1)
                pack, feats = FreshPack(batch, eng.dev), eng.feats(batch)
                timed[n] = elapsed(lambda: packed_forward(eng, pack, feats))
                print(n, timed[n], flush=True)
            res["results"]["median_ms"] = timed
            save()
        check_outputs(res["results"]["fixed_features"])
        check_outputs(res["results"]["video"])
        return
    if a.stage == "graphs_limits":
        res["results"] = graph_limits(eng, frames)
        save()
        return
    if a.stage == "gdn_tile64":
        # The public SM120 wrapper hardcodes 128. Split the independent value lanes in two
        # to stay under this device's per-CTA shared memory limit. Scope the override to this probe.
        with narrow_flash_qla():
            res["results"] = delta_sweep(eng, frames, flash_qla_single_card)
        save()
        return
    if a.stage in ("gdn_optional", "gdn_single_card"):
        flash_kernel = flash_qla_single_card
        if a.stage == "gdn_optional":
            from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

            flash_kernel = chunk_gated_delta_rule
        res["results"] = delta_sweep(eng, frames, flash_kernel)
        save()
        return
    if a.stage in ("graphs", "graphs32", "gdn", "fixed_features"):
        res["results"] = {
            "graphs": graph_checks,
            "graphs32": graph_checks,
            "gdn": delta_sweep,
            "fixed_features": fixed_features,
        }[a.stage](eng, frames)
        save()
        if a.stage == "fixed_features":
            check_outputs(res["results"])
        elif a.stage.startswith("graphs"):
            for row in res["results"]:
                check_outputs([row["vs_eager"], row["vs_original"]])
        return
    q = dict(type="noul", instr=f"Is there {QUERIES[0]} in the image?")
    eng.fv.graphs = False
    if a.stage.startswith("stages_"):
        for n in (32, 64, 128, 256):
            batch = collate_mixed(eng._items(frames[:n], [[q]] * n, None), eng.tok, pad_to=1)
            pack, feats = FreshPack(batch, eng.dev), eng.feats(batch)
            res["results"][n] = stage_breakdown(eng, pack, feats)
            print(n, res["results"][n], flush=True)
            save()
        return
    if a.stage.startswith("ncu_"):
        n = a.n
        batch = collate_mixed(eng._items(frames[:n], [[q]] * n, None), eng.tok, pad_to=1)
        pack, feats = FreshPack(batch, eng.dev), eng.feats(batch)
        for _ in range(2):
            packed_forward(eng, pack, feats)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        packed_forward(eng, pack, feats)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    for n in (32, 64, 128, 256):
        batch = collate_mixed(eng._items(frames[:n], [[q]] * n, None), eng.tok, pad_to=1)
        pack, feats = FreshPack(batch, eng.dev), eng.feats(batch)

        def run():
            return packed_forward(eng, pack, feats)

        run()
        row = dict(tokens=len(pack.ids), ms=elapsed(run), kernels=trace(run))
        res["results"][n] = row
        print(n, len(pack.ids), row["ms"], flush=True)
        save()
        del batch, pack, feats


if __name__ == "__main__":
    main()
