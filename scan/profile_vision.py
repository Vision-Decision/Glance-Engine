"""Fresh-frame stage measurements. Use the serialized runner in scan/vm/vision_prep.sh."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scan.bench_scan import Bench
from scan.data import load_images
import scan.engine as engine
from scan.profile_fresh import kernel_group


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


class Timers:
    def __init__(self):
        self.cpu = defaultdict(float)
        self.gpu = defaultdict(list)

    def host(self, name, fn):
        def call(*args, **kw):
            start = time.perf_counter()
            result = fn(*args, **kw)
            self.cpu[name] += (time.perf_counter() - start) * 1000
            return result
        return call

    def device(self, name, fn):
        def call(*args, **kw):
            start = event()
            result = fn(*args, **kw)
            self.gpu[name].append((start, event()))
            return result
        return call

    def result(self):
        torch.cuda.synchronize()
        return dict(self.cpu) | {name: sum(s.elapsed_time(e) for s, e in pairs) for name, pairs in self.gpu.items()}


def profile_call(eng, images, q, after):
    # Instrument the eager tower. Graph replay does not execute Python hooks.
    times = Timers()
    layers = eng.fv.v.blocks
    with ExitStack() as stack:
        def wrap(obj, attr, name, method):
            stack.enter_context(patch.object(obj, attr, method(name, getattr(obj, attr))))
        wrap(type(eng.proc.image_processor), '__call__', 'hf_processor', times.host)
        wrap(engine, 'question_block', 'tokenize', times.host)
        wrap(engine, 'collate_mixed', 'collate', times.host)
        wrap(F, 'scaled_dot_product_attention', 'sdpa', times.device)
        wrap(eng.fv.v.merger, 'forward', 'merger', times.device)
        projection = getattr(eng.frame_tower, 'mlp_projection', None) if after else None
        if projection is not None:
            wrap(eng.frame_tower, 'mlp_projection', 'block_modules', times.device)
        for blk in layers:
            mlp = (blk.mlp.linear_fc1, blk.mlp.act_fn, blk.mlp.linear_fc2) if projection is not None else (blk.mlp,)
            for mod in (blk.norm1, blk.norm2, blk.attn.qkv, blk.attn.proj, *mlp):
                wrap(mod, 'forward', 'block_modules', times.device)
        wrap(eng.fv.Q, 'apply_rotary_pos_emb_vision', 'rope', times.device)
        if after:
            import speed.engine as se
            wrap(se, 'vision_rotary', 'rope', times.device)
            wrap(se, 'image_attention', 'attention_wall', times.device)
            wrap(eng.fv.v.patch_embed, 'forward', 'patch_embed', times.device)
        # The only F.linear using this particular weight is the unfused patch embedding.
        linear = F.linear
        def timed_linear(x, w, b=None):
            if w is eng.fv.W:
                return times.device('patch_gemm', linear)(x, w, b)
            return linear(x, w, b)
        stack.enter_context(patch.object(F, 'linear', timed_linear))
        wrap(eng, 'fresh_vision' if after else 'vision', 'tower', times.device)
        # The in-scope feature scatter is the first index_copy_ in each packed_forward.
        packed = engine.packed_forward
        def with_scatter(*args, **kw):
            original = torch.Tensor.index_copy_
            # Identify this operation by its source tensor, before any LM block runs.
            feats = args[2]
            def select(x, dim, idx, src):
                if src.data_ptr() == feats.data_ptr():
                    return times.device('feature_scatter', original)(x, dim, idx, src)
                return original(x, dim, idx, src)
            with patch.object(torch.Tensor, 'index_copy_', select):
                return packed(*args, **kw)
        stack.enter_context(patch.object(engine, 'packed_forward', with_scatter))
        tensor_to = torch.Tensor.to
        def copy(t, *a, **kw):
            if t.device.type == 'cpu' and t.numel() > 0 and t.dtype in (torch.float32, torch.uint8) and t.ndim >= 2:
                target = a[0] if a else kw.get('device')
                if isinstance(target, (torch.device, str)) and torch.device(target).type == 'cuda' and (t.shape[-1] == 1536 or t.dtype == torch.uint8):
                    return times.device('pixel_h2d', tensor_to)(t, *a, **kw)
            return tensor_to(t, *a, **kw)
        stack.enter_context(patch.object(torch.Tensor, 'to', copy))
        if after:
            from scan import vision_prep
            wrap(vision_prep, 'upload_frames', 'frame_ingest', times.host)
            wrap(vision_prep, 'frame_layout', 'collate', times.host)
            wrap(vision_prep, 'patch_pixels', 'patchify', times.device)
            if eng.frame_prep.fused:
                from speed import frame_pixels
                wrap(frame_pixels, 'patches', 'patchify', times.device)
        wall = {}
        eng.fresh(images, q, timings=wall)
    row = times.result()
    return row | {f'wall_{k}': v for k, v in wall.items()}


def trace(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    kernels = defaultdict(float)
    groups = defaultdict(float)
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            ms = e.time_range.elapsed_us() / 1000
            kernels[e.name] += ms
            groups[kernel_group(e.name)] += ms
    return dict(groups=dict(groups), kernels=dict(sorted(kernels.items(), key=lambda pair: -pair[1])))


def elapsed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = event()
        fn()
        end = event()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


def errors(x, y):
    diff = (x.float() - y.float()).abs()
    return dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                relative_max=(diff.max() / y.float().abs().max()).item(), equal=torch.equal(x, y))


def check_question_layouts(eng, ims):
    from scan.vision_prep import frame_layout
    from scan.varlen import FreshPack
    variants = (dict(type='box', text='clock'), dict(type='noul', instr='Is there a person?', block='short'),
                dict(type='choice', instr='What color?', options=['red', 'green', 'blue']),
                dict(type='score', instr='Rate the lighting.', options=['poor', 'good']))
    for q in variants:
        for first in (None, q.get('text') or q.get('instr')):
            items = eng._items(ims, [[q]] * len(ims), first)
            batch = engine.collate_mixed(items, eng.tok, pad_to=1)
            fast = frame_layout(eng.frame_prep.template(ims[0].size, q, first), len(ims))
            expected, actual = (FreshPack(b, eng.dev) for b in (batch, fast))
            for name in ('ids', 'pos', 'img_rows', 'p_idx', 'q_idx'):
                assert getattr(expected, name).equal(getattr(actual, name)), (q, first, name)
            assert all(t.equal(actual.heads[k]) for k, t in expected.heads.items())
    return len(variants) * 2


def check_features(eng, frames):
    from scan.vision_prep import frame_layout, upload_frames
    from scan.varlen import FreshPack
    q = dict(type='noul', instr='Is a person present?')
    res = {}
    layout_checks = check_question_layouts(eng, frames[:3])
    for n in (1, 8, 32, 64, 128):
        ims = frames[:n]
        items = eng._items(ims, [[q]] * n, None)
        batch = engine.collate_mixed(items, eng.tok, pad_to=1)
        ref_px = batch['pixel_values'].cuda()
        grid = batch['grid_thw']
        prep = eng.frame_prep
        assert prep.supports(ims)
        host = upload_frames(ims)
        px = prep.pixels(host)
        device_px = prep.pixels(host, dtype=eng.dtype) if prep.fused else px
        expected_px = ref_px.to(device_px.dtype)
        assert torch.equal(device_px, expected_px)
        layout = frame_layout(prep.template(ims[0].size, q, None), n)
        p, r = (FreshPack(b, eng.dev) for b in (batch, layout))
        for name in ('ids', 'pos', 'img_rows', 'p_idx', 'q_idx'):
            assert torch.equal(getattr(p, name), getattr(r, name)), name
        assert all(torch.equal(p.heads[k], r.heads[k]) for k in p.heads)
        assert torch.equal(px, ref_px)
        ref = eng.net.visual(ref_px, grid.cuda()).pooler_output
        legacy = eng.fv(px, grid)
        actual = eng.fresh_vision(device_px, grid)
        single = torch.cat([eng.net.visual(v, g[None].cuda()).pooler_output
                            for v, g in zip(ref_px.chunk(n), grid)])
        res[str(n)] = dict(pixel=errors(px, ref_px), device_pixel=errors(device_px, expected_px), layout_equal=True, question_layout_checks=layout_checks, old_vs_hf=errors(legacy, ref),
                           new_vs_hf=errors(actual, ref), hf_batch_vs_single=errors(ref, single),
                           new_vs_single=errors(actual, single))
    return res


def check_graph(eng, frames):
    from scan.vision_prep import upload_frames
    eng.frame_tower.graphs = True
    rows = []
    before = len(eng.frame_tower._graphs)
    for images in (frames, frames[::-1]):
        px = eng.frame_prep.pixels(upload_frames(images), dtype=eng.dtype)
        grid = eng.frame_prep.template(images[0].size, dict(type='box', text='person'), None)['grid_thw']
        grid = grid.expand(len(images), -1)
        expected = eng.net.visual(px, grid.cuda()).pooler_output
        actual = eng.fresh_vision(px, grid)
        rows.append(dict(**errors(actual, expected), graph_entries=len(eng.frame_tower._graphs)))
        assert rows[-1]['equal'] and len(eng.frame_tower._graphs) == before + 1, rows[-1]
    eng.free_graphs()
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', required=True)
    a = ap.parse_args()
    after = a.stage.startswith('features') or a.stage in ('profile_after', 'profile_final', 'profile_fast', 'graph_features')
    a.vision_prep = 'fused' if after else 'reference'
    bench = Bench(a)
    eng = bench.to_engine()
    eng.fv.graphs = False
    if getattr(eng, "frame_tower", None) is not None:
        eng.frame_tower.graphs = False
    q = dict(type='noul', instr='Is there a person standing in the image?')
    res = {}
    for tv in (128, 256):
        count = 256 if a.stage.startswith('profile') else 128
        frames = [r['img'] for r in load_images('data/prep/eval_refcoco_val.jsonl', '/content/data/coco/train2014',
                                                32 if a.stage == 'tune' else count, tv, frame=True)]
        if a.stage == 'graph_features':
            n = 128 if tv == 128 else 64
            res[f'Tv{tv}_N{n}'] = check_graph(eng, frames[:n])
        elif a.stage.startswith('features'):
            res[str(tv)] = check_features(eng, frames)
            Path(a.out).write_text(json.dumps(res, indent=1))
            assert all(row['new_vs_hf']['equal'] for row in res[str(tv)].values()), 'vision differs from batched HF'
        elif a.stage == 'tune':
            batch = eng.build(frames, [[q]] * len(frames))[0]
            pixels = batch['pixel_values'].cuda()
            grid = batch['grid_thw']
            rows = {}
            for backend in ('cublas', 'cublaslt'):
                try:
                    torch.backends.cuda.preferred_blas_library(backend)
                    for n in (1, 2, 4, 8, 16, 32):
                        pv = pixels[:pixels.shape[0] * n // len(frames)]
                        fn = lambda: eng.vision(pv, grid[:n])
                        rows[f'{backend}_{n}'] = elapsed(fn)
                        if n == 8:
                            rows[f'{backend}_trace'] = trace(fn)
                except RuntimeError as exc:
                    rows[backend + '_error'] = str(exc)
            res[str(tv)] = rows
        else:
            for n in (32, 64, 128, 256):
                ims = frames[:n]
                eng.fresh(ims, q)
                samples = [profile_call(eng, ims, q, after) for _ in range(3)]
                res[f'Tv{tv}_N{n}'] = {k: median([s[k] for s in samples]) for k in samples[0]}
                print(tv, n, res[f'Tv{tv}_N{n}'], flush=True)
                Path(a.out).write_text(json.dumps(res, indent=1))
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
