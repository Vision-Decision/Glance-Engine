"""Inspect vision attention dispatch and test native-target BF16 GEMMs on the E1b shapes."""
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from scan.bench_scan import Bench
from scan.data import load_images
from scan.profile_vision import elapsed, errors, trace


@triton.jit
def _projection(X, W, Bias, Y, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    jn = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for b in range(K // BK):
        x = tl.load(X + im[:, None] * K + b * BK + kk[None, :], im[:, None] < M, 0)
        w = tl.load(W + jn[:, None] * K + b * BK + kk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    acc += tl.load(Bias + jn)[None, :]
    tl.store(Y + im[:, None] * N + jn[None, :], acc, im[:, None] < M)


def projections(eng):
    torch.manual_seed(0)
    result = {}
    modules = eng.fv.v.blocks[0]
    for m in (3744, 14976, 59904):
        for mod in (modules.attn.qkv, modules.attn.proj, modules.mlp.linear_fc1, modules.mlp.linear_fc2):
            n, k = mod.weight.shape
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            y = torch.empty((m, n), device='cuda', dtype=x.dtype)
            ref = mod(x)
            row = dict(pytorch_ms=elapsed(lambda: mod(x)), candidates={})
            for bm, bn, bk, warps in ((64, 128, 32, 4), (128, 128, 64, 8), (128, 256, 32, 8), (64, 64, 64, 4)):
                def run():
                    return _projection[(triton.cdiv(m, bm), n // bn)](x, mod.weight, mod.bias, y, m, n, k, bm, bn, bk,
                                                                       num_warps=warps, num_stages=3)
                compiled = run()
                row['target'] = [s.strip() for s in compiled.asm['ptx'].splitlines() if s.strip().startswith('.target')][0]
                row['candidates'][str((bm, bn, bk, warps))] = dict(ms=elapsed(run), max_abs=(y-ref).abs().max().item())
            result[f'{m}x{k}x{n}'] = row
    return result


def attention(eng, frames):
    ims = frames[:8]
    q = dict(type='box', text='dog')
    batch = eng.build(ims, [[q]] * len(ims))[0]
    px, grid = batch['pixel_values'].cuda(), batch['grid_thw']
    _, (pos, cos, sin, lens) = eng.fv._consts(grid, px.device)
    v, b = eng.fv.v, eng.fv.v.blocks[0]
    x = v.patch_embed(px)
    x = b.norm1(x + pos.to(x.dtype))
    h = b.attn.num_heads
    q, k, val = b.attn.qkv(x).reshape(len(x), 3, h, -1).unbind(1)
    from speed.engine import vision_rotary
    qr, kr = eng.fv.Q.apply_rotary_pos_emb_vision(q, k, cos, sin)
    qf, kf = vision_rotary(q, k, cos, sin)
    res = dict(rotary_exact=bool(qf.equal(qr) and kf.equal(kr)), q_strides=[qr.stride(), qf.stride()])
    def shaped(t):
        return t.reshape(len(lens), lens[0], h, -1).transpose(1, 2)
    inp = tuple(map(shaped, (qr, kr, val)))
    def one(i):
        return F.scaled_dot_product_attention(*(t[i:i+1] for t in inp), scale=b.attn.scaling)
    def every():
        return torch.cat([one(i) for i in range(len(ims))])
    ref = every()
    res['single_kernels'] = trace(lambda: one(0))['kernels']
    res['single_loop_ms'] = elapsed(every)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.nn.attention.varlen import varlen_attn
    import inspect
    res['varlen_signature'] = str(inspect.signature(varlen_attn))
    for backend in ('FLASH_ATTENTION', 'CUDNN_ATTENTION', 'EFFICIENT_ATTENTION'):
        try:
            with sdpa_kernel(getattr(SDPBackend, backend)):
                fn = lambda: F.scaled_dot_product_attention(*inp, scale=b.attn.scaling)
                res[backend] = dict(error=errors(fn(), ref), ms=elapsed(fn), kernels=trace(fn)['kernels'])
        except RuntimeError as exc:
            res[backend] = str(exc)
    cu = torch.arange(len(ims)+1, dtype=torch.int32, device='cuda') * lens[0]
    for splits in (1, 2, 4, 8, 16):
        try:
            fn = lambda: shaped(varlen_attn(qr, kr, val, cu, cu, lens[0], lens[0], scale=b.attn.scaling, num_splits=splits))
            res[f'varlen_{splits}'] = dict(error=errors(fn(), ref), ms=elapsed(fn))
        except (RuntimeError, TypeError, ValueError) as exc:
            res[f'varlen_{splits}'] = str(exc)
    bs = len(lens)
    length = lens[0]
    pages = (length+255)//256
    table = torch.arange(bs*pages, dtype=torch.int32, device='cuda').reshape(bs, pages)
    used = torch.full((bs,), length, dtype=torch.int32, device='cuda')
    def paged(t):
        unmerged = t.reshape(bs, length, h, -1)
        return F.pad(unmerged, (0, 0, 0, 0, 0, pages*256-length)).reshape(bs*pages, 256, h, -1)
    kp, vp = paged(kr), paged(val)
    for splits in (1, 2, 3, 4):
        try:
            fn = lambda: shaped(varlen_attn(qr, kp, vp, cu, cu, length, length, scale=b.attn.scaling,
                                             seqused_k=used, block_table=table, num_splits=splits))
            res[f'paged_{splits}'] = dict(error=errors(fn(), ref), ms=elapsed(fn), kernels=trace(fn)['kernels'])
        except (RuntimeError, TypeError, ValueError) as exc:
            res[f'paged_{splits}'] = str(exc)
    # Unequal sequence control using the same first-layer activations.
    lengths = [length if i%2 == 0 else length//2 for i in range(bs)]
    seq = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device='cuda')
    parts = [[t[i, :, :lengths[i]].transpose(0, 1) for i in range(bs)] for t in inp]
    packed = tuple(torch.cat(p) for p in parts)
    def per_image():
        return torch.cat([F.scaled_dot_product_attention(*(t.transpose(0,1)[None] for t in unit),
                         scale=b.attn.scaling)[0].transpose(0,1) for unit in zip(*parts)])
    def varlen():
        return varlen_attn(*packed, seq, seq, length, length, scale=b.attn.scaling)
    res['mixed'] = dict(lengths=lengths, error=errors(varlen(), per_image()), reference_ms=elapsed(per_image),
                        varlen_ms=elapsed(varlen))
    dense = tuple(t.transpose(1, 2) for t in inp)
    for splits in (1, 2, 3, 4, 8):
        try:
            fn = lambda: torch.ops.aten._flash_attention_forward(*dense, None, None, lens[0], lens[0],
                0., False, False, scale=b.attn.scaling, num_splits=splits)[0].transpose(1, 2)
            res[f'dense_{splits}'] = dict(error=errors(fn(), ref), ms=elapsed(fn), kernels=trace(fn)['kernels'])
        except (RuntimeError, TypeError, ValueError) as exc:
            res[f'dense_{splits}'] = str(exc)
    return res


def patch_embed(eng, frames):
    batch = eng.build(frames, [[dict(type='box', text='person')]] * len(frames))[0]
    x = batch['pixel_values'].cuda().to(eng.fv.W.dtype)
    v = eng.fv.v.patch_embed
    w, b = eng.fv.W, eng.fv.b
    view = x.view(-1, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size)
    conv = v.proj
    ref = v(x)
    def channels_last():
        return conv(view.contiguous(memory_format=torch.channels_last_3d)).reshape_as(ref)
    def conv1x1():
        # One patch is one 1x1 Conv2d row. Preserve Conv's separate BF16 bias add.
        y = F.conv2d(x.T[None, :, :, None], w[:, :, None, None], b)
        return y[0, :, :, 0].T
    rows = {}
    for name, op in {'hf': lambda: v(x), 'linear': lambda: F.linear(x,w,b),
                     'linear_bias_separate': lambda: F.linear(x,w) + b,
                     'conv1x1': conv1x1, 'channels_last': channels_last}.items():
        rows[name] = dict(ms=elapsed(op), error=errors(op(), ref))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch-only', action='store_true')
    ap.add_argument('--attention-only', action='store_true')
    a = ap.parse_args()
    eng = Bench(a).to_engine()
    res = {}
    for tv in (128, 256):
        frames = [r['img'] for r in load_images('data/prep/eval_refcoco_val.jsonl','/content/data/coco/train2014',32 if a.patch_only else 8,tv,frame=True)]
        res[str(tv)] = patch_embed(eng, frames) if a.patch_only else attention(eng, frames)
        Path(a.out).write_text(json.dumps(res, indent=1))
    if not (a.patch_only or a.attention_only):
        res['gemm'] = projections(eng)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
