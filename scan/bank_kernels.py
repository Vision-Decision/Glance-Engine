"""GPU decode of an exact exponent window and gather into the clip BMM layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def state_values(Low, Mid, Nibble, Tag, Escape, idx):
    low = tl.load(Low + idx).to(tl.uint16, bitcast=True).to(tl.uint32)
    mid = tl.load(Mid + idx).to(tl.uint32)
    t = tl.load(Tag + idx // 128)
    code = (tl.load(Nibble + idx // 2).to(tl.int32) >> (4 * (idx % 2))) & 15
    esc = tl.load(Escape + (-1 - t) * 128 + idx % 128, t < 0, 0).to(tl.int32)
    ex = tl.where(t >= 0, t + code, esc).to(tl.uint32)
    return (((mid & 128) << 24) | (ex << 23) | ((mid & 127) << 16) | low).to(tl.float32, bitcast=True)


@triton.jit
def _unpack(Low, Mid, Nibble, Tag, Escape, Out, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Out + i, state_values(Low, Mid, Nibble, Tag, Escape, i))


def unpack(low, mid, nibble, tag, escape, out=None):
    if out is None:
        out = torch.empty_like(low, dtype=torch.float32)
    block = 1024 if low.numel() % 1024 == 0 else 128
    _unpack[(out.numel() // block,)](low, mid, nibble, tag, escape, out, block)
    return out


@triton.jit
def _gather(
    K,
    V,
    Ids,
    Ok,
    Ov,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(1)
    h = tl.program_id(2)
    j = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, D)
    slot = tl.load(Ids + row * L + j, j < L, 0)
    src = (slot[:, None] * H + h) * D + d[None, :]
    dst = ((row * H + h) * L + j[:, None]) * D + d[None, :]
    tl.store(Ok + dst, tl.load(K + src), (j < L)[:, None])
    tl.store(Ov + dst, tl.load(V + src), (j < L)[:, None])


def gather(k, v, ids):
    n, l = ids.shape
    _, h, d = k.shape
    out = k.new_empty((n, h, l, d))
    val = torch.empty_like(out)
    _gather[(triton.cdiv(l, 8), n, h)](k, v, ids, out, val, l, h, d, 8)
    return out, val
