"""One fixed reduction tree per patch, independent of batch, question count and spatial padding."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(
    Q,
    K,
    Valid,
    Images,
    Slots,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    MAPPED: tl.constexpr,
    COMPACT: tl.constexpr,
    SCALE: tl.constexpr,
    PATCHES: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    patch = tl.program_id(1) * PATCHES + tl.arange(0, PATCHES)
    channel = tl.arange(0, WIDTH)
    image = tl.load(Images + row) if MAPPED else row
    image = image.to(tl.int64)
    position = image * SV0 + patch * SV1
    valid = tl.load(Valid + position, mask=patch < N, other=False)
    if COMPACT:
        slot = tl.load(Slots + image * N + patch, mask=valid, other=0)
        base = slot.to(tl.int64) * SK0
    else:
        base = image * SK0 + patch * SK1
    key = tl.load(
        K + base[:, None] + channel[None, :] * SK2, mask=valid[:, None] & (channel[None, :] < D), other=0
    ).to(tl.float64)
    query = tl.load(Q + row * SQ0 + channel * SQ1, mask=channel < D, other=0).to(tl.float64)
    score = tl.sum(key * query[None, :], axis=1) * SCALE
    tl.store(Out + row * N + patch, tl.where(valid, score, -1.0e4), mask=patch < N)


def cuda_scores(query, keys, valid, images, slots):
    rows, width = query.shape
    patches = valid.shape[1]
    out = query.new_empty((rows, patches), dtype=torch.float64)
    if slots is None:
        batch_stride, patch_stride, channel_stride = keys.stride()
    else:
        batch_stride, channel_stride = keys.stride()
        patch_stride = 0
        slots = slots.contiguous()
    if images is not None:
        images = images.contiguous()
    tile = 8
    with torch.cuda.device(query.device):
        _scores[(rows, triton.cdiv(patches, tile))](
            query,
            keys,
            valid,
            images,
            slots,
            out,
            patches,
            width,
            *query.stride(),
            batch_stride,
            patch_stride,
            channel_stride,
            *valid.stride(),
            images is not None,
            slots is not None,
            1 / math.sqrt(width),
            tile,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return out


@triton.jit
def _project(
    X, W, Bias, Out, M, D: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    dim = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(D, BK)):
        k = base * BK + dim
        x = tl.load(X + rows[:, None] * D + k[None, :], mask=(rows[:, None] < M) & (k[None, :] < D), other=0)
        w = tl.load(W + cols[None, :] * D + k[:, None], mask=(cols[None, :] < H) & (k[:, None] < D), other=0)
        acc = tl.dot(x, w, acc, input_precision="ieee")
    bias = tl.load(Bias + cols, mask=cols < H, other=0)
    tl.store(
        Out + rows[:, None] * H + cols[None, :],
        acc + bias[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < H),
    )


def fixed_project(layer, inputs):
    """FP32 center projection with the same reduction order for singleton, batched and padded rows."""
    h, d = layer.weight.shape
    x = inputs.reshape(-1, d).contiguous()
    y = x.new_empty((len(x), h))
    with torch.cuda.device(x.device):
        _project[(triton.cdiv(len(x), 16), triton.cdiv(h, 32))](
            x, layer.weight.contiguous(), layer.bias.contiguous(), y, len(x), d, h, 16, 32, 16
        )
    return y.view(*inputs.shape[:-1], h)
