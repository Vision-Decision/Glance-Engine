"""Residual and 1,024-wide LayerNorm with ATen's BF16 rounding and Welford order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_norm(X, Update, Weight, Bias, Residual, Norm, EPS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 128)
    mu = tl.full((128,), 0, tl.float32)
    m2 = tl.full((128,), 0, tl.float32)
    # ATen assigns two 4-value vectors to each of 128 logical lanes.
    for i in tl.static_range(8):
        col = (i // 4) * 512 + lane * 4 + i % 4
        offset = row * 1024 + col
        val = (
            tl.load(X + offset).to(tl.float32) + tl.load(Update + offset).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(Residual + offset, val)
        val = val.to(tl.float32)
        delta = val - mu
        mu = tl.fma(delta, 1.0 / (i + 1), mu)
        m2 = tl.fma(delta, val - mu, m2)
    # Each warp reduces its own lane groups, then the four warp leaders merge.
    for i in tl.static_range(7):
        peer = (16 >> i) if i < 5 else (64 >> (i - 5))
        other_mu = tl.gather(mu, lane ^ peer, 0)
        other_m2 = tl.gather(m2, lane ^ peer, 0)
        delta = mu - other_mu
        m2 = tl.fma((delta * delta) * (8 << i), 0.5, other_m2 + m2)
        mu = tl.fma(mu, 0.5, other_mu * 0.5)
    mean = tl.sum(tl.where(lane == 0, mu, 0.0), 0)
    variance = tl.sum(tl.where(lane == 0, m2, 0.0), 0) / 1024.0
    scale = tl.rsqrt(variance + EPS)
    for i in tl.static_range(8):
        col = (i // 4) * 512 + lane * 4 + i % 4
        offset = row * 1024 + col
        val = (
            (
                tl.load(X + offset).to(tl.float32)
                + tl.load(Update + offset).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        w, b = tl.load(Weight + col).to(tl.float32), tl.load(Bias + col).to(tl.float32)
        tl.store(Norm + offset, tl.fma(w, scale * (val - mean), b))


def add_norm(x, update, norm):
    residual, y = torch.empty_like(x), torch.empty_like(x)
    _add_norm[(len(x),)](
        x, update, norm.weight, norm.bias, residual, y, norm.eps, num_warps=4
    )
    return residual, y
