"""Qwen vision RoPE on Q and K from one strided QKV projection."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary(Q, K, Cos, Sin, OutQ, OutK, N, H: tl.constexpr, D: tl.constexpr,
            TOKEN_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    # Each lane owns a pair across the two head halves, so each source value is read once.
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = pair // (H * (D // 2))
    head = pair // (D // 2) % H
    col = pair % (D // 2)
    mask = token < N
    offs = token * TOKEN_STRIDE + head * D + col
    table = token * D + col
    c0 = tl.load(Cos + table, mask, 0).to(tl.float32)
    c1 = tl.load(Cos + table + D // 2, mask, 0).to(tl.float32)
    s0 = tl.load(Sin + table, mask, 0).to(tl.float32)
    s1 = tl.load(Sin + table + D // 2, mask, 0).to(tl.float32)
    q0 = tl.load(Q + offs, mask, 0).to(tl.float32)
    q1 = tl.load(Q + offs + D // 2, mask, 0).to(tl.float32)
    k0 = tl.load(K + offs, mask, 0).to(tl.float32)
    k1 = tl.load(K + offs + D // 2, mask, 0).to(tl.float32)
    out = (token * H + head) * D + col
    tl.store(OutQ + out, q0 * c0 - q1 * s0, mask)
    tl.store(OutQ + out + D // 2, q1 * c1 + q0 * s1, mask)
    tl.store(OutK + out, k0 * c0 - k1 * s0, mask)
    tl.store(OutK + out + D // 2, k1 * c1 + k0 * s1, mask)


def apply(q, k, cos, sin):
    n, h, d = q.shape
    oq, ok = (torch.empty((n, h, d), dtype=q.dtype, device=q.device) for _ in range(2))
    block = 512
    _rotary[(triton.cdiv(n * h * (d // 2), block),)](
        q, k, cos.contiguous(), sin.contiguous(), oq, ok, n, h, d, q.stride(0), block,
        enable_fp_fusion=False)
    return oq, ok
