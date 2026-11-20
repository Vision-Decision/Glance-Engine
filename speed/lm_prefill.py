"""BF16 LM prefill kernels. Load this module only on the optimized CUDA path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_norm(
    X, Z, W, Y, N, H: tl.constexpr, D: tl.constexpr, ZS: tl.constexpr, EPS: tl.constexpr, ROWS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, D)
    off = row[:, None] * D + col[None, :]
    x = tl.load(X + off, row[:, None] < N, 0).to(tl.float32)
    zoff = ((row // H) * ZS + (row % H) * D)[:, None] + col[None, :]
    z = tl.load(Z + zoff, row[:, None] < N, 0).to(tl.float32)
    w = tl.load(W + col)
    r = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    # HF rounds both the normalized vector and its learned scale to the input dtype.
    xn = (x * r[:, None]).to(tl.bfloat16).to(tl.float32)
    scaled = (xn * w[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + off, scaled * (z * tl.sigmoid(z)), row[:, None] < N)


def gated_norm(m, x, z):
    d = m.head_v_dim
    h = m.value_dim // d
    z = z.reshape(-1, m.value_dim)
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel() // d
    _gated_norm[(triton.cdiv(n, 16),)](x, z, m.norm.weight, out, n, h, d, z.stride(0), m.norm.variance_epsilon, 16)
    return out


@triton.jit
def _conv(
    X,
    W,
    Bias,
    Start,
    Q,
    K,
    V,
    N,
    SX: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    TAPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    start = tl.load(Start + t, t < N, 0)
    acc = tl.full((BT, BC), 0, tl.float32)
    for i in tl.static_range(TAPS):
        ti = t + i + 1 - TAPS
        x = tl.load(X + ti[:, None] * SX + c[None, :], ((ti >= start) & (t < N))[:, None], 0).to(tl.float32)
        w = tl.load(W + c * TAPS + i).to(tl.float32)
        acc = tl.fma(x, w[None, :], acc)
    if HAS_BIAS:
        acc += tl.load(Bias + c)[None, :]
    x = acc.to(tl.bfloat16).to(tl.float32)
    y = (x * tl.sigmoid(x)).to(tl.bfloat16).to(tl.float32)
    if tl.program_id(1) * BC < 2 * KD:
        # Match FLA's BF16 input and per-head FP32 l2norm_fwd, including its epsilon.
        y = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=1) + 1.0e-6))[:, None]
    if tl.program_id(1) * BC < KD:
        tl.store(Q + t[:, None] * KD + c[None, :], y, t[:, None] < N)
    elif tl.program_id(1) * BC < 2 * KD:
        tl.store(K + t[:, None] * KD + c[None, :] - KD, y, t[:, None] < N)
    else:
        tl.store(V + t[:, None] * VD + c[None, :] - 2 * KD, y, t[:, None] < N)


def conv_qkv(m, mixed, starts):
    n = len(mixed)
    kd, vd = m.key_dim, m.value_dim
    q, k = (mixed.new_empty((n, kd)) for _ in range(2))
    v = mixed.new_empty((n, vd))
    w, b = m.conv1d.weight, m.conv1d.bias
    _conv[(triton.cdiv(n, 16), (2 * kd + vd) // 128)](
        mixed, w, b, starts, q, k, v, n, mixed.stride(0), kd, vd, w.shape[-1], b is not None, 16, 128
    )
    return (q.view(1, n, -1, m.head_k_dim), k.view(1, n, -1, m.head_k_dim), v.view(1, n, -1, m.head_v_dim))


@triton.jit
def _head_norm(
    X,
    W,
    Cos,
    Sin,
    Y,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    XS: tl.constexpr,
    HS: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    t = row // H
    c = tl.arange(0, D)
    x = tl.load(X + (t * XS + row % H * HS)[:, None] + c[None, :], t[:, None] < N, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    w = tl.load(W + c).to(tl.float32)
    v = (x * r[:, None] * (1.0 + w[None, :])).to(tl.bfloat16)
    if ROT:
        peer = tl.gather(v, tl.broadcast_to(((c + ROT // 2) % ROT)[None, :], (R, D)), 1)
        table = t[:, None] * ROT + c[None, :]
        co = tl.load(Cos + table, (t[:, None] < N) & (c[None, :] < ROT), 0)
        si = tl.load(Sin + table, (t[:, None] < N) & (c[None, :] < ROT), 0)
        other = tl.where(c[None, :] < ROT // 2, -peer, peer)
        # The HF elementwise products round separately before addition.
        a = (v.to(tl.float32) * co.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        b = (other.to(tl.float32) * si.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        v = tl.where(c[None, :] < ROT, (a + b).to(tl.bfloat16), v)
    tl.store(Y + (row[:, None] * D + c[None, :]), v, t[:, None] < N)


def head_norm(x, norm, rope=None):
    n, h, d = x.shape
    out = torch.empty((n, h, d), dtype=x.dtype, device=x.device)
    cos, sin = (None, None) if rope is None else rope
    rot = cos.shape[-1] if cos is not None else 0
    _head_norm[(triton.cdiv(n * h, 8),)](
        x, norm.weight, cos, sin, out, n, h, d, x.stride(0), x.stride(1), rot, norm.eps, 8, enable_fp_fusion=False
    )
    return out


@triton.jit(do_not_specialize=["M"])
def _mm(
    X,
    W,
    Y,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GATED: tl.constexpr,
):
    # Reuse a weight tile across up to eight M tiles before moving along N.
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    first = pid // (8 * nn) * 8
    size = tl.minimum(nm - first, 8)
    im = (first + (pid % (8 * nn)) % size) * BM + tl.arange(0, BM)
    jn = ((pid % (8 * nn)) // size) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(K // BK):
        x = tl.load(X + im[:, None] * K + i * BK + kk[None, :], im[:, None] < M, 0)
        if GATED:
            local = tl.arange(0, BN)
            weight_row = (jn // BN) * (BN // 2) + local % (BN // 2) + (local // (BN // 2)) * (N // 2)
        else:
            weight_row = jn
        w = tl.load(W + weight_row[:, None] * K + i * BK + kk[None, :], jn[:, None] < N, 0)
        acc = tl.dot(x, tl.trans(w), acc)
    if GATED:
        pair = acc.to(tl.bfloat16).to(tl.float32).reshape(BM, 2, BN // 2).permute(0, 2, 1)
        gate, up = tl.split(pair)
        val = gate * tl.sigmoid(gate) * up
        col = ((pid % (8 * nn)) // size) * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(Y + im[:, None] * (N // 2) + col[None, :], val, im[:, None] < M)
    else:
        tl.store(Y + im[:, None] * N + jn[None, :], acc, (im[:, None] < M) & (jn[None, :] < N))


def gate_up_direct(h, weight):
    m, k = h.shape
    n = weight.shape[0]
    bm, bn, bk, warps = (128, 128, 32, 4) if m < 8192 else (128, 256, 64, 8)
    out = h.new_empty((m, n // 2))
    _mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        h, weight, out, m, k, n, bm, bn, bk, True, num_warps=warps, num_stages=3
    )
    return out


@triton.jit(do_not_specialize=["M"])
def _tma_gate(X, W, Y, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Load two adjacent-in-N projection tiles from the original gate-then-up weight.
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    first = pid // (8 * nn) * 8
    size = tl.minimum(nm - first, 8)
    row = (first + (pid % (8 * nn)) % size) * BM
    col = ((pid % (8 * nn)) // size) * BN
    gate = tl.full((BM, BN), 0, tl.float32)
    up = tl.full((BM, BN), 0, tl.float32)
    for i in range(K // BK):
        x = X.load([row, i * BK])
        wg = W.load([col, i * BK])
        wu = W.load([col + N, i * BK])
        gate = tl.dot(x, tl.trans(wg), gate)
        up = tl.dot(x, tl.trans(wu), up)
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    Y.store([row, col], (g * tl.sigmoid(g) * u).to(tl.bfloat16))


def gate_up(h, weight):
    # The workload is gated on SM120 and on aligned, contiguous BF16 projection shapes by the caller.
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        return gate_up_direct(h, weight)
    m, k = h.shape
    n = weight.shape[0] // 2
    bm, bn, bk = 128, 64, 32
    out = h.new_empty((m, n))
    xd = TensorDescriptor.from_tensor(h, [bm, bk])
    wd = TensorDescriptor.from_tensor(weight, [bn, bk])
    yd = TensorDescriptor.from_tensor(out, [bm, bn])
    _tma_gate[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        xd, wd, yd, m, k, n, bm, bn, bk, num_warps=4, num_stages=3
    )
    return out
