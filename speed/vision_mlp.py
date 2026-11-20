"""Fuse the first vision MLP projection and its BF16 activation on SM120."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit(do_not_specialize=["M"])
def _fc1(
    X,
    W,
    Bias,
    Table,
    Out,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), N // BN
    first = pid // (8 * nn) * 8
    size = tl.minimum(nm - first, 8)
    row = (first + pid % (8 * nn) % size) * BM
    col = (pid % (8 * nn) // size) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(K // BK):
        acc = tl.dot(X.load([row, i * BK]), tl.trans(W.load([col, i * BK])), acc)
    biased = acc + tl.load(Bias + col + tl.arange(0, BN))[None, :].to(tl.float32)
    bits = biased.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)
    Out.store([row, col], tl.load(Table + bits))


def activation_table(mlp):
    # The ordinary BF16 linear rounds before GELU. Evaluate every possible input
    # with the model's actual activation, retaining that rounding in the epilogue.
    bits = torch.arange(65536, dtype=torch.int32, device=mlp.linear_fc1.weight.device)
    return mlp.act_fn(bits.to(torch.int16).view(torch.bfloat16))


def fc1(x, linear, table, tile=(128, 128, 32, 4)):
    bm, bn, bk, warps = tile
    m, k = x.shape
    n = len(linear.weight)
    result = x.new_empty((m, n))
    xd = TensorDescriptor.from_tensor(x, [bm, bk])
    wd = TensorDescriptor.from_tensor(linear.weight, [bn, bk])
    yd = TensorDescriptor.from_tensor(result, [bm, bn])
    _fc1[(triton.cdiv(m, bm) * (n // bn),)](
        xd,
        wd,
        linear.bias,
        table,
        yd,
        m,
        k,
        n,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=3,
    )
    return result
