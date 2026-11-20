"""Shape-specific dispatch and native-target GEMM / SwiGLU epilogue controls."""

import torch
import triton
import triton.language as tl


from speed.lm_prefill import _mm, _tma_gate


def sweep(m, weight, elapsed, trace):
    import torch.nn.functional as F
    from speed.engine import fused

    n, k = weight.shape
    x = torch.randn((m, k), dtype=weight.dtype, device=weight.device)
    y = x.new_empty((m, n))
    ref = F.linear(x, weight)
    row = dict(
        shape=[m, k, n],
        strides=[x.stride(), weight.stride()],
        pointer_mod_256=[t.data_ptr() % 256 for t in (x, weight, y)],
        candidates={},
    )
    old = torch.backends.cuda.preferred_blas_library()
    for lib in ("cublas", "cublaslt"):
        torch.backends.cuda.preferred_blas_library(lib)
        fn = lambda: F.linear(x, weight)
        row[lib] = dict(ms=elapsed(fn), kernels=[r["kernel"] for r in trace(fn)])
    torch.backends.cuda.preferred_blas_library(old)
    for bm, bn, bk, warps in ((64, 128, 64, 4), (128, 128, 32, 4), (128, 256, 64, 8)):
        for gated in (False, True) if n == 18432 else (False,):
            if gated:
                w = weight
                out = x.new_empty((m, n // 2))
                expected = fused("silu_mul")(ref)
                row["cublas_with_gate_ms"] = elapsed(lambda: fused("silu_mul")(F.linear(x, weight)))
            else:
                w, out, expected = weight, y, ref

            def run():
                return _mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                    x, w, out, m, k, n, bm, bn, bk, gated, num_warps=warps, num_stages=3
                )

            compiled = run()
            row["candidates"][str((bm, bn, bk, warps, gated))] = dict(
                ms=elapsed(run),
                max_abs=(expected - out).abs().max().item(),
                target=next(s.strip() for s in compiled.asm["ptx"].splitlines() if s.strip().startswith(".target")),
            )
    return row


def tma_sweep(m, weight, elapsed):
    from triton.tools.tensor_descriptor import TensorDescriptor
    from speed.lm_prefill import gate_up_direct
    from speed.engine import fused

    x = torch.randn((m, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    n = weight.shape[0] // 2
    out = x.new_empty((m, n))
    expected = fused("silu_mul")(torch.nn.functional.linear(x, weight))
    rows = dict(deployed_ms=elapsed(lambda: gate_up_direct(x, weight)), candidates={})
    for bm, bn, bk, warps, stages in (
        (128, 128, 64, 8, 2),
        (128, 128, 32, 8, 3),
        (128, 64, 64, 4, 2),
        (128, 64, 32, 4, 3),
        (64, 128, 64, 4, 2),
        (128, 128, 64, 8, 3),
    ):
        key = str((bm, bn, bk, warps, stages))
        xd = TensorDescriptor.from_tensor(x, [bm, bk])
        wd = TensorDescriptor.from_tensor(weight, [bn, bk])
        yd = TensorDescriptor.from_tensor(out, [bm, bn])

        def run():
            return _tma_gate[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                xd, wd, yd, m, x.shape[1], n, bm, bn, bk, num_warps=warps, num_stages=stages
            )

        try:
            run()
            rows["candidates"][key] = dict(ms=elapsed(run), max_abs=(out - expected).abs().max().item())
        except triton.runtime.errors.OutOfResources as exc:
            rows["candidates"][key] = dict(error=str(exc))
    return rows
