"""Attach useful-work rates to a packed LM trace. Bytes are logical payload, not DRAM transactions."""

import argparse
import csv
import json
from math import prod
from pathlib import Path


SIZE = {
    "c10::BFloat16": 2,
    "c10::Half": 2,
    "float": 4,
    "double": 8,
    "long int": 8,
    "long unsigned int": 8,
    "int": 4,
    "bool": 1,
}
BANDWIDTH = 1463.21
BF16 = 401.73
FP32 = 67.4


def work(row, n, t, prefix, before):
    op, kernel = row["op"], row["kernel"]
    dims, types = row["dims"], row["types"]
    length = t // n
    if row.get("transfer_bytes"):
        return "bytes", row["transfer_bytes"] / row["calls"], "transfer payload"
    if row.get("tflops") is not None:
        return (
            "bf16" if types[0] == "c10::BFloat16" else "fp32",
            row["tflops"] * row["ms"] * 1e9 / row["calls"],
            "2 M N K, incl. batch",
        )
    if kernel in ("_mm", "_tma_gate"):
        return "bf16", 2 * t * 2560 * 18432, "gate and up GEMM, excludes epilogue"
    if kernel == "_conv" or op == "CausalConv1dFunction":
        return "bytes", 2 * (2 * t * 8192 + 4 * 8192), "one input, tap weights and one output, ideal reuse"
    if kernel == "_gated_norm":
        return "bytes", 2 * (3 * t * 4096 + 128), "output, gate, scale and normalized result"
    if kernel == "_head_norm":
        return (
            "bytes",
            2 * (2 * t * (4096 + 1024) / 2 + 2 * t * 64 + 256),
            "mean Q/K head width across equal call counts, rotary tables once",
        )
    if kernel == "tilelang_fused_chunk_gdr_fwd_kernel_kernel":
        return "bf16", 6 * t * 32 * 128**2, "useful state update and readout; excludes within-chunk corrections"
    if kernel == "tilelang_kkt_solve_kernel_kernel":
        pairs32 = n * sum(min(32, length - i) * (min(32, length - i) + 1) // 2 for i in range(0, length, 32))
        return "bf16", 2 * 32 * 128 * pairs32, "valid 32-token KKT products; excludes triangular solve"
    if kernel == "tilelang_chunk_local_cumsum_kernel_kernel":
        return "bytes", 2 * 4 * t * 32, "FP32 log-decay input and cumulative output"
    if kernel == "l2norm_fwd_kernel":
        return "bytes", t * 16 * (2 * 128 * 2 + 4), "one Q or K read + write, FP32 inverse norm"
    if kernel == "chunk_local_cumsum_scalar_kernel":
        return "bytes", 2 * 4 * t * 32, "FP32 g read and cumulative g write"
    if kernel == "chunk_gated_delta_rule_fwd_kernel_h_blockdim64":
        return "bf16", 4 * t * 32 * 128**2, "two state contractions per valid token"
    if kernel == "chunk_fwd_kernel_o":
        return "bf16", 2 * t * 32 * 128**2, "state readout contraction, excludes within-chunk correction"
    # Use only valid causal pairs inside each chunk, including its diagonal.
    pairs = n * sum(min(64, length - i) * (min(64, length - i) + 1) // 2 for i in range(0, length, 64))
    if kernel == "chunk_gated_delta_rule_fwd_kkt_solve_kernel":
        return "bf16", 2 * 32 * 128 * pairs, "useful KKT products, excludes triangular solve"
    if kernel == "recompute_w_u_fwd_kernel":
        return "bf16", 2 * 32 * (128 + 128) * pairs, "valid triangular transform applied to key and value"
    if op == "aten::_flash_attention_forward":
        tq, tk = dims[0][0] // n, dims[1][0] // n
        pairs = n * tq * (tq + 1) // 2 if tq == tk else n * tq * tk
        return "bf16", 4 * 4096 * pairs, "valid score and value products"
    if op.startswith("triton_"):
        return (
            "bytes",
            sum(prod(d) * SIZE[dt] for dt, d in zip(types, dims) if dt in SIZE and d),
            "compiled kernel tensor arguments, logical once",
        )
    if op == "aten::cat":
        # All concatenations of LM activations in this trace are BF16. FP32 head
        # concatenations have last-axis inputs of width 512 / 2560 or 64 bin features.
        shapes = dims[0]
        itemsize = 4 if len(shapes[0]) == 2 and shapes[0][0] == n else 2
        return "bytes", 2 * itemsize * sum(prod(s) for s in shapes), "read each concatenand and write result"
    if not dims:
        return None, None, "trace exposes no tensor extent"
    elt = SIZE.get(types[0], 0)
    if kernel.startswith("void cublasLt::") or "scal_kernel" in kernel:
        lhs, rhs = dims[1:3] if op == "aten::addmm" else dims[:2]
        count = prod(lhs[:-1]) * rhs[-1]
        return "bytes", count * elt, "lower bound: one result write, vendor workspace reads unknown"
    if op == "aten::arange":
        return "bytes", 16 * 4, "16 FP32 Fourier frequencies per edge/IoU head call"
    if op == "aten::fill_" and not dims[0] and elt:
        return "bytes", elt, "one scalar write"
    if op == "aten::pow" and not dims[0] and len(dims) > 1 and dims[1]:
        return "bytes", 8 + 2 * prod(dims[1]) * SIZE[types[1]], "scalar base, vector exponent and output"
    if not dims[0] or not elt:
        return None, None, "tensor extent or dtype unavailable"
    first = prod(dims[0])
    if op == "aten::copy_":
        return "bytes", first * (elt + SIZE[types[1]]), "source and destination, logical dst shape"
    if op == "aten::fill_":
        return "bytes", first * elt, "destination writes"
    if op == "aten::index_copy_":
        return "bytes", 2 * prod(dims[3]) * elt + 8 * prod(dims[2]), "source, indexed destinations and row indices"
    if op == "aten::gather" and len(dims) > 2:
        # aten::embedding passes an expanded index view with hidden-axis stride zero.
        num = prod(dims[2])
        indices = dims[2][0] if dims[0][0] == 248320 else num
        return "bytes", 2 * num * elt + 8 * indices, "gather source, result and indices"
    if op == "aten::index":
        if dims[0][0] == t and len(dims[0]) == 3:
            if before and dims[0][1] == 16:
                rows = t / 2  # prefix and question gathers aggregated into the same name
            else:
                rows = n * prefix if before else t - n * prefix
        elif dims[0] == [t, 2560]:
            rows = (n + n * 117) / 2  # final question and visual rows
        elif dims[0] in ([n, 117, 512], [n, 117]):
            return "bytes", 2 * first * elt + n * 8, "one complete key/valid-mask row per question"
        elif dims[0] in ([n, 117, 2560], [n, 117, 2]):
            return (
                "bytes",
                2 * n * dims[0][-1] * elt + n * 16,
                "selected visual state / center, two indices per question",
            )
        elif dims[0] == [n, 30]:
            return "bytes", n * 28 * 2 * 2 * elt + 28 * 2 * 8, "typed groups, two candidate labels per group"
        else:
            return None, None, "advanced-index extent absent"
        return "bytes", rows * (8 + 2 * prod(dims[0][1:]) * elt), "gathered rows and indices"
    if op == "aten::native_layer_norm":
        rows = first / dims[0][-1]
        return (
            "bytes",
            2 * first * elt + 2 * rows * 4 + 2 * dims[0][-1] * elt,
            "layernorm x/y, scale/bias, mean/rstd",
        )
    if op in ("aten::mean", "aten::sum", "aten::amax", "aten::argmax"):
        return (
            "bytes",
            first * elt + prod(dims[0][:-1]) * (8 if op == "aten::argmax" else elt),
            "last-axis reduction in/out",
        )
    if op in (
        "aten::mul",
        "aten::add",
        "aten::add_",
        "aten::sub",
        "aten::div",
        "aten::silu",
        "aten::sigmoid",
        "aten::pow",
        "aten::neg",
        "aten::rsqrt",
        "aten::cos",
        "aten::sin",
        "aten::exp",
        "aten::exp_",
        "aten::gelu",
        "aten::lt",
        "aten::eq",
        "aten::masked_fill_",
        "aten::log_",
        "aten::clamp",
        "aten::_softmax",
        "aten::bitwise_not",
        "aten::abs",
    ):
        inputs = [(prod(d), SIZE[dt]) for d, dt in zip(dims, types) if d and dt in SIZE]
        output_size = (
            1 if op in ("aten::lt", "aten::eq", "aten::bitwise_not") else max(s for count, s in inputs if count)
        )
        return (
            "bytes",
            sum(count * s for count, s in inputs) + max(count for count, _ in inputs) * output_size,
            "logical inputs and output, broadcast inputs once",
        )
    return None, None, "opaque launch"


def dram_rates(raw):
    """One warm launch per kernel/grid/block configuration, from Nsight Compute."""
    with (raw / "item3_dram_rates.tsv").open("w") as dest:
        writer = csv.writer(dest, delimiter="\t")
        writer.writerow(
            [
                "pass",
                "actual_tokens",
                "kernel",
                "grid",
                "block",
                "dram_bytes",
                "duration_ns",
                "GB/s",
                "peak_GB/s",
                "peak_percent",
            ]
        )
        for path in sorted(raw.glob("item3_*ncu_*_*.csv")):
            prefix, name, ns = path.stem.rsplit("_", 2)
            if prefix == "item3_accel_ncu":
                name = "flashqla"
            with path.open() as source:
                rows = csv.DictReader(line for line in source if line.startswith('"'))
                for row in rows:
                    if not row["ID"]:
                        continue
                    size = int(row["dram__bytes.sum"].replace(",", ""))
                    duration = int(row["gpu__time_duration.sum"].replace(",", ""))
                    rate = size / duration
                    writer.writerow(
                        [
                            name,
                            146 * int(ns),
                            row["Kernel Name"],
                            row["Grid Size"],
                            row["Block Size"],
                            size,
                            duration,
                            round(rate, 2),
                            BANDWIDTH,
                            round(100 * rate / BANDWIDTH, 2),
                        ]
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    dram_rates(args.raw)
    with (args.raw / "item3_kernel_rates.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "pass",
                "nominal_visual_tokens",
                "actual_tokens",
                "calls",
                "ms",
                "unit",
                "achieved",
                "peak",
                "percent",
                "work_definition",
                "operator",
                "kernel",
            ]
        )
        traces = {"before": "profile_before", "production": "profile_production"}
        if (args.raw / "item3_accel_profile_production.json").exists():
            traces["flashqla"] = "accel_profile_production"
        for stage, stem in traces.items():
            p = json.loads((args.raw / f"item3_{stem}.json").read_text())
            for ns, row in p["results"].items():
                n, t = int(ns), row["tokens"]
                for r in row["kernels"]:
                    kind, count, definition = work(r, n, t, 122, stage == "before")
                    rate = peak = pct = ""
                    if kind:
                        rate = count * r["calls"] / r["ms"] / (1e6 if kind == "bytes" else 1e9)
                        peak = {"bytes": BANDWIDTH, "bf16": BF16, "fp32": FP32}[kind]
                        pct = round(rate / peak * 100, 2)
                        rate = round(rate, 3)
                    writer.writerow(
                        [
                            stage,
                            128 * n,
                            t,
                            r["calls"],
                            round(r["ms"], 6),
                            {"bytes": "logical GB/s", "bf16": "BF16 TFLOP/s", "fp32": "FP32 TFLOP/s", None: ""}[
                                kind
                            ],
                            rate,
                            peak,
                            pct,
                            definition,
                            r["op"],
                            r["kernel"],
                        ]
                    )


if __name__ == "__main__":
    main()
