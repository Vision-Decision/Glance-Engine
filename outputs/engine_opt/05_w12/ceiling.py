"""Recompute item 5's per-call G4 floors from the archived checkpoints, shapes and timings.

From the repository root: PYTHONPATH=. python outputs/engine_opt/05_w12/ceiling.py
"""

import json
import math
from pathlib import Path
from statistics import median

from speed.ceiling_model import call_floor

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "05_w12/raw"
SPEED_FIELDS = {
    "total": "total_ms_median",
    "prep": "prep_ms_median",
    "vision": "vision_ms_median",
    "LM": "lm_ms_median",
    "model": "model_ms_median",
}
SCAN_FIELDS = {
    "total": "total_ms",
    "prep": "prep_ms",
    "vision": "vision_ms",
    "LM": "lm_ms",
    "model": "model_ms",
}


def read(path):
    return json.loads(path.read_text())


def item(name):
    return read(RAW / f"item5_{name}.json")


def serving(paths):
    return {
        key: row.get("total_ms_median", row.get("total_ms"))
        for path in paths
        for key, row in read(path)["modes"].items()
    }


def compact_rows(n):
    if n <= 64:
        return max(16, 2 ** (n - 1).bit_length())
    if n <= 96:
        return 96
    return math.ceil(n / 64) * 64


def floors(key, calls, rates, matrices):
    # The historical dispatch is fixed for this report. The cutoff depends on (output, input).
    limits = {
        (18432, 2560): 128,
        (12352, 2560): 96,
        (10240, 2560): 96,
        (2560, 9216): 40,
        (2560, 4096): 24,
    }
    rows_tested = {1, 8, 16, 24, 32, 40, 48, 64, 96, 128}
    dense_weights = sum(r["weights"] for r in matrices)
    per_call = []
    for units in calls:
        prefix = sum(n * p for n, _, p, _ in units)
        question = sum(n * sum(q) for n, _, _, q in units)
        if key.startswith(("followup", "Tv")):
            tokens = question
            physical = compact_rows(question)
        else:
            tokens = prefix + question
            # Videos use packed independent sequences. Single-image speed calls pad both axes.
            physical = (
                64 * (math.ceil(prefix / 64) + math.ceil(question / 64)) if key.startswith("bs1") else tokens
            )

        compressed = [
            r for r in matrices if physical in rows_tested and physical <= limits[tuple(r["shape"])]
        ]
        weight_bytes = dense_weights * 2
        for matrix in compressed:
            n, k = matrix["shape"]
            padded = math.ceil(n / 128) * 128
            # GPU stream reads planes, scales and CSR offsets/columns/corrections.
            packed = padded * k * 3 // 2 + 4 * padded + 4 * (padded + 1) + 8 * matrix["escapes"]
            weight_bytes += packed - matrix["weights"] * 2

        baseline, _ = call_floor(key, units, rates)
        if key.startswith("fresh"):
            baseline["H2D"] /= 8  # Item 2 transfers one uint8 RGB frame.
        w12 = baseline.copy()
        w12["LM dense"] = max(
            2 * dense_weights * tokens / (rates["bf16"] * 1e9),
            (weight_bytes + 4 * 32 * 2560 * tokens) / (rates["copy"] * 1e6),
        )
        per_call.append((physical, tokens, len(compressed), baseline, w12))
    physical, useful, counts, base_stages, stages = zip(*per_call)
    stage_ms = {s: median(row[s] for row in stages) for s in stages[0]}
    return {
        "w12_rows_median": median(physical),
        "useful_rows_median": median(useful),
        "w12_matrix_count_median": median(counts),
        "bf16_floor_ms": sum(median(row[s] for row in base_stages) for s in base_stages[0]),
        "w12_stages_floor_ms": stage_ms,
        "w12_total_floor_ms": median(sum(row.values()) for row in stages),
        "w12_lm_floor_ms": sum(stage_ms[s] for s in ("LM dense", "GDN", "LM attention", "heads", "launch")),
        "w12_vision_floor_ms": sum(stage_ms.get(s, 0) for s in ("H2D", "ViT dense", "ViT attention")),
    }


def scan_calls(shapes, key, kind, count):
    if kind == "fresh":
        tv, size = key.split("_")
        # The scan harness resizes every frame to the same 4:3 grid.
        return [[[int(size[1:]), *call[0][1:]]] for call in shapes[f"fresh_{tv}_N32"][:count]]
    if key in shapes:
        return shapes[key][:count]
    _, size, readout = key.split("_")
    # A smaller bank contains the first N images. The same query is broadcast to all of them.
    calls = []
    for query in shapes[f"Tv640_N64_{readout}"][:count]:
        units = [
            [1, v, p, query[0][3]] for image in shapes["bs1_box_N1"][: int(size[1:])] for _, v, p, _ in image
        ]
        calls.append(units)
    return calls


def main():
    peaks = read(ROOT / "00_ceiling/raw/c0_peaks.json")
    rates = {
        dtype: max(r["tflops"] for r in peaks["gemm"] if r["dtype"] == torch_dtype)
        for dtype, torch_dtype in (
            ("bf16", "torch.bfloat16"),
            ("fp32", "torch.float32"),
        )
    }
    rates.update(
        copy=peaks["copy"]["GBs"],
        h2d=peaks["pinned"]["GBs"],
        launch_ms=peaks["replay"]["ms"],
    )
    matrices = list(item("encode")["matrices"].values())
    shapes = read(ROOT / "00_ceiling/raw/c0_shapes.json")
    vllm = serving(
        [
            ROOT.parent / "sprint_speed/raw/serve_vllm_async_g4.json",
            ROOT.parent / "scan/raw/scan-a/vllm_bf16.json",
            ROOT / "02_vision_prep/raw/item2_cold_vllm.json",
        ]
    )
    sglang = serving(
        [
            ROOT / "00_ceiling/raw/c0_sglang_clean.json",
            ROOT / "00_ceiling/raw/c0_sglang_bank.json",
            ROOT / "02_vision_prep/raw/item2_serving_sglang.json",
            RAW / "item5_sglang.json",
        ]
    )
    result = {}

    def add(key, calls, endpoint, runs, fields):
        row = floors(key, calls, rates, matrices)
        for mode, run in zip(("bf16", "w12", "fp8"), runs):
            row[mode] = {label: run[field] for label, field in fields.items() if field in run}
        row.update(vllm=vllm[endpoint], sglang=sglang[endpoint])
        times = row["w12"]
        row["w12_total_efficiency_pct"] = 100 * row["w12_total_floor_ms"] / times["total"]
        row["w12_lm_efficiency_pct"] = (
            100 * row["w12_lm_floor_ms"] / (times["LM"] if "LM" in times else times["model"])
        )
        row["w12_vision_efficiency_pct"] = (
            100 * row["w12_vision_floor_ms"] / times["vision"] if "vision" in times else None
        )
        result[key] = row

    speed = [
        item(name)["modes"][mode]
        for name, mode in (
            ("bf16", "eng_fused"),
            ("verified_w12", "eng_w12"),
            ("fp8", "eng_fp8"),
        )
    ]
    for key in speed[0]:
        if key in shapes:
            runs = [data[key] for data in speed]
            add(key, shapes[key][: runs[1]["n_calls"]], key, runs, SPEED_FIELDS)
    scan = [item(name) for name in ("scan_bf16", "verified_scan_w12", "scan_fp8")]
    for kind in ("cached", "fresh"):
        for key, value in scan[0][kind].items():
            if isinstance(value, dict):
                runs = [data[kind][key] for data in scan]
                calls = scan_calls(shapes, key, kind, scan[0]["args"]["n_calls"])
                label = "fresh_" + key if kind == "fresh" else key
                endpoint = label + "_yesno" if kind == "fresh" else "cached_" + key.replace("_noul", "_yesno")
                add(label, calls, endpoint, runs, SCAN_FIELDS)
    (RAW / "item5_workload_floors.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
