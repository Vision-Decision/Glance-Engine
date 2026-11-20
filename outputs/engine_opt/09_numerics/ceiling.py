"""Apply the measured FP64 rate to the box score arithmetic in the C0 useful-work model.

From the worktree root run PYTHONPATH=. python outputs/engine_opt/09_numerics/ceiling.py.
"""

import json
from pathlib import Path
from statistics import median

from speed.ceiling_model import call_floor


def main():
    raw = Path(__file__).parent / "raw"
    previous = raw.parents[1]
    c0 = previous / "00_ceiling" / "raw"
    shapes = json.loads((c0 / "c0_shapes.json").read_text())
    rates = json.loads((previous / "08_bank_memory/raw/item8_floors.json").read_text())["rates"]
    rates["fp64"] = json.loads((raw / "item9_tiled_micro.json").read_text())["fp64_gemm"]["tflops"]
    keys = ["Tv640_N256_box", "Tv640_N256_noul", "fresh_Tv128_N128"]
    keys += ["bs1_box_N1", "bs1_box_N8", "bs1_box_N32", "bs1_mixed_N8", "bs1_mixed_N32"]
    keys += [f"followup_{kind}_N{n}" for kind in ("box", "mixed") for n in (1, 8, 32)]
    result = {}
    for key in keys:
        bank = key.startswith("Tv")
        calls = shapes[key][:15] if bank else shapes[key]
        floors = []
        for units in calls:
            stages, _ = call_floor(key, units, rates)
            if key.startswith("fresh"):
                stages["H2D"] /= 8
            visual = sum(n * v for n, v, _, qs in units)
            questions = sum(n * len(qs) for n, v, _, qs in units)
            scores = sum(n * v * len(qs) for n, v, _, qs in units)
            d, h = 2560, 512
            readout = d * h + h + d * h + (2 * d + 64) * 3 * h + 2 * h * 256 + h + d * (30 + 28)
            wp = 0 if bank else d * h
            bytes_ = 4 * (readout + wp + visual * (h if bank else d) + questions * d)
            fp32 = 2 * (readout * questions + wp * visual)
            fp64 = 2 * h * scores
            stages["heads"] = max(
                bytes_ / (rates["copy"] * 1e6), fp32 / (rates["fp32"] * 1e9), fp64 / (rates["fp64"] * 1e9)
            )
            floors.append(stages)
        result[key] = {
            "calls": len(calls),
            "ms": median(sum(row.values()) for row in floors),
            "stage_ms": {k: median(row[k] for row in floors) for k in floors[0]},
        }
    (raw / "item9_floors.json").write_text(json.dumps({"rates": rates, "workloads": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
