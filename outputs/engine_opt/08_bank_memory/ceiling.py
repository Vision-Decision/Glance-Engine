"""Recompute bank floors from the archived G4 peaks, call shapes and byte inventory.

From the repository root run PYTHONPATH=. python outputs/engine_opt/08_bank_memory/ceiling.py
"""

import json
from pathlib import Path
from statistics import median

from speed.ceiling_model import call_floor


def main():
    raw = Path(__file__).parent / "raw"
    c0 = raw.parents[1] / "00_ceiling" / "raw"
    peaks = json.loads((c0 / "c0_peaks.json").read_text())
    shapes = json.loads((c0 / "c0_shapes.json").read_text())
    sizes = json.loads((raw / "item8_before_bytes.json").read_text())
    before = json.loads((raw / "item8_before.json").read_text())
    rates = {
        name: max(x["tflops"] for x in peaks["gemm"] if x["dtype"] == dtype)
        for name, dtype in (("bf16", "torch.bfloat16"), ("fp32", "torch.float32"))
    }
    rates.update(copy=peaks["copy"]["GBs"], h2d=peaks["pinned"]["GBs"], launch_ms=peaks["replay"]["ms"])
    times = {}
    fixed = {}
    for family in ("cached", "clips"):
        for key, timeset in before[family].items():
            if not isinstance(timeset, dict):
                continue
            prefix, suffix = key.split("_N")
            # All three native video formats share the clip readout and question lengths.
            template = f"f4_v96_N{suffix}" if family == "clips" else key
            floors = []
            for units in shapes[template][: before["args"]["n_calls"]]:
                if family == "clips":
                    tokens = sizes[prefix]["valid_tokens"][0]
                    units = [[n, visual, tokens, qs] for n, visual, _, qs in units]
                floors.append(call_floor(template, units, rates)[0])
            times[key] = {
                "median_ms": median(sum(s.values()) for s in floors),
                "stage_ms": {k: median(s[k] for s in floors) for k in floors[0]},
            }
            fixed[key] = floors[0]
    result = {"rates": rates, "first_15_calls": times, "fixed_first_question": fixed}
    (raw / "item8_floors.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
