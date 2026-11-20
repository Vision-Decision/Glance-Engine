"""Build outputs/sprint_speed/{tables.md, speed_summary.json, cost_vs_q_<gpu>.png} from raw benchmark JSONs.

python speed/report.py --raw outputs/sprint_speed/raw --out outputs/sprint_speed
Raw files: glance_<gpu>.json (speed/bench_glance.py), serve_<backend>[_<variant>]_<gpu>.json (speed/bench_serve.py),
hw_<gpu>.json (speed/hwbench.py).  <gpu> in {g4, l4, ...}.
"""
import argparse, glob, json, os, re
from collections import OrderedDict

QS = [1, 2, 4, 8, 16, 32, 64]
GLANCE_LABEL = OrderedDict(ref_lora="Glance, shipped path (PEFT LoRA, eager)", ref_merged="Glance, LoRA merged (eager)",
                           eng_eager="Glance engine (eager)", eng_graph="Glance engine (CUDA graphs, bf16)",
                           eng_fused="Glance engine (CUDA graphs + fused ops, bf16)",
                           eng_fp8="Glance engine (CUDA graphs, FP8 W8A8)")
# assumed on-demand $/GPU-hour (only used for the $ column; GPU-seconds per 1k questions is price-free)
PRICE = dict(g4=2.00, l4=0.70, h100=6.00, a100=2.50)   # illustrative on-demand $/GPU-hour


def load(raw):
    G, S, H = {}, {}, {}
    for p in sorted(glob.glob(f"{raw}/*.json")):
        b = os.path.basename(p)[:-5]
        d = json.load(open(p))
        if b.startswith("glance"):           # glance_<gpu>, glance2_<gpu>, ... : modes are merged per GPU
            gpu = b.split("_", 1)[1]
            if gpu not in G:
                G[gpu] = dict(d, modes={})
            for m, v in d.get("modes", {}).items():      # deep merge: regimes of one mode may come from several runs
                G[gpu]["modes"].setdefault(m, {}).update(v)
        elif b.startswith("serve_"):
            parts = b[6:].split("_")
            gpu = parts[-1]
            S.setdefault(gpu, OrderedDict())["_".join(parts[:-1])] = d
        elif b.startswith("hw_"):
            H[b[3:]] = d
    return G, S, H


def serve_label(name, d):
    be = d.get("backend", name.split("_")[0])
    v = d.get("version", "")
    a = d.get("args", {})
    extra = []
    if a.get("spec"):
        extra.append(f"MTP spec. decoding k={a.get('spec_k')}")
    if a.get("async_sched"):
        extra.append("async scheduling")
    if a.get("one_token"):
        return (f"Engine-matched decode-free bound: {'vLLM' if be == 'vllm' else 'SGLang'} {v} prefill + 1 token per "
                f"question{', FP8' if a.get('quant') else ''} (no box output)")
    if a.get("quant"):
        extra.append(a["quant"].upper())
    s = f"Qwen3.5-4B generative, {'vLLM' if be == 'vllm' else 'SGLang'} {v} + prefix cache"
    return s + (", " + ", ".join(extra) if extra else "")


def cell(d, k, key="total_ms_median"):
    v = d.get(k, {}).get(key)
    return None if v is None else float(v)


def fmt(v, nd=0):
    return "–" if v is None else (f"{v:.{nd}f}")


DECODE_BYTES = 8.1e9   # Qwen3.5-4B LM weights + tied LM head read once per decode step, bf16 (~4.05 B params)


def roofline(G, S, H, gpu):
    """AR lower bound per regime: the same prefill as the Glance engine (bf16, graphs) + generated tokens x one full
    weight read at the measured copy bandwidth (any decoder without speculative decoding; batching the N questions
    shares the weight reads, so the floor is per decode step, not per question)."""
    hw = H.get(gpu)
    if not hw or gpu not in G or "eng_graph" not in G[gpu]["modes"]:
        return None
    step_ms = DECODE_BYTES / (hw["bw_gbs"] * 1e9) * 1e3
    toks = None
    for d in S.get(gpu, {}).values():
        t = d["modes"].get("bs1_box_N1", {}).get("gen_tokens_mean")
        if t:
            toks = t
            break
    toks = toks or 19.5
    g = G[gpu]["modes"]["eng_graph"]
    out = {k: {"total_ms_median": v["total_ms_median"] + toks * step_ms}
           for k, v in g.items() if isinstance(v, dict) and "total_ms_median" in v and "_box_" in k and k.startswith("bs1")}
    return out, step_ms, toks


def systems(G, S, gpu, H=None):
    out = OrderedDict()
    if gpu in G:
        for m, lab in GLANCE_LABEL.items():
            if m in G[gpu]["modes"]:
                out[lab] = G[gpu]["modes"][m]
    for n, d in S.get(gpu, {}).items():
        lab = serve_label(n, d)            # several files with the same settings (extra regimes) are merged
        out.setdefault(lab, {}).update(d["modes"])
    r = roofline(G, S, H or {}, gpu)
    if r:
        out[f"AR lower bound w/o speculative decoding (Glance prefill + {r[2]:.1f} tok x {r[1]:.1f} ms weight read)"] = r[0]
    return out


def best_baseline(sysm, k, key="total_ms_median", higher=False):
    vals = [(cell(d, k, key), lab) for lab, d in sysm.items()
            if not lab.startswith(("Glance", "AR lower", "Engine-matched")) and cell(d, k, key) is not None]
    if not vals:
        return None, None
    return (max if higher else min)(vals)


def table(sysm, cols, key="total_ms_median", nd=0, head=None):
    lines = ["| system | " + " | ".join(head or cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for lab, d in sysm.items():
        lines.append(f"| {lab} | " + " | ".join(fmt(cell(d, c, key), nd) for c in cols) + " |")
    return "\n".join(lines)


def _pick(sysm):
    """<= 6 series for the report figure (fixed palette slots, reference palette): shipped Glance, best bf16 engine,
    FP8 engine, vLLM bf16 + MTP (fair bf16 pair; falls back to the best bf16 vLLM), vLLM FP8 + MTP (fair FP8 pair),
    and the AR roofline. SGLang is dominated by vLLM in every regime and stays in the tables."""
    k = [f"bs1_box_N{q}" for q in QS]
    mean = lambda d: sum(cell(d, c) or 1e9 for c in k) / len(k)
    out = []
    if "Glance, shipped path (PEFT LoRA, eager)" in sysm:
        out.append(("Glance, shipped path", sysm["Glance, shipped path (PEFT LoRA, eager)"], "#2a78d6", "--", "o"))
    bf = [l for l in sysm if l.startswith("Glance engine") and "FP8" not in l and "graphs" in l]
    if bf:
        out.append(("Glance engine, bf16", sysm[min(bf, key=lambda l: mean(sysm[l]))], "#2a78d6", "-", "o"))
    if "Glance engine (CUDA graphs, FP8 W8A8)" in sysm:
        out.append(("Glance engine, FP8", sysm["Glance engine (CUDA graphs, FP8 W8A8)"], "#1baf7a", "-", "o"))
    vb = [l for l in sysm if ", vLLM " in l and "FP8" not in l and "1 token" not in l]
    if vb:
        l = min(vb, key=lambda l: mean(sysm[l]))
        out.append(("vLLM, bf16" + (" + MTP" if "MTP" in l else ""), sysm[l], "#eb6834", "-", "s"))
    vf = [l for l in sysm if ", vLLM " in l and "FP8" in l and "1 token" not in l]
    if vf:
        out.append(("vLLM, FP8 + MTP", sysm[min(vf, key=lambda l: mean(sysm[l]))], "#e34948", "--", "s"))
    c = [l for l in sysm if l.startswith("AR lower")]
    if c:
        out.append(("AR floor, no spec. decoding", sysm[c[0]], "#8a8984", ":", None))
    return out


def plot(sysm, gpu, out, hw=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for lab, d, col, ls, mk in _pick(sysm):
        xs = [q for q in QS if cell(d, f"bs1_box_N{q}") is not None]
        if not xs:
            continue
        ys = [cell(d, f"bs1_box_N{q}") for q in xs]
        kw = dict(color=col, ls=ls, lw=2.0 if ls == "-" else 1.4, marker=mk, ms=4.5, label=lab)
        ax[0].plot(xs, ys, **kw)
        ax[1].plot(xs, [y / q for y, q in zip(ys, xs)], **kw)
    for a_, t in zip(ax, ("latency per image, all N answers (ms)", "GPU time per question (ms)")):
        a_.set_xscale("log", base=2); a_.set_yscale("log")
        a_.set_xticks(QS); a_.set_xticklabels([str(q) for q in QS])
        a_.set_xlabel("questions per image (N)"); a_.set_ylabel(t)
        a_.grid(True, which="major", color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    gname = (hw or {}).get("gpu", gpu)
    fig.suptitle(f"One <=640-token image, batch 1, {gname}  (vLLM: prefix caching, CUDA graphs, async scheduling)",
                 fontsize=9.5, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    p = f"{out}/cost_vs_q_{gpu}.png"
    fig.savefig(p, dpi=170); fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/sprint_speed/raw")
    ap.add_argument("--out", default="outputs/sprint_speed")
    a = ap.parse_args()
    G, S, H = load(a.raw)
    md, summ = [], {}
    for gpu in sorted(set(G) | set(S)):
        sysm = systems(G, S, gpu, H)
        gname = (H.get(gpu) or {}).get("gpu") or (G.get(gpu) or {}).get("gpu", gpu)
        md.append(f"## {gpu.upper()} ({gname})\n")
        s = summ[gpu] = dict(gpu=gname, latency={}, ratios={}, throughput={}, cost={})
        cols = [f"bs1_box_N{q}" for q in QS]
        md += ["### Latency per call, 1 image (<=640 visual tokens), N box questions: median ms\n",
               table(sysm, cols, head=[f"N={q}" for q in QS]), ""]
        md += ["p90:\n", table(sysm, cols, key="total_ms_p90", head=[f"N={q}" for q in QS]), ""]
        md += ["Time to first answer (ms; Glance: all answers arrive together):\n",
               table(sysm, cols, key="ttfa_ms_median", head=[f"N={q}" for q in QS]), ""]
        mcols = [f"bs1_mixed_N{q}" for q in (4, 8, 16, 32, 64)]
        md += ["### Mixed typed questions (box : yes/no : choice : score = 1:1:1:1), 1 image: median ms\n",
               table(sysm, mcols, head=[f"N={q}" for q in (4, 8, 16, 32, 64)]), ""]
        fcols = [f"followup_{k}_N{q}" for q in (1, 8, 32) for k in ("box", "mixed")]
        md += ["### Follow-up questions on an already-seen image (warm prefix cache), median ms\n",
               table(sysm, fcols, head=[c.replace("followup_", "") for c in fcols]), ""]
        tcols = [f"bs{b}_box_N{n}" for b in (2, 4, 8, 16, 32) for n in (1, 8)] + \
                [f"bs{b}_box_N{n}" for b, n in ((4, 32), (8, 32), (16, 32), (4, 64), (8, 64))]
        md += ["### Batched images (throughput regime): questions/s\n",
               table(sysm, tcols, key="q_per_s", head=[c.replace("_box", "") for c in tcols]), ""]
        vcols = [f"video_f{b}_N{n}" for b in (8, 32, 64) for n in (1, 4)]
        md += ["### Video-style: F frames (<=256 visual tokens each) x N questions per frame, one call: median ms\n",
               table(sysm, vcols, head=[c.replace("video_", "") for c in vcols]), ""]
        md += ["(questions/s)\n", table(sysm, vcols, key="q_per_s", nd=0, head=[c.replace("video_", "") for c in vcols]), ""]
        gcols = [f"gui{nv}_box_N{n}" for nv in (2048, 4096) for n in (1, 8, 32)]
        md += ["### GUI screenshots (ScreenSpot-Pro, <=2048 / <=4096 visual tokens), 1 image: median ms\n",
               table(sysm, gcols, head=[c.replace("_box", "") for c in gcols]), ""]
        # ratios vs strongest baseline
        refs = ["Glance engine (CUDA graphs, bf16)", "Glance engine (CUDA graphs + fused ops, bf16)", "Glance engine (eager)"]
        rows = []
        for c in cols + mcols + fcols + gcols + vcols:
            gs = [cell(sysm.get(r_, {}), c) for r_ in refs]
            gs = [x for x in gs if x]
            g = min(gs) if gs else None
            bv, bl = best_baseline(sysm, c)
            if g and bv:
                s["ratios"][c] = dict(glance_ms=g, best_baseline_ms=bv, best_baseline=bl, speedup=bv / g)
                rows.append(f"| {c} | {g:.1f} | {bv:.1f} | {bl.replace('Qwen3.5-4B generative, ', '')} | {bv / g:.2f}x |")
        md += ["### Speed-up of the Glance engine (best bf16 variant; numerically equivalent) over the strongest generative baseline per regime\n",
               "| regime | Glance ms | best baseline ms | best baseline | speed-up |", "|---|---|---|---|---|"] + rows + [""]
        # saturation throughput and cost per 1k questions
        crow = []
        for lab, d in sysm.items():
            for n in (1, 8, 32, 64):
                cand = [(d[k]["q_per_s"], k) for k in d if isinstance(d.get(k), dict) and re.fullmatch(rf"bs\d+_box_N{n}", k) and "q_per_s" in d[k]]
                if not cand:
                    continue
                qps, k = max(cand)
                gpu_s = 1000 / qps
                usd = gpu_s / 3600 * PRICE.get(gpu, float("nan"))
                s["cost"].setdefault(lab, {})[f"N{n}"] = dict(q_per_s=qps, at=k, gpu_s_per_1k=gpu_s, usd_per_1k=usd)
                crow.append(f"| {lab} | {n} | {qps:.0f} ({k}) | {gpu_s:.2f} | {usd:.4f} |")
        md += [f"### Saturation throughput and cost per 1k questions (assumed ${PRICE.get(gpu, float('nan')):.2f}/GPU-hour)\n",
               "| system | questions/image | best q/s (config) | GPU-s per 1k q | $ per 1k q |", "|---|---|---|---|---|"] + crow + [""]
        if sysm:
            s["plot"] = plot(sysm, gpu, a.out, H.get(gpu))
        s["latency"] = {lab: {c: cell(d, c) for c in cols + mcols + fcols + gcols + vcols if cell(d, c) is not None} for lab, d in sysm.items()}
        if gpu in G:
            s["equiv"] = G[gpu].get("equiv")
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/tables.md", "w").write("# Sprint speed: generated tables\n\n" + "\n".join(md))
    json.dump(dict(summary=summ, hw=H), open(f"{a.out}/speed_summary.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
