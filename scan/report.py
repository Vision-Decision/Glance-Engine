"""Phase-1 tables from bench_scan.py / bench_vllm.py JSON.   python -m scan.report GLANCE.json [VLLM.json ...] > table.md"""
import json, sys


def fmt(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    g = json.load(open(sys.argv[1]))
    vs = [json.load(open(p)) for p in sys.argv[2:]]
    vtag = lambda v: ("vLLM " + ("FP8" if v["args"].get("quant") else "bf16") + ("+MTP" if v["args"].get("spec") else ""))
    print(f"GPU: {g['gpu']}\n")
    if "equiv" in g:
        print("## Exactness vs each image alone (shipped path, bf16)\n\n| set / query / path | max ΔP | mean ΔP | decision agree | max Δbox |\n|---|---|---|---|---|")
        for k, d in g["equiv"].items():
            if "p_absdiff_max" in d:
                print(f"| {k} | {d['p_absdiff_max']:.4f} | {d['p_absdiff_mean']:.5f} | {d['decision_agree']:.3f} | {d['box_absdiff_max']:.4f} |")
            else:
                print(f"| {k} | {d['prob_absdiff_max']:.4f} | | {d['decision_agree']:.3f} | |")
        print()
    for tag in ("cached", "cached_fp8"):
        if tag not in g:
            continue
        c = g[tag]
        print(f"## (A) {tag}: one new query over N cached images (ms, median; p90)\n")
        for Tv in (640, 256, 64):
            rows = [(k, v) for k, v in c.items() if k.startswith(f"Tv{Tv}_N") and isinstance(v, dict)]
            if not rows:
                continue
            print(f"Tv={Tv}: bank {fmt(c.get(f'Tv{Tv}_bank_MB_per_image'))} MB/image, encode {fmt(c.get(f'Tv{Tv}_encode_ms_per_image'), 2)} ms/image\n")
            print("| N | readout | Glance total | p90 | " + " | ".join(f"{vtag(v)} {k}" for v in vs for k in ("yesno", "box")) + " |")
            print("|---" * (4 + 2 * len(vs)) + "|")
            for k, v in rows:
                N, kind = k.split("_")[1][1:], k.split("_")[2]
                cells = []
                for vv in vs:
                    for kk in ("yesno", "box"):
                        m = vv.get("modes", {}).get(f"cached_Tv{Tv}_N{N}_{kk}")
                        cells.append(fmt(m["total_ms"]) if m else "—")
                print(f"| {N} | {kind} | {fmt(v['total_ms'])} | {fmt(v['total_ms_p90'])} | " + " | ".join(cells) + " |")
            print()
    for tag in ("fresh", "fresh_fp8", "fresh_qfirst"):
        if tag not in g:
            continue
        f = g[tag]
        Tvs = sorted({int(k.split("_")[0][2:]) for k in f})
        Ns = sorted({int(k.split("_")[1][1:]) for k in f})
        print(f"## (B) {tag}: N new frames + 1 query, end to end (ms; < 100 ms in bold)\n")
        print("| Tv \\ N | " + " | ".join(map(str, Ns)) + " |\n|---" + "|---" * len(Ns) + "|")
        for Tv in Tvs:
            cells = []
            for N in Ns:
                r = f.get(f"Tv{Tv}_N{N}")
                cells.append("—" if r is None else (f"**{r['total_ms']:.0f}**" if r["total_ms"] < 100 else f"{r['total_ms']:.0f}"))
            print(f"| {Tv} | " + " | ".join(cells) + " |")
        print()
        for vv in vs:
            print(f"{vtag(vv)} fresh (yes/no 1 token | box): ")
            for Tv in Tvs:
                cells = []
                for N in Ns:
                    y = vv.get("modes", {}).get(f"fresh_Tv{Tv}_N{N}_yesno"); b = vv.get("modes", {}).get(f"fresh_Tv{Tv}_N{N}_box")
                    cells.append(f"{fmt(y['total_ms'], 0) if y else '—'}/{fmt(b['total_ms'], 0) if b else '—'}")
                print(f"  Tv={Tv}: " + "  ".join(f"N{N}={c}" for N, c in zip(Ns, cells)))
            print()
    if "acc_vs_tokens" in g:
        print("## Accuracy vs visual tokens per image\n\n| Tv | RefCOCO Acc@0.5 | gRefCOCO presence AUROC | gRefCOCO FA@.5 | gRefCOCO miss@.5 |\n|---|---|---|---|---|")
        for k, r in g["acc_vs_tokens"].items():
            a, b = r["refcoco_val"], r["grefcoco_val"]
            print(f"| {k} | {fmt(a['acc50'], 3)} | {fmt(b.get('auroc'), 3)} | {fmt(b.get('false_alarm_at_05'), 3)} | {fmt(b.get('miss_at_05'), 3)} |")
        print()


if __name__ == "__main__":
    main()
