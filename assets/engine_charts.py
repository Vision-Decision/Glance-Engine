"""Engine latency charts. Every number is copied from
outputs/engine_opt/CEILING.md, section "Final state (frozen at 8bc7d58)", final-state table
(columns Final Glance / vLLM BF16 / SGLang BF16 / Best measured full-output vLLM [FP8+MTP only]).
Warm median ms, one RTX PRO 6000 Blackwell."""
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import render

# (family, cell, glance, vllm_bf16, sglang_bf16, vllm_fp8_mtp or None)
CELLS = []
def add(fam, names, g, v, s, f=None):
    f = f or [None] * len(g)
    for row in zip(names, g, v, s, f):
        CELLS.append((fam,) + row)
add("one image", ["box Q1", "box Q8", "box Q32", "mixed Q8", "mixed Q32"],
    [37.7, 38.5, 48.9, 46.2, 69.2], [166.5, 221.0, 334.0, 199.2, 278.2],
    [187.3, 395.8, 658.0, 375.3, 655.4], [99.2, 134.0, 229.2, 122.1, 177.0])
add("GUI", ["2048 Q1", "2048 Q8", "2048 Q32", "4096 Q1", "4096 Q8", "4096 Q32"],
    [119.2, 122.6, 146.1, 277.9, 282.9, 311.9], [226.2, 311.7, 572.2, 353.7, 418.4, 566.4],
    [301.7, 566.9, 1406.2, 489.0, 1305.1, 2398.4], [149.5, 208.0, 418.6, 262.4, 299.5, 435.7])
add("video", ["32 frames", "64 frames"], [290.1, 587.7], [549.4, 928.1], [666.3, 1052.0], [387.1, 695.7])
add("cached scan", ["Tv640 box N64", "Tv640 box N256", "Tv640 yn N64", "Tv640 yn N256",
                    "Tv256 box N64", "Tv256 box N256", "Tv64 box N64", "Tv64 box N256",
                    "4f v96 N64", "4f v96 N256", "4f v48 N64", "4f v48 N256", "2f v96 N64", "2f v96 N256"],
    [23.7, 96.7, 45.4, 175.4, 22.5, 84.5, 22.0, 82.6, 55.7, 241.4, 55.0, 234.9, 54.9, 234.3],
    [899.1, 3151.6, 680.6, 2679.9, 653.1, 2950.1, 401.3, 1112.6, 342.7, 1355.1, 236.0, 907.7, 214.2, 823.3],
    [1556.6, 7732.9, 1287.3, 5670.7, 592.2, 2981.4, 446.1, 1440.6, 620.7, 2203.7, 556.5, 2002.2, 479.1, 1784.3],
    [720.7, 2672.2, 582.8, 2292.3, 479.4] + [None] * 9)
add("fresh scan", [f"Tv{tv} N{n}" for tv in (128, 256) for n in (32, 64, 128, 256)],
    [142.9, 287.5, 581.5, 1165.3, 301.7, 602.1, 1207.5, 2420.5],
    [177.9, 332.1, 637.8, 1252.8, 322.8, 629.4, 1244.7, 2495.9],
    [259.3, 431.4, 815.3, 1572.9, 461.0, 848.4, 1567.6, 2997.1])
add("follow-up", ["box Q1", "box Q8", "box Q32", "mixed Q1", "mixed Q8", "mixed Q32"],
    [8.8, 10.3, 19.6, 8.8, 15.0, 39.9], [145.2, 196.6, 316.9, 145.1, 175.3, 254.9],
    [162.0, 346.1, 606.5, 161.7, 342.2, 561.8], [78.3, 112.9, 211.8, 78.5, 96.6, 157.4])
for w, g, v, s in [
    ("cold box", [32.2, 104.0, 362.7, 1424.3], [166.2, 264.9, 578.6, 1828.0], [188.1, 322.8, 715.4, 2330.0]),
    ("cold yes/no", [32.2, 104.3, 370.3, 1456.3], [39.5, 111.1, 381.7, 1505.3], [53.8, 157.2, 508.6, 1946.8]),
    ("8 mixed", [45.3, 141.1, 534.0, 2144.3], [199.6, 351.1, 979.4, 3499.5], [378.3, 888.8, 2734.1, 12305.1]),
    ("cached box", [8.7, 17.3, 47.4, 173.1], [140.3, 182.0, 253.3, 1019.0], [167.1, 248.2, 452.8, 1801.8]),
    ("GUI box", [98.3, 354.1, 1384.3, 5515.6], [233.7, 521.2, 1607.2, 5983.4], [297.6, 705.5, 2221.4, 8640.1]),
    ("32 fresh frames", [157.5, 609.3, 2422.7, 9695.5], [181.1, 635.2, 2552.3, 10603.3], [248.9, 809.8, 3069.3, 12724.0])]:
    add("clients", [f"{w} C{c}" for c in (1, 4, 16, 64)], g, v, s)
assert len(CELLS) == 65, len(CELLS)
assert sum(c[5] is not None for c in CELLS) == 24

# ---------- headline: the README workloads ----------
HEAD = [  # label, glance, vllm bf16, sglang bf16, fp8+mtp
    ("One image,\n32 box questions", 48.9, 334.0, 658.0, 229.2),
    ("Follow-up on a cached image,\n32 box questions", 19.6, 316.9, 606.5, 211.8),
    ("Video, 64 frames,\none box question each", 587.7, 928.1, 1052.0, 695.7),
    ("Cached scan: one box question\nover 256 images", 96.7, 3151.6, 7732.9, 2672.2),
    ("64 concurrent clients,\none cold box question each", 1424.3, 1828.0, 2330.0, None),
    ("GUI screenshot (4,096 tokens),\n1 box question", 277.9, 353.7, 489.0, 262.4),
]

def headline(t, mode):
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    fig.subplots_adjust(left=0.25, right=0.84, top=0.86, bottom=0.12)
    n = len(HEAD)
    for i, (lab, g, v, s, f) in enumerate(HEAD):
        y = n - 1 - i
        others = [x for x in (v, s, f) if x]
        ax.plot([g, max(others)], [y, y], color=t["grid"], lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([s], [y], s=70, color=t["aqua"], edgecolor=t["axis"] if False else "none", zorder=3)
        ax.scatter([v], [y], s=70, color=t["orange"], zorder=3)
        if f:
            ax.scatter([f], [y], s=70, facecolor="none", edgecolor=t["violet"], lw=2.2, zorder=6)
        ax.scatter([g], [y], s=150, color=t["blue"], zorder=5)
        best = min(x for x in (v, s, f) if x)
        ratio = best / g
        if ratio >= 1:
            txt, col, w = f"{ratio:.1f}× faster" if ratio >= 1.95 else f"{ratio:.2f}× faster", t["ink"], "bold"
        else:
            txt, col, w = f"{ratio:.2f}×  (FP8+MTP wins)", t["ink2"], "normal"
        ax.text(1.015, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=11, weight=w, color=col)
        if f and f < g:
            ax.text(g * 1.1, y + 0.26, f"{g:,.1f}", ha="left", va="bottom", fontsize=9.5, color=t["blue"], weight="bold")
        else:
            ax.text(g / 1.12, y, f"{g:,.1f}", ha="right", va="center", fontsize=9.5, color=t["blue"], weight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels([h[0] for h in HEAD][::-1], fontsize=10, color=t["ink"])
    ax.set_xscale("log"); ax.set_xlim(8, 12000)
    ticks = [10, 30, 100, 300, 1000, 3000, 10000]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{x:,}" for x in ticks])
    ax.minorticks_off()
    ax.grid(axis="x", color=t["grid"], lw=1); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("warm median latency, ms (log scale, lower is better)", fontsize=10, color=t["ink2"])
    ax.text(1.015, 1.02, "vs fastest baseline\n(any precision)", transform=ax.transAxes, fontsize=9, color=t["muted"], va="bottom")
    h = [Line2D([], [], marker="o", ls="", ms=11, color=t["blue"], label="Glance Engine (BF16)"),
         Line2D([], [], marker="o", ls="", ms=8, color=t["orange"], label="vLLM BF16"),
         Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=t["violet"], mew=2, label="vLLM FP8 + MTP (lossy)"),
         Line2D([], [], marker="o", ls="", ms=8, color=t["aqua"], label="SGLang BF16")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.25, 0.985), ncol=4, frameon=False,
               fontsize=10, labelcolor=t["ink"], handletextpad=0.3, columnspacing=1.6)
    return fig

# ---------- all 65 cells: speedup strip ----------
def strip(t, mode):
    import random
    rnd = random.Random(0)
    rows = [("vs vLLM BF16", t["orange"], False, [(c[3] / c[2], c) for c in CELLS]),
            ("vs SGLang BF16", t["aqua"], False, [(c[4] / c[2], c) for c in CELLS]),
            ("vs vLLM FP8 + MTP\n(lossy, 24 matched cells)", t["violet"], True,
             [(c[5] / c[2], c) for c in CELLS if c[5]])]
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    fig.subplots_adjust(left=0.2, right=0.82, top=0.9, bottom=0.17)
    ax.axvspan(0.6, 1.0, color=t["red"], alpha=0.07 if mode == "light" else 0.18, lw=0)
    ax.axvline(1.0, color=t["ink2"], lw=1.2)
    ax.text(0.985, 2.62, "baseline\nfaster", ha="right", va="center", fontsize=9, color=t["red"])
    ax.text(1.04, 2.62, "Glance faster →", ha="left", va="center", fontsize=9, color=t["ink2"])
    for i, (lab, col, hollow, pts) in enumerate(rows):
        y = 2 - i
        for r, c in pts:
            yy = y + rnd.uniform(-0.22, 0.22) if r >= 1 else y
            if hollow:
                ax.scatter(r, yy, s=34, facecolor="none", edgecolor=col, lw=1.6, zorder=3)
            else:
                ax.scatter(r, yy, s=34, color=col, alpha=0.9, edgecolor="none", zorder=3)
        rs = [r for r, _ in pts]
        lo, hi = min(rs), max(rs)
        wins = sum(r > 1 for r in rs)
        ax.text(1.02, y + 0.12, f"faster in {wins} / {len(rs)}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=11, weight="bold", color=t["ink"])
        ax.text(1.02, y - 0.2, f"range {lo:.2f}× to {hi:.0f}×", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9.5, color=t["ink2"])
        if lo < 1:
            c = min(pts)[1]
            ax.annotate(f"GUI screenshot, 4,096 tokens, 1 question:\nGlance {c[2]} ms, vLLM FP8+MTP {c[5]} ms", xy=(lo, y), xytext=(1.12, y + 0.62),
                        fontsize=8.5, color=t["ink"], ha="left",
                        arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=t["ink"])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xscale("log"); ax.set_xlim(0.7, 110)
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{x:g}×" for x in ticks]); ax.minorticks_off()
    ax.grid(axis="x", color=t["grid"], lw=1); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("speedup = baseline latency / Glance latency, per workload cell (log scale)", fontsize=10, color=t["ink2"])
    return fig

if __name__ == "__main__":
    for c in CELLS:
        pass
    print("min vs vllm", min(c[3]/c[2] for c in CELLS), "max", max(c[3]/c[2] for c in CELLS))
    print("min vs sglang", min(c[4]/c[2] for c in CELLS), "max", max(c[4]/c[2] for c in CELLS))
    print("fp8", sorted(c[5]/c[2] for c in CELLS if c[5])[:3])
    render("engine-latency", headline)
    render("engine-all-cells", strip)
