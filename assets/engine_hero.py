"""Hero: one prefill + one forward for N typed questions, vs token-by-token decoding.
Measured times: outputs/engine_opt/CEILING.md, final-state table, row
"One image, <=640 visual tokens", box Q=32 (Glance 48.9; vLLM BF16 334.0; SGLang BF16 658.0 ms)."""
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.pyplot as plt
from figstyle import render

def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.9, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def image_icon(ax, x, y, s, t, accent):
    box(ax, x, y, s, s * 0.8, t["card"], t["cardline"], r=0.7)
    ax.add_patch(Polygon([(x + s*.12, y + s*.12), (x + s*.42, y + s*.52), (x + s*.62, y + s*.3),
                          (x + s*.74, y + s*.42), (x + s*.9, y + s*.12)], closed=True,
                         fc=accent, ec="none", alpha=.85, zorder=3))
    ax.add_patch(plt.Circle((x + s*.72, y + s*.6), s*.08, fc=accent, ec="none", alpha=.85, zorder=3))

def arrow(ax, x0, x1, y, t):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.3, mutation_scale=11))

def draw(t, mode):
    fig = plt.figure(figsize=(12.8, 5.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(0, 56); ax.axis("off")

    # ---------- lane 1: generative server ----------
    yA = 36.0
    ax.text(2, 53, "Generative server", fontsize=13.5, weight="bold", color=t["ink"], va="top")
    ax.text(2, 49.6, "vLLM, SGLang: every question decodes its answer as text, one token per forward step",
            fontsize=10.5, color=t["ink2"], va="top")
    image_icon(ax, 2, yA, 8, t, t["gray"])
    ax.text(6, yA - 2.2, "image +\n32 questions", ha="center", va="top", fontsize=9, color=t["ink2"])
    arrow(ax, 10.8, 13.3, yA + 3.2, t)
    box(ax, 13.5, yA, 12, 6.4, t["gray_t"], t["cardline"])
    ax.text(19.5, yA + 3.2, "vision +\nprefill", ha="center", va="center", fontsize=9.5, color=t["ink"])
    arrow(ax, 26.2, 28.6, yA + 3.2, t)
    toks = ["[", "412", ",", "87", ",", "655", ",", "903", "]", "eos"]
    cw, gap = 5.7, 0.7
    rows = [(yA + 5.0, "Q1"), (yA + 2.6, "Q2"), (yA - 1.4, "Q32")]
    for ry, lab in rows:
        ax.text(30.6, ry + 0.9, lab, ha="right", va="center", fontsize=8, color=t["muted"])
        for i, tok in enumerate(toks):
            x = 31 + i * (cw + gap)
            box(ax, x, ry, cw, 1.8, t["orange_t"], t["orange"], lw=0.9, r=0.4)
            ax.text(x + cw/2, ry + 0.9, tok, ha="center", va="center", fontsize=8, color=t["ink"])
    ax.text(31 + 4.5*(cw+gap), yA + 1.25, "⋮", ha="center", va="center", fontsize=11, color=t["muted"])
    xend = 31 + len(toks) * (cw + gap)
    ax.annotate("", xy=(xend - gap, yA - 3.2), xytext=(31, yA - 3.2),
                arrowprops=dict(arrowstyle="-|>", color=t["orange"], lw=1.3, mutation_scale=10))
    ax.text(31, yA - 4.4, "one decode step per token (a box allows up to 24), then parse the text",
            fontsize=9, color=t["ink2"], va="top")
    ax.text(127, yA + 4.6, "334.0 ms", ha="right", va="center", fontsize=20, weight="bold", color=t["ink2"])
    ax.text(127, yA + 1.4, "vLLM BF16  ·  fastest vLLM (FP8+MTP): 229.2 ms", ha="right", va="center", fontsize=9.5, color=t["ink2"])

    ax.plot([2, 126], [25.0, 25.0], color=t["grid"], lw=1)

    # ---------- lane 2: Glance Engine ----------
    yB = 6.5
    ax.text(2, 21.6, "Glance Engine", fontsize=13.5, weight="bold", color=t["blue"], va="top")
    ax.text(2, 18.2, "one packed forward over the image and every question block; heads read the answers",
            fontsize=10.5, color=t["ink2"], va="top")
    image_icon(ax, 2, yB, 8, t, t["blue"])
    ax.text(6, yB - 2.2, "image +\n32 questions", ha="center", va="top", fontsize=9, color=t["ink2"])
    arrow(ax, 10.8, 13.3, yB + 3.2, t)
    box(ax, 13.5, yB, 12, 6.4, t["blue_t"], t["blue"])
    ax.text(19.5, yB + 3.2, "vision +\nprefill", ha="center", va="center", fontsize=9.5, color=t["ink"])
    # branches: all question blocks in the same forward
    bx = 29
    for i, yy in enumerate([yB + 5.4, yB + 3.8, yB + 2.2, yB - 1.0]):
        ax.plot([25.5, bx], [yB + 3.2, yy + 0.7], color=t["blue"], lw=0.9, alpha=.7, zorder=1)
        box(ax, bx, yy, 7.5, 1.4, t["blue"], t["blue"], lw=0, r=0.35)
        ax.text(bx + 3.75, yy + 0.7, ["Q1", "Q2", "Q3", "Q32"][i], ha="center", va="center",
                fontsize=7.5, color=t["onfill"], weight="bold")
    ax.text(bx + 3.75, yB + 1.05, "⋮", ha="center", va="center", fontsize=9, color=t["muted"])
    ax.text(bx, yB - 2.8, "1 forward, no decode loop", fontsize=9, color=t["ink2"], va="top")
    arrow(ax, 37.2, 40.2, yB + 3.2, t)
    box(ax, 40.4, yB + 0.4, 7.2, 5.6, t["blue_t"], t["blue"])
    ax.text(44, yB + 3.2, "heads", ha="center", va="center", fontsize=9.5, color=t["ink"])
    arrow(ax, 47.8, 50.8, yB + 3.2, t)
    answers = [("Box", "P(present), box [x1, y1, x2, y2], P(IoU ≥ 0.5)"),
               ("Yes / no", "P(yes)"),
               ("Choice", "a probability for every option")]
    for i, (k, v) in enumerate(answers):
        yy = yB + 4.9 - i * 2.45
        box(ax, 51, yy - 0.95, 42, 1.95, t["card"], t["cardline"], lw=0.9, r=0.45)
        ax.text(52.2, yy, k, va="center", fontsize=9, weight="bold", color=t["blue"])
        ax.text(60.0, yy, v, va="center", fontsize=9, color=t["ink"])
    ax.text(127, yB + 4.6, "48.9 ms", ha="right", va="center", fontsize=26, weight="bold", color=t["blue"])
    ax.text(127, yB + 1.0, "6.8× faster than vLLM BF16", ha="right", va="center", fontsize=10.5,
            color=t["ink"])
    ax.text(127, yB - 1.6, "32 box questions, one image, RTX PRO 6000", ha="right", va="center",
            fontsize=8.5, color=t["muted"])
    return fig

if __name__ == "__main__":
    render("engine-hero", draw)
