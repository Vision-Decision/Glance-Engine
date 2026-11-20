"""Render docs/assets/architecture-{light,dark}.svg (python docs/assets/make_architecture.py).

The README shows the variant matching the reader's GitHub theme through <picture>.
"""

from pathlib import Path

PALETTES = {
    "light": dict(text="#1f2328", muted="#59636e", line="#818b98", edge="#d1d9e0", panel="#f6f8fa",
                  sched="#bc4c00", sched_bg="#fff1e5", prefix="#0969da", prefix_bg="#ddf4ff",
                  q="#8250df", q_bg="#fbefff", out="#1a7f37", out_bg="#dafbe1", bank="#9a6700", bank_bg="#fff8c5"),
    "dark": dict(text="#f0f6fc", muted="#9198a1", line="#9198a1", edge="#3d444d", panel="#151b23",
                 sched="#f0883e", sched_bg="#3d1f0b", prefix="#4493f8", prefix_bg="#0c2d6b",
                 q="#ab7df8", q_bg="#2e1a4f", out="#3fb950", out_bg="#0f3d1f", bank="#d29922", bank_bg="#3b2e0a"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W, H = 1000, 540


def box(x, y, w, h, fill, stroke, r=10, width=1.5):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def text(x, y, s, color, size=14, weight=400, anchor="start", italic=False):
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{style}>{s}</text>')


def arrow(x1, y1, x2, y2, color, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} '
            f'marker-end="url(#head)"/>')


def curve(x1, y1, x2, y2, color):
    mx = (x1 + x2) / 2
    return (f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" marker-end="url(#head)"/>')


def render(p):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="{FONT}" role="img" aria-labelledby="t d">',
           '<title id="t">Glance engine architecture</title>',
           '<desc id="d">A scheduler batches requests into one device step. Each image runs the vision tower and '
           'a shared prefix prefill once; question blocks branch from the shared state and heads read out '
           'answers without decoding. Retained prefixes form a bank that one new question scans in one pass.</desc>',
           f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{p["line"]}"/></marker></defs>']
    # 1. scheduler lane
    out += [box(20, 16, 960, 62, p["sched_bg"], p["sched"]),
            text(40, 43, "SCHEDULER", p["sched"], 13, 700),
            text(40, 63, "one device owner", p["muted"], 13),
            text(190, 43, "Requests from many clients are admitted in order, gathered for up to 0.5 ms and packed",
                 p["text"], 14),
            text(190, 63, "under a token budget into one device step. Python calls and HTTP requests share this path.",
                 p["text"], 14),
            arrow(500, 78, 500, 112, p["line"])]
    # 2. main lane panel
    out += [box(20, 112, 960, 290, p["panel"], p["edge"], 12, 1)]
    # image
    out += [box(40, 190, 100, 100, p["panel"], p["line"], 8),
            f'<polygon points="52,276 82,232 104,262 116,248 130,276" fill="{p["edge"]}"/>',
            f'<circle cx="114" cy="214" r="9" fill="{p["edge"]}"/>',
            text(90, 312, "image", p["muted"], 13, anchor="middle"),
            arrow(140, 240, 168, 240, p["line"])]
    # vision tower
    out += [box(170, 195, 112, 90, p["prefix_bg"], p["prefix"]),
            text(226, 234, "vision", p["text"], 14, 600, "middle"),
            text(226, 252, "tower", p["text"], 14, 600, "middle"),
            text(226, 306, "once per image", p["muted"], 13, anchor="middle"),
            arrow(282, 240, 310, 240, p["line"])]
    # shared prefix
    out += [box(312, 150, 196, 180, p["prefix_bg"], p["prefix"], 12, 2),
            text(410, 182, "shared prefix", p["prefix"], 15, 700, "middle"),
            text(410, 210, "prefilled once", p["text"], 14, anchor="middle"),
            text(410, 236, "Gated DeltaNet state", p["text"], 14, anchor="middle"),
            text(410, 256, "+ attention K/V", p["text"], 14, anchor="middle"),
            text(410, 290, "kept in the prefix", p["muted"], 13, anchor="middle"),
            text(410, 306, "cache for follow-ups", p["muted"], 13, anchor="middle")]
    # question blocks and readouts
    qs = [("Q1 box", "“the red car”", "P(present) · box · P(IoU≥0.5)"),
          ("Q2 yes/no", "“is it raining?”", "P(yes)"),
          ("Q3 choice", "day / night", "P(day), P(night)"),
          ("Qn …", "any typed question", "one distribution each")]
    text_y0 = 150
    out += [text(660, 140, "question blocks branch from the shared state", p["muted"], 13, anchor="middle"),
            text(875, 140, "heads read out answers", p["muted"], 13, anchor="middle")]
    for i, (name, detail, ans) in enumerate(qs):
        y = text_y0 + i * 58
        out += [curve(508, 240, 568, y + 22, p["q"]),
                box(570, y, 180, 44, p["q_bg"], p["q"], 8),
                text(582, y + 19, name, p["q"], 13, 700),
                text(582, y + 36, detail, p["text"], 13),
                arrow(750, y + 22, 776, y + 22, p["line"]),
                box(778, y, 194, 44, p["out_bg"], p["out"], 8),
                text(875, y + 27, ans, p["text"], 13, anchor="middle")]
    out += [text(767, 392, "one forward, every answer at once · no decode loop", p["out"], 13, 600, "middle")]
    # 3. bank lane
    out += [arrow(410, 330, 410, 428, p["bank"], dash=True),
            text(420, 386, "retain", p["bank"], 13, italic=True),
            box(312, 430, 196, 92, p["bank_bg"], p["bank"], 12, 2),
            text(410, 458, "bank of N prefixes", p["bank"], 15, 700, "middle"),
            text(410, 482, "GPU-resident or", p["text"], 13, anchor="middle"),
            text(410, 500, "pinned host memory", p["text"], 13, anchor="middle"),
            arrow(508, 476, 568, 476, p["line"]),
            box(570, 430, 394, 92, p["bank_bg"], p["bank"], 12),
            text(588, 458, "CACHED SCAN", p["bank"], 13, 700),
            text(588, 482, "one new question × N images or clips → N answers", p["text"], 14),
            text(588, 502, "in one packed forward; no vision tower, no prefill", p["text"], 14),
            text(40, 458, "Same engine,", p["muted"], 13),
            text(40, 478, "the other direction:", p["muted"], 13),
            text(40, 498, "one question, many images", p["muted"], 13)]
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    here = Path(__file__).parent
    for name, palette in PALETTES.items():
        (here / f"architecture-{name}.svg").write_text(render(palette))
