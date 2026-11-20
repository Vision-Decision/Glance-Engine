"""Render demo/glance_vs_vllm.gif from measured timelines (demo/data/race_*.json).

python3 demo/render_race.py --image path/to/000000275198.jpg [--slow 10]

Replays one measured call per system: the call whose total time is the median of its 50 timed calls. A box is drawn
at the moment its answer reached the host in that call, with the time axis slowed by --slow. Boxes are the systems'
own outputs from the measured run (Glance: shown when P(present) > 0.5; vLLM: the parsed '[x1, y1, x2, y2]', or
nothing for 'none').
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

from render_common import BG, PANEL, LINE, TEXT, MUTED, GLANCE, VLLM, font, palette, text_w, pill, save_gif

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 960, 640


def median_call(run):
    calls = run["calls"]
    order = np.argsort([c["total"] for c in calls])
    return calls[int(order[len(order) // 2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--slow", type=float, default=10.0)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "glance_vs_vllm.gif"))
    a = ap.parse_args()
    g = json.load(open(os.path.join(HERE, "data/race_glance.json")))
    v = json.load(open(os.path.join(HERE, "data/race_vllm_fp8mtp.json")))
    gr, vr = g["runs"][f"glance_bf16_N{a.N}"], v["runs"][f"vllm_fp8mtp_N{a.N}"]
    gc, vc = median_call(gr), median_call(vr)
    g_t = np.array(gc["answers_at_ms"][:a.N])
    v_t = np.array(vc["answers_at_ms"][:a.N])
    g_boxes = [x["box"] if x["p_present"] > 0.5 else None for x in gr["answers"]]
    # vLLM boxes: parse the texts of the replayed call itself
    from_text = __import__("re").findall
    v_boxes = []
    for t in vc["texts"]:
        m = from_text(r"-?\d+(?:\.\d+)?", t)
        v_boxes.append(None if ("none" in t.lower() or len(m) < 4) else [float(x) / 1000 for x in m[:4]])
    g_total, v_total = float(gc["total"]), float(vc["total"])
    g_med, v_med = gr["total_ms_median"], vr["total_ms_median"]
    cols = palette(a.N)

    img = Image.open(a.image).convert("RGB")
    pw = 450
    ph = int(round(img.height * pw / img.width))
    photo = img.resize((pw, ph), Image.LANCZOS)
    dim = Image.blend(photo, Image.new("RGB", photo.size, (0, 0, 0)), 0.18)
    px = [20, W - 20 - pw]
    py = 104
    axis_y0 = py + ph + 26
    t_max = v_total * 1.04
    ax_x0, ax_x1 = 112, W - 28

    f_title, f_hdr, f_small, f_mono, f_big = font("bold", 22), font("bold", 16), font("sans", 14), font("mono", 15), font("mono", 26)

    def tx(t):
        return ax_x0 + (ax_x1 - ax_x0) * min(t, t_max) / t_max

    def draw_boxes(d, x0, boxes, times, t, flash):
        for k, (b, tk) in enumerate(zip(boxes, times)):
            if b is None or tk > t:
                continue
            x1, y1, x2, y2 = [max(0.0, min(1.0, z)) for z in b]
            r = [x0 + x1 * pw, py + y1 * ph, x0 + x2 * pw, py + y2 * ph]
            fresh = (t - tk) < flash
            d.rectangle([r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1], outline=(0, 0, 0), width=1)
            d.rectangle(r, outline=(255, 255, 255) if fresh else cols[k], width=3 if fresh else 2)

    def frame(t, final=False):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((20, 16), f"{a.N} box questions about one image", font=f_title, fill=TEXT)
        d.text((20, 46), "Every box appears when its answer reaches the host.", font=f_small, fill=MUTED)
        clock = f"t = {min(t, v_total):6.1f} ms"
        d.text((W - 20 - text_w(d, clock, f_big), 12), clock, font=f_big, fill=TEXT)
        sm = f"{a.slow:g}× SLOW MOTION"
        fb = font("bold", 15)
        pill(d, (W - 20 - text_w(d, sm, fb) - 16, 44), sm, (240, 246, 252), fg=BG, f=fb)
        for side, (x0, name, color, times, boxes, total) in enumerate(
                [(px[0], "Glance  ·  bf16, one forward pass", GLANCE, g_t, g_boxes, g_total),
                 (px[1], "vLLM  ·  FP8 + MTP speculative decoding", VLLM, v_t, v_boxes, v_total)]):
            n_done = int((times <= t).sum())
            d.text((x0, py - 30), name, font=f_hdr, fill=color)
            cnt = f"{n_done}/{a.N} answered"
            d.text((x0 + pw - text_w(d, cnt, f_small), py - 27), cnt, font=f_small, fill=TEXT if n_done == a.N else MUTED)
            im.paste(photo if n_done else dim, (x0, py))
            draw_boxes(d, x0, boxes, times, t, flash=0.0 if final else 3 * 40.0 / a.slow)
            d.rectangle([x0 - 1, py - 1, x0 + pw, py + ph], outline=color if n_done == a.N else LINE, width=2 if n_done == a.N else 1)
            if n_done == a.N:
                nb = sum(b is not None for b in boxes)
                no = "P(present) < 0.5" if side == 0 else "'none'"
                pill(d, (x0 + 8, py + ph - 30), f"{total:.0f} ms · {nb} boxes · {a.N - nb} answered {no}", color)
        # shared time axis with one lane per system
        d.line([ax_x0, axis_y0, ax_x1, axis_y0], fill=LINE, width=1)
        step = 50 if t_max < 600 else 100
        for ms in range(0, int(t_max) + 1, step):
            x = tx(ms)
            d.line([x, axis_y0 - 3, x, axis_y0 + 3], fill=MUTED)
            lab = f"{ms} ms"
            d.text((x - text_w(d, lab, f_small) / 2, axis_y0 + 6), lab, font=f_small, fill=MUTED)
        for lane, (name, color, times) in enumerate([("Glance", GLANCE, g_t), ("vLLM", VLLM, v_t)]):
            y = axis_y0 + 34 + lane * 28
            d.text((20, y - 9), name, font=f_hdr, fill=color)
            d.rounded_rectangle([ax_x0, y - 7, ax_x1, y + 7], radius=7, fill=PANEL)
            d.rounded_rectangle([ax_x0, y - 7, max(ax_x0 + 8, tx(min(t, times.max()))), y + 7], radius=7,
                                fill=tuple(int(c * 0.35) for c in color))
            for k, tk in enumerate(times):
                if tk <= t:
                    x = tx(tk)
                    d.line([x, y - 7, x, y + 7], fill=cols[k], width=2)
        cx = tx(t)
        d.line([cx, axis_y0 - 2, cx, axis_y0 + 34 + 28 + 10], fill=TEXT, width=1)
        if final:
            msg = f"{v_med / g_med:.1f}× faster: {g_med:.0f} ms vs {v_med:.0f} ms (medians of 50 calls)"
            f = font("bold", 24)
            d.text(((W - text_w(d, msg, f)) / 2, axis_y0 + 86), msg, font=f, fill=TEXT)
        cap1 = (f"Same GPU (RTX PRO 6000), same image and {a.N} questions, image to answers on the host. "
                f"Replay of each system's median call, {a.slow:g}× slow motion.")
        cap2 = ("Glance-4B bf16 vs Qwen3.5-4B in vLLM 0.34 (prefix caching, CUDA graphs, async scheduling). "
                "Photo: COCO val2017 #275198, CC BY 2.0.")
        d.text((20, H - 44), cap1, font=f_small, fill=MUTED)
        d.text((20, H - 24), cap2, font=f_small, fill=MUTED)
        d.rectangle([0, 0, W - 1, H - 1], outline=LINE, width=1)
        return im

    real_per_frame = 40.0 / a.slow           # 40 ms GIF frames (25 fps)
    frames, durs = [frame(0.0)], [600]
    t = 0.0
    end = v_total
    while t < end:
        t = min(end, t + real_per_frame)
        frames.append(frame(t))
        durs.append(40)
    frames.append(frame(end, final=True)); durs.append(3200)
    size = save_gif(frames, durs, a.out, pal_frames=[frames[0], frames[-1]])
    print(a.out, f"{size / 1e6:.2f} MB", len(frames), "frames", f"{sum(durs) / 1e3:.1f} s",
          f"glance call {g_total:.1f} ms (median {g_med:.1f}), vllm call {v_total:.1f} ms (median {v_med:.1f})")


if __name__ == "__main__":
    main()
