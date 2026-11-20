"""Render demo/search_256.gif from the measured search runs (demo/data/search_*.json), in real time (1x).

python3 demo/render_search.py [--present "a giraffe"] [--absent "a bird"]

Two queries over the same 256 COCO images, each replaying one measured call per system (the call with the median
total time). Glance's call returns all 256 answers at once; the vLLM bar advances as each of its 256 requests
completes. Hits and the verdict are Glance's measured outputs: images selected by the calibrated rule, boxes from
the same call.
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

from render_common import BG, PANEL, LINE, TEXT, MUTED, GLANCE, VLLM, font, text_w, pill, save_gif

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
W, H = 960, 640
TILE, GAP = 28, 2
GX, GY = 20, 104


def median_call(calls, key="total"):
    order = np.argsort([c[key] for c in calls])
    return calls[int(order[len(order) // 2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--present", default=None)
    ap.add_argument("--absent", default=None)
    ap.add_argument("--vllm", default=None, help="bf16 | fp8mtp (default: the faster median)")
    ap.add_argument("--out", default=os.path.join(HERE, "search_256.gif"))
    a = ap.parse_args()
    g = json.load(open(os.path.join(D, "search_glance.json")))
    pick = json.load(open(os.path.join(D, "coco_pick.json")))["search"]
    vs = {c: json.load(open(os.path.join(D, f"search_vllm_{c}.json"))) for c in ("bf16", "fp8mtp")
          if os.path.exists(os.path.join(D, f"search_vllm_{c}.json"))}
    qn = list(g["queries"])
    present = a.present or qn[0]
    # "a bird" is a poor absent query: COCO leaves bird logos and far-off birds unannotated
    absent = a.absent or "a toaster"
    if a.vllm is None:
        a.vllm = min(vs, key=lambda c: np.mean([vs[c]["queries"][q]["total_ms_median"] for q in (present, absent)]))
    v = vs[a.vllm]
    vname = {"bf16": "vLLM bf16", "fp8mtp": "vLLM FP8 + MTP"}[a.vllm]
    sprite = Image.open(os.path.join(D, "search_sprite.jpg")).convert("RGB")
    st = sprite.width // 16
    tiles = [sprite.crop(((k % 16) * st, (k // 16) * st, (k % 16 + 1) * st, (k // 16 + 1) * st)).resize((TILE, TILE), Image.LANCZOS)
             for k in range(256)]
    grid_bright = Image.new("RGB", (16 * (TILE + GAP), 16 * (TILE + GAP)), BG)
    grid_dim = grid_bright.copy()
    for k, t in enumerate(tiles):
        xy = ((k % 16) * (TILE + GAP), (k // 16) * (TILE + GAP))
        grid_bright.paste(t, xy)
        grid_dim.paste(Image.blend(t, Image.new("RGB", t.size, BG), 0.78), xy)

    f_title, f_hdr, f_small, f_big, f_mid = font("bold", 22), font("bold", 16), font("sans", 14), font("mono", 26), font("bold", 18)
    RX = GX + 16 * (TILE + GAP) + 24
    RW = W - 20 - RX

    def beat(q):
        gq = g["queries"][q]
        gc = median_call(gq["calls"])
        vq = v["queries"][q]
        vc = median_call(vq["calls"])
        sel = gq["selected"]
        larges = []
        for k in sel[:4]:
            rid = pick["bank"][k]["id"]
            im = Image.open(os.path.join(D, "search_large", f"{rid}.jpg")).convert("RGB")
            larges.append((k, im, gq["box"][k], gq["p_present"][k]))
        return dict(q=q, g_t=float(gc["total"]), g_med=gq["total_ms_median"], v_t=np.sort(vc["answers_at_ms"]),
                    v_total=float(vc["total"]), v_med=vq["total_ms_median"], sel=sel, none=gq["none_conformal"],
                    larges=larges, truth=gq["truth_present"], p_max=max(gq["p_present"]), v_hits=sum(x["box"] is not None for x in vq["answers"]))

    beats = [beat(present), beat(absent)]

    def frame(b, t, idx, final=False):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((20, 16), "One question, 256 images", font=f_title, fill=TEXT)
        d.text((20, 46), "Both have seen every image before (Glance: 121 ms and 75 MB per image, once). Only the new question is timed.", font=f_small, fill=MUTED)
        clock = f"t = {min(t, b['v_total']) / 1e3:5.2f} s"
        d.text((W - 20 - text_w(d, clock, f_big), 12), clock, font=f_big, fill=TEXT)
        sm = "REAL TIME"
        fb = font("bold", 15)
        pill(d, (W - 20 - text_w(d, sm, fb) - 16, 44), sm, (240, 246, 252), fg=BG, f=fb)
        x = pill(d, (GX, 72), f"question {idx + 1} of 2", PANEL, fg=MUTED, f=f_small)
        d.text((x + 10, 72), f"Where is {b['q']}?", font=f_hdr, fill=TEXT)
        g_done = t >= b["g_t"]
        im.paste(grid_dim if g_done else grid_bright, (GX, GY))
        if g_done:
            for k in b["sel"]:
                x0, y0 = GX + (k % 16) * (TILE + GAP), GY + (k // 16) * (TILE + GAP)
                im.paste(tiles[k], (x0, y0))
                d.rectangle([x0 - 2, y0 - 2, x0 + TILE + 1, y0 + TILE + 1], outline=GLANCE, width=2)
        # lanes
        y = GY + 6
        for name, color, done_frac, label in (
                ("Glance  ·  bf16, cached scan", GLANCE, 1.0 if g_done else 0.0,
                 f"256/256 · {b['g_t']:.0f} ms" if g_done else "0/256"),
                (f"{vname}  ·  prefix-cached", VLLM, float((b["v_t"] <= t).mean()),
                 f"{int((b['v_t'] <= t).sum())}/256" + (f" · {b['v_total'] / 1e3:.2f} s · returned boxes for {b['v_hits']} image{'s' if b['v_hits'] != 1 else ''}"
                                                         if t >= b["v_total"] else ""))):
            d.text((RX, y), name, font=f_hdr, fill=color)
            d.rounded_rectangle([RX, y + 26, RX + RW, y + 40], radius=7, fill=PANEL)
            if done_frac > 0:
                d.rounded_rectangle([RX, y + 26, RX + max(14, RW * done_frac), y + 40], radius=7, fill=color)
            d.text((RX + RW - text_w(d, label, f_small), y + 45), label, font=f_small, fill=TEXT)
            y += 72
        # verdict card
        cy = y + 6
        d.rounded_rectangle([RX, cy, RX + RW, cy + 236], radius=10, fill=PANEL, outline=GLANCE if g_done else LINE)
        if not g_done:
            d.text((RX + 16, cy + 16), "Glance answer: waiting…", font=f_hdr, fill=MUTED)
        elif b["sel"]:
            d.text((RX + 16, cy + 14), f"Found in {len(b['sel'])} of 256 images", font=f_mid, fill=GLANCE)
            n = len(b["larges"])
            tw = (RW - 32 - (n - 1) * 8) // max(n, 1)
            for j, (k, big, box, p) in enumerate(b["larges"]):
                th = int(big.height * tw / big.width)
                thumb = big.resize((tw, th), Image.LANCZOS)
                x0, y0 = RX + 16 + j * (tw + 8), cy + 48
                im.paste(thumb, (x0, y0))
                bx = [x0 + box[0] * tw, y0 + box[1] * th, x0 + box[2] * tw, y0 + box[3] * th]
                d.rectangle(bx, outline=GLANCE, width=2)
                d.text((x0, y0 + th + 4), f"P = {p:.2f}", font=f_small, fill=TEXT)
            d.text((RX + 16, cy + 184), "Threshold set on 34 other absent-object queries:", font=f_small, fill=MUTED)
            d.text((RX + 16, cy + 202), "≤ 10% false “found”, on average over queries.", font=f_small, fill=MUTED)
        else:
            d.text((RX + 16, cy + 14), "In none of the 256 images", font=f_mid, fill=GLANCE)
            d.text((RX + 16, cy + 48), "No image passes the calibrated", font=f_small, fill=TEXT)
            d.text((RX + 16, cy + 66), "threshold (max-statistic, α = 0.1).", font=f_small, fill=TEXT)
            d.text((RX + 16, cy + 94), f"Highest P(present) in any image: {b['p_max']:.2f}", font=f_small, fill=TEXT)
            d.text((RX + 16, cy + 184), "Threshold set on 34 other absent-object queries:", font=f_small, fill=MUTED)
            d.text((RX + 16, cy + 202), "≤ 10% false “found”, on average over queries.", font=f_small, fill=MUTED)
        if final:
            msg = f"{b['v_med'] / b['g_med']:.0f}× faster: {b['g_med']:.0f} ms vs {b['v_med'] / 1e3:.2f} s (medians)"
            d.text((RX, cy + 246), msg, font=font("bold", 24), fill=TEXT)
        cap1 = "Same GPU (RTX PRO 6000), same 256 COCO val2017 images (≤640 visual tokens), question text to all answers on host."
        cap2 = (f"Glance-4B bf16 (one forward over cached image states) vs Qwen3.5-4B on vLLM 0.34 "
                f"{'bf16' if a.vllm == 'bf16' else 'FP8 + MTP'} (256 requests). CC BY 2.0.")
        d.text((20, H - 44), cap1, font=f_small, fill=MUTED)
        d.text((20, H - 24), cap2, font=f_small, fill=MUTED)
        d.rectangle([0, 0, W - 1, H - 1], outline=LINE, width=1)
        return im

    frames, durs = [], []
    for i, b in enumerate(beats):
        frames.append(frame(b, 0.0, i)); durs.append(250)
        t = 0.0
        while t < b["v_total"]:
            t = min(b["v_total"], t + 40.0)
            frames.append(frame(b, t, i)); durs.append(40)
        frames.append(frame(b, b["v_total"], i, final=True)); durs.append(2000)
    size = save_gif(frames, durs, a.out, pal_frames=[frames[0], frames[-1], frames[len(frames) // 2 - 1]])
    print(a.out, f"{size / 1e6:.2f} MB", len(frames), "frames", f"{sum(durs) / 1e3:.1f} s", "vllm config", a.vllm,
          [(b["q"], round(b["g_t"], 1), round(b["v_total"]), b["sel"], b["truth"], b["none"]) for b in beats])


if __name__ == "__main__":
    main()
