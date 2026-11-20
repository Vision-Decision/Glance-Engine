"""Shared drawing helpers for the README GIFs (PIL only)."""
import colorsys, os
from PIL import Image, ImageDraw, ImageFont

# A dark "screen" card: reads as a deliberate panel on GitHub's light and dark themes alike.
BG = (13, 17, 23)
PANEL = (22, 27, 34)
LINE = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
GLANCE = (63, 185, 80)     # green
VLLM = (210, 153, 34)      # amber
ACCENT = (88, 166, 255)

_FONT_CANDIDATES = {
    "sans": ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}
_cache = {}


def font(kind, size):
    key = (kind, size)
    if key not in _cache:
        for p in _FONT_CANDIDATES[kind]:
            if os.path.exists(p):
                idx = 1 if (kind == "bold" and p.endswith(".ttc")) else 0
                try:
                    _cache[key] = ImageFont.truetype(p, size, index=idx)
                    break
                except OSError:
                    continue
        else:
            _cache[key] = ImageFont.load_default(size)
    return _cache[key]


def palette(n, s=0.62, v=0.98):
    """n distinct, bright hues (golden-angle order so neighbours differ)."""
    out = []
    for k in range(n):
        h = (k * 0.618034) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def text_w(d, s, f):
    return d.textbbox((0, 0), s, font=f)[2]


def pill(d, xy, label, fill, fg=BG, f=None, pad=(8, 3)):
    f = f or font("bold", 13)
    x, y = xy
    w = text_w(d, label, f)
    h = f.size + 2 * pad[1] + 2
    d.rounded_rectangle([x, y, x + w + 2 * pad[0], y + h], radius=h // 2, fill=fill)
    d.text((x + pad[0], y + pad[1]), label, font=f, fill=fg)
    return x + w + 2 * pad[0]


def save_gif(frames, durations, path, colors=255, pal_frames=None, dither=True):
    """frames: list of RGB PIL images; durations in ms (GIF uses 10 ms units; >= 20 ms renders reliably).
    One global palette, built from pal_frames (default: first and last frame side by side)."""
    src = pal_frames or [frames[0], frames[-1]]
    w, h = src[0].size
    sheet = Image.new("RGB", (w * len(src), h))
    for k, f in enumerate(src):
        sheet.paste(f, (k * w, 0))
    pal_src = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal_src, dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    return os.path.getsize(path)
