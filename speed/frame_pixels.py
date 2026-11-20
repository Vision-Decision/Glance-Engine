"""Patchify spatially merged RGB frames directly from compact device bytes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _patches(
    RGB,
    Lut,
    Layout,
    Out,
    WIDTH: tl.constexpr,
    COUNT: tl.constexpr,
    MIXED: tl.constexpr,
    WIDE: tl.constexpr,
):
    patch, image = tl.program_id(0), tl.program_id(1)
    if WIDE:
        patch = patch.to(tl.int64)
        image = image.to(tl.int64)
    if MIXED:
        start = tl.load(Layout + image * 3)
        count = tl.load(Layout + image * 3 + 1)
        width = tl.load(Layout + image * 3 + 2)
    else:
        start, count, width = image * COUNT, COUNT, WIDTH
    if WIDE:
        start = start.to(tl.int64)
    if patch < count:
        # HF merge order is tile-y, tile-x, then the four spatial patches of a tile.
        tile = patch // 4
        y = (tile // (width // 32)) * 32 + ((patch % 4) // 2) * 16
        x = (tile % (width // 32)) * 32 + (patch % 2) * 16
        i = tl.arange(0, 1024)
        channel, pixel = i // 256, i % 256
        offset = start * 256 + (y + pixel // 16) * width + x + pixel % 16
        value = tl.load(RGB + offset * 3 + channel, i < 768, 0).to(tl.int32)
        value = tl.load(Lut + channel * 256 + value, i < 768, 0)
        dst = Out + (start + patch) * 1536 + channel * 512 + pixel
        tl.store(dst, value, i < 768)
        tl.store(dst + 256, value, i < 768)


def patches(raw, lut, layout=None, *, count=None, maximum=None, dtype=None):
    if layout is None:
        n, h, w, _ = raw.shape
        maximum = h * w // 256
        count = maximum * n
    else:
        n, w = len(layout), 0
    result = torch.empty((count, 1536), device=raw.device, dtype=dtype or lut.dtype)
    _patches[(maximum, n)](
        raw, lut, layout, result, w, maximum, layout is not None, count * 1536 >= 2**31
    )
    return result
