"""Preprocessing for a broadcast question over already resized, equal-grid RGB frames."""
from functools import lru_cache
import os

import numpy as np
import torch
from PIL import Image

from dv.data.mixed import collate_mixed


def frame_layout(example, n):
    """Broadcast the reference collator's singleton metadata, preserving its positions and labels."""
    batch = {k: example[k].expand(n, -1) for k in
             ('p_ids', 'p_valid', 'p_img_mask', 'q_ids', 'q_valid', 'vis_valid', 'lab_idx', 'grid_thw')}
    batch['centers'] = example['centers'].expand(n, -1, -1)
    for k in ('q_pos', 'p_pos'):
        batch[k] = example[k].expand(-1, n, -1)
    batch['q_img'] = torch.arange(n)
    return batch


def upload_frames(images):
    w, h = images[0].size
    host = torch.empty((len(images), h, w, 3), dtype=torch.uint8, pin_memory=True)
    dest = host.numpy()
    for i, im in enumerate(images):
        dest[i] = np.asarray(im)
    return host


def patch_pixels(raw, lut, processor):
    # Index the processor's exact FP32 value for each of the 256 possible channel values.
    channels = raw.permute(0, 3, 1, 2)
    normalized = torch.stack([lut[c][channels[:, c].long()] for c in range(3)], dim=1)
    patches, _, _ = processor.patchify(normalized, processor.patch_size, processor.merge_size,
                                       processor.temporal_patch_size)
    return patches.flatten(0, 1)


class FramePrep:
    def __init__(self, engine, *, device=None):
        self.tok = engine.tok
        self.device = engine.dev if device is None else device
        self.processor = p = engine.proc.image_processor
        self.fused = (self.device.type == "cuda" and getattr(p, "patch_size", None) == 16
                      and getattr(p, "merge_size", None) == 2 and getattr(p, "temporal_patch_size", None) == 2
                      and os.getenv("GLANCE_FRAME_OPT", "1") != "0")
        self.example = lru_cache(maxsize=16)(self._example)
        self.normalization_table = lru_cache(maxsize=1)(self._normalization_table)

    def supports(self, images):
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        except ImportError:
            return False
        p = self.processor
        if type(p) is not Qwen2VLImageProcessor or not hasattr(p, "patchify"):
            return False
        if len(images) == 0 or not isinstance(images[0], Image.Image):
            return False
        w, h = images[0].size
        step = p.patch_size * p.merge_size
        return (p.merge_size == 2 and w % step == h % step == 0
                and all(isinstance(im, Image.Image) and im.mode == 'RGB' and im.size == (w, h) for im in images))

    def template(self, size, q, qf):
        p = self.processor.patch_size
        grid = (1, size[1] // p, size[0] // p)
        options = tuple(map(str, q['options'])) if q['type'] in ('choice', 'score') else ()
        return self.example(grid, q['type'], q.get('text'), q.get('instr'), q.get('block'),
                            options, qf)

    def _example(self, grid, kind, text, instr, block, options, qf):
        from scan.engine import image_item
        question = dict(type=kind, text=text, instr=instr, block=block, options=options)
        item = image_item(self.tok, torch.tensor(grid), None, [question], qf)
        return collate_mixed([item], self.tok, pad_to=1)

    def _normalization_table(self, rescale, factor, normalize, mean, std):
        ramp = torch.arange(256, dtype=torch.uint8).expand(1, 3, 1, 256).contiguous()
        lut = self.processor.rescale_and_normalize(ramp, rescale, factor, normalize, mean, std)
        return lut.reshape(3, 256).to(self.device)

    def mixed_pixels(self, images, dtype=None):
        """Upload the original RGB bytes once, preserving each image's patch offset and width."""
        from speed.frame_pixels import patches
        counts = [im.width * im.height // 256 for im in images]
        host = torch.empty((sum(counts) * 256, 3), dtype=torch.uint8, pin_memory=True)
        dest = host.numpy()
        rows, start = [], 0
        for im, count in zip(images, counts):
            rows.append((start, count, im.width))
            dest[start * 256:(start + count) * 256] = np.asarray(im).reshape(-1, 3)
            start += count
        layout = torch.tensor(rows, dtype=torch.int32, device=self.device)
        return patches(host.to(self.device, non_blocking=True), self._lut(), layout,
                       count=start, maximum=max(counts), dtype=dtype)

    def _lut(self):
        p = self.processor
        mean = tuple(p.image_mean) if isinstance(p.image_mean, list) else p.image_mean
        std = tuple(p.image_std) if isinstance(p.image_std, list) else p.image_std
        return self.normalization_table(p.do_rescale, p.rescale_factor, p.do_normalize, mean, std)

    def pixels(self, host, dtype=None):
        raw = host.to(self.device, non_blocking=True)
        if self.fused:
            from speed.frame_pixels import patches
            return patches(raw, self._lut(), dtype=dtype)
        return patch_pixels(raw, self._lut(), self.processor)
