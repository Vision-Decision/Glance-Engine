"""Batched BF16 image decisions. All model and cache mutations run on one device thread."""

import os
import struct
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from itertools import groupby
from time import perf_counter
from typing import NamedTuple

import torch
import torch.nn.functional as F
from dv.data.mixed import collate_mixed

from scan.engine import image_item
from scan.varlen import FreshPack, SharedPrefixes, packed_forward
from scan.vision_prep import frame_layout, upload_frames
from speed.engine import prepare


@dataclass(frozen=True)
class ImageRef:
    digest: bytes


@dataclass
class Decision:
    images: list
    questions: list
    cache: bool = False
    image_ids: list = field(init=False, default_factory=list)

    batch_kind = "image"

    @property
    def prefer_immediate(self):
        return (len(self.images) == 1 and (self.cache or isinstance(self.images[0], ImageRef))
                and os.getenv("GLANCE_FRAME_OPT", "1") != "0")

    @property
    def tokens(self):
        # Admission estimate. The worker cuts again using actual tokenized block lengths.
        q = sum(
            16 + len(x.get("text", x.get("instr", ""))) // 3 + sum(len(s) // 3 + 3 for s in x.get("options", ()))
            for x in self.questions
        )
        return sum(q if isinstance(im, ImageRef) else im.width * im.height // 1024 + 5 + q for im in self.images)


class PrefixCache:
    """Device-owned byte-bounded LRU, shared by image prefixes and named scan banks."""

    def __init__(self, budget):
        if budget < 0:
            raise ValueError("negative cache budget")
        self.budget = budget
        self.entries = OrderedDict()
        self.used = 0
        self.last_rgb = None

    def clear(self):
        self.last_rgb = None
        self.entries.clear()
        self.used = 0

    def key_for_image(self, im):
        # A new decode or an in-place pixel edit must always be checked. Keep one
        # bounded byte snapshot to avoid hashing a recently resent RGB image again.
        raw = im.tobytes()
        if self.last_rgb is not None:
            size, previous, key = self.last_rgb
            if size == im.size and raw == previous:
                return key
        key = image_key(im, raw)
        if len(raw) <= min(8 << 20, self.budget):
            self.last_rgb = (im.size, raw, key)
        return key

    def get(self, key):
        found = self.entries.get(key)
        if found is not None:
            self.entries.move_to_end(key)
            return found[0]

    def put(self, key, value, size):
        if size > self.budget:
            return False
        if key in self.entries:
            self.used -= self.entries.pop(key)[1]
        while self.used + size > self.budget:
            self.used -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = (value, size)
        self.used += size
        return True


def image_key(im, raw=None):
    digest = sha256(struct.pack("!II", *im.size))
    digest.update(im.tobytes() if raw is None else raw)
    return ("RGB", digest.digest())


@dataclass
class Prefix:
    layers: list
    vis: torch.Tensor
    size: tuple
    spatial: dict = field(default_factory=dict)
    q_start: int = 0

    @classmethod
    def extract(cls, layers, vis, i, plen, nv, types, size, padded=False):
        # Clone compact slices. A view would keep the entire batched graph or projection alive.
        values = []
        for (a, b), kind in zip(layers, types):
            pair = (a[i], b[i]) if kind == "linear_attention" or padded else (a[i, :, -plen:], b[i, :, -plen:])
            values.append(tuple(t.clone(memory_format=torch.contiguous_format) for t in pair))
        return cls(values, (vis[i] if padded else vis[i, :nv]).clone(memory_format=torch.contiguous_format), size)

    def nbytes(self):
        return self.vis.nbytes + sum(t.nbytes for layer in self.layers for t in layer) + sum(t.nbytes for t in self.spatial.values())

    @cached_property
    def singleton(self):
        return [(a.unsqueeze(0), b.unsqueeze(0)) for a, b in self.layers], self.vis.unsqueeze(0)

    @staticmethod
    def assemble(prefixes, host, types):
        layers = []
        for i, kind in enumerate(types):
            pair = []
            for component in range(2):
                tensors = [p.layers[i][component] for p in prefixes]
                if kind != "linear_attention":
                    tensors = [
                        F.pad(t, (0, 0, host["Lp"] - t.shape[-2], 0)) if host["Lp"] != t.shape[-2] else t
                        for t in tensors
                    ]
                pair.append(torch.stack(tensors) if len(tensors) > 1 else tensors[0][None])
            layers.append(tuple(pair))
        vis = [
            F.pad(p.vis, (0, 0, 0, host["Nv"] - len(p.vis))) if host["Nv"] != len(p.vis) else p.vis
            for p in prefixes
        ]
        return layers, vis[0][None] if len(vis) == 1 else torch.stack(vis)


class Unit(NamedTuple):
    request_index: int
    image_index: int
    image: object
    item: dict
    key: tuple | None
    prefix: Prefix | None


class DecisionWorker:
    def __init__(self, engine, max_tokens=8192, cache_bytes=8 << 30, packed_shared=True, group_vision=True):
        self.engine, self.max_tokens = engine, max_tokens
        self.packed_shared = packed_shared
        self.group_vision = group_vision
        self.cache = PrefixCache(cache_bytes)
        self.reset_timings()
        self.video = None
        engine.graph_limit = 64
        for tower in (engine.fv, engine.frame_tower):
            if tower is not None:
                tower.graph_limit = 32

    @classmethod
    def load(cls, ckpt, *, graphs=True, **kw):
        from scan.pilot_charades import load_glance

        eng, _ = load_glance(ckpt)
        eng.graphs = graphs
        for tower in (eng.fv, eng.frame_tower):
            if tower is not None:
                tower.graphs = graphs
        return cls(eng, **kw)

    def reset_timings(self):
        self.timings = Counter()

    @staticmethod
    def is_oom(exc):
        return isinstance(exc, torch.OutOfMemoryError)

    def recover(self):
        self.engine.free_graphs()

    @torch.no_grad()
    def __call__(self, requests):
        if requests[0].batch_kind != "image":
            return [self.video(r) for r in requests]
        eng = self.engine
        results = [[[] for _ in r.images] for r in requests]
        groups = defaultdict(list)
        started = perf_counter()
        patch = eng.proc.image_processor.patch_size
        for i, req in enumerate(requests):
            req.image_ids = [None] * len(req.images)
            items_by_size = {}
            for j, im in enumerate(req.images):
                ref = isinstance(im, ImageRef)
                key = ("RGB", im.digest) if ref else None
                if not ref and req.cache:
                    key = self.cache.key_for_image(im) if req.prefer_immediate else image_key(im)
                prefix = self.cache.get(key) if key else None
                if ref and prefix is None:
                    raise KeyError("unknown or evicted image prefix")
                size = prefix.size if ref else im.size
                if key:
                    req.image_ids[j] = key[1].hex()
                if size not in items_by_size:
                    grid = torch.tensor([1, size[1] // patch, size[0] // patch])
                    items_by_size[size] = image_item(eng.tok, grid, size, req.questions)
                item = items_by_size[size]
                # A neighboring multi-question client must not switch an independent image
                # from its contiguous one-shot prefill to a state-producing branch prefill.
                hit = prefix is not None
                independent = not hit and key is None and len(req.questions) == 1
                unit = Unit(i, j, im, item, key, prefix)
                room = self.max_tokens - (0 if hit else eng._plen(item))
                lengths = [len(ids) for ids in item["q_ids"]]
                if max(lengths) > room:
                    raise ValueError("one image and one question exceed the token budget")
                # A retained recurrent state can outlive the other requests in this step.
                # Produce it with a stable singleton layout on cache insertion.
                retain = not hit and key is not None
                if sum(lengths) <= room:
                    groups[(hit, independent, retain)].append(unit)
                else:
                    start, used = 0, 0
                    for end, length in enumerate(lengths):
                        if used + length > room:
                            partial = image_item(eng.tok, item["grid_thw"], size, req.questions[start:end])
                            groups[(hit, False, retain)].append(unit._replace(item=partial))
                            start, used = end, 0
                        used += length
                    partial = image_item(eng.tok, item["grid_thw"], size, req.questions[start:])
                    groups[(hit, False, retain)].append(unit._replace(item=partial))
        self.timings["prep"] += (perf_counter() - started) * 1000
        for (hit, _, retain), units in groups.items():
            if retain:
                for unit in units:
                    self._step([unit], False, results)
                continue
            step, tokens = [], 0
            for unit in units:
                item = unit.item
                cost = sum(map(len, item["q_ids"])) + (0 if hit else eng._plen(item))
                if step and tokens + cost > self.max_tokens:
                    self._step(step, hit, results)
                    step, tokens = [], 0
                step.append(unit)
                tokens += cost
            if step:
                self._step(step, hit, results)
        for req in requests:
            if req.cache:
                req.image_ids = [
                    s if s and ("RGB", bytes.fromhex(s)) in self.cache.entries else None for s in req.image_ids
                ]
        return results

    def _pixels(self, units):
        eng = self.engine
        images = [u.image for u in units]
        if eng.frame_prep is None or not all(eng.frame_prep.supports([im]) for im in images):
            b = eng.proc.image_processor(images=images, do_resize=False, return_tensors="pt")
            return b["pixel_values"].to(eng.dev)
        if eng.frame_prep.fused and any(im.size != images[0].size for im in images):
            return eng.frame_prep.mixed_pixels(images, dtype=eng.dtype)
        # Consecutive equal grids can share the same patchify launch.
        pixels = [
            eng.frame_prep.pixels(upload_frames(list(group)), dtype=eng.dtype)
            for _, group in groupby(images, key=lambda im: im.size)
        ]
        return pixels[0] if len(pixels) == 1 else torch.cat(pixels)

    def _vision(self, units, grid):
        eng = self.engine
        if not self.group_vision or len(units) == 1:
            return eng.fresh_vision(self._pixels(units), grid)
        groups = defaultdict(list)
        for index, unit in enumerate(units):
            groups[unit.image.size].append(index)
        if len(groups) == 1:
            return eng.fresh_vision(self._pixels(units), grid)
        parts = [None] * len(units)
        for indices in groups.values():
            pixels = self._pixels([units[i] for i in indices])
            group = eng.fresh_vision(pixels, grid[indices])
            # Vision graphs may reuse the same pool across different grids.
            views = group.clone().chunk(len(indices))
            for i, feature in zip(indices, views):
                parts[i] = feature
        return torch.cat(parts)

    def _step(self, units, hit, results):
        try:
            self._forward(units, hit, results)
            return
        except torch.OutOfMemoryError:
            if len(units) == 1:
                raise
        self.recover()
        middle = len(units) // 2
        self._step(units[:middle], hit, results)
        self._step(units[middle:], hit, results)

    def _singleton_inputs(self, unit):
        prefix, item = unit.prefix, unit.item
        ids, lab = item["q_ids"][0], item["labs"][0]
        E, host = self.engine.question_inputs(ids, lab, item["qtype"][0], 1, True)
        E.update(prefix.spatial)
        E["q_pos"] = (prefix.q_start + E.pop("q_off")).view(1, 1, -1).expand(3, -1, -1)
        E["q_img"] = torch.zeros(1, dtype=torch.long)
        E["lab_idx"] = F.pad(E["lab_idx"], (0, 26 - E["lab_idx"].shape[1]), value=-1)
        E["q_offsets"] = torch.tensor([0, len(ids)], dtype=torch.int32)
        host.update(B=1, Lp=E["p_valid"].shape[1], Nv=E["vis_valid"].shape[1],
                    nopad=self.engine._plen(item) == E["p_valid"].shape[1], n_qtok=len(ids))
        return E, host

    def _forward(self, units, hit, results):
        eng = self.engine
        start = perf_counter()
        items = [u.item for u in units]
        packed = not hit and all(len(u.item["qs"]) == 1 and u.key is None for u in units)
        single_hit = (hit and len(units) == 1 and len(items[0]["qs"]) == 1
                      and units[0].prefix.spatial and os.getenv("GLANCE_FRAME_OPT", "1") != "0")
        if single_hit:
            E, host = self._singleton_inputs(units[0])
        elif packed and len(items) > 1 and all(it is items[0] for it in items) and eng.frame_prep is not None:
            template = eng.frame_prep.template(units[0].image.size, items[0]["qs"][0], None)
            batch = frame_layout(template, len(items))
        else:
            batch = collate_mixed(items, eng.tok, pad_to=1)
        if not packed and not single_hit:
            nv = batch["vis_valid"].shape[1]
            E, host = prepare(batch, Nv=((nv + 63) // 64) * 64)
        t1 = perf_counter()
        if not hit:
            feats = self._vision(units, batch["grid_thw"])
            if eng.dev.type == "cuda":
                torch.cuda.synchronize(eng.dev)
        t2 = perf_counter()
        if packed:
            out = packed_forward(eng, FreshPack(batch, eng.dev), feats, use_graph=eng.graphs and eng.lm_prefill)
        elif hit:
            layers, vis = (units[0].prefix.singleton if single_hit else
                           Prefix.assemble([u.prefix for u in units], host, eng.types))
            inputs = {k: v for k, v in E.items() if k != "img_flat"}
            if single_hit:
                out = eng.run_q(inputs, host, layers, vis, host_inputs=True)
            else:
                out = eng.run_q({k: v.to(eng.dev) for k, v in inputs.items()}, host, layers, vis)
        else:
            inputs = eng.to_device(E, host, feats)
            if self.packed_shared and eng.lm_prefill and eng.question_branch is not None and len(units) > 1:
                taps = next(
                    layer.linear_attn.conv_kernel_size
                    for layer, kind in zip(eng.layers, eng.types)
                    if kind == "linear_attention"
                )
                out, layers, vis = SharedPrefixes(batch, host, eng.dev, taps).forward(eng, inputs, host)
            else:
                out, layers, vis = eng.run(inputs, host)
            for i, (_, _, _, item, key, _) in enumerate(units):
                if key is not None:
                    prefix = Prefix.extract(
                        layers,
                        vis,
                        i,
                        eng._plen(item),
                        int(batch["vis_valid"][i].sum()),
                        eng.types,
                        item["orig_wh"],
                        padded=len(units) == 1 and os.getenv("GLANCE_FRAME_OPT", "1") != "0",
                    )
                    if len(units) == 1 and os.getenv("GLANCE_FRAME_OPT", "1") != "0":
                        prefix.spatial = {k: inputs[k] for k in ("p_valid", "vis_valid", "centers")}
                        prefix.q_start = int(batch["q_pos"][0, 0, 0])
                        # Construct the batch-dimension views once at insertion.
                        _ = prefix.singleton
                    self.cache.put(key, prefix, prefix.nbytes())
        host_out = {k: v.float().cpu() for k, v in out.items()}
        presence = host_out["presence_logit"].sigmoid().tolist()
        iou = host_out["iou_logit"].sigmoid().tolist()
        boxes = host_out["box_pred"].tolist()
        probabilities = host_out["typed_logits"].softmax(-1).tolist()
        offset = 0
        for i, j, _, item, _, _ in units:
            row = []
            for q, labels in zip(item["qs"], item["labs"]):
                if q["type"] == "box":
                    row.append({"p_present": presence[offset], "box": boxes[offset], "p_iou50": iou[offset]})
                else:
                    row.append({"probs": probabilities[offset][: len(labels)]})
                offset += 1
            results[i][j].extend(row)
        self.timings.update(
            prep=(t1 - start) * 1000, vision=(t2 - t1) * 1000, lm=(perf_counter() - t2) * 1000, steps=1
        )
