"""The Glance-X engine: image prefixes computed once, request blocks answered against them, in three directions.

``image_first`` is split the way the v1 engine splits a prefill, which makes it cacheable:

* ``encode(images)``: vision tower, the image stream through the modality stage (layers 1..K) and the image prefix
  through the fusion stage (layers K+1..N). Per image it keeps an :class:`XPrefix`: every fusion layer's Gated
  DeltaNet state and conv tail, every attention layer's post-RoPE keys and values, and the final image tokens. Left
  padding is trimmed, so a prefix does not depend on the batch it was encoded in.
* ``answer(prefixes, rows)``: any number of request blocks against any of those prefixes in one forward. Each
  distinct request runs the modality stage once and is shared by every image it is asked about; the fusion stage
  then continues each block from its own image's state (glancex's ``gdn_block`` / ``attn_block``, which equal a
  plain forward of ``[image, request]``).

A fresh ``image_first`` question is ``answer(encode(images), rows)``: the same arithmetic as glancex's
``GlanceXBackbone.forward(..., "image_first")``, which ``tests/test_x_engine.py`` checks. ``text_first`` and ``joint``
cannot be cached (the image tokens read the request), so ``pairs`` runs glancex's own forward for them.
"""

from __future__ import annotations

import contextlib
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DIRECTIONS = ("image_first", "text_first", "joint")

_SHORT = {"image_first": "i2t", "text_first": "t2i", "joint": "bi"}


@dataclass
class XPrefix:
    """One encoded image (``image_first``). ``states``: per fusion layer, (S, conv tail) or (K, V) without padding."""

    states: list
    vis: torch.Tensor        # [n, D] final image tokens
    centers: torch.Tensor    # [n, 2] patch centres in 0..1 (x, y)
    size: tuple              # (w, h) of the encoded (resized) image
    length: int              # valid prefix tokens: <vision_start> + n patches + <vision_end>
    spatial: dict = field(default_factory=dict)  # unused; keeps the PrefixCache entry shape of v1

    def nbytes(self):
        return self.vis.nbytes + self.centers.nbytes + sum(t.nbytes for s in self.states for t in s)


@dataclass
class XBank:
    """Prefixes stacked into glancex's batched bank layout, ready for ``answer`` (left-padded keys, padded tokens)."""

    states: list
    p_valid: torch.Tensor
    vis: torch.Tensor
    vis_valid: torch.Tensor
    centers: torch.Tensor

    def __len__(self):
        return self.p_valid.shape[0]


class XEngine:
    family = "x"
    # v1 engine attributes that callers probe; Glance-X has no frame tower or clip path.
    fv = frame_tower = frame_prep = None
    clip_max_vis = None

    def __init__(self, net, heads, proc, cfg, device, source="", gpu_prep=None):
        from glancex.data import Tok

        self.net, self.heads, self.proc, self.cfg = net, heads, proc, cfg
        self.tok = Tok(proc, cfg.max_q_tokens)
        self.dev = torch.device(device)
        self.dtype = next(net.lm.parameters()).dtype
        self.source = source
        self.layers = list(zip(net.lm.layers, net.layer_types))
        self.K = cfg.bidir_k
        self.max_vis = cfg.max_vis_tokens
        weights = [float(x) for x in cfg.mode_probs.split(",")]
        self.directions = tuple(d for d, w in zip(DIRECTIONS, weights) if w > 0)
        self.graphs = False   # CUDA-graph replay of encode / answer (set by the loader; never on the exact path)
        self.graph_limit = 64
        self._graph_cache = None
        self.obj_thr = 0.5
        # Normalise and patchify on the GPU (glancex's fast path). The exact path keeps the CPU image processor.
        self.gpu_prep = self.dev.type == "cuda" if gpu_prep is None else gpu_prep
        self._pool = None

    # ---------------------------------------------------------------------------------------- helpers
    def direction(self, d):
        """Canonical direction name; refuses directions the checkpoint was not trained in."""
        d = {"i2t": "image_first", "t2i": "text_first", "bi": "joint"}.get(d, d)
        if d not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}; got {d!r}")
        if d not in self.directions:
            raise ValueError(f"this Glance-X checkpoint was not trained in the {d!r} direction "
                             f"(trained: {', '.join(self.directions)})")
        return d

    def _autocast(self):
        if self.dev.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def fit(self, image):
        """RGB image at most ``max_vis`` tokens with sides multiple of 32 (resized as in training if needed)."""
        from glancex.data import resize_dims

        image = image.convert("RGB")
        w, h = image.size
        if w % 32 == 0 and h % 32 == 0 and (w // 32) * (h // 32) <= self.max_vis:
            return image
        return image.resize(resize_dims(w, h, self.max_vis, self.cfg.max_upscale), Image.Resampling.BICUBIC)

    def map(self, fn, xs):
        """``fn`` over ``xs`` on worker threads (PIL resizing and pinned uploads release the GIL)."""
        xs = list(xs)
        if len(xs) < 4:
            return [fn(x) for x in xs]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))
        return list(self._pool.map(fn, xs))

    def fit_many(self, images, max_vis=None):
        """``fit`` for many images on worker threads, optionally to fewer than ``max_vis`` tokens."""
        if max_vis is None or max_vis >= self.max_vis:
            return self.map(self.fit, images)
        from glancex.data import resize_dims

        def one(im):
            im = im.convert("RGB")
            size = resize_dims(*im.size, max_vis, self.cfg.max_upscale)
            return im if size == im.size else im.resize(size, Image.Resampling.BICUBIC)

        return self.map(one, images)

    def text_ids(self, text):
        return self.tok.text(text)

    def _items(self, images, requests):
        """glancex collate items for already fitted images. ``requests``: per image, a list of request strings."""
        if self.gpu_prep:
            ts = self.map(lambda im: torch.from_numpy(np.asarray(im)).permute(2, 0, 1).contiguous().pin_memory()
                          .to(self.dev, non_blocking=True), images)
            out = self.proc.image_processor(images=ts, do_resize=False, return_tensors="pt", device=self.dev)
            out["pixel_values"] = out["pixel_values"].to(torch.bfloat16)
            out["image_grid_thw"] = out["image_grid_thw"].cpu()
        else:
            out = self.proc.image_processor(images=list(images), do_resize=False, return_tensors="pt")
        grids = out["image_grid_thw"]
        chunks = out["pixel_values"].split(grids.prod(-1).tolist())
        items = []
        for im, px, grid, rq in zip(images, chunks, grids, requests):
            qs = [dict(text=t, present=True) for t in rq]
            items.append(dict(pixel_values=px, grid_thw=grid, qs=qs, q_ids=[self.text_ids(t) for t in rq],
                              present=[True] * len(qs), boxes=[[] for _ in qs], rec={}, orig_wh=im.size))
        return items

    def _to_dev(self, batch):
        return {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ---------------------------------------------------------------------------------------- image_first
    @torch.no_grad()
    def encode(self, images) -> list:
        """Vision tower + image prefix for fitted images. Returns one :class:`XPrefix` per image."""
        from glancex.data import collate

        if not images:
            return []
        B = len(images)
        items = self._items(images, [["x"]] * B)  # collate needs one request per image; it is unused
        n_img = [int(it["grid_thw"].prod()) // 4 for it in items]
        graph = self._use_graphs() and _bucket(B, _B_BUCKETS) is not None
        cfg = self.cfg.update(dict(pad_img_to=_round(max(n_img) + 2, 64))) if graph else self.cfg
        b = collate(items, self.tok, cfg, "i2t")
        b.pop("meta")
        b = self._to_dev(b)
        feats = self.net.image_features(b["pixel_values"], b["grid_thw"])
        lengths = b["img_valid"].sum(1).tolist()
        names = ("img_ids", "img_valid", "img_mask", "img_pos", "vis_valid")
        if graph:
            Bb, Nb = _bucket(B, _B_BUCKETS), _round(max(n_img), 64)
            rows = list(range(B)) + [B - 1] * (Bb - B)          # padding rows repeat the last image
            idx = torch.tensor(rows, device=self.dev)
            inputs = [b[k][:, idx] if k == "img_pos" else b[k][idx] for k in names]
            inputs[4] = F.pad(inputs[4], (0, Nb - inputs[4].shape[1]))
            flat = torch.cat([feats[i] for i in rows])
            inputs.append(F.pad(flat, (0, 0, 0, Bb * Nb - len(flat))))
            outs = self._graphs(("encode", Bb, inputs[0].shape[1], Nb), self._encode_core, inputs)
        else:
            outs = self._encode_core(*[b[k] for k in names], torch.cat(list(feats)))
        *flat_states, vis = outs
        out = []
        for i, (im, L, n) in enumerate(zip(images, lengths, n_img)):
            states = []
            for li, (_, kind) in enumerate(self.layers[self.K:]):
                a, c = flat_states[2 * li], flat_states[2 * li + 1]
                pair = (a[i], c[i]) if kind == "linear_attention" else (a[i, :, -L:], c[i, :, -L:])
                states.append(tuple(t.clone(memory_format=torch.contiguous_format) for t in pair))
            out.append(XPrefix(states, vis[i, :n].clone(), b["centers"][i, :n].clone(), im.size, int(L)))
        return out

    def _encode_core(self, img_ids, img_valid, img_mask, img_pos, vis_valid, feats):
        """The image prefix through every layer (glancex ``encode_images``). Static shapes: CUDA-graph safe."""
        batch = dict(img_ids=img_ids, img_valid=img_valid, img_mask=img_mask, img_pos=img_pos, vis_valid=vis_valid)
        with self._autocast():
            d = self.net.encode_images(batch, feats)
        return [t for st in d["states"] for t in st] + [d["vis"]]

    def stack(self, prefixes, Lp=None, N=None) -> XBank:
        """Stack prefixes into the batched layout of glancex's image bank (left-padded keys, right-padded tokens)."""
        Lp = Lp or max(p.length for p in prefixes)
        N = N or max(len(p.vis) for p in prefixes)
        states = []
        for li, (_, kind) in enumerate(self.layers[self.K:]):
            pair = []
            for c in range(2):
                ts = [p.states[li][c] for p in prefixes]
                if kind != "linear_attention":
                    ts = [F.pad(t, (0, 0, Lp - t.shape[-2], 0)) if t.shape[-2] != Lp else t for t in ts]
                pair.append(torch.stack(ts))
            states.append(tuple(pair))
        p_valid = torch.zeros(len(prefixes), Lp, dtype=torch.bool, device=self.dev)
        vis_valid = torch.zeros(len(prefixes), N, dtype=torch.bool, device=self.dev)
        for i, p in enumerate(prefixes):
            p_valid[i, Lp - p.length:] = True
            vis_valid[i, : len(p.vis)] = True
        vis = torch.stack([F.pad(p.vis, (0, 0, 0, N - len(p.vis))) for p in prefixes])
        centers = torch.stack([F.pad(p.centers, (0, 0, 0, N - len(p.centers))) for p in prefixes])
        return XBank(states, p_valid, vis, vis_valid, centers)

    def _text_stream(self, ids_list, Lt=None):
        """LEFT-padded request streams from ``cfg.text_pos0`` (as glancex.data.collate)."""
        Lt = Lt or max(len(q) for q in ids_list)
        M = len(ids_list)
        txt = torch.full((M, Lt), self.tok.pad, dtype=torch.long)
        tv = torch.zeros(M, Lt, dtype=torch.bool)
        tpos = torch.zeros(3, M, Lt, dtype=torch.long)
        for m, q in enumerate(ids_list):
            txt[m, Lt - len(q):] = torch.tensor(q)
            tv[m, Lt - len(q):] = True
            tpos[:, m, Lt - len(q):] = self.cfg.text_pos0 + torch.arange(len(q))
        return txt.to(self.dev), tv.to(self.dev), tpos.to(self.dev)

    @torch.no_grad()
    def answer(self, prefixes, rows):
        """Request blocks against encoded images. ``prefixes``: a list of :class:`XPrefix` or a stacked
        :class:`XBank`. ``rows``: (prefix index, request ids). Returns one result per row.

        Each distinct request runs the modality stage once; the fusion stage runs every row against its own prefix.
        Small calls (at most 16 images and 256 rows) replay a CUDA graph per shape bucket; padding rows and images
        are masked or dropped, so the answers do not depend on the bucket.
        """
        uniq, row_text = {}, []
        for _, ids in rows:
            row_text.append(uniq.setdefault(tuple(ids), len(uniq)))
        texts = [list(k) for k in uniq]
        img = [i for i, _ in rows]
        M = len(rows)
        if (self._use_graphs() and not isinstance(prefixes, XBank) and _bucket(len(prefixes), _B_BUCKETS)
                and _bucket(M, _M_BUCKETS)):
            Bb, Mb, Ub = _bucket(len(prefixes), _B_BUCKETS), _bucket(M, _M_BUCKETS), _bucket(len(texts), _M_BUCKETS)
            Lp = _round(max(p.length for p in prefixes), 64)
            N = _round(max(len(p.vis) for p in prefixes), 64)
            bank = self.stack(list(prefixes) + [prefixes[0]] * (Bb - len(prefixes)), Lp, N)
            txt, tv, tpos = self._text_stream(texts + [texts[0]] * (Ub - len(texts)),
                                              _round(max(map(len, texts)), 8))
            sel = torch.tensor(row_text + [0] * (Mb - M), device=self.dev)
            q_img = torch.tensor(img + [0] * (Mb - M), device=self.dev)
            inputs = [txt, tv, tpos, sel, q_img, bank.p_valid, bank.vis, bank.vis_valid, bank.centers]
            inputs += [t for st in bank.states for t in st]
            outs = self._graphs(("answer", Bb, Lp, N, Ub, txt.shape[1], Mb), self._answer_core, inputs)
            return self._rows([t[:M] for t in outs])
        bank = prefixes if isinstance(prefixes, XBank) else self.stack(prefixes)
        txt, tv, tpos = self._text_stream(texts)
        sel = torch.tensor(row_text, device=self.dev)
        q_img = torch.tensor(img, device=self.dev)
        # row m asks prefix m (a scan): continue each block from the bank in place instead of gathering a copy
        identity = img == list(range(len(bank)))
        outs = self._answer_core(txt, tv, tpos, sel, q_img, bank.p_valid, bank.vis, bank.vis_valid, bank.centers,
                                 *[t for st in bank.states for t in st], identity=identity)
        return self._rows(outs)

    def _answer_core(self, txt, tv, tpos, sel, q_img, p_valid, vis, vis_valid, centers, *flat_states,
                     identity=False):
        """Modality stage per distinct request, fusion stage per row, then the heads. Static shapes, no host sync."""
        from glancex._layers import attn_block, block_mask, gdn_block
        from glancex.backbone import pack

        net = self.net
        gather = None if identity else q_img
        with self._autocast():
            xt = net.lm.embed_tokens(txt)
            if self.K:
                xt = net.unimodal(xt, tv, tpos, self.layers[: self.K])
            xq, qv, qpos, _, last = pack([(xt[sel], tv[sel], tpos[:, sel], None)], tv.shape[1])
            rope_q, mask_q = net.rope(xq, qpos), block_mask(p_valid[q_img], qv, self.cfg.bidir_block_mask)
            for li, (layer, kind) in enumerate(self.layers[self.K:]):
                st = (flat_states[2 * li], flat_states[2 * li + 1])
                h = layer.input_layernorm(xq)
                if kind == "linear_attention":
                    o = gdn_block(layer.linear_attn, h, st, gather)
                else:
                    o = attn_block(layer.self_attn, h, rope_q, mask_q, st, gather)
                xq = xq + o
                xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
            hq = net.lm.norm(xq)
        r = hq[torch.arange(hq.shape[0], device=hq.device), last]
        return self._heads(r, vis, vis_valid, centers, q_img)

    # ---------------------------------------------------------------------------------------- text_first / joint
    @torch.no_grad()
    def pairs(self, images, requests, direction):
        """Fresh (image, request) pairs in ``text_first`` or ``joint`` (glancex's own forward, then the heads).

        ``requests``: per image, a list of request strings. Results are in image-major order. Up to 64 images and
        256 pairs replay a CUDA graph per shape bucket (padding images and requests repeat the last one; dropped).
        """
        from glancex.data import collate

        d = self.direction(direction)
        mode = _SHORT[d]
        items = self._items(images, requests)
        M = sum(len(it["q_ids"]) for it in items)
        graph = self._use_graphs() and _bucket(len(items), _P_BUCKETS) and _bucket(M, _M_BUCKETS)
        cfg = self.cfg
        if graph:
            Bb = _bucket(len(items), _P_BUCKETS)
            items = list(items) + [items[-1]] * (Bb - len(items))
            extra = _bucket(sum(len(it["q_ids"]) for it in items), _M_BUCKETS) - sum(len(it["q_ids"]) for it in items)
            if extra:
                last = dict(items[-1])
                for k in ("q_ids", "qs", "present", "boxes"):
                    last[k] = list(last[k]) + [last[k][-1]] * extra
                items[-1] = last
            n_img = max(int(it["grid_thw"].prod()) // 4 for it in items)
            lt = max(len(q) for it in items for q in it["q_ids"])
            cfg = cfg.update(dict(pad_img_to=_round(n_img + 2, 64), pad_txt_to=_round(lt, 8)))
        b = collate(items, self.tok, cfg, mode)
        b.pop("meta")
        b = self._to_dev(b)
        feats = torch.cat(list(self.net.image_features(b["pixel_values"], b["grid_thw"])))
        names = ("img_ids", "img_valid", "img_mask", "img_pos", "txt_ids", "txt_valid", "txt_pos", "q_img",
                 "vis_valid", "centers")
        inputs = [b[k] for k in names]
        if not graph:
            return self._rows(self._pairs_core(mode, *inputs, feats))
        Nb = _round(inputs[8].shape[1], 64)
        inputs[8] = F.pad(inputs[8], (0, Nb - inputs[8].shape[1]))
        inputs[9] = F.pad(inputs[9], (0, 0, 0, Nb - inputs[9].shape[1]))
        inputs.append(F.pad(feats, (0, 0, 0, len(items) * Nb - len(feats))))
        key = ("pairs", mode, len(items), inputs[0].shape[1], inputs[4].shape[0], inputs[4].shape[1], Nb)
        outs = self._graphs(key, functools.partial(self._pairs_core, mode), inputs)
        return self._rows([t[:M] for t in outs])

    def _pairs_core(self, mode, img_ids, img_valid, img_mask, img_pos, txt_ids, txt_valid, txt_pos, q_img, vis_valid,
                    centers, feats):
        batch = dict(img_ids=img_ids, img_valid=img_valid, img_mask=img_mask, img_pos=img_pos, txt_ids=txt_ids,
                     txt_valid=txt_valid, txt_pos=txt_pos, q_img=q_img, vis_valid=vis_valid)
        with self._autocast():
            o = self.net(batch, feats, mode)
        if mode == "i2t":
            valid, cent = vis_valid, centers
        else:  # one request-conditioned image per request
            valid, cent = vis_valid[q_img], centers[q_img]
        return self._heads(o["r"], o["vis"], valid, cent, o["vis_index"])

    # ---------------------------------------------------------------------------------------- readout
    def _heads(self, r, vis, vis_valid, centers, q_img):
        """Heads in FP32 (as glancex): presence, box, IoU and the multi-box decode, on the device."""
        out = self.heads(r, vis, vis_valid, centers, q_img)
        boxes, scores, keep = self.heads.multi_tensors(r, vis, vis_valid, centers, q_img, out, thr=self.obj_thr)
        return [out["presence_logit"], out["iou_logit"], out["box_pred"], boxes, scores, keep]

    @staticmethod
    def _rows(outs):
        """One device-to-host copy, then one result dict per request."""
        pres, iou, box, mb, ms = (t.float().cpu() for t in outs[:5])
        keep = outs[5].cpu()
        p, piou = pres.sigmoid().tolist(), iou.sigmoid().tolist()
        box = box.clamp(0, 1).tolist()
        return [{"p_present": p[m], "box": box[m], "p_iou50": piou[m], "boxes": mb[m][keep[m]].tolist(),
                 "box_scores": ms[m][keep[m]].tolist(), "presence_logit": float(pres[m])} for m in range(len(p))]

    # ---------------------------------------------------------------------------------------- CUDA graphs
    def _use_graphs(self):
        return self.graphs and self.dev.type == "cuda"

    def _graphs(self, key, fn, inputs):
        if self._graph_cache is None:
            self._graph_cache = GraphCache(self.dev, limit=self.graph_limit or 64)
        return self._graph_cache(key, fn, inputs)

    def free_graphs(self):
        self._graph_cache = None
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()


def image_tokens(image) -> int:
    """Visual tokens of a fitted image (one per 32x32 pixels)."""
    w, h = image.size
    return (w // 32) * (h // 32)


def as_image(x):
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)
    return x


# ------------------------------------------------------------------------------------------------ CUDA graphs
_B_BUCKETS = (1, 2, 4, 8, 16)
_M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_P_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _bucket(n, buckets):
    return next((b for b in buckets if n <= b), None)


def _round(n, m):
    return (n + m - 1) // m * m


class GraphCache:
    """CUDA graphs keyed by shape bucket, LRU-bounded. Input buffers are shared between graphs of equal input
    shapes (replays are sequential on one stream); outputs are only valid until the next replay."""

    def __init__(self, device, limit=64):
        self.dev, self.limit = device, limit
        self.entries, self.buffers, self.pool = {}, {}, None

    def _buffer(self, i, t):
        key = (i, tuple(t.shape), t.dtype)
        if key not in self.buffers:
            self.buffers[key] = torch.empty_like(t)
        return self.buffers[key]

    def __call__(self, key, fn, inputs):
        entry = self.entries.pop(key, None)
        if entry is None:
            static = [self._buffer(i, t) for i, t in enumerate(inputs)]
            for s, t in zip(static, inputs):
                s.copy_(t)
            stream = torch.cuda.Stream(self.dev)
            stream.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm up: Triton autotuning and lazy initialisation stay outside the capture
                    fn(*static)
            torch.cuda.current_stream(self.dev).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outs = fn(*static)
            self.pool = graph.pool()
            entry = (graph, static, outs)
            while len(self.entries) >= self.limit:
                self.entries.pop(next(iter(self.entries)))
        graph, static, outs = entry
        self.entries[key] = entry
        for s, t in zip(static, inputs):
            s.copy_(t)
        graph.replay()
        return outs
