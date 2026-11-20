"""Glance scan engine: ONE text query against MANY images, per-image typed answers in one packed forward.

Builds on speed/engine.py (merged LoRA, fused ops, CUDA graphs, FP8 option). Three modes:

  cached (A)  encode(images) -> ImageBank: every image's prefix  <|im_start|>user\\n<vision_start><image_pad>*n<vision_end>
              is prefilled ONCE; the bank keeps, per image, the GDN recurrent state + conv tail of every linear-attention
              layer, the post-RoPE K/V of every full-attention layer, the final visual hidden states and the box head's
              query-independent key projection c_k(v_norm(vis)).  query(bank, q) then runs N short branches (one copy
              of the question block per image) in ONE forward: token-wise ops over the packed N*|q| tokens, GDN branches
              start from their own image's state (no gather copy), attention branches read their own image's K/V
              (split softmax over [prefix keys | block keys], no K/V concatenation copy).  No vision tower, no prefill.
  fresh (B)   fresh(images, q): vision tower over all N images (batched), packed prefill of the N prefixes, and the N
              question blocks. Each sequence is its own prefix followed by its own block.
  query-first (C)  the query text is placed in front of the image inside the prefix:
              <|im_start|>user\\n{query}\\n<vision_start>...<vision_end>  + the usual question block.  The shared query
              part is at most ~3% of the tokens, so latency = fresh; it exists to measure accuracy (out of training
              distribution for v1: v1 always saw image-first prompts).

Exactness: every per-image answer equals running that image alone through dv.model (SharedPrefixLM + heads):
the branch math is the reference math (same causal GDN continuation, same masked attention over prefix + own block),
only reorganised.  scan/test_scan_equivalence.py checks it on a tiny random config (fp32, CPU) against both the
per-image reference and the plain HF forward.
"""
import math, os, time
import torch
import torch.nn.functional as F

from scan import vision_prep as frames
from scan.bank import FORMATS, StoredBank
from scan.varlen import FreshPack, PackedGraphs, packed_forward
from dv.data.mixed import QTYPES, collate_mixed
from dv.model.typed import typed_logits
import speed.engine as SE
from speed.box_readout import center_scores, geometry, mlp, project
from speed.engine import GlanceEngine, prepare, fused, _gdn_core_noproj


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ branch GDN
BRANCH_GDR = os.environ.get("SCAN_BRANCH_GDR", "recurrent")  # "recurrent": fused recurrent kernel, no final state (branches are short and their final
                             # state is never used); "chunk": the reference chunked kernel (as the prefix)


def _gdr_branch(q, k, v, g, beta, S):
    """Gated delta rule over short branch blocks from per-branch initial states; returns outputs only."""
    if BRANCH_GDR == "recurrent":
        try:
            from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fr
            o, _ = fr(q, k, v, g=g, beta=beta, initial_state=S, output_final_state=False, use_qk_l2norm_in_kernel=True)
            return o
        except ImportError:
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule as tr
            o, _ = tr(q, k, v, g=g, beta=beta, initial_state=S, output_final_state=False, use_qk_l2norm_in_kernel=True)
            return o
    from dv.model.backbone import _gdr
    return _gdr(q, k, v, g, beta, S)[0]


def _gdn_core_branch(m, conv_out, z, beta, g, S):
    """= _gdn_core_noproj with the branch delta rule."""
    B, _, T = conv_out.shape
    conv_out = F.silu(conv_out).transpose(1, 2)
    q, k, v = torch.split(conv_out, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    o = _gdr_branch(q.reshape(B, T, -1, m.head_k_dim), k.reshape(B, T, -1, m.head_k_dim), v.reshape(B, T, -1, m.head_v_dim),
                    g, beta, S)
    return m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)


# ------------------------------------------------------------------------------------------------ items
def question_block(tok, q):
    """q: dict(type='box', text=...) or typed (noul / choice / score) in dv/data/mixed.py record format."""
    if q["type"] == "box":
        return tok.box_block(q["text"]), []
    return tok.typed_block(q)


def make_item(tok, proc, img, qs, query_first=None):
    """One collate_mixed item: image prefix (+ optional query text in front of the image) and question blocks."""
    pv = proc.image_processor(images=[img], do_resize=False, return_tensors="pt")
    item = image_item(tok, pv["image_grid_thw"][0], img.size, qs, query_first)
    item["pixel_values"] = pv["pixel_values"]
    return item


def image_item(tok, grid, size, qs, query_first=None):
    segs = [("ids", tok.user_open)]
    if query_first:
        segs.append(("ids", tok.enc(query_first.strip() + "\n")))
    segs += [("ids", [tok.vs]), ("img", None), ("ids", [tok.ve])]
    q_ids, qtype, labs = [], [], []
    for q in qs:
        ids, lab = question_block(tok, q)
        q_ids.append(ids); qtype.append(QTYPES[q["type"]]); labs.append(lab)
    n = len(qs)
    return dict(rec={}, segs=segs, grid_thw=grid, q_ids=q_ids,
                qtype=qtype, labs=labs, tgts=[[0.0] * len(l) for l in labs], present=[False] * n, has_box=[False] * n,
                box=[[0, 0, 0, 0]] * n, points=[[] for _ in range(n)], qs=qs, orig_wh=size)


def make_clip_item(tok, proc, frames, qs, dt=0.5, max_vis=96, query_first=None):
    """A short clip as a NATIVE Qwen3.5 video prefix, token for token as dv/data/video.video_segs (Glance-Video):
    <|im_start|>user\n<vs> {<t seconds><vs><video_pad>*n<ve>} x ceil(T/2) <ve>   (timestamps local to the clip).
    frames: list of PIL images or a uint8 array [T,H,W,3], time order, dt seconds apart."""
    import numpy as np
    from dv.data import video as V
    arr = frames if isinstance(frames, np.ndarray) else np.stack([np.asarray(f.convert("RGB")) for f in frames])
    if len(arr) % 2:
        arr = np.concatenate([arr, arr[-1:]])
    pv, g = V.video_inputs(arr, proc, max_vis)
    T, gh, gw = int(g[0]), int(g[1]) // 2, int(g[2]) // 2
    ts = V.timestamps(len(arr), dt)
    head = tok.user_open + (tok.enc(query_first.strip() + "\n") if query_first else [])
    segs = [("ids", head + [tok.vs])]
    for k in range(T):
        segs += [("ids", tok.enc(f"<{ts[k]:.1f} seconds>") + [tok.vs]), ("vid", (gh, gw)), ("ids", [tok.ve])]
    segs.append(("ids", [tok.ve]))
    q_ids, qtype, labs = [], [], []
    for q in qs:
        ids, lab = question_block(tok, q)
        q_ids.append(ids); qtype.append(QTYPES[q["type"]]); labs.append(lab)
    n = len(qs)
    return dict(rec={}, segs=segs, pixel_values_videos=pv, video_grid_thw=g, q_ids=q_ids, qtype=qtype, labs=labs,
                tgts=[[0.0] * len(l) for l in labs], present=[False] * n, has_box=[False] * n, box=[[0, 0, 0, 0]] * n,
                points=[[] for _ in range(n)], qs=qs, orig_wh=None)


def is_clip(u):
    """A scan unit is an image (PIL) or a clip (list/tuple of PIL frames or a uint8 array [T,H,W,3])."""
    return isinstance(u, (list, tuple)) or (hasattr(u, "ndim") and getattr(u, "ndim", 0) == 4)


DUMMY_Q = dict(type="box", text="x")


# ------------------------------------------------------------------------------------------------ bank
class ImageBank:
    """Cached per-image state for N images (dim 0 = image). Built by ScanEngine.encode; concat() merges banks
    (attention K/V and the prefix mask are LEFT-padded to a common length: pads are masked keys, exact)."""

    def __init__(self, cache, types, p_valid, q_start, vis, vis_valid, centers, ck, grid_hw, has_clips=None):
        self.cache, self.types = cache, types
        self.p_valid, self.q_start = p_valid, q_start          # [N, Lp] bool, [N] long (first text position)
        self.vis, self.vis_valid, self.centers = vis, vis_valid, centers   # [N, Nv, D], [N, Nv], [N, Nv, 2]
        self.ck = ck                                            # [N, Nv, d] fp32  c_k(v_norm(vis))
        self.has_clips = vis.shape[1] == 1 if has_clips is None else has_clips
        self.grid_hw = grid_hw                                  # [N, 2]
        self._graphs = {}
        self._pool = None

    @property
    def N(self):
        return self.p_valid.shape[0]

    @property
    def Lp(self):
        return self.p_valid.shape[1]

    def layer(self, i):
        return self.cache[i]

    def nbytes(self):
        t = [x for c in self.cache for x in c] + [self.vis, self.ck, self.grid_hw, self.centers, self.vis_valid, self.q_start, self.p_valid]
        return sum(x.numel() * x.element_size() for x in t)

    @staticmethod
    def concat(banks):
        """Merge banks (CONSUMES the inputs' state tensors to keep the peak at ~1x)."""
        if len(banks) == 1:
            return banks[0]
        # Materialize a compact input once. Repeated access to its dense compatibility
        # view inside the layer loop would reconstruct the entire bank for each layer.
        if any(isinstance(b, StoredBank) for b in banks):
            banks = [ImageBank(b.cache, b.types, b.p_valid, b.q_start, b.vis, b.vis_valid, b.centers, b.ck,
                               b.grid_hw, b.has_clips) if isinstance(b, StoredBank) else b for b in banks]
        Lp = max(b.Lp for b in banks)
        Nv = max(b.vis.shape[1] for b in banks)
        lpad = lambda t, n, v=0: F.pad(t, (n, 0), value=v) if n else t
        cache = []
        for li, lt in enumerate(banks[0].types):
            if lt == "linear_attention":
                cache.append((torch.cat([b.cache[li][0] for b in banks]), torch.cat([b.cache[li][1] for b in banks])))
            else:   # [N, H, Lp, D]: left pad along Lp
                cache.append(tuple(torch.cat([F.pad(b.cache[li][j], (0, 0, Lp - b.Lp, 0)) for b in banks]) for j in (0, 1)))
            for b in banks:                     # consume the inputs layer by layer (peak ~1x the bank, not 2x)
                b.cache[li] = None
        pv = torch.cat([lpad(b.p_valid, Lp - b.Lp, False) for b in banks])
        rp = lambda t, n: F.pad(t, (0, 0, 0, n)) if n else t        # right pad the visual-token axis
        vis = torch.cat([rp(b.vis, Nv - b.vis.shape[1]) for b in banks])
        ck = torch.cat([rp(b.ck, Nv - b.ck.shape[1]) for b in banks])
        cen = torch.cat([rp(b.centers, Nv - b.centers.shape[1]) for b in banks])
        vv = torch.cat([F.pad(b.vis_valid, (0, Nv - b.vis_valid.shape[1]), value=False) for b in banks])
        return ImageBank(cache, banks[0].types, pv, torch.cat([b.q_start for b in banks]), vis, vv, cen, ck,
                         torch.cat([b.grid_hw for b in banks]), any(b.has_clips for b in banks))

    def head(self, n):
        """The first n images as VIEWS (no copy). Graphs captured on it belong to the view."""
        cache = [tuple(x[:n] for x in c) for c in self.cache]
        return ImageBank(cache, self.types, self.p_valid[:n], self.q_start[:n], self.vis[:n], self.vis_valid[:n],
                         self.centers[:n], self.ck[:n], self.grid_hw[:n], self.has_clips)

    def select(self, idx):
        """Sub-bank of images idx (a copy)."""
        idx = torch.as_tensor(idx, device=self.p_valid.device)
        cache = [tuple(x[idx] for x in c) for c in self.cache]
        return ImageBank(cache, self.types, self.p_valid[idx], self.q_start[idx], self.vis[idx], self.vis_valid[idx],
                         self.centers[idx], self.ck[idx], self.grid_hw[idx], self.has_clips)


# ------------------------------------------------------------------------------------------------ engine
class ScanEngine(GlanceEngine):
    def __init__(self, net, heads, tok, proc, graphs=True, fast_ops=False, fast_vision=True, fresh_varlen=True, vision_prep="fused", lm_prefill=True):
        super().__init__(net, heads, graphs=graphs, fast_vision=fast_vision, fast_ops=fast_ops, optimized_singleton=False, lm_prefill=lm_prefill)
        self.tok, self.proc = tok, proc
        self.fresh_varlen = fresh_varlen
        self.packed_graphs = PackedGraphs()
        self._bank_decode_stream = None
        self.frame_prep = frames.FramePrep(self) if self.dev.type == "cuda" and vision_prep != "reference" else None
        self.frame_tower = SE.FastVision(net.visual, graphs=graphs, optimized=True) if (
            self.frame_prep is not None and fast_vision and vision_prep == "fused") else None
        self.exact_len = True                          # no prefix padding when all units have the same length
        self.dt, self.clip_max_vis = 0.5, 96          # clip units: seconds between frames, visual tokens per 2-frame group
        self.dtype = torch.bfloat16 if self.dev.type == "cuda" else next(self.lm.parameters()).dtype

    def free_graphs(self, *banks):
        """Drop every captured CUDA graph (engine, vision tower, given banks) and return the memory."""
        import gc
        self._graphs.clear(); self._pool = None
        self.packed_graphs.clear()
        if self.fv is not None:
            self.fv._graphs.clear(); self.fv._pool = None
        if self.frame_tower is not None:
            self.frame_tower._graphs.clear(); self.frame_tower._pool = None
        for b in banks:
            if hasattr(b, "free_graphs"):
                b.free_graphs()
                continue
            b._graphs.clear()
            b._pool = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- overrides: feature buffer in the model dtype (bf16 on GPU as speed/engine; fp32 in CPU tests)
    def to_device(self, eng, host, feats):
        E = {k: v.to(self.dev, non_blocking=True) for k, v in eng.items()}
        B, Lp = host["B"], host["Lp"]
        fb = torch.zeros(B * Lp, self.net.hidden, dtype=self.dtype, device=self.dev)
        rm = torch.zeros(B * Lp, dtype=torch.bool, device=self.dev)
        if feats is not None:
            fb.index_copy_(0, E["img_flat"], feats.to(self.dtype))
            rm[E["img_flat"]] = True
        E.pop("img_flat")
        E["featbuf"], E["img_rowmask"] = fb, rm
        return E

    @torch.no_grad()
    def run(self, E, host):
        if self.dev.type != "cuda":
            return self._core(E, host)
        return super().run(E, host)

    def _core(self, E, host, typed_prefix=False):
        with self._autocast():
            P, Q, x = self._inputs(E, host)
            h, cache = self.lm_forward(x, P, Q)
        if typed_prefix:
            return None, cache, None
        Np = host["B"] * host["Lp"]
        out, vis = self.heads_forward(h[:Np], h[Np:], E)
        return out, cache, vis

    # ---------------- host-side batch building
    def _items(self, images, qss, query_first):
        assert len({is_clip(im) for im in images}) == 1, "one call holds either images or clips"
        return [make_clip_item(self.tok, self.proc, im, qs, self.dt, self.clip_max_vis, query_first) if is_clip(im)
                 else make_item(self.tok, self.proc, im, qs, query_first) for im, qs in zip(images, qss)]

    def build(self, images, qss, query_first=None, Nv=None):
        items = self._items(images, qss, query_first)
        # equal-length prefixes (video frames / clips of one size): no padding at all -> the prefill runs exactly the
        # real tokens and takes the unmasked causal (flash) attention path; otherwise pad to 64 (shape buckets)
        lens = {self._plen(it) for it in items}
        bucket = 1 if (len(lens) == 1 and self.exact_len) else 64
        batch = collate_mixed(items, self.tok, pad_to=bucket)
        batch.pop("meta", None)
        if batch.get("pixel_values_videos") is not None:
            # clips carry no box targets (vis_valid is all False): prepare with room for every video token, then
            # shrink the box head's visual inputs to one masked slot (the typed head is the readout for clips)
            eng, host = prepare(batch, Lp_bucket=bucket, Nv=int(batch["p_img_mask"].sum(1).max()))
            eng["vis_gather"], eng["vis_valid"], eng["centers"] = eng["vis_gather"][:, :1], eng["vis_valid"][:, :1], eng["centers"][:, :1]
            host["Nv"] = 1
            return batch, eng, host
        eng, host = prepare(batch, Lp_bucket=bucket, Nv=Nv)
        return batch, eng, host

    @staticmethod
    def _plen(it, merge=2):
        """prefix length of a collate item (as collate_mixed computes it)"""
        g = it.get("grid_thw")
        gh, gw = (int(g[1] // merge), int(g[2] // merge)) if g is not None else (0, 0)
        n = 0
        for kind, val in it["segs"]:
            n += len(val) if kind == "ids" else (gh * gw if kind == "img" else val[0] * val[1])
        return n

    def feats(self, batch, image_tower=None):
        """Vision tower over the batch's images or clips. Qwen3.5's tower attends within each temporal group of a
        video (cu_seqlens per t), so a clip grid [t,h,w] is run as t grids [1,h,w]: identical math, and the fast
        (graph-captured, batched-SDPA) path applies."""
        if batch.get("pixel_values_videos") is not None:
            g = batch["video_grid_thw"]
            g1 = torch.repeat_interleave(torch.cat([torch.ones_like(g[:, :1]), g[:, 1:]], 1), g[:, 0], dim=0)
            return self.vision(batch["pixel_values_videos"].to(self.dev, non_blocking=True), g1)
        vision = self.vision if image_tower is None else image_tower
        return vision(batch["pixel_values"].to(self.dev, non_blocking=True), batch["grid_thw"])

    # ---------------- (B) fresh images: one call, vision + prefill + one block per image
    def fresh_vision(self, pixels, grid):
        if self.frame_tower is None:
            return self.vision(pixels, grid)
        return self.frame_tower(pixels, grid)

    def _fresh_token_budget(self):
        """Bound the active layer workspace using model widths and available device memory."""
        if self.dev.type != "cuda":
            return math.inf
        cfg = self.lm.config
        # Fused gate/up, activated intermediate, residual/norm buffers and mixer temporaries.
        bytes_per_token = self.dtype.itemsize * (3 * cfg.intermediate_size + 8 * cfg.hidden_size)
        free, _ = torch.cuda.mem_get_info(self.dev)
        # Keep room for the vision tower and graph pools; limit the eager working set on large GPUs.
        workspace = min(free // 4, 2 << 30)
        return max(1, workspace // bytes_per_token)

    @torch.no_grad()
    def fresh(self, images, q, query_first=False, keep_bank=False, timings=None):
        """N new images and one query -> per-image CPU outputs in input order."""
        if keep_bank or not self.fresh_varlen:
            return self._fresh_reference(images, q, query_first, keep_bank, timings)
        _sync()
        start = time.perf_counter()
        qf = (q.get("text") or q.get("instr")) if query_first else None
        fast_frames = self.frame_prep is not None and self.frame_prep.supports(images)
        if fast_frames:
            example = self.frame_prep.template(images[0].size, q, qf)
            lengths = [example["n_tokens"]] * len(images)
        else:
            items = self._items(images, [[q]] * len(images), qf)
            lengths = [self._plen(item) + len(item["q_ids"][0]) for item in items]
        budget = self._fresh_token_budget()
        if sum(lengths) <= budget:
            cuts = [0, len(images)]
        elif min(lengths) == max(lengths):
            # Stable batch sizes let same-grid frames reuse the existing vision graphs across questions.
            count = max(1, budget // lengths[0])
            step = 1 << (count.bit_length() - 1)
            cuts = [*range(0, len(images), step), len(images)]
        else:
            cuts, used = [0], 0
            for i, size in enumerate(lengths):
                if used and used + size > budget:
                    cuts.append(i)
                    used = 0
                used += size
            cuts.append(len(images))
        prep, vision, lm = time.perf_counter() - start, 0., 0.
        outputs = []
        for lo, hi in zip(cuts, cuts[1:]):
            t0 = time.perf_counter()
            if fast_frames:
                batch = frames.frame_layout(example, hi - lo)
                host = frames.upload_frames(images[lo:hi])
            else:
                batch = collate_mixed(items[lo:hi], self.tok, pad_to=1)
            t1 = time.perf_counter()
            if fast_frames:
                pixels = self.frame_prep.pixels(host, dtype=self.dtype)
                feats = self.fresh_vision(pixels, batch["grid_thw"])
                del host, pixels
            else:
                feats = self.feats(batch)
            _sync()
            t2 = time.perf_counter()
            out = packed_forward(self, FreshPack(batch, self.dev), feats,
                                 use_graph=bool(self.graphs and self.lm_prefill and fast_frames and not qf))
            outputs.append({k: v.cpu() for k, v in out.items()})
            del feats, out, batch
            t3 = time.perf_counter()
            prep += t1 - t0
            vision += t2 - t1
            lm += t3 - t2
        result = outputs[0] if len(outputs) == 1 else {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
        if timings is not None:
            timings.update(prep=prep * 1e3, vision=vision * 1e3, lm=lm * 1e3,
                           total=(time.perf_counter() - start) * 1e3)
        self.last_fresh_chunks = len(outputs)
        return result

    def _fresh_reference(self, images, q, query_first, keep_bank, timings):
        T = timings if timings is not None else {}
        _sync(); t0 = time.perf_counter()
        qf = (q.get("text") or q.get("instr")) if query_first else None
        batch, eng, host = self.build(images, [[q]] * len(images), query_first=qf)
        t1 = time.perf_counter()
        feats = self.feats(batch)
        _sync(); t2 = time.perf_counter()
        E = self.to_device(eng, host, feats)
        res, cache, vis = self.run(E, host)
        out = {k: v[:host["M0"]].cpu() for k, v in res.items()}
        t3 = time.perf_counter()
        T.update(prep=(t1 - t0) * 1e3, vision=(t2 - t1) * 1e3, lm=(t3 - t2) * 1e3, total=(t3 - t0) * 1e3)
        self.last_fresh_chunks = 1
        if keep_bank:
            bank = self._bank_from(batch, E, host, cache, vis, clone=True)
            mode = os.getenv("GLANCE_BANK_FORMAT", "compact")
            out["_bank"] = bank if mode == "legacy" else StoredBank.pack(bank, mode)
        return out

    # ---------------- (A) build the bank
    def _bank_from(self, batch, E, host, cache, vis, clone=False):
        B = host["B"]
        # ALWAYS materialise: the conv tail and V are views into the whole prefix projection buffers (keeping them
        # alive costs ~0.5 MB per prefix token); clone=True additionally detaches from CUDA-graph static outputs
        cp = lambda t: t.contiguous().clone()
        cache = [tuple(cp(x) for x in c) for c in cache]
        if vis is None:
            vis = torch.empty((B, 0, 0), device=self.dev, dtype=self.dtype)
            ck = vis
        else:
            vis = cp(vis[:B])
            H = self.heads.box
            with torch.no_grad():
                ck = project(H.c_k, H.v_norm(vis.float()), self.box_reduction, 512)
        vv = E["vis_valid"][:B, :vis.shape[1]]
        centers = E["centers"][:B, :vis.shape[1]]
        q_start = batch["q_pos"][0, :, 0][: B].clone() if batch["q_ids"].shape[0] == B else \
            torch.stack([batch["q_pos"][0, int((batch["q_img"] == b).nonzero()[0, 0]), 0] for b in range(B)])
        ghw = batch["grid_thw"][:, 1:] // 2 if batch.get("grid_thw") is not None else torch.zeros(B, 2, dtype=torch.long)
        return ImageBank(cache, self.types, E["p_valid"][:B].clone(), q_start.to(self.dev), vis, vv.clone(),
                         centers.clone(), ck, ghw.to(self.dev), batch.get("pixel_values_videos") is not None)

    @torch.no_grad()
    def encode(self, images, chunk=32, query_first=None, *, bank_format=None, readout="all", tier="gpu"):
        """Prefill read-only prefixes. `bank_format` overrides GLANCE_BANK_FORMAT.
        GPU banks default to compact KV and host banks also pack FP32 state losslessly.
        `readout="typed"` omits spatial outputs from compact banks. `tier="cpu"`
        pins each chunk as it is encoded and streams it on query.
        """
        mode = bank_format or os.getenv("GLANCE_BANK_FORMAT", "state_packed" if tier == "cpu" else "compact")
        if mode not in FORMATS or readout not in ("all", "typed") or tier not in ("gpu", "cpu"):
            raise ValueError("unsupported bank format, readout or tier")
        if mode == "legacy" and tier != "gpu":
            raise ValueError("host tier requires a compact bank format")
        if not len(images) or chunk <= 0:
            raise ValueError("encode requires at least one unit and a positive chunk size")
        if tier == "cpu":
            from scan.host_bank import HostBank
            return HostBank.from_chunks(self.encode(images[s:s+chunk], chunk, query_first,
                                                    bank_format=mode, readout=readout)
                                        for s in range(0, len(images), chunk))
        banks = []
        for s in range(0, len(images), chunk):
            ims = images[s:s + chunk]
            batch, eng, host = self.build(ims, [[DUMMY_Q]] * len(ims), query_first=query_first)
            # Match fresh image embeddings when caching spatial box keys. Clips use the video path.
            feats = self.feats(batch, self.frame_tower if self.lm_prefill else None)
            E = self.to_device(eng, host, feats)
            res, cache, vis = self._core(E, host, typed_prefix=readout == "typed" and mode != "legacy")
            banks.append(self._bank_from(batch, E, host, cache, vis))
            del res, cache, vis, E, feats
        bank = ImageBank.concat(banks)
        return bank if mode == "legacy" else StoredBank.pack(bank, mode, readout)

    # ---------------- (A) the branch forward
    def branch_inputs(self, bank, q):
        """Host-side inputs for one question block replicated on every image of the bank."""
        ids, lab = question_block(self.tok, q)
        return self.question_inputs(ids, lab, QTYPES[q["type"]], bank.N, self.compact_questions)

    @staticmethod
    def question_inputs(ids, lab, qtype, N, compact):
        L = len(ids)
        Tq = -(-L // 16) * 16
        nq = N * L
        Nqb = SE.question_rows(nq, compact)
        MT = N * Tq
        q_tok_idx = torch.full((Nqb,), MT, dtype=torch.long)
        q_tok_idx[:nq] = (torch.arange(N)[:, None] * Tq + torch.arange(L)[None]).reshape(-1)
        q_ids_packed = torch.zeros(Nqb, dtype=torch.long)
        q_ids_packed[:nq] = torch.tensor(ids).repeat(N)
        q_valid = torch.zeros(N, Tq, dtype=torch.bool)
        q_valid[:, :L] = True
        q_last_packed = torch.arange(N) * L + (L - 1)
        nl = max(1, len(lab))
        lab_idx = torch.full((N, nl), -1, dtype=torch.long)
        if lab:
            lab_idx[:, :len(lab)] = torch.tensor(lab)
        E = dict(q_tok_idx=q_tok_idx, q_ids_packed=q_ids_packed, q_valid=q_valid, q_last_packed=q_last_packed,
                 lab_idx=lab_idx, q_off=torch.arange(Tq))
        host = dict(M=N, M0=N, Tq=Tq, Nqb=Nqb, L=L, qtype=qtype)
        return E, host

    def _mix_gdn_branch(self, m, mixed, z, beta, g, M, Tq, S, tail):
        w = m.conv1d.weight
        mixed = mixed.transpose(1, 2)
        mq = torch.cat([tail, mixed.to(tail.dtype)], dim=-1).to(w.dtype)
        conv = SE.conv_taps(mq, w, m.conv1d.bias) if SE.USE_TAPS else \
            F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
        o = _gdn_core_branch(m, conv, z.reshape(M, Tq, -1, m.head_v_dim), beta, g, S)
        return o

    @staticmethod
    def _split_attn(qq, kq, vq, kp, vp, key_valid_p, mask_qq, scale):
        """Attention of each block over [its own image's prefix keys | its own block keys], GQA, without
        materialising the concatenated K/V. qq [M,Hq,Tq,D]; kq,vq [M,Hk,Tq,D]; kp,vp [M,Hk,Lp,D];
        key_valid_p [M,Lp] bool; mask_qq [M,Tq,Tq] bool (True = may attend). Softmax in fp32."""
        M, Hq, Tq, D = qq.shape
        Hk = kp.shape[1]
        r = Hq // Hk
        q = qq.reshape(M, Hk, r * Tq, D)
        sp = torch.matmul(q, kp.transpose(-1, -2)).float() * scale              # [M,Hk,rTq,Lp]
        sq = torch.matmul(q, kq.transpose(-1, -2)).float() * scale              # [M,Hk,rTq,Tq]
        sp = sp.masked_fill(~key_valid_p[:, None, None, :], float("-inf"))
        sq = sq.masked_fill(~mask_qq.repeat(1, r, 1)[:, None], float("-inf"))
        mx = torch.maximum(sp.amax(-1, keepdim=True), sq.amax(-1, keepdim=True))
        ep, eq = (sp - mx).exp(), (sq - mx).exp()
        den = ep.sum(-1, keepdim=True) + eq.sum(-1, keepdim=True)
        o = torch.matmul(ep.to(vp.dtype), vp).float() + torch.matmul(eq.to(vq.dtype), vq).float()
        return (o / den).to(qq.dtype).reshape(M, Hq, Tq, D)

    def _branch_lm(self, x, Q, bank):
        """x: [Nqb, D] packed embedded branch tokens. Same per-token math as GlanceEngine.lm_forward's question path."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        M, Tq, qidx = Q["M"], Q["Tq"], Q["q_tok_idx"]
        MT = M * Tq
        fast_attention = self.branch_attention is not None and not bank.has_clips
        state_reader = None
        if isinstance(bank, StoredBank) and bank.mode == "state_packed" and self.dev.type == "cuda":
            from scan.bank_codec import StateReader
            if self._bank_decode_stream is None:
                self._bank_decode_stream = torch.cuda.Stream(device=self.dev)
            state_reader = StateReader.make(bank, self._bank_decode_stream)

        def scat(t):
            buf = t.new_zeros((MT + 1,) + tuple(t.shape[1:]))
            buf.index_copy_(0, qidx, t)
            return buf[:MT].view((M, Tq) + tuple(t.shape[1:]))

        def gath(t):
            flat = t.reshape((MT,) + tuple(t.shape[2:]))
            flat = torch.cat([flat, flat.new_zeros((1,) + tuple(flat.shape[1:]))])
            return flat.index_select(0, qidx)

        fo = self.fast_ops
        eps = self.layers[0].input_layernorm.eps
        h_next = fused("rms")(x, self.layers[0].input_layernorm.weight, eps) if fo else None
        for li, (layer, lt) in enumerate(zip(self.layers, self.types)):
            h = h_next if fo else layer.input_layernorm(x)
            if lt == "linear_attention":
                m = layer.linear_attn
                if state_reader is None:
                    S, tail = bank.layer(li)
                else:
                    S, tail = state_reader.take(li), bank.tensors[li, "tail"]
                if bank.has_clips and self.branch_output is not None and BRANCH_GDR == "recurrent":
                    # Preserve FLA arithmetic on native videos, while removing right padding
                    # and the four per-layer projection scatters.
                    length = Q["L"]
                    count = M * length
                    mixed, z, beta, g = self._gdn_projections(m, h)
                    rows = [t[:count].reshape(M, length, -1) for t in (mixed, z, beta, g)]
                    oq = self._mix_gdn_branch(m, *rows, M, length, S, tail).reshape(count, -1)
                    if count < len(h):
                        oq = F.pad(oq, (0, 0, 0, len(h) - count))
                elif self.branch_output is not None and BRANCH_GDR == "recurrent" and Tq <= 128:
                    mixed, z, ab = self._gdn_project_raw(m, h)
                    oq = self.branch_output(m, mixed, z, None, None, S, tail, M, Q["L"], raw_ab=ab)
                else:
                    mixed, z, beta, g = self._gdn_projections(m, h)
                    oq = gath(self._mix_gdn_branch(m, scat(mixed), scat(z), scat(beta), scat(g), M, Tq, S, tail))
                o = m.out_proj(oq)
            else:
                m = layer.self_attn
                D2 = m.head_dim
                if fast_attention and Tq <= 128:
                    qa, kk, vv, gate = self._attention_projections(m, h, Q["packed_rope"])
                    kp, vp, tokens = bank.attention(li) if isinstance(bank, StoredBank) else (*bank.layer(li), None)
                    oq = self.branch_attention(qa, kk, vv, gate, kp, vp, bank.p_valid, m.scaling, M, Q["L"],
                                               self.mask_mode != "block_bidir", tokens=tokens)
                    o = m.o_proj(oq)
                else:
                    if fo:
                        y = m._qkv(h)
                        nq_, nk_ = m.q_proj.out_features, m.k_proj.out_features
                        qg, kk, vv = y[:, :nq_], y[:, nq_:nq_ + nk_], y[:, nq_ + nk_:]
                    else:
                        qg, kk, vv = m.q_proj(h), m.k_proj(h), m.v_proj(h)
                    qa, gate = torch.chunk(qg.view(-1, qg.shape[-1] // (2 * D2), 2 * D2), 2, dim=-1)
                    qa = m.q_norm(qa)
                    kk = m.k_norm(kk.view(-1, kk.shape[-1] // D2, D2))
                    vv = vv.view(-1, vv.shape[-1] // D2, D2)
                    gate = gate.reshape(gate.shape[0], -1)
                    kp, vp = bank.layer(li)
                    qq, kq = apply_rotary_pos_emb(scat(qa).transpose(1, 2), scat(kk).transpose(1, 2), *Q["rope"])
                    vq = scat(vv).transpose(1, 2)
                    oq = self._split_attn(qq, kq.to(kp.dtype), vq.to(vp.dtype), kp, vp, bank.p_valid, Q["mask_qq"], m.scaling)
                    oq = gath(oq.transpose(1, 2).reshape(M, Tq, -1))
                    o = m.o_proj(oq * torch.sigmoid(gate))
            if fo:
                x, h2 = fused("add_rms")(x, o, layer.post_attention_layernorm.weight, eps)
                if fast_attention:
                    update = self._mlp_update(layer, h2)
                    norm = self.layers[li + 1].input_layernorm if li + 1 < len(self.layers) else self.lm.norm
                    x, h_next = fused("add_rms")(x, update, norm.weight, norm.eps)
                    continue
                x = x + layer.mlp.down_proj(fused("silu_mul")(layer._gu(h2)))
                nxt = self.layers[li + 1].input_layernorm.weight if li + 1 < len(self.layers) else None
                if nxt is not None:
                    h_next = fused("rms")(x, nxt, eps)
            else:
                x = x + o
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        return h_next if fast_attention else self.lm.norm(x)

    def _branch_heads(self, hq_last, bank, E):
        """= BoxHeads.forward + typed readout with q_img = arange(N), using the bank's cached c_k(v_norm(vis))."""
        if getattr(bank, "readout", "all") == "typed":
            return {"typed_logits": typed_logits(self.heads.typed(hq_last), E["lab_idx"]).float()}
        H = self.heads.box
        hq = H.q_norm(hq_last.float())
        pres = mlp(H.presence, hq, self.box_reduction).squeeze(-1)
        if self.box_reduction != "legacy":
            query = project(H.c_q, hq, self.box_reduction)
            if isinstance(bank, StoredBank) and "pixels" in bank.tensors:
                cl = center_scores(query, bank.tensors["ck"], bank.vis_valid, slots=bank.tensors["pixels"])
            else:
                cl = center_scores(query, bank.ck, bank.vis_valid)
        else:
            cl = torch.einsum("md,mnd->mn", H.c_q(hq), bank.ck) / math.sqrt(H.c_q.out_features)
            cl = cl.masked_fill(~bank.vis_valid, -1e4)
        pi = cl.argmax(-1)
        ar = torch.arange(hq.shape[0], device=hq.device)
        if isinstance(bank, StoredBank):
            v_pred = H.v_norm(bank.spatial("vis", pi).float())
            pc = bank.spatial("centers", pi)
        else:
            v_pred = H.v_norm(bank.vis[ar, pi].float())
            pc = bank.centers[ar, pi]
        if self.box_reduction != "legacy":
            box, iou = geometry(H, hq, v_pred, pc, self.box_reduction)
        else:
            el = H.edges_logits(hq, v_pred, pc)
            box = H.expected_box(el, pc)
            iou = H.iou(H._edge_feats(hq, v_pred, pc)).squeeze(-1)
        tl = typed_logits(self.heads.typed(hq_last), E["lab_idx"])
        return dict(presence_logit=pres.float(), box_pred=box.float(), iou_logit=iou.float(), typed_logits=tl.float(),
                    center_idx=pi)

    def _core_branch(self, E, host, bank):
        lm = self.lm
        M, Tq = host["M"], host["Tq"]
        with self._autocast():
            xq = lm.embed_tokens(E["q_ids_packed"])
            pos = (bank.q_start[:, None] + E["q_off"][None]).expand(3, M, Tq)
            qv = E["q_valid"]
            if self.mask_mode == "block_bidir":
                within = qv[:, None, :].expand(M, Tq, Tq)
            else:
                within = torch.ones(Tq, Tq, dtype=torch.bool, device=qv.device).tril()[None] & qv[:, None, :]
            within = within | torch.eye(Tq, dtype=torch.bool, device=qv.device)[None]
            Q = dict(M=M, Tq=Tq, L=host["L"], q_tok_idx=E["q_tok_idx"], rope=self.net.rope(xq[None], pos), mask_qq=within)
            if self.branch_attention is not None and not bank.has_clips and Tq <= 128:
                idx = E["q_tok_idx"].clamp_max(M * Tq - 1)
                Q["packed_rope"] = tuple(r.reshape(-1, r.shape[-1])[idx] for r in Q["rope"])
            h = self._branch_lm(xq, Q, bank)
        return self._branch_heads(h[E["q_last_packed"]], bank, E)

    @torch.no_grad()
    def query(self, bank, q, timings=None):
        """One query against every image of the bank -> dict of [N, ...] CPU tensors (+ timings)."""
        from scan.host_bank import HostBank
        if isinstance(bank, HostBank):
            return bank.query(self, q, timings)
        if getattr(bank, "readout", "all") == "typed" and q["type"] == "box":
            raise ValueError("a typed-only bank cannot answer a box query")
        T = timings if timings is not None else {}
        _sync(); t0 = time.perf_counter()
        E_cpu, host = self.branch_inputs(bank, q)
        t1 = time.perf_counter()
        outs = self._launch_branch(bank, E_cpu, host)
        out = {k: v.float().cpu() if v.is_floating_point() else v.cpu() for k, v in outs.items()}
        t2 = time.perf_counter()
        T.update(prep=(t1 - t0) * 1e3, model=(t2 - t1) * 1e3, total=(t2 - t0) * 1e3)
        return out

    def _launch_branch(self, bank, E_cpu, host):
        E = {k: v.to(self.dev, non_blocking=True) for k, v in E_cpu.items()}
        key = ("branch", self.box_reduction, host["M"], host["Tq"], host["Nqb"], host["L"], E_cpu["lab_idx"].shape[1])
        if not self.graphs:
            outs = self._core_branch(E, host, bank)
        else:
            if key not in bank._graphs:
                bank._graphs[key] = self._capture(lambda S: self._core_branch(S, host, bank), E, owner=bank)
            g, S, outs = bank._graphs[key]
            for k, v in E.items():
                S[k].copy_(v)
            g.replay()
        return outs

    # ---------------- reference: each image alone (separate runs of the speed engine's full path, eager)
    @torch.no_grad()
    def separate(self, images, q, query_first=None):
        outs = []
        for im in images:
            batch, eng, host = self.build([im], [[q]], query_first=query_first)
            E = self.to_device(eng, host, self.feats(batch))
            res, _, _ = self._core(E, host)
            outs.append({k: v[:1].cpu() for k, v in res.items()})
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def answers(out, q, presence_T=1.0):
    """Per-image answer dicts from engine outputs (CPU)."""
    N = next(iter(out.values())).shape[0]
    res = []
    for i in range(N):
        if q["type"] == "box":
            res.append(dict(p_present=float(torch.sigmoid(out["presence_logit"][i] / presence_T)),
                            box=[round(float(x), 4) for x in out["box_pred"][i]],
                            p_iou50=float(torch.sigmoid(out["iou_logit"][i]))))
        else:
            tl = out["typed_logits"][i]
            p = torch.softmax(tl[torch.isfinite(tl)], -1)
            res.append(dict(probs=[round(float(x), 5) for x in p]))
    return res
