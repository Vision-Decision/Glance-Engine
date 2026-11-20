"""Fast Glance inference engine (sprint_speed).

Same math as dv.model.backbone.SharedPrefixLM + dv.model.v1forward.run_model_v1 (inference only), reorganised for speed:

  merge     LoRA merged into the bf16 base weights and the PEFT wrappers removed (no adapter matmuls at inference).
  fuse      prefix tokens and question-block tokens share ONE token buffer [Np+Nq, D]: every norm / projection / MLP
            runs once over all tokens (one GEMM instead of two; for few questions the second GEMM re-reads all
            weights for ~30 rows). Only the sequence mixers (GDN conv + delta rule, attention) see the two parts
            separately, exactly as in the reference.
  nosync    no boolean indexing / masked_scatter (host syncs): all gather/scatter indices are built on the host
            during collation, so the whole language model + heads is CUDA-graph capturable.
  graphs    the language model + heads are captured in a CUDA graph per static shape bucket (B, Lp, Nv, M, Tq);
            Lp is a multiple of 64 (left padding is exact: padded GDN inputs are zero, pads are masked keys),
            M a power of two (dummy blocks are ignored), Tq a multiple of 16.
  prefix cache  prefill() returns the per-layer prefix state (GDN recurrent state + conv tail, attention K/V) and
            the visual hidden states; ask() answers any number of new questions from a cached prefix
            (follow-up questions about the same screen skip the vision tower and the prefill).
  box       fixed-size FP32 affine blocks and a fixed FP64 center dot retain the hard patch argmax.
            GLANCE_BOX_REDUCTION=legacy selects the shipped readout.
  fp8       optional: W8A8 FP8 (e4m3) linears in the language model, per-output-channel weight scales and dynamic
            per-token activation scales (torch._scaled_mm). Changes numerics; reported separately.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dv.model import backbone as bb
from dv.model.typed import typed_logits
from speed.box_readout import box_forward


# ------------------------------------------------------------------------------------------------ weights
def merge_lora_(hf):
    """Merge every PEFT LoRA layer into its base weight and replace the wrapper by the base nn.Linear."""
    from peft.tuners.lora import LoraLayer
    n = 0
    for name, m in list(hf.named_modules()):
        if isinstance(m, LoraLayer):
            m.merge()
            parent, attr = name.rsplit(".", 1)
            setattr(hf.get_submodule(parent), attr, m.get_base_layer())
            n += 1
    return n


class FP8Linear(nn.Module):
    """y = x W^T (+b) with W in e4m3 and x quantised on the fly (torch._scaled_mm).
    rowwise: per-output-channel weight scales and per-token activation scales (needs a GPU/torch with rowwise
    scaled-mm support); otherwise one scale per tensor (dynamic activation amax)."""
    ROWWISE = None

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        if FP8Linear.ROWWISE is None:
            FP8Linear.ROWWISE = _probe_rowwise(w.device)
        if FP8Linear.ROWWISE:
            s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0          # [out, 1]
            self.register_buffer("ws", s.t().contiguous())                       # [1, out]
        else:
            s = w.abs().amax().clamp(min=1e-12) / 448.0
            self.register_buffer("ws", s.reshape(()).float())
        self.register_buffer("w8", (w / s).to(torch.float8_e4m3fn).contiguous())   # [out, in] row-major; .t() is col-major
        self.bias = None if lin.bias is None else lin.bias.data.to(torch.bfloat16)
        self.out_features = lin.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if FP8Linear.ROWWISE:
            x8, xs = quant_rowwise(x2)
        else:
            xs = (x2.abs().amax().float().clamp(min=1e-12) / 448.0).reshape(())
            x8 = (x2.float() / xs).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16,
                             bias=None if self.bias is None else self.bias.to(torch.bfloat16))
        return y.reshape(*shp[:-1], self.out_features).to(x.dtype if x.dtype != torch.float32 else torch.bfloat16)


def _quant_rowwise(x2):
    xs = x2.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (x2.float() / xs).to(torch.float8_e4m3fn), xs


_quant_compiled = None


def quant_rowwise(x2):
    """per-token e4m3 quantisation in one fused pass (torch.compile)"""
    global _quant_compiled
    if _quant_compiled is None:
        _quant_compiled = torch.compile(_quant_rowwise, dynamic=True, fullgraph=True)
    return _quant_compiled(x2)


def _probe_rowwise(dev):
    try:
        a = torch.randn(32, 64, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(48, 64, device=dev).to(torch.float8_e4m3fn)
        torch._scaled_mm(a, b.t(), scale_a=torch.ones(32, 1, device=dev), scale_b=torch.ones(1, 48, device=dev),
                         out_dtype=torch.bfloat16)
        return True
    except Exception as e:  # pragma: no cover
        print("fp8 rowwise scaled_mm unavailable, using tensorwise:", str(e)[:200], flush=True)
        return False


def fp8_language_model_(lm, skip=("in_proj_a", "in_proj_b", "._ab")):
    """(also used for the vision tower: pass net.visual). If fused projections exist (fuse_weights_), only the fused
    modules are converted; the unfused originals (views into them) stay bf16 and unused."""
    n = 0
    if any(hasattr(l, "_gu") for l in getattr(lm, "layers", [])):
        skip = tuple(skip) + ("mlp.gate_proj", "mlp.up_proj", "q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")
    for name, m in list(lm.named_modules()):
        if isinstance(m, nn.Linear) and not any(s in name for s in skip) and m.in_features % 16 == 0 and m.out_features % 16 == 0:
            parent, attr = name.rsplit(".", 1)
            setattr(lm.get_submodule(parent), attr, FP8Linear(m))
            n += 1
    return n


def _conv_taps(xpad, w, b):
    """Depthwise causal conv as a sum of K shifted taps, accumulated in fp32, result in w.dtype.
    xpad [N, C, T+K-1] -> [N, C, T]; same result as F.conv1d(xpad, w, b, groups=C) up to fp32 summation order."""
    K = w.shape[-1]
    T = xpad.shape[-1] - (K - 1)
    wf = w[:, 0, :].float()
    acc = xpad[..., 0:T].float() * wf[:, 0, None]
    for k in range(1, K):
        acc = acc + xpad[..., k:k + T].float() * wf[:, k, None]
    if b is not None:
        acc = acc + b.float()[:, None]
    return acc.to(w.dtype)


_conv_compiled = None


def conv_taps(xpad, w, b):
    global _conv_compiled
    if _conv_compiled is None:
        try:
            _conv_compiled = torch.compile(_conv_taps, dynamic=True, fullgraph=True)
        except Exception:  # pragma: no cover
            _conv_compiled = _conv_taps
    return _conv_compiled(xpad, w, b)


USE_TAPS = True


def _gdn_core_noproj(m, conv_out, z, beta, g, init, prefill=False):
    """= dv.model.backbone._gdn_core without the final out_proj (applied once over all tokens by the caller)."""
    B, _, T = conv_out.shape
    conv_out = F.silu(conv_out).transpose(1, 2)
    q, k, v = torch.split(conv_out, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    q = q.reshape(B, T, -1, m.head_k_dim)
    k = k.reshape(B, T, -1, m.head_k_dim)
    v = v.reshape(B, T, -1, m.head_v_dim)
    o, S = bb._gdr(q, k, v, g, beta, init)
    if prefill:
        from speed.lm_prefill import gated_norm
        o = gated_norm(m, o, z).reshape(B, T, -1)
    else:
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
    return o, S


# ------------------------------------------------------------------------------------------------ fused ops
def _cat_linear(mods):
    """One nn.Linear whose weight is the row-concatenation of mods' weights (no bias in Qwen3.5 LM linears)."""
    W = torch.cat([m.weight.data for m in mods], 0)
    lin = nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=W.dtype)
    lin.weight.data = W
    o = 0
    for m in mods:                       # originals become views into the fused weight (no extra memory)
        n = m.weight.shape[0]
        m.weight.data = W[o:o + n]
        o += n
    return lin


def fuse_weights_(lm, layer_types, keep_ab_separate=False):
    """Attach concatenated projections: layer._gu = [gate; up], attn._qkv = [q; k; v], gdn._in = [qkv; z; b; a]
    (or [qkv; z] + gdn._ab = [b; a] when keep_ab_separate, so FP8 can leave b/a in bf16). Pure re-layout."""
    for layer, lt in zip(lm.layers, layer_types):
        layer._gu = _cat_linear([layer.mlp.gate_proj, layer.mlp.up_proj])
        if lt == "linear_attention":
            m = layer.linear_attn
            if keep_ab_separate:
                m._in = _cat_linear([m.in_proj_qkv, m.in_proj_z])
                m._ab = _cat_linear([m.in_proj_b, m.in_proj_a])
            else:
                m._in = _cat_linear([m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a])
                m._ab = None
        else:
            a = layer.self_attn
            a._qkv = _cat_linear([a.q_proj, a.k_proj, a.v_proj])


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def _add_rms(x, o, w, eps):
    s = x + o
    return s, _rms(s, w, eps)


def _silu_mul(gu):
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


def _attention_gate(output, gate):
    return output * gate.sigmoid()


def _gdn_gates(b, a, A_log, dt_bias):
    return b.sigmoid(), -A_log.float().exp() * F.softplus(a.float() + dt_bias)


_FUSED = {}


def fused(name):
    if name not in _FUSED:
        fn = dict(rms=_rms, add_rms=_add_rms, silu_mul=_silu_mul, gdn_gates=_gdn_gates, attention_gate=_attention_gate)[name]
        _FUSED[name] = torch.compile(fn, dynamic=True, fullgraph=True)
    return _FUSED[name]


# ------------------------------------------------------------------------------------------------ collation
def bucket_pow2(n):
    return 1 << max(0, (n - 1).bit_length())


def question_rows(n, compact=False):
    if compact and n <= 64:
        return max(16, bucket_pow2(n))
    if compact and 64 < n <= 96:
        return 96
    return -(-max(n, 1) // 64) * 64


def prepare(batch, Lp_bucket=64, M_bucket=True, Nv=None, nl=26, compact_questions=False):
    """Host-side: turn a collate_mixed batch (CPU) into static-shape engine inputs + host metadata.
    Adds left padding of the prefix to a multiple of Lp_bucket, pads the question blocks to a power of two,
    pads the visual-token axis to Nv and the label axis to nl, and precomputes every gather/scatter index."""
    B, Lp0 = batch["p_ids"].shape
    Lp = -(-Lp0 // Lp_bucket) * Lp_bucket
    padL = Lp - Lp0
    lpad = lambda t, v: F.pad(t, (padL, 0), value=v) if padL else t
    p_ids, p_valid, p_img = lpad(batch["p_ids"], 0), lpad(batch["p_valid"], False), lpad(batch["p_img_mask"], False)
    p_pos = F.pad(batch["p_pos"], (padL, 0)) if padL else batch["p_pos"]
    M0, Tq = batch["q_ids"].shape
    M = bucket_pow2(M0) if M_bucket else M0
    padM = M - M0
    q_ids = F.pad(batch["q_ids"], (0, 0, 0, padM))
    q_valid = F.pad(batch["q_valid"], (0, 0, 0, padM))
    q_pos = F.pad(batch["q_pos"], (0, 0, 0, padM))
    q_img = F.pad(batch["q_img"], (0, padM))
    q_last = F.pad(batch["q_last"], (0, padM))
    nv0 = batch["vis_valid"].shape[1]
    Nv = Nv or nv0
    assert Nv >= nv0
    vis_valid = F.pad(batch["vis_valid"], (0, Nv - nv0), value=False)
    centers = F.pad(batch["centers"], (0, 0, 0, Nv - nv0))
    # flat gather index of visual tokens in the [B*Lp] prefix (pads -> 0, masked)
    vis_gather = torch.zeros(B, Nv, dtype=torch.long)
    img_flat = []
    for b in range(B):
        pos = p_img[b].nonzero()[:, 0]
        vis_gather[b, :len(pos)] = b * Lp + pos
        img_flat.append(b * Lp + pos)
    img_flat = torch.cat(img_flat) if img_flat else torch.zeros(0, dtype=torch.long)
    lab = batch["lab_idx"]
    lab_idx = torch.full((M, nl), -1, dtype=torch.long)
    lab_idx[:M0, :lab.shape[1]] = lab
    # Pack valid question tokens for the linears. Padding rows point at the unused M*Tq slot.
    MT = M * Tq
    vidx = q_valid.reshape(-1).nonzero()[:, 0]
    nq = len(vidx)
    Nqb = question_rows(nq, compact_questions)
    q_tok_idx = torch.full((Nqb,), MT, dtype=torch.long)
    q_tok_idx[:nq] = vidx
    q_ids_packed = torch.zeros(Nqb, dtype=torch.long)
    q_ids_packed[:nq] = q_ids.reshape(-1)[vidx]
    pos_in_packed = torch.zeros(MT, dtype=torch.long)
    pos_in_packed[vidx] = torch.arange(nq)
    q_last_packed = pos_in_packed[torch.arange(M) * Tq + q_last]
    eng = dict(p_ids=p_ids, p_valid=p_valid, p_pos=p_pos, img_flat=img_flat, q_valid=q_valid, q_pos=q_pos,
               q_img=q_img, vis_valid=vis_valid, centers=centers, vis_gather=vis_gather, lab_idx=lab_idx,
               q_tok_idx=q_tok_idx, q_ids_packed=q_ids_packed, q_last_packed=q_last_packed,
               q_offsets=F.pad(q_valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0)))
    if B > 1:
        # Graph replay can change both question order and the number per image.
        eng["branch_order"] = torch.argsort(q_img, stable=True).to(torch.int32)
        eng["image_offsets"] = F.pad(torch.bincount(q_img, minlength=B).cumsum(0, dtype=torch.int32), (1, 0))
    host = dict(M0=M0, qtype=batch["qtype"], B=B, Lp=Lp, M=M, Tq=Tq, Nv=Nv, Nqb=Nqb, n_qtok=nq, nopad=bool(p_valid.all()))
    return eng, host


GRAPH_MAX_TOKENS = 3072      # language-model tokens per call above which the engine runs eager (GPU-bound anyway)
GRAPH_MAX_PATCHES = 16384    # vision patches per call above which (or with mixed grids) the vision tower runs eager


def evict_if_full(obj, frac=0.8):
    """Small GPUs: drop every captured graph of obj (and its pool) when reserved memory exceeds frac of the device;
    graphs are re-captured on demand (capture happens in the untimed warm-up passes of the benchmark)."""
    tot = torch.cuda.get_device_properties(0).total_memory
    if torch.cuda.memory_reserved() > frac * tot and obj._graphs:
        obj._graphs.clear()
        obj._pool = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ vision
def vision_rotary(q, k, cos, sin):
    from speed.vision_rotary import apply
    return apply(q, k, cos, sin)


def image_attention(q, k, v, streams, scale=None):
    """Apply HF's per-image SDPA on disjoint rows. Join all writes before returning to the caller."""
    batch = q.shape[0]
    # Allocate the result on the caller stream. Worker-local SDPA temporaries die on their own stream.
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    main = torch.cuda.current_stream(q.device)
    for i, stream in enumerate(streams):
        if stream != main:
            stream.wait_stream(main)
        with torch.cuda.stream(stream):
            for j in range(i, batch, len(streams)):
                value = F.scaled_dot_product_attention(q[j].transpose(0, 1)[None],
                         k[j].transpose(0, 1)[None], v[j].transpose(0, 1)[None], scale=scale)
                out[j].copy_(value[0].transpose(0, 1))
    for stream in streams:
        if stream != main:
            main.wait_stream(stream)
    return out


class FastVision:
    """Graph-capturable Qwen vision tower with grid-keyed position and rotary tables.
    The fused path preserves HF's patch convolution and per-image attention reduction.
    The legacy path uses a flattened patch GEMM and batches SDPA for equal grids."""

    def __init__(self, visual, graphs=True, optimized=False, optimized_singleton=False, fused_rotary=False):
        import transformers.models.qwen3_5.modeling_qwen3_5 as Q
        self.Q, self.v, self.graphs = Q, visual, graphs
        pe = visual.patch_embed.proj
        self.W = pe.weight.reshape(pe.weight.shape[0], -1)
        self.fused_rotary = fused_rotary
        self.mlp_table = None
        self.mlp_projection = None
        if ((optimized or fused_rotary) and os.getenv("GLANCE_FRAME_OPT", "1") != "0"
                and self.W.is_cuda and torch.cuda.get_device_capability(self.W.device) == (12, 0)
                and getattr(visual.config, "hidden_act", "") in ("gelu", "gelu_pytorch_tanh", "gelu_new")
                and all(type(b.mlp.linear_fc1) is nn.Linear and b.mlp.linear_fc1.weight.shape == (4096, 1024)
                        and b.mlp.linear_fc1.weight.dtype == torch.bfloat16 and b.mlp.linear_fc1.bias is not None
                        and b.mlp.linear_fc1.weight.is_contiguous() for b in visual.blocks)):
            try:
                from speed.vision_mlp import activation_table, fc1
            except ImportError:
                pass  # Earlier Triton versions have no tensor descriptors.
            else:
                self.mlp_projection = fc1
                self.mlp_table = activation_table(visual.blocks[0].mlp)
        self.optimized = optimized
        self.optimized_singleton = optimized_singleton
        self.attention_streams = []
        self.b = pe.bias
        self.tables, self._graphs, self._pool = {}, {}, None
        self.graph_limit = None
        self.H = visual.blocks[0].attn.num_heads

    def _consts(self, grid_cpu, dev):
        key = tuple(tuple(int(x) for x in g) for g in grid_cpu.tolist())
        if key not in self.tables:
            v, Q = self.v, self.Q
            g = grid_cpu.to(dev)
            idx, w = Q.get_vision_interpolation_indices_and_weights(
                g, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
                align_corners=v.interpolation_align_corners, spatial_merge_size=v.config.spatial_merge_size, kwargs={})
            pos = (v.pos_embed(idx) * w[:, :, None]).sum(1)
            pid = Q.get_vision_position_ids(g, v.spatial_merge_size, kwargs={})
            cos, sin = v.rotary_pos_emb(pos, pid)
            lens = [int(t * h * w_) for t, h, w_ in key]
            self.tables[key] = (pos, cos, sin, lens)
        return key, self.tables[key]

    def _forward(self, px, pos, cos, sin, lens):
        v, Q = self.v, self.Q
        use_fused = self.optimized or (self.optimized_singleton and len(lens) == 1)
        if use_fused:
            # The reference Conv3d and the equivalent flattened GEMM can round differently in BF16.
            x = v.patch_embed(px)
        else:
            x = F.linear(px.to(self.W.dtype), self.W, self.b)
        x = x + pos.to(x.dtype)
        N = x.shape[0]
        same = len(set(lens)) == 1
        streams = ()
        if same and self.optimized and len(lens) > 1 and x.is_cuda:
            streams = self.attention_streams[:self._worker_count(lens)]
        for blk in v.blocks:
            h = blk.norm1(x)
            q, k, vv = blk.attn.qkv(h).reshape(N, 3, self.H, -1).permute(1, 0, 2, 3).unbind(0)
            if (use_fused or self.fused_rotary) and q.is_cuda and q.shape[2] % 2 == 0:
                q, k = vision_rotary(q, k, cos, sin)
            else:
                q, k = Q.apply_rotary_pos_emb_vision(q, k, cos, sin)
            # HF issues SDPA per image. Batched flash can change the split-KV reduction order.
            if streams:
                shape = (len(lens), lens[0], self.H, -1)
                o = image_attention(q.reshape(shape), k.reshape(shape), vv.reshape(shape), streams, blk.attn.scaling)
                o = o.reshape(N, -1)
            elif same and (not self.optimized or len(lens) == 1):
                B, L = len(lens), lens[0]
                f = lambda t: t.reshape(B, L, self.H, -1).transpose(1, 2)
                o = F.scaled_dot_product_attention(f(q), f(k), f(vv), scale=blk.attn.scaling)
                o = o.transpose(1, 2).reshape(N, -1)
            else:
                outs, s0 = [], 0
                for L in lens:
                    f = lambda t: t[s0:s0 + L].transpose(0, 1)[None]
                    outs.append(F.scaled_dot_product_attention(f(q), f(k), f(vv), scale=blk.attn.scaling)[0].transpose(0, 1).reshape(L, -1))
                    s0 += L
                o = torch.cat(outs)
            x = x + blk.attn.proj(o)
            h = blk.norm2(x)
            if self.mlp_projection is not None and N >= 4096 and len(lens) > 1:
                intermediate = self.mlp_projection(h, blk.mlp.linear_fc1, self.mlp_table)
                x = x + blk.mlp.linear_fc2(intermediate)
            else:
                x = x + blk.mlp(h)
        return v.merger(x)

    @staticmethod
    def _worker_count(lens):
        if lens[0] > 512:
            return min(len(lens), 2)
        return min(len(lens), 4 if len(lens) <= 32 else 8)

    @torch.no_grad()
    def __call__(self, pixel_values, grid_cpu):
        if self.graph_limit is not None and len(self.tables) > self.graph_limit:
            # Graph captures retain device pointers into these position tables.
            self.tables = {k: v for k, v in self.tables.items() if k in self._graphs}
        key, (pos, cos, sin, lens) = self._consts(grid_cpu, pixel_values.device)
        if self.optimized and pixel_values.is_cuda and len(lens) > 1:
            # Short spatial groups leave more idle SMs per singleton attention launch.
            workers = self._worker_count(lens)
            while len(self.attention_streams) < workers:
                self.attention_streams.append(torch.cuda.Stream(device=pixel_values.device))
        small = len(set(key)) == 1 and sum(lens) <= GRAPH_MAX_PATCHES
        if self.graph_limit is not None and key not in self._graphs and len(self._graphs) >= self.graph_limit:
            small = False
        if self.graphs and small and key not in self._graphs:
            evict_if_full(self)
        if self.graphs and not small:
            evict_if_full(self, 0.5)
        if not self.graphs or not small:
            return self._forward(pixel_values, pos, cos, sin, lens)
        if key not in self._graphs:
            S = pixel_values.clone()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._forward(S, pos, cos, sin, lens)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self._forward(S, pos, cos, sin, lens)
            self._graphs[key] = (g, S, out)
        g, S, out = self._graphs[key]
        S.copy_(pixel_values)
        g.replay()
        return out


# ------------------------------------------------------------------------------------------------ engine
def _supports_prefill(layer, kind):
    if not isinstance(layer._gu, nn.Linear) or layer._gu.weight.dtype != torch.bfloat16:
        return False
    if kind != "linear_attention":
        return layer.self_attn.head_dim == 256
    m = layer.linear_attn
    return (m.head_v_dim == m.head_k_dim == 128 and m.key_dim % 128 == m.value_dim % 128 == 0
            and hasattr(m.norm, "variance_epsilon") and m.norm.weight.dtype == torch.bfloat16
            and not isinstance(m._in, FP8Linear))


class GlanceEngine:
    def __init__(self, net, heads, graphs=True, fast_vision=True, fast_ops=False, optimized_singleton=True, lm_prefill=True):
        self.net, self.heads = net, heads
        self.box_reduction = os.getenv("GLANCE_BOX_REDUCTION", "blocks")
        if self.box_reduction not in ("fixed", "tiled", "blocks", "legacy"):
            raise ValueError("GLANCE_BOX_REDUCTION must be fixed, tiled, blocks or legacy")
        self.trace_centers = os.getenv("GLANCE_TRACE_CENTERS") == "1"
        self.lm = net.lm
        self.layers = list(self.lm.layers)
        self.types = net.layer_types
        self.graphs = graphs
        self.compact_questions = bool(getattr(self.lm, "_w12_compact", False))
        self.dev = next(self.lm.parameters()).device
        self._graphs = {}
        self.graph_limit = None
        self._pool = None
        self.mask_mode = net.cfg.attn_mask
        frame_rotary = bool(fast_ops and optimized_singleton and os.getenv("GLANCE_FRAME_OPT", "1") != "0")
        self.fv = (FastVision(net.visual, graphs=graphs, optimized_singleton=optimized_singleton,
                              fused_rotary=frame_rotary) if fast_vision else None)
        self.fast_ops = fast_ops and all(hasattr(l, "_gu") for l in self.layers)
        self.lm_prefill = lm_prefill and self.fast_ops and self.dev.type == "cuda" and (
            self.lm.embed_tokens.weight.dtype == torch.bfloat16 and all(
                _supports_prefill(layer, kind) for layer, kind in zip(self.layers, self.types)))
        self.question_branch = None
        self.branch_output = None
        self.branch_attention = None
        self.native_gate_up = None
        self.flash_gdn = None
        if self.lm_prefill and torch.cuda.get_device_capability(self.dev) == (12, 0):
            from speed.lm_prefill import gate_up
            from speed.flash_gdn import load_flash_gdn
            branch_mode = os.environ.get("GLANCE_BRANCH_MODE", "fused")
            if branch_mode in ("wy", "fused"):
                from speed.gdn_branch import branch_output
                self.branch_output = branch_output
                self.question_branch = branch_output
            if branch_mode == "fused":
                from speed.gdn_branch import branch_attention, load_chunk_branch
                self.branch_attention = branch_attention
                self.question_branch = load_chunk_branch()
            self.native_gate_up = gate_up
            self.flash_gdn = load_flash_gdn(uniform_frames=os.getenv("GLANCE_FRAME_OPT", "1") != "0")

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda")

    @torch.no_grad()
    def answer_independent(self, batch, feats):
        """Answer one question per image in collator order, without retaining any prefix states."""
        from scan.varlen import FreshPack, packed_forward
        return packed_forward(self, FreshPack(batch, self.dev), feats)

    # ---------------- vision
    @torch.no_grad()
    def vision(self, pixel_values, grid_cpu):
        """-> [sum of merged visual tokens, D]. grid_cpu: [n_img, 3] CPU tensor."""
        if self.fv is not None:
            return self.fv(pixel_values, grid_cpu)
        return torch.cat(self.net.image_features(pixel_values, grid_cpu.to(pixel_values.device)))

    # ---------------- language model core (graph-capturable; all inputs on device)
    def _mix_gdn_prefix(self, m, mixed, z, beta, g, B, Lp):
        K = m.conv_kernel_size
        w = m.conv1d.weight
        mixed = mixed.view(B, Lp, -1).transpose(1, 2)
        if USE_TAPS:
            conv = conv_taps(F.pad(mixed.to(w.dtype), (K - 1, 0)), w, m.conv1d.bias)
        else:
            conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :Lp]
        o, S = _gdn_core_noproj(m, conv, z.view(B, Lp, -1, m.head_v_dim), beta.view(B, Lp, -1), g.view(B, Lp, -1), None, self.lm_prefill)
        tail = mixed[:, :, Lp - (K - 1):]
        return o, S, tail

    def _mix_gdn_q(self, m, mixed, z, beta, g, M, Tq, S, tail, q_img):
        w = m.conv1d.weight
        mixed = mixed.transpose(1, 2)
        mq = torch.cat([tail[q_img], mixed.to(tail.dtype)], dim=-1).to(w.dtype)
        conv = conv_taps(mq, w, m.conv1d.bias) if USE_TAPS else F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])
        o, _ = _gdn_core_noproj(m, conv, z.reshape(M, Tq, -1, m.head_v_dim), beta, g, S[q_img].contiguous())
        return o

    def _qkv(self, m, h, cos, sin, n, T):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        q, gate = torch.chunk(m.q_proj(h).view(n, T, -1, m.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(n, T, -1)
        q = m.q_norm(q).transpose(1, 2)
        k = m.k_norm(m.k_proj(h).view(n, T, -1, m.head_dim)).transpose(1, 2)
        v = m.v_proj(h).view(n, T, -1, m.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, gate

    @staticmethod
    def _gdn_project_raw(m, h):
        y = m._in(h)
        c, z = m.in_proj_qkv.out_features, m.in_proj_z.out_features
        ab = m._ab(h) if m._ab is not None else y[:, c + z:]
        return y[:, :c], y[:, c:c + z], ab

    def _gdn_projections(self, m, h):
        if self.fast_ops:
            mixed, z, ab = self._gdn_project_raw(m, h)
            nb = m.in_proj_b.out_features
            beta, g = fused("gdn_gates")(ab[:, :nb], ab[:, nb:], m.A_log, m.dt_bias)
        else:
            mixed = m.in_proj_qkv(h)
            z = m.in_proj_z(h)
            beta = m.in_proj_b(h).sigmoid()
            g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(h).float() + m.dt_bias)
        return mixed, z, beta, g

    def _attention_projections(self, m, h, rope=None):
        D2 = m.head_dim
        if self.fast_ops:
            y = m._qkv(h)
            nq_, nk_ = m.q_proj.out_features, m.k_proj.out_features
            qg, kk, vv = y[:, :nq_], y[:, nq_:nq_ + nk_], y[:, nq_ + nk_:]
        else:
            qg, kk, vv = m.q_proj(h), m.k_proj(h), m.v_proj(h)
        qa, gate = torch.chunk(qg.view(-1, qg.shape[-1] // (2 * D2), 2 * D2), 2, dim=-1)
        kk = kk.view(-1, kk.shape[-1] // D2, D2)
        if self.lm_prefill:
            from speed.lm_prefill import head_norm
            qa = head_norm(qa, m.q_norm, rope)
            kk = head_norm(kk, m.k_norm, rope)
        else:
            qa = m.q_norm(qa)
            kk = m.k_norm(kk)
        vv = vv.view(-1, vv.shape[-1] // D2, D2)
        return qa, kk, vv, (gate if rope is not None else gate.reshape(gate.shape[0], -1))

    def _mlp_update(self, layer, h):
        if (self.lm_prefill and self.native_gate_up is not None and len(h) >= 4096 and h.shape[1] == 2560
                and h.is_contiguous() and layer._gu.bias is None and layer._gu.weight.shape == (18432, 2560)
                and layer._gu.weight.is_contiguous()):
            return layer.mlp.down_proj(self.native_gate_up(h, layer._gu.weight))
        return layer.mlp.down_proj(fused("silu_mul")(layer._gu(h)))

    def lm_forward(self, x, P, Q, cache=None):
        """x: [Np+Nqb, D] embedded tokens: prefix rows [B*Lp] first (Np = 0 when cache is given), then the PACKED
        valid question tokens. P: prefix inputs (B, Lp, p_valid, rope, mask_p, nopad) or None.
        Q: question inputs (M, Tq, q_img, rope [padded layout], mask_q, q_tok_idx).
        Token-wise ops run on all rows at once; the mixers see the prefix as [B, Lp] and the questions scattered
        into the padded block layout [M, Tq] (pads are zero rows; they never influence valid tokens: GDN is causal
        and pads are masked keys in attention). Returns final-normed hidden [Np+Nqb, D] and the prefix cache."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        packed = P.get("packed") if P is not None else None
        Np = len(packed.rows) if packed is not None else (P["B"] * P["Lp"] if P is not None else 0)
        if Np and packed is None:
            B, Lp = P["B"], P["Lp"]
            pv = torch.cat([P["p_valid"].reshape(-1), torch.ones(x.shape[0] - Np, dtype=torch.bool, device=x.device)])
            pvf = pv[:, None].to(x.dtype)
        M, Tq, q_img, qidx = Q["M"], Q["Tq"], Q["q_img"], Q["q_tok_idx"]
        MT = M * Tq

        def scat(t):                       # [Nqb, ...] packed -> [M, Tq, ...] padded (zeros at pads)
            buf = t.new_zeros((MT + 1,) + tuple(t.shape[1:]))
            buf.index_copy_(0, qidx, t)
            return buf[:MT].view((M, Tq) + tuple(t.shape[1:]))

        def gath(t):                       # [M, Tq, ...] padded -> [Nqb, ...] packed
            flat = t.reshape((MT,) + tuple(t.shape[2:]))
            flat = torch.cat([flat, flat.new_zeros((1,) + tuple(flat.shape[1:]))])
            return flat.index_select(0, qidx)

        new_cache = []
        fo = self.fast_ops
        eps = self.layers[0].input_layernorm.eps
        h_next = fused("rms")(x, self.layers[0].input_layernorm.weight, eps) if fo else None
        for li, (layer, lt) in enumerate(zip(self.layers, self.types)):
            h = h_next if fo else layer.input_layernorm(x)
            if lt == "linear_attention":
                m = layer.linear_attn
                if Np and packed is None:
                    h = h * pvf                                  # reference: pads of the prefix are zeroed
                mixed, z, beta, g = self._gdn_projections(m, h)
                outs = []
                if Np:
                    args = (m, mixed[:Np], z[:Np], beta[:Np], g[:Np])
                    op, S, tail = (packed.gdn(*args) if packed is not None else self._mix_gdn_prefix(*args, B, Lp))
                    outs.append(op.reshape(Np, -1))
                    new_cache.append((S, tail))
                else:
                    S, tail = cache[li]
                if (self.question_branch is not None and Q.get("offsets") is not None and Tq <= 128
                        and tail.shape[-1] == m.conv_kernel_size - 1):
                    oq = self.question_branch(m, mixed[Np:], z[Np:], beta[Np:], g[Np:], S, tail,
                                            M, Tq, Q["offsets"], q_img, groups=Q.get("groups", (None, None)))
                else:
                    oq = gath(self._mix_gdn_q(m, scat(mixed[Np:]), scat(z[Np:]), scat(beta[Np:]),
                                              scat(g[Np:]), M, Tq, S, tail, q_img))
                outs.append(oq)
                o = m.out_proj(torch.cat(outs) if len(outs) > 1 else outs[0])
            else:
                m = layer.self_attn
                D2 = m.head_dim
                qa, kk, vv, gate = self._attention_projections(m, h)
                outs = []
                if packed is not None:
                    op, kp, vp = packed.attention(m, qa[:Np], kk[:Np], vv[:Np], gate[:Np], P["rope"])
                    outs.append(op)
                    new_cache.append((kp, vp))
                elif Np:
                    qp, kp = apply_rotary_pos_emb(qa[:Np].view(B, Lp, -1, D2).transpose(1, 2),
                                                  kk[:Np].view(B, Lp, -1, D2).transpose(1, 2), *P["rope"])
                    vp = vv[:Np].view(B, Lp, -1, D2).transpose(1, 2)
                    if P["nopad"]:
                        op = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True, scale=m.scaling, enable_gqa=True)
                    else:
                        op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=P["mask_p"], scale=m.scaling, enable_gqa=True)
                    outs.append((op.transpose(1, 2).reshape(Np, -1) * torch.sigmoid(gate[:Np])))
                    new_cache.append((kp, vp))
                else:
                    kp, vp = cache[li]
                if self.branch_attention is not None and "packed_rope" in Q:
                    # The prefix above already used normalized Q/K. Use its unrotated
                    # question heads here and apply the packed rotary tables directly.
                    qq, kq = apply_rotary_pos_emb(qa[Np:].transpose(0, 1)[None], kk[Np:].transpose(0, 1)[None],
                                                  *Q["packed_rope"])
                    qgate = gate[Np:].reshape(-1, qa.shape[1], D2)
                    oq = self.branch_attention(qq[0].transpose(0, 1).contiguous(), kq[0].transpose(0, 1).contiguous(),
                                               vv[Np:], qgate, kp, vp, Q["p_valid"], m.scaling, M, Tq,
                                               self.mask_mode != "block_bidir", Q["offsets"], q_img)
                    outs.append(oq)
                else:
                    qq, kq = apply_rotary_pos_emb(scat(qa[Np:]).transpose(1, 2), scat(kk[Np:]).transpose(1, 2), *Q["rope"])
                    vq = scat(vv[Np:]).transpose(1, 2)
                    k_all = torch.cat([kp[q_img], kq], dim=2)
                    v_all = torch.cat([vp[q_img], vq], dim=2)
                    oq = F.scaled_dot_product_attention(qq, k_all, v_all, attn_mask=Q["mask_q"], scale=m.scaling, enable_gqa=True)
                    oq = gath(oq.transpose(1, 2).reshape(M, Tq, -1))
                    outs.append(oq * torch.sigmoid(gate[Np:]))
                o = m.o_proj(torch.cat(outs) if len(outs) > 1 else outs[0])
            if fo:
                x, h2 = fused("add_rms")(x, o, layer.post_attention_layernorm.weight, eps)
                update = self._mlp_update(layer, h2)
                if self.lm_prefill:
                    norm = self.layers[li + 1].input_layernorm if li + 1 < len(self.layers) else self.lm.norm
                    x, h_next = fused("add_rms")(x, update, norm.weight, norm.eps)
                    continue
                x = x + update
                nxt = self.layers[li + 1].input_layernorm.weight if li + 1 < len(self.layers) else None
                if nxt is not None:
                    h_next = fused("rms")(x, nxt, eps)
            else:
                x = x + o
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        return (h_next if self.lm_prefill else self.lm.norm(x)), new_cache

    # ---------------- heads (graph-capturable)
    def heads_forward(self, hp_flat, hq, E, vis_override=None):
        hq_last = hq[E["q_last_packed"]]
        if vis_override is None:
            vis = hp_flat[E["vis_gather"]] * E["vis_valid"][..., None].to(hp_flat.dtype)
        else:
            vis = vis_override
        args = (hq_last, vis, E["vis_valid"], E["centers"], E["q_img"])
        if self.box_reduction != "legacy":
            box = box_forward(self.heads.box, *args, mode=self.box_reduction)
        else:
            box = self.heads.box(*args)
        lab = self.heads.typed(hq_last)
        tl = typed_logits(lab, E["lab_idx"])
        out = dict(presence_logit=box["presence_logit"].float(), box_pred=box["box_pred"].float(),
                   iou_logit=box["iou_logit"].float(), typed_logits=tl.float())
        if self.trace_centers:
            out["center_idx"] = box["pred_idx"]
        return out, vis

    # ---------------- inputs
    def _inputs(self, E, host, with_prefix=True):
        lm = self.lm
        Q = dict(M=host["M"], Tq=host["Tq"], q_img=E["q_img"], q_tok_idx=E["q_tok_idx"])
        xq = lm.embed_tokens(E["q_ids_packed"])
        Q["offsets"] = E.get("q_offsets")
        if self.branch_output is not None and Q["offsets"] is None:
            lengths = E["q_valid"].sum(1, dtype=torch.int32)
            Q["offsets"] = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        Q["groups"] = (E.get("branch_order"), E.get("image_offsets"))
        Q["rope"] = self.net.rope(xq[None], E["q_pos"])
        p_valid = E["p_valid"]
        if self.branch_attention is not None and host["Tq"] <= 128:
            Q["p_valid"] = p_valid
            idx = E["q_tok_idx"].clamp_max(host["M"] * host["Tq"] - 1)
            Q["packed_rope"] = tuple(r.reshape(-1, r.shape[-1])[idx][None] for r in Q["rope"])
        else:
            to_prefix = p_valid[E["q_img"]][:, None, :].expand(host["M"], host["Tq"], host["Lp"])
            qv = E["q_valid"]
            if self.mask_mode == "block_bidir":
                within = qv[:, None, :].expand(host["M"], host["Tq"], host["Tq"])
            else:
                within = torch.ones(host["Tq"], host["Tq"], dtype=torch.bool, device=qv.device).tril()[None] & qv[:, None, :]
            within = within | torch.eye(host["Tq"], dtype=torch.bool, device=qv.device)[None]
            Q["mask_q"] = torch.cat([to_prefix, within], dim=-1)[:, None]
        if not with_prefix:
            return None, Q, xq
        B, Lp = host["B"], host["Lp"]
        P = dict(B=B, Lp=Lp, p_valid=p_valid, nopad=host["nopad"])
        xp = lm.embed_tokens(E["p_ids"]).reshape(B * Lp, -1)
        xp = torch.where(E["img_rowmask"][:, None], E["featbuf"], xp)
        P["rope"] = self.net.rope(xp.view(B, Lp, -1), E["p_pos"])
        if not host["nopad"]:
            causal = torch.ones(Lp, Lp, dtype=torch.bool, device=xp.device).tril()
            eye = torch.eye(Lp, dtype=torch.bool, device=xp.device)
            P["mask_p"] = ((causal[None] & p_valid[:, None, :]) | eye[None])[:, None]
        return P, Q, torch.cat([xp, xq])

    def _core(self, E, host):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P, Q, x = self._inputs(E, host)
            h, cache = self.lm_forward(x, P, Q)
        Np = host["B"] * host["Lp"]
        out, vis = self.heads_forward(h[:Np], h[Np:], E)     # heads in fp32, outside autocast (as the reference)
        return out, cache, vis

    def _core_q(self, E, host):
        cache = [(E[f"c{i}a"], E[f"c{i}b"]) for i in range(len(self.layers))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, Q, x = self._inputs(E, host, with_prefix=False)
            h, _ = self.lm_forward(x, None, Q, cache=cache)
        out, _ = self.heads_forward(None, h, E, vis_override=E["vis"])
        return out

    # ---------------- public API
    def to_device(self, eng, host, feats):
        """Copy host inputs to the GPU and place the visual features into a [B*Lp, D] buffer (outside any graph)."""
        E = {k: v.to(self.dev, non_blocking=True) for k, v in eng.items()}
        B, Lp = host["B"], host["Lp"]
        D = self.net.hidden
        fb = torch.zeros(B * Lp, D, dtype=torch.bfloat16, device=self.dev)
        rm = torch.zeros(B * Lp, dtype=torch.bool, device=self.dev)
        if feats is not None:
            fb.index_copy_(0, E["img_flat"], feats.to(torch.bfloat16))
            rm[E["img_flat"]] = True
        E.pop("img_flat")                      # variable length: used here only, never inside a graph
        E["featbuf"], E["img_rowmask"] = fb, rm
        return E

    def _key(self, host, kind):
        return (kind, self.box_reduction, self.trace_centers, host["B"], host["Lp"], host["Nv"], host["M"], host["Tq"], host["Nqb"], host["nopad"])

    @torch.no_grad()
    def run(self, E, host):
        """Full call: prefix + question blocks + heads. Returns (outputs dict, prefix cache, vis).
        In graph mode the returned tensors are static graph outputs: clone them (keep_prefix) before the next call."""
        if not self.graphs or host["B"] * host["Lp"] + host["Nqb"] > GRAPH_MAX_TOKENS:
            evict_if_full(self, 0.5)          # large eager calls need room: drop captured graphs first if memory is tight
            return self._core(E, host)      # large calls are GPU-bound: eager, no per-shape capture
        key = self._key(host, "full")
        if key not in self._graphs:
            if self.graph_limit is not None and len(self._graphs) >= self.graph_limit:
                return self._core(E, host)
            self._graphs[key] = self._capture(lambda S: self._core(S, host), E)
        g, S, outs = self._graphs[key]
        for k, v in E.items():
            S[k].copy_(v)
        g.replay()
        return outs

    @torch.no_grad()
    def run_q(self, E, host, cache, vis, *, host_inputs=False):
        """Question blocks only, from a cached prefix (cache, vis from a previous run())."""
        key = self._key(host, "singleton_q" if host_inputs else "q")
        if host_inputs and (not self.graphs or key not in self._graphs or host["Nqb"] > GRAPH_MAX_TOKENS):
            E = {k: v.to(self.dev, non_blocking=True) for k, v in E.items()}
        E = dict(E, vis=vis, **{f"c{i}a": c[0] for i, c in enumerate(cache)}, **{f"c{i}b": c[1] for i, c in enumerate(cache)})
        if not self.graphs or host["Nqb"] > GRAPH_MAX_TOKENS:
            return self._core_q(E, host)
        if key not in self._graphs:
            if self.graph_limit is not None and len(self._graphs) >= self.graph_limit:
                return self._core_q(E, host)
            self._graphs[key] = self._capture(lambda S: self._core_q(S, host), E)
        g, S, outs = self._graphs[key]
        for k, v in E.items():
            S[k].copy_(v, non_blocking=True)
        g.replay()
        return outs

    def _capture(self, fn, E, owner=None):
        # The pool handle must have the same lifetime as the graphs using it.
        owner = self if owner is None else owner
        evict_if_full(owner)
        S = {k: v.clone() for k, v in E.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):           # warm up (Triton autotune, cuBLAS handles) outside the graph
                fn(S)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if not owner._graphs or owner._pool is None:
            owner._pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=owner._pool):
            outs = fn(S)
        torch.cuda.synchronize()
        return g, S, outs
