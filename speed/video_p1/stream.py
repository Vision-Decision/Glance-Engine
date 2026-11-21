"""Streaming one-pass step for Qwen3.5 (P1 latency prototype; timing only, untrained heads).

Each call processes one new frame per stream and N question blocks in ONE pass over the weights:
  * every token-wise op (norms, projections, MLP) runs once over [B*T frame rows + B*M*Tq question rows];
  * GDN layers: the frame continues the persistent recurrent state S and conv tail (incremental, no prefix recompute);
    the question blocks fork from the post-frame state (S_new, tail_new) and are never written back;
  * full-attention layers: frame queries see the KV window (ring buffer, W slots) + causal frame keys; question
    queries see window + whole frame + their own block (block-bidirectional, as Glance's default attn_mask);
    only the frame's K/V are written into the ring (continuous RoPE: keys are rotated at their absolute position).
Readout follows Glance: presence/box heads (dv.model.heads.BoxHeads via speed.box_readout) and the typed yes/no head
on each block's last token, plus a 30 Hz dot-product gate. Heads are randomly initialised.

Reuses glance-engine's fused-weight layout, fused elementwise ops, conv taps, FP8 linears and the FastVision tower.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from speed.engine import fuse_weights_, fused, conv_taps, fp8_language_model_, FP8Linear

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
except ImportError:  # pragma: no cover
    fla_gdr = None


_WIN = {}


def window_attention(q, k, v, scale):
    """Unmasked attention that also returns the log-sum-exp [B, H, Lq] (fp32). Picks the fastest available fused
    backend once (flash, cuDNN, memory-efficient)."""
    def flash(q, k, v):
        r = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scale)
        return r[0], r[1]

    def cudnn(q, k, v):
        r = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, False, False, scale=scale)
        return r[0], r[1]

    def eff(q, k, v):
        r = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False, scale=scale)
        return r[0], r[1][..., :q.shape[2]]

    key = (tuple(q.shape), k.shape[2])
    if key not in _WIN:
        import time
        ref = F.scaled_dot_product_attention(q, k, v, scale=scale).float()
        best = None
        for name, fn in (("flash", flash), ("cudnn", cudnn), ("efficient", eff)):
            try:
                o, l = fn(q, k, v)
                if l.shape != q.shape[:3] or (o.float() - ref).abs().max() > 2e-2:
                    continue
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(20):
                    fn(q, k, v)
                torch.cuda.synchronize()
                t = time.perf_counter() - t
                if best is None or t < best[0]:
                    best = (t, name, fn)
            except Exception:
                continue
        _WIN[key] = best
        print("window_attention backend", key, best[1], f"{best[0] / 20 * 1e3:.3f} ms", flush=True)
    o, l = _WIN[key][2](q, k, v)
    return o, l.float()


def gdr(q, k, v, g, beta, init):
    return fla_gdr(q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True)


class StreamLM:
    def __init__(self, hf, fp8=False, attn_impl="split"):
        inner = hf.model
        self.lm = inner.language_model
        self.cfg = hf.config.text_config
        self.types = list(self.cfg.layer_types)
        self.layers = list(self.lm.layers)
        fuse_weights_(self.lm, self.types, keep_ab_separate=fp8)
        self.fp8 = fp8
        if fp8:
            self.n_fp8 = fp8_language_model_(self.lm)
        self.D = self.cfg.hidden_size
        self.dev = self.lm.embed_tokens.weight.device
        self.attn_impl = attn_impl
        self.eps = self.layers[0].input_layernorm.eps
        self.n_attn = sum(t != "linear_attention" for t in self.types)

    # ------------------------------------------------------------------ state
    def alloc(self, B, W, T, M, Tq, seed=0):
        """Persistent per-stream state, initialised to a 'warm' steady state (random values, window full)."""
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.B, self.W, self.T, self.M, self.Tq = B, W, T, M, Tq
        self.S, self.tail, self.K, self.V = {}, {}, {}, {}
        for li, (layer, lt) in enumerate(zip(self.layers, self.types)):
            if lt == "linear_attention":
                m = layer.linear_attn
                Hv = m.num_v_heads
                self.S[li] = torch.randn(B, Hv, m.head_k_dim, m.head_v_dim, device=self.dev, generator=gen) * 0.02
                C = m.conv1d.weight.shape[0]
                self.tail[li] = torch.randn(B, C, m.conv_kernel_size - 1, device=self.dev, generator=gen,
                                            dtype=torch.bfloat16) * 0.1
            else:
                a = layer.self_attn
                Hkv = self.cfg.num_key_value_heads
                self.K[li] = torch.randn(B, Hkv, W, a.head_dim, device=self.dev, dtype=torch.bfloat16, generator=gen)
                self.V[li] = torch.randn(B, Hkv, W, a.head_dim, device=self.dev, dtype=torch.bfloat16, generator=gen)
        self.wptr = torch.zeros((), dtype=torch.long, device=self.dev)
        self.pos = torch.full((), W, dtype=torch.long, device=self.dev)
        self.win_valid = torch.ones(W, dtype=torch.bool, device=self.dev)
        self._build_mask()

    def _build_mask(self):
        T, M, Tq, W = self.T, self.M, self.Tq, self.W
        G = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        Rq = M * Tq
        R, C = T + Rq, W + T + Rq
        mask = torch.zeros(R, C, dtype=torch.bool, device=self.dev)
        mask[:, :W] = self.win_valid[None]
        mask[:T, W:W + T] = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
        if Rq:
            mask[T:, W:W + T] = True
            blk = torch.arange(Rq, device=self.dev) // Tq
            mask[T:, W + T:] = blk[:, None] == blk[None, :]
            if getattr(self, "q_causal", False):                                         # HF-equivalent (check only)
                mask[T:, W + T:] &= torch.ones(Rq, Rq, dtype=torch.bool, device=self.dev).tril()
        self.R, self.C, self.G = R, C, G
        # GQA folded into query rows: [Hkv, G*R] queries share one K/V head -> repeat the mask G times along rows
        self.mask = mask.repeat(G, 1)[None, None]

    def state_bytes(self):
        s = sum(t.numel() * t.element_size() for t in self.S.values())
        c = sum(t.numel() * t.element_size() for t in self.tail.values())
        kv = sum(t.numel() * t.element_size() for t in list(self.K.values()) + list(self.V.values()))
        return dict(gdn_state=s // self.B, conv_tail=c // self.B, kv_window=kv // self.B)

    # ------------------------------------------------------------------ pieces
    def _gdn_proj(self, m, h):
        if self.fp8:
            y = m._in(h)
            c, z = m.in_proj_qkv.out_features, m.in_proj_z.out_features
            mixed, zz, ab = y[:, :c], y[:, c:c + z], m._ab(h)
        else:
            y = m._in(h)
            c, z = m.in_proj_qkv.out_features, m.in_proj_z.out_features
            mixed, zz, ab = y[:, :c], y[:, c:c + z], y[:, c + z:]
        nb = m.in_proj_b.out_features
        beta, g = fused("gdn_gates")(ab[:, :nb], ab[:, nb:], m.A_log, m.dt_bias)
        return mixed, zz, beta, g

    def _gdn_core(self, m, conv, z, beta, g, init):
        n, _, L = conv.shape
        x = F.silu(conv).transpose(1, 2)
        q, k, v = torch.split(x, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(n, L, -1, m.head_k_dim)
        k = k.reshape(n, L, -1, m.head_k_dim)
        v = v.reshape(n, L, -1, m.head_v_dim)
        o, S = gdr(q, k, v, g.view(n, L, -1), beta.view(n, L, -1), init)
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
        return o.reshape(n * L, -1), S

    def _gdn(self, li, m, h):
        B, T, M, Tq = self.B, self.T, self.M, self.Tq
        Nf = B * T
        mixed, z, beta, g = self._gdn_proj(m, h)
        K = m.conv_kernel_size
        w, bias = m.conv1d.weight, m.conv1d.bias
        mf = mixed[:Nf].view(B, T, -1).transpose(1, 2)
        xin = torch.cat([self.tail[li], mf.to(self.tail[li].dtype)], dim=-1)            # [B, C, K-1+T]
        new_tail = xin[:, :, -(K - 1):]
        of, S_new = self._gdn_core(m, conv_taps(xin, w, bias), z[:Nf], beta[:Nf], g[:Nf], self.S[li])
        outs = [of]
        if M:
            mq = mixed[Nf:].view(B * M, Tq, -1).transpose(1, 2)
            qin = torch.cat([new_tail.repeat_interleave(M, 0), mq.to(new_tail.dtype)], dim=-1)
            init = S_new.repeat_interleave(M, 0)                                         # fork: snapshot per block
            oq, _ = self._gdn_core(m, conv_taps(qin, w, bias), z[Nf:], beta[Nf:], g[Nf:], init)
            outs.append(oq)
        self.tail[li].copy_(new_tail)                                                   # write: frame only
        self.S[li].copy_(S_new)
        return m.out_proj(torch.cat(outs) if M else of)

    def _attn(self, li, a, h, rope):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        B, T, M, Tq, W = self.B, self.T, self.M, self.Tq, self.W
        D2 = a.head_dim
        H, Hkv, G, R = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.G, self.R
        y = a._qkv(h)
        nq_, nk_ = a.q_proj.out_features, a.k_proj.out_features
        qg, kk, vv = y[:, :nq_], y[:, nq_:nq_ + nk_], y[:, nq_ + nk_:]
        qa, gate = torch.chunk(qg.view(-1, H, 2 * D2), 2, dim=-1)
        qa = a.q_norm(qa).view(B, R, H, D2).transpose(1, 2)
        kk = a.k_norm(kk.view(-1, Hkv, D2)).view(B, R, Hkv, D2).transpose(1, 2)
        vv = vv.view(B, R, Hkv, D2).transpose(1, 2)
        qa, kk = apply_rotary_pos_emb(qa, kk, *rope)
        qf = qa.reshape(B, Hkv, G * R, D2)                                               # GQA folded into rows
        if self.attn_impl == "split":
            # window part: every slot visible -> unmasked fused kernel with LSE; local part (frame + blocks, R keys)
            # masked in a small explicit softmax; merge the two by log-sum-exp. Exact.
            o1, l1 = window_attention(qf, self.K[li], self.V[li], a.scaling)
            s2 = torch.matmul(qf.float(), kk.float().transpose(-1, -2)) * a.scaling
            s2 = s2.masked_fill(~self.mask[..., self.W:], float("-inf"))
            l2 = torch.logsumexp(s2, -1)
            o2 = torch.matmul(torch.exp(s2 - l2[..., None]), vv.float())
            l = torch.logaddexp(l1, l2)
            o = (o1.float() * torch.exp(l1 - l)[..., None] + o2 * torch.exp(l2 - l)[..., None]).to(qf.dtype)
        else:
            k_all = torch.cat([self.K[li], kk], dim=2)                                   # [B, Hkv, W+R, D]
            v_all = torch.cat([self.V[li], vv], dim=2)
            if self.attn_impl == "sdpa_nomask":                                          # lower bound (wrong mask)
                o = F.scaled_dot_product_attention(qf, k_all, v_all, scale=a.scaling)
            else:
                o = F.scaled_dot_product_attention(qf, k_all, v_all, attn_mask=self.mask, scale=a.scaling)
        o = o.reshape(B, H, R, D2).transpose(1, 2).reshape(B * R, H * D2)
        # rows were [B, R] per stream; reorder to the global [frame rows | question rows] layout
        o = self._from_stream_major(o)
        o = o * torch.sigmoid(self._from_stream_major(gate.reshape(B * R, -1)))
        slots = (self.wptr + torch.arange(T, device=self.dev)) % W
        self.K[li].index_copy_(2, slots, kk[:, :, :T])                                   # write: frame K/V only
        self.V[li].index_copy_(2, slots, vv[:, :, :T])
        return a.o_proj(o)

    # Global token layout is [B*T frame rows][B*M*Tq question rows]; attention wants per-stream [T + M*Tq].
    def _to_stream_major(self, x):
        B, T = self.B, self.T
        if not self.M:
            return x
        f, q = x[:B * T].view(B, T, -1), x[B * T:].view(B, self.M * self.Tq, -1)
        return torch.cat([f, q], 1).reshape(B * self.R, -1)

    def _from_stream_major(self, x):
        B, T = self.B, self.T
        if not self.M:
            return x
        x = x.view(B, self.R, -1)
        return torch.cat([x[:, :T].reshape(B * T, -1), x[:, T:].reshape(B * self.M * self.Tq, -1)])

    # ------------------------------------------------------------------ one step
    def step(self, x):
        """x: [B*T + B*M*Tq, D] embedded rows (frame features, then question tokens). Returns final-normed hidden."""
        B, T, M, Tq = self.B, self.T, self.M, self.Tq
        pf = self.pos + torch.arange(T, device=self.dev)
        pq = self.pos + T + torch.arange(Tq, device=self.dev).repeat(M)
        p1 = torch.cat([pf, pq])[None]                                                   # same positions per stream
        rope = self.lm.rotary_emb(x[None, :self.R], p1[None].expand(3, 1, -1))
        h = fused("rms")(x, self.layers[0].input_layernorm.weight, self.eps)
        for li, (layer, lt) in enumerate(zip(self.layers, self.types)):
            if lt == "linear_attention":
                o = self._gdn(li, layer.linear_attn, h)
            else:
                o = self._attn(li, layer.self_attn, self._to_stream_major(h), rope)
            x, h2 = fused("add_rms")(x, o, layer.post_attention_layernorm.weight, self.eps)
            upd = layer.mlp.down_proj(fused("silu_mul")(layer._gu(h2)))
            norm = self.layers[li + 1].input_layernorm if li + 1 < len(self.layers) else self.lm.norm
            x, h = fused("add_rms")(x, upd, norm.weight, norm.eps)
        self.pos.add_(T)
        self.wptr.copy_((self.wptr + T) % self.W)
        return h


class Readout(nn.Module):
    """Glance-structured heads (random init): box/presence head over the frame's visual tokens, typed yes/no head on
    the block's last token, and a 30 Hz gate (registered question vector . mean frame hidden)."""

    def __init__(self, D, embed_w, dev):
        super().__init__()
        from dv.config import Config
        from dv.model.heads import BoxHeads
        from dv.model.typed import TypedHead
        self.box = BoxHeads(D, Config()).to(dev)
        self.typed = TypedHead(embed_w, [[9370, 2610], [13048, 9454]], D).to(dev)   # token ids arbitrary (timing)

    def forward(self, h_vis, h_qlast, q_img, gate_vec):
        from speed.box_readout import box_forward
        B, T, D = h_vis.shape
        valid = torch.ones(B, T, dtype=torch.bool, device=h_vis.device)
        side = int(T ** 0.5)
        ii = torch.arange(T, device=h_vis.device)
        centers = torch.stack([(ii % side + 0.5) / side, (ii // side + 0.5) / side], -1)[None].expand(B, T, 2)
        box = box_forward(self.box, h_qlast, h_vis, valid, centers, q_img, mode="blocks")
        yn = self.typed(h_qlast)
        gate = torch.sigmoid((gate_vec.float() * h_vis.float().mean(1, keepdim=True)).sum(-1))
        return torch.cat([torch.sigmoid(box["presence_logit"])[:, None], box["box_pred"],
                          torch.sigmoid(box["iou_logit"])[:, None], yn.softmax(-1)], -1), gate


class Pipeline:
    """Graph-capturable LM step + readout for fixed (B, T, M, Tq, W)."""

    def __init__(self, slm, heads, B, T, M, Tq, W, graphs=True):
        self.slm, self.heads, self.graphs = slm, heads, graphs
        slm.alloc(B, W, T, M, Tq)
        dev = slm.dev
        self.B, self.T, self.M, self.Tq = B, T, M, Tq
        self.x_frame = torch.randn(B * T, slm.D, device=dev, dtype=torch.bfloat16)
        self.q_ids = torch.randint(0, 200000, (B * M * Tq,), device=dev)
        self.q_img = torch.arange(B, device=dev).repeat_interleave(M)
        self.gate_vec = torch.randn(B, max(M, 1), slm.D, device=dev, dtype=torch.bfloat16)
        self.g = None

    def _run(self):
        slm = self.slm
        xq = slm.lm.embed_tokens(self.q_ids)
        x = torch.cat([self.x_frame, xq]) if self.M else self.x_frame
        h = slm.step(x)
        Nf = self.B * self.T
        if not self.M:
            return h[:Nf].view(self.B, self.T, -1)[:, -1].float().sum(-1)
        hq_last = h[Nf:].view(self.B * self.M, self.Tq, -1)[:, -1]
        out, gate = self.heads(h[:Nf].view(self.B, self.T, -1), hq_last, self.q_img, self.gate_vec)
        return torch.cat([out.reshape(-1), gate.reshape(-1)])

    @torch.no_grad()
    def __call__(self, x_frame=None):
        if x_frame is not None:
            self.x_frame.copy_(x_frame)
        if not self.graphs:
            return self._run()
        if self.g is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.out = self._run()
            torch.cuda.synchronize()
        self.g.replay()
        return self.out

    def release(self):
        self.g = None
        self.slm.S, self.slm.tail, self.slm.K, self.slm.V = {}, {}, {}, {}
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def patchify(frames, p=16, tp=2, m=2):
    """frames [B, tp, C, H, W] (normalised) -> Qwen pixel_values [B*gh*gw, C*tp*p*p] in the processor's order."""
    B, _, C, H, W = frames.shape
    gh, gw = H // p, W // p
    x = frames.reshape(B, 1, tp, C, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return x.reshape(B * gh * gw, C * tp * p * p)


__all__ = ["StreamLM", "Readout", "Pipeline", "patchify", "FP8Linear"]
