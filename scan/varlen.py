"""Fresh LM prefill. A sequence is one image prefix immediately followed by its question."""
from itertools import accumulate
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dv.model import backbone
from speed.engine import fused

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None


class Segments:
    def __init__(self, lengths, device, *, fixed_width=False):
        self.fixed_width = fixed_width
        self._starts = None
        self._chunk_offsets32 = None
        self.lengths = lengths
        self.minimum = min(lengths)
        self.maximum = max(lengths)
        self.cpu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
        self.cu = self.cpu.to(device)

    def token_starts(self):
        if self._starts is None:
            lengths = self.cpu.diff().long()
            self._starts = self.cpu[:-1].repeat_interleave(lengths).to(self.cu.device)
        return self._starts

    def chunk_offsets32(self):
        if self._chunk_offsets32 is None:
            chunks = [(n + 31) // 32 for n in self.lengths]
            self._chunk_offsets32 = torch.tensor([0, *accumulate(chunks)], dtype=torch.int32).to(self.cu.device)
        return self._chunk_offsets32


class FreshPack:
    """Translate the collator's positions and head indices into a pad-free, image-major layout."""
    def __init__(self, batch, device):
        p, q = batch["p_valid"], batch["q_valid"]
        if not torch.equal(batch["q_img"], torch.arange(len(p))):
            raise ValueError("Fresh prefill requires one question per image, in input order")
        plens, qlens = p.sum(1).tolist(), q.sum(1).tolist()
        self.prefix = Segments(plens, device)
        self.question = Segments(qlens, device)
        self.full = Segments([a + b for a, b in zip(plens, qlens)], device)
        valid = torch.cat([p, q], dim=1)
        self.ids = torch.cat([batch["p_ids"], batch["q_ids"]], dim=1)[valid].to(device)
        self.pos = torch.cat([batch["p_pos"], batch["q_pos"]], dim=2)[:, valid].unsqueeze(1).to(device)
        rows = torch.arange(len(self.ids))
        is_prefix = torch.cat([p, torch.zeros_like(q)], dim=1)[valid]
        self.p_idx = rows[is_prefix].to(device)
        self.q_idx = rows[~is_prefix].to(device)
        image_in_prefix = batch["p_img_mask"][p]
        img_rows = rows[is_prefix][image_in_prefix]
        self.img_rows = img_rows.to(device)

        # Only head inputs use a rectangular visual axis. Clips have no box targets.
        vv = batch["vis_valid"]
        vis_rows = torch.zeros_like(vv, dtype=torch.long)
        if vv.any():
            vis_rows[vv] = img_rows
        self.heads = dict(vis_gather=vis_rows.to(device), vis_valid=vv.to(device),
                          centers=batch["centers"].to(device), q_img=batch["q_img"].to(device),
                          q_last_packed=(self.full.cu[1:].long() - 1),
                          lab_idx=F.pad(batch["lab_idx"], (0, max(0, 26 - batch["lab_idx"].shape[1])), value=-1).to(device))


def _attention(q, k, v, queries, keys, causal, scale, used_keys=None):
    if q.is_cuda and varlen_attn is not None:
        return varlen_attn(q, k, v, queries.cu, keys.cu, queries.maximum, keys.maximum,
                           scale=scale, window_size=(-1, 0 if causal else -1), enable_gqa=True,
                           **({"seqused_k": used_keys} if used_keys is not None else {}))
    outputs = []
    for qi, ki, vi in zip(q.split(queries.lengths), k.split(keys.lengths), v.split(keys.lengths)):
        o = F.scaled_dot_product_attention(qi.transpose(0, 1)[None], ki.transpose(0, 1)[None],
                                           vi.transpose(0, 1)[None], scale=scale, is_causal=causal, enable_gqa=True)
        outputs.append(o[0].transpose(0, 1))
    return torch.cat(outputs)


def _attention_layer(engine, m, h, pack, rope):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    q, k, v, gate = engine._attention_projections(m, h, rope if engine.lm_prefill else None)
    if not engine.lm_prefill:
        q, k = apply_rotary_pos_emb(q[None], k[None], *rope, unsqueeze_dim=2)
        q, k = q[0], k[0]
    if engine.mask_mode == "block_bidir":
        # Prefix queries see only their causal prefix. Block queries see their entire sequence.
        pidx, qidx = pack.p_idx, pack.q_idx
        if engine.lm_prefill:
            # Prefix outputs of the full causal call are already correct in their final rows.
            out = _attention(q, k, v, pack.full, pack.full, True, m.scaling)
        else:
            prefix = _attention(q[pidx], k[pidx], v[pidx], pack.prefix, pack.prefix, True, m.scaling)
            out = torch.empty_like(q)
            out.index_copy_(0, pidx, prefix)
        question = _attention(q[qidx], k, v, pack.question, pack.full, False, m.scaling,
                              getattr(pack, "used_keys", None))
        out.index_copy_(0, qidx, question)
    else:
        out = _attention(q, k, v, pack.full, pack.full, True, m.scaling)
    if engine.lm_prefill:
        return m.o_proj(fused("attention_gate")(out, gate).flatten(1))
    return m.o_proj(out.flatten(1) * gate.sigmoid())


def _gdn_layer(engine, m, h, seq):
    mixed, z, beta, g = engine._gdn_projections(m, h)
    if engine.lm_prefill and backbone.fla_chunk_gdr is not None:
        from speed.lm_prefill import conv_qkv
        q, k, v = conv_qkv(m, mixed, seq.token_starts())
    else:
        w, bias = m.conv1d.weight, m.conv1d.bias
        if mixed.is_cuda and backbone.fla_chunk_gdr is not None:
            from fla.modules.causal_conv1d import causal_conv1d
            conv, _ = causal_conv1d(mixed[None], w[:, 0], bias,
                                    cu_seqlens=seq.cu, cu_seqlens_cpu=seq.cpu)
        else:
            parts = []
            for x in mixed.split(seq.lengths):
                y = F.conv1d(x.T[None].to(w.dtype), w, bias, padding=w.shape[-1] - 1, groups=w.shape[0])
                parts.append(y[0, :, :len(x)].T)
            conv = torch.cat(parts)[None]
        # Match the original bf16 rounding between convolution and activation.
        q, k, v = F.silu(conv).split([m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q, k = (x.reshape(1, len(h), -1, m.head_k_dim) for x in (q, k))
        v = v.reshape(1, len(h), -1, m.head_v_dim)
    if seq.fixed_width:
        # The batch dimension marks image boundaries, including the constant-width right tail.
        # This also avoids FLA's global varlen metadata cache during graph replay.
        shape = (len(seq.lengths), seq.maximum)
        out, _ = backbone.fla_chunk_gdr(*(a.reshape(*shape, *a.shape[2:]) for a in (q, k, v)),
                                        g=g.reshape(*shape, -1), beta=beta.reshape(*shape, -1),
                                        use_qk_l2norm_in_kernel=False, output_final_state=False)
    elif (engine.lm_prefill and engine.flash_gdn is not None and len(h) >= 4096
          and q.shape[-2:] == (16, 128) and v.shape[-2:] == (32, 128) and backbone.fla_chunk_gdr is not None):
        out = engine.flash_gdn(q, k, v, g[None], beta[None], seq)
    elif h.is_cuda and backbone.fla_chunk_gdr is not None:
        out, _ = backbone.fla_chunk_gdr(q, k, v, g=g[None], beta=beta[None], cu_seqlens=seq.cu,
                                        cu_seqlens_cpu=seq.cpu, use_qk_l2norm_in_kernel=not engine.lm_prefill, output_final_state=False)
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        parts = [x.split(seq.lengths, dim=1) for x in (q, k, v, g[None], beta[None])]
        out = torch.cat([torch_chunk_gated_delta_rule(*args, output_final_state=False,
                                                      use_qk_l2norm_in_kernel=True)[0] for args in zip(*parts)], dim=1)
    if engine.lm_prefill:
        from speed.lm_prefill import gated_norm
        out = gated_norm(m, out, z)
    else:
        out = m.norm(out.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    return m.out_proj(out.reshape(len(h), -1))


def _packed_layers(engine, pack, x, rope):
    h_next = None
    for i, (layer, kind) in enumerate(zip(engine.layers, engine.types)):
        norm = layer.input_layernorm
        h = h_next if h_next is not None else (fused("rms")(x, norm.weight, norm.eps) if engine.fast_ops else norm(x))
        if kind == "linear_attention":
            o = _gdn_layer(engine, layer.linear_attn, h, pack.full)
        else:
            o = _attention_layer(engine, layer.self_attn, h, pack, rope)
        norm = layer.post_attention_layernorm
        if engine.fast_ops:
            x, h = fused("add_rms")(x, o, norm.weight, norm.eps)
            update = engine._mlp_update(layer, h)
            if engine.lm_prefill:
                nxt = engine.layers[i + 1].input_layernorm if i + 1 < len(engine.layers) else engine.lm.norm
                x, h_next = fused("add_rms")(x, update, nxt.weight, nxt.eps)
            else:
                x = x + update
        else:
            x = x + o
            x = x + layer.mlp(norm(x))
    return h_next if h_next is not None else engine.lm.norm(x)


class PackedGraphs:
    """Capture fixed-grid LM layers. Trailing question slots are invisible to every real query."""
    def __init__(self, max_tokens=1024, max_graphs=8):
        self.max_tokens = max_tokens
        self.max_graphs = max_graphs
        self._graphs = {}
        self._pool = None

    def clear(self):
        self._graphs.clear()
        self._pool = None

    def run(self, engine, pack, x, rope):
        n = len(pack.full.lengths)
        prefix = pack.prefix.maximum
        question = pack.question.maximum
        # The reserve depends only on the image layout. Long questions take the eager path.
        width = (prefix + 32 + 15) // 16 * 16
        key = (engine.mask_mode, n, prefix)
        if (varlen_attn is None or backbone.fla_chunk_gdr is None
                or n * width > self.max_tokens or question > width - prefix
                or min(pack.prefix.lengths) != prefix or min(pack.question.lengths) != question
                or (key not in self._graphs and len(self._graphs) >= self.max_graphs)):
            return None
        if key not in self._graphs:
            dev = x.device
            full = Segments([width] * n, dev, fixed_width=True)
            rows = torch.arange(n * width, device=dev).view(n, width)
            static = SimpleNamespace(full=full, prefix=Segments([prefix] * n, dev),
                                     question=Segments([width - prefix] * n, dev),
                                     p_idx=rows[:, :prefix].reshape(-1), q_idx=rows[:, prefix:].reshape(-1))
            # The input/output buffers and their entire pool share the cache's lifetime.
            inputs = dict(x=x.new_zeros((n, width, x.shape[-1])),
                          cos=rope[0].new_zeros((n, width, rope[0].shape[-1])),
                          sin=rope[1].new_zeros((n, width, rope[1].shape[-1])),
                          used=torch.full((n,), width, device=dev, dtype=torch.int32))
            def forward(buf):
                static.used_keys = buf["used"]
                tables = (buf["cos"].flatten(0, 1)[None], buf["sin"].flatten(0, 1)[None])
                return _packed_layers(engine, static, buf["x"].flatten(0, 1), tables)
            # Indices and sequence offsets are allocated outside the private pool. Own them
            # for as long as the graph holds their device addresses.
            self._graphs[key] = (*engine._capture(forward, inputs, owner=self), static)
        graph, buf, hidden, _ = self._graphs[key]
        length = prefix + question
        for dst, src in ((buf["x"], x), (buf["cos"], rope[0]), (buf["sin"], rope[1])):
            dst[:, :length].copy_(src.reshape(n, length, -1))
        buf["used"].fill_(length)
        graph.replay()
        # Padded suffix states may depend on old inputs. Only prefixes and real question rows reach the heads.
        return hidden.view(n, width, -1)[:, :length].reshape(len(x), -1)


def packed_forward(engine, pack, feats, use_graph=False):
    with engine._autocast():
        x = engine.lm.embed_tokens(pack.ids)
        x.index_copy_(0, pack.img_rows, feats.to(x.dtype))
        rope = engine.net.rope(x[None], pack.pos)
        h = engine.packed_graphs.run(engine, pack, x, rope) if use_graph else None
        if h is None:
            h = _packed_layers(engine, pack, x, rope)
    return engine.heads_forward(h, h, pack.heads)[0]


class SharedPrefixes:
    """Pad-free prefill for independent images with multiple question branches per image.

    The packed prefix mixers export one GDN state and conv tail per image. Only
    K/V is scattered to the left-padded layout consumed by the existing branch
    attention kernel. No prefix is duplicated when multiple questions refer to it.
    """
    def __init__(self, batch, host, device, taps):
        valid = F.pad(batch["p_valid"], (host["Lp"] - batch["p_valid"].shape[1], 0))
        lengths = valid.sum(1).tolist()
        self.seq = Segments(lengths, device)
        # The shipped branch prefill left-pads each isolated prefix to 64. FLA
        # rounds at chunk boundaries, so discarding that leading offset can move
        # typed probabilities even with identical causal dependencies.
        padded = [(n + 63) // 64 * 64 for n in lengths]
        self.gdn_seq = Segments(padded, device)
        self.gdn_rows = torch.cat([torch.arange(end-n, end) for n, end in zip(lengths, accumulate(padded))]).to(device)
        self.rows = valid.flatten().nonzero().flatten().to(device)
        self.tail_rows = self.seq.cu[1:, None].long() + torch.arange(1 - taps, 0, device=device)
        self.width = host["Lp"]

    def gdn(self, m, mixed, z, beta, g):
        from speed.lm_prefill import conv_qkv, gated_norm
        seq, rows = self.gdn_seq, self.gdn_rows
        length = int(seq.cpu[-1])
        padded_mixed = mixed.new_zeros((length, mixed.shape[1]))
        padded_mixed.index_copy_(0, rows, mixed)
        q, k, v = conv_qkv(m, padded_mixed, seq.token_starts())
        # Gates for a zeroed reference input (all GDN projections are bias-free).
        padded_g = (-m.A_log.float().exp() * F.softplus(m.dt_bias.float())).expand(length, -1).clone()
        padded_g.index_copy_(0, rows, g)
        padded_beta = beta.new_full((length, beta.shape[1]), 0.5)
        padded_beta.index_copy_(0, rows, beta)
        out, state = backbone.fla_chunk_gdr(q, k, v, g=padded_g[None], beta=padded_beta[None], cu_seqlens=seq.cu,
                                           cu_seqlens_cpu=seq.cpu, use_qk_l2norm_in_kernel=False,
                                           output_final_state=True)
        return gated_norm(m, out[:, rows], z), state, mixed[self.tail_rows].transpose(1, 2)

    def attention(self, m, q, k, v, gate, rope):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q[None], k[None], *rope, unsqueeze_dim=2)
        out = _attention(q[0], k[0], v, self.seq, self.seq, True, m.scaling)
        def rectangular(t):
            b = len(self.seq.lengths)
            target = t.new_zeros((b*self.width, *t.shape[1:]))
            target.index_copy_(0, self.rows, t)
            return target.view(b, self.width, *t.shape[1:]).transpose(1, 2)
        return fused("attention_gate")(out, gate.view_as(out)).flatten(1), rectangular(k[0]), rectangular(v)

    def forward(self, eng, inputs, host):
        rows = self.rows
        with eng._autocast():
            _, questions, xq = eng._inputs(inputs, host, with_prefix=False)
            xp = eng.lm.embed_tokens(inputs['p_ids'].flatten()[rows])
            xp = torch.where(inputs['img_rowmask'][rows,None], inputs['featbuf'][rows], xp)
            pos = inputs['p_pos'].flatten(1)[:,rows].unsqueeze(1)
            prefix = dict(packed=self, B=host['B'], Lp=host['Lp'], rope=eng.net.rope(xp[None], pos))
            hidden, cache = eng.lm_forward(torch.cat([xp,xq]), prefix, questions)
        # Convert head indices from padded slots to packed rows. Invalid head slots are masked.
        dense_to_packed = inputs['p_valid'].flatten().long().cumsum(0).sub_(1).clamp_min_(0)
        metadata = dict(inputs, vis_gather=dense_to_packed[inputs['vis_gather']])
        out, vis = eng.heads_forward(hidden[:len(rows)], hidden[len(rows):], metadata)
        return out, cache, vis
