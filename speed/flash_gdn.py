"""Inference-only SM120 chunked GDN using the optional FlashQLA 0.1.3 kernels."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=2)
def load_flash_gdn(*, uniform_frames=True):
    # The value tile is an internal compiler argument of this version. Other versions use FLA.
    try:
        if version("flash-qla") != "0.1.3" or version("tilelang") != "0.1.15":
            return None
        from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120 import kkt_solve
        from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_fwd import tilelang_fused_chunk_gdr_fwd
        from flash_qla.ops.utils import chunk_local_cumsum
    except (PackageNotFoundError, ImportError, OSError):
        return None

    def forward(q, k, v, g, beta, seq, *, dense=None, value_tile=64):
        cu = seq.cu
        shape = v.shape
        if dense is None:
            dense = uniform_frames and len(seq.lengths) >= 32 and seq.maximum <= 160 and seq.minimum == seq.maximum
        if dense:
            width = seq.lengths[0]
            if seq.minimum != seq.maximum:
                raise ValueError("dense GDN requires equal sequence lengths")
            q, k, v, g, beta = (t.reshape(-1, width, *t.shape[2:]) for t in (q, k, v, g, beta))
        boundaries = None if dense else cu
        decays = chunk_local_cumsum(g=g, cu_seqlens=boundaries, chunk_size=32)
        triangular = kkt_solve(k=k, b=beta, cu_seqlens=boundaries, chunk_size=32)
        _, _, key_heads, key_dim = k.shape
        _, _, value_heads, value_dim = v.shape
        kernel = tilelang_fused_chunk_gdr_fwd(
            value_heads,
            key_heads,
            key_dim,
            value_dim,
            32,
            key_dim**-0.5,
            accum_dtype="float32",
            qkva_dtype=q.dtype,
            g_dtype=decays.dtype,
            b_dtype=beta.dtype,
            h0_dtype=torch.float32,
            ht_dtype=torch.float32,
            h_dtype=q.dtype,
            o_dtype=q.dtype,
            seqlen_dtype=cu.dtype,
            use_initial_state=False,
            store_final_state=False,
            store_h=False,
            store_o=True,
            is_varlen=not dense,
            is_cp=False,
            state_v_first=False,
            block_DV=value_tile,
        )
        # The compiled interface checks these shapes even when state IO is compiled out.
        # Both state arguments can share storage and the chunk-state axis has length zero.
        state = v.new_empty((len(seq.lengths), value_heads, key_dim, value_dim), dtype=torch.float32)
        chunk_states = v.new_empty((v.shape[0], 0, value_heads, key_dim, value_dim))
        out = torch.empty_like(v)
        kernel(
            q,
            k,
            v,
            triangular,
            decays,
            beta,
            state,
            cu,
            seq.chunk_offsets32(),
            cu[:-1],
            cu,
            out,
            chunk_states,
            state,
        )
        return out.reshape(shape)

    return forward
