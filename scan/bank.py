"""Compact storage for read-only scan prefixes. Query arithmetic stays in the model's native dtype."""

from collections import defaultdict

import torch

FORMATS = (
    "legacy",
    "compact",
    "state_packed",
    "state_bf16",
    "state_fp8",
    "state_int8",
    "kv_fp8",
    "state_kv_fp8",
)


def _scaled(x, kind):
    dtype, limit = (torch.int8, 127) if kind == "int8" else (torch.float8_e4m3fn, 448)
    scale = x.float().abs().amax(-1, keepdim=True) / limit
    scale = torch.where(scale == 0, 1.0, scale)
    y = x.float() / scale
    if kind == "int8":
        y = y.round()
    return y.clamp(-limit, limit).to(dtype), scale


def _token_layout(bank):
    """Only share contiguous leading positions whose stored K and V bits agree in every layer."""
    mask = bank.p_valid
    row, col = mask.nonzero(as_tuple=True)
    lengths = mask.sum(1)
    offsets = lengths.cumsum(0) - lengths
    width = min(16, int(lengths.min()))
    same = torch.ones(width, device=mask.device, dtype=torch.bool)
    if bank.N > 1 and width:
        seq = col[offsets[:, None] + torch.arange(width, device=mask.device)]
        item = torch.arange(bank.N, device=mask.device)[:, None]
        for kind, pair in zip(bank.types, bank.cache):
            if kind != "linear_attention":
                for t in pair:
                    bits = t.transpose(1, 2)[item, seq].contiguous().view(torch.uint8)
                    same &= (bits == bits[:1]).flatten(2).all(2).all(0)
        common = int(same.long().cumprod(0).sum())
    else:
        common = 0
    ordinal = torch.arange(len(row), device=mask.device) - offsets[row]
    keep = (ordinal >= common) | (row == 0)
    index = keep.int().cumsum(0, dtype=torch.int32)
    ids = torch.zeros_like(mask, dtype=torch.int32)
    ids[row, col] = torch.where(ordinal < common, ordinal + 1, index).int()
    return ids, row[keep], col[keep], common


class StoredBank:
    """Layer tensors and position maps. Entry zero of a token table represents masked padding."""

    def __init__(self, tensors, types, has_clips, mode, readout, shared, kv_dtype, state_dtype):
        self.tensors = tensors
        self.types, self.has_clips = types, has_clips
        self.mode, self.readout, self.shared = mode, readout, shared
        self.kv_dtype, self.state_dtype = kv_dtype, state_dtype
        self._graphs, self._pool = {}, None

    @classmethod
    def pack(cls, bank, mode="compact", readout="all"):
        if mode not in FORMATS[1:] or readout not in ("all", "typed"):
            raise ValueError("unsupported bank format or readout")
        tensors = {k: getattr(bank, k) for k in ("p_valid", "q_start", "grid_hw")}
        ids, rows, cols, shared = _token_layout(bank)
        tensors["tokens"] = ids
        kv_dtype = state_dtype = None
        for i, (kind, (a, b)) in enumerate(zip(bank.types, bank.cache)):
            if kind == "linear_attention":
                state_dtype = a.dtype
                if mode == "state_packed" and a.dtype == torch.float32 and a.shape[-1] == 128:
                    from scan.bank_codec import encode

                    a, *planes = encode(a)
                    for name, plane in zip(("mid", "nibble", "tag", "escape"), planes):
                        tensors[i, name] = plane
                elif mode == "state_bf16":
                    a = a.bfloat16()
                elif mode in ("state_fp8", "state_int8", "state_kv_fp8"):
                    a, tensors[i, "state_scale"] = _scaled(a, "int8" if mode == "state_int8" else "fp8")
                tensors[i, "state"], tensors[i, "tail"] = a, b
            else:
                kv_dtype = a.dtype
                for name, t in (("key", a), ("value", b)):
                    flat = t.new_empty((len(rows) + 1, t.shape[1], t.shape[3]))
                    flat[0].zero_()
                    flat[1:] = t.transpose(1, 2)[rows, cols]
                    if mode in ("kv_fp8", "state_kv_fp8"):
                        flat, tensors[i, name + "_scale"] = _scaled(flat, "fp8")
                    tensors[i, name] = flat
        if readout == "all":
            mask = bank.vis_valid
            tensors["vis_valid"] = mask
            if bank.has_clips and not mask.any():
                # Every center is masked. index zero is the deterministic winner, and the
                # masked visual vector/center is zero. Retain the original first slot for API views.
                for name in ("vis", "centers", "ck"):
                    tensors[name] = getattr(bank, name)[:1, :1].clone()
            elif mask.all():
                for name in ("vis", "centers", "ck"):
                    tensors[name] = getattr(bank, name)
            else:
                # Row zero preserves the masked spatial slot on native video in a mixed bank.
                tensors["pixels"] = torch.where(mask, mask.flatten().cumsum(0).reshape_as(mask), 0).int()
                for name in ("vis", "centers", "ck"):
                    t = getattr(bank, name)
                    tensors[name] = torch.cat((t.new_zeros((1, t.shape[-1])), t[mask]))
        return cls(
            tensors,
            bank.types,
            bank.has_clips,
            mode,
            readout,
            shared,
            kv_dtype,
            state_dtype,
        )

    @property
    def p_valid(self):
        return self.tensors["p_valid"]

    @property
    def q_start(self):
        return self.tensors["q_start"]

    @property
    def grid_hw(self):
        return self.tensors["grid_hw"]

    @property
    def N(self):
        return self.p_valid.shape[0]

    @property
    def Lp(self):
        return self.p_valid.shape[1]

    @property
    def vis_valid(self):
        return self.tensors["vis_valid"]

    def spatial(self, name, selected=None):
        t = self.tensors[name]
        if "pixels" in self.tensors:
            ids = self.tensors["pixels"]
            if selected is not None:
                ids = ids[torch.arange(self.N, device=t.device), selected]
            return t[ids.long()]
        t = t.expand(self.N, self.vis_valid.shape[1], -1)
        return t if selected is None else t[torch.arange(self.N, device=t.device), selected]

    vis = property(lambda self: self.spatial("vis"))
    centers = property(lambda self: self.spatial("centers"))
    ck = property(lambda self: self.spatial("ck"))

    def _decode(self, i, name, dtype):
        t = self.tensors[i, name]
        scale = self.tensors.get((i, name + "_scale"))
        if scale is not None:
            t = t.float() * scale
        return t.to(dtype)

    def attention(self, i):
        return (
            self._decode(i, "key", self.kv_dtype),
            self._decode(i, "value", self.kv_dtype),
            self.tensors["tokens"],
        )

    def planes(self, i):
        if (i, "mid") in self.tensors:
            return tuple(self.tensors[i, k] for k in ("mid", "nibble", "tag", "escape"))
        return None

    def layer(self, i):
        if self.types[i] == "linear_attention":
            planes = self.planes(i)
            if planes is not None:
                from scan.bank_codec import decode

                return decode((self.tensors[i, "state"], *planes)), self.tensors[i, "tail"]
            return self._decode(i, "state", self.state_dtype), self.tensors[i, "tail"]
        k, v, ids = self.attention(i)
        if k.is_cuda and k.shape[-1] == 256:
            from scan.bank_kernels import gather

            return gather(k, v, ids)
        return tuple(t[ids.long()].transpose(1, 2).contiguous() for t in (k, v))

    @property
    def cache(self):
        """Dense compatibility view for callers inspecting the original prefix tensors."""
        return [self.layer(i) for i in range(len(self.types))]

    def components(self):
        sizes = defaultdict(int)
        for k, t in self.tensors.items():
            sizes[k[1] if isinstance(k, tuple) else k] += t.nbytes
        return dict(sizes)

    def nbytes(self):
        return sum(self.components().values())

    def with_tensors(self, tensors):
        return StoredBank(
            tensors,
            self.types,
            self.has_clips,
            self.mode,
            self.readout,
            self.shared,
            self.kv_dtype,
            self.state_dtype,
        )

    def _rows(self, idx, copy):
        t = {}
        for k, v in self.tensors.items():
            flat_kv = isinstance(k, tuple) and self.types[k[0]] != "linear_attention"
            flat_spatial = k in ("vis", "ck", "centers") and ("pixels" in self.tensors or v.shape[0] != self.N)
            if flat_kv or flat_spatial or (isinstance(k, tuple) and k[1] == "escape"):
                t[k] = v
            elif copy and v.dtype == torch.float8_e4m3fn:
                t[k] = v.view(torch.uint8)[idx].view(v.dtype)
            else:
                t[k] = v[idx]
        if copy:
            # Preserve repeated row order and stored quantization exactly, including scale tensors.
            kv_fields = [k for k in t if isinstance(k, tuple) and self.types[k[0]] != "linear_attention"]
            for table, fields in (("tokens", kv_fields), ("pixels", ("vis", "ck", "centers"))):
                if table not in t:
                    continue
                used, inv = torch.unique(t[table], return_inverse=True)
                t[table] = inv.int()
                for k in fields:
                    v = t[k]
                    t[k] = (
                        v.view(torch.uint8)[used.long()].view(v.dtype)
                        if v.dtype == torch.float8_e4m3fn
                        else v[used.long()]
                    )
            for i in range(len(self.types)):
                if (i, "escape") not in t:
                    continue
                tag = t[i, "tag"]
                wide = tag < 0
                used, inv = torch.unique(-1 - tag[wide], return_inverse=True)
                tag[wide] = -1 - inv.int()
                escape = t[i, "escape"]
                t[i, "escape"] = torch.cat((escape[:-1].view(-1, 128)[used.long()].flatten(), escape[-1:]))
        return self.with_tensors(t)

    @staticmethod
    def concat(banks):
        """Merge full-readout banks into the legacy dense representation."""
        from scan.engine import ImageBank

        return ImageBank.concat(banks)

    def head(self, n):
        return self._rows(slice(0, n), False)

    def select(self, idx):
        return self._rows(torch.as_tensor(idx, device=self.p_valid.device), True)
