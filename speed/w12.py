"""Lossless storage and inference for the merged Glance BF16 language projections.

The SM120 record layout follows SpecPilot D2 (see specpilot/LICENSE). Each record
covers 128 output channels and 64 inputs. The exponent window and full BF16 escape
values are stored alongside the records. BF16 remains available for large GEMMs.
"""

from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        "glance_w12_sm120",
        [str(Path(__file__).with_name("specpilot") / "w12.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"],
        extra_ldflags=["-lcuda"],
    )


def projections(lm):
    """Only the four linears executed per fused layer. Originals alias fused storage."""
    for i, layer in enumerate(lm.layers):
        if not hasattr(layer, "_gu"):
            raise ValueError("W12 requires merged, fused BF16 language projections")
        names = ["_gu", "mlp.down_proj"]
        if hasattr(layer, "linear_attn"):
            if layer.linear_attn._ab is not None:
                raise ValueError("W12 requires the BF16 GDN fusion")
            names += ["linear_attn._in", "linear_attn.out_proj"]
        else:
            names += ["self_attn._qkv", "self_attn.o_proj"]
        for name in names:
            yield f"layers.{i}.{name}", layer.get_submodule(name)


def _record_order(x):
    rows, k = x.shape
    return (
        x.view(rows // 16, 2, 8, k // 64, 4, 2, 4, 2)
        .permute(0, 3, 2, 6, 4, 5, 1, 7)
        .reshape(rows // 16, k // 64, 32, 32)
    )


def _planes(dense, nibble):
    # Each lane's 32 weights are two 16-byte vectors plus one 16-byte nibble vector.
    rows, k = dense.shape
    rb, kb = rows // 128, k // 64
    q = _record_order(dense).reshape(rb * 8, kb, 32, 2, 16).transpose(2, 3)
    b = _record_order(nibble).reshape(rb * 8, kb, 32, 4, 2, 4)
    b = b[..., 0, :] | (b[..., 1, :] << 4)
    q = q.reshape(rb, 8, kb, 1024).transpose(1, 2).reshape(rb, kb, 8192)
    b = b.reshape(rb, 8, kb, 512).transpose(1, 2).reshape(rb, kb, 4096)
    return torch.cat((q, b), dim=-1).reshape(rb, -1)


@torch.no_grad()
def encode(w):
    rows, k = w.shape
    if w.dtype != torch.bfloat16 or k % 64 or k > 32768:
        raise ValueError("expected a BF16 matrix with K divisible by 64 and <= 32768")
    planes, tops, counts, columns, values = [], [], [], [], []
    for chunk in w.split(1024):
        wc = F.pad(chunk, (0, 0, 0, (-len(chunk)) % 128), value=1).contiguous()
        bits = wc.view(torch.int16).int() & 65535
        exp = (bits >> 7) & 255
        hist = torch.zeros((len(wc), 256), device=w.device, dtype=torch.int32)
        hist.scatter_add_(1, exp.long(), torch.ones_like(exp))
        # Exceptional values, zeros and subnormals use their original 16 bits.
        hist[:, 0] = 0
        hist[:, 255] = 0
        prefix = F.pad(hist.cumsum(1), (1, 0))
        coverage = prefix[:, 17:256] - prefix[:, 1:240]
        top = coverage.argmax(dim=1).int() + 16
        code = exp - (top[:, None] - 15)
        esc = (code < 0) | (code > 15) | (exp == 255)
        dense = ((code & 1) << 7) | (bits & 0x70) | ((bits >> 12) & 8) | ((code >> 1) & 7)
        planes.append(_planes(dense.masked_fill(esc, 0).byte(), (bits & 15).masked_fill(esc, 0).byte()))
        tops.append(top.byte())
        counts.append(esc.sum(1))
        row, col = esc.nonzero(as_tuple=True)
        columns.append(col.short())
        values.append(wc[row, col])
    offsets = F.pad(torch.cat(counts).cumsum(0), (1, 0)).int()
    return {
        "shape": (rows, k),
        "planes": torch.cat(planes),
        "top": torch.cat(tops),
        "offsets": offsets,
        "columns": torch.cat(columns),
        "values": torch.cat(values),
    }


def decode_record(record):
    """Portable integer reference for the offline format, including arbitrary BF16 bit patterns."""
    n, k = record["shape"]
    tiles = record["planes"].view(-1, k // 64, 12288)
    decoded = []
    for t0 in range(0, len(tiles), 8):
        data = tiles[t0 : t0 + 8]
        rb, kb = data.shape[:2]
        rows = rb * 128
        q = data[..., :8192].reshape(rb, kb, 8, 1024).transpose(1, 2)
        q = q.reshape(rb * 8, kb, 2, 32, 16).transpose(2, 3).reshape(rb * 8, kb, 32, 32)
        b = data[..., 8192:].reshape(rb, kb, 8, 512).transpose(1, 2).reshape(rb * 8, kb, 32, 4, 4)
        low = torch.stack((b & 15, b >> 4), dim=-2).reshape_as(q)
        # Undo the per-warp fragment permutation used by the SM120 MMA instructions.
        code = ((q & 7) << 1) | (q >> 7)
        base = ((q.int() & 8) << 12) | (code.int() << 7) | (q & 112) | low
        bits = base.reshape(rows // 16, kb, 8, 4, 4, 2, 2, 2).permute(0, 6, 2, 1, 4, 5, 3, 7).reshape(rows, k)
        exp = record["top"][t0 * 128 : t0 * 128 + rows].int() - 15
        decoded.append((bits + (exp[:, None] << 7)).short())
    out = torch.cat(decoded)[:n]
    counts = record["offsets"].diff().long()
    row = torch.arange(len(counts), device=out.device).repeat_interleave(counts)
    out[row, record["columns"].long()] = record["values"].view(torch.int16)
    return out.view(torch.bfloat16)


def validate_record(record):
    n, k = record["shape"]
    if min(n, k) < 1 or k % 64 or k > 32768 or n * k >= 2**31:
        raise ValueError("invalid W12 matrix dimensions")
    tiles = (n + 127) // 128
    escapes = record["columns"].numel()
    expected = {
        "planes": (torch.uint8, (tiles, k // 64 * 12288)),
        "top": (torch.uint8, (tiles * 128,)),
        "offsets": (torch.int32, (tiles * 128 + 1,)),
        "columns": (torch.int16, (escapes,)),
        "values": (torch.bfloat16, (escapes,)),
    }
    device = record["planes"].device
    for key, (dtype, shape) in expected.items():
        v = record[key]
        if v.dtype != dtype or v.shape != shape or v.device != device or not v.is_contiguous():
            raise ValueError(f"invalid W12 {key}")
    offsets, cols, top = record["offsets"], record["columns"], record["top"]
    if (
        offsets[0] != 0
        or offsets[n] != escapes
        or offsets[-1] != escapes
        or (offsets[1:] < offsets[:-1]).any()
        or ((cols < 0) | (cols >= k)).any()
        or ((top < 16) | (top > 254)).any()
    ):
        raise ValueError("invalid W12 escape indices or exponent window")


class PackedWeight(nn.Module):
    """Device representation. A private workspace per call is safe under graph capture and replay."""

    def __init__(self, record):
        super().__init__()
        validate_record(record)
        self.n, self.k = record["shape"]
        self.tiles = (self.n + 127) // 128
        self.register_buffer("planes", record["planes"])
        self.register_buffer("offsets", record["offsets"])
        self.register_buffer("columns", record["columns"].int())
        values = record["values"]
        counts = self.offsets.diff().long()
        rows = torch.arange(len(counts), device=values.device).repeat_interleave(counts)
        self.register_buffer("indices", rows * self.k + self.columns)
        if (self.indices[1:] <= self.indices[:-1]).any():
            raise ValueError("W12 escape columns must increase within each row")
        self.register_buffer("values", values)
        self.register_buffer("scale", torch.exp2((record["top"].int() - 127).double()).float())
        normalized = values.double() / self.scale[rows].double()
        self.register_buffer("correction", normalized.float())
        self.exact_correction = torch.equal(self.correction.double(), normalized)
        sms = torch.cuda.get_device_properties(self.planes.device).multi_processor_count
        self.blocks = min(sms, self.tiles * max(1, sms // self.tiles))
        stages = self.k // 64
        total = self.tiles * stages

        def owner(pos):
            return ((pos + 1) * self.blocks - 1) // total

        self.max_segments = max(owner((t + 1) * stages - 1) - owner(t * stages) + 1 for t in range(self.tiles))
        self.kernel = extension()

    def decode(self, verify_placeholders=False):
        out = self.values.new_empty((self.n, self.k))
        self.kernel.decode(self.planes, self.scale, out, self.n, self.k, 0)
        if self.values.numel():
            if verify_placeholders:
                rows = self.indices // self.k
                expected = (self.scale[rows] * 2**-15).to(torch.bfloat16)
                if not torch.equal(out.flatten()[self.indices].view(torch.int16), expected.view(torch.int16)):
                    raise ValueError("W12 escape placeholder differs from the GEMM correction")
            out.flatten().index_copy_(0, self.indices, self.values)
        return out

    def multiply(self, x, whole=False, split_stages=0, epilogue=False):
        if x.shape[-1] != self.k:
            raise ValueError("W12 input width differs from the packed matrix")
        x = x.reshape(-1, self.k).contiguous()
        m = x.shape[0]
        if (
            not 0 < m <= (64 if split_stages else 128)
            or x.dtype != torch.bfloat16
            or x.device != self.planes.device
        ):
            raise ValueError("W12 GEMM received an unsupported number of BF16 rows on the packed weight device")
        if not self.exact_correction:
            raise ValueError("escape correction exceeds FP32 range")
        if x.data_ptr() % 16:
            x = x.clone()
        out = x.new_empty((m, self.n))
        if split_stages:
            splits = (self.k // 64 + split_stages - 1) // split_stages
            partial = x.new_empty((splits, m, self.n), dtype=torch.float32)
            self.kernel.direct(
                x, self.planes, self.scale, self.offsets, self.columns, self.correction, out, partial, split_stages
            )
            return out
        nt = (m + 7) // 8 * 8 if m <= 48 else next(n for n in (64, 96, 128) if m <= n)
        workspace = x.new_empty(self.tiles * self.max_segments * 128 * nt, dtype=torch.float32)
        counters = x.new_zeros(self.tiles, dtype=torch.int32)
        self.kernel.gemm(
            x,
            self.planes,
            self.scale,
            self.offsets,
            self.columns,
            self.correction,
            out,
            workspace,
            counters,
            self.blocks,
            self.max_segments,
            whole,
            epilogue,
        )
        return out


# Cold-weight graph medians on the 188-SM G4, see item5_micro_epilogue.json.
# Unmeasured widths and row counts use the original Linear.
_CUTOFF = {(18432, 2560): 128, (12352, 2560): 96, (10240, 2560): 96, (2560, 9216): 40, (2560, 4096): 24}
_ROWS = (1, 8, 16, 24, 32, 40, 48, 64, 96, 128)


class W12Linear(nn.Linear):
    """Keep the existing BF16 parameter and its aliases for the large-M SM120 path."""

    def __init__(self, linear, packed, max_rows=64):
        nn.Module.__init__(self)
        self.weight, self.bias = linear.weight, linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.packed = packed
        self.rows = frozenset(m for m in _ROWS if m <= max_rows)

    def forward(self, x):
        m = x.numel() // self.in_features
        if (
            m in self.rows
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.shape[-1] == self.in_features
            and not torch.is_grad_enabled()
        ):
            y = self.packed.multiply(x)
            return y.view(*x.shape[:-1], self.out_features)
        return F.linear(x, self.weight, self.bias)


@torch.no_grad()
def install(lm, folder):
    """Attach verified offline records before constructing any engine or capturing graphs."""
    import json

    device = lm.embed_tokens.weight.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("W12 kernels require an SM120 device")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    modules = dict(projections(lm))
    if manifest["version"] != 1 or set(manifest["matrices"]) != set(modules):
        raise ValueError("W12 manifest does not match the fused model layout")
    replacements = []
    for name, mod in modules.items():
        if type(mod) != nn.Linear or mod.bias is not None or mod.weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a bias-free BF16 linear")
        record = torch.load(folder / f"{name}.pt", weights_only=True, map_location=mod.weight.device)
        if tuple(record["shape"]) != tuple(mod.weight.shape):
            raise ValueError(f"W12 matrix shape differs at {name}")
        packed = PackedWeight(record)
        if not packed.exact_correction or not torch.equal(
            packed.decode(verify_placeholders=True).view(torch.int16), mod.weight.view(torch.int16)
        ):
            raise ValueError(f"W12 artifact differs from merged weights at {name}")
        cutoff = _CUTOFF.get(tuple(mod.weight.shape), 0)
        if cutoff:
            replacements.append((name, W12Linear(mod, packed, cutoff)))
    for name, replacement in replacements:
        lm.set_submodule(name, replacement)
    lm._w12_compact = True
    return len(replacements)
