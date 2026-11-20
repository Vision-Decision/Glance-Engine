"""Lossless FP32 recurrent-state planes. Each 128-value group stores a local exponent window."""

import torch


def encode(x):
    bits = x.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 255).byte().view(-1, 128)
    base = exp.amin(-1)
    wide = (exp.amax(-1) - base) > 15
    tag = torch.where(wide, -wide.cumsum(0, dtype=torch.int32), base.int())
    nibble = ((exp - base[:, None]) & 15).reshape(*x.shape[:-1], -1, 2)
    # CUDA accepts a valid pointer even when every tile uses the four-bit window.
    escape = torch.cat((exp[wide].flatten(), exp.new_zeros(1)))
    mid = (((bits >> 16) & 127) | ((bits >> 24) & 128)).byte()
    return (
        bits.short(),
        mid,
        nibble[..., 0] | nibble[..., 1] << 4,
        tag.view(*x.shape[:-1], -1),
        escape,
    )


def decode(planes):
    low, mid, nibble, tag, escape = planes
    if low.is_cuda:
        from scan.bank_kernels import unpack

        return unpack(*planes)
    idx = torch.arange(low.numel()).reshape(low.shape)
    t = tag.reshape(-1)[idx // 128]
    ex = t + ((nibble.flatten()[idx // 2].int() >> (4 * (idx % 2))) & 15)
    ex = torch.where(
        t >= 0,
        ex,
        escape[((-1 - t) * 128 + idx % 128).clamp(0, escape.numel() - 1)].int(),
    )
    return (
        (((mid.int() & 128) << 24) | (ex << 23) | ((mid.int() & 127) << 16) | (low.int() & 65535))
        .int()
        .view(torch.float32)
    )


class StateReader:
    """Decode one layer ahead into two scratch buffers with stable graph-captured addresses."""

    @classmethod
    def make(cls, bank, stream):
        layers = [i for i, kind in enumerate(bank.types) if kind == "linear_attention"]
        if not layers or any(bank.planes(i) is None for i in layers):
            return None
        first = bank.tensors[layers[0], "state"]
        if any(bank.tensors[i, "state"].shape != first.shape for i in layers):
            return None
        return cls(bank, stream, layers, first)

    def __init__(self, bank, stream, layers, first):
        self.bank, self.stream, self.layers = bank, stream, iter(layers)
        self.buffers = [torch.empty_like(first, dtype=torch.float32) for _ in range(2)]
        self.slot = 0
        self._next()

    def _next(self):
        from scan.bank_kernels import unpack

        self.layer = next(self.layers, None)
        if self.layer is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.bank.q_start.device))
            with torch.cuda.stream(self.stream):
                unpack(
                    self.bank.tensors[self.layer, "state"],
                    *self.bank.planes(self.layer),
                    out=self.buffers[self.slot],
                )

    def take(self, layer):
        if self.layer != layer:
            raise RuntimeError("prefetched recurrent layer is out of order")
        torch.cuda.current_stream(self.bank.q_start.device).wait_stream(self.stream)
        result = self.buffers[self.slot]
        self.slot ^= 1
        self._next()
        return result
