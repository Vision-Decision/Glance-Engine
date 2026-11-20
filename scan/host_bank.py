"""Pinned, chunk-addressable scan banks with two reusable device staging slots."""

import time
from collections import defaultdict

import torch


def _layout(fields):
    result = []
    end = 0
    for key, shape, dtype in fields:
        offset = (end + 15) // 16 * 16
        size = shape.numel() * dtype.itemsize
        result.append((key, shape, dtype, offset, size))
        end = offset + size
    return tuple(result)


def _size(layout):
    return layout[-1][3] + layout[-1][4]


def _views(layout, slab):
    return {k: slab[offset : offset + size].view(dtype).view(shape) for k, shape, dtype, offset, size in layout}


def _copy_ranges(source, target):
    """Coalesce adjacent fields with the same displacement, including alignment padding."""
    ranges = []
    for (_, _, _, src, size), (_, _, _, dst, _) in zip(source, target):
        if ranges and src - dst == ranges[-1][0] - ranges[-1][1]:
            start, dest, _ = ranges[-1]
            ranges[-1] = (start, dest, src + size - start)
        elif size:
            ranges.append((src, dst, size))
    return ranges


class HostChunk:
    def __init__(self, bank):
        # Variable-length exception tables come last. Changing one table size must not
        # split the large fixed-state transfer into a separate copy for every layer.
        fields = sorted(bank.tensors.items(), key=lambda item: isinstance(item[0], tuple) and item[0][1] == "escape")
        self.layout = _layout((k, t.shape, t.dtype) for k, t in fields)
        self.slab = torch.empty(_size(self.layout), device="cpu", dtype=torch.uint8, pin_memory=True)
        views = _views(self.layout, self.slab)
        for k, t in bank.tensors.items():
            views[k].copy_(t, non_blocking=True)
        torch.cuda.current_stream(bank.q_start.device).synchronize()
        self.bank = bank.with_tensors(views)

    def geometry(self):
        """The number of physical records can vary without changing query tensor shapes."""
        fields = []
        for k, shape, dtype, _, _ in self.layout:
            variable = (
                isinstance(k, tuple)
                and (self.bank.types[k[0]] != "linear_attention" or k[1] == "escape")
                or "pixels" in self.bank.tensors
                and k in ("vis", "ck", "centers")
            )
            fields.append((k, (None, *shape[1:]) if variable else shape, dtype))
        return self.bank.has_clips, self.bank.mode, self.bank.readout, tuple(fields)


class HostBank:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("empty host bank")
        self.N = sum(c.bank.N for c in self.chunks)
        groups = defaultdict(list)
        for chunk in dict.fromkeys(self.chunks):
            groups[chunk.geometry()].append(chunk)
        self.layouts = {}
        self.copies = {}
        for group in groups.values():
            fields = []
            for entries in zip(*(chunk.layout for chunk in group)):
                k, shape, dtype, _, _ = max(entries, key=lambda e: e[4])
                fields.append((k, shape, dtype))
            layout = _layout(fields)
            self.layouts.update((chunk, layout) for chunk in group)
            self.copies.update((chunk, _copy_ranges(chunk.layout, layout)) for chunk in group)
        self._slots = [None, None]
        self._copy = None

    @classmethod
    def from_chunks(cls, banks):
        return cls(HostChunk(bank) for bank in banks)

    def nbytes(self):
        return sum(c.slab.nbytes for c in self.chunks)

    def free_graphs(self):
        self._slots = [None, None]
        self._copy = None

    def _stage(self, index, chunk, device):
        layout = self.layouts[chunk]
        slot = self._slots[index]
        if slot is None:
            capacity = max(_size(self.layouts[c]) for c in self.chunks[index::2])
            slot = {
                "slab": torch.empty(capacity, dtype=torch.uint8, device=device),
                "ready": torch.cuda.Event(),
                "used": torch.cuda.Event(),
                "owners": {},
            }
            self._slots[index] = slot
        self._copy.wait_event(slot["used"])
        owners = slot["owners"]
        key = (layout, chunk.bank.mode, chunk.bank.has_clips, chunk.bank.readout)
        bank = owners.get(key)
        if bank is None:
            # A full chunk and the final partial chunk can share one slot across repeated queries.
            if len(owners) >= 2:
                owners.pop(next(iter(owners)))
            bank = owners[key] = chunk.bank.with_tensors(_views(layout, slot["slab"]))
        for src, dst, size in self.copies[chunk]:
            slot["slab"][dst : dst + size].copy_(chunk.slab[src : src + size], non_blocking=True)
        slot["ready"].record()
        return slot, bank

    def query(self, engine, q, timings):
        if self.chunks[0].bank.readout == "typed" and q["type"] == "box":
            raise ValueError("a typed-only bank cannot answer a box query")
        compute = torch.cuda.current_stream(engine.dev)
        if self._copy is None:
            self._copy = torch.cuda.Stream(device=engine.dev)
        start = time.perf_counter()
        prep = 0.0
        outputs = []
        for i, chunk in enumerate(self.chunks):
            with torch.cuda.stream(self._copy):
                slot, bank = self._stage(i % 2, chunk, engine.dev)
            compute.wait_event(slot["ready"])
            slot["slab"].record_stream(compute)
            try:
                t = time.perf_counter()
                inputs, host = engine.branch_inputs(bank, q)
                prep += time.perf_counter() - t
                out = engine._launch_branch(bank, inputs, host)
                # Replaying a graph overwrites its output buffers.
                outputs.append({k: v.clone() for k, v in out.items()})
            finally:
                slot["used"].record(compute)
        result = {k: torch.cat([o[k] for o in outputs]).cpu() for k in outputs[0]}
        total = time.perf_counter() - start
        if timings is not None:
            timings.update(prep=prep * 1000, model=(total - prep) * 1000, total=total * 1000)
        return result
