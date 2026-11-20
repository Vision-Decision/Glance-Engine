"""Batched version of the exact singleton split-KV attention used by the fixed G4 frame grids."""

import warnings
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load():
    # The extension links a private FA2 launcher. Its parameter layout and heuristic
    # belong to this exact torch revision. Other builds use the per-image SDPA path.
    if torch.version.git_version != "5c4886908584029761b579af026dcfb627c84070":
        return None
    from torch.utils.cpp_extension import load as build

    try:
        build(
            name="glance_vision_splitkv",
            sources=[str(Path(__file__).with_name("flash_vision") / "attention.cpp")],
            with_cuda=True,
            is_python_module=False,
        )
    except (OSError, RuntimeError) as exc:
        warnings.warn(f"Using per-image vision SDPA: {exc}", RuntimeWarning)
        return None
    return torch.ops.glance_vision.split_attention


def batched(op, q, k, v, scale):
    splits = 2 if q.shape[1] == 468 else 4
    parts = [op(qi, ki, vi, splits, scale) for qi, ki, vi in zip(q.split(16), k.split(16), v.split(16))]
    return parts[0] if len(parts) == 1 else torch.cat(parts)
