"""Custom kernels and the environment they need. ``python -m glance_engine.kernels`` prints a report.

Every kernel here is optional. When a requirement is missing, the engine keeps the reference
PyTorch/FLA operation for that stage and the answers stay within the same equivalence gates.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

KERNELS = [
    # name, module, what it does, requirement
    ("lm_prefill", "speed/lm_prefill.py",
     "packed BF16 LM prefill: gated norm, conv taps, head norm, fused SwiGLU epilogue", "Triton, SM120"),
    ("gdn_branch", "speed/gdn_branch.py",
     "output-only GDN continuations of many question branches from one read-only FP32 state", "Triton, SM120"),
    ("flash_gdn", "speed/flash_gdn.py",
     "chunked GDN prefill with FlashQLA's 64-wide value tile on large packed calls", "[sm120-prefill] extra, SM120"),
    ("box_scores", "speed/box_scores_kernel.py",
     "box centre scores with one fixed reduction tree per patch (layout-independent)", "Triton"),
    ("vision_mlp", "speed/vision_mlp.py",
     "fused vision MLP projection and GELU with reference rounding", "Triton, SM120"),
    ("vision_rotary", "speed/vision_rotary.py", "vision RoPE on Q and K from one strided QKV projection", "Triton"),
    ("frame_pixels", "speed/frame_pixels.py", "patchify compact uint8 RGB frames on the device", "Triton"),
    ("bank_decode", "scan/bank_kernels.py",
     "decode lossless exponent-window bank state and gather into the clip BMM layout", "Triton"),
    ("w12", "speed/w12.py + speed/specpilot/w12.cu",
     "opt-in lossless packed BF16 weights (exponent window + escapes) for short, bandwidth-bound projections",
     "nvcc + Ninja, SM120"),
    ("fla", "flash-linear-attention 0.6.0", "reference fused GDN chunk/recurrent kernels", "[gpu] extra"),
]


def _ver(dist):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def report() -> dict:
    """Detected environment: device, capability and the optional packages each kernel needs."""
    import torch

    cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if cuda else None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": _ver("transformers"),
        "glance-core": _ver("glance-core"),
        "cuda_device": torch.cuda.get_device_name() if cuda else None,
        "sm120": cap == (12, 0),
        "triton": importlib.util.find_spec("triton") is not None,
        "flash-linear-attention": _ver("flash-linear-attention"),
        "flash-qla": _ver("flash-qla"),
        "tilelang": _ver("tilelang"),
        "aiohttp": _ver("aiohttp"),
        "nvcc": shutil.which("nvcc") is not None,
        "sm120_prefill_ready": _ver("flash-qla") == "0.1.3" and _ver("tilelang") == "0.1.15",
    }


def main():
    env = report()
    width = max(map(len, env))
    for k, v in env.items():
        print(f"{k:<{width}}  {v}")
    print()
    for name, where, what, needs in KERNELS:
        print(f"{name:<13} {needs:<30} {what}  [{where}]")
    if not env["cuda_device"]:
        print("\nNo CUDA device: only Glance.tiny() and the CPU tests run here.")


if __name__ == "__main__":
    main()
