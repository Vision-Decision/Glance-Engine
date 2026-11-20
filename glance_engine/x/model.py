"""Load Glance-X checkpoints into an :class:`~glance_engine.x.engine.XEngine`.

Accepted sources, as for v1 (see ``glance_engine.checkpoints``):

``"Vision-Decision/Glance-X-Lite"``, ``"Vision-Decision/Glance-X-4B"``
    A Hugging Face repository in the glancex release format: ``glancex_config.json``,
    ``adapter_model.safetensors`` (LoRA) and ``heads.safetensors``. A local directory with the same files works too.
``"ckpt_final.pt"`` or ``"org/repo:path/ckpt_final.pt"``
    A glancex training checkpoint (``torch.save`` dict with ``cfg``, ``lora`` and ``heads``; ``cfg`` has
    ``bidir_k``). This is how unreleased X-series weights (for example Glance-X2) are loaded.
``"tiny-x"``
    A tiny random Glance-X (8 LM layers, K=2) in FP32 on CPU, for tests and smoke runs. Answers are meaningless.

Real checkpoints run in BF16 on CUDA with the LoRA merged into the base weights (``exact=False``, the default), as in
glancex's fast path. ``exact=True`` keeps the LoRA unmerged: the training-time computation.
"""

from __future__ import annotations

import json
import os

import torch

X_CONFIG = "glancex_config.json"
_X_FILES = (X_CONFIG, "adapter_model.safetensors", "heads.safetensors")
TINY_BASE = "Qwen/Qwen3.5-4B"  # only its config and processor are downloaded (the same files the v1 tiny model uses)


def is_x_engine(engine) -> bool:
    return getattr(engine, "family", None) == "x"


# ------------------------------------------------------------------------------------------------ source detection
def x_state(spec: str):
    """Return the glancex state dict (``cfg``, ``lora``, ``heads``) if ``spec`` names a Glance-X checkpoint, else None.

    Only cheap parts are read to decide: a ``.pt`` is memory-mapped, a Hub repo is asked for ``glancex_config.json``.
    """
    if spec == "tiny-x":
        return None
    if spec.endswith(".pt"):
        from glance_engine.checkpoints import _fetch_pt

        try:
            sd = torch.load(_fetch_pt(spec), map_location="cpu", weights_only=False, mmap=True)
        except Exception:  # unreadable here (for example a legacy, non-zip file): the v1 loader decides
            return None
        cfg = sd.get("cfg") if isinstance(sd, dict) else None
        return sd if isinstance(cfg, dict) and "bidir_k" in cfg else None
    if os.path.isdir(spec):
        return _read_release(spec) if os.path.exists(os.path.join(spec, X_CONFIG)) else None
    if "/" not in spec or spec.count("/") != 1:
        return None
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(spec, X_CONFIG)
    except Exception:  # not a glancex release (or not reachable): let the v1 loader report it
        return None
    from huggingface_hub import snapshot_download

    return _read_release(snapshot_download(spec, allow_patterns=list(_X_FILES)))


def _read_release(root):
    from safetensors.torch import load_file

    missing = [f for f in _X_FILES if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"{root} lacks {missing}")
    with open(os.path.join(root, X_CONFIG)) as f:
        cfg = json.load(f)
    lora, heads = (load_file(os.path.join(root, f)) for f in _X_FILES[1:])
    return dict(cfg=cfg, lora=lora, heads=heads)


# ------------------------------------------------------------------------------------------------ loading
def load(state: dict, *, device: str | None = None, exact: bool = False, graphs: bool = True, base: str | None = None,
         source: str = ""):
    """Build the backbone and heads from a glancex state dict and return an ``XEngine`` (BF16 on CUDA).

    ``graphs`` replays the image-first encode and answer steps as CUDA graphs per shape bucket (not with ``exact``).
    """
    from glancex.build import build, load_ckpt
    from glancex.config import Config

    from glance_engine.x.engine import XEngine

    device = device or "cuda"
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real Glance-X checkpoints run in BF16 on CUDA; use model='tiny-x' for a CPU smoke test")
    cfg = Config().update(state["cfg"]).update(_INFERENCE)
    if base:
        cfg = cfg.update(dict(model_id=base))
    net, heads, proc = build(cfg, device, dtype=torch.bfloat16)
    load_ckpt(state, net, heads)
    if not exact:
        from glancex.inference import merge_lora

        merge_lora(net.hf)
    eng = XEngine(net.eval(), heads.eval(), proc, cfg, device, source=source, gpu_prep=not exact)
    eng.graphs = graphs and not exact
    return eng


# glancex pads to fixed shapes during training (its Hopper kernels compile per shape); inference uses batch maxima.
_INFERENCE = dict(grad_ckpt=False, pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0, pad_t2i_block_to=0,
                  pad_bi_block_to=0)


def tiny(device: str = "cpu", seed: int = 0, max_vis: int = 64, bidir_k: int = 2, mode_probs: str = "0.5,0.25,0.25"):
    """Tiny random Glance-X: 8 LM layers of width 128, K=2, one vision block, FP32. Same code paths as a real one."""
    from glancex.backbone import GlanceXBackbone
    from glancex.config import Config
    from glancex.heads import GlanceXHeads
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from glance_engine.x.engine import XEngine

    torch.manual_seed(seed)
    proc = AutoProcessor.from_pretrained(TINY_BASE)
    c = AutoConfig.from_pretrained(TINY_BASE)
    t = c.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
    t.layer_types = t.layer_types[:8]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    v = c.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
    hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(device).eval()
    cfg = Config().update(dict(model_id=TINY_BASE, lora_r=0, bidir_k=bidir_k, max_vis_tokens=max_vis, head_dim=64,
                               mode_probs=mode_probs, **_INFERENCE))
    net = GlanceXBackbone(hf, cfg).eval()
    heads = GlanceXHeads(net.hidden, cfg).to(device).eval()
    return XEngine(net, heads, proc, cfg, device, source="tiny-x")
