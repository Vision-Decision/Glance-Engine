"""Build a ready-to-run scan engine from a checkpoint.

Three sources are accepted:

``"Vision-Decision/Glance-4B-preview"``
    A Hugging Face repository in the public release format: ``glance_config.json``,
    ``adapter_model.safetensors`` (LoRA) and ``heads.safetensors``. A local directory with
    the same three files also works. The config's post-hoc temperatures are returned so
    answers can be calibrated.
``"ckpt_final.pt"`` or ``"Vision-Decision/glance-train-ckpts-legacy:v1/ckpt_final.pt"``
    A training checkpoint (``torch.save`` dict with ``cfg``, ``lora`` and ``heads``), local or
    ``repo:path`` on the Hub. This is the format the benchmark scripts load.
``"tiny"``
    A tiny random Qwen3.5 configuration in FP32 on CPU. Answers are meaningless; it runs
    the same engine code paths and is what the CPU tests and the quickstart smoke test use.

Glance-X checkpoints (``glancex_config.json`` releases such as ``"Vision-Decision/Glance-X-Lite"``,
glancex training ``.pt`` files, and ``"tiny-x"``) are recognised here and built by
:mod:`glance_engine.x` instead; see that module.

Real checkpoints load the Qwen3.5-4B base weights from the Hub, merge the LoRA into
BF16 weights and fuse projections, as in the benchmarks. They require CUDA.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

DEFAULT_MODEL = "Vision-Decision/Glance-4B-preview"
BASE_MODEL = "Qwen/Qwen3.5-4B"
_RELEASE_FILES = ("glance_config.json", "adapter_model.safetensors", "heads.safetensors")


@dataclass
class Loaded:
    engine: object
    temperatures: dict = field(default_factory=dict)
    max_vis: int = 640
    source: str = ""
    family: str = "v1"  # "x" for Glance-X models (glance_engine.x)


def load(model: str = DEFAULT_MODEL, *, device: str | None = None, fp8: bool = False, w12: str | None = None,
         graphs: bool = True, exact: bool = False) -> Loaded:
    """Resolve ``model`` and return the engine plus its calibration temperatures.

    ``exact`` (Glance-X only) keeps the LoRA unmerged, the training-time computation.
    """
    if model == "tiny":
        return tiny(device=device or "cpu")
    if model == "tiny-x":
        return tiny_x(device=device or "cpu")
    x = _x_state(model)
    if x is not None:
        from glance_engine.x import model as X

        if fp8 or w12:
            raise ValueError("fp8 and w12 weight formats are not available for Glance-X models")
        eng = X.load(x, device=device, exact=exact, graphs=graphs, source=model)
        return Loaded(eng, {}, eng.max_vis, model, "x")
    if exact:
        raise ValueError("exact=True applies to Glance-X models")
    device = device or "cuda"
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real Glance checkpoints run in BF16 on CUDA; use model='tiny' for a CPU smoke test")
    if fp8 and w12:
        raise ValueError("choose one LM weight format: fp8 or w12")
    if model.endswith(".pt"):
        cfg, net, heads, proc, temps = _load_training_ckpt(_fetch_pt(model), device)
    else:
        cfg, net, heads, proc, temps = _load_release(_fetch_release(model), device)
    _optimize_weights(net, fp8=fp8, w12=w12)
    eng = _engine(cfg, net, heads, proc)
    eng.graphs = graphs
    for tower in (eng.fv, eng.frame_tower):
        if tower is not None:
            tower.graphs = graphs
    return Loaded(eng, temps, cfg.max_vis_tokens, model)


def tiny(device: str = "cpu", seed: int = 0, max_vis: int = 64) -> Loaded:
    """Tiny random model (8 LM layers of width 128, one vision block). Downloads only the base config and processor."""
    from dv.config import Config
    from dv.data.mixed import ChatTok
    from dv.model.backbone import SharedPrefixLM
    from dv.model.build import V1Heads
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    import speed.engine as SE
    from scan.engine import ScanEngine

    torch.manual_seed(seed)
    proc = AutoProcessor.from_pretrained(BASE_MODEL)
    c = AutoConfig.from_pretrained(BASE_MODEL)
    t = c.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
    t.layer_types = t.layer_types[:8]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    v = c.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
    hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(device).eval()
    SE.USE_TAPS = False  # the compiled conv taps are a GPU fast path; CPU uses F.conv1d
    cfg = Config().update(dict(task="v1", attn_mask="block_bidir", max_vis_tokens=max_vis, grad_ckpt=False))
    net = SharedPrefixLM(hf, cfg).eval()
    tok = ChatTok(proc, cfg)
    heads = V1Heads(net.hidden, cfg, hf.get_output_embeddings().weight, tok.label_token_groups()).to(device).eval()
    for p in heads.parameters():  # non-trivial typed residual (zero-initialised in training)
        if p.dim() == 2 and p.shape[0] < 40:
            torch.nn.init.normal_(p, std=0.02)
    eng = ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=False)
    return Loaded(eng, {}, max_vis, "tiny")


def tiny_x(device: str = "cpu", seed: int = 0, max_vis: int = 64) -> Loaded:
    """Tiny random Glance-X (8 LM layers of width 128, K=2), all three directions. See glance_engine.x.model.tiny."""
    from glance_engine.x import model as X

    eng = X.tiny(device=device, seed=seed, max_vis=max_vis)
    return Loaded(eng, {}, max_vis, "tiny-x", "x")


def _x_state(model):
    try:
        from glance_engine.x.model import x_state
    except ImportError:  # pragma: no cover
        return None
    return x_state(model)


# ------------------------------------------------------------------------------------------------ sources
def _fetch_pt(spec):
    if os.path.exists(spec):
        return spec
    repo, sep, path = spec.partition(":")
    if not sep:
        raise FileNotFoundError(f"{spec}: not a local file; use 'org/repo:path/to/ckpt.pt' for the Hub")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path)


def _fetch_release(spec):
    if os.path.isdir(spec):
        missing = [f for f in _RELEASE_FILES if not os.path.exists(os.path.join(spec, f))]
        if missing:
            raise FileNotFoundError(f"{spec} lacks {missing}")
        return spec
    from huggingface_hub import snapshot_download

    return snapshot_download(spec, allow_patterns=list(_RELEASE_FILES))


def _engine(cfg, net, heads, proc):
    from dv.data.mixed import ChatTok

    from scan.engine import ScanEngine

    net.eval()
    heads.eval()
    return ScanEngine(net, heads, ChatTok(proc, cfg), proc, graphs=False, fast_ops=True)


def _load_training_ckpt(path, device):
    from dv.config import Config
    from dv.model.build import build, load_ckpt

    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
    net, heads, proc = build(cfg, device)
    load_ckpt(path, net, heads)
    return cfg, net, heads, proc, {}


def release_config(gc: dict):
    """Map a release ``glance_config.json`` onto the training Config."""
    from dv.config import Config

    fields = dict(
        task="v1", model_id=gc.get("base_model", BASE_MODEL), attn_mask=gc["attn_mask"], grad_ckpt=False,
        lora_r=gc["lora"]["r"], lora_alpha=gc["lora"]["lora_alpha"], lora_target=gc["lora"]["target_modules"],
    )
    fields.update({k: gc["heads"][k] for k in ("head_dim", "edge_bins", "edge_lo", "edge_hi") if k in gc["heads"]})
    for k in ("max_vis_tokens", "max_upscale", "max_q_tokens", "max_q_tokens_typed", "max_state_tokens"):
        if k in gc:
            fields[k] = gc[k]
    return Config().update(fields)


def load_release_weights(net, heads, adapter_file, heads_file):
    """Load release-format LoRA and head tensors, refusing any key mismatch."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    lora = load_file(adapter_file)
    res = set_peft_model_state_dict(net.hf, lora)
    unexpected = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
    if unexpected:
        raise KeyError(f"unexpected LoRA keys: {unexpected[:5]}")
    expected = sum(1 for n, _ in net.hf.named_parameters() if "lora_" in n)
    if expected != len(lora):
        raise KeyError(f"checkpoint has {len(lora)} LoRA tensors, model expects {expected}")
    res = heads.load_state_dict(load_file(heads_file), strict=False)
    bad = [k for k in res.missing_keys if ".multi." not in k] + list(res.unexpected_keys)
    if bad:
        raise KeyError(f"head keys do not match the checkpoint: {bad[:5]}")


def _load_release(root, device):
    from dv.model.build import build

    with open(os.path.join(root, "glance_config.json")) as f:
        gc = json.load(f)
    cfg = release_config(gc)
    net, heads, proc = build(cfg, device)
    load_release_weights(net, heads, os.path.join(root, "adapter_model.safetensors"),
                         os.path.join(root, "heads.safetensors"))
    temps = {k: float(v) for k, v in gc.get("temperatures", {}).items()}
    return cfg, net, heads, proc, temps


def _optimize_weights(net, *, fp8, w12):
    """The benchmark weight path: merged LoRA, fused projections (as scan.pilot_charades.load_glance)."""
    from speed.engine import fp8_language_model_, fuse_weights_, merge_lora_

    merge_lora_(net.hf)
    fuse_weights_(net.lm, net.layer_types, keep_ab_separate=fp8)
    if fp8:
        fp8_language_model_(net.lm)
    if w12:
        from speed.w12 import install

        install(net.lm, w12)
