"""Release-format checkpoints (glance_config.json + adapter/heads safetensors) round-trip on the tiny model."""

import pytest
import torch

from glance_engine.checkpoints import load_release_weights, release_config

PREVIEW_CONFIG = {  # Vision-Decision/Glance-4B-preview glance_config.json (v1, step 3844)
    "base_model": "Qwen/Qwen3.5-4B", "attn_mask": "block_bidir",
    "lora": {"r": 64, "lora_alpha": 128, "target_modules": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
             r"in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"},
    "heads": {"head_dim": 512, "edge_bins": 64, "edge_lo": -0.1, "edge_hi": 1.0},
    "max_vis_tokens": 640, "max_upscale": 2.0, "max_q_tokens": 40, "max_q_tokens_typed": 384,
    "max_state_tokens": 1536, "temperatures": {"presence": 0.948836, "choice": 1.046105, "noul": 0.889068},
}


def test_release_config_maps_onto_training_config():
    cfg = release_config(PREVIEW_CONFIG)
    assert (cfg.task, cfg.lora_r, cfg.lora_alpha) == ("v1", 64, 128)
    assert (cfg.attn_mask, cfg.max_vis_tokens) == ("block_bidir", 640)
    assert (cfg.head_dim, cfg.edge_bins, cfg.edge_lo, cfg.edge_hi) == (512, 64, -0.1, 1.0)


def _tiny_with_lora(seed):
    from peft import LoraConfig, inject_adapter_in_model

    from glance_engine.checkpoints import tiny

    loaded = tiny(seed=seed)
    net, heads = loaded.engine.net, loaded.engine.heads
    lc = LoraConfig(r=4, lora_alpha=8, target_modules=release_config(PREVIEW_CONFIG).lora_target, bias="none")
    inject_adapter_in_model(lc, net.hf)
    torch.manual_seed(seed)
    for n, p in net.hf.named_parameters():
        if "lora_" in n:
            torch.nn.init.normal_(p, std=0.02)
    return net, heads


def test_adapter_and_heads_round_trip(tmp_path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    src_net, src_heads = _tiny_with_lora(seed=1)
    save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(src_net.hf).items()}, tmp_path / "a.safetensors")
    save_file({k: v.contiguous() for k, v in src_heads.state_dict().items()}, tmp_path / "h.safetensors")

    dst_net, dst_heads = _tiny_with_lora(seed=2)
    load_release_weights(dst_net, dst_heads, tmp_path / "a.safetensors", tmp_path / "h.safetensors")
    src = get_peft_model_state_dict(src_net.hf)
    dst = get_peft_model_state_dict(dst_net.hf)
    assert src.keys() == dst.keys() and all(torch.equal(src[k], dst[k]) for k in src)
    assert all(torch.equal(a, b) for a, b in zip(src_heads.state_dict().values(), dst_heads.state_dict().values()))

    save_file({k: v for k, v in list(src.items())[:-2]}, tmp_path / "short.safetensors")
    with pytest.raises(KeyError):
        load_release_weights(dst_net, dst_heads, tmp_path / "short.safetensors", tmp_path / "h.safetensors")
