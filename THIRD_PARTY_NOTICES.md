# Third-party notices

## Vendored source

| Path | Upstream | License | Changes | Used by |
|---|---|---|---|---|
| [`speed/specpilot/w12.cu`](speed/specpilot/w12.cu) | [StargazerX0/SpecPilot](https://github.com/StargazerX0/SpecPilot) @ `1058dc2`, `specpilot_d2_w12.py` (PRO6000 D2) | Apache-2.0, [text](speed/specpilot/LICENSE) | Standalone PyTorch binding, exact linear specializations only, direct driver API, Glance-selected stream-K geometry (header comment in the file) | Opt-in W12 lossless weights (`speed/w12.py`) |
| [`scan/flash_vision/params.h`](scan/flash_vision/params.h) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) @ `14c3779` (the PyTorch submodule), Copyright (c) 2023 Tri Dao | BSD-3-Clause, [text](scan/flash_vision/LICENSE) | Forward argument layout only | Split-KV attention probe (`scan/splitkv_probe.py`); not on the production path |
| [`scan/flash_vision/attention.cpp`](scan/flash_vision/attention.cpp) | Launcher written here against the FA2 kernels built into PyTorch | BSD-3-Clause (with the header it includes) | — | Same probe |

Both license files ship in the wheel (`[tool.setuptools.package-data]` in `pyproject.toml`). The [`NOTICE`](NOTICE) file carries the attribution required by Apache-2.0 §4(d).

## Runtime dependencies

Installed from PyPI or GitHub, not vendored. Licenses as declared in each package's metadata.

| Package | Version | License | Needed for |
|---|---|---|---|
| glance-core (`dv`) | v0.2.0 | Vision-Decision, Apache-2.0 | Model definition, data collation, reference forward |
| PyTorch | ≥ 2.6 | BSD-3-Clause | Everything |
| transformers | 5.23.0 | Apache-2.0 | Qwen3.5 model and processor |
| peft | ≥ 0.17 | Apache-2.0 | LoRA loading before merge |
| flash-linear-attention | 0.6.0 | MIT | `[gpu]` fused GDN kernels |
| flash-qla | 0.1.3 | MIT | `[sm120-prefill]` |
| tilelang | 0.1.15 | MIT | `[sm120-prefill]` |
| aiohttp | ≥ 3.9 | Apache-2.0 AND MIT | `[serve]` HTTP server |

The serving baselines (vLLM 0.34.0, Apache-2.0; SGLang 0.5.23, Apache-2.0) run in separate virtual environments and are never imported by the engine.

## Models and data

- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0, downloaded at run time.
- Glance weights: [Vision-Decision/Glance-4B-preview](https://huggingface.co/Vision-Decision/Glance-4B-preview), CC BY-NC 4.0 per its model card.
- Demo images in [`demo/`](demo/): COCO val2017 photographs under CC BY 2.0 or "no known copyright restrictions", one credit line per image in [`demo/data/credits.csv`](demo/data/credits.csv).
