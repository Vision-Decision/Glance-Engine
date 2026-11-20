# Development

## Install

pip installs glance-core, a dependency, from its public GitHub repository at a pinned tag.

```bash
git clone git@github.com:Vision-Decision/glance-engine.git && cd glance-engine
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' -c constraints/cpu-tested.txt          # CPU: tests and the tiny model
pip install -e '.[gpu,serve]'                                   # CUDA: FLA kernels and the HTTP server
pip install -e '.[gpu,serve,sm120-prefill]'                     # RTX PRO 6000 / SM120: also FlashQLA GDN prefill
```

On SM120 with `[sm120-prefill]`, launch with `FLA_FLASH_QLA=0`. On Colab, `pip uninstall -y torchao` first: the preinstalled torchao 0.10.0 makes peft refuse to import. `python -m glance_engine.kernels` prints which optional kernels the environment supports.

## Tests

```bash
pytest                       # everything that runs on CPU (about a minute, most of it the tiny exactness suite)
pytest -m "not slow"         # skip the subprocess exactness suite
ruff check glance_engine tests examples conftest.py
```

| Test | What it checks |
|---|---|
| [`tests/test_equivalence_cpu.py`](../tests/test_equivalence_cpu.py) | Runs `python -m scan.test_scan_equivalence`: cached, fresh, query-first and clip scans, packed varlen prefill and the Scanner against the reference forward, tiny FP32 model |
| [`tests/test_api_tiny.py`](../tests/test_api_tiny.py) | `Glance` on the tiny model: typed answers, cached follow-ups equal fresh answers, broadcast equals one-by-one, eviction, temperatures, Scanner |
| [`tests/test_serve_tiny.py`](../tests/test_serve_tiny.py) | The real worker behind the aiohttp app: decisions, image-id follow-ups, 404/400 paths, chat completions |
| [`tests/test_release_format.py`](../tests/test_release_format.py) | Release-format checkpoint mapping and LoRA/head round trip |
| [`tests/test_questions.py`](../tests/test_questions.py) | Wire format, validation, temperature scaling |
| [`speed/test_scheduler.py`](../speed/test_scheduler.py), [`speed/test_server.py`](../speed/test_server.py), [`speed/test_w12.py`](../speed/test_w12.py), [`scan/test_bank.py`](../scan/test_bank.py) | Model-free scheduler, transport, W12 codec and bank storage tests |

The first run downloads the Qwen3.5-4B config and processor files (a few MB, no weights). On CPU, transformers warns that flash-linear-attention is missing and torch warns that CUDA autocast is disabled; both are expected.

GPU equivalence gates need real weights and are run by hand; see [exactness](exactness.md#running-the-gates).

## Continuous integration

[`.github/workflows/cpu-tests.yml`](../.github/workflows/cpu-tests.yml) runs ruff and pytest on Python 3.10 and 3.12 with CPU torch. It needs no secrets: glance-core and glancex are public and install straight from GitHub.

There are no GPU jobs.

## Conventions

- Engine changes keep the reference path available behind a flag or environment variable (for example `GLANCE_BOX_REDUCTION=legacy`, `--lm_prefill reference`, `bank_format="legacy"`) and are accepted only with the equivalence gates passing.
- Benchmark claims cite a raw JSON whose `.log` records the command, GPU and source hashes.
- Large files (over 5 MB) are not committed; [`LARGE_FILES.tsv`](../LARGE_FILES.tsv) lists the ones restored with `python scripts/fetch_large_files.py`.

## Regenerating figures

```bash
python docs/assets/make_architecture.py          # docs/assets/architecture-{light,dark}.svg
python demo/render_search.py                                       # search_256.gif from demo/data/
python demo/render_race.py --image path/to/000000275198.jpg       # glance_vs_vllm.gif; the COCO val2017 photo is not committed
```
