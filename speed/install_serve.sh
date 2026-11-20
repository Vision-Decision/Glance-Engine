#!/usr/bin/env bash
# VM side: separate venvs for the generative baselines (verbose log). bash speed/install_serve.sh > /content/serve_install.log 2>&1
# Colab exports a system-python default for `uv pip`, so every install names its interpreter explicitly.
set -x
pip install -q uv
uv python install 3.12
uv venv --allow-existing -p 3.12 /content/vllm_env
/content/vllm_env/bin/python -c "import vllm" 2>/dev/null || uv pip install --python /content/vllm_env/bin/python vllm 2>&1 | tail -15
/content/vllm_env/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
uv venv --allow-existing -p 3.12 /content/sglang_env
if ! /content/sglang_env/bin/python -c "import sglang" 2>/dev/null; then
  uv pip install --python /content/sglang_env/bin/python sglang 2>&1 | tail -5
  if ! /content/sglang_env/bin/python -c "import sglang" 2>/dev/null; then   # outlines-core may need a Rust build
    command -v cargo || (curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1)
    export PATH="$HOME/.cargo/bin:$PATH"
    uv pip install --python /content/sglang_env/bin/python sglang 2>&1 | tail -15
  fi
fi
/content/sglang_env/bin/python -c "import sglang,torch;print('sglang',sglang.__version__,'torch',torch.__version__)"
echo SERVE_INSTALL_DONE
