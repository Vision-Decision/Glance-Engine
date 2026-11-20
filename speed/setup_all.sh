#!/usr/bin/env bash
# VM side: base setup, then the vLLM / SGLang venvs.  bash speed/setup_all.sh   (logs: /content/setup.log, /content/serve_install.log)
bash /content/dv/speed/setup_vm.sh > /content/setup.log 2>&1
bash /content/dv/speed/install_serve.sh > /content/serve_install.log 2>&1
