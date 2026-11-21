#!/usr/bin/env bash
# P1 video-latency VM setup (Colab G4). bash speed/video_p1/setup_vm.sh > /content/p1_setup.log 2>&1
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" accelerate
CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
pip install -q torchcodec --index-url https://download.pytorch.org/whl/$CU || pip install -q torchcodec
python -c "import transformers, fla, torchcodec; print('transformers', transformers.__version__, 'fla', fla.__version__, 'torchcodec', torchcodec.__version__)"
ffmpeg -version | head -1
# 60 s 1080p30 H.264 test stream: moving test pattern + noise (realistic bitrate), no B-frames (live-stream setting)
[ -f /content/v1080.mp4 ] || ffmpeg -loglevel error -y -f lavfi -i "testsrc2=size=1920x1080:rate=30,noise=alls=12:allf=t" \
  -t 60 -c:v libx264 -preset veryfast -profile:v high -pix_fmt yuv420p -bf 0 -g 60 -b:v 8M /content/v1080.mp4
ls -la /content/v1080.mp4
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"]:
    print(snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
PY
echo SETUP_DONE
