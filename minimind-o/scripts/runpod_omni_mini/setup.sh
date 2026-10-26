#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 PIP_CACHE_DIR=/workspace/pip-cache
export OMNI_RUN_ROOT=/workspace/omni-mini-20261025
python -m pip install --break-system-packages 'transformers==4.57.6' 'huggingface_hub>=0.34,<1' 'pyarrow>=19' 'Pillow==11.3.0' soundfile librosa onnxruntime funasr addict modelscope datasets
python - <<'PY'
import torch
print('TORCH',torch.__version__,'CUDA',torch.version.cuda)
assert torch.cuda.is_available()
print('GPU',torch.cuda.get_device_name(), torch.cuda.get_device_capability())
x=torch.randn(256,256,device='cuda');print('CUDA matmul:',float((x@x).sum()))
PY
python /workspace/omni-mini-20261025/repo/scripts/runpod_omni_mini/prepare.py
