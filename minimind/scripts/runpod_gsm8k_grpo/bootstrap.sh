#!/usr/bin/env bash
set -euo pipefail
cd /tmp/minimind-gsm8k-grpo-source
python3 -m pip install --quiet --break-system-packages 'transformers==4.57.6' 'huggingface_hub==0.36.0' 'numpy==1.26.4' 'safetensors>=0.4' 'jinja2==3.1.2'
python3 -m pip uninstall -y torchvision >/dev/null 2>&1 || true
export HF_HOME=/shared-volume/minimind-gsm8k-grpo-20261025/hf-home
export TOKENIZERS_PARALLELISM=false
exec python3 scripts/runpod_gsm8k_grpo/run.py --kinds "$@"
