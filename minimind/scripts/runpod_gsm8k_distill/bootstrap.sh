#!/usr/bin/env bash
set -euo pipefail
cd /workspace/moe-dense-gsm8k-distill-20261025
python3 -m venv --system-site-packages venv
venv/bin/pip install 'transformers==4.57.6' 'huggingface_hub<1' 'numpy<3' pyarrow safetensors sentencepiece > install.log 2>&1
export HF_HOME=/workspace/moe-dense-gsm8k-distill-20261025/hf-cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
set +e
venv/bin/python -u code/run.py > training.log 2>&1
code=$?
printf '%s\n' "$code" > exit_code
exit "$code"
