#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
export OMNI_RUN_ROOT=/workspace/omni-mini-20261025
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
cd "$OMNI_RUN_ROOT/repo"
python scripts/runpod_omni_mini/train.py
python scripts/runpod_omni_mini/evaluate.py
