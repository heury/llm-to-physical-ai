#!/bin/bash
set -euo pipefail
export GRPO_ROOT=/shared-volume/minimind-grpo-ultrafeedback-20261020
export GRPO_RUNTIME=/tmp/minimind-grpo-ultrafeedback-20261020
export HF_HOME="$GRPO_RUNTIME/hf-cache" HF_DATASETS_CACHE="$GRPO_RUNTIME/datasets-cache"
export PYTHONPATH="$GRPO_RUNTIME/repo" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
trap 'echo $? > "$GRPO_ROOT/smoke_exit_code"' EXIT
/tmp/minimind-checkpoint-test-20261020/venv/bin/python "$GRPO_RUNTIME/smoke.py"
