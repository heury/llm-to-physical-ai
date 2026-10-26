#!/usr/bin/env bash
set -euo pipefail
ROOT=/shared-volume/minimind-pretrain-rtx5090-20261023
RUNTIME=/tmp/minimind-pretrain-rtx5090
mkdir -p "$ROOT/wandb" "$RUNTIME/repo"
exec 9>"$RUNTIME/run.lock"
flock -n 9 || exit 1
python -m pip install --break-system-packages --disable-pip-version-check 'transformers==4.57.6' 'datasets==3.6.0' 'wandb==0.22.3' > "$ROOT/install.log" 2>&1
python -u "$RUNTIME/repo/scripts/runpod_pretrain/run.py" > "$ROOT/training.log" 2>&1
