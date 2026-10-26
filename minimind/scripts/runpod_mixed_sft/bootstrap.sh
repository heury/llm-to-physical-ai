#!/usr/bin/env bash
set -euo pipefail
kind=$1
cd /tmp/minimind-mixed-sft-source
python3 -m pip install --quiet --break-system-packages 'transformers==4.57.6' 'datasets==3.6.0' 'wandb==0.22.3' 'numpy==1.26.4' 'pyarrow==19.0.1' 'jinja2==3.1.2' 'psutil==5.9.8' 'pyahocorasick==2.1.0'
python3 -m pip uninstall -y torchvision >/dev/null 2>&1 || true
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
if [ "$kind" = dense ]; then
 python3 scripts/runpod_mixed_sft/prepare.py > /shared-volume/minimind-mixed-sft-20261024/prepare.log 2>&1
fi
export TOKENIZERS_PARALLELISM=false
exec python3 scripts/runpod_mixed_sft/run.py --kind "$kind"
