#!/usr/bin/env bash
set -euo pipefail
kind=$1
cd /tmp/minimind-gsm8k-source
python3 -m pip install --quiet --break-system-packages 'transformers==4.57.6' 'datasets==3.6.0' 'wandb==0.22.3' 'numpy==1.26.4' 'pyarrow==19.0.1' 'jinja2==3.1.2' 'psutil==5.9.8'
# torchvision is not used by these language models; remove incompatible preinstalled copy.
python3 -m pip uninstall -y torchvision >/dev/null 2>&1 || true
if [ "$kind" = dense ]; then
  set +e
  python3 scripts/runpod_gsm8k/prepare.py > /shared-volume/minimind-gsm8k-sft-20261024/prepare.log 2>&1
  code=$?
  echo "$code" > /shared-volume/minimind-gsm8k-sft-20261024/prepare.exit
  set -e
  if [ "$code" != 0 ]; then exit "$code"; fi
fi
exec python3 scripts/runpod_gsm8k/run.py --kind "$kind"
