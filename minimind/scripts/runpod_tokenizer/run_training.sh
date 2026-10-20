#!/usr/bin/env bash
# Upload this repository's trainer/train_tokenizer.py and scripts/runpod_tokenizer
# preserving their relative paths, then run this script on the provisioned Pod.
set -euo pipefail
umask 077

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_root="${RUN_ROOT:-/shared-volume/minimind-tokenizer-6400}"
mountpoint -q /shared-volume || { echo '/shared-volume is not mounted' >&2; exit 2; }
case "$run_root" in
  /shared-volume/*) ;;
  *) echo 'RUN_ROOT must be under /shared-volume' >&2; exit 2 ;;
esac
mkdir -p "$run_root"
# Global Volumes use an object-storage FUSE filesystem. Keep package installs,
# caches, and advisory locks on the Pod's local disk; persist all results below.
local_root="${LOCAL_ROOT:-/tmp/minimind-tokenizer-runtime}"
mkdir -p "$local_root"
exec 9>"$local_root/training.lock"
flock -n 9 || { echo 'Another training process holds the lock' >&2; exit 2; }
test ! -e "$run_root/model/training_info.json" || {
  echo 'Training results already exist; select a new RUN_ROOT to preserve them' >&2
  exit 2
}
echo "$$" > "$run_root/launcher.pid"
trap 'result=$?; echo "$result" > "$run_root/exit_code"' EXIT

python3 - <<'PY'
from pathlib import Path
memory = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemTotal:'))) * 1024
for filename in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
    path = Path(filename)
    if path.exists():
        value = path.read_text().strip()
        if value.isdigit():
            memory = min(memory, int(value))
print(f'Available container RAM limit: {memory / 1024**3:.1f} GiB', flush=True)
if memory < 60 * 1024**3:
    raise SystemExit('Full-corpus BPE requires a larger host; provision at least 64 GiB RAM')
PY

python3 -m venv "$local_root/venv"
python_bin="$local_root/venv/bin/python"
"$python_bin" -m pip install --disable-pip-version-check --no-cache-dir \
  datasets==3.6.0 tokenizers==0.22.2 transformers==4.57.6 jinja2==3.1.6
"$python_bin" -m pip freeze > "$run_root/requirements-resolved.txt"
cp "$source_root/trainer/train_tokenizer.py" "$run_root/train_tokenizer.py"
cp "$source_root/scripts/runpod_tokenizer/validate.py" "$run_root/validate.py"

export HF_HOME="$local_root/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
"$python_bin" "$run_root/train_tokenizer.py" \
  --dataset HAERAE-HUB/KOREAN-WEBTEXT \
  --revision 2ad96e4983923d91350ab2214e059bd57219eddd \
  --split train --max-lines 0 --vocab-size 6400 \
  --tokenizer-dir "$run_root/model" --skip-eval \
  > "$run_root/training.log" 2>&1
"$python_bin" "$run_root/validate.py" "$run_root/model" \
  > "$run_root/validation.log" 2>&1
cp "$run_root/requirements-resolved.txt" "$run_root/model/requirements-resolved.txt"
tar -czf "$run_root/tokenizer-artifacts.tar.gz" -C "$run_root/model" \
  tokenizer.json tokenizer_config.json vocab.json merges.txt \
  training_info.json validation.json requirements-resolved.txt
sync
echo "Training and validation completed: $run_root/tokenizer-artifacts.tar.gz"
