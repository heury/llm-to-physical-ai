#!/usr/bin/env bash
# Run inside a dedicated Pod; keep all evidence and environments on the volume.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 NOTEBOOK JOB_NAME ATTEMPT [EXTRA]" >&2
    exit 2
fi
notebook=$1
job_name=$2
attempt=$3
extra=${4:-}
source_dir=${UNSLOTH_VALIDATION_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
job_dir="/workspace/unsloth-validation/jobs/$job_name"
project_dir="$job_dir/project"
result_dir="$job_dir/$attempt"
mkdir -p "$project_dir/scripts" "$result_dir"
cp "$source_dir"/*.ipynb "$source_dir/pyproject.toml" "$source_dir/uv.lock" \
    "$source_dir/.python-version" "$project_dir/"
cp "$source_dir/scripts/execute_notebook.py" "$project_dir/scripts/"
exec > >(tee "$result_dir/launch.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$result_dir/exit_code"' EXIT

export HF_HOME=/workspace/unsloth-validation/hf-cache
export UV_CACHE_DIR=/workspace/unsloth-validation/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/unsloth-validation/uv-python
export UV_PROJECT_ENVIRONMENT="$job_dir/.venv"
export IPYTHONDIR="$job_dir/ipython"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TZ=Asia/Seoul
cd "$project_dir"
if ! command -v uv >/dev/null; then
    python3 -m pip install uv
fi
sync_args=(--locked --reinstall-package xformers)
if [[ -n $extra ]]; then sync_args+=(--extra "$extra"); fi
uv sync "${sync_args[@]}" > "$result_dir/install.log" 2>&1
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$result_dir/pip-check.log" 2>&1
uv run --no-sync python scripts/execute_notebook.py "$notebook" \
    --output-dir "$result_dir"
