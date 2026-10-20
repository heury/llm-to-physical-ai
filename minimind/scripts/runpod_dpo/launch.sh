#!/bin/bash
set -euo pipefail
umask 077
export DPO_ROOT=/shared-volume/minimind-dpo-ultrafeedback-20261020
export DPO_RUNTIME=/tmp/minimind-dpo-ultrafeedback-20261020
mkdir -p "$DPO_ROOT"
exec 9>"$DPO_RUNTIME/run.lock"
flock -n 9
trap 'echo $? > "$DPO_ROOT/exit_code"' EXIT
test ! -e "$DPO_ROOT/SUCCESS"
export HF_HOME="$DPO_RUNTIME/hf-cache" HF_DATASETS_CACHE="$DPO_RUNTIME/datasets-cache"
export WANDB_DIR="$DPO_RUNTIME/wandb" WANDB_CACHE_DIR="$DPO_RUNTIME/wandb-cache" WANDB_MODE=online
export PYTHONPATH="$DPO_RUNTIME/repo" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$WANDB_DIR"
python=/tmp/minimind-checkpoint-test-20261020/venv/bin/python
"$python" -m pip freeze > "$DPO_ROOT/requirements-resolved.txt"
"$python" "$DPO_RUNTIME/prepare.py"
DPO_PHASE=smoke "$python" "$DPO_RUNTIME/evaluate.py" > "$DPO_ROOT/evaluation_smoke.log" 2>&1
DPO_PHASE=before "$python" "$DPO_RUNTIME/evaluate.py" > "$DPO_ROOT/evaluation_before.log" 2>&1
cd "$DPO_RUNTIME/repo/trainer"
"$python" train_dpo.py --hidden_size 768 --num_hidden_layers 8 --epochs 1 \
 --batch_size 32 --accumulation_steps 2 --learning_rate 0.0000005 --beta 0.1 --max_seq_len 1024 \
 --dtype bfloat16 --num_workers 4 --log_interval 10 --save_interval 250 \
 --data_path "$DPO_RUNTIME/train_prefs.jsonl" --from_weight full_sft \
 --save_dir "$DPO_ROOT/out" --save_weight dpo --use_wandb --wandb_project MiniMind-UltraFeedback-DPO > "$DPO_ROOT/training.log" 2>&1
DPO_PHASE=after "$python" "$DPO_RUNTIME/evaluate.py" > "$DPO_ROOT/evaluation_after.log" 2>&1
"$python" "$DPO_RUNTIME/package.py"
echo completed > "$DPO_ROOT/SUCCESS"
