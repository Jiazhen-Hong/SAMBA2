#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${REPOSITORY:-$HOME/Github/SAMBA2}"
PYTHON="${PYTHON:-$HOME/venvs/eeg26-track2-clean/bin/python}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/data/SAMBA-tmp/python312-packages}"
NEURALBENCH_CONFIG="${NEURALBENCH_CONFIG:-/data/SAMBA-tmp/neuralbench-config.json}"
CHECKPOINT="${CHECKPOINT:-$REPOSITORY/checkpoints/SAMBA-100s}"
GPU="${GPU:-0}"
TASKS="${TASKS:-pathology depression_diagnosis schizophrenia_diagnosis}"

cd "$REPOSITORY"
mkdir -p logs

if [[ ! -f "$CHECKPOINT/config.json" || ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "Incomplete SAMBA-100s checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$NEURALBENCH_CONFIG" ]]; then
  echo "Missing NeuralBench config: $NEURALBENCH_CONFIG" >&2
  exit 1
fi

for task in $TASKS; do
  echo "===== SAMBA-100s linear probe: $task ====="
  env \
    NEURALBENCH_CONFIG="$NEURALBENCH_CONFIG" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$PACKAGE_ROOT:$REPOSITORY/src" \
    "$PYTHON" scripts/evaluate_neuralbench.py \
      "$CHECKPOINT" \
      --task "$task" \
      --name samba-100s \
      --no-download \
      --no-prepare \
      2>&1 | tee "logs/samba-100s-${task}.log"
done
