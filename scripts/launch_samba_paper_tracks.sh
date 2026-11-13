#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${REPOSITORY:-$HOME/Github/SAMBA2}"
PYTHON="${PYTHON:-$HOME/venvs/eeg26-track2-clean/bin/python}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/data/SAMBA-tmp/python312-packages}"
DATA_ROOT="${DATA_ROOT:-/data/SAMBA-tmp}"
SAMBA_T_GPU="${SAMBA_T_GPU:-0}"
SAMBA_E_GPU="${SAMBA_E_GPU:-1}"

cd "$REPOSITORY"
mkdir -p logs checkpoints/SAMBA-T checkpoints/SAMBA-E

for manifest in \
  "$DATA_ROOT/samba-t-tuab100-manifest.json" \
  "$DATA_ROOT/samba-e-short-manifest.json"
do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest" >&2
    echo "Run: bash scripts/prepare_paper_track_manifests.sh" >&2
    exit 1
  fi
done

if pgrep -u "$USER" -f '[s]cripts/train_samba.py' >/dev/null; then
  echo "A SAMBA training process owned by $USER is already running." >&2
  echo "Stop or preserve it before launching the two new tracks." >&2
  exit 1
fi

for session in samba-t samba-e; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

tmux new-session -d -s samba-t \
  "cd '$REPOSITORY' && env CUDA_VISIBLE_DEVICES='$SAMBA_T_GPU' PYTHONPATH='$PACKAGE_ROOT:$REPOSITORY/src' '$PYTHON' scripts/train_samba.py --config configs/samba_t_paper.yaml --manifest '$DATA_ROOT/samba-t-tuab100-manifest.json' --output-root checkpoints/SAMBA-T/paper-tuab100-001 --tensorboard-dir checkpoints/SAMBA-T/paper-tuab100-001/tensorboard 2>&1 | tee logs/samba-t-paper.log"

tmux new-session -d -s samba-e \
  "cd '$REPOSITORY' && env CUDA_VISIBLE_DEVICES='$SAMBA_E_GPU' PYTHONPATH='$PACKAGE_ROOT:$REPOSITORY/src' '$PYTHON' scripts/train_samba.py --config configs/samba_e_short.yaml --manifest '$DATA_ROOT/samba-e-short-manifest.json' --output-root checkpoints/SAMBA-E/short-spatial-001 --tensorboard-dir checkpoints/SAMBA-E/short-spatial-001/tensorboard 2>&1 | tee logs/samba-e-short.log"

echo "Started SAMBA-T on GPU $SAMBA_T_GPU in tmux session samba-t"
echo "Started SAMBA-E on GPU $SAMBA_E_GPU in tmux session samba-e"
echo "Inspect with:"
echo "  tmux capture-pane -pt samba-t -S -30"
echo "  tmux capture-pane -pt samba-e -S -30"
