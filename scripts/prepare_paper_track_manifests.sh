#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${REPOSITORY:-$HOME/Github/SAMBA2}"
PYTHON="${PYTHON:-$HOME/venvs/eeg26-track2-clean/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/SAMBA-tmp}"

cd "$REPOSITORY"

"$PYTHON" scripts/export_multicontext_data.py \
  --config configs/pretrain_samba_t.yaml \
  --output-root "$DATA_ROOT" \
  --manifest-output "$DATA_ROOT/samba-t-tuab100-manifest.json"

"$PYTHON" scripts/export_multicontext_data.py \
  --config configs/pretrain_samba_e_short.yaml \
  --output-root "$DATA_ROOT" \
  --manifest-output "$DATA_ROOT/samba-e-short-manifest.json"

echo "Prepared manifests:"
echo "  $DATA_ROOT/samba-t-tuab100-manifest.json"
echo "  $DATA_ROOT/samba-e-short-manifest.json"
