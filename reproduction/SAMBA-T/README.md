# SAMBA-T

Paper-aligned reproduction using the TUAB BIOT train, validation, and test recording split. Each example is a non-overlapping 100-second window with 16 bipolar channels at 128 Hz.

Expected release directory: `checkpoints/SAMBA-T/`.

The retained `processed_Raw` files contain continuous 14-channel signals and cannot reproduce the paper's 16 bipolar channels. `TUAB_16ch_10s` is also mislabeled and contains 256-sample windows without recording boundaries. Therefore, paper reproduction starts from the TUAB EDF files.

Generate a lightweight manifest using the original 16 bipolar derivations, the retained train/validation/test recording IDs, 128 Hz, and non-overlapping 100-second windows:

```bash
python scripts/prepare_tuab_edf_manifest.py \
  --edf-root /data/neuralbench_data/tuh_eeg/tuab/v3.0.1/edf \
  --split-reference-root /data/data_downstream_task/TUAB/processed_Raw \
  --output-root data/manifests/TUAB_EDF_16ch_100s
```

Training must not start unless the manifest has no unresolved recordings and resolves exactly 39,810 windows.

Run a two-GPU integration smoke test before the full experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_samba_t.py \
  --config reproduction/SAMBA-T/config.yaml \
  --manifest-root data/manifests/TUAB_EDF_16ch_100s \
  --output-root checkpoints/SAMBA-T/smoke-test \
  --smoke-test
```

Then start the complete 200-epoch paper run:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_samba_t.py \
  --config reproduction/SAMBA-T/config.yaml \
  --manifest-root data/manifests/TUAB_EDF_16ch_100s \
  --output-root checkpoints/SAMBA-T/run-001
```
