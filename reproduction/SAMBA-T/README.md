# SAMBA-T

Paper-aligned reproduction using the existing TUAB BIOT train, validation, and test recording split. Each example is a non-overlapping 100-second window with 16 channels at 128 Hz.

Use `scripts/prepare_tuab_100s.py --manifest-only` to avoid duplicating approximately 31 GiB of EEG. Training reads each indexed window from its source recording.

Expected release directory: `checkpoints/SAMBA-T/`.

## Axon pretraining

Generate a recording-safe manifest without duplicating EEG:

```bash
python scripts/prepare_tuab_100s.py \
  --input-root /data/data_downstream_task/TUAB/processed_Raw \
  --output-root data/manifests/TUAB_16ch_100s \
  --sample-rate 128 --channels 16 \
  --duration-seconds 100 --stride-seconds 100 \
  --manifest-only
```

After the manifest resolves exactly 39,810 windows, start paper-aligned two-GPU training:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_samba_t.py \
  --config reproduction/SAMBA-T/config.yaml \
  --manifest-root data/manifests/TUAB_16ch_100s \
  --output-root checkpoints/SAMBA-T/smoke-test \
  --smoke-test
```

If the smoke test succeeds, start the complete run:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_samba_t.py \
  --config reproduction/SAMBA-T/config.yaml \
  --manifest-root data/manifests/TUAB_16ch_100s \
  --output-root checkpoints/SAMBA-T/run-001
```
