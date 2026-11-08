# SAMBA-T

Paper-aligned reproduction using the existing TUAB BIOT train, validation, and test recording split. Each example is a non-overlapping 100-second window with 16 channels at 128 Hz.

Use `scripts/prepare_tuab_100s.py --manifest-only` to avoid duplicating approximately 31 GiB of EEG. Training reads each indexed window from its source recording.

Expected release directory: `checkpoints/SAMBA-T/`.
