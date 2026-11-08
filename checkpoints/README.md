# Checkpoint release

Large weight files are not committed to Git. The repository publishes one current checkpoint:

```text
checkpoints/
└── SAMBA/
    ├── config.json
    ├── model.safetensors
    ├── training_metadata.json
    └── README.md
```

The new `SAMBA` checkpoint will be trained with the rebuilt joint pipeline and will not be presented as a strict reproduction of the lost SAMBA-E or SAMBA-T checkpoints. See `docs/TRAINING_AND_RELEASE.md` for the data and release requirements.
