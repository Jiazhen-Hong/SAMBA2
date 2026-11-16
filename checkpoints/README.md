# Checkpoint releases

Large weight files are not committed to Git. The two public releases use the
standard Hugging Face model-repository layout:

```text
checkpoints/
├── SAMBA-2s/
│   ├── config.json
│   ├── model.safetensors
│   ├── training_metadata.json
│   ├── SHA256SUMS
│   └── README.md
└── SAMBA-100s/
    ├── config.json
    ├── model.safetensors
    ├── training_metadata.json
    ├── SHA256SUMS
    └── README.md
```

`SAMBA-2s` is pretrained on spatially diverse two-second recordings.
`SAMBA-100s` is pretrained on 100-second TUAB windows. The suffix describes
the pretraining duration, not a downstream input restriction. Both encoders
accept variable channel layouts and temporal lengths.

Create both release folders from completed Axon runs with:

```bash
python scripts/package_huggingface_checkpoints.py
```

The command copies each selected `best/` checkpoint. It does not remove or
modify the original training runs. See `docs/TRAINING_AND_RELEASE.md` for the
data and release requirements.
