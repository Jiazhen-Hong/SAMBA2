# Checkpoint releases

Large weight files are not committed to Git. Each released checkpoint is uploaded as a complete Hugging Face-compatible directory:

```text
checkpoints/
├── SAMBA-E/
│   ├── config.json
│   ├── model.safetensors
│   ├── training_metadata.json
│   └── README.md
├── SAMBA-T/
│   └── ...
└── SAMBA-opt/
    └── ...
```

SAMBA-E and SAMBA-T must satisfy `reproduction/README.md`. SAMBA-opt must be labeled experimental and is never substituted for a paper checkpoint.
