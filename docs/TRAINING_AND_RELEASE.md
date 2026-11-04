# SAMBA-E and SAMBA-T training and release contract

This document prevents a checkpoint from becoming detached from its data and code history.

## Paper-aligned definitions

SAMBA-E uses 14-channel, 128 Hz, two-second Emotiv windows. SAMBA-T uses 16-channel, 128 Hz, 100-second TUAB windows. Both recipes use 50% Temporal Semantic Random masking, 200 epochs, AdamW with weight decay 0.01, and the paper's OneCycle schedule.

The paper reports 311,011 samples for the six-corpus SAMBA-E pretraining set and 39,810 100-second samples for SAMBA-T. Training must stop if the resolved manifest counts do not match the intended release. Do not silently resample, discard subjects, or substitute a random sample split.

## Required manifest fields

Every data source must record:

- stable sample or shard path;
- subject identifier and train, validation, or test assignment;
- ordered channel names and three-dimensional coordinates;
- sampling rate, window length, stride, filtering, referencing, and normalization;
- source dataset version and preprocessing code revision.

The split must be subject-wise unless a named benchmark protocol requires otherwise. The manifest content hash is stored in `training_metadata.json`.

## Required release artifacts

A public checkpoint directory must contain:

- `config.json`, including target positions and reconstruction channel count;
- `model.safetensors` or `pytorch_model.bin`;
- `training_metadata.json`, including the Git revision, manifest hash, seed, optimizer, scheduler, epoch, and validation metric;
- a model card describing intended use, preprocessing, input units, limitations, and benchmark results.

The final weights must be selected using validation data only. Test data is evaluated once after model selection. Resume checkpoints must also store optimizer, scheduler, scaler, epoch, global step, and random number generator states.

## Compatibility checks before upload

Run:

```bash
PYTHONPATH=src python -m pytest -q
```

Then verify loading on CPU, a runtime montage with missing-channel masks, both paper sequence lengths, and the NeuralBench adapter. Upload the complete directory to Hugging Face rather than uploading a bare `.pth` file.
