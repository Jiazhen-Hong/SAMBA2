# SAMBA training and release contract

This repository publishes one current checkpoint named `SAMBA`.

## Historical checkpoint status

SAMBA-E and SAMBA-T were trained during a 2025 internship. Their checkpoint files were lost when the associated personal company account was removed after the internship. The internal preprocessing directories were later reorganized. The surviving data do not fully establish the original filtering, normalization, window boundaries, and sampling metadata.

For this reason, the current checkpoint is not labeled as a strict reproduction of SAMBA-E or SAMBA-T. The original paper results remain unchanged, but new weights must be evaluated and reported separately.

## Current data contract

The rebuilt pipeline may jointly train on verified Emotiv and TUAB sources. Every source must record:

- source dataset and version;
- subject identifier and split assignment;
- ordered channel names and 3D coordinates;
- sampling rate, units, reference, filtering, and normalization;
- window length, stride, and recording boundaries;
- preprocessing code revision and manifest hash.

A directory name is not accepted as duration metadata. In particular, the recovered `TUAB_16ch_10s` arrays have 256 samples per window, but the surviving files do not yet verify that these samples represent 10 seconds.

Sources with different montages or lengths use homogeneous source-duration
microbatches. They are mapped to a shared coordinate-driven reconstruction space
before the reconstruction objective is evaluated. Different buckets accumulate
gradients in the same optimizer step. The physical batch size is inversely
proportional to temporal length, which keeps the approximate number of processed
time points constant without padding short signals to the longest duration.

The multi-context recipe includes Alpha, Attention, Crowdsource,
DriverDistraction, STEW, TUAB, HBN resting state, Stieger motor imagery, Wang
SSVEP, and ERP CORE N170. The recipe records explicit source weights, task
families, device families, temporal lengths, and data caps. NeuralBench
validation and test splits are excluded from self-supervised pretraining.

All Emotiv sources are normalized per window and channel by the absolute 95th percentile. Constant or nonfinite channels are zeroed and marked invalid in the channel mask. The recovered TUAB arrays already use the same scale rule.

## Required release artifacts

The released `SAMBA` directory must contain:

- `config.json`;
- `model.safetensors`;
- `training_metadata.json` with the Git revision, manifest hash, seed, optimizer, scheduler, epoch, and validation metric;
- `README.md` describing preprocessing, input units, intended use, limitations, and benchmark results.

The final checkpoint is selected using validation data only. Test data is evaluated after model selection. A release is not called pretrained until all artifacts are published together.

## Compatibility checks

Run:

```bash
PYTHONPATH=src python -m pytest -q
```

Then verify CPU loading, missing-channel masks, multiple channel counts, multiple temporal lengths, Hugging Face serialization, and the NeuralBench adapter.
