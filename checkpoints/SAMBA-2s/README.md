---
library_name: samba-eeg
pipeline_tag: feature-extraction
tags:
  - eeg
  - brain-computer-interface
  - self-supervised-learning
  - mamba
  - biosignal
  - pytorch
---

# SAMBA-2s

SAMBA-2s is an EEG encoder pretrained with two-second windows sampled at
128 Hz. The pretraining corpus combines five surviving Emotiv datasets with
HBN resting-state, Muse sleep-onset, and DICE eyes-closed EEG. Source batches
remain montage-homogeneous and cover 4, 14, 19, and 129 channel devices.

The `2s` suffix describes the pretraining window. It is not a restriction on
downstream input duration. The encoder accepts variable temporal lengths,
channel counts, channel masks, and electrode coordinates.

## Files

- `config.json`: architecture and input metadata;
- `model.safetensors`: selected validation-best encoder and decoder weights;
- `training_metadata.json`: manifest identity, training provenance, and
  validation selection information;
- `SHA256SUMS`: release-file checksums.

## Usage

Install SAMBA and its Mamba dependency from the source repository, then load
the local or downloaded model directory:

```python
import torch
from samba_eeg import SambaModel

model = SambaModel.from_pretrained("SAMBA-2s").eval()
signals = torch.randn(2, 14, 256)
positions = load_your_electrode_coordinates()  # [14, 3], same channel order

with torch.inference_mode():
    output = model(signals=signals, input_positions=positions)

features = output.last_hidden_state
```

`signals` has shape `[batch, channels, samples]`. `input_positions` may have
shape `[channels, 3]` or `[batch, channels, 3]`. Coordinates must correspond
to the input-channel order.

## Evaluation status

NeuralBench linear-probe evaluation is in progress. Report SAMBA-2s results
separately from SAMBA-100s because their pretraining corpora and context
durations differ.

## Intended use and limitations

This checkpoint is intended for EEG representation extraction, linear
probing, and research fine-tuning. It is not a medical device and must not be
used for clinical diagnosis without independent validation.

The historical SAMBA-E checkpoint was lost. SAMBA-2s uses a rebuilt and more
spatially diverse data pipeline, so it is not a strict reproduction of the
paper's original SAMBA-E weights. DREAMER was excluded because its surviving
prepared data could not be validated.

## Citation

```bibtex
@article{hong2026samba,
  title={SAMBA: Toward an EEG Foundation Model via Differential Mamba and Coordinate-Driven Spatial Embedding},
  author={Hong, Jiazhen and Mackellar, Geoffrey and Ghane, Soheila},
  journal={arXiv preprint arXiv:2511.18571},
  year={2026}
}
```
