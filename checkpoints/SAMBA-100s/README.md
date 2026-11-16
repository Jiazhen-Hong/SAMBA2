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
  - neuralbench
---

# SAMBA-100s

SAMBA-100s is an EEG encoder pretrained without labels on 16-channel,
100-second TUAB windows sampled at 128 Hz. It follows the published SAMBA
architecture, Temporal Semantic Random masking, time-frequency objective, and
optimization settings.

The `100s` suffix describes the pretraining window. It is not a restriction on
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

model = SambaModel.from_pretrained("SAMBA-100s").eval()
signals = torch.randn(2, 16, 1280)
positions = load_your_electrode_coordinates()  # [16, 3], same channel order

with torch.inference_mode():
    output = model(signals=signals, input_positions=positions)

features = output.last_hidden_state
```

`signals` has shape `[batch, channels, samples]`. `input_positions` may have
shape `[channels, 3]` or `[batch, channels, 3]`. Coordinates must correspond
to the input-channel order.

## NeuralBench evaluation

The encoder is frozen and evaluated with NeuralBench 0.4.0 using its standard
`linear_probe_mean` wrapper. Values are the mean and sample standard deviation
over seeds 33, 34, and 35. The NeuralBench headline classification metric is
`test/bal_acc`.

| Task | Dataset | Balanced accuracy | AUROC |
|---|---|---:|---:|
| Pathology | Lopez2017Tuab | 0.7144 ± 0.0003 | 0.7914 ± 0.0003 |
| Depression diagnosis | Mumtaz2018Machine | 0.7981 ± 0.0056 | 0.8127 ± 0.0262 |
| Schizophrenia diagnosis | Albrecht2019Increased | 0.6251 ± 0.0018 | 0.6553 ± 0.0011 |

Pathology is an in-domain check with unlabeled TUAB pretraining overlap.
Depression and schizophrenia are cross-dataset transfer checks. These results
must not be compared directly with the paper's fine-tuning table because the
downstream protocol differs.

## Intended use and limitations

This checkpoint is intended for EEG representation extraction, linear
probing, and research fine-tuning. It is not a medical device and must not be
used for clinical diagnosis without independent validation.

The historical SAMBA-T checkpoint was lost. The recovered raw TUAB training
split was processed again and divided into a new recording-disjoint
self-supervised validation split. SAMBA-100s is therefore paper-aligned but
not a byte-identical reproduction of the original checkpoint.

## Citation

```bibtex
@article{hong2026samba,
  title={SAMBA: Toward an EEG Foundation Model via Differential Mamba and Coordinate-Driven Spatial Embedding},
  author={Hong, Jiazhen and Mackellar, Geoffrey and Ghane, Soheila},
  journal={arXiv preprint arXiv:2511.18571},
  year={2026}
}
```
