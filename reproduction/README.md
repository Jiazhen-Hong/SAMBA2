# Paper reproduction

This directory is the source of truth for checkpoints that reproduce the SAMBA paper. Engineering changes may improve packaging and runtime compatibility, but they must not change the architecture, objective, data mixture, or duration used for pretraining.

| Checkpoint | Pretraining data | Input | Reconstruction target |
| --- | --- | --- | --- |
| SAMBA-E | Six Emotiv corpora | 14 channels, 128 Hz, 2 s | Original 14-channel montage |
| SAMBA-T | TUAB BIOT split | 16 channels, 128 Hz, 100 s | Original 16-channel montage |

The two checkpoints are trained independently. Mixed-duration training belongs under `SAMBA-opt/` and must not be reported as SAMBA-E or SAMBA-T.
