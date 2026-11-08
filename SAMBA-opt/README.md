# SAMBA-opt

SAMBA-opt is the experimental compatibility track. It is intentionally separated from the paper reproduction checkpoints.

Planned changes may include:

- mixed-duration pretraining with homogeneous-length microbatches;
- a shared reconstruction space across montages;
- duration-normalized spectral objectives;
- duration-aware sampling and gradient accumulation;
- direct NeuralBench development and evaluation.

Weights trained with these changes must be published as `SAMBA-opt`, not as SAMBA-E or SAMBA-T. Until its design is frozen and validated, this directory is experimental and must not be used to reproduce paper tables.

NeuralBench compatibility that does not change model mathematics, such as the runtime-coordinate adapter, remains available to every checkpoint through `samba_eeg.integrations.SambaNeuralBenchBackbone`.
