# Paper-aligned SAMBA pretraining tracks

The historical SAMBA checkpoints are unavailable, so these tracks reproduce
the published architecture and optimization settings without claiming that
the recovered data split is byte-identical to the original run.

## What “64” means

The architecture uses 64 feature channels in its first encoder stage. It does
not force every EEG recording to 64 electrodes. The paper uses 16-channel
TUAB, 14-channel Emotiv, and several 64-channel downstream datasets. The MNE
`standard_1020` target file in this repository contains 94 coordinate entries.

## SAMBA-T

- TUAB only
- 16 input and reconstruction channels
- 128 Hz, 12,800 samples, 100 seconds
- stationary reconstruction target: the original TUAB window
- paper TSR mask, L1 plus unnormalized full-window rFFT loss
- AdamW and the published OneCycle schedule
- FP32
- effective batch 64 through 4-window microbatches and 16-step accumulation

The new SSL validation split is recording-disjoint and drawn only from the
recovered TUAB training split. This avoids downstream test leakage but differs
from the unavailable historical pipeline.

## SAMBA-E

- every source is exactly 128 Hz and 256 samples (2 seconds)
- five surviving Emotiv datasets plus HBN, Muse, and DICE
- 4, 14, 19, and 129 channel devices
- equal source weights
- coordinate-driven output decoding reconstructs each source's original EEG
  channels, keeping the target stationary while sharing one encoder
- paper TSR mask, L1 plus unnormalized full-window rFFT loss
- AdamW and the published OneCycle schedule
- FP32
- nominal effective batch 4,096 through 256-window microbatches and 16-step
  accumulation

DREAMER is excluded because its surviving prepared data could not be validated.
The recovered 256-sample TUAB arrays are excluded because their physical
duration is not verifiable.

## Axon commands

```bash
cd ~/Github/SAMBA2
git pull --ff-only

bash scripts/prepare_paper_track_manifests.sh
bash scripts/launch_samba_paper_tracks.sh
```

The launcher refuses to start when another SAMBA training process owned by the
same user is running. By default SAMBA-T uses GPU 0 and SAMBA-E uses GPU 1.
Override this explicitly when needed:

```bash
SAMBA_T_GPU=0 SAMBA_E_GPU=1 bash scripts/launch_samba_paper_tracks.sh
```

Monitor both jobs:

```bash
tmux capture-pane -pt samba-t -S -30
tmux capture-pane -pt samba-e -S -30
nvidia-smi
```

Start one TensorBoard instance for both tracks:

```bash
~/venvs/eeg26-track2-clean/bin/tensorboard \
  --logdir ~/Github/SAMBA2/checkpoints \
  --host 127.0.0.1 \
  --port 6006
```
