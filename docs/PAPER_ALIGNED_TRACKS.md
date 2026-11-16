# Paper-aligned SAMBA pretraining tracks

The historical SAMBA checkpoints are unavailable, so these tracks reproduce
the published architecture and optimization settings without claiming that
the recovered data split is byte-identical to the original run.

## What “64” means

The architecture uses 64 feature channels in its first encoder stage. It does
not force every EEG recording to 64 electrodes. The paper uses 16-channel
TUAB, 14-channel Emotiv, and several 64-channel downstream datasets. The MNE
`standard_1020` target file in this repository contains 94 coordinate entries.

## SAMBA-100s (paper SAMBA-T track)

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

## SAMBA-2s (extended SAMBA-E track)

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

## Extending a completed run

A completed OneCycle schedule cannot be continued by only increasing
`--epochs`, because its learning rate has already reached the final value.
Resume the optimizer and model while starting a smaller OneCycle schedule for
the additional epochs. Reuse the same TensorBoard directory to append to the
existing curves.

For example, extend the 200-epoch short run to epoch 300 on GPU 1:

```bash
cd ~/Github/SAMBA2

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/data/SAMBA-tmp/python312-packages:$PWD/src \
~/venvs/eeg26-track2-clean/bin/python scripts/train_samba.py \
  --config configs/samba_e_short.yaml \
  --manifest /data/SAMBA-tmp/samba-e-short-manifest.json \
  --output-root checkpoints/SAMBA-E/short-spatial-001 \
  --resume checkpoints/SAMBA-E/short-spatial-001/last.pt \
  --epochs 300 \
  --restart-scheduler \
  --initial-lr 5e-5 \
  --max-lr 1e-4 \
  --final-lr 1e-6 \
  --tensorboard-dir checkpoints/SAMBA-E/short-spatial-001/tensorboard
```

If that extension is interrupted, resume it with the same command but remove
`--restart-scheduler` and the three learning-rate overrides. The scheduler
state saved by the extension will then continue instead of starting over.

## Initial SAMBA-100s linear probes

The first binary checks use three NeuralBench tasks: TUAB pathology detection,
depression diagnosis, and schizophrenia diagnosis. Pathology is an in-domain
check with unlabeled pretraining overlap; the other two are cross-dataset
transfer checks. Run them sequentially on one GPU with:

```bash
tmux new-session -d -s samba-100s-probes \
  "bash $HOME/Github/SAMBA2/scripts/evaluate_samba_100s_binary.sh"
```

The script reuses local datasets and NeuralBench caches, runs the standard
`linear_probe_mean` wrapper, and writes one log per task under `logs/`.
