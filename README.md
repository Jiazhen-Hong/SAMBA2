# SAMBA: Toward an EEG Foundation Model via Differential Mamba and Coordinate-Driven Spatial Embedding
**Update: Jan. 2026**

**SAMBA** is a scalable self-supervised framework for long-sequence EEG foundation modeling. It integrates 3D coordinate-driven spatial embedding and differential Mamba modules to enable robust, efficient, and generalizable EEG representation learning across diverse recording configurations and cognitive tasks.

<p align="center">
  <img src="./Figures/SAMBA.png" alt="SAMBA Overview" width="900"/>
</p>

---

## Abstract

Modeling long EEG sequences is critical for developing generalizable neural representations, particularly due to the high temporal resolution and extended durations often required to capture brain dynamics. While transformer-based models have shown success on short EEG segments, their quadratic complexity prevents effective scaling to long contexts. Additionally, the diversity in EEG montages and subject variability presents significant generalization challenges.

We introduce **SAMBA**, a self-supervised learning framework featuring a U-shaped encoder-decoder architecture built on the linear-time **Mamba** module. SAMBA incorporates:

1. **Temporal Semantic Random Masking** — to reconstruct semantically masked segments in long sequences;
2. **Multi-Head Differential Mamba** — to reduce redundancy and enhance salient temporal features;
3. **Coordinate-Driven Spatial Embedding (CDSE)** — to learn robust 3D spatial representations across heterogeneous EEG devices.

Evaluations across **13 EEG datasets** covering a range of tasks, montages, and sequence lengths demonstrate that SAMBA consistently outperforms state-of-the-art baselines, while maintaining low memory usage and fast inference. In addition, the learned spatial weights exhibit strong alignment with task-relevant neurophysiological regions, suggesting that SAMBA is both **learnable** and **interpretable**.

<!-- > 📌 The code is available at: https://github.com/anon1280/SAMBA -->

---

## Learnable Spatial Embedding Visualization

The figure below illustrates the **Coordinate-Driven Spatial Embedding (CDSE)** module and its alignment with neurophysiological topology:

<p align="center">
  <img src="./Figures/CDSE.png" alt="CDSE Overview" width="800"/>
</p>


<table align="center">
  <tr>
    <td align="center">
        <img src="./Figures/Topo_gif/EyesState_GIF.gif" alt="Eyes State" width="250"/><br/>
      <b>Eyes Close/Open</b><br/>
      Emphasizes the <u>frontal lobe</u>, consistent with alpha modulation during eye closure[1].
    </td>
    <td align="center">
      <img src="./Figures/Topo_gif/DriverDistraction_GIF.gif" alt="Driver Distraction" width="250"/><br/>
      <b>Driver Distraction</b><br/>
      Highlights the <u>left temporal lobe</u>, associated with auditory processing (conversation during driving) and cognitive control under distraction[2].
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./Figures/Topo_gif/STEW_GIF.gif" alt="Workload Estimation" width="250"/><br/>
      <b>Workload Estimation</b><br/>
      Focuses on the <u>left frontal region</u>, linked to stress-induced workload processing[3].
    </td>
    <td align="center">
      <img src="./Figures/Topo_gif/TUAB_GIF.gif" alt="Seizure" width="250"/><br/>
      <b>Abnormal Detection (Seizure)</b><br/>
      Shows <u>occipital dominance</u>, consistent with findings that occipital electrodes yield better performance for distinguishing abnormal EEG[4].
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./Figures/Topo_gif/MI_GIF.gif" alt="Motor Imagery" width="250"/><br/>
      <b>Motor Imagery</b><br/>
      Localizes around the <u>motor cortex</u>, reflecting sensorimotor activity during imagery tasks[5].
    </td>
  </tr>
</table>

#### References
<!-- <div style="font-size: smaller"> -->
<sub>
[1] S. Hoffmann et al. The correction of eye blink artefacts in the EEG: a comparison of two prominent methods. PLOS ONE, 3(8):e3004, 2008.  <br/>
[2] G. Li et al. Drivers’ EEG responses to different distraction tasks. Auto. Innov., 6(1):20–31, 2023.  <br/>
[3] G. Berretz et al. Acute stress increases left hemispheric activity measured via changes in frontal alpha asymmetries. iScience, 25(2), 2022.  <br/>
[4] S. Lopez et al. Automated identification of abnormal adult EEGs. In Proc. IEEE SPMB, pp. 1–5, 2015.  <br/>
[5] J. Hong et al. A deep learning framework based on dynamic channel selection for early classification of left and right hand motor imagery tasks. In Proc. IEEE EMBC, pp. 3550–3553, 2022.  <br/>
</sub>
<!-- </div> -->

## Model and Checkpoints

- The full architecture is implemented under `Models/`.
- Two pretrained checkpoints are provided under `Checkpoints/`:
  - `SAMBA-E`: trained using PyTorch Lightning
  - `SAMBA-T`: trained using native PyTorch
- The accompanying paper is currently in preparation. More features and documentation will be released soon.

---

## Repository Structure
```plaintext
SAMBA/
├── Env-requirement/     # Environment configs with dated backups
├── Checkpoints/         # Pretrained SAMBA models (SAMBA-E, SAMBA-T)
├── Models/              # Model architecture implementations
├── Experiments/         # PyTorch Lightning training modules
├── utility/             # Supporting functions: data loading, masking, loss, evaluation
├── Figures/             # Diagrams and visualizations
├── Montage/             # EEG montage metadata for multiple devices
└── README.md            # Project overview
```

## Axon command reference

Run these commands from the repository root after pulling the latest code.

### Check the environment

```bash
cd ~/Github/SAMBA2
git pull
CUDA_VISIBLE_DEVICES=1 python -c "import torch, mamba_ssm; print(torch.__version__, torch.version.cuda, mamba_ssm.__version__); print(torch.cuda.get_device_name(0))"
```

### Standalone fair efficiency benchmark

This benchmark is independent of SAMBA. It compares simple sequence blocks using matched input shape, width, layer count, precision, and batch size. The plain Transformer explicitly uses mathematical attention and cannot silently select a Flash kernel. Full details are in [`benchmarks/sequence_efficiency/README.md`](benchmarks/sequence_efficiency/README.md).

Install FlashAttention 4 for the Blackwell GPU. It is currently a prerelease, so the version is pinned for reproducibility:

```bash
pip install "flash-attn-4==4.0.0b33"
```

Inspect the available attention backends:

```bash
python benchmarks/sequence_efficiency/benchmark.py --gpu 1 --list-backends
```

Run the width-matched experiment on GPU 1:

```bash
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 \
  --protocol width \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer flash4 \
  --lengths 256 512 1024 2048 4096 8192 16384 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 \
  --d-state 64 --headdim 64 --dtype bfloat16
```

Run the parameter-matched experiment:

```bash
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol params \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer flash4 \
  --lengths 256 512 1024 2048 4096 8192 16384 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 \
  --d-state 64 --headdim 64 --dtype bfloat16
```

Run a short smoke test before the full benchmark:

```bash
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol width \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer flash4 \
  --lengths 256 1024 --warmup 2 --repeats 3 --train-repeats 2
```

Run only inference:

```bash
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol both \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer flash4 \
  --skip-training
```

Results are written to `Results/sequence_efficiency/`. If FlashAttention 4 is not importable, it remains in the CSV with an explicit `UNSUPPORTED` status.

### Existing EEG benchmark

The older EEG-oriented benchmark remains available for reproducing the earlier figures:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --models mamba1 mamba2 mamba3 transformer linear_transformer
```

Its options and output files are documented in [`notebook/README.md`](notebook/README.md).
