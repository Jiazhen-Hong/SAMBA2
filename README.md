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

## Axon experiment command reference

This section is the copy-and-run record for Axon. Run every command from `~/Github/SAMBA2`. The standalone benchmark does not import the SAMBA architecture.

### 1. Update the repository

```bash
cd ~/Github/SAMBA2
git pull
```

### 2. Keep two isolated environments

Mamba-3 and FlashAttention-4 require incompatible TVM-FFI versions. Never install FA4 into the `mamba` environment.

The tested Mamba environment uses:

```text
mamba-ssm          2.3.2.post1
tilelang           0.1.8
apache-tvm-ffi     0.1.9
quack-kernels      0.6.5
nvidia-cutlass-dsl 4.8.0
```

Verify it with:

```bash
conda activate mamba
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import tilelang, mamba_ssm
from mamba_ssm import Mamba, Mamba2, Mamba3
print("TileLang:", tilelang.__version__)
print("mamba_ssm:", mamba_ssm.__version__)
print("GPU imports: OK")
PY
```

FA4 is installed only in the separate `fa4` environment:

```bash
conda activate fa4
python -m pip install "flash-attn-4==4.0.0b33"
python benchmarks/sequence_efficiency/benchmark.py --gpu 1 --list-backends
```

### 3. Width-matched experiment

This protocol fixes `d_model=256`, four layers, batch size, sequence length, dtype, and GPU. Parameter count and FLOPs are measured outcomes.

Run Mamba and the non-FA4 Transformer models:

```bash
conda activate mamba
cd ~/Github/SAMBA2
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol width \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer \
  --lengths 256 512 1024 2048 4096 8192 16384 32768 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 --ffn-mult 4 \
  --d-state 64 --headdim 64 --d-conv 4 --expand 2 \
  --dtype bfloat16 --warmup 10 --repeats 50 --train-repeats 10 \
  --memory-limit-gib 90 \
  --output Results/sequence_efficiency_width_mamba
```

Run FA4 with exactly the same experimental arguments:

```bash
conda activate fa4
cd ~/Github/SAMBA2
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol width --models flash4 \
  --lengths 256 512 1024 2048 4096 8192 16384 32768 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 --ffn-mult 4 \
  --d-state 64 --headdim 64 --d-conv 4 --expand 2 \
  --dtype bfloat16 --warmup 10 --repeats 50 --train-repeats 10 \
  --memory-limit-gib 90 \
  --output Results/sequence_efficiency_width_fa4
```

Merge and redraw without rerunning the models:

```bash
python benchmarks/sequence_efficiency/merge_results.py \
  Results/sequence_efficiency_width_mamba \
  Results/sequence_efficiency_width_fa4 \
  --output Results/sequence_efficiency_width_combined
```

### 4. Parameter-matched experiment

This protocol uses the four-layer, width-256 mathematical Transformer as the parameter target. It automatically searches a legal width for every model, then records the selected width, actual parameters, and matching error. Layers, sequence lengths, batch size, dtype, and GPU remain equal. FLOPs are reported rather than matched.

Run Mamba and the non-FA4 Transformer models:

```bash
conda activate mamba
cd ~/Github/SAMBA2
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol params \
  --models mamba1 mamba2 mamba3 transformer_math pytorch_sdpa linear_transformer \
  --lengths 256 512 1024 2048 4096 8192 16384 32768 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 --ffn-mult 4 \
  --d-state 64 --headdim 64 --d-conv 4 --expand 2 \
  --dtype bfloat16 --warmup 10 --repeats 50 --train-repeats 10 \
  --memory-limit-gib 90 --parameter-tolerance 0.03 \
  --output Results/sequence_efficiency_params_mamba
```

Run parameter-matched FA4:

```bash
conda activate fa4
cd ~/Github/SAMBA2
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol params --models flash4 \
  --lengths 256 512 1024 2048 4096 8192 16384 32768 \
  --batch-size 1 --d-model 256 --layers 4 --heads 8 --ffn-mult 4 \
  --d-state 64 --headdim 64 --d-conv 4 --expand 2 \
  --dtype bfloat16 --warmup 10 --repeats 50 --train-repeats 10 \
  --memory-limit-gib 90 --parameter-tolerance 0.03 \
  --output Results/sequence_efficiency_params_fa4
```

Merge the parameter-matched results:

```bash
python benchmarks/sequence_efficiency/merge_results.py \
  Results/sequence_efficiency_params_mamba \
  Results/sequence_efficiency_params_fa4 \
  --output Results/sequence_efficiency_params_combined
```

### 5. Result files

Each raw run contains `results.csv`, `metadata.json`, and its own figure. Each combined directory contains:

```text
results_combined.csv
metadata_combined.json
efficiency_summary_combined_linear.png
efficiency_summary_combined_log.png
```

The log-scale figure is usually easier to read. Figures use EEG duration in seconds at 128 Hz. The plotting-only horizontal offsets expose overlapping curves and do not modify stored sequence lengths. OOM events are marked directly on each panel.

Redraw either existing combined CSV into a new folder without using the GPU:

```bash
python benchmarks/sequence_efficiency/plot_results.py \
  Results/sequence_efficiency_params_combined/results_combined.csv \
  --sampling-rate 128 \
  --output Results/sequence_efficiency_params_128hz
```

For the width-matched result, replace `params` with `width` in both paths. This command generates `efficiency_128hz_linear.png` and `efficiency_128hz_log.png`.

See [`benchmarks/sequence_efficiency/README.md`](benchmarks/sequence_efficiency/README.md) for metric definitions and complete Axon commands.
