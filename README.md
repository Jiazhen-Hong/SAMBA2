# SAMBA: Toward an EEG Foundation Model via Differential Mamba and Coordinate-Driven Spatial Embedding
**Update: Nov. 2026**

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

### Checkpoint history

The original SAMBA-E and SAMBA-T checkpoints were produced during a 2025 internship. After the internship ended, the personal company account used for this work was removed, and the checkpoint files were not retained. The internal data pipeline was later reorganized. Some recovered arrays still match the reported shapes and sample counts, but the original filtering, normalization, and window construction cannot be fully verified. Therefore, this repository no longer claims that a newly trained checkpoint is a strict reproduction of the original SAMBA-E or SAMBA-T weights.

The paper results remain the record for the original experiments. New weights must not be used as direct replacements for reproducing the paper tables.

### Current release direction

The repository now targets one checkpoint named **SAMBA**. The rebuilt pipeline is designed for joint training with verified Emotiv and TUAB sources, source-homogeneous microbatches, a shared coordinate-driven reconstruction space, and explicit preprocessing metadata. It also improves support for variable channel layouts and temporal lengths in downstream use.

The recovered TUAB array currently has 256 samples per window. Its directory name contains `10s`, but the surviving files do not provide enough metadata to verify a 10-second duration. The configuration records this duration as unknown rather than inferring it from the directory name.

The current evidence and blocking data issues are recorded in [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md).

The new checkpoint will be released only after the rebuilt data manifest, preprocessing record, validation result, code revision, and model weights are available together. Until then, pretrained weights are **not yet included**.

The reusable implementation is under `src/samba_eeg/`. The research prototype under `Models/` remains temporarily available for comparison with the 2025 code.

### Public model API

```python
import torch
from samba_eeg import SambaModel

model = SambaModel.from_pretrained("path/to/SAMBA")
output = model(
    signals=torch.randn(2, 14, 256),
    input_positions=torch.randn(14, 3),
)

# Encoder representation used by the paper's linear probing protocol: [B, T/8, 128]
features = output.last_hidden_state
```

`input_positions` may be `[C, 3]` for a shared montage or `[B, C, 3]` for a batch with per-recording coordinates. `channel_mask` supports padded or missing electrodes. This removes the prototype's fixed channel-order assumption.

The checkpoint directory follows the common Hugging Face layout:

```text
SAMBA/
├── config.json
├── model.safetensors        # preferred
├── README.md                # model card
└── training_metadata.json   # data manifest hash, code revision, and metrics
```

The optional `SambaNeuralBenchBackbone` adapter returns a plain `[B, T, D]` tensor and accepts runtime channel coordinates. The encoder accepts different channel counts and temporal lengths without changing checkpoint parameters.

```python
from neuralbench import evaluate_model
from samba_eeg.integrations import SambaNeuralBenchBackbone

scores = evaluate_model(
    SambaNeuralBenchBackbone(model),
    "eeg",
    "all",
    name="samba",
    downstream_wrapper="linear_probe_mean",
    debug=True,
)
```

NeuralBench runs the model itself; uploading a checkpoint alone does not produce a score. Its bring-your-own-model route expects one model instance to accept arbitrary channel counts and window lengths, and passes `channel_positions` into `forward`. SAMBA's adapter follows that interface. See the [official NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/quickstart/03_evaluate_your_own_model.html).

For manually installed datasets such as TUAB, skip NeuralBench's automatic download
step while retaining cache preparation:

```bash
python scripts/evaluate_neuralbench.py path/to/SAMBA \
  --task pathology \
  --debug \
  --no-download
```

### Multi-context pretraining pipeline

The current pipeline trains short and long windows in one run without padding
every sample to the longest duration. Each source-duration pair is a homogeneous
bucket. Buckets may have different montages and may provide one coordinate set
per source or per recording. Gradient accumulation mixes buckets within an
optimizer step. A fixed temporal budget automatically changes the physical
batch size: with the default settings, 2-second windows use 256 examples,
4-second windows use 128, 10-second windows use 51, and 100-second windows use
5. This keeps long windows from exhausting memory and avoids wasting compute on
padding.

The initial diverse corpus recipe is in `configs/pretrain_sources.yaml`:

- five 14-channel Emotiv tasks at 2 seconds;
- TUAB clinical EEG at 10 seconds;
- HBN resting-state EEG at 10 seconds;
- Stieger motor imagery and Wang SSVEP at 4 seconds;
- ERP CORE N170 at 1 second.

NeuralBench subject splits are used to locate the data, but only its training
split is eligible for export. A fixed subset of those training windows becomes
the self-supervised validation set. NeuralBench validation and test examples are
not read by pretraining. Source weights are explicit, so HBN or TUAB cannot
dominate simply because they contain more recordings.

Prepare the five recovered Emotiv sources first:

```bash
python scripts/prepare_samba_data.py \
  --input-root /data/data_downstream_task \
  --output-root /data/SAMBA-tmp/Emotiv
```

Prepare true 10-second TUAB windows directly from the raw EDF files. This path
uses the paper's 16 bipolar channels, limits processing to eight workers, and is
resumable at recording boundaries. It does not build a NeuralBench task cache.

```bash
python scripts/prepare_tuab_10s.py \
  --edf-root /data/neuralbench_data/Lopez2017Tuab/edf \
  --split-reference-root /data/data_downstream_task/TUAB/processed_Raw \
  --output-root /data/SAMBA-tmp/TUAB10 \
  --workers 8
```

Use the Python 3.12 NeuralBench environment to check each additional source and
batch schema before exporting it. For example, inspect HBN without writing the
pretraining arrays:

```bash
NEURALBENCH_CONFIG=/data/SAMBA-tmp/neuralbench-config.json \
python scripts/export_multicontext_data.py \
  --config configs/pretrain_sources.yaml \
  --output-root /data/SAMBA-tmp/pretrain-v2 \
  --manifest-output /data/SAMBA-tmp/pretrain-v2/manifest.json \
  --workers 8 \
  --check-only \
  --only 'HBN/resting/10s'
```

After the check succeeds, remove `--check-only`. The export is resumable at the
source/split level and writes the final manifest only after all sources finish.
It caps large sources by hours to keep the prepared corpus within a controlled
disk budget.

Run a small single-GPU integration test in the SAMBA Mamba environment:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_samba.py \
  --config configs/samba_multicontext.yaml \
  --manifest /data/SAMBA-tmp/pretrain-v2/manifest.json \
  --output-root checkpoints/SAMBA/smoke-test \
  --smoke-test
```

Then start the full run in `tmux`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_samba.py \
  --config configs/samba_multicontext.yaml \
  --manifest /data/SAMBA-tmp/pretrain-v2/manifest.json \
  --output-root checkpoints/SAMBA/multicontext-001
```

The loader uses no more than 24 workers in total, with pinned memory and
prefetching. Resampling, filtering, normalization, and coordinate extraction are
completed before training. Validation uses fixed windows and a fixed masking
seed, so checkpoint selection is comparable across epochs.

The previous `configs/samba.yaml` command remains available only to inspect the
256-sample pilot checkpoint. Its recovered directory name did not verify a
10-second TUAB duration and it should not be used for the new multi-context run.

On shared systems with limited `/dev/shm`, pass `--workers-per-source 0`. The dataset implements vectorized batch reads, so this mode avoids multiprocessing shared memory without returning to slow per-sample disk access.

---

## Repository Structure
```plaintext
SAMBA/
├── src/samba_eeg/       # Stable package API
├── configs/samba.yaml   # Current single-checkpoint configuration
├── checkpoints/         # SAMBA release layout documentation
├── scripts/             # Configuration and evaluation tools
├── tests/               # CPU compatibility tests
├── Models/              # Legacy research prototype
├── utility/             # Legacy experiment utilities
├── Figures/             # Diagrams and visualizations
├── Montage/             # EEG montage metadata
└── README.md
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
