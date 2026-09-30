# EEG sequence-model benchmark

`benchmark_mamba_eeg.py` compares five sequence blocks inside the same simulated-EEG reconstruction network:

- Mamba-1
- Mamba-2
- Mamba-3
- Transformer
- Linear Transformer

Every network uses the same 14-channel input, input projection, model width, number of layers, output normalization, and EEG reconstruction head. Plot legends report the total parameter count of the complete network.

## Models

The Transformer is a pre-norm causal encoder block implemented with PyTorch `scaled_dot_product_attention`. Each layer contains QKV projection, exact causal softmax attention, output projection, and a two-layer GELU feed-forward network with hidden width `2 * d_model`.

The Linear Transformer uses the same QKV projection, output projection, normalization, residual connections, and feed-forward network. Its attention replaces softmax with the positive feature map `ELU(x) + 1`. Causal key and key-value states are accumulated in chunks, so memory grows linearly with sequence length instead of constructing a full attention matrix. It is implemented only with PyTorch and does not require an additional linear-attention package.

Mamba-2 automatically uses its fused memory-efficient path when all required `causal-conv1d` functions are callable. Otherwise it uses the supported non-fused path and is labeled `Mamba-2 (fallback)` in the figures.

## Run

Run all five models on physical GPU 1:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1
```

Run selected models:

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models mamba1 mamba2 mamba3
```

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models transformer linear_transformer
```

Use custom sequence lengths:

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models mamba1 mamba2 mamba3 transformer linear_transformer \
  --lengths 256,512,1024,2048,4096,8192,16384
```

Disable the short synthetic pretraining stage:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --pretrain-steps 0
```

Skip the separate training-memory curve:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --skip-training-memory
```

Force a Mamba-2 execution path:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --mamba2-path fallback
python notebook/benchmark_mamba_eeg.py --gpu 1 --mamba2-path fused
```

Exact attention has quadratic compute cost. By default, Transformer inference is limited to 8,192 samples and its training-memory test is limited to 4,096 samples. The chunked Linear Transformer defaults to 65,536 and 32,768 samples respectively to avoid extremely long runs. Longer points are marked as safety-limited rather than OOM. Set the corresponding limits to `0` to remove them:

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models transformer \
  --transformer-max-length 0 \
  --transformer-training-max-length 0
```

The Linear Transformer equivalents are `--linear-transformer-max-length` and `--linear-transformer-training-max-length`.

## Measurements

Inference memory is measured under `torch.inference_mode()` with `torch.cuda.max_memory_allocated()`. It includes the model, input, output, and temporary inference tensors allocated by PyTorch, but not memory owned by unrelated GPU processes.

Training memory measures one complete masked-reconstruction step: forward pass, FP32 loss calculation, backward pass, and AdamW update. It therefore includes saved activations, gradients, and optimizer states and is normally higher than inference memory.

Inference latency is measured with `torch.cuda.Event`. CUDA is synchronized before reading event times, and the reported latency is the median over repeated runs. Throughput is calculated as:

```text
batch_size * sequence_length / median_latency_ms
```

Synthetic EEG generation, CPU-to-GPU transfer, CSV writing, and plotting are outside the timed region.

## Outputs

The default output directory is `Results/eeg_sequence_benchmark/`:

- `eeg_models_inference_memory.png`: inference peak memory
- `eeg_models_training_memory.png`: masked-pretraining peak memory
- `eeg_models_inference_speed.png`: inference throughput
- `eeg_models_combined.png`: combined inference memory and speed
- `benchmark_results.csv`: raw values, status, OOM, and errors
- `run_metadata.json`: arguments, package versions, GPU information, and Mamba-2 path
- `checkpoints/`: checkpoints produced by synthetic pretraining

`Results/` is ignored by Git.
