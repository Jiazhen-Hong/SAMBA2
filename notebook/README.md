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

The Linear Transformer uses the same QKV projection, output projection, normalization, residual connections, and feed-forward network. Its attention replaces softmax with the positive feature map `ELU(x) + 1`. Causal key and key-value states are accumulated in chunks, so memory grows linearly with sequence length instead of constructing a full attention matrix. It is implemented only with PyTorch and does not require an additional linear-attention package. During training, non-reentrant activation checkpointing recomputes each chunk in backward instead of retaining its large rank-5 key-value intermediates. The plots still label it as a PyTorch reference implementation because it does not use a custom fused CUDA/Triton kernel.

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

Run a matched-state Mamba comparison so that all three versions use `d_state=64`:

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models mamba1 mamba2 mamba3 \
  --mamba1-d-state 64 \
  --mamba2-d-state 64 \
  --mamba3-d-state 64
```

The defaults remain Mamba-1/2/3 state sizes 16/64/128. Use the matched-state command when the goal is implementation efficiency rather than each version's default configuration.

Disable the short synthetic pretraining stage:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --pretrain-steps 0
```

Skip the separate training-memory curve:

```bash
python notebook/benchmark_mamba_eeg.py --gpu 1 --skip-training-memory
```

Disable the default Linear Transformer chunk checkpointing to reproduce the higher-memory reference graph:

```bash
python notebook/benchmark_mamba_eeg.py \
  --gpu 1 \
  --models linear_transformer \
  --disable-linear-attention-checkpoint
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

Forward computational cost is reported as estimated GFLOPs. One multiply-add counts as two FLOPs. Dense projections and Conv1d operations are counted from the instantiated model weights. Exact causal Transformer attention and the current Linear Transformer core use analytical formulas. The Mamba scan core uses the same comparative proxy for all three versions:

```text
9 * batch_size * sequence_length * d_inner * d_state
```

per layer. Normalization, bias, activation, masking, and loss operations are omitted. The estimate is useful for architecture comparison but is not an exact hardware-instruction count, especially for fused Mamba-2/3 Triton kernels. `effective_tflops` in the CSV divides estimated forward GFLOPs by measured latency; it should be interpreted as an estimated model-level utilization metric.

## Outputs

The default output directory is `Results/eeg_sequence_benchmark/`:

- `eeg_models_inference_memory.png`: inference peak memory
- `eeg_models_training_memory.png`: masked-pretraining peak memory
- `eeg_models_inference_speed.png`: inference throughput
- `eeg_models_forward_flops.png`: estimated forward computational cost
- `eeg_models_combined.png`: combined inference memory, speed, and estimated FLOPs
- `benchmark_results.csv`: raw values, status, OOM, and errors
- `run_metadata.json`: arguments, package versions, GPU information, and Mamba-2 path
- `checkpoints/`: checkpoints produced by synthetic pretraining

`Results/` is ignored by Git.
