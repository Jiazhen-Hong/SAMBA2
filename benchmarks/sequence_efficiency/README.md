# Standalone sequence efficiency benchmark

This benchmark does not import SAMBA. Every model uses the same sequence length, batch size, layer count, dtype, and GPU.

## Methods

- `mamba1`, `mamba2`, `mamba3`: official `mamba_ssm` blocks.
- `transformer_math`: handwritten causal softmax attention that explicitly materializes the attention matrix.
- `pytorch_sdpa`: the same QKV, output projection, residual, and FFN structure, with `torch.nn.functional.scaled_dot_product_attention` as its attention operation. PyTorch selects its optimized CUDA backend at runtime.
- `linear_transformer`: handwritten causal linear attention with the ELU+1 feature map.
- `flash4`: FlashAttention 4 through `flash_attn.cute.flash_attn_func`.

The three exact-attention methods share the same surrounding Transformer block. Only the attention operation changes. Linear attention keeps the same surrounding block but changes the attention algorithm. Mamba uses its native residual block.

## Comparison protocols

`--protocol width` keeps `d_model`, layers, sequence length, batch size, dtype, and hardware equal. Parameters and FLOPs are measured outcomes rather than matching targets.

`--protocol params` uses the four-layer, width-256 mathematical Transformer as the default parameter target. It searches legal widths in increments of 32 for every method and records the selected width, actual parameter count, and matching error.

`--protocol both` runs both protocols. Equal width and equal parameter count answer different questions, so their curves are labeled separately.

No experiment can simultaneously hold width, depth, parameter count, and FLOPs equal across these architectures. FLOPs therefore remain a reported result in both protocols.

## Metrics and outputs

The benchmark records parameter count, analytical forward FLOPs, median and p95 inference latency, throughput, inference peak memory, full AdamW training-step latency and peak memory, effective TFLOP/s, and failure status.

CUDA events measure GPU time. `torch.cuda.max_memory_allocated()` measures PyTorch tensor memory, not whole-process GPU memory. A multiply-add counts as two FLOPs. The Mamba scan count is an explicit analytical proxy.

Outputs under `Results/sequence_efficiency/`:

- `results.csv`
- `metadata.json`
- `efficiency_summary.png`

## FlashAttention 4 on Axon

FlashAttention 4 is currently published as a prerelease. Install a fixed version for reproducibility:

```bash
pip install "flash-attn-4==4.0.0b33"
```

Check the import and detected backends:

```bash
python benchmarks/sequence_efficiency/benchmark.py --gpu 1 --list-backends
```

The backend listing checks the installed FA4 package without importing it. This is intentional: FA4 and Mamba-3 both use TVM-related runtimes, and importing FA4 before Mamba-3 can cause duplicate global type registration. The actual FA4 import and CUDA execution are tested when the `flash4` benchmark runs.

Run a short FA4 test:

```bash
python benchmarks/sequence_efficiency/benchmark.py \
  --gpu 1 --protocol width --models flash4 \
  --lengths 256 1024 --warmup 2 --repeats 3 --train-repeats 2 \
  --output Results/sequence_efficiency_flash4_smoke
```

See the complete command list at the end of the repository root `README.md`.

## Merge isolated-environment results

Run the Mamba/Transformer group and FA4 group with identical benchmark settings but different output directories. Then merge them from either environment:

```bash
python benchmarks/sequence_efficiency/merge_results.py \
  Results/sequence_efficiency_mamba \
  Results/sequence_efficiency_fa4 \
  --output Results/sequence_efficiency_combined
```

The script refuses to merge runs with different lengths, batch size, layers, heads, dtype, or architecture settings. It writes `results_combined.csv`, `metadata_combined.json`, and `efficiency_summary_combined.png`.
