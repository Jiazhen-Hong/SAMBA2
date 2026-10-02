# Standalone sequence efficiency benchmark

This benchmark does not import or run SAMBA. Every model receives the same random tensor with shape `[batch, sequence, d_model]`.

The attention models share the same pre-norm residual block, QKV projection, output projection, and FFN. Only the attention implementation changes. The Mamba models use the same width, layer count, expansion ratio, convolution width, and state size. Parameter counts and analytical FLOPs are reported because equal width does not imply equal parameter count or equal work.

## What is compared

- `mamba1`, `mamba2`, `mamba3`: official classes from the installed `mamba_ssm` package.
- `transformer`: a plain mathematical causal attention baseline that explicitly materializes the attention matrix. It cannot silently select a Flash kernel.
- `linear_transformer`: causal ELU feature-map linear attention written in PyTorch. This is a different attention algorithm.
- `flash1` to `flash4`: version-specific official FlashAttention kernels when importable.

FlashAttention is an implementation of exact attention, not a separate model architecture. FlashAttention generations are not normally all available from one Python environment. FlashAttention 1 requires a legacy 1.x environment. FlashAttention 3 uses the Hopper package. FlashAttention 4 uses the CuTeDSL package and targets Hopper and Blackwell. Missing or incompatible kernels are recorded as `UNSUPPORTED`; they are never replaced silently.

## Metrics

- Median and p95 inference latency after warmup
- Token throughput
- Peak allocated inference memory
- Full training-step latency and peak memory, including forward, backward, and AdamW
- Parameter count
- Analytical forward FLOPs and effective TFLOP/s
- OOM, runtime error, unsupported backend, and safety-limit status

CUDA events measure GPU time. `torch.cuda.max_memory_allocated()` measures tensor memory allocated by PyTorch, not the whole GPU process. FLOPs count a multiply-add as two operations. Mamba selective-scan FLOPs are a documented analytical proxy, so comparisons with profiler-reported hardware FLOPs should be labeled separately.

## Outputs

The default output directory is `Results/sequence_efficiency/`:

- `results.csv`: all raw measurements and statuses
- `metadata.json`: command settings, package versions, GPU, and backend detection
- `efficiency_summary.png`: latency, throughput, memory, and FLOPs plots

See the Axon command reference at the end of the repository root `README.md`.
