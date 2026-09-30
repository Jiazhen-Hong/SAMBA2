#!/usr/bin/env python3
"""Benchmark Mamba-1, Mamba-2, and Mamba-3 on simulated EEG.

This script is intended to run on an NVIDIA GPU. It compares otherwise
identical EEG reconstruction networks whose sequence block is Mamba-1,
Mamba-2, or Mamba-3. It can optionally perform a short masked-reconstruction
pretraining stage before measuring inference memory and speed.

Example on Axon (physical GPU 1):
    python notebook/benchmark_mamba_eeg.py --gpu 1

Outputs are written to Results/mamba123_benchmark by default. The directory
contains two requested plots, a combined plot, raw CSV results, run metadata,
and optional locally trained checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


def _gpu_from_cli() -> str:
    """Read --gpu before importing torch so CUDA visibility is set in time."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="1")
    known, _ = parser.parse_known_args()
    return str(known.gpu)


PHYSICAL_GPU = _gpu_from_cli()
os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


MODEL_LABELS = {
    "mamba1": "Mamba-1",
    "mamba2": "Mamba-2",
    "mamba3": "Mamba-3",
    "transformer": "Transformer",
    "linear_transformer": "Linear Transformer",
}
MODEL_COLORS = {
    "mamba1": "#4472C4",
    "mamba2": "#ED7D31",
    "mamba3": "#70AD47",
    "transformer": "#A64D79",
    "linear_transformer": "#00A6A6",
}
MODEL_MARKERS = {
    "mamba1": "o",
    "mamba2": "s",
    "mamba3": "^",
    "transformer": "D",
    "linear_transformer": "P",
}


@dataclass
class Result:
    model: str
    sequence_length: int
    status: str
    peak_memory_gib: float | None = None
    latency_ms: float | None = None
    throughput_samples_per_ms: float | None = None
    training_status: str = "not_run"
    peak_training_memory_gib: float | None = None
    parameter_count: int | None = None
    error: str = ""
    training_error: str = ""


def parse_lengths(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sequence lengths must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Mamba-1/2/3 with simulated EEG data."
    )
    parser.add_argument(
        "--gpu",
        default="1",
        help=(
            "Physical CUDA GPU index exposed to this process (default: 1). "
            "Inside the script it is mapped to cuda:0."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_LABELS),
        default=list(MODEL_LABELS),
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths(
            "256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144"
        ),
        help="Comma-separated EEG sequence lengths.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=14)
    parser.add_argument("--sample-rate", type=float, default=128.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
        help="Number of heads for Transformer and Linear Transformer.",
    )
    parser.add_argument(
        "--linear-attention-chunk-size",
        type=int,
        default=256,
        help="Chunk size for causal Linear Transformer prefix accumulation.",
    )
    parser.add_argument(
        "--mamba2-path",
        choices=("auto", "fused", "fallback"),
        default="auto",
        help=(
            "Mamba-2 execution path. 'auto' uses the fused path only when the "
            "causal-conv1d CUDA function is importable."
        ),
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--skip-training-memory",
        action="store_true",
        help="Skip the additional forward/backward/AdamW peak-memory curve.",
    )
    parser.add_argument(
        "--transformer-max-length",
        type=int,
        default=16384,
        help="Inference safety limit for exact Transformer; 0 disables it.",
    )
    parser.add_argument(
        "--transformer-training-max-length",
        type=int,
        default=8192,
        help="Training-memory safety limit for exact Transformer; 0 disables it.",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=5,
        help="Masked EEG reconstruction steps per model; use 0 to disable.",
    )
    parser.add_argument("--pretrain-length", type=int, default=4096)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing mamba1.pt, mamba2.pt, and/or "
            "mamba3.pt. Missing files fall back to short synthetic pretraining."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Results/mamba123_benchmark")
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="Try larger lengths after an OOM; by default they are marked skipped_oom.",
    )
    args = parser.parse_args()

    positive_fields = (
        "batch_size",
        "channels",
        "sample_rate",
        "d_model",
        "layers",
        "expand",
        "d_conv",
        "headdim",
        "attention_heads",
        "linear_attention_chunk_size",
        "repeats",
        "pretrain_length",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.pretrain_steps < 0:
        parser.error("--warmup and --pretrain-steps cannot be negative")
    if args.transformer_max_length < 0 or args.transformer_training_max_length < 0:
        parser.error("Transformer length limits cannot be negative")
    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask-ratio must be between 0 and 1")
    if args.d_model * args.expand % args.headdim != 0:
        parser.error("d-model * expand must be divisible by headdim")
    if args.d_model % args.attention_heads != 0:
        parser.error("d-model must be divisible by attention-heads")
    return args


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def causal_conv1d_status() -> tuple[bool, str]:
    """Return whether Mamba-2's fused causal-conv1d functions are usable."""
    try:
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.triton import ssd_combined

        functions = {
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_fwd_function": ssd_combined.causal_conv1d_fwd_function,
            "causal_conv1d_bwd_function": ssd_combined.causal_conv1d_bwd_function,
        }
        missing = [name for name, function in functions.items() if not callable(function)]
        if missing:
            return False, "non-callable fused function(s): " + ", ".join(missing)
        try:
            version = importlib.metadata.version("causal-conv1d")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return True, f"causal-conv1d {version}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {' '.join(str(exc).split())}"


def use_mamba2_fused_path(args: argparse.Namespace) -> bool:
    available, detail = causal_conv1d_status()
    if args.mamba2_path == "fused" and not available:
        raise RuntimeError(
            "--mamba2-path fused was requested, but the causal-conv1d CUDA "
            f"function is unavailable ({detail}). Use --mamba2-path fallback "
            "or reinstall a causal-conv1d build compatible with this PyTorch/CUDA."
        )
    if args.mamba2_path == "fallback":
        return False
    return available


def build_mamba(kind: str, args: argparse.Namespace, dtype: torch.dtype) -> nn.Module:
    try:
        from mamba_ssm import Mamba, Mamba2, Mamba3
    except ImportError as exc:
        raise RuntimeError(
            "mamba_ssm with Mamba, Mamba2, and Mamba3 is required. "
            "This benchmark targets mamba_ssm 2.3.2.post1."
        ) from exc

    common = dict(d_model=args.d_model, expand=args.expand, dtype=dtype)
    if kind == "mamba1":
        return Mamba(d_state=16, d_conv=args.d_conv, **common)
    if kind == "mamba2":
        return Mamba2(
            d_state=64,
            d_conv=args.d_conv,
            headdim=args.headdim,
            use_mem_eff_path=use_mamba2_fused_path(args),
            **common,
        )
    if kind == "mamba3":
        # SISO avoids the optional TileLang MIMO kernel and is the most portable
        # Mamba-3 path. Official guidance uses chunk 64 for BF16 SISO and 32 for FP32.
        chunk_size = 64 if dtype == torch.bfloat16 else 32
        return Mamba3(
            d_state=128,
            headdim=args.headdim,
            is_mimo=False,
            chunk_size=chunk_size,
            **common,
        )
    raise ValueError(f"Unknown model kind: {kind}")


class MambaResidualBlock(nn.Module):
    def __init__(self, kind: str, args: argparse.Namespace, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.LayerNorm(args.d_model, dtype=dtype)
        self.mixer = build_mamba(kind, args, dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mixer(self.norm(hidden))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, dtype=dtype),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, dtype=dtype),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class CausalSelfAttention(nn.Module):
    """Exact causal softmax attention using PyTorch SDPA kernels."""

    def __init__(self, args: argparse.Namespace, dtype: torch.dtype):
        super().__init__()
        self.num_heads = args.attention_heads
        self.head_dim = args.d_model // self.num_heads
        self.qkv = nn.Linear(args.d_model, 3 * args.d_model, dtype=dtype)
        self.out_projection = nn.Linear(args.d_model, args.d_model, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = hidden.shape
        qkv = self.qkv(hidden).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, d_model)
        return self.out_projection(attended)


class CausalLinearAttention(nn.Module):
    """Causal ELU+1 linear attention with bounded-size prefix chunks."""

    def __init__(self, args: argparse.Namespace, dtype: torch.dtype):
        super().__init__()
        self.num_heads = args.attention_heads
        self.head_dim = args.d_model // self.num_heads
        self.chunk_size = args.linear_attention_chunk_size
        self.qkv = nn.Linear(args.d_model, 3 * args.d_model, dtype=dtype)
        self.out_projection = nn.Linear(args.d_model, args.d_model, dtype=dtype)

    @staticmethod
    def feature_map(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(tensor).add_(1.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = hidden.shape
        qkv = self.qkv(hidden).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.feature_map(query.transpose(1, 2))
        key = self.feature_map(key.transpose(1, 2))
        value = value.transpose(1, 2)

        key_state = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        key_value_state = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            query_chunk = query[:, :, start:stop]
            key_chunk = key[:, :, start:stop]
            value_chunk = value[:, :, start:stop]
            key_prefix = key_chunk.cumsum(dim=2) + key_state.unsqueeze(2)
            key_value = torch.einsum(
                "bhld,bhle->bhlde", key_chunk, value_chunk
            )
            key_value_prefix = key_value.cumsum(dim=2) + key_value_state.unsqueeze(2)
            numerator = torch.einsum(
                "bhld,bhlde->bhle", query_chunk, key_value_prefix
            )
            denominator = torch.einsum(
                "bhld,bhld->bhl", query_chunk, key_prefix
            ).unsqueeze(-1)
            outputs.append(numerator / denominator.clamp_min(1e-6))
            key_state = key_prefix[:, :, -1]
            key_value_state = key_value_prefix[:, :, -1]

        attended = torch.cat(outputs, dim=2)
        attended = attended.transpose(1, 2).reshape(batch, length, d_model)
        return self.out_projection(attended)


class AttentionResidualBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        args: argparse.Namespace,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.d_model, dtype=dtype)
        self.attention = attention
        self.ffn_norm = nn.LayerNorm(args.d_model, dtype=dtype)
        self.ffn = FeedForward(args.d_model, dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.ffn(self.ffn_norm(hidden))


def build_sequence_block(
    kind: str, args: argparse.Namespace, dtype: torch.dtype
) -> nn.Module:
    if kind in ("mamba1", "mamba2", "mamba3"):
        return MambaResidualBlock(kind, args, dtype)
    if kind == "transformer":
        return AttentionResidualBlock(CausalSelfAttention(args, dtype), args, dtype)
    if kind == "linear_transformer":
        return AttentionResidualBlock(CausalLinearAttention(args, dtype), args, dtype)
    raise ValueError(f"Unknown model kind: {kind}")


class EEGMambaModel(nn.Module):
    """Small channel-to-channel EEG reconstruction model."""

    def __init__(self, kind: str, args: argparse.Namespace, dtype: torch.dtype):
        super().__init__()
        self.input_projection = nn.Linear(
            args.channels, args.d_model, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            build_sequence_block(kind, args, dtype) for _ in range(args.layers)
        )
        self.output_norm = nn.LayerNorm(args.d_model, dtype=dtype)
        self.output_projection = nn.Linear(
            args.d_model, args.channels, dtype=dtype
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # Input and output: [batch, channels, time]. Mamba: [batch, time, d_model].
        hidden = self.input_projection(eeg.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        reconstructed = self.output_projection(self.output_norm(hidden))
        return reconstructed.transpose(1, 2)


def simulate_eeg(
    batch: int,
    channels: int,
    length: int,
    sample_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate reproducible EEG-like mixtures without storing a dataset."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    time_axis = torch.arange(length, device=device, dtype=torch.float32)
    time_axis = time_axis / sample_rate

    # Delta, theta, alpha, beta, and low-gamma components with channel-specific mix.
    frequencies = torch.tensor(
        [2.0, 6.0, 10.0, 20.0, 35.0], device=device, dtype=torch.float32
    )
    phase = 2.0 * torch.pi * torch.rand(
        batch, channels, len(frequencies), 1, device=device, generator=generator
    )
    amplitude = 0.2 + torch.rand(
        batch, channels, len(frequencies), 1, device=device, generator=generator
    )
    waves = torch.sin(
        2.0 * torch.pi * frequencies.view(1, 1, -1, 1) * time_axis + phase
    )
    signal = (amplitude * waves).sum(dim=2)

    # Add shared activity and sensor noise. Scaling approximates normalized EEG,
    # not voltage in microvolts, which is sufficient for architecture benchmarking.
    shared = torch.randn(batch, 1, length, device=device, generator=generator)
    noise = torch.randn(batch, channels, length, device=device, generator=generator)
    signal = signal + 0.25 * shared + 0.15 * noise
    signal = signal / signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return signal.to(dtype=dtype)


def contiguous_mask(
    batch: int,
    channels: int,
    length: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Make shared temporal spans that resemble masked EEG pretraining."""
    span = max(1, min(32, length // 8))
    start_probability = min(1.0, ratio / span)
    starts = torch.rand(batch, 1, length, device=device) < start_probability
    expanded = torch.nn.functional.max_pool1d(
        starts.float(), kernel_size=span, stride=1, padding=span // 2
    )[..., :length]
    return expanded.bool().expand(batch, channels, length)


def checkpoint_path(args: argparse.Namespace, kind: str) -> Path | None:
    if args.checkpoint_dir is None:
        return None
    candidate = args.checkpoint_dir / f"{kind}.pt"
    return candidate if candidate.is_file() else None


def load_checkpoint(model: nn.Module, path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


def pretrain_model(
    model: nn.Module,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    source = checkpoint_path(args, kind)
    if source is not None:
        load_checkpoint(model, source)
        return {"source": str(source), "steps": 0, "final_loss": None}
    if args.pretrain_steps == 0:
        return {"source": "random_initialization", "steps": 0, "final_loss": None}

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    final_loss = None
    for step in range(args.pretrain_steps):
        target = simulate_eeg(
            args.batch_size,
            args.channels,
            args.pretrain_length,
            args.sample_rate,
            device,
            dtype,
            args.seed + 1000 + step,
        )
        mask = contiguous_mask(
            args.batch_size,
            args.channels,
            args.pretrain_length,
            args.mask_ratio,
            device,
        )
        corrupted = target.masked_fill(mask, 0.0)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(corrupted)
        # Compute loss in FP32 for stable reporting under BF16/FP16.
        squared_error = (prediction.float() - target.float()).square()
        loss = squared_error.masked_select(mask).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        print(
            f"  pretrain {kind}: step {step + 1}/{args.pretrain_steps}, "
            f"masked MSE={final_loss:.6f}",
            flush=True,
        )
        del target, mask, corrupted, prediction, squared_error, loss

    checkpoint = args.output_dir / "checkpoints" / f"{kind}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kind": kind,
            "args": serializable_args(args),
        },
        checkpoint,
    )
    del optimizer
    torch.cuda.empty_cache()
    return {
        "source": "synthetic_masked_reconstruction",
        "steps": args.pretrain_steps,
        "final_loss": final_loss,
        "saved_checkpoint": str(checkpoint),
    }


def is_cuda_oom(exc: BaseException) -> bool:
    oom_class = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_class) or "out of memory" in str(exc).lower()


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_one_length(
    model: nn.Module,
    kind: str,
    length: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    parameter_count: int,
) -> Result:
    eeg = simulate_eeg(
        args.batch_size,
        args.channels,
        length,
        args.sample_rate,
        device,
        dtype,
        args.seed + length,
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(eeg)
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        for start, end in zip(start_events, end_events):
            start.record()
            output = model(eeg)
            end.record()
        torch.cuda.synchronize()

    timings = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    latency_ms = float(np.median(timings))
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    throughput = args.batch_size * length / latency_ms
    del eeg, output, start_events, end_events
    return Result(
        model=kind,
        sequence_length=length,
        status="ok",
        peak_memory_gib=peak_gib,
        latency_ms=latency_ms,
        throughput_samples_per_ms=throughput,
        parameter_count=parameter_count,
    )


def benchmark_training_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    length: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Measure a complete masked-reconstruction training step."""
    target = simulate_eeg(
        args.batch_size,
        args.channels,
        length,
        args.sample_rate,
        device,
        dtype,
        args.seed + 500_000 + length,
    )
    mask = contiguous_mask(
        args.batch_size,
        args.channels,
        length,
        args.mask_ratio,
        device,
    )
    corrupted = target.masked_fill(mask, 0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    prediction = model(corrupted)
    squared_error = (prediction.float() - target.float()).square()
    loss = squared_error.masked_select(mask).mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    optimizer.zero_grad(set_to_none=True)
    del target, mask, corrupted, prediction, squared_error, loss
    return peak_gib


def exceeds_transformer_limit(
    kind: str,
    length: int,
    args: argparse.Namespace,
    training: bool,
) -> bool:
    if kind != "transformer":
        return False
    limit = (
        args.transformer_training_max_length
        if training
        else args.transformer_max_length
    )
    return limit > 0 and length > limit


def compact_error(exc: BaseException, limit: int = 400) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def write_results(results: Iterable[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(Result.__dataclass_fields__)
    with (output_dir / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def plot_metric(
    results: list[Result],
    models: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    log_x: bool = True,
    status_field: str = "status",
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    successful_values = [
        getattr(row, metric)
        for row in results
        if getattr(row, status_field) == "ok" and getattr(row, metric) is not None
    ]
    y_min = min(successful_values) if successful_values else 1.0
    y_max = max(successful_values) if successful_values else 1.0
    is_memory_metric = "memory" in metric
    oom_y = y_max * 1.12 if is_memory_metric else max(y_min * 0.7, 1e-6)
    error_y = y_max * 1.22 if is_memory_metric else max(y_min * 0.5, 1e-6)

    for kind in models:
        rows = [row for row in results if row.model == kind]
        ok_rows = [row for row in rows if getattr(row, status_field) == "ok"]
        if ok_rows:
            axis.plot(
                [row.sequence_length for row in ok_rows],
                [getattr(row, metric) for row in ok_rows],
                label=MODEL_LABELS[kind],
                color=MODEL_COLORS[kind],
                marker=MODEL_MARKERS[kind],
                linewidth=2,
                markersize=6,
            )
        oom_rows = [
            row
            for row in rows
            if getattr(row, status_field) in ("oom", "skipped_oom")
        ]
        if oom_rows:
            axis.scatter(
                [row.sequence_length for row in oom_rows],
                [oom_y] * len(oom_rows),
                color=MODEL_COLORS[kind],
                marker="X",
                s=75,
                zorder=4,
            )
            first = min(oom_rows, key=lambda row: row.sequence_length)
            axis.annotate(
                "OOM",
                (first.sequence_length, oom_y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color=MODEL_COLORS[kind],
                fontsize=8,
            )
        error_rows = [
            row for row in rows if getattr(row, status_field) == "error"
        ]
        if error_rows:
            axis.scatter(
                [row.sequence_length for row in error_rows],
                [error_y] * len(error_rows),
                color=MODEL_COLORS[kind],
                marker="x",
                s=65,
                zorder=4,
            )
        limited_rows = [
            row for row in rows if getattr(row, status_field) == "skipped_limit"
        ]
        if limited_rows:
            first = min(limited_rows, key=lambda row: row.sequence_length)
            axis.annotate(
                "safety limit",
                (first.sequence_length, error_y),
                rotation=90,
                va="center",
                ha="center",
                color=MODEL_COLORS[kind],
                fontsize=8,
            )

    if log_x:
        axis.set_xscale("log", base=2)
    axis.set_xlabel("EEG sequence length (samples)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_all(results: list[Result], models: list[str], output_dir: Path) -> None:
    plot_metric(
        results,
        models,
        "peak_memory_gib",
        "Peak allocated GPU memory (GiB)",
        "EEG sequence-model inference memory",
        output_dir / "mamba123_memory.png",
    )
    plot_metric(
        results,
        models,
        "throughput_samples_per_ms",
        "Throughput (batch × samples / ms)",
        "EEG sequence-model inference speed",
        output_dir / "mamba123_speed.png",
    )
    plot_metric(
        results,
        models,
        "peak_training_memory_gib",
        "Peak allocated GPU memory (GiB)",
        "EEG masked-pretraining memory (forward + backward + AdamW)",
        output_dir / "mamba123_training_memory.png",
        status_field="training_status",
    )

    memory_image = plt.imread(output_dir / "mamba123_memory.png")
    speed_image = plt.imread(output_dir / "mamba123_speed.png")
    figure, axes = plt.subplots(1, 2, figsize=(18, 6.3), constrained_layout=True)
    axes[0].imshow(memory_image)
    axes[1].imshow(speed_image)
    for axis, label in zip(axes, ("(a) Memory usage", "(b) Inference speed")):
        axis.axis("off")
        axis.set_title(label, y=-0.06, fontsize=13)
    figure.savefig(output_dir / "mamba123_combined.png", dpi=180)
    plt.close(figure)


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    import mamba_ssm

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    causal_available, causal_detail = causal_conv1d_status()
    if args.mamba2_path == "fallback":
        selected_mamba2_path = "fallback"
    elif args.mamba2_path == "fused":
        selected_mamba2_path = "fused" if causal_available else "unavailable"
    else:
        selected_mamba2_path = "fused" if causal_available else "fallback"
    return {
        "arguments": serializable_args(args),
        "physical_gpu_requested": PHYSICAL_GPU,
        "visible_cuda_device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": total_bytes / (1024**3),
        "gpu_free_memory_at_start_gib": free_bytes / (1024**3),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "causal_conv1d_available": causal_available,
        "causal_conv1d_detail": causal_detail,
        "mamba2_execution_path": selected_mamba2_path,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def main() -> int:
    args = parse_args()
    if str(args.gpu) != PHYSICAL_GPU:
        raise RuntimeError("--gpu changed after CUDA initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CUDA device is visible")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = torch_dtype(args.dtype)
    metadata = environment_metadata(args)
    print(json.dumps(metadata, indent=2), flush=True)
    if metadata["mamba2_execution_path"] == "fallback":
        MODEL_LABELS["mamba2"] = "Mamba-2 (fallback)"
        print(
            "\nMamba-2 fused causal-conv1d kernels are unavailable; "
            "using the supported fallback path for this run.",
            flush=True,
        )

    results: list[Result] = []
    pretraining: dict[str, Any] = {}
    for model_index, kind in enumerate(args.models):
        print(f"\n=== {MODEL_LABELS[kind]} ===", flush=True)
        release_cuda()
        try:
            model = EEGMambaModel(kind, args, dtype).to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(f"parameters: {parameter_count:,}", flush=True)
            try:
                pretraining[kind] = pretrain_model(
                    model, kind, args, device, dtype
                )
            except BaseException as exc:
                if not is_cuda_oom(exc):
                    raise
                pretraining[kind] = {
                    "source": "random_initialization_after_pretrain_oom",
                    "steps": 0,
                    "error": compact_error(exc),
                }
                print("  pretraining OOM; continuing with current weights", flush=True)
                model.zero_grad(set_to_none=True)
                release_cuda()

            saw_oom = False
            for length in args.lengths:
                if saw_oom and not args.continue_after_oom:
                    result = Result(
                        model=kind,
                        sequence_length=length,
                        status="skipped_oom",
                        parameter_count=parameter_count,
                        error="Skipped because a shorter sequence already caused OOM.",
                    )
                    results.append(result)
                    continue
                release_cuda()
                try:
                    result = benchmark_one_length(
                        model,
                        kind,
                        length,
                        args,
                        device,
                        dtype,
                        parameter_count,
                    )
                    print(
                        f"  L={length:>7}: {result.peak_memory_gib:.3f} GiB, "
                        f"{result.latency_ms:.3f} ms, "
                        f"{result.throughput_samples_per_ms:.2f} samples/ms",
                        flush=True,
                    )
                except BaseException as exc:
                    status = "oom" if is_cuda_oom(exc) else "error"
                    result = Result(
                        model=kind,
                        sequence_length=length,
                        status=status,
                        parameter_count=parameter_count,
                        error=compact_error(exc),
                    )
                    print(f"  L={length:>7}: {status.upper()} - {result.error}", flush=True)
                    if status == "oom":
                        saw_oom = True
                    else:
                        traceback.print_exc()
                results.append(result)
                write_results(results, args.output_dir)
                plot_all(results, args.models, args.output_dir)
        except BaseException as exc:
            # A construction or pretraining error is recorded for every requested
            # length so the plots/CSV show that the model did not silently vanish.
            status = "oom" if is_cuda_oom(exc) else "error"
            print(f"  model-level {status.upper()}: {compact_error(exc)}", flush=True)
            traceback.print_exc()
            for length in args.lengths:
                results.append(
                    Result(
                        model=kind,
                        sequence_length=length,
                        status=status,
                        error=compact_error(exc),
                    )
                )
        finally:
            if "model" in locals():
                del model
            release_cuda()
            write_results(results, args.output_dir)
            plot_all(results, args.models, args.output_dir)

    metadata["pretraining"] = pretraining
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nSaved results to: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
