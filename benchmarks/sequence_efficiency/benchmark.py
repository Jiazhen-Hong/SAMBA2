#!/usr/bin/env python3
"""Standalone sequence-block efficiency benchmark. It does not import SAMBA."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


def _early_gpu() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default="1")
    a, _ = p.parse_known_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)


_early_gpu()
import torch
from torch import nn
from torch.nn import functional as F


ALL_MODELS = [
    "mamba1", "mamba2", "mamba3", "transformer", "linear_transformer",
    "flash1", "flash2", "flash3", "flash4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="1", help="Physical GPU index before CUDA remapping")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--lengths", nargs="+", type=int,
                   default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--d-state", type=int, default=64,
                   help="Shared Mamba state size for a matched-width comparison")
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--train-repeats", type=int, default=5)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--memory-limit-gib", type=float, default=90.0,
                   help="Stop a model before the next length after this measured peak")
    p.add_argument("--output", type=Path, default=Path("Results/sequence_efficiency"))
    p.add_argument("--list-backends", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if a.d_model % a.heads:
        p.error("--d-model must be divisible by --heads")
    if min(a.lengths) <= 0:
        p.error("sequence lengths must be positive")
    return a


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def flash_status() -> dict[str, str]:
    version = package_version("flash-attn")
    out = {"flash1": "requires a separate flash-attn 1.x environment",
           "flash2": "flash-attn 2.x not installed",
           "flash3": "flash_attn_3 not importable",
           "flash4": "flash-attn-4 / flash_attn.cute not importable"}
    if version:
        major = version.split(".", 1)[0]
        if major == "1":
            out["flash1"] = f"available: flash-attn {version}"
        elif major == "2":
            out["flash2"] = f"available: flash-attn {version}"
    try:
        from flash_attn_3 import flash_attn_interface  # noqa: F401
        out["flash3"] = "available: flash_attn_3"
    except Exception:
        pass
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401
        out["flash4"] = "available: flash_attn.cute"
    except Exception:
        pass
    return out


class MambaResidual(nn.Module):
    def __init__(self, mixer: nn.Module, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class AttentionCore(nn.Module):
    def __init__(self, d_model: int, heads: int, backend: str):
        super().__init__()
        self.heads, self.head_dim, self.backend = heads, d_model // heads, backend
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _qkv(self, x: torch.Tensor):
        b, l, _ = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.heads, self.head_dim)
        return qkv.unbind(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(x)
        if self.backend == "transformer":
            # Deliberately materialize the causal attention matrix. This is the
            # plain mathematical baseline and cannot silently select a Flash kernel.
            qt, kt, vt = (z.transpose(1, 2) for z in (q, k, v))
            scores = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril_()
            scores.masked_fill_(~causal, torch.finfo(scores.dtype).min)
            y = torch.matmul(scores.softmax(dim=-1), vt).transpose(1, 2)
        elif self.backend == "linear_transformer":
            # Causal linear attention with phi(x)=elu(x)+1. Prefix sums keep O(L) complexity.
            qp, kp = F.elu(q) + 1, F.elu(k) + 1
            kv = torch.einsum("blhd,blhe->blhde", kp, v).cumsum(dim=1)
            ks = kp.cumsum(dim=1)
            y = torch.einsum("blhd,blhde->blhe", qp, kv)
            y = y / (torch.einsum("blhd,blhd->blh", qp, ks)[..., None] + 1e-6)
        elif self.backend == "flash2":
            from flash_attn import flash_attn_func
            y = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        elif self.backend == "flash3":
            from flash_attn_3 import flash_attn_interface
            y = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
        elif self.backend == "flash4":
            from flash_attn.cute import flash_attn_func
            y = flash_attn_func(q, k, v, causal=True)
        elif self.backend == "flash1":
            from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
            b, l, h, d = q.shape
            packed = torch.stack((q, k, v), dim=2).reshape(b * l, 3, h, d)
            cu = torch.arange(0, (b + 1) * l, l, device=x.device, dtype=torch.int32)
            y = flash_attn_unpadded_qkvpacked_func(packed, cu, l, 0.0, causal=True).view(b, l, h, d)
        else:
            raise ValueError(self.backend)
        if isinstance(y, tuple):
            y = y[0]
        return self.out(y.reshape(x.shape))


class AttentionLayer(nn.Module):
    def __init__(self, args: argparse.Namespace, backend: str):
        super().__init__()
        hidden = args.d_model * args.ffn_mult
        self.n1, self.attn = nn.LayerNorm(args.d_model), AttentionCore(args.d_model, args.heads, backend)
        self.n2 = nn.LayerNorm(args.d_model)
        self.ffn = nn.Sequential(nn.Linear(args.d_model, hidden, bias=False), nn.GELU(),
                                 nn.Linear(hidden, args.d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Stack(nn.Module):
    def __init__(self, layers: list[nn.Module], d_model: int):
        super().__init__()
        self.layers, self.norm = nn.ModuleList(layers), nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def build_model(name: str, args: argparse.Namespace, dtype: torch.dtype) -> nn.Module:
    if name.startswith("mamba"):
        from mamba_ssm import Mamba, Mamba2, Mamba3
        cls = {"mamba1": Mamba, "mamba2": Mamba2, "mamba3": Mamba3}[name]
        kwargs = dict(d_model=args.d_model, d_state=args.d_state, expand=args.expand, dtype=dtype)
        if name == "mamba1":
            kwargs["d_conv"] = args.d_conv
        elif name == "mamba2":
            kwargs.update(d_conv=args.d_conv, headdim=args.headdim, use_mem_eff_path=False)
        else:
            kwargs.update(headdim=args.headdim, is_mimo=False,
                          chunk_size=64 if dtype == torch.bfloat16 else 32)
        layers = [MambaResidual(cls(**kwargs), args.d_model) for _ in range(args.layers)]
    else:
        layers = [AttentionLayer(args, name) for _ in range(args.layers)]
    return Stack(layers, args.d_model).cuda().to(dtype=dtype)


@dataclass
class Result:
    model: str
    length: int
    status: str
    parameters: int | None = None
    inference_median_ms: float | None = None
    inference_p95_ms: float | None = None
    tokens_per_second: float | None = None
    inference_peak_gib: float | None = None
    training_median_ms: float | None = None
    training_peak_gib: float | None = None
    forward_gflops: float | None = None
    effective_tflops: float | None = None
    error: str | None = None


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def timed_forward(model: nn.Module, x: torch.Tensor, warmup: int, repeats: int):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(x); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.median(times), percentile(times, .95), torch.cuda.max_memory_allocated() / 2**30


def timed_training(model: nn.Module, x: torch.Tensor, repeats: int):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    model(x).float().square().mean().backward(); opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); times = []
    for _ in range(repeats):
        opt.zero_grad(set_to_none=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); y = model(x); y.float().square().mean().backward(); opt.step(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), torch.cuda.max_memory_allocated() / 2**30


def estimate_flops(name: str, a: argparse.Namespace, length: int) -> float:
    """Analytical forward FLOPs. One multiply-add counts as two FLOPs."""
    b, l, d, n, h, s, e = a.batch_size, length, a.d_model, a.layers, a.heads, a.d_state, a.expand
    if name.startswith("mamba"):
        inner = d * e
        # Projections + depthwise convolution + selective scan proxy. Mamba2/3 scan is approximate.
        per = 2*b*l*d*inner*2 + 2*b*l*inner*a.d_conv + 9*b*l*inner*s
    else:
        projections = 2*b*l*(3*d*d + d*d)
        ffn = 4*b*l*d*(d*a.ffn_mult)
        if name == "linear_transformer":
            attention = 4*b*l*h*(d//h)**2 + 4*b*l*h*(d//h)
        else:
            attention = 4*b*h*l*l*(d//h)
        per = projections + ffn + attention
    return n * per / 1e9


def unsupported_reason(name: str, statuses: dict[str, str]) -> str | None:
    if name.startswith("flash") and not statuses[name].startswith("available"):
        return statuses[name]
    return None


def write_results(rows: list[Result], out: Path, metadata: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=Result.__dataclass_fields__.keys()); w.writeheader()
        w.writerows(asdict(r) for r in rows)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = [("inference_median_ms", "Inference latency (ms)"),
               ("tokens_per_second", "Throughput (tokens/s)"),
               ("inference_peak_gib", "Inference peak memory (GiB)"),
               ("training_peak_gib", "Training peak memory (GiB)"),
               ("forward_gflops", "Estimated forward GFLOPs")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flat
    for ax, (field, title) in zip(axes, metrics):
        for name in metadata["args"]["models"]:
            rs = [r for r in rows if r.model == name and getattr(r, field) is not None]
            if rs: ax.plot([r.length for r in rs], [getattr(r, field) for r in rs], marker="o", label=name)
        ax.set_xscale("log", base=2); ax.set_title(title); ax.set_xlabel("Sequence length"); ax.grid(alpha=.3)
    axes[5].axis("off"); handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center")
    fig.suptitle("Standalone sequence block efficiency benchmark"); fig.tight_layout()
    fig.savefig(out / "efficiency_summary.png", dpi=180); plt.close(fig)


def main() -> None:
    a = parse_args(); statuses = flash_status()
    if a.list_backends:
        print(json.dumps(statuses, indent=2)); return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dtype = getattr(torch, a.dtype)
    rows: list[Result] = []
    for name in a.models:
        reason = unsupported_reason(name, statuses)
        if reason:
            for length in a.lengths: rows.append(Result(name, length, "UNSUPPORTED", error=reason))
            print(f"{name}: UNSUPPORTED ({reason})"); continue
        stop = False
        for length in a.lengths:
            if stop:
                rows.append(Result(name, length, "SAFETY_LIMIT", error="previous peak exceeded memory limit")); continue
            model = x = None
            try:
                model = build_model(name, a, dtype)
                params = sum(p.numel() for p in model.parameters())
                x = torch.randn(a.batch_size, length, a.d_model, device="cuda", dtype=dtype)
                med, p95, mem = timed_forward(model, x, a.warmup, a.repeats)
                train_ms = train_mem = None
                if not a.skip_training:
                    train_ms, train_mem = timed_training(model, x, a.train_repeats)
                gf = estimate_flops(name, a, length)
                row = Result(name, length, "OK", params, med, p95,
                             a.batch_size*length/(med/1000), mem, train_ms, train_mem,
                             gf, gf/(med/1000)/1000)
                rows.append(row); print(f"{name:18s} L={length:7d} {med:8.3f} ms {mem:7.2f} GiB")
                stop = max(mem, train_mem or 0) >= a.memory_limit_gib
            except torch.cuda.OutOfMemoryError as exc:
                rows.append(Result(name, length, "OOM", error=str(exc).splitlines()[0])); stop = True
                print(f"{name:18s} L={length:7d} OOM")
            except Exception as exc:
                rows.append(Result(name, length, "ERROR", error=f"{type(exc).__name__}: {exc}")); stop = True
                print(f"{name:18s} L={length:7d} ERROR: {exc}")
            finally:
                del model, x
                torch.cuda.empty_cache()
    metadata = {"args": vars(a) | {"output": str(a.output)}, "flash_backends": statuses,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "mamba_ssm": package_version("mamba-ssm"),
                "flop_definition": "analytical forward FLOPs; multiply-add=2; Mamba scan is an explicit proxy"}
    write_results(rows, a.output, metadata)
    print(f"Saved to {a.output}")


if __name__ == "__main__":
    main()
