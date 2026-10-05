#!/usr/bin/env python3
"""Redraw existing sequence-efficiency CSV results without rerunning models."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


MODEL_STYLES = {
    # Mamba generations: one blue family, light to dark.
    "mamba1": ("#9ECAE1", "o", "-"),
    "mamba2": ("#4292C6", "s", "--"),
    "mamba3": ("#08519C", "^", "-."),
    # Handwritten attention variants: one orange family.
    "transformer_math": ("#FEC44F", "X", ":"),
    "linear_transformer": ("#D95F0E", "P", "-."),
    # Optimized exact attention: one green family.
    "pytorch_sdpa": ("#74C476", "D", "--"),
    "flash4": ("#006D2C", "v", "-"),
}
DISPLAY_NAMES = {
    "mamba1": "Mamba-1",
    "mamba2": "Mamba-2",
    "mamba3": "Mamba-3",
    "transformer_math": "Transformer (Math)",
    "linear_transformer": "Linear Transformer (ELU+1)",
    "pytorch_sdpa": "Transformer (PyTorch SDPA)",
    "flash4": "Transformer (FlashAttention-4)",
}
NUMERIC_FIELDS = {
    "length": int,
    "d_model": int,
    "parameters": int,
    "parameter_error_percent": float,
    "inference_median_ms": float,
    "inference_p95_ms": float,
    "tokens_per_second": float,
    "inference_peak_gib": float,
    "training_median_ms": float,
    "training_peak_gib": float,
    "forward_gflops": float,
    "effective_tflops": float,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="results.csv or results_combined.csv")
    parser.add_argument("--output", type=Path, required=True, help="New figure directory")
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--prefix", default="efficiency_128hz")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def value(row: dict, field: str):
    raw = row.get(field, "")
    if raw in ("", None, "None"):
        return None
    return NUMERIC_FIELDS.get(field, str)(raw)


def format_seconds(seconds: float) -> str:
    return f"{seconds:g} s"


def style_axis(axis, title: str, sampling_rate: float, log_y: bool) -> None:
    axis.set_xscale("log", base=2)
    if log_y:
        axis.set_yscale("log")
    axis.set_title(title, fontsize=14, fontweight="bold", pad=10)
    axis.set_xlabel(f"EEG duration at {sampling_rate:g} Hz (seconds)",
                    fontsize=12, fontweight="bold")
    axis.grid(which="both", alpha=0.22, linewidth=0.8)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(10)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)


def plot_rows(
    rows: list[dict], output: Path, sampling_rate: float = 128.0,
    prefix: str = "efficiency_128hz", log_y: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    if sampling_rate <= 0:
        raise ValueError("sampling rate must be positive")
    # Column order requested for reading: latency, memory, then other metrics.
    metrics = [
        ("inference_median_ms", "Inference latency (ms)"),
        ("inference_peak_gib", "Inference peak memory (GiB)"),
        ("tokens_per_second", "Throughput (tokens/s)"),
        ("training_median_ms", "Training-step latency (ms)"),
        ("training_peak_gib", "Training peak memory (GiB)"),
        ("forward_gflops", "Estimated forward GFLOPs"),
    ]
    model_order = list(MODEL_STYLES)
    series = sorted(
        {(row.get("protocol", "unknown"), row["model"]) for row in rows},
        key=lambda item: (item[0], model_order.index(item[1])
                          if item[1] in model_order else len(model_order)),
    )
    protocols = {protocol for protocol, _ in series}
    offsets = {item: index - (len(series) - 1) / 2 for index, item in enumerate(series)}
    fig, axes = plt.subplots(2, 3, figsize=(22, 12.5))
    axis_list = axes.flat
    all_lengths = sorted({value(row, "length") for row in rows if value(row, "length")})
    tick_seconds = [length / sampling_rate for length in all_lengths]

    for axis, (field, title) in zip(axis_list, metrics):
        for protocol, model in series:
            model_rows = [row for row in rows if row.get("protocol", "unknown") == protocol
                          and row["model"] == model]
            selected = [row for row in model_rows if value(row, field) is not None]
            selected.sort(key=lambda row: value(row, "length"))
            if not model_rows:
                continue
            reference = selected[0] if selected else model_rows[0]
            width, parameters = value(reference, "d_model"), value(reference, "parameters")
            name = DISPLAY_NAMES.get(model, model)
            detail = f"d={width}, {parameters / 1e6:.2f}M" if width and parameters else ""
            label = f"{name} ({detail})" if detail else name
            if len(protocols) > 1:
                label += f" [{protocol}]"
            color, marker, linestyle = MODEL_STYLES.get(model, ("#555555", "o", "-"))
            shift = offsets[(protocol, model)] * 0.012
            if selected:
                x_values = [(value(row, "length") / sampling_rate) * (2 ** shift)
                            for row in selected]
                axis.plot(
                    x_values, [value(row, field) for row in selected],
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=2.6, markersize=8.5, markerfacecolor="white",
                    markeredgewidth=2.1, alpha=0.96, label=label,
                )
            oom_rows = [row for row in model_rows if row.get("status") == "OOM"]
            for oom_index, row in enumerate(oom_rows):
                x_oom = (value(row, "length") / sampling_rate) * (2 ** shift)
                y_fraction = 0.94 - 0.055 * (model_order.index(model) % 4)
                axis.scatter(
                    [x_oom], [y_fraction], transform=axis.get_xaxis_transform(),
                    marker="X", s=115, color=color, edgecolor="black",
                    linewidth=0.8, zorder=20, clip_on=False,
                )
                axis.annotate(
                    "OOM", xy=(x_oom, y_fraction), xycoords=axis.get_xaxis_transform(),
                    xytext=(0, 8 + 2 * oom_index), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=color,
                )
        style_axis(axis, title, sampling_rate, log_y)
        axis.set_xticks(tick_seconds)
        axis.set_xticklabels([format_seconds(seconds) for seconds in tick_seconds],
                             rotation=35, ha="right")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(3, max(1, len(labels))),
        frameon=True, fancybox=True, framealpha=0.98, fontsize=13,
        borderpad=0.8, handlelength=2.8, columnspacing=1.6,
    )
    scale_name = "log y-axis" if log_y else "linear y-axis"
    fig.suptitle(
        f"Sequence-model efficiency for EEG duration at {sampling_rate:g} Hz ({scale_name})",
        fontsize=18, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.018,
        "Small horizontal offsets only reveal overlapping curves; CSV values are unchanged. "
        "X markers denote recorded CUDA out-of-memory events.",
        ha="center", fontsize=10.5, fontweight="bold", color="#333333",
    )
    fig.tight_layout(rect=(0.025, 0.13, 0.99, 0.955), w_pad=2.2, h_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "log" if log_y else "linear"
    destination = output / f"{prefix}_{suffix}.png"
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    linear = plot_rows(rows, args.output, args.sampling_rate, args.prefix, log_y=False)
    log = plot_rows(rows, args.output, args.sampling_rate, args.prefix, log_y=True)
    print(f"Saved {linear}")
    print(f"Saved {log}")


if __name__ == "__main__":
    main()
