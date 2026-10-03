#!/usr/bin/env python3
"""Merge benchmark runs from isolated environments and redraw common figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
CHECKED_ARGS = [
    "lengths", "batch_size", "layers", "heads", "ffn_mult", "d_state",
    "headdim", "d_conv", "expand", "dtype",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Run directories containing results.csv and metadata.json")
    parser.add_argument("--output", type=Path,
                        default=Path("Results/sequence_efficiency_combined"))
    return parser.parse_args()


def load_run(path: Path) -> tuple[list[dict], dict]:
    result_path, metadata_path = path / "results.csv", path / "metadata.json"
    if not result_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{path} must contain results.csv and metadata.json")
    with result_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    metadata = json.loads(metadata_path.read_text())
    for row in rows:
        row["source_run"] = str(path)
    return rows, metadata


def validate_runs(metadata: list[dict]) -> None:
    reference = metadata[0]["args"]
    differences = []
    for index, item in enumerate(metadata[1:], start=2):
        current = item["args"]
        for key in CHECKED_ARGS:
            if current.get(key) != reference.get(key):
                differences.append(
                    f"input {index}: {key}={current.get(key)!r}, expected {reference.get(key)!r}"
                )
    if differences:
        raise ValueError("Runs are not comparable:\n  " + "\n  ".join(differences))


def value(row: dict, field: str):
    raw = row.get(field, "")
    if raw in ("", None, "None"):
        return None
    return NUMERIC_FIELDS.get(field, str)(raw)


def write_plot(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV was merged without a figure")
        return
    metrics = [
        ("inference_median_ms", "Inference latency (ms)"),
        ("tokens_per_second", "Throughput (tokens/s)"),
        ("inference_peak_gib", "Inference peak memory (GiB)"),
        ("training_median_ms", "Training-step latency (ms)"),
        ("training_peak_gib", "Training peak memory (GiB)"),
        ("forward_gflops", "Estimated forward GFLOPs"),
    ]
    series = sorted({(row["protocol"], row["model"]) for row in rows})
    protocols = {protocol for protocol, _ in series}
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for axis, (field, title) in zip(axes.flat, metrics):
        for protocol, model in series:
            selected = [row for row in rows if row["protocol"] == protocol
                        and row["model"] == model and value(row, field) is not None]
            selected.sort(key=lambda row: value(row, "length"))
            if selected:
                label = f"{model} [{protocol}]" if len(protocols) > 1 else model
                axis.plot([value(row, "length") for row in selected],
                          [value(row, field) for row in selected], marker="o", label=label)
        axis.set_xscale("log", base=2)
        axis.set_title(title)
        axis.set_xlabel("Sequence length")
        axis.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))))
    fig.suptitle("Sequence efficiency across isolated environments")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output / "efficiency_summary_combined.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    all_rows, all_metadata = [], []
    for path in args.inputs:
        rows, metadata = load_run(path)
        all_rows.extend(rows)
        all_metadata.append(metadata)
    validate_runs(all_metadata)
    seen, duplicates = set(), []
    for row in all_rows:
        key = (row["protocol"], row["model"], row["length"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate protocol/model/length rows: {duplicates[:5]}")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with (args.output / "results_combined.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    combined_metadata = {
        "source_runs": [str(path) for path in args.inputs],
        "validated_args": {key: all_metadata[0]["args"].get(key) for key in CHECKED_ARGS},
        "source_metadata": all_metadata,
    }
    (args.output / "metadata_combined.json").write_text(
        json.dumps(combined_metadata, indent=2) + "\n"
    )
    write_plot(all_rows, args.output)
    print(f"Merged {len(all_rows)} rows into {args.output}")


if __name__ == "__main__":
    main()
