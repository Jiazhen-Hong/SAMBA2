"""Read-only audit of recovered Emotiv and TUAB arrays for the SAMBA pipeline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np


EMOTIV_DATASETS = ("Alpha", "Attention", "Crowdsource", "DREAMER", "DriverDistraction", "STEW")
TUAB_SIGNAL_FILES = ("All_train_data.npy", "train_data.npy", "val_data.npy", "test_data.npy")
TUAB_LABEL_FILES = ("All_train_label.npy", "train_label.npy", "val_label.npy", "test_label.npy")


def load_array(path: Path, mmap: bool = True):
    if mmap:
        try:
            return np.load(path, mmap_mode="r", allow_pickle=False)
        except (TypeError, ValueError):
            pass
    return np.load(path, allow_pickle=True)


def sampled_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count, dtype=np.int64)


def signal_statistics(array: np.ndarray, max_windows: int = 256) -> dict[str, Any]:
    if array.ndim != 3:
        raise ValueError(f"expected [windows, channels, samples], got {array.shape}")
    indices = sampled_indices(len(array), max_windows)
    sample = np.asarray(array[indices], dtype=np.float32)
    finite = np.isfinite(sample)
    finite_values = sample[finite]
    if not finite_values.size:
        raise ValueError("sample contains no finite values")

    safe = np.where(finite, sample, np.nan)
    channel_mean = np.nanmean(safe, axis=-1)
    channel_std = np.nanstd(safe, axis=-1)
    near_zscore = (np.abs(channel_mean) <= 0.1) & (channel_std >= 0.8) & (channel_std <= 1.2)
    quantiles = np.quantile(finite_values, [0.01, 0.05, 0.5, 0.95, 0.99])
    std_quantiles = np.quantile(channel_std[np.isfinite(channel_std)], [0.05, 0.5, 0.95])

    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sampled_windows": int(len(indices)),
        "finite_fraction": float(finite.mean()),
        "value_quantiles": {
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
        },
        "channel_mean_abs_median": float(np.nanmedian(np.abs(channel_mean))),
        "channel_std_quantiles": {
            "q05": float(std_quantiles[0]),
            "q50": float(std_quantiles[1]),
            "q95": float(std_quantiles[2]),
        },
        "near_zscore_channel_fraction": float(near_zscore.mean()),
        "constant_channel_fraction": float((channel_std < 1e-8).mean()),
    }


def array_description(path: Path, max_windows: int) -> dict[str, Any]:
    loaded = load_array(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
    }
    if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
        loaded = loaded.item()
    if isinstance(loaded, dict):
        result["container"] = "npy_dict"
        result["keys"] = sorted(str(key) for key in loaded)
        arrays = {}
        for key, value in loaded.items():
            if not isinstance(value, np.ndarray):
                arrays[str(key)] = {"type": type(value).__name__}
            elif value.ndim == 3:
                arrays[str(key)] = signal_statistics(value, max_windows)
            else:
                arrays[str(key)] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        result["arrays"] = arrays
    elif isinstance(loaded, np.ndarray):
        result["container"] = "array"
        if loaded.ndim == 3:
            result["signal"] = signal_statistics(loaded, max_windows)
        else:
            result["shape"] = list(loaded.shape)
            result["dtype"] = str(loaded.dtype)
            if loaded.ndim == 1 and loaded.size:
                values = np.asarray(loaded[sampled_indices(len(loaded), min(max_windows * 4, len(loaded)))])
                result["sample_unique_count"] = int(len(np.unique(values)))
    else:
        result["container"] = type(loaded).__name__
    del loaded
    gc.collect()
    return result


def choose_emotiv_files(root: Path, dataset: str) -> list[Path]:
    directory = root / dataset
    if not directory.exists():
        return []
    x_file = directory / f"{dataset}_X.npy"
    if x_file.exists():
        files = [x_file]
        files.extend(path for path in (directory / f"{dataset}_y.npy", directory / f"{dataset}_id.npy") if path.exists())
        return files
    main = directory / f"{dataset}.npy"
    return [main] if main.exists() else []


def inventory_processed(root: Path) -> list[dict[str, Any]]:
    processed = root / "Processed"
    if not processed.exists():
        return []
    inventory = []
    candidates = list(processed.glob("*")) + list(processed.glob("*/*"))
    for path in sorted(candidates):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix,
                }
            )
    return inventory


def audit_emotiv(root: Path, max_windows: int) -> dict[str, Any]:
    datasets = {}
    for dataset in EMOTIV_DATASETS:
        files = choose_emotiv_files(root, dataset)
        datasets[dataset] = {
            "files": [array_description(path, max_windows) for path in files],
            "status": "found" if files else "missing",
        }
        print(f"Emotiv {dataset}: {datasets[dataset]['status']}", flush=True)
    return {
        "root": str(root.resolve()),
        "datasets": datasets,
        "processed_inventory": inventory_processed(root),
    }


def audit_tuab(root: Path, max_windows: int, assumed_rate: float) -> dict[str, Any]:
    signals = {}
    labels = {}
    for name in TUAB_SIGNAL_FILES:
        path = root / name
        if path.exists():
            signals[name] = array_description(path, max_windows)
            print(f"TUAB {name}: {signals[name]['signal']['shape']}", flush=True)
    for name in TUAB_LABEL_FILES:
        path = root / name
        if path.exists():
            labels[name] = array_description(path, max_windows)

    lengths = {
        name: item["signal"]["shape"][0]
        for name, item in signals.items()
    }
    split_arithmetic = None
    if {"All_train_data.npy", "train_data.npy", "val_data.npy"}.issubset(lengths):
        split_arithmetic = lengths["All_train_data.npy"] == lengths["train_data.npy"] + lengths["val_data.npy"]
    sample_counts = sorted({item["signal"]["shape"][-1] for item in signals.values()})
    return {
        "root": str(root.resolve()),
        "signals": signals,
        "labels": labels,
        "all_train_equals_train_plus_val": split_arithmetic,
        "sample_counts": sample_counts,
        "seconds_if_rate_assumption_is_correct": [count / assumed_rate for count in sample_counts],
        "assumed_rate_hz": assumed_rate,
        "rate_verified_from_array_metadata": False,
        "recording_or_subject_ids_present": any("id" in path.name.lower() for path in root.iterdir()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotiv-root", type=Path, required=True)
    parser.add_argument("--tuab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--assumed-sample-rate", type=float, default=128.0)
    args = parser.parse_args()
    if args.max_windows < 1:
        raise ValueError("--max-windows must be positive")

    report = {
        "read_only": True,
        "emotiv": audit_emotiv(args.emotiv_root, args.max_windows),
        "tuab": audit_tuab(args.tuab_root, args.max_windows, args.assumed_sample_rate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
