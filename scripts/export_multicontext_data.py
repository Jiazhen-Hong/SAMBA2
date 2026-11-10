"""Export train/validation EEG windows into SAMBA's multi-context format.

NeuralBench supplies subject-disjoint splits, preprocessing, channel positions,
and access to many heterogeneous datasets.  This exporter keeps only train and
validation windows, applies SAMBA's per-window/channel absolute-q95 scaling, and
writes memory-mapped arrays for fast pretraining.  Test windows are never read.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml


FORMAT = "samba-pretrain-manifest-v2"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_abs_q95(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 3:
        raise ValueError(f"signals must be [B, C, T], got {signals.shape}")
    finite = np.isfinite(signals).all(axis=-1)
    safe = np.where(np.isfinite(signals), signals, 0.0)
    scale = np.quantile(np.abs(safe), 0.95, axis=-1).astype(np.float32)
    valid = finite & (scale > 1e-8)
    safe /= np.where(valid, scale, 1.0)[..., None]
    safe[~valid] = 0.0
    if not valid.any(axis=1).all():
        raise ValueError("a window contains no finite, nonconstant EEG channel")
    return safe.astype(np.float32, copy=False), valid


def batch_mapping(batch) -> Mapping:
    if isinstance(batch, Mapping):
        return batch
    if hasattr(batch, "_asdict"):
        return batch._asdict()
    if hasattr(batch, "__dict__"):
        return vars(batch)
    raise TypeError(f"unsupported NeuralBench batch type: {type(batch)!r}")


def tensor_array(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_neural_batch(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = batch_mapping(batch)
    signal = None
    for key in ("eeg", "neuro", "signals", "x"):
        candidate = tensor_array(values.get(key))
        if candidate is not None and candidate.ndim == 3:
            signal = candidate
            break
    if signal is None:
        candidates = []
        for key, value in values.items():
            candidate = tensor_array(value)
            if candidate is not None and candidate.ndim == 3 and candidate.shape[-1] > 3:
                candidates.append((key, candidate))
        if len(candidates) != 1:
            shapes = {
                key: getattr(value, "shape", None) for key, value in values.items()
            }
            raise ValueError(f"cannot identify EEG tensor in batch: {shapes}")
        signal = candidates[0][1]

    positions = None
    for key in ("channel_positions", "positions", "input_positions"):
        candidate = tensor_array(values.get(key))
        if candidate is not None:
            positions = candidate
            break
    if positions is None:
        raise ValueError(f"batch has no channel positions; keys={list(values)}")
    if positions.ndim == 2:
        positions = np.broadcast_to(positions[None], (len(signal), *positions.shape))
    if positions.shape != (signal.shape[0], signal.shape[1], 3):
        raise ValueError(
            f"positions {positions.shape} do not match EEG batch {signal.shape}"
        )

    supplied_mask = None
    for key in ("channel_mask", "channels_mask", "mask"):
        candidate = tensor_array(values.get(key))
        if candidate is not None and candidate.shape == signal.shape[:2]:
            supplied_mask = candidate.astype(np.bool_)
            break
    normalized, data_mask = normalize_abs_q95(signal)
    position_mask = np.isfinite(positions).all(axis=-1)
    mask = data_mask & position_mask
    if supplied_mask is not None:
        mask &= supplied_mask
    normalized[~mask] = 0.0
    positions = np.nan_to_num(positions).astype(np.float32, copy=False)
    return normalized, mask, positions


def complete_paths(root: Path, split: str) -> dict[str, str] | None:
    paths = {
        "signals": root / f"{split}_signals.npy",
        "channel_mask": root / f"{split}_channel_mask.npy",
        "channel_positions": root / f"{split}_channel_positions.npy",
    }
    if all(path.exists() for path in paths.values()):
        return {key: str(path.resolve()) for key, path in paths.items()}
    return None


def export_loader(loader, root: Path, split: str, max_hours: float) -> dict[str, str]:
    existing = complete_paths(root, split)
    if existing is not None:
        print(f"{root.name}/{split}: already complete", flush=True)
        return existing

    iterator = iter(loader)
    first = next(iterator)
    first_signals, first_mask, first_positions = extract_neural_batch(first)
    window_samples = first_signals.shape[-1]
    sample_rate = 128
    maximum = max(1, int(max_hours * 3600 * sample_rate // window_samples))
    count = min(len(loader.dataset), maximum)
    root.mkdir(parents=True, exist_ok=True)
    partial = {
        "signals": root / f"{split}_signals.partial.npy",
        "channel_mask": root / f"{split}_channel_mask.partial.npy",
        "channel_positions": root / f"{split}_channel_positions.partial.npy",
    }
    for path in partial.values():
        path.unlink(missing_ok=True)
    outputs = {
        "signals": np.lib.format.open_memmap(
            partial["signals"],
            mode="w+",
            dtype=np.float32,
            shape=(count, first_signals.shape[1], window_samples),
        ),
        "channel_mask": np.lib.format.open_memmap(
            partial["channel_mask"],
            mode="w+",
            dtype=np.bool_,
            shape=(count, first_signals.shape[1]),
        ),
        "channel_positions": np.lib.format.open_memmap(
            partial["channel_positions"],
            mode="w+",
            dtype=np.float32,
            shape=(count, first_signals.shape[1], 3),
        ),
    }

    written = 0

    def write(signals, mask, positions) -> None:
        nonlocal written
        take = min(len(signals), count - written)
        if take <= 0:
            return
        outputs["signals"][written : written + take] = signals[:take]
        outputs["channel_mask"][written : written + take] = mask[:take]
        outputs["channel_positions"][written : written + take] = positions[:take]
        written += take

    write(first_signals, first_mask, first_positions)
    for batch in iterator:
        if written >= count:
            break
        write(*extract_neural_batch(batch))
        if written % 10000 == 0:
            print(f"{root.name}/{split}: {written}/{count}", flush=True)
    if written != count:
        raise RuntimeError(f"loader ended after {written} of {count} requested windows")

    for array in outputs.values():
        array.flush()
    del outputs
    final = {
        key: root / f"{split}_{key}.npy"
        for key in ("signals", "channel_mask", "channel_positions")
    }
    for key in final:
        os.replace(partial[key], final[key])
    print(f"{root.name}/{split}: wrote {count} windows", flush=True)
    return {key: str(path.resolve()) for key, path in final.items()}


def add_prepared_source(source: dict) -> dict:
    root = Path(source["root"]).expanduser()
    splits = {}
    for split in ("train", "val"):
        signals = root / f"{split}_signals.npy"
        mask = root / f"{split}_channel_mask.npy"
        if not signals.exists() or not mask.exists():
            raise FileNotFoundError(f"incomplete prepared source: {root}/{split}")
        splits[split] = {
            "signals": str(signals.resolve()),
            "channel_mask": str(mask.resolve()),
        }
    probe = np.load(splits["train"]["signals"], mmap_mode="r", allow_pickle=False)
    return {
        "name": source["name"],
        "weight": float(source["weight"]),
        "provider": "prepared_npy",
        "task_family": source["task_family"],
        "device_family": source["device_family"],
        "window_samples": int(probe.shape[-1]),
        "static_positions": source["static_positions"],
        "splits": splits,
    }


def add_neuralbench_source(
    source: dict, output_root: Path, workers: int, check_only: bool = False
) -> dict | None:
    from neuralbench.data import get_default_dataloaders
    import torch
    from torch.utils.data import DataLoader, Subset

    duration = float(source["duration_seconds"])
    stride = source.get("stride_seconds")
    overrides = {
        "duration": duration,
        "stride": float(stride) if stride is not None else None,
        "batch_size": int(source.get("export_batch_size", 32)),
        "val_batch_size": int(source.get("export_batch_size", 32)),
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
        "seed": int(source.get("seed", 3407)),
        "neuro.frequency": 128.0,
        "neuro.filter": [0.5, 45.0],
        "neuro.infra.cluster": None,
    }
    loaders = get_default_dataloaders(
        "eeg",
        source["task"],
        dataset=source.get("dataset"),
        **overrides,
    )
    # Do not expose NeuralBench validation or test examples to self-supervised
    # pretraining. Build a fixed, window-disjoint SSL validation subset solely
    # from NeuralBench's training split.
    base_loader = loaders["train"]
    total = len(base_loader.dataset)
    samples = int(round(duration * 128))
    requested_train = max(
        1, int(float(source["max_train_hours"]) * 3600 * 128 // samples)
    )
    requested_val = max(
        1, int(float(source["max_val_hours"]) * 3600 * 128 // samples)
    )
    val_count = min(requested_val, max(1, total // 10))
    train_count = min(requested_train, total - val_count)
    if train_count < 1:
        raise ValueError(f"{source['name']} has too few NeuralBench training windows")
    generator = np.random.default_rng(int(source.get("seed", 3407)))
    selected = generator.choice(total, size=train_count + val_count, replace=False)

    def subset_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        torch_generator = torch.Generator().manual_seed(
            int(source.get("seed", 3407)) + (0 if shuffle else 1000)
        )
        return DataLoader(
            Subset(base_loader.dataset, indices.tolist()),
            collate_fn=base_loader.collate_fn,
            batch_size=int(source.get("export_batch_size", 32)),
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
            generator=torch_generator,
        )

    ssl_loaders = {
        "train": subset_loader(selected[:train_count], True),
        "val": subset_loader(selected[train_count:], False),
    }
    if check_only:
        signals, mask, positions = extract_neural_batch(next(iter(ssl_loaders["train"])))
        print(
            f"{source['name']}: signals={signals.shape} mask={mask.shape} "
            f"positions={positions.shape} train_pool={total}",
            flush=True,
        )
        return None
    source_root = output_root / source["name"].replace("/", "__")
    splits = {
        "train": export_loader(
            ssl_loaders["train"],
            source_root,
            "train",
            float(source["max_train_hours"]),
        ),
        "val": export_loader(
            ssl_loaders["val"], source_root, "val", float(source["max_val_hours"])
        ),
    }
    probe = np.load(splits["train"]["signals"], mmap_mode="r", allow_pickle=False)
    return {
        "name": source["name"],
        "weight": float(source["weight"]),
        "provider": "neuralbench",
        "neuralbench_task": source["task"],
        "neuralbench_dataset": source.get("dataset"),
        "task_family": source["task_family"],
        "device_family": source["device_family"],
        "window_samples": int(probe.shape[-1]),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_sources.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="With --check-only, inspect only these source names.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if int(config["sampling_rate"]) != 128:
        raise ValueError("the current exporter requires a 128-Hz canonical rate")
    sources = []
    for source in config["sources"]:
        provider = source["provider"]
        if provider == "prepared_npy":
            prepared = add_prepared_source(source)
            if args.check_only:
                shape = np.load(
                    prepared["splits"]["train"]["signals"],
                    mmap_mode="r",
                    allow_pickle=False,
                ).shape
                print(f"{source['name']}: signals={shape} prepared", flush=True)
            else:
                sources.append(prepared)
        elif provider == "neuralbench":
            if args.only and source["name"] not in args.only:
                continue
            exported = add_neuralbench_source(
                source, args.output_root, args.workers, check_only=args.check_only
            )
            if exported is not None:
                sources.append(exported)
        else:
            raise ValueError(f"unsupported provider {provider!r}")
    if args.check_only:
        return
    if args.only:
        raise ValueError("--only is supported only with --check-only")
    weights = sum(float(source["weight"]) for source in sources)
    if not np.isclose(weights, 1.0):
        raise ValueError(
            f"selected source weights sum to {weights}; export the complete configured set"
        )
    manifest = {
        "format": FORMAT,
        "sampling_rate": 128,
        "normalization": "per_window_per_channel_abs_q95",
        "split_policy": "NeuralBench train split only; fixed window-disjoint SSL train/val; downstream val/test excluded",
        "sources": sources,
        "limitations": [
            "NeuralBench validation and test splits are excluded from self-supervised pretraining.",
            "Dataset overlap must be disclosed when evaluating an in-domain downstream task.",
        ],
    }
    atomic_json(manifest, args.manifest_output)
    print(f"wrote {args.manifest_output}", flush=True)


if __name__ == "__main__":
    main()
