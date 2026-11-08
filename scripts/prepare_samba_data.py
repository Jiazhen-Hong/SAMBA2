"""Prepare the verified Emotiv subset for the rebuilt single SAMBA checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


EXISTING_SPLIT_SOURCES = ("Alpha", "Attention")
SUBJECT_SPLIT_SOURCES = ("Crowdsource", "DriverDistraction", "STEW")
ALL_SOURCES = EXISTING_SPLIT_SOURCES + SUBJECT_SPLIT_SOURCES


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def python_value(value):
    return value.item() if isinstance(value, np.generic) else value


def deterministic_subject_split(ids: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    if ids.ndim != 1:
        raise ValueError(f"subject IDs must be one-dimensional, got {ids.shape}")
    subjects = np.unique(ids)
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required for train, validation, and test splits")
    generator = np.random.default_rng(seed)
    subjects = subjects[generator.permutation(len(subjects))]
    test_count = max(1, int(round(len(subjects) * 0.2)))
    val_count = max(1, int(round(len(subjects) * 0.1)))
    if test_count + val_count >= len(subjects):
        test_count, val_count = 1, 1
    split_subjects = {
        "test": subjects[:test_count],
        "val": subjects[test_count : test_count + val_count],
        "train": subjects[test_count + val_count :],
    }
    indices = {split: np.flatnonzero(np.isin(ids, selected)) for split, selected in split_subjects.items()}
    serializable = {
        split: [python_value(value) for value in selected.tolist()]
        for split, selected in split_subjects.items()
    }
    sets = [set(values) for values in serializable.values()]
    if any(first & second for i, first in enumerate(sets) for second in sets[i + 1 :]):
        raise AssertionError("subject split overlap")
    return indices, serializable


def normalize_abs_q95(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 3 or signals.shape[1:] != (14, 256):
        raise ValueError(f"expected [N, 14, 256], got {signals.shape}")
    finite_channels = np.isfinite(signals).all(axis=-1)
    safe = np.where(np.isfinite(signals), signals, 0.0)
    scale = np.quantile(np.abs(safe), 0.95, axis=-1).astype(np.float32)
    valid = finite_channels & (scale > 1e-8)
    safe /= np.where(valid, scale, 1.0)[..., None]
    safe[~valid] = 0.0
    if not valid.any(axis=1).all():
        bad = int((~valid.any(axis=1)).sum())
        raise ValueError(f"{bad} windows contain no valid EEG channel")
    return safe.astype(np.float32, copy=False), valid


def validate_materialized(signals_path: Path, mask_path: Path, count: int) -> None:
    signals = np.load(signals_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if signals.shape != (count, 14, 256) or signals.dtype != np.float32:
        raise ValueError(f"invalid existing signals {signals_path}: {signals.shape}/{signals.dtype}")
    if mask.shape != (count, 14) or mask.dtype != np.bool_:
        raise ValueError(f"invalid existing mask {mask_path}: {mask.shape}/{mask.dtype}")


def materialize_split(
    source: str,
    split: str,
    array: np.ndarray,
    output_root: Path,
    indices: np.ndarray | None,
    chunk_size: int,
) -> dict[str, Any]:
    source_root = output_root / source
    source_root.mkdir(parents=True, exist_ok=True)
    count = len(array) if indices is None else len(indices)
    signals_path = source_root / f"{split}_signals.npy"
    mask_path = source_root / f"{split}_channel_mask.npy"
    partial_signals = source_root / f"{split}_signals.partial.npy"
    partial_mask = source_root / f"{split}_channel_mask.partial.npy"

    if signals_path.exists() != mask_path.exists():
        signals_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
    if signals_path.exists():
        validate_materialized(signals_path, mask_path, count)
        print(f"{source}/{split}: already complete ({count} windows)", flush=True)
    else:
        partial_signals.unlink(missing_ok=True)
        partial_mask.unlink(missing_ok=True)
        signals_out = np.lib.format.open_memmap(
            partial_signals, mode="w+", dtype=np.float32, shape=(count, 14, 256)
        )
        mask_out = np.lib.format.open_memmap(
            partial_mask, mode="w+", dtype=np.bool_, shape=(count, 14)
        )
        invalid_channels = 0
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            selection = slice(start, stop) if indices is None else indices[start:stop]
            normalized, valid = normalize_abs_q95(array[selection])
            signals_out[start:stop] = normalized
            mask_out[start:stop] = valid
            invalid_channels += int((~valid).sum())
        signals_out.flush()
        mask_out.flush()
        del signals_out, mask_out
        os.replace(partial_signals, signals_path)
        os.replace(partial_mask, mask_path)
        print(f"{source}/{split}: wrote {count} windows", flush=True)
        return {"windows": count, "invalid_channels": invalid_channels}

    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    return {"windows": count, "invalid_channels": int((~mask).sum())}


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_existing_split_source(
    input_root: Path,
    output_root: Path,
    source: str,
    chunk_size: int,
) -> dict[str, Any]:
    path = input_root / source / f"{source}.npy"
    container = np.load(path, allow_pickle=True).item()
    splits = {}
    for split in ("train", "val", "test"):
        array = container[f"{split}_data"]
        splits[split] = materialize_split(source, split, array, output_root, None, chunk_size)
        subject_key = f"{split}_subjects"
        if subject_key in container:
            splits[split]["source_subjects"] = [python_value(value) for value in container[subject_key]]
    del container
    gc.collect()
    return {
        "source_file": file_identity(path),
        "split_method": "recovered_subject_split",
        "splits": splits,
    }


def prepare_subject_split_source(
    input_root: Path,
    output_root: Path,
    source: str,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    directory = input_root / source
    signals_path = directory / f"{source}_X.npy"
    ids_path = directory / f"{source}_id.npy"
    signals = np.load(signals_path, mmap_mode="r", allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=True)
    if len(signals) != len(ids):
        raise ValueError(f"{source} signals and subject IDs have different lengths")
    indices, subjects = deterministic_subject_split(ids, seed)
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = materialize_split(
            source, split, signals, output_root, indices[split], chunk_size
        )
        splits[split]["subjects"] = subjects[split]
    del signals, ids
    gc.collect()
    return {
        "source_files": [file_identity(signals_path), file_identity(ids_path)],
        "split_method": "deterministic_subject_split_70_10_20",
        "split_seed": seed,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_files = []
    for source in EXISTING_SPLIT_SOURCES:
        source_files.append(file_identity(args.input_root / source / f"{source}.npy"))
    for source in SUBJECT_SPLIT_SOURCES:
        directory = args.input_root / source
        source_files.extend(
            [
                file_identity(directory / f"{source}_X.npy"),
                file_identity(directory / f"{source}_id.npy"),
            ]
        )
    plan = {
        "format": "samba-emotiv-abs-q95-v1",
        "input_root": str(args.input_root.resolve()),
        "seed": args.seed,
        "normalization": "per_window_per_channel_abs_q95",
        "sources": list(ALL_SOURCES),
        "source_files": source_files,
    }
    plan_path = args.output_root / "materialization_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"existing materialization plan does not match: {plan_path}")
    atomic_json(plan, plan_path)

    sources = {}
    for source in EXISTING_SPLIT_SOURCES:
        sources[source] = prepare_existing_split_source(
            args.input_root, args.output_root, source, args.chunk_size
        )
    for offset, source in enumerate(SUBJECT_SPLIT_SOURCES):
        sources[source] = prepare_subject_split_source(
            args.input_root, args.output_root, source, args.seed + offset, args.chunk_size
        )

    metadata = {
        "format": "samba-emotiv-abs-q95-v1",
        "complete": True,
        "normalization": "per_window_per_channel_abs_q95",
        "constant_channel_policy": "mask",
        "excluded_sources": {"DREAMER": "systematic constant channels in recovered file"},
        "input_root": str(args.input_root.resolve()),
        "plan": plan,
        "sources": sources,
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(metadata, args.output_root / "materialization.json")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
