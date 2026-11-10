"""Prepare true 10-second, 16-channel TUAB windows with bounded CPU use.

This bypasses NeuralBench task-cache construction. It reads only recordings in
the recovered TUAB training split, creates a fixed recording-grouped SSL
train/validation split, applies the paper's 16 bipolar derivations, resamples to
128 Hz, band-passes 0.5--45 Hz, and normalizes each window/channel by abs-q95.
The materialization is resumable at recording boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np


BIPOLAR_PAIRS = (
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalized_channel_map(channel_names: list[str]) -> dict[str, str]:
    return {name.strip().upper(): name for name in channel_names}


def index_edf_files(root: Path) -> dict[str, Path]:
    indexed = {}
    for path in root.rglob("*.edf"):
        if path.stem in indexed:
            raise ValueError(f"duplicate EDF stem: {path.stem}")
        indexed[path.stem] = path
    if not indexed:
        raise FileNotFoundError(f"no EDF files below {root}")
    return indexed


def reference_group(path: Path) -> str:
    with path.open("rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict) or "id" not in record:
        raise ValueError(f"unexpected split reference: {path}")
    return str(record["id"])


def stable_validation_groups(groups: set[str], fraction: float, seed: int) -> set[str]:
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    count = max(1, int(round(len(ordered) * fraction)))
    return set(ordered[:count])


def recording_header(path: Path, target_rate: int, window_samples: int) -> dict:
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    lookup = normalized_channel_map(raw.ch_names)
    required = list(dict.fromkeys(channel for pair in BIPOLAR_PAIRS for channel in pair))
    missing = [channel for channel in required if channel not in lookup]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    source_rate = float(raw.info["sfreq"])
    resampled = math.ceil(raw.n_times * target_rate / source_rate)
    return {
        "stem": path.stem,
        "path": str(path.resolve()),
        "source_rate": source_rate,
        "windows": resampled // window_samples,
    }


def process_recording(job: dict) -> tuple[str, np.ndarray]:
    # Prevent each worker from creating another large thread pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import mne
    from scipy.signal import butter, resample_poly, sosfiltfilt

    raw = mne.io.read_raw_edf(job["path"], preload=False, verbose="ERROR")
    lookup = normalized_channel_map(raw.ch_names)
    required = list(dict.fromkeys(channel for pair in BIPOLAR_PAIRS for channel in pair))
    actual = [lookup[channel] for channel in required]
    referential = raw.get_data(picks=actual).astype(np.float32, copy=False)
    source_rate = float(raw.info["sfreq"])
    target_rate = int(job["target_rate"])
    if not np.isclose(source_rate, target_rate):
        ratio = Fraction(target_rate, int(round(source_rate))).limit_denominator(1000)
        referential = resample_poly(
            referential, ratio.numerator, ratio.denominator, axis=-1
        ).astype(np.float32, copy=False)
    position = {name: index for index, name in enumerate(required)}
    bipolar = np.stack(
        [
            referential[position[first]] - referential[position[second]]
            for first, second in BIPOLAR_PAIRS
        ]
    ).astype(np.float32, copy=False)
    sos = butter(4, [0.5, 45.0], btype="bandpass", fs=target_rate, output="sos")
    bipolar = sosfiltfilt(sos, bipolar, axis=-1).astype(np.float32, copy=False)

    window_samples = int(job["window_samples"])
    count = min(int(job["windows"]), bipolar.shape[-1] // window_samples)
    windows = bipolar[:, : count * window_samples].reshape(16, count, window_samples)
    windows = np.transpose(windows, (1, 0, 2)).copy()
    scale = np.quantile(np.abs(windows), 0.95, axis=-1).astype(np.float32)
    valid = np.isfinite(windows).all(axis=-1) & (scale > 1e-8)
    if not valid.all():
        raise ValueError(f"{job['path']} contains invalid or constant bipolar channels")
    windows /= scale[..., None]
    return str(job["stem"]), windows.astype(np.float32, copy=False)


def materialize_split(
    jobs: list[dict], output_root: Path, split: str, workers: int
) -> dict:
    signals_path = output_root / f"{split}_signals.npy"
    mask_path = output_root / f"{split}_channel_mask.npy"
    partial_path = output_root / f"{split}_signals.partial.npy"
    plan_path = output_root / f"{split}_plan.json"
    progress_path = output_root / f"{split}_progress.json"

    offset = 0
    plan_jobs = []
    for job in jobs:
        item = {**job, "start": offset, "stop": offset + int(job["windows"])}
        plan_jobs.append(item)
        offset = item["stop"]
    plan = {"split": split, "windows": offset, "jobs": plan_jobs}
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"existing plan differs: {plan_path}")
    atomic_json(plan, plan_path)

    if signals_path.exists() and mask_path.exists():
        array = np.load(signals_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (offset, 16, 1280) or array.dtype != np.float32:
            raise ValueError(f"invalid completed split: {signals_path}")
        return {"windows": offset, "recordings": len(jobs)}

    if partial_path.exists():
        output = np.lib.format.open_memmap(partial_path, mode="r+")
    else:
        output = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(offset, 16, 1280)
        )
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"complete_stems": []}
    )
    complete = set(progress["complete_stems"])
    pending = [job for job in plan_jobs if job["stem"] not in complete]
    lookup = {job["stem"]: job for job in plan_jobs}
    print(
        f"{split}: {len(complete)}/{len(jobs)} recordings already complete, "
        f"{offset} windows total",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_recording, job): job for job in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            stem, windows = future.result()
            job = lookup[stem]
            expected = int(job["stop"]) - int(job["start"])
            if windows.shape != (expected, 16, 1280):
                raise ValueError(
                    f"{stem} produced {windows.shape}, expected {(expected, 16, 1280)}"
                )
            output[int(job["start"]) : int(job["stop"])] = windows
            complete.add(stem)
            if number % 10 == 0 or number == len(pending):
                output.flush()
                atomic_json({"complete_stems": sorted(complete)}, progress_path)
                print(
                    f"{split}: {len(complete)}/{len(jobs)} recordings",
                    flush=True,
                )
    output.flush()
    del output
    os.replace(partial_path, signals_path)
    mask = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(offset, 16)
    )
    mask[:] = True
    mask.flush()
    del mask
    progress_path.unlink(missing_ok=True)
    return {"windows": offset, "recordings": len(jobs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--split-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("--workers must be between 1 and 24")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    references = sorted((args.split_reference_root / "train").glob("*.pkl"))
    edf_index = index_edf_files(args.edf_root)
    reference_rows = []
    for reference in references:
        edf = edf_index.get(reference.stem)
        if edf is None:
            raise FileNotFoundError(f"no EDF for {reference.stem}")
        reference_rows.append((edf, reference_group(reference)))
    validation_groups = stable_validation_groups(
        {group for _, group in reference_rows}, args.validation_fraction, args.seed
    )

    jobs = {"train": [], "val": []}
    for number, (edf, group) in enumerate(reference_rows, start=1):
        header = recording_header(edf, 128, 1280)
        header.update({"target_rate": 128, "window_samples": 1280, "group": group})
        jobs["val" if group in validation_groups else "train"].append(header)
        if number % 250 == 0:
            print(f"indexed {number}/{len(reference_rows)} recordings", flush=True)

    required_bytes = sum(job["windows"] for split in jobs.values() for job in split)
    required_bytes *= 16 * 1280 * np.dtype(np.float32).itemsize
    allocated_bytes = sum(
        path.stat().st_size
        for pattern in ("*_signals.npy", "*_signals.partial.npy")
        for path in args.output_root.glob(pattern)
    )
    remaining_bytes = max(0, required_bytes - allocated_bytes)
    free_bytes = shutil.disk_usage(args.output_root).free
    reserve = int(args.minimum_free_gib * 1024**3)
    print(
        f"required={required_bytes / 1024**3:.2f} GiB "
        f"remaining={remaining_bytes / 1024**3:.2f} GiB "
        f"free={free_bytes / 1024**3:.2f} GiB workers={args.workers}",
        flush=True,
    )
    if free_bytes < remaining_bytes + reserve:
        raise OSError("insufficient disk space for TUAB10 plus requested reserve")

    splits = {
        split: materialize_split(split_jobs, args.output_root, split, args.workers)
        for split, split_jobs in jobs.items()
    }
    metadata = {
        "format": "samba-tuab10-16ch-v1",
        "sample_rate": 128,
        "duration_seconds": 10,
        "window_samples": 1280,
        "channels": 16,
        "bipolar_pairs": BIPOLAR_PAIRS,
        "filter_hz": [0.5, 45.0],
        "normalization": "per-window per-channel abs-q95",
        "source_split": "recovered TUAB train only",
        "ssl_validation": "stable recording-group split from source train",
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "splits": splits,
    }
    atomic_json(metadata, args.output_root / "materialization.json")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
