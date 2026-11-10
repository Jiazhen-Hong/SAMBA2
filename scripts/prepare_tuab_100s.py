"""Materialize 100-second TUAB windows for multi-context pretraining.

Only recordings referenced by the recovered TUAB training split are read. A
fixed recording-group split supplies self-supervised train and validation data;
the downstream TUAB validation and test recordings are never used. Processing
uses bounded worker processes and can resume at recording boundaries.
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


SAMPLE_RATE = 128
WINDOW_SECONDS = 100
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
CHANNELS = 16
EXPECTED_SOURCE_TRAIN_WINDOWS = 32477
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


def channel_lookup(names: list[str]) -> dict[str, str]:
    return {name.strip().upper(): name for name in names}


def index_edfs(root: Path) -> dict[str, Path]:
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


def recording_header(path: Path) -> dict:
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    lookup = channel_lookup(raw.ch_names)
    required = list(dict.fromkeys(channel for pair in BIPOLAR_PAIRS for channel in pair))
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"{path} missing channels: {missing}")
    source_rate = float(raw.info["sfreq"])
    target_samples = math.ceil(raw.n_times * SAMPLE_RATE / source_rate)
    return {
        "stem": path.stem,
        "path": str(path.resolve()),
        "source_rate": source_rate,
        "windows": target_samples // WINDOW_SAMPLES,
    }


def process_recording(job: dict) -> tuple[str, np.ndarray]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import mne
    from scipy.signal import butter, resample_poly, sosfiltfilt

    raw = mne.io.read_raw_edf(job["path"], preload=False, verbose="ERROR")
    lookup = channel_lookup(raw.ch_names)
    required = list(dict.fromkeys(channel for pair in BIPOLAR_PAIRS for channel in pair))
    referential = raw.get_data(picks=[lookup[name] for name in required]).astype(
        np.float32, copy=False
    )
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(source_rate, SAMPLE_RATE):
        ratio = Fraction(SAMPLE_RATE, int(round(source_rate))).limit_denominator(1000)
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
    sos = butter(4, [0.5, 45.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    bipolar = sosfiltfilt(sos, bipolar, axis=-1).astype(np.float32, copy=False)

    count = min(int(job["windows"]), bipolar.shape[-1] // WINDOW_SAMPLES)
    windows = bipolar[:, : count * WINDOW_SAMPLES].reshape(
        CHANNELS, count, WINDOW_SAMPLES
    )
    windows = np.transpose(windows, (1, 0, 2)).copy()
    scale = np.quantile(np.abs(windows), 0.95, axis=-1).astype(np.float32)
    valid = np.isfinite(windows).all(axis=-1) & (scale > 1e-8)
    if not valid.all():
        raise ValueError(f"{job['path']} contains invalid or constant channels")
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
    planned = []
    for job in jobs:
        item = {**job, "start": offset, "stop": offset + int(job["windows"])}
        planned.append(item)
        offset = item["stop"]
    plan = {"split": split, "windows": offset, "jobs": planned}
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"existing plan differs: {plan_path}")
    atomic_json(plan, plan_path)

    expected_shape = (offset, CHANNELS, WINDOW_SAMPLES)
    if signals_path.exists() and mask_path.exists():
        array = np.load(signals_path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"invalid completed split: {signals_path}")
        return {"windows": offset, "recordings": len(jobs)}

    if partial_path.exists():
        output = np.lib.format.open_memmap(partial_path, mode="r+")
        if output.shape != expected_shape or output.dtype != np.float32:
            raise ValueError(f"invalid partial split: {partial_path}")
    else:
        output = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"complete_stems": []}
    )
    complete = set(progress["complete_stems"])
    pending = [job for job in planned if job["stem"] not in complete]
    lookup = {job["stem"]: job for job in planned}
    print(
        f"{split}: {len(complete)}/{len(jobs)} recordings complete; "
        f"{offset} windows total",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_recording, job): job for job in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            stem, windows = future.result()
            job = lookup[stem]
            expected = int(job["stop"]) - int(job["start"])
            if windows.shape != (expected, CHANNELS, WINDOW_SAMPLES):
                raise ValueError(
                    f"{stem} produced {windows.shape}; expected "
                    f"{(expected, CHANNELS, WINDOW_SAMPLES)}"
                )
            output[int(job["start"]) : int(job["stop"])] = windows
            complete.add(stem)
            if number % 10 == 0 or number == len(pending):
                output.flush()
                atomic_json({"complete_stems": sorted(complete)}, progress_path)
                print(f"{split}: {len(complete)}/{len(jobs)} recordings", flush=True)
    output.flush()
    del output
    os.replace(partial_path, signals_path)
    mask = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(offset, CHANNELS)
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
    parser.add_argument(
        "--output-root", type=Path, default=Path("/data/SAMBA-tmp/TUAB100")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("--workers must be between 1 and 24")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    references = sorted((args.split_reference_root / "train").glob("*.pkl"))
    if not references:
        raise FileNotFoundError("no recovered TUAB training references")
    edfs = index_edfs(args.edf_root)
    rows = []
    for reference in references:
        edf = edfs.get(reference.stem)
        if edf is None:
            raise FileNotFoundError(f"no EDF for {reference.stem}")
        rows.append((edf, reference_group(reference)))
    validation_groups = stable_validation_groups(
        {group for _, group in rows}, args.validation_fraction, args.seed
    )

    jobs = {"train": [], "val": []}
    for number, (edf, group) in enumerate(rows, start=1):
        header = recording_header(edf)
        header["group"] = group
        jobs["val" if group in validation_groups else "train"].append(header)
        if number % 250 == 0 or number == len(rows):
            print(f"indexed {number}/{len(rows)} recordings", flush=True)

    required = sum(job["windows"] for split in jobs.values() for job in split)
    if required != EXPECTED_SOURCE_TRAIN_WINDOWS:
        raise ValueError(
            f"found {required} windows, expected {EXPECTED_SOURCE_TRAIN_WINDOWS}; "
            "refusing to materialize a different TUAB corpus"
        )
    required *= CHANNELS * WINDOW_SAMPLES * np.dtype(np.float32).itemsize
    allocated = sum(
        path.stat().st_size
        for pattern in ("*_signals.npy", "*_signals.partial.npy")
        for path in args.output_root.glob(pattern)
    )
    remaining = max(0, required - allocated)
    free = shutil.disk_usage(args.output_root).free
    reserve = int(args.minimum_free_gib * 1024**3)
    print(
        f"required={required / 1024**3:.2f} GiB "
        f"remaining={remaining / 1024**3:.2f} GiB "
        f"free={free / 1024**3:.2f} GiB workers={args.workers}",
        flush=True,
    )
    if free < remaining + reserve:
        raise OSError("insufficient disk space for TUAB100 and requested reserve")

    splits = {
        split: materialize_split(split_jobs, args.output_root, split, args.workers)
        for split, split_jobs in jobs.items()
    }
    metadata = {
        "format": "samba-tuab100-16ch-v1",
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "expected_source_train_windows": EXPECTED_SOURCE_TRAIN_WINDOWS,
        "channels": CHANNELS,
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
