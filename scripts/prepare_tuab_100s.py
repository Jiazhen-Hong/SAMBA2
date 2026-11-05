"""Create recording-safe 100-second TUAB windows from processed recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")


def _as_channel_time(value: Any, channels: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    signal = np.asarray(value)
    if signal.ndim != 2:
        raise ValueError(f"expected a 2-D EEG array, got {signal.shape}")
    if signal.shape[0] < channels and signal.shape[1] >= channels:
        signal = signal.T
    if signal.shape[0] < channels:
        raise ValueError(f"recording has only {signal.shape[0]} channels")
    return np.asarray(signal[:channels], dtype=np.float32)


def _load_recording(path: Path, channels: int) -> tuple[np.ndarray, Any]:
    with path.open("rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f"expected a dict, got {type(record).__name__}")
    signal_key = next((key for key in ("X", "x", "data", "signal") if key in record), None)
    label_key = next((key for key in ("y", "Y", "label", "target") if key in record), None)
    if signal_key is None or label_key is None:
        raise ValueError(f"missing signal/label key; available keys: {sorted(record)}")
    return _as_channel_time(record[signal_key], channels), record[label_key]


def _atomic_pickle(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def prepare_split(
    input_directory: Path,
    output_directory: Path,
    channels: int,
    sample_rate: int,
    duration_seconds: int,
    stride_seconds: int,
    resume: bool,
) -> dict[str, int]:
    window = sample_rate * duration_seconds
    stride = sample_rate * stride_seconds
    source_files = sorted(input_directory.glob("*.pkl"))
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    counts = {"recordings": len(source_files), "windows": 0, "too_short": 0, "skipped": 0, "errors": 0}

    for source in source_files:
        try:
            signal, label = _load_recording(source, channels)
            if signal.shape[-1] < window:
                counts["too_short"] += 1
                continue
            window_index = 0
            for start in range(0, signal.shape[-1] - window + 1, stride):
                name = f"{source.stem}_w{window_index:04d}.pkl"
                destination = output_directory / name
                row = {
                    "file": name,
                    "source_recording": source.name,
                    "start_sample": start,
                    "stop_sample": start + window,
                    "label": int(np.asarray(label).reshape(-1)[0]),
                }
                if destination.exists() and resume:
                    counts["skipped"] += 1
                else:
                    _atomic_pickle(
                        {
                            "X": np.ascontiguousarray(signal[:, start : start + window]),
                            "y": row["label"],
                            "source_recording": source.name,
                            "start_sample": start,
                            "sample_rate": sample_rate,
                        },
                        destination,
                    )
                manifest_rows.append(row)
                counts["windows"] += 1
                window_index += 1
        except Exception as error:
            counts["errors"] += 1
            print(f"ERROR {source}: {error}")

    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--duration-seconds", type=int, default=100)
    parser.add_argument("--stride-seconds", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.input_root.resolve() == args.output_root.resolve():
        raise ValueError("input-root and output-root must be different")
    if args.stride_seconds <= 0 or args.duration_seconds <= 0:
        raise ValueError("duration and stride must be positive")

    summary: dict[str, Any] = {
        "input_root": str(args.input_root.resolve()),
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "duration_seconds": args.duration_seconds,
        "stride_seconds": args.stride_seconds,
        "splits": {},
    }
    for split in SPLITS:
        input_directory = args.input_root / split
        if not input_directory.is_dir():
            raise FileNotFoundError(f"missing split directory: {input_directory}")
        print(f"Preparing {split} from {input_directory}")
        counts = prepare_split(
            input_directory,
            args.output_root / split,
            args.channels,
            args.sample_rate,
            args.duration_seconds,
            args.stride_seconds,
            args.resume,
        )
        summary["splits"][split] = counts
        print(f"{split}: {counts}")

    manifest_hash = hashlib.sha256()
    for split in SPLITS:
        manifest_hash.update((args.output_root / split / "manifest.jsonl").read_bytes())
    summary["manifest_sha256"] = manifest_hash.hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
