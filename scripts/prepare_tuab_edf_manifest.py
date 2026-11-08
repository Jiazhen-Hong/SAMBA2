"""Build the paper-aligned SAMBA-T 100-second manifest from TUAB EDF files."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


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


def normalized_channel_map(channel_names: list[str]) -> dict[str, int]:
    return {name.strip().upper(): index for index, name in enumerate(channel_names)}


def validate_channels(channel_names: list[str]) -> None:
    available = normalized_channel_map(channel_names)
    required = {channel for pair in BIPOLAR_PAIRS for channel in pair}
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"missing required referential channels: {missing}")


def window_count(n_times: int, source_rate: float, target_rate: int, window_samples: int) -> int:
    resampled_times = int(round(n_times * target_rate / source_rate))
    return max(0, resampled_times // window_samples)


def reference_metadata(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict) or not {"y", "id"}.issubset(record):
        raise ValueError(f"unexpected split reference format in {path}")
    return int(record["y"]), str(record["id"])


def index_edf_files(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in root.rglob("*.edf"):
        if path.stem in indexed:
            raise ValueError(f"duplicate EDF stem: {path.stem}")
        indexed[path.stem] = path
    if not indexed:
        raise FileNotFoundError(f"no EDF files found under {root}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--split-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--duration-seconds", type=int, default=100)
    parser.add_argument("--expected-windows", type=int, default=39810)
    args = parser.parse_args()

    import mne

    window_samples = args.sample_rate * args.duration_seconds
    edf_index = index_edf_files(args.edf_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "edf_root": str(args.edf_root.resolve()),
        "split_reference_root": str(args.split_reference_root.resolve()),
        "sample_rate": args.sample_rate,
        "duration_seconds": args.duration_seconds,
        "window_samples": window_samples,
        "expected_windows": args.expected_windows,
        "bipolar_pairs": BIPOLAR_PAIRS,
        "splits": {},
    }
    manifest_hash = hashlib.sha256()

    for split in ("train", "val", "test"):
        references = sorted((args.split_reference_root / split).glob("*.pkl"))
        rows = []
        errors = []
        for reference in references:
            try:
                label, recording_id = reference_metadata(reference)
                edf = edf_index.get(reference.stem)
                if edf is None:
                    raise FileNotFoundError(f"no EDF matching {reference.stem}")
                path_label = 1 if "abnormal" in edf.parts else 0 if "normal" in edf.parts else None
                if path_label is None or path_label != label:
                    raise ValueError(f"label mismatch: reference={label}, path={path_label}")
                raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
                validate_channels(raw.ch_names)
                source_rate = float(raw.info["sfreq"])
                count = window_count(raw.n_times, source_rate, args.sample_rate, window_samples)
                for window_index in range(count):
                    start = window_index * window_samples
                    rows.append(
                        {
                            "source_edf": str(edf.resolve()),
                            "source_recording": edf.name,
                            "recording_id": recording_id,
                            "label": label,
                            "source_sample_rate": source_rate,
                            "target_sample_rate": args.sample_rate,
                            "start_sample": start,
                            "stop_sample": start + window_samples,
                            "window_index": window_index,
                        }
                    )
            except Exception as error:
                errors.append({"reference": reference.name, "error": str(error)})

        manifest = args.output_root / f"{split}.jsonl"
        with manifest.open("w") as stream:
            for row in rows:
                line = json.dumps(row, sort_keys=True) + "\n"
                stream.write(line)
                manifest_hash.update(line.encode())
        summary["splits"][split] = {
            "reference_recordings": len(references),
            "resolved_recordings": len(references) - len(errors),
            "windows": len(rows),
            "errors": len(errors),
        }
        if errors:
            (args.output_root / f"{split}_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        print(split, summary["splits"][split], flush=True)

    summary["total_windows"] = sum(item["windows"] for item in summary["splits"].values())
    summary["manifest_sha256"] = manifest_hash.hexdigest()
    summary["matches_paper_count"] = summary["total_windows"] == args.expected_windows
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(item["errors"] for item in summary["splits"].values()):
        raise SystemExit("Manifest contains unresolved recordings; inspect *_errors.json")
    if not summary["matches_paper_count"]:
        raise SystemExit(
            f"Resolved {summary['total_windows']} windows, expected {args.expected_windows}; do not train yet"
        )


if __name__ == "__main__":
    main()
