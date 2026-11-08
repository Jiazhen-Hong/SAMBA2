"""Materialize the verified TUAB 100-second manifest as resumable NPY shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from samba_eeg import TUABEDFWindowDataset, TUABNpyShardDataset


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_existing_shard(path: Path, count: int, window_shape: tuple[int, int]) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (count, *window_shape)
    if array.dtype != np.float32 or array.shape != expected:
        raise ValueError(f"existing shard {path} has {array.shape}/{array.dtype}; expected {expected}/float32")


def materialize_split(
    manifest_root: Path,
    output_root: Path,
    split: str,
    shard_size: int,
) -> dict:
    dataset = TUABEDFWindowDataset(
        manifest_root / f"{split}.jsonl",
        cache_recordings=1,
        normalize=True,
    )
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    window_shape = (16, 12_800)
    shards = []

    for shard_index, start in enumerate(range(0, len(dataset), shard_size)):
        stop = min(start + shard_size, len(dataset))
        count = stop - start
        name = f"shard-{shard_index:05d}.npy"
        final_path = split_root / name
        partial_path = split_root / f"shard-{shard_index:05d}.partial.npy"
        if final_path.exists():
            validate_existing_shard(final_path, count, window_shape)
            print(f"{split} {name}: already complete", flush=True)
        else:
            partial_path.unlink(missing_ok=True)
            array = np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(count, *window_shape),
            )
            for offset, sample_index in enumerate(range(start, stop)):
                window = dataset[sample_index].numpy()
                if window.shape != window_shape or not np.isfinite(window).all():
                    raise ValueError(f"invalid normalized window at {split}[{sample_index}]")
                array[offset] = window
            array.flush()
            del array
            os.replace(partial_path, final_path)
            print(f"{split} {name}: wrote {count} windows", flush=True)
        shards.append(
            {
                "path": str(final_path.relative_to(output_root)),
                "start": start,
                "stop": stop,
            }
        )

    index = {
        "split": split,
        "windows": len(dataset),
        "window_shape": list(window_shape),
        "shard_size": shard_size,
        "shards": shards,
        "recording_groups": dataset.recording_groups,
    }
    atomic_json(index, split_root / "index.json")
    return {"windows": len(dataset), "shards": len(shards)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    args = parser.parse_args()

    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    source_summary = json.loads((args.manifest_root / "summary.json").read_text())
    if source_summary.get("total_windows") != 39_810 or not source_summary.get("matches_paper_count"):
        raise ValueError("source manifest is not the verified 39,810-window SAMBA-T manifest")
    if any(item["errors"] for item in source_summary["splits"].values()):
        raise ValueError("source manifest contains unresolved recordings")

    args.output_root.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(source_summary["total_windows"]) * 16 * 12_800 * np.dtype(np.float32).itemsize
    completed_bytes = sum(path.stat().st_size for path in args.output_root.glob("*/*.npy"))
    remaining_bytes = max(0, expected_bytes - completed_bytes)
    free_bytes = shutil.disk_usage(args.output_root).free
    reserve_bytes = int(args.minimum_free_gib * 1024**3)
    print(
        f"expected={expected_bytes / 1024**3:.2f} GiB "
        f"completed={completed_bytes / 1024**3:.2f} GiB "
        f"free={free_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    if free_bytes < remaining_bytes + reserve_bytes:
        raise OSError(
            f"insufficient space: need {remaining_bytes / 1024**3:.2f} GiB plus "
            f"{args.minimum_free_gib:.2f} GiB reserve"
        )

    plan = {
        "format": TUABNpyShardDataset.FORMAT,
        "dtype": "float32",
        "normalized": True,
        "normalization": "per-window per-channel abs 0.95 quantile",
        "source_manifest_sha256": source_summary["manifest_sha256"],
        "source_total_windows": source_summary["total_windows"],
        "shard_size": args.shard_size,
    }
    plan_path = args.output_root / "materialization_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"existing materialization plan does not match: {plan_path}")
    atomic_json(plan, plan_path)

    splits = {
        split: materialize_split(args.manifest_root, args.output_root, split, args.shard_size)
        for split in ("train", "val", "test")
    }
    materialization = {
        **plan,
        "source_summary": source_summary,
        "splits": splits,
        "complete": True,
    }
    atomic_json(materialization, args.output_root / "materialization.json")
    print(json.dumps(materialization, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
