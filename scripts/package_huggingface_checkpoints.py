"""Package completed SAMBA runs as Hugging Face model repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASES = {
    "SAMBA-2s": {
        "source": Path("checkpoints/SAMBA-E/short-spatial-001/best"),
        "historical_track": "SAMBA-E",
        "pretraining_window_seconds": 2,
    },
    "SAMBA-100s": {
        "source": Path("checkpoints/SAMBA-T/paper-tuab100-001/best"),
        "historical_track": "SAMBA-T",
        "pretraining_window_seconds": 100,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_local_paths(value: Any) -> Any:
    """Remove machine-local paths while retaining file identity metadata."""
    if isinstance(value, dict):
        return {
            key: sanitize_local_paths(item)
            for key, item in value.items()
            if key not in {"path", "input_root", "output_root"}
        }
    if isinstance(value, list):
        return [sanitize_local_paths(item) for item in value]
    return value


def git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def package_checkpoint(
    repository: Path,
    release_name: str,
    source: Path,
    destination: Path,
    historical_track: str,
    window_seconds: int,
    overwrite: bool = False,
) -> None:
    source = source if source.is_absolute() else repository / source
    destination = destination if destination.is_absolute() else repository / destination
    required = ("config.json", "model.safetensors", "training_metadata.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} is missing: {', '.join(missing)}")

    model_card = destination / "README.md"
    if not model_card.is_file():
        raise FileNotFoundError(f"missing release model card: {model_card}")
    occupied = [name for name in required if (destination / name).exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            f"{destination} already contains {', '.join(occupied)}; use --overwrite"
        )

    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config["checkpoint_variant"] = release_name
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(source / "model.safetensors", destination / "model.safetensors")

    metadata = sanitize_local_paths(
        json.loads((source / "training_metadata.json").read_text())
    )
    try:
        source_checkpoint = str(source.relative_to(repository))
    except ValueError:
        source_checkpoint = str(source)
    metadata["release"] = {
        "name": release_name,
        "historical_track": historical_track,
        "pretraining_window_seconds": window_seconds,
        "source_checkpoint": source_checkpoint,
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repository),
        "source_training_metadata_sha256": sha256(
            source / "training_metadata.json"
        ),
    }
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    checksum_files = (
        "README.md",
        "config.json",
        "model.safetensors",
        "training_metadata.json",
    )
    checksums = "".join(
        f"{sha256(destination / name)}  {name}\n" for name in checksum_files
    )
    (destination / "SHA256SUMS").write_text(checksums)
    print(f"Packaged {release_name}: {destination}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--samba-2s-source",
        type=Path,
        default=RELEASES["SAMBA-2s"]["source"],
    )
    parser.add_argument(
        "--samba-100s-source", type=Path, default=RELEASES["SAMBA-100s"]["source"]
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing packaged config, weights, and metadata files.",
    )
    args = parser.parse_args()

    sources = {
        "SAMBA-2s": args.samba_2s_source,
        "SAMBA-100s": args.samba_100s_source,
    }
    for release_name, specification in RELEASES.items():
        package_checkpoint(
            repository=repository,
            release_name=release_name,
            source=sources[release_name],
            destination=args.output_root / release_name,
            historical_track=specification["historical_track"],
            window_seconds=specification["pretraining_window_seconds"],
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
