import pickle
from pathlib import Path

import numpy as np

from scripts.prepare_tuab_100s import prepare_split


def test_prepare_tuab_windows_do_not_cross_recordings(tmp_path: Path):
    source = tmp_path / "raw"
    output = tmp_path / "windows"
    for split in ("train", "val", "test"):
        directory = source / split
        directory.mkdir(parents=True)
        with (directory / f"subject_{split}.pkl").open("wb") as stream:
            pickle.dump({"X": np.ones((20, 270)), "y": 1}, stream)

    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = prepare_split(
            source / split,
            output / split,
            channels=16,
            sample_rate=10,
            duration_seconds=10,
            stride_seconds=10,
            resume=False,
            manifest_only=False,
        )
    assert summaries["train"]["windows"] == 2
    files = sorted((output / "train").glob("*_w*.pkl"))
    assert len(files) == 2
    with files[0].open("rb") as stream:
        sample = pickle.load(stream)
    assert sample["X"].shape == (16, 100)
    assert sample["source_recording"] == "subject_train.pkl"


def test_manifest_only_does_not_duplicate_eeg(tmp_path: Path):
    source = tmp_path / "raw" / "train"
    output = tmp_path / "windows" / "train"
    source.mkdir(parents=True)
    with (source / "recording.pkl").open("wb") as stream:
        pickle.dump({"X": np.ones((16, 250)), "y": 0}, stream)

    counts = prepare_split(
        source,
        output,
        channels=16,
        sample_rate=10,
        duration_seconds=10,
        stride_seconds=10,
        resume=False,
        manifest_only=True,
    )

    assert counts["windows"] == 2
    assert not list(output.glob("*_w*.pkl"))
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 2
