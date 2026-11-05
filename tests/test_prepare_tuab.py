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
        )
    assert summaries["train"]["windows"] == 2
    files = sorted((output / "train").glob("*_w*.pkl"))
    assert len(files) == 2
    with files[0].open("rb") as stream:
        sample = pickle.load(stream)
    assert sample["X"].shape == (16, 100)
    assert sample["source_recording"] == "subject_train.pkl"
