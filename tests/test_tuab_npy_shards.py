import json

import numpy as np
import pytest

from samba_eeg import TUABNpyShardDataset


def write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def test_materialized_tuab_dataset_reads_across_shards(tmp_path):
    split_root = tmp_path / "train"
    split_root.mkdir()
    first = np.zeros((2, 16, 12_800), dtype=np.float32)
    second = np.ones((1, 16, 12_800), dtype=np.float32)
    first[1] = 2
    np.save(split_root / "shard-00000.npy", first)
    np.save(split_root / "shard-00001.npy", second)
    write_json(
        tmp_path / "materialization.json",
        {
            "format": TUABNpyShardDataset.FORMAT,
            "dtype": "float32",
            "normalized": True,
        },
    )
    write_json(
        split_root / "index.json",
        {
            "windows": 3,
            "recording_groups": [[0, 1], [2]],
            "shards": [
                {"path": "train/shard-00000.npy", "start": 0, "stop": 2},
                {"path": "train/shard-00001.npy", "start": 2, "stop": 3},
            ],
        },
    )

    dataset = TUABNpyShardDataset(tmp_path, "train")
    assert len(dataset) == 3
    assert dataset.recording_groups == [[0, 1], [2]]
    assert np.array_equal(dataset[0].numpy(), first[0])
    assert np.array_equal(dataset[2].numpy(), second[0])
    assert np.array_equal(dataset[-1].numpy(), second[0])
    with pytest.raises(IndexError):
        dataset[3]


def test_materialized_tuab_dataset_rejects_incomplete_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete materialized dataset"):
        TUABNpyShardDataset(tmp_path, "train")
