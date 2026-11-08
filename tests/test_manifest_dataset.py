import json
import pickle

import numpy as np
import torch

from samba_eeg.data import TUABManifestDataset


def test_tuab_manifest_dataset_slices_time_first_recording(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    signal = np.arange(300 * 20, dtype=np.float32).reshape(300, 20)
    with (source / "recording.pkl").open("wb") as stream:
        pickle.dump({"X": signal, "y": 1}, stream)
    row = {
        "source_recording": "recording.pkl",
        "start_sample": 50,
        "stop_sample": 150,
        "label": 1,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")

    dataset = TUABManifestDataset(manifest, source, channels=16, normalize=False)
    window = dataset[0]
    assert window.shape == (16, 100)
    assert window.dtype == torch.float32
    np.testing.assert_array_equal(window.numpy(), signal[50:150, :16].T)
