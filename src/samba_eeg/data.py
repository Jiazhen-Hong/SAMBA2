from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def as_channel_time(value: Any, channels: int) -> np.ndarray:
    """Convert a recording to [channels, time] without silently guessing badly."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    signal = np.asarray(value)
    if signal.ndim != 2:
        raise ValueError(f"expected a 2-D EEG array, got {signal.shape}")
    first, second = signal.shape
    first_is_channels = channels <= first <= 256 and second > first
    second_is_channels = channels <= second <= 256 and first > second
    if second_is_channels and not first_is_channels:
        signal = signal.T
    elif not first_is_channels and not second_is_channels:
        raise ValueError(f"cannot infer channel axis for shape {signal.shape}")
    if signal.shape[0] < channels:
        raise ValueError(f"recording has only {signal.shape[0]} channels")
    return np.asarray(signal[:channels], dtype=np.float32)


class TUABManifestDataset(Dataset[torch.Tensor]):
    """Read indexed 100-second windows directly from TUAB source recordings."""

    def __init__(
        self,
        manifest: str | Path,
        source_root: str | Path,
        channels: int = 16,
        normalize: bool = True,
        cache_recordings: int = 2,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.source_root = Path(source_root)
        self.channels = channels
        self.normalize = normalize
        self.cache_recordings = cache_recordings
        self.rows = [json.loads(line) for line in self.manifest_path.read_text().splitlines() if line.strip()]
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _recording(self, name: str) -> tuple[np.ndarray, int]:
        if name in self._cache:
            value = self._cache.pop(name)
            self._cache[name] = value
            return value
        with (self.source_root / name).open("rb") as stream:
            record = pickle.load(stream)
        if not isinstance(record, dict):
            raise ValueError(f"{name} is not a dictionary")
        signal_key = next((key for key in ("X", "x", "data", "signal") if key in record), None)
        label_key = next((key for key in ("y", "Y", "label", "target") if key in record), None)
        if signal_key is None or label_key is None:
            raise ValueError(f"{name} lacks signal or label keys")
        value = as_channel_time(record[signal_key], self.channels), int(np.asarray(record[label_key]).reshape(-1)[0])
        self._cache[name] = value
        while len(self._cache) > self.cache_recordings:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        signal, label = self._recording(row["source_recording"])
        start, stop = int(row["start_sample"]), int(row["stop_sample"])
        window = np.array(signal[:, start:stop], dtype=np.float32, copy=True)
        if window.shape[-1] != stop - start:
            raise ValueError(f"short window in {row['source_recording']}: {window.shape}")
        if int(row["label"]) != label:
            raise ValueError(f"label mismatch in {row['source_recording']}")
        if self.normalize:
            scale = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
            window /= scale.astype(np.float32) + 1e-8
        return torch.from_numpy(window)
