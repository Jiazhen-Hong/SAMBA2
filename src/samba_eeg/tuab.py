from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


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


def _channel_lookup(names: list[str]) -> dict[str, str]:
    return {name.strip().upper(): name for name in names}


class TUABEDFWindowDataset(Dataset[torch.Tensor]):
    """Load paper-aligned 16-channel TUAB windows directly from EDF recordings."""

    def __init__(self, manifest: str | Path, cache_recordings: int = 4, normalize: bool = True) -> None:
        self.manifest_path = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest_path}")
        self.cache_recordings = cache_recordings
        self.normalize = normalize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[row["source_edf"]].append(index)
        self.recording_groups = list(groups.values())

    def __len__(self) -> int:
        return len(self.rows)

    def _load_recording(self, path: str, target_rate: int) -> np.ndarray:
        if path in self._cache:
            signal = self._cache.pop(path)
            self._cache[path] = signal
            return signal

        import mne

        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        lookup = _channel_lookup(raw.ch_names)
        required = list(dict.fromkeys(channel for pair in BIPOLAR_PAIRS for channel in pair))
        missing = [channel for channel in required if channel not in lookup]
        if missing:
            raise ValueError(f"{path} is missing channels {missing}")
        actual_names = [lookup[channel] for channel in required]
        raw.pick(actual_names)
        if float(raw.info["sfreq"]) != float(target_rate):
            raw.resample(target_rate)
        referential = raw.get_data().astype(np.float32, copy=False)
        position = {name.strip().upper(): index for index, name in enumerate(raw.ch_names)}
        signal = np.stack(
            [referential[position[first]] - referential[position[second]] for first, second in BIPOLAR_PAIRS]
        ).astype(np.float32, copy=False)
        self._cache[path] = signal
        while len(self._cache) > self.cache_recordings:
            self._cache.popitem(last=False)
        return signal

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        signal = self._load_recording(row["source_edf"], int(row["target_sample_rate"]))
        start, stop = int(row["start_sample"]), int(row["stop_sample"])
        window = np.array(signal[:, start:stop], dtype=np.float32, copy=True)
        if window.shape != (16, stop - start):
            raise ValueError(f"invalid window {window.shape} from {row['source_edf']}")
        if self.normalize:
            scale = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True).astype(np.float32)
            window /= scale + 1e-8
        return torch.from_numpy(window)


class RecordingGroupedDistributedSampler(Sampler[int]):
    """Shuffle recordings while preserving EDF locality and equal DDP step counts."""

    def __init__(
        self,
        dataset: TUABEDFWindowDataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 3407,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self._indices: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        groups = [list(group) for group in self.dataset.recording_groups]
        generator = random.Random(self.seed + epoch)
        if self.shuffle:
            generator.shuffle(groups)
            for group in groups:
                generator.shuffle(group)
        rank_groups: list[list[int]] = [[] for _ in range(self.world_size)]
        sizes = [0] * self.world_size
        for group in groups:
            target = min(range(self.world_size), key=sizes.__getitem__)
            rank_groups[target].extend(group)
            sizes[target] += len(group)
        usable = min(sizes)
        usable -= usable % self.batch_size
        self._indices = rank_groups[self.rank][:usable]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)
