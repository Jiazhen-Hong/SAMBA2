from __future__ import annotations

import json
import random
from bisect import bisect_right
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


class TUABNpyShardDataset(Dataset[torch.Tensor]):
    """Read pre-normalized TUAB windows from resumable float32 NPY shards."""

    FORMAT = "samba-tuab-npy-shards-v1"

    def __init__(self, root: str | Path, split: str, cache_shards: int = 8) -> None:
        self.root = Path(root)
        self.split = split
        metadata_path = self.root / "materialization.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"incomplete materialized dataset: {metadata_path} is missing; "
                "rerun scripts/materialize_tuab_100s.py"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != self.FORMAT:
            raise ValueError(f"unsupported TUAB materialization format in {metadata_path}")
        if metadata.get("dtype") != "float32" or metadata.get("normalized") is not True:
            raise ValueError("paper reproduction requires normalized float32 materialized windows")

        index_path = self.root / split / "index.json"
        index = json.loads(index_path.read_text())
        self.shards = index["shards"]
        self._stops = [int(shard["stop"]) for shard in self.shards]
        self.recording_groups = [[int(item) for item in group] for group in index["recording_groups"]]
        self.cache_shards = cache_shards
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._length = int(index["windows"])
        if not self.shards or self._stops[-1] != self._length:
            raise ValueError(f"invalid shard index: {index_path}")

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, relative_path: str) -> np.ndarray:
        if relative_path in self._cache:
            array = self._cache.pop(relative_path)
            self._cache[relative_path] = array
            return array
        path = self.root / relative_path
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[1:] != (16, 12_800):
            raise ValueError(f"invalid materialized shard {path}: {array.shape}/{array.dtype}")
        self._cache[relative_path] = array
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return array

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect_right(self._stops, index)
        shard = self.shards[shard_index]
        array = self._load_shard(shard["path"])
        local_index = index - int(shard["start"])
        # Memmaps are read-only. Copying makes the returned tensor safe for PyTorch.
        window = np.array(array[local_index], dtype=np.float32, copy=True)
        return torch.from_numpy(window)


class RecordingGroupedDistributedSampler(Sampler[int]):
    """Shuffle recordings while preserving EDF locality and equal DDP step counts."""

    def __init__(
        self,
        dataset: TUABEDFWindowDataset | TUABNpyShardDataset,
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
