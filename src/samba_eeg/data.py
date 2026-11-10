from __future__ import annotations

import bisect
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class NpyWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy memory-mapped EEG windows with an optional valid-channel mask."""

    def __init__(self, signals: str | Path, channel_mask: str | Path | None = None) -> None:
        self.signals_path = Path(signals)
        self.mask_path = Path(channel_mask) if channel_mask is not None else None
        probe = np.load(self.signals_path, mmap_mode="r", allow_pickle=False)
        if probe.ndim != 3:
            raise ValueError(f"signals must be [N, C, T], got {probe.shape}")
        if probe.dtype != np.float32:
            raise ValueError(f"signals must be float32, got {probe.dtype}")
        self.shape = tuple(int(value) for value in probe.shape)
        del probe
        if self.mask_path is not None:
            mask = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)
            if mask.dtype != np.bool_ or mask.shape != self.shape[:2]:
                raise ValueError(f"invalid channel mask {mask.shape}/{mask.dtype} for {self.shape}")
            del mask
        self._signals = None
        self._mask = None

    def __len__(self) -> int:
        return self.shape[0]

    def _open(self) -> None:
        if self._signals is None:
            self._signals = np.load(self.signals_path, mmap_mode="r", allow_pickle=False)
        if self.mask_path is not None and self._mask is None:
            self._mask = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        signal = np.array(self._signals[index], dtype=np.float32, copy=True)
        if self._mask is None:
            mask = np.ones(self.shape[1], dtype=np.bool_)
        else:
            mask = np.array(self._mask[index], dtype=np.bool_, copy=True)
        return torch.from_numpy(signal), torch.from_numpy(mask)

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Fetch a complete DataLoader batch with one memory-map operation.

        PyTorch uses this method when automatic batching is enabled. It keeps the
        zero-worker path efficient and avoids multiprocessing shared-memory use.
        """
        self._open()
        signals = np.array(self._signals[indices], dtype=np.float32, copy=True)
        if self._mask is None:
            masks = np.ones((len(indices), self.shape[1]), dtype=np.bool_)
        else:
            masks = np.array(self._mask[indices], dtype=np.bool_, copy=True)
        return [
            (torch.from_numpy(signals[index]), torch.from_numpy(masks[index]))
            for index in range(len(indices))
        ]


class PositionedNpyWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Memory-mapped windows with either static or per-window EEG positions.

    This is the canonical SAMBA v2 pretraining format.  Every dataset instance
    has one temporal length, so a DataLoader batch never needs temporal padding.
    Different instances may use different lengths and channel counts and can be
    mixed safely through gradient accumulation.
    """

    def __init__(
        self,
        signals: str | Path,
        *,
        static_positions: np.ndarray | torch.Tensor | None = None,
        positions: str | Path | None = None,
        channel_mask: str | Path | None = None,
    ) -> None:
        if (static_positions is None) == (positions is None):
            raise ValueError("provide exactly one of static_positions or positions")
        self.signals_path = Path(signals)
        self.positions_path = Path(positions) if positions is not None else None
        self.mask_path = Path(channel_mask) if channel_mask is not None else None

        probe = np.load(self.signals_path, mmap_mode="r", allow_pickle=False)
        if probe.ndim != 3 or probe.dtype != np.float32:
            raise ValueError(
                f"signals must be float32 [N, C, T], got {probe.shape}/{probe.dtype}"
            )
        self.shape = tuple(int(value) for value in probe.shape)
        del probe

        if static_positions is not None:
            static = np.asarray(static_positions, dtype=np.float32)
            if static.shape != (self.shape[1], 3):
                raise ValueError(
                    f"static positions must be [C, 3], got {static.shape} for {self.shape}"
                )
            self._static_positions = torch.from_numpy(static.copy())
        else:
            self._static_positions = None
            position_probe = np.load(
                self.positions_path, mmap_mode="r", allow_pickle=False
            )
            if position_probe.dtype != np.float32 or position_probe.shape != (
                self.shape[0],
                self.shape[1],
                3,
            ):
                raise ValueError(
                    "per-window positions must be float32 [N, C, 3], got "
                    f"{position_probe.shape}/{position_probe.dtype}"
                )
            del position_probe

        if self.mask_path is not None:
            mask_probe = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)
            if mask_probe.dtype != np.bool_ or mask_probe.shape != self.shape[:2]:
                raise ValueError(
                    f"invalid channel mask {mask_probe.shape}/{mask_probe.dtype} "
                    f"for {self.shape}"
                )
            del mask_probe
        self._signals = None
        self._positions = None
        self._mask = None

    @property
    def channels(self) -> int:
        return self.shape[1]

    @property
    def window_samples(self) -> int:
        return self.shape[2]

    def __len__(self) -> int:
        return self.shape[0]

    def _open(self) -> None:
        if self._signals is None:
            self._signals = np.load(
                self.signals_path, mmap_mode="r", allow_pickle=False
            )
        if self.positions_path is not None and self._positions is None:
            self._positions = np.load(
                self.positions_path, mmap_mode="r", allow_pickle=False
            )
        if self.mask_path is not None and self._mask is None:
            self._mask = np.load(self.mask_path, mmap_mode="r", allow_pickle=False)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open()
        signal = torch.from_numpy(
            np.array(self._signals[index], dtype=np.float32, copy=True)
        )
        if self._mask is None:
            mask = torch.ones(self.channels, dtype=torch.bool)
        else:
            mask = torch.from_numpy(
                np.array(self._mask[index], dtype=np.bool_, copy=True)
            )
        if self._static_positions is not None:
            positions = self._static_positions
        else:
            positions = torch.from_numpy(
                np.array(self._positions[index], dtype=np.float32, copy=True)
            )
        return signal, mask, positions

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self._open()
        signals = np.array(self._signals[indices], dtype=np.float32, copy=True)
        masks = (
            np.ones((len(indices), self.channels), dtype=np.bool_)
            if self._mask is None
            else np.array(self._mask[indices], dtype=np.bool_, copy=True)
        )
        if self._static_positions is None:
            positions = np.array(
                self._positions[indices], dtype=np.float32, copy=True
            )
        return [
            (
                torch.from_numpy(signals[row]),
                torch.from_numpy(masks[row]),
                self._static_positions
                if self._static_positions is not None
                else torch.from_numpy(positions[row]),
            )
            for row in range(len(indices))
        ]


def read_mne_recording(path: str | Path, reader: str):
    """Open one supported existing EEG recording without preloading it."""
    import mne

    path = str(path)
    readers = {
        "eeglab": mne.io.read_raw_eeglab,
        "brainvision": mne.io.read_raw_brainvision,
        "edf": mne.io.read_raw_edf,
    }
    if reader not in readers:
        raise ValueError(f"unsupported MNE reader: {reader}")
    return readers[reader](path, preload=False, verbose="ERROR")


class RawMneWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Zero-copy windows from existing EEGLAB, BrainVision, or EDF recordings.

    One fully filtered and resampled recording is cached per DataLoader worker.
    Recording-grouped batches therefore avoid reopening a raw file for every
    window while still creating no persistent EEG copy.
    """

    def __init__(self, source: dict[str, Any], split: str) -> None:
        self.reader = str(source["reader"])
        self.target_rate = int(source["sample_rate"])
        self.window_samples = int(source["window_samples"])
        self.channel_names = [str(name) for name in source["channel_names"]]
        positions = np.asarray(source["channel_positions"], dtype=np.float32)
        if positions.shape != (len(self.channel_names), 3):
            raise ValueError(
                f"raw positions must be [C, 3], got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("raw channel positions must be finite")
        self._positions = torch.from_numpy(positions.copy())
        self.records = list(source["splits"][split]["recordings"])
        if not self.records:
            raise ValueError(f"raw source has no {split} recordings")
        self._stops = []
        offset = 0
        self.record_ranges = []
        for record in self.records:
            path = Path(record["path"])
            stat = path.stat()
            if stat.st_size != int(record["size_bytes"]):
                raise ValueError(f"raw recording size changed: {path}")
            if stat.st_mtime_ns != int(record["mtime_ns"]):
                raise ValueError(f"raw recording modification time changed: {path}")
            count = int(record["windows"])
            if count < 1:
                raise ValueError(f"raw recording contains no full windows: {path}")
            self.record_ranges.append((offset, offset + count))
            offset += count
            self._stops.append(offset)
        self._length = offset
        self._cached_record = None
        self._cached_windows = None
        self._cached_masks = None

    @property
    def channels(self) -> int:
        return len(self.channel_names)

    def __len__(self) -> int:
        return self._length

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        record_index = bisect.bisect_right(self._stops, index)
        start = 0 if record_index == 0 else self._stops[record_index - 1]
        return record_index, index - start

    def _load_recording(self, record_index: int) -> None:
        if self._cached_record == record_index:
            return
        from scipy.signal import butter, resample_poly, sosfiltfilt

        record = self.records[record_index]
        raw = read_mne_recording(record["path"], self.reader)
        if any(name not in raw.ch_names for name in self.channel_names):
            missing = [name for name in self.channel_names if name not in raw.ch_names]
            raise ValueError(f"{record['path']} is missing {missing}")
        signals = raw.get_data(picks=self.channel_names).astype(np.float32, copy=False)
        source_rate = float(raw.info["sfreq"])
        if not np.isclose(source_rate, self.target_rate):
            ratio = Fraction(self.target_rate / source_rate).limit_denominator(1000)
            signals = resample_poly(
                signals, ratio.numerator, ratio.denominator, axis=-1
            ).astype(np.float32, copy=False)
        sos = butter(
            4, [0.5, 45.0], btype="bandpass", fs=self.target_rate, output="sos"
        )
        signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32, copy=False)
        count = int(record["windows"])
        required = count * self.window_samples
        if signals.shape[-1] < required:
            raise ValueError(
                f"{record['path']} yielded {signals.shape[-1]} samples; need {required}"
            )
        windows = signals[:, :required].reshape(
            self.channels, count, self.window_samples
        )
        windows = np.transpose(windows, (1, 0, 2)).copy()
        scale = np.quantile(np.abs(windows), 0.95, axis=-1).astype(np.float32)
        masks = np.isfinite(windows).all(axis=-1) & (scale > 1e-8)
        windows = np.nan_to_num(windows)
        windows /= np.where(masks, scale, 1.0)[..., None]
        windows[~masks] = 0.0
        if not masks.any(axis=1).all():
            raise ValueError(f"{record['path']} contains an empty EEG window")
        self._cached_record = record_index
        self._cached_windows = windows.astype(np.float32, copy=False)
        self._cached_masks = masks

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index, local_index = self.locate(index)
        self._load_recording(record_index)
        return (
            torch.from_numpy(self._cached_windows[local_index].copy()),
            torch.from_numpy(self._cached_masks[local_index].copy()),
            self._positions,
        )

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [self[index] for index in indices]


class RecordingBatchSampler(Sampler[list[int]]):
    """Shuffle recordings while keeping each raw-file batch contiguous."""

    def __init__(
        self,
        record_ranges: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.record_ranges = list(record_ranges)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.iteration = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        ranges = list(self.record_ranges)
        if self.shuffle:
            rng.shuffle(ranges)
        for start, stop in ranges:
            indices = list(range(start, stop))
            if self.shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset : offset + self.batch_size]

    def __len__(self) -> int:
        return sum(
            math.ceil((stop - start) / self.batch_size)
            for start, stop in self.record_ranges
        )


def token_budget_batch_size(
    window_samples: int,
    *,
    reference_samples: int,
    reference_batch_size: int,
    minimum: int = 1,
) -> int:
    """Keep approximately constant temporal work across duration buckets."""
    if min(window_samples, reference_samples, reference_batch_size, minimum) < 1:
        raise ValueError("batch sizing arguments must be positive")
    budget = reference_samples * reference_batch_size
    return max(minimum, min(reference_batch_size, budget // window_samples))


def load_pretrain_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a canonical SAMBA multi-context manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("format") != "samba-pretrain-manifest-v2":
        raise ValueError("unsupported pretraining manifest format")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pretraining manifest must contain at least one source")
    names = [source.get("name") for source in sources]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every source needs a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("pretraining source names must be unique")
    weights = [float(source.get("weight", 0.0)) for source in sources]
    if any(weight <= 0 for weight in weights):
        raise ValueError("every source weight must be positive")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"source weights must sum to one, got {sum(weights)}")
    return payload
