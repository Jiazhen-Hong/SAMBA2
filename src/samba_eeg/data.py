from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
