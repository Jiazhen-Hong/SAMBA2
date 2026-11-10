from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
