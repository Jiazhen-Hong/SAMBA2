from __future__ import annotations

import torch
from torch import nn


class TimeFrequencyReconstructionLoss(nn.Module):
    """L1 time loss plus squared complex rFFT error from the SAMBA paper."""

    def __init__(
        self,
        time_weight: float = 1.0,
        frequency_weight: float = 1.0,
        fft_norm: str | None = None,
    ) -> None:
        super().__init__()
        if fft_norm not in {None, "ortho"}:
            raise ValueError("fft_norm must be None or 'ortho'")
        self.time_weight = time_weight
        self.frequency_weight = frequency_weight
        self.fft_norm = fft_norm

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        time_loss = (prediction - target).abs().mean()
        frequency_error = torch.fft.rfft(
            prediction, dim=-1, norm=self.fft_norm
        ) - torch.fft.rfft(target, dim=-1, norm=self.fft_norm)
        spectral_power = frequency_error.abs().square()
        if self.fft_norm == "ortho":
            # rFFT stores only the nonnegative frequencies. Double the interior
            # bins so the one-sided sum satisfies Parseval exactly, then divide
            # by T to obtain a per-sample energy that is comparable across
            # temporal lengths. DC and (for even T) Nyquist are not doubled.
            weights = torch.ones(
                spectral_power.shape[-1],
                dtype=spectral_power.dtype,
                device=spectral_power.device,
            )
            if prediction.shape[-1] % 2 == 0:
                weights[1:-1] = 2
            else:
                weights[1:] = 2
            frequency_loss = (
                spectral_power * weights
            ).sum(dim=-1).div(prediction.shape[-1]).mean()
        else:
            frequency_loss = spectral_power.mean()
        return self.time_weight * time_loss + self.frequency_weight * frequency_loss
