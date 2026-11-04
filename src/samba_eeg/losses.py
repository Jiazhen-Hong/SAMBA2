from __future__ import annotations

import torch
from torch import nn


class TimeFrequencyReconstructionLoss(nn.Module):
    """L1 time loss plus squared complex rFFT error from the SAMBA paper."""

    def __init__(self, time_weight: float = 1.0, frequency_weight: float = 1.0) -> None:
        super().__init__()
        self.time_weight = time_weight
        self.frequency_weight = frequency_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        time_loss = (prediction - target).abs().mean()
        frequency_error = torch.fft.rfft(prediction, dim=-1) - torch.fft.rfft(target, dim=-1)
        frequency_loss = frequency_error.abs().square().mean()
        return self.time_weight * time_loss + self.frequency_weight * frequency_loss
