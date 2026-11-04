from __future__ import annotations

import torch
from torch import nn

from .modeling_samba import SambaModel


class SambaNeuralBenchBackbone(nn.Module):
    """Thin NeuralBench adapter returning a standard [B, T, D] tensor."""

    def __init__(self, model: SambaModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.model(x, channel_positions, channel_mask).last_hidden_state
