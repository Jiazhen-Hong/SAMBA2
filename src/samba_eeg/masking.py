from __future__ import annotations

import torch


def temporal_semantic_random_mask(
    signals: torch.Tensor,
    mask_ratio: float = 0.5,
    num_visible_blocks: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask time spans while preserving exactly the requested number of samples.

    Returns the masked signal and a boolean mask where True denotes a masked value.
    The same time mask is applied to all channels in one recording.
    """
    if signals.ndim != 3:
        raise ValueError("signals must have shape [batch, channels, time]")
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    batch, channels, length = signals.shape
    visible = length - int(round(length * mask_ratio))
    blocks = max(1, min(num_visible_blocks, visible))
    raw_lengths = torch.rand((batch, blocks), device=signals.device, generator=generator)
    block_lengths = torch.floor(
        raw_lengths / raw_lengths.sum(dim=1, keepdim=True) * (visible - blocks)
    ).long() + 1
    block_lengths[:, -1] += visible - block_lengths.sum(dim=1)

    gap_total = length - visible
    raw_gaps = torch.rand((batch, blocks + 1), device=signals.device, generator=generator)
    gaps = torch.floor(raw_gaps / raw_gaps.sum(dim=1, keepdim=True) * gap_total).long()
    gaps[:, -1] += gap_total - gaps.sum(dim=1)

    starts = torch.empty_like(block_lengths)
    starts[:, 0] = gaps[:, 0]
    if blocks > 1:
        starts[:, 1:] = gaps[:, :1] + torch.cumsum(
            block_lengths[:, :-1] + gaps[:, 1:blocks], dim=1
        )
    ends = starts + block_lengths
    time = torch.arange(length, device=signals.device)[None, None, :]
    visible_mask = ((time >= starts[..., None]) & (time < ends[..., None])).any(dim=1)
    mask = (~visible_mask)[:, None, :].expand(batch, channels, length)
    return signals.masked_fill(mask, 0), mask
