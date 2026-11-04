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
    mask = torch.ones((batch, length), dtype=torch.bool, device=signals.device)

    for index in range(batch):
        raw = torch.rand(blocks, device=signals.device, generator=generator)
        block_lengths = torch.floor(raw / raw.sum() * (visible - blocks)).long() + 1
        block_lengths[-1] += visible - int(block_lengths.sum())
        gap_total = length - visible
        raw_gaps = torch.rand(blocks + 1, device=signals.device, generator=generator)
        gaps = torch.floor(raw_gaps / raw_gaps.sum() * gap_total).long()
        gaps[-1] += gap_total - int(gaps.sum())
        cursor = int(gaps[0])
        for block_index, block_length in enumerate(block_lengths.tolist()):
            mask[index, cursor : cursor + block_length] = False
            cursor += block_length + int(gaps[block_index + 1])

    mask = mask[:, None, :].expand(batch, channels, length)
    return signals.masked_fill(mask, 0), mask
