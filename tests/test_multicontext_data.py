import json

import numpy as np
import torch

from samba_eeg.data import (
    PositionedNpyWindowDataset,
    load_pretrain_manifest,
    token_budget_batch_size,
)
from scripts.export_multicontext_data import extract_neural_batch
from scripts.prepare_tuab_10s import stable_validation_groups
from samba_eeg.losses import TimeFrequencyReconstructionLoss


def test_token_budget_changes_batch_size_with_duration():
    assert token_budget_batch_size(
        128, reference_samples=256, reference_batch_size=256
    ) == 256
    assert token_budget_batch_size(
        256, reference_samples=256, reference_batch_size=256
    ) == 256
    assert token_budget_batch_size(
        512, reference_samples=256, reference_batch_size=256
    ) == 128
    assert token_budget_batch_size(
        1280, reference_samples=256, reference_batch_size=256
    ) == 51
    assert token_budget_batch_size(
        12800, reference_samples=256, reference_batch_size=256
    ) == 5


def test_positioned_dataset_supports_per_window_positions(tmp_path):
    signals = np.random.default_rng(0).normal(size=(4, 3, 128)).astype(np.float32)
    positions = np.random.default_rng(1).normal(size=(4, 3, 3)).astype(np.float32)
    mask = np.ones((4, 3), dtype=np.bool_)
    np.save(tmp_path / "signals.npy", signals)
    np.save(tmp_path / "positions.npy", positions)
    np.save(tmp_path / "mask.npy", mask)
    dataset = PositionedNpyWindowDataset(
        tmp_path / "signals.npy",
        positions=tmp_path / "positions.npy",
        channel_mask=tmp_path / "mask.npy",
    )
    batch = dataset.__getitems__([0, 2])
    assert dataset.window_samples == 128
    assert dataset.channels == 3
    assert batch[0][0].shape == (3, 128)
    assert batch[1][2].shape == (3, 3)


def test_extract_neural_batch_masks_invalid_channels():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(2, 3, 64)).astype(np.float32)
    signal[0, 1] = 0
    positions = rng.normal(size=(2, 3, 3)).astype(np.float32)
    positions[1, 2] = np.nan
    normalized, mask, clean_positions = extract_neural_batch(
        {"eeg": signal, "channel_positions": positions}
    )
    assert not mask[0, 1]
    assert not mask[1, 2]
    assert np.all(normalized[0, 1] == 0)
    assert np.isfinite(clean_positions).all()


def test_manifest_requires_normalized_positive_weights(tmp_path):
    manifest = {
        "format": "samba-pretrain-manifest-v2",
        "sampling_rate": 128,
        "sources": [
            {"name": "short", "weight": 0.4},
            {"name": "long", "weight": 0.6},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_pretrain_manifest(path)["sources"][1]["name"] == "long"


def test_orthonormal_spectral_loss_is_comparable_across_lengths():
    loss = TimeFrequencyReconstructionLoss(0.0, 1.0, fft_norm="ortho")
    short_target = torch.zeros(2, 3, 128)
    short_prediction = torch.ones_like(short_target)
    long_target = torch.zeros(2, 3, 1280)
    long_prediction = torch.ones_like(long_target)
    short_loss = loss(short_prediction, short_target)
    long_loss = loss(long_prediction, long_target)
    assert torch.allclose(short_loss, torch.tensor(1.0), rtol=1e-6)
    assert torch.allclose(short_loss, long_loss, rtol=1e-6)


def test_tuab_ssl_validation_groups_are_stable_and_disjoint():
    groups = {f"subject-{index:03d}" for index in range(100)}
    first = stable_validation_groups(groups, 0.05, 3407)
    second = stable_validation_groups(groups, 0.05, 3407)
    assert first == second
    assert len(first) == 5
    assert first < groups
