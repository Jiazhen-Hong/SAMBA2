import torch

from samba_eeg import SambaConfig, SambaModel, temporal_semantic_random_mask
from samba_eeg.integrations import SambaNeuralBenchBackbone
from samba_eeg.probing import temporal_statistics


def config() -> SambaConfig:
    return SambaConfig(
        base_channels=8,
        bottleneck_channels=16,
        d_state=4,
        d_conv=2,
        expand=2,
        head_dim=8,
        target_channel_names=["A", "B", "C", "D"],
        target_positions=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0]],
        reconstruction_channels=3,
        mixer_backend="gru",
    )


def test_runtime_montage_and_neuralbench_shape():
    model = SambaModel(config())
    signals = torch.randn(2, 3, 32)
    positions = torch.randn(2, 3, 3)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    output = model(signals, positions, valid)
    assert output.reconstruction.shape == (2, 3, 32)
    assert output.last_hidden_state.shape == (2, 4, 16)
    assert output.spatial_weights.shape == (2, 4, 3)
    assert torch.allclose(output.spatial_weights[0, :, 2], torch.zeros(4))
    assert SambaNeuralBenchBackbone(model)(signals, positions, valid).shape == (2, 4, 16)


def test_nan_position_is_treated_as_missing_channel():
    model = SambaModel(config())
    signals = torch.randn(1, 3, 16)
    positions = torch.randn(3, 3)
    positions[1] = torch.nan
    output = model(signals, positions)
    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.equal(output.spatial_weights[0, :, 1], torch.zeros(4))


def test_mask_is_exact_for_short_sequences():
    signals = torch.ones(3, 2, 16)
    masked, mask = temporal_semantic_random_mask(signals, mask_ratio=0.5, num_visible_blocks=8)
    assert mask[:, 0].sum(dim=1).tolist() == [8, 8, 8]
    assert torch.equal(masked == 0, mask)


def test_paper_statistics_use_time_axis():
    features = temporal_statistics(torch.randn(2, 5, 7))
    assert features.shape == (2, 7 * 9)


def test_checkpoint_round_trip(tmp_path):
    model = SambaModel(config())
    model.save_pretrained(tmp_path, safe_serialization=False)
    restored = SambaModel.from_pretrained(tmp_path)
    assert restored.config.output_channels == 3
    assert set(model.state_dict()) == set(restored.state_dict())
