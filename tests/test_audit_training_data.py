import numpy as np

from scripts.audit_training_data import sampled_indices, signal_statistics


def test_sampled_indices_cover_both_ends():
    assert sampled_indices(10, 4).tolist() == [0, 3, 6, 9]


def test_signal_statistics_detects_channelwise_zscore():
    rng = np.random.default_rng(7)
    signals = rng.normal(size=(32, 4, 256)).astype(np.float32)
    signals -= signals.mean(axis=-1, keepdims=True)
    signals /= signals.std(axis=-1, keepdims=True)
    stats = signal_statistics(signals, max_windows=16)
    assert stats["finite_fraction"] == 1.0
    assert stats["near_zscore_channel_fraction"] == 1.0
    assert stats["constant_channel_fraction"] == 0.0
