from scripts.prepare_tuab_edf_manifest import BIPOLAR_PAIRS, validate_channels, window_count


def test_original_bipolar_definition_has_sixteen_derivations():
    assert len(BIPOLAR_PAIRS) == 16


def test_window_count_uses_target_sampling_rate():
    assert window_count(200_000, source_rate=200, target_rate=128, window_samples=12_800) == 10


def test_required_channels_accept_original_tuab_names():
    names = sorted({channel for pair in BIPOLAR_PAIRS for channel in pair})
    validate_channels(names)
