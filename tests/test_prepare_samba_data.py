import numpy as np

from scripts.prepare_samba_data import deterministic_subject_split, normalize_abs_q95


def test_subject_split_is_deterministic_and_disjoint():
    ids = np.repeat(np.arange(20), 3)
    first_indices, first_subjects = deterministic_subject_split(ids, seed=3407)
    second_indices, second_subjects = deterministic_subject_split(ids, seed=3407)
    assert first_subjects == second_subjects
    assert all(np.array_equal(first_indices[key], second_indices[key]) for key in first_indices)
    split_sets = [set(values) for values in first_subjects.values()]
    assert not split_sets[0].intersection(split_sets[1])
    assert not split_sets[0].intersection(split_sets[2])
    assert not split_sets[1].intersection(split_sets[2])
    assert sum(len(indices) for indices in first_indices.values()) == len(ids)


def test_abs_q95_normalization_and_constant_channel_mask():
    rng = np.random.default_rng(9)
    signals = rng.normal(size=(3, 14, 256)).astype(np.float32) * 17
    signals[0, 2] = 0
    normalized, mask = normalize_abs_q95(signals)
    assert normalized.dtype == np.float32
    assert mask.shape == (3, 14)
    assert not mask[0, 2]
    assert np.all(normalized[0, 2] == 0)
    scales = np.quantile(np.abs(normalized), 0.95, axis=-1)
    assert np.allclose(scales[mask], 1.0, atol=1e-5)
