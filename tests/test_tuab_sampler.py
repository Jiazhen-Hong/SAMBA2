from samba_eeg.tuab import RecordingGroupedDistributedSampler


class Dataset:
    recording_groups = [list(range(0, 5)), list(range(5, 12)), list(range(12, 20))]


def test_distributed_sampler_has_equal_batch_aligned_lengths():
    first = RecordingGroupedDistributedSampler(Dataset(), batch_size=2, rank=0, world_size=2)
    second = RecordingGroupedDistributedSampler(Dataset(), batch_size=2, rank=1, world_size=2)
    assert len(first) == len(second)
    assert len(first) % 2 == 0
    assert set(first).isdisjoint(set(second))
