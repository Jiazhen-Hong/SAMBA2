from pathlib import Path

import yaml


def load_recipe():
    return yaml.safe_load(Path("configs/samba.yaml").read_text())


def test_single_checkpoint_config_is_not_labeled_as_paper_reproduction():
    recipe = load_recipe()
    assert recipe["variant"] == "SAMBA"
    assert recipe["paper_reproduction"] is False
    assert recipe["status"] == "data_policy_frozen"
    assert recipe["reconstruction_space"] == "common_cdse_target"
    assert recipe["data"]["batching"] == "source_homogeneous_microbatches"
    assert recipe["data"]["normalization"] == "per_window_per_channel_abs_q95"
    assert recipe["data"]["mixture"] == {"TUAB": 0.5, "Emotiv_total": 0.5}
    assert recipe["training"]["micro_batch_size"] * recipe["training"]["gradient_accumulation_steps"] == 1024
    assert recipe["training"]["workers_per_source"] * 6 == 12
    assert recipe["training"]["save_every"] == 50
    assert recipe["objective"]["frequency_loss"] == "full_window_rfft"


def test_recovered_tuab_duration_is_not_inferred_from_directory_name():
    recipe = load_recipe()
    tuab = next(source for source in recipe["data"]["sources"] if source["name"] == "TUAB")
    assert tuab["window_samples"] == 256
    assert tuab["duration_seconds"] is None


def test_dreamer_is_excluded_and_driver_uses_subject_ids():
    recipe = load_recipe()
    emotiv = next(source for source in recipe["data"]["sources"] if source["name"] == "Emotiv")
    assert "DREAMER" in emotiv["excluded_datasets"]
    assert emotiv["driver_distraction_source"] == "DriverDistraction_X.npy"
    assert emotiv["split_policy"] == "subject_wise"
