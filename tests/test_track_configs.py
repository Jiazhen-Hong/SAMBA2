from pathlib import Path

import yaml


def load_recipe():
    return yaml.safe_load(Path("configs/samba.yaml").read_text())


def test_single_checkpoint_config_is_not_labeled_as_paper_reproduction():
    recipe = load_recipe()
    assert recipe["variant"] == "SAMBA"
    assert recipe["paper_reproduction"] is False
    assert recipe["status"] == "pipeline_rebuild"
    assert recipe["reconstruction_space"] == "common_cdse_target"
    assert recipe["data"]["batching"] == "source_homogeneous_microbatches"


def test_recovered_tuab_duration_is_not_inferred_from_directory_name():
    recipe = load_recipe()
    tuab = next(source for source in recipe["data"]["sources"] if source["name"] == "TUAB")
    assert tuab["window_samples"] == 256
    assert tuab["duration_seconds"] is None
