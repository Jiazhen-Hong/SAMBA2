from pathlib import Path

import yaml


def load(path: str):
    return yaml.safe_load(Path(path).read_text())


def test_paper_tracks_are_locked_and_single_duration():
    samba_e = load("reproduction/SAMBA-E/config.yaml")
    samba_t = load("reproduction/SAMBA-T/config.yaml")

    assert samba_e["paper_locked"] is True
    assert samba_t["paper_locked"] is True
    assert samba_e["duration_mixing"] is False
    assert samba_t["duration_mixing"] is False
    assert samba_e["reconstruction_space"] == "input_montage"
    assert samba_t["reconstruction_space"] == "input_montage"
    assert samba_e["window_samples"] == 256
    assert samba_t["window_samples"] == 12800


def test_optimized_track_cannot_be_mistaken_for_paper_checkpoint():
    samba_opt = load("SAMBA-opt/config.yaml")
    assert samba_opt["track"] == "experimental"
    assert samba_opt["paper_locked"] is False
    assert samba_opt["variant"] == "SAMBA-opt"
