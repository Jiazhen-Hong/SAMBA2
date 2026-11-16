import json

import pytest

from scripts.package_huggingface_checkpoints import package_checkpoint, sha256


def make_source(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "samba", "checkpoint_variant": "SAMBA-T"})
    )
    (path / "model.safetensors").write_bytes(b"test-safe-weights")
    (path / "training_metadata.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "abc",
                "tuab_identity": {
                    "path": "/data/private/train.npy",
                    "shape": [10, 16, 12800],
                },
            }
        )
    )


def test_package_checkpoint_creates_huggingface_layout(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "checkpoints" / "SAMBA-100s"
    destination.mkdir(parents=True)
    (destination / "README.md").write_text("---\nlibrary_name: samba-eeg\n---\n")
    make_source(source)
    monkeypatch.setattr(
        "scripts.package_huggingface_checkpoints.git_revision",
        lambda repository: "revision",
    )

    package_checkpoint(
        repository=tmp_path,
        release_name="SAMBA-100s",
        source=source,
        destination=destination,
        historical_track="SAMBA-T",
        window_seconds=100,
    )

    assert json.loads((destination / "config.json").read_text())[
        "checkpoint_variant"
    ] == "SAMBA-100s"
    metadata = json.loads((destination / "training_metadata.json").read_text())
    assert "path" not in metadata["tuab_identity"]
    assert metadata["tuab_identity"]["shape"] == [10, 16, 12800]
    assert metadata["release"]["name"] == "SAMBA-100s"
    assert metadata["release"]["git_revision"] == "revision"
    sums = (destination / "SHA256SUMS").read_text()
    assert f"{sha256(destination / 'model.safetensors')}  model.safetensors" in sums


def test_package_checkpoint_refuses_to_overwrite(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "README.md").write_text("model card")
    (destination / "config.json").write_text("existing")
    make_source(source)

    with pytest.raises(FileExistsError):
        package_checkpoint(
            repository=tmp_path,
            release_name="SAMBA-2s",
            source=source,
            destination=destination,
            historical_track="SAMBA-E",
            window_seconds=2,
        )
