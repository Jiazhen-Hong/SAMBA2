"""Train SAMBA on source-homogeneous, duration-bucketed EEG windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from samba_eeg import (
    NpyWindowDataset,
    SambaConfig,
    SambaModel,
    temporal_semantic_random_mask,
)
from samba_eeg.data import (
    PositionedNpyWindowDataset,
    RawMneWindowDataset,
    RecordingBatchSampler,
    load_pretrain_manifest,
    token_budget_batch_size,
)
from samba_eeg.losses import TimeFrequencyReconstructionLoss


EMOTIV_SOURCES = ("Alpha", "Attention", "Crowdsource", "DriverDistraction", "STEW")


def load_positions(path: Path) -> tuple[list[str], list[list[float]]]:
    frame = pd.read_excel(path)
    required = ["Channel", "X", "Y", "Z"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} must contain {required}")
    return frame["Channel"].astype(str).tolist(), frame[["X", "Y", "Z"]].astype(float).values.tolist()


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def file_identity(path: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def plain_file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def combined_manifest_hash(recipe: dict, emotiv_metadata: dict, tuab_root: Path) -> tuple[str, dict]:
    tuab_identity = {
        split: file_identity(tuab_root / f"{split}_data.npy")
        for split in ("train", "val")
    }
    identity = {
        "recipe": recipe,
        "emotiv_manifest_sha256": emotiv_metadata["manifest_sha256"],
        "tuab": tuab_identity,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), tuab_identity


def resolve_data_path(value: str, manifest_path: Path, repository: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    return manifest_relative if manifest_relative.exists() else repository / path


def multi_context_manifest_hash(
    recipe: dict, manifest: dict, manifest_path: Path, repository: Path
) -> tuple[str, dict]:
    identities = {}
    for source in manifest["sources"]:
        source_identity = {}
        if source.get("static_positions"):
            source_identity["static_positions"] = plain_file_identity(
                resolve_data_path(source["static_positions"], manifest_path, repository)
            )
        for split in ("train", "val"):
            entry = source["splits"][split]
            split_identity = {}
            for key in ("signals", "channel_mask", "channel_positions"):
                if entry.get(key):
                    path = resolve_data_path(entry[key], manifest_path, repository)
                    split_identity[key] = file_identity(path)
            source_identity[split] = split_identity
        identities[source["name"]] = source_identity
    identity = {"recipe": recipe, "manifest": manifest, "files": identities}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), identities


def make_loaders(
    emotiv_root: Path,
    tuab_root: Path,
    split: str,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    seed: int,
) -> dict[str, DataLoader]:
    datasets = {
        f"Emotiv/{source}": NpyWindowDataset(
            emotiv_root / source / f"{split}_signals.npy",
            emotiv_root / source / f"{split}_channel_mask.npy",
        )
        for source in EMOTIV_SOURCES
    }
    datasets["TUAB"] = NpyWindowDataset(tuab_root / f"{split}_data.npy")
    loaders = {}
    for offset, (name, dataset) in enumerate(datasets.items()):
        generator = torch.Generator().manual_seed(seed + offset)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=workers if split == "train" else 0,
            pin_memory=True,
            persistent_workers=split == "train" and workers > 0,
            prefetch_factor=prefetch_factor if split == "train" and workers > 0 else None,
            generator=generator,
        )
    return loaders


def make_manifest_loaders(
    manifest: dict,
    manifest_path: Path,
    repository: Path,
    split: str,
    reference_batch_size: int,
    reference_samples: int,
    workers: int,
    max_total_workers: int,
    prefetch_factor: int,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    loaders: dict[str, DataLoader] = {}
    batch_sizes: dict[str, int] = {}
    workers = min(workers, max_total_workers // max(1, len(manifest["sources"])))
    for offset, source in enumerate(manifest["sources"]):
        entry = source["splits"][split]
        if source.get("provider") == "raw_mne":
            dataset = RawMneWindowDataset(source, split)
            declared_samples = int(source["window_samples"])
            if dataset.window_samples != declared_samples:
                raise ValueError(
                    f"{source['name']} declares {declared_samples} samples but "
                    f"contains {dataset.window_samples}"
                )
            batch_size = token_budget_batch_size(
                dataset.window_samples,
                reference_samples=reference_samples,
                reference_batch_size=reference_batch_size,
            )
            sampler = RecordingBatchSampler(
                dataset.record_ranges,
                batch_size,
                shuffle=split == "train",
                seed=seed + offset,
            )
            raw_workers = workers if split == "train" else 0
            loaders[source["name"]] = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=raw_workers,
                pin_memory=True,
                persistent_workers=raw_workers > 0,
                prefetch_factor=(
                    prefetch_factor if split == "train" and raw_workers > 0 else None
                ),
            )
            batch_sizes[source["name"]] = batch_size
            continue
        static_positions = None
        positions_path = None
        if source.get("static_positions"):
            _, values = load_positions(
                resolve_data_path(source["static_positions"], manifest_path, repository)
            )
            static_positions = np.asarray(values, dtype=np.float32)
        elif entry.get("channel_positions"):
            positions_path = resolve_data_path(
                entry["channel_positions"], manifest_path, repository
            )
        else:
            raise ValueError(
                f"{source['name']} needs static_positions or per-window channel_positions"
            )
        dataset = PositionedNpyWindowDataset(
            resolve_data_path(entry["signals"], manifest_path, repository),
            static_positions=static_positions,
            positions=positions_path,
            channel_mask=(
                resolve_data_path(entry["channel_mask"], manifest_path, repository)
                if entry.get("channel_mask")
                else None
            ),
        )
        declared_samples = int(source["window_samples"])
        if dataset.window_samples != declared_samples:
            raise ValueError(
                f"{source['name']} declares {declared_samples} samples but contains "
                f"{dataset.window_samples}"
            )
        batch_size = token_budget_batch_size(
            dataset.window_samples,
            reference_samples=reference_samples,
            reference_batch_size=reference_batch_size,
        )
        generator = torch.Generator().manual_seed(seed + offset)
        loaders[source["name"]] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=workers if split == "train" else 0,
            pin_memory=True,
            persistent_workers=split == "train" and workers > 0,
            prefetch_factor=prefetch_factor if split == "train" and workers > 0 else None,
            generator=generator,
        )
        batch_sizes[source["name"]] = batch_size
    return loaders, batch_sizes


def source_probabilities(recipe: dict) -> tuple[list[str], list[float]]:
    tuab_weight = float(recipe["data"]["mixture"]["TUAB"])
    emotiv_weight = float(recipe["data"]["mixture"]["Emotiv_total"])
    names = [f"Emotiv/{source}" for source in EMOTIV_SOURCES] + ["TUAB"]
    weights = [emotiv_weight / len(EMOTIV_SOURCES)] * len(EMOTIV_SOURCES) + [tuab_weight]
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"source mixture must sum to one, got {sum(weights)}")
    return names, weights


def manifest_source_probabilities(manifest: dict) -> tuple[list[str], list[float]]:
    names = [source["name"] for source in manifest["sources"]]
    weights = [float(source["weight"]) for source in manifest["sources"]]
    return names, weights


def apply_channel_dropout(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    keep = torch.rand(mask.shape, device=mask.device) >= probability
    result = mask & keep
    empty = ~result.any(dim=1)
    if empty.any():
        for row in empty.nonzero(as_tuple=False).flatten().tolist():
            first = mask[row].nonzero(as_tuple=False)[0, 0]
            result[row, first] = True
    return result


def next_source_batch(name: str, loaders: dict, iterators: dict):
    try:
        return next(iterators[name])
    except StopIteration:
        iterators[name] = iter(loaders[name])
        return next(iterators[name])


def move_batch_to_device(batch, static_positions, device):
    if len(batch) == 2:
        signals, channel_mask = batch
        positions = static_positions
    elif len(batch) == 3:
        signals, channel_mask, positions = batch
    else:
        raise ValueError(f"expected a 2- or 3-tensor EEG batch, got {len(batch)}")
    return (
        signals.to(device, non_blocking=True),
        channel_mask.to(device, non_blocking=True),
        positions.to(device, non_blocking=True),
    )


def reconstruction_step(
    model: SambaModel,
    signals: torch.Tensor,
    channel_mask: torch.Tensor,
    positions: torch.Tensor,
    loss_function,
    mask_ratio: float,
    channel_dropout: float,
    autocast_enabled: bool,
    temporal_generator: torch.Generator | None = None,
) -> torch.Tensor:
    signals = signals.masked_fill(~channel_mask[..., None], 0)
    effective_mask = apply_channel_dropout(channel_mask, channel_dropout)
    masked, _ = temporal_semantic_random_mask(
        signals,
        mask_ratio,
        generator=temporal_generator,
    )
    masked = masked.masked_fill(~effective_mask[..., None], 0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        target, _ = model.cdse(signals, positions, channel_mask)
        prediction = model(masked, positions, effective_mask).reconstruction
    return loss_function(prediction.float(), target.detach().float())


@torch.no_grad()
def validate(
    model: SambaModel,
    loaders: dict[str, DataLoader],
    positions: dict[str, torch.Tensor],
    source_weights: dict[str, float],
    loss_function,
    mask_ratio: float,
    batches_per_source: int,
    device: torch.device,
    seed: int,
    autocast_enabled: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = {}
    generator = torch.Generator(device=device).manual_seed(seed)
    for name, loader in loaders.items():
        total, count = 0.0, 0
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches_per_source:
                break
            signals, channel_mask, batch_positions = move_batch_to_device(
                batch, positions.get(name), device
            )
            loss = reconstruction_step(
                model,
                signals,
                channel_mask,
                batch_positions,
                loss_function,
                mask_ratio,
                0.0,
                autocast_enabled,
                generator,
            )
            total += float(loss) * len(signals)
            count += len(signals)
        losses[name] = total / max(1, count)
    weighted = sum(losses[name] * source_weights[name] for name in losses)
    return weighted, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/samba.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--emotiv-root", type=Path, default=None)
    parser.add_argument("--tuab-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--workers-per-source",
        type=int,
        default=None,
        help="Persistent workers per source; defaults to the recipe (12 workers total).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimizer-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    recipe = yaml.safe_load(args.config.read_text())
    if recipe.get("variant") != "SAMBA" or recipe.get("status") not in {
        "data_policy_frozen",
        "multi_context_v2",
    }:
        raise ValueError("unsupported SAMBA training recipe")
    if args.manifest is None and (args.emotiv_root is None or args.tuab_root is None):
        parser.error("use --manifest, or provide both --emotiv-root and --tuab-root")
    if args.manifest is not None and (
        args.emotiv_root is not None or args.tuab_root is not None
    ):
        parser.error("--manifest cannot be combined with legacy data roots")

    repository = Path(__file__).resolve().parents[1]
    multi_context_manifest = None
    emotiv_metadata = None
    tuab_identity = None
    source_identities = None
    if args.manifest is not None:
        multi_context_manifest = load_pretrain_manifest(args.manifest)
        if int(multi_context_manifest["sampling_rate"]) != int(recipe["sampling_rate"]):
            raise ValueError("manifest and recipe sampling rates differ")
        manifest_hash, source_identities = multi_context_manifest_hash(
            recipe, multi_context_manifest, args.manifest, repository
        )
    else:
        emotiv_metadata = json.loads(
            (args.emotiv_root / "materialization.json").read_text()
        )
        if (
            emotiv_metadata.get("format") != "samba-emotiv-abs-q95-v1"
            or not emotiv_metadata.get("complete")
        ):
            raise ValueError("Emotiv materialization is incomplete or unsupported")
        manifest_hash, tuab_identity = combined_manifest_hash(
            recipe, emotiv_metadata, args.tuab_root
        )

    if not torch.cuda.is_available():
        raise RuntimeError("SAMBA pretraining requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    training = recipe["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    source_rng = random.Random(seed)

    target_names, target_positions = load_positions(repository / recipe["target_montage"])
    model_config = SambaConfig(
        **recipe["model"],
        target_channel_names=target_names,
        target_positions=target_positions,
        reconstruction_channels=len(target_names),
        sampling_rate=float(recipe["sampling_rate"]),
        checkpoint_variant="SAMBA",
    )
    model = SambaModel(model_config).to(device)
    positions = {}
    if multi_context_manifest is None:
        _, emotiv_positions_list = load_positions(repository / "Montage/Emotiv.xlsx")
        _, tuab_positions_list = load_positions(repository / "Montage/TUAB_16.xlsx")
        positions = {
            **{
                f"Emotiv/{source}": torch.tensor(
                    emotiv_positions_list, dtype=torch.float32, device=device
                )
                for source in EMOTIV_SOURCES
            },
            "TUAB": torch.tensor(
                tuab_positions_list, dtype=torch.float32, device=device
            ),
        }

    batch_size = args.micro_batch_size or int(training["micro_batch_size"])
    accumulation = args.accumulation_steps or int(training["gradient_accumulation_steps"])
    epochs = 1 if args.smoke_test else (args.epochs or int(training["epochs"]))
    steps_per_epoch = 4 if args.smoke_test else (
        args.optimizer_steps_per_epoch or int(training["optimizer_steps_per_epoch"])
    )
    validation_batches = 1 if args.smoke_test else int(training["validation_batches_per_source"])
    workers_per_source = (
        args.workers_per_source
        if args.workers_per_source is not None
        else int(training["workers_per_source"])
    )
    prefetch_factor = int(training["prefetch_factor"])
    if multi_context_manifest is None:
        names, weights = source_probabilities(recipe)
    else:
        names, weights = manifest_source_probabilities(multi_context_manifest)
    source_weights = dict(zip(names, weights))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["initial_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training["max_lr"]),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.5 if args.smoke_test else float(training["warmup_fraction"]),
        anneal_strategy="cos",
        div_factor=float(training["max_lr"]) / float(training["initial_lr"]),
        final_div_factor=float(training["initial_lr"]) / float(training["final_lr"]),
    )
    objective = recipe["objective"]
    loss_function = TimeFrequencyReconstructionLoss(
        float(objective["time_weight"]),
        float(objective["frequency_weight"]),
        objective.get("frequency_normalization"),
    )
    autocast_enabled = training["precision"] == "bf16"
    start_epoch, best_validation = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("resume checkpoint was trained with a different data manifest")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        source_rng.setstate(checkpoint["source_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])

    if multi_context_manifest is None:
        train_loaders = make_loaders(
            args.emotiv_root,
            args.tuab_root,
            "train",
            batch_size,
            workers_per_source,
            prefetch_factor,
            seed,
        )
        val_loaders = make_loaders(
            args.emotiv_root,
            args.tuab_root,
            "val",
            batch_size,
            0,
            prefetch_factor,
            seed + 1000,
        )
        bucket_batch_sizes = {name: batch_size for name in names}
    else:
        reference_samples = int(training["reference_window_samples"])
        max_total_workers = int(training["max_total_workers"])
        train_loaders, bucket_batch_sizes = make_manifest_loaders(
            multi_context_manifest,
            args.manifest,
            repository,
            "train",
            batch_size,
            reference_samples,
            workers_per_source,
            max_total_workers,
            prefetch_factor,
            seed,
        )
        val_loaders, _ = make_manifest_loaders(
            multi_context_manifest,
            args.manifest,
            repository,
            "val",
            batch_size,
            reference_samples,
            0,
            max_total_workers,
            prefetch_factor,
            seed + 1000,
        )
        print(
            "duration buckets: " + json.dumps(bucket_batch_sizes, sort_keys=True),
            flush=True,
        )
    iterators = {name: iter(loader) for name, loader in train_loaders.items()}

    args.output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, microbatches = 0.0, 0
        started = time.monotonic()
        for optimizer_step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                source = source_rng.choices(names, weights=weights, k=1)[0]
                batch = next_source_batch(source, train_loaders, iterators)
                signals, channel_mask, batch_positions = move_batch_to_device(
                    batch, positions.get(source), device
                )
                loss = reconstruction_step(
                    model,
                    signals,
                    channel_mask,
                    batch_positions,
                    loss_function,
                    float(objective["mask_ratio"]),
                    float(training["channel_dropout"]),
                    autocast_enabled,
                )
                (loss / accumulation).backward()
                running_loss += float(loss.detach())
                microbatches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (optimizer_step + 1) % args.log_every == 0 or optimizer_step == 0:
                elapsed = time.monotonic() - started
                print(
                    f"epoch={epoch + 1}/{epochs} step={optimizer_step + 1}/{steps_per_epoch} "
                    f"train={running_loss / microbatches:.7g} lr={scheduler.get_last_lr()[0]:.3g} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        validation_loss, source_validation = validate(
            model,
            val_loaders,
            positions,
            source_weights,
            loss_function,
            float(objective["mask_ratio"]),
            validation_batches,
            device,
            seed + 100_000,
            autocast_enabled,
        )
        print(
            f"epoch={epoch + 1}/{epochs} train={running_loss / microbatches:.7g} "
            f"val={validation_loss:.7g} sources={json.dumps(source_validation, sort_keys=True)}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation": min(best_validation, validation_loss),
            "manifest_sha256": manifest_hash,
            "source_rng_state": source_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "recipe": recipe,
        }
        atomic_save(state, args.output_root / "last.pt")
        if not args.smoke_test and (epoch + 1) % int(training["save_every"]) == 0:
            atomic_save(state, args.output_root / f"epoch-{epoch + 1:03d}.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            model.save_pretrained(args.output_root / "best")
            metadata = {
                "variant": "SAMBA",
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "source_validation_loss": source_validation,
                "manifest_sha256": manifest_hash,
                "emotiv_manifest_sha256": (
                    emotiv_metadata["manifest_sha256"] if emotiv_metadata else None
                ),
                "tuab_identity": tuab_identity,
                "source_identities": source_identities,
                "duration_bucket_batch_sizes": bucket_batch_sizes,
                "paper_reproduction": False,
                "excluded_sources": (
                    ["DREAMER"] if multi_context_manifest is None else []
                ),
                "limitations": (
                    [
                        "TUAB duration is not verified from surviving array metadata.",
                        "Recovered TUAB arrays do not retain recording or subject IDs.",
                    ]
                    if multi_context_manifest is None
                    else multi_context_manifest.get("limitations", [])
                ),
            }
            (args.output_root / "best" / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )


if __name__ == "__main__":
    main()
