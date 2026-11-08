"""Paper-aligned SAMBA-T pretraining from the verified TUAB EDF manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from samba_eeg import (
    RecordingGroupedDistributedSampler,
    SambaConfig,
    SambaModel,
    TUABEDFWindowDataset,
    TUABNpyShardDataset,
    temporal_semantic_random_mask,
)
from samba_eeg.losses import TimeFrequencyReconstructionLoss


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


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


def reduce_mean(total: float, count: int, device: torch.device) -> float:
    value = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value)
    return float(value[0] / value[1].clamp_min(1))


@torch.no_grad()
def validate(model, loader, positions, loss_function, device, seed: int, max_batches: int | None = None) -> float:
    model.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, signals in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        signals = signals.to(device, non_blocking=True)
        masked, _ = temporal_semantic_random_mask(signals, 0.5, generator=generator)
        prediction = model(masked, positions).reconstruction
        loss = loss_function(prediction.float(), signals.float())
        total += float(loss) * signals.shape[0]
        count += signals.shape[0]
    return reduce_mean(total, count, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("reproduction/SAMBA-T/config.yaml"))
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--manifest-root", type=Path, help="Verified EDF manifest for on-demand loading.")
    data_source.add_argument(
        "--materialized-root",
        type=Path,
        help="Pre-normalized NPY shards created by scripts/materialize_tuab_100s.py.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Global batch size; defaults to the paper recipe.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true", help="Run four train batches and one validation batch.")
    args = parser.parse_args()

    recipe = yaml.safe_load(args.config.read_text())
    if recipe.get("variant") != "SAMBA-T" or recipe.get("paper_locked") is not True:
        raise ValueError("this trainer only accepts the locked SAMBA-T reproduction recipe")
    if args.materialized_root:
        materialization = json.loads((args.materialized_root / "materialization.json").read_text())
        summary = materialization["source_summary"]
        if materialization.get("format") != TUABNpyShardDataset.FORMAT:
            raise ValueError("unsupported materialized TUAB format")
        if materialization.get("dtype") != "float32" or materialization.get("normalized") is not True:
            raise ValueError("paper reproduction requires normalized float32 materialized windows")
    else:
        summary = json.loads((args.manifest_root / "summary.json").read_text())
    expected = int(recipe["data"]["expected_samples"])
    if summary.get("total_windows") != expected or not summary.get("matches_paper_count"):
        raise ValueError(f"verified SAMBA-T manifest must contain exactly {expected} windows")
    if any(split["errors"] for split in summary["splits"].values()):
        raise ValueError("manifest contains unresolved recordings")

    rank, world_size, local_rank = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("SAMBA-T pretraining requires CUDA")
    device = torch.device("cuda", local_rank)
    seed = int(recipe["training"]["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    repository = Path(__file__).resolve().parents[1]
    target_names, target_positions = load_positions(repository / recipe["target_montage"])
    _, input_positions_list = load_positions(repository / "Montage/TUAB_16.xlsx")
    input_positions = torch.tensor(input_positions_list, dtype=torch.float32, device=device)
    model_config = SambaConfig(
        **recipe["model"],
        target_channel_names=target_names,
        target_positions=target_positions,
        reconstruction_channels=int(recipe["input_channels"]),
        sampling_rate=float(recipe["sampling_rate"]),
        checkpoint_variant=recipe["variant"],
    )
    model = SambaModel(model_config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.materialized_root:
        datasets = {
            split: TUABNpyShardDataset(args.materialized_root, split)
            for split in ("train", "val")
        }
    else:
        datasets = {
            split: TUABEDFWindowDataset(args.manifest_root / f"{split}.jsonl")
            for split in ("train", "val")
        }
    global_batch = args.batch_size or int(recipe["training"]["batch_size"])
    if global_batch % world_size:
        raise ValueError(f"global batch size {global_batch} is not divisible by world size {world_size}")
    local_batch = global_batch // world_size
    samplers = {
        split: RecordingGroupedDistributedSampler(
            dataset,
            batch_size=local_batch,
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=samplers[split],
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )
        for split, dataset in datasets.items()
    }

    training = recipe["training"]
    epochs = 1 if args.smoke_test else (args.epochs or int(training["epochs"]))
    steps_per_epoch = 4 if args.smoke_test else len(loaders["train"])
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
        float(objective["time_weight"]), float(objective["frequency_weight"])
    )
    start_epoch, best_validation = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        core = model.module if isinstance(model, DistributedDataParallel) else model
        core.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        samplers["train"].set_epoch(epoch)
        model.train()
        train_total, train_count = 0.0, 0
        for batch_index, signals in enumerate(loaders["train"]):
            if args.smoke_test and batch_index >= 4:
                break
            signals = signals.to(device, non_blocking=True)
            masked, _ = temporal_semantic_random_mask(signals, float(objective["mask_ratio"]))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(masked, input_positions).reconstruction
            loss = loss_function(prediction.float(), signals.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_total += float(loss.detach()) * signals.shape[0]
            train_count += signals.shape[0]

        train_loss = reduce_mean(train_total, train_count, device)
        validation_loss = validate(
            model,
            loaders["val"],
            input_positions,
            loss_function,
            device,
            seed + epoch,
            max_batches=1 if args.smoke_test else None,
        )
        core = model.module if isinstance(model, DistributedDataParallel) else model
        if rank == 0:
            print(f"epoch={epoch + 1}/{epochs} train={train_loss:.8g} val={validation_loss:.8g}", flush=True)
            state = {
                "epoch": epoch,
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_validation": min(best_validation, validation_loss),
                "recipe": recipe,
                "manifest_sha256": summary["manifest_sha256"],
            }
            atomic_save(state, args.output_root / "last.pt")
            if not args.smoke_test and (epoch + 1) % args.save_every == 0:
                atomic_save(state, args.output_root / f"epoch-{epoch + 1:03d}.pt")
            if validation_loss < best_validation:
                best_validation = validation_loss
                core.save_pretrained(args.output_root / "best")
                metadata = {
                    "variant": "SAMBA-T",
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                    "manifest_sha256": summary["manifest_sha256"],
                    "paper_locked": True,
                }
                (args.output_root / "best" / "training_metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                )
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
