"""Build a self-contained config.json from a recipe and montage workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text())
    montage_path = Path(recipe["target_montage"])
    montage = pd.read_excel(montage_path)
    required = ["Channel", "X", "Y", "Z"]
    if not set(required).issubset(montage.columns):
        raise ValueError(f"{montage_path} must contain {required}")
    model = dict(recipe["model"])
    model.update(
        target_channel_names=montage["Channel"].astype(str).tolist(),
        target_positions=montage[["X", "Y", "Z"]].astype(float).values.tolist(),
        reconstruction_channels=int(recipe["input_channels"]),
        sampling_rate=float(recipe["sampling_rate"]),
        checkpoint_variant=recipe["variant"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(model, indent=2) + "\n")
    print(f"Wrote {args.output / 'config.json'} ({len(montage)} target channels)")


if __name__ == "__main__":
    main()
