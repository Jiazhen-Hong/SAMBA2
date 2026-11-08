"""Evaluate a released SAMBA checkpoint with NeuralBench."""

from __future__ import annotations

import argparse

from neuralbench import evaluate_model

from samba_eeg import SambaModel
from samba_eeg.integrations import SambaNeuralBenchBackbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="all")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--name", default="samba")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model = SambaNeuralBenchBackbone(SambaModel.from_pretrained(args.checkpoint))
    scores = evaluate_model(
        model,
        "eeg",
        args.task,
        dataset=args.dataset,
        name=args.name,
        downstream_wrapper="linear_probe_mean",
        debug=args.debug,
    )
    print(scores)


if __name__ == "__main__":
    main()
