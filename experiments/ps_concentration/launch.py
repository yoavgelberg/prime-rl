"""Configure or launch the seed-11 Matching Pairs GRPO parent run."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .hero_config import HeroConfig


def configure(args):
    """Resolve machine paths without changing the recorded parent recipe."""
    config = HeroConfig(
        model=str(args.model.resolve()),
        source=str(args.source.resolve()),
        output=str(args.output.resolve()),
        node_dir=str(args.node_dir.resolve()),
        source_sha256=args.source_sha256,
        deadline=args.deadline,
        gpus=args.gpus,
        wandb_id=args.wandb_id,
    )
    if not Path(config.model, "config.json").is_file():
        raise ValueError("--model must point to the pinned local Hugging Face snapshot")
    if not Path(config.source, "1_matching_pairs_new").is_dir():
        raise ValueError("--source must contain the native 1_matching_pairs_new directory")
    from .overlap_config import protocol_lock

    protocol_lock(config)
    destination = Path(config.output) / "launch-config.json"
    values = json.loads(json.dumps(asdict(config), allow_nan=False))
    if destination.exists():
        if json.loads(destination.read_text()) != values:
            raise ValueError("Existing launch configuration differs; use a new output directory")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            handle.write(json.dumps(values, indent=2, allow_nan=False) + "\n")
    return config, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Pinned native RNG-Bench source")
    parser.add_argument("--source-sha256", required=True, help="SHA256 of the experiment implementation archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True, help="Fresh node-local control directory")
    parser.add_argument("--deadline", type=float, required=True, help="Allocation end time as Unix seconds")
    parser.add_argument("--gpus", type=int, choices=(4, 8), default=8)
    parser.add_argument("--wandb-id", required=True, help="Unique W&B ID; reuse only when resuming the same run")
    parser.add_argument("--prepare-only", action="store_true", help="Write configuration without starting workers")
    args = parser.parse_args(argv)
    if not args.prepare_only and os.environ.get("SLURM_JOB_PARTITION") != "yejin":
        parser.error("Launch only inside an allocated regular yejin job")
    _, path = configure(args)
    print(path, flush=True)
    if not args.prepare_only:
        from .hero_v2 import main as train

        train(["--config", str(path)])


if __name__ == "__main__":
    main()
