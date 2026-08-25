"""CLI for sequential training-run × checkpoint × fixed-RoI evaluation."""

from __future__ import annotations

import argparse
import json

from checkpoint_roi_sweep import (
    build_checkpoint_roi_sweep_plan,
    execute_checkpoint_roi_sweep,
    public_sweep_plan,
)
from experiment_config import DEFAULT_TRAINING_SEED, FORMAL_EXPERIMENT_DEFAULTS


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more learned training runs over a checkpoint × RoI matrix"
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        dest="run_directories",
        help="completed training run directory; repeat for multiple runs",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-episode", type=int)
    checkpoint.add_argument("--checkpoint-episodes", type=int, nargs="+")
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--roi-count", type=int)
    roi.add_argument("--roi-counts", type=int, nargs="+")
    parser.add_argument(
        "--episodes",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS[
            "evaluation_episodes_per_trained_seed"
        ],
    )
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"],
    )
    parser.add_argument("--manifest-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument(
        "--output-root", default="results/checkpoint_roi_sweeps"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = build_checkpoint_roi_sweep_plan(
        args.run_directories,
        checkpoint_episode=args.checkpoint_episode,
        checkpoint_episodes=args.checkpoint_episodes,
        roi_count=args.roi_count,
        roi_counts=args.roi_counts,
        evaluation_episodes=args.episodes,
        episode_seconds=args.episode_seconds,
        manifest_seed=args.manifest_seed,
        output_root=args.output_root,
    )
    print(json.dumps(public_sweep_plan(plan), indent=2, ensure_ascii=False))
    result = execute_checkpoint_roi_sweep(plan)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
