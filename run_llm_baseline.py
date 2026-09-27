"""Command-line entry point for fixed offline baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_baseline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTANCE_EPSILON,
    DEFAULT_REWARD_EPSILON,
    DEFAULT_SAMPLES_PER_SOURCE,
    DEFAULT_SEED,
    run_baseline,
)


DEFAULT_OUTPUT_ROOT = Path("results") / "llm_baselines"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reload a fixed stratified transition sample and compute "
            "finite-sample baseline Lipschitz estimates."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "sampling output directory containing joint_replay.npz, metadata.json, "
            "and scenario_manifest.json; repeat for multiple checkpoints"
        ),
    )
    parser.add_argument(
        "--fixed-sample",
        help=(
            "existing baseline output containing fixed_samples.json/.npz; reloads "
            "without sampling (optional --source values only verify original hashes)"
        ),
    )
    parser.add_argument(
        "--samples-per-source", type=int, default=DEFAULT_SAMPLES_PER_SOURCE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=None,
        metavar="MBIT_PER_J",
        help=(
            "lambda list; new samples use the documented five-value default when "
            "omitted, while reloads reuse the saved list"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--distance-epsilon",
        type=float,
        default=None,
        help=(
            f"primary-pair distance threshold (new-run default {DEFAULT_DISTANCE_EPSILON:g}; "
            "reload default is the saved value)"
        ),
    )
    parser.add_argument(
        "--reward-epsilon",
        type=float,
        default=None,
        help=(
            f"reward-conflict tolerance (new-run default {DEFAULT_REWARD_EPSILON:g}; "
            "reload default is the saved value)"
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--output-dir",
        help="exact output directory; must not already exist",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixed_sample is None and not args.source:
        raise SystemExit("at least one --source is required unless --fixed-sample is used")
    result = run_baseline(
        source_directories=args.source,
        fixed_sample_directory=args.fixed_sample,
        samples_per_source=args.samples_per_source,
        seed=args.seed,
        lambdas=args.lambdas,
        batch_size=args.batch_size,
        distance_epsilon=args.distance_epsilon,
        reward_epsilon=args.reward_epsilon,
        output_root=args.output_root,
        output_dir=args.output_dir,
    )
    report = result["report"]
    print(
        "Baseline complete: "
        f"samples={report['lipschitz']['sample_count']} "
        f"pairs={report['lipschitz']['all_pair_count']} "
        f"primary_pairs={report['lipschitz']['primary_pair_count']}"
    )
    for value, estimate in report["lipschitz"][
        "estimates_by_lambda_mbit_per_joule"
    ].items():
        print(f"  lambda={value} L_hat={estimate['l_hat']} status={estimate['status']}")
    print(f"Output: {result['output_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
