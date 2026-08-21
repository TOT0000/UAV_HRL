"""CLI for one-method paper evaluation suites; this never trains a model."""

import argparse
import json

from experiment_config import METHOD_REGISTRY
from paper_evaluation import PAPER_EVALUATION_SUITES, run_paper_evaluation


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one semantic paper suite without starting training"
    )
    parser.add_argument("method", choices=tuple(METHOD_REGISTRY))
    parser.add_argument(
        "--run-dir",
        help="completed training run (omit only for the pure-random baseline)",
    )
    parser.add_argument("--suite", required=True, choices=tuple(PAPER_EVALUATION_SUITES))
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument(
        "--target-uav-id",
        type=int,
        help="explicit target UAV for uav_trajectory_snapshots",
    )
    parser.add_argument("--output-root", default="results/paper_evaluations")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_paper_evaluation(
        args.method,
        run_directory=args.run_dir,
        suite=args.suite,
        manifest_path=args.manifest,
        manifest_seed=args.manifest_seed,
        episodes=args.episodes,
        episode_seconds=args.episode_seconds,
        target_uav_id=args.target_uav_id,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
