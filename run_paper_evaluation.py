"""CLI for one-method paper evaluation suites; this never trains a model."""

import argparse
import json

from evaluation_selection import resolve_checkpoint_episodes, resolve_roi_counts
from experiment_config import METHOD_REGISTRY, MethodSpec
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
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-episode", type=int)
    checkpoint.add_argument("--checkpoint-episodes", type=int, nargs="+")
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--roi-count", type=int)
    roi.add_argument("--roi-counts", type=int, nargs="+")
    parser.add_argument(
        "--target-uav-id",
        type=int,
        help="explicit target UAV for uav_trajectory_snapshots",
    )
    parser.add_argument("--output-root", default="results/paper_evaluations")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint_episodes = resolve_checkpoint_episodes(
        args.checkpoint_episode, args.checkpoint_episodes
    )
    explicit_checkpoint = (
        args.checkpoint_episode is not None
        or args.checkpoint_episodes is not None
    )
    explicit_roi = args.roi_count is not None or args.roi_counts is not None
    if explicit_roi and args.suite != "fixed_roi":
        raise ValueError("RoI selectors are available only for the fixed_roi suite")
    roi_counts = (
        resolve_roi_counts(args.roi_count, args.roi_counts)
        if args.suite == "fixed_roi"
        else None
    )
    method = MethodSpec.parse(args.method)
    if explicit_checkpoint and not (
        method.learns_movement or method.learns_routing
    ):
        raise ValueError("a pure-random method has no checkpoint episode selector")
    if len(checkpoint_episodes) > 1:
        if args.suite != "fixed_roi":
            raise ValueError(
                "multi-checkpoint paper evaluation is available only for fixed_roi"
            )
        if args.run_dir is None:
            raise ValueError("multi-checkpoint evaluation requires --run-dir")
        from checkpoint_roi_sweep import (
            build_checkpoint_roi_sweep_plan,
            execute_checkpoint_roi_sweep,
            public_sweep_plan,
        )

        plan = build_checkpoint_roi_sweep_plan(
            (args.run_dir,),
            checkpoint_episodes=checkpoint_episodes,
            roi_counts=roi_counts,
            evaluation_episodes=args.episodes,
            episode_seconds=args.episode_seconds,
            manifest_seed=args.manifest_seed,
            output_root=args.output_root,
        )
        print(json.dumps(public_sweep_plan(plan), indent=2, ensure_ascii=False))
        result = execute_checkpoint_roi_sweep(plan)
        print(json.dumps(result, ensure_ascii=False))
        return 0
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
        checkpoint_episode=checkpoint_episodes[0],
        roi_counts=roi_counts,
        allow_registered_fixed_roi_method=bool(
            args.suite == "fixed_roi" and (explicit_checkpoint or explicit_roi)
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
