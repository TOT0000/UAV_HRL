"""CLI for one-method paper evaluation suites; this never trains a model."""

import argparse
import json

from evaluation_selection import (
    resolve_checkpoint_episodes,
    resolve_episode_horizons_s,
    resolve_environment_sizes_m,
    resolve_roi_counts,
)
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    METHOD_REGISTRY,
    MethodSpec,
)
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
    episode_horizon = parser.add_mutually_exclusive_group()
    episode_horizon.add_argument("--episode-horizon-s", type=int)
    episode_horizon.add_argument("--episode-horizons-s", type=int, nargs="+")
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        nargs="+",
        help="custom thresholds for the target-delay deadline suite only",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-episode", type=int)
    checkpoint.add_argument("--checkpoint-episodes", type=int, nargs="+")
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--roi-count", type=int)
    roi.add_argument("--roi-counts", type=int, nargs="+")
    environment_size = parser.add_mutually_exclusive_group()
    environment_size.add_argument("--environment-size-m", type=int)
    environment_size.add_argument("--environment-sizes-m", type=int, nargs="+")
    parser.add_argument(
        "--target-uav-id",
        type=int,
        help="explicit target UAV for uav_trajectory_snapshots",
    )
    parser.add_argument("--output-root", default="results/paper_evaluations")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        args.deadline_seconds is not None
        and args.suite != "task_type_delay_violation_vs_target_delay"
    ):
        raise ValueError(
            "--deadline-seconds is available only for the "
            "task_type_delay_violation_vs_target_delay suite"
        )
    checkpoint_episodes = resolve_checkpoint_episodes(
        args.checkpoint_episode, args.checkpoint_episodes
    )
    explicit_checkpoint = (
        args.checkpoint_episode is not None
        or args.checkpoint_episodes is not None
    )
    explicit_roi = args.roi_count is not None or args.roi_counts is not None
    if explicit_roi and args.suite not in {"fixed_roi", "environment_size"}:
        raise ValueError(
            "RoI selectors are available only for fixed_roi or environment_size"
        )
    if args.suite == "environment_size" and args.roi_counts is not None:
        raise ValueError("environment_size accepts --roi-count, not --roi-counts")
    roi_counts = None
    if args.suite == "fixed_roi":
        roi_counts = resolve_roi_counts(args.roi_count, args.roi_counts)
    elif args.suite == "environment_size":
        roi_counts = resolve_roi_counts(
            args.roi_count if args.roi_count is not None else 8,
            None,
        )
    explicit_environment_size = (
        args.environment_size_m is not None
        or args.environment_sizes_m is not None
    )
    if explicit_environment_size and args.suite != "environment_size":
        raise ValueError(
            "environment-size selectors are available only for environment_size"
        )
    environment_sizes_m = (
        resolve_environment_sizes_m(
            args.environment_size_m,
            args.environment_sizes_m,
        )
        if args.suite == "environment_size"
        else None
    )
    explicit_episode_horizon = (
        args.episode_horizon_s is not None
        or args.episode_horizons_s is not None
    )
    if explicit_episode_horizon and args.suite != "environment_size":
        raise ValueError(
            "episode-horizon selectors are available only for environment_size"
        )
    if explicit_episode_horizon and args.episode_seconds is not None:
        raise ValueError(
            "--episode-seconds cannot be combined with episode-horizon selectors"
        )
    episode_horizons_s = (
        resolve_episode_horizons_s(
            args.episode_horizon_s,
            args.episode_horizons_s,
        )
        if args.suite == "environment_size" and args.episode_seconds is None
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
    if args.suite == "environment_size":
        summary_horizons = (
            episode_horizons_s
            if episode_horizons_s is not None
            else (int(args.episode_seconds),)
        )
        episode_count = (
            FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
            if args.episodes is None
            else int(args.episodes)
        )
        print(
            f"{len(environment_sizes_m)} environment sizes × "
            f"{len(summary_horizons)} horizons × {episode_count} episodes = "
            f"{len(environment_sizes_m) * len(summary_horizons) * episode_count} "
            "episodes"
        )
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
        environment_sizes_m=environment_sizes_m,
        episode_horizons_s=episode_horizons_s,
        deadline_seconds=args.deadline_seconds,
        allow_registered_fixed_roi_method=bool(
            args.suite == "fixed_roi" and (explicit_checkpoint or explicit_roi)
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
