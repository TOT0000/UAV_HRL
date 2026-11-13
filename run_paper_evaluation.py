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
from paper_evaluation import (
    PAPER_EVALUATION_SUITES,
    evaluation_sweep_points,
    run_paper_evaluation,
)


COMMUNICATION_RESOURCE_SUITES = {
    "bandwidth_sweep",
    "sr_transmit_power_sweep",
    "uav_transmit_power_sweep",
}
SR_SPEED_SUITE = "sr_speed_sweep"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one semantic paper suite without starting training"
    )
    parser.add_argument("method", choices=tuple(METHOD_REGISTRY))
    parser.add_argument(
        "--run-dir",
        help=(
            "training run containing the selected complete checkpoint "
            "(omit only for the pure-random baseline)"
        ),
    )
    parser.add_argument("--suite", required=True, choices=tuple(PAPER_EVALUATION_SUITES))
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument("--collect-search-diagnostics", action="store_true")
    episode_horizon = parser.add_mutually_exclusive_group()
    episode_horizon.add_argument("--episode-horizon-s", type=int)
    episode_horizon.add_argument("--episode-horizons-s", type=int, nargs="+")
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        nargs="+",
        help="custom thresholds for the target-delay deadline suite only",
    )
    parser.add_argument(
        "--bandwidth-mhz",
        type=float,
        nargs="+",
        help="bandwidth sweep values, or one fixed value for a power sweep",
    )
    parser.add_argument(
        "--sr-tx-power-dbm",
        type=float,
        nargs="+",
        help="SR power sweep values, or one fixed value for another resource sweep",
    )
    parser.add_argument(
        "--uav-tx-power-dbm",
        type=float,
        nargs="+",
        help="UAV power sweep values, or one fixed value for another resource sweep",
    )
    parser.add_argument(
        "--sr-speeds-mps",
        type=float,
        nargs="+",
        help="one or more arbitrary finite positive SR speeds for sr_speed_sweep",
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
    parser.add_argument(
        "--output-directory",
        help=(
            "explicit directory for a single flattened evaluation point; "
            "ordinary evaluations should continue to use --output-root"
        ),
    )
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
    if explicit_roi and args.suite not in {
        "fixed_roi",
        "environment_size",
        *COMMUNICATION_RESOURCE_SUITES,
        SR_SPEED_SUITE,
    }:
        raise ValueError(
            "RoI selectors require fixed_roi, environment_size, or a "
            "communication-resource suite"
        )
    if (
        args.suite == "environment_size"
        or args.suite in COMMUNICATION_RESOURCE_SUITES
    ) and args.roi_counts is not None:
        raise ValueError(f"{args.suite} accepts --roi-count, not --roi-counts")
    roi_counts = None
    if args.suite == "fixed_roi":
        roi_counts = resolve_roi_counts(args.roi_count, args.roi_counts)
    elif args.suite == SR_SPEED_SUITE:
        roi_counts = resolve_roi_counts(args.roi_count, args.roi_counts)
    elif (
        args.suite == "environment_size"
        or args.suite in COMMUNICATION_RESOURCE_SUITES
    ):
        roi_counts = resolve_roi_counts(
            args.roi_count if args.roi_count is not None else 8,
            None,
        )
    resource_values_present = any(
        value is not None
        for value in (
            args.bandwidth_mhz,
            args.sr_tx_power_dbm,
            args.uav_tx_power_dbm,
        )
    )
    if resource_values_present and args.suite not in COMMUNICATION_RESOURCE_SUITES:
        raise ValueError(
            "communication-resource arguments require a communication-resource suite"
        )
    if args.suite in COMMUNICATION_RESOURCE_SUITES:
        evaluation_sweep_points(
            args.suite,
            roi_counts=roi_counts,
            bandwidth_mhz=args.bandwidth_mhz,
            sr_tx_power_dbm=args.sr_tx_power_dbm,
            uav_tx_power_dbm=args.uav_tx_power_dbm,
        )
    if args.suite == SR_SPEED_SUITE:
        evaluation_sweep_points(
            args.suite,
            roi_counts=roi_counts,
            sr_speeds_mps=args.sr_speeds_mps,
        )
    elif args.sr_speeds_mps is not None:
        raise ValueError("--sr-speeds-mps is available only for sr_speed_sweep")
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
    if args.collect_search_diagnostics and args.suite != "environment_size":
        raise ValueError(
            "--collect-search-diagnostics is available only for environment_size"
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
        if args.output_directory is not None:
            raise ValueError("--output-directory requires a single checkpoint")
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
        output_directory=args.output_directory,
        flatten_single_point=args.output_directory is not None,
        checkpoint_episode=checkpoint_episodes[0],
        roi_counts=roi_counts,
        environment_sizes_m=environment_sizes_m,
        episode_horizons_s=episode_horizons_s,
        collect_search_diagnostics=args.collect_search_diagnostics,
        deadline_seconds=args.deadline_seconds,
        bandwidth_mhz=args.bandwidth_mhz,
        sr_tx_power_dbm=args.sr_tx_power_dbm,
        uav_tx_power_dbm=args.uav_tx_power_dbm,
        sr_speeds_mps=args.sr_speeds_mps,
        allow_registered_fixed_roi_method=bool(
            args.suite == "fixed_roi" and (explicit_checkpoint or explicit_roi)
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
