"""Unified CLI for manifest-driven comparison experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path

from com_capacity_calibration import load_com_capacity_reference
from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from dinkelbach_blocks import DINKELBACH_TRAINING_STATE_FIELDS
from experiment_config import (
    DEFAULT_TRAINING_SEED,
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
)
from experiment_paths import (
    prepare_run_directory,
    training_run_directory,
    training_run_identity,
    validate_run_directory_preflight,
    write_run_status,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    formal_training_config,
    smoke_training_config,
    train,
)
from resume_recovery import (
    execute_resume_reconciliation,
    plan_resume_reconciliation,
)
from scenario_manifest import ScenarioManifest, generate_manifest
from training_checkpoint import (
    FULL_RESUME_LOGGING_STATE_FIELDS,
    inspect_full_resume_checkpoint,
    inspect_model_checkpoint,
)
from training_history import (
    preflight_resume_training_history,
    training_history_identity,
)


DEFAULT_OUTPUT_DIR = Path("runs") / "comparison"


def _method(value):
    try:
        return MethodSpec.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _num_gt(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("num_GT must be an integer") from exc
    if not ROI_COUNT_MIN <= number <= ROI_COUNT_MAX:
        raise argparse.ArgumentTypeError(
            f"num_GT must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
        )
    return number


def _load_manifest(path, expected_split=None):
    if path is None:
        raise ValueError("this command requires --manifest")
    manifest = ScenarioManifest.load(path)
    if expected_split is not None and manifest.split != expected_split:
        raise ValueError(
            f"manifest split mismatch: manifest={manifest.split}, "
            f"requested={expected_split}"
        )
    return manifest


def _write_run_metadata(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    path.write_text(
        json.dumps(result["run_metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def command_generate_manifest(args):
    manifest = generate_manifest(
        args.split, args.manifest_seed, args.episodes, num_gt=args.num_gt
    )
    path = Path(args.manifest) if args.manifest else (
        Path(args.output_dir)
        / "manifests"
        / (
            f"{args.split}.json"
            if args.num_gt is None
            else f"{args.split}-num-gt-{args.num_gt}.json"
        )
    )
    manifest.save(path)
    print(
        json.dumps(
            {"manifest": str(path), "content_hash": manifest.content_hash},
            ensure_ascii=False,
        )
    )
    return 0


def command_smoke(args):
    method = args.method
    manifest = (
        _load_manifest(args.manifest, expected_split=args.split)
        if args.manifest
        else generate_manifest(
            args.split,
            manifest_seed=DEFAULT_TRAINING_SEED,
            episode_count=args.episodes,
        )
    )
    config = replace(
        smoke_training_config(),
        total_episodes=args.episodes,
        random_seed=args.training_seed,
    )
    result = train(config, scenario_manifest=manifest, method_spec=method)
    _write_run_metadata(args.output_dir, result)
    print(json.dumps(result["run_metadata"], ensure_ascii=False))
    return 0


def command_train(args):
    preflight = _training_preflight(args)
    method = preflight["method"]
    manifest = preflight["manifest"]
    identity = preflight["identity"]
    run_dir = prepare_run_directory(
        preflight["run_directory"],
        identity,
        resume_checkpoint=args.resume,
    )
    try:
        reconciliation_plan = preflight["reconciliation_plan"]
        reconciliation = (
            execute_resume_reconciliation(reconciliation_plan)
            if reconciliation_plan is not None
            else None
        )
        write_run_status(run_dir, "RUNNING")
        result = train(
            preflight["config"],
            scenario_manifest=manifest,
            method_spec=method,
        )
        result["run_metadata"].update(
            {
                "run_directory": str(run_dir),
                "run_identity": identity,
                "run_status_file": str(run_dir / "run_status.json"),
            }
        )
        if reconciliation is not None:
            result["run_metadata"]["resume_reconciliation"] = reconciliation
        _write_run_metadata(run_dir, result)
        write_run_status(run_dir, "COMPLETED")
    except BaseException as exc:
        try:
            write_run_status(run_dir, "FAILED", exception=exc)
        except BaseException:
            pass
        raise
    return 0


def _training_preflight(args):
    """Validate every formal training input before canonical run creation."""

    method = args.method
    MethodSpec(**{
        key: value
        for key, value in method.to_dict().items()
        if key != "method_id"
    })
    manifest = _load_manifest(args.manifest, expected_split=args.split)
    if args.checkpoint is not None:
        raise ValueError(
            "training checkpoint output is managed under the canonical run "
            "directory; do not pass --checkpoint"
        )
    if manifest.episode_count < int(args.episodes):
        raise ValueError(
            "scenario manifest has fewer entries than requested episodes"
        )
    identity = training_run_identity(method, manifest, args.training_seed)
    run_dir = training_run_directory(
        args.output_dir, method, manifest, args.training_seed
    )
    config = formal_training_config(
        args.episodes,
        random_seed=args.training_seed,
        resume_dir=args.resume,
        checkpoint_root=str(run_dir / "checkpoints"),
        enable_plots=False,
        enable_csv=False,
        run_directory=str(run_dir),
    )
    _, calibration = load_com_capacity_reference()
    validate_run_directory_preflight(
        run_dir, identity, resume_checkpoint=args.resume
    )
    reconciliation_plan = None
    if args.resume is not None:
        expected_experiment = {
            "method_spec_fingerprint": method.fingerprint,
            "manifest_hash": manifest.content_hash,
            "training_seed": int(args.training_seed),
        }
        inspect_full = partial(
            inspect_full_resume_checkpoint,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            td3_gamma=1.0,
            ddqn_gamma=0.99,
            calibration=calibration,
            expected_experiment_metadata=expected_experiment,
            expected_formal_config=asdict(config),
            require_episode_directory=True,
        )
        inspect_model = partial(
            inspect_model_checkpoint,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            td3_gamma=1.0,
            ddqn_gamma=0.99,
            calibration=calibration,
            expected_experiment_metadata=expected_experiment,
            expected_formal_config=asdict(config),
            require_episode_directory=True,
        )
        reconciliation_plan = plan_resume_reconciliation(
            run_dir,
            args.resume,
            inspect_full=inspect_full,
            inspect_model=inspect_model,
        )
        training_state = reconciliation_plan.resume_training_state
        required_training_state = {
            "total_joint_transitions",
            "global_routing_slot",
            "td3_post_warmup_transition",
            "ddqn_schedule_slot",
            "td3_noise_log",
            "routing_epsilon_log",
            "training_history_rows",
        } | set(FULL_RESUME_LOGGING_STATE_FIELDS)
        if method.uses_dinkelbach:
            required_training_state |= set(DINKELBACH_TRAINING_STATE_FIELDS)
        missing = required_training_state.difference(training_state)
        if missing:
            raise RuntimeError(
                "exact-resume checkpoint training state is incomplete: "
                f"{sorted(missing)}"
            )
        history_rows = preflight_resume_training_history(
            run_dir,
            training_history_identity(
                method.method_id, args.training_seed, manifest.content_hash
            ),
            checkpoint_rows=training_state["training_history_rows"],
        )
        if len(history_rows) != reconciliation_plan.resume_episode:
            raise RuntimeError(
                "exact-resume checkpoint history length does not match its "
                f"episode: rows={len(history_rows)}, "
                f"episode={reconciliation_plan.resume_episode}"
            )
    return {
        "method": method,
        "manifest": manifest,
        "identity": identity,
        "run_directory": Path(run_dir).resolve(),
        "config": config,
        "calibration": calibration,
        "reconciliation_plan": reconciliation_plan,
    }


def command_evaluate(args):
    from evaluation_metrics import run_evaluation_command

    return run_evaluation_command(args)


def command_collect_design_dataset(args):
    from design_dataset import run_design_dataset_command

    return run_design_dataset_command(args)


def command_aggregate(args):
    from evaluation_metrics import run_aggregate_command

    return run_aggregate_command(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manifest-driven corrected UAV-HRL comparison runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-manifest")
    generate.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    generate.add_argument("--manifest")
    generate.add_argument("--manifest-seed", type=int, required=True)
    generate.add_argument("--episodes", type=int, required=True)
    generate.add_argument("--num-gt", type=_num_gt)
    generate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    generate.set_defaults(handler=command_generate_manifest)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--method", type=_method, default=MethodSpec())
    smoke.add_argument("--split", default="test", choices=("train", "validation", "test"))
    smoke.add_argument("--manifest")
    smoke.add_argument(
        "--training-seed", type=int, default=DEFAULT_TRAINING_SEED
    )
    smoke.add_argument("--episodes", type=int, default=1)
    smoke.add_argument("--checkpoint")
    smoke.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "smoke"))
    smoke.add_argument("--resume")
    smoke.set_defaults(handler=command_smoke)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--method", type=_method, default=MethodSpec())
    train_parser.add_argument("--split", default="train", choices=("train",))
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--training-seed", type=int, required=True)
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"],
    )
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    train_parser.add_argument("--resume")
    train_parser.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--method", type=_method, default=MethodSpec())
    evaluate.add_argument("--split", default="test", choices=("validation", "test"))
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--training-seed", type=int, required=True)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    evaluate.add_argument("--resume")
    evaluate.set_defaults(handler=command_evaluate)

    design = subparsers.add_parser("collect-design-dataset")
    design.add_argument("--method", type=_method, default=MethodSpec())
    design.add_argument(
        "--split", default="validation", choices=("validation", "test")
    )
    design.add_argument("--manifest", required=True)
    design.add_argument("--training-seed", type=int, required=True)
    design.add_argument(
        "--episodes",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS[
            "evaluation_episodes_per_trained_seed"
        ],
    )
    design.add_argument("--checkpoint", required=True)
    design.add_argument("--output-dir", required=True)
    design.add_argument("--reference-per-episode")
    design.set_defaults(handler=command_collect_design_dataset)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--method", type=_method, default=MethodSpec())
    aggregate.add_argument("--split", default="test", choices=("validation", "test"))
    aggregate.add_argument("--manifest")
    aggregate.add_argument("--training-seed", type=int)
    aggregate.add_argument("--episodes", type=int)
    aggregate.add_argument("--checkpoint")
    aggregate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "aggregate"))
    aggregate.add_argument("--resume")
    aggregate.add_argument("--input-dir", required=True)
    aggregate.add_argument(
        "--expected-seed-count",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS["training_seed_count"],
    )
    aggregate.add_argument(
        "--expected-episodes-per-seed",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS[
            "evaluation_episodes_per_trained_seed"
        ],
    )
    aggregate.set_defaults(handler=command_aggregate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
