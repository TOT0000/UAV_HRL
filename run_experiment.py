"""Train, exactly resume, or evaluate one registered trajectory method."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from com_capacity_calibration import load_com_capacity_reference
from evaluation_metrics import write_evaluation_outputs
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    FORMAL_CHECKPOINT_EPISODE,
    METHOD_REGISTRY,
    MethodSpec,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    ROUTING_WARMUP_TRANSITIONS,
    effective_training_config,
    exploration_schedule_configuration,
    comparison_method_configuration,
    movement_agent_configuration,
    routing_agent_configuration,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    TrainingConfig,
    formal_training_config,
    train,
)
from observation_strategy import masked_observation_metadata
from resume_recovery import (
    execute_resume_reconciliation,
    plan_resume_reconciliation,
)
from scenario_manifest import (
    ScenarioManifest,
    extend_training_manifest,
    generate_manifest,
    resolve_training_manifest,
)
from training_checkpoint import (
    inspect_full_resume_checkpoint,
    inspect_model_checkpoint,
)
from training_history import (
    preflight_resume_training_history,
    training_history_identity,
)


def _git_short_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_unique_leaf(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-{attempt:02d}"
        candidate = parent / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate.resolve()
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique directory below {parent}")


def create_unique_run_directory(output_root, method_key, seed, git_sha=None):
    """Atomically create a new leaf; a collision is never silently reused."""

    parent = Path(output_root) / str(method_key)
    sha = str(git_sha or _git_short_sha())[:12]
    return _create_unique_leaf(parent, f"run-seed{int(seed)}-{sha}")


def _roi_count(value):
    value = int(value)
    if not ROI_COUNT_MIN <= value <= ROI_COUNT_MAX:
        raise argparse.ArgumentTypeError(
            f"RoI count must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
        )
    return value


def _add_training_options(parser):
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument("--roi-count", type=_roi_count)
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one non-formal episode/transition by default, without checkpoints",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one controlled trajectory experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for key in METHOD_REGISTRY:
        method_parser = commands.add_parser(key, help=MethodSpec.parse(key).label)
        _add_training_options(method_parser)
        method_parser.set_defaults(command="train", method=MethodSpec.parse(key))

    resume = commands.add_parser("resume", help="exactly resume one run directory")
    resume.add_argument("run_directory")
    resume.add_argument(
        "--target-episodes",
        type=int,
        help="explicitly extend the planned training horizon before exact resume",
    )
    resume.set_defaults(method=None)

    evaluate = commands.add_parser(
        "evaluate", help="evaluate one checkpoint inferred from a run directory"
    )
    evaluate.add_argument("run_directory")
    evaluate.add_argument("--checkpoint-episode", type=int, default=FORMAL_CHECKPOINT_EPISODE)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--episode-seconds", type=int)
    evaluate.add_argument("--manifest-seed", type=int)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument(
        "--smoke",
        action="store_true",
        help="run a one-transition non-formal evaluation unless explicitly overridden",
    )
    evaluate.set_defaults(method=None)
    return parser


def _resolved_values(args):
    defaults = FORMAL_EXPERIMENT_DEFAULTS
    episodes = args.episodes
    if episodes is None:
        episodes = 1 if args.smoke else defaults["training_episodes_per_seed"]
    episode_seconds = args.episode_seconds
    if episode_seconds is None:
        episode_seconds = 1 if args.smoke else defaults["episode_seconds"]
    seed = defaults["training_seed"] if args.seed is None else args.seed
    checkpoint_interval = (
        defaults["checkpoint_interval_episodes"]
        if args.checkpoint_interval is None
        else args.checkpoint_interval
    )
    output_root = args.output_root or defaults["output_root"]
    if int(episodes) <= 0 or int(episode_seconds) <= 0:
        raise ValueError("episodes and episode-seconds must be positive")
    if int(checkpoint_interval) <= 0:
        raise ValueError("checkpoint interval must be positive")
    return {
        "episodes": int(episodes),
        "episode_seconds": int(episode_seconds),
        "seed": int(seed),
        "checkpoint_interval": int(checkpoint_interval),
        "output_root": str(output_root),
    }


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"run metadata is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"run metadata must be an object: {path}")
    return value


def _load_run_context(run_directory):
    run_dir = Path(run_directory).resolve()
    resolved = _read_json(run_dir / "resolved_config.json")
    method = MethodSpec.parse(resolved["method"])
    if resolved.get("method_spec") != method.to_dict():
        raise RuntimeError("run method metadata is incompatible with the registry")
    _, manifest = resolve_training_manifest(run_dir, resolved)
    if int(resolved["seed"]) != int(manifest.manifest_seed):
        raise RuntimeError("run seed is incompatible with its scenario manifest")
    return run_dir, resolved, method, manifest


def _training_config_from_resolved(resolved, **overrides):
    allowed = {field.name for field in fields(TrainingConfig)}
    values = {
        key: value
        for key, value in resolved["training_config"].items()
        if key in allowed
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _base_resolved(run_dir, method, manifest, config, values, args, git_sha):
    comparison = comparison_method_configuration(method)
    return {
        "status": "RUNNING",
        "method": method.method_key,
        "method_spec": method.to_dict(),
        "agent": method.agent,
        "reward_mode": method.reward_mode,
        "task_potential_enabled": method.task_potential_enabled,
        **comparison,
        "masked_state_fields": (
            masked_observation_metadata()
            if method.task_observation == "masked"
            else None
        ),
        "seed": values["seed"],
        "episodes": values["episodes"],
        "num_uav": FORMAL_EXPERIMENT_DEFAULTS["num_uav"],
        "roi_count": args.roi_count if args.roi_count is not None else [ROI_COUNT_MIN, ROI_COUNT_MAX],
        "checkpoint_interval": values["checkpoint_interval"],
        "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
        "movement_hyperparameters": FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"],
        "effective_movement_agent_configuration": movement_agent_configuration(
            method, config
        ),
        "effective_routing_agent_configuration": routing_agent_configuration(
            method, config
        ),
        "exploration_schedule_configuration": exploration_schedule_configuration(
            config, method
        ),
        "training_config": effective_training_config(config, method),
        "training_manifest_hash": manifest.content_hash,
        "training_manifest_path": "scenario_manifest.json",
        "training_history_manifest_hash": manifest.content_hash,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "run_directory": str(run_dir),
    }


def run(args):
    """Create and train one new isolated run."""

    values = _resolved_values(args)
    method = args.method
    git_sha = _git_short_sha()
    run_dir = create_unique_run_directory(
        values["output_root"], method.method_key, values["seed"], git_sha
    )
    checkpoints_enabled = not args.smoke
    config = formal_training_config(
        values["episodes"],
        mode="smoke" if args.smoke else "train",
        episode_seconds=values["episode_seconds"],
        random_seed=values["seed"],
        warmup_joint_transitions=(
            0
            if args.smoke
            else FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["warmup_joint_transitions"]
        ),
        routing_warmup_transitions=(
            1 if args.smoke else ROUTING_WARMUP_TRANSITIONS
        ),
        batch_size=(
            1
            if args.smoke
            else FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["batch_size"]
        ),
        replay_max_size=FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["replay_size"],
        model_checkpoint_every=values["checkpoint_interval"],
        full_resume_every=values["checkpoint_interval"],
        checkpoint_root=str(run_dir / "checkpoints"),
        enable_model_checkpoints=checkpoints_enabled,
        enable_full_resume=checkpoints_enabled,
        enable_plots=False,
        enable_csv=False,
        run_directory=str(run_dir),
    )
    manifest = generate_manifest(
        "train",
        manifest_seed=values["seed"],
        episode_count=values["episodes"],
        num_gt=args.roi_count,
    )
    manifest.save(run_dir / "scenario_manifest.json")
    resolved = _base_resolved(run_dir, method, manifest, config, values, args, git_sha)
    _write_json(run_dir / "resolved_config.json", resolved)
    try:
        result = train(config, scenario_manifest=manifest, method_spec=method)
        resolved.update(
            status="COMPLETED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            history_rows=len(result["training_history_rows"]),
            dinkelbach_update_count=result["dinkelbach_update_count"],
        )
        _write_json(run_dir / "run_metadata.json", {**result["run_metadata"], **resolved})
        _write_json(run_dir / "resolved_config.json", resolved)
    except BaseException as exc:
        resolved.update(
            status="FAILED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_json(run_dir / "resolved_config.json", resolved)
        raise
    print(json.dumps({"run_directory": str(run_dir), "status": "COMPLETED"}))
    return 0


def _episode_directories(root):
    candidates = []
    root = Path(root)
    if root.is_dir():
        for path in root.iterdir():
            match = re.fullmatch(r"ep_(\d+)", path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path.resolve()))
    return sorted(candidates, reverse=True)


def run_resume(args):
    run_dir, resolved, method, manifest = _load_run_context(args.run_directory)
    try:
        previous_total = int(resolved["training_config"]["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("run training horizon is invalid") from exc
    if previous_total <= 0 or manifest.episode_count != previous_total:
        raise RuntimeError(
            "run training manifest length disagrees with its planned horizon"
        )

    target_episodes = args.target_episodes
    extension_provenance = None
    active_manifest = manifest
    active_manifest_path, _ = resolve_training_manifest(run_dir, resolved)
    if target_episodes is not None:
        target_episodes = int(target_episodes)
        if target_episodes <= previous_total:
            raise ValueError(
                "target-episodes must exceed the current planned training horizon: "
                f"target={target_episodes}, current={previous_total}"
            )
        active_manifest, manifest_provenance = extend_training_manifest(
            manifest, target_episodes
        )
        relative_manifest_path = Path("scenario_manifests") / (
            f"train_ep_{target_episodes:04d}_{active_manifest.content_hash[:12]}.json"
        )
        active_manifest_path = (run_dir / relative_manifest_path).resolve()
        if active_manifest_path.exists():
            recovered = ScenarioManifest.load(active_manifest_path)
            if recovered.content_hash != active_manifest.content_hash:
                raise FileExistsError(
                    "existing extended manifest path has incompatible content: "
                    f"{active_manifest_path}"
                )
        extension_provenance = {
            "previous_total_episodes": previous_total,
            "target_total_episodes": target_episodes,
            **manifest_provenance,
            "previous_manifest_path": str(
                resolve_training_manifest(run_dir, resolved)[0]
            ),
            "extended_manifest_path": str(active_manifest_path),
        }

    config = _training_config_from_resolved(
        resolved,
        **(
            {"total_episodes": target_episodes}
            if target_episodes is not None
            else {}
        ),
        resume_dir=None,
        checkpoint_root=str(run_dir / "checkpoints"),
        run_directory=str(run_dir),
        enable_plots=False,
        enable_csv=False,
    )
    formal_config = effective_training_config(config, method)
    _, calibration = load_com_capacity_reference()
    expected_experiment = {
        "method_spec_fingerprint": method.compatible_fingerprints,
        "manifest_hash": active_manifest.content_hash,
        "training_seed": int(resolved["seed"]),
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
        expected_formal_config=formal_config,
        current_training_manifest=active_manifest,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
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
        expected_formal_config=formal_config,
        current_training_manifest=active_manifest,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
    )
    checkpoint = None
    failures = []
    for _, candidate in _episode_directories(run_dir / "checkpoints" / "full"):
        try:
            inspect_full(candidate)
            checkpoint = candidate
            break
        except RuntimeError as exc:
            if "legacy ratio checkpoint uses per-transition B/E" in str(exc):
                raise
            failures.append(f"{candidate.name}: {exc}")
    if checkpoint is None:
        detail = "; ".join(failures) if failures else "no checkpoint directories"
        raise RuntimeError(f"no valid full-resume checkpoint found: {detail}")

    plan = plan_resume_reconciliation(
        run_dir,
        checkpoint,
        inspect_full=inspect_full,
        inspect_model=inspect_model,
    )
    if target_episodes is not None and target_episodes <= plan.resume_episode:
        raise ValueError(
            "target-episodes must exceed the latest completed episode: "
            f"target={target_episodes}, completed={plan.resume_episode}"
        )
    history_manifest_hash = str(
        resolved.get("training_history_manifest_hash", manifest.content_hash)
    )
    preflight_resume_training_history(
        run_dir,
        training_history_identity(
            method.method_id,
            int(resolved["seed"]),
            history_manifest_hash,
        ),
        checkpoint_rows=plan.resume_training_state.get(
            "training_history_rows", ()
        ),
    )
    reconciliation = execute_resume_reconciliation(plan)
    if extension_provenance is not None:
        if not active_manifest_path.exists():
            active_manifest.save_atomic(active_manifest_path)
        extension_provenance.update(
            extended_at=datetime.now(timezone.utc).isoformat(),
            resume_checkpoint=str(checkpoint),
            resume_episode=plan.resume_episode,
        )
    config.resume_dir = str(checkpoint)
    resolved.update(
        status="RUNNING",
        resumed_at=datetime.now(timezone.utc).isoformat(),
        resume_checkpoint=str(checkpoint),
        resume_episode=plan.resume_episode,
        episodes=int(config.total_episodes),
        training_config=formal_config,
        training_manifest_hash=active_manifest.content_hash,
        training_manifest_path=str(active_manifest_path.relative_to(run_dir)),
        training_history_manifest_hash=history_manifest_hash,
        effective_movement_agent_configuration=movement_agent_configuration(
            method, config
        ),
    )
    if extension_provenance is not None:
        history = list(resolved.get("horizon_extension_history") or ())
        history.append(extension_provenance)
        resolved.update(
            horizon_extension_provenance=extension_provenance,
            horizon_extension_history=history,
        )
    _write_json_atomic(run_dir / "resolved_config.json", resolved)
    try:
        result = train(
            config,
            scenario_manifest=active_manifest,
            method_spec=method,
            training_history_manifest_hash=history_manifest_hash,
        )
        resolved.update(
            status="COMPLETED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            history_rows=len(result["training_history_rows"]),
            dinkelbach_update_count=result["dinkelbach_update_count"],
            resume_reconciliation=reconciliation,
        )
        _write_json(run_dir / "run_metadata.json", {**result["run_metadata"], **resolved})
        _write_json_atomic(run_dir / "resolved_config.json", resolved)
    except BaseException as exc:
        resolved.update(
            status="FAILED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_json_atomic(run_dir / "resolved_config.json", resolved)
        raise
    print(json.dumps({"run_directory": str(run_dir), "status": "COMPLETED", "resumed_from": str(checkpoint)}))
    return 0


def _write_evaluation_plots(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir()
    x = list(range(1, len(rows) + 1))
    metrics = {
        "energy_efficiency_mbit_per_j": "Energy efficiency (Mbit/J)",
        "timely_goodput_mbits": "Timely goodput (Mbit)",
        "total_mobility_energy_j": "Mobility energy (J)",
    }
    paths = []
    for key, label in metrics.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(x, [float(row[key]) for row in rows])
        axis.set(xlabel="Evaluation episode", ylabel=label)
        axis.grid(True)
        figure.tight_layout()
        path = plot_dir / f"{key}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def run_evaluate(args):
    run_dir, resolved, method, training_manifest = _load_run_context(args.run_directory)
    checkpoint_episode = int(args.checkpoint_episode)
    if checkpoint_episode <= 0:
        raise ValueError("checkpoint episode must be positive")
    checkpoint = run_dir / "checkpoints" / "models" / f"ep_{checkpoint_episode:04d}"
    evaluation_episodes = (
        int(args.episodes)
        if args.episodes is not None
        else (1 if args.smoke else FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"])
    )
    episode_seconds = (
        int(args.episode_seconds)
        if args.episode_seconds is not None
        else (1 if args.smoke else FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"])
    )
    if evaluation_episodes <= 0 or episode_seconds <= 0:
        raise ValueError("evaluation episodes and episode-seconds must be positive")
    manifest_seed = (
        int(args.manifest_seed)
        if args.manifest_seed is not None
        else int(resolved["seed"])
    )
    fixed_num_gt = training_manifest.generation_profile.get("fixed_num_gt")
    manifest = generate_manifest(
        args.split,
        manifest_seed=manifest_seed,
        episode_count=evaluation_episodes,
        num_gt=fixed_num_gt,
    )
    expected_training_config = dict(resolved["training_config"])
    _, calibration = load_com_capacity_reference()
    inspected = inspect_model_checkpoint(
        checkpoint,
        movement_state_dim=MOVEMENT_STATE_DIM,
        joint_action_dim=JOINT_ACTION_DIM,
        routing_state_dim=ROUTING_STATE_DIM,
        td3_gamma=1.0,
        ddqn_gamma=0.99,
        calibration=calibration,
        expected_experiment_metadata={
            "method_spec_fingerprint": method.compatible_fingerprints,
            "training_seed": int(resolved["seed"]),
        },
        expected_completed_episodes=checkpoint_episode,
        expected_formal_config=expected_training_config,
        current_training_manifest=training_manifest,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
    )
    output_dir = _create_unique_leaf(
        run_dir / "evaluation" / f"ep_{checkpoint_episode}", "eval"
    )
    manifest.save(output_dir / "scenario_manifest.json")
    config = TrainingConfig(
        total_episodes=evaluation_episodes,
        mode="custom",
        episode_seconds=episode_seconds,
        routing_slot_seconds=FORMAL_EXPERIMENT_DEFAULTS["routing_slot_seconds"],
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=int(resolved["seed"]),
    )
    result = train(
        config,
        scenario_manifest=manifest,
        method_spec=method,
        evaluation=True,
        checkpoint_dir=checkpoint,
        expected_checkpoint_episodes=checkpoint_episode,
        expected_checkpoint_formal_config=expected_training_config,
        expected_checkpoint_training_manifest=training_manifest,
    )
    formal_evaluation = bool(
        not args.smoke
        and checkpoint_episode == FORMAL_CHECKPOINT_EPISODE
        and evaluation_episodes
        == FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
        and episode_seconds == FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"]
    )
    metadata = {
        **result["run_metadata"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_episode": checkpoint_episode,
        **inspected["horizon_compatibility"],
        "evaluation_invariants": result["evaluation_invariants"],
        "evaluation_manifest": str((output_dir / "scenario_manifest.json").resolve()),
        "evaluation_run_directory": str(output_dir),
        "formal_evaluation": formal_evaluation,
        "smoke_evaluation": bool(args.smoke),
    }
    paths = write_evaluation_outputs(output_dir, result["episode_metrics"], metadata)
    plot_paths = _write_evaluation_plots(output_dir, result["episode_metrics"])
    print(json.dumps({"evaluation_directory": str(output_dir), "outputs": {key: str(value) for key, value in paths.items()}, "plots": plot_paths}))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return run(args)
    if args.command == "resume":
        return run_resume(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
