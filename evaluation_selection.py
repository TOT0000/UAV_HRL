"""Canonical checkpoint and RoI selectors for evaluation workflows."""

from __future__ import annotations

import json
from pathlib import Path

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from com_capacity_calibration import load_com_capacity_reference
from experiment_config import (
    FORMAL_CHECKPOINT_EPISODE,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    MethodSpec,
)
from HRL_task_aware import ROUTING_STATE_DIM
from scenario_manifest import (
    ScenarioManifest,
    resolve_training_manifest_segment,
    resolve_training_manifest_segments_from_metadata,
)
from training_checkpoint import (
    checkpoint_artifact_provenance,
    inspect_model_checkpoint,
)


DEFAULT_FIXED_ROI_COUNTS = tuple(range(ROI_COUNT_MIN, ROI_COUNT_MAX + 1))


def _exclusive_values(single, multiple, *, label, default):
    if single is not None and multiple is not None:
        raise ValueError(f"use either --{label} or --{label}s, not both")
    values = multiple if multiple is not None else (
        (single,) if single is not None else default
    )
    if isinstance(values, (str, bytes)):
        values = (values,)
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{label.replace('-', ' ')} must be an integer")
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label.replace('-', ' ')} must be an integer"
            ) from exc
        if resolved != value:
            raise ValueError(f"{label.replace('-', ' ')} must be an integer")
        if resolved not in result:
            result.append(resolved)
    if not result:
        raise ValueError(f"at least one {label.replace('-', ' ')} is required")
    return tuple(result)


def resolve_checkpoint_episodes(
    checkpoint_episode=None, checkpoint_episodes=None
):
    """Resolve singular/batch checkpoint selectors with the formal default."""

    episodes = _exclusive_values(
        checkpoint_episode,
        checkpoint_episodes,
        label="checkpoint-episode",
        default=(FORMAL_CHECKPOINT_EPISODE,),
    )
    invalid = [episode for episode in episodes if episode <= 0]
    if invalid:
        raise ValueError(
            f"checkpoint episodes must be positive integers: {invalid}"
        )
    return episodes


def resolve_roi_counts(roi_count=None, roi_counts=None):
    """Resolve singular/batch RoI selectors with the fixed-paper default."""

    counts = _exclusive_values(
        roi_count,
        roi_counts,
        label="roi-count",
        default=DEFAULT_FIXED_ROI_COUNTS,
    )
    invalid = [
        count
        for count in counts
        if not ROI_COUNT_MIN <= count <= ROI_COUNT_MAX
    ]
    if invalid:
        raise ValueError(
            "RoI counts must be in the inclusive range "
            f"[{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]: {invalid}"
        )
    return counts


def _read_json_object(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _training_total_episodes(resolved):
    config = resolved.get("training_config")
    if not isinstance(config, dict):
        raise RuntimeError("training run resolved config lacks training_config")
    try:
        total = int(config["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "training run resolved config has invalid total_episodes"
        ) from exc
    if total <= 0:
        raise RuntimeError("training total episodes must be positive")
    if resolved.get("episodes") is not None and int(resolved["episodes"]) != total:
        raise RuntimeError(
            "training run episode count disagrees with resolved training config"
        )
    return total, dict(config)


def resolve_training_run_checkpoint(
    run_directory,
    checkpoint_episode,
    *,
    expected_method=None,
    require_run_metadata=False,
):
    """Validate one run/checkpoint without loading policy weights."""

    if run_directory is None:
        raise ValueError("a learned checkpoint evaluation requires --run-dir")
    run_dir = Path(run_directory).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"training run directory is missing: {run_dir}")
    resolved = _read_json_object(
        run_dir / "resolved_config.json", "training resolved config"
    )
    run_metadata_path = run_dir / "run_metadata.json"
    run_metadata = (
        _read_json_object(run_metadata_path, "training run metadata")
        if require_run_metadata or run_metadata_path.is_file()
        else None
    )
    try:
        method = MethodSpec.parse(resolved["method"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "training run method is missing or absent from the current registry"
        ) from exc
    if expected_method is not None:
        expected = MethodSpec.parse(expected_method)
        if method.method_id != expected.method_id:
            raise RuntimeError(
                "paper method/run mismatch: "
                f"requested={expected.method_id}, run={method.method_id}"
            )
    if resolved.get("method_spec") != method.to_dict():
        raise RuntimeError("training run method metadata is incompatible")
    if resolved.get("status") != "COMPLETED":
        raise RuntimeError("evaluation requires a completed training run")
    if run_metadata is not None:
        metadata_method = run_metadata.get(
            "method_id", run_metadata.get("method")
        )
        if metadata_method is not None and metadata_method != method.method_id:
            raise RuntimeError(
                "training run metadata method disagrees with resolved config"
            )
    if not (method.learns_movement or method.learns_routing):
        raise ValueError(
            f"{method.method_id} has no learned checkpoint to evaluate"
        )
    training_total, expected_training_config = _training_total_episodes(resolved)
    (
        training_manifest_path,
        training_manifest,
        training_manifest_segments,
    ) = resolve_training_manifest_segments_from_metadata(run_dir, resolved)
    if training_manifest.episode_count != training_total:
        raise RuntimeError(
            "training manifest length disagrees with resolved training horizon"
        )
    checkpoint_episode = resolve_checkpoint_episodes(
        checkpoint_episode=checkpoint_episode
    )[0]
    if checkpoint_episode > training_total:
        raise ValueError(
            "checkpoint episode exceeds training total episodes: "
            f"checkpoint={checkpoint_episode}, training_total={training_total}"
        )
    checkpoint = (
        run_dir
        / "checkpoints"
        / "models"
        / f"ep_{checkpoint_episode:04d}"
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"checkpoint ep_{checkpoint_episode:04d} is missing: {checkpoint}"
        )
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
        current_training_manifest_segments=training_manifest_segments,
        training_run_directory=run_dir,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
    )
    if int(inspected["completed_episode"]) != checkpoint_episode:
        raise RuntimeError(
            "checkpoint completed episode disagrees with selector: "
            f"checkpoint={inspected['completed_episode']}, "
            f"selected={checkpoint_episode}"
        )
    artifact_provenance = checkpoint_artifact_provenance(
        checkpoint, metadata=inspected["metadata"]
    )
    return {
        "run_dir": run_dir,
        "training_run_id": run_dir.name,
        "resolved": resolved,
        "run_metadata": run_metadata,
        "method": method,
        "training_seed": int(resolved["seed"]),
        "training_total_episodes": training_total,
        "checkpoint_episode": checkpoint_episode,
        "checkpoint": checkpoint.resolve(),
        "checkpoint_metadata": inspected["metadata"],
        "checkpoint_artifact_provenance": artifact_provenance,
        "training_manifest": training_manifest,
        "training_manifest_path": training_manifest_path,
        "training_manifest_segments": training_manifest_segments,
        "checkpoint_training_manifest_segment": resolve_training_manifest_segment(
            run_dir,
            training_manifest_segments,
            checkpoint_episode,
            current_total_episodes=training_total,
        ),
        **(inspected.get("horizon_compatibility") or {}),
        "expected_training_config": expected_training_config,
        "checkpoint_required": True,
    }


def validate_fixed_roi_manifest(manifest, roi_count, episodes, manifest_seed):
    """Assert a shared manifest is exactly the requested fixed-RoI scenario set."""

    if not isinstance(manifest, ScenarioManifest):
        raise TypeError("shared fixed-RoI manifest must be a ScenarioManifest")
    roi_count = resolve_roi_counts(roi_count=roi_count)[0]
    if manifest.split != "test":
        raise ValueError("checkpoint RoI sweep requires test manifests")
    if int(manifest.manifest_seed) != int(manifest_seed):
        raise ValueError("shared fixed-RoI manifest seed is incompatible")
    if int(manifest.episode_count) != int(episodes):
        raise ValueError("shared fixed-RoI manifest episode count is incompatible")
    if manifest.generation_profile.get("fixed_num_gt") != roi_count:
        raise ValueError("shared manifest fixed RoI count is incompatible")
    for entry in manifest.episodes:
        if (
            int(entry.get("num_GT", -1)) != roi_count
            or len(entry.get("ground_targets", ())) != roi_count
            or len(entry.get("sr_teams", ())) != roi_count
        ):
            raise ValueError(
                "shared manifest scenario has an incompatible actual RoI count"
            )
    return manifest
