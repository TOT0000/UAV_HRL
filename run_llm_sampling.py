"""Collect enriched movement replay from one compatible TD3/Safe-DDQN model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from observation_strategy import ROUTING_STATE_DIM
from replay_auxiliary import replay_auxiliary_metadata
from scenario_manifest import ScenarioManifest, generate_manifest
from training_checkpoint import (
    MODEL_CHECKPOINT_TYPE,
    checkpoint_artifact_provenance,
)
from utils_update_v2 import ReplayBufferJoint


DEFAULT_OUTPUT_ROOT = Path("results/llm_samples")
SAMPLING_METADATA_SCHEMA_VERSION = "uav-hrl-llm-sampling-v1"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Collect auxiliary movement replay without training or invoking an LLM"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--sampling-seed", type=int, default=20260927)
    parser.add_argument("--manifest", help="reuse an existing test/validation manifest")
    parser.add_argument("--manifest-output", help="optional additional manifest copy")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    metadata_path = checkpoint / "metadata.json"
    models_path = checkpoint / "models.pt"
    if not metadata_path.is_file() or not models_path.is_file():
        raise FileNotFoundError(
            "checkpoint must contain metadata.json and models.pt: "
            f"{checkpoint}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_type") != MODEL_CHECKPOINT_TYPE:
        raise ValueError("sampling requires a model-only models/ep_xxxx checkpoint")
    experiment = metadata.get("experiment") or {}
    method_payload = experiment.get("method_spec") or {}
    method_id = (
        method_payload.get("method_id")
        or method_payload.get("method_key")
        or experiment.get("method_id")
    )
    if not method_id:
        raise RuntimeError("checkpoint metadata has no method id")
    method = MethodSpec.parse(method_id)
    if method.agent != "td3" or method.routing != "safe_ddqn":
        raise ValueError(
            "unsupported sampling checkpoint: expected TD3 movement with "
            f"safe-DDQN routing, got {method.agent}/{method.routing}"
        )
    training_seed = experiment.get("training_seed")
    if training_seed is None:
        raise RuntimeError("checkpoint metadata has no training seed")
    formal_config = experiment.get("formal_config") or {}
    episode_seconds = int(formal_config.get("episode_seconds", 0))
    if episode_seconds <= 0:
        raise RuntimeError("checkpoint metadata has no valid episode horizon")
    completed_episodes = int(metadata.get("episode", -1)) + 1
    if completed_episodes <= 0:
        raise RuntimeError("checkpoint episode metadata is invalid")
    return {
        "path": checkpoint,
        "metadata": metadata,
        "method": method,
        "training_seed": int(training_seed),
        "episode_seconds": episode_seconds,
        "completed_episodes": completed_episodes,
        "artifact_provenance": checkpoint_artifact_provenance(
            checkpoint, metadata=metadata
        ),
    }


def _resolve_manifest(path, episodes, sampling_seed):
    if path is None:
        return generate_manifest(
            "test",
            int(sampling_seed),
            int(episodes),
            balanced_num_gt=True,
        ), "generated_balanced_2_to_8"
    manifest = ScenarioManifest.load(path)
    if manifest.split not in {"test", "validation"}:
        raise ValueError("sampling manifest must use the test or validation split")
    if manifest.episode_count != int(episodes):
        raise ValueError(
            "sampling manifest episode_count must exactly match --episodes: "
            f"{manifest.episode_count} != {episodes}"
        )
    return manifest, "user_supplied"


def _allocate_output_directory(root, method_id, checkpoint_episode):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    method = method_id.replace("td3_dinkelbach", "td3d").replace("_", "-")[:20]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        suffix = uuid.uuid4().hex[:8]
        candidate = root / f"{method}-ep{checkpoint_episode}-{stamp}-{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique directory below {root}")


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def collect_samples(
    checkpoint,
    *,
    episodes=100,
    noise_std=0.1,
    sampling_seed=20260927,
    manifest_path=None,
    manifest_output=None,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    episodes = int(episodes)
    noise_std = float(noise_std)
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative")

    source = _read_checkpoint(checkpoint)
    manifest, manifest_source = _resolve_manifest(
        manifest_path, episodes, sampling_seed
    )
    if manifest_output is not None and Path(manifest_output).exists():
        raise FileExistsError(
            f"refusing to overwrite manifest output: {manifest_output}"
        )
    output = _allocate_output_directory(
        output_root, source["method"].method_id, source["completed_episodes"]
    )
    local_manifest = output / "scenario_manifest.json"
    manifest.save_atomic(local_manifest)
    if manifest_output is not None:
        manifest.save_atomic(manifest_output)

    capacity = episodes * source["episode_seconds"]
    collector = ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=capacity,
        rng=np.random.default_rng(int(sampling_seed)),
        record_auxiliary=True,
    )
    scenario_ids = [str(entry["scenario_id"]) for entry in manifest.episodes]
    base_metadata = {
        "schema_version": SAMPLING_METADATA_SCHEMA_VERSION,
        "status": "running",
        "complete": False,
        "sample_generation_mode": "independent_checkpoint_sampling",
        "method_id": source["method"].method_id,
        "checkpoint_path": str(source["path"]),
        "checkpoint_completed_episodes": source["completed_episodes"],
        "source_training_seed": source["training_seed"],
        "sampling_scenario_seed": int(sampling_seed),
        "action_noise_seed": int(sampling_seed),
        "noise_std": noise_std,
        "requested_episodes": episodes,
        "episode_seconds": source["episode_seconds"],
        "collector_capacity": capacity,
        "manifest_source": manifest_source,
        "scenario_manifest_path": str(local_manifest.resolve()),
        "scenario_manifest_hash": manifest.content_hash,
        "scenario_ids_by_index": scenario_ids,
        "source_checkpoint_has_auxiliary_replay": False,
        "source_checkpoint_artifact_provenance": source[
            "artifact_provenance"
        ],
        "source_checkpoint_contract": {
            key: source["metadata"].get(key)
            for key in (
                "checkpoint_schema_version",
                "state_contract",
                "movement_state_dim",
                "joint_action_dim",
                "routing_state_dim",
                "routing_action_dim",
                "num_uav",
                "visual_sensing_contract_version",
                "visual_sensing_configuration",
                "movement_feature_schema_version",
                "movement_state_feature_schema",
                "movement_agent_configuration",
                "routing_agent_configuration",
                "movement_action_projection_contract_version",
                "movement_replay_contract_version",
                "channel_contract",
                "channel_model_version",
                "channel_configuration",
                "ground_station_position_m",
                "communication_range_contract_version",
                "communication_range_boundary_rule",
                "maximum_3d_communication_distance_m",
                "production_task_deadline_seconds",
                "production_episode_horizon_seconds",
                "packet_injection_cutoff_seconds",
                "task_potential_configuration",
            )
        },
        "source_training_environment_contract": {
            key: (source["metadata"].get("experiment") or {}).get(key)
            for key in (
                "training_environment_width_m",
                "training_environment_height_m",
                "coordinate_normalization",
                "remaining_time_normalization",
                "remaining_time_normalization_horizon_s",
            )
        },
        "collector_starts_empty": True,
        "routing_exploration": "disabled; valid-action mask unchanged",
        "learning_updates": "disabled",
        "dinkelbach_lambda": "fixed from source checkpoint",
        "routing_cost_multiplier": "fixed from source checkpoint",
        "action_semantics": "executed post-displacement action, matching movement replay",
        "replay_auxiliary": replay_auxiliary_metadata(),
        "joint_replay_field_order": list(collector.all_fields),
        "field_role_contract": {
            "current_snapshot": "decision-available data aligned with state",
            "action": "executed post-displacement movement action",
            "next_snapshot": "post-action boundary data aligned with next_state",
            "provenance": (
                "episode/step/scenario identifiers and Dinkelbach lambda are "
                "trace-only and are not policy inputs"
            ),
        },
        "git_sha": _git_sha(),
    }
    _write_json(output / "metadata.json", base_metadata)

    def observe(record):
        if collector.total_added >= collector.max_size:
            raise RuntimeError(
                "sampling collector capacity exhausted; refusing silent overwrite"
            )
        collector.add(
            record["state"],
            record["executed_action"],
            record["next_state"],
            done=record["done"],
            delivered_mbits=record["delivered_mbits"],
            total_mobility_energy=record["total_mobility_energy_j"],
            c9_penalty=record["c9_penalty_mean"],
            c10_penalty=record["c10_penalty_mean"],
            com_range_penalty=record["com_range_penalty_mean"],
            ratio_objective_reward=record["ratio_objective_reward"],
            current_movement_mask=record["current_movement_mask"],
            next_movement_mask=record["next_movement_mask"],
            current_auxiliary_snapshot=record["current_auxiliary_snapshot"],
            next_auxiliary_snapshot=record["next_auxiliary_snapshot"],
            episode_id=record["episode_index"],
            td3_step=record["movement_step"],
            global_transition_id=record["global_transition_index"],
            scenario_index=record["scenario_index"],
            scenario_id=record["scenario_id"],
            dinkelbach_lambda=record["dinkelbach_lambda"],
        )

    config = TrainingConfig(
        total_episodes=episodes,
        mode="custom",
        episode_seconds=source["episode_seconds"],
        warmup_joint_transitions=0,
        batch_size=1,
        replay_max_size=capacity,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=source["training_seed"],
    )
    try:
        result = train(
            config,
            scenario_manifest=manifest,
            method_spec=source["method"],
            evaluation=True,
            checkpoint_dir=str(source["path"]),
            expected_checkpoint_episodes=source["completed_episodes"],
            transition_observer=observe,
            sampling_mode=True,
            sampling_noise_std=noise_std,
            sampling_seed=int(sampling_seed),
        )
        expected = episodes * source["episode_seconds"]
        if collector.size != expected or collector.total_added != expected:
            raise RuntimeError(
                "sampling transition count disagrees with executed episode horizon: "
                f"{collector.size} != {expected}"
            )
        if not collector.auxiliary_valid[: collector.size].all():
            raise RuntimeError("sampling produced incomplete auxiliary snapshots")
        replay_path = output / "joint_replay.npz"
        collector.save_npz(replay_path)
        with np.load(replay_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(collector.all_fields):
                raise RuntimeError("saved sampling replay field set is incomplete")
        valid_counts = {
            field: int(np.count_nonzero(getattr(collector, field)[: collector.size]))
            for field in collector.auxiliary_fields
            if getattr(collector, field).dtype == np.bool_
        }
        summary = {
            "status": "complete",
            "complete": True,
            "completed_episodes": int(result["episodes_run"]),
            "transition_count": int(collector.size),
            "collector_wrapped": bool(collector.total_added > collector.max_size),
            "collector_diagnostics": collector.diagnostics(),
            "field_validity_counts": valid_counts,
            "evaluation_invariants": result["evaluation_invariants"],
            "sampling_actor_updates": 0,
            "sampling_critic_updates": 0,
            "sampling_routing_updates": 0,
            "sampling_dinkelbach_updates": 0,
            "loaded_checkpoint_cumulative_counters": {
                "actor_updates": int(result["actor_updates"]),
                "critic_updates": int(result["critic_updates"]),
                "routing_updates": int(result["ddqn_training_updates"]),
                "dinkelbach_updates": int(result["dinkelbach_update_count"]),
            },
            "lambda": result["lambda"],
            "lambda_cost": result["lambda_cost"],
            "output_directory": str(output),
            "joint_replay": str(replay_path.resolve()),
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "metadata.json", {**base_metadata, **summary})
        print(
            f"Sampling complete: episodes={episodes} transitions={collector.size} "
            f"output={output}"
        )
        return summary
    except BaseException as exc:
        failure = {
            **base_metadata,
            "status": "failed",
            "complete": False,
            "partial_transition_count": int(collector.size),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
        _write_json(output / "metadata.json", failure)
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    collect_samples(
        args.checkpoint,
        episodes=args.episodes,
        noise_std=args.noise_std,
        sampling_seed=args.sampling_seed,
        manifest_path=args.manifest,
        manifest_output=args.manifest_output,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
