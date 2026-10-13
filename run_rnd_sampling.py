"""Collect frozen RND diagnostics and enriched trajectories from checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from rnd_exploration import RNDConfig, RNDExplorer, RND_METHOD_ID
from scenario_manifest import ScenarioManifest, generate_manifest
from training_checkpoint import checkpoint_artifact_provenance
from utils_update_v2 import ReplayBufferJoint


SAMPLING_SCHEMA_VERSION = "uav-hrl-rnd-sampling-v1"
DEFAULT_CHECKPOINT_EPISODES = (100, 250, 500, 750, 1000, 1500)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect complete zero-noise episodes from RND TD3 checkpoints"
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument(
        "--checkpoint-episodes", type=int, nargs="+", default=DEFAULT_CHECKPOINT_EPISODES
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--sampling-seed", type=int, default=20260927)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root", default="results/rnd_samples")
    return parser


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document must be an object: {path}")
    return value


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _save_npz_atomic(path, arrays):
    path = Path(path)
    temporary = path.parent / f".{path.stem}.tmp-{uuid.uuid4().hex}.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_source(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    metadata = _json(checkpoint / "metadata.json")
    experiment = metadata.get("experiment") or {}
    method_id = experiment.get("method_id") or (
        experiment.get("method_spec") or {}
    ).get("method_id")
    if method_id != RND_METHOD_ID:
        raise RuntimeError(
            f"sampling requires {RND_METHOD_ID}, checkpoint contains {method_id!r}"
        )
    rnd_metadata = metadata.get("rnd")
    if not isinstance(rnd_metadata, dict):
        raise RuntimeError("checkpoint is missing RND metadata")
    payload = torch.load(
        checkpoint / "models.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict) or "rnd" not in payload:
        raise RuntimeError("checkpoint is missing frozen RND state")
    config = RNDConfig.from_dict(rnd_metadata.get("configuration"))
    state_dim = int(metadata.get("movement_state_dim", -1))
    explorer = RNDExplorer(state_dim, config, device="cpu")
    explorer.load_state_dict(payload["rnd"], require_full=False)
    completed = int(metadata.get("episode", -1)) + 1
    formal = experiment.get("formal_config") or {}
    training_seed = experiment.get("training_seed")
    if completed <= 0 or training_seed is None:
        raise RuntimeError("checkpoint training provenance is incomplete")
    return {
        "path": checkpoint,
        "metadata": metadata,
        "method": MethodSpec.parse(method_id),
        "rnd_config": config,
        "rnd": explorer,
        "completed_episodes": completed,
        "training_seed": int(training_seed),
        "episode_seconds": int(formal.get("episode_seconds", 0)),
        "artifact": checkpoint_artifact_provenance(checkpoint, metadata=metadata),
    }


def _resolve_manifest(path, episodes, seed, batch_dir):
    local = batch_dir / "scenario_manifest.json"
    if path:
        manifest = ScenarioManifest.load(path)
    elif local.exists():
        manifest = ScenarioManifest.load(local)
    else:
        manifest = generate_manifest(
            "test", int(seed), int(episodes), balanced_num_gt=True
        )
    if manifest.split not in {"test", "validation"}:
        raise ValueError("sampling manifest must be test or validation")
    if manifest.episode_count != int(episodes):
        raise ValueError("sampling manifest length must equal --episodes")
    if local.exists():
        existing = ScenarioManifest.load(local)
        if existing.content_hash != manifest.content_hash:
            raise RuntimeError("existing batch manifest is incompatible")
    else:
        manifest.save_atomic(local)
    return manifest, local


def _completed_output_is_valid(output, *, checkpoint_episode, manifest_hash, episodes):
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = _json(metadata_path)
        if not metadata.get("complete"):
            return False
        if int(metadata["checkpoint_completed_episodes"]) != int(checkpoint_episode):
            return False
        if metadata["manifest_hash"] != manifest_hash:
            return False
        if int(metadata["completed_episodes"]) != int(episodes):
            return False
        for name, expected_hash in metadata["file_sha256"].items():
            path = output / name
            if not path.is_file() or _sha256(path) != expected_hash:
                return False
        with np.load(output / "joint_replay.npz", allow_pickle=False) as replay:
            if int(replay["state"].shape[0]) != int(metadata["transition_count"]):
                return False
        with np.load(output / "rnd_diagnostics.npz", allow_pickle=False) as diagnostics:
            if int(diagnostics["raw_error"].shape[0]) != int(metadata["transition_count"]):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def collect_checkpoint(source, manifest, output, episodes, sampling_seed):
    output.mkdir(parents=True, exist_ok=True)
    capacity = int(episodes) * int(source["episode_seconds"])
    collector = ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=capacity,
        rng=np.random.default_rng(int(sampling_seed)),
        record_auxiliary=True,
    )
    diagnostics = {
        name: []
        for name in (
            "raw_error", "normalized_bonus", "weighted_bonus",
            "original_reward", "diagnostic_training_reward", "done", "truncated",
            "episode", "step", "scenario_index",
        )
    }
    scenario_ids = []
    rnd = source["rnd"]
    rnd_before = rnd.fingerprint()

    def observe(record):
        if collector.total_added >= collector.max_size:
            raise RuntimeError("sampling collector capacity exhausted")
        collector.add(
            record["state"], record["executed_action"], record["next_state"],
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
        value = rnd.diagnostic(record["next_state"])
        original = float(record["existing_movement_reward"])
        diagnostics["raw_error"].append(value["raw_error"])
        diagnostics["normalized_bonus"].append(value["normalized_bonus"])
        diagnostics["weighted_bonus"].append(value["weighted_bonus"])
        diagnostics["original_reward"].append(original)
        diagnostics["diagnostic_training_reward"].append(
            original + value["weighted_bonus"]
        )
        diagnostics["done"].append(bool(record["done"]))
        diagnostics["truncated"].append(False)
        diagnostics["episode"].append(int(record["episode_index"]))
        diagnostics["step"].append(int(record["movement_step"]))
        diagnostics["scenario_index"].append(int(record["scenario_index"]))
        scenario_ids.append(str(record["scenario_id"]))

    config = TrainingConfig(
        total_episodes=int(episodes), mode="custom",
        episode_seconds=int(source["episode_seconds"]),
        warmup_joint_transitions=0, batch_size=1,
        replay_max_size=capacity,
        enable_model_checkpoints=False, enable_full_resume=False,
        enable_plots=False, enable_csv=False,
        random_seed=int(source["training_seed"]),
    )
    base = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "status": "running", "complete": False,
        "checkpoint_path": str(source["path"]),
        "checkpoint_completed_episodes": int(source["completed_episodes"]),
        "manifest_hash": manifest.content_hash,
        "requested_episodes": int(episodes),
        "sampling_seed": int(sampling_seed),
        "action_noise_std": 0.0,
        "rnd_configuration": source["rnd_config"].to_dict(),
        "source_rnd_metadata": source["metadata"]["rnd"],
        "rnd_reward_source": "checkpoint_frozen_diagnostic_not_applied_to_actions",
        "source_checkpoint_artifact": source["artifact"],
    }
    _write_json_atomic(output / "metadata.json", base)
    result = train(
        config,
        scenario_manifest=manifest,
        method_spec=source["method"],
        evaluation=True,
        checkpoint_dir=str(source["path"]),
        expected_checkpoint_episodes=source["completed_episodes"],
        transition_observer=observe,
        sampling_mode=True,
        sampling_noise_std=0.0,
        sampling_seed=int(sampling_seed),
        rnd_config=source["rnd_config"],
    )
    if rnd.fingerprint() != rnd_before:
        raise AssertionError("frozen sampling RND state changed")
    if result["rnd_fingerprint"] != result["rnd_evaluation_fingerprint_before"]:
        raise AssertionError("train evaluation changed its loaded RND state")
    if not collector.auxiliary_valid[: collector.size].all():
        raise RuntimeError("sampling produced incomplete auxiliary snapshots")

    replay_path = output / "joint_replay.npz"
    temporary_replay = output / f".joint-replay-{uuid.uuid4().hex}.npz"
    collector.save_npz(temporary_replay)
    os.replace(temporary_replay, replay_path)
    typed = {
        "raw_error": np.asarray(diagnostics["raw_error"], dtype=np.float32),
        "normalized_bonus": np.asarray(diagnostics["normalized_bonus"], dtype=np.float32),
        "weighted_bonus": np.asarray(diagnostics["weighted_bonus"], dtype=np.float32),
        "original_reward": np.asarray(diagnostics["original_reward"], dtype=np.float32),
        "diagnostic_training_reward": np.asarray(diagnostics["diagnostic_training_reward"], dtype=np.float32),
        "done": np.asarray(diagnostics["done"], dtype=bool),
        "truncated": np.asarray(diagnostics["truncated"], dtype=bool),
        "episode": np.asarray(diagnostics["episode"], dtype=np.int32),
        "step": np.asarray(diagnostics["step"], dtype=np.int16),
        "scenario_index": np.asarray(diagnostics["scenario_index"], dtype=np.int32),
        "scenario_id": np.asarray(scenario_ids, dtype=np.str_),
    }
    diagnostic_path = output / "rnd_diagnostics.npz"
    _save_npz_atomic(diagnostic_path, typed)
    episode_path = output / "episodes.json"
    _write_json_atomic(
        episode_path,
        {
            "episodes": result["episode_metrics"],
            "scenario_ids": [str(entry["scenario_id"]) for entry in manifest.episodes],
        },
    )
    file_hashes = {
        path.name: _sha256(path)
        for path in (replay_path, diagnostic_path, episode_path)
    }
    complete = {
        **base,
        "status": "complete", "complete": True,
        "completed_episodes": int(result["episodes_run"]),
        "transition_count": int(collector.size),
        "file_sha256": file_hashes,
        "actor_updates_during_sampling": 0,
        "critic_updates_during_sampling": 0,
        "routing_updates_during_sampling": 0,
        "rnd_updates_during_sampling": 0,
        "rnd_fingerprint_before": rnd_before,
        "rnd_fingerprint_after": rnd.fingerprint(),
        "raw_error_mean": float(np.mean(typed["raw_error"])),
        "raw_error_max": float(np.max(typed["raw_error"])),
        "normalized_bonus_mean": float(np.mean(typed["normalized_bonus"])),
        "normalized_bonus_max": float(np.max(typed["normalized_bonus"])),
    }
    _write_json_atomic(output / "metadata.json", complete)
    return complete


def run(args):
    run_dir = Path(args.run_directory).resolve()
    episodes = int(args.episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    checkpoints = sorted({int(value) for value in args.checkpoint_episodes})
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("checkpoint episodes must be positive")
    run_identity = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:8]
    batch = Path(args.output_root).resolve() / f"r{run_identity}-s{int(args.sampling_seed)}-n{episodes}"
    batch.mkdir(parents=True, exist_ok=True)
    manifest, manifest_path = _resolve_manifest(
        args.manifest, episodes, args.sampling_seed, batch
    )
    summary = []
    for checkpoint_episode in checkpoints:
        checkpoint = run_dir / "checkpoints" / "models" / f"ep_{checkpoint_episode:04d}"
        source = _load_source(checkpoint)
        if source["completed_episodes"] != checkpoint_episode:
            raise RuntimeError("checkpoint directory episode disagrees with metadata")
        output = batch / f"ep{checkpoint_episode:04d}"
        if _completed_output_is_valid(
            output,
            checkpoint_episode=checkpoint_episode,
            manifest_hash=manifest.content_hash,
            episodes=episodes,
        ):
            print(f"[RND sampling] ep={checkpoint_episode} complete; verified and skipped")
            summary.append({"checkpoint_episode": checkpoint_episode, "status": "skipped_complete"})
            continue
        print(f"[RND sampling] ep={checkpoint_episode} episodes={episodes} output={output}")
        result = collect_checkpoint(
            source, manifest, output, episodes, int(args.sampling_seed)
        )
        summary.append({
            "checkpoint_episode": checkpoint_episode,
            "status": "completed",
            "transitions": result["transition_count"],
        })
    _write_json_atomic(
        batch / "batch_summary.json",
        {
            "schema_version": SAMPLING_SCHEMA_VERSION,
            "run_directory": str(run_dir),
            "manifest_path": str(manifest_path),
            "manifest_hash": manifest.content_hash,
            "requested_checkpoint_episodes": checkpoints,
            "requested_episodes_per_checkpoint": episodes,
            "items": summary,
        },
    )
    print(json.dumps({"batch_directory": str(batch), "items": summary}))
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
