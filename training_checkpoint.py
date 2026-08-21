import hashlib
import json
import random
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import torch

from centralized_movement import MOVEMENT_STATE_DIM, movement_mask_from_state

from dinkelbach_blocks import (
    DINKELBACH_CONFIG_FIELDS,
    DinkelbachBlockState,
    dinkelbach_config_metadata,
)

CHECKPOINT_SCHEMA_VERSION = 4
PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION = 3
LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION = 2
MODEL_CHECKPOINT_TYPE = "model-only"
FULL_CHECKPOINT_TYPE = "full-resume"
FULL_RESUME_LOGGING_SCHEMA_VERSION = 1

FULL_RESUME_LOGGING_STATE_FIELDS = (
    "full_resume_logging_schema_version",
    "reward_log",
    "delivered_log",
    "energy_log",
    "lambda_used_log",
    "lambda_after_episode_log",
)

LEGACY_JOINT_REPLAY_FIELDS = (
    "state",
    "action",
    "next_state",
    "not_done",
    "delivered_mbits",
    "total_mobility_energy",
    "phi_search_t",
    "phi_search_t1",
    "phi_vs_t",
    "phi_vs_t1",
    "phi_com_t",
    "phi_com_t1",
)
PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS = (
    *LEGACY_JOINT_REPLAY_FIELDS,
    "ratio_objective_reward",
)
JOINT_REPLAY_FIELDS = (
    *PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS,
    "current_movement_mask",
    "next_movement_mask",
    "movement_mask_valid",
)
ROUTING_REPLAY_FIELDS = (
    "state",
    "action",
    "next_state",
    "reward",
    "cost",
    "not_done",
    "tag_gt",
)

FORMAL_CORE_CONFIG_FIELDS = (
    "mode",
    "total_episodes",
    "episode_seconds",
    "routing_slot_seconds",
    "warmup_joint_transitions",
    "batch_size",
    "policy_delay",
    "beta_search",
    "beta_vs",
    "beta_com",
    "search_coverage_threshold",
    "replay_max_size",
    "dinkelbach_initial_lambda",
    "dinkelbach_update_interval_episodes",
    "dinkelbach_update_rule",
    "dinkelbach_numerator_unit",
    "dinkelbach_denominator_unit",
    "assignment_strategy",
    "assignment_rounds",
    "movement_policy",
    "movement_objective",
    "routing_policy",
    "task_observation_mode",
    "fov_com_pair_max_distance_m",
    "search_utility",
    "utility_normalization_mode",
    "task_compatibility_policy",
    "hover_assignment_candidate",
    "assignment_dummy_utility",
)

FULL_RESUME_CONFIG_FIELDS = (
    *FORMAL_CORE_CONFIG_FIELDS,
    "model_checkpoint_every",
    "full_resume_every",
    "full_resume_keep_last",
    "formal_evaluation_episode",
    "random_seed",
)


def calibration_fingerprint(calibration):
    canonical = json.dumps(
        calibration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def checkpoint_metadata_fingerprint(metadata):
    return calibration_fingerprint(metadata)


def checkpoint_episode_schedule(total_episodes, every):
    """Return periodic episodes plus a non-duplicate normal-completion save."""

    total_episodes = int(total_episodes)
    every = int(every)
    if total_episodes <= 0 or every <= 0:
        return []
    episodes = list(range(every, total_episodes + 1, every))
    if not episodes or episodes[-1] != total_episodes:
        episodes.append(total_episodes)
    return episodes


def _atomic_checkpoint_write(checkpoint_dir, writer):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    temporary = checkpoint_dir.parent / (
        f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        writer(temporary)
        temporary.replace(checkpoint_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return checkpoint_dir


def prune_full_resume_checkpoints(full_root, keep_last):
    full_root = Path(full_root).resolve()
    keep_last = int(keep_last)
    if keep_last <= 0:
        raise ValueError("full-resume retention must keep at least one checkpoint")
    if not full_root.is_dir():
        return []
    checkpoints = []
    for candidate in full_root.iterdir():
        match = re.fullmatch(r"ep_(\d+)", candidate.name)
        if match and candidate.is_dir() and candidate.resolve().parent == full_root:
            checkpoints.append((int(match.group(1)), candidate.resolve()))
    checkpoints.sort()
    removed = []
    for _, checkpoint in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def _agent_kind(agent):
    return str(getattr(agent, "agent_kind", "td3"))


def _routing_agent_kind(agent):
    return str(getattr(agent, "routing_agent_kind", "safe_ddqn"))


def _movement_agent_configuration(agent):
    kind = _agent_kind(agent)
    return {
        "movement_agent_kind": kind,
        "movement_agent_gamma": float(agent.gamma),
        "tau": float(agent.tau) if hasattr(agent, "tau") else None,
        "policy_delay": getattr(agent, "policy_delay", None),
        "target_policy_noise": (
            float(agent.policy_noise)
            if kind == "td3"
            else getattr(agent, "target_policy_noise", None)
        ),
        "target_noise_clip": (
            float(agent.noise_clip)
            if kind == "td3"
            else getattr(agent, "target_noise_clip", None)
        ),
        "twin_critics": kind == "td3",
    }


def _validate_effective_formal_movement_config(formal_config, kind, metadata=None):
    if not isinstance(formal_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    expected_delay = {"td3": 2, "ddpg": 1, "random": None}.get(kind)
    if kind not in {"td3", "ddpg", "random"}:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")
    if not _metadata_value_matches(formal_config.get("policy_delay"), expected_delay):
        raise RuntimeError(
            "checkpoint formal training config is incompatible with effective "
            f"{kind} policy delay: checkpoint={formal_config.get('policy_delay')}, "
            f"expected={expected_delay}"
        )
    nested = formal_config.get("movement_agent_configuration")
    if nested is not None and metadata is not None:
        actual = metadata.get("movement_agent_configuration")
        if nested != actual:
            raise RuntimeError(
                "checkpoint formal movement-agent configuration is incompatible "
                "with checkpoint metadata"
            )


def _movement_network_states(agent):
    kind = _agent_kind(agent)
    if kind == "td3":
        return {
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic_1": agent.critic_1.state_dict(),
            "critic_1_target": agent.critic_1_target.state_dict(),
            "critic_2": agent.critic_2.state_dict(),
            "critic_2_target": agent.critic_2_target.state_dict(),
        }
    if kind == "ddpg":
        return {
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
        }
    if kind == "random":
        return {}
    raise ValueError(f"unsupported movement agent kind: {kind}")


def _network_states(td3, ddqn):
    kind = _agent_kind(td3)
    routing_kind = _routing_agent_kind(ddqn)
    movement_key = "td3" if kind == "td3" else "movement_agent"
    states = {
        movement_key: {
            **({"kind": kind} if movement_key == "movement_agent" else {}),
            **_movement_network_states(td3),
        },
    }
    if routing_kind == "safe_ddqn":
        states["ddqn"] = {
            "q_network": ddqn.q_network.state_dict(),
            "target_q_network": ddqn.target_q_network.state_dict(),
            "cost_network": ddqn.cost_network.state_dict(),
            "target_cost_network": ddqn.target_cost_network.state_dict(),
        }
    elif routing_kind == "dqn":
        states["routing_agent"] = {
            "kind": "dqn",
            "q_network": ddqn.q_network.state_dict(),
            "target_q_network": ddqn.target_q_network.state_dict(),
        }
    elif routing_kind == "random":
        states["routing_agent"] = {"kind": "random"}
    else:
        raise ValueError(f"unsupported routing agent kind: {routing_kind}")
    return states


def _load_network_states(payload, td3, ddqn):
    kind = _agent_kind(td3)
    if kind == "td3":
        td3_state = payload["td3"]
        td3.actor.load_state_dict(td3_state["actor"])
        td3.actor_target.load_state_dict(td3_state["actor_target"])
        td3.critic_1.load_state_dict(td3_state["critic_1"])
        td3.critic_1_target.load_state_dict(td3_state["critic_1_target"])
        td3.critic_2.load_state_dict(td3_state["critic_2"])
        td3.critic_2_target.load_state_dict(td3_state["critic_2_target"])
    elif kind == "ddpg":
        state = payload["movement_agent"]
        if state.get("kind") != "ddpg":
            raise RuntimeError("checkpoint movement agent kind is incompatible")
        td3.actor.load_state_dict(state["actor"])
        td3.actor_target.load_state_dict(state["actor_target"])
        td3.critic.load_state_dict(state["critic"])
        td3.critic_target.load_state_dict(state["critic_target"])
    elif kind == "random":
        state = payload.get("movement_agent") or {}
        if state.get("kind") != "random":
            raise RuntimeError("checkpoint movement agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")

    routing_kind = _routing_agent_kind(ddqn)
    if routing_kind == "safe_ddqn":
        ddqn_state = payload["ddqn"]
        ddqn.q_network.load_state_dict(ddqn_state["q_network"])
        ddqn.target_q_network.load_state_dict(ddqn_state["target_q_network"])
        ddqn.cost_network.load_state_dict(ddqn_state["cost_network"])
        ddqn.target_cost_network.load_state_dict(ddqn_state["target_cost_network"])
    elif routing_kind == "dqn":
        state = payload.get("routing_agent") or {}
        if state.get("kind") != "dqn" or "cost_network" in state:
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        ddqn.q_network.load_state_dict(state["q_network"])
        ddqn.target_q_network.load_state_dict(state["target_q_network"])
    elif routing_kind == "random":
        state = payload.get("routing_agent") or {}
        if state != {"kind": "random"}:
            raise RuntimeError("checkpoint routing agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported routing agent kind: {routing_kind}")


def _base_metadata(
    checkpoint_type,
    episode,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3,
    ddqn,
    calibration,
    experiment_metadata=None,
):
    kind = _agent_kind(td3)
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": checkpoint_type,
        "episode": int(episode),
        "movement_state_dim": int(movement_state_dim),
        "joint_action_dim": int(joint_action_dim),
        "routing_state_dim": int(routing_state_dim),
        "movement_agent_kind": kind,
        "movement_agent_gamma": float(td3.gamma),
        "movement_agent_configuration": _movement_agent_configuration(td3),
        "routing_ddqn_gamma": float(ddqn.gamma),
        "routing_agent_kind": _routing_agent_kind(ddqn),
        "com_calibration_fingerprint": calibration_fingerprint(calibration),
    }
    if kind == "td3":
        metadata["centralized_td3_gamma"] = float(td3.gamma)
    if experiment_metadata is not None:
        metadata["experiment"] = dict(experiment_metadata)
    return metadata


def save_model_checkpoint(
    checkpoint_dir,
    *,
    episode,
    td3,
    ddqn,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    experiment_metadata=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    metadata = _base_metadata(
        MODEL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
        experiment_metadata,
    )
    experiment = metadata.get("experiment") or {}
    if "formal_config" in experiment:
        _validate_effective_formal_movement_config(
            experiment["formal_config"], _agent_kind(td3), metadata
        )
    def write(temporary):
        torch.save(_network_states(td3, ddqn), temporary / "models.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    return _atomic_checkpoint_write(checkpoint_dir, write)


def _metadata_value_matches(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return bool(np.isclose(float(actual), float(expected)))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _checkpoint_directory_episode(checkpoint_dir, metadata):
    match = re.fullmatch(r"ep_(\d+)", Path(checkpoint_dir).name)
    if match is None:
        raise RuntimeError(
            f"checkpoint directory is not an episode checkpoint: {checkpoint_dir}"
        )
    try:
        metadata_episode = int(metadata["episode"]) + 1
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint episode metadata is invalid") from exc
    directory_episode = int(match.group(1))
    if directory_episode != metadata_episode:
        raise RuntimeError(
            "checkpoint directory episode is incompatible with metadata: "
            f"directory={directory_episode}, metadata={metadata_episode}"
        )
    return directory_episode


def _validate_formal_config(actual_config, expected_config, fields):
    if not isinstance(actual_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    mismatches = {
        key: (actual_config.get(key), expected_config.get(key))
        for key in fields
        if not _metadata_value_matches(
            actual_config.get(key), expected_config.get(key)
        )
    }
    if mismatches:
        raise RuntimeError(
            f"checkpoint formal training config is incompatible: {mismatches}"
        )


def _validate_dinkelbach_checkpoint_metadata(metadata, expected_formal_config):
    experiment = metadata.get("experiment") or {}
    actual_formal_config = experiment.get("formal_config")
    if not isinstance(actual_formal_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    config = expected_formal_config or actual_formal_config
    try:
        expected_config = dinkelbach_config_metadata(config)
        actual_config = dinkelbach_config_metadata(actual_formal_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint Dinkelbach configuration is incomplete"
        ) from exc
    if actual_config != expected_config:
        raise RuntimeError(
            "checkpoint Dinkelbach configuration is incompatible: "
            f"checkpoint={actual_config}, expected={expected_config}"
        )
    provenance_mismatches = {
        field: (experiment.get(field), actual_config[field])
        for field in DINKELBACH_CONFIG_FIELDS
        if experiment.get(field) != actual_config[field]
    }
    if provenance_mismatches:
        raise RuntimeError(
            "checkpoint Dinkelbach provenance is incompatible: "
            f"{provenance_mismatches}"
        )
    try:
        completed_episodes = int(metadata["episode"]) + 1
        state = DinkelbachBlockState.from_training_state(
            experiment.get("dinkelbach_state", {}),
            actual_formal_config,
            expected_completed_episodes=completed_episodes,
        )
        stored_lambda = float(experiment["lambda_ee"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint Dinkelbach metadata is incomplete"
        ) from exc
    if not np.isfinite(stored_lambda) or not np.isclose(
        stored_lambda, state.current_lambda
    ):
        raise RuntimeError(
            "checkpoint Dinkelbach lambda metadata is incompatible with block state"
        )
    return state


def _checkpoint_uses_dinkelbach(metadata):
    experiment = metadata.get("experiment") or {}
    method_spec = experiment.get("method_spec") or {}
    return method_spec.get("reward_mode", experiment.get("reward_mode", "dinkelbach")) == "dinkelbach"


def _validate_checkpoint_schema(metadata):
    schema = metadata.get("checkpoint_schema_version")
    if schema in {
        CHECKPOINT_SCHEMA_VERSION,
        PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION,
    }:
        return int(schema)
    try:
        legacy_schema = (
            schema is None
            or int(schema) < PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        )
    except (TypeError, ValueError):
        legacy_schema = False
    if (
        not _checkpoint_uses_dinkelbach(metadata)
        and legacy_schema
    ):
        raise RuntimeError(
            "legacy ratio checkpoint uses per-transition B/E and cannot be "
            "resumed under the terminal ratio-of-sums reward contract"
        )
    if schema == LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION:
        return int(schema)
    raise RuntimeError(
        "checkpoint checkpoint_schema_version is incompatible: "
        f"checkpoint={schema}, expected={CHECKPOINT_SCHEMA_VERSION}"
    )


def _formal_config_for_validation(metadata, actual_config, expected_config):
    """Normalize documented legacy fields without weakening current checks."""

    if (
        metadata.get("checkpoint_schema_version")
        == LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION
        and metadata.get("movement_agent_kind", "td3") == "ddpg"
        and isinstance(actual_config, dict)
        and isinstance(expected_config, dict)
    ):
        actual_config = dict(actual_config)
        actual_config["policy_delay"] = expected_config.get("policy_delay")
    experiment = metadata.get("experiment") or {}
    legacy_method_spec = experiment.get("method_spec") or {}
    if (
        isinstance(actual_config, dict)
        and isinstance(expected_config, dict)
        and "task_observation" not in legacy_method_spec
    ):
        actual_config = dict(actual_config)
        for field in FORMAL_CORE_CONFIG_FIELDS:
            if field not in actual_config and field in expected_config:
                actual_config[field] = expected_config[field]
    return actual_config


def _movement_gamma_key(metadata, schema):
    kind = metadata.get("movement_agent_kind", "td3")
    if kind == "td3" and "centralized_td3_gamma" in metadata:
        return "centralized_td3_gamma"
    return (
        "movement_agent_gamma"
        if schema >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        else "centralized_td3_gamma"
    )


def _validate_td3_gamma_alias(metadata):
    if metadata.get("movement_agent_kind", "td3") != "td3":
        return
    if "movement_agent_gamma" not in metadata or "centralized_td3_gamma" not in metadata:
        return
    if not _metadata_value_matches(
        metadata["movement_agent_gamma"], metadata["centralized_td3_gamma"]
    ):
        raise RuntimeError(
            "checkpoint centralized_td3_gamma is incompatible with "
            "movement_agent_gamma"
        )


def validate_model_checkpoint_metadata(
    metadata,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    td3_gamma=None,
    ddqn_gamma=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
    movement_agent_kind=None,
):
    """Validate evaluation provenance before any model payload is loaded."""

    schema = _validate_checkpoint_schema(metadata)
    checks = {"checkpoint_type": MODEL_CHECKPOINT_TYPE}
    optional_checks = {
        "movement_state_dim": movement_state_dim,
        "joint_action_dim": joint_action_dim,
        "routing_state_dim": routing_state_dim,
        "routing_ddqn_gamma": ddqn_gamma,
    }
    checks.update(
        {key: value for key, value in optional_checks.items() if value is not None}
    )
    if movement_agent_kind is not None:
        actual_kind = metadata.get("movement_agent_kind", "td3")
        if actual_kind != movement_agent_kind:
            raise RuntimeError(
                "checkpoint movement_agent_kind is incompatible: "
                f"checkpoint={actual_kind}, expected={movement_agent_kind}"
            )
    for key, expected in checks.items():
        actual = metadata.get(key)
        if not _metadata_value_matches(actual, expected):
            raise RuntimeError(
                f"checkpoint {key} is incompatible: "
                f"checkpoint={actual}, expected={expected}"
            )
    _validate_td3_gamma_alias(metadata)
    if td3_gamma is not None:
        gamma_key = _movement_gamma_key(metadata, schema)
        actual_gamma = metadata.get(gamma_key)
        if not _metadata_value_matches(actual_gamma, td3_gamma):
            raise RuntimeError(
                f"checkpoint {gamma_key} is incompatible: "
                f"checkpoint={actual_gamma}, expected={td3_gamma}"
            )

    if calibration is not None:
        expected_calibration = calibration_fingerprint(calibration)
        if metadata.get("com_calibration_fingerprint") != expected_calibration:
            raise RuntimeError("checkpoint COM calibration fingerprint is incompatible")

    if expected_experiment_metadata is not None:
        validate_checkpoint_experiment_metadata(
            metadata, expected_experiment_metadata
        )

    if expected_completed_episodes is not None:
        try:
            completed_episodes = int(metadata["episode"]) + 1
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint episode metadata is invalid") from exc
        if completed_episodes != int(expected_completed_episodes):
            raise RuntimeError(
                "checkpoint completed training episodes is incompatible: "
                f"checkpoint={completed_episodes}, "
                f"expected={int(expected_completed_episodes)}"
            )

    if expected_formal_config is not None:
        experiment = metadata.get("experiment") or {}
        _validate_formal_config(
            _formal_config_for_validation(
                metadata, experiment.get("formal_config"), expected_formal_config
            ),
            expected_formal_config,
            FORMAL_CORE_CONFIG_FIELDS,
        )
        if _checkpoint_uses_dinkelbach(metadata):
            _validate_dinkelbach_checkpoint_metadata(
                metadata, expected_formal_config
            )
    experiment = metadata.get("experiment") or {}
    if (
        schema >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        and "formal_config" in experiment
    ):
        _validate_effective_formal_movement_config(
            experiment["formal_config"], metadata.get("movement_agent_kind", "td3"), metadata
        )
    return metadata


def inspect_model_checkpoint(
    checkpoint_dir,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    td3_gamma=None,
    ddqn_gamma=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
    require_episode_directory=False,
    movement_agent_kind=None,
):
    """Validate model metadata and required files without loading weights."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    models_path = checkpoint_dir / "models.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_model_checkpoint_metadata(
        metadata,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3_gamma,
        ddqn_gamma=ddqn_gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_completed_episodes=expected_completed_episodes,
        expected_formal_config=expected_formal_config,
        movement_agent_kind=movement_agent_kind,
    )
    if not models_path.is_file():
        raise RuntimeError(f"checkpoint model payload is missing: {models_path}")
    completed_episode = (
        _checkpoint_directory_episode(checkpoint_dir, metadata)
        if require_episode_directory
        else int(metadata["episode"]) + 1
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "completed_episode": completed_episode,
        "metadata": metadata,
    }


def load_model_checkpoint(
    checkpoint_dir,
    td3,
    ddqn,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    inspected = inspect_model_checkpoint(
        checkpoint_dir,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3.gamma,
        ddqn_gamma=ddqn.gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_completed_episodes=expected_completed_episodes,
        expected_formal_config=expected_formal_config,
        movement_agent_kind=_agent_kind(td3),
    )
    metadata = inspected["metadata"]
    payload = torch.load(
        checkpoint_dir / "models.pt", map_location="cpu", weights_only=False
    )
    _load_network_states(payload, td3, ddqn)
    return metadata


def _save_replay(path, replay, fields):
    size = int(replay.size)
    arrays = {field: np.asarray(getattr(replay, field)[:size]) for field in fields}
    np.savez_compressed(path, **arrays)
    return {
        "ptr": int(replay.ptr),
        "size": size,
        "max_size": int(replay.max_size),
        "n_step": int(replay.n_step),
        "gamma": float(getattr(replay, "gamma", 1.0)),
        "n_step_buffer": list(getattr(replay, "n_step_buffer", [])),
    }


def _load_replay(path, replay, fields, metadata):
    expected = {
        "max_size": int(replay.max_size),
        "n_step": int(replay.n_step),
    }
    for key, expected_value in expected.items():
        actual = int(metadata[key])
        if actual != expected_value:
            raise RuntimeError(
                f"replay {key} mismatch: checkpoint={actual}, current={expected_value}"
            )
    if hasattr(replay, "gamma"):
        checkpoint_gamma = float(metadata["gamma"])
        if not np.isclose(checkpoint_gamma, replay.gamma):
            raise RuntimeError(
                "replay gamma mismatch: "
                f"checkpoint={checkpoint_gamma}, current={replay.gamma}"
            )
    size = int(metadata["size"])
    ptr = int(metadata["ptr"])
    if not 0 <= size <= replay.max_size or not 0 <= ptr < replay.max_size:
        raise RuntimeError(f"invalid replay size/ptr in checkpoint: {size}/{ptr}")
    with np.load(path, allow_pickle=False) as arrays:
        for field in fields:
            saved = arrays[field]
            target = getattr(replay, field)
            if saved.shape != target[:size].shape:
                raise RuntimeError(
                    f"replay field {field} shape mismatch: "
                    f"checkpoint={saved.shape}, current={target[:size].shape}"
                )
            target[:size] = saved
    replay.size = size
    replay.ptr = ptr
    replay.n_step_buffer = list(metadata.get("n_step_buffer", []))


def _checkpoint_task_observation_mode(metadata):
    experiment = metadata.get("experiment") or {}
    method_spec = experiment.get("method_spec") or {}
    formal_config = experiment.get("formal_config") or {}
    return str(
        method_spec.get(
            "task_observation",
            formal_config.get("task_observation_mode", "full"),
        )
    )


def _validate_joint_replay_projection_masks(checkpoint_dir, metadata):
    """Validate mask availability before exact-resume state can be mutated."""

    schema = int(metadata["checkpoint_schema_version"])
    observation_mode = _checkpoint_task_observation_mode(metadata)
    if schema < CHECKPOINT_SCHEMA_VERSION:
        if observation_mode == "masked":
            raise RuntimeError(
                "legacy masked-observation full-resume checkpoint lacks true "
                "movement projection masks and cannot be resumed safely"
            )
        return

    replay_path = Path(checkpoint_dir) / "joint_replay.npz"
    required = {
        "current_movement_mask",
        "next_movement_mask",
        "movement_mask_valid",
    }
    with np.load(replay_path, allow_pickle=False) as arrays:
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise RuntimeError(
                "full-resume checkpoint is missing movement projection mask "
                f"fields: {missing}"
            )
        if int(metadata.get("movement_state_dim", -1)) == MOVEMENT_STATE_DIM:
            validity = np.asarray(arrays["movement_mask_valid"], dtype=bool)
            if not validity.all():
                raise RuntimeError(
                    "full-resume checkpoint contains joint transitions without "
                    "authoritative movement projection masks"
                )


def _reconstruct_legacy_full_observation_masks(replay, metadata):
    """Migrate pre-mask full-observation replay without guessing assignments."""

    if _checkpoint_task_observation_mode(metadata) != "full":
        raise RuntimeError(
            "legacy masked-observation full-resume checkpoint lacks true "
            "movement projection masks and cannot be resumed safely"
        )
    if replay.state.shape[1] != MOVEMENT_STATE_DIM:
        raise RuntimeError(
            "legacy full-observation replay cannot reconstruct movement masks "
            f"from state dimension {replay.state.shape[1]}"
        )
    size = int(replay.size)
    replay.current_movement_mask[:size] = movement_mask_from_state(
        replay.state[:size]
    )
    replay.next_movement_mask[:size] = movement_mask_from_state(
        replay.next_state[:size]
    )
    replay.movement_mask_valid[:size] = True


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _movement_training_payload(agent):
    kind = _agent_kind(agent)
    counters = {
        "critic_updates": int(agent.num_critic_update_iteration),
        "actor_updates": int(agent.num_actor_update_iteration),
        "training_updates": int(agent.num_training),
    }
    if kind == "td3":
        return {
            "td3_optimizers": {
                "actor": agent.actor_optimizer.state_dict(),
                "critic_1": agent.critic_1_optimizer.state_dict(),
                "critic_2": agent.critic_2_optimizer.state_dict(),
            },
            "td3_counters": counters,
            "td3_hyperparameters": {
                "gamma": float(agent.gamma),
                "tau": float(agent.tau),
                "policy_delay": int(agent.policy_delay),
                "policy_noise": float(agent.policy_noise),
                "noise_clip": float(agent.noise_clip),
                "max_action": float(agent.max_action),
            },
        }
    if kind == "ddpg":
        return {
            "movement_agent_optimizers": {
                "actor": agent.actor_optimizer.state_dict(),
                "critic": agent.critic_optimizer.state_dict(),
            },
            "movement_agent_counters": counters,
            "movement_agent_hyperparameters": {
                "gamma": float(agent.gamma),
                "tau": float(agent.tau),
                "policy_delay": int(agent.policy_delay),
                "target_policy_noise": agent.target_policy_noise,
                "target_noise_clip": agent.target_noise_clip,
                "twin_critics": False,
                "max_action": float(agent.max_action),
            },
        }
    if kind == "random":
        return {
            "movement_agent_optimizers": {},
            "movement_agent_counters": counters,
            "movement_agent_hyperparameters": {"gamma": float(agent.gamma)},
        }
    raise ValueError(f"unsupported movement agent kind: {kind}")


def _restore_movement_training_payload(agent, payload):
    kind = _agent_kind(agent)
    if kind == "td3":
        agent.actor_optimizer.load_state_dict(payload["td3_optimizers"]["actor"])
        agent.critic_1_optimizer.load_state_dict(
            payload["td3_optimizers"]["critic_1"]
        )
        agent.critic_2_optimizer.load_state_dict(
            payload["td3_optimizers"]["critic_2"]
        )
        counters = payload["td3_counters"]
        hyperparameters = payload["td3_hyperparameters"]
    elif kind == "ddpg":
        optimizers = payload["movement_agent_optimizers"]
        agent.actor_optimizer.load_state_dict(optimizers["actor"])
        agent.critic_optimizer.load_state_dict(optimizers["critic"])
        counters = payload["movement_agent_counters"]
        hyperparameters = payload["movement_agent_hyperparameters"]
    elif kind == "random":
        counters = payload["movement_agent_counters"]
        hyperparameters = payload["movement_agent_hyperparameters"]
    else:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")
    agent.num_critic_update_iteration = int(counters["critic_updates"])
    agent.num_actor_update_iteration = int(counters["actor_updates"])
    agent.num_training = int(counters["training_updates"])
    for name, value in hyperparameters.items():
        setattr(agent, name, value)


def _routing_training_payload(agent):
    kind = _routing_agent_kind(agent)
    state = {
        "kind": kind,
        "gamma": float(agent.gamma),
        "tau": float(agent.tau) if agent.tau is not None else None,
        "training_updates": int(agent.num_training),
        "target_update_count": int(getattr(agent, "target_update_count", 0)),
        "loss_log": list(agent.loss_log),
    }
    if kind == "safe_ddqn":
        return {
            "ddqn_optimizers": {
                "reward": agent.optimizer.state_dict(),
                "cost": agent.cost_optimizer.state_dict(),
            },
            "ddqn_state": {
                **state,
                "eta": float(agent.eta),
                "cost_loss_log": list(agent.cost_loss_log),
            },
        }
    if kind == "dqn":
        return {
            "routing_agent_optimizer": agent.optimizer.state_dict(),
            "routing_agent_state": state,
        }
    if kind == "random":
        return {"routing_agent_state": state}
    raise ValueError(f"unsupported routing agent kind: {kind}")


def _restore_routing_training_payload(agent, payload):
    kind = _routing_agent_kind(agent)
    if kind == "safe_ddqn":
        agent.optimizer.load_state_dict(payload["ddqn_optimizers"]["reward"])
        agent.cost_optimizer.load_state_dict(payload["ddqn_optimizers"]["cost"])
        state = payload["ddqn_state"]
        if state.get("kind", "safe_ddqn") != "safe_ddqn":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        agent.eta = float(state["eta"])
        agent.cost_loss_log = list(state["cost_loss_log"])
    elif kind == "dqn":
        state = payload.get("routing_agent_state") or {}
        if state.get("kind") != "dqn":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        agent.optimizer.load_state_dict(payload["routing_agent_optimizer"])
    elif kind == "random":
        state = payload.get("routing_agent_state") or {}
        if state.get("kind") != "random":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported routing agent kind: {kind}")
    agent.gamma = float(state["gamma"])
    if state.get("tau") is not None:
        agent.tau = float(state["tau"])
    agent.num_training = int(state["training_updates"])
    agent.target_update_count = int(state.get("target_update_count", 0))
    agent.loss_log = list(state["loss_log"])


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_full_resume_checkpoint(
    checkpoint_dir,
    *,
    episode,
    td3,
    ddqn,
    joint_replay,
    routing_replay,
    training_state,
    formal_config,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    experiment_metadata=None,
    keep_last=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    _validate_full_resume_logging_state(training_state, int(episode) + 1)
    metadata = _base_metadata(
        FULL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
        experiment_metadata,
    )
    _validate_effective_formal_movement_config(
        formal_config, _agent_kind(td3), metadata
    )
    experiment = metadata.get("experiment") or {}
    if "formal_config" in experiment:
        _validate_effective_formal_movement_config(
            experiment["formal_config"], _agent_kind(td3), metadata
        )
    def write(temporary):
        replay_metadata = {
            "joint": _save_replay(
                temporary / "joint_replay.npz", joint_replay, JOINT_REPLAY_FIELDS
            ),
            "routing": _save_replay(
                temporary / "routing_replay.npz",
                routing_replay,
                ROUTING_REPLAY_FIELDS,
            ),
        }
        payload = {
            "networks": _network_states(td3, ddqn),
            **_movement_training_payload(td3),
            **_routing_training_payload(ddqn),
            "training_state": dict(training_state),
            "formal_config": dict(formal_config),
            "replay_metadata": replay_metadata,
            "rng_state": _rng_state(),
        }
        torch.save(payload, temporary / "training_state.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    saved = _atomic_checkpoint_write(checkpoint_dir, write)
    if keep_last is not None:
        prune_full_resume_checkpoints(saved.parent, keep_last)
    return saved


def _validate_full_resume_logging_state(training_state, completed_episode):
    if not isinstance(training_state, dict):
        raise RuntimeError("checkpoint training state is invalid")
    schema_version = training_state.get("full_resume_logging_schema_version")
    if schema_version != FULL_RESUME_LOGGING_SCHEMA_VERSION:
        raise RuntimeError(
            "checkpoint full-resume logging schema is incompatible: "
            f"checkpoint={schema_version}, "
            f"expected={FULL_RESUME_LOGGING_SCHEMA_VERSION}"
        )
    expected_length = int(completed_episode)
    dinkelbach_active = training_state.get("dinkelbach_active", True) is not False
    lambda_fields = {"lambda_used_log", "lambda_after_episode_log"}
    for field in FULL_RESUME_LOGGING_STATE_FIELDS[1:]:
        values = training_state.get(field)
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(
                f"checkpoint full-resume logging state is incomplete: {field}"
            )
        if len(values) != expected_length:
            raise RuntimeError(
                "checkpoint full-resume logging length is incompatible: "
                f"{field}={len(values)}, completed_episodes={expected_length}"
            )
        if not dinkelbach_active and field in lambda_fields:
            finite = all(value is None for value in values)
        else:
            try:
                finite = all(np.isfinite(float(value)) for value in values)
            except (TypeError, ValueError):
                finite = False
        if not finite:
            raise RuntimeError(
                f"checkpoint full-resume logging state is non-finite: {field}"
            )


def _validate_full_metadata(
    metadata,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
    expected_experiment_metadata=None,
    movement_agent_kind=None,
):
    if metadata.get("checkpoint_type") != FULL_CHECKPOINT_TYPE:
        raise RuntimeError(
            "model-only checkpoint can only be used for evaluation, not exact resume"
        )
    schema = _validate_checkpoint_schema(metadata)
    checks = {
        "movement_state_dim": int(movement_state_dim),
        "joint_action_dim": int(joint_action_dim),
        "routing_state_dim": int(routing_state_dim),
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint {key} is incompatible: checkpoint={actual}, current={expected}"
            )
    if movement_agent_kind is not None:
        actual_kind = metadata.get("movement_agent_kind", "td3")
        if actual_kind != movement_agent_kind:
            raise RuntimeError(
                "checkpoint movement_agent_kind is incompatible: "
                f"checkpoint={actual_kind}, current={movement_agent_kind}"
            )
    _validate_td3_gamma_alias(metadata)
    movement_gamma_key = _movement_gamma_key(metadata, schema)
    gamma_checks = {
        movement_gamma_key: float(td3_gamma),
        "routing_ddqn_gamma": float(ddqn_gamma),
    }
    for key, expected in gamma_checks.items():
        actual = float(metadata.get(key, float("nan")))
        if not np.isclose(actual, expected):
            raise RuntimeError(
                f"checkpoint {key} is incompatible: checkpoint={actual}, current={expected}"
            )
    expected_fingerprint = calibration_fingerprint(calibration)
    if metadata.get("com_calibration_fingerprint") != expected_fingerprint:
        raise RuntimeError("checkpoint COM calibration fingerprint is incompatible")
    if expected_experiment_metadata is not None:
        validate_checkpoint_experiment_metadata(
            metadata, expected_experiment_metadata
        )


def validate_checkpoint_experiment_metadata(metadata, expected):
    """Validate only requested experiment identity fields for compatibility."""

    actual = metadata.get("experiment")
    if actual is None:
        raise RuntimeError("checkpoint has no experiment identity metadata")
    mismatches = {}
    for key, value in expected.items():
        actual_value = actual.get(key)
        if isinstance(value, (tuple, list, set, frozenset)):
            matches = actual_value in value
        else:
            matches = actual_value == value
        if not matches:
            mismatches[key] = (actual_value, value)
    if mismatches:
        raise RuntimeError(
            f"checkpoint experiment metadata is incompatible: {mismatches}"
        )


def inspect_full_resume_checkpoint(
    checkpoint_dir,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
    expected_experiment_metadata=None,
    expected_formal_config=None,
    require_episode_directory=False,
    movement_agent_kind=None,
):
    """Validate an exact-resume checkpoint without mutating training state."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        legacy_metadata_path = checkpoint_dir / "meta.json"
        if legacy_metadata_path.is_file():
            legacy = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
            legacy_dim = legacy.get("movement_state_dim", "unknown")
            raise RuntimeError(
                "legacy/model-only checkpoint "
                f"({legacy_dim}-D movement state) can only be used for "
                "evaluation and is incompatible with exact resume"
            )
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_full_metadata(
        metadata,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3_gamma,
        ddqn_gamma=ddqn_gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        movement_agent_kind=movement_agent_kind,
    )
    completed_episode = (
        _checkpoint_directory_episode(checkpoint_dir, metadata)
        if require_episode_directory
        else int(metadata["episode"]) + 1
    )
    required_paths = (
        checkpoint_dir / "training_state.pt",
        checkpoint_dir / "joint_replay.npz",
        checkpoint_dir / "routing_replay.npz",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"checkpoint has incomplete full-resume state; missing={missing}"
        )
    _validate_joint_replay_projection_masks(checkpoint_dir, metadata)
    payload = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint full-resume payload is invalid")
    training_state = payload.get("training_state")
    formal_config = payload.get("formal_config")
    if not isinstance(training_state, dict):
        raise RuntimeError("checkpoint training state is invalid")
    try:
        state_completed = int(training_state["completed_episode_index"]) + 1
        state_next = int(training_state["next_episode_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint training episode state is invalid") from exc
    if state_completed != completed_episode or state_next != completed_episode:
        raise RuntimeError(
            "checkpoint training episode state is incompatible with metadata: "
            f"completed={state_completed}, next={state_next}, "
            f"metadata={completed_episode}"
        )
    if expected_formal_config is not None:
        experiment = metadata.get("experiment") or {}
        _validate_formal_config(
            _formal_config_for_validation(
                metadata, experiment.get("formal_config"), expected_formal_config
            ),
            expected_formal_config,
            FULL_RESUME_CONFIG_FIELDS,
        )
        _validate_formal_config(
            _formal_config_for_validation(
                metadata, formal_config, expected_formal_config
            ),
            expected_formal_config,
            FULL_RESUME_CONFIG_FIELDS,
        )
        if _checkpoint_uses_dinkelbach(metadata):
            _validate_dinkelbach_checkpoint_metadata(
                metadata, expected_formal_config
            )
    if not isinstance(formal_config, dict):
        raise RuntimeError("checkpoint formal training configuration is invalid")
    if (
        metadata["checkpoint_schema_version"]
        >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
    ):
        kind = metadata.get("movement_agent_kind", "td3")
        experiment = metadata.get("experiment") or {}
        _validate_effective_formal_movement_config(formal_config, kind, metadata)
        if "formal_config" in experiment:
            _validate_effective_formal_movement_config(
                experiment["formal_config"], kind, metadata
            )
    _validate_full_resume_logging_state(training_state, completed_episode)
    if _checkpoint_uses_dinkelbach(metadata):
        dinkelbach_state = DinkelbachBlockState.from_training_state(
            training_state,
            formal_config,
            expected_completed_episodes=completed_episode,
        )
        if completed_episode > 0 and not np.isclose(
            float(training_state["lambda_after_episode_log"][-1]),
            dinkelbach_state.current_lambda,
        ):
            raise RuntimeError(
                "checkpoint lambda-after-episode log is incompatible with "
                "Dinkelbach block state"
            )
    return {
        "checkpoint_dir": checkpoint_dir,
        "completed_episode": completed_episode,
        "metadata": metadata,
        "training_state": training_state,
        "formal_config": formal_config,
        "payload": payload,
    }


def load_full_resume_checkpoint(
    checkpoint_dir,
    *,
    td3,
    ddqn,
    joint_replay,
    routing_replay,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    expected_experiment_metadata=None,
    expected_formal_config=None,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    inspected = inspect_full_resume_checkpoint(
        checkpoint_dir,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3.gamma,
        ddqn_gamma=ddqn.gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_formal_config=expected_formal_config,
        movement_agent_kind=_agent_kind(td3),
    )
    metadata = inspected["metadata"]
    payload = inspected["payload"]
    _load_network_states(payload["networks"], td3, ddqn)
    _restore_movement_training_payload(td3, payload)
    _restore_routing_training_payload(ddqn, payload)

    replay_metadata = payload["replay_metadata"]
    _load_replay(
        checkpoint_dir / "joint_replay.npz",
        joint_replay,
        (
            JOINT_REPLAY_FIELDS
            if metadata["checkpoint_schema_version"] >= CHECKPOINT_SCHEMA_VERSION
            else (
                PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS
                if metadata["checkpoint_schema_version"]
                >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
                else LEGACY_JOINT_REPLAY_FIELDS
            )
        ),
        replay_metadata["joint"],
    )
    if metadata["checkpoint_schema_version"] < CHECKPOINT_SCHEMA_VERSION:
        _reconstruct_legacy_full_observation_masks(joint_replay, metadata)
    _load_replay(
        checkpoint_dir / "routing_replay.npz",
        routing_replay,
        ROUTING_REPLAY_FIELDS,
        replay_metadata["routing"],
    )
    _restore_rng_state(payload["rng_state"])
    return {
        "metadata": metadata,
        "training_state": payload["training_state"],
        "formal_config": payload["formal_config"],
    }
