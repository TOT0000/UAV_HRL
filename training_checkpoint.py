import hashlib
import json
import random
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
MODEL_CHECKPOINT_TYPE = "model-only"
FULL_CHECKPOINT_TYPE = "full-resume"

JOINT_REPLAY_FIELDS = (
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


def _network_states(td3, ddqn):
    return {
        "td3": {
            "actor": td3.actor.state_dict(),
            "actor_target": td3.actor_target.state_dict(),
            "critic_1": td3.critic_1.state_dict(),
            "critic_1_target": td3.critic_1_target.state_dict(),
            "critic_2": td3.critic_2.state_dict(),
            "critic_2_target": td3.critic_2_target.state_dict(),
        },
        "ddqn": {
            "q_network": ddqn.q_network.state_dict(),
            "target_q_network": ddqn.target_q_network.state_dict(),
            "cost_network": ddqn.cost_network.state_dict(),
            "target_cost_network": ddqn.target_cost_network.state_dict(),
        },
    }


def _load_network_states(payload, td3, ddqn):
    td3_state = payload["td3"]
    td3.actor.load_state_dict(td3_state["actor"])
    td3.actor_target.load_state_dict(td3_state["actor_target"])
    td3.critic_1.load_state_dict(td3_state["critic_1"])
    td3.critic_1_target.load_state_dict(td3_state["critic_1_target"])
    td3.critic_2.load_state_dict(td3_state["critic_2"])
    td3.critic_2_target.load_state_dict(td3_state["critic_2_target"])

    ddqn_state = payload["ddqn"]
    ddqn.q_network.load_state_dict(ddqn_state["q_network"])
    ddqn.target_q_network.load_state_dict(ddqn_state["target_q_network"])
    ddqn.cost_network.load_state_dict(ddqn_state["cost_network"])
    ddqn.target_cost_network.load_state_dict(ddqn_state["target_cost_network"])


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
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": checkpoint_type,
        "episode": int(episode),
        "movement_state_dim": int(movement_state_dim),
        "joint_action_dim": int(joint_action_dim),
        "routing_state_dim": int(routing_state_dim),
        "centralized_td3_gamma": float(td3.gamma),
        "routing_ddqn_gamma": float(ddqn.gamma),
        "com_calibration_fingerprint": calibration_fingerprint(calibration),
    }
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
):
    """Validate evaluation provenance before any model payload is loaded."""

    checks = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": MODEL_CHECKPOINT_TYPE,
    }
    optional_checks = {
        "movement_state_dim": movement_state_dim,
        "joint_action_dim": joint_action_dim,
        "routing_state_dim": routing_state_dim,
        "centralized_td3_gamma": td3_gamma,
        "routing_ddqn_gamma": ddqn_gamma,
    }
    checks.update(
        {key: value for key, value in optional_checks.items() if value is not None}
    )
    for key, expected in checks.items():
        actual = metadata.get(key)
        if not _metadata_value_matches(actual, expected):
            raise RuntimeError(
                f"checkpoint {key} is incompatible: "
                f"checkpoint={actual}, expected={expected}"
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
            experiment.get("formal_config"),
            expected_formal_config,
            FORMAL_CORE_CONFIG_FIELDS,
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


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


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
            "td3_optimizers": {
                "actor": td3.actor_optimizer.state_dict(),
                "critic_1": td3.critic_1_optimizer.state_dict(),
                "critic_2": td3.critic_2_optimizer.state_dict(),
            },
            "td3_counters": {
                "critic_updates": int(td3.num_critic_update_iteration),
                "actor_updates": int(td3.num_actor_update_iteration),
                "training_updates": int(td3.num_training),
            },
            "td3_hyperparameters": {
                "gamma": float(td3.gamma),
                "tau": float(td3.tau),
                "policy_delay": int(td3.policy_delay),
                "policy_noise": float(td3.policy_noise),
                "noise_clip": float(td3.noise_clip),
                "max_action": float(td3.max_action),
            },
            "ddqn_optimizers": {
                "reward": ddqn.optimizer.state_dict(),
                "cost": ddqn.cost_optimizer.state_dict(),
            },
            "ddqn_state": {
                "eta": float(ddqn.eta),
                "tau": float(ddqn.tau),
                "gamma": float(ddqn.gamma),
                "training_updates": int(ddqn.num_training),
                "loss_log": list(ddqn.loss_log),
                "cost_loss_log": list(ddqn.cost_loss_log),
            },
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
):
    if metadata.get("checkpoint_type") != FULL_CHECKPOINT_TYPE:
        raise RuntimeError(
            "model-only checkpoint can only be used for evaluation, not exact resume"
        )
    checks = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
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
    gamma_checks = {
        "centralized_td3_gamma": float(td3_gamma),
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
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
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
            experiment.get("formal_config"),
            expected_formal_config,
            FULL_RESUME_CONFIG_FIELDS,
        )
        _validate_formal_config(
            formal_config,
            expected_formal_config,
            FULL_RESUME_CONFIG_FIELDS,
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
    )
    metadata = inspected["metadata"]
    payload = inspected["payload"]
    _load_network_states(payload["networks"], td3, ddqn)
    td3.actor_optimizer.load_state_dict(payload["td3_optimizers"]["actor"])
    td3.critic_1_optimizer.load_state_dict(payload["td3_optimizers"]["critic_1"])
    td3.critic_2_optimizer.load_state_dict(payload["td3_optimizers"]["critic_2"])
    td3.num_critic_update_iteration = int(
        payload["td3_counters"]["critic_updates"]
    )
    td3.num_actor_update_iteration = int(payload["td3_counters"]["actor_updates"])
    td3.num_training = int(payload["td3_counters"]["training_updates"])
    for name, value in payload["td3_hyperparameters"].items():
        setattr(td3, name, value)

    ddqn.optimizer.load_state_dict(payload["ddqn_optimizers"]["reward"])
    ddqn.cost_optimizer.load_state_dict(payload["ddqn_optimizers"]["cost"])
    ddqn_state = payload["ddqn_state"]
    ddqn.eta = float(ddqn_state["eta"])
    ddqn.tau = float(ddqn_state["tau"])
    ddqn.gamma = float(ddqn_state["gamma"])
    ddqn.num_training = int(ddqn_state["training_updates"])
    ddqn.loss_log = list(ddqn_state["loss_log"])
    ddqn.cost_loss_log = list(ddqn_state["cost_loss_log"])

    replay_metadata = payload["replay_metadata"]
    _load_replay(
        checkpoint_dir / "joint_replay.npz",
        joint_replay,
        JOINT_REPLAY_FIELDS,
        replay_metadata["joint"],
    )
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
