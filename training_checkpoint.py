import hashlib
import json
import random
from pathlib import Path

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


def calibration_fingerprint(calibration):
    canonical = json.dumps(
        calibration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
):
    return {
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
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = _base_metadata(
        MODEL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
    )
    torch.save(_network_states(td3, ddqn), checkpoint_dir / "models.pt")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint_dir


def load_model_checkpoint(checkpoint_dir, td3, ddqn):
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
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
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = _base_metadata(
        FULL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
    )
    replay_metadata = {
        "joint": _save_replay(
            checkpoint_dir / "joint_replay.npz", joint_replay, JOINT_REPLAY_FIELDS
        ),
        "routing": _save_replay(
            checkpoint_dir / "routing_replay.npz",
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
    torch.save(payload, checkpoint_dir / "training_state.pt")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint_dir


def _validate_full_metadata(
    metadata,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
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
):
    checkpoint_dir = Path(checkpoint_dir)
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
        td3_gamma=td3.gamma,
        ddqn_gamma=ddqn.gamma,
        calibration=calibration,
    )
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.is_file():
        raise RuntimeError(
            "checkpoint has no full-resume state; it can only be used for evaluation"
        )
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
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
