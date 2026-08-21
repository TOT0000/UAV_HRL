import argparse
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass
import hashlib
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DDQN import DDQN
from dinkelbach_blocks import (
    DINKELBACH_DENOMINATOR_UNIT,
    DINKELBACH_INITIAL_LAMBDA,
    DINKELBACH_NUMERATOR_UNIT,
    DINKELBACH_UPDATE_INTERVAL_EPISODES,
    DINKELBACH_UPDATE_RULE,
    DinkelbachBlockState,
    dinkelbach_config_metadata,
    validate_dinkelbach_config,
)
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    apply_joint_movement_proposals,
    build_joint_movement_proposals,
    calculate_movement_potentials,
    get_global_movement_state,
    project_joint_action,
)
from com_capacity_calibration import load_com_capacity_reference
from exploration_schedules import (
    ddqn_decay_steps,
    ddqn_epsilon,
    movement_behavior_noise,
    movement_decay_steps,
)
from evaluation_metrics import safe_energy_efficiency
from experiment_config import (
    DEFAULT_TRAINING_SEED,
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    NUM_UAV,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
)
from movement_agents import create_movement_agent
from training_checkpoint import (
    FULL_RESUME_LOGGING_SCHEMA_VERSION,
    checkpoint_episode_schedule,
    checkpoint_metadata_fingerprint,
    load_full_resume_checkpoint,
    load_model_checkpoint,
    save_full_resume_checkpoint,
    save_model_checkpoint,
)
from training_history import (
    TRAINING_HISTORY_COMMIT,
    TRAINING_HISTORY_CSV,
    TRAINING_HISTORY_JSONL,
    build_training_history_row,
    prepare_training_history,
    training_history_identity,
    write_training_history,
)
import utils_update_v2


MOVEMENT_CONTROL_INTERVAL = 4
ROUTING_STATE_DIM = 126
PRODUCTION_WARMUP_TRANSITIONS = FORMAL_EXPERIMENT_DEFAULTS[
    "movement_hyperparameters"
]["warmup_joint_transitions"]
PRODUCTION_BATCH_SIZE = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
    "batch_size"
]
PRODUCTION_POLICY_DELAY = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
    "td3"
]["policy_delay"]
SMOKE_RANDOM_SEED = DEFAULT_TRAINING_SEED


def _seed_training_rng(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_movement_decision(slot, interval=MOVEMENT_CONTROL_INTERVAL):
    return slot % interval == 0


def _is_last_movement_decision(
    slot, total_slots, interval=MOVEMENT_CONTROL_INTERVAL
):
    return _is_movement_decision(slot, interval) and slot + interval >= total_slots


def _is_episode_end(slot, total_slots):
    return slot == total_slots - 1


def _search_transition_done(search_done, movement_episode_end):
    return bool(search_done or movement_episode_end)


def _uses_warmup_random_action(total_joint_transitions, warmup_transitions):
    return int(total_joint_transitions) < int(warmup_transitions)


def _create_active_replay_buffers(state_dim, routing_dim, max_size=int(2e5)):
    """Legacy test helper; the active movement path uses ReplayBufferJoint."""
    routing_buffer = utils_update_v2.ReplayBufferDiscrete(
        state_dim,
        action_dim=routing_dim,
        max_size=max_size,
        n_step=1,
        gamma=0.99,
    )
    movement_buffer_search = utils_update_v2.ReplayBufferContinuous(
        state_dim, action_dim=3, max_size=max_size, n_step=1, gamma=0.99
    )
    movement_buffer_fov = utils_update_v2.ReplayBufferContinuous(
        state_dim, action_dim=3, max_size=max_size, n_step=1, gamma=0.99
    )
    return routing_buffer, movement_buffer_search, movement_buffer_fov


@dataclass
class TrainingConfig:
    total_episodes: int
    mode: str = "train"
    episode_seconds: int = 60
    routing_slot_seconds: float = 0.25
    warmup_joint_transitions: int = PRODUCTION_WARMUP_TRANSITIONS
    batch_size: int = PRODUCTION_BATCH_SIZE
    policy_delay: int = PRODUCTION_POLICY_DELAY
    replay_max_size: int = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
        "replay_size"
    ]
    beta_search: float = 1.0
    beta_vs: float = 1.0
    beta_com: float = 1.0
    search_coverage_threshold: float = 0.99
    dinkelbach_initial_lambda: float = DINKELBACH_INITIAL_LAMBDA
    dinkelbach_update_interval_episodes: int = DINKELBACH_UPDATE_INTERVAL_EPISODES
    dinkelbach_update_rule: str = DINKELBACH_UPDATE_RULE
    dinkelbach_numerator_unit: str = DINKELBACH_NUMERATOR_UNIT
    dinkelbach_denominator_unit: str = DINKELBACH_DENOMINATOR_UNIT
    model_checkpoint_every: int = 50
    full_resume_every: int = 50
    full_resume_keep_last: int = 2
    formal_evaluation_episode: int = FORMAL_CHECKPOINT_EPISODE
    checkpoint_root: str = "checkpoints_centralized_td3"
    resume_dir: str | None = None
    enable_model_checkpoints: bool = True
    enable_full_resume: bool = True
    enable_plots: bool = True
    enable_csv: bool = True
    random_seed: int | None = None
    run_directory: str | None = None

    def __post_init__(self):
        if self.mode not in {"smoke", "train", "custom"}:
            raise ValueError(f"unsupported training mode: {self.mode}")
        slots = 1.0 / float(self.routing_slot_seconds)
        if not np.isclose(slots, MOVEMENT_CONTROL_INTERVAL):
            raise ValueError("movement interval must contain exactly four routing slots")
        if self.episode_seconds <= 0 or self.total_episodes <= 0:
            raise ValueError("episode_seconds and total_episodes must be positive")
        if self.warmup_joint_transitions < 0 or self.batch_size <= 0:
            raise ValueError("warmup must be non-negative and batch_size positive")
        if self.full_resume_keep_last <= 0:
            raise ValueError("full_resume_keep_last must be positive")
        if self.formal_evaluation_episode <= 0:
            raise ValueError("formal_evaluation_episode must be positive")
        validate_dinkelbach_config(self)


def smoke_training_config():
    return TrainingConfig(
        total_episodes=1,
        mode="smoke",
        episode_seconds=60,
        routing_slot_seconds=0.25,
        warmup_joint_transitions=0,
        batch_size=1,
        policy_delay=2,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=SMOKE_RANDOM_SEED,
    )


def formal_training_config(total_episodes=None, **overrides):
    if total_episodes is None:
        total_episodes = FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"]
    values = {
        "total_episodes": int(total_episodes),
        "mode": "train",
        "episode_seconds": FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"],
        "routing_slot_seconds": FORMAL_EXPERIMENT_DEFAULTS["routing_slot_seconds"],
        "warmup_joint_transitions": PRODUCTION_WARMUP_TRANSITIONS,
        "batch_size": PRODUCTION_BATCH_SIZE,
        "policy_delay": PRODUCTION_POLICY_DELAY,
        "dinkelbach_initial_lambda": DINKELBACH_INITIAL_LAMBDA,
        "dinkelbach_update_interval_episodes": DINKELBACH_UPDATE_INTERVAL_EPISODES,
        "dinkelbach_update_rule": DINKELBACH_UPDATE_RULE,
        "dinkelbach_numerator_unit": DINKELBACH_NUMERATOR_UNIT,
        "dinkelbach_denominator_unit": DINKELBACH_DENOMINATOR_UNIT,
        "model_checkpoint_every": FORMAL_EXPERIMENT_DEFAULTS[
            "checkpoint_interval_episodes"
        ],
        "full_resume_every": FORMAL_EXPERIMENT_DEFAULTS[
            "checkpoint_interval_episodes"
        ],
        "full_resume_keep_last": 2,
        "formal_evaluation_episode": FORMAL_CHECKPOINT_EPISODE,
        "random_seed": DEFAULT_TRAINING_SEED,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _routing_masks(env):
    return {
        uav_id: env.get_routing_action_mask(uav_id).astype(bool)
        for uav_id in range(env.num_UAV)
    }


def _active_backlog(packet_engine):
    return defaultdict(float, packet_engine.backlog_bits)


def _routing_transition_done(episode_done, next_hol):
    """Cut DDQN bootstrap when this UAV has no next routing decision."""

    return bool(episode_done or next_hol is None)


def _run_routing_slot(
    env,
    packet_engine,
    ddqn,
    routing_buffer,
    routing_masks,
    current_time,
    done,
    delay_bound_steps,
    violation_stats,
    epsilon,
    write_replay=True,
):
    step_time = float(packet_engine.step_time)
    env.current_time = float(current_time)
    # Source generation precedes cleanup by contract. Prior-slot expirations are
    # normally already gone; this makes the boundary explicit and idempotent.
    packet_engine.inject_packets(
        env, delay_bound_steps, env.current_time, step_time=step_time
    )
    packet_engine.expire_packets(env.current_time, inclusive=True)
    packet_engine.drop_expired_packets(env.current_time)
    backlog_before = _active_backlog(packet_engine)
    uavs_with_packets = packet_engine.nonempty_uav_ids()
    effective_masks = {
        uid: packet_engine.get_effective_action_mask(
            env, uid, routing_masks[uid]
        )
        for uid in uavs_with_packets
    }

    states = {
        uid: packet_engine.get_state_ta(
            env,
            uid,
            backlog_bits=backlog_before,
            action_mask=effective_masks[uid],
        )
        for uid in uavs_with_packets
    }
    for uid, state in states.items():
        if state.shape != (ROUTING_STATE_DIM,):
            raise AssertionError(
                f"routing state for UAV {uid} has shape {state.shape}, "
                f"expected ({ROUTING_STATE_DIM},)"
            )

    next_hops = _select_routing_actions(
        ddqn, states, effective_masks, epsilon=epsilon
    )

    proposed_links = {
        sender: receiver
        for sender, receiver in next_hops.items()
        if receiver != sender and packet_engine.get_hol_packet(sender) is not None
    }
    active_capacities, _ = env.allocate_active_link_capacities(
        proposed_links
    )
    slot_result = packet_engine.serve_active_links(
        env,
        next_hops,
        active_capacities,
        current_time=env.current_time,
    )
    violation_count = sum(
        bool(outcome["violated"]) for outcome in slot_result["outcomes"]
    )
    attributed_cost = float(sum(slot_result["cost_by_sender"].values()))
    if not np.isclose(attributed_cost, float(violation_count)):
        raise AssertionError(
            "deadline violation cost attribution mismatch: "
            f"violations={violation_count}, cost={attributed_cost}"
        )
    env.current_time = float(current_time) + step_time
    for outcome in slot_result["outcomes"]:
        task_type = outcome["task_type"]
        if task_type not in violation_stats:
            continue
        if outcome["violated"]:
            violation_stats[task_type]["deadline_violated_packets"] += 1
        else:
            violation_stats[task_type]["timely_delivered_packets"] += 1

    backlog_after = _active_backlog(packet_engine)
    next_states = {
        uid: packet_engine.get_state_ta(
            env,
            uid,
            backlog_bits=backlog_after,
            action_mask=packet_engine.get_effective_action_mask(
                env, uid, routing_masks[uid]
            ),
        )
        for uid in uavs_with_packets
    }
    if write_replay:
        for uid in uavs_with_packets:
            next_hol = packet_engine.get_hol_packet(uid)
            transition_done = _routing_transition_done(done, next_hol)
            routing_buffer.add(
                states[uid],
                int(next_hops.get(uid, uid)),
                next_states.get(uid, states[uid]),
                float(slot_result["reward_by_sender"][uid]),
                float(slot_result["cost_by_sender"][uid]),
                transition_done,
                tag_gt=env.num_GT,
            )
    return (
        float(slot_result["timely_goodput_bits"]),
        float(sum(slot_result["reward_by_sender"].values())),
        len(next_hops),
    )


def _select_routing_actions(ddqn, states, routing_masks, epsilon):
    """Select all routing actions with one shared slot-level epsilon."""

    return {
        uid: int(
            ddqn.select_action(
                states[uid],
                uid,
                routing_masks[uid],
                visited_nodes=None,
                epsilon=epsilon,
                logits_noise_std=0.0,
            )
        )
        for uid in sorted(states)
    }


def _mark_search_observations(env):
    if getattr(env, "_search_phase_over", False):
        return
    for uav_id in range(env.num_UAV):
        has_search = any(
            task.get("task_type") == "Search"
            for task in env.multi_tasks.get(uav_id, [])
        )
        if has_search:
            env.update_visited_grid(uav_id)
            env.mark_search_coverage(uav_id)


def _dinkelbach_update(delivered_mbits, total_energy, previous_lambda):
    delivered_mbits = float(delivered_mbits)
    total_energy = float(total_energy)
    previous_lambda = float(previous_lambda)
    if (
        not np.isfinite(delivered_mbits)
        or not np.isfinite(total_energy)
        or total_energy <= 0.0
    ):
        return float(previous_lambda)
    ratio = delivered_mbits / total_energy
    return float(ratio) if np.isfinite(ratio) else float(previous_lambda)


def _interval_reward(
    delivered_mbits,
    energy,
    current_lambda,
    gamma,
    potentials_t,
    potentials_t1,
    done,
    config,
    reward_mode="dinkelbach",
    task_potential_enabled=True,
):
    next_values = (0.0, 0.0, 0.0) if done else potentials_t1
    shaping = float(bool(task_potential_enabled)) * (
        config.beta_search * (gamma * next_values[0] - potentials_t[0])
        + config.beta_vs * (gamma * next_values[1] - potentials_t[1])
        + config.beta_com * (gamma * next_values[2] - potentials_t[2])
    )
    if reward_mode == "dinkelbach":
        objective = float(delivered_mbits) - float(current_lambda) * float(energy)
    elif reward_mode == "ratio":
        objective = safe_energy_efficiency(delivered_mbits, energy)
    else:
        raise ValueError(f"unsupported reward mode: {reward_mode}")
    return float(objective + shaping)


def _is_checkpoint_episode(completed_episode, total_episodes, every):
    return int(completed_episode) in checkpoint_episode_schedule(
        total_episodes, every
    )


def _append_lambda_history(
    lambda_used_log,
    lambda_after_episode_log,
    *,
    lambda_used,
    lambda_after_episode,
    dinkelbach_active=True,
):
    if dinkelbach_active:
        lambda_used_log.append(float(lambda_used))
        lambda_after_episode_log.append(float(lambda_after_episode))
    else:
        lambda_used_log.append(None)
        lambda_after_episode_log.append(None)


def _inactive_dinkelbach_event(episode):
    """Compatibility row for the legacy history schema; never updates lambda."""

    return {
        "dinkelbach_lambda_used": 0.0,
        "dinkelbach_lambda_after_episode": 0.0,
        "dinkelbach_lambda_updated": False,
        "dinkelbach_update_status": "disabled_for_reward_mode",
        "dinkelbach_block_index": int(episode) + 1,
        "dinkelbach_block_episode": 1,
        "dinkelbach_block_timely_mbits_so_far": 0.0,
        "dinkelbach_block_energy_joules_so_far": 0.0,
        "dinkelbach_block_completed": False,
    }


def _legacy_training_frame(
    *,
    initial_log_length,
    reward_log,
    delivered_log,
    energy_log,
    lambda_used_log,
    lambda_after_episode_log,
):
    lengths = {
        len(reward_log),
        len(delivered_log),
        len(energy_log),
        len(lambda_used_log),
        len(lambda_after_episode_log),
    }
    if len(lengths) != 1:
        raise RuntimeError("legacy training logs have inconsistent lengths")
    return pd.DataFrame(
        {
            "episode": np.arange(initial_log_length, len(reward_log)),
            "reward": reward_log[initial_log_length:],
            "delivered_mbits": delivered_log[initial_log_length:],
            "mobility_energy": energy_log[initial_log_length:],
            "lambda_used": lambda_used_log[initial_log_length:],
            "lambda_after_episode": lambda_after_episode_log[
                initial_log_length:
            ],
        }
    )


def _full_training_state(
    *,
    episode,
    dinkelbach_state,
    reward_log,
    delivered_log,
    energy_log,
    lambda_used_log,
    lambda_after_episode_log,
    total_joint_transitions,
    routing_slots_executed,
    td3_noise_log,
    routing_epsilon_log,
    warmup_joint_transitions,
    training_history_rows,
    dinkelbach_active=True,
):
    return {
        "completed_episode_index": int(episode),
        "next_episode_index": int(episode) + 1,
        "full_resume_logging_schema_version": (
            FULL_RESUME_LOGGING_SCHEMA_VERSION
        ),
        **(
            dinkelbach_state.training_state()
            if dinkelbach_active
            else {"dinkelbach_active": False}
        ),
        "reward_log": list(reward_log),
        "delivered_log": list(delivered_log),
        "energy_log": list(energy_log),
        "lambda_used_log": list(lambda_used_log),
        "lambda_after_episode_log": list(lambda_after_episode_log),
        "total_joint_transitions": int(total_joint_transitions),
        "global_routing_slot": int(routing_slots_executed),
        "td3_post_warmup_transition": max(
            int(total_joint_transitions) - int(warmup_joint_transitions), 0
        ),
        "ddqn_schedule_slot": int(routing_slots_executed),
        "td3_noise_log": list(td3_noise_log),
        "movement_noise_log": list(td3_noise_log),
        "routing_epsilon_log": list(routing_epsilon_log),
        "training_history_rows": list(training_history_rows),
    }


def _experiment_identity(method_spec, scenario_manifest, training_seed, config):
    return {
        "method_id": method_spec.method_id,
        "method_spec": method_spec.to_dict(),
        "method_spec_fingerprint": method_spec.fingerprint,
        "manifest_hash": (
            scenario_manifest.content_hash
            if scenario_manifest is not None
            else None
        ),
        "manifest_split": (
            scenario_manifest.split if scenario_manifest is not None else None
        ),
        "manifest_generation_profile": (
            scenario_manifest.generation_profile
            if scenario_manifest is not None
            else None
        ),
        "training_seed": (
            int(training_seed) if training_seed is not None else None
        ),
        "movement_agent": method_spec.agent,
        "reward_mode": method_spec.reward_mode,
        "task_potential_enabled": bool(method_spec.task_potential_enabled),
        **(
            dinkelbach_config_metadata(config)
            if method_spec.uses_dinkelbach
            else {"dinkelbach_active": False}
        ),
    }


def _evaluation_state_snapshot(
    movement_agent, ddqn, joint_replay, routing_replay, dinkelbach_state
):
    """Copy all learning state that evaluation is forbidden to mutate."""

    def replay_snapshot(replay, fields):
        size = int(replay.size)
        snapshot = {
            "size": size,
            "ptr": int(replay.ptr),
            "fields": {
                field: np.asarray(getattr(replay, field))[:size].copy()
                for field in fields
            },
        }
        if hasattr(replay, "n_step_buffer"):
            snapshot["n_step_buffer"] = copy.deepcopy(replay.n_step_buffer)
        return snapshot

    kind = movement_agent.agent_kind
    online = {
        "q_network": ddqn.q_network.state_dict(),
        "cost_network": ddqn.cost_network.state_dict(),
    }
    targets = {
        "target_q_network": ddqn.target_q_network.state_dict(),
        "target_cost_network": ddqn.target_cost_network.state_dict(),
    }
    optimizers = {
        "ddqn_reward": ddqn.optimizer.state_dict(),
        "ddqn_cost": ddqn.cost_optimizer.state_dict(),
    }
    if kind in {"td3", "ddpg"}:
        online["actor"] = movement_agent.actor.state_dict()
        targets["actor_target"] = movement_agent.actor_target.state_dict()
        optimizers["actor"] = movement_agent.actor_optimizer.state_dict()
    if kind == "td3":
        online.update(
            critic_1=movement_agent.critic_1.state_dict(),
            critic_2=movement_agent.critic_2.state_dict(),
        )
        targets.update(
            critic_1_target=movement_agent.critic_1_target.state_dict(),
            critic_2_target=movement_agent.critic_2_target.state_dict(),
        )
        optimizers.update(
            critic_1=movement_agent.critic_1_optimizer.state_dict(),
            critic_2=movement_agent.critic_2_optimizer.state_dict(),
        )
    elif kind == "ddpg":
        online["critic"] = movement_agent.critic.state_dict()
        targets["critic_target"] = movement_agent.critic_target.state_dict()
        optimizers["critic"] = movement_agent.critic_optimizer.state_dict()
    return {
        "online_networks": copy.deepcopy(online),
        "target_networks": copy.deepcopy(targets),
        "optimizers": copy.deepcopy(optimizers),
        "replays": {
            "joint": replay_snapshot(
                joint_replay,
                (
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
                ),
            ),
            "routing": replay_snapshot(
                routing_replay,
                ("state", "action", "next_state", "reward", "cost", "not_done", "tag_gt"),
            ),
        },
        "update_counters": {
            "movement_agent_kind": kind,
            "movement_critic": int(movement_agent.num_critic_update_iteration),
            "movement_actor": int(movement_agent.num_actor_update_iteration),
            "movement_training": int(movement_agent.num_training),
            "ddqn_training": int(ddqn.num_training),
            "ddqn_loss_log": list(ddqn.loss_log),
            "ddqn_cost_loss_log": list(ddqn.cost_loss_log),
        },
        "dinkelbach_state": copy.deepcopy(dinkelbach_state.training_state()),
    }


def _learning_state_fingerprint(value):
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            update(array)
        elif isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            digest.update(b"array")
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _nested_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_nested_state_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _evaluation_invariants(before, after, routing_epsilon_log, td3_noise_log):
    checks = {
        "online_networks_unchanged": _nested_state_equal(
            before["online_networks"], after["online_networks"]
        ),
        "target_networks_unchanged": _nested_state_equal(
            before["target_networks"], after["target_networks"]
        ),
        "optimizer_states_unchanged": _nested_state_equal(
            before["optimizers"], after["optimizers"]
        ),
        "replay_state_unchanged": _nested_state_equal(
            before["replays"], after["replays"]
        ),
        "update_counters_unchanged": _nested_state_equal(
            before["update_counters"], after["update_counters"]
        ),
        "dinkelbach_state_unchanged": _nested_state_equal(
            before["dinkelbach_state"], after["dinkelbach_state"]
        ),
        "exploration_disabled": (
            not td3_noise_log
            and all(float(epsilon) == 0.0 for epsilon in routing_epsilon_log)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"evaluation mutated learning state: {failed}")
    return checks


def train(
    config=None,
    *,
    scenario_manifest=None,
    method_spec=None,
    evaluation=False,
    checkpoint_dir=None,
    expected_checkpoint_episodes=None,
    expected_checkpoint_formal_config=None,
    transition_observer=None,
    episode_observer=None,
):
    if config is None:
        raise ValueError(
            "training config is required; use smoke_training_config() or "
            "formal_training_config(total_episodes)"
        )
    method_spec = method_spec or MethodSpec()
    # Reconstructing validates even subclasses or deserialized specifications.
    MethodSpec(**{
        key: value
        for key, value in method_spec.to_dict().items()
        if key != "method_id"
    })
    if scenario_manifest is not None:
        if scenario_manifest.episode_count < config.total_episodes:
            raise ValueError(
                "scenario manifest has fewer entries than requested episodes"
            )
        if config.mode == "train" and scenario_manifest.split != "train":
            raise ValueError("formal training requires a train scenario manifest")
        if evaluation and scenario_manifest.split not in {"validation", "test"}:
            raise ValueError("evaluation requires validation or test scenarios")
    if evaluation and checkpoint_dir is None:
        raise ValueError("evaluation requires a model checkpoint")
    if evaluation and config.random_seed is None:
        raise ValueError("evaluation requires the checkpoint training seed")
    if evaluation and config.resume_dir is not None:
        raise ValueError("evaluation cannot load a full-resume training state")
    if transition_observer is not None and not evaluation:
        raise ValueError("transition collection is available only in evaluation")
    _seed_training_rng(config.random_seed)

    c_ref_com, calibration = load_com_capacity_reference()
    env = Simulator(num_UAV=NUM_UAV)
    packet_engine = PacketEngine(
        num_uav=NUM_UAV, step_time=config.routing_slot_seconds
    )
    movement_agent = create_movement_agent(
        method_spec,
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        config,
    )
    ddqn = DDQN(ROUTING_STATE_DIM, env.num_UAV + 1)
    joint_replay = utils_update_v2.ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=config.replay_max_size,
    )
    routing_replay = utils_update_v2.ReplayBufferDiscrete(
        ROUTING_STATE_DIM,
        env.num_UAV + 1,
        max_size=config.replay_max_size,
        n_step=1,
        gamma=0.99,
    )

    experiment_identity = _experiment_identity(
        method_spec, scenario_manifest, config.random_seed, config
    )
    history_identity = None
    training_history_rows = []
    if config.run_directory is not None:
        if scenario_manifest is None or config.random_seed is None:
            raise ValueError(
                "persistent training history requires a manifest and training seed"
            )
        history_identity = training_history_identity(
            method_spec.method_id,
            config.random_seed,
            scenario_manifest.content_hash,
        )
    loaded_checkpoint_metadata = None
    checkpoint_provenance = {}
    if evaluation:
        loaded_checkpoint_metadata = load_model_checkpoint(
            checkpoint_dir,
            movement_agent,
            ddqn,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=calibration,
            expected_experiment_metadata={
                "method_spec_fingerprint": method_spec.fingerprint,
                "training_seed": int(config.random_seed),
            },
            expected_completed_episodes=expected_checkpoint_episodes,
            expected_formal_config=expected_checkpoint_formal_config,
        )
        checkpoint_experiment = loaded_checkpoint_metadata["experiment"]
        checkpoint_provenance = {
            "training_manifest_hash": checkpoint_experiment.get("manifest_hash"),
            "evaluation_manifest_hash": (
                scenario_manifest.content_hash
                if scenario_manifest is not None
                else None
            ),
            "checkpoint_completed_episodes": (
                int(loaded_checkpoint_metadata["episode"]) + 1
            ),
            "checkpoint_training_seed": checkpoint_experiment.get(
                "training_seed"
            ),
            "checkpoint_method_spec_fingerprint": checkpoint_experiment.get(
                "method_spec_fingerprint"
            ),
            "checkpoint_metadata_path": os.path.abspath(
                os.path.join(checkpoint_dir, "metadata.json")
            ),
            "checkpoint_metadata_fingerprint": (
                checkpoint_metadata_fingerprint(loaded_checkpoint_metadata)
            ),
            "checkpoint_dinkelbach_config": (
                dinkelbach_config_metadata(checkpoint_experiment["formal_config"])
                if method_spec.uses_dinkelbach
                else None
            ),
            "checkpoint_dinkelbach_state": (
                dict(checkpoint_experiment["dinkelbach_state"])
                if method_spec.uses_dinkelbach
                else None
            ),
        }

    dinkelbach_state = DinkelbachBlockState.from_config(config)
    lambda_ee = float(dinkelbach_state.current_lambda)
    if loaded_checkpoint_metadata is not None and method_spec.uses_dinkelbach:
        checkpoint_experiment = loaded_checkpoint_metadata["experiment"]
        dinkelbach_state = DinkelbachBlockState.from_training_state(
            checkpoint_experiment.get("dinkelbach_state", {}),
            checkpoint_experiment.get("formal_config", {}),
            expected_completed_episodes=(
                int(loaded_checkpoint_metadata["episode"]) + 1
            ),
        )
        lambda_ee = float(dinkelbach_state.current_lambda)
    start_episode = 0
    reward_log = []
    delivered_log = []
    energy_log = []
    lambda_used_log = []
    lambda_after_episode_log = []
    total_joint_transitions = 0
    routing_slots_executed = 0
    td3_noise_log = []
    routing_epsilon_log = []
    episode_metrics = []
    if config.resume_dir is not None:
        if not os.path.isdir(config.resume_dir):
            raise FileNotFoundError(f"centralized checkpoint not found: {config.resume_dir}")
        restored = load_full_resume_checkpoint(
            config.resume_dir,
            td3=movement_agent,
            ddqn=ddqn,
            joint_replay=joint_replay,
            routing_replay=routing_replay,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=calibration,
            expected_experiment_metadata={
                "method_spec_fingerprint": method_spec.fingerprint,
                "training_seed": (
                    int(config.random_seed)
                    if config.random_seed is not None
                    else None
                ),
                **(
                    {"manifest_hash": scenario_manifest.content_hash}
                    if scenario_manifest is not None
                    else {}
                ),
            },
            expected_formal_config=asdict(config),
        )
        training_state = restored["training_state"]
        start_episode = int(training_state["next_episode_index"])
        if method_spec.uses_dinkelbach:
            dinkelbach_state = DinkelbachBlockState.from_training_state(
                training_state,
                config,
                expected_completed_episodes=start_episode,
            )
            lambda_ee = float(dinkelbach_state.current_lambda)
        reward_log = list(training_state["reward_log"])
        delivered_log = list(training_state["delivered_log"])
        energy_log = list(training_state["energy_log"])
        lambda_used_log = list(training_state["lambda_used_log"])
        lambda_after_episode_log = list(
            training_state["lambda_after_episode_log"]
        )
        total_joint_transitions = int(training_state["total_joint_transitions"])
        routing_slots_executed = int(training_state["global_routing_slot"])
        td3_noise_log = list(training_state["td3_noise_log"])
        routing_epsilon_log = list(training_state["routing_epsilon_log"])
        if history_identity is not None:
            if "training_history_rows" not in training_state:
                raise RuntimeError(
                    "exact-resume checkpoint has no persistent training history"
                )
            training_history_rows = prepare_training_history(
                config.run_directory,
                history_identity,
                checkpoint_rows=training_state["training_history_rows"],
            )
        expected_post_warmup = max(
            total_joint_transitions - config.warmup_joint_transitions, 0
        )
        if int(training_state["td3_post_warmup_transition"]) != expected_post_warmup:
            raise RuntimeError("TD3 exploration counter is inconsistent with replay history")
        if int(training_state["ddqn_schedule_slot"]) != routing_slots_executed:
            raise RuntimeError("DDQN exploration counter is inconsistent with slot history")
    elif history_identity is not None:
        training_history_rows = prepare_training_history(
            config.run_directory, history_identity
        )

    evaluation_state_before = (
        _evaluation_state_snapshot(
            movement_agent,
            ddqn,
            joint_replay,
            routing_replay,
            dinkelbach_state,
        )
        if evaluation
        else None
    )

    initial_log_length = len(reward_log)
    duplicate_target_assertions = 0
    environment_actor_calls = 0
    proposal_batches = 0
    energy_evaluations = 0
    terminal_joint_transitions = 0
    td3_schedule_decay = movement_decay_steps(
        config.total_episodes,
        config.episode_seconds,
        config.warmup_joint_transitions,
    )
    ddqn_schedule_decay = ddqn_decay_steps(
        config.total_episodes,
        config.episode_seconds,
        MOVEMENT_CONTROL_INTERVAL,
    )
    delay_bound_steps = int(5.0 / config.routing_slot_seconds)

    ddqn_action_selections = 0
    executed_scenario_ids = []
    for episode in range(start_episode, config.total_episodes):
        if scenario_manifest is None:
            env.num_GT = int(
                np.random.randint(ROI_COUNT_MIN, ROI_COUNT_MAX + 1)
            )
            env.reset_environment()
            scenario_id = None
        else:
            scenario_entry = scenario_manifest.episodes[episode]
            env.apply_scenario_entry(scenario_entry)
            scenario_id = str(scenario_entry["scenario_id"])
        executed_scenario_ids.append(scenario_id)
        packet_engine.reset_packet_state()
        env.lambda_EE_global = float(lambda_ee)
        episode_lambda = float(lambda_ee)
        episode_delivered_mbits = 0.0
        episode_energy = 0.0
        episode_reward = 0.0
        episode_routing_reward = 0.0
        violation_stats = {
            "FOV": {
                "timely_delivered_packets": 0,
                "deadline_violated_packets": 0,
            },
            "COM": {
                "timely_delivered_packets": 0,
                "deadline_violated_packets": 0,
            },
        }

        for interval in range(config.episode_seconds):
            if interval == 0:
                # Match the existing slot-0 SR update. Later SR updates are part
                # of the preceding one-second transition so S_{t+1} == S_t next.
                env.advance_sr_teams()
            if getattr(env, "need_reassign", False):
                env.assign_tasks()
                env.need_reassign = False

            backlog_before = _active_backlog(packet_engine)
            try:
                state = get_global_movement_state(
                    env,
                    packet_engine,
                    backlog_before,
                    c_ref_com,
                    remaining_time=(config.episode_seconds - interval)
                    / config.episode_seconds,
                )
                potentials_t = calculate_movement_potentials(env, c_ref_com)
            except ValueError as exc:
                if "duplicate" in str(exc):
                    duplicate_target_assertions += 1
                raise

            if method_spec.agent == "random":
                raw_joint_action = np.random.uniform(
                    -1.0, 1.0, size=JOINT_ACTION_DIM
                ).astype(np.float32)
            elif evaluation:
                raw_joint_action = movement_agent.select_action(
                    state, add_noise=False, noise_std=0.0
                )
                environment_actor_calls += 1
            elif _uses_warmup_random_action(
                total_joint_transitions, config.warmup_joint_transitions
            ):
                raw_joint_action = np.random.uniform(
                    -1.0, 1.0, size=JOINT_ACTION_DIM
                ).astype(np.float32)
            else:
                behavior_noise = movement_behavior_noise(
                    total_joint_transitions - config.warmup_joint_transitions,
                    td3_schedule_decay,
                )
                raw_joint_action = movement_agent.select_action(
                    state, add_noise=True, noise_std=behavior_noise
                )
                td3_noise_log.append(behavior_noise)
                environment_actor_calls += 1
            projected_action = project_joint_action(raw_joint_action, state)

            # Phase 1 is read-only: all sixteen proposals are built from one snapshot.
            proposals = build_joint_movement_proposals(
                env, movement_agent, projected_action
            )
            proposal_batches += 1
            # Phase 2 mutates positions only after every proposal exists.
            interval_energies = apply_joint_movement_proposals(env, proposals)
            energy_evaluations += int(interval_energies.size)
            interval_energy = float(interval_energies.sum())

            _mark_search_observations(env)
            if getattr(env, "need_reassign", False):
                env.assign_tasks()
                env.need_reassign = False
            if (
                not getattr(env, "_search_phase_over", False)
                and float(env.visited_bitmap.mean())
                >= config.search_coverage_threshold
            ):
                env._search_phase_over = True
                env.convert_search_to_hovering()
            env.update_source_uavs()
            env.update_u2u_channels()
            env.update_u2g_channels()
            masks = _routing_masks(env)

            interval_delivered_bits = 0.0
            for routing_slot in range(MOVEMENT_CONTROL_INTERVAL):
                slot_epsilon = (
                    0.0
                    if evaluation
                    else ddqn_epsilon(
                        routing_slots_executed, ddqn_schedule_decay
                    )
                )
                routing_epsilon_log.append(slot_epsilon)
                routing_slots_executed += 1
                absolute_slot = interval * MOVEMENT_CONTROL_INTERVAL + routing_slot
                final_slot = (
                    interval == config.episode_seconds - 1
                    and routing_slot == MOVEMENT_CONTROL_INTERVAL - 1
                )
                delivered_bits, routing_reward, action_selections = _run_routing_slot(
                    env,
                    packet_engine,
                    ddqn,
                    routing_replay,
                    masks,
                    current_time=absolute_slot * config.routing_slot_seconds,
                    done=final_slot,
                    delay_bound_steps=delay_bound_steps,
                    violation_stats=violation_stats,
                    epsilon=slot_epsilon,
                    write_replay=not evaluation,
                )
                ddqn_action_selections += action_selections
                interval_delivered_bits += delivered_bits
                episode_routing_reward += routing_reward

            interval_delivered_mbits = interval_delivered_bits / 1e6
            done = interval == config.episode_seconds - 1
            if not done:
                env.advance_sr_teams()
            backlog_after = _active_backlog(packet_engine)
            potentials_t1 = calculate_movement_potentials(env, c_ref_com)
            next_state = get_global_movement_state(
                env,
                packet_engine,
                backlog_after,
                c_ref_com,
                remaining_time=(config.episode_seconds - (interval + 1))
                / config.episode_seconds,
            )
            terminal_joint_transitions += int(done)
            if not evaluation and method_spec.learns_movement:
                joint_replay.add(
                    state,
                    projected_action,
                    next_state,
                    done=done,
                    delivered_mbits=interval_delivered_mbits,
                    total_mobility_energy=interval_energy,
                    phi_search_t=potentials_t[0],
                    phi_search_t1=potentials_t1[0],
                    phi_vs_t=potentials_t[1],
                    phi_vs_t1=potentials_t1[1],
                    phi_com_t=potentials_t[2],
                    phi_com_t1=potentials_t1[2],
                )
            global_transition_index = total_joint_transitions
            total_joint_transitions += 1
            episode_delivered_mbits += interval_delivered_mbits
            episode_energy += interval_energy
            interval_reward = _interval_reward(
                interval_delivered_mbits,
                interval_energy,
                episode_lambda,
                movement_agent.gamma,
                potentials_t,
                potentials_t1,
                done,
                config,
                reward_mode=method_spec.reward_mode,
                task_potential_enabled=method_spec.task_potential_enabled,
            )
            episode_reward += interval_reward
            if transition_observer is not None:
                effective_potentials_t1 = (
                    (0.0, 0.0, 0.0) if done else potentials_t1
                )
                transition_observer(
                    {
                        "state": state.copy(),
                        "projected_joint_action": projected_action.copy(),
                        "next_state": next_state.copy(),
                        "done": done,
                        "not_done": 1.0 - float(done),
                        "delivered_mbits": interval_delivered_mbits,
                        "total_mobility_energy_j": interval_energy,
                        "phi_search_t": potentials_t[0],
                        "phi_search_t1": effective_potentials_t1[0],
                        "phi_vs_t": potentials_t[1],
                        "phi_vs_t1": effective_potentials_t1[1],
                        "phi_com_t": potentials_t[2],
                        "phi_com_t1": effective_potentials_t1[2],
                        "reward_at_checkpoint_lambda": interval_reward,
                        "checkpoint_lambda": (
                            episode_lambda if method_spec.uses_dinkelbach else None
                        ),
                        "episode_index": episode,
                        "movement_step": interval,
                        "global_transition_index": global_transition_index,
                        "scenario_index": episode,
                        "scenario_id": scenario_id,
                    }
                )

            if (
                not evaluation
                and method_spec.learns_movement
                and total_joint_transitions >= config.warmup_joint_transitions
                and joint_replay.size >= config.batch_size
            ):
                movement_agent.update_joint(
                    joint_replay,
                    current_lambda=episode_lambda,
                    batch_size=config.batch_size,
                    beta_search=config.beta_search,
                    beta_vs=config.beta_vs,
                    beta_com=config.beta_com,
                    reward_mode=method_spec.reward_mode,
                    task_potential_enabled=method_spec.task_potential_enabled,
                )

        if not evaluation and routing_replay.size >= config.batch_size:
            ddqn.train(routing_replay, config.batch_size)
        if not evaluation:
            ddqn.update_target()

        timely_goodput_mbits = float(packet_engine.timely_goodput_bits) / 1e6
        raw_final_hop_mbits = float(packet_engine.raw_final_hop_bits) / 1e6
        dinkelbach_event = None
        if not evaluation and method_spec.uses_dinkelbach:
            dinkelbach_event = dinkelbach_state.record_episode(
                timely_goodput_mbits,
                episode_energy,
            )
            lambda_ee = float(dinkelbach_state.current_lambda)
        elif not evaluation:
            dinkelbach_event = _inactive_dinkelbach_event(episode)
        env.lambda_EE_global = lambda_ee
        reward_log.append(episode_reward)
        delivered_log.append(episode_delivered_mbits)
        energy_log.append(episode_energy)
        _append_lambda_history(
            lambda_used_log,
            lambda_after_episode_log,
            lambda_used=episode_lambda,
            lambda_after_episode=lambda_ee,
            dinkelbach_active=method_spec.uses_dinkelbach,
        )
        coverage = float(env.visited_bitmap.mean())
        found_gt_ratio = (
            float(env.count_found_targets()) / float(env.num_GT)
            if int(env.num_GT) > 0
            else 0.0
        )
        episode_metrics.append(
            {
                "method_id": method_spec.method_id,
                "training_seed": (
                    int(config.random_seed)
                    if config.random_seed is not None
                    else None
                ),
                "evaluation_split": (
                    scenario_manifest.split
                    if evaluation and scenario_manifest is not None
                    else None
                ),
                "scenario_id": scenario_id,
                "evaluation_manifest_hash": (
                    scenario_manifest.content_hash
                    if scenario_manifest is not None
                    else None
                ),
                "training_manifest_hash": (
                    checkpoint_provenance.get("training_manifest_hash")
                    if evaluation
                    else (
                        scenario_manifest.content_hash
                        if scenario_manifest is not None
                        else None
                    )
                ),
                "checkpoint_completed_episodes": checkpoint_provenance.get(
                    "checkpoint_completed_episodes"
                ),
                "checkpoint_metadata_fingerprint": checkpoint_provenance.get(
                    "checkpoint_metadata_fingerprint"
                ),
                "num_GT": int(env.num_GT),
                "timely_goodput_mbits": timely_goodput_mbits,
                "raw_final_hop_mbits": raw_final_hop_mbits,
                "total_mobility_energy_j": float(episode_energy),
                "energy_efficiency_mbit_per_j": safe_energy_efficiency(
                    timely_goodput_mbits, episode_energy
                ),
                "fov_timely_delivered_packets": int(packet_engine.fov_delivered),
                "com_timely_delivered_packets": int(packet_engine.com_delivered),
                "fov_deadline_violations": int(packet_engine.fov_violated),
                "com_deadline_violations": int(packet_engine.com_violated),
                "total_deadline_violations": int(packet_engine.total_violated),
                "coverage": coverage,
                "found_GT_ratio": found_gt_ratio,
                "routing_wait_count": int(packet_engine.wait_actions),
                "partial_transmission_count": int(
                    packet_engine.partial_transmissions
                ),
                "slot_budget_violation_count": int(
                    packet_engine.link_slot_budget_violations
                ),
            }
        )
        if episode_observer is not None:
            episode_observer(
                {
                    **episode_metrics[-1],
                    "episode": episode + 1,
                    "reward": float(episode_reward),
                    "delivered_mbits": float(episode_delivered_mbits),
                    "mobility_energy_j": float(episode_energy),
                    "dinkelbach_lambda": (
                        float(lambda_ee) if method_spec.uses_dinkelbach else None
                    ),
                    "reward_mode": method_spec.reward_mode,
                    "task_potential_enabled": bool(
                        method_spec.task_potential_enabled
                    ),
                }
            )
        if history_identity is not None:
            training_history_rows.append(
                build_training_history_row(
                    history_identity,
                    episode=episode + 1,
                    reward=episode_reward,
                    timely_goodput_mbits=timely_goodput_mbits,
                    mobility_energy_j=episode_energy,
                    **dinkelbach_event,
                )
            )
            training_history_rows = write_training_history(
                config.run_directory,
                training_history_rows,
                history_identity,
            )
        lambda_summary = (
            f"lambda_used={episode_lambda:.9f} lambda_after={lambda_ee:.9f}"
            if method_spec.uses_dinkelbach
            else "dinkelbach=disabled"
        )
        print(
            f"[Episode {episode + 1}] joint_transitions={config.episode_seconds} "
            f"reward={episode_reward:.6f} delivered={episode_delivered_mbits:.6f} Mbit "
            f"energy={episode_energy:.6f} J {lambda_summary} "
            f"routing_reward={episode_routing_reward:.6f} "
            f"raw_final_bits={packet_engine.raw_final_hop_bits:.0f} "
            f"timely_bits={packet_engine.timely_goodput_bits:.0f} "
            f"wait={packet_engine.wait_actions} partial={packet_engine.partial_transmissions} "
            f"deadline_drops={packet_engine.deadline_drops} "
            f"critic_updates={movement_agent.num_critic_update_iteration} "
            f"actor_updates={movement_agent.num_actor_update_iteration}"
        )
        if dinkelbach_event is not None and dinkelbach_event[
            "dinkelbach_block_completed"
        ]:
            first_episode = (
                (dinkelbach_event["dinkelbach_block_index"] - 1)
                * config.dinkelbach_update_interval_episodes
                + 1
            )
            print(
                f"[Dinkelbach block "
                f"{dinkelbach_event['dinkelbach_block_index']}]\n"
                f"episodes={first_episode}-{episode + 1}\n"
                f"timely_mbits="
                f"{dinkelbach_event['dinkelbach_block_timely_mbits_so_far']:.9f}\n"
                f"mobility_energy_j="
                f"{dinkelbach_event['dinkelbach_block_energy_joules_so_far']:.9f}\n"
                f"lambda_old={dinkelbach_event['dinkelbach_lambda_used']:.12g}\n"
                f"lambda_new="
                f"{dinkelbach_event['dinkelbach_lambda_after_episode']:.12g}\n"
                f"status={dinkelbach_event['dinkelbach_update_status']}"
            )

        if (
            not evaluation
            and
            config.enable_model_checkpoints
            and _is_checkpoint_episode(
                episode + 1,
                config.total_episodes,
                config.model_checkpoint_every,
            )
        ):
            save_model_checkpoint(
                os.path.join(
                    config.checkpoint_root, "models", f"ep_{episode + 1:04d}"
                ),
                episode=episode,
                td3=movement_agent,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata={
                    **experiment_identity,
                    "lambda_ee": (
                        float(lambda_ee) if method_spec.uses_dinkelbach else None
                    ),
                    "dinkelbach_state": (
                        dinkelbach_state.training_state()
                        if method_spec.uses_dinkelbach
                        else {"active": False, "update_count": 0}
                    ),
                    "formal_config": asdict(config),
                },
            )
        if (
            not evaluation
            and
            config.enable_full_resume
            and _is_checkpoint_episode(
                episode + 1,
                config.total_episodes,
                config.full_resume_every,
            )
        ):
            save_full_resume_checkpoint(
                os.path.join(
                    config.checkpoint_root, "full", f"ep_{episode + 1:04d}"
                ),
                episode=episode,
                td3=movement_agent,
                ddqn=ddqn,
                joint_replay=joint_replay,
                routing_replay=routing_replay,
                training_state=_full_training_state(
                    episode=episode,
                    dinkelbach_state=dinkelbach_state,
                    reward_log=reward_log,
                    delivered_log=delivered_log,
                    energy_log=energy_log,
                    lambda_used_log=lambda_used_log,
                    lambda_after_episode_log=lambda_after_episode_log,
                    total_joint_transitions=total_joint_transitions,
                    routing_slots_executed=routing_slots_executed,
                    td3_noise_log=td3_noise_log,
                    routing_epsilon_log=routing_epsilon_log,
                    warmup_joint_transitions=config.warmup_joint_transitions,
                    training_history_rows=training_history_rows,
                    dinkelbach_active=method_spec.uses_dinkelbach,
                ),
                formal_config=asdict(config),
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata={
                    **experiment_identity,
                    "lambda_ee": (
                        float(lambda_ee) if method_spec.uses_dinkelbach else None
                    ),
                    "dinkelbach_state": (
                        dinkelbach_state.training_state()
                        if method_spec.uses_dinkelbach
                        else {"active": False, "update_count": 0}
                    ),
                    "formal_config": asdict(config),
                },
                keep_last=config.full_resume_keep_last,
            )

    if config.enable_csv:
        os.makedirs("results", exist_ok=True)
        csv_frame = _legacy_training_frame(
            initial_log_length=initial_log_length,
            reward_log=reward_log,
            delivered_log=delivered_log,
            energy_log=energy_log,
            lambda_used_log=lambda_used_log,
            lambda_after_episode_log=lambda_after_episode_log,
        )
        csv_path = "results/centralized_td3_training.csv"
        append = config.resume_dir is not None and os.path.isfile(csv_path)
        csv_frame.to_csv(
            csv_path,
            mode="a" if append else "w",
            header=not append,
            index=False,
        )
    if config.enable_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(reward_log, label="Centralized TD3 reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    evaluation_invariants = None
    evaluation_state_fingerprints = None
    if evaluation:
        evaluation_state_after = _evaluation_state_snapshot(
            movement_agent,
            ddqn,
            joint_replay,
            routing_replay,
            dinkelbach_state,
        )
        evaluation_invariants = _evaluation_invariants(
            evaluation_state_before,
            evaluation_state_after,
            routing_epsilon_log,
            td3_noise_log,
        )
        evaluation_state_fingerprints = {
            "before": _learning_state_fingerprint(evaluation_state_before),
            "after": _learning_state_fingerprint(evaluation_state_after),
        }
        if (
            evaluation_state_fingerprints["before"]
            != evaluation_state_fingerprints["after"]
        ):
            raise AssertionError("evaluation learning-state fingerprint changed")

    backlog_invariant_passed = None
    deadline_counter_consistent = None
    if config.mode == "smoke":
        backlog_invariant_passed = all(
            np.isclose(
                float(packet_engine.backlog_bits.get(uav_id, 0.0)),
                packet_engine.recompute_backlog_for_assertion(uav_id),
            )
            for uav_id in range(env.num_UAV)
        )
        deadline_counter_consistent = (
            packet_engine.total_violated == packet_engine.deadline_drops
            == packet_engine.fov_violated + packet_engine.com_violated
        )
        if not backlog_invariant_passed:
            raise AssertionError("incremental backlog invariant failed")
        if not deadline_counter_consistent:
            raise AssertionError("deadline counters are inconsistent")
        print(
            "[Smoke checks] "
            f"backlog_invariant={backlog_invariant_passed} "
            f"deadline_counters={deadline_counter_consistent}"
        )

    return {
        "episodes": len(reward_log),
        "episodes_run": len(reward_log) - initial_log_length,
        "joint_transitions": total_joint_transitions,
        "critic_updates": movement_agent.num_critic_update_iteration,
        "actor_updates": movement_agent.num_actor_update_iteration,
        "routing_state_dim": ROUTING_STATE_DIM,
        "movement_state_dim": MOVEMENT_STATE_DIM,
        "joint_action_dim": JOINT_ACTION_DIM,
        "centralized_td3_gamma": movement_agent.gamma,
        "movement_agent_kind": movement_agent.agent_kind,
        "routing_ddqn_gamma": ddqn.gamma,
        "joint_replay_size": joint_replay.size,
        "routing_replay_size": routing_replay.size,
        "lambda": float(lambda_ee) if method_spec.uses_dinkelbach else None,
        "dinkelbach_update_count": (
            int(dinkelbach_state.update_count) if method_spec.uses_dinkelbach else 0
        ),
        "dinkelbach_state": (
            dinkelbach_state.training_state()
            if method_spec.uses_dinkelbach
            else {"active": False, "update_count": 0}
        ),
        "calibration": calibration,
        "duplicate_target_assertions": duplicate_target_assertions,
        "environment_actor_calls": environment_actor_calls,
        "proposal_batches": proposal_batches,
        "energy_evaluations": energy_evaluations,
        "terminal_joint_transitions": terminal_joint_transitions,
        "routing_slots_executed": routing_slots_executed,
        "ddqn_action_selections": ddqn_action_selections,
        "ddqn_training_updates": ddqn.num_training,
        "td3_noise_log": td3_noise_log,
        "movement_noise_log": td3_noise_log,
        "routing_epsilon_log": routing_epsilon_log,
        "raw_final_hop_bits": packet_engine.raw_final_hop_bits,
        "timely_goodput_bits": packet_engine.timely_goodput_bits,
        "timely_delivered_packets": packet_engine.total_delivered,
        "deadline_violated_packets": packet_engine.total_violated,
        "routing_wait_actions": packet_engine.wait_actions,
        "partial_transmissions": packet_engine.partial_transmissions,
        "deadline_drops": packet_engine.deadline_drops,
        "link_slot_budget_violations": packet_engine.link_slot_budget_violations,
        "backlog_invariant_passed": backlog_invariant_passed,
        "deadline_counter_consistent": deadline_counter_consistent,
        "evaluation": bool(evaluation),
        "evaluation_invariants": evaluation_invariants,
        "evaluation_state_fingerprints": evaluation_state_fingerprints,
        "scenario_ids": executed_scenario_ids,
        "episode_metrics": episode_metrics,
        "run_metadata": {
            **experiment_identity,
            **checkpoint_provenance,
            "formal_config": asdict(config),
            "dinkelbach_config": (
                dinkelbach_config_metadata(config)
                if method_spec.uses_dinkelbach
                else None
            ),
            "dinkelbach_state": (
                dinkelbach_state.training_state()
                if method_spec.uses_dinkelbach
                else {"active": False, "update_count": 0}
            ),
            "evaluation": bool(evaluation),
            "training_history": (
                {
                    "row_count": len(training_history_rows),
                    "last_episode": (
                        training_history_rows[-1]["episode"]
                        if training_history_rows
                        else 0
                    ),
                    "identity": history_identity,
                    "canonical_format": "jsonl",
                    "jsonl_file": TRAINING_HISTORY_JSONL,
                    "csv_file": TRAINING_HISTORY_CSV,
                    "commit_file": TRAINING_HISTORY_COMMIT,
                }
                if history_identity is not None
                else None
            ),
        },
        "training_history_rows": training_history_rows,
        "reward_log": reward_log,
        "delivered_log": delivered_log,
        "energy_log": energy_log,
        "lambda_used_log": lambda_used_log if method_spec.uses_dinkelbach else [],
        "lambda_after_episode_log": (
            lambda_after_episode_log if method_spec.uses_dinkelbach else []
        ),
    }


def parse_training_config(argv=None):
    parser = argparse.ArgumentParser(
        description="Corrected no-LLM centralized TD3 training"
    )
    parser.add_argument("--mode", required=True, choices=("smoke", "train"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-dir")
    parser.add_argument(
        "--checkpoint-root", default="checkpoints_centralized_td3"
    )
    args = parser.parse_args(argv)
    if args.mode == "smoke":
        if args.episodes is not None:
            parser.error("smoke mode fixes --episodes to 1; do not pass --episodes")
        if args.seed is not None:
            parser.error(
                f"smoke mode fixes --seed to {DEFAULT_TRAINING_SEED}; "
                "do not pass --seed"
            )
        if args.resume_dir is not None:
            parser.error("smoke mode does not support full resume")
        return smoke_training_config()
    if args.episodes is None:
        parser.error("formal train mode requires --episodes")
    if args.seed is None:
        parser.error("formal train mode requires --seed")
    return formal_training_config(
        args.episodes,
        random_seed=args.seed,
        resume_dir=args.resume_dir,
        checkpoint_root=args.checkpoint_root,
    )


def main(argv=None):
    return train(parse_training_config(argv))


if __name__ == "__main__":
    main()
