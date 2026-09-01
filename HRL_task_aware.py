import argparse
from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
import os
import random
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Fov_model_phase import FovModel
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
from Packet_scheduler_v1 import (
    EPISODE_INJECTION_CUTOFF_SECONDS,
    PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION,
    PacketEngine,
    TASK_DEADLINE_SECONDS,
)
from Simulator import Simulator
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    apply_joint_movement_proposals,
    build_velocity_substep_proposals,
    calculate_movement_potentials,
    decode_joint_velocity_commands,
    executed_joint_action_from_displacement,
    get_global_movement_state,
    movement_mask_from_state,
    project_joint_action,
)
from com_capacity_calibration import load_com_capacity_reference
from exploration_schedules import movement_behavior_noise
from evaluation_metrics import safe_energy_efficiency
from experiment_config import (
    COM_SESSION_LIFECYCLE_VERSION,
    COM_PACKET_SIZE_BITS,
    COM_OFFERED_RATE_BPS,
    DEFAULT_TRAINING_SEED,
    FOV_EMA_LIFECYCLE_VERSION,
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
    MOVEMENT_EXPLORATION_DECAY_EPISODES,
    MOVEMENT_INTERVAL_SECONDS,
    MOVEMENT_REPLAY_CAPACITY,
    MethodSpec,
    NUM_UAV,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    ROUTING_EPSILON_DECAY_EPISODES,
    ROUTING_GRADIENT_STEPS_PER_UPDATE,
    ROUTING_REPLAY_CAPACITY,
    ROUTING_SLOT_SECONDS,
    ROUTING_UPDATE_INTERVAL_SLOTS,
    ROUTING_WARMUP_TRANSITIONS,
    SR_ROUTE_LIFECYCLE_VERSION,
    effective_training_config,
    exploration_schedule_configuration,
    movement_agent_configuration,
    comparison_method_configuration,
    routing_agent_configuration,
)
from movement_agents import create_movement_agent, sample_random_joint_action
from observation_strategy import (
    ROUTING_STATE_DIM,
    apply_observation_strategy,
    masked_observation_metadata,
)
from packet_outcome_artifacts import (
    MAX_BOUNDED_PACKET_OUTCOME_EPISODES,
    PACKET_OUTCOME_ARTIFACT_MODES,
    PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
    PACKET_OUTCOME_MODE_BOUNDED,
    PACKET_OUTCOME_MODE_DISABLED,
    PACKET_OUTCOME_MODE_STREAMING,
    PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
    PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS,
    packet_outcome_episode_record,
)
from routing_q_score_diagnostics import (
    ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION,
    ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS,
    RoutingQScoreDiagnosticAccumulator,
)
from routing_agents import create_routing_agent
from routing_lifecycle import RoutingLearnerLifecycle
from routing_transition_ledger import RoutingTransitionLedger
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    FULL_RESUME_LOGGING_SCHEMA_VERSION,
    checkpoint_artifact_provenance,
    checkpoint_episode_schedule,
    checkpoint_training_provenance,
    checkpoint_run_compatibility_from_metadata,
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
from rng_contract import CHANNEL_RNG_STREAMS, NamedRNGStreams, RNG_CONTRACT_VERSION
from scenario_manifest import validate_manifest_initial_topologies


MOVEMENT_CONTROL_INTERVAL = int(round(MOVEMENT_INTERVAL_SECONDS / ROUTING_SLOT_SECONDS))
if not np.isclose(
    MOVEMENT_CONTROL_INTERVAL * ROUTING_SLOT_SECONDS,
    MOVEMENT_INTERVAL_SECONDS,
    rtol=0.0,
    atol=1e-12,
):
    raise RuntimeError("movement interval must contain an integer routing-slot count")
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
    """Return the local RNG registry; formal execution never seeds globals."""

    streams = NamedRNGStreams(0 if seed is None else int(seed))
    # Materialization consumes no draws and guarantees full-resume captures
    # both training and evaluation channel generator states explicitly.
    for stream_name in CHANNEL_RNG_STREAMS:
        streams.numpy(stream_name)
    return streams


def _git_commit_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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


def _create_active_replay_buffers(
    state_dim, routing_dim, max_size=MOVEMENT_REPLAY_CAPACITY
):
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
    routing_slot_seconds: float = ROUTING_SLOT_SECONDS
    warmup_joint_transitions: int = PRODUCTION_WARMUP_TRANSITIONS
    batch_size: int = PRODUCTION_BATCH_SIZE
    policy_delay: int = PRODUCTION_POLICY_DELAY
    replay_max_size: int = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
        "replay_size"
    ]
    routing_warmup_transitions: int = ROUTING_WARMUP_TRANSITIONS
    routing_update_interval_slots: int = ROUTING_UPDATE_INTERVAL_SLOTS
    routing_gradient_steps_per_update: int = ROUTING_GRADIENT_STEPS_PER_UPDATE
    movement_exploration_decay_episodes: int = (
        MOVEMENT_EXPLORATION_DECAY_EPISODES
    )
    routing_epsilon_decay_episodes: int = ROUTING_EPSILON_DECAY_EPISODES
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
    collect_packet_outcomes: bool = False
    packet_outcome_artifact_mode: str = PACKET_OUTCOME_MODE_DISABLED
    packet_outcome_collection_limit: int = 0

    def __post_init__(self):
        if self.mode not in {"smoke", "train", "custom"}:
            raise ValueError(f"unsupported training mode: {self.mode}")
        if self.packet_outcome_artifact_mode not in PACKET_OUTCOME_ARTIFACT_MODES:
            raise ValueError(
                "unsupported packet outcome artifact mode: "
                f"{self.packet_outcome_artifact_mode}"
            )
        if type(self.collect_packet_outcomes) is not bool:
            raise TypeError("collect_packet_outcomes must be boolean")
        bounded = self.packet_outcome_artifact_mode == PACKET_OUTCOME_MODE_BOUNDED
        if self.collect_packet_outcomes != bounded:
            raise ValueError(
                "collect_packet_outcomes is true only for bounded_memory mode"
            )
        if self.mode == "train" and self.packet_outcome_artifact_mode != (
            PACKET_OUTCOME_MODE_DISABLED
        ):
            raise ValueError(
                "formal training requires packet outcome artifact mode disabled"
            )
        if self.resume_dir is not None and self.packet_outcome_artifact_mode != (
            PACKET_OUTCOME_MODE_DISABLED
        ):
            raise ValueError(
                "full-resume training requires packet outcome artifact mode disabled"
            )
        if bounded:
            if not (
                1
                <= int(self.packet_outcome_collection_limit)
                <= MAX_BOUNDED_PACKET_OUTCOME_EPISODES
            ):
                raise ValueError(
                    "bounded packet outcome collection requires a positive limit "
                    f"no greater than {MAX_BOUNDED_PACKET_OUTCOME_EPISODES}"
                )
            if int(self.total_episodes) > int(
                self.packet_outcome_collection_limit
            ):
                raise ValueError(
                    "bounded packet outcome collection limit is below total episodes"
                )
        elif int(self.packet_outcome_collection_limit) != 0:
            raise ValueError(
                "packet outcome collection limit is valid only in bounded_memory mode"
            )
        if not np.isclose(
            float(self.routing_slot_seconds),
            ROUTING_SLOT_SECONDS,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("routing slot duration is fixed at 0.25 seconds")
        slots = MOVEMENT_INTERVAL_SECONDS / float(self.routing_slot_seconds)
        if not np.isclose(slots, MOVEMENT_CONTROL_INTERVAL):
            raise ValueError("movement interval must contain exactly four routing slots")
        if self.episode_seconds <= 0 or self.total_episodes <= 0:
            raise ValueError("episode_seconds and total_episodes must be positive")
        if self.warmup_joint_transitions < 0 or self.batch_size <= 0:
            raise ValueError("warmup must be non-negative and batch_size positive")
        if int(self.replay_max_size) <= 0:
            raise ValueError("replay_max_size must be positive")
        if (
            self.mode == "train"
            and int(self.total_episodes) >= int(
                FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"]
            )
            and (
                int(self.warmup_joint_transitions)
                != PRODUCTION_WARMUP_TRANSITIONS
                or int(self.replay_max_size)
                != FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
                    "replay_size"
                ]
            )
        ):
            raise ValueError(
                "formal movement warmup/replay capacity are fixed by contract"
            )
        if (
            self.routing_warmup_transitions <= 0
            or self.routing_update_interval_slots <= 0
            or self.routing_gradient_steps_per_update <= 0
            or self.movement_exploration_decay_episodes <= 0
            or self.routing_epsilon_decay_episodes <= 0
        ):
            raise ValueError("routing lifecycle and exploration horizons must be positive")
        if (
            self.routing_update_interval_slots != ROUTING_UPDATE_INTERVAL_SLOTS
            or self.routing_gradient_steps_per_update
            != ROUTING_GRADIENT_STEPS_PER_UPDATE
            or self.movement_exploration_decay_episodes
            != MOVEMENT_EXPLORATION_DECAY_EPISODES
            or self.routing_epsilon_decay_episodes
            != ROUTING_EPSILON_DECAY_EPISODES
        ):
            raise ValueError(
                "production cadence and exploration horizons are fixed; use "
                "synthetic lifecycle counters for shortened tests"
            )
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
        routing_slot_seconds=ROUTING_SLOT_SECONDS,
        warmup_joint_transitions=0,
        routing_warmup_transitions=1,
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


def _normalize_evaluation_overrides(overrides):
    values = dict(overrides or {})
    allowed = {
        "fov_rate_packets_per_second",
        "com_rate_packets_per_second",
        "fov_deadline_seconds",
        "com_deadline_seconds",
        "packet_injection_cutoff_seconds",
    }
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown evaluation override fields: {sorted(unknown)}")
    rates = {
        "FOV": values.get("fov_rate_packets_per_second"),
        "COM": values.get("com_rate_packets_per_second"),
    }
    for task_type, value in tuple(rates.items()):
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{task_type} packet rate must be finite and non-negative")
        rates[task_type] = value
    deadlines = {
        "FOV": float(
            values.get("fov_deadline_seconds", TASK_DEADLINE_SECONDS["FOV"])
        ),
        "COM": float(
            values.get("com_deadline_seconds", TASK_DEADLINE_SECONDS["COM"])
        ),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in deadlines.values()):
        raise ValueError("evaluation deadlines must be finite positive seconds")
    injection_cutoff = float(
        values.get(
            "packet_injection_cutoff_seconds",
            EPISODE_INJECTION_CUTOFF_SECONDS,
        )
    )
    if not np.isfinite(injection_cutoff) or injection_cutoff < 0.0:
        raise ValueError("packet injection cutoff must be finite and non-negative")
    return {
        "traffic_rates_packets_per_second": rates,
        "task_deadlines_seconds": deadlines,
        "packet_injection_cutoff_seconds": injection_cutoff,
        "units": {
            "traffic_rate": "packets/s",
            "deadline": "seconds",
            "packet_injection_cutoff": "seconds",
        },
    }


def _uav_task_phase(env, uav_id):
    tasks = [
        str(task.get("task_type", "Hovering"))
        for task in env.multi_tasks.get(int(uav_id), [])
    ]
    active = [task for task in tasks if task != "Hovering"]
    if "FOV" in active and "COM" in active:
        return "FOV+COM"
    if active:
        return active[0]
    return "Hover"


def _sensing_footprint(env, uav_id):
    """Return the exact rectangular footprint used by search coverage."""

    uav = env.uav_dict[int(uav_id)]
    model = FovModel(
        f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80
    )
    width, height = model.get_ground_fov_size(uav.z_u)
    x_min = max(0.0, float(uav.x_u) - float(width) / 2.0)
    x_max = min(float(env.env_width), float(uav.x_u) + float(width) / 2.0)
    y_min = max(0.0, float(uav.y_u) - float(height) / 2.0)
    y_max = min(float(env.env_height), float(uav.y_u) + float(height) / 2.0)
    return {
        "uav_id": int(uav_id),
        "geometry": "axis_aligned_ground_rectangle",
        "center_x": float(uav.x_u),
        "center_y": float(uav.y_u),
        "ground_z": 0.0,
        "width_m": float(width),
        "height_m": float(height),
        "clipped_bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "model": {"f_m": 0.004, "image_width_m": 0.008, "image_length_m": 0.012},
    }


def _trajectory_state(
    env, *, actual_time_seconds, target_uav_id=None, active_links=None
):
    target_uav_id = (
        None if target_uav_id is None else int(target_uav_id)
    )
    return {
        "actual_time_seconds": float(actual_time_seconds),
        "target_uav_id": target_uav_id,
        "target_uav_phase": (
            _uav_task_phase(env, target_uav_id)
            if target_uav_id is not None
            else None
        ),
        "uavs": [
            {
                "uav_id": int(uav_id),
                "x": float(uav.x_u),
                "y": float(uav.y_u),
                "z": float(uav.z_u),
                "task_phase": _uav_task_phase(env, uav_id),
                "assigned_tasks": copy.deepcopy(env.multi_tasks.get(uav_id, [])),
            }
            for uav_id, uav in sorted(env.uav_dict.items())
        ],
        "sr_teams": [
            {
                "sr_id": int(sr.id),
                "x": float(sr.x),
                "y": float(sr.y),
                "z": float(sr.z),
                "assigned_gt_id": (
                    int(sr.assigned_gt_id)
                    if sr.assigned_gt_id is not None
                    else None
                ),
                "arrived": bool(sr.arrived),
            }
            for sr in sorted(env.SR_teams, key=lambda item: item.id)
        ],
        "ground_targets": [
            {
                "gt_id": int(gt.id),
                "x": float(gt.x),
                "y": float(gt.y),
                "z": float(gt.z),
                "radius_m": float(gt.radius),
                "detected": bool(gt.is_found),
                "detected_by_uav_id": (
                    int(gt.found_by) if gt.found_by is not None else None
                ),
            }
            for gt in sorted(env.gts, key=lambda item: item.id)
        ],
        "ground_station": {
            "gs_id": int(env.GS_ID),
            "x": float(env.GS_pos[0]),
            "y": float(env.GS_pos[1]),
            "z": float(env.GS_pos[2]),
        },
        "active_links": copy.deepcopy(list(active_links or [])),
        "assignment_metadata": copy.deepcopy(env.assignment_metadata()),
        "sensing_coverage": (
            [_sensing_footprint(env, target_uav_id)]
            if target_uav_id is not None
            else []
        ),
    }


def _routing_transition_done(episode_done, next_hol):
    """Cut DDQN bootstrap when this UAV has no next routing decision."""

    return bool(episode_done or next_hol is None)


def _attribute_routing_transition_cost(
    routing_buffer, pending_transitions, sender, cost
):
    """Attribute cleanup cost before a pending transition enters replay."""

    sender = int(sender)
    value = float(cost)
    if pending_transitions is not None and sender in pending_transitions:
        pending_transitions[sender]["cost"] += value
        return True
    return routing_buffer.attribute_latest_cost(sender, value)


def _finalize_pending_routing_transitions(
    routing_buffer,
    pending_transitions,
    states,
    start_of_slot_hol_by_sender,
):
    """Finalize prior-slot transitions at the next real decision snapshot.

    Injection, expiration, HOL eligibility, masks, and observations have already
    been resolved when this function is called.  A sender without a decision is
    truncated according to the existing routing-transition terminal contract.
    """

    if pending_transitions is None:
        return 0
    finalized = 0
    for uid in sorted(tuple(pending_transitions)):
        transition = pending_transitions.pop(uid)
        next_hol = start_of_slot_hol_by_sender.get(uid)
        transition_done = _routing_transition_done(False, next_hol)
        next_state = states.get(uid, transition["state"])
        routing_buffer.add(
            transition["state"],
            transition["action"],
            next_state,
            transition["reward"],
            transition["cost"],
            transition_done,
            tag_gt=transition["tag_gt"],
            agent_id=uid,
        )
        finalized += 1
    return finalized


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
    task_observation_mode="full",
    traffic_rate_overrides=None,
    pending_routing_transitions=None,
    routing_transition_ledger=None,
    routing_q_score_accumulator=None,
    routing_q_score_context=None,
):
    del routing_masks
    if write_replay and int(getattr(routing_buffer, "n_step", -1)) != 1:
        raise ValueError("formal routing replay requires n_step=1")
    step_time = float(packet_engine.step_time)
    env.current_time = float(current_time)
    absolute_slot = int(round(env.current_time / step_time))
    env.prepare_channel_routing_slot(absolute_slot)
    # Source generation precedes cleanup by contract. Prior-slot expirations are
    # normally already gone; this makes the boundary explicit and idempotent.
    packet_engine.inject_packets(
        env,
        delay_bound_steps,
        env.current_time,
        step_time=step_time,
        rate_overrides=traffic_rate_overrides,
    )
    pre_slot_violations = packet_engine.expire_packets(
        env.current_time, inclusive=True
    )
    pre_slot_violations.extend(
        packet_engine.drop_expired_packets(env.current_time)
    )
    for violation in pre_slot_violations:
        task_type = violation["task_type"]
        if task_type in violation_stats:
            violation_stats[task_type]["deadline_violated_packets"] += 1
        sender = int(violation["attributed_sender"])
        if write_replay:
            if routing_transition_ledger is not None:
                transition_id = violation.get("routing_transition_id")
                if transition_id is None:
                    # The packet engine already recorded this formal system
                    # violation as pre-routing/unattributed. It must not enter
                    # safe-DDQN replay.
                    continue
                if routing_transition_ledger.add_cost(transition_id, 1.0):
                    packet_engine.replay_attributed_violation_cost_count += 1.0
                else:
                    raise AssertionError(
                        "stable routing transition ID rejected delayed cost"
                    )
            elif sender >= 0 and _attribute_routing_transition_cost(
                routing_buffer, pending_routing_transitions, sender, 1.0
            ):
                packet_engine.replay_attributed_violation_cost_count += 1.0
    backlog_before = _active_backlog(packet_engine)
    start_of_slot_hol_by_sender = {
        int(uid): packet_engine.get_hol_packet(uid)
        for uid in packet_engine.nonempty_uav_ids()
    }
    start_of_slot_hol_by_sender = {
        uid: pkt
        for uid, pkt in start_of_slot_hol_by_sender.items()
        if pkt is not None
    }
    routing_decision_uav_ids = sorted(start_of_slot_hol_by_sender)
    start_of_slot_eligible_packet_ids = {
        uid: {
            int(pkt["id"])
            for pkt in packet_engine.get_queue_packets(uid)
        }
        for uid in routing_decision_uav_ids
    }
    physical_masks = {
        uid: env.get_routing_action_mask(uid).astype(bool)
        for uid in routing_decision_uav_ids
    }
    effective_masks = {
        uid: packet_engine.get_effective_action_mask(
            env, uid, physical_masks[uid]
        )
        for uid in routing_decision_uav_ids
    }

    physical_states = {
        uid: packet_engine.get_state_ta(
            env,
            uid,
            backlog_bits=backlog_before,
            action_mask=effective_masks[uid],
        )
        for uid in routing_decision_uav_ids
    }
    states = {
        uid: np.asarray(
            apply_observation_strategy(state, task_observation_mode, "routing"),
            dtype=np.float32,
        )
        for uid, state in physical_states.items()
    }
    for uid, state in states.items():
        if state.shape != (ROUTING_STATE_DIM,):
            raise AssertionError(
                f"routing state for UAV {uid} has shape {state.shape}, "
                f"expected ({ROUTING_STATE_DIM},)"
            )

    if write_replay:
        if routing_transition_ledger is not None:
            routing_transition_ledger.finalize_causality(
                states, start_of_slot_hol_by_sender
            )
            routing_transition_ledger.commit_ready(
                routing_buffer,
                packet_engine.routing_transition_reference_counts(),
            )
        else:
            _finalize_pending_routing_transitions(
                routing_buffer,
                pending_routing_transitions,
                states,
                start_of_slot_hol_by_sender,
            )

    next_hops = _select_routing_actions(
        ddqn, states, effective_masks, epsilon=epsilon
    )
    if routing_q_score_accumulator is not None:
        if not np.isclose(float(epsilon), 0.0, rtol=0.0, atol=0.0):
            raise AssertionError(
                "routing Q-score diagnostics require evaluation epsilon=0"
            )
        if getattr(ddqn, "routing_agent_kind", None) != "safe_ddqn":
            raise TypeError("routing Q-score diagnostics require safe-DDQN")
        if routing_q_score_context is None:
            raise ValueError("routing Q-score diagnostics require slot context")
        for uid in routing_decision_uav_ids:
            inspection = ddqn.inspect_action_scores(
                states[uid], effective_masks[uid]
            )
            routing_q_score_accumulator.add_decision(
                inspection,
                selected_action=next_hops[uid],
                sender_uav_id=uid,
                hol_task_type=start_of_slot_hol_by_sender[uid]["task_type"],
                scenario_id=routing_q_score_context["scenario_id"],
                episode_index=routing_q_score_context["episode_index"],
                slot_index=routing_q_score_context["slot_index"],
                time_seconds=current_time,
            )

    proposed_links = {
        sender: receiver
        for sender, receiver in next_hops.items()
        if receiver != sender and sender in start_of_slot_hol_by_sender
    }
    requested_s2u_links = packet_engine.active_s2u_links(env)
    active_capacities, _ = env.allocate_active_link_capacities(
        proposed_links, s2u_links=requested_s2u_links
    )
    resolved_s2u_links = None
    if packet_engine.enable_packet_diagnostic_artifacts:
        resolved_s2u_links = {
            int(sr_id): int(receiver)
            for sr_id, receiver in env.active_s2u_capacities
        }
    routing_transition_ids = {}
    if write_replay and routing_transition_ledger is not None:
        for uid in routing_decision_uav_ids:
            routing_transition_ids[uid] = routing_transition_ledger.create(
                agent_id=uid,
                state=states[uid],
                action=int(next_hops.get(uid, uid)),
                tag_gt=int(env.num_GT),
            )
    slot_result = packet_engine.serve_active_links(
        env,
        next_hops,
        active_capacities,
        current_time=env.current_time,
        start_of_slot_hol_by_sender=start_of_slot_hol_by_sender,
        start_of_slot_eligible_packet_ids=start_of_slot_eligible_packet_ids,
        start_of_slot_backlog_bits_by_sender={
            uid: float(backlog_before.get(uid, 0.0))
            for uid in routing_decision_uav_ids
        },
        routing_transition_ids_by_sender=routing_transition_ids,
        start_of_slot_physical_masks_by_sender=physical_masks,
        start_of_slot_effective_masks_by_sender=effective_masks,
        block_capacity_profiles=env.active_link_capacity_profiles_mbps,
        s2u_block_capacity_profiles=env.active_s2u_capacity_profiles_mbps,
        resolved_s2u_links=resolved_s2u_links,
    )
    attributable_violation_count = sum(
        bool(outcome["violated"])
        for outcome in slot_result["outcomes"]
    )
    deferred_cost = float(
        sum(slot_result["deferred_cost_by_sender"].values())
    )
    attributed_cost = float(
        sum(slot_result["cost_by_sender"].values()) + deferred_cost
    )
    if not np.isclose(attributed_cost, float(attributable_violation_count)):
        raise AssertionError(
            "deadline violation cost attribution mismatch: "
            f"attributable_violations={attributable_violation_count}, "
            f"cost={attributed_cost}"
        )
    if write_replay:
        if routing_transition_ledger is not None:
            for outcome in slot_result["outcomes"]:
                if not outcome["violated"]:
                    continue
                transition_id = outcome.get("routing_transition_id")
                if transition_id is None:
                    continue
                if routing_transition_ledger.add_cost(transition_id, 1.0):
                    packet_engine.replay_attributed_violation_cost_count += 1.0
                else:
                    raise AssertionError(
                        "stable routing transition ID rejected slot cost"
                    )
        else:
            for sender, cost in sorted(
                slot_result["deferred_cost_by_sender"].items()
            ):
                if sender >= 0 and routing_buffer.attribute_latest_cost(
                    sender, cost
                ):
                    packet_engine.replay_attributed_violation_cost_count += float(cost)
            packet_engine.replay_attributed_violation_cost_count += float(
                sum(slot_result["cost_by_sender"].values())
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

    if write_replay:
        if routing_transition_ledger is not None:
            for uid, transition_id in routing_transition_ids.items():
                routing_transition_ledger.set_reward(
                    transition_id, slot_result["reward_by_sender"][uid]
                )
            if done:
                routing_transition_ledger.finalize_causality(
                    {}, {}, terminal=True
                )
            routing_transition_ledger.commit_ready(
                routing_buffer,
                packet_engine.routing_transition_reference_counts(),
            )
            immediate_next_states = None
        else:
            immediate_next_states = {}
        if routing_transition_ledger is None and pending_routing_transitions is None:
            backlog_after = _active_backlog(packet_engine)
            physical_next_states = {
                uid: packet_engine.get_state_ta(
                    env,
                    uid,
                    backlog_bits=backlog_after,
                    action_mask=packet_engine.get_effective_action_mask(
                        env, uid, env.get_routing_action_mask(uid).astype(bool)
                    ),
                )
                for uid in routing_decision_uav_ids
            }
            immediate_next_states = {
                uid: apply_observation_strategy(
                    state, task_observation_mode, "routing"
                )
                for uid, state in physical_next_states.items()
            }
        for uid in routing_decision_uav_ids if routing_transition_ledger is None else ():
            next_hol = packet_engine.get_hol_packet(uid)
            transition_done = _routing_transition_done(done, next_hol)
            transition = {
                "state": states[uid].copy(),
                "action": int(next_hops.get(uid, uid)),
                "reward": float(slot_result["reward_by_sender"][uid]),
                "cost": float(slot_result["cost_by_sender"][uid]),
                "tag_gt": int(env.num_GT),
            }
            if pending_routing_transitions is not None and not transition_done:
                if uid in pending_routing_transitions:
                    raise AssertionError("routing transition was staged twice")
                pending_routing_transitions[uid] = transition
            else:
                routing_buffer.add(
                    transition["state"],
                    transition["action"],
                    immediate_next_states.get(uid, transition["state"]),
                    transition["reward"],
                    transition["cost"],
                    transition_done,
                    tag_gt=transition["tag_gt"],
                    agent_id=uid,
                )
    selected_links = [
        {
            **dict(item),
            "capacity_bits_per_second": float(item["capacity_mbps"]) * 1e6,
        }
        for item in env.active_link_diagnostics
    ]
    return (
        float(slot_result["timely_goodput_bits"]),
        float(sum(slot_result["reward_by_sender"].values())),
        len(next_hops),
        selected_links,
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
        return ()
    search_uav_ids = [
        uav_id
        for uav_id in range(env.num_UAV)
        if any(
            task.get("task_type") == "Search"
            for task in env.multi_tasks.get(uav_id, [])
        )
    ]
    if not search_uav_ids:
        return ()
    visited_precommit = env.visited_bitmap.copy()
    # Discovery/geometry is evaluated first and does not mutate coverage.
    for uav_id in search_uav_ids:
        env.update_visited_grid(uav_id)
    search_uav_ids = frozenset(search_uav_ids)
    # Observation participation is intentionally broader than coverage
    # contribution: every UAV freezes its raw FOV sample from the same V_pre.
    transitions = tuple(
        env.mark_search_coverage(
            uav_id,
            visited_snapshot=visited_precommit,
            commit=False,
            coverage_contributor=uav_id in search_uav_ids,
        )
        for uav_id in range(env.num_UAV)
    )
    # Atomically commit only the Search-UAV union after every participant's raw
    # observation has been frozen.
    committed = env.visited_bitmap.copy()
    for transition in transitions:
        if not transition.coverage_contributor:
            continue
        if transition.current_footprint is None:
            continue
        bx_min, bx_max, by_min, by_max = transition.current_footprint
        committed[bx_min : bx_max + 1, by_min : by_max + 1] = True
    env.visited_bitmap[:, :] = committed
    return transitions


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


def terminal_ratio_objective(
    reward_mode,
    done,
    cumulative_delivered_mbits,
    cumulative_energy_j,
):
    """Return the direct-ratio objective only on an episode's terminal step."""

    if reward_mode not in {"dinkelbach", "ratio"}:
        raise ValueError(f"unsupported reward mode: {reward_mode}")
    if reward_mode != "ratio" or not done:
        return 0.0
    # safe_energy_efficiency is an evaluation Mbit/J metric.  The learning
    # objective is explicitly bit/J and therefore converts its numerator.
    return safe_energy_efficiency(
        cumulative_delivered_mbits,
        cumulative_energy_j,
    ) * 1e6


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
    ratio_objective_reward=0.0,
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
        objective = float(ratio_objective_reward)
        if not np.isfinite(objective):
            objective = 0.0
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
        "dinkelbach_lambda_used": None,
        "dinkelbach_lambda_after_episode": None,
        "dinkelbach_lambda_updated": False,
        "dinkelbach_update_status": "disabled_for_reward_mode",
        "dinkelbach_block_index": None,
        "dinkelbach_block_episode": None,
        "dinkelbach_block_timely_mbits_so_far": None,
        "dinkelbach_block_energy_joules_so_far": None,
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
    lambda_cost_used_log=None,
    lambda_cost_after_episode_log=None,
    fov_ema_state=None,
    sr_route_state=None,
    routing_lifecycle_state=None,
    exploration_state,
    named_rng_state=None,
    channel_lifecycle_state=None,
    routing_transition_state=None,
    packet_engine_state=None,
):
    completed_episode_count = int(episode) + 1
    if lambda_cost_used_log is None:
        lambda_cost_used_log = [0.0] * completed_episode_count
    if lambda_cost_after_episode_log is None:
        lambda_cost_after_episode_log = [0.0] * completed_episode_count
    if fov_ema_state is None:
        fov_ema_state = {
            "lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
            "values": {},
            "initialized_uav_ids": [],
            "previous_footprints": {},
            "transition_marker": None,
            "footprint_transition_marker": None,
            "update_count": 0,
        }
    if sr_route_state is None:
        sr_route_state = {
            "lifecycle_version": SR_ROUTE_LIFECYCLE_VERSION,
            "teams": [],
            "trajectory": {},
            "checkpoint_scope": "episode_boundary_terminal_snapshot",
            "mid_episode_checkpoint_supported": False,
        }
    if routing_transition_state is None:
        routing_transition_state = RoutingTransitionLedger().state_dict()
    if packet_engine_state is None:
        packet_engine_state = {
            "schema_version": PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_scope": "episode_boundary_terminal_snapshot",
            "mid_episode_checkpoint_supported": False,
            "next_packet_id": 0,
            "active_packets": [],
            "inject_buffer": {},
            "source_buffer": {},
            "uav_queue_packet_ids": {
                str(uid): [] for uid in range(NUM_UAV)
            },
            "sr_queue_packet_ids": {},
            "routing_transition_reference_counts": {},
            "com_session_state": {
                "lifecycle_version": COM_SESSION_LIFECYCLE_VERSION,
                "sessions": {},
            },
            "generated_packet_counts": {"FOV": 0, "COM": 0},
            "eligible_packet_counts": {"FOV": 0, "COM": 0},
            "raw_final_hop_bits": 0.0,
            "timely_goodput_bits": 0.0,
            "fov_generated_raw_bits": 0.0,
            "fov_timely_delivered_raw_bits": 0.0,
            "fov_timely_useful_bits": 0.0,
            "fov_capture_coverage_sum": 0.0,
            "fov_capture_coverage_count": 0,
            "fov_zero_coverage_packet_count": 0,
            "com_timely_delivered_bits": 0.0,
            "total_timely_useful_bits": 0.0,
            "pending_terminal_violation_events": [],
            "system_qos_eligible_packet_count": 0,
            "system_qos_violation_count": 0,
            "routing_credit_eligible_packet_count": 0,
            "routing_credit_violation_count": 0,
            "replay_attributed_violation_cost_count": 0.0,
            "unattributed_transition_violation_count": 0,
            "unattributed_pre_routing_violation_count": 0,
        }
    movement_post_warmup = max(
        int(total_joint_transitions) - int(warmup_joint_transitions), 0
    )
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
        "td3_post_warmup_transition": movement_post_warmup,
        "movement_post_warmup_transition_count": movement_post_warmup,
        "ddqn_schedule_slot": (
            int(routing_lifecycle_state["routing_global_slot_count"])
            if routing_lifecycle_state is not None
            else 0
        ),
        "routing_lifecycle_state": copy.deepcopy(routing_lifecycle_state),
        **(
            copy.deepcopy(routing_lifecycle_state)
            if routing_lifecycle_state is not None
            else {}
        ),
        **copy.deepcopy(exploration_state),
        "routing_epsilon_decay_start_slot": (
            routing_lifecycle_state["routing_epsilon_decay_start_slot"]
            if routing_lifecycle_state is not None
            else None
        ),
        "td3_noise_log": list(td3_noise_log),
        "movement_noise_log": list(td3_noise_log),
        "routing_epsilon_log": list(routing_epsilon_log),
        "lambda_cost_used_log": list(lambda_cost_used_log),
        "lambda_cost_after_episode_log": list(
            lambda_cost_after_episode_log
        ),
        "fov_ema_state": copy.deepcopy(fov_ema_state),
        "sr_route_state": copy.deepcopy(sr_route_state),
        "training_history_rows": list(training_history_rows),
        "named_rng_state": copy.deepcopy(named_rng_state),
        "channel_lifecycle_state": copy.deepcopy(channel_lifecycle_state),
        "routing_transition_state": copy.deepcopy(routing_transition_state),
        "packet_engine_state": copy.deepcopy(packet_engine_state),
    }


def _experiment_identity(
    method_spec,
    scenario_manifest,
    training_seed,
    config,
    *,
    evaluation=False,
    rng_contract_metadata=None,
):
    comparison = comparison_method_configuration(method_spec)
    resolved_exploration = exploration_schedule_configuration(config, method_spec)
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
        "training_episode_count": (
            None if evaluation else int(config.total_episodes)
        ),
        "evaluation_episode_count": (
            int(config.total_episodes) if evaluation else None
        ),
        "episode_horizon_seconds": int(config.episode_seconds),
        "movement_interval_seconds": MOVEMENT_INTERVAL_SECONDS,
        "routing_slot_seconds": float(config.routing_slot_seconds),
        "num_uav": NUM_UAV,
        "collect_packet_outcomes": bool(config.collect_packet_outcomes),
        "packet_outcome_artifact_mode": config.packet_outcome_artifact_mode,
        "packet_outcome_collection_limit": int(
            config.packet_outcome_collection_limit
        ),
        "packet_outcome_artifact_schema_version": (
            PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION
            if config.packet_outcome_artifact_mode
            != PACKET_OUTCOME_MODE_DISABLED
            else None
        ),
        "packet_routing_diagnostic_contract_version": (
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION
            if config.packet_outcome_artifact_mode
            != PACKET_OUTCOME_MODE_DISABLED
            else None
        ),
        **{
            field: (
                definition
                if config.packet_outcome_artifact_mode
                != PACKET_OUTCOME_MODE_DISABLED
                else None
            )
            for field, definition in PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS.items()
        },
        "routing_slots_per_episode": resolved_exploration[
            "routing_slots_per_episode"
        ],
        "git_sha": _git_commit_sha(),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "rng_contract": copy.deepcopy(rng_contract_metadata),
        "movement_agent": method_spec.agent,
        "movement_agent_configuration": movement_agent_configuration(
            method_spec, config
        ),
        "routing_agent_configuration": routing_agent_configuration(
            method_spec, config
        ),
        "exploration_schedule_configuration": resolved_exploration,
        "reward_mode": method_spec.reward_mode,
        "goodput_metric_metadata": {
            "timely_goodput_bits": {
                "unit": "bit",
                "definition": "alias of total_timely_useful_bits",
            },
            "total_timely_useful_bits": {
                "unit": "bit",
                "definition": (
                    "timely FOV physical size times capture-time coverage "
                    "plus timely COM physical size"
                ),
            },
            "fov_generated_raw_bits": {"unit": "bit"},
            "fov_timely_delivered_raw_bits": {"unit": "bit"},
            "fov_timely_useful_bits": {"unit": "bit"},
            "fov_mean_capture_coverage": {
                "unit": "ratio",
                "missing_when": "no generated FOV packets",
            },
            "fov_zero_coverage_packet_count": {"unit": "packet"},
            "com_timely_delivered_bits": {"unit": "bit"},
            "coverage_snapshot_timing": "FOV packet generation/capture time",
            "physical_packet_service_weighted_by_coverage": False,
        },
        "movement_objective_definition": (
            {
                "objective_unit": "bit_per_j",
                "numerator": "sum of episode timely useful bits",
                "denominator": "sum of episode mobility energy joules",
                "semantics": "terminal-only ratio of sums",
            }
            if method_spec.reward_mode == "ratio"
            else {
                "objective_unit": "Mbit_minus_lambda_Mbit_per_J_times_J",
                "numerator": "timely useful Mbit",
                "semantics": "Dinkelbach residual",
            }
        ),
        "task_potential_enabled": bool(method_spec.task_potential_enabled),
        **comparison,
        "masked_state_fields": (
            masked_observation_metadata()
            if method_spec.task_observation == "masked"
            else None
        ),
        **(
            dinkelbach_config_metadata(config)
            if method_spec.uses_dinkelbach
            else {"dinkelbach_active": False}
        ),
    }


def _evaluation_runtime_provenance(
    method_spec,
    scenario_manifest,
    config,
    resolved_evaluation,
    routing_lifecycle,
    ddqn,
    evaluation_git_sha,
):
    return {
        "evaluation_episode_count": int(config.total_episodes),
        "evaluation_git_sha": str(evaluation_git_sha),
        "resolved_evaluation_config": {
            "method_id": method_spec.method_id,
            "method_spec": method_spec.to_dict(),
            "evaluation_episode_count": int(config.total_episodes),
            "evaluation_seed": (
                int(config.random_seed)
                if config.random_seed is not None
                else None
            ),
            "evaluation_manifest_hash": (
                scenario_manifest.content_hash
                if scenario_manifest is not None
                else None
            ),
            "evaluation_manifest_split": (
                scenario_manifest.split
                if scenario_manifest is not None
                else None
            ),
            "episode_horizon_seconds": float(config.episode_seconds),
            "movement_interval_seconds": MOVEMENT_INTERVAL_SECONDS,
            "routing_slot_seconds": float(config.routing_slot_seconds),
            "evaluation_overrides": copy.deepcopy(resolved_evaluation),
            "learning_state_frozen": True,
        },
        "routing_lifecycle": (
            routing_lifecycle.state_dict()
            if routing_lifecycle is not None
            else None
        ),
        "lambda_cost_source": (
            "checkpoint_frozen"
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "safe_ddqn_constraint_state": (
            ddqn.constraint_state()
            if method_spec.routing == "safe_ddqn"
            else None
        ),
    }


def _evaluation_provenance_aliases(checkpoint_training, evaluation_runtime):
    if not isinstance(evaluation_runtime, dict):
        raise ValueError("evaluation runtime provenance is required")
    lifecycle = (
        checkpoint_training.get("routing_lifecycle")
        if checkpoint_training is not None
        else None
    )
    return {
        "training_episode_count": (
            int(checkpoint_training["training_episode_count"])
            if checkpoint_training is not None
            else None
        ),
        "evaluation_episode_count": int(
            evaluation_runtime["evaluation_episode_count"]
        ),
        "checkpoint_training_episode_count": (
            int(checkpoint_training["training_episode_count"])
            if checkpoint_training is not None
            else None
        ),
        "checkpoint_training_git_sha": (
            checkpoint_training["training_git_sha"]
            if checkpoint_training is not None
            else None
        ),
        "evaluation_git_sha": evaluation_runtime["evaluation_git_sha"],
        "routing_optimizer_update_count": (
            int(lifecycle["routing_optimizer_update_count"])
            if lifecycle is not None
            else 0
        ),
        "routing_target_update_count": (
            int(lifecycle["routing_target_update_count"])
            if lifecycle is not None
            else 0
        ),
        "routing_epsilon_decay_start_slot": (
            lifecycle["routing_epsilon_decay_start_slot"]
            if lifecycle is not None
            else None
        ),
    }


def _evaluation_state_snapshot(
    movement_agent,
    ddqn,
    joint_replay,
    routing_replay,
    dinkelbach_state,
    schedule_counters=None,
):
    """Copy all learning state that evaluation is forbidden to mutate."""

    def replay_snapshot(replay, fields):
        if replay is None:
            return None
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
    online = {}
    targets = {}
    optimizers = {}
    if hasattr(ddqn, "q_network"):
        online["q_network"] = ddqn.q_network.state_dict()
        targets["target_q_network"] = ddqn.target_q_network.state_dict()
        optimizers["routing_reward"] = ddqn.optimizer.state_dict()
    if hasattr(ddqn, "cost_network"):
        online["cost_network"] = ddqn.cost_network.state_dict()
        targets["target_cost_network"] = ddqn.target_cost_network.state_dict()
        optimizers["routing_cost"] = ddqn.cost_optimizer.state_dict()
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
                    "current_movement_mask",
                    "next_movement_mask",
                    "movement_mask_valid",
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
            "ddqn_cost_loss_log": list(getattr(ddqn, "cost_loss_log", [])),
            "routing_target_updates": int(
                getattr(ddqn, "target_update_count", 0)
            ),
            "routing_agent_kind": str(ddqn.routing_agent_kind),
            "routing_constraint_state": (
                copy.deepcopy(ddqn.constraint_state())
                if ddqn.routing_agent_kind == "safe_ddqn"
                else None
            ),
        },
        "dinkelbach_state": copy.deepcopy(dinkelbach_state.training_state()),
        "schedule_counters": copy.deepcopy(schedule_counters),
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
        "schedule_counters_unchanged": _nested_state_equal(
            before["schedule_counters"], after["schedule_counters"]
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
    expected_checkpoint_training_manifest=None,
    training_history_manifest_hash=None,
    training_run_provenance=None,
    transition_observer=None,
    episode_observer=None,
    evaluation_overrides=None,
    trajectory_snapshot_times=None,
    trajectory_target_uav_id=None,
    packet_outcome_sink=None,
    collect_routing_q_score_diagnostics=False,
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
    formal_config = effective_training_config(config, method_spec)
    resolved_exploration = exploration_schedule_configuration(config, method_spec)
    resolved_routing = routing_agent_configuration(method_spec, config)
    packet_outcome_mode = config.packet_outcome_artifact_mode
    if packet_outcome_mode == PACKET_OUTCOME_MODE_STREAMING:
        if packet_outcome_sink is None or not callable(packet_outcome_sink):
            raise ValueError(
                "stream_jsonl packet outcome mode requires a callable per-episode sink"
            )
    elif packet_outcome_sink is not None:
        raise ValueError(
            "a packet outcome sink is valid only in stream_jsonl mode"
        )
    collect_routing_q_score_diagnostics = bool(
        collect_routing_q_score_diagnostics
    )
    if collect_routing_q_score_diagnostics and (
        not evaluation or method_spec.routing != "safe_ddqn"
    ):
        raise ValueError(
            "routing Q-score diagnostics are evaluation-only and require safe-DDQN"
        )
    if scenario_manifest is not None:
        if scenario_manifest.episode_count < config.total_episodes:
            raise ValueError(
                "scenario manifest has fewer entries than requested episodes"
            )
        if config.mode == "train" and scenario_manifest.split != "train":
            raise ValueError("formal training requires a train scenario manifest")
        if evaluation and scenario_manifest.split not in {"validation", "test"}:
            raise ValueError("evaluation requires validation or test scenarios")
        validate_manifest_initial_topologies(
            scenario_manifest, episode_count=config.total_episodes
        )
    checkpoint_required = bool(
        method_spec.learns_movement or method_spec.learns_routing
    )
    if evaluation and checkpoint_required and checkpoint_dir is None:
        raise ValueError("evaluation requires a model checkpoint")
    if evaluation and not checkpoint_required and checkpoint_dir is not None:
        raise ValueError("pure-random evaluation must not use a model checkpoint")
    if evaluation and config.random_seed is None:
        raise ValueError("evaluation requires the checkpoint training seed")
    if evaluation and config.resume_dir is not None:
        raise ValueError("evaluation cannot load a full-resume training state")
    if transition_observer is not None and not evaluation:
        raise ValueError("transition collection is available only in evaluation")
    if not evaluation and (
        evaluation_overrides
        or trajectory_snapshot_times
        or trajectory_target_uav_id is not None
    ):
        raise ValueError("paper evaluation overrides are unavailable during training")
    resolved_evaluation = _normalize_evaluation_overrides(evaluation_overrides)
    requested_snapshot_times = tuple(
        sorted({float(value) for value in (trajectory_snapshot_times or ())})
    )
    if any(
        not np.isfinite(value) or value < 0.0
        for value in requested_snapshot_times
    ):
        raise ValueError("trajectory snapshot times must be finite and non-negative")
    if requested_snapshot_times and requested_snapshot_times[-1] > float(
        config.episode_seconds
    ):
        raise ValueError("trajectory snapshot time exceeds the evaluation horizon")
    if requested_snapshot_times and trajectory_target_uav_id is None:
        raise ValueError("trajectory snapshots require an explicit target_uav_id")
    if trajectory_target_uav_id is not None and not (
        0 <= int(trajectory_target_uav_id) < NUM_UAV
    ):
        raise ValueError(f"target_uav_id must be in [0, {NUM_UAV - 1}]")
    rng_streams = _seed_training_rng(config.random_seed)

    c_ref_com, calibration = load_com_capacity_reference()
    env = Simulator(
        num_UAV=NUM_UAV, rng_streams=rng_streams, evaluation=evaluation
    )
    env.configure_method(method_spec)
    # Formal execution initializes interval zero only after the canonical
    # slot-0 SR boundary movement. Direct Simulator callers retain the eager
    # reset behavior unless they explicitly opt into this two-phase lifecycle.
    env.defer_initial_channel_boundary = True
    packet_engine = PacketEngine(
        num_uav=NUM_UAV,
        step_time=config.routing_slot_seconds,
        task_deadlines_seconds=resolved_evaluation["task_deadlines_seconds"],
        injection_cutoff_seconds=resolved_evaluation[
            "packet_injection_cutoff_seconds"
        ],
        enable_packet_diagnostic_artifacts=(
            packet_outcome_mode != PACKET_OUTCOME_MODE_DISABLED
        ),
    )
    movement_agent = create_movement_agent(
        method_spec,
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        config,
        rng_streams=rng_streams,
    )
    ddqn = create_routing_agent(
        method_spec,
        ROUTING_STATE_DIM,
        env.num_UAV + 1,
        rng_streams=rng_streams,
        evaluation=evaluation,
    )
    joint_replay = (
        utils_update_v2.ReplayBufferJoint(
            MOVEMENT_STATE_DIM,
            JOINT_ACTION_DIM,
            max_size=config.replay_max_size,
            rng=rng_streams.numpy("movement_replay_sampling"),
        )
        if method_spec.learns_movement
        else None
    )
    routing_replay = (
        utils_update_v2.ReplayBufferDiscrete(
            ROUTING_STATE_DIM,
            env.num_UAV + 1,
            max_size=ROUTING_REPLAY_CAPACITY,
            n_step=1,
            gamma=0.99,
            rng=rng_streams.numpy(
                "safe_ddqn_replay_sampling"
                if method_spec.routing == "safe_ddqn"
                else "standard_dqn_replay_sampling"
            ),
        )
        if method_spec.learns_routing
        else None
    )
    routing_lifecycle = (
        RoutingLearnerLifecycle(
            update_interval_slots=config.routing_update_interval_slots,
            gradient_steps_per_update=config.routing_gradient_steps_per_update,
            warmup_transitions=config.routing_warmup_transitions,
        )
        if method_spec.learns_routing
        else None
    )
    routing_transition_ledger = (
        RoutingTransitionLedger() if method_spec.learns_routing else None
    )
    routing_q_score_accumulator = (
        RoutingQScoreDiagnosticAccumulator()
        if collect_routing_q_score_diagnostics
        else None
    )

    experiment_identity = _experiment_identity(
        method_spec,
        scenario_manifest,
        config.random_seed,
        config,
        evaluation=evaluation,
        rng_contract_metadata=rng_streams.metadata(),
    )
    if training_run_provenance is not None:
        if evaluation:
            raise ValueError(
                "training run provenance cannot be supplied during evaluation"
            )
        if not isinstance(training_run_provenance, dict):
            raise TypeError("training run provenance must be an object")
        required_provenance = {
            "training_manifest_segments",
            "training_history_identity_manifest_hash",
            "training_history_manifest_hash",
            "initial_training_git_sha",
            "latest_training_git_sha",
            "git_sha",
        }
        missing_provenance = sorted(
            required_provenance.difference(training_run_provenance)
        )
        if missing_provenance:
            raise ValueError(
                f"training run provenance is incomplete: {missing_provenance}"
            )
        if (
            training_run_provenance["training_history_manifest_hash"]
            != training_run_provenance[
                "training_history_identity_manifest_hash"
            ]
        ):
            raise ValueError(
                "legacy training history manifest hash must alias its stable identity"
            )
        if (
            training_run_provenance["git_sha"]
            != training_run_provenance["latest_training_git_sha"]
        ):
            raise ValueError("legacy training Git SHA must identify the latest invocation")
        experiment_identity.update(copy.deepcopy(training_run_provenance))
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
            (
                scenario_manifest.content_hash
                if training_history_manifest_hash is None
                else str(training_history_manifest_hash)
            ),
        )
    loaded_checkpoint_metadata = None
    checkpoint_provenance = {}
    if evaluation and checkpoint_required:
        loaded_checkpoint_metadata = load_model_checkpoint(
            checkpoint_dir,
            movement_agent,
            ddqn,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=calibration,
            expected_experiment_metadata={
                "method_spec_fingerprint": method_spec.compatible_fingerprints,
                "training_seed": int(config.random_seed),
            },
            expected_completed_episodes=expected_checkpoint_episodes,
            expected_formal_config=expected_checkpoint_formal_config,
            current_training_manifest=expected_checkpoint_training_manifest,
        )
        checkpoint_experiment = loaded_checkpoint_metadata["experiment"]
        loaded_training_provenance = checkpoint_training_provenance(
            loaded_checkpoint_metadata
        )
        artifact_provenance = checkpoint_artifact_provenance(
            checkpoint_dir, metadata=loaded_checkpoint_metadata
        )
        horizon_compatibility = (
            checkpoint_run_compatibility_from_metadata(
                loaded_checkpoint_metadata,
                expected_checkpoint_formal_config,
                checkpoint_episode=expected_checkpoint_episodes,
                current_training_manifest=expected_checkpoint_training_manifest,
            )
            if expected_checkpoint_formal_config is not None
            else {}
        )
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
            "checkpoint_training_provenance": loaded_training_provenance,
            "checkpoint_training_episode_count": int(
                loaded_training_provenance["training_episode_count"]
            ),
            "checkpoint_training_git_sha": loaded_training_provenance[
                "training_git_sha"
            ],
            **horizon_compatibility,
            "checkpoint_schema_version": loaded_checkpoint_metadata.get(
                "checkpoint_schema_version"
            ),
            "checkpoint_training_exploration_schedule_version": (
                checkpoint_experiment.get("formal_config", {})
                .get("exploration_schedule_configuration", {})
                .get("exploration_schedule_version")
                or "legacy_or_unrecorded"
            ),
            "checkpoint_training_schedule_conforms_to_current": bool(
                loaded_checkpoint_metadata.get("checkpoint_schema_version")
                == CHECKPOINT_SCHEMA_VERSION
                and checkpoint_experiment.get("formal_config", {})
                .get("exploration_schedule_configuration", {})
                .get("exploration_schedule_version")
                == resolved_exploration["exploration_schedule_version"]
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
            **artifact_provenance,
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
    elif evaluation:
        checkpoint_provenance = {
            "training_manifest_hash": None,
            "evaluation_manifest_hash": (
                scenario_manifest.content_hash
                if scenario_manifest is not None
                else None
            ),
            "checkpoint_required": False,
            "checkpoint_completed_episodes": None,
            "checkpoint_training_seed": None,
            "checkpoint_method_spec_fingerprint": None,
            "checkpoint_metadata_path": None,
            "checkpoint_metadata_fingerprint": None,
            "checkpoint_models_sha256": None,
            "checkpoint_artifact_fingerprint": None,
            "checkpoint_training_provenance": None,
            "checkpoint_training_episode_count": None,
            "checkpoint_training_git_sha": None,
            "no_checkpoint_reason": (
                "method learns neither movement nor routing; no neural state exists"
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
    lambda_cost_used_log = []
    lambda_cost_after_episode_log = []
    episode_metrics = []
    trajectory_artifacts = []
    packet_outcome_artifacts = (
        [] if packet_outcome_mode == PACKET_OUTCOME_MODE_BOUNDED else None
    )
    packet_outcome_streamed_episode_count = 0
    resume_checkpoint_compatibility = None
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
                "method_spec_fingerprint": method_spec.compatible_fingerprints,
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
            expected_formal_config=formal_config,
            current_training_manifest=scenario_manifest,
            current_training_manifest_segments=experiment_identity.get(
                "training_manifest_segments"
            ),
            training_run_directory=config.run_directory,
        )
        training_state = restored["training_state"]
        if routing_transition_ledger is not None:
            packet_state = training_state["packet_engine_state"]
            routing_transition_ledger.load_state_dict(
                training_state["routing_transition_state"],
                reference_counts=packet_state[
                    "routing_transition_reference_counts"
                ],
            )
        try:
            rng_streams.load_state_dict(training_state["named_rng_state"])
        except KeyError as exc:
            raise RuntimeError(
                "resume checkpoint lacks named subsystem RNG state"
            ) from exc
        try:
            env.load_channel_state_dict(
                training_state["channel_lifecycle_state"]
            )
        except KeyError as exc:
            raise RuntimeError(
                "resume checkpoint lacks channel lifecycle state"
            ) from exc
        resume_checkpoint_compatibility = restored["horizon_compatibility"]
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
        if method_spec.learns_routing:
            routing_lifecycle = RoutingLearnerLifecycle.from_state(
                training_state.get("routing_lifecycle_state"),
                update_interval_slots=config.routing_update_interval_slots,
                gradient_steps_per_update=(
                    config.routing_gradient_steps_per_update
                ),
                warmup_transitions=config.routing_warmup_transitions,
            )
            if routing_lifecycle.global_slot_count != routing_slots_executed:
                raise RuntimeError(
                    "routing lifecycle slot counter is inconsistent with history"
                )
            if (
                routing_lifecycle.optimizer_update_count != ddqn.num_training
                or routing_lifecycle.target_update_count
                != ddqn.target_update_count
            ):
                raise RuntimeError(
                    "routing lifecycle update counters are inconsistent with agent state"
                )
        elif training_state.get("routing_lifecycle_state") is not None:
            raise RuntimeError(
                "random-routing checkpoint must not contain learner lifecycle state"
            )
        try:
            packet_engine.load_fov_ema_state(training_state["fov_ema_state"])
        except KeyError as exc:
            raise RuntimeError(
                "resume checkpoint lacks FOV EMA lifecycle state"
            ) from exc
        try:
            saved_sr_route_state = training_state["sr_route_state"]
        except KeyError as exc:
            raise RuntimeError(
                "resume checkpoint lacks SR route lifecycle state"
            ) from exc
        if (
            saved_sr_route_state.get("lifecycle_version")
            != SR_ROUTE_LIFECYCLE_VERSION
            or saved_sr_route_state.get("checkpoint_scope")
            != "episode_boundary_terminal_snapshot"
            or bool(saved_sr_route_state.get("mid_episode_checkpoint_supported"))
        ):
            raise RuntimeError("resume checkpoint SR route state is incompatible")
        if method_spec.routing == "safe_ddqn":
            try:
                lambda_cost_used_log = list(
                    training_state["lambda_cost_used_log"]
                )
                lambda_cost_after_episode_log = list(
                    training_state["lambda_cost_after_episode_log"]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "safe-DDQN resume checkpoint lacks adaptive multiplier logs"
                ) from exc
            if not (
                len(lambda_cost_used_log)
                == len(lambda_cost_after_episode_log)
                == start_episode
            ):
                raise RuntimeError(
                    "safe-DDQN multiplier history is inconsistent with resume episode"
                )
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
        expected_ddqn_slot = (
            routing_lifecycle.global_slot_count
            if routing_lifecycle is not None
            else 0
        )
        if int(training_state["ddqn_schedule_slot"]) != expected_ddqn_slot:
            raise RuntimeError("DDQN exploration counter is inconsistent with slot history")
        expected_exploration_state = {
            key: resolved_exploration[key]
            for key in (
                "exploration_schedule_version",
                "movement_exploration_decay_episodes",
                "routing_epsilon_decay_episodes",
                "resolved_movement_decay_steps",
                "resolved_routing_decay_slots",
                "movement_noise_start",
                "movement_noise_end",
                "routing_epsilon_start",
                "routing_epsilon_end",
            )
        }
        mismatches = {
            key: (training_state.get(key), value)
            for key, value in expected_exploration_state.items()
            if training_state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "checkpoint exploration schedule is incompatible: "
                f"{mismatches}"
            )
    elif history_identity is not None:
        training_history_rows = prepare_training_history(
            config.run_directory, history_identity
        )

    schedule_counters_before = {
        "movement_post_warmup_transition_count": max(
            total_joint_transitions - config.warmup_joint_transitions, 0
        ),
        "routing_lifecycle_state": (
            routing_lifecycle.state_dict()
            if routing_lifecycle is not None
            else None
        ),
    }
    evaluation_state_before = (
        _evaluation_state_snapshot(
            movement_agent,
            ddqn,
            joint_replay,
            routing_replay,
            dinkelbach_state,
            schedule_counters_before,
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
    evaluation_observation_transition_index = 0
    td3_schedule_decay = resolved_exploration[
        "resolved_movement_decay_steps"
    ]
    ddqn_schedule_decay = resolved_exploration["resolved_routing_decay_slots"]
    delay_bound_steps = int(5.0 / config.routing_slot_seconds)

    ddqn_action_selections = 0
    executed_scenario_ids = []
    for episode in range(start_episode, config.total_episodes):
        if scenario_manifest is None:
            env.num_GT = int(
                env.environment_rng.integers(ROI_COUNT_MIN, ROI_COUNT_MAX + 1)
            )
            env.reset_environment()
            scenario_id = None
        else:
            scenario_entry = scenario_manifest.episodes[episode]
            env.apply_scenario_entry(scenario_entry)
            scenario_id = str(scenario_entry["scenario_id"])
        executed_scenario_ids.append(scenario_id)
        env.prepare_initial_movement_interval()
        packet_engine.reset_packet_state()
        packet_engine.update_fov_ema(
            env, transition_marker=f"episode={episode},map_reset"
        )
        scenario_entry = (
            scenario_manifest.episodes[episode]
            if scenario_manifest is not None
            else None
        )
        scenario_rates = dict(
            (scenario_entry or {}).get("traffic_primitives", {})
        )
        resolved_episode_rates = {
            "FOV": (
                resolved_evaluation["traffic_rates_packets_per_second"]["FOV"]
                if resolved_evaluation["traffic_rates_packets_per_second"]["FOV"]
                is not None
                else float(scenario_rates.get("base_fov_packets_per_second", 5.0))
            ),
            "COM": (
                resolved_evaluation["traffic_rates_packets_per_second"]["COM"]
                if resolved_evaluation["traffic_rates_packets_per_second"]["COM"]
                is not None
                else float(scenario_rates.get("base_com_packets_per_second", 50.0))
            ),
        }
        env.com_offered_rate_bps = (
            float(resolved_episode_rates["COM"]) * COM_PACKET_SIZE_BITS
        )
        latest_active_links = []
        trajectory_history = [
            _trajectory_state(
                env,
                actual_time_seconds=0.0,
                target_uav_id=trajectory_target_uav_id,
                active_links=latest_active_links,
            )
        ]
        pending_snapshot_times = list(requested_snapshot_times)
        episode_snapshots = []
        env.lambda_EE_global = float(lambda_ee)
        episode_lambda = float(lambda_ee)
        episode_delivered_mbits = 0.0
        episode_energy = 0.0
        episode_reward = 0.0
        episode_routing_reward = 0.0
        episode_lambda_cost = (
            float(ddqn.lambda_cost)
            if method_spec.routing == "safe_ddqn"
            else None
        )
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
        pending_routing_transitions = (
            {} if routing_transition_ledger is None else None
        )
        expected_next_movement_state = None
        expected_next_movement_mask = None
        expected_next_movement_potentials = None

        for interval in range(config.episode_seconds):
            if env.channel.movement_interval_index != interval:
                raise AssertionError(
                    "movement decision does not reuse its boundary-prepared channel state"
                )

            backlog_before = _active_backlog(packet_engine)
            try:
                physical_state = get_global_movement_state(
                    env,
                    packet_engine,
                    backlog_before,
                    c_ref_com,
                    remaining_time=(config.episode_seconds - interval)
                    / config.episode_seconds,
                )
                state = apply_observation_strategy(
                    physical_state,
                    method_spec.task_observation,
                    "movement",
                )
                potentials_t = calculate_movement_potentials(env, c_ref_com)
                current_movement_mask = movement_mask_from_state(physical_state)
            except ValueError as exc:
                if "duplicate" in str(exc):
                    duplicate_target_assertions += 1
                raise
            if expected_next_movement_state is not None:
                if not np.array_equal(state, expected_next_movement_state):
                    raise AssertionError(
                        "movement replay next_state differs from the next policy observation"
                    )
                if not np.array_equal(
                    current_movement_mask, expected_next_movement_mask
                ):
                    raise AssertionError(
                        "movement replay next mask differs from the next policy mask"
                    )
                if tuple(potentials_t) != tuple(expected_next_movement_potentials):
                    raise AssertionError(
                        "movement replay next potentials differ from the next transition"
                    )
                expected_next_movement_state = None
                expected_next_movement_mask = None
                expected_next_movement_potentials = None

            if method_spec.agent == "random":
                raw_joint_action = sample_random_joint_action(
                    JOINT_ACTION_DIM,
                    rng_streams.numpy(
                        "evaluation_random_movement"
                        if evaluation
                        else "random_movement"
                    ),
                )
            elif evaluation:
                raw_joint_action = movement_agent.select_action(
                    state, add_noise=False, noise_std=0.0
                )
                environment_actor_calls += 1
            elif _uses_warmup_random_action(
                total_joint_transitions, config.warmup_joint_transitions
            ):
                raw_joint_action = rng_streams.numpy(
                    "movement_exploration"
                ).uniform(-1.0, 1.0, size=JOINT_ACTION_DIM).astype(np.float32)
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
            projected_action = project_joint_action(
                raw_joint_action, movement_mask=current_movement_mask
            )
            velocity_commands = decode_joint_velocity_commands(
                movement_agent, projected_action
            )
            interval_initial_positions = np.asarray(
                [env.uav_dict[uav_id].get_position() for uav_id in range(env.num_UAV)],
                dtype=np.float64,
            )
            env.update_source_uavs()
            interval_energies = np.zeros(env.num_UAV, dtype=np.float64)
            interval_delivered_bits = 0.0
            for routing_slot in range(MOVEMENT_CONTROL_INTERVAL):
                # All UAV proposals come from one substep snapshot. The same
                # decoded command is held across all four 0.25-second slots.
                proposals = build_velocity_substep_proposals(
                    env, velocity_commands, config.routing_slot_seconds
                )
                proposal_batches += 1
                substep_energies = apply_joint_movement_proposals(
                    env, proposals, step_time=config.routing_slot_seconds
                )
                interval_energies += substep_energies
                energy_evaluations += int(substep_energies.size)
                # Channel geometry is refreshed after each actual projected
                # displacement and before that substep's routing decision.
                env.update_u2u_channels()
                env.update_u2g_channels()
                slot_epsilon = (
                    0.0
                    if evaluation or not method_spec.learns_routing
                    else routing_lifecycle.epsilon(ddqn_schedule_decay)
                )
                routing_epsilon_log.append(slot_epsilon)
                absolute_slot = interval * MOVEMENT_CONTROL_INTERVAL + routing_slot
                final_slot = (
                    interval == config.episode_seconds - 1
                    and routing_slot == MOVEMENT_CONTROL_INTERVAL - 1
                )
                (
                    delivered_bits,
                    routing_reward,
                    action_selections,
                    latest_active_links,
                ) = _run_routing_slot(
                    env,
                    packet_engine,
                    ddqn,
                    routing_replay,
                    None,
                    current_time=absolute_slot * config.routing_slot_seconds,
                    done=final_slot,
                    delay_bound_steps=delay_bound_steps,
                    violation_stats=violation_stats,
                    epsilon=slot_epsilon,
                    write_replay=(not evaluation and method_spec.learns_routing),
                    task_observation_mode=method_spec.task_observation,
                    traffic_rate_overrides=resolved_episode_rates,
                    pending_routing_transitions=pending_routing_transitions,
                    routing_transition_ledger=routing_transition_ledger,
                    routing_q_score_accumulator=routing_q_score_accumulator,
                    routing_q_score_context=(
                        {
                            "scenario_id": scenario_id,
                            "episode_index": episode,
                            "slot_index": absolute_slot,
                        }
                        if routing_q_score_accumulator is not None
                        else None
                    ),
                )
                ddqn_action_selections += action_selections
                interval_delivered_bits += delivered_bits
                episode_routing_reward += routing_reward
                routing_slots_executed += 1
                if not evaluation and method_spec.learns_routing:
                    routing_lifecycle.complete_slot(
                        ddqn, routing_replay, config.batch_size
                    )

            executed_action = executed_joint_action_from_displacement(
                interval_initial_positions,
                np.asarray(
                    [
                        env.uav_dict[uav_id].get_position()
                        for uav_id in range(env.num_UAV)
                    ],
                    dtype=np.float64,
                ),
                MOVEMENT_INTERVAL_SECONDS,
            )
            interval_energy = float(interval_energies.sum())
            # Search observation, RoI discovery, and task assignment remain
            # one-second boundary events and therefore execute exactly once.
            fov_transitions = _mark_search_observations(env)
            if fov_transitions:
                packet_engine.process_fov_transitions(
                    env,
                    transition_marker=f"episode={episode},interval={interval}",
                    footprint_transitions=fov_transitions,
                )
            if (
                not getattr(env, "_search_phase_over", False)
                and float(env.visited_bitmap.mean())
                >= config.search_coverage_threshold
            ):
                env.convert_search_to_hovering(defer_assignment=True)

            interval_delivered_mbits = interval_delivered_bits / 1e6
            done = interval == config.episode_seconds - 1
            if not done:
                env.prepare_next_movement_interval(interval + 1)
            env.update_source_uavs()
            actual_time_seconds = float(interval + 1)
            trajectory_history.append(
                _trajectory_state(
                    env,
                    actual_time_seconds=actual_time_seconds,
                    target_uav_id=trajectory_target_uav_id,
                    active_links=latest_active_links,
                )
            )
            while (
                pending_snapshot_times
                and actual_time_seconds >= pending_snapshot_times[0]
            ):
                requested_time = pending_snapshot_times.pop(0)
                episode_snapshots.append(
                    {
                        "requested_time_seconds": requested_time,
                        **copy.deepcopy(trajectory_history[-1]),
                    }
                )
            backlog_after = _active_backlog(packet_engine)
            potentials_t1 = calculate_movement_potentials(env, c_ref_com)
            physical_next_state = get_global_movement_state(
                env,
                packet_engine,
                backlog_after,
                c_ref_com,
                remaining_time=(config.episode_seconds - (interval + 1))
                / config.episode_seconds,
            )
            next_state = apply_observation_strategy(
                physical_next_state,
                method_spec.task_observation,
                "movement",
            )
            next_movement_mask = movement_mask_from_state(physical_next_state)
            if not done:
                expected_next_movement_state = next_state.copy()
                expected_next_movement_mask = next_movement_mask.copy()
                expected_next_movement_potentials = tuple(potentials_t1)
            terminal_joint_transitions += int(done)
            episode_delivered_mbits += interval_delivered_mbits
            episode_energy += interval_energy
            ratio_objective_reward = terminal_ratio_objective(
                method_spec.reward_mode,
                done,
                episode_delivered_mbits,
                episode_energy,
            )
            if not evaluation and method_spec.learns_movement:
                joint_replay.add(
                    state,
                    executed_action,
                    next_state,
                    done=done,
                    delivered_mbits=interval_delivered_mbits,
                    total_mobility_energy=interval_energy,
                    ratio_objective_reward=ratio_objective_reward,
                    phi_search_t=potentials_t[0],
                    phi_search_t1=potentials_t1[0],
                    phi_vs_t=potentials_t[1],
                    phi_vs_t1=potentials_t1[1],
                    phi_com_t=potentials_t[2],
                    phi_com_t1=potentials_t1[2],
                    current_movement_mask=current_movement_mask,
                    next_movement_mask=next_movement_mask,
                )
            global_transition_index = (
                evaluation_observation_transition_index
                if evaluation
                else total_joint_transitions
            )
            if not evaluation and method_spec.learns_movement:
                total_joint_transitions += 1
            if evaluation:
                evaluation_observation_transition_index += 1
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
                ratio_objective_reward=ratio_objective_reward,
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
                        "ratio_objective_reward": ratio_objective_reward,
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

        if pending_routing_transitions:
            raise AssertionError("terminal routing transitions remained pending")
        packet_metrics = packet_engine.finalize_episode(
            float(config.episode_seconds)
        )
        if not evaluation and method_spec.learns_routing:
            for violation in packet_engine.pending_terminal_violation_events:
                transition_id = violation.get("routing_transition_id")
                if transition_id is None:
                    continue
                if routing_transition_ledger.add_cost(transition_id, 1.0):
                    packet_engine.replay_attributed_violation_cost_count += 1.0
                else:
                    raise AssertionError(
                        "stable routing transition ID rejected terminal cost"
                    )
            routing_transition_ledger.finalize_causality(
                {}, {}, terminal=True
            )
            routing_transition_ledger.commit_ready(
                routing_replay,
                packet_engine.routing_transition_reference_counts(),
            )
            routing_transition_ledger.assert_drained(
                packet_engine.routing_transition_reference_counts()
            )
            packet_engine.pending_terminal_violation_events.clear()
            packet_engine.pending_terminal_cost_by_sender.clear()
        else:
            packet_engine.pending_terminal_violation_events.clear()
            packet_engine.pending_terminal_cost_by_sender.clear()
        (
            episode_system_violation_count,
            episode_system_eligible_packet_count,
        ) = packet_engine.system_qos_counts()
        (
            episode_routing_cost_sum,
            episode_routing_eligible_packet_count,
        ) = packet_engine.routing_constraint_counts()
        packet_engine.assert_violation_credit_conservation()
        episode_routing_cost_sum = float(episode_routing_cost_sum)
        if (
            not evaluation
            and method_spec.learns_routing
            and not np.isclose(
                episode_routing_cost_sum,
                packet_engine.replay_attributed_violation_cost_count,
            )
        ):
            raise AssertionError(
                "replay-attributed costs differ from routing-credit violations"
            )
        episode_violation_probability = (
            episode_system_violation_count
            / float(episode_system_eligible_packet_count)
            if episode_system_eligible_packet_count
            else None
        )
        lambda_cost_after_episode = episode_lambda_cost
        if method_spec.routing == "safe_ddqn":
            if not evaluation:
                lambda_cost_after_episode = ddqn.update_cost_multiplier(
                    episode_routing_cost_sum,
                    episode_routing_eligible_packet_count,
                )
            else:
                lambda_cost_after_episode = float(ddqn.lambda_cost)
            lambda_cost_used_log.append(float(episode_lambda_cost))
            lambda_cost_after_episode_log.append(
                float(lambda_cost_after_episode)
            )
        if packet_outcome_mode != PACKET_OUTCOME_MODE_DISABLED:
            packet_outcome_record = packet_outcome_episode_record(
                scenario_id,
                packet_metrics,
                packet_engine.packet_outcomes,
            )
            if packet_outcome_mode == PACKET_OUTCOME_MODE_BOUNDED:
                packet_outcome_artifacts.append(
                    copy.deepcopy(packet_outcome_record)
                )
            else:
                packet_outcome_sink(packet_outcome_record)
                packet_outcome_streamed_episode_count += 1
        if requested_snapshot_times:
            trajectory_artifacts.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_manifest_hash": (
                        scenario_manifest.content_hash
                        if scenario_manifest is not None
                        else None
                    ),
                    "requested_times_seconds": list(requested_snapshot_times),
                    "target_uav_id": int(trajectory_target_uav_id),
                    "snapshots": episode_snapshots,
                    "trajectory_history": trajectory_history,
                    "uav_paths": {
                        str(uav_id): [
                            {
                                "actual_time_seconds": state["actual_time_seconds"],
                                "x": uav["x"],
                                "y": uav["y"],
                                "z": uav["z"],
                                "task_phase": uav["task_phase"],
                                "assigned_tasks": copy.deepcopy(
                                    uav.get("assigned_tasks", [])
                                ),
                            }
                            for state in trajectory_history
                            for uav in state["uavs"]
                            if uav["uav_id"] == uav_id
                        ]
                        for uav_id in range(env.num_UAV)
                    },
                    "sr_paths": {
                        str(sr_id): [
                            {
                                "actual_time_seconds": state["actual_time_seconds"],
                                "x": sr["x"],
                                "y": sr["y"],
                                "z": sr["z"],
                            }
                            for state in trajectory_history
                            for sr in state["sr_teams"]
                            if sr["sr_id"] == sr_id
                        ]
                        for sr_id in sorted(
                            {sr["sr_id"] for sr in trajectory_history[0]["sr_teams"]}
                        )
                    },
                    "ground_targets": copy.deepcopy(
                        (scenario_entry or {}).get("ground_targets", [])
                    ),
                    "initial_sr_teams": copy.deepcopy(
                        (scenario_entry or {}).get("sr_teams", [])
                    ),
                }
            )
        timely_goodput_mbits = float(packet_engine.timely_goodput_bits) / 1e6
        total_timely_useful_bits = float(
            packet_engine.total_timely_useful_bits
        )
        if not np.isclose(
            packet_engine.timely_goodput_bits,
            total_timely_useful_bits,
            rtol=0.0,
            atol=1e-9,
        ):
            raise AssertionError("timely-goodput alias diverged from useful bits")
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
                "total_timely_useful_mbits": total_timely_useful_bits / 1e6,
                "fov_generated_raw_bits": float(
                    packet_engine.fov_generated_raw_bits
                ),
                "fov_timely_delivered_raw_bits": float(
                    packet_engine.fov_timely_delivered_raw_bits
                ),
                "fov_timely_useful_bits": float(
                    packet_engine.fov_timely_useful_bits
                ),
                "fov_mean_capture_coverage": (
                    float(packet_engine.fov_capture_coverage_sum)
                    / int(packet_engine.fov_capture_coverage_count)
                    if packet_engine.fov_capture_coverage_count
                    else None
                ),
                "fov_zero_coverage_packet_count": int(
                    packet_engine.fov_zero_coverage_packet_count
                ),
                "com_timely_delivered_bits": float(
                    packet_engine.com_timely_delivered_bits
                ),
                "total_timely_useful_bits": total_timely_useful_bits,
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
                **{
                    f"{task.lower()}_{field}": value
                    for task, metrics in packet_metrics.items()
                    for field, value in metrics.items()
                },
                "fov_rate_packets_per_second": float(
                    resolved_episode_rates["FOV"]
                ),
                "com_rate_packets_per_second": float(
                    resolved_episode_rates["COM"]
                ),
                "fov_deadline_seconds": float(
                    resolved_evaluation["task_deadlines_seconds"]["FOV"]
                ),
                "com_deadline_seconds": float(
                    resolved_evaluation["task_deadlines_seconds"]["COM"]
                ),
                "packet_injection_cutoff_seconds": float(
                    resolved_evaluation["packet_injection_cutoff_seconds"]
                ),
                "episode_horizon_seconds": float(config.episode_seconds),
                "routing_cost_sum": episode_routing_cost_sum,
                "eligible_packet_count": episode_system_eligible_packet_count,
                "delay_violation_probability": episode_violation_probability,
                "sr_admission_drop_count": int(
                    packet_engine.sr_admission_drop_count
                ),
                "system_qos_violation_count": episode_system_violation_count,
                "system_qos_eligible_packets": (
                    episode_system_eligible_packet_count
                ),
                "routing_cost_eligible_packets": (
                    episode_routing_eligible_packet_count
                ),
                "routing_credit_violation_count": int(
                    packet_engine.routing_credit_violation_count
                ),
                "replay_attributed_violation_cost_count": float(
                    packet_engine.replay_attributed_violation_cost_count
                ),
                "unattributed_transition_violation_count": int(
                    packet_engine.unattributed_transition_violation_count
                ),
                "unattributed_pre_routing_violation_count": int(
                    packet_engine.unattributed_pre_routing_violation_count
                ),
                "lambda_cost_used": episode_lambda_cost,
                "lambda_cost_after_episode": lambda_cost_after_episode,
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
                    total_timely_useful_mbits=timely_goodput_mbits,
                    mobility_energy_j=episode_energy,
                    eligible_packet_count=episode_system_eligible_packet_count,
                    delay_violation_count=int(episode_system_violation_count),
                    delay_violation_probability=episode_violation_probability,
                    lambda_cost_used=episode_lambda_cost,
                    lambda_cost_after_episode=lambda_cost_after_episode,
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
            checkpoint_required
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
                    "formal_config": formal_config,
                },
                routing_lifecycle_state=(
                    routing_lifecycle.state_dict()
                    if routing_lifecycle is not None
                    else None
                ),
            )
        if (
            not evaluation
            and
            checkpoint_required
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
                    lambda_cost_used_log=lambda_cost_used_log,
                    lambda_cost_after_episode_log=(
                        lambda_cost_after_episode_log
                    ),
                    fov_ema_state=packet_engine.fov_ema_state(),
                    sr_route_state=env.sr_route_state(),
                    routing_lifecycle_state=(
                        routing_lifecycle.state_dict()
                        if routing_lifecycle is not None
                        else None
                    ),
                    exploration_state={
                        key: resolved_exploration[key]
                        for key in (
                            "exploration_schedule_version",
                            "movement_exploration_decay_episodes",
                            "routing_epsilon_decay_episodes",
                            "resolved_movement_decay_steps",
                            "resolved_routing_decay_slots",
                            "movement_noise_start",
                            "movement_noise_end",
                            "routing_epsilon_start",
                            "routing_epsilon_end",
                        )
                    },
                    warmup_joint_transitions=config.warmup_joint_transitions,
                    training_history_rows=training_history_rows,
                    dinkelbach_active=method_spec.uses_dinkelbach,
                    named_rng_state=rng_streams.state_dict(),
                    channel_lifecycle_state=env.channel_state_dict(),
                    routing_transition_state=(
                        routing_transition_ledger.state_dict(
                            packet_engine.routing_transition_reference_counts()
                        )
                        if routing_transition_ledger is not None
                        else RoutingTransitionLedger().state_dict()
                    ),
                    packet_engine_state=packet_engine.checkpoint_state(),
                ),
                formal_config=formal_config,
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
                    "formal_config": formal_config,
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
            {
                "movement_post_warmup_transition_count": max(
                    total_joint_transitions - config.warmup_joint_transitions,
                    0,
                ),
                "routing_lifecycle_state": (
                    routing_lifecycle.state_dict()
                    if routing_lifecycle is not None
                    else None
                ),
            },
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
            packet_engine.total_violated
            == packet_engine.deadline_drops
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

    checkpoint_training = checkpoint_provenance.get(
        "checkpoint_training_provenance"
    )
    evaluation_runtime = (
        _evaluation_runtime_provenance(
            method_spec,
            scenario_manifest,
            config,
            resolved_evaluation,
            routing_lifecycle,
            ddqn,
            experiment_identity["git_sha"],
        )
        if evaluation
        else None
    )
    evaluation_aliases = (
        _evaluation_provenance_aliases(checkpoint_training, evaluation_runtime)
        if evaluation
        else None
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
        "num_uav": NUM_UAV,
        "reserved_search_uav_ids": list(env.reserved_search_uav_ids),
        "search_release_time_seconds": env.search_release_time,
        "search_release_coverage": env.search_release_coverage,
        "assignment_invocations": int(env.assignment_invocations),
        "movement_agent_kind": movement_agent.agent_kind,
        "movement_agent_gamma": movement_agent.gamma,
        "movement_agent_configuration": movement_agent_configuration(
            method_spec, config
        ),
        **(
            {"centralized_td3_gamma": movement_agent.gamma}
            if movement_agent.agent_kind == "td3"
            else {}
        ),
        "routing_ddqn_gamma": ddqn.gamma,
        "routing_agent_kind": ddqn.routing_agent_kind,
        "routing_policy": method_spec.routing,
        "joint_replay_size": (
            int(joint_replay.size) if joint_replay is not None else 0
        ),
        "joint_replay_diagnostics": (
            joint_replay.diagnostics() if joint_replay is not None else None
        ),
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "rng_contract": rng_streams.metadata(),
        "routing_replay_size": (
            int(routing_replay.size) if routing_replay is not None else 0
        ),
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
        "routing_training_global_slot_count": (
            routing_lifecycle.global_slot_count
            if routing_lifecycle is not None
            else 0
        ),
        "ddqn_action_selections": ddqn_action_selections,
        "ddqn_training_updates": ddqn.num_training,
        "routing_target_update_count": getattr(
            ddqn, "target_update_count", 0
        ),
        "routing_reward_optimizer_update_count": getattr(
            ddqn, "reward_optimizer_update_count", 0
        ),
        "routing_cost_optimizer_update_count": getattr(
            ddqn, "cost_optimizer_update_count", 0
        ),
        "routing_reward_target_update_count": getattr(
            ddqn, "reward_target_update_count", 0
        ),
        "routing_cost_target_update_count": getattr(
            ddqn, "cost_target_update_count", 0
        ),
        "routing_lifecycle_state": (
            routing_lifecycle.state_dict()
            if routing_lifecycle is not None
            else None
        ),
        "exploration_schedule_configuration": resolved_exploration,
        "movement_post_warmup_transition_count": max(
            total_joint_transitions - config.warmup_joint_transitions, 0
        ),
        "lambda_cost": (
            float(ddqn.lambda_cost)
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "safe_ddqn_constraint_state": (
            ddqn.constraint_state()
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "lambda_cost_used_log": lambda_cost_used_log,
        "lambda_cost_after_episode_log": lambda_cost_after_episode_log,
        "fov_ema_state": packet_engine.fov_ema_state(),
        "sr_route_state": env.sr_route_state(),
        "channel_lifecycle_state": env.channel_state_dict(),
        "channel_profile_generation_seconds": (
            env.channel.last_profile_generation_seconds
        ),
        "td3_noise_log": td3_noise_log,
        "movement_noise_log": td3_noise_log,
        "routing_epsilon_log": routing_epsilon_log,
        "raw_final_hop_bits": packet_engine.raw_final_hop_bits,
        "timely_goodput_bits": packet_engine.timely_goodput_bits,
        "fov_generated_raw_bits": packet_engine.fov_generated_raw_bits,
        "fov_timely_delivered_raw_bits": (
            packet_engine.fov_timely_delivered_raw_bits
        ),
        "fov_timely_useful_bits": packet_engine.fov_timely_useful_bits,
        "fov_mean_capture_coverage": (
            packet_engine.fov_capture_coverage_sum
            / packet_engine.fov_capture_coverage_count
            if packet_engine.fov_capture_coverage_count
            else None
        ),
        "fov_zero_coverage_packet_count": (
            packet_engine.fov_zero_coverage_packet_count
        ),
        "com_timely_delivered_bits": packet_engine.com_timely_delivered_bits,
        "total_timely_useful_bits": packet_engine.total_timely_useful_bits,
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
        "packet_outcome_artifacts": packet_outcome_artifacts,
        "packet_outcome_streamed_episode_count": (
            packet_outcome_streamed_episode_count
        ),
        "routing_q_score_diagnostics": (
            routing_q_score_accumulator.summary()
            if routing_q_score_accumulator is not None
            else None
        ),
        "routing_q_score_voluntary_waits": (
            list(routing_q_score_accumulator.voluntary_wait_events)
            if routing_q_score_accumulator is not None
            else None
        ),
        "trajectory_artifacts": [
            {
                **artifact,
                "method_id": method_spec.method_id,
                "method_spec": method_spec.to_dict(),
                "checkpoint_path": (
                    os.path.abspath(checkpoint_dir) if checkpoint_dir else None
                ),
                "checkpoint_required": checkpoint_required,
                "checkpoint_fingerprint": checkpoint_provenance.get(
                    "checkpoint_metadata_fingerprint"
                ),
                "checkpoint_metadata_fingerprint": checkpoint_provenance.get(
                    "checkpoint_metadata_fingerprint"
                ),
                "checkpoint_models_sha256": checkpoint_provenance.get(
                    "checkpoint_models_sha256"
                ),
                "checkpoint_artifact_fingerprint": checkpoint_provenance.get(
                    "checkpoint_artifact_fingerprint"
                ),
                "checkpoint_training_provenance": copy.deepcopy(
                    checkpoint_training
                ),
                "evaluation_runtime_provenance": copy.deepcopy(
                    evaluation_runtime
                ),
            }
            for artifact in trajectory_artifacts
        ],
        "run_metadata": {
            **experiment_identity,
            **checkpoint_provenance,
            "resume_checkpoint_compatibility": resume_checkpoint_compatibility,
            "checkpoint_training_provenance": copy.deepcopy(
                checkpoint_training
            ),
            "evaluation_runtime_provenance": copy.deepcopy(
                evaluation_runtime
            ),
            **(
                evaluation_aliases
                if evaluation
                else {
                    "training_episode_count": int(
                        experiment_identity["training_episode_count"]
                    ),
                    "evaluation_episode_count": None,
                    "checkpoint_training_episode_count": None,
                    "checkpoint_training_git_sha": None,
                    "evaluation_git_sha": None,
                }
            ),
            "formal_config": formal_config,
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
            "safe_ddqn_constraint_state": (
                ddqn.constraint_state()
                if method_spec.routing == "safe_ddqn"
                else None
            ),
            "routing_mask_scope": "every_slot",
            **resolved_routing,
            **resolved_exploration,
            "routing_optimizer_update_count": (
                evaluation_aliases["routing_optimizer_update_count"]
                if evaluation
                else int(ddqn.num_training)
            ),
            "routing_target_update_count": (
                evaluation_aliases["routing_target_update_count"]
                if evaluation
                else int(getattr(ddqn, "target_update_count", 0))
            ),
            "routing_epsilon_decay_start_slot": (
                evaluation_aliases["routing_epsilon_decay_start_slot"]
                if evaluation
                else (
                    routing_lifecycle.epsilon_decay_start_slot
                    if routing_lifecycle is not None
                    else None
                )
            ),
            "lambda_cost_source": (
                "checkpoint_frozen"
                if evaluation and method_spec.routing == "safe_ddqn"
                else None
            ),
            "fov_ema_state": packet_engine.fov_ema_state(),
            "sr_route_state": env.sr_route_state(),
            "resolved_packet_configuration": {
                **copy.deepcopy(resolved_evaluation),
                "episode_horizon_seconds": float(config.episode_seconds),
                "com_packet_size_bits": COM_PACKET_SIZE_BITS,
                "com_offered_rate_bps": (
                    float(resolved_evaluation["traffic_rates_packets_per_second"]["COM"])
                    * COM_PACKET_SIZE_BITS
                    if resolved_evaluation["traffic_rates_packets_per_second"]["COM"]
                    is not None
                    else COM_OFFERED_RATE_BPS
                ),
                "communication_bandwidth_pool_hz": 10e6,
                "fdma_policy": "equal-across-active-S2U-U2U-U2G",
            },
            "evaluation": bool(evaluation),
            "checkpoint_required": checkpoint_required,
            "packet_outcome_streamed_episode_count": (
                packet_outcome_streamed_episode_count
            ),
            "routing_q_score_diagnostics_enabled": bool(
                routing_q_score_accumulator is not None
            ),
            "routing_q_score_diagnostic_contract_version": (
                ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION
                if routing_q_score_accumulator is not None
                else None
            ),
            **{
                field: (
                    definition
                    if routing_q_score_accumulator is not None
                    else None
                )
                for field, definition in (
                    ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS.items()
                )
            },
            "evaluation_overrides": (
                resolved_evaluation if evaluation else None
            ),
            "packet_metric_definition": (
                {
                    "average_e2e_delay": (
                        "mean(GS arrival time - generation time) over packets "
                        "that reached GS, in seconds"
                    ),
                    "violation_probability": (
                        "canonical eligible deadline violations / "
                        "(generated FOV + every activated COM); missing when zero eligible"
                    ),
                    "sr_admission_drop": "excluded from numerator and denominator",
                    "terminal_outcomes_mutually_exclusive": True,
                }
                if evaluation
                else None
            ),
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
