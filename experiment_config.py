"""Central configuration and method registry for formal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType

from Channel_model import (
    reference_s2u_max_capacity_mbps,
    reference_u2g_max_capacity_mbps,
    reference_u2u_max_capacity_mbps,
)


NUM_UAV = 10
ROI_COUNT_MIN = 2
ROI_COUNT_MAX = 8
RESERVED_SEARCH_UAV_IDS = (0, NUM_UAV - 1)
DEFAULT_TRAINING_SEED = 20260817
FORMAL_TRAINING_EPISODES = 1500
FORMAL_CHECKPOINT_EPISODE = FORMAL_TRAINING_EPISODES
SEARCH_COVERAGE_THRESHOLD = 0.99
FOV_COM_PAIR_MAX_DISTANCE_M = 200.0
TOTAL_COMMUNICATION_BANDWIDTH_HZ = 10e6
REFERENCE_COM_BANDWIDTH_HZ = TOTAL_COMMUNICATION_BANDWIDTH_HZ / (
    NUM_UAV + ROI_COUNT_MAX
)
COM_PACKET_RATE_PER_SECOND = 50.0
COM_PACKET_SIZE_BITS = 256.0
COM_OFFERED_RATE_BPS = COM_PACKET_RATE_PER_SECOND * COM_PACKET_SIZE_BITS
ASSIGNMENT_DUMMY_UTILITY = -1e-9
UTILITY_NORMALIZATION_MODE = "fov_global_minmax_com_fixed_theoretical_capacity-v2"
COM_UTILITY_CONTRACT_VERSION = "fixed-s2u-theoretical-maximum-v1"
TASK_COMPATIBILITY_POLICY = "fov_com_only_with_distance_limit"
FOV_ASSIGNMENT_UTILITY_VERSION = "coverage_times_reciprocal_image_quality-v1"
FOV_QUALITY_TRANSFORM = (
    "q(I)=0 for non-finite or I<=0; I for 0<I<=1; 1/I for I>1"
)
FOV_COVERAGE_SOURCE = (
    "centralized_movement.fov_task_metrics circle-ROI/rectangular-FOV "
    "intersection ratio [0,1]"
)
SAFE_DDQN_QOS_TARGET_PROBABILITY = 0.1
SAFE_DDQN_INITIAL_LAMBDA_COST = 0.0
SAFE_DDQN_ETA_C = 0.01
SAFE_DDQN_LAMBDA_UPDATE_SCOPE = "episode_end"
SAFE_DDQN_EVALUATION_LAMBDA_MODE = "checkpoint_frozen"
ROUTING_MASK_SCOPE = "every_slot"
MOVEMENT_INTERVAL_SECONDS = 1.0
EXPLORATION_SCHEDULE_VERSION = "linear_v2_1000ep"
EXPLORATION_SCHEDULE_TYPE = "linear"
MOVEMENT_EXPLORATION_DECAY_EPISODES = 1000
ROUTING_EPSILON_DECAY_EPISODES = 1000
ROUTING_EPSILON_START = 1.0
ROUTING_EPSILON_END = 0.05
ROUTING_WARMUP_TRANSITIONS = 1000
ROUTING_UPDATE_INTERVAL_SLOTS = 4
ROUTING_GRADIENT_STEPS_PER_UPDATE = 1
ROUTING_LEARNING_RATE = 1e-3
ROUTING_GAMMA = 0.99
ROUTING_TAU = 0.005
ROUTING_OPTIMIZER_UPDATE_SCOPE = "every_4_routing_slots"
FOV_EMA_LIFECYCLE_VERSION = "no-map-footprint-progression-v3"
SR_ROUTE_LIFECYCLE_VERSION = "assigned-and-arrived-derived-state-v2"
PACKET_QOS_CONTRACT_VERSION = "eligible-fov-plus-next-slot-s2u-admitted-com-v3"
PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION = "start-of-slot-hol-routing-v1"
QOS_AGGREGATE_CONTRACT_VERSION = "pooled-fov-com-eligible-v1"
ROUTING_REWARD_CONTRACT_VERSION = "capacity-minus-actual-hol-wait-v3"
ROUTING_REWARD_ALPHA_CAPACITY = 1.0
ROUTING_REWARD_ALPHA_DELAY = 0.5
ROUTING_CAPACITY_EPSILON_BPS = 1e-9
MOVEMENT_CHANNEL_TIMING_VERSION = "held-command-four-synchronous-substeps-v2"
PROPULSION_MODEL_ID = "canonical-3d-quadrotor-v1"
PROPULSION_PARAMETERS = MappingProxyType(
    {
        "n_r": 4,
        "rho": 1.293,
        "S_FP": 0.01,
        "g": 9.8,
        "m": 2.0,
        "delta": 0.012,
        "c_T": 0.302,
        "c_s": 0.0955,
        "c_f": 0.131,
        "A": 0.0314,
        "d_0": 0.834,
    }
)
PRODUCTION_TASK_DEADLINE_SECONDS = {"FOV": 1.5, "COM": 1.0}
PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS = 58.5

CURRENT_METHOD_ID = "td3_dinkelbach"

_COMMON_METHOD = {
    "assignment": "k_km",
    "assignment_rounds": 2,
    "routing": "safe_ddqn",
    "task_observation": "full",
    "task_potential_enabled": True,
}

_METHOD_DEFINITIONS = {
    "td3_dinkelbach": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "label": "TD3 + Dinkelbach",
    },
    "ddpg_dinkelbach": {
        **_COMMON_METHOD,
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "label": "DDPG + Dinkelbach",
    },
    "td3_ratio": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "ratio",
        "label": "TD3 + Direct ratio",
    },
    "ddpg_ratio": {
        **_COMMON_METHOD,
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "ratio",
        "label": "DDPG + Direct ratio",
    },
    "random_action": {
        **_COMMON_METHOD,
        "agent": "random",
        "movement": "random_action",
        "reward_mode": "ratio",
        "label": "Random selected",
    },
    "td3_dinkelbach_no_task_potential": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": False,
        "label": "TD3 + Dinkelbach without task potential",
    },
    "ddpg_dinkelbach_no_task_potential": {
        **_COMMON_METHOD,
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": False,
        "label": "DDPG + Dinkelbach without task potential",
    },
    "td3_dinkelbach_wo_ta": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "task_observation": "masked",
        "label": "TD3 + Dinkelbach without task-assignment observations",
    },
    "td3_dinkelbach_dqn": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "routing": "dqn",
        "label": "TD3 + Dinkelbach + controlled DQN routing",
    },
    "kkm_random_action_random_routing": {
        **_COMMON_METHOD,
        "agent": "random",
        "movement": "random_action",
        "reward_mode": "ratio",
        "routing": "random",
        "label": "K-KM + random movement + random routing",
    },
    "km_td3_dinkelbach": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "assignment": "km",
        "assignment_rounds": 1,
        "label": "KM + TD3 + Dinkelbach",
    },
    "random_assignment_td3_dinkelbach": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "assignment": "random_one_to_one",
        "assignment_rounds": 1,
        "label": "Random assignment + TD3 + Dinkelbach",
    },
    "km_ddpg_dinkelbach": {
        **_COMMON_METHOD,
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "assignment": "km",
        "assignment_rounds": 1,
        "label": "KM + DDPG + Dinkelbach",
    },
    "ddpg_dinkelbach_wo_ta": {
        **_COMMON_METHOD,
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "task_observation": "masked",
        "label": "DDPG + Dinkelbach without task-assignment observations",
    },
    "td3_dinkelbach_random_routing": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "routing": "random",
        "label": "TD3 + Dinkelbach + random routing",
    },
    "td3_dinkelbach_dqn_wo_ta": {
        **_COMMON_METHOD,
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "routing": "dqn",
        "task_observation": "masked",
        "label": "TD3 + Dinkelbach + controlled DQN without task-assignment observations",
    },
}
METHOD_REGISTRY = MappingProxyType(_METHOD_DEFINITIONS)
_LEGACY_METHOD_IDS = frozenset(tuple(_METHOD_DEFINITIONS)[:7])


@dataclass(frozen=True)
class MethodSpec:
    """Validated registry entry used by the shared experiment flow."""

    method_key: str = CURRENT_METHOD_ID
    assignment: str = "k_km"
    movement: str = "centralized_td3"
    routing: str = "safe_ddqn"
    task_observation: str = "full"
    assignment_rounds: int = 2
    lambda_mode: str = "dinkelbach"
    llm_enabled: bool = False
    agent: str = "td3"
    reward_mode: str = "dinkelbach"
    task_potential_enabled: bool = True
    label: str = "TD3 + Dinkelbach"

    def __post_init__(self):
        key = str(self.method_key).strip().lower()
        definition = METHOD_REGISTRY.get(key)
        if definition is None:
            raise ValueError(
                f"unsupported method {self.method_key!r}; choose one of "
                f"{', '.join(METHOD_REGISTRY)}"
            )
        expected = {
            "assignment": definition["assignment"],
            "movement": definition["movement"],
            "routing": definition["routing"],
            "task_observation": definition["task_observation"],
            "assignment_rounds": definition["assignment_rounds"],
            "lambda_mode": definition["reward_mode"],
            "llm_enabled": False,
            "agent": definition["agent"],
            "reward_mode": definition["reward_mode"],
            "task_potential_enabled": definition["task_potential_enabled"],
            "label": definition["label"],
        }
        actual = {
            field: getattr(self, field)
            for field in expected
        }
        if actual != expected:
            raise ValueError(
                "unsupported comparison method specification; "
                f"requested={actual}, registry={expected}"
            )
        object.__setattr__(self, "method_key", key)

    @property
    def method_id(self) -> str:
        return self.method_key

    @property
    def uses_dinkelbach(self) -> bool:
        return self.reward_mode == "dinkelbach"

    @property
    def learns_movement(self) -> bool:
        return self.agent in {"td3", "ddpg"}

    @property
    def learns_routing(self) -> bool:
        return self.routing in {"safe_ddqn", "dqn"}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def compatible_fingerprints(self) -> tuple[str, ...]:
        """Accept the pre-strategy fingerprint only for the original methods."""

        fingerprints = [self.fingerprint]
        if self.method_id in _LEGACY_METHOD_IDS:
            legacy = {
                "method_id": self.method_id,
                "method_key": self.method_id,
                "assignment": "current_k_km",
                "movement": self.movement,
                "routing": "safe_ddqn",
                "lambda_mode": self.reward_mode,
                "llm_enabled": False,
                "agent": self.agent,
                "reward_mode": self.reward_mode,
                "task_potential_enabled": self.task_potential_enabled,
                "label": self.label,
            }
            encoded = json.dumps(
                legacy, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            fingerprints.append(hashlib.sha256(encoded).hexdigest())
        return tuple(fingerprints)

    def to_dict(self) -> dict:
        return {"method_id": self.method_id, **asdict(self)}

    @classmethod
    def parse(cls, value: str) -> "MethodSpec":
        normalized = str(value).strip().lower()
        if normalized == "current":
            normalized = CURRENT_METHOD_ID
        definition = METHOD_REGISTRY.get(normalized)
        if definition is None:
            raise ValueError(
                f"unsupported method {value!r}; choose one of "
                f"{', '.join(METHOD_REGISTRY)}"
            )
        return cls(
            method_key=normalized,
            assignment=definition["assignment"],
            movement=definition["movement"],
            routing=definition["routing"],
            task_observation=definition["task_observation"],
            assignment_rounds=definition["assignment_rounds"],
            lambda_mode=definition["reward_mode"],
            agent=definition["agent"],
            reward_mode=definition["reward_mode"],
            task_potential_enabled=definition["task_potential_enabled"],
            label=definition["label"],
        )


FORMAL_EXPERIMENT_DEFAULTS = {
    "training_episodes_per_seed": FORMAL_TRAINING_EPISODES,
    "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
    "training_seed": DEFAULT_TRAINING_SEED,
    "training_seed_count": 1,
    "num_uav": NUM_UAV,
    "roi_count_min": ROI_COUNT_MIN,
    "roi_count_max": ROI_COUNT_MAX,
    "episode_seconds": 60,
    "routing_slot_seconds": 0.25,
    "evaluation_episodes_per_trained_seed": 100,
    "checkpoint_interval_episodes": 50,
    "output_root": "results",
    "movement_hyperparameters": {
        # Derived from the authoritative feature/action schemas.  These values
        # are asserted against centralized_movement at experiment preflight.
        "state_dim": NUM_UAV * 17 + 16 * 16 + 3,
        "joint_action_dim": NUM_UAV * 3,
        "max_action": 1.0,
        "hidden_layers": [256, 256, 256, 256],
        "actor_learning_rate": 6e-5,
        "critic_learning_rate": 2e-4,
        "batch_size": 64,
        "replay_size": 200_000,
        "warmup_joint_transitions": 1000,
        "gamma": 1.0,
        "tau": 0.005,
        "exploration_noise_start": 0.20,
        "exploration_noise_end": 0.05,
        "td3": {
            "policy_delay": 2,
            "target_policy_noise": 0.20,
            "target_noise_clip": 0.50,
            "twin_critics": True,
        },
        "ddpg": {
            "policy_delay": 1,
            "target_policy_noise": None,
            "target_noise_clip": None,
            "twin_critics": False,
        },
    },
}


def movement_agent_configuration(method_spec: MethodSpec, training_config=None) -> dict:
    """Return the algorithm settings that the selected controller really uses."""

    method_spec = MethodSpec.parse(method_spec.method_id)
    common = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]
    algorithm = common.get(method_spec.agent, {})
    resolved_training = (
        dict(training_config)
        if isinstance(training_config, dict)
        else asdict(training_config)
        if training_config is not None
        else {}
    )
    batch_size = int(resolved_training.get("batch_size", common["batch_size"]))
    replay_capacity = int(
        resolved_training.get("replay_max_size", common["replay_size"])
    )
    warmup = int(
        resolved_training.get(
            "warmup_joint_transitions", common["warmup_joint_transitions"]
        )
    )
    return {
        "movement_agent_kind": method_spec.agent,
        "movement_agent_gamma": common["gamma"],
        "tau": common["tau"] if method_spec.learns_movement else None,
        "actor_learning_rate": (
            common["actor_learning_rate"] if method_spec.learns_movement else None
        ),
        "critic_learning_rate": (
            common["critic_learning_rate"] if method_spec.learns_movement else None
        ),
        "hidden_layers": (
            list(common["hidden_layers"]) if method_spec.learns_movement else None
        ),
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "warmup_joint_transitions": warmup,
        "behavior_exploration_enabled": bool(method_spec.learns_movement),
        "exploration_schedule_version": (
            EXPLORATION_SCHEDULE_VERSION if method_spec.learns_movement else None
        ),
        "exploration_noise_start": (
            common["exploration_noise_start"] if method_spec.learns_movement else None
        ),
        "exploration_noise_end": (
            common["exploration_noise_end"] if method_spec.learns_movement else None
        ),
        "policy_delay": algorithm.get("policy_delay"),
        "target_policy_noise": algorithm.get("target_policy_noise"),
        "target_noise_clip": algorithm.get("target_noise_clip"),
        "twin_critics": bool(algorithm.get("twin_critics", False)),
        "optimizer_update_scope": "every_movement_transition_after_warmup",
    }


def exploration_schedule_configuration(config, method_spec: MethodSpec) -> dict:
    """Resolve fixed-episode exploration horizons from the shared time grid."""

    method_spec = MethodSpec.parse(method_spec.method_id)
    values = asdict(config) if not isinstance(config, dict) else dict(config)
    episode_seconds = float(values["episode_seconds"])
    routing_slot_seconds = float(values["routing_slot_seconds"])
    movement_decay_episodes = int(
        values.get(
            "movement_exploration_decay_episodes",
            MOVEMENT_EXPLORATION_DECAY_EPISODES,
        )
    )
    routing_decay_episodes = int(
        values.get(
            "routing_epsilon_decay_episodes", ROUTING_EPSILON_DECAY_EPISODES
        )
    )
    movement_steps_per_episode = int(
        round(episode_seconds / MOVEMENT_INTERVAL_SECONDS)
    )
    routing_slots_per_episode = int(round(episode_seconds / routing_slot_seconds))
    return {
        "exploration_schedule_version": EXPLORATION_SCHEDULE_VERSION,
        "exploration_schedule_type": EXPLORATION_SCHEDULE_TYPE,
        "movement_exploration_enabled": bool(method_spec.learns_movement),
        "routing_epsilon_enabled": bool(method_spec.learns_routing),
        "movement_exploration_decay_episodes": movement_decay_episodes,
        "routing_epsilon_decay_episodes": routing_decay_episodes,
        "movement_noise_start": (
            FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
                "exploration_noise_start"
            ]
            if method_spec.learns_movement
            else None
        ),
        "movement_noise_end": (
            FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
                "exploration_noise_end"
            ]
            if method_spec.learns_movement
            else None
        ),
        "routing_epsilon_start": (
            ROUTING_EPSILON_START if method_spec.learns_routing else None
        ),
        "routing_epsilon_end": (
            ROUTING_EPSILON_END if method_spec.learns_routing else None
        ),
        "resolved_movement_decay_steps": (
            movement_decay_episodes * movement_steps_per_episode
            if method_spec.learns_movement
            else None
        ),
        "resolved_routing_decay_slots": (
            routing_decay_episodes * routing_slots_per_episode
            if method_spec.learns_routing
            else None
        ),
        "movement_transitions_per_episode": movement_steps_per_episode,
        "routing_slots_per_episode": routing_slots_per_episode,
        "movement_interval_seconds": MOVEMENT_INTERVAL_SECONDS,
        "movement_warmup_transitions": int(
            values.get(
                "warmup_joint_transitions",
                FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
                    "warmup_joint_transitions"
                ],
            )
        ),
        "exploration_counter_scope": {
            "movement": "post_warmup_joint_movement_transitions",
            "routing": "global_routing_slots_after_replay_warmup",
        },
        "evaluation_exploration_mode": "disabled",
    }


def routing_agent_configuration(method_spec: MethodSpec, training_config=None) -> dict:
    """Return the resolved lifecycle and optimizer settings for routing."""

    method_spec = MethodSpec.parse(method_spec.method_id)
    resolved = (
        asdict(training_config)
        if training_config is not None and not isinstance(training_config, dict)
        else dict(training_config or {})
    )
    learned = method_spec.learns_routing
    return {
        "routing_agent_kind": method_spec.routing,
        "routing_learner_enabled": bool(learned),
        "routing_optimizer_update_scope": (
            ROUTING_OPTIMIZER_UPDATE_SCOPE if learned else None
        ),
        "routing_update_interval_slots": (
            int(
                resolved.get(
                    "routing_update_interval_slots", ROUTING_UPDATE_INTERVAL_SLOTS
                )
            )
            if learned
            else None
        ),
        "routing_gradient_steps_per_update": (
            int(
                resolved.get(
                    "routing_gradient_steps_per_update",
                    ROUTING_GRADIENT_STEPS_PER_UPDATE,
                )
            )
            if learned
            else None
        ),
        "routing_warmup_transitions": (
            int(
                resolved.get(
                    "routing_warmup_transitions", ROUTING_WARMUP_TRANSITIONS
                )
            )
            if learned
            else None
        ),
        "routing_warmup_counter_source": (
            "routing_replay_size" if learned else None
        ),
        "batch_size": (
            int(resolved.get("batch_size", 64)) if learned else None
        ),
        "replay_capacity": (
            int(resolved.get("replay_max_size", 200_000)) if learned else None
        ),
        "learning_rate": ROUTING_LEARNING_RATE if learned else None,
        "gamma": ROUTING_GAMMA,
        "tau": ROUTING_TAU if learned else None,
        "target_update_scope": "after_each_optimizer_event" if learned else None,
        "routing_mask_scope": ROUTING_MASK_SCOPE,
    }


def effective_training_config(config, method_spec: MethodSpec) -> dict:
    """Serialize formal config with method-specific movement settings resolved."""

    values = asdict(config) if not isinstance(config, dict) else dict(config)
    movement = movement_agent_configuration(method_spec, config)
    values["policy_delay"] = movement["policy_delay"]
    values["movement_agent_configuration"] = movement
    values["routing_agent_configuration"] = routing_agent_configuration(
        method_spec, config
    )
    values["exploration_schedule_configuration"] = (
        exploration_schedule_configuration(config, method_spec)
    )
    values.update(comparison_method_configuration(method_spec))
    return values


def comparison_method_configuration(method_spec: MethodSpec) -> dict:
    """Resolve orthogonal comparison strategies and shared assignment constants."""

    method_spec = MethodSpec.parse(method_spec.method_id)
    return {
        "assignment_strategy": method_spec.assignment,
        "assignment_rounds": int(method_spec.assignment_rounds),
        "movement_policy": method_spec.agent,
        "movement_objective": method_spec.reward_mode,
        "routing_policy": method_spec.routing,
        "task_observation_mode": method_spec.task_observation,
        "fov_com_pair_max_distance_m": FOV_COM_PAIR_MAX_DISTANCE_M,
        "reserved_search_uav_ids": list(RESERVED_SEARCH_UAV_IDS),
        "search_coverage_threshold": SEARCH_COVERAGE_THRESHOLD,
        "service_assignment_only": True,
        "utility_normalization_mode": UTILITY_NORMALIZATION_MODE,
        "com_utility_contract_version": COM_UTILITY_CONTRACT_VERSION,
        "reference_com_bandwidth_hz": REFERENCE_COM_BANDWIDTH_HZ,
        "reference_s2u_max_capacity_mbps": (
            reference_s2u_max_capacity_mbps(REFERENCE_COM_BANDWIDTH_HZ)
        ),
        "task_compatibility_policy": TASK_COMPATIBILITY_POLICY,
        "hover_assignment_candidate": False,
        "assignment_dummy_utility": ASSIGNMENT_DUMMY_UTILITY,
        "fov_assignment_utility_version": FOV_ASSIGNMENT_UTILITY_VERSION,
        "fov_quality_transform": FOV_QUALITY_TRANSFORM,
        "fov_coverage_source": FOV_COVERAGE_SOURCE,
        "safe_ddqn_qos_target_probability": (
            SAFE_DDQN_QOS_TARGET_PROBABILITY
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "safe_ddqn_initial_lambda_cost": (
            SAFE_DDQN_INITIAL_LAMBDA_COST
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "safe_ddqn_eta_c": (
            SAFE_DDQN_ETA_C if method_spec.routing == "safe_ddqn" else None
        ),
        "safe_ddqn_lambda_update_scope": (
            SAFE_DDQN_LAMBDA_UPDATE_SCOPE
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "safe_ddqn_evaluation_lambda_mode": (
            SAFE_DDQN_EVALUATION_LAMBDA_MODE
            if method_spec.routing == "safe_ddqn"
            else None
        ),
        "routing_mask_scope": ROUTING_MASK_SCOPE,
        "fov_ema_lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
        "sr_route_lifecycle_version": SR_ROUTE_LIFECYCLE_VERSION,
        "packet_qos_contract_version": PACKET_QOS_CONTRACT_VERSION,
        "packet_routing_causality_contract_version": (
            PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION
        ),
        "qos_aggregate_contract_version": QOS_AGGREGATE_CONTRACT_VERSION,
        "routing_reward_contract_version": ROUTING_REWARD_CONTRACT_VERSION,
        "routing_reward_alpha_capacity": ROUTING_REWARD_ALPHA_CAPACITY,
        "routing_reward_alpha_delay": ROUTING_REWARD_ALPHA_DELAY,
        "routing_capacity_epsilon_bps": ROUTING_CAPACITY_EPSILON_BPS,
        "reference_u2u_max_capacity_mbps": (
            reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
        ),
        "reference_u2g_max_capacity_mbps": (
            reference_u2g_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
        ),
        "propulsion_model_id": PROPULSION_MODEL_ID,
        "propulsion_parameters": dict(PROPULSION_PARAMETERS),
        "movement_channel_timing_version": MOVEMENT_CHANNEL_TIMING_VERSION,
        "movement_substeps_per_interval": 4,
        "movement_substep_seconds": 0.25,
        "resolved_fov_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["FOV"],
        "resolved_com_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["COM"],
        "packet_injection_cutoff_seconds": (
            PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS
        ),
    }
