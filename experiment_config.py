"""Central configuration and method registry for formal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType

from communication_contract import (
    COMMUNICATION_RANGE_BOUNDARY_RULE,
    MAX_3D_COMMUNICATION_DISTANCE_M,
    validate_communication_range_aliases,
)
from evaluation_aggregation import EVALUATION_AGGREGATION_SCHEMA_VERSION

from Channel_model import (
    CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
    CHANNEL_FAIRNESS_CONTRACT_VERSION,
    CHANNEL_MODEL_VERSION,
    CHANNEL_NORMALIZATION_VERSION,
    FADING_BLOCK_SECONDS,
    FADING_BLOCKS_PER_ROUTING_SLOT,
    LARGE_SCALE_STATE_SECONDS,
    RICIAN_K_DB,
    RICIAN_K_LINEAR,
    ROUTING_SLOT_SECONDS,
    channel_configuration_metadata,
    reference_s2u_max_capacity_mbps,
    reference_u2g_max_capacity_mbps,
    reference_u2u_max_capacity_mbps,
)


NUM_UAV = 10
ROI_COUNT_MIN = 2
ROI_COUNT_MAX = 8
RESERVED_SEARCH_UAV_IDS = (0, NUM_UAV - 1)
GROUND_STATION_POSITION_M = (0.0, 0.0, 0.0)
PERMANENT_GS_GATEWAY_UAV_ID = 0
GS_GATEWAY_SOFT_RADIUS_M = 360.0
GS_GATEWAY_HARD_RADIUS_M = MAX_3D_COMMUNICATION_DISTANCE_M
GS_GATEWAY_PROJECTION_MODE = "gs_only"
GS_GATEWAY_CONTRACT_VERSION = (
    "permanent-uav0-search-to-hover-altitude-feasible-3d-soft360-hard400-v2"
)
CANONICAL_UAV_INITIAL_XY_M = (
    (50.0, 50.0),
    (300.0, 250.0),
    (500.0, 250.0),
    (700.0, 250.0),
    (900.0, 250.0),
    (100.0, 750.0),
    (300.0, 750.0),
    (500.0, 750.0),
    (700.0, 750.0),
    (900.0, 750.0),
)
UAV_INITIAL_LAYOUT_VERSION = "gs-reachable-gateway-grid-v2"
INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION = (
    "finite-3d-inclusive-unified-400m-gs-component-min-two-uavs-v2"
)
ENVIRONMENT_WIDTH_M = 1000.0
ENVIRONMENT_HEIGHT_M = 1000.0
GROUND_ALTITUDE_M = 0.0
UAV_MAX_ALTITUDE_M = 150.0
TASK_POTENTIAL_NORMALIZATION_EPSILON = 1e-12
TASK_POTENTIAL_CONTRACT_VERSION = (
    "vs-horizontal-proximity-com-400m-range-gap-blend-v2"
)
VS_SENSING_POTENTIAL_WEIGHT = 0.5
VS_DISTANCE_POTENTIAL_WEIGHT = 0.5
COM_CAPACITY_POTENTIAL_WEIGHT = 0.5
COM_DISTANCE_POTENTIAL_WEIGHT = 0.5
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
COMMUNICATION_RANGE_M = MAX_3D_COMMUNICATION_DISTANCE_M
A2G_COMMUNICATION_RANGE_M = COMMUNICATION_RANGE_M
A2A_COMMUNICATION_RANGE_M = COMMUNICATION_RANGE_M
S2U_COMMUNICATION_RANGE_M = COMMUNICATION_RANGE_M
U2G_COMMUNICATION_RANGE_M = COMMUNICATION_RANGE_M
U2U_COMMUNICATION_RANGE_M = COMMUNICATION_RANGE_M
COMMUNICATION_RANGE_CONTRACT_VERSION = (
    "slot-start-3d-inclusive-s2u-u2g-u2u-400m-v2"
)
COM_SESSION_LIFECYCLE_VERSION = (
    "400m-activated-generation-immediate-e2e-qos-persistent-v3"
)
validate_communication_range_aliases(
    A2G_COMMUNICATION_RANGE_M,
    A2A_COMMUNICATION_RANGE_M,
    S2U_COMMUNICATION_RANGE_M,
    U2G_COMMUNICATION_RANGE_M,
    U2U_COMMUNICATION_RANGE_M,
)
ASSIGNMENT_DUMMY_UTILITY = -1e-9
UTILITY_NORMALIZATION_MODE = "fov-global-minmax-com-fading-aware-reference-v3"
COM_UTILITY_CONTRACT_VERSION = "fixed-s2u-los-rician-expected-maximum-v2"
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
MOVEMENT_WARMUP_TRANSITIONS = 10_000
MOVEMENT_REPLAY_CAPACITY = 50_000
TD3_TARGET_POLICY_NOISE = 0.10
TD3_TARGET_NOISE_CLIP = 0.25
EXPLORATION_SCHEDULE_VERSION = "linear_v2_1000ep"
EXPLORATION_SCHEDULE_TYPE = "linear"
MOVEMENT_EXPLORATION_DECAY_EPISODES = 1000
ROUTING_EPSILON_DECAY_EPISODES = 1000
ROUTING_EPSILON_START = 1.0
ROUTING_EPSILON_END = 0.05
ROUTING_WARMUP_TRANSITIONS = 1000
ROUTING_REPLAY_CAPACITY = 200_000
ROUTING_UPDATE_INTERVAL_SLOTS = 4
ROUTING_GRADIENT_STEPS_PER_UPDATE = 1
ROUTING_LEARNING_RATE = 1e-3
ROUTING_GAMMA = 0.99
ROUTING_TAU = 0.005
ROUTING_OPTIMIZER_UPDATE_SCOPE = "every_4_routing_slots"
FOV_EMA_LIFECYCLE_VERSION = "all-participant-precommit-search-union-v5"
SR_ROUTE_LIFECYCLE_VERSION = "assigned-and-arrived-derived-state-v2"
PACKET_QOS_CONTRACT_VERSION = (
    "assigned-fov-and-activated-com-immediate-qos-v7"
)
FOV_PACKET_GENERATION_CONTRACT_VERSION = (
    "assigned-source-rate-integrator-capture-coverage-snapshot-v2"
)
TIMELY_USEFUL_GOODPUT_CONTRACT_VERSION = (
    "fov-capture-coverage-weighted-com-full-timely-bits-v1"
)
PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION = (
    "transition-id-causality-credit-pending-one-step-v3"
)
ROUTING_COST_ATTRIBUTION_CONTRACT_VERSION = (
    "system-qos-versus-routing-credit-stable-id-v2"
)
PACKET_SERVICE_CONTRACT_VERSION = (
    "unified-400m-slot-start-fifty-5ms-block-cumulative-service-v2"
)
QOS_AGGREGATE_CONTRACT_VERSION = (
    "canonical-single-result-seed-ratio-of-sums-student-t-v3"
)
ROUTING_REWARD_CONTRACT_VERSION = (
    "unified-400m-slot-start-other-backlog-over-fading-effective-capacity-v6"
)
ROUTING_REWARD_ALPHA_CAPACITY = 1.0
ROUTING_REWARD_ALPHA_DELAY = 0.5
MOVEMENT_CHANNEL_TIMING_VERSION = (
    "boundary-prepared-held-command-four-slots-fifty-fading-blocks-v4"
)
PROPULSION_MODEL_ID = "canonical-3d-quadrotor-v1"
MOVEMENT_ACTION_PROJECTION_CONTRACT_VERSION = (
    "fieldwise-clamp-heading-wrap-mask-uav0-altitude-feasible-gs-3d-position-v3"
)
MOVEMENT_REPLAY_CONTRACT_VERSION = (
    "executed-net-displacement-action-boundary-aligned-next-state-capacity-50000-v3"
)
MOVEMENT_WARMUP_CONTRACT_VERSION = "global-joint-transition-boundary-10000-v1"
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


def validate_task_potential_weights():
    """Fail fast unless both authoritative blend pairs are convex weights."""

    groups = {
        "VS": (VS_SENSING_POTENTIAL_WEIGHT, VS_DISTANCE_POTENTIAL_WEIGHT),
        "COM": (COM_CAPACITY_POTENTIAL_WEIGHT, COM_DISTANCE_POTENTIAL_WEIGHT),
    }
    for name, weights in groups.items():
        numeric = tuple(float(weight) for weight in weights)
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in numeric):
            raise ValueError(
                f"{name} task-potential weights must be finite and non-negative"
            )
        if not math.isclose(sum(numeric), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} task-potential weights must sum to 1")
    return groups


def task_potential_contract_metadata():
    """Return the shared reward/config/checkpoint task-potential contract."""

    validate_task_potential_weights()
    horizontal_reference = math.hypot(
        ENVIRONMENT_WIDTH_M, ENVIRONMENT_HEIGHT_M
    )
    distance_3d_reference = math.sqrt(
        ENVIRONMENT_WIDTH_M**2
        + ENVIRONMENT_HEIGHT_M**2
        + (UAV_MAX_ALTITUDE_M - GROUND_ALTITUDE_M) ** 2
    )
    range_gap_reference = max(
        distance_3d_reference - S2U_COMMUNICATION_RANGE_M,
        TASK_POTENTIAL_NORMALIZATION_EPSILON,
    )
    return {
        "contract_version": TASK_POTENTIAL_CONTRACT_VERSION,
        "search": {
            "unchanged": True,
            "definition": "mean(visited_bitmap)",
            "target_distance_used": False,
        },
        "vs": {
            "sensing_weight": VS_SENSING_POTENTIAL_WEIGHT,
            "distance_weight": VS_DISTANCE_POTENTIAL_WEIGHT,
            "distance_dimensionality": "horizontal_2d",
            "normalization": "environment_xy_diagonal",
            "environment_width_m": ENVIRONMENT_WIDTH_M,
            "environment_height_m": ENVIRONMENT_HEIGHT_M,
            "distance_normalization_reference_m": horizontal_reference,
        },
        "com": {
            "capacity_weight": COM_CAPACITY_POTENTIAL_WEIGHT,
            "distance_weight": COM_DISTANCE_POTENTIAL_WEIGHT,
            "distance_dimensionality": "three_dimensional_3d",
            "s2u_range_m": S2U_COMMUNICATION_RANGE_M,
            "normalization": "positive_range_gap_over_maximum_environment_gap",
            "environment_width_m": ENVIRONMENT_WIDTH_M,
            "environment_height_m": ENVIRONMENT_HEIGHT_M,
            "uav_max_altitude_m": UAV_MAX_ALTITUDE_M,
            "ground_altitude_m": GROUND_ALTITUDE_M,
            "maximum_3d_distance_reference_m": distance_3d_reference,
            "range_gap_normalization_reference_m": range_gap_reference,
            "normalization_epsilon": TASK_POTENTIAL_NORMALIZATION_EPSILON,
        },
        "lifecycle": {
            "form": "beta * (gamma * phi_next - phi_current)",
            "delivery_or_connectivity_potential": False,
        },
    }


validate_task_potential_weights()

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
    "routing_slot_seconds": ROUTING_SLOT_SECONDS,
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
        "replay_size": MOVEMENT_REPLAY_CAPACITY,
        "warmup_joint_transitions": MOVEMENT_WARMUP_TRANSITIONS,
        "gamma": 1.0,
        "tau": 0.005,
        "exploration_noise_start": 0.20,
        "exploration_noise_end": 0.05,
        "td3": {
            "policy_delay": 2,
            "target_policy_noise": TD3_TARGET_POLICY_NOISE,
            "target_noise_clip": TD3_TARGET_NOISE_CLIP,
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
    learned = bool(method_spec.learns_movement)
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
        "batch_size": batch_size if learned else None,
        "replay_capacity": replay_capacity if learned else None,
        "warmup_joint_transitions": warmup if learned else None,
        "warmup_counter_source": "global_joint_transition_count" if learned else None,
        "exploration_decay_origin": "first_post_warmup_transition" if learned else None,
        "behavior_exploration_enabled": learned,
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
        "optimizer_update_scope": (
            "every_movement_transition_after_warmup" if learned else None
        ),
        "action_projection_contract_version": (
            MOVEMENT_ACTION_PROJECTION_CONTRACT_VERSION
        ),
        "heading_projection": "periodic-wrap-to-[-1,1)",
        "replay_action_semantics": (
            "inverse-encoded-executed-net-displacement" if learned else None
        ),
        "replay_contract_version": (
            MOVEMENT_REPLAY_CONTRACT_VERSION if learned else None
        ),
        "warmup_contract_version": (
            MOVEMENT_WARMUP_CONTRACT_VERSION if learned else None
        ),
        "capabilities": {
            "learned_off_policy": learned,
            "replay": learned,
            "warmup": learned,
            "target_policy_smoothing": method_spec.agent == "td3",
        },
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
            ROUTING_REPLAY_CAPACITY if learned else None
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
        "task_potential_enabled": bool(method_spec.task_potential_enabled),
        "task_potential_contract_version": TASK_POTENTIAL_CONTRACT_VERSION,
        "task_potential_configuration": task_potential_contract_metadata(),
        "ground_station_position_m": list(GROUND_STATION_POSITION_M),
        "permanent_gs_gateway_uav_id": PERMANENT_GS_GATEWAY_UAV_ID,
        "gs_gateway_soft_radius_m": GS_GATEWAY_SOFT_RADIUS_M,
        "gs_gateway_hard_radius_m": GS_GATEWAY_HARD_RADIUS_M,
        "gs_gateway_projection_mode": GS_GATEWAY_PROJECTION_MODE,
        "gs_gateway_contract_version": GS_GATEWAY_CONTRACT_VERSION,
        "uav_initial_layout_version": UAV_INITIAL_LAYOUT_VERSION,
        "initial_communication_topology_contract_version": (
            INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION
        ),
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
        "fov_packet_generation_contract_version": (
            FOV_PACKET_GENERATION_CONTRACT_VERSION
        ),
        "timely_useful_goodput_contract_version": (
            TIMELY_USEFUL_GOODPUT_CONTRACT_VERSION
        ),
        "timely_goodput_definition": "total timely useful bits",
        "fov_coverage_snapshot_timing": "packet generation/capture time",
        "fov_physical_packet_bits_coverage_weighted": False,
        "com_session_lifecycle_version": COM_SESSION_LIFECYCLE_VERSION,
        "communication_range_contract_version": (
            COMMUNICATION_RANGE_CONTRACT_VERSION
        ),
        "communication_range_boundary_rule": COMMUNICATION_RANGE_BOUNDARY_RULE,
        "maximum_3d_communication_distance_m": (
            MAX_3D_COMMUNICATION_DISTANCE_M
        ),
        "s2u_communication_range_m": S2U_COMMUNICATION_RANGE_M,
        "u2g_communication_range_m": U2G_COMMUNICATION_RANGE_M,
        "u2u_communication_range_m": U2U_COMMUNICATION_RANGE_M,
        "a2g_communication_range_m": A2G_COMMUNICATION_RANGE_M,
        "a2a_communication_range_m": A2A_COMMUNICATION_RANGE_M,
        "packet_routing_causality_contract_version": (
            PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION
        ),
        "routing_cost_attribution_contract_version": (
            ROUTING_COST_ATTRIBUTION_CONTRACT_VERSION
        ),
        "packet_service_contract_version": PACKET_SERVICE_CONTRACT_VERSION,
        "qos_aggregate_contract_version": QOS_AGGREGATE_CONTRACT_VERSION,
        "evaluation_aggregation_schema_version": (
            EVALUATION_AGGREGATION_SCHEMA_VERSION
        ),
        "routing_reward_contract_version": ROUTING_REWARD_CONTRACT_VERSION,
        "routing_reward_alpha_capacity": ROUTING_REWARD_ALPHA_CAPACITY,
        "routing_reward_alpha_delay": ROUTING_REWARD_ALPHA_DELAY,
        "reference_u2u_max_capacity_mbps": (
            reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
        ),
        "reference_u2g_max_capacity_mbps": (
            reference_u2g_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
        ),
        "propulsion_model_id": PROPULSION_MODEL_ID,
        "propulsion_parameters": dict(PROPULSION_PARAMETERS),
        "movement_channel_timing_version": MOVEMENT_CHANNEL_TIMING_VERSION,
        "movement_replay_contract_version": MOVEMENT_REPLAY_CONTRACT_VERSION,
        "movement_substeps_per_interval": 4,
        "movement_substep_seconds": ROUTING_SLOT_SECONDS,
        "channel_model_version": CHANNEL_MODEL_VERSION,
        "channel_environment_contract_version": (
            CHANNEL_ENVIRONMENT_CONTRACT_VERSION
        ),
        "channel_fairness_contract_version": CHANNEL_FAIRNESS_CONTRACT_VERSION,
        "channel_normalization_version": CHANNEL_NORMALIZATION_VERSION,
        "channel_configuration": channel_configuration_metadata(),
        "large_scale_state_seconds": LARGE_SCALE_STATE_SECONDS,
        "fading_block_seconds": FADING_BLOCK_SECONDS,
        "fading_blocks_per_routing_slot": FADING_BLOCKS_PER_ROUTING_SLOT,
        "rician_k_linear": RICIAN_K_LINEAR,
        "rician_k_db": RICIAN_K_DB,
        "resolved_fov_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["FOV"],
        "resolved_com_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["COM"],
        "packet_injection_cutoff_seconds": (
            PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS
        ),
    }
