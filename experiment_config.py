"""Central configuration and method registry for formal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType


NUM_UAV = 16
ROI_COUNT_MIN = 2
ROI_COUNT_MAX = 8
DEFAULT_TRAINING_SEED = 20260817
FORMAL_TRAINING_EPISODES = 2500
FORMAL_CHECKPOINT_EPISODE = FORMAL_TRAINING_EPISODES
SEARCH_UTILITY = 0.05
SEARCH_COVERAGE_THRESHOLD = 0.99
FOV_COM_PAIR_MAX_DISTANCE_M = 200.0
ASSIGNMENT_DUMMY_UTILITY = 0.0
UTILITY_NORMALIZATION_MODE = "per_task_type_global_minmax_feasible_only"
TASK_COMPATIBILITY_POLICY = "fov_com_only_with_distance_limit"
FOV_ASSIGNMENT_UTILITY_VERSION = "coverage_times_reciprocal_image_quality-v1"
FOV_QUALITY_TRANSFORM = (
    "q(I)=0 for non-finite or I<=0; I for 0<I<=1; 1/I for I>1"
)
FOV_COVERAGE_SOURCE = (
    "centralized_movement.fov_task_metrics circle-ROI/rectangular-FOV "
    "intersection ratio [0,1]"
)
SAFE_DDQN_QOS_COST_BUDGET = 12.0
SAFE_DDQN_INITIAL_LAMBDA_COST = 0.0
SAFE_DDQN_ETA_C = 0.01
SAFE_DDQN_LAMBDA_UPDATE_SCOPE = "episode_end"
SAFE_DDQN_EVALUATION_LAMBDA_MODE = "checkpoint_frozen"
ROUTING_MASK_SCOPE = "every_slot"
FOV_EMA_LIFECYCLE_VERSION = "no-map-footprint-progression-v3"
SR_ROUTE_LIFECYCLE_VERSION = "no-duplicate-start-exact-endpoint-v1"
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
        "state_dim": 532,
        "joint_action_dim": 48,
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
        "exploration_noise_start": common["exploration_noise_start"],
        "exploration_noise_end": common["exploration_noise_end"],
        "policy_delay": algorithm.get("policy_delay"),
        "target_policy_noise": algorithm.get("target_policy_noise"),
        "target_noise_clip": algorithm.get("target_noise_clip"),
        "twin_critics": bool(algorithm.get("twin_critics", False)),
        "optimizer_update_scope": "every_movement_transition_after_warmup",
    }


def effective_training_config(config, method_spec: MethodSpec) -> dict:
    """Serialize formal config with method-specific movement settings resolved."""

    values = asdict(config) if not isinstance(config, dict) else dict(config)
    movement = movement_agent_configuration(method_spec, config)
    values["policy_delay"] = movement["policy_delay"]
    values["movement_agent_configuration"] = movement
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
        "search_utility": SEARCH_UTILITY,
        "search_coverage_threshold": SEARCH_COVERAGE_THRESHOLD,
        "utility_normalization_mode": UTILITY_NORMALIZATION_MODE,
        "task_compatibility_policy": TASK_COMPATIBILITY_POLICY,
        "hover_assignment_candidate": False,
        "assignment_dummy_utility": ASSIGNMENT_DUMMY_UTILITY,
        "fov_assignment_utility_version": FOV_ASSIGNMENT_UTILITY_VERSION,
        "fov_quality_transform": FOV_QUALITY_TRANSFORM,
        "fov_coverage_source": FOV_COVERAGE_SOURCE,
        "safe_ddqn_qos_cost_budget": (
            SAFE_DDQN_QOS_COST_BUDGET
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
        "resolved_fov_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["FOV"],
        "resolved_com_deadline_seconds": PRODUCTION_TASK_DEADLINE_SECONDS["COM"],
        "packet_injection_cutoff_seconds": (
            PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS
        ),
    }
