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

CURRENT_METHOD_ID = "td3_dinkelbach"

_METHOD_DEFINITIONS = {
    "td3_dinkelbach": {
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": True,
        "label": "TD3 + Dinkelbach",
    },
    "ddpg_dinkelbach": {
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": True,
        "label": "DDPG + Dinkelbach",
    },
    "td3_ratio": {
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "ratio",
        "task_potential_enabled": True,
        "label": "TD3 + Direct ratio",
    },
    "ddpg_ratio": {
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "ratio",
        "task_potential_enabled": True,
        "label": "DDPG + Direct ratio",
    },
    "random_action": {
        "agent": "random",
        "movement": "random_action",
        "reward_mode": "ratio",
        "task_potential_enabled": True,
        "label": "Random selected",
    },
    "td3_dinkelbach_no_task_potential": {
        "agent": "td3",
        "movement": "centralized_td3",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": False,
        "label": "TD3 + Dinkelbach without task potential",
    },
    "ddpg_dinkelbach_no_task_potential": {
        "agent": "ddpg",
        "movement": "centralized_ddpg",
        "reward_mode": "dinkelbach",
        "task_potential_enabled": False,
        "label": "DDPG + Dinkelbach without task potential",
    },
}
METHOD_REGISTRY = MappingProxyType(_METHOD_DEFINITIONS)


@dataclass(frozen=True)
class MethodSpec:
    """Validated registry entry used by the shared experiment flow."""

    method_key: str = CURRENT_METHOD_ID
    assignment: str = "current_k_km"
    movement: str = "centralized_td3"
    routing: str = "safe_ddqn"
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
            "assignment": "current_k_km",
            "movement": definition["movement"],
            "routing": "safe_ddqn",
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
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

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
            movement=definition["movement"],
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


def movement_agent_configuration(method_spec: MethodSpec) -> dict:
    """Return the algorithm settings that the selected controller really uses."""

    method_spec = MethodSpec.parse(method_spec.method_id)
    common = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]
    algorithm = common.get(method_spec.agent, {})
    return {
        "movement_agent_kind": method_spec.agent,
        "movement_agent_gamma": common["gamma"],
        "tau": common["tau"] if method_spec.learns_movement else None,
        "policy_delay": algorithm.get("policy_delay"),
        "target_policy_noise": algorithm.get("target_policy_noise"),
        "target_noise_clip": algorithm.get("target_noise_clip"),
        "twin_critics": bool(algorithm.get("twin_critics", False)),
    }


def effective_training_config(config, method_spec: MethodSpec) -> dict:
    """Serialize formal config with method-specific movement settings resolved."""

    values = asdict(config) if not isinstance(config, dict) else dict(config)
    movement = movement_agent_configuration(method_spec)
    values["policy_delay"] = movement["policy_delay"]
    values["movement_agent_configuration"] = movement
    return values
