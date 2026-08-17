"""Method and formal comparison configuration for the unified runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scenario_manifest import sha256_json


CURRENT_METHOD_ID = "current-k-km-centralized-td3-safe-ddqn-dinkelbach-no-llm"


@dataclass(frozen=True)
class MethodSpec:
    assignment: str = "current_k_km"
    movement: str = "centralized_td3"
    routing: str = "safe_ddqn"
    lambda_mode: str = "dinkelbach"
    llm_enabled: bool = False

    def __post_init__(self):
        expected = {
            "assignment": "current_k_km",
            "movement": "centralized_td3",
            "routing": "safe_ddqn",
            "lambda_mode": "dinkelbach",
            "llm_enabled": False,
        }
        actual = asdict(self)
        if actual != expected:
            raise ValueError(
                "unsupported comparison method specification; "
                f"requested={actual}, supported={expected}"
            )

    @property
    def method_id(self) -> str:
        return CURRENT_METHOD_ID

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict:
        return {"method_id": self.method_id, **asdict(self)}

    @classmethod
    def parse(cls, value: str) -> "MethodSpec":
        normalized = str(value).strip().lower()
        if normalized not in {"current", CURRENT_METHOD_ID}:
            raise ValueError(
                f"unsupported method {value!r}; only {CURRENT_METHOD_ID!r} "
                "is executable in this framework version"
            )
        return cls()


FORMAL_EXPERIMENT_DEFAULTS = {
    "training_episodes_per_seed": 1500,
    "training_seed_count": 5,
    "episode_seconds": 60,
    "routing_slot_seconds": 0.25,
    "evaluation_episodes_per_trained_seed": 100,
}
