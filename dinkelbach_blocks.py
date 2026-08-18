"""Deterministic non-overlapping Dinkelbach outer-block state."""

from __future__ import annotations

from dataclasses import dataclass
import math


DINKELBACH_BLOCK_STATE_SCHEMA_VERSION = 1
DINKELBACH_INITIAL_LAMBDA = 0.0
DINKELBACH_UPDATE_INTERVAL_EPISODES = 50
DINKELBACH_UPDATE_RULE = "ratio_of_block_sums"
DINKELBACH_NUMERATOR_UNIT = "timely_delivered_mbits"
DINKELBACH_DENOMINATOR_UNIT = "mobility_energy_joules"

DINKELBACH_CONFIG_FIELDS = (
    "dinkelbach_initial_lambda",
    "dinkelbach_update_interval_episodes",
    "dinkelbach_update_rule",
    "dinkelbach_numerator_unit",
    "dinkelbach_denominator_unit",
)

DINKELBACH_TRAINING_STATE_FIELDS = (
    "dinkelbach_block_state_schema_version",
    "current_dinkelbach_lambda",
    "dinkelbach_initial_lambda",
    "dinkelbach_update_interval_episodes",
    "dinkelbach_block_index",
    "dinkelbach_block_completed_episodes",
    "dinkelbach_block_timely_delivered_mbits",
    "dinkelbach_block_mobility_energy_joules",
    "dinkelbach_update_count",
    "dinkelbach_block_inputs_valid",
    "dinkelbach_last_update_status",
)


def dinkelbach_config_metadata(config=None):
    if config is None:
        return {
            "dinkelbach_initial_lambda": DINKELBACH_INITIAL_LAMBDA,
            "dinkelbach_update_interval_episodes": (
                DINKELBACH_UPDATE_INTERVAL_EPISODES
            ),
            "dinkelbach_update_rule": DINKELBACH_UPDATE_RULE,
            "dinkelbach_numerator_unit": DINKELBACH_NUMERATOR_UNIT,
            "dinkelbach_denominator_unit": DINKELBACH_DENOMINATOR_UNIT,
        }
    return {
        field: getattr(config, field) if hasattr(config, field) else config[field]
        for field in DINKELBACH_CONFIG_FIELDS
    }


def validate_dinkelbach_config(config):
    values = dinkelbach_config_metadata(config)
    initial_lambda = float(values["dinkelbach_initial_lambda"])
    if not math.isfinite(initial_lambda):
        raise ValueError("dinkelbach_initial_lambda must be finite")
    interval = int(values["dinkelbach_update_interval_episodes"])
    if interval <= 0:
        raise ValueError("dinkelbach_update_interval_episodes must be positive")
    expected = dinkelbach_config_metadata()
    for field in (
        "dinkelbach_update_rule",
        "dinkelbach_numerator_unit",
        "dinkelbach_denominator_unit",
    ):
        if values[field] != expected[field]:
            raise ValueError(
                f"unsupported {field}: {values[field]!r}; expected={expected[field]!r}"
            )
    return values


def dinkelbach_full_block_count(total_episodes, interval_episodes):
    total_episodes = int(total_episodes)
    interval_episodes = int(interval_episodes)
    if total_episodes < 0 or interval_episodes <= 0:
        raise ValueError("episode count must be non-negative and interval positive")
    return total_episodes // interval_episodes


@dataclass
class DinkelbachBlockState:
    current_lambda: float = DINKELBACH_INITIAL_LAMBDA
    initial_lambda: float = DINKELBACH_INITIAL_LAMBDA
    update_interval_episodes: int = DINKELBACH_UPDATE_INTERVAL_EPISODES
    block_index: int = 1
    block_completed_episodes: int = 0
    block_timely_delivered_mbits: float = 0.0
    block_mobility_energy_joules: float = 0.0
    update_count: int = 0
    block_inputs_valid: bool = True
    last_update_status: str = "not_started"

    @classmethod
    def from_config(cls, config):
        values = validate_dinkelbach_config(config)
        initial_lambda = float(values["dinkelbach_initial_lambda"])
        return cls(
            current_lambda=initial_lambda,
            initial_lambda=initial_lambda,
            update_interval_episodes=int(
                values["dinkelbach_update_interval_episodes"]
            ),
        )

    @classmethod
    def from_training_state(
        cls,
        training_state,
        config,
        *,
        expected_completed_episodes=None,
    ):
        missing = set(DINKELBACH_TRAINING_STATE_FIELDS).difference(training_state)
        if missing:
            raise RuntimeError(
                "exact-resume checkpoint is missing Dinkelbach block state: "
                f"{sorted(missing)}"
            )
        if (
            training_state["dinkelbach_block_state_schema_version"]
            != DINKELBACH_BLOCK_STATE_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "exact-resume Dinkelbach block state schema is incompatible"
            )
        values = validate_dinkelbach_config(config)
        state = cls(
            current_lambda=float(training_state["current_dinkelbach_lambda"]),
            initial_lambda=float(training_state["dinkelbach_initial_lambda"]),
            update_interval_episodes=int(
                training_state["dinkelbach_update_interval_episodes"]
            ),
            block_index=int(training_state["dinkelbach_block_index"]),
            block_completed_episodes=int(
                training_state["dinkelbach_block_completed_episodes"]
            ),
            block_timely_delivered_mbits=float(
                training_state["dinkelbach_block_timely_delivered_mbits"]
            ),
            block_mobility_energy_joules=float(
                training_state["dinkelbach_block_mobility_energy_joules"]
            ),
            update_count=int(training_state["dinkelbach_update_count"]),
            block_inputs_valid=bool(
                training_state["dinkelbach_block_inputs_valid"]
            ),
            last_update_status=str(
                training_state["dinkelbach_last_update_status"]
            ),
        )
        if state.initial_lambda != float(values["dinkelbach_initial_lambda"]):
            raise RuntimeError(
                "exact-resume Dinkelbach initial lambda is incompatible"
            )
        if state.update_interval_episodes != int(
            values["dinkelbach_update_interval_episodes"]
        ):
            raise RuntimeError(
                "exact-resume Dinkelbach update interval is incompatible"
            )
        state.validate(expected_completed_episodes=expected_completed_episodes)
        return state

    def validate(self, *, expected_completed_episodes=None):
        finite_values = {
            "current_dinkelbach_lambda": self.current_lambda,
            "dinkelbach_initial_lambda": self.initial_lambda,
            "dinkelbach_block_timely_delivered_mbits": (
                self.block_timely_delivered_mbits
            ),
            "dinkelbach_block_mobility_energy_joules": (
                self.block_mobility_energy_joules
            ),
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"{name} must be finite")
        if self.update_interval_episodes <= 0:
            raise RuntimeError("Dinkelbach block interval must be positive")
        if self.block_index <= 0:
            raise RuntimeError("Dinkelbach block index must be positive")
        if not 0 <= self.block_completed_episodes < self.update_interval_episodes:
            raise RuntimeError("Dinkelbach partial block episode count is invalid")
        if not 0 <= self.update_count <= self.block_index - 1:
            raise RuntimeError("Dinkelbach update count is inconsistent")
        completed_episodes = (
            (self.block_index - 1) * self.update_interval_episodes
            + self.block_completed_episodes
        )
        if (
            expected_completed_episodes is not None
            and completed_episodes != int(expected_completed_episodes)
        ):
            raise RuntimeError(
                "Dinkelbach block state does not match checkpoint episode: "
                f"state={completed_episodes}, "
                f"checkpoint={int(expected_completed_episodes)}"
            )
        return self

    def training_state(self):
        self.validate()
        return {
            "dinkelbach_block_state_schema_version": (
                DINKELBACH_BLOCK_STATE_SCHEMA_VERSION
            ),
            "current_dinkelbach_lambda": float(self.current_lambda),
            "dinkelbach_initial_lambda": float(self.initial_lambda),
            "dinkelbach_update_interval_episodes": int(
                self.update_interval_episodes
            ),
            "dinkelbach_block_index": int(self.block_index),
            "dinkelbach_block_completed_episodes": int(
                self.block_completed_episodes
            ),
            "dinkelbach_block_timely_delivered_mbits": float(
                self.block_timely_delivered_mbits
            ),
            "dinkelbach_block_mobility_energy_joules": float(
                self.block_mobility_energy_joules
            ),
            "dinkelbach_update_count": int(self.update_count),
            "dinkelbach_block_inputs_valid": bool(self.block_inputs_valid),
            "dinkelbach_last_update_status": str(self.last_update_status),
        }

    def record_episode(self, timely_delivered_mbits, mobility_energy_joules):
        lambda_used = float(self.current_lambda)
        block_index = int(self.block_index)
        block_episode = int(self.block_completed_episodes) + 1
        timely = float(timely_delivered_mbits)
        energy = float(mobility_energy_joules)

        if math.isfinite(timely) and timely >= 0.0:
            candidate = self.block_timely_delivered_mbits + timely
            if math.isfinite(candidate):
                self.block_timely_delivered_mbits = candidate
            else:
                self.block_inputs_valid = False
        else:
            self.block_inputs_valid = False
        if math.isfinite(energy):
            candidate = self.block_mobility_energy_joules + energy
            if math.isfinite(candidate):
                self.block_mobility_energy_joules = candidate
            else:
                self.block_inputs_valid = False
            if energy < 0.0:
                self.block_inputs_valid = False
        else:
            self.block_inputs_valid = False

        self.block_completed_episodes = block_episode
        block_timely = float(self.block_timely_delivered_mbits)
        block_energy = float(self.block_mobility_energy_joules)
        updated = False
        completed = block_episode == self.update_interval_episodes
        status = (
            "accumulating"
            if self.block_inputs_valid
            else "accumulating_invalid_inputs"
        )
        if completed:
            if not self.block_inputs_valid:
                status = "invalid_block_inputs"
            elif block_energy <= 0.0:
                status = "invalid_denominator"
            else:
                ratio = block_timely / block_energy
                if math.isfinite(ratio):
                    self.current_lambda = float(ratio)
                    self.update_count += 1
                    updated = True
                    status = "updated"
                else:
                    status = "non_finite_ratio"
            self.last_update_status = status
            self.block_index += 1
            self.block_completed_episodes = 0
            self.block_timely_delivered_mbits = 0.0
            self.block_mobility_energy_joules = 0.0
            self.block_inputs_valid = True

        return {
            "dinkelbach_lambda_used": lambda_used,
            "dinkelbach_lambda_after_episode": float(self.current_lambda),
            "dinkelbach_lambda_updated": bool(updated),
            "dinkelbach_update_status": status,
            "dinkelbach_block_index": block_index,
            "dinkelbach_block_episode": block_episode,
            "dinkelbach_block_timely_mbits_so_far": block_timely,
            "dinkelbach_block_energy_joules_so_far": block_energy,
            "dinkelbach_block_completed": bool(completed),
        }
