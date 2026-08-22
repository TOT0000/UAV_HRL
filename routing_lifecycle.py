"""Shared routing learner cadence and replay-gated epsilon lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from exploration_schedules import ddqn_epsilon
from experiment_config import (
    ROUTING_GRADIENT_STEPS_PER_UPDATE,
    ROUTING_UPDATE_INTERVAL_SLOTS,
    ROUTING_WARMUP_TRANSITIONS,
)


@dataclass
class RoutingLearnerLifecycle:
    update_interval_slots: int = ROUTING_UPDATE_INTERVAL_SLOTS
    gradient_steps_per_update: int = ROUTING_GRADIENT_STEPS_PER_UPDATE
    warmup_transitions: int = ROUTING_WARMUP_TRANSITIONS
    global_slot_count: int = 0
    optimizer_update_count: int = 0
    target_update_count: int = 0
    epsilon_decay_start_slot: int | None = None
    last_optimizer_update_slot: int | None = None

    def __post_init__(self):
        if (
            int(self.update_interval_slots) <= 0
            or int(self.gradient_steps_per_update) <= 0
            or int(self.warmup_transitions) <= 0
        ):
            raise ValueError("routing lifecycle settings must be positive")
        for value in (
            self.global_slot_count,
            self.optimizer_update_count,
            self.target_update_count,
        ):
            if int(value) < 0:
                raise ValueError("routing lifecycle counters must be non-negative")
        if self.epsilon_decay_start_slot is not None and not (
            0 < int(self.epsilon_decay_start_slot) <= int(self.global_slot_count)
        ):
            raise ValueError("routing epsilon marker is outside completed slots")
        if (self.optimizer_update_count == 0) != (
            self.last_optimizer_update_slot is None
        ):
            raise ValueError("routing last-update marker disagrees with update count")
        if self.last_optimizer_update_slot is not None and (
            not 0 < int(self.last_optimizer_update_slot) <= int(self.global_slot_count)
            or int(self.last_optimizer_update_slot)
            % int(self.update_interval_slots)
            != 0
        ):
            raise ValueError("routing last-update marker is not a cadence boundary")

    @property
    def update_phase(self):
        return int(self.global_slot_count) % int(self.update_interval_slots)

    @property
    def warmup_complete(self):
        return self.epsilon_decay_start_slot is not None

    @property
    def slots_since_last_update(self):
        baseline = self.last_optimizer_update_slot or 0
        return int(self.global_slot_count) - int(baseline)

    def epsilon(self, decay_slots):
        if self.epsilon_decay_start_slot is None:
            return 1.0
        post_warmup_slot = max(
            int(self.global_slot_count) - int(self.epsilon_decay_start_slot), 0
        )
        return ddqn_epsilon(post_warmup_slot, decay_slots)

    def complete_slot(self, agent, replay, batch_size):
        """Complete one training slot and run at most one optimizer event."""

        self.global_slot_count += 1
        if (
            self.epsilon_decay_start_slot is None
            and int(replay.size) >= int(self.warmup_transitions)
        ):
            self.epsilon_decay_start_slot = int(self.global_slot_count)

        cadence_boundary = self.update_phase == 0
        if not cadence_boundary or int(replay.size) < int(self.warmup_transitions):
            return False

        before_training = int(agent.num_training)
        before_target = int(agent.target_update_count)
        before_reward = int(getattr(agent, "reward_optimizer_update_count", 0))
        before_cost = int(getattr(agent, "cost_optimizer_update_count", 0))
        for _ in range(int(self.gradient_steps_per_update)):
            agent.train(replay, batch_size)
        agent.update_target()
        expected_steps = int(self.gradient_steps_per_update)
        if (
            int(agent.num_training) - before_training != expected_steps
            or int(agent.target_update_count) - before_target != 1
            or int(getattr(agent, "reward_optimizer_update_count", 0))
            - before_reward
            != expected_steps
        ):
            raise AssertionError("routing optimizer/target event counts diverged")
        if (
            getattr(agent, "routing_agent_kind", None) == "safe_ddqn"
            and int(getattr(agent, "cost_optimizer_update_count", 0))
            - before_cost
            != expected_steps
        ):
            raise AssertionError("safe-DDQN reward/cost update counts diverged")
        self.optimizer_update_count += 1
        self.target_update_count += 1
        self.last_optimizer_update_slot = int(self.global_slot_count)
        return True

    def state_dict(self):
        update_scope = f"every_{int(self.update_interval_slots)}_routing_slots"
        return {
            "routing_optimizer_update_scope": update_scope,
            "routing_update_interval_slots": int(self.update_interval_slots),
            "routing_gradient_steps_per_update": int(
                self.gradient_steps_per_update
            ),
            "routing_warmup_transitions": int(self.warmup_transitions),
            "routing_global_slot_count": int(self.global_slot_count),
            "routing_update_phase": int(self.update_phase),
            "routing_optimizer_update_count": int(self.optimizer_update_count),
            "routing_target_update_count": int(self.target_update_count),
            "routing_slots_since_last_update": int(self.slots_since_last_update),
            "routing_warmup_complete": bool(self.warmup_complete),
            "routing_epsilon_decay_start_slot": self.epsilon_decay_start_slot,
            "routing_last_optimizer_update_slot": self.last_optimizer_update_slot,
        }

    @classmethod
    def from_state(
        cls,
        state,
        *,
        update_interval_slots=ROUTING_UPDATE_INTERVAL_SLOTS,
        gradient_steps_per_update=ROUTING_GRADIENT_STEPS_PER_UPDATE,
        warmup_transitions=ROUTING_WARMUP_TRANSITIONS,
    ):
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint routing lifecycle state is missing")
        required = {
            "routing_optimizer_update_scope",
            "routing_update_interval_slots",
            "routing_gradient_steps_per_update",
            "routing_warmup_transitions",
            "routing_global_slot_count",
            "routing_update_phase",
            "routing_optimizer_update_count",
            "routing_target_update_count",
            "routing_slots_since_last_update",
            "routing_warmup_complete",
            "routing_epsilon_decay_start_slot",
            "routing_last_optimizer_update_slot",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise RuntimeError(
                f"checkpoint routing lifecycle state is incomplete: {missing}"
            )
        expected = {
            "routing_optimizer_update_scope": (
                f"every_{int(update_interval_slots)}_routing_slots"
            ),
            "routing_update_interval_slots": int(update_interval_slots),
            "routing_gradient_steps_per_update": int(
                gradient_steps_per_update
            ),
            "routing_warmup_transitions": int(warmup_transitions),
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"checkpoint routing lifecycle configuration is incompatible: {mismatches}"
            )
        lifecycle = cls(
            update_interval_slots=update_interval_slots,
            gradient_steps_per_update=gradient_steps_per_update,
            warmup_transitions=warmup_transitions,
            global_slot_count=int(state["routing_global_slot_count"]),
            optimizer_update_count=int(state["routing_optimizer_update_count"]),
            target_update_count=int(state["routing_target_update_count"]),
            epsilon_decay_start_slot=state["routing_epsilon_decay_start_slot"],
            last_optimizer_update_slot=state[
                "routing_last_optimizer_update_slot"
            ],
        )
        computed = lifecycle.state_dict()
        for key in (
            "routing_update_phase",
            "routing_slots_since_last_update",
            "routing_warmup_complete",
        ):
            if computed[key] != state[key]:
                raise RuntimeError(
                    f"checkpoint routing lifecycle counter is inconsistent: {key}"
                )
        if lifecycle.optimizer_update_count != lifecycle.target_update_count:
            raise RuntimeError(
                "checkpoint routing optimizer/target update counts diverge"
            )
        return lifecycle
