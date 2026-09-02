"""Stable routing-transition causality with slot-immediate costs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION = (
    "global-id-causality-immediate-cost-v2"
)


class RoutingTransitionLedger:
    """Hold transitions until their next observation is causally available."""

    def __init__(self, next_transition_id=0):
        self.next_transition_id = int(next_transition_id)
        self.entries = {}

    def create(self, *, agent_id, state, action, tag_gt):
        transition_id = self.next_transition_id
        self.next_transition_id += 1
        self.entries[transition_id] = {
            "transition_id": transition_id,
            "agent_id": int(agent_id),
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": int(action),
            "tag_gt": int(tag_gt),
            "reward": None,
            "cost": None,
            "next_state": None,
            "done": None,
            "causality_pending": True,
        }
        return transition_id

    def set_reward(self, transition_id, reward):
        entry = self._entry(transition_id)
        if entry["reward"] is not None:
            raise AssertionError("routing transition reward was assigned twice")
        value = float(reward)
        if not np.isfinite(value):
            raise ValueError("routing transition reward must be finite")
        entry["reward"] = value

    def set_cost(self, transition_id, cost):
        entry = self._entry(transition_id)
        if entry["cost"] is not None:
            raise AssertionError("routing transition cost was assigned twice")
        value = float(cost)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("routing transition cost must be finite and non-negative")
        entry["cost"] = value

    def finalize_causality(self, states, start_of_slot_hol_by_sender, *, terminal=False):
        """Resolve every prior transition from the next real slot snapshot."""

        states = {int(key): value for key, value in dict(states).items()}
        active_senders = {
            int(key) for key in dict(start_of_slot_hol_by_sender)
        }
        finalized = 0
        for transition_id in sorted(self.entries):
            entry = self.entries[transition_id]
            if not entry["causality_pending"]:
                continue
            agent_id = entry["agent_id"]
            entry["next_state"] = np.asarray(
                states.get(agent_id, entry["state"]), dtype=np.float32
            ).copy()
            entry["done"] = bool(terminal or agent_id not in active_senders)
            entry["causality_pending"] = False
            finalized += 1
        return finalized

    def commit_ready(self, replay):
        """Move causally closed, fully specified transitions into replay."""

        committed = []
        for transition_id in sorted(tuple(self.entries)):
            entry = self.entries[transition_id]
            if (
                entry["causality_pending"]
                or entry["reward"] is None
                or entry["cost"] is None
            ):
                continue
            replay.add(
                entry["state"],
                entry["action"],
                entry["next_state"],
                entry["reward"],
                entry["cost"],
                entry["done"],
                tag_gt=entry["tag_gt"],
                agent_id=entry["agent_id"],
                transition_id=transition_id,
            )
            committed.append(transition_id)
            del self.entries[transition_id]
        return committed

    def state_dict(self):
        return {
            "schema_version": ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION,
            "next_transition_id": int(self.next_transition_id),
            "entries": deepcopy(self.entries),
        }

    def load_state_dict(self, state):
        validated = validate_routing_transition_ledger_state(state)
        self.next_transition_id = validated["next_transition_id"]
        self.entries = deepcopy(validated["entries"])

    def assert_drained(self):
        if self.entries:
            raise AssertionError("routing transition ledger remained at episode end")

    def _entry(self, transition_id):
        try:
            return self.entries[int(transition_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"unknown routing transition ID: {transition_id}") from exc


def validate_routing_transition_ledger_state(state):
    if not isinstance(state, dict):
        raise RuntimeError("routing transition ledger checkpoint state is missing")
    if state.get("schema_version") != ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION:
        raise RuntimeError("routing transition ledger checkpoint schema is incompatible")
    next_transition_id = state.get("next_transition_id")
    if isinstance(next_transition_id, bool) or not isinstance(next_transition_id, int):
        raise RuntimeError("routing transition next ID is invalid")
    entries = state.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError("routing transition ledger entries are invalid")
    normalized = {}
    for raw_id, raw_entry in entries.items():
        transition_id = int(raw_id)
        if transition_id < 0 or transition_id >= next_transition_id:
            raise RuntimeError("routing transition ledger ID is out of bounds")
        if not isinstance(raw_entry, dict):
            raise RuntimeError("routing transition ledger entry is invalid")
        entry = deepcopy(raw_entry)
        if int(entry.get("transition_id", -1)) != transition_id:
            raise RuntimeError("routing transition ledger entry ID is inconsistent")
        required = {
            "agent_id",
            "state",
            "action",
            "tag_gt",
            "reward",
            "cost",
            "next_state",
            "done",
            "causality_pending",
        }
        if set(entry) != required | {"transition_id"}:
            raise RuntimeError("routing transition ledger entry schema is incompatible")
        normalized[transition_id] = entry
    return {
        "schema_version": ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION,
        "next_transition_id": next_transition_id,
        "entries": normalized,
    }
