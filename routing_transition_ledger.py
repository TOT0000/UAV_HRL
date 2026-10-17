"""Independent reward causality and packet-path cost causality for routing."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION = (
    "all-uav-reward-multi-packet-path-cost-v4"
)


class RoutingTransitionLedger:
    """Hold sender-time reward transitions and packet-time cost links."""

    def __init__(self, next_transition_id=0):
        self.next_transition_id = int(next_transition_id)
        self.entries = {}
        self.decision_snapshots = {}
        self.packet_pending = {}
        self.cost_closures = []
        self.terminal_packet_ids = set()
        self.episode_decision_count = 0
        self.episode_cost_transition_count = 0
        self.episode_terminal_cost_sum = 0.0
        self.episode_terminal_without_decision_count = 0

    def begin_episode(self):
        if self.entries or self.packet_pending or self.cost_closures:
            raise AssertionError("routing transition ledger was not drained")
        self.decision_snapshots.clear()
        self.terminal_packet_ids.clear()
        self.episode_decision_count = 0
        self.episode_cost_transition_count = 0
        self.episode_terminal_cost_sum = 0.0
        self.episode_terminal_without_decision_count = 0

    def create(self, *, agent_id, state, action, tag_gt, packet_id=None):
        transition_id = self.next_transition_id
        self.next_transition_id += 1
        state_array = np.asarray(state, dtype=np.float32).copy()
        self.entries[transition_id] = {
            "transition_id": transition_id,
            "agent_id": int(agent_id),
            "state": state_array,
            "action": int(action),
            "tag_gt": int(tag_gt),
            "reward": None,
            "next_state": None,
            "done": None,
            "causality_pending": True,
        }
        self.decision_snapshots[transition_id] = {
            "state": state_array.copy(),
            "action": int(action),
            "agent_id": int(agent_id),
            "tag_gt": int(tag_gt),
        }
        if packet_id is not None:
            self.attribute_packet(packet_id, transition_id)
        return transition_id

    def attribute_packet(self, packet_id, transition_id):
        """Link one actually affected packet to a preselected slot action."""

        packet_id, transition_id = int(packet_id), int(transition_id)
        if packet_id in self.terminal_packet_ids:
            raise AssertionError("terminal packet received another routing decision")
        snapshot = self.decision_snapshots.get(transition_id)
        if snapshot is None:
            raise KeyError(f"unknown routing transition ID: {transition_id}")
        previous = self.packet_pending.get(packet_id)
        if previous is not None and int(previous["transition_id"]) == transition_id:
            return False
        if previous is not None:
            self._close_cost_transition(
                previous["transition_id"],
                snapshot["state"],
                cost=0.0,
                done=False,
                packet_id=packet_id,
            )
        self.packet_pending[packet_id] = {
            "transition_id": transition_id,
            "state": np.asarray(snapshot["state"], dtype=np.float32).copy(),
        }
        self.episode_decision_count += 1
        return True

    def set_reward(self, transition_id, reward):
        entry = self._entry(transition_id)
        if entry["reward"] is not None:
            raise AssertionError("routing transition reward was assigned twice")
        value = float(reward)
        if not np.isfinite(value):
            raise ValueError("routing transition reward must be finite")
        entry["reward"] = value

    def finalize_packet(self, packet_id, *, violated):
        """Close one packet chain once with its terminal outcome."""

        packet_id = int(packet_id)
        if packet_id in self.terminal_packet_ids:
            return False
        self.terminal_packet_ids.add(packet_id)
        pending = self.packet_pending.pop(packet_id, None)
        if pending is None:
            self.episode_terminal_without_decision_count += 1
            return False
        cost = 1.0 if bool(violated) else 0.0
        self._close_cost_transition(
            pending["transition_id"],
            pending["state"],
            cost=cost,
            done=True,
            packet_id=packet_id,
        )
        self.episode_terminal_cost_sum += cost
        return True

    def _close_cost_transition(
        self, transition_id, next_state, *, cost, done, packet_id
    ):
        transition_id = int(transition_id)
        snapshot = self.decision_snapshots.get(transition_id)
        if snapshot is None:
            raise KeyError(f"unknown routing transition ID: {transition_id}")
        self.cost_closures.append({
            "transition_id": transition_id,
            "packet_id": int(packet_id),
            "state": np.asarray(snapshot["state"], dtype=np.float32).copy(),
            "action": int(snapshot["action"]),
            "tag_gt": int(snapshot["tag_gt"]),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
            "cost": float(cost),
            "done": bool(done),
        })
        self.episode_cost_transition_count += 1

    def finalize_causality(self, states, start_of_slot_hol_by_sender, *, terminal=False):
        """Resolve reward transitions from the next real sender snapshot."""

        states = {int(key): value for key, value in dict(states).items()}
        del start_of_slot_hol_by_sender
        finalized = 0
        for transition_id in sorted(self.entries):
            entry = self.entries[transition_id]
            if not entry["causality_pending"]:
                continue
            agent_id = entry["agent_id"]
            entry["next_state"] = np.asarray(
                states.get(agent_id, entry["state"]), dtype=np.float32
            ).copy()
            entry["done"] = bool(terminal)
            entry["causality_pending"] = False
            finalized += 1
        return finalized

    def commit_ready(self, replay):
        """Commit reward rows, then attach independently closed cost links."""

        committed = []
        for transition_id in sorted(tuple(self.entries)):
            entry = self.entries[transition_id]
            if entry["causality_pending"] or entry["reward"] is None:
                continue
            replay.add(
                entry["state"], entry["action"], entry["next_state"],
                entry["reward"], 0.0, entry["done"],
                tag_gt=entry["tag_gt"], agent_id=entry["agent_id"],
                transition_id=transition_id,
            )
            committed.append(transition_id)
            del self.entries[transition_id]
        remaining = []
        for closure in self.cost_closures:
            if not replay.add_cost_transition(
                closure["state"],
                closure["action"],
                closure["next_state"],
                closure["cost"],
                closure["done"],
                tag_gt=closure["tag_gt"],
                transition_id=closure["transition_id"],
                packet_id=closure["packet_id"],
            ):
                remaining.append(closure)
        self.cost_closures = remaining
        return committed

    def episode_diagnostics(self):
        return {
            "packet_path_decision_count": int(self.episode_decision_count),
            "packet_path_cost_transition_count": int(self.episode_cost_transition_count),
            "packet_path_terminal_cost_sum": float(self.episode_terminal_cost_sum),
            "pre_routing_terminal_without_decision_count": int(
                self.episode_terminal_without_decision_count
            ),
        }

    def state_dict(self):
        return {
            "schema_version": ROUTING_TRANSITION_LEDGER_SCHEMA_VERSION,
            "next_transition_id": int(self.next_transition_id),
            "entries": deepcopy(self.entries),
            "decision_snapshots": deepcopy(self.decision_snapshots),
            "packet_pending": deepcopy(self.packet_pending),
            "cost_closures": deepcopy(self.cost_closures),
            "terminal_packet_ids": sorted(self.terminal_packet_ids),
            "episode_diagnostics": self.episode_diagnostics(),
        }

    def load_state_dict(self, state):
        validated = validate_routing_transition_ledger_state(state)
        self.next_transition_id = validated["next_transition_id"]
        self.entries = deepcopy(validated["entries"])
        self.decision_snapshots = deepcopy(validated["decision_snapshots"])
        self.packet_pending = deepcopy(validated["packet_pending"])
        self.cost_closures = deepcopy(validated["cost_closures"])
        self.terminal_packet_ids = set(validated["terminal_packet_ids"])
        diagnostics = validated["episode_diagnostics"]
        self.episode_decision_count = diagnostics["packet_path_decision_count"]
        self.episode_cost_transition_count = diagnostics["packet_path_cost_transition_count"]
        self.episode_terminal_cost_sum = diagnostics["packet_path_terminal_cost_sum"]
        self.episode_terminal_without_decision_count = diagnostics[
            "pre_routing_terminal_without_decision_count"
        ]

    def assert_drained(self):
        if self.entries or self.packet_pending or self.cost_closures:
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
        raise RuntimeError(
            "routing transition ledger checkpoint uses the incompatible "
            "pre-packet-path cost contract; retraining is required"
        )
    required = {
        "schema_version", "next_transition_id", "entries", "decision_snapshots", "packet_pending",
        "cost_closures", "terminal_packet_ids", "episode_diagnostics",
    }
    if set(state) != required:
        raise RuntimeError("routing transition ledger state is incomplete")
    next_transition_id = state["next_transition_id"]
    if isinstance(next_transition_id, bool) or not isinstance(next_transition_id, int):
        raise RuntimeError("routing transition next ID is invalid")
    for field in ("entries", "decision_snapshots", "packet_pending"):
        if not isinstance(state[field], dict):
            raise RuntimeError(f"routing transition ledger {field} is invalid")
    if not isinstance(state["cost_closures"], list):
        raise RuntimeError("routing transition ledger cost_closures is invalid")
    if not isinstance(state["terminal_packet_ids"], list):
        raise RuntimeError("routing terminal packet IDs are invalid")
    if not isinstance(state["episode_diagnostics"], dict):
        raise RuntimeError("routing packet-path diagnostics are invalid")
    return deepcopy(state)
