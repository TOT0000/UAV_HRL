"""Read-only routing foldback and physical-connectivity diagnostics.

The collector is deliberately outside the policy and packet-outcome contracts.
It observes already-selected slot actions and already-completed hops, streams raw
records, and retains only aggregate keys plus the minimum state needed to trace
an in-progress hop across slot boundaries.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION = "routing-loop-diagnostics-v2"
ROUTING_LOOP_EVENT_DEFINITION = (
    "A completed U2U hop A->B immediately followed for the same packet by a "
    "completed U2U hop B->A; overlapping events are counted (A->B->A->B is two)."
)
ROUTING_LOOP_EVENT_FIELD_SEMANTICS = {
    "was_original_decision_slot_start_hol": (
        "packet identity matched the HOL captured in the referenced decision snapshot"
    ),
    "packet_decision_relation": (
        "how this departure used the referenced decision: first HOL departure, "
        "same-packet return reuse, other-packet reuse, empty-queue reuse, or unknown"
    ),
    "reused_same_packet_preselected_action": (
        "the same packet had already completed a departure from this sender using "
        "the referenced decision, including a prior-slot partial hop completed in "
        "that decision slot"
    ),
    "reused_preselected_action": (
        "this departure did not receive a packet-specific new policy decision"
    ),
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


class _JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self._handle = None
        self.row_count = 0

    def write(self, record):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        self._handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            + "\n"
        )
        self.row_count += 1

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        elif not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()


def _new_stats():
    return {
        "event_count": 0,
        "packet_keys": set(),
        "decision_keys": set(),
        "with_alternative_actual_legal": 0,
        "without_alternative_actual_legal": 0,
        "unknown_actual_legal": 0,
        "physical_alternative_but_not_actual_legal": 0,
        "completion_slot_physical_alternative_but_not_actual_legal": 0,
        "gs_reachable": 0,
        "gs_unreachable": 0,
        "gs_reachability_unknown": 0,
        "gs_reachable_with_alternative": 0,
        "gs_reachable_without_alternative": 0,
        "gs_unreachable_with_alternative": 0,
        "gs_unreachable_without_alternative": 0,
        "slot_start_hol": 0,
        "first_departure_slot_start_hol": 0,
        "reused_same_packet_action": 0,
        "reused_other_packet_action": 0,
        "reused_empty_queue_action": 0,
        "decision_relation_unknown": 0,
        "cross_slot_transmission": 0,
        "loop_packets_on_time_delivered": 0,
        "loop_packets_late_delivered": 0,
        "loop_packets_not_delivered": 0,
        "twenty_hop_terminal_packet_count": 0,
        "twenty_hop_all_completed_in_one_slot": 0,
        "twenty_hop_delivered_to_gs": 0,
        "twenty_hop_hop_limit_not_delivered": 0,
    }


class RoutingLoopDiagnosticsCollector:
    """Streaming observer for routing decisions and immediate U2U foldbacks."""

    def __init__(self, output_directory, *, metadata=None):
        self.output_directory = Path(output_directory).resolve()
        self.metadata = dict(metadata or {})
        self.snapshot_writer = _JsonlWriter(
            self.output_directory / "routing_loop_slot_snapshots.jsonl"
        )
        self.event_writer = _JsonlWriter(
            self.output_directory / "routing_loop_events.jsonl"
        )
        self._stats = defaultdict(_new_stats)
        self._pair_stats = defaultdict(_new_stats)
        self._episode = None
        self._current_decisions = {}
        self._hop_origins = {}
        self._hop_slot_state = {}
        self._decision_departures = defaultdict(set)
        self._finalized = False
        self._initialized_pair_groups = set()

    @staticmethod
    def _packet_key(roi_count, scenario_id, packet_id):
        return (int(roi_count), str(scenario_id), int(packet_id))

    def begin_episode(self, *, roi_count, scenario_id, episode_index):
        if self._episode is not None:
            raise RuntimeError("routing-loop diagnostic episode already active")
        if scenario_id is None:
            raise ValueError("routing-loop diagnostics require a scenario_id")
        self._episode = {
            "roi_count": int(roi_count),
            "scenario_id": str(scenario_id),
            "episode_index": int(episode_index),
        }
        self._current_decisions = {}
        self._hop_origins = {}
        self._hop_slot_state = {}
        self._decision_departures = defaultdict(set)
        for group in self._group_keys(int(roi_count), "FOV") + self._group_keys(
            int(roi_count), "COM"
        ):
            self._stats[group]

    @staticmethod
    def _mask_values(mask):
        return np.asarray(mask, dtype=bool).astype(int).tolist()

    @staticmethod
    def _physical_graph(env):
        num_uav = int(env.num_UAV)
        gs_id = int(env.GS_ID)
        adjacency = {node: set() for node in range(num_uav + 1)}
        edges = []
        for left in range(num_uav):
            for right in range(left + 1, num_uav):
                if bool(env.is_routing_link_in_range(left, right)):
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                    edges.append([left, right])
            if bool(env.is_routing_link_in_range(left, gs_id)):
                adjacency[left].add(gs_id)
                adjacency[gs_id].add(left)
                edges.append([left, gs_id])
        reachable = {gs_id}
        frontier = [gs_id]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        return adjacency, edges, reachable

    def observe_slot(
        self,
        *,
        env,
        packet_engine,
        current_time,
        slot_index,
        physical_masks,
        effective_masks,
        selection_masks,
        selected_actions,
        start_of_slot_hol_by_sender,
    ):
        del packet_engine
        if self._episode is None:
            raise RuntimeError("routing-loop diagnostic episode is not active")
        num_uav = int(env.num_UAV)
        gs_id = int(env.GS_ID)
        adjacency, edges, reachable = self._physical_graph(env)
        groups = set(self._group_keys(self._episode["roi_count"], "FOV"))
        groups.update(self._group_keys(self._episode["roi_count"], "COM"))
        for group in groups.difference(self._initialized_pair_groups):
            for left in range(num_uav):
                for right in range(left + 1, num_uav):
                    self._pair_stats[(group, (left, right))]
            self._initialized_pair_groups.add(group)
        snapshot_id = (
            f"roi={self._episode['roi_count']}|scenario={self._episode['scenario_id']}"
            f"|slot={int(slot_index)}"
        )
        hol_rows = {}
        for sender in range(num_uav):
            hol = start_of_slot_hol_by_sender.get(sender)
            hol_rows[str(sender)] = {
                "packet_id": None if hol is None else int(hol["id"]),
                "task_type": None if hol is None else str(hol.get("task_type")),
                "hop_receiver_lock": (
                    None
                    if hol is None or hol.get("hop_receiver") is None
                    else int(hol["hop_receiver"])
                ),
            }
            selection = np.asarray(selection_masks[sender], dtype=bool)
            physical_neighbors = sorted(
                int(value) for value in adjacency[sender] if int(value) != sender
            )
            self._current_decisions[sender] = {
                "snapshot_id": snapshot_id,
                "decision_key": (snapshot_id, int(sender)),
                "sender_uav_id": int(sender),
                "selected_action": int(selected_actions[sender]),
                "slot_index": int(slot_index),
                "time_seconds": float(current_time),
                "hol_packet_id": None if hol is None else int(hol["id"]),
                "hol_task_type": None if hol is None else str(hol.get("task_type")),
                "hol_receiver_lock": (
                    None
                    if hol is None or hol.get("hop_receiver") is None
                    else int(hol["hop_receiver"])
                ),
                "selection_legal_actions": np.flatnonzero(selection).astype(int).tolist(),
                "physical_neighbors": physical_neighbors,
                "physical_gs_reachable": bool(sender in reachable),
            }
        record = {
            "schema_version": ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
            "record_type": "routing_slot_snapshot",
            "snapshot_id": snapshot_id,
            **self._episode,
            "slot_index": int(slot_index),
            "time_seconds": float(current_time),
            "gs_id": gs_id,
            "positions": {
                "uavs": [
                    {
                        "uav_id": uid,
                        "xyz_m": [float(value) for value in env.uav_dict[uid].get_position()],
                    }
                    for uid in range(num_uav)
                ],
                "ground_station": {
                    "gs_id": gs_id,
                    "xyz_m": [float(value) for value in env.GS_pos],
                },
            },
            "physical_masks": {
                str(uid): self._mask_values(physical_masks[uid]) for uid in range(num_uav)
            },
            "effective_masks": {
                str(uid): self._mask_values(effective_masks[uid]) for uid in range(num_uav)
            },
            "selection_masks": {
                str(uid): self._mask_values(selection_masks[uid]) for uid in range(num_uav)
            },
            "selected_next_hops": {
                str(uid): int(selected_actions[uid]) for uid in range(num_uav)
            },
            "slot_start_hol": hol_rows,
            "physical_topology": {
                "definition": "inclusive authoritative 3-D routing-range predicate",
                "slot_geometry_constant": True,
                "undirected_edges": edges,
                "gs_component_nodes": sorted(reachable),
                "gs_reachable_by_uav": {
                    str(uid): bool(uid in reachable) for uid in range(num_uav)
                },
            },
            "selected_forwarding_edges": [
                [uid, int(selected_actions[uid])]
                for uid in range(num_uav)
                if int(selected_actions[uid]) != uid
            ],
        }
        self.snapshot_writer.write(record)

    def _origin_from_current_decision(self, event):
        sender = int(event["sender_uav_id"])
        decision = self._current_decisions.get(sender)
        if decision is None:
            return None
        packet_id = int(event["packet_id"])
        return {
            **decision,
            "was_original_decision_slot_start_hol": (
                decision["hol_packet_id"] == packet_id
            ),
        }

    @staticmethod
    def _decision_relation(origin, reused_same_packet):
        """Separate decision-time HOL identity from reuse after a return.

        Completed-departure uses are keyed by decision snapshot.  In
        particular this identifies a repeat departure when the first hop began
        in an earlier slot and completed under the current slot's held action.
        """

        if origin is None:
            return "unknown"
        if reused_same_packet:
            return "reused_same_packet_preselected_action"
        if origin["was_original_decision_slot_start_hol"]:
            return "first_departure_slot_start_hol"
        if origin["hol_packet_id"] is None:
            return "reused_empty_queue_preselected_action"
        return "reused_other_packet_preselected_action"

    def observe_hop_event(self, event):
        if self._episode is None:
            raise RuntimeError("routing-loop diagnostic episode is not active")
        phase = str(event.get("phase"))
        packet_key = self._packet_key(
            self._episode["roi_count"],
            self._episode["scenario_id"],
            event["packet_id"],
        )
        hop_number = int(event["hop_sequence_number"])
        hop_key = (packet_key, hop_number)
        if phase == "started":
            self._hop_origins[hop_key] = self._origin_from_current_decision(event)
            return
        if phase != "completed":
            return

        if str(event.get("link_type")) not in {"U2U", "U2G"}:
            return

        slot_decision = self._current_decisions.get(int(event["sender_uav_id"]))
        completion_slot = None if slot_decision is None else int(slot_decision["slot_index"])
        state = self._hop_slot_state.setdefault(
            packet_key,
            {"count": 0, "first_slot": completion_slot, "all_same_slot": True},
        )
        state["count"] += 1
        if completion_slot != state["first_slot"]:
            state["all_same_slot"] = False

        origin = self._hop_origins.pop(hop_key, None)
        completed_hops = list(event.get("completed_hops") or ())
        sender = int(event["sender_uav_id"])
        origin_departure_key = (
            None
            if origin is None
            else (packet_key, sender, origin["decision_key"])
        )
        completion_departure_key = (
            None
            if slot_decision is None
            else (packet_key, sender, slot_decision["decision_key"])
        )
        reused_for_origin = bool(
            origin_departure_key is not None
            and self._decision_departures[origin_departure_key]
        )
        reused_for_completion_slot = bool(
            completion_departure_key is not None
            and self._decision_departures[completion_departure_key]
        )
        # A completion is a use of both its traced start decision and the
        # current slot's held action.  Recording the latter is what makes a
        # prior-slot partial completion count before a same-slot return.
        for departure_key in {origin_departure_key, completion_departure_key}:
            if departure_key is not None:
                self._decision_departures[departure_key].add(hop_number)
        if str(event.get("link_type")) != "U2U" or len(completed_hops) < 2:
            return
        previous = completed_hops[-2]
        current = completed_hops[-1]
        if str(previous.get("link_type", "U2U")) != "U2U":
            return
        try:
            is_foldback = (
                int(previous["from"]) == int(current["to"])
                and int(previous["to"]) == int(current["from"])
            )
        except (TypeError, ValueError):
            is_foldback = False
        if not is_foldback:
            return

        receiver = int(event["receiver_id"])
        if origin is None:
            actual_alternatives = None
            physical_alternatives = None
            relation = "unknown"
            decision_snapshot_id = None
            decision_key = None
            gs_reachable = None
            decision_slot = None
        else:
            actual_alternatives = sorted(
                action
                for action in origin["selection_legal_actions"]
                if action not in {sender, receiver}
            )
            physical_alternatives = sorted(
                action
                for action in origin["physical_neighbors"]
                if action not in {sender, receiver}
            )
            relation = self._decision_relation(origin, reused_for_origin)
            decision_snapshot_id = origin["snapshot_id"]
            decision_key = origin["decision_key"]
            gs_reachable = bool(origin["physical_gs_reachable"])
            decision_slot = int(origin["slot_index"])
        if slot_decision is None:
            completion_slot_alternatives = None
            completion_slot_physical_alternatives = None
            completion_slot_relation = "unknown"
        else:
            completion_slot_alternatives = sorted(
                action
                for action in slot_decision["selection_legal_actions"]
                if action not in {sender, receiver}
            )
            completion_slot_physical_alternatives = sorted(
                action
                for action in slot_decision["physical_neighbors"]
                if action not in {sender, receiver}
            )
            completion_slot_relation = self._decision_relation(
                {
                    **slot_decision,
                    "was_original_decision_slot_start_hol": (
                        slot_decision["hol_packet_id"] == int(event["packet_id"])
                    ),
                },
                reused_for_completion_slot,
            )
        event_id = (
            f"{packet_key[0]}|{packet_key[1]}|packet={packet_key[2]}|hop={hop_number}"
        )
        record = {
            "schema_version": ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
            "record_type": "immediate_u2u_foldback",
            "event_id": event_id,
            **self._episode,
            "packet_id": int(event["packet_id"]),
            "task_type": str(event["task_type"]),
            "previous_uav_id": receiver,
            "previous_hop_sender_uav_id": int(previous["from"]),
            "previous_hop_receiver_uav_id": int(previous["to"]),
            "sender_uav_id": sender,
            "receiver_uav_id": receiver,
            "completion_time_seconds": float(event["completion_time_seconds"]),
            "hop_sequence_number": hop_number,
            "completion_slot_index": completion_slot,
            "decision_slot_index": decision_slot,
            "decision_snapshot_id": decision_snapshot_id,
            "service_slot_snapshot_id": (
                None if slot_decision is None else slot_decision["snapshot_id"]
            ),
            "decision_trace_status": "known" if origin is not None else "unknown",
            "packet_decision_relation": relation,
            "was_slot_start_hol": (
                origin["was_original_decision_slot_start_hol"]
                if origin
                else None
            ),
            "was_original_decision_slot_start_hol": (
                origin["was_original_decision_slot_start_hol"]
                if origin
                else None
            ),
            "completion_slot_packet_decision_relation": completion_slot_relation,
            "was_completion_slot_start_hol": (
                None
                if slot_decision is None
                else slot_decision["hol_packet_id"] == int(event["packet_id"])
            ),
            "reused_preselected_action": (
                relation.startswith("reused_") if origin else None
            ),
            "reused_same_packet_preselected_action": (
                relation == "reused_same_packet_preselected_action"
                if origin
                else None
            ),
            "cross_slot_transmission": bool(event["cross_slot_transmission"]),
            "existing_receiver_lock_traced_to_prior_decision": bool(
                origin is not None
                and bool(event["cross_slot_transmission"])
                and decision_slot != completion_slot
            ),
            "receiver_lock_at_original_decision": (
                None if origin is None else origin["hol_receiver_lock"]
            ),
            "receiver_lock_at_completion_slot_start": (
                None if slot_decision is None else slot_decision["hol_receiver_lock"]
            ),
            "had_existing_receiver_lock_at_completion_slot_start": (
                None
                if slot_decision is None
                else slot_decision["hol_receiver_lock"] == receiver
            ),
            "alternative_actual_legal_next_hops": actual_alternatives,
            "alternative_actual_legal_next_hop_count": (
                None if actual_alternatives is None else len(actual_alternatives)
            ),
            "alternative_physical_neighbors": physical_alternatives,
            "alternative_physical_neighbor_count": (
                None if physical_alternatives is None else len(physical_alternatives)
            ),
            "physical_alternative_blocked_by_mask_or_lock": (
                None
                if actual_alternatives is None
                else bool(physical_alternatives and not actual_alternatives)
            ),
            "completion_slot_alternative_actual_legal_next_hops": (
                completion_slot_alternatives
            ),
            "completion_slot_alternative_actual_legal_next_hop_count": (
                None
                if completion_slot_alternatives is None
                else len(completion_slot_alternatives)
            ),
            "completion_slot_alternative_physical_neighbors": (
                completion_slot_physical_alternatives
            ),
            "completion_slot_physical_alternative_blocked_by_mask_or_lock": (
                None
                if completion_slot_alternatives is None
                else bool(
                    completion_slot_physical_alternatives
                    and not completion_slot_alternatives
                )
            ),
            "physical_gs_reachable_at_decision": gs_reachable,
        }
        self.event_writer.write(record)
        self._add_event(record, packet_key, decision_key)

    @staticmethod
    def _group_keys(roi_count, task_type):
        task = "VS" if task_type == "FOV" else "COM"
        roi = int(roi_count)
        return ((None, "ALL"), (None, task), (roi, "ALL"), (roi, task))

    @staticmethod
    def _group_label(group):
        roi_count, task = group
        return task if roi_count is None else f"ROI_{roi_count}_{task}"

    @staticmethod
    def _group_sort_key(group):
        roi_count, task = group
        task_order = {"ALL": 0, "VS": 1, "COM": 2}
        return (-1 if roi_count is None else int(roi_count), task_order[task])

    def _add_event(self, record, packet_key, decision_key):
        pair = tuple(sorted((int(record["sender_uav_id"]), int(record["receiver_uav_id"]))))
        for group in self._group_keys(record["roi_count"], record["task_type"]):
            self._update_event_stats(self._stats[group], record, packet_key, decision_key)
            self._update_event_stats(
                self._pair_stats[(group, pair)], record, packet_key, decision_key
            )

    @staticmethod
    def _update_event_stats(stats, record, packet_key, decision_key):
        stats["event_count"] += 1
        stats["packet_keys"].add(packet_key)
        if decision_key is not None:
            stats["decision_keys"].add(decision_key)
        alternatives = record["alternative_actual_legal_next_hop_count"]
        reachable = record["physical_gs_reachable_at_decision"]
        if alternatives is None:
            stats["unknown_actual_legal"] += 1
        elif alternatives:
            stats["with_alternative_actual_legal"] += 1
        else:
            stats["without_alternative_actual_legal"] += 1
        if record["physical_alternative_blocked_by_mask_or_lock"]:
            stats["physical_alternative_but_not_actual_legal"] += 1
        if record["completion_slot_physical_alternative_blocked_by_mask_or_lock"]:
            stats["completion_slot_physical_alternative_but_not_actual_legal"] += 1
        if reachable is None:
            stats["gs_reachability_unknown"] += 1
        else:
            stats["gs_reachable" if reachable else "gs_unreachable"] += 1
            if alternatives is not None:
                suffix = "with_alternative" if alternatives else "without_alternative"
                prefix = "gs_reachable" if reachable else "gs_unreachable"
                stats[f"{prefix}_{suffix}"] += 1
        relation_key = {
            "first_departure_slot_start_hol": "first_departure_slot_start_hol",
            "reused_same_packet_preselected_action": "reused_same_packet_action",
            "reused_other_packet_preselected_action": "reused_other_packet_action",
            "reused_empty_queue_preselected_action": "reused_empty_queue_action",
        }.get(record["packet_decision_relation"], "decision_relation_unknown")
        stats[relation_key] += 1
        stats["slot_start_hol"] += int(
            bool(record.get("was_original_decision_slot_start_hol"))
        )
        stats["cross_slot_transmission"] += int(record["cross_slot_transmission"])

    def end_episode(self, packet_outcomes):
        if self._episode is None:
            raise RuntimeError("routing-loop diagnostic episode is not active")
        roi_count = self._episode["roi_count"]
        scenario_id = self._episode["scenario_id"]
        for outcome in packet_outcomes:
            packet_key = self._packet_key(roi_count, scenario_id, outcome["packet_id"])
            task = str(outcome.get("task_type", "COM"))
            groups = self._group_keys(roi_count, task)
            for group in groups:
                stats = self._stats[group]
                if packet_key in stats["packet_keys"]:
                    if outcome.get("outcome") == "on_time_delivered":
                        stats["loop_packets_on_time_delivered"] += 1
                    elif outcome.get("outcome") == "late_delivered":
                        stats["loop_packets_late_delivered"] += 1
                    else:
                        stats["loop_packets_not_delivered"] += 1
            for (group, _pair), stats in self._pair_stats.items():
                if group not in groups or packet_key not in stats["packet_keys"]:
                    continue
                if outcome.get("outcome") == "on_time_delivered":
                    stats["loop_packets_on_time_delivered"] += 1
                elif outcome.get("outcome") == "late_delivered":
                    stats["loop_packets_late_delivered"] += 1
                else:
                    stats["loop_packets_not_delivered"] += 1
            completed = int(outcome.get("completed_uav_hop_count", 0))
            if completed == 20:
                hop_state = self._hop_slot_state.get(packet_key)
                delivered = bool(outcome.get("delivered_to_gs", False))

                def update_twenty_hop(stats):
                    stats["twenty_hop_terminal_packet_count"] += 1
                    stats["twenty_hop_all_completed_in_one_slot"] += int(
                        hop_state is not None
                        and hop_state["count"] == 20
                        and hop_state["all_same_slot"]
                    )
                    stats["twenty_hop_delivered_to_gs"] += int(delivered)
                    stats["twenty_hop_hop_limit_not_delivered"] += int(
                        not delivered
                    )

                for group in groups:
                    update_twenty_hop(self._stats[group])
                for (group, _pair), stats in self._pair_stats.items():
                    if group in groups and packet_key in stats["packet_keys"]:
                        update_twenty_hop(stats)
        self._episode = None
        self._current_decisions = {}
        self._hop_origins = {}
        self._hop_slot_state = {}
        self._decision_departures = defaultdict(set)

    @staticmethod
    def _ratio(numerator, denominator):
        return None if not denominator else float(numerator) / float(denominator)

    def _public_stats(self, group, stats):
        event_count = int(stats["event_count"])
        twenty_count = int(stats["twenty_hop_terminal_packet_count"])
        row = {
            "group": self._group_label(group),
            "roi_count": group[0],
            "task": group[1],
            "event_count": event_count,
            "distinct_packet_count": len(stats["packet_keys"]),
            "distinct_decision_count": len(stats["decision_keys"]),
        }
        for key, value in stats.items():
            if key not in {"event_count", "packet_keys", "decision_keys"}:
                row[key] = int(value)
        row.update(
            {
                "with_alternative_actual_legal_ratio": self._ratio(
                    stats["with_alternative_actual_legal"], event_count
                ),
                "gs_reachable_ratio": self._ratio(stats["gs_reachable"], event_count),
                "slot_start_hol_ratio": self._ratio(stats["slot_start_hol"], event_count),
                "reused_preselected_action_ratio": self._ratio(
                    stats["reused_same_packet_action"]
                    + stats["reused_other_packet_action"]
                    + stats["reused_empty_queue_action"],
                    event_count,
                ),
                "twenty_hop_all_completed_in_one_slot_ratio": self._ratio(
                    stats["twenty_hop_all_completed_in_one_slot"], twenty_count
                ),
            }
        )
        return row

    @staticmethod
    def _write_csv(path, rows):
        rows = list(rows)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def finalize(self, *, extra_metadata=None):
        if self._finalized:
            raise RuntimeError("routing-loop diagnostics already finalized")
        if self._episode is not None:
            raise RuntimeError("cannot finalize during an active diagnostic episode")
        self.snapshot_writer.close()
        self.event_writer.close()
        summary_rows = [
            self._public_stats(group, self._stats[group])
            for group in sorted(self._stats, key=self._group_sort_key)
        ]
        pair_rows = []
        for (group, pair), stats in sorted(
            self._pair_stats.items(),
            key=lambda item: (self._group_sort_key(item[0][0]), item[0][1]),
        ):
            pair_rows.append(
                {
                    **self._public_stats(group, stats),
                    "uav_a": int(pair[0]),
                    "uav_b": int(pair[1]),
                    "is_uav_11_15": bool(pair == (11, 15)),
                }
            )
        summary = {
            "schema_version": ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
            "event_definition": ROUTING_LOOP_EVENT_DEFINITION,
            "event_field_semantics": ROUTING_LOOP_EVENT_FIELD_SEMANTICS,
            "denominator_notes": {
                "event_ratios": "completed immediate-foldback events",
                "distinct_packet_count": "scenario-qualified packets with at least one event",
                "distinct_decision_count": "sender decisions referenced by at least one event",
                "twenty_hop_ratio": "terminal packets with exactly 20 completed UAV hops",
                "slot_start_hol_ratio": (
                    "events whose packet was the original decision's slot-start HOL; "
                    "this identity is independent of whether the departure reused the decision"
                ),
                "decision_use_categories": (
                    "first departure of the slot-start HOL, same packet returned and reused, "
                    "other packet reused, empty-queue action reused, or unknown; event denominator"
                ),
                "pair_twenty_hop": (
                    "distinct 20-completed-hop packets with at least one immediate foldback "
                    "for that pair; a packet may count once for each involved pair"
                ),
                "overlap": "packet classifications are not mutually exclusive event categories",
            },
            "groups": summary_rows,
            "uav_11_15": [
                self._public_stats(group, stats)
                for (group, pair), stats in sorted(
                    self._pair_stats.items(),
                    key=lambda item: (
                        self._group_sort_key(item[0][0]), item[0][1]
                    ),
                )
                if pair == (11, 15)
            ],
        }
        metadata = {
            "schema_version": ROUTING_LOOP_DIAGNOSTIC_SCHEMA_VERSION,
            "collector_enabled": True,
            "read_only_observer": True,
            "event_definition": ROUTING_LOOP_EVENT_DEFINITION,
            "event_field_semantics": ROUTING_LOOP_EVENT_FIELD_SEMANTICS,
            "slot_snapshot_count": self.snapshot_writer.row_count,
            "foldback_event_count": self.event_writer.row_count,
            "files": {
                "slot_snapshots": "routing_loop_slot_snapshots.jsonl",
                "events": "routing_loop_events.jsonl",
                "summary_json": "routing_loop_summary.json",
                "summary_csv": "routing_loop_summary.csv",
                "pairs_csv": "routing_loop_pairs.csv",
            },
            **self.metadata,
            **dict(extra_metadata or {}),
        }
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "routing_loop_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.output_directory / "routing_loop_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_csv(self.output_directory / "routing_loop_summary.csv", summary_rows)
        self._write_csv(self.output_directory / "routing_loop_pairs.csv", pair_rows)
        self._finalized = True
        return metadata

    def close(self):
        self.snapshot_writer.close()
        self.event_writer.close()


def compare_json_values(actual, expected, *, path="$", float_rtol=1e-12, float_atol=1e-12):
    """Compare streamed episode rows: discrete values exact, floats tightly tolerant."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: object keys differ")
        for key in expected:
            compare_json_values(
                actual[key], expected[key], path=f"{path}.{key}",
                float_rtol=float_rtol, float_atol=float_atol,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list shape differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_json_values(
                actual_item, expected_item, path=f"{path}[{index}]",
                float_rtol=float_rtol, float_atol=float_atol,
            )
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=float_rtol, abs_tol=float_atol
        ):
            raise AssertionError(f"{path}: float differs: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: value differs: {actual!r} != {expected!r}")
    if type(actual) is not type(expected):
        raise AssertionError(
            f"{path}: discrete JSON type differs: "
            f"{type(actual).__name__} != {type(expected).__name__}"
        )


def verify_episode_jsonl_prefix(actual_path, source_path, expected_scenario_ids):
    """Stream-compare exactly the selected source episodes in manifest order."""

    expected_ids = list(expected_scenario_ids)
    compared = 0
    with Path(actual_path).open("r", encoding="utf-8") as actual_handle, Path(
        source_path
    ).open("r", encoding="utf-8") as source_handle:
        for scenario_id in expected_ids:
            actual_line = actual_handle.readline()
            source_line = source_handle.readline()
            if not actual_line or not source_line:
                raise AssertionError("per-episode JSONL ended before selected prefix")
            actual = json.loads(actual_line)
            expected = json.loads(source_line)
            if actual.get("scenario_id") != scenario_id or expected.get("scenario_id") != scenario_id:
                raise AssertionError("per-episode scenario order differs from manifest prefix")
            # A canonical prefix is a different manifest artifact by design;
            # its hash is verified separately against the saved prefix.  Every
            # simulation result field remains part of this strict comparison.
            actual.pop("evaluation_manifest_hash", None)
            expected.pop("evaluation_manifest_hash", None)
            compare_json_values(actual, expected)
            compared += 1
        if actual_handle.readline().strip():
            raise AssertionError("diagnostic evaluation produced extra episodes")
    return compared
