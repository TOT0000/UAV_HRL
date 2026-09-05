"""Deterministic expected-capacity Relay assignment and movement metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from Channel_model import reference_u2u_max_capacity_mbps
from experiment_config import (
    PERMANENT_GS_GATEWAY_UAV_ID,
    RELAY_FORWARD_REFERENCE_SECONDS,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)


@dataclass(frozen=True)
class RelayMetrics:
    candidate_uav_id: int
    receive_score: float
    forward_score: float
    utility: float
    receive_centroid: tuple[float, float, float] | None
    forward_target: tuple[float, float, float] | None
    first_next_hop: int | None
    shortest_path: tuple[int, ...]
    shortest_path_cost_seconds: float | None
    reachable: bool
    zero_backlog_fallback: bool
    source_uav_ids: tuple[int, ...]
    source_weights: tuple[tuple[int, float], ...]
    normalized_receive_capacities: tuple[tuple[int, float], ...]

    def metadata(self):
        payload = asdict(self)
        for key in (
            "receive_centroid",
            "forward_target",
            "shortest_path",
            "source_uav_ids",
        ):
            value = payload[key]
            payload[key] = None if value is None else list(value)
        payload["source_weights"] = {
            str(source): float(weight) for source, weight in self.source_weights
        }
        payload["normalized_receive_capacities"] = {
            str(source): float(capacity)
            for source, capacity in self.normalized_receive_capacities
        }
        return payload


def requested_relay_count(discovered_roi_count):
    discovered = max(int(discovered_roi_count), 0)
    return discovered // 2


def _available_uav_ids(env):
    ids = tuple(sorted({int(uid) for uid in env.get_available_uav_ids()}))
    return tuple(uid for uid in ids if 0 <= uid < int(env.num_UAV))


def _backlog_snapshot(env, backlog_bits):
    snapshot = (
        getattr(env, "assignment_backlog_snapshot", {})
        if backlog_bits is None
        else backlog_bits
    )
    clean = {}
    for uid in _available_uav_ids(env):
        try:
            value = float(snapshot.get(uid, snapshot.get(str(uid), 0.0)))
        except (AttributeError, TypeError, ValueError):
            value = 0.0
        clean[uid] = value if math.isfinite(value) and value > 0.0 else 0.0
    return clean


def _expected_u2u_capacity_mbps(env, sender, receiver):
    sender, receiver = int(sender), int(receiver)
    if sender == receiver or not env.is_u2u_in_range(sender, receiver):
        return 0.0
    capacity = float(np.asarray(env.Capacity_matrix)[sender, receiver])
    return capacity if math.isfinite(capacity) and capacity > 0.0 else 0.0


def _expected_u2g_capacity_mbps(env, sender):
    sender = int(sender)
    if not env.is_u2g_in_range(sender):
        return 0.0
    capacity = float(np.asarray(env.gs_capacity)[sender])
    return capacity if math.isfinite(capacity) and capacity > 0.0 else 0.0


def _shortest_forward_path(env, start, available, backlog):
    gs_id = int(env.GS_ID)
    slot_seconds = float(env.dt)
    if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
        raise ValueError("routing slot duration must be positive and finite")
    start = int(start)
    best = {start: (0.0, (start,))}
    heap = [(0.0, (start,), start)]
    while heap:
        cost, path, node = heapq.heappop(heap)
        if best.get(node) != (cost, path):
            continue
        if node == gs_id:
            return cost, path
        sender_backlog = float(backlog.get(node, 0.0))
        neighbors = []
        for receiver in available:
            if receiver == node or receiver in path:
                continue
            capacity = _expected_u2u_capacity_mbps(env, node, receiver)
            if capacity > 0.0:
                neighbors.append((receiver, capacity))
        gs_capacity = _expected_u2g_capacity_mbps(env, node)
        if gs_capacity > 0.0:
            neighbors.append((gs_id, gs_capacity))
        for receiver, capacity in sorted(neighbors):
            edge_cost = slot_seconds + sender_backlog / (capacity * 1e6)
            if not math.isfinite(edge_cost) or edge_cost < 0.0:
                continue
            candidate = (cost + edge_cost, path + (receiver,))
            previous = best.get(receiver)
            if previous is None or candidate < previous:
                best[receiver] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], receiver))
    return math.inf, ()


def relay_metrics(env, candidate_uav_id, backlog_bits=None):
    candidate = int(candidate_uav_id)
    available = _available_uav_ids(env)
    backlog = _backlog_snapshot(env, backlog_bits)
    gateway = int(
        getattr(env, "permanent_gs_gateway_uav_id", PERMANENT_GS_GATEWAY_UAV_ID)
    )
    sources = tuple(uid for uid in available if uid not in {candidate, gateway})
    total_backlog = sum(backlog.get(uid, 0.0) for uid in sources)
    uniform = bool(sources and total_backlog <= 0.0)
    if total_backlog > 0.0:
        weights = {uid: backlog.get(uid, 0.0) / total_backlog for uid in sources}
    elif sources:
        weights = {uid: 1.0 / len(sources) for uid in sources}
    else:
        weights = {}

    reference = float(reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ))
    if not math.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("canonical U2U capacity reference is invalid")
    normalized = {}
    centroid_numerator = np.zeros(3, dtype=float)
    centroid_denominator = 0.0
    for source in sources:
        value = float(
            np.clip(
                _expected_u2u_capacity_mbps(env, source, candidate) / reference,
                0.0,
                1.0,
            )
        )
        normalized[source] = value
        coefficient = weights[source] * value
        centroid_denominator += coefficient
        centroid_numerator += coefficient * np.asarray(
            env.uav_dict[source].get_position(), dtype=float
        )
    receive_score = float(np.clip(centroid_denominator, 0.0, 1.0))
    receive_centroid = (
        tuple(map(float, centroid_numerator / centroid_denominator))
        if centroid_denominator > 0.0
        else None
    )

    path_cost, path = _shortest_forward_path(env, candidate, available, backlog)
    reachable = bool(path and path[-1] == int(env.GS_ID) and math.isfinite(path_cost))
    if reachable:
        forward_score = float(
            np.clip(
                1.0 / (1.0 + path_cost / RELAY_FORWARD_REFERENCE_SECONDS),
                0.0,
                1.0,
            )
        )
        first_hop = int(path[1])
        target = (
            tuple(map(float, env.GS_pos))
            if first_hop == int(env.GS_ID)
            else tuple(map(float, env.uav_dict[first_hop].get_position()))
        )
    else:
        forward_score = 0.0
        first_hop = None
        target = None
        path_cost = None
        path = ()
    utility = float(np.clip(min(receive_score, forward_score), 0.0, 1.0))
    values = (receive_score, forward_score, utility)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AssertionError("Relay metrics must be finite and within [0, 1]")
    return RelayMetrics(
        candidate_uav_id=candidate,
        receive_score=receive_score,
        forward_score=forward_score,
        utility=utility,
        receive_centroid=receive_centroid,
        forward_target=target,
        first_next_hop=first_hop,
        shortest_path=tuple(path),
        shortest_path_cost_seconds=path_cost,
        reachable=reachable,
        zero_backlog_fallback=uniform,
        source_uav_ids=sources,
        source_weights=tuple(sorted(weights.items())),
        normalized_receive_capacities=tuple(sorted(normalized.items())),
    )


def relay_metrics_by_candidate(env, candidate_uav_ids, backlog_bits=None):
    return {
        int(uid): relay_metrics(env, int(uid), backlog_bits=backlog_bits)
        for uid in sorted({int(uid) for uid in candidate_uav_ids})
    }
