"""Deterministic expected-capacity Relay assignment and movement metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from Channel_model import reference_u2u_max_capacity_mbps
from experiment_config import (
    COMMUNICATION_RANGE_M,
    ENVIRONMENT_HEIGHT_M,
    ENVIRONMENT_WIDTH_M,
    GROUND_ALTITUDE_M,
    PERMANENT_GS_GATEWAY_UAV_ID,
    RELAY_FORWARD_DISTANCE_POTENTIAL_WEIGHT,
    RELAY_FORWARD_PATH_POTENTIAL_WEIGHT,
    RELAY_FORWARD_REFERENCE_SECONDS,
    RELAY_RECEIVE_CAPACITY_POTENTIAL_WEIGHT,
    RELAY_RECEIVE_DISTANCE_POTENTIAL_WEIGHT,
    TASK_POTENTIAL_NORMALIZATION_EPSILON,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
    UAV_MAX_ALTITUDE_M,
)


@dataclass(frozen=True)
class RelayMetrics:
    candidate_uav_id: int
    receive_score: float
    forward_score: float
    utility: float
    receive_distance_progress: float
    forward_distance_progress: float
    movement_receive_score: float
    movement_forward_score: float
    movement_utility: float
    receive_centroid: tuple[float, float, float] | None
    forward_target: tuple[float, float, float] | None
    receive_direction_target: tuple[float, float, float] | None
    forward_direction_target: tuple[float, float, float] | None
    forward_distance_target_node: int | None
    forward_range_gap_m: float
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
            "receive_direction_target",
            "forward_direction_target",
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


def _maximum_range_gap_m():
    maximum_distance = math.sqrt(
        ENVIRONMENT_WIDTH_M**2
        + ENVIRONMENT_HEIGHT_M**2
        + (UAV_MAX_ALTITUDE_M - GROUND_ALTITUDE_M) ** 2
    )
    return max(
        maximum_distance - COMMUNICATION_RANGE_M,
        TASK_POTENTIAL_NORMALIZATION_EPSILON,
    )


def _distance_progress_from_gap(range_gap_m):
    progress = 1.0 - float(
        np.clip(float(range_gap_m) / _maximum_range_gap_m(), 0.0, 1.0)
    )
    return float(np.clip(progress, 0.0, 1.0))


def _gs_connected_component(env, available):
    """Return UAVs with a positive expected-capacity directed path to GS."""

    component = {
        int(uid)
        for uid in available
        if _expected_u2g_capacity_mbps(env, int(uid)) > 0.0
    }
    frontier = list(sorted(component, reverse=True))
    while frontier:
        connected = frontier.pop()
        for candidate in available:
            candidate = int(candidate)
            if candidate in component or candidate == connected:
                continue
            if _expected_u2u_capacity_mbps(env, candidate, connected) > 0.0:
                component.add(candidate)
                frontier.append(candidate)
    return tuple(sorted(component))


def _nearest_forward_distance_target(env, candidate, available):
    candidate_position = tuple(map(float, env.uav_dict[candidate].get_position()))
    gs_id = int(env.GS_ID)
    targets = [(gs_id, tuple(map(float, env.GS_pos)))]
    targets.extend(
        (uid, tuple(map(float, env.uav_dict[uid].get_position())))
        for uid in _gs_connected_component(env, available)
        if uid != candidate
    )
    if not targets:
        return None, None, _maximum_range_gap_m()
    node, position = min(
        targets,
        key=lambda item: (math.dist(candidate_position, item[1]), int(item[0])),
    )
    distance = math.dist(candidate_position, position)
    return int(node), position, max(distance - COMMUNICATION_RANGE_M, 0.0)


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
    total_backlog = math.fsum(backlog.get(uid, 0.0) for uid in sources)
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
    source_centroid = (
        tuple(
            float(
                math.fsum(
                    weights[source]
                    * env.uav_dict[source].get_position()[axis]
                    for source in sources
                )
            )
            for axis in range(3)
        )
        if weights
        else None
    )
    weighted_receive_gap = math.fsum(
        weights[source]
        * max(
            math.dist(
                env.uav_dict[source].get_position(),
                env.uav_dict[candidate].get_position(),
            )
            - COMMUNICATION_RANGE_M,
            0.0,
        )
        for source in sources
    )
    receive_distance_progress = (
        _distance_progress_from_gap(weighted_receive_gap) if sources else 0.0
    )

    path_cost, path = _shortest_forward_path(env, candidate, available, backlog)
    reachable = bool(path and path[-1] == int(env.GS_ID) and math.isfinite(path_cost))
    distance_target_node, distance_target, forward_range_gap = (
        _nearest_forward_distance_target(env, candidate, available)
    )
    forward_distance_progress = _distance_progress_from_gap(forward_range_gap)
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
    receive_direction_target = receive_centroid or source_centroid
    forward_direction_target = target or distance_target
    utility = float(np.clip(min(receive_score, forward_score), 0.0, 1.0))
    movement_receive_score = float(
        RELAY_RECEIVE_CAPACITY_POTENTIAL_WEIGHT * receive_score
        + RELAY_RECEIVE_DISTANCE_POTENTIAL_WEIGHT * receive_distance_progress
    )
    movement_forward_score = float(
        RELAY_FORWARD_PATH_POTENTIAL_WEIGHT * forward_score
        + RELAY_FORWARD_DISTANCE_POTENTIAL_WEIGHT * forward_distance_progress
    )
    movement_utility = float(
        np.clip(min(movement_receive_score, movement_forward_score), 0.0, 1.0)
    )
    values = (
        receive_score,
        forward_score,
        utility,
        receive_distance_progress,
        forward_distance_progress,
        movement_receive_score,
        movement_forward_score,
        movement_utility,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AssertionError("Relay metrics must be finite and within [0, 1]")
    return RelayMetrics(
        candidate_uav_id=candidate,
        receive_score=receive_score,
        forward_score=forward_score,
        utility=utility,
        receive_distance_progress=receive_distance_progress,
        forward_distance_progress=forward_distance_progress,
        movement_receive_score=movement_receive_score,
        movement_forward_score=movement_forward_score,
        movement_utility=movement_utility,
        receive_centroid=receive_centroid,
        forward_target=target,
        receive_direction_target=receive_direction_target,
        forward_direction_target=forward_direction_target,
        forward_distance_target_node=distance_target_node,
        forward_range_gap_m=float(forward_range_gap),
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
