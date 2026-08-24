"""Centralized full/masked observation strategies for controlled baselines."""

from __future__ import annotations

import numpy as np
import torch

from centralized_movement import MOVEMENT_STATE_DIM, movement_state_feature_schema
from experiment_config import NUM_UAV


ROUTING_STATE_DIM = 6 * NUM_UAV + 30
_MOVEMENT_TASK_SUFFIXES = {
    "task_search",
    "task_fov",
    "task_com",
    "task_hovering",
    "fov_error",
    "fov_target_x",
    "fov_target_y",
    "fov_target_z",
    "com_target_x",
    "com_target_y",
    "com_target_z",
    "com_capacity",
}


def routing_state_feature_names():
    """Return the authoritative derived task-aware routing layout."""

    names = [f"uav_id_one_hot[{index}]" for index in range(NUM_UAV)]
    names.extend(
        [
            "energy",
            "backlog",
            "assigned_task_search",
            "assigned_task_fov",
            "assigned_task_com",
            "assigned_task_hovering",
            "assigned_task_contains_fov",
            "assigned_source_uav",
        ]
    )
    for group in (
        "effective_action_mask",
        "link_delay",
        "link_capacity",
        "next_hop_backlog",
        "next_hop_assigned_fov",
    ):
        names.extend(f"{group}[{index}]" for index in range(NUM_UAV + 1))
    names.extend(
        [
            "position_x",
            "position_y",
            "position_z",
            "assigned_task_altitude_error",
            "vertical_velocity",
            "assigned_fov_quality",
            "assigned_fov_error",
            "distance_to_ground_station",
            "eta_to_ground_station",
            "dinkelbach_lambda",
            "coverage_overlap_ema",
            "coverage_unvisited_ema",
            "coverage_frontier_ema",
            "hol_assigned_task_fov",
            "hol_assigned_task_com",
            "hol_deadline_remaining",
            "hol_packet_fraction_remaining",
        ]
    )
    if len(names) != ROUTING_STATE_DIM:
        raise AssertionError(
            f"routing feature layout does not cover {ROUTING_STATE_DIM} dimensions"
        )
    return tuple(names)


def _movement_task_indices():
    schema = movement_state_feature_schema()
    return tuple(
        item["index"]
        for item in schema["features"]
        if item["name"].rsplit(".", 1)[-1] in _MOVEMENT_TASK_SUFFIXES
    )


def _routing_task_indices():
    prefixes = (
        "assigned_task_",
        "primary_task_",
        "assigned_source_",
        "next_hop_assigned_",
        "hol_assigned_",
    )
    return tuple(
        index
        for index, name in enumerate(routing_state_feature_names())
        if name.startswith(prefixes) or name.startswith("assigned_fov_")
    )


MOVEMENT_TASK_ASSIGNMENT_INDICES = _movement_task_indices()
ROUTING_TASK_ASSIGNMENT_INDICES = _routing_task_indices()


def _contiguous_ranges(indices):
    ranges = []
    for index in indices:
        if not ranges or index != ranges[-1][1] + 1:
            ranges.append([index, index])
        else:
            ranges[-1][1] = index
    return tuple(tuple(item) for item in ranges)


def masked_observation_metadata():
    movement_schema = movement_state_feature_schema()["features"]
    routing_names = routing_state_feature_names()
    return {
        "movement": {
            "dimension": MOVEMENT_STATE_DIM,
            "indices": list(MOVEMENT_TASK_ASSIGNMENT_INDICES),
            "index_ranges_inclusive": [
                list(item) for item in _contiguous_ranges(MOVEMENT_TASK_ASSIGNMENT_INDICES)
            ],
            "fields": [
                movement_schema[index]["name"]
                for index in MOVEMENT_TASK_ASSIGNMENT_INDICES
            ],
        },
        "routing": {
            "dimension": ROUTING_STATE_DIM,
            "indices": list(ROUTING_TASK_ASSIGNMENT_INDICES),
            "index_ranges_inclusive": [
                list(item) for item in _contiguous_ranges(ROUTING_TASK_ASSIGNMENT_INDICES)
            ],
            "fields": [routing_names[index] for index in ROUTING_TASK_ASSIGNMENT_INDICES],
        },
    }


def apply_observation_strategy(state, mode, observation_kind):
    """Return a copy with only direct task-assignment fields neutralized."""

    if mode not in {"full", "masked"}:
        raise ValueError(f"unsupported task observation mode: {mode}")
    if observation_kind == "movement":
        dimension = MOVEMENT_STATE_DIM
        indices = MOVEMENT_TASK_ASSIGNMENT_INDICES
    elif observation_kind == "routing":
        dimension = ROUTING_STATE_DIM
        indices = ROUTING_TASK_ASSIGNMENT_INDICES
    else:
        raise ValueError(f"unsupported observation kind: {observation_kind}")
    if state.shape[-1] != dimension:
        raise ValueError(
            f"{observation_kind} state must end in {dimension} features"
        )
    if torch.is_tensor(state):
        output = state.clone()
        if mode == "masked":
            output[..., list(indices)] = 0
        return output
    output = np.array(state, copy=True)
    if mode == "masked":
        output[..., list(indices)] = 0
    return output
