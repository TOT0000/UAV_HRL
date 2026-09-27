"""Side-effect-free auxiliary snapshots for movement replay transitions."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np

from centralized_movement import fov_task_geometry
from experiment_config import (
    MAX_3D_COMMUNICATION_DISTANCE_M,
    NUM_UAV,
    REFERENCE_COM_BANDWIDTH_HZ,
    ROI_COUNT_MAX,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)
from visual_sensing import vs_c10_edge_distances


REPLAY_AUXILIARY_SCHEMA_VERSION = "uav-hrl-joint-replay-aux-v1"
MAX_TASKS_PER_UAV = 2
TASK_TYPE_ENCODING = {
    "NONE": 0,
    "Search": 1,
    "FOV": 2,
    "COM": 3,
    "Hovering": 4,
}
PACKET_TYPE_ENCODING = {"NONE": 0, "FOV": 1, "COM": 2}


def _spec(shape, dtype, unit, semantics, *, mask=None):
    return {
        "shape": tuple(shape),
        "dtype": np.dtype(dtype),
        "unit": unit,
        "semantics": semantics,
        "mask": mask,
    }


SNAPSHOT_FIELD_SPECS = OrderedDict(
    (
        ("snapshot_valid", _spec((1,), np.bool_, None, "snapshot was recorded")),
        ("snapshot_time_s", _spec((1,), np.float32, "seconds", "episode time at the policy observation")),
        ("sr_id", _spec((ROI_COUNT_MAX,), np.int16, None, "SR id; -1 for padding", mask="sr_observable")),
        ("sr_object_exists", _spec((ROI_COUNT_MAX,), np.bool_, None, "an observable SR occupies this row")),
        ("sr_observable", _spec((ROI_COUNT_MAX,), np.bool_, None, "SR and its discovered RoI association are decision-visible")),
        ("sr_position_valid", _spec((ROI_COUNT_MAX,), np.bool_, None, "sr_position_m is valid")),
        ("sr_position_m", _spec((ROI_COUNT_MAX, 3), np.float32, "meters", "observable SR xyz", mask="sr_position_valid")),
        ("sr_roi_id", _spec((ROI_COUNT_MAX,), np.int16, None, "known corresponding discovered RoI id", mask="sr_roi_mapping_valid")),
        ("sr_roi_mapping_valid", _spec((ROI_COUNT_MAX,), np.bool_, None, "SR-to-RoI mapping is decision-visible")),
        ("sr_queue_valid", _spec((ROI_COUNT_MAX,), np.bool_, None, "SR FIFO summary is observable")),
        ("sr_queue_empty", _spec((ROI_COUNT_MAX,), np.bool_, None, "observable SR FIFO is empty", mask="sr_queue_valid")),
        ("sr_backlog_bits", _spec((ROI_COUNT_MAX,), np.float32, "bits", "sum of remaining bits in the SR FIFO", mask="sr_queue_valid")),
        ("sr_packet_count", _spec((ROI_COUNT_MAX,), np.int32, "packets", "active packets in the SR FIFO", mask="sr_queue_valid")),
        ("sr_hol_deadline_valid", _spec((ROI_COUNT_MAX,), np.bool_, None, "SR FIFO HOL deadline value is present")),
        ("sr_hol_remaining_deadline_s", _spec((ROI_COUNT_MAX,), np.float32, "seconds", "absolute E2E deadline minus snapshot time", mask="sr_hol_deadline_valid")),
        ("uav_queue_valid", _spec((NUM_UAV,), np.bool_, None, "UAV aggregate FIFO summary is valid")),
        ("uav_queue_empty", _spec((NUM_UAV,), np.bool_, None, "UAV aggregate FIFO is empty", mask="uav_queue_valid")),
        ("uav_backlog_bits", _spec((NUM_UAV,), np.float32, "bits", "unclipped aggregate FIFO remaining bits", mask="uav_queue_valid")),
        ("uav_vs_backlog_bits", _spec((NUM_UAV,), np.float32, "bits", "FOV packet remaining bits", mask="uav_queue_valid")),
        ("uav_com_backlog_bits", _spec((NUM_UAV,), np.float32, "bits", "COM packet remaining bits", mask="uav_queue_valid")),
        ("uav_vs_packet_count", _spec((NUM_UAV,), np.int32, "packets", "FOV packets in aggregate FIFO", mask="uav_queue_valid")),
        ("uav_com_packet_count", _spec((NUM_UAV,), np.int32, "packets", "COM packets in aggregate FIFO", mask="uav_queue_valid")),
        ("uav_hol_type", _spec((NUM_UAV,), np.int8, None, "NONE=0,FOV=1,COM=2", mask="uav_hol_valid")),
        ("uav_hol_valid", _spec((NUM_UAV,), np.bool_, None, "actual aggregate FIFO HOL packet exists")),
        ("uav_hol_remaining_deadline_s", _spec((NUM_UAV,), np.float32, "seconds", "HOL absolute E2E deadline minus snapshot time", mask="uav_hol_valid")),
        ("uav_vs_min_deadline_valid", _spec((NUM_UAV,), np.bool_, None, "minimum FOV deadline is present")),
        ("uav_vs_min_remaining_deadline_s", _spec((NUM_UAV,), np.float32, "seconds", "minimum FOV absolute deadline minus snapshot time", mask="uav_vs_min_deadline_valid")),
        ("uav_com_min_deadline_valid", _spec((NUM_UAV,), np.bool_, None, "minimum COM deadline is present")),
        ("uav_com_min_remaining_deadline_s", _spec((NUM_UAV,), np.float32, "seconds", "minimum COM absolute deadline minus snapshot time", mask="uav_com_min_deadline_valid")),
        ("roi_id", _spec((ROI_COUNT_MAX,), np.int16, None, "discovered RoI id; -1 for padding", mask="roi_observable")),
        ("roi_object_exists", _spec((ROI_COUNT_MAX,), np.bool_, None, "a decision-visible discovered RoI occupies this row")),
        ("roi_observable", _spec((ROI_COUNT_MAX,), np.bool_, None, "RoI has been discovered; unknown RoIs are not serialized")),
        ("roi_position_m", _spec((ROI_COUNT_MAX, 3), np.float32, "meters", "discovered RoI xyz", mask="roi_observable")),
        ("task_type", _spec((NUM_UAV, MAX_TASKS_PER_UAV), np.int8, None, "task type numeric code", mask="task_pair_valid")),
        ("task_target_id", _spec((NUM_UAV, MAX_TASKS_PER_UAV), np.int16, None, "FOV RoI id or COM SR id; -1 when absent", mask="task_target_valid")),
        ("task_pair_valid", _spec((NUM_UAV, MAX_TASKS_PER_UAV), np.bool_, None, "assignment slot contains a task")),
        ("task_target_valid", _spec((NUM_UAV, MAX_TASKS_PER_UAV), np.bool_, None, "assignment has a decision-visible service target")),
        ("vs_pair_valid", _spec((NUM_UAV,), np.bool_, None, "assigned UAV-RoI VS pair exists")),
        ("vs_uav_id", _spec((NUM_UAV,), np.int16, None, "UAV endpoint id", mask="vs_pair_valid")),
        ("vs_roi_id", _spec((NUM_UAV,), np.int16, None, "discovered RoI endpoint id", mask="vs_pair_valid")),
        ("vs_coverage_ratio", _spec((NUM_UAV,), np.float32, "ratio", "canonical circular-RoI coverage", mask="vs_pair_valid")),
        ("vs_image_quantity", _spec((NUM_UAV,), np.float32, "ratio", "raw unclipped image quantity I", mask="vs_pair_valid")),
        ("vs_relative_altitude_m", _spec((NUM_UAV,), np.float32, "meters", "h", mask="vs_pair_valid")),
        ("vs_horizontal_distance_m", _spec((NUM_UAV,), np.float32, "meters", "d", mask="vs_pair_valid")),
        ("vs_c9_margin_m", _spec((NUM_UAV,), np.float32, "meters", "b1*h-d; non-negative satisfies C9", mask="vs_pair_valid")),
        ("vs_geometry_valid", _spec((NUM_UAV,), np.bool_, None, "C9 model range is valid")),
        ("vs_c10_geometry_valid", _spec((NUM_UAV,), np.bool_, None, "finite canonical d_L and d_R exist")),
        ("vs_d_left_m", _spec((NUM_UAV,), np.float32, "meters", "canonical C10 d_L", mask="vs_c10_geometry_valid")),
        ("vs_d_right_m", _spec((NUM_UAV,), np.float32, "meters", "canonical C10 d_R", mask="vs_c10_geometry_valid")),
        ("vs_c10_margin_m", _spec((NUM_UAV,), np.float32, "meters", "min(d_L,d_R)-RoI radius; non-negative satisfies C10", mask="vs_c10_geometry_valid")),
        ("vs_capture_valid", _spec((NUM_UAV,), np.bool_, None, "canonical sensing_valid_now flag")),
        ("u2u_link_valid", _spec((NUM_UAV, NUM_UAV), np.bool_, None, "directed non-self endpoint pair and cached decision CSI are valid")),
        ("u2u_distance_m", _spec((NUM_UAV, NUM_UAV), np.float32, "meters", "decision-time 3-D distance", mask="u2u_link_valid")),
        ("u2u_in_range", _spec((NUM_UAV, NUM_UAV), np.bool_, None, "inclusive canonical range test", mask="u2u_link_valid")),
        ("u2u_reference_capacity_mbps", _spec((NUM_UAV, NUM_UAV), np.float32, "Mbps", "cached expected capacity at full reference bandwidth, not scheduled service", mask="u2u_link_valid")),
        ("u2g_link_valid", _spec((NUM_UAV,), np.bool_, None, "UAV-GS endpoint and cached decision CSI are valid")),
        ("u2g_distance_m", _spec((NUM_UAV,), np.float32, "meters", "decision-time 3-D distance", mask="u2g_link_valid")),
        ("u2g_in_range", _spec((NUM_UAV,), np.bool_, None, "inclusive canonical range test", mask="u2g_link_valid")),
        ("u2g_reference_capacity_mbps", _spec((NUM_UAV,), np.float32, "Mbps", "cached expected capacity at full reference bandwidth, not scheduled service", mask="u2g_link_valid")),
        ("s2u_link_valid", _spec((ROI_COUNT_MAX, NUM_UAV), np.bool_, None, "observable SR-UAV endpoint pair and cached large-scale state are valid")),
        ("s2u_distance_m", _spec((ROI_COUNT_MAX, NUM_UAV), np.float32, "meters", "decision-time 3-D distance", mask="s2u_link_valid")),
        ("s2u_in_range", _spec((ROI_COUNT_MAX, NUM_UAV), np.bool_, None, "inclusive canonical range test", mask="s2u_link_valid")),
        ("s2u_reference_capacity_mbps", _spec((ROI_COUNT_MAX, NUM_UAV), np.float32, "Mbps", "expected S2U capacity at reference COM bandwidth, not scheduled service", mask="s2u_link_valid")),
    )
)

TRANSITION_FIELD_SPECS = OrderedDict(
    (
        ("auxiliary_valid", _spec((1,), np.bool_, None, "both current and next snapshots were recorded")),
        ("episode_id", _spec((1,), np.int32, None, "zero-based episode id")),
        ("td3_step", _spec((1,), np.int32, None, "zero-based movement decision within episode")),
        ("global_transition_id", _spec((1,), np.int64, None, "monotonic transition id in the producing run")),
        ("scenario_index", _spec((1,), np.int32, None, "manifest episode index")),
        ("scenario_id_hash", _spec((1,), np.int64, None, "stable 63-bit hash; metadata maps index to full id")),
        ("dinkelbach_lambda", _spec((1,), np.float64, None, "lambda used for this transition")),
    )
)

REPLAY_AUXILIARY_FIELDS = tuple(
    [f"current_{name}" for name in SNAPSHOT_FIELD_SPECS]
    + [f"next_{name}" for name in SNAPSHOT_FIELD_SPECS]
    + list(TRANSITION_FIELD_SPECS)
)


def empty_snapshot():
    snapshot = {
        name: np.zeros(spec["shape"], dtype=spec["dtype"])
        for name, spec in SNAPSHOT_FIELD_SPECS.items()
    }
    for name in ("sr_id", "sr_roi_id", "roi_id", "task_target_id", "vs_uav_id", "vs_roi_id"):
        snapshot[name].fill(-1)
    return snapshot


def stable_scenario_id_hash(value):
    if value is None:
        return -1
    raw = hashlib.sha256(str(value).encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big", signed=False) & ((1 << 63) - 1)


def _active_packets(queue):
    return [packet for packet in queue if packet is not None and not packet.get("done", False)]


def _remaining_deadline(packet, snapshot_time):
    value = float(packet["deadline_abs"]) - float(snapshot_time)
    if not math.isfinite(value):
        raise RuntimeError("packet absolute deadline is not finite")
    return value


def capture_replay_snapshot(env, packet_engine, snapshot_time_s):
    """Capture policy-time observable state without mutating env, channel, or RNG."""

    snapshot_time = float(snapshot_time_s)
    if not math.isfinite(snapshot_time) or snapshot_time < 0.0:
        raise ValueError("snapshot time must be finite and non-negative")
    result = empty_snapshot()
    result["snapshot_valid"][0] = True
    result["snapshot_time_s"][0] = snapshot_time

    discovered_ids = {
        int(gt.id) for gt in env.gts if bool(getattr(gt, "is_found", False))
    }
    for row, gt in enumerate(sorted(
        (gt for gt in env.gts if int(gt.id) in discovered_ids),
        key=lambda item: int(item.id),
    )):
        if row >= ROI_COUNT_MAX:
            raise RuntimeError("discovered RoI count exceeds replay padding")
        result["roi_id"][row] = int(gt.id)
        result["roi_object_exists"][row] = True
        result["roi_observable"][row] = True
        result["roi_position_m"][row] = np.asarray(gt.get_position(), dtype=np.float32)

    for sr in sorted(env.SR_teams, key=lambda item: int(item.id)):
        sr_id = int(sr.id)
        roi_id = getattr(sr, "assigned_gt_id", None)
        observable = roi_id is not None and int(roi_id) in discovered_ids
        if not observable:
            continue
        if not 0 <= sr_id < ROI_COUNT_MAX:
            raise RuntimeError("observable SR id exceeds replay padding")
        row = sr_id
        result["sr_id"][row] = sr_id
        result["sr_object_exists"][row] = True
        result["sr_observable"][row] = True
        result["sr_position_valid"][row] = True
        result["sr_position_m"][row] = np.asarray(sr.get_position(), dtype=np.float32)
        result["sr_roi_id"][row] = int(roi_id)
        result["sr_roi_mapping_valid"][row] = True
        packets = _active_packets(packet_engine.sr_queues[sr_id])
        result["sr_queue_valid"][row] = True
        result["sr_queue_empty"][row] = not packets
        result["sr_backlog_bits"][row] = math.fsum(
            max(float(packet.get("rem_bits", 0.0)), 0.0) for packet in packets
        )
        result["sr_packet_count"][row] = len(packets)
        if packets:
            result["sr_hol_deadline_valid"][row] = True
            result["sr_hol_remaining_deadline_s"][row] = _remaining_deadline(
                packets[0], snapshot_time
            )

    for uav_id in range(NUM_UAV):
        packets = _active_packets(packet_engine.uav_queues[uav_id])
        result["uav_queue_valid"][uav_id] = True
        result["uav_queue_empty"][uav_id] = not packets
        by_type = {"FOV": [], "COM": []}
        for packet in packets:
            task_type = "FOV" if packet.get("task_type") == "FOV" else "COM"
            by_type[task_type].append(packet)
        for task_type, prefix in (("FOV", "vs"), ("COM", "com")):
            typed = by_type[task_type]
            bits = math.fsum(
                max(float(packet.get("rem_bits", 0.0)), 0.0) for packet in typed
            )
            result[f"uav_{prefix}_backlog_bits"][uav_id] = bits
            result[f"uav_{prefix}_packet_count"][uav_id] = len(typed)
            if typed:
                result[f"uav_{prefix}_min_deadline_valid"][uav_id] = True
                result[f"uav_{prefix}_min_remaining_deadline_s"][uav_id] = min(
                    _remaining_deadline(packet, snapshot_time) for packet in typed
                )
        result["uav_backlog_bits"][uav_id] = (
            result["uav_vs_backlog_bits"][uav_id]
            + result["uav_com_backlog_bits"][uav_id]
        )
        if packets:
            hol = packets[0]
            result["uav_hol_valid"][uav_id] = True
            result["uav_hol_type"][uav_id] = PACKET_TYPE_ENCODING[
                "FOV" if hol.get("task_type") == "FOV" else "COM"
            ]
            result["uav_hol_remaining_deadline_s"][uav_id] = _remaining_deadline(
                hol, snapshot_time
            )

        def task_target_id(task):
            target_id = task.get("target_obj_id")
            return -1 if target_id is None else int(target_id)

        tasks = sorted(
            env.multi_tasks.get(uav_id, ()),
            key=lambda task: (
                TASK_TYPE_ENCODING.get(task.get("task_type"), 127),
                task_target_id(task),
            ),
        )
        if len(tasks) > MAX_TASKS_PER_UAV:
            raise RuntimeError("UAV task count exceeds replay padding")
        for slot, task in enumerate(tasks):
            task_type = str(task.get("task_type"))
            if task_type not in TASK_TYPE_ENCODING:
                raise RuntimeError(f"unsupported replay task type: {task_type}")
            result["task_type"][uav_id, slot] = TASK_TYPE_ENCODING[task_type]
            result["task_pair_valid"][uav_id, slot] = True
            if task_type in {"FOV", "COM"}:
                target_id = int(task["target_obj_id"])
                if task_type == "FOV" and target_id not in discovered_ids:
                    raise RuntimeError("assigned FOV task exposes an undiscovered RoI")
                result["task_target_id"][uav_id, slot] = target_id
                result["task_target_valid"][uav_id, slot] = True
            if task_type == "FOV":
                geometry = fov_task_geometry(env, uav_id, task)
                target = env.gts[int(task["target_obj_id"])]
                result["vs_pair_valid"][uav_id] = True
                result["vs_uav_id"][uav_id] = uav_id
                result["vs_roi_id"][uav_id] = int(target.id)
                result["vs_coverage_ratio"][uav_id] = float(geometry.coverage_ratio)
                result["vs_image_quantity"][uav_id] = float(geometry.image_quantity)
                result["vs_relative_altitude_m"][uav_id] = float(geometry.relative_altitude)
                result["vs_horizontal_distance_m"][uav_id] = float(geometry.horizontal_distance)
                result["vs_c9_margin_m"][uav_id] = float(
                    geometry.b1 * geometry.relative_altitude
                    - geometry.horizontal_distance
                )
                result["vs_geometry_valid"][uav_id] = bool(geometry.model_range_valid)
                result["vs_capture_valid"][uav_id] = bool(geometry.sensing_valid_now)
                edges = vs_c10_edge_distances(geometry)
                if edges is not None:
                    d_left, d_right = map(float, edges)
                    result["vs_c10_geometry_valid"][uav_id] = True
                    result["vs_d_left_m"][uav_id] = d_left
                    result["vs_d_right_m"][uav_id] = d_right
                    result["vs_c10_margin_m"][uav_id] = (
                        min(d_left, d_right) - float(target.radius)
                    )

    positions = np.asarray(
        [env.uav_dict[uav_id].get_position() for uav_id in range(NUM_UAV)],
        dtype=np.float64,
    )
    if positions.shape == (NUM_UAV, 3) and np.isfinite(positions).all():
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        cached = np.asarray(getattr(env, "u2u_nominal_capacity", ()), dtype=float)
        if cached.shape == (NUM_UAV, NUM_UAV) and np.isfinite(cached).all():
            valid = ~np.eye(NUM_UAV, dtype=bool)
            result["u2u_link_valid"][:] = valid
            result["u2u_distance_m"][:] = distances.astype(np.float32)
            result["u2u_in_range"][:] = valid & (
                distances <= MAX_3D_COMMUNICATION_DISTANCE_M
            )
            result["u2u_reference_capacity_mbps"][:] = cached.astype(np.float32)

        gs_position = np.asarray(env.GS_pos, dtype=np.float64)
        cached_u2g = np.asarray(getattr(env, "u2g_nominal_capacity", ()), dtype=float)
        if (
            gs_position.shape == (3,)
            and np.isfinite(gs_position).all()
            and cached_u2g.shape == (NUM_UAV,)
            and np.isfinite(cached_u2g).all()
        ):
            distances_gs = np.linalg.norm(positions - gs_position[None, :], axis=1)
            result["u2g_link_valid"][:] = True
            result["u2g_distance_m"][:] = distances_gs.astype(np.float32)
            result["u2g_in_range"][:] = (
                distances_gs <= MAX_3D_COMMUNICATION_DISTANCE_M
            )
            result["u2g_reference_capacity_mbps"][:] = cached_u2g.astype(np.float32)

    for sr_id in np.flatnonzero(result["sr_observable"]):
        sr_position = np.asarray(env.SR_teams[int(sr_id)].get_position(), dtype=float)
        for uav_id in range(NUM_UAV):
            try:
                capacity = float(env.get_sr_uav_reference_capacity_mbps(uav_id, sr_id))
            except (IndexError, RuntimeError, TypeError, ValueError):
                continue
            distance = float(np.linalg.norm(positions[uav_id] - sr_position))
            if not math.isfinite(capacity) or not math.isfinite(distance):
                continue
            result["s2u_link_valid"][sr_id, uav_id] = True
            result["s2u_distance_m"][sr_id, uav_id] = distance
            result["s2u_in_range"][sr_id, uav_id] = (
                distance <= MAX_3D_COMMUNICATION_DISTANCE_M
            )
            result["s2u_reference_capacity_mbps"][sr_id, uav_id] = capacity

    return {name: value.copy() for name, value in result.items()}


def replay_auxiliary_metadata():
    def public(spec):
        return {
            "shape_per_transition": list(spec["shape"]),
            "dtype": spec["dtype"].str,
            "unit": spec["unit"],
            "semantics": spec["semantics"],
            "mask": spec["mask"],
        }

    current_next = {
        name: public(spec) for name, spec in SNAPSHOT_FIELD_SPECS.items()
    }
    transition = {
        name: public(spec) for name, spec in TRANSITION_FIELD_SPECS.items()
    }
    bytes_per_snapshot = sum(
        int(np.prod(spec["shape"], dtype=np.int64)) * spec["dtype"].itemsize
        for spec in SNAPSHOT_FIELD_SPECS.values()
    )
    bytes_per_transition = 2 * bytes_per_snapshot + sum(
        int(np.prod(spec["shape"], dtype=np.int64)) * spec["dtype"].itemsize
        for spec in TRANSITION_FIELD_SPECS.values()
    )
    return {
        "schema_version": REPLAY_AUXILIARY_SCHEMA_VERSION,
        "snapshot_alignment": {
            "current": "same decision boundary as state, before movement action",
            "next": "same boundary as next_state, after service and any next-boundary reassignment",
        },
        "current_and_next_fields": current_next,
        "transition_fields": transition,
        "task_type_encoding": TASK_TYPE_ENCODING,
        "packet_type_encoding": PACKET_TYPE_ENCODING,
        "link_axis_encoding": {
            "u2u": "[sender_uav_id, receiver_uav_id]",
            "u2g": "[sender_uav_id], receiver is the fixed GS",
            "s2u": "[sr_id, receiver_uav_id]",
        },
        "link_capacity_contract": {
            "u2u_u2g_bandwidth_hz": TOTAL_COMMUNICATION_BANDWIDTH_HZ,
            "s2u_bandwidth_hz": REFERENCE_COM_BANDWIDTH_HZ,
            "meaning": "decision-time expected reference capacity; not scheduling or realized service",
            "rng_draws": "none; consumes cached large-scale state only",
        },
        "unknown_roi_policy": "undiscovered RoI ids and positions are not serialized",
        "fixed_roi_radius_storage": "metadata only; per-pair radius is not repeated",
        "bytes_per_snapshot": int(bytes_per_snapshot),
        "auxiliary_bytes_per_transition": int(bytes_per_transition),
    }


def allocate_auxiliary_arrays(max_size):
    arrays = {}
    for prefix in ("current_", "next_"):
        for name, spec in SNAPSHOT_FIELD_SPECS.items():
            arrays[prefix + name] = np.zeros(
                (int(max_size), *spec["shape"]), dtype=spec["dtype"]
            )
    for name, spec in TRANSITION_FIELD_SPECS.items():
        arrays[name] = np.zeros(
            (int(max_size), *spec["shape"]), dtype=spec["dtype"]
        )
    for prefix in ("current_", "next_"):
        for name in ("sr_id", "sr_roi_id", "roi_id", "task_target_id", "vs_uav_id", "vs_roi_id"):
            arrays[prefix + name].fill(-1)
    arrays["episode_id"].fill(-1)
    arrays["td3_step"].fill(-1)
    arrays["global_transition_id"].fill(-1)
    arrays["scenario_index"].fill(-1)
    arrays["scenario_id_hash"].fill(-1)
    return arrays


def write_auxiliary_transition(
    replay,
    index,
    current_snapshot,
    next_snapshot,
    *,
    episode_id=-1,
    td3_step=-1,
    global_transition_id=-1,
    scenario_index=-1,
    scenario_id=None,
    dinkelbach_lambda=0.0,
):
    index = int(index)
    for name in REPLAY_AUXILIARY_FIELDS:
        target = getattr(replay, name)
        target[index] = False if target.dtype == np.bool_ else 0
    for prefix in ("current_", "next_"):
        for name in ("sr_id", "sr_roi_id", "roi_id", "task_target_id", "vs_uav_id", "vs_roi_id"):
            getattr(replay, prefix + name)[index].fill(-1)
    replay.episode_id[index, 0] = int(episode_id)
    replay.td3_step[index, 0] = int(td3_step)
    replay.global_transition_id[index, 0] = int(global_transition_id)
    replay.scenario_index[index, 0] = int(scenario_index)
    replay.scenario_id_hash[index, 0] = stable_scenario_id_hash(scenario_id)
    replay.dinkelbach_lambda[index, 0] = float(dinkelbach_lambda)
    if current_snapshot is None or next_snapshot is None:
        return
    for prefix, snapshot in (("current_", current_snapshot), ("next_", next_snapshot)):
        if set(snapshot) != set(SNAPSHOT_FIELD_SPECS):
            raise ValueError("auxiliary snapshot fields do not match the active schema")
        for name, spec in SNAPSHOT_FIELD_SPECS.items():
            value = np.asarray(snapshot[name], dtype=spec["dtype"])
            if value.shape != spec["shape"]:
                raise ValueError(
                    f"auxiliary field {name} has shape {value.shape}, expected {spec['shape']}"
                )
            getattr(replay, prefix + name)[index] = value
    replay.auxiliary_valid[index, 0] = bool(
        current_snapshot["snapshot_valid"][0]
        and next_snapshot["snapshot_valid"][0]
    )
