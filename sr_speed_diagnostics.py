"""Evaluation-only SR movement and dual-task service diagnostics."""

import csv
import json
from pathlib import Path

from visual_sensing import vs_geometry


SR_SPEED_DIAGNOSTIC_SCHEMA_VERSION = "sr-speed-evaluation-diagnostics-v2"
DUAL_TASK_SAMPLING = "routing-slot state, integrated at 0.25-second resolution"


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _mean(total, count):
    return float(total) / float(count) if count else None


def _event_elapsed(event, event_name):
    """Resolve elapsed time only when the discrete observation order supports it."""

    event_time = event.get(f"{event_name}_time_seconds")
    if event_time is None:
        return {
            "elapsed_seconds": None,
            "valid": False,
            "reason": "event_not_observed",
            "basis": None,
        }
    assignment_time = float(event["assignment_time_seconds"])
    event_time = float(event_time)
    source = event.get(f"{event_name}_event_source")
    if source == "assignment_state":
        return {
            "elapsed_seconds": 0.0,
            "valid": True,
            "reason": "already_satisfied_at_assignment",
            "basis": "assignment_state_observation",
        }
    if event_time < assignment_time:
        return {
            "elapsed_seconds": None,
            "valid": False,
            "reason": "event_observation_precedes_assignment",
            "basis": "discrete_state_observation",
        }
    if event_time == assignment_time:
        return {
            "elapsed_seconds": None,
            "valid": False,
            "reason": "same_timestamp_discrete_movement_after_assignment",
            "basis": "discrete_state_observation",
        }
    return {
        "elapsed_seconds": event_time - assignment_time,
        "valid": True,
        "reason": "observed_at_later_discrete_movement_update",
        "basis": "discrete_state_observation",
    }


def build_sr_episode_diagnostics(env, *, episode_index, scenario_id, episode_end_s):
    """Build per-SR rows and a pooled episode summary from executed state."""

    rows = []
    for sr in sorted(env.SR_teams, key=lambda item: int(item.id)):
        event = env.sr_movement_events.get(int(sr.id))
        assigned = event is not None
        roi_id = int(event["assigned_roi_id"]) if assigned else None
        target = env.gts[roi_id] if assigned else None
        remaining = (
            ((float(sr.x) - float(target.x)) ** 2 + (float(sr.y) - float(target.y)) ** 2)
            ** 0.5
            if assigned
            else None
        )
        assignment_time = event["assignment_time_seconds"] if assigned else None
        entered_time = event["entered_roi_time_seconds"] if assigned else None
        arrived_time = event["arrived_center_time_seconds"] if assigned else None
        entered_elapsed = (
            _event_elapsed(event, "entered_roi")
            if assigned
            else {
                "elapsed_seconds": None,
                "valid": False,
                "reason": "sr_not_assigned",
                "basis": None,
            }
        )
        arrived_elapsed = (
            _event_elapsed(event, "arrived_center")
            if assigned
            else {
                "elapsed_seconds": None,
                "valid": False,
                "reason": "sr_not_assigned",
                "basis": None,
            }
        )
        rows.append(
            {
                "schema_version": SR_SPEED_DIAGNOSTIC_SCHEMA_VERSION,
                "episode_index": int(episode_index),
                "episode_number": int(episode_index) + 1,
                "scenario_id": scenario_id,
                "sr_id": int(sr.id),
                "assigned_roi_id": roi_id,
                "sr_speed_mps": float(env.sr_speed_mps),
                "received_rescue_mission": bool(assigned),
                "mission_assignment_time_seconds": assignment_time,
                "mission_assignment_movement_update_index": (
                    event.get("assignment_movement_update_index") if assigned else None
                ),
                "mission_assignment_position_m": (
                    event["assignment_position_m"] if assigned else None
                ),
                "mission_assignment_distance_to_roi_center_m": (
                    event["assignment_distance_to_roi_center_m"] if assigned else None
                ),
                "entered_roi": entered_time is not None,
                "first_entered_roi_time_seconds": entered_time,
                "first_entered_roi_observation_time_seconds": entered_time,
                "entered_roi_event_source": (
                    event.get("entered_roi_event_source") if assigned else None
                ),
                "entered_roi_movement_update_index": (
                    event.get("entered_roi_movement_update_index") if assigned else None
                ),
                "arrived_center": arrived_time is not None,
                "first_arrived_center_time_seconds": arrived_time,
                "first_arrived_center_observation_time_seconds": arrived_time,
                "arrived_center_event_source": (
                    event.get("arrived_center_event_source") if assigned else None
                ),
                "arrived_center_movement_update_index": (
                    event.get("arrived_center_movement_update_index") if assigned else None
                ),
                "assignment_to_roi_entry_seconds": entered_elapsed["elapsed_seconds"],
                "assignment_to_roi_entry_time_valid": entered_elapsed["valid"],
                "assignment_to_roi_entry_time_reason": entered_elapsed["reason"],
                "assignment_to_roi_entry_time_basis": entered_elapsed["basis"],
                "assignment_to_center_arrival_seconds": arrived_elapsed[
                    "elapsed_seconds"
                ],
                "assignment_to_center_arrival_time_valid": arrived_elapsed["valid"],
                "assignment_to_center_arrival_time_reason": arrived_elapsed["reason"],
                "assignment_to_center_arrival_time_basis": arrived_elapsed["basis"],
                "assignment_observation_resolution_seconds": float(env.dt),
                "movement_state_update_resolution_seconds": float(
                    env.sr_update_seconds
                ),
                "episode_end_time_seconds": float(episode_end_s),
                "episode_end_position_m": [float(sr.x), float(sr.y), float(sr.z)],
                "episode_end_remaining_distance_to_roi_center_m": remaining,
            }
        )
    return rows, summarize_sr_rows(rows, episode_index=episode_index, scenario_id=scenario_id)


def summarize_sr_rows(rows, *, episode_index=None, scenario_id=None):
    assigned = [row for row in rows if row["received_rescue_mission"]]
    entered = [row for row in assigned if row["entered_roi"]]
    arrived = [row for row in assigned if row["arrived_center"]]
    valid_entry_times = [
        row for row in entered if row["assignment_to_roi_entry_time_valid"]
    ]
    valid_arrival_times = [
        row for row in arrived if row["assignment_to_center_arrival_time_valid"]
    ]
    entry_sum = sum(
        float(row["assignment_to_roi_entry_seconds"])
        for row in valid_entry_times
    )
    arrival_sum = sum(
        float(row["assignment_to_center_arrival_seconds"])
        for row in valid_arrival_times
    )
    return {
        "schema_version": SR_SPEED_DIAGNOSTIC_SCHEMA_VERSION,
        "episode_index": episode_index,
        "episode_number": None if episode_index is None else int(episode_index) + 1,
        "scenario_id": scenario_id,
        "total_sr_team_count": len(rows),
        "assigned_sr_count": len(assigned),
        "unassigned_sr_count": len(rows) - len(assigned),
        "event_ratio_denominator_assigned_sr_count": len(assigned),
        "entered_roi_count": len(entered),
        "arrived_center_count": len(arrived),
        "entered_roi_ratio": _ratio(len(entered), len(assigned)),
        "arrived_center_ratio": _ratio(len(arrived), len(assigned)),
        "roi_entry_elapsed_seconds_sum": entry_sum,
        "roi_entry_elapsed_seconds_sample_count": len(valid_entry_times),
        "roi_entry_elapsed_seconds_unresolved_event_count": (
            len(entered) - len(valid_entry_times)
        ),
        "mean_assignment_to_roi_entry_seconds": _mean(
            entry_sum, len(valid_entry_times)
        ),
        "mean_assignment_to_roi_entry_sample_scope": (
            "entered events with valid observed discrete-state elapsed time"
        ),
        "center_arrival_elapsed_seconds_sum": arrival_sum,
        "center_arrival_elapsed_seconds_sample_count": len(valid_arrival_times),
        "center_arrival_elapsed_seconds_unresolved_event_count": (
            len(arrived) - len(valid_arrival_times)
        ),
        "mean_assignment_to_center_arrival_seconds": _mean(
            arrival_sum, len(valid_arrival_times)
        ),
        "mean_assignment_to_center_arrival_sample_scope": (
            "arrival events with valid observed discrete-state elapsed time"
        ),
        "zero_assigned_denominator": len(assigned) == 0,
    }


def empty_dual_task_accumulator():
    return {
        group: {
            "dual_task_uav_seconds": 0.0,
            "both_valid_uav_seconds": 0.0,
            "vs_only_uav_seconds": 0.0,
            "com_only_uav_seconds": 0.0,
            "neither_valid_uav_seconds": 0.0,
        }
        for group in ("all", "same_roi", "different_roi")
    }


def observe_dual_task_service(env, accumulator, duration_seconds):
    """Integrate authoritative VS geometry and queue-independent S2U feasibility."""

    duration = float(duration_seconds)
    for uav_id in range(env.num_UAV):
        tasks = env.multi_tasks.get(uav_id, ())
        fov = next((task for task in tasks if task.get("task_type") == "FOV"), None)
        com = next((task for task in tasks if task.get("task_type") == "COM"), None)
        if fov is None or com is None:
            continue
        roi_id = int(fov["target_obj_id"])
        sr_id = int(com["target_obj_id"])
        sr = env.SR_teams[sr_id]
        com_roi_id = sr.assigned_gt_id
        group = "same_roi" if com_roi_id == roi_id else "different_roi"
        gt = env.gts[roi_id]
        vs_valid = bool(
            vs_geometry(
                env.uav_dict[uav_id].get_position(),
                gt.get_position(),
                gt.radius,
            ).sensing_valid_now
        )
        com_valid = bool(
            com_roi_id is not None and env.is_s2u_in_range(sr_id, uav_id)
        )
        category = (
            "both_valid_uav_seconds"
            if vs_valid and com_valid
            else "vs_only_uav_seconds"
            if vs_valid
            else "com_only_uav_seconds"
            if com_valid
            else "neither_valid_uav_seconds"
        )
        for selected_group in ("all", group):
            accumulator[selected_group]["dual_task_uav_seconds"] += duration
            accumulator[selected_group][category] += duration


def summarize_dual_task_accumulator(accumulator, *, episode_index=None, scenario_id=None):
    rows = []
    for group in ("all", "same_roi", "different_roi"):
        values = accumulator[group]
        denominator = float(values["dual_task_uav_seconds"])
        rows.append(
            {
                "schema_version": SR_SPEED_DIAGNOSTIC_SCHEMA_VERSION,
                "episode_index": episode_index,
                "episode_number": None if episode_index is None else int(episode_index) + 1,
                "scenario_id": scenario_id,
                "roi_relationship": group,
                **{key: float(value) for key, value in values.items()},
                "simultaneously_valid_ratio": _ratio(
                    values["both_valid_uav_seconds"], denominator
                ),
                "zero_dual_task_denominator": denominator == 0.0,
                "time_unit": "UAV-seconds",
                "sampling_resolution": DUAL_TASK_SAMPLING,
            }
        )
    return rows


def aggregate_diagnostics(sr_rows, dual_rows):
    sr = summarize_sr_rows(sr_rows)
    pooled_dual = empty_dual_task_accumulator()
    for row in dual_rows:
        group = row["roi_relationship"]
        if group not in pooled_dual:
            continue
        for key in pooled_dual[group]:
            pooled_dual[group][key] += float(row[key])
    return {
        "schema_version": SR_SPEED_DIAGNOSTIC_SCHEMA_VERSION,
        "sr_movement": sr,
        "dual_task_service": summarize_dual_task_accumulator(pooled_dual),
        "definitions": {
            "sr_event_denominator": "SR teams assigned a rescue mission",
            "absolute_event_times": (
                "simulation timestamps at assignment-state or discrete movement-state observation"
            ),
            "elapsed_time_validity": (
                "elapsed time is null when an outside-at-assignment SR first satisfies "
                "an event after a movement update sharing the assignment timestamp"
            ),
            "mean_event_times": (
                "pooled valid observed elapsed-time sum / valid elapsed-time sample count; "
                "observed events with unresolved elapsed time are excluded and counted separately"
            ),
            "dual_task_time_unit": "UAV-seconds",
            "dual_task_sampling": DUAL_TASK_SAMPLING,
            "com_serviceable": "assigned COM SR is within the existing 400 m S2U range; queue occupancy is ignored",
        },
    }


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def write_sr_speed_diagnostic_artifacts(directory, sr_rows, sr_episode_rows, dual_rows):
    root = Path(directory)
    summary = aggregate_diagnostics(sr_rows, dual_rows)
    paths = {
        "sr_movement_per_sr_jsonl": root / "sr_movement_per_sr.jsonl",
        "sr_movement_per_sr_csv": root / "sr_movement_per_sr.csv",
        "sr_movement_per_episode_jsonl": root / "sr_movement_per_episode.jsonl",
        "dual_task_service_per_episode_jsonl": root / "dual_task_service_per_episode.jsonl",
        "sr_speed_diagnostic_summary_json": root / "sr_speed_diagnostic_summary.json",
    }
    _write_jsonl(paths["sr_movement_per_sr_jsonl"], sr_rows)
    _write_csv(paths["sr_movement_per_sr_csv"], sr_rows)
    _write_jsonl(paths["sr_movement_per_episode_jsonl"], sr_episode_rows)
    _write_jsonl(paths["dual_task_service_per_episode_jsonl"], dual_rows)
    paths["sr_speed_diagnostic_summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary, paths
