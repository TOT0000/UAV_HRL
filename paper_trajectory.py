"""Self-contained standalone trajectory source-data serialization."""

from __future__ import annotations

import copy
import json
import math

from experiment_config import NUM_UAV
from training_checkpoint import (
    CHECKPOINT_PROVENANCE_FIELDS,
    EVALUATION_PROVENANCE_FIELDS,
)


STANDALONE_TRAJECTORY_SCHEMA_VERSION = "uav-hrl-standalone-trajectory-v1"


def _time(value, field):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"standalone trajectory {field} must be finite and non-negative")
    return value


def _truncate_paths(paths, actual_time, *, entity_type):
    result = {}
    for entity_id, values in paths.items():
        selected = [
            copy.deepcopy(value)
            for value in values
            if _time(value["actual_time_seconds"], "path time") <= actual_time
        ]
        if not selected:
            raise ValueError(
                f"standalone trajectory {entity_type} path is empty through "
                f"actual_time={actual_time}: entity={entity_id}"
            )
        if any(
            float(value["actual_time_seconds"]) > actual_time for value in selected
        ):
            raise AssertionError("future path state escaped truncation")
        result[str(entity_id)] = selected
    return result


def build_standalone_trajectory_source(
    *,
    figure_id,
    method_id,
    artifact,
    snapshot,
    requested_time_seconds,
    git_sha,
    checkpoint_provenance,
    camera,
    style,
):
    requested = _time(requested_time_seconds, "requested time")
    actual = _time(snapshot["actual_time_seconds"], "actual time")
    if actual < requested:
        raise ValueError(
            f"standalone trajectory actual time precedes request: "
            f"requested={requested}, actual={actual}"
        )
    history = artifact.get("trajectory_history") or []
    eligible = [
        _time(state["actual_time_seconds"], "history time")
        for state in history
        if _time(state["actual_time_seconds"], "history time") >= requested
    ]
    if not eligible or actual != eligible[0]:
        raise ValueError(
            "standalone trajectory snapshot is not the first state at or after "
            f"the request: requested={requested}, expected={eligible[0] if eligible else None}, "
            f"actual={actual}"
        )
    uavs = copy.deepcopy(snapshot.get("uavs") or [])
    if len(uavs) != NUM_UAV or {int(row["uav_id"]) for row in uavs} != set(
        range(NUM_UAV)
    ):
        raise ValueError(
            f"standalone trajectory requires all {NUM_UAV} UAV snapshot states"
        )
    uav_paths = _truncate_paths(
        artifact.get("uav_paths") or {}, actual, entity_type="UAV"
    )
    if set(uav_paths) != {str(value) for value in range(NUM_UAV)}:
        raise ValueError(
            f"standalone trajectory requires all {NUM_UAV} UAV paths"
        )
    sr_teams = copy.deepcopy(snapshot.get("sr_teams") or [])
    sr_paths = _truncate_paths(
        artifact.get("sr_paths") or {}, actual, entity_type="SR"
    )
    if set(sr_paths) != {str(int(row["sr_id"])) for row in sr_teams}:
        raise ValueError("standalone trajectory SR snapshot/path identities differ")
    checkpoint = {
        "checkpoint_path": checkpoint_provenance.get("checkpoint_path"),
        **{
            field: checkpoint_provenance.get(field)
            for field in (
                *CHECKPOINT_PROVENANCE_FIELDS,
                *EVALUATION_PROVENANCE_FIELDS,
            )
        },
    }
    if any(value is None for value in checkpoint.values()):
        raise ValueError(
            "standalone learned trajectory requires complete checkpoint provenance"
        )
    source = {
        "schema_version": STANDALONE_TRAJECTORY_SCHEMA_VERSION,
        "figure_id": str(figure_id),
        "method_key": str(method_id),
        "requested_time_seconds": requested,
        "actual_time_seconds": actual,
        "actual_phase": str(snapshot["target_uav_phase"]),
        "scenario_id": str(artifact["scenario_id"]),
        "scenario_manifest_hash": str(artifact["scenario_manifest_hash"]),
        **checkpoint,
        "git_sha": str(git_sha),
        "target_uav_id": int(artifact["target_uav_id"]),
        "ground_station": copy.deepcopy(snapshot["ground_station"]),
        "ground_targets": copy.deepcopy(snapshot.get("ground_targets") or []),
        "uavs": uavs,
        "uav_paths": uav_paths,
        "sr_teams": sr_teams,
        "sr_paths": sr_paths,
        "active_links": copy.deepcopy(snapshot.get("active_links") or []),
        "sensing_coverage": copy.deepcopy(
            snapshot.get("sensing_coverage") or []
        ),
        "render_contract": {
            "camera": copy.deepcopy(camera),
            "axis_limits": {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [0.0, 180.0]},
            "axis_labels": {"x": "X(m)", "y": "Y(m)", "z": "Z(m)"},
            "style_reference": "PLOT_STYLES.uav_trajectory_snapshots",
            "style": copy.deepcopy(style),
        },
    }
    validate_standalone_trajectory_source(source)
    return source


def validate_standalone_trajectory_source(source):
    if source.get("schema_version") != STANDALONE_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported standalone trajectory schema: {source.get('schema_version')}"
        )
    actual = _time(source.get("actual_time_seconds"), "actual time")
    requested = _time(source.get("requested_time_seconds"), "requested time")
    if actual < requested:
        raise ValueError("standalone trajectory actual time precedes requested time")
    if len(source.get("uavs") or []) != NUM_UAV:
        raise ValueError(f"standalone trajectory must contain {NUM_UAV} UAVs")
    if set(source.get("uav_paths") or {}) != {str(value) for value in range(NUM_UAV)}:
        raise ValueError(f"standalone trajectory must contain {NUM_UAV} UAV paths")
    for group in ("uav_paths", "sr_paths"):
        for entity_id, path in source[group].items():
            if not path or any(
                _time(row["actual_time_seconds"], "path time") > actual
                for row in path
            ):
                raise ValueError(
                    f"standalone trajectory {group}/{entity_id} is empty or contains future state"
                )
    for field in (
        "figure_id",
        "method_key",
        "actual_phase",
        "scenario_id",
        "scenario_manifest_hash",
        "git_sha",
        "ground_station",
        "ground_targets",
        "sr_teams",
        "active_links",
        "sensing_coverage",
        "render_contract",
    ):
        if field not in source:
            raise ValueError(f"standalone trajectory source lacks {field}")
    for field in CHECKPOINT_PROVENANCE_FIELDS:
        value = source.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"standalone trajectory source has invalid {field}")
    training = source.get("checkpoint_training_provenance")
    runtime = source.get("evaluation_runtime_provenance")
    if not isinstance(training, dict) or not isinstance(runtime, dict):
        raise ValueError(
            "standalone trajectory source lacks separated evaluation provenance"
        )
    if (
        int(source.get("checkpoint_training_episode_count", -1))
        != int(training.get("training_episode_count", -2))
        or source.get("checkpoint_training_git_sha")
        != training.get("training_git_sha")
        or int(source.get("evaluation_episode_count", -1))
        != int(runtime.get("evaluation_episode_count", -2))
        or source.get("evaluation_git_sha") != runtime.get("evaluation_git_sha")
    ):
        raise ValueError(
            "standalone trajectory source provenance aliases are inconsistent"
        )
    return source


def standalone_trajectory_csv_rows(source):
    validate_standalone_trajectory_source(source)
    common = {
        "figure_id": source["figure_id"],
        "method_key": source["method_key"],
        "requested_time_seconds": source["requested_time_seconds"],
        "actual_time_seconds": source["actual_time_seconds"],
        "actual_phase": source["actual_phase"],
        "scenario_id": source["scenario_id"],
        "scenario_manifest_hash": source["scenario_manifest_hash"],
        "checkpoint_path": source["checkpoint_path"],
        **{field: source[field] for field in CHECKPOINT_PROVENANCE_FIELDS},
        "checkpoint_training_episode_count": source[
            "checkpoint_training_episode_count"
        ],
        "evaluation_episode_count": source["evaluation_episode_count"],
        "checkpoint_training_git_sha": source["checkpoint_training_git_sha"],
        "evaluation_git_sha": source["evaluation_git_sha"],
        "checkpoint_training_provenance_json": json.dumps(
            source["checkpoint_training_provenance"], sort_keys=True
        ),
        "evaluation_runtime_provenance_json": json.dumps(
            source["evaluation_runtime_provenance"], sort_keys=True
        ),
        "git_sha": source["git_sha"],
    }
    rows = []
    for uav_id, path in source["uav_paths"].items():
        for state in path:
            rows.append(
                {
                    **common,
                    "record_type": "uav_path",
                    "entity_id": int(uav_id),
                    "time_seconds": state["actual_time_seconds"],
                    "x": state["x"],
                    "y": state["y"],
                    "z": state["z"],
                    "task_phase": state.get("task_phase"),
                    "assigned_tasks_json": json.dumps(
                        state.get("assigned_tasks", []),
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                }
            )
    for uav in source["uavs"]:
        rows.append(
            {
                **common,
                "record_type": "uav_snapshot",
                "entity_id": int(uav["uav_id"]),
                "time_seconds": source["actual_time_seconds"],
                "x": uav["x"],
                "y": uav["y"],
                "z": uav["z"],
                "task_phase": uav.get("task_phase"),
                "assigned_tasks_json": json.dumps(
                    uav.get("assigned_tasks", []),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        )
    for sr_id, path in source["sr_paths"].items():
        for state in path:
            rows.append(
                {
                    **common,
                    "record_type": "sr_path",
                    "entity_id": int(sr_id),
                    "time_seconds": state["actual_time_seconds"],
                    "x": state["x"],
                    "y": state["y"],
                    "z": state["z"],
                }
            )
    for sr in source["sr_teams"]:
        rows.append(
            {
                **common,
                "record_type": "sr_snapshot",
                "entity_id": int(sr["sr_id"]),
                "time_seconds": source["actual_time_seconds"],
                "x": sr["x"],
                "y": sr["y"],
                "z": sr["z"],
                "active": sr.get("active"),
            }
        )
    gs = source["ground_station"]
    rows.append(
        {
            **common,
            "record_type": "ground_station",
            "entity_id": gs.get("gs_id"),
            "x": gs["x"],
            "y": gs["y"],
            "z": gs["z"],
        }
    )
    for gt in source["ground_targets"]:
        rows.append(
            {
                **common,
                "record_type": "ground_target",
                "entity_id": gt["gt_id"],
                "x": gt["x"],
                "y": gt["y"],
                "z": gt["z"],
                "radius_m": gt["radius_m"],
                "detected": gt.get("detected"),
                "detected_by_uav_id": gt.get("detected_by_uav_id"),
            }
        )
    for link in source["active_links"]:
        rows.append(
            {
                **common,
                "record_type": "active_link",
                "source_id": link["sender_id"],
                "destination_id": link["receiver_id"],
                "link_type": link["link_type"],
                "capacity_bits_per_second": link.get("capacity_bits_per_second"),
            }
        )
    for coverage in source["sensing_coverage"]:
        rows.append(
            {
                **common,
                "record_type": "sensing_coverage",
                "entity_id": coverage["uav_id"],
                "coverage_json": json.dumps(
                    coverage, sort_keys=True, ensure_ascii=False
                ),
            }
        )
    return rows
