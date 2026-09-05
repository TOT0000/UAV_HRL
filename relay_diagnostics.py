"""Validated, JSON-safe Relay diagnostics artifacts shared by all run outputs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid


RELAY_DIAGNOSTICS_FILENAME = "relay_diagnostics.json"
RELAY_DIAGNOSTICS_OUTPUT_CONTRACT_VERSION = (
    "relay-assignment-forwarding-per-episode-json-v1"
)
RELAY_FORWARDING_GROUPS = (
    "assigned_relay_forwarding",
    "nonassigned_uav_forwarding",
    "traversed_assigned_relay",
)
RELAY_FORWARDING_COUNTER_FIELDS = (
    "bits",
    "packets",
    "completed_packet_hops",
)


def _validate_finite_json_value(value, path="relay_diagnostics"):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Relay diagnostics contain a non-finite value at {path}")


def _validate_forwarding(forwarding, label):
    if not isinstance(forwarding, dict):
        raise ValueError(f"{label} must be an object")
    if set(forwarding) != set(RELAY_FORWARDING_GROUPS):
        raise ValueError(f"{label} has incompatible forwarding groups")
    for group in RELAY_FORWARDING_GROUPS:
        counters = forwarding[group]
        if not isinstance(counters, dict):
            raise ValueError(f"{label}.{group} must be an object")
        if set(counters) != set(RELAY_FORWARDING_COUNTER_FIELDS):
            raise ValueError(f"{label}.{group} has incompatible counters")
        if float(counters["bits"]) < 0.0:
            raise ValueError(f"{label}.{group}.bits must be non-negative")
        for field in ("packets", "completed_packet_hops"):
            value = counters[field]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{label}.{group}.{field} must be non-negative")


def validate_relay_diagnostics(diagnostics):
    """Validate one complete train/resume/evaluation Relay diagnostic payload."""

    if not isinstance(diagnostics, dict):
        raise ValueError("Relay diagnostics must be an object")
    for field in (
        "forwarding_packet_semantics",
        "traversed_relay_semantics",
        "episodes",
        "relay_role_change_count",
        "forwarding",
    ):
        if field not in diagnostics:
            raise ValueError(f"Relay diagnostics are missing {field}")
    if not isinstance(diagnostics["episodes"], list) or not diagnostics["episodes"]:
        raise ValueError("Relay diagnostics must contain at least one episode")
    if not isinstance(diagnostics["forwarding_packet_semantics"], str):
        raise ValueError("Relay forwarding packet semantics must be text")
    if not isinstance(diagnostics["traversed_relay_semantics"], str):
        raise ValueError("Relay traversal semantics must be text")

    expected_role_changes = 0
    expected_forwarding = {
        group: {field: 0.0 for field in RELAY_FORWARDING_COUNTER_FIELDS}
        for group in RELAY_FORWARDING_GROUPS
    }
    for expected_index, episode in enumerate(diagnostics["episodes"]):
        label = f"relay_diagnostics.episodes[{expected_index}]"
        if not isinstance(episode, dict):
            raise ValueError(f"{label} must be an object")
        if episode.get("episode_index") != expected_index:
            raise ValueError(
                "Relay diagnostic episode indexes must be contiguous and unique"
            )
        if "scenario_id" not in episode:
            raise ValueError(f"{label} is missing scenario_id")
        assignment = episode.get("assignment")
        if not isinstance(assignment, dict):
            raise ValueError(f"{label}.assignment must be an object")
        for field in (
            "relay_assignment_history",
            "relay_candidate_metrics",
            "selected_relay_uav_ids",
            "relay_role_change_count",
        ):
            if field not in assignment:
                raise ValueError(f"{label}.assignment is missing {field}")
        if not isinstance(assignment["relay_assignment_history"], list):
            raise ValueError(f"{label}.assignment history must be a list")
        if not isinstance(assignment["relay_candidate_metrics"], dict):
            raise ValueError(f"{label}.candidate metrics must be an object")
        if not isinstance(assignment["selected_relay_uav_ids"], list):
            raise ValueError(f"{label}.selected Relay IDs must be a list")
        expected_role_changes += int(assignment["relay_role_change_count"])

        forwarding = episode.get("forwarding")
        _validate_forwarding(forwarding, f"{label}.forwarding")
        for group in RELAY_FORWARDING_GROUPS:
            for field in RELAY_FORWARDING_COUNTER_FIELDS:
                expected_forwarding[group][field] += float(
                    forwarding[group][field]
                )

    if int(diagnostics["relay_role_change_count"]) != expected_role_changes:
        raise ValueError("Relay role-change summary disagrees with episode records")
    _validate_forwarding(diagnostics["forwarding"], "relay_diagnostics.forwarding")
    for group in RELAY_FORWARDING_GROUPS:
        for field in RELAY_FORWARDING_COUNTER_FIELDS:
            actual = float(diagnostics["forwarding"][group][field])
            expected = expected_forwarding[group][field]
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "Relay forwarding summary disagrees with episode records: "
                    f"{group}.{field}"
                )
    _validate_finite_json_value(diagnostics)
    return diagnostics


def relay_diagnostics_metadata(diagnostics):
    validate_relay_diagnostics(diagnostics)
    return {
        "relay_diagnostics_filename": RELAY_DIAGNOSTICS_FILENAME,
        "relay_diagnostics_output_contract_version": (
            RELAY_DIAGNOSTICS_OUTPUT_CONTRACT_VERSION
        ),
        "relay_diagnostics_episode_count": len(diagnostics["episodes"]),
        "relay_diagnostics_episode_semantics": (
            "one zero-based, contiguous record per executed episode; exact resume "
            "contains checkpoint history followed once by newly executed episodes"
        ),
        "relay_diagnostics_assignment_semantics": (
            "per-episode final assignment snapshot plus complete event assignment "
            "history and Relay candidate metrics"
        ),
        "relay_diagnostics_forwarding_semantics": {
            "packets": diagnostics["forwarding_packet_semantics"],
            "traversed_assigned_relay": diagnostics["traversed_relay_semantics"],
        },
        "relay_diagnostics_summary": {
            "relay_role_change_count": int(
                diagnostics["relay_role_change_count"]
            ),
            "forwarding": diagnostics["forwarding"],
            "episode_scenarios": [
                {
                    "episode_index": int(episode["episode_index"]),
                    "scenario_id": episode["scenario_id"],
                }
                for episode in diagnostics["episodes"]
            ],
        },
    }


def write_relay_diagnostics(output_directory, diagnostics):
    """Atomically write the canonical artifact and return metadata plus its path."""

    validate_relay_diagnostics(diagnostics)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / RELAY_DIAGNOSTICS_FILENAME
    temporary = output_directory / f".{RELAY_DIAGNOSTICS_FILENAME}.{uuid.uuid4().hex}.tmp"
    payload = (
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return relay_diagnostics_metadata(diagnostics), destination.resolve()
