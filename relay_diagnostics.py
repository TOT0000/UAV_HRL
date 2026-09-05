"""Validated, JSON-safe Relay diagnostics artifacts shared by all run outputs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid


RELAY_DIAGNOSTICS_FILENAME = "relay_diagnostics.json"
RELAY_DIAGNOSTICS_OUTPUT_CONTRACT_VERSION = (
    "relay-assignment-forwarding-stable-bits-sum-per-episode-json-v2"
)
RELAY_BITS_SUM_REL_TOL = 1e-15
RELAY_BITS_SUM_ABS_TOL = 1e-6
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
        try:
            bits = float(counters["bits"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{group}.bits must be numeric") from exc
        if not math.isfinite(bits):
            raise ValueError(f"{label}.{group}.bits is non-finite")
        if bits < 0.0:
            raise ValueError(f"{label}.{group}.bits must be non-negative")
        for field in ("packets", "completed_packet_hops"):
            value = counters[field]
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{label}.{group}.{field} must be a non-negative integer"
                )


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
    episode_forwarding = {
        group: {field: [] for field in RELAY_FORWARDING_COUNTER_FIELDS}
        for group in RELAY_FORWARDING_GROUPS
    }
    for expected_index, episode in enumerate(diagnostics["episodes"]):
        label = f"relay_diagnostics.episodes[{expected_index}]"
        if not isinstance(episode, dict):
            raise ValueError(f"{label} must be an object")
        if type(episode.get("episode_index")) is not int or episode.get(
            "episode_index"
        ) != expected_index:
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
        role_changes = assignment["relay_role_change_count"]
        if type(role_changes) is not int or role_changes < 0:
            raise ValueError(
                f"{label}.assignment.relay_role_change_count must be a "
                "non-negative integer"
            )
        expected_role_changes += role_changes

        forwarding = episode.get("forwarding")
        _validate_forwarding(forwarding, f"{label}.forwarding")
        for group in RELAY_FORWARDING_GROUPS:
            for field in RELAY_FORWARDING_COUNTER_FIELDS:
                episode_forwarding[group][field].append(
                    forwarding[group][field]
                )

    if (
        type(diagnostics["relay_role_change_count"]) is not int
        or diagnostics["relay_role_change_count"] != expected_role_changes
    ):
        raise ValueError("Relay role-change summary disagrees with episode records")
    _validate_forwarding(diagnostics["forwarding"], "relay_diagnostics.forwarding")
    for group in RELAY_FORWARDING_GROUPS:
        actual_bits = float(diagnostics["forwarding"][group]["bits"])
        expected_bits = math.fsum(
            float(value) for value in episode_forwarding[group]["bits"]
        )
        if not math.isclose(
            actual_bits,
            expected_bits,
            rel_tol=RELAY_BITS_SUM_REL_TOL,
            abs_tol=RELAY_BITS_SUM_ABS_TOL,
        ):
            raise ValueError(
                "Relay forwarding summary disagrees with episode records: "
                f"{group}.bits"
            )
        for field in ("packets", "completed_packet_hops"):
            actual = diagnostics["forwarding"][group][field]
            expected = sum(episode_forwarding[group][field])
            if actual != expected:
                raise ValueError(
                    "Relay forwarding summary disagrees with episode records: "
                    f"{group}.{field}"
                )
    _validate_finite_json_value(diagnostics)
    return diagnostics


def aggregate_relay_episode_diagnostics(episodes):
    """Build the stable run summary from one ordered record per episode."""

    records = list(episodes)
    diagnostics = {
        "forwarding_packet_semantics": "completed packet hops",
        "traversed_relay_semantics": (
            "bits and completed packet hops received by a UAV while assigned Relay"
        ),
        "episodes": records,
        "relay_role_change_count": sum(
            int(episode["assignment"]["relay_role_change_count"])
            for episode in records
        ),
        "forwarding": {
            group: {
                "bits": math.fsum(
                    float(episode["forwarding"][group]["bits"])
                    for episode in records
                ),
                "completed_packet_hops": sum(
                    int(episode["forwarding"][group]["completed_packet_hops"])
                    for episode in records
                ),
                "packets": sum(
                    int(episode["forwarding"][group]["packets"])
                    for episode in records
                ),
            }
            for group in RELAY_FORWARDING_GROUPS
        },
    }
    return validate_relay_diagnostics(diagnostics)


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
            "bits_summation": "math.fsum over per-episode bit counters",
            "bits_validation_relative_tolerance": RELAY_BITS_SUM_REL_TOL,
            "bits_validation_absolute_tolerance": RELAY_BITS_SUM_ABS_TOL,
            "integer_counter_validation": "exact",
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
