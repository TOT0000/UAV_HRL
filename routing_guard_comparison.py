"""Streamed OFF/ON comparison for the evaluation-only routing guard."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


COMPARISON_SCHEMA_VERSION = "routing-guard-off-on-comparison-v2"


def _new_totals():
    return {
        "complete_useful_mbits": 0.0,
        "mobility_energy_j": 0.0,
        "eligible_packet_count": 0,
        "violation_packet_count": 0,
        "on_time_delivered_packet_count": 0,
        "late_delivered_packet_count": 0,
        "twenty_hop_terminal_packet_count": 0,
        "episode_end_at_sr_packet_count": 0,
        "episode_end_at_uav_packet_count": 0,
        "immediate_foldback_event_count": 0,
        "distinct_foldback_packet_count": 0,
        "forced_wait_no_forwarding_candidate_count": 0,
        "guard_induced_forced_wait_count": 0,
        "slot_start_hol_guard_induced_forced_wait_decision_count": 0,
        "uav_preselection_guard_induced_forced_wait_decision_count": None,
    }


def _keys(roi_count, task_type):
    task = "VS" if str(task_type) == "FOV" else "COM"
    return ((None, "ALL"), (None, task), (int(roi_count), "ALL"), (int(roi_count), task))


def _initialize_groups(roi_counts):
    totals = defaultdict(_new_totals)
    for roi in (None, *[int(value) for value in roi_counts]):
        for task in ("ALL", "VS", "COM"):
            totals[(roi, task)]
    return totals


def _read_selected_rows(path, selected_scenario_ids):
    expected = list(map(str, selected_scenario_ids))
    seen = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(seen) >= len(expected):
                break
            row = json.loads(line)
            scenario_id = str(row.get("scenario_id"))
            if scenario_id != expected[len(seen)]:
                raise AssertionError(
                    f"scenario order mismatch in {path}: expected "
                    f"{expected[len(seen)]!r}, got {scenario_id!r}"
                )
            seen.append(scenario_id)
            yield row
    if seen != expected:
        raise AssertionError(f"selected scenario prefix is incomplete in {path}")


def _foldback_count(packet):
    completed = [
        hop for hop in packet.get("per_hop", ())
        if str(hop.get("link_type")) == "U2U"
    ]
    count = 0
    for previous, current in zip(completed, completed[1:]):
        try:
            count += int(
                int(previous["from"]) == int(current["to"])
                and int(previous["to"]) == int(current["from"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return count


def _accumulate_evaluation(
    root, roi_counts, selected_ids_by_roi, *, episode_end_seconds=60.0
):
    totals = _initialize_groups(roi_counts)
    root = Path(root)
    for roi_count in roi_counts:
        roi_count = int(roi_count)
        selected = selected_ids_by_roi[str(roi_count)]
        point = root / f"roi_{roi_count}"
        for episode in _read_selected_rows(point / "per_episode.jsonl", selected):
            energy = float(episode["total_mobility_energy_j"])
            useful = {
                "ALL": float(episode["total_all_useful_mbits"]),
                "VS": (
                    float(episode["fov_timely_useful_bits"])
                    + float(episode["fov_late_useful_bits"])
                ) / 1e6,
                "COM": (
                    float(episode["com_timely_delivered_bits"])
                    + float(episode["com_late_delivered_bits"])
                ) / 1e6,
            }
            for group_roi in (None, roi_count):
                for task in ("ALL", "VS", "COM"):
                    totals[(group_roi, task)]["complete_useful_mbits"] += useful[task]
                    totals[(group_roi, task)]["mobility_energy_j"] += energy

        for episode in _read_selected_rows(point / "packet_outcomes.jsonl", selected):
            for packet in episode.get("packet_outcomes", ()):
                foldbacks = _foldback_count(packet)
                delivered = bool(packet.get("delivered_to_gs", False))
                outcome = str(packet.get("outcome"))
                ended_at_episode_boundary = bool(
                    not delivered
                    and packet.get("finish_time_seconds") is not None
                    and math.isclose(
                        float(packet["finish_time_seconds"]),
                        float(episode_end_seconds),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
                for group in _keys(roi_count, packet.get("task_type")):
                    stats = totals[group]
                    eligible = bool(packet.get("qos_eligible", False))
                    stats["eligible_packet_count"] += int(eligible)
                    stats["violation_packet_count"] += int(
                        eligible and outcome != "on_time_delivered"
                    )
                    stats["on_time_delivered_packet_count"] += int(
                        outcome == "on_time_delivered"
                    )
                    stats["late_delivered_packet_count"] += int(
                        outcome == "late_delivered"
                    )
                    stats["twenty_hop_terminal_packet_count"] += int(
                        int(packet.get("completed_uav_hop_count", 0)) == 20
                        and not delivered
                    )
                    terminal = str(packet.get("terminal_node_type"))
                    stats["episode_end_at_sr_packet_count"] += int(
                        ended_at_episode_boundary and terminal == "SR"
                    )
                    stats["episode_end_at_uav_packet_count"] += int(
                        ended_at_episode_boundary and terminal == "UAV"
                    )
                    stats["immediate_foldback_event_count"] += int(foldbacks)
                    stats["distinct_foldback_packet_count"] += int(bool(foldbacks))
                    stats["forced_wait_no_forwarding_candidate_count"] += int(
                        packet.get(
                            "routing_only_wait_no_available_link_slot_count", 0
                        )
                    )
    return totals


def _inject_guard_wait_counts(totals, routing_summary_path):
    summary = json.loads(Path(routing_summary_path).read_text(encoding="utf-8"))
    if summary.get("schema_version") != "routing-loop-diagnostics-v4":
        raise ValueError(
            "guarded ON evaluation requires routing-loop diagnostics v4 "
            "with held-action packet WAIT counts"
        )
    rows = list(summary.get("groups", ()))
    if not rows:
        raise ValueError("guarded ON evaluation routing summary has no groups")
    seen = set()
    for row in rows:
        key = (row.get("roi_count"), str(row["task"]))
        if key in totals:
            required = (
                "held_action_packet_guard_induced_forced_wait_count",
                "slot_start_hol_guard_induced_forced_wait_decision_count",
            )
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(
                    "guarded ON routing summary lacks required WAIT diagnostics: "
                    + ", ".join(missing)
                )
            totals[key]["guard_induced_forced_wait_count"] = int(
                row["held_action_packet_guard_induced_forced_wait_count"]
            )
            totals[key][
                "slot_start_hol_guard_induced_forced_wait_decision_count"
            ] = int(
                row["slot_start_hol_guard_induced_forced_wait_decision_count"]
            )
            seen.add(key)
    if seen != set(totals):
        raise ValueError("guarded ON routing summary does not contain every comparison group")
    preselection = {
        row.get("roi_count"): row
        for row in summary.get("uav_preselection_decisions", ())
    }
    expected_rois = {key[0] for key in totals}
    if set(preselection) != expected_rois:
        raise ValueError(
            "guarded ON routing summary lacks per-RoI UAV preselection diagnostics"
        )
    for (roi_count, task), stats in totals.items():
        if task == "ALL":
            stats[
                "uav_preselection_guard_induced_forced_wait_decision_count"
            ] = int(
                preselection[roi_count][
                    "uav_preselection_guard_induced_forced_wait_decision_count"
                ]
            )


def _public(group, stats):
    useful = float(stats["complete_useful_mbits"])
    energy = float(stats["mobility_energy_j"])
    eligible = int(stats["eligible_packet_count"])
    violations = int(stats["violation_packet_count"])
    return {
        "group": group[1] if group[0] is None else f"ROI_{group[0]}_{group[1]}",
        "roi_count": group[0],
        "task": group[1],
        "complete_useful_mbits": useful,
        "mobility_energy_j": energy,
        "energy_efficiency_mbit_per_j": None if energy == 0.0 else useful / energy,
        "eligible_packet_count": eligible,
        "violation_packet_count": violations,
        "violation_rate": None if eligible == 0 else violations / eligible,
        "on_time_delivered_packet_count": int(stats["on_time_delivered_packet_count"]),
        "late_delivered_packet_count": int(stats["late_delivered_packet_count"]),
        "twenty_hop_terminal_packet_count": int(stats["twenty_hop_terminal_packet_count"]),
        "episode_end_at_sr_packet_count": int(stats["episode_end_at_sr_packet_count"]),
        "episode_end_at_uav_packet_count": int(stats["episode_end_at_uav_packet_count"]),
        "immediate_foldback_event_count": int(stats["immediate_foldback_event_count"]),
        "distinct_foldback_packet_count": int(
            stats["distinct_foldback_packet_count"]
        ),
        "forced_wait_no_forwarding_candidate_count": int(
            stats["forced_wait_no_forwarding_candidate_count"]
        ),
        "guard_induced_forced_wait_count": int(
            stats["guard_induced_forced_wait_count"]
        ),
        "slot_start_hol_guard_induced_forced_wait_decision_count": int(
            stats["slot_start_hol_guard_induced_forced_wait_decision_count"]
        ),
        "uav_preselection_guard_induced_forced_wait_decision_count": (
            None
            if stats["uav_preselection_guard_induced_forced_wait_decision_count"] is None
            else int(stats["uav_preselection_guard_induced_forced_wait_decision_count"])
        ),
    }


def _sort_key(group):
    task_order = {"ALL": 0, "VS": 1, "COM": 2}
    return (-1 if group[0] is None else int(group[0]), task_order[group[1]])


def _verified_strategy(root, *, enabled):
    from mutual_next_hop_guard import evaluation_routing_strategy_info

    expected = evaluation_routing_strategy_info(enabled)
    metadata_path = Path(root) / "paper_evaluation_metadata.json"
    if not metadata_path.is_file():
        if enabled:
            raise ValueError(
                "guarded ON evaluation lacks paper evaluation strategy metadata"
            )
        return expected, "legacy_off_missing_strategy_metadata"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata.get("evaluation_routing_strategy")
    if actual is None:
        actual = metadata.get("mutual_next_hop_guard")
    if actual is None:
        if enabled:
            raise ValueError("guarded ON evaluation lacks routing strategy identity")
        return expected, "legacy_off_missing_strategy_fields"
    actual_enabled = bool(
        actual.get(
            "mutual_next_hop_guard_enabled", actual.get("enabled", False)
        )
    )
    if actual_enabled != bool(enabled):
        raise ValueError("evaluation routing strategy guard mode is inconsistent")
    if enabled:
        for field in ("strategy_version", "strategy_fingerprint"):
            if actual.get(field) != expected[field]:
                raise ValueError(
                    f"guarded ON evaluation routing strategy {field} is inconsistent"
                )
    return expected, "explicit_strategy_metadata"


def build_routing_guard_comparison(
    *, source_evaluation_dir, guarded_evaluation_dir, roi_counts,
    selected_scenario_ids_by_roi,
):
    off_strategy, off_strategy_compatibility = _verified_strategy(
        source_evaluation_dir, enabled=False
    )
    on_strategy, on_strategy_compatibility = _verified_strategy(
        guarded_evaluation_dir, enabled=True
    )
    off = _accumulate_evaluation(
        source_evaluation_dir, roi_counts, selected_scenario_ids_by_roi
    )
    on = _accumulate_evaluation(
        guarded_evaluation_dir, roi_counts, selected_scenario_ids_by_roi
    )
    _inject_guard_wait_counts(
        on, Path(guarded_evaluation_dir) / "routing_loop_summary.json"
    )
    for stats in off.values():
        # A confirmed legacy source has no guard, so guard-caused counts are
        # logically zero even though it predates the diagnostic fields.
        stats["guard_induced_forced_wait_count"] = 0
        stats["slot_start_hol_guard_induced_forced_wait_decision_count"] = 0
    for (roi_count, task), stats in off.items():
        if task == "ALL":
            stats["uav_preselection_guard_induced_forced_wait_decision_count"] = 0
    rows = []
    for group in sorted(off, key=_sort_key):
        off_row = _public(group, off[group])
        on_row = _public(group, on[group])
        if (
            on_row["guard_induced_forced_wait_count"]
            > on_row["forced_wait_no_forwarding_candidate_count"]
        ):
            raise AssertionError(
                f"{on_row['group']}: guard-induced packet WAIT exceeds all forced WAIT"
            )
        row = {key: off_row[key] for key in ("group", "roi_count", "task")}
        row.update(
            {
                "off_strategy_version": off_strategy["strategy_version"],
                "off_strategy_fingerprint": off_strategy["strategy_fingerprint"],
                "off_mutual_next_hop_guard_enabled": False,
                "on_strategy_version": on_strategy["strategy_version"],
                "on_strategy_fingerprint": on_strategy["strategy_fingerprint"],
                "on_mutual_next_hop_guard_enabled": True,
                "evaluation_strategy_scope": "evaluation_only",
            }
        )
        for key in off_row.keys() - {"group", "roi_count", "task"}:
            before, after = off_row[key], on_row[key]
            row[f"off_{key}"] = before
            row[f"on_{key}"] = after
            row[f"delta_{key}"] = (
                None if before is None or after is None else after - before
            )
            row[f"relative_change_{key}"] = (
                None
                if before in (None, 0) or after is None
                else (after - before) / before
            )
        rows.append(row)

    output = Path(guarded_evaluation_dir)
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "off_source_evaluation_directory": str(Path(source_evaluation_dir).resolve()),
        "on_guarded_evaluation_directory": str(output.resolve()),
        "off_evaluation_routing_strategy": off_strategy,
        "on_evaluation_routing_strategy": on_strategy,
        "strategy_metadata_compatibility": {
            "off": off_strategy_compatibility,
            "on": on_strategy_compatibility,
        },
        "aggregation": {
            "energy_efficiency": "sum complete useful Mbit / sum mobility energy J",
            "violation_rate": "sum violations / sum eligible packets",
            "task_energy_denominator": (
                "the same complete UAV mobility energy is reported for ALL, VS, and COM"
            ),
            "episode_end_location": (
                "not delivered, terminal node SR/UAV, and finish time equals the 60 s horizon"
            ),
            "relative_change": "(ON - OFF) / OFF; null when OFF is zero or missing",
            "forced_wait_counts": (
                "all and guard-induced counts both use actual packet HOL uses of held "
                "actions, deduplicated by RoI/scenario/slot/sender/packet"
            ),
        },
        "groups": rows,
    }
    json_path = output / "routing_guard_off_on_comparison.json"
    csv_path = output / "routing_guard_off_on_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve())}
