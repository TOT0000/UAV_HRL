"""Bounded or streaming persistence for per-packet episode outcomes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION = "uav-hrl-packet-outcomes-jsonl-v6"
PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION = (
    "uav-hrl-packet-routing-diagnostics-v3"
)
PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS = {
    "routing_wait_definition": (
        "routing_decision_slot_count increments once when a routing-eligible "
        "packet is the frozen sender HOL and enters the routing decision path; "
        "routing_wait_slot_count increments when that decision selects receiver "
        "== sender, and routing_wait_seconds adds the authoritative slot duration"
    ),
    "routing_wait_decomposition_definition": (
        "voluntary Wait means selected receiver == sender while at least one "
        "effective-mask action other than sender is True; only-Wait means selected "
        "receiver == sender while no effective non-Wait action is True; the two "
        "categories are mutually exclusive and sum to routing_wait_slot_count"
    ),
    "forced_locked_wait_definition": (
        "a routing-eligible frozen HOL packet selected Wait, had a locked "
        "hop_receiver outside the authoritative inclusive 400 m 3-D routing "
        "range according to env.is_routing_link_in_range, and its effective "
        "mask contained only sender/Wait; capacity unavailability while in "
        "range is excluded; this is a strict subcategory of only-Wait; aggregate "
        "fraction uses all routing decision slots as its denominator"
    ),
    "s2u_hol_opportunity_definition": (
        "a COM packet is counted at most once per routing slot when it is the "
        "actual SR FIFO HOL and enters that slot's canonical S2U service "
        "opportunity path; non-HOL queue packets are not scanned or counted"
    ),
    "s2u_hol_service_definition": (
        "an S2U HOL service slot requires actual packet bits to be consumed and "
        "the packet's actual positive-capacity S2U airtime to increase; active "
        "link or nominal capacity alone is not service"
    ),
    "s2u_hol_no_service_reason_priority": (
        "each unserved S2U HOL opportunity is assigned exactly one reason in "
        "priority order: no locked or assigned receiver; receiver outside the "
        "authoritative inclusive 400 m 3-D S2U range; no matching canonical "
        "resolved active S2U link; no finite positive allocated capacity/block "
        "budget; or positive capacity but no actual service"
    ),
    "pre_s2u_violation_decomposition_definition": (
        "incomplete-S2U violations are mutually partitioned into never became "
        "HOL (zero S2U HOL opportunities), became HOL but never started (positive "
        "HOL opportunities and zero actual S2U airtime), or partial S2U service "
        "(positive actual S2U airtime without S2U completion)"
    ),
    "loop_definition": (
        "at least one UAV ID occurs more than once in path; the SR source entry "
        "and GS are excluded; repeated_uav_count is the number of distinct "
        "repeated UAV IDs"
    ),
    "terminal_node_definition": (
        "an unfinished partial hop remains at its sender UAV; pre-S2U packets "
        "remain at SR; completed delivery terminates at GS"
    ),
    "uav_queue_delay_definition": (
        "sum of queue_s for completed U2U/U2G hops plus elapsed queue waiting "
        "for an unfinished terminal UAV hop; current queue waiting ends at the "
        "first actual service start, or at terminal finish_time if service has "
        "not started"
    ),
    "uav_tx_delay_definition": (
        "actual U2U/U2G transmission airtime accumulated as bits consumed "
        "divided by each positive instantaneous block capacity, including a "
        "terminal partial hop and excluding Wait, zero-capacity blocks, and "
        "wall-clock gaps; per_hop.tx_s remains the legacy service span while "
        "per_hop.actual_tx_s is actual airtime"
    ),
    "s2u_delay_definition": (
        "for COM, S2U queue delay ends at first actual S2U service or terminal "
        "finish_time if service never starts, and S2U tx delay is actual "
        "positive-capacity block airtime including incomplete S2U service; both "
        "fields are null for FOV"
    ),
    "loop_hop_statistic_definition": (
        "mean_hops_loop_packets and mean_hops_non_loop_packets use completed "
        "UAV-side U2U plus U2G hops only; S2U and terminal partial hops are "
        "excluded"
    ),
}
PACKET_OUTCOME_MODE_DISABLED = "disabled"
PACKET_OUTCOME_MODE_BOUNDED = "bounded_memory"
PACKET_OUTCOME_MODE_STREAMING = "stream_jsonl"
PACKET_OUTCOME_ARTIFACT_MODES = frozenset(
    {
        PACKET_OUTCOME_MODE_DISABLED,
        PACKET_OUTCOME_MODE_BOUNDED,
        PACKET_OUTCOME_MODE_STREAMING,
    }
)
MAX_BOUNDED_PACKET_OUTCOME_EPISODES = 16

PACKET_OUTCOME_REQUIRED_FIELDS = frozenset(
    {
        "packet_id",
        "task_type",
        "source_kind",
        "source_uav_id",
        "source_sr_id",
        "outcome",
        "generation_time_seconds",
        "finish_time_seconds",
        "deadline_seconds",
        "e2e_delay_seconds",
        "s2u_completed",
        "s2u_completion_time_seconds",
        "routing_eligible_time_seconds",
        "terminal_node_type",
        "terminal_node_id",
        "terminal_uav_id",
        "locked_hop_receiver_at_terminal",
        "path",
        "per_hop",
        "path_hop_count",
        "completed_uav_hop_count",
        "unique_uav_count",
        "has_repeated_uav",
        "repeated_uav_count",
        "repeated_uav_ids",
        "routing_decision_slot_count",
        "routing_wait_slot_count",
        "routing_voluntary_wait_with_legal_nonwait_slot_count",
        "routing_only_wait_no_available_link_slot_count",
        "routing_wait_seconds",
        "locked_receiver_out_of_range_wait_slot_count",
        "locked_receiver_out_of_range_wait_seconds",
        "cumulative_uav_queue_delay_seconds",
        "cumulative_uav_tx_delay_seconds",
        "s2u_queue_delay_seconds",
        "s2u_tx_delay_seconds",
        "s2u_hol_opportunity_slot_count",
        "s2u_hol_service_slot_count",
        "s2u_hol_no_receiver_slot_count",
        "s2u_hol_receiver_out_of_range_slot_count",
        "s2u_hol_no_active_link_slot_count",
        "s2u_hol_no_positive_capacity_slot_count",
        "s2u_hol_positive_capacity_but_no_service_slot_count",
        "qos_eligible",
    }
)

S2U_HOL_COUNT_FIELDS = (
    "s2u_hol_opportunity_slot_count",
    "s2u_hol_service_slot_count",
    "s2u_hol_no_receiver_slot_count",
    "s2u_hol_receiver_out_of_range_slot_count",
    "s2u_hol_no_active_link_slot_count",
    "s2u_hol_no_positive_capacity_slot_count",
    "s2u_hol_positive_capacity_but_no_service_slot_count",
)
S2U_HOL_NO_SERVICE_COUNT_FIELDS = S2U_HOL_COUNT_FIELDS[2:]


def _validate_nonnegative_number(row, field, *, nullable=False):
    value = row.get(field)
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"packet outcome field {field} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(
            f"packet outcome field {field} must be finite and non-negative"
        )


def validate_packet_outcome(row):
    if not isinstance(row, dict):
        raise TypeError("each packet outcome must be a dictionary")
    missing = PACKET_OUTCOME_REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(
            "packet outcome lacks required fields: " + ", ".join(sorted(missing))
        )
    if row["task_type"] not in {"COM", "FOV"}:
        raise ValueError("packet outcome task_type must be COM or FOV")
    if row["source_kind"] not in {"SR", "UAV"}:
        raise ValueError("packet outcome source_kind must be SR or UAV")
    if row["outcome"] not in {
        "on_time_delivered",
        "late_delivered",
        "expired_dropped",
        "sr_admission_drop",
    }:
        raise ValueError("packet outcome outcome is invalid")
    if type(row["qos_eligible"]) is not bool:
        raise ValueError("packet outcome qos_eligible must be boolean")
    if row["source_kind"] == "SR":
        if row["source_uav_id"] is not None or not isinstance(
            row["source_sr_id"], int
        ):
            raise ValueError("SR packet source fields are inconsistent")
        if type(row["s2u_completed"]) is not bool:
            raise ValueError("SR packet s2u_completed must be boolean")
        if row["s2u_queue_delay_seconds"] is None:
            raise ValueError("SR packet S2U queue delay must be numeric")
        if row["s2u_tx_delay_seconds"] is None:
            raise ValueError("SR packet S2U transmission delay must be numeric")
    else:
        if row["source_sr_id"] is not None or not isinstance(
            row["source_uav_id"], int
        ):
            raise ValueError("UAV packet source fields are inconsistent")
        if row["s2u_completed"] is not None:
            raise ValueError("UAV packet s2u_completed must be null")
        if row["s2u_completion_time_seconds"] is not None:
            raise ValueError("UAV packet S2U completion time must be null")
        if row["s2u_queue_delay_seconds"] is not None:
            raise ValueError("UAV packet S2U queue delay must be null")
        if row["s2u_tx_delay_seconds"] is not None:
            raise ValueError("UAV packet S2U transmission delay must be null")
    if row["s2u_completed"] is True and (
        row["s2u_completion_time_seconds"] is None
    ):
        raise ValueError("completed S2U packet lacks completion time")
    if row["s2u_completed"] is False and (
        row["s2u_completion_time_seconds"] is not None
    ):
        raise ValueError("incomplete S2U packet has a completion time")
    if row["terminal_node_type"] not in {"SR", "UAV", "GS"}:
        raise ValueError("packet outcome terminal_node_type is invalid")
    if row["terminal_node_type"] == "UAV":
        if row["terminal_uav_id"] != row["terminal_node_id"]:
            raise ValueError("terminal UAV fields disagree")
    elif row["terminal_uav_id"] is not None:
        raise ValueError("non-UAV terminal outcome cannot have terminal_uav_id")
    if not isinstance(row["path"], list) or not isinstance(row["per_hop"], list):
        raise ValueError("packet outcome path and per_hop must be lists")
    if (
        isinstance(row["terminal_node_id"], bool)
        or not isinstance(row["terminal_node_id"], int)
        or row["terminal_node_id"] < 0
    ):
        raise ValueError("terminal_node_id must be a non-negative integer")
    locked_receiver = row["locked_hop_receiver_at_terminal"]
    if locked_receiver is not None and (
        isinstance(locked_receiver, bool)
        or not isinstance(locked_receiver, int)
        or locked_receiver < 0
    ):
        raise ValueError(
            "locked_hop_receiver_at_terminal must be a non-negative integer or null"
        )
    delivered = row["outcome"] in {"on_time_delivered", "late_delivered"}
    if delivered != (row["terminal_node_type"] == "GS"):
        raise ValueError("delivered outcome and terminal node type disagree")
    if type(row["has_repeated_uav"]) is not bool:
        raise ValueError("packet outcome has_repeated_uav must be boolean")
    if row["s2u_completed"] is not None and type(row["s2u_completed"]) is not bool:
        raise ValueError("packet outcome s2u_completed must be boolean or null")
    if not isinstance(row["repeated_uav_ids"], list):
        raise ValueError("packet outcome repeated_uav_ids must be a list")
    if any(
        isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0
        for node_id in row["repeated_uav_ids"]
    ):
        raise ValueError("repeated_uav_ids must contain non-negative integers")
    if row["repeated_uav_ids"] != sorted(set(row["repeated_uav_ids"])):
        raise ValueError("repeated_uav_ids must be sorted and unique")
    for hop in row["per_hop"]:
        if not isinstance(hop, dict):
            raise ValueError("per_hop entries must be dictionaries")
        missing_hop_fields = {
            "from",
            "to",
            "link_type",
            "queue_s",
            "tx_s",
            "actual_tx_s",
        }.difference(hop)
        if missing_hop_fields:
            raise ValueError(
                "per_hop entry lacks required fields: "
                + ", ".join(sorted(missing_hop_fields))
            )
        if hop["link_type"] not in {"S2U", "U2U", "U2G"}:
            raise ValueError("per_hop link_type is invalid")
        _validate_nonnegative_number(hop, "queue_s")
        _validate_nonnegative_number(hop, "tx_s")
        _validate_nonnegative_number(hop, "actual_tx_s")
    count_fields = (
        "path_hop_count",
        "completed_uav_hop_count",
        "unique_uav_count",
        "repeated_uav_count",
        "routing_decision_slot_count",
        "routing_wait_slot_count",
        "routing_voluntary_wait_with_legal_nonwait_slot_count",
        "routing_only_wait_no_available_link_slot_count",
        "locked_receiver_out_of_range_wait_slot_count",
    )
    for field in count_fields:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"packet outcome field {field} must be a non-negative integer"
            )
    if row["repeated_uav_count"] != len(row["repeated_uav_ids"]):
        raise ValueError("repeated_uav_count disagrees with repeated_uav_ids")
    if row["routing_wait_slot_count"] > row["routing_decision_slot_count"]:
        raise ValueError("routing wait slots exceed routing decision slots")
    if row["routing_wait_slot_count"] != (
        row["routing_voluntary_wait_with_legal_nonwait_slot_count"]
        + row["routing_only_wait_no_available_link_slot_count"]
    ):
        raise ValueError(
            "routing wait slots do not equal voluntary plus only-Wait slots"
        )
    if (
        row["locked_receiver_out_of_range_wait_slot_count"]
        > row["routing_only_wait_no_available_link_slot_count"]
    ):
        raise ValueError("forced locked wait slots exceed only-Wait slots")
    for field in S2U_HOL_COUNT_FIELDS:
        value = row[field]
        if row["task_type"] == "FOV":
            if value is not None:
                raise ValueError(f"FOV packet outcome field {field} must be null")
        elif (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"COM packet outcome field {field} must be a non-negative integer"
            )
    if row["task_type"] == "COM" and row["s2u_hol_opportunity_slot_count"] != (
        row["s2u_hol_service_slot_count"]
        + sum(row[field] for field in S2U_HOL_NO_SERVICE_COUNT_FIELDS)
    ):
        raise ValueError(
            "S2U HOL opportunities do not equal service plus no-service reasons"
        )
    for field in (
        "generation_time_seconds",
        "finish_time_seconds",
        "deadline_seconds",
        "routing_wait_seconds",
        "locked_receiver_out_of_range_wait_seconds",
        "cumulative_uav_queue_delay_seconds",
        "cumulative_uav_tx_delay_seconds",
    ):
        _validate_nonnegative_number(row, field)
    for field in (
        "e2e_delay_seconds",
        "s2u_completion_time_seconds",
        "routing_eligible_time_seconds",
        "s2u_queue_delay_seconds",
        "s2u_tx_delay_seconds",
    ):
        _validate_nonnegative_number(row, field, nullable=True)
    return row


def validate_packet_outcome_episode_record(record):
    if not isinstance(record, dict):
        raise TypeError("packet outcome episode record must be a dictionary")
    if record.get("artifact_schema_version") != (
        PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("packet outcome artifact schema version is invalid")
    for field in ("scenario_id", "summary", "packet_outcomes"):
        if field not in record:
            raise ValueError(
                f"packet outcome episode record lacks required field: {field}"
            )
    if not isinstance(record["summary"], dict):
        raise TypeError("packet outcome summary must be a dictionary")
    if not isinstance(record["packet_outcomes"], list):
        raise TypeError("packet outcomes must be an episode-local list")
    for row in record["packet_outcomes"]:
        validate_packet_outcome(row)
    return record


def packet_outcome_episode_record(scenario_id, summary, packet_outcomes):
    """Build one traceable episode record without copying packet dictionaries."""

    if not isinstance(packet_outcomes, list):
        raise TypeError("packet outcomes must be an episode-local list")
    if not isinstance(summary, dict):
        raise TypeError("packet outcome summary must be a dictionary")
    record = {
        "artifact_schema_version": PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "summary": summary,
        "packet_outcomes": packet_outcomes,
    }
    return validate_packet_outcome_episode_record(record)


def _empty_diagnostic_group():
    return {
        "eligible_packets": 0,
        "violated_packets": 0,
        "pre_s2u_violation_count": 0,
        "pre_s2u_violation_never_became_hol_count": 0,
        "pre_s2u_violation_became_hol_never_started_count": 0,
        "pre_s2u_violation_partial_service_count": 0,
        "post_s2u_violation_count": 0,
        "expired_at_sr_count": 0,
        "expired_at_uav_count": 0,
        "loop_packet_count": 0,
        "loop_violation_count": 0,
        "non_loop_violation_count": 0,
        "total_routing_decision_slots": 0,
        "total_wait_slots": 0,
        "total_voluntary_wait_with_legal_nonwait_slots": 0,
        "total_only_wait_no_available_link_slots": 0,
        "packets_with_voluntary_wait": 0,
        "packets_with_only_wait_no_available_link": 0,
        "violations_with_voluntary_wait": 0,
        "violations_with_only_wait_no_available_link": 0,
        "forced_locked_out_of_range_wait_slots": 0,
        "packets_with_forced_locked_wait": 0,
        "violations_with_forced_locked_wait": 0,
        "sum_completed_uav_hops": 0.0,
        "sum_cumulative_uav_queue_delay_seconds": 0.0,
        "sum_cumulative_uav_tx_delay_seconds": 0.0,
        "sum_wait_slots": 0.0,
        "sum_forced_locked_wait_slots": 0.0,
        "total_s2u_hol_opportunity_slots": 0,
        "total_s2u_hol_service_slots": 0,
        "total_s2u_hol_no_receiver_slots": 0,
        "total_s2u_hol_receiver_out_of_range_slots": 0,
        "total_s2u_hol_no_active_link_slots": 0,
        "total_s2u_hol_no_positive_capacity_slots": 0,
        "total_s2u_hol_positive_capacity_but_no_service_slots": 0,
        "loop_hop_sum": 0.0,
        "non_loop_hop_sum": 0.0,
        "non_loop_packet_count": 0,
        "expired_packet_count_by_terminal_uav": {},
    }


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


class PacketRoutingDiagnosticAccumulator:
    """Incrementally aggregate packet diagnostics without retaining packet rows."""

    def __init__(self):
        self._groups = {
            group: _empty_diagnostic_group() for group in ("ALL", "COM", "FOV")
        }

    def add_packet_outcomes(self, packet_outcomes, *, validate=True):
        for row in packet_outcomes:
            if validate:
                validate_packet_outcome(row)
            if not bool(row["qos_eligible"]):
                continue
            for group_name in ("ALL", row["task_type"]):
                self._add_row(self._groups[group_name], row)

    @staticmethod
    def _add_row(group, row):
        group["eligible_packets"] += 1
        violated = row["outcome"] in {"late_delivered", "expired_dropped"}
        if violated:
            group["violated_packets"] += 1
            if row["task_type"] == "COM":
                if row["s2u_completed"] is True:
                    group["post_s2u_violation_count"] += 1
                elif row["s2u_completed"] is False:
                    group["pre_s2u_violation_count"] += 1
                    hol_opportunities = int(
                        row["s2u_hol_opportunity_slot_count"]
                    )
                    if hol_opportunities == 0:
                        group[
                            "pre_s2u_violation_never_became_hol_count"
                        ] += 1
                    elif float(row["s2u_tx_delay_seconds"]) == 0.0:
                        group[
                            "pre_s2u_violation_became_hol_never_started_count"
                        ] += 1
                    else:
                        group[
                            "pre_s2u_violation_partial_service_count"
                        ] += 1
            if row["has_repeated_uav"]:
                group["loop_violation_count"] += 1
            else:
                group["non_loop_violation_count"] += 1
        if row["outcome"] == "expired_dropped":
            if row["terminal_node_type"] == "SR":
                group["expired_at_sr_count"] += 1
            elif row["terminal_node_type"] == "UAV":
                group["expired_at_uav_count"] += 1
                terminal_uav = str(int(row["terminal_uav_id"]))
                distribution = group["expired_packet_count_by_terminal_uav"]
                distribution[terminal_uav] = distribution.get(terminal_uav, 0) + 1
        if row["has_repeated_uav"]:
            group["loop_packet_count"] += 1
            group["loop_hop_sum"] += float(row["completed_uav_hop_count"])
        else:
            group["non_loop_packet_count"] += 1
            group["non_loop_hop_sum"] += float(
                row["completed_uav_hop_count"]
            )
        decision_slots = int(row["routing_decision_slot_count"])
        wait_slots = int(row["routing_wait_slot_count"])
        voluntary_slots = int(
            row["routing_voluntary_wait_with_legal_nonwait_slot_count"]
        )
        only_wait_slots = int(
            row["routing_only_wait_no_available_link_slot_count"]
        )
        forced_slots = int(
            row["locked_receiver_out_of_range_wait_slot_count"]
        )
        group["total_routing_decision_slots"] += decision_slots
        group["total_wait_slots"] += wait_slots
        group["total_voluntary_wait_with_legal_nonwait_slots"] += voluntary_slots
        group["total_only_wait_no_available_link_slots"] += only_wait_slots
        group["forced_locked_out_of_range_wait_slots"] += forced_slots
        group["sum_wait_slots"] += wait_slots
        group["sum_forced_locked_wait_slots"] += forced_slots
        if forced_slots > 0:
            group["packets_with_forced_locked_wait"] += 1
            if violated:
                group["violations_with_forced_locked_wait"] += 1
        if voluntary_slots > 0:
            group["packets_with_voluntary_wait"] += 1
            if violated:
                group["violations_with_voluntary_wait"] += 1
        if only_wait_slots > 0:
            group["packets_with_only_wait_no_available_link"] += 1
            if violated:
                group["violations_with_only_wait_no_available_link"] += 1
        if row["task_type"] == "COM":
            group["total_s2u_hol_opportunity_slots"] += int(
                row["s2u_hol_opportunity_slot_count"]
            )
            group["total_s2u_hol_service_slots"] += int(
                row["s2u_hol_service_slot_count"]
            )
            group["total_s2u_hol_no_receiver_slots"] += int(
                row["s2u_hol_no_receiver_slot_count"]
            )
            group["total_s2u_hol_receiver_out_of_range_slots"] += int(
                row["s2u_hol_receiver_out_of_range_slot_count"]
            )
            group["total_s2u_hol_no_active_link_slots"] += int(
                row["s2u_hol_no_active_link_slot_count"]
            )
            group["total_s2u_hol_no_positive_capacity_slots"] += int(
                row["s2u_hol_no_positive_capacity_slot_count"]
            )
            group[
                "total_s2u_hol_positive_capacity_but_no_service_slots"
            ] += int(
                row["s2u_hol_positive_capacity_but_no_service_slot_count"]
            )
        group["sum_completed_uav_hops"] += float(
            row["completed_uav_hop_count"]
        )
        group["sum_cumulative_uav_queue_delay_seconds"] += float(
            row["cumulative_uav_queue_delay_seconds"]
        )
        group["sum_cumulative_uav_tx_delay_seconds"] += float(
            row["cumulative_uav_tx_delay_seconds"]
        )

    def summary(self):
        groups = {}
        for name, raw in self._groups.items():
            eligible = raw["eligible_packets"]
            violations = raw["violated_packets"]
            com_only = name == "COM"
            if raw["total_wait_slots"] != (
                raw["total_voluntary_wait_with_legal_nonwait_slots"]
                + raw["total_only_wait_no_available_link_slots"]
            ):
                raise AssertionError(
                    f"{name} aggregate routing Wait decomposition failed"
                )
            if raw["forced_locked_out_of_range_wait_slots"] > raw[
                "total_only_wait_no_available_link_slots"
            ]:
                raise AssertionError(
                    f"{name} forced locked Wait subtype conservation failed"
                )
            s2u_no_service_slots = sum(
                raw[field]
                for field in (
                    "total_s2u_hol_no_receiver_slots",
                    "total_s2u_hol_receiver_out_of_range_slots",
                    "total_s2u_hol_no_active_link_slots",
                    "total_s2u_hol_no_positive_capacity_slots",
                    "total_s2u_hol_positive_capacity_but_no_service_slots",
                )
            )
            if raw["total_s2u_hol_opportunity_slots"] != (
                raw["total_s2u_hol_service_slots"] + s2u_no_service_slots
            ):
                raise AssertionError(
                    f"{name} aggregate S2U HOL decomposition failed"
                )
            if raw["pre_s2u_violation_count"] != (
                raw["pre_s2u_violation_never_became_hol_count"]
                + raw["pre_s2u_violation_became_hol_never_started_count"]
                + raw["pre_s2u_violation_partial_service_count"]
            ):
                raise AssertionError(
                    f"{name} pre-S2U violation decomposition failed"
                )
            groups[name] = {
                "task_type": name,
                "eligible_packets": eligible,
                "violated_packets": violations,
                "violation_probability": _ratio(violations, eligible),
                "pre_s2u_violation_count": (
                    raw["pre_s2u_violation_count"] if com_only else None
                ),
                "pre_s2u_violation_never_became_hol_count": (
                    raw["pre_s2u_violation_never_became_hol_count"]
                    if com_only
                    else None
                ),
                "pre_s2u_violation_never_became_hol_share": (
                    _ratio(
                        raw["pre_s2u_violation_never_became_hol_count"],
                        raw["pre_s2u_violation_count"],
                    )
                    if com_only
                    else None
                ),
                "pre_s2u_violation_became_hol_never_started_count": (
                    raw["pre_s2u_violation_became_hol_never_started_count"]
                    if com_only
                    else None
                ),
                "pre_s2u_violation_became_hol_never_started_share": (
                    _ratio(
                        raw[
                            "pre_s2u_violation_became_hol_never_started_count"
                        ],
                        raw["pre_s2u_violation_count"],
                    )
                    if com_only
                    else None
                ),
                "pre_s2u_violation_partial_service_count": (
                    raw["pre_s2u_violation_partial_service_count"]
                    if com_only
                    else None
                ),
                "pre_s2u_violation_partial_service_share": (
                    _ratio(
                        raw["pre_s2u_violation_partial_service_count"],
                        raw["pre_s2u_violation_count"],
                    )
                    if com_only
                    else None
                ),
                "post_s2u_violation_count": (
                    raw["post_s2u_violation_count"] if com_only else None
                ),
                "post_s2u_violation_share": (
                    _ratio(raw["post_s2u_violation_count"], violations)
                    if com_only
                    else None
                ),
                "expired_at_sr_count": raw["expired_at_sr_count"],
                "expired_at_uav_count": raw["expired_at_uav_count"],
                "loop_packet_count": raw["loop_packet_count"],
                "loop_packet_probability": _ratio(
                    raw["loop_packet_count"], eligible
                ),
                "loop_violation_count": raw["loop_violation_count"],
                "non_loop_violation_count": raw["non_loop_violation_count"],
                "loop_share_among_violations": _ratio(
                    raw["loop_violation_count"], violations
                ),
                "total_routing_decision_slots": raw[
                    "total_routing_decision_slots"
                ],
                "total_wait_slots": raw["total_wait_slots"],
                "total_voluntary_wait_with_legal_nonwait_slots": raw[
                    "total_voluntary_wait_with_legal_nonwait_slots"
                ],
                "total_only_wait_no_available_link_slots": raw[
                    "total_only_wait_no_available_link_slots"
                ],
                "voluntary_wait_share_among_wait_slots": _ratio(
                    raw["total_voluntary_wait_with_legal_nonwait_slots"],
                    raw["total_wait_slots"],
                ),
                "only_wait_no_available_link_share_among_wait_slots": _ratio(
                    raw["total_only_wait_no_available_link_slots"],
                    raw["total_wait_slots"],
                ),
                "voluntary_wait_fraction_of_routing_decisions": _ratio(
                    raw["total_voluntary_wait_with_legal_nonwait_slots"],
                    raw["total_routing_decision_slots"],
                ),
                "only_wait_no_available_link_fraction_of_routing_decisions": _ratio(
                    raw["total_only_wait_no_available_link_slots"],
                    raw["total_routing_decision_slots"],
                ),
                "packets_with_voluntary_wait": raw[
                    "packets_with_voluntary_wait"
                ],
                "packets_with_only_wait_no_available_link": raw[
                    "packets_with_only_wait_no_available_link"
                ],
                "violations_with_voluntary_wait": raw[
                    "violations_with_voluntary_wait"
                ],
                "violations_with_only_wait_no_available_link": raw[
                    "violations_with_only_wait_no_available_link"
                ],
                "share_of_violations_with_voluntary_wait": _ratio(
                    raw["violations_with_voluntary_wait"], violations
                ),
                "share_of_violations_with_only_wait_no_available_link": _ratio(
                    raw["violations_with_only_wait_no_available_link"],
                    violations,
                ),
                "wait_slot_fraction": _ratio(
                    raw["total_wait_slots"], raw["total_routing_decision_slots"]
                ),
                "forced_locked_out_of_range_wait_slots": raw[
                    "forced_locked_out_of_range_wait_slots"
                ],
                "forced_locked_out_of_range_wait_fraction": _ratio(
                    raw["forced_locked_out_of_range_wait_slots"],
                    raw["total_routing_decision_slots"],
                ),
                "mean_wait_slots_per_packet": _ratio(
                    raw["sum_wait_slots"], eligible
                ),
                "mean_forced_locked_wait_slots_per_packet": _ratio(
                    raw["sum_forced_locked_wait_slots"], eligible
                ),
                "packets_with_forced_locked_wait": raw[
                    "packets_with_forced_locked_wait"
                ],
                "violations_with_forced_locked_wait": raw[
                    "violations_with_forced_locked_wait"
                ],
                "share_of_violations_with_forced_locked_wait": _ratio(
                    raw["violations_with_forced_locked_wait"], violations
                ),
                "mean_completed_uav_hops": _ratio(
                    raw["sum_completed_uav_hops"], eligible
                ),
                "mean_cumulative_uav_queue_delay_seconds": _ratio(
                    raw["sum_cumulative_uav_queue_delay_seconds"], eligible
                ),
                "mean_cumulative_uav_tx_delay_seconds": _ratio(
                    raw["sum_cumulative_uav_tx_delay_seconds"], eligible
                ),
                "mean_hops_loop_packets": _ratio(
                    raw["loop_hop_sum"], raw["loop_packet_count"]
                ),
                "mean_hops_non_loop_packets": _ratio(
                    raw["non_loop_hop_sum"], raw["non_loop_packet_count"]
                ),
                "total_s2u_hol_opportunity_slots": (
                    raw["total_s2u_hol_opportunity_slots"]
                    if com_only
                    else None
                ),
                "total_s2u_hol_service_slots": (
                    raw["total_s2u_hol_service_slots"] if com_only else None
                ),
                "total_s2u_hol_no_receiver_slots": (
                    raw["total_s2u_hol_no_receiver_slots"]
                    if com_only
                    else None
                ),
                "total_s2u_hol_receiver_out_of_range_slots": (
                    raw["total_s2u_hol_receiver_out_of_range_slots"]
                    if com_only
                    else None
                ),
                "total_s2u_hol_no_active_link_slots": (
                    raw["total_s2u_hol_no_active_link_slots"]
                    if com_only
                    else None
                ),
                "total_s2u_hol_no_positive_capacity_slots": (
                    raw["total_s2u_hol_no_positive_capacity_slots"]
                    if com_only
                    else None
                ),
                "total_s2u_hol_positive_capacity_but_no_service_slots": (
                    raw[
                        "total_s2u_hol_positive_capacity_but_no_service_slots"
                    ]
                    if com_only
                    else None
                ),
                "s2u_hol_service_fraction": (
                    _ratio(
                        raw["total_s2u_hol_service_slots"],
                        raw["total_s2u_hol_opportunity_slots"],
                    )
                    if com_only
                    else None
                ),
                "s2u_no_receiver_share_among_no_service_hol_slots": (
                    _ratio(raw["total_s2u_hol_no_receiver_slots"], s2u_no_service_slots)
                    if com_only
                    else None
                ),
                "s2u_receiver_out_of_range_share_among_no_service_hol_slots": (
                    _ratio(
                        raw["total_s2u_hol_receiver_out_of_range_slots"],
                        s2u_no_service_slots,
                    )
                    if com_only
                    else None
                ),
                "s2u_no_active_link_share_among_no_service_hol_slots": (
                    _ratio(
                        raw["total_s2u_hol_no_active_link_slots"],
                        s2u_no_service_slots,
                    )
                    if com_only
                    else None
                ),
                "s2u_no_positive_capacity_share_among_no_service_hol_slots": (
                    _ratio(
                        raw["total_s2u_hol_no_positive_capacity_slots"],
                        s2u_no_service_slots,
                    )
                    if com_only
                    else None
                ),
                "s2u_positive_capacity_but_no_service_share_among_no_service_hol_slots": (
                    _ratio(
                        raw[
                            "total_s2u_hol_positive_capacity_but_no_service_slots"
                        ],
                        s2u_no_service_slots,
                    )
                    if com_only
                    else None
                ),
                "expired_packet_count_by_terminal_uav": dict(
                    sorted(
                        raw["expired_packet_count_by_terminal_uav"].items(),
                        key=lambda item: int(item[0]),
                    )
                ),
            }
        return {
            "packet_routing_diagnostic_contract_version": (
                PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION
            ),
            **PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS,
            "groups": groups,
        }


def packet_routing_diagnostics_from_outcomes(packet_outcomes):
    accumulator = PacketRoutingDiagnosticAccumulator()
    accumulator.add_packet_outcomes(packet_outcomes)
    return accumulator.summary()


def write_packet_routing_diagnostic_artifacts(output_directory, diagnostics):
    output_directory = Path(output_directory)
    json_path = output_directory / "packet_routing_diagnostics.json"
    csv_path = output_directory / "packet_routing_diagnostics.csv"
    terminal_path = output_directory / "terminal_uav_distribution.csv"
    json_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    groups = diagnostics["groups"]
    csv_rows = []
    for name in ("ALL", "COM", "FOV"):
        row = dict(groups[name])
        row["expired_packet_count_by_terminal_uav"] = json.dumps(
            row["expired_packet_count_by_terminal_uav"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        csv_rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    terminal_rows = []
    for name in ("ALL", "COM", "FOV"):
        group = groups[name]
        for terminal_uav_id, count in group[
            "expired_packet_count_by_terminal_uav"
        ].items():
            terminal_rows.append(
                {
                    "task_type": name,
                    "terminal_uav_id": int(terminal_uav_id),
                    "expired_packet_count": int(count),
                    "share_of_task_violations": _ratio(
                        count, group["violated_packets"]
                    ),
                }
            )
    with terminal_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "task_type",
            "terminal_uav_id",
            "expired_packet_count",
            "share_of_task_violations",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(terminal_rows)
    return {
        "packet_routing_diagnostics_json": json_path.resolve(),
        "packet_routing_diagnostics_csv": csv_path.resolve(),
        "terminal_uav_distribution_csv": terminal_path.resolve(),
    }


class PacketOutcomeJsonlWriter:
    """Write and flush one complete episode per JSONL record."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None
        self.episode_count = 0
        self._routing_diagnostics = PacketRoutingDiagnosticAccumulator()

    @property
    def closed(self):
        return self._handle is None or self._handle.closed

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("packet outcome writer is already open")
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def write_episode(self, record):
        if self._handle is None or self._handle.closed:
            raise RuntimeError("packet outcome writer is not open")
        validate_packet_outcome_episode_record(record)
        json.dump(
            record,
            self._handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()
        self._routing_diagnostics.add_packet_outcomes(
            record["packet_outcomes"], validate=False
        )
        self.episode_count += 1

    def routing_diagnostics(self):
        return self._routing_diagnostics.summary()

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._handle.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
