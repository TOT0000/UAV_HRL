"""Annealed LLM guidance for TD3 movement exploration only.

This comparison method deliberately keeps safe-DDQN on its original action
selection path.  Routing records are retained only to compute compact,
movement-relevant evidence and never become a routing control input.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_explorer import (
    ExplorerValidationError,
    LLMExplorerConfig,
    LLMExplorerController,
    NUM_UAV,
    ROUTING_ACTION_DIM,
    PromptBudgetError,
    _atomic_json,
    _compact_json,
    _json_clone,
    environment_constants_and_definitions,
    prompt_token_budget,
    render_prompt,
    validate_td3_bias,
)
from llm_explorer_v3 import (
    LLM_EXPLORER_V3_ANNEAL_END_EPISODE,
    LLM_EXPLORER_V3_MAX_INPUT_TOKENS,
    LLM_EXPLORER_V3_TARGET_INPUT_TOKENS,
    LLMExplorerV3Controller,
    _diverse_case_subset,
    _numeric_summary,
    build_observable_initial_state_v3,
    first_eligible_episode,
    llm_guidance_weight,
    movement_episode_summary,
)


LLM_EXPLORER_TD3_ONLY_METHOD_ID = "td3_dinkelbach_llm_explorer_td3_only"
LLM_EXPLORER_TD3_ONLY_VERSION = "uav-hrl-llm-explorer-td3-only-v1"
LLM_EXPLORER_TD3_ONLY_PROMPT_VERSION = (
    "uav-hrl-llm-explorer-td3-only-prompts-v3"
)
LLM_EXPLORER_TD3_ONLY_SUMMARY_VERSION = (
    "shared-goodput-movement-communication-summary-v3"
)
LLM_EXPLORER_TD3_ONLY_CASE_VERSION = "movement-service-cases-v1"
LLM_EXPLORER_TD3_ONLY_GUIDANCE_CONTEXT_VERSION = (
    "self-contained-raw-td3-only-v1"
)
LLM_EXPLORER_TD3_ONLY_ANNEAL_END_EPISODE = (
    LLM_EXPLORER_V3_ANNEAL_END_EPISODE
)
LLM_EXPLORER_TD3_ONLY_MAX_INPUT_TOKENS = LLM_EXPLORER_V3_MAX_INPUT_TOKENS
LLM_EXPLORER_TD3_ONLY_TARGET_INPUT_TOKENS = (
    LLM_EXPLORER_V3_TARGET_INPUT_TOKENS
)


@dataclass(frozen=True)
class LLMExplorerTD3OnlyConfig(LLMExplorerConfig):
    prompt_version: str = LLM_EXPLORER_TD3_ONLY_PROMPT_VERSION
    input_hard_limit_tokens: int = LLM_EXPLORER_TD3_ONLY_MAX_INPUT_TOKENS
    annealing_end_episode: int = LLM_EXPLORER_TD3_ONLY_ANNEAL_END_EPISODE
    episode_joint_transitions: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_version != LLM_EXPLORER_TD3_ONLY_PROMPT_VERSION:
            raise ValueError("TD3-only prompt version is fixed by the method contract")
        if int(self.input_hard_limit_tokens) != LLM_EXPLORER_TD3_ONLY_MAX_INPUT_TOKENS:
            raise ValueError("TD3-only input hard limit must remain 20,000 tokens")
        if int(self.annealing_end_episode) != LLM_EXPLORER_TD3_ONLY_ANNEAL_END_EPISODE:
            raise ValueError("TD3-only annealing end episode must remain 600")
        if int(self.episode_joint_transitions) <= 0:
            raise ValueError("episode_joint_transitions must be positive")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExplorerValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_guidance_response_td3_only(
    content: str, sigma_episode_start: float
) -> tuple[np.ndarray | None, dict[str, str]]:
    """Validate the exact single-field TD3-only numerical contract."""

    try:
        value = json.loads(str(content), object_pairs_hook=_reject_duplicate_pairs)
    except (TypeError, json.JSONDecodeError, ExplorerValidationError) as exc:
        return None, {"td3": f"invalid JSON: {type(exc).__name__}: {exc}"}
    if not isinstance(value, dict):
        return None, {"td3": "response root is not an object"}
    if set(value) != {"td3_bias"}:
        return None, {"td3": "response must contain exactly td3_bias"}
    matrix = value["td3_bias"]
    if not isinstance(matrix, list) or len(matrix) != NUM_UAV:
        return None, {"td3": "td3_bias must have shape [16,3]"}
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            return None, {"td3": "td3_bias must have shape [16,3]"}
        if any(isinstance(item, (bool, np.bool_)) for item in row):
            return None, {"td3": "td3_bias must contain JSON numbers, not booleans"}
        if any(not isinstance(item, (int, float)) for item in row):
            return None, {"td3": "td3_bias must contain finite JSON numbers only"}
    try:
        return validate_td3_bias(matrix, sigma_episode_start), {}
    except (TypeError, ValueError, ExplorerValidationError) as exc:
        return None, {"td3": str(exc)}


def output_schema_td3_only() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["td3_bias"],
        "additionalProperties": False,
        "properties": {
            "td3_bias": {
                "type": "array",
                "minItems": NUM_UAV,
                "maxItems": NUM_UAV,
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
            }
        },
        "semantic_constraints": [
            "Rows are UAV IDs 0 through 15.",
            "Columns are speed, heading, vertical.",
            "Every value is a finite JSON number within the supplied current sigma bound.",
            "Booleans, strings, duplicate keys, and additional fields are forbidden.",
        ],
    }


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (size,) or not np.isfinite(result).all():
        return None
    return result


def _component_summaries(values: list[np.ndarray]) -> list[dict[str, Any]]:
    if not values:
        return [_numeric_summary([]) for _ in range(3)]
    matrix = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    return [_numeric_summary(matrix[:, index].tolist(), 6) for index in range(3)]


def movement_episode_summary_td3_only(
    records: list[dict[str, Any]], guidance: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extend v3 full-record movement statistics with executed/applied values."""

    summaries, saturation_cases = movement_episode_summary(records, guidance)
    geometry_cases: list[dict[str, Any]] = []
    for uav_id in range(NUM_UAV):
        executed: list[np.ndarray] = []
        applied: list[np.ndarray] = []
        actor: list[np.ndarray] = []
        for record in records:
            uavs = record.get("uavs")
            if not isinstance(uavs, list) or uav_id >= len(uavs):
                continue
            item = uavs[uav_id]
            if not isinstance(item, dict) or not bool(item.get("movement_mask", False)):
                continue
            actor_value = _finite_vector(item.get("actor_normalized_action"), 3)
            executed_value = _finite_vector(item.get("executed_normalized_action"), 3)
            applied_value = _finite_vector(item.get("applied_llm_bias"), 3)
            if actor_value is not None:
                actor.append(actor_value)
            if executed_value is not None:
                executed.append(executed_value)
            if applied_value is not None:
                applied.append(applied_value)
            targets = item.get("observable_targets")
            if executed_value is not None and isinstance(targets, list) and targets:
                relative = _finite_vector(targets[0].get("relative_position_m"), 3)
                if relative is not None:
                    geometry_cases.append(
                        {
                            "domain": "movement",
                            "kind": "controlled_movement_with_observable_service_geometry",
                            "episode": int(record.get("episode_index", -1)) + 1,
                            "time_s": record.get("time_seconds"),
                            "uav_id": uav_id,
                            "tasks": item.get("task_types", []),
                            "target_kind": targets[0].get("kind"),
                            "relative_target_m": relative.round(3).tolist(),
                            "target_distance_m": round(float(np.linalg.norm(relative)), 3),
                            "actor_action": None if actor_value is None else actor_value.round(5).tolist(),
                            "executed_action": executed_value.round(5).tolist(),
                            "applied_effective_bias": None if applied_value is None else applied_value.round(6).tolist(),
                        }
                    )
        entry = summaries[f"UAV_{uav_id}"]
        entry["executed_normalized_action_by_component"] = _component_summaries(executed)
        entry["actor_abs_ge_0.95_ratio_by_component"] = (
            None
            if not actor
            else np.mean(np.abs(np.asarray(actor)) >= 0.95, axis=0).round(6).tolist()
        )
        entry["executed_abs_ge_0.95_ratio_by_component"] = (
            None
            if not executed
            else np.mean(np.abs(np.asarray(executed)) >= 0.95, axis=0).round(6).tolist()
        )
        entry["actually_applied_effective_bias_by_component"] = _component_summaries(applied)
        entry["actually_applied_effective_bias_count"] = len(applied)
    candidates = saturation_cases + geometry_cases
    return summaries, _diverse_case_subset(candidates, 3)


def _summary_tuple(value: Any) -> list[Any]:
    if not isinstance(value, dict):
        return [None, None, None, None]
    return [value.get(key) for key in ("count", "mean", "min", "max")]


def compact_movement_summaries_td3_only(value: Any) -> dict[str, Any]:
    summaries = value if isinstance(value, dict) else {}
    rows: list[list[Any]] = []
    for uav_id in range(NUM_UAV):
        item = summaries.get(f"UAV_{uav_id}")
        if not isinstance(item, dict):
            rows.append([uav_id] + [None] * 17)
            continue
        tasks = item.get("task_occurrence_counts")
        applied = item.get("actually_applied_effective_bias_by_component")
        rows.append(
            [
                uav_id,
                None if not isinstance(tasks, dict) else [[str(key), int(count)] for key, count in sorted(tasks.items())],
                item.get("movement_mask_true_count"),
                item.get("td3_controlled_actor_decision_count"),
                _summary_tuple(item.get("actor_horizontal_speed_mps")),
                _summary_tuple(item.get("actor_vertical_velocity_mps")),
                item.get("actor_heading_circular_mean_rad"),
                item.get("actor_near_10_mps_ratio"),
                item.get("actor_abs_ge_0.95_ratio_by_component"),
                [
                    _summary_tuple(component)
                    for component in item.get("executed_normalized_action_by_component", [])
                ],
                item.get("executed_abs_ge_0.95_ratio_by_component"),
                item.get("raw_bias"),
                item.get("effective_bias"),
                item.get("episode_start_sigma"),
                item.get("actually_applied_effective_bias_count"),
                None if not isinstance(applied, list) else [_summary_tuple(component) for component in applied],
                _summary_tuple(item.get("observable_target_distance_m")),
                _summary_tuple(item.get("movement_energy_j")),
            ]
        )
    return {
        "columns": [
            "uav_id",
            "role_occurrence_[name,count]",
            "movement_mask_true_count",
            "td3_actor_decision_count",
            "actor_horizontal_speed_mps_[count,mean,min,max]",
            "actor_vertical_velocity_mps_[count,mean,min,max]",
            "actor_heading_circular_mean_rad",
            "actor_speed_ge_9.5_ratio",
            "actor_abs_normalized_ge_0.95_ratio_[speed,heading,vertical]",
            "executed_normalized_[speed,heading,vertical]_each_[count,mean,min,max]",
            "executed_abs_normalized_ge_0.95_ratio_[speed,heading,vertical]",
            "requested_raw_bias_[speed,heading,vertical]",
            "requested_effective_w_times_raw_[speed,heading,vertical]",
            "episode_start_sigma",
            "actually_applied_effective_bias_count",
            "actually_applied_effective_[speed,heading,vertical]_each_[count,mean,min,max]",
            "observable_target_distance_m_[count,mean,min,max]",
            "movement_energy_j_[count,mean,min,max]",
        ],
        "semantics": (
            "All frequency statistics use the complete recorded episode. null means unknown; "
            "numeric zero is observed zero. Actor/executed/applied rows count only moments "
            "whose movement_mask was true; Search/Hover controller actions are not attributed to TD3."
        ),
        "rows": rows,
    }


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def communication_bottleneck_summary(
    records: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate movement-relevant communication evidence from all decisions."""

    per_sender: list[list[Any]] = []
    all_backlogs: list[float] = []
    all_deadlines: list[float] = []
    all_legal_transmit_counts: list[float] = []
    all_reference_capacities: list[float] = []
    valid_mask_decisions = 0
    known_backlog_decisions = 0
    positive_backlog_decisions = 0
    known_deadline_decisions = 0
    deadline_zero_decisions = 0
    missing_deadline_decisions = 0
    gs_legal_decisions = 0
    no_legal_transmit_decisions = 0
    zero_reference_capacity_links = 0
    senders_observed: set[int] = set()

    for sender in range(NUM_UAV):
        sender_records = [
            record
            for record in records
            if isinstance(record, dict) and record.get("sender_uav_id") == sender
        ]
        sender_backlogs: list[float] = []
        sender_deadlines: list[float] = []
        sender_legal_counts: list[float] = []
        sender_capacities: list[float] = []
        sender_valid_masks = 0
        sender_gs_legal = 0
        sender_no_transmit = 0
        sender_deadline_zero = 0
        sender_deadline_missing = 0
        if sender_records:
            senders_observed.add(sender)
        for record in sender_records:
            backlog = _safe_float(record.get("sender_total_backlog_bits"))
            if backlog is not None:
                sender_backlogs.append(backlog)
                all_backlogs.append(backlog)
                known_backlog_decisions += 1
                positive_backlog_decisions += int(backlog > 0.0)
            deadline = _safe_float(record.get("hol_remaining_deadline_seconds"))
            if deadline is None:
                sender_deadline_missing += 1
                missing_deadline_decisions += 1
            else:
                sender_deadlines.append(deadline)
                all_deadlines.append(deadline)
                known_deadline_decisions += 1
                sender_deadline_zero += int(deadline <= 0.0)
                deadline_zero_decisions += int(deadline <= 0.0)
            mask_value = record.get("legal_action_mask")
            try:
                mask = np.asarray(mask_value, dtype=bool)
            except (TypeError, ValueError):
                mask = np.asarray([], dtype=bool)
            if mask.shape == (ROUTING_ACTION_DIM,):
                sender_valid_masks += 1
                valid_mask_decisions += 1
                legal_transmissions = [
                    int(action)
                    for action in np.flatnonzero(mask)
                    if int(action) != sender
                ]
                sender_legal_counts.append(float(len(legal_transmissions)))
                all_legal_transmit_counts.append(float(len(legal_transmissions)))
                sender_no_transmit += int(not legal_transmissions)
                no_legal_transmit_decisions += int(not legal_transmissions)
                sender_gs_legal += int(bool(mask[NUM_UAV]))
                gs_legal_decisions += int(bool(mask[NUM_UAV]))
                references = record.get("reference_capacity_bps_by_legal_action")
                references = references if isinstance(references, dict) else {}
                for action in legal_transmissions:
                    capacity = _safe_float(
                        references.get(str(action), references.get(action))
                    )
                    if capacity is not None:
                        sender_capacities.append(capacity)
                        all_reference_capacities.append(capacity)
                        zero_reference_capacity_links += int(capacity == 0.0)
        per_sender.append(
            [
                sender,
                len(sender_records),
                _summary_tuple(_numeric_summary(sender_backlogs, 2)),
                _summary_tuple(_numeric_summary(sender_deadlines, 4)),
                sender_deadline_missing,
                sender_deadline_zero,
                sender_valid_masks,
                _summary_tuple(_numeric_summary(sender_legal_counts, 3)),
                sender_no_transmit,
                sender_gs_legal,
                _summary_tuple(_numeric_summary(sender_capacities, 2)),
            ]
        )

    system_eligible = metrics.get("system_qos_eligible_packets")
    system_violations = metrics.get("system_qos_violation_count")
    return {
        "scope": (
            "Full recorded episode. Per-sender rows describe observed access, queue, and "
            "HOL-deadline conditions only. System QoS is not attributed to individual UAVs."
        ),
        "units_and_definitions": {
            "backlog": "bits in the sender queue before the routing slot",
            "deadline": "seconds remaining for the observed head-of-line packet; null is unknown",
            "legal_transmit_action": "legal action excluding sender self-action WAIT",
            "reference_capacity": (
                "program-computed legal-link reference capacity in bit/s; it is not "
                "realized throughput, relay service, or eventual GS delivery"
            ),
            "denominators": "Counts are included explicitly; ratios are null when their denominator is zero",
        },
        "system": {
            "recorded_routing_decisions": len(records),
            "observed_sender_count_out_of_16": [len(senders_observed), NUM_UAV],
            "known_backlog_decisions": known_backlog_decisions,
            "positive_backlog_decisions": positive_backlog_decisions,
            "positive_backlog_ratio_of_known": _ratio(positive_backlog_decisions, known_backlog_decisions),
            "backlog_bits_[count,mean,min,max]": _summary_tuple(_numeric_summary(all_backlogs, 2)),
            "known_hol_deadline_decisions": known_deadline_decisions,
            "missing_hol_deadline_decisions": missing_deadline_decisions,
            "hol_remaining_deadline_s_[count,mean,min,max]": _summary_tuple(_numeric_summary(all_deadlines, 4)),
            "hol_deadline_at_or_below_zero_decisions": deadline_zero_decisions,
            "hol_deadline_at_or_below_zero_ratio_of_known": _ratio(deadline_zero_decisions, known_deadline_decisions),
            "valid_legal_mask_decisions": valid_mask_decisions,
            "legal_transmit_actions_per_valid_mask_[count,mean,min,max]": _summary_tuple(_numeric_summary(all_legal_transmit_counts, 3)),
            "no_legal_transmit_decisions": no_legal_transmit_decisions,
            "no_legal_transmit_ratio_of_valid_masks": _ratio(no_legal_transmit_decisions, valid_mask_decisions),
            "gs_legal_decisions": gs_legal_decisions,
            "gs_legal_ratio_of_valid_masks": _ratio(gs_legal_decisions, valid_mask_decisions),
            "legal_link_reference_capacity_bps_[count,mean,min,max]": _summary_tuple(_numeric_summary(all_reference_capacities, 2)),
            "zero_reference_capacity_legal_links": zero_reference_capacity_links,
            "system_qos_eligible_packets": system_eligible,
            "system_qos_violation_count": system_violations,
            "system_qos_violation_probability": metrics.get("system_qos_violation_probability"),
            "system_qos_scope": "system-wide packet outcome; not per-UAV attribution",
        },
        "per_sender_columns": [
            "sender_uav_id",
            "decision_count",
            "backlog_bits_[count,mean,min,max]",
            "hol_remaining_deadline_s_[count,mean,min,max]",
            "missing_hol_deadline_count",
            "hol_deadline_at_or_below_zero_count",
            "valid_legal_mask_count",
            "legal_transmit_actions_[count,mean,min,max]",
            "no_legal_transmit_count",
            "gs_legal_count",
            "legal_link_reference_capacity_bps_[count,mean,min,max]",
        ],
        "per_sender_rows": per_sender,
        "unavailable": [
            "per-UAV timely GS-delivery attribution",
            "per-UAV deadline-violation outcome attribution",
            "realized per-link service or throughput from these decision records",
        ],
    }


def _td3_bias_rows(value: Any) -> list[list[float]] | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (NUM_UAV, 3) or not np.isfinite(matrix).all():
        return None
    return matrix.round(6).tolist()


def recorded_raw_td3_bias(previous: dict[str, Any]) -> dict[str, Any]:
    guidance_value = previous.get("applied_guidance")
    guidance = guidance_value if isinstance(guidance_value, dict) else {}
    explicitly_unavailable = (
        guidance.get("enabled") is False
        or guidance.get("td3_fallback") is True
    )
    rows = (
        None
        if explicitly_unavailable
        else _td3_bias_rows(guidance.get("td3_bias_raw"))
    )
    source = "applied_guidance.td3_bias_raw"
    if rows is None and not explicitly_unavailable:
        movement = previous.get("movement_summaries")
        movement = movement if isinstance(movement, dict) else {}
        rows = _td3_bias_rows(
            [
                movement.get(f"UAV_{uav_id}", {}).get("raw_bias")
                if isinstance(movement.get(f"UAV_{uav_id}"), dict)
                else None
                for uav_id in range(NUM_UAV)
            ]
        )
        source = "movement_summaries.UAV_k.raw_bias"
    if (
        rows is None
        and not explicitly_unavailable
        and previous.get("schema_version") == LLM_EXPLORER_TD3_ONLY_VERSION
    ):
        # This method persists td3_bias as the validated raw model output and
        # creates td3_bias_effective separately after multiplying by w once.
        rows = _td3_bias_rows(guidance.get("td3_bias"))
        source = "legacy_td3_only_applied_guidance.td3_bias_confirmed_raw"
    if rows is None:
        source = (
            "unknown_no_valid_previous_intervention"
            if explicitly_unavailable
            else "unknown_not_recorded"
        )
    return {
        "source": source,
        "uav_id_order": list(range(NUM_UAV)),
        "columns": ["speed", "heading", "vertical"],
        "rows": rows,
        "semantics": (
            "previous-episode raw TD3 bias before multiplication by its annealing "
            "weight; rows=null means unknown, not zero"
        ),
    }


def compact_previous_movement_guidance(
    previous: dict[str, Any], *, include_application_rows: bool
) -> dict[str, Any]:
    guidance_value = previous.get("applied_guidance")
    guidance = guidance_value if isinstance(guidance_value, dict) else {}
    result = {
        key: _json_clone(guidance[key])
        for key in (
            "enabled",
            "td3_fallback",
            "td3_reason",
            "analysis_attempt",
            "td3_guidance_attempt",
            "annealing_weight",
            "sigma_episode_start",
        )
        if key in guidance
    }
    result["routing_guidance"] = "disabled_not_requested"
    if include_application_rows:
        summaries = previous.get("movement_summaries")
        summaries = summaries if isinstance(summaries, dict) else {}
        rows = []
        for uav_id in range(NUM_UAV):
            item = summaries.get(f"UAV_{uav_id}")
            item = item if isinstance(item, dict) else {}
            applied = item.get("actually_applied_effective_bias_by_component")
            means = (
                [component.get("mean") if isinstance(component, dict) else None for component in applied]
                if isinstance(applied, list)
                else None
            )
            rows.append(
                [
                    uav_id,
                    item.get("td3_controlled_actor_decision_count"),
                    item.get("actually_applied_effective_bias_count"),
                    means,
                ]
            )
        result["application_columns"] = [
            "uav_id",
            "td3_controlled_actor_decision_count",
            "actually_applied_effective_bias_count",
            "actually_applied_effective_bias_mean_[speed,heading,vertical]",
        ]
        result["application_rows"] = rows
    else:
        result["bias_and_application_location"] = (
            "per-UAV movement summary contains requested raw, effective w*raw, "
            "and actually applied effective bias statistics"
        )
    return result


def environment_definitions_td3_only(**kwargs: Any) -> dict[str, Any]:
    base = environment_constants_and_definitions(**kwargs)
    visual = base["visual_sensing"]
    return {
        "schema_version": LLM_EXPLORER_TD3_ONLY_VERSION,
        "num_uav": NUM_UAV,
        "uav_id_range_inclusive": [0, NUM_UAV - 1],
        "episode_seconds": base["episode_seconds"],
        "movement_interval_seconds": base["movement_interval_seconds"],
        "routing_slot_seconds": base["routing_slot_seconds"],
        "operating_area_m": base["operating_area_m"],
        "uav_altitude_bounds_m": base["uav_altitude_bounds_m"],
        "communication_range_3d_m_inclusive": base["communication_range_3d_m_inclusive"],
        "task_deadlines_seconds": base["task_deadlines_seconds"],
        "packet_generation_model": base["packet_generation_model"],
        "routing_decision_model": base["routing_decision_model"],
        "packet_service_model": base["packet_service_model"],
        "bandwidth_model": base["bandwidth_model"],
        "fading_model": base["fading_model"],
        "action_decode": base["action_decode"],
        "service_conditions": {
            "FOV": (
                "Visual-sensing service for an assigned discovered RoI. Packets are "
                "generated only while the oblique sensing geometry is valid."
            ),
            "COM": (
                "Communication service for an observable SR team assigned to a discovered "
                "RoI; assigned UAV-SR range uses three-dimensional distance."
            ),
            "Search": "Existing search controller owns movement; TD3 movement_mask is false.",
            "Hovering": "Existing hover controller owns movement; TD3 movement_mask is false.",
        },
        "movement_constraints": {
            "C9": (
                "FOV model-range/proximity condition over currently assigned FOV pairs; "
                "penalty is 1 minus clipped proximity."
            ),
            "C10": (
                "For C9-valid FOV pairs with finite geometry, the assigned RoI must fit "
                "within both along-tilt footprint margins; penalty is 1 minus clipped coverage factor."
            ),
            "COM_range": (
                "Assigned COM UAV-SR three-dimensional range condition; penalty increases "
                "outside the configured service range."
            ),
        },
        "visual_sensing": {
            "model": visual["vs_model"],
            "geometry_validity": visual["geometry_validity"],
            "packet_generation": visual["packet_generation"],
            "invalid_sensing": visual["invalid_sensing"],
        },
        "summary_definitions": {
            "full_episode_statistics": "computed from every recorded decision before selecting cases",
            "reference_capacity": "observable program reference in bit/s, not realized throughput",
            "representative_cases": "deterministic illustrations, not frequency estimates or proven optima",
            "system_qos": "system-level packet outcomes; not individual-UAV attribution",
        },
        "routing_control": "safe-DDQN remains host-controlled; no LLM routing prior or mixing",
        "validity": base["validity"],
    }


class DeterministicMockExplorerTD3OnlyClient:
    def chat(self, *, model: str, prompt: str, **_: Any) -> dict[str, Any]:
        if "Analyze the previous episode" in prompt:
            content = (
                "Key evidence: the offline mock supplies no directional causal evidence. "
                "Movement recommendations: retain neutral raw TD3 biases for all UAVs. "
                "Uncertainty: this deterministic mock is not live model inference."
            )
        else:
            content = _compact_json(
                {"td3_bias": [[0.0, 0.0, 0.0] for _ in range(NUM_UAV)]}
            )
        return {
            "content": content,
            "actual_model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }


class LLMExplorerTD3OnlyController(LLMExplorerV3Controller):
    schema_version = LLM_EXPLORER_TD3_ONLY_VERSION
    artifact_directory_name = "llm_explorer_td3_only"

    def __init__(
        self,
        *,
        config: LLMExplorerTD3OnlyConfig | None = None,
        mock: bool = False,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        resolved = config or LLMExplorerTD3OnlyConfig()
        if mock and client is None:
            client = DeterministicMockExplorerTD3OnlyClient()
            mock = False
        super().__init__(config=resolved, mock=mock, client=client, **kwargs)
        self.config: LLMExplorerTD3OnlyConfig

    @staticmethod
    def _unguided(reason: str) -> dict[str, Any]:
        value = LLMExplorerController._unguided(reason)
        value.update(
            routing_prior=None,
            routing_fallback=False,
            routing_reason="routing_guidance_disabled",
            routing_guidance_attempt=None,
        )
        return value

    @staticmethod
    def _disabled(reason: str, *, weight: float, alpha: float = 0.0) -> dict[str, Any]:
        del alpha
        value = LLMExplorerTD3OnlyController._unguided(reason)
        value.update(
            td3_fallback=False,
            annealing_weight=float(weight),
            routing_alpha=0.0,
            td3_bias_raw=np.zeros((NUM_UAV, 3), dtype=np.float32),
            td3_bias_effective=np.zeros((NUM_UAV, 3), dtype=np.float32),
            sigma_episode_start=None,
        )
        return value

    def _record_fallback_episode(
        self, episode_index: int, *, td3: bool, routing: bool
    ) -> None:
        # Routing is outside this method's guidance contract.  Calls inherited
        # from shared failure paths cannot turn its absence into a failure.
        del routing
        super()._record_fallback_episode(episode_index, td3=td3, routing=False)

    def _templates(self) -> tuple[str, str, str]:
        root = Path(__file__).resolve().parent / "prompts"
        return tuple(
            (root / name).read_text(encoding="utf-8")
            for name in (
                "llm_explorer_td3_only_system.txt",
                "llm_explorer_td3_only_analysis.txt",
                "llm_explorer_td3_only_guidance.txt",
            )
        )

    def _render_system_prompt(self, template: str, environment_constants: dict[str, Any]) -> str:
        return render_prompt(
            template,
            {"VERIFIED_MECHANISMS_AND_FIELD_DEFINITIONS": environment_constants},
        )

    def _cache_identity(self, **kwargs: Any) -> dict[str, Any]:
        identity = LLMExplorerController._cache_identity(self, **kwargs)
        identity.update(
            schema_version=LLM_EXPLORER_TD3_ONLY_VERSION,
            summary_version=LLM_EXPLORER_TD3_ONLY_SUMMARY_VERSION,
            case_selection_version=LLM_EXPLORER_TD3_ONLY_CASE_VERSION,
            annealing={
                "end_episode": self.config.annealing_end_episode,
                "episode_joint_transitions": self.config.episode_joint_transitions,
                "warmup_joint_transitions": self.config.warmup_joint_transitions,
            },
            routing_guidance="disabled",
        )
        if kwargs.get("stage") in {"guidance", "episode_result"}:
            identity["guidance_context_version"] = (
                LLM_EXPLORER_TD3_ONLY_GUIDANCE_CONTEXT_VERSION
            )
        return identity

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        raw = recorded_raw_td3_bias(previous)
        cases = previous.get("representative_cases")
        cases = [case for case in cases if isinstance(case, dict)] if isinstance(cases, list) else []
        cases = _diverse_case_subset(cases, 3)
        return {
            "metrics": _json_clone(previous.get("metrics", {})),
            "movement_summaries": compact_movement_summaries_td3_only(
                previous.get("movement_summaries", {})
            ),
            "communication_bottleneck_summary": _json_clone(
                previous.get("communication_bottleneck_summary", {})
            ),
            "representative_cases": {
                "selection": "deterministic diverse real records; maximum 3",
                "semantics": "illustrative only; complete summaries determine frequencies",
                "items": _json_clone(cases),
            },
            "movement_guidance_analysis": compact_previous_movement_guidance(
                previous, include_application_rows=False
            ),
            "movement_guidance_guidance": compact_previous_movement_guidance(
                previous, include_application_rows=True
            ),
            "previous_raw_td3_bias": raw,
            "_case_candidates": cases,
        }

    def _exploration_settings(self, *, sigma_episode_start: float, routing_epsilon: float) -> dict[str, Any]:
        del routing_epsilon
        return {
            "upcoming_episode": getattr(self, "_upcoming_episode", None),
            "sigma_episode_start": float(sigma_episode_start),
            "guidance_weight_w": float(self._current_weight),
            "td3_raw_bias_bound_inclusive": [
                -float(sigma_episode_start),
                float(sigma_episode_start),
            ],
            "first_eligible_episode": first_eligible_episode(
                self.config.warmup_joint_transitions,
                self.config.episode_joint_transitions,
            ),
            "annealing_end_episode": self.config.annealing_end_episode,
            "guidance_fixed_for_episode": True,
            "host_application": "effective_bias = w * raw_bias exactly once",
            "routing_guidance": "disabled; safe-DDQN original epsilon-greedy flow",
        }

    def _analysis_replacements(
        self,
        *,
        previous: dict[str, Any],
        current_observable_initial_state: dict[str, Any],
        settings: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "EPISODE_METRICS": previous["metrics"],
            "MOVEMENT_SUMMARIES": previous["movement_summaries"],
            "COMMUNICATION_BOTTLENECK_SUMMARY": previous[
                "communication_bottleneck_summary"
            ],
            "REPRESENTATIVE_CASES": previous["representative_cases"],
            "PREVIOUS_MOVEMENT_GUIDANCE": previous["movement_guidance_analysis"],
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
        }

    def _render_analysis_prompt(
        self,
        *,
        system_prompt: str,
        analysis_template: str,
        previous: dict[str, Any],
        environment_constants: dict[str, Any],
        current_observable_initial_state: dict[str, Any],
        settings: dict[str, Any],
    ) -> str:
        del environment_constants
        cases = list(previous.get("_case_candidates", []))
        last_prompt: str | None = None
        smallest_hard_limit_prompt: str | None = None
        # Fixed, reproducible compression order.  Core summaries and bounds are
        # never truncated; only real illustrative cases are reduced.
        for maximum in (3, 2, 1, 0):
            candidate = dict(previous)
            candidate["representative_cases"] = {
                "selection": f"deterministic diverse real records; maximum {maximum}",
                "semantics": "illustrative only; complete summaries determine frequencies",
                "items": _json_clone(_diverse_case_subset(cases, maximum)),
            }
            prompt = render_prompt(
                analysis_template,
                self._analysis_replacements(
                    previous=candidate,
                    current_observable_initial_state=current_observable_initial_state,
                    settings=settings,
                ),
            )
            budget = prompt_token_budget(
                system_prompt,
                prompt,
                reserved_output_tokens=self.config.analysis_max_output_tokens,
                context_window_tokens=self.config.context_window_tokens,
                safety_margin_tokens=self.config.safety_margin_tokens,
                encoding_name=self.config.tokenizer_encoding,
            )
            last_prompt = prompt
            if budget["input_tokens"] <= self.config.input_hard_limit_tokens:
                smallest_hard_limit_prompt = prompt
            if budget["input_tokens"] <= LLM_EXPLORER_TD3_ONLY_TARGET_INPUT_TOKENS:
                return prompt
        return smallest_hard_limit_prompt or str(last_prompt)

    def _guidance_replacements(
        self,
        *,
        analysis: str,
        previous: dict[str, Any],
        current_observable_initial_state: dict[str, Any],
        settings: dict[str, Any],
        validation_feedback: str,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "PREVIOUS_MOVEMENT_GUIDANCE": previous["movement_guidance_guidance"],
            "PREVIOUS_RAW_TD3_BIAS": previous["previous_raw_td3_bias"],
            "OUTPUT_JSON_SCHEMA": output_schema_td3_only(),
            "VALIDATION_FEEDBACK_OR_EMPTY": validation_feedback,
        }

    def _routing_guidance_required(self) -> bool:
        return False

    def _parse_guidance_response(
        self, content: str, sigma_episode_start: float
    ) -> tuple[np.ndarray | None, None, dict[str, str]]:
        td3, errors = parse_guidance_response_td3_only(
            content, sigma_episode_start
        )
        return td3, None, errors

    def _guidance_cache_content(
        self, td3_bias: np.ndarray, routing_prior: np.ndarray | None
    ) -> str:
        del routing_prior
        return _compact_json({"td3_bias": np.asarray(td3_bias).tolist()})

    def _load_episode_result(
        self, episode_index: int, identity: dict[str, Any], sigma: float
    ) -> dict[str, Any] | None:
        path = self._episode_result_path(episode_index)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version:
                return None
            if value.get("identity") != identity:
                return None
            guidance = self._restore_guidance(value["guidance"])
            if not guidance.get("td3_fallback"):
                guidance["td3_bias"] = validate_td3_bias(
                    guidance["td3_bias"], sigma
                )
            guidance.update(
                routing_prior=None,
                routing_fallback=False,
                routing_reason="routing_guidance_disabled",
                routing_guidance_attempt=None,
                reused=True,
                cache_source="episode_result",
            )
            return guidance
        except (OSError, KeyError, TypeError, ValueError, ExplorerValidationError):
            return None

    def _save_episode_result(
        self,
        *,
        episode_index: int,
        identity: dict[str, Any],
        guidance: dict[str, Any],
    ) -> None:
        path = self._episode_result_path(episode_index)
        if path is None:
            return
        safe = self._serializable_guidance(guidance)
        safe.update(
            routing_prior=None,
            routing_fallback=False,
            routing_reason="routing_guidance_disabled",
            routing_guidance_attempt=None,
        )
        try:
            _atomic_json(
                path,
                {
                    "schema_version": self.schema_version,
                    "identity": identity,
                    "guidance": safe,
                },
            )
        except OSError as exc:
            self._record_local_failure(
                episode_index=episode_index,
                stage="episode_result",
                attempt=0,
                category="artifact_write",
                reason=f"episode result cache: {type(exc).__name__}: {exc}",
            )

    def _prompt_budget(self, **kwargs: Any) -> dict[str, Any]:
        budget = LLMExplorerController._prompt_budget(self, **kwargs)
        budget.update(
            input_hard_limit_tokens=int(self.config.input_hard_limit_tokens),
            target_input_tokens=LLM_EXPLORER_TD3_ONLY_TARGET_INPUT_TOKENS,
            within_target_input_tokens=(
                int(budget["input_tokens"])
                <= LLM_EXPLORER_TD3_ONLY_TARGET_INPUT_TOKENS
            ),
            fits_input_hard_limit=(
                int(budget["input_tokens"])
                <= int(self.config.input_hard_limit_tokens)
            ),
        )
        attempt_root = kwargs.get("attempt_root")
        if attempt_root is not None:
            try:
                _atomic_json(Path(attempt_root) / "token_budget.json", budget)
            except OSError as exc:
                self._record_local_failure(
                    episode_index=kwargs["episode_index"],
                    stage=kwargs["stage"],
                    attempt=kwargs["attempt"],
                    category="artifact_write",
                    reason=f"TD3-only token budget artifact: {type(exc).__name__}: {exc}",
                )
        if int(budget["input_tokens"]) > int(self.config.input_hard_limit_tokens):
            reason = (
                f"{kwargs['stage']} input requires {budget['input_tokens']} tokens; "
                f"TD3-only hard limit is {self.config.input_hard_limit_tokens}"
            )
            self._record_local_failure(
                episode_index=kwargs["episode_index"],
                stage=kwargs["stage"],
                attempt=kwargs["attempt"],
                category="local_preflight",
                reason=reason,
            )
            raise PromptBudgetError(reason)
        return budget

    def begin_episode(
        self,
        *,
        episode_index: int,
        total_joint_transitions: int,
        sigma_episode_start: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.movement_records = []
        self.routing_records = []
        episode = int(episode_index) + 1
        self._upcoming_episode = episode
        start = first_eligible_episode(
            self.config.warmup_joint_transitions,
            self.config.episode_joint_transitions,
        )
        weight = llm_guidance_weight(
            episode,
            start_episode=start,
            end_episode=self.config.annealing_end_episode,
        )
        self._current_weight = weight
        self._current_alpha = 0.0
        if self.evaluation:
            self.active_guidance = self._disabled("evaluation_disabled", weight=0.0)
            return self.active_guidance
        if episode >= self.config.annealing_end_episode:
            self.active_guidance = self._disabled("annealing_finished", weight=0.0)
            return self.active_guidance
        if int(total_joint_transitions) < int(self.config.warmup_joint_transitions):
            self.active_guidance = self._disabled("warmup", weight=0.0)
            return self.active_guidance
        if weight <= 0.0:
            self.active_guidance = self._disabled("annealing_finished", weight=0.0)
            return self.active_guidance

        result = LLMExplorerController.begin_episode(
            self,
            episode_index=episode_index,
            total_joint_transitions=total_joint_transitions,
            sigma_episode_start=sigma_episode_start,
            **kwargs,
        )
        raw = np.asarray(result["td3_bias"], dtype=np.float32).reshape(NUM_UAV, 3)
        effective = (float(weight) * raw).astype(np.float32)
        result.update(
            annealing_weight=float(weight),
            routing_alpha=0.0,
            sigma_episode_start=float(sigma_episode_start),
            td3_bias_raw=raw,
            td3_bias_effective=effective,
            routing_prior=None,
            routing_fallback=False,
            routing_reason="routing_guidance_disabled",
            routing_guidance_attempt=None,
        )
        self.active_guidance = result
        print(
            f"[LLM explorer TD3-only] episode={episode} w={weight:.6f} "
            f"enabled={str(bool(result['enabled'])).lower()} "
            f"reason={result.get('td3_reason') or 'active'} "
            f"raw_bias_abs_max={float(np.max(np.abs(raw))):.6f} "
            f"effective_bias_abs_max={float(np.max(np.abs(effective))):.6f} "
            "routing_guidance=disabled",
            flush=True,
        )
        return result

    def complete_episode(self, *, episode_index: int, metrics: dict[str, Any]) -> None:
        movement_summary, cases = movement_episode_summary_td3_only(
            self.movement_records, self.active_guidance
        )
        communication = communication_bottleneck_summary(
            self.routing_records, metrics
        )
        action_sources = Counter(
            str(record.get("action_source", "unknown"))
            for record in self.movement_records
        )
        verified_metrics = {
            "episode": int(episode_index) + 1,
            **_json_clone(metrics),
            "movement_action_source_counts": dict(sorted(action_sources.items())),
            "previous_episode_guidance_weight": float(
                self.active_guidance.get("annealing_weight", 0.0)
            ),
            "guidance_enabled": bool(self.active_guidance.get("enabled", False)),
            "td3_fallback": bool(self.active_guidance.get("td3_fallback", False)),
            "routing_guidance": "disabled_not_requested",
            "disabled_reason": self.active_guidance.get("td3_reason"),
        }
        feedback = {
            "schema_version": LLM_EXPLORER_TD3_ONLY_VERSION,
            "summary_version": LLM_EXPLORER_TD3_ONLY_SUMMARY_VERSION,
            "case_selection_version": LLM_EXPLORER_TD3_ONLY_CASE_VERSION,
            "metrics": verified_metrics,
            "movement_summaries": movement_summary,
            "communication_bottleneck_summary": communication,
            "representative_cases": cases,
            "case_semantics": "illustrative only; full summaries are authoritative for frequency",
            "applied_guidance": self._serializable_guidance(self.active_guidance),
        }
        self.previous_episode_feedback = _json_clone(feedback)
        if self.root is not None:
            episode_root = self.root / f"episode_{episode_index + 1:06d}"
            _atomic_json(
                episode_root / "feedback.json",
                {
                    **feedback,
                    "movement_decisions": self.movement_records,
                    "routing_decisions": self.routing_records,
                },
            )
            _atomic_json(self.root / "state.json", self.state_dict())
        print(
            f"[LLM explorer TD3-only] episode={episode_index + 1} "
            f"w={self.active_guidance.get('annealing_weight', 0.0):.6f} "
            f"enabled={str(bool(self.active_guidance.get('enabled', False))).lower()} "
            f"reason={self.active_guidance.get('td3_reason') or 'active'} "
            f"movement_decisions={len(self.movement_records)} "
            f"routing_diagnostics={len(self.routing_records)} cases={len(cases)} "
            f"requests_total={self.statistics['requests_total']} "
            f"tokens_total={self.statistics['token_usage']['total_tokens']} "
            f"failed_attempts={self.statistics['failed_attempts_total']} "
            f"td3_fallback_episodes={self.statistics['td3_fallback_episodes_total']} "
            "routing_guidance=disabled",
            flush=True,
        )


def explorer_metadata_td3_only(
    config: LLMExplorerTD3OnlyConfig,
) -> dict[str, Any]:
    return {
        "schema_version": LLM_EXPLORER_TD3_ONLY_VERSION,
        "prompt_version": config.prompt_version,
        "summary_version": LLM_EXPLORER_TD3_ONLY_SUMMARY_VERSION,
        "case_selection_version": LLM_EXPLORER_TD3_ONLY_CASE_VERSION,
        "guidance_context_version": LLM_EXPLORER_TD3_ONLY_GUIDANCE_CONTEXT_VERSION,
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "episode_joint_transitions": config.episode_joint_transitions,
        "first_eligible_episode": first_eligible_episode(
            config.warmup_joint_transitions, config.episode_joint_transitions
        ),
        "annealing_end_episode": config.annealing_end_episode,
        "target_input_tokens": LLM_EXPLORER_TD3_ONLY_TARGET_INPUT_TOKENS,
        "input_hard_limit_tokens": config.input_hard_limit_tokens,
        "td3_intervention": "episode-fixed w * raw bias exactly once before projection",
        "routing_guidance": "disabled",
        "routing_intervention": "none_original_safe_ddqn_epsilon_greedy",
        "evaluation_calls_llm": False,
    }


build_observable_initial_state_td3_only = build_observable_initial_state_v3
