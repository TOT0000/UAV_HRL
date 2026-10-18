"""State-dependent LLM rule guidance for TD3 movement exploration only."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiment_config import (
    COM_PACKET_SIZE_BITS,
    REFERENCE_COM_BANDWIDTH_HZ,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)
from visual_sensing import VS_PACKET_MAX_BITS
from llm_explorer import (
    ExplorerValidationError,
    LLMExplorerConfig,
    LLMExplorerController,
    NUM_UAV,
    _atomic_json,
    _json_clone,
    environment_constants_and_definitions,
    prompt_token_budget,
    render_prompt,
)
from llm_explorer_v3 import _diverse_case_subset, _numeric_summary, first_eligible_episode
from llm_explorer_state_rule import (
    STATE_RULE_FEATURE_CATALOG_VERSION,
    STATE_RULE_LANGUAGE_VERSION,
    STATE_RULE_SCHEMA_VERSION,
    CompiledStateRule,
    build_runtime_feature_rows,
    compile_state_rule,
    evaluate_rule_for_joint_action,
    parse_and_compile_state_rule,
    runtime_feature_catalog,
    state_rule_output_schema,
)


LLM_STATE_RULE_METHOD_ID = "td3_dinkelbach_llm_explorer_state_rule"
LLM_STATE_RULE_VERSION = "uav-hrl-llm-state-rule-explorer-v2"
LLM_STATE_RULE_PROMPT_VERSION = "uav-hrl-llm-state-rule-prompts-v2"
LLM_STATE_RULE_SUMMARY_VERSION = "state-rule-observed-service-summary-v2"
LLM_STATE_RULE_CACHE_VERSION = "state-rule-complete-input-cache-v2"
LLM_STATE_RULE_ANALYSIS_OUTPUT_TOKENS = 4096
LLM_STATE_RULE_RULE_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class LLMStateRuleConfig(LLMExplorerConfig):
    prompt_version: str = LLM_STATE_RULE_PROMPT_VERSION
    analysis_max_output_tokens: int = LLM_STATE_RULE_ANALYSIS_OUTPUT_TOKENS
    guidance_max_output_tokens: int = LLM_STATE_RULE_RULE_OUTPUT_TOKENS
    episode_joint_transitions: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_version != LLM_STATE_RULE_PROMPT_VERSION:
            raise ValueError("state-rule prompt version is fixed by the method contract")
        if int(self.episode_joint_transitions) <= 0:
            raise ValueError("episode_joint_transitions must be positive")
        if int(self.safety_margin_tokens) != 4_096:
            raise ValueError("state-rule requests require a fixed 4,096-token safety margin")
        for name, value in (
            ("analysis_max_output_tokens", self.analysis_max_output_tokens),
            ("guidance_max_output_tokens", self.guidance_max_output_tokens),
        ):
            if not 1 <= int(value) <= 16_384:
                raise ValueError(f"{name} must fit the gpt-4o-mini 16,384-token output limit")


class DeterministicMockStateRuleClient:
    """Offline two-stage client; never performs a paid request."""

    def chat(self, *, model: str, prompt: str, **_: Any) -> dict[str, Any]:
        if "Use the following evidence to design" in prompt:
            content = (
                "1. Main problems and numerical evidence\n"
                "The offline mock has no evidence supporting directional intervention.\n"
                "2. Shared rule design\nReturn zero on all three components.\n"
                "3. Expected outcomes and possible costs\nPreserve baseline exploration.\n"
                "4. Uncertainty and metrics to inspect next episode\nMock inference only."
            )
        else:
            content = json.dumps(
                {
                    "schema_version": STATE_RULE_SCHEMA_VERSION,
                    "speed": 0.0,
                    "heading": 0.0,
                    "vertical": 0.0,
                },
                separators=(",", ":"),
            )
        return {
            "content": content,
            "actual_model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }


def environment_definitions_state_rule(**kwargs: Any) -> dict[str, Any]:
    base = environment_constants_and_definitions(**kwargs)
    visual = base["visual_sensing"]
    return {
        "schema_version": LLM_STATE_RULE_VERSION,
        "resolved_system_settings": {
            **base,
            "state_rule": {
                "weight_w": 1.0,
                "rule_update_interval_episodes": 1,
                "warmup_joint_transitions": 10_000,
                "evaluation_bias_enabled": False,
                "routing_guidance": "disabled",
            },
        },
        "resolved_traffic_settings": {
            "com_packet_bits": float(COM_PACKET_SIZE_BITS),
            "fov_max_packet_bits": float(VS_PACKET_MAX_BITS),
            "task_deadlines_seconds": base["task_deadlines_seconds"],
            "generation_model": base["packet_generation_model"],
            "reference_com_bandwidth_hz": float(REFERENCE_COM_BANDWIDTH_HZ),
            "total_shared_bandwidth_hz": float(TOTAL_COMMUNICATION_BANDWIDTH_HZ),
        },
        "resolved_reward_and_metric_definitions": {
            "movement_reward": base["movement_reward"],
            "routing_reward": base["routing_reward"],
            "safe_ddqn": base["safe_ddqn"],
            "energy_efficiency": "timely useful GS-delivered bits divided by all-UAV movement energy in joules",
            "visual_sensing": visual,
        },
        "runtime_feature_catalog": runtime_feature_catalog(),
        "rule_language": {
            "schema_version": STATE_RULE_SCHEMA_VERSION,
            "semantics_version": STATE_RULE_LANGUAGE_VERSION,
            "output_schema": state_rule_output_schema(),
        },
    }


def _summaries(values: list[list[float]]) -> list[dict[str, Any]]:
    if not values:
        return [_numeric_summary([]) for _ in range(3)]
    array = np.asarray(values, dtype=float).reshape(-1, 3)
    return [_numeric_summary(array[:, index].tolist(), 6) for index in range(3)]


def _count_ratio(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "ratio": (
            None
            if int(denominator) == 0
            else round(float(count) / float(denominator), 6)
        ),
    }


def _service_geometry_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    coverage: list[float] = []
    image_quantity: list[float] = []
    com_distances: list[float] = []
    com_capacities: list[float] = []
    sr_backlogs: list[float] = []
    sr_packet_counts: list[float] = []
    sr_hol_deadlines: list[float] = []
    counters["decision_count"] = len(items)
    for item in items:
        features = item.get("features")
        if not isinstance(features, dict):
            counters["feature_unavailable_decisions"] += 1
            continue
        counters["feature_available_decisions"] += 1
        actor_valid = bool(features.get("actor_action_valid", False))
        counters["actor_action_valid_decisions"] += int(actor_valid)
        counters["actor_action_unavailable_decisions"] += int(not actor_valid)

        has_fov = bool(features.get("has_fov_task", False))
        has_com = bool(features.get("has_com_task", False))
        counters["fov_assigned_decisions"] += int(has_fov)
        counters["com_assigned_decisions"] += int(has_com)
        counters["dual_fov_com_assigned_decisions"] += int(has_fov and has_com)
        if has_fov:
            fov_valid = bool(features.get("fov_target_valid", False))
            counters["fov_target_known_decisions"] += int(fov_valid)
            if fov_valid:
                counters["fov_c9_satisfied_decisions"] += int(
                    bool(features.get("fov_c9_satisfied", False))
                )
                c10_valid = bool(features.get("fov_c10_geometry_valid", False))
                counters["fov_c10_geometry_valid_decisions"] += int(c10_valid)
                if c10_valid:
                    counters["fov_c10_satisfied_decisions"] += int(
                        bool(features.get("fov_c10_satisfied", False))
                    )
                counters["fov_capture_valid_decisions"] += int(
                    bool(features.get("fov_capture_valid", False))
                )
                coverage.append(float(features.get("fov_coverage_ratio", 0.0)))
                image_quantity.append(float(features.get("fov_image_quantity", 0.0)))

        if has_com:
            com_valid = bool(features.get("com_target_valid", False))
            counters["com_target_known_decisions"] += int(com_valid)
            if com_valid:
                in_range = bool(features.get("com_in_range", False))
                counters["com_in_range_decisions"] += int(in_range)
                counters["com_out_of_range_decisions"] += int(not in_range)
                com_distances.append(float(features.get("com_distance_3d_m", 0.0)))
                capacity_valid = bool(
                    features.get("com_reference_capacity_valid", False)
                )
                counters["com_capacity_valid_decisions"] += int(capacity_valid)
                if capacity_valid:
                    com_capacities.append(
                        float(features.get("com_reference_capacity_mbps", 0.0))
                    )
                queue_valid = bool(features.get("com_sr_queue_valid", False))
                counters["com_sr_queue_valid_decisions"] += int(queue_valid)
                if queue_valid:
                    queue_empty = bool(features.get("com_sr_queue_empty", False))
                    counters["com_sr_queue_empty_decisions"] += int(queue_empty)
                    counters["com_sr_queue_nonempty_decisions"] += int(not queue_empty)
                    sr_backlogs.append(float(features.get("com_sr_backlog_bits", 0.0)))
                    sr_packet_counts.append(
                        float(features.get("com_sr_packet_count", 0.0))
                    )
                    hol_valid = bool(
                        features.get("com_sr_hol_deadline_valid", False)
                    )
                    counters["com_sr_hol_deadline_valid_decisions"] += int(
                        hol_valid
                    )
                    if hol_valid:
                        sr_hol_deadlines.append(
                            float(
                                features.get(
                                    "com_sr_hol_remaining_deadline_s", 0.0
                                )
                            )
                        )

    available = counters["feature_available_decisions"] > 0
    fov_known = counters["fov_target_known_decisions"]
    com_known = counters["com_target_known_decisions"]
    c10_valid = counters["fov_c10_geometry_valid_decisions"]
    queue_valid = counters["com_sr_queue_valid_decisions"]
    return {
        "available": bool(available),
        "unavailable_reason": (
            None
            if available
            else "no compatible decision-boundary runtime feature rows"
        ),
        "decision_count": int(counters["decision_count"]),
        "feature_availability": _count_ratio(
            counters["feature_available_decisions"], counters["decision_count"]
        ),
        "feature_unavailable_decisions": int(
            counters["feature_unavailable_decisions"]
        ),
        "actor_action_availability": _count_ratio(
            counters["actor_action_valid_decisions"],
            counters["feature_available_decisions"],
        ),
        "actor_action_unavailable_decisions": int(
            counters["actor_action_unavailable_decisions"]
        ),
        "dual_fov_com_assigned_decisions": int(
            counters["dual_fov_com_assigned_decisions"]
        ),
        "fov": {
            "assigned_decisions": int(counters["fov_assigned_decisions"]),
            "assigned_target_known": _count_ratio(
                fov_known, counters["fov_assigned_decisions"]
            ),
            "c9_satisfied": _count_ratio(
                counters["fov_c9_satisfied_decisions"], fov_known
            ),
            "c10_geometry_valid": _count_ratio(c10_valid, fov_known),
            "c10_satisfied_given_valid_geometry": _count_ratio(
                counters["fov_c10_satisfied_decisions"], c10_valid
            ),
            "capture_valid": _count_ratio(
                counters["fov_capture_valid_decisions"], fov_known
            ),
            "coverage_ratio_when_target_known": _numeric_summary(coverage, 6),
            "image_quantity_when_target_known": _numeric_summary(
                image_quantity, 6
            ),
        },
        "com": {
            "assigned_decisions": int(counters["com_assigned_decisions"]),
            "assigned_target_known": _count_ratio(
                com_known, counters["com_assigned_decisions"]
            ),
            "in_range": _count_ratio(
                counters["com_in_range_decisions"], com_known
            ),
            "out_of_range": _count_ratio(
                counters["com_out_of_range_decisions"], com_known
            ),
            "distance_3d_m_when_target_known": _numeric_summary(
                com_distances, 4
            ),
            "current_expected_capacity_valid": _count_ratio(
                counters["com_capacity_valid_decisions"], com_known
            ),
            "current_expected_capacity_mbps_when_valid": _numeric_summary(
                com_capacities, 6
            ),
            "sr_queue_valid": _count_ratio(queue_valid, com_known),
            "sr_queue_empty_given_valid": _count_ratio(
                counters["com_sr_queue_empty_decisions"], queue_valid
            ),
            "sr_queue_nonempty_given_valid": _count_ratio(
                counters["com_sr_queue_nonempty_decisions"], queue_valid
            ),
            "sr_backlog_bits_when_queue_valid": _numeric_summary(
                sr_backlogs, 2
            ),
            "sr_packet_count_when_queue_valid": _numeric_summary(
                sr_packet_counts, 3
            ),
            "sr_hol_deadline_valid_given_queue_valid": _count_ratio(
                counters["com_sr_hol_deadline_valid_decisions"], queue_valid
            ),
            "sr_hol_remaining_deadline_s_when_valid": _numeric_summary(
                sr_hol_deadlines, 6
            ),
        },
    }


def _diverse_state_rule_cases(
    cases: list[dict[str, Any]], maximum: int = 3
) -> list[dict[str, Any]]:
    """Select real cases by service role, UAV, time region, then behavior."""

    maximum = max(0, int(maximum))
    if not cases or maximum == 0:
        return []
    candidates = [_json_clone(case) for case in cases]
    def numeric_time(case: dict[str, Any]) -> float:
        try:
            value = float(case.get("time_s"))
        except (OverflowError, TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    finite_times = [
        value
        for value in (numeric_time(case) for case in candidates)
        if np.isfinite(value)
    ]
    lower = min(finite_times) if finite_times else None
    upper = max(finite_times) if finite_times else None

    def segment(case: dict[str, Any]) -> int | None:
        if lower is None or upper is None:
            return None
        value = numeric_time(case)
        if not np.isfinite(value):
            return None
        if upper == lower:
            return 0
        return min(2, int(3.0 * (value - lower) / (upper - lower)))

    def canonical(index: int) -> tuple[Any, ...]:
        case = candidates[index]
        return (
            str(case.get("service_role", "")),
            str(case.get("rule_behavior", "")),
            int(case.get("uav_id", -1)),
            numeric_time(case),
            json.dumps(case, ensure_ascii=False, sort_keys=True),
        )

    remaining = set(range(len(candidates)))
    selected: list[int] = []
    seen_roles: set[str] = set()
    seen_behaviors: set[str] = set()
    seen_uavs: set[int] = set()
    seen_segments: set[int] = set()
    while remaining and len(selected) < maximum:
        def rank(index: int) -> tuple[int, ...]:
            case = candidates[index]
            time_segment = segment(case)
            new_role = int(str(case.get("service_role")) not in seen_roles)
            new_behavior = int(
                str(case.get("rule_behavior")) not in seen_behaviors
            )
            new_uav = int(int(case.get("uav_id", -1)) not in seen_uavs)
            new_segment = int(
                time_segment is not None and time_segment not in seen_segments
            )
            return (
                new_role,
                new_uav + new_segment,
                new_behavior,
                new_uav,
                new_segment,
            )

        best_rank = max(rank(index) for index in remaining)
        chosen = min(
            (index for index in remaining if rank(index) == best_rank),
            key=canonical,
        )
        selected.append(chosen)
        remaining.remove(chosen)
        case = candidates[chosen]
        seen_roles.add(str(case.get("service_role")))
        seen_behaviors.add(str(case.get("rule_behavior")))
        seen_uavs.add(int(case.get("uav_id", -1)))
        time_segment = segment(case)
        if time_segment is not None:
            seen_segments.add(time_segment)
    return [candidates[index] for index in selected]


def state_rule_movement_summary(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate every decision; cases are illustrative and separately labeled."""

    by_uav: dict[str, Any] = {}
    all_case_candidates: list[dict[str, Any]] = []
    totals = Counter()
    for uav_id in range(NUM_UAV):
        actors: list[list[float]] = []
        executed: list[list[float]] = []
        raw: list[list[float]] = []
        bounded: list[list[float]] = []
        effective: list[list[float]] = []
        energies: list[float] = []
        role_counts: Counter[str] = Counter()
        max_role_run: Counter[str] = Counter()
        current_role_run: Counter[str] = Counter()
        clipped = zero = failures = controlled = 0
        conditions = conditions_true = short_circuits = 0
        service_items: list[dict[str, Any]] = []
        td3_service_items: list[dict[str, Any]] = []
        other_service_items: list[dict[str, Any]] = []
        for record in records:
            uavs = record.get("uavs", [])
            if uav_id >= len(uavs) or not isinstance(uavs[uav_id], dict):
                continue
            item = uavs[uav_id]
            tasks = [str(task) for task in item.get("task_types", [])]
            active_roles = set(tasks)
            for role in list(current_role_run):
                if role not in active_roles:
                    current_role_run[role] = 0
            for role in active_roles:
                role_counts[role] += 1
                current_role_run[role] += 1
                max_role_run[role] = max(max_role_run[role], current_role_run[role])
            actor = item.get("actor_normalized_action")
            actual = item.get("executed_normalized_action")
            rule = item.get("state_rule") if isinstance(item.get("state_rule"), dict) else {}
            service_item = {
                "features": rule.get("features"),
                "task_types": tasks,
            }
            service_items.append(service_item)
            if bool(item.get("movement_mask", False)):
                td3_service_items.append(service_item)
            else:
                other_service_items.append(service_item)
            applied = np.asarray(
                rule.get("effective_bias", [0.0, 0.0, 0.0]), dtype=float
            )
            runtime_failure = rule.get("runtime_failure")
            clipped_this_decision = int(rule.get("clipped_components", 0))
            if runtime_failure is not None:
                behavior = "runtime_fallback"
            elif clipped_this_decision:
                behavior = "clipped"
            elif np.any(np.abs(applied) > 1e-12):
                behavior = "nonzero_bias"
            else:
                behavior = "zero_bias"
            role = (
                "dual"
                if {"FOV", "COM"}.issubset(active_roles)
                else "fov"
                if "FOV" in active_roles
                else "com"
                if "COM" in active_roles
                else "other"
            )
            all_case_candidates.append(
                {
                    "domain": "movement",
                    "kind": f"movement_{role}_{behavior}",
                    "service_role": role,
                    "rule_behavior": behavior,
                    "episode": int(record.get("episode_index", -1)) + 1,
                    "time_s": record.get("time_seconds"),
                    "uav_id": uav_id,
                    "tasks": tasks,
                    "movement_mask": bool(item.get("movement_mask", False)),
                    "evaluation_status": rule.get("evaluation_status"),
                    "features": rule.get("features"),
                    "actor_action": actor,
                    "raw_bias": rule.get("raw_bias"),
                    "bounded_bias": rule.get("bounded_bias"),
                    "effective_bias": rule.get("effective_bias"),
                    "executed_action": actual,
                    "runtime_failure": runtime_failure,
                }
            )
            if not bool(item.get("movement_mask", False)):
                continue
            controlled += 1
            if actor is not None:
                actors.append(list(map(float, actor)))
            if actual is not None:
                executed.append(list(map(float, actual)))
            for target, destination in (
                (rule.get("raw_bias"), raw),
                (rule.get("bounded_bias"), bounded),
                (rule.get("effective_bias"), effective),
            ):
                if isinstance(target, list) and len(target) == 3:
                    destination.append(list(map(float, target)))
            clipped += clipped_this_decision
            failures += int(runtime_failure is not None)
            trace = rule.get("trace") if isinstance(rule.get("trace"), dict) else {}
            conditions += int(trace.get("conditions_evaluated", 0))
            conditions_true += int(trace.get("conditions_true", 0))
            short_circuits += int(trace.get("short_circuits", 0))
            zero += int(not np.any(np.abs(applied) > 1e-12))
            if item.get("movement_energy_j") is not None:
                energies.append(float(item["movement_energy_j"]))
        totals.update(
            controlled_decisions=controlled,
            clipped_components=clipped,
            zero_bias_decisions=zero,
            runtime_failures=failures,
            conditions_evaluated=conditions,
            conditions_true=conditions_true,
            short_circuits=short_circuits,
        )
        by_uav[f"UAV_{uav_id}"] = {
            "role_decision_counts": dict(sorted(role_counts.items())),
            "longest_consecutive_role_duration_s": {
                key: int(value) for key, value in sorted(max_role_run.items())
            },
            "td3_controlled_decisions": controlled,
            "actor_action_by_component": _summaries(actors),
            "raw_bias_by_component": _summaries(raw),
            "bounded_bias_by_component": _summaries(bounded),
            "effective_bias_by_component": _summaries(effective),
            "executed_action_by_component": _summaries(executed),
            "clipped_component_count": clipped,
            "zero_effective_bias_decision_count": zero,
            "runtime_failure_count": failures,
            "condition_evaluation_count": conditions,
            "condition_true_count": conditions_true,
            "short_circuit_count": short_circuits,
            "movement_energy_j": _numeric_summary(energies, 4),
            "service_geometry_by_control_scope": {
                "all_decisions": _service_geometry_summary(service_items),
                "td3_controlled_decisions": _service_geometry_summary(
                    td3_service_items
                ),
                "other_controller_decisions": _service_geometry_summary(
                    other_service_items
                ),
            },
        }
    return {
        "scope": "complete episode; one row per UAV; service geometry uses every recorded legal decision boundary and durations use one-second movement decisions",
        "totals": dict(totals),
        "by_uav": by_uav,
    }, _diverse_state_rule_cases(all_case_candidates, 3)


def _packet_id_from_event(item: Any) -> int | None:
    if not isinstance(item, dict):
        return None
    packet = item.get("packet") if isinstance(item.get("packet"), dict) else item
    top = item.get("packet_id")
    nested = packet.get("id") if isinstance(packet, dict) else None
    if top is not None and nested is not None and int(top) != int(nested):
        raise ExplorerValidationError(
            f"packet event has conflicting packet_id={top} and packet.id={nested}"
        )
    value = top if top is not None else nested
    return None if value is None else int(value)


def state_rule_slot_service_record(
    slot_result: Mapping[str, Any],
    *,
    pre_slot_violations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    current_time: float,
) -> dict[str, Any]:
    """Normalize authoritative PacketEngine event rows without changing service."""

    packet_use_counts = Counter(
        packet_id
        for packet_id in (
            _packet_id_from_event(item)
            for item in slot_result.get("packet_decision_uses", ())
        )
        if packet_id is not None
    )
    serviced_packet_ids = set(packet_use_counts)
    for key in ("s2u_arrivals", "relay_arrivals", "outcomes"):
        for item in slot_result.get(key, ()):
            packet_id = _packet_id_from_event(item)
            if packet_id is not None:
                serviced_packet_ids.add(packet_id)

    terminal_by_packet: dict[int, tuple[str, str]] = {}
    for item in (*tuple(pre_slot_violations), *tuple(slot_result.get("outcomes", ()))):
        if not isinstance(item, dict):
            continue
        packet = item.get("packet")
        if not isinstance(packet, dict):
            continue
        packet_id = _packet_id_from_event(item)
        outcome = packet.get("terminal_outcome")
        if packet_id is None or outcome is None:
            continue
        outcome = str(outcome)
        if outcome not in {
            "on_time_delivered",
            "late_delivered",
            "expired_dropped",
            "sr_admission_drop",
        }:
            raise ExplorerValidationError(
                f"unsupported canonical terminal_outcome {outcome!r}"
            )
        task_type = str(item.get("task_type", packet.get("task_type", "unknown"))).upper()
        previous = terminal_by_packet.setdefault(packet_id, (outcome, task_type))
        if previous != (outcome, task_type):
            raise ExplorerValidationError(
                f"packet {packet_id} has conflicting terminal outcomes {previous} and {(outcome, task_type)}"
            )

    terminal_counts = Counter(outcome for outcome, _ in terminal_by_packet.values())
    by_task: dict[str, dict[str, int]] = {}
    for task in ("FOV", "COM"):
        counts = Counter(
            outcome
            for outcome, task_type in terminal_by_packet.values()
            if task_type == task
        )
        by_task[task] = {
            "on_time_delivered_packets": int(counts["on_time_delivered"]),
            "late_delivered_packets": int(counts["late_delivered"]),
            "expired_dropped_packets": int(counts["expired_dropped"]),
            "sr_admission_drop_packets": int(counts["sr_admission_drop"]),
        }
    active_links = [
        {
            "sender_id": int(item["sender_id"]),
            "receiver_id": int(item["receiver_id"]),
            "link_type": str(item["link_type"]),
            "capacity_bits_per_second": float(item["capacity_bits_per_second"]),
            "transmitted_bits": float(item["transmitted_bits"]),
            "active_seconds": float(item["active_seconds"]),
        }
        for item in slot_result.get("active_links", ())
        if isinstance(item, dict)
    ]
    return {
        "time_seconds": float(current_time),
        "start_of_slot_nonempty_sender_ids": list(
            map(int, slot_result.get("start_of_slot_routing_sender_ids", ()))
        ),
        "serviced_packet_ids": sorted(serviced_packet_ids),
        "terminal_packet_ids": sorted(terminal_by_packet),
        "s2u_completed_packets": len(slot_result.get("s2u_arrivals", ())),
        "uav_relay_completions": len(slot_result.get("relay_arrivals", ())),
        "on_time_gs_delivery_packets": int(
            terminal_counts["on_time_delivered"]
        ),
        "late_gs_delivery_packets": int(terminal_counts["late_delivered"]),
        "gs_delivery_outcomes": int(
            terminal_counts["on_time_delivered"]
            + terminal_counts["late_delivered"]
        ),
        "expired_dropped_packets": int(terminal_counts["expired_dropped"]),
        "sr_admission_drop_packets": int(terminal_counts["sr_admission_drop"]),
        "terminal_outcomes_by_task": by_task,
        "total_timely_useful_bits": float(
            slot_result.get("total_timely_useful_bits", 0.0)
        ),
        "fov_timely_delivered_raw_bits": float(
            slot_result.get("fov_timely_delivered_raw_bits", 0.0)
        ),
        "fov_timely_useful_bits": float(
            slot_result.get("fov_timely_useful_bits", 0.0)
        ),
        "com_timely_delivered_bits": float(
            slot_result.get("com_timely_delivered_bits", 0.0)
        ),
        "same_slot_multi_hop_packets": sum(
            count >= 2 for count in packet_use_counts.values()
        ),
        "transmitted_bits_by_link": {
            str(key): float(value)
            for key, value in slot_result.get("transmitted_bits_by_link", {}).items()
        },
        "active_links": active_links,
    }


def slot_service_summary(
    records: list[dict[str, Any]], metrics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    totals = Counter()
    transmitted_bits: list[float] = []
    active_capacity_by_type: dict[str, list[float]] = {}
    active_seconds_by_type: dict[str, list[float]] = {}
    transmitted_by_type: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = {
        "FOV": Counter(),
        "COM": Counter(),
    }
    slots_with_empty_start_and_service = 0
    for record in records:
        start_senders = set(map(int, record.get("start_of_slot_nonempty_sender_ids", [])))
        service_packet_ids = set(map(int, record.get("serviced_packet_ids", [])))
        totals["slot_count"] += 1
        totals["s2u_completed_packets"] += int(record.get("s2u_completed_packets", 0))
        totals["uav_relay_completions"] += int(record.get("uav_relay_completions", 0))
        totals["on_time_gs_delivery_packets"] += int(
            record.get("on_time_gs_delivery_packets", 0)
        )
        totals["late_gs_delivery_packets"] += int(
            record.get("late_gs_delivery_packets", 0)
        )
        totals["gs_delivery_outcomes"] += int(record.get("gs_delivery_outcomes", 0))
        totals["expired_dropped_packets_observed_in_routing_slots"] += int(
            record.get("expired_dropped_packets", 0)
        )
        totals["sr_admission_drop_packets_observed_in_routing_slots"] += int(
            record.get("sr_admission_drop_packets", 0)
        )
        totals["same_slot_multi_hop_packets"] += int(record.get("same_slot_multi_hop_packets", 0))
        for key in (
            "total_timely_useful_bits",
            "fov_timely_delivered_raw_bits",
            "fov_timely_useful_bits",
            "com_timely_delivered_bits",
        ):
            totals[key] += float(record.get(key, 0.0))
        for task, values in record.get("terminal_outcomes_by_task", {}).items():
            if task not in by_task or not isinstance(values, dict):
                continue
            for key, value in values.items():
                by_task[task][str(key)] += int(value)
        if service_packet_ids and not start_senders:
            # This is deliberately only a slot-level fact. Packet IDs and
            # sender IDs are separate namespaces, so no per-UAV attribution is
            # inferred from an empty start followed by actual service.
            slots_with_empty_start_and_service += 1
        transmitted_bits.extend(float(value) for value in record.get("transmitted_bits_by_link", {}).values())
        for link in record.get("active_links", ()):
            if not isinstance(link, dict):
                continue
            link_type = str(link.get("link_type", "unknown"))
            active_capacity_by_type.setdefault(link_type, []).append(
                float(link.get("capacity_bits_per_second", 0.0))
            )
            active_seconds_by_type.setdefault(link_type, []).append(
                float(link.get("active_seconds", 0.0))
            )
            transmitted_by_type[link_type] += float(
                link.get("transmitted_bits", 0.0)
            )
    metrics = metrics if isinstance(metrics, Mapping) else {}

    def metric_number(name: str) -> float | None:
        value = metrics.get(name)
        return None if value is None else float(value)

    authoritative = {
        "on_time_gs_delivery_packets": sum(
            int(metrics.get(f"{task}_on_time_delivered_packets", 0) or 0)
            for task in ("fov", "com")
        ) if metrics else None,
        "late_gs_delivery_packets": sum(
            int(metrics.get(f"{task}_late_delivered_packets", 0) or 0)
            for task in ("fov", "com")
        ) if metrics else None,
        "total_timely_useful_bits": metric_number("timely_useful_delivery_bits"),
        "fov_timely_delivered_raw_bits": metric_number(
            "fov_timely_delivered_raw_bits"
        ),
        "fov_timely_useful_bits": metric_number("fov_timely_useful_bits"),
        "com_timely_delivered_bits": metric_number("com_timely_delivered_bits"),
    }
    observed_for_alignment = {
        key: float(totals[key])
        for key in (
            "on_time_gs_delivery_packets",
            "late_gs_delivery_packets",
            "total_timely_useful_bits",
            "fov_timely_delivered_raw_bits",
            "fov_timely_useful_bits",
            "com_timely_delivered_bits",
        )
    }
    alignment = {
        key: (
            None
            if authoritative[key] is None
            else bool(np.isclose(observed_for_alignment[key], authoritative[key]))
        )
        for key in authoritative
    }
    result = {
        "scope": "actual event-time service over complete recorded routing slots; system/link evidence, not per-UAV causal attribution",
        **dict(totals),
        "gs_delivery_outcomes_definition": "unique packets delivered to GS, on time plus late; timely useful bits include on-time complete deliveries only",
        "terminal_outcomes_by_task": {
            task: dict(sorted(values.items()))
            for task, values in by_task.items()
        },
        "slots_with_no_starting_uav_hol_but_actual_service": slots_with_empty_start_and_service,
        "transmitted_bits_per_active_link": _numeric_summary(transmitted_bits, 2),
        "actual_shared_fdma_service_by_link_type": {
            link_type: {
                "active_link_observation_count": len(capacities),
                "mean_service_capacity_bits_per_second": _numeric_summary(
                    capacities, 2
                ),
                "active_seconds": _numeric_summary(
                    active_seconds_by_type.get(link_type, []), 6
                ),
                "transmitted_bits_total": float(transmitted_by_type[link_type]),
            }
            for link_type, capacities in sorted(active_capacity_by_type.items())
        },
        "authoritative_episode_totals": authoritative,
        "recorded_slot_sum_matches_authoritative_episode_total": alignment,
        "note": "slot-start empty queues do not imply no service; generation, S2U access, and same-slot forwarding can occur later in the slot",
    }
    return result


def routing_expected_capacity_observation(
    env: Any, sender_uav_id: int, legal_actions: Any
) -> dict[str, Any]:
    """Read the already refreshed decision-boundary capacity caches."""

    sender = int(sender_uav_id)
    gs_id = int(env.GS_ID)
    actions = [int(action) for action in legal_actions]
    u2u = np.asarray(env.u2u_nominal_capacity, dtype=float)
    u2g = np.asarray(env.u2g_nominal_capacity, dtype=float)
    positions = np.asarray(
        [env.uav_dict[uav_id].get_position() for uav_id in range(int(env.num_UAV))],
        dtype=float,
    )
    gs_position = np.asarray(env.GS_pos, dtype=float)
    los_state = np.asarray(env.channel.u2g_los_state, dtype=bool)
    current: dict[str, float] = {}
    normalizers: dict[str, float] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for action in actions:
        key = str(action)
        if action == sender:
            current[key] = 0.0
            normalizers[key] = 0.0
            contexts[key] = {
                "link_type": "WAIT",
                "sender_uav_id": sender,
                "receiver_id": sender,
                "distance_3d_m": 0.0,
                "reference_bandwidth_hz": 0.0,
                "fading_or_los_treatment": "not applicable",
            }
        elif action == gs_id:
            current[key] = float(u2g[sender]) * 1e6
            normalizers[key] = float(
                env.routing_capacity_reference_mbps(sender, action)
            ) * 1e6
            contexts[key] = {
                "link_type": "U2G",
                "sender_uav_id": sender,
                "receiver_id": gs_id,
                "distance_3d_m": float(
                    np.linalg.norm(positions[sender] - gs_position)
                ),
                "reference_bandwidth_hz": float(env.B_tot),
                "fading_or_los_treatment": "installed U2G LoS/NLoS state with expected fading",
                "installed_los_state": bool(los_state[sender]),
            }
        else:
            current[key] = float(u2u[sender, action]) * 1e6
            normalizers[key] = float(
                env.routing_capacity_reference_mbps(sender, action)
            ) * 1e6
            contexts[key] = {
                "link_type": "U2U",
                "sender_uav_id": sender,
                "receiver_id": action,
                "distance_3d_m": float(
                    np.linalg.norm(positions[sender] - positions[action])
                ),
                "reference_bandwidth_hz": float(env.B_tot),
                "fading_or_los_treatment": "expected Rician fading; no new channel sample",
            }
    return {
        "current_expected_capacity_bps_by_legal_action": current,
        "maximum_reference_normalizer_bps_by_legal_action": normalizers,
        "capacity_context_by_legal_action": contexts,
    }


def communication_bottleneck_summary_state_rule(
    records: list[dict[str, Any]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    expected_by_type: dict[str, list[float]] = {"U2U": [], "U2G": []}
    selected_by_type: dict[str, list[float]] = {"U2U": [], "U2G": []}
    normalizer_by_type: dict[str, list[float]] = {"U2U": [], "U2G": []}
    distance_by_type: dict[str, list[float]] = {"U2U": [], "U2G": []}
    bandwidth_by_type: dict[str, list[float]] = {"U2U": [], "U2G": []}
    fading_treatments: dict[str, set[str]] = {"U2U": set(), "U2G": set()}
    u2g_los_states: Counter[str] = Counter()
    records_with_current = 0
    for record in records:
        current = record.get("current_expected_capacity_bps_by_legal_action")
        normalizers = record.get(
            "maximum_reference_normalizer_bps_by_legal_action"
        )
        contexts = record.get("capacity_context_by_legal_action")
        if not all(isinstance(value, dict) for value in (current, normalizers, contexts)):
            continue
        records_with_current += 1
        for action, value in current.items():
            context = contexts.get(str(action), {})
            link_type = str(context.get("link_type", "unknown"))
            if link_type not in expected_by_type:
                continue
            expected_by_type[link_type].append(float(value))
            normalizer_by_type[link_type].append(
                float(normalizers.get(str(action), 0.0))
            )
            distance_by_type[link_type].append(
                float(context.get("distance_3d_m", 0.0))
            )
            bandwidth_by_type[link_type].append(
                float(context.get("reference_bandwidth_hz", 0.0))
            )
            fading_treatments[link_type].add(
                str(context.get("fading_or_los_treatment", "unavailable"))
            )
            if link_type == "U2G" and context.get("installed_los_state") is not None:
                u2g_los_states[
                    "LoS" if bool(context["installed_los_state"]) else "NLoS"
                ] += 1
        selected = str(record.get("selected_action"))
        selected_context = contexts.get(selected, {})
        selected_type = str(selected_context.get("link_type", "unknown"))
        if selected_type in selected_by_type and selected in current:
            selected_by_type[selected_type].append(float(current[selected]))
    available = records_with_current > 0
    return {
        "available": bool(available),
        "unavailable_reason": (
            None
            if available
            else "records predate decision-boundary expected-capacity observation; later geometry is not used to reconstruct them"
        ),
        "scope": "routing decisions from the complete episode",
        "decision_count": len(records),
        "records_with_current_expected_capacity": records_with_current,
        "capacity_definition": {
            "current_expected_capacity": "decision-boundary expected capacity at the link reference bandwidth, read from already refreshed channel caches",
            "actual_shared_fdma_service": "reported separately from PacketEngine active-link service observations",
            "maximum_reference_normalizer": "fixed maximum used only for normalization; excluded from current-capacity distributions",
            "units": "bits per second",
        },
        "current_expected_capacity_bps_by_link_type": {
            link_type: _numeric_summary(values, 2)
            for link_type, values in expected_by_type.items()
        },
        "selected_current_expected_capacity_bps_by_link_type": {
            link_type: _numeric_summary(values, 2)
            for link_type, values in selected_by_type.items()
        },
        "maximum_reference_normalizer_bps_by_link_type": {
            link_type: _numeric_summary(values, 2)
            for link_type, values in normalizer_by_type.items()
        },
        "link_observation_context_by_type": {
            link_type: {
                "endpoint_identity": "sender_uav_id and receiver_id retained in each saved routing decision",
                "distance_3d_m": _numeric_summary(
                    distance_by_type[link_type], 4
                ),
                "reference_bandwidth_hz": _numeric_summary(
                    bandwidth_by_type[link_type], 2
                ),
                "fading_or_los_treatment": sorted(
                    fading_treatments[link_type]
                ),
                **(
                    {"installed_los_state_observation_counts": dict(u2g_los_states)}
                    if link_type == "U2G"
                    else {}
                ),
            }
            for link_type in ("U2U", "U2G")
        },
        "episode_context": {
            key: metrics.get(key)
            for key in (
                "routing_stage_eligible_packets",
                "routing_stage_violation_count",
                "routing_immediate_cost_sum",
            )
        },
    }


class LLMStateRuleController(LLMExplorerController):
    schema_version = LLM_STATE_RULE_VERSION
    artifact_directory_name = "llm_rule"

    def __init__(
        self,
        *,
        config: LLMStateRuleConfig | None = None,
        mock: bool = False,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        resolved = config or LLMStateRuleConfig()
        if mock and client is None:
            client = DeterministicMockStateRuleClient()
            mock = False
        super().__init__(config=resolved, mock=mock, client=client, **kwargs)
        self.config: LLMStateRuleConfig
        self._compiled_rule: CompiledStateRule | None = None
        self._last_runtime_rows: list[dict[str, Any]] = []
        self.slot_service_records: list[dict[str, Any]] = []
        self.runtime_failure_total = 0

    @staticmethod
    def _unguided(reason: str) -> dict[str, Any]:
        failed = str(reason) in {"analysis_stage_failed", "guidance_invalid_or_failed"}
        return {
            "enabled": False,
            "rule": None,
            "td3_fallback": failed,
            "td3_reason": str(reason),
            "routing_prior": None,
            "routing_fallback": False,
            "routing_reason": "routing_guidance_disabled",
            "analysis_attempt": None,
            "td3_guidance_attempt": None,
            "routing_guidance_attempt": None,
            "reused": False,
            "guidance_weight_w": 0.0,
            "sigma_episode_start": None,
        }

    @staticmethod
    def _serializable_guidance(guidance: dict[str, Any]) -> dict[str, Any]:
        result = dict(guidance)
        candidate = result.get("td3_bias")
        if isinstance(candidate, dict):
            result["rule"] = candidate
            result.pop("td3_bias", None)
        for key, value in tuple(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return _json_clone(result)

    @staticmethod
    def _restore_guidance(value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        rule = result.get("rule")
        if rule is not None:
            result["rule"] = compile_state_rule(rule).rule
        result.pop("td3_bias", None)
        result["routing_prior"] = None
        result["routing_fallback"] = False
        result["routing_reason"] = "routing_guidance_disabled"
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        rule = self.active_guidance.get("rule")
        self._compiled_rule = None if rule is None else compile_state_rule(rule)
        self.runtime_failure_total = int(state.get("runtime_failure_total", 0))

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result.update(
            runtime_failure_total=int(self.runtime_failure_total),
            state_rule_source_episode=self.active_guidance.get("source_episode"),
            state_rule_intervention_active=bool(
                self.active_guidance.get("enabled")
                and not self.active_guidance.get("td3_fallback")
            ),
        )
        return result

    def _templates(self) -> tuple[str, str, str]:
        root = Path(__file__).resolve().parent / "prompts"
        return tuple(
            (root / name).read_text(encoding="utf-8")
            for name in (
                "llm_state_rule_system.txt",
                "llm_state_rule_analysis.txt",
                "llm_state_rule_generation.txt",
            )
        )

    def _render_system_prompt(self, template: str, environment_constants: dict[str, Any]) -> str:
        return render_prompt(
            template,
            {
                "RESOLVED_SYSTEM_SETTINGS": environment_constants["resolved_system_settings"],
                "RESOLVED_TRAFFIC_SETTINGS": environment_constants["resolved_traffic_settings"],
                "RESOLVED_REWARD_AND_METRIC_DEFINITIONS": environment_constants["resolved_reward_and_metric_definitions"],
                "RUNTIME_FEATURE_CATALOG": environment_constants["runtime_feature_catalog"],
            },
        )

    def _cache_identity(self, **kwargs: Any) -> dict[str, Any]:
        identity = super()._cache_identity(**kwargs)
        identity.update(
            schema_version=LLM_STATE_RULE_VERSION,
            cache_version=LLM_STATE_RULE_CACHE_VERSION,
            rule_schema_version=STATE_RULE_SCHEMA_VERSION,
            rule_language_version=STATE_RULE_LANGUAGE_VERSION,
            feature_catalog_version=STATE_RULE_FEATURE_CATALOG_VERSION,
            routing_guidance="disabled",
        )
        return identity

    def _routing_guidance_required(self) -> bool:
        return False

    def _stage_display_name(self, stage: str) -> str:
        return "rule" if stage == "guidance" else str(stage)

    def _before_request(self, *, episode_index: int, stage: str, attempt: int) -> None:
        print(
            f"[LLM state rule] episode={episode_index + 1} "
            f"stage={self._stage_display_name(stage)} "
            f"attempt={attempt} request_start retry={str(attempt > 1).lower()} "
            f"failed_attempts={self.statistics['failed_attempts_total']}",
            flush=True,
        )

    def _parse_guidance_response(
        self, content: str, sigma_episode_start: float
    ) -> tuple[dict[str, Any] | None, None, dict[str, str]]:
        del sigma_episode_start
        try:
            return parse_and_compile_state_rule(content).rule, None, {}
        except (
            ExplorerValidationError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            return None, None, {"td3": str(exc)}

    def _guidance_cache_content(self, td3_bias: Any, routing_prior: Any) -> str:
        del routing_prior
        return json.dumps(
            compile_state_rule(td3_bias).rule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _load_episode_result(
        self, episode_index: int, identity: dict[str, Any], sigma: float
    ) -> dict[str, Any] | None:
        del sigma
        path = self._episode_result_path(episode_index)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version or value.get("identity") != identity:
                return None
            guidance = self._restore_guidance(value["guidance"])
            if guidance.get("enabled") and not guidance.get("td3_fallback"):
                rule = compile_state_rule(guidance["rule"]).rule
                guidance["rule"] = rule
                guidance["td3_bias"] = rule
            guidance["reused"] = True
            guidance["cache_source"] = "episode_result"
            return guidance
        except (
            OSError,
            KeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ExplorerValidationError,
        ):
            return None

    def _save_episode_result(
        self, *, episode_index: int, identity: dict[str, Any], guidance: dict[str, Any]
    ) -> None:
        path = self._episode_result_path(episode_index)
        if path is None:
            return
        try:
            _atomic_json(
                path,
                {
                    "schema_version": self.schema_version,
                    "identity": identity,
                    "guidance": self._serializable_guidance(guidance),
                },
            )
        except OSError as exc:
            self._record_local_failure(
                episode_index=episode_index, stage="episode_result", attempt=0,
                category="artifact_write", reason=f"episode result cache: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _previous_rule(previous: Mapping[str, Any]) -> dict[str, Any]:
        guidance = previous.get("applied_guidance")
        guidance = guidance if isinstance(guidance, dict) else {}
        rule = guidance.get("rule")
        if rule is None or guidance.get("td3_fallback") or not guidance.get("enabled"):
            return {
                "available": False,
                "reason": guidance.get("td3_reason") or "no reliable prior complete rule",
                "rule": None,
            }
        return {"available": True, "source_episode": guidance.get("source_episode"), "rule": _json_clone(rule)}

    @staticmethod
    def _application_status(previous: Mapping[str, Any]) -> dict[str, Any]:
        guidance = previous.get("applied_guidance")
        guidance = guidance if isinstance(guidance, dict) else {}
        movement = previous.get("movement_and_rule_application_summaries")
        totals = movement.get("totals", {}) if isinstance(movement, dict) else {}
        return {
            "enabled": bool(guidance.get("enabled", False)),
            "fallback": bool(guidance.get("td3_fallback", False)),
            "disabled_reason": guidance.get("td3_reason"),
            "reused_from_cache": bool(guidance.get("reused", False)),
            "source_episode": guidance.get("source_episode"),
            "actual_application_totals": _json_clone(totals),
            "routing_guidance": "disabled_not_requested",
        }

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        return {
            "metrics": _json_clone(previous.get("metrics", {})),
            "previous_complete_rule": self._previous_rule(previous),
            "previous_rule_application_status": self._application_status(previous),
            "observed_task_and_traffic_summary": _json_clone(previous.get("observed_task_and_traffic_summary", {})),
            "movement_and_rule_application_summaries": _json_clone(previous.get("movement_and_rule_application_summaries", {})),
            "fov_com_and_slot_service_summaries": _json_clone(previous.get("fov_com_and_slot_service_summaries", {})),
            "representative_cases": _json_clone(previous.get("representative_cases", [])),
        }

    def _exploration_settings(self, *, sigma_episode_start: float, routing_epsilon: float) -> dict[str, Any]:
        del routing_epsilon
        return {
            "upcoming_episode": getattr(self, "_upcoming_episode", None),
            "total_joint_transitions": getattr(self, "_total_joint_transitions", None),
            "warmup_joint_transitions": self.config.warmup_joint_transitions,
            "first_eligible_episode_if_every_episode_has_configured_length": first_eligible_episode(
                self.config.warmup_joint_transitions, self.config.episode_joint_transitions
            ),
            "activation_uses_actual_joint_transition_count": True,
            "sigma_episode_start": float(sigma_episode_start),
            "raw_bias_host_clip": [-float(sigma_episode_start), float(sigma_episode_start)],
            "guidance_weight_w": 1.0,
            "weight_schedule": "fixed at 1 through the configured training horizon; no episode-600 cutoff",
            "rule_structure_fixed_for_episode": True,
            "rule_output_varies_with_latest_decision_state": True,
            "routing_guidance": "disabled; original safe-DDQN flow",
        }

    def _analysis_replacements(self, *, previous: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "PREVIOUS_COMPLETE_RULE": previous["previous_complete_rule"],
            "PREVIOUS_RULE_APPLICATION_STATUS": previous["previous_rule_application_status"],
            "OBSERVED_TASK_AND_TRAFFIC_SUMMARY": previous["observed_task_and_traffic_summary"],
            "VERIFIED_EPISODE_METRICS": previous["metrics"],
            "MOVEMENT_AND_RULE_APPLICATION_SUMMARIES": previous["movement_and_rule_application_summaries"],
            "FOV_COM_AND_SLOT_SERVICE_SUMMARIES": previous["fov_com_and_slot_service_summaries"],
            "REPRESENTATIVE_CASES": previous["representative_cases"],
        }

    def _render_analysis_prompt(self, *, system_prompt: str, analysis_template: str, previous: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        cases = previous.get("representative_cases", [])
        last_prompt = ""
        for maximum in (3, 2, 1, 0):
            candidate = dict(previous)
            candidate["representative_cases"] = _diverse_case_subset(
                [item for item in cases if isinstance(item, dict)], maximum
            )
            last_prompt = render_prompt(
                analysis_template,
                self._analysis_replacements(
                    previous=candidate,
                    current_observable_initial_state=current_observable_initial_state,
                    settings=settings,
                ),
            )
            budget = prompt_token_budget(
                system_prompt, last_prompt,
                reserved_output_tokens=self.config.analysis_max_output_tokens,
                context_window_tokens=self.config.context_window_tokens,
                safety_margin_tokens=self.config.safety_margin_tokens,
                encoding_name=self.config.tokenizer_encoding,
            )
            if budget["fits"]:
                return last_prompt
        return last_prompt

    def _guidance_replacements(self, *, analysis: str, previous: dict[str, Any], environment_constants: dict[str, Any], current_observable_initial_state: dict[str, Any], settings: dict[str, Any], validation_feedback: str) -> dict[str, Any]:
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "VERIFIED_KEY_EVIDENCE_SUMMARY": {
                "episode_metrics": previous["metrics"],
                "movement_and_rule_application": previous["movement_and_rule_application_summaries"],
                "service": previous["fov_com_and_slot_service_summaries"],
            },
            "RUNTIME_FEATURE_CATALOG": environment_constants["runtime_feature_catalog"],
            "RULE_LANGUAGE_AND_OUTPUT_SCHEMA": environment_constants["rule_language"],
            "PREVIOUS_COMPLETE_RULE": previous["previous_complete_rule"],
            "PREVIOUS_RULE_APPLICATION_STATUS": previous["previous_rule_application_status"],
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "VALIDATION_FEEDBACK_OR_EMPTY": validation_feedback or "No local validation feedback is present.",
        }

    def begin_episode(self, *, episode_index: int, total_joint_transitions: int, sigma_episode_start: float, **kwargs: Any) -> dict[str, Any]:
        self.movement_records = []
        self.routing_records = []
        self.slot_service_records = []
        self._last_runtime_rows = []
        self._compiled_rule = None
        self._upcoming_episode = int(episode_index) + 1
        self._total_joint_transitions = int(total_joint_transitions)
        if self.evaluation:
            self.active_guidance = self._unguided("evaluation_disabled")
            return self.active_guidance
        if int(total_joint_transitions) < int(self.config.warmup_joint_transitions):
            self.active_guidance = self._unguided("warmup")
            return self.active_guidance
        result = LLMExplorerController.begin_episode(
            self,
            episode_index=episode_index,
            total_joint_transitions=total_joint_transitions,
            sigma_episode_start=sigma_episode_start,
            **kwargs,
        )
        candidate = result.get("td3_bias")
        if result.get("enabled") and not result.get("td3_fallback") and isinstance(candidate, dict):
            self._compiled_rule = compile_state_rule(candidate)
            result["rule"] = self._compiled_rule.rule
            result["source_episode"] = int(episode_index) + 1
        else:
            result["rule"] = None
        result.pop("td3_bias", None)
        result.update(
            guidance_weight_w=(1.0 if self._compiled_rule is not None else 0.0),
            sigma_episode_start=float(sigma_episode_start),
            routing_prior=None,
            routing_fallback=False,
            routing_reason="routing_guidance_disabled",
        )
        self.active_guidance = result
        print(
            f"[LLM state rule] episode={episode_index + 1} stage=complete "
            f"enabled={str(bool(result.get('enabled'))).lower()} "
            f"fallback={str(bool(result.get('td3_fallback'))).lower()} w={result['guidance_weight_w']:.1f} "
            f"failed_attempts={self.statistics['failed_attempts_total']}",
            flush=True,
        )
        return result

    def evaluate_movement_bias(
        self,
        env: Any,
        snapshot: Mapping[str, Any],
        movement_mask: Any,
        actor_action: Any,
        *,
        sigma_episode_start: float,
        episode_seconds: float,
    ) -> np.ndarray:
        if self._compiled_rule is None:
            self._last_runtime_rows = []
            return np.zeros(NUM_UAV * 3, dtype=np.float32)
        try:
            features = build_runtime_feature_rows(
                env, snapshot, movement_mask, actor_action,
                episode_seconds=episode_seconds,
            )
            bias, rows = evaluate_rule_for_joint_action(
                self._compiled_rule,
                features,
                movement_mask,
                sigma_episode_start=sigma_episode_start,
                weight=1.0,
            )
            for row in rows:
                row["evaluation_status"] = (
                    "rule_evaluated"
                    if bool(row.get("movement_mask", False))
                    else "not_evaluated_other_controller"
                )
        except (ArithmeticError, ExplorerValidationError, TypeError, ValueError) as exc:
            mask = np.asarray(movement_mask, dtype=bool).reshape(NUM_UAV)
            bias = np.zeros(NUM_UAV * 3, dtype=np.float32)
            rows = [
                {
                    "uav_id": uav_id,
                    "movement_mask": bool(mask[uav_id]),
                    "raw_bias": [0.0, 0.0, 0.0],
                    "bounded_bias": [0.0, 0.0, 0.0],
                    "effective_bias": [0.0, 0.0, 0.0],
                    "clipped_components": 0,
                    "runtime_failure": (
                        f"{type(exc).__name__}: {exc}" if mask[uav_id] else None
                    ),
                    "evaluation_status": (
                        "runtime_fallback"
                        if mask[uav_id]
                        else "not_evaluated_other_controller"
                    ),
                    "trace": {
                        "conditions_evaluated": 0,
                        "conditions_true": 0,
                        "short_circuits": 0,
                    },
                    "features": None,
                }
                for uav_id in range(NUM_UAV)
            ]
        self._last_runtime_rows = rows
        failures = sum(item.get("runtime_failure") is not None for item in rows)
        self.runtime_failure_total += failures
        return bias

    def observe_movement_boundary(
        self,
        env: Any,
        snapshot: Mapping[str, Any],
        movement_mask: Any,
        actor_action: Any,
        *,
        episode_seconds: float,
    ) -> None:
        """Capture service geometry even before guidance or during fallback."""

        if self._compiled_rule is not None and len(self._last_runtime_rows) == NUM_UAV:
            return
        mask = np.asarray(movement_mask, dtype=bool).reshape(NUM_UAV)
        try:
            features = build_runtime_feature_rows(
                env,
                snapshot,
                mask,
                actor_action,
                episode_seconds=episode_seconds,
            )
            self._last_runtime_rows = [
                {
                    "uav_id": uav_id,
                    "movement_mask": bool(mask[uav_id]),
                    "raw_bias": [0.0, 0.0, 0.0],
                    "bounded_bias": [0.0, 0.0, 0.0],
                    "effective_bias": [0.0, 0.0, 0.0],
                    "clipped_components": 0,
                    "runtime_failure": None,
                    "evaluation_status": "not_evaluated_no_active_rule",
                    "trace": {
                        "conditions_evaluated": 0,
                        "conditions_true": 0,
                        "short_circuits": 0,
                    },
                    "features": _json_clone(features[uav_id]),
                }
                for uav_id in range(NUM_UAV)
            ]
        except (ArithmeticError, ExplorerValidationError, TypeError, ValueError) as exc:
            self._last_runtime_rows = [
                {
                    "uav_id": uav_id,
                    "movement_mask": bool(mask[uav_id]),
                    "raw_bias": [0.0, 0.0, 0.0],
                    "bounded_bias": [0.0, 0.0, 0.0],
                    "effective_bias": [0.0, 0.0, 0.0],
                    "clipped_components": 0,
                    "runtime_failure": None,
                    "evaluation_status": "feature_capture_unavailable",
                    "feature_capture_error": f"{type(exc).__name__}: {exc}",
                    "trace": {
                        "conditions_evaluated": 0,
                        "conditions_true": 0,
                        "short_circuits": 0,
                    },
                    "features": None,
                }
                for uav_id in range(NUM_UAV)
            ]

    def consume_last_runtime_rows(self) -> list[dict[str, Any]]:
        return _json_clone(self._last_runtime_rows)

    def record_slot_service(self, record: dict[str, Any]) -> None:
        self.slot_service_records.append(_json_clone(record))

    def complete_episode(self, *, episode_index: int, metrics: dict[str, Any]) -> None:
        movement, cases = state_rule_movement_summary(self.movement_records)
        communication = communication_bottleneck_summary_state_rule(
            self.routing_records, metrics
        )
        service = slot_service_summary(self.slot_service_records, metrics)
        roles = {
            key: value["role_decision_counts"]
            for key, value in movement["by_uav"].items()
        }
        feedback = {
            "schema_version": LLM_STATE_RULE_VERSION,
            "summary_version": LLM_STATE_RULE_SUMMARY_VERSION,
            "metrics": _json_clone(metrics),
            "observed_task_and_traffic_summary": {
                "role_decision_counts_by_uav": roles,
                "source_and_task_metrics": {
                    key: value for key, value in metrics.items()
                    if key.startswith(("fov_", "com_"))
                },
            },
            "movement_and_rule_application_summaries": movement,
            "fov_com_and_slot_service_summaries": {
                "routing_decision_observations": communication,
                "actual_within_slot_service": service,
            },
            "representative_cases": cases,
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
                    "slot_service_records": self.slot_service_records,
                },
            )
            _atomic_json(self.root / "state.json", self.state_dict())
        print(
            f"[LLM state rule] episode={episode_index + 1} summary "
            f"controlled={movement['totals'].get('controlled_decisions', 0)} "
            f"runtime_failures={movement['totals'].get('runtime_failures', 0)} "
            f"requests={self.statistics['requests_total']} "
            f"failed_attempts={self.statistics['failed_attempts_total']} "
            f"fallback_episodes={self.statistics['td3_fallback_episodes_total']}",
            flush=True,
        )


def explorer_metadata_state_rule(config: LLMStateRuleConfig) -> dict[str, Any]:
    return {
        "schema_version": LLM_STATE_RULE_VERSION,
        "method_id": LLM_STATE_RULE_METHOD_ID,
        "prompt_version": config.prompt_version,
        "summary_version": LLM_STATE_RULE_SUMMARY_VERSION,
        "cache_version": LLM_STATE_RULE_CACHE_VERSION,
        "rule_schema_version": STATE_RULE_SCHEMA_VERSION,
        "rule_language_version": STATE_RULE_LANGUAGE_VERSION,
        "feature_catalog_version": STATE_RULE_FEATURE_CATALOG_VERSION,
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "episode_joint_transitions": config.episode_joint_transitions,
        "update_interval_episodes": 1,
        "guidance_weight_w": 1.0,
        "annealing_end_episode": None,
        "analysis_reserved_output_tokens": config.analysis_max_output_tokens,
        "rule_reserved_output_tokens": config.guidance_max_output_tokens,
        "context_window_tokens": config.context_window_tokens,
        "safety_margin_tokens": config.safety_margin_tokens,
        "legacy_input_hard_limit_tokens": None,
        "td3_intervention": "latest-state shared rule; raw then host clip to episode-start sigma; fixed w=1",
        "routing_intervention": "none_original_safe_ddqn_epsilon_greedy",
        "evaluation_calls_llm": False,
    }
