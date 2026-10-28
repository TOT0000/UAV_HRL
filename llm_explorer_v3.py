"""Compact, annealed LLM exploration guidance for the isolated v3 method."""

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
    ROUTING_PRIOR_WEIGHT,
    PromptBudgetError,
    _atomic_json,
    _compact_json,
    _json_clone,
    _routing_state_named_values,
    environment_constants_and_definitions,
    prompt_token_budget,
    render_prompt,
    stratified_routing_samples,
    validate_td3_bias,
)


LLM_EXPLORER_V3_METHOD_ID = "td3_dinkelbach_llm_explorer_v3"
LLM_EXPLORER_V3_VERSION = "uav-hrl-llm-explorer-v3"
LLM_EXPLORER_V3_PROMPT_VERSION = "uav-hrl-llm-explorer-prompts-v5"
LLM_EXPLORER_V3_SUMMARY_VERSION = "shared-goodput-event-time-episode-summary-v4"
LLM_EXPLORER_V3_CASE_VERSION = "deterministic-diverse-cases-v2"
LLM_EXPLORER_V3_GUIDANCE_CONTEXT_VERSION = "self-contained-raw-td3-v1"
LLM_EXPLORER_V3_ANNEAL_END_EPISODE = 600
LLM_EXPLORER_V3_MAX_INPUT_TOKENS = 20_000
LLM_EXPLORER_V3_TARGET_INPUT_TOKENS = 12_000


@dataclass(frozen=True)
class LLMExplorerV3Config(LLMExplorerConfig):
    prompt_version: str = LLM_EXPLORER_V3_PROMPT_VERSION
    input_hard_limit_tokens: int = LLM_EXPLORER_V3_MAX_INPUT_TOKENS
    annealing_end_episode: int = LLM_EXPLORER_V3_ANNEAL_END_EPISODE
    episode_joint_transitions: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_version != LLM_EXPLORER_V3_PROMPT_VERSION:
            raise ValueError("v3 prompt version is fixed by the method contract")
        if int(self.input_hard_limit_tokens) != LLM_EXPLORER_V3_MAX_INPUT_TOKENS:
            raise ValueError("v3 input hard limit must remain 20,000 tokens")
        if int(self.annealing_end_episode) != LLM_EXPLORER_V3_ANNEAL_END_EPISODE:
            raise ValueError("v3 annealing end episode must remain 600")
        if int(self.episode_joint_transitions) <= 0:
            raise ValueError("episode_joint_transitions must be positive")


def first_eligible_episode(
    warmup_joint_transitions: int, episode_joint_transitions: int
) -> int:
    """Return the first 1-based episode whose start is past movement warmup."""

    warmup = int(warmup_joint_transitions)
    per_episode = int(episode_joint_transitions)
    if warmup < 0 or per_episode <= 0:
        raise ValueError("warmup must be non-negative and episode length positive")
    return int(math.ceil(warmup / per_episode)) + 1


def llm_guidance_weight(
    episode: int, *, start_episode: int, end_episode: int = 600
) -> float:
    """Fixed-horizon, 1-based annealing weight independent of run length."""

    episode = int(episode)
    start = int(start_episode)
    end = int(end_episode)
    if episode < start or start >= end:
        return 0.0
    return float(np.clip((end - episode) / float(end - start), 0.0, 1.0))


def routing_action_name(sender: int, action: int) -> str:
    sender = int(sender)
    action = int(action)
    if action == sender:
        return "WAIT"
    if action == NUM_UAV:
        return "GS"
    if 0 <= action < NUM_UAV:
        return f"UAV_{action}"
    raise ExplorerValidationError(f"routing action index {action} is invalid")


def named_routing_prior_to_matrix(value: Any) -> np.ndarray:
    if not isinstance(value, dict):
        raise ExplorerValidationError("routing_prior must be an object")
    expected_senders = {f"UAV_{index}" for index in range(NUM_UAV)}
    if set(value) != expected_senders:
        missing = sorted(expected_senders.difference(value))
        extra = sorted(set(value).difference(expected_senders))
        raise ExplorerValidationError(
            f"routing_prior must contain every sender exactly once; missing={missing}, extra={extra}"
        )
    matrix = np.zeros((NUM_UAV, ROUTING_ACTION_DIM), dtype=np.float64)
    allowed = {"GS", "WAIT", *(f"UAV_{index}" for index in range(NUM_UAV))}
    for sender in range(NUM_UAV):
        row = value[f"UAV_{sender}"]
        if not isinstance(row, dict) or not row:
            raise ExplorerValidationError(
                f"routing_prior.UAV_{sender} must be a non-empty object"
            )
        unknown = sorted(set(row).difference(allowed))
        if unknown:
            raise ExplorerValidationError(
                f"routing_prior.UAV_{sender} has unknown actions {unknown}"
            )
        if f"UAV_{sender}" in row:
            raise ExplorerValidationError(
                f"routing_prior.UAV_{sender} must name its self-action WAIT"
            )
        for name, raw_probability in row.items():
            if isinstance(raw_probability, (bool, np.bool_)):
                raise ExplorerValidationError("routing probabilities must not be booleans")
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError) as exc:
                raise ExplorerValidationError(
                    f"routing probability {name} for UAV_{sender} is not numeric"
                ) from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ExplorerValidationError(
                    f"routing probability {name} for UAV_{sender} must be finite in [0,1]"
                )
            action = (
                NUM_UAV
                if name == "GS"
                else sender
                if name == "WAIT"
                else int(name.removeprefix("UAV_"))
            )
            matrix[sender, action] = probability
        total = float(matrix[sender].sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ExplorerValidationError(
                f"routing_prior.UAV_{sender} probabilities must sum to 1"
            )
        matrix[sender] /= total
    return matrix


def routing_matrix_to_named(value: Any, *, include_zeros: bool = False) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (NUM_UAV, ROUTING_ACTION_DIM):
        raise ExplorerValidationError("routing prior matrix must have shape [16,17]")
    result: dict[str, Any] = {}
    for sender in range(NUM_UAV):
        row: dict[str, float] = {}
        for action, probability in enumerate(matrix[sender]):
            if include_zeros or float(probability) != 0.0:
                row[routing_action_name(sender, action)] = float(probability)
        result[f"UAV_{sender}"] = row
    return result


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExplorerValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_guidance_response_v3(
    content: str, sigma_episode_start: float
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, str]]:
    errors: dict[str, str] = {}
    try:
        value = json.loads(str(content), object_pairs_hook=_reject_duplicate_pairs)
    except (TypeError, json.JSONDecodeError, ExplorerValidationError) as exc:
        reason = f"invalid JSON: {type(exc).__name__}: {exc}"
        return None, None, {"td3": reason, "routing": reason}
    if not isinstance(value, dict):
        return None, None, {
            "td3": "response root is not an object",
            "routing": "response root is not an object",
        }
    if set(value) != {"td3_bias", "routing_prior"}:
        reason = "response must contain exactly td3_bias and routing_prior"
        return None, None, {"td3": reason, "routing": reason}
    try:
        td3 = validate_td3_bias(value["td3_bias"], sigma_episode_start)
    except (TypeError, ValueError, ExplorerValidationError) as exc:
        td3 = None
        errors["td3"] = str(exc)
    try:
        routing = named_routing_prior_to_matrix(value["routing_prior"])
    except (TypeError, ValueError, ExplorerValidationError) as exc:
        routing = None
        errors["routing"] = str(exc)
    return td3, routing, errors


def _numeric_summary(values: list[float], digits: int = 5) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values], dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": round(float(clean.mean()), digits),
        "min": round(float(clean.min()), digits),
        "max": round(float(clean.max()), digits),
    }


def _case_domain(case: dict[str, Any]) -> str:
    if case.get("domain") in {"movement", "routing"}:
        return str(case["domain"])
    return (
        "movement"
        if str(case.get("kind", "")).startswith("movement_")
        else "routing"
    )


def _case_entity(case: dict[str, Any]) -> str:
    if case.get("uav_id") is not None:
        return f"UAV_{int(case['uav_id'])}"
    return str(case.get("sender", "unknown"))


def _case_time_segments(cases: list[dict[str, Any]]) -> dict[int, int | None]:
    timed: list[tuple[int, float]] = []
    for index, case in enumerate(cases):
        try:
            value = float(case.get("time_s"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            timed.append((index, value))
    if not timed:
        return {index: None for index in range(len(cases))}
    lower = min(value for _, value in timed)
    upper = max(value for _, value in timed)
    result: dict[int, int | None] = {
        index: None for index in range(len(cases))
    }
    for index, value in timed:
        if upper == lower:
            result[index] = 0
        else:
            result[index] = min(
                2, int(3.0 * (value - lower) / (upper - lower))
            )
    return result


def _diverse_case_subset(
    cases: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Select real cases deterministically by kind, entity, then time region."""

    maximum = max(0, int(maximum))
    if maximum == 0 or not cases:
        return []
    candidates = [_json_clone(case) for case in cases]
    segments = _case_time_segments(candidates)

    def canonical(index: int) -> tuple[Any, ...]:
        case = candidates[index]
        raw_time = case.get("time_s")
        try:
            time_value = float(raw_time)
        except (TypeError, ValueError):
            time_value = math.inf
        return (
            str(case.get("kind", "")),
            _case_entity(case),
            time_value,
            json.dumps(case, ensure_ascii=False, sort_keys=True),
        )

    remaining = set(range(len(candidates)))
    selected: list[int] = []
    seen_kinds: set[str] = set()
    seen_entities: set[str] = set()
    seen_segments: set[int] = set()
    while remaining and len(selected) < maximum:
        def rank(index: int) -> tuple[Any, ...]:
            case = candidates[index]
            segment = segments[index]
            new_kind = int(str(case.get("kind", "")) not in seen_kinds)
            new_entity = int(_case_entity(case) not in seen_entities)
            new_segment = int(segment is not None and segment not in seen_segments)
            # Entity and time diversity jointly outrank another adjacent event.
            return (
                new_kind,
                new_entity + new_segment,
                new_entity,
                new_segment,
            )

        best_rank = max(rank(index) for index in remaining)
        eligible = [index for index in remaining if rank(index) == best_rank]
        chosen = min(eligible, key=canonical)
        selected.append(chosen)
        remaining.remove(chosen)
        selected_case = candidates[chosen]
        seen_kinds.add(str(selected_case.get("kind", "")))
        seen_entities.add(_case_entity(selected_case))
        if segments[chosen] is not None:
            seen_segments.add(int(segments[chosen]))
    return [candidates[index] for index in selected]


def select_representative_cases(
    movement_cases: list[dict[str, Any]],
    routing_cases: list[dict[str, Any]],
    *,
    movement_maximum: int = 3,
    routing_maximum: int = 5,
) -> list[dict[str, Any]]:
    """Keep a small diverse set and reserve a real ordinary routing example."""

    movement = _diverse_case_subset(movement_cases, movement_maximum)
    routing_maximum = max(0, int(routing_maximum))
    ordinary = [
        case
        for case in routing_cases
        if case.get("kind") == "ordinary_recorded_routing_decision"
    ]
    anomalies = [
        case
        for case in routing_cases
        if case.get("kind") != "ordinary_recorded_routing_decision"
    ]
    if routing_maximum == 0:
        routing: list[dict[str, Any]] = []
    elif ordinary:
        # Ordinary means only that no configured anomaly filter matched; it is
        # illustrative and is never treated as an optimal action.
        routing = _diverse_case_subset(anomalies, routing_maximum - 1)
        routing.extend(
            _diverse_case_subset(ordinary, routing_maximum - len(routing))
        )
    else:
        routing = _diverse_case_subset(anomalies, routing_maximum)
    return movement + routing


def movement_episode_summary(
    records: list[dict[str, Any]], guidance: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_bias = np.asarray(
        guidance.get("td3_bias_raw", guidance.get("td3_bias", np.zeros((16, 3)))),
        dtype=np.float64,
    ).reshape(NUM_UAV, 3)
    effective_bias = np.asarray(
        guidance.get("td3_bias_effective", np.zeros((16, 3))), dtype=np.float64
    ).reshape(NUM_UAV, 3)
    by_uav: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    for uav_id in range(NUM_UAV):
        task_counts: Counter[str] = Counter()
        mask_true = 0
        actor_actions: list[np.ndarray] = []
        target_distances: list[float] = []
        energy: list[float] = []
        nonzero_bias_count = 0
        for record in records:
            uavs = record.get("uavs", [])
            if uav_id >= len(uavs):
                continue
            item = uavs[uav_id]
            for task in item.get("task_types", []):
                task_counts[str(task)] += 1
            controlled = bool(item.get("movement_mask", False))
            mask_true += int(controlled)
            if controlled and item.get("actor_normalized_action") is not None:
                action = np.asarray(item["actor_normalized_action"], dtype=np.float64)
                if action.shape == (3,) and np.isfinite(action).all():
                    actor_actions.append(action)
                    if np.any(np.abs(effective_bias[uav_id]) > 1e-12):
                        nonzero_bias_count += 1
                    speed_mps = 5.0 * (float(action[0]) + 1.0)
                    if (
                        speed_mps >= 9.5
                        and float(raw_bias[uav_id, 0]) > 0.0
                    ):
                        cases.append(
                            {
                                "domain": "movement",
                                "kind": "movement_speed_saturation_with_positive_bias",
                                "episode": int(record.get("episode_index", -1)) + 1,
                                "time_s": record.get("time_seconds"),
                                "uav_id": uav_id,
                                "tasks": item.get("task_types", []),
                                "movement_mask": True,
                                "action_source": record.get("action_source"),
                                "actor_speed_mps": round(speed_mps, 4),
                                "raw_speed_bias": round(float(raw_bias[uav_id, 0]), 6),
                            }
                        )
            for target in item.get("observable_targets", []):
                relative = np.asarray(target.get("relative_position_m", []), dtype=float)
                if relative.shape == (3,) and np.isfinite(relative).all():
                    target_distances.append(float(np.linalg.norm(relative)))
            if item.get("movement_energy_j") is not None:
                energy.append(float(item["movement_energy_j"]))
        actions = np.asarray(actor_actions, dtype=np.float64)
        speeds = (5.0 * (actions[:, 0] + 1.0)).tolist() if actions.size else []
        vertical = (2.0 * actions[:, 2]).tolist() if actions.size else []
        if actions.size:
            heading_radians = np.pi * actions[:, 1]
            resultant = np.mean(np.exp(1j * heading_radians))
            circular_heading = (
                None
                if abs(resultant) < 1e-12
                else round(float(np.angle(resultant)), 5)
            )
            near_limit = float(np.mean(np.asarray(speeds) >= 9.5))
        else:
            circular_heading = None
            near_limit = None
        by_uav[f"UAV_{uav_id}"] = {
            "task_occurrence_counts": dict(sorted(task_counts.items())),
            "movement_mask_true_count": int(mask_true),
            "td3_controlled_actor_decision_count": len(actor_actions),
            "actor_horizontal_speed_mps": _numeric_summary(speeds),
            "actor_vertical_velocity_mps": _numeric_summary(vertical),
            "actor_heading_circular_mean_rad": circular_heading,
            "actor_near_10_mps_ratio": near_limit,
            "raw_bias": raw_bias[uav_id].round(6).tolist(),
            "effective_bias": effective_bias[uav_id].round(6).tolist(),
            "episode_start_sigma": guidance.get("sigma_episode_start"),
            "nonzero_effective_bias_controlled_count": int(nonzero_bias_count),
            "observable_target_distance_m": _numeric_summary(target_distances, 3),
            "movement_energy_j": _numeric_summary(energy, 4),
        }
    return by_uav, cases


def routing_episode_summary(
    records: list[dict[str, Any]], maximum_cases: int = 8, episode: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_sender: dict[str, Any] = {}
    anomaly_cases: list[dict[str, Any]] = []
    ordinary_cases: list[dict[str, Any]] = []
    for sender in range(NUM_UAV):
        sender_records = [
            record for record in records if int(record["sender_uav_id"]) == sender
        ]
        branch_rows = {
            "epsilon_exploration": [
                record
                for record in sender_records
                if record.get("decision_branch") == "epsilon_exploration"
            ],
            "safe_q_greedy": [
                record
                for record in sender_records
                if record.get("decision_branch") == "safe_q_greedy"
            ],
        }
        legal_counts: Counter[str] = Counter()
        capacity: dict[str, list[float]] = {}
        next_backlog: dict[str, list[float]] = {}
        gs_progress: dict[str, list[float]] = {}
        backlogs: list[float] = []
        deadlines: list[float] = []
        for record in sender_records:
            named = _routing_state_named_values(record["compact_routing_state"])
            mask = np.asarray(record["legal_action_mask"], dtype=bool)
            for action in np.flatnonzero(mask):
                action_name = routing_action_name(sender, int(action))
                legal_counts[action_name] += 1
                capacity.setdefault(action_name, []).append(
                    float(named[f"link_capacity[{action}]"])
                )
                next_backlog.setdefault(action_name, []).append(
                    float(named[f"next_hop_backlog[{action}]"])
                )
                gs_progress.setdefault(action_name, []).append(
                    float(named[f"gs_progress_norm[{action}]"])
                )
            backlogs.append(float(record.get("sender_total_backlog_bits", 0.0)))
            if record.get("hol_remaining_deadline_seconds") is not None:
                deadlines.append(float(record["hol_remaining_deadline_seconds"]))
            selected = int(record["selected_action"])
            matched_anomaly = False
            if (
                record.get("decision_branch") == "epsilon_exploration"
                and selected == sender
                and float(record.get("sender_total_backlog_bits", 0.0)) > 0.0
                and any(action not in (sender,) for action in np.flatnonzero(mask))
            ):
                distribution = record.get("exploration_distribution")
                wait_probability = (
                    None if distribution is None else float(distribution[sender])
                )
                if wait_probability is None or wait_probability >= 0.5:
                    matched_anomaly = True
                    anomaly_cases.append(
                        {
                            "domain": "routing",
                            "kind": "exploration_wait_with_backlog_and_other_legal_action",
                            "episode": episode,
                            "time_s": record.get("time_seconds"),
                            "sender": f"UAV_{sender}",
                            "selected_action": "WAIT",
                            "backlog_bits": record.get("sender_total_backlog_bits"),
                            "remaining_deadline_s": record.get(
                                "hol_remaining_deadline_seconds"
                            ),
                            "legal_actions": [
                                routing_action_name(sender, int(action))
                                for action in np.flatnonzero(mask)
                            ],
                            "wait_exploration_probability": wait_probability,
                        }
                    )
            if bool(record.get("zero_legal_prior_mass_fallback", False)):
                matched_anomaly = True
                anomaly_cases.append(
                    {
                        "domain": "routing",
                        "kind": "masked_prior_zero_mass",
                        "episode": episode,
                        "time_s": record.get("time_seconds"),
                        "sender": f"UAV_{sender}",
                        "selected_action": routing_action_name(sender, selected),
                        "legal_actions": [
                            routing_action_name(sender, int(action))
                            for action in np.flatnonzero(mask)
                        ],
                    }
                )
            if not matched_anomaly:
                ordinary_cases.append(
                    {
                        "domain": "routing",
                        "kind": "ordinary_recorded_routing_decision",
                        "episode": episode,
                        "time_s": record.get("time_seconds"),
                        "sender": f"UAV_{sender}",
                        "branch": record.get("decision_branch"),
                        "selected_action": routing_action_name(sender, selected),
                    }
                )
        branch_summary: dict[str, Any] = {}
        for branch, branch_records in branch_rows.items():
            count = len(branch_records)
            waits = sum(
                int(record["selected_action"]) == sender for record in branch_records
            )
            branch_summary[branch] = {
                "decision_count": count,
                "wait_count": waits,
                "wait_ratio": None if count == 0 else waits / count,
                "next_hop_count": count - waits,
                "zero_legal_prior_mass_count": sum(
                    bool(record.get("zero_legal_prior_mass_fallback", False))
                    for record in branch_records
                ),
            }
            zero_count = branch_summary[branch]["zero_legal_prior_mass_count"]
            branch_summary[branch]["zero_legal_prior_mass_ratio"] = (
                None if count == 0 else zero_count / count
            )
        by_sender[f"UAV_{sender}"] = {
            "decision_count": len(sender_records),
            "branches": branch_summary,
            "selected_action_counts": dict(
                Counter(
                    routing_action_name(sender, int(record["selected_action"]))
                    for record in sender_records
                )
            ),
            "legal_action_occurrence_counts": dict(sorted(legal_counts.items())),
            "legal_action_observable_summaries": {
                action: {
                    "reference_capacity_norm": _numeric_summary(capacity[action], 4),
                    "next_hop_backlog_log_norm": _numeric_summary(
                        next_backlog[action], 4
                    ),
                    "gs_progress_norm": _numeric_summary(gs_progress[action], 4),
                }
                for action in sorted(legal_counts)
            },
            "sender_backlog_bits": _numeric_summary(backlogs, 2),
            "hol_remaining_deadline_s": _numeric_summary(deadlines, 4),
            "packet_delivery_attribution": "unknown_not_recorded_per_decision",
        }
    return by_sender, select_representative_cases(
        [],
        anomaly_cases + ordinary_cases,
        movement_maximum=0,
        routing_maximum=maximum_cases,
    )


def _summary_tuple(value: Any) -> list[Any] | None:
    if not isinstance(value, dict):
        return None
    return [
        value.get("count"),
        value.get("mean"),
        value.get("min"),
        value.get("max"),
    ]


def _compact_movement_summaries(value: Any) -> dict[str, Any]:
    summaries = value if isinstance(value, dict) else {}
    rows: list[list[Any]] = []
    for uav_id in range(NUM_UAV):
        item = summaries.get(f"UAV_{uav_id}")
        if not isinstance(item, dict):
            rows.append([uav_id] + [None] * 13)
            continue
        task_counts = item.get("task_occurrence_counts")
        rows.append(
            [
                uav_id,
                (
                    None
                    if not isinstance(task_counts, dict)
                    else [
                        [str(key), int(count)]
                        for key, count in sorted(task_counts.items())
                    ]
                ),
                item.get("movement_mask_true_count"),
                item.get("td3_controlled_actor_decision_count"),
                _summary_tuple(item.get("actor_horizontal_speed_mps")),
                _summary_tuple(item.get("actor_vertical_velocity_mps")),
                item.get("actor_heading_circular_mean_rad"),
                item.get("actor_near_10_mps_ratio"),
                item.get("raw_bias"),
                item.get("effective_bias"),
                item.get("episode_start_sigma"),
                item.get("nonzero_effective_bias_controlled_count"),
                _summary_tuple(item.get("observable_target_distance_m")),
                _summary_tuple(item.get("movement_energy_j")),
            ]
        )
    return {
        "columns": [
            "uav_id",
            "task_occurrence_[name,count]",
            "movement_mask_true_count",
            "td3_actor_decision_count",
            "horizontal_speed_mps_[count,mean,min,max]",
            "vertical_velocity_mps_[count,mean,min,max]",
            "heading_circular_mean_rad",
            "speed_ge_9.5_ratio",
            "raw_bias_[speed,heading,vertical]",
            "effective_bias_w_times_raw_[speed,heading,vertical]",
            "episode_start_sigma",
            "nonzero_effective_bias_actor_count",
            "observable_target_distance_m_[count,mean,min,max]",
            "movement_energy_j_[count,mean,min,max]",
        ],
        "rows": rows,
    }


def _routing_count_vector(
    item: dict[str, Any], key: str, sender: int
) -> list[int] | None:
    value = item.get(key)
    if not isinstance(value, dict):
        return None
    return [
        int(value.get(routing_action_name(sender, action), 0))
        for action in range(ROUTING_ACTION_DIM)
    ]


def _routing_observable_means(
    item: dict[str, Any], field: str, sender: int
) -> list[Any] | None:
    values = item.get("legal_action_observable_summaries")
    if not isinstance(values, dict):
        return None
    result: list[Any] = []
    for action in range(ROUTING_ACTION_DIM):
        action_value = values.get(routing_action_name(sender, action))
        summary = (
            action_value.get(field) if isinstance(action_value, dict) else None
        )
        result.append(summary.get("mean") if isinstance(summary, dict) else None)
    return result


def _compact_branch(value: Any) -> list[Any] | None:
    if not isinstance(value, dict):
        return None
    return [
        value.get("decision_count"),
        value.get("wait_count"),
        value.get("next_hop_count"),
        value.get("zero_legal_prior_mass_count"),
    ]


def _compact_routing_summaries(value: Any) -> dict[str, Any]:
    summaries = value if isinstance(value, dict) else {}
    rows: list[list[Any]] = []
    for sender in range(NUM_UAV):
        item = summaries.get(f"UAV_{sender}")
        if not isinstance(item, dict):
            rows.append([sender] + [None] * 11)
            continue
        branches = item.get("branches")
        branches = branches if isinstance(branches, dict) else {}
        rows.append(
            [
                sender,
                item.get("decision_count"),
                _compact_branch(branches.get("epsilon_exploration")),
                _compact_branch(branches.get("safe_q_greedy")),
                _routing_count_vector(item, "selected_action_counts", sender),
                _routing_count_vector(
                    item, "legal_action_occurrence_counts", sender
                ),
                _routing_observable_means(
                    item, "reference_capacity_norm", sender
                ),
                _routing_observable_means(
                    item, "next_hop_backlog_log_norm", sender
                ),
                _routing_observable_means(item, "gs_progress_norm", sender),
                _summary_tuple(item.get("sender_backlog_bits")),
                _summary_tuple(item.get("hol_remaining_deadline_s")),
                item.get("packet_delivery_attribution"),
            ]
        )
    return {
        "action_index_semantics": (
            "indices 0..15 name UAV_k except sender index means WAIT; index 16 means GS"
        ),
        "unknown_and_zero": (
            "null means missing/unknown; numeric 0 is observed zero; "
            "count vectors give denominators"
        ),
        "columns": [
            "sender_uav_id",
            "decision_count",
            "epsilon_[decision,wait,next_hop,masked_prior_zero_mass]",
            "greedy_[decision,wait,next_hop,masked_prior_zero_mass]",
            "selected_count_by_action_index",
            "legal_occurrence_count_by_action_index",
            "reference_capacity_mean_by_action_index",
            "next_hop_backlog_mean_by_action_index",
            "gs_progress_mean_by_action_index",
            "sender_backlog_bits_[count,mean,min,max]",
            "hol_remaining_deadline_s_[count,mean,min,max]",
            "per_decision_delivery_attribution",
        ],
        "rows": rows,
    }


def _routing_prior_matrix_from_guidance(guidance: dict[str, Any]) -> Any:
    value = guidance.get("routing_prior")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = np.asarray([], dtype=np.float64)
    if matrix.shape == (NUM_UAV, ROUTING_ACTION_DIM) and np.isfinite(matrix).all():
        return matrix.round(6).tolist()
    named = guidance.get("routing_prior_named")
    if isinstance(named, dict):
        try:
            return named_routing_prior_to_matrix(named).round(6).tolist()
        except (TypeError, ValueError, ExplorerValidationError):
            pass
    return None


def _td3_bias_rows(value: Any) -> list[list[float]] | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (NUM_UAV, 3) or not np.isfinite(matrix).all():
        return None
    return matrix.round(6).tolist()


def _recorded_raw_td3_bias(previous: dict[str, Any]) -> dict[str, Any]:
    guidance_value = previous.get("applied_guidance")
    guidance = guidance_value if isinstance(guidance_value, dict) else {}
    rows = _td3_bias_rows(guidance.get("td3_bias_raw"))
    source = "applied_guidance.td3_bias_raw"

    if rows is None:
        movement_value = previous.get("movement_summaries")
        movement = movement_value if isinstance(movement_value, dict) else {}
        recorded_rows = [
            (
                movement[f"UAV_{uav_id}"].get("raw_bias")
                if isinstance(movement.get(f"UAV_{uav_id}"), dict)
                else None
            )
            for uav_id in range(NUM_UAV)
        ]
        rows = _td3_bias_rows(recorded_rows)
        source = "movement_summaries.UAV_k.raw_bias"

    if rows is None and previous.get("schema_version") == LLM_EXPLORER_V3_VERSION:
        # In persisted v3 guidance, td3_bias is the validated raw model output;
        # v3 creates td3_bias_effective separately after applying w exactly once.
        rows = _td3_bias_rows(guidance.get("td3_bias"))
        source = "legacy_v3_applied_guidance.td3_bias_confirmed_raw"

    if rows is None:
        source = "unknown_not_recorded"
    return {
        "source": source,
        "uav_id_order": list(range(NUM_UAV)),
        "columns": ["speed", "heading", "vertical"],
        "rows": rows,
        "semantics": (
            "previous-episode recorded raw TD3 bias before multiplication by the "
            "previous annealing weight; rows=null means unknown, not zero"
        ),
        "current_episode_constraint": (
            "use only as historical context; every new raw bias must satisfy the "
            "current episode sigma bound and must not be copied blindly"
        ),
    }


def _compact_applied_guidance(
    guidance_value: Any,
    movement_summaries: Any,
    raw_td3_bias: dict[str, Any],
) -> dict[str, Any]:
    guidance = guidance_value if isinstance(guidance_value, dict) else {}
    result = {
        key: _json_clone(guidance[key])
        for key in (
            "enabled",
            "td3_fallback",
            "routing_fallback",
            "td3_reason",
            "routing_reason",
            "analysis_attempt",
            "td3_source_attempt",
            "routing_source_attempt",
            "annealing_weight",
            "routing_alpha",
            "sigma_episode_start",
        )
        if key in guidance
    }
    result["routing_prior_action_index_semantics"] = (
        "rows are sender UAV IDs 0..15; columns 0..15 are UAV action indices "
        "with the sender column meaning WAIT; column 16 is GS"
    )
    result["routing_prior_rows"] = _routing_prior_matrix_from_guidance(guidance)
    movement = movement_summaries if isinstance(movement_summaries, dict) else {}
    has_recorded_bias = all(
        isinstance(movement.get(f"UAV_{uav_id}"), dict)
        and "raw_bias" in movement[f"UAV_{uav_id}"]
        and "effective_bias" in movement[f"UAV_{uav_id}"]
        for uav_id in range(NUM_UAV)
    )
    if has_recorded_bias:
        result["td3_bias_location"] = (
            "movement summary rows contain raw and actually applied w*raw biases"
        )
    else:
        effective = guidance.get("td3_bias_effective")
        for key, matrix_value in (
            ("td3_raw_bias_rows", raw_td3_bias.get("rows")),
            ("td3_effective_bias_rows", effective),
        ):
            rows = _td3_bias_rows(matrix_value)
            if rows is not None:
                result[key] = rows
    return result


def _compact_cases(cases: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    items = []
    for case in cases:
        item = {
            key: value
            for key, value in case.items()
            if key != "interpretation"
        }
        items.append(_json_clone(item))
    return {
        "selection_profile": profile,
        "semantics": (
            "deterministic real-record illustrations only; ordinary means no configured "
            "anomaly filter matched, not that the action was optimal; frequencies "
            "come from full summaries"
        ),
        "items": items,
    }


def build_observable_initial_state_v3(env: Any, packet_engine: Any) -> dict[str, Any]:
    """Compact current-episode observations without carrying discoveries forward."""

    known_rois = {
        int(gt.id): gt for gt in env.gts if bool(getattr(gt, "is_found", False))
    }
    observable_srs = {
        int(sr.id): sr
        for sr in env.SR_teams
        if getattr(sr, "assigned_gt_id", None) is not None
        and int(sr.assigned_gt_id) in known_rois
    }
    backlog = getattr(packet_engine, "backlog_bits", {})
    rows = []
    for uav_id, uav in sorted(env.uav_dict.items()):
        tasks = []
        for task in env.multi_tasks.get(uav_id, ()):
            kind = str(task.get("task_type", "Hovering"))
            target = task.get("target_obj_id")
            observable = (
                kind == "FOV" and target is not None and int(target) in known_rois
            ) or (
                kind == "COM" and target is not None and int(target) in observable_srs
            )
            tasks.append([kind, int(target) if observable else None])
        rows.append(
            [
                int(uav_id),
                round(float(uav.x_u), 2),
                round(float(uav.y_u), 2),
                round(float(uav.z_u), 2),
                round(float(uav.energy), 2),
                round(float(backlog.get(uav_id, 0.0)), 2),
                tasks,
            ]
        )
    return _json_clone(
        {
            "time": "episode_start_before_first_action",
            "hidden_roi_count_disclosed": False,
            "uav_columns": [
                "id",
                "x_m",
                "y_m",
                "z_m",
                "energy_j",
                "backlog_bits",
                "observable_tasks_[type,target_or_null]",
            ],
            "uav_rows": rows,
            "known_rois_current_episode": [
                [
                    roi_id,
                    round(float(gt.x), 2),
                    round(float(gt.y), 2),
                    round(float(gt.z), 2),
                    round(float(gt.radius), 2),
                ]
                for roi_id, gt in sorted(known_rois.items())
            ],
            "observable_sr_current_episode": [
                [
                    sr_id,
                    round(float(sr.x), 2),
                    round(float(sr.y), 2),
                    round(float(sr.z), 2),
                    int(sr.assigned_gt_id),
                    bool(getattr(sr, "arrived", False)),
                ]
                for sr_id, sr in sorted(observable_srs.items())
            ],
            "ground_station_position_m": [float(value) for value in env.GS_pos],
        }
    )


def environment_definitions_v3(**kwargs: Any) -> dict[str, Any]:
    base = environment_constants_and_definitions(**kwargs)
    return {
        "schema_version": LLM_EXPLORER_V3_VERSION,
        "num_uav": NUM_UAV,
        "episode_seconds": base["episode_seconds"],
        "movement_interval_seconds": base["movement_interval_seconds"],
        "routing_slot_seconds": base["routing_slot_seconds"],
        "operating_area_m": base["operating_area_m"],
        "uav_altitude_bounds_m": base["uav_altitude_bounds_m"],
        "communication_range_3d_m_inclusive": base[
            "communication_range_3d_m_inclusive"
        ],
        "task_deadlines_seconds": base["task_deadlines_seconds"],
        "packet_generation_model": base["packet_generation_model"],
        "routing_decision_model": base["routing_decision_model"],
        "packet_service_model": base["packet_service_model"],
        "bandwidth_model": base["bandwidth_model"],
        "fading_model": base["fading_model"],
        "action_decode": base["action_decode"],
        "routing_actions": {
            "GS": 16,
            "WAIT": "sender index",
            "UAV_k": "index k; sender self-action must be named WAIT",
        },
        "summary_definitions": {
            "full_episode_statistics": "computed from every recorded decision before sampling",
            "movement_actor_statistics": "only rows whose movement_mask is true and actor action is recorded",
            "heading_mean": "circular mean in radians",
            "reference_capacity": "observable normalized reference, not realized service",
            "representative_cases": "deterministic illustrations, never frequency estimates or proven optima",
            "qos": "system QoS and routing-stage QoS are distinct",
        },
        "validity": base["validity"],
    }


def output_schema_v3() -> dict[str, Any]:
    sender_names = [f"UAV_{sender}" for sender in range(NUM_UAV)]
    return {
        "type": "object",
        "required": ["td3_bias", "routing_prior"],
        "additionalProperties": False,
        "properties": {
            "td3_bias": {
                "type": "array",
                "minItems": 16,
                "maxItems": 16,
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
            },
            "routing_prior": {
                "type": "object",
                "required": sender_names,
                "additionalProperties": False,
                "patternProperties": {
                    "^UAV_(?:[0-9]|1[0-5])$": {"$ref": "#/$defs/sender_prior"}
                },
            },
        },
        "$defs": {
            "sender_prior": {
                "type": "object",
                "description": (
                    "Sparse probabilities; omitted actions are zero. The validator "
                    "rejects the sender's own UAV_k name and requires WAIT instead."
                ),
                "minProperties": 1,
                "additionalProperties": False,
                "patternProperties": {
                    "^(?:GS|WAIT|UAV_(?:[0-9]|1[0-5]))$": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    }
                },
            }
        },
        "semantic_constraints": [
            "Each routing sender row sums to one within 1e-6.",
            "Allowed action names are GS, WAIT, and UAV_0 through UAV_15.",
            "The sender's own UAV_k name is forbidden; use WAIT.",
            "All values are finite JSON numbers; booleans are forbidden.",
        ],
    }


class DeterministicMockExplorerV3Client:
    def chat(self, *, model: str, prompt: str, **_: Any) -> dict[str, Any]:
        if "Analyze the previous episode" in prompt:
            content = (
                "Evidence: Offline mock inputs provide no directional evidence. "
                "Movement recommendations: retain neutral raw biases. "
                "Routing recommendations: retain approximately uniform named priors. "
                "Uncertainty: this is a deterministic local mock, not model inference."
            )
        else:
            uniform = 1.0 / ROUTING_ACTION_DIM
            routing = {}
            for sender in range(NUM_UAV):
                routing[f"UAV_{sender}"] = {
                    routing_action_name(sender, action): uniform
                    for action in range(ROUTING_ACTION_DIM)
                }
            content = _compact_json(
                {
                    "td3_bias": [[0.0, 0.0, 0.0] for _ in range(NUM_UAV)],
                    "routing_prior": routing,
                }
            )
        return {
            "content": content,
            "actual_model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }


class LLMExplorerV3Controller(LLMExplorerController):
    schema_version = LLM_EXPLORER_V3_VERSION
    artifact_directory_name = "llm_explorer_v3"

    def __init__(self, *, config: LLMExplorerV3Config | None = None, mock=False, client=None, **kwargs):
        resolved = config or LLMExplorerV3Config()
        if mock and client is None:
            client = DeterministicMockExplorerV3Client()
            mock = False
        super().__init__(
            config=resolved,
            mock=mock,
            client=client,
            lazy_client=True,
            **kwargs,
        )
        self.config: LLMExplorerV3Config
        self._current_weight = 0.0
        self._current_alpha = 0.0

    @staticmethod
    def _empty_statistics():
        statistics = LLMExplorerController._empty_statistics()
        statistics["token_usage_by_stage"] = {
            "analysis": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "guidance": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return statistics

    def _rebuild_statistics(self):
        super()._rebuild_statistics()
        by_stage = self.statistics["token_usage_by_stage"]
        for event in self._request_records.values():
            stage = str(event.get("stage"))
            usage = event.get("usage")
            if stage not in by_stage or not isinstance(usage, dict):
                continue
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    by_stage[stage][key] += int(usage[key])

    def _templates(self) -> tuple[str, str, str]:
        root = Path(__file__).resolve().parent / "prompts"
        return tuple(
            (root / name).read_text(encoding="utf-8")
            for name in (
                "llm_explorer_v3_system.txt",
                "llm_explorer_v3_analysis.txt",
                "llm_explorer_v3_guidance.txt",
            )
        )

    def _render_system_prompt(self, template, environment_constants):
        return render_prompt(
            template,
            {
                "VERIFIED_MECHANISMS_AND_FIELD_DEFINITIONS": environment_constants
            },
        )

    def _cache_identity(self, **kwargs):
        identity = super()._cache_identity(**kwargs)
        identity.update(
            schema_version=LLM_EXPLORER_V3_VERSION,
            summary_version=LLM_EXPLORER_V3_SUMMARY_VERSION,
            case_selection_version=LLM_EXPLORER_V3_CASE_VERSION,
            annealing={
                "end_episode": self.config.annealing_end_episode,
                "episode_joint_transitions": self.config.episode_joint_transitions,
                "warmup_joint_transitions": self.config.warmup_joint_transitions,
            },
        )
        if kwargs.get("stage") in {"guidance", "episode_result"}:
            identity["guidance_context_version"] = (
                LLM_EXPLORER_V3_GUIDANCE_CONTEXT_VERSION
            )
        return identity

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        original_cases = previous.get("representative_cases", [])
        original_cases = (
            original_cases if isinstance(original_cases, list) else []
        )
        movement_cases = [
            case
            for case in original_cases
            if isinstance(case, dict) and _case_domain(case) == "movement"
        ]
        routing_cases = [
            case
            for case in original_cases
            if isinstance(case, dict) and _case_domain(case) == "routing"
        ]
        cases = select_representative_cases(
            movement_cases,
            routing_cases,
            movement_maximum=3,
            routing_maximum=5,
        )
        raw_td3_bias = _recorded_raw_td3_bias(previous)
        return {
            "metrics": _json_clone(previous.get("metrics", {})),
            "movement_summaries": _compact_movement_summaries(
                previous.get("movement_summaries", {})
            ),
            "routing_summaries": _compact_routing_summaries(
                previous.get("routing_summaries", {})
            ),
            "representative_cases": _compact_cases(cases, "movement=3,routing=5"),
            "applied_guidance": _compact_applied_guidance(
                previous.get("applied_guidance", {}),
                previous.get("movement_summaries", {}),
                raw_td3_bias,
            ),
            "_previous_raw_td3_bias": raw_td3_bias,
            "_case_candidates": cases,
        }

    def _exploration_settings(self, *, sigma_episode_start, routing_epsilon):
        return {
            "upcoming_episode": getattr(self, "_upcoming_episode", None),
            "sigma_episode_start": float(sigma_episode_start),
            "routing_epsilon_at_episode_start": float(routing_epsilon),
            "guidance_weight_w": float(self._current_weight),
            "routing_alpha": float(self._current_alpha),
            "first_eligible_episode": first_eligible_episode(
                self.config.warmup_joint_transitions,
                self.config.episode_joint_transitions,
            ),
            "annealing_end_episode": self.config.annealing_end_episode,
            "td3_bias_bound": [
                -float(sigma_episode_start),
                float(sigma_episode_start),
            ],
            "guidance_fixed_for_episode": True,
            "host_scaling": "td3 effective=w*raw exactly once; routing alpha=0.8*w",
        }

    def _analysis_replacements(self, *, previous, environment_constants, current_observable_initial_state, settings):
        del environment_constants
        return {
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "EPISODE_METRICS": previous["metrics"],
            "MOVEMENT_SUMMARIES": previous["movement_summaries"],
            "ROUTING_SUMMARIES": previous["routing_summaries"],
            "REPRESENTATIVE_CASES": previous["representative_cases"],
            "PREVIOUS_GUIDANCE": previous["applied_guidance"],
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
        }

    def _render_analysis_prompt(self, *, system_prompt, analysis_template, previous, environment_constants, current_observable_initial_state, settings):
        cases = list(previous.get("_case_candidates", []))
        movement_cases = [
            case for case in cases if _case_domain(case) == "movement"
        ]
        routing_cases = [
            case for case in cases if _case_domain(case) == "routing"
        ]
        last_prompt = None
        smallest_hard_limit_prompt = None
        # Fixed finite compression order. Each step reselects for diversity;
        # it never takes a prefix or truncates rendered text/JSON.
        profiles = ((3, 5), (2, 4), (1, 2), (0, 1), (0, 0))
        for movement_limit, routing_limit in profiles:
            selected = select_representative_cases(
                movement_cases,
                routing_cases,
                movement_maximum=movement_limit,
                routing_maximum=routing_limit,
            )
            candidate = {
                key: value
                for key, value in previous.items()
                if key != "_case_candidates"
            }
            candidate["representative_cases"] = _compact_cases(
                selected,
                f"movement={movement_limit},routing={routing_limit}",
            )
            prompt = render_prompt(
                analysis_template,
                self._analysis_replacements(
                    previous=candidate,
                    environment_constants=environment_constants,
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
            if budget["input_tokens"] <= LLM_EXPLORER_V3_TARGET_INPUT_TOKENS:
                return prompt
        return smallest_hard_limit_prompt or last_prompt

    def _guidance_replacements(self, *, analysis, previous, current_observable_initial_state, settings, validation_feedback, **_):
        previous_guidance = _json_clone(previous["applied_guidance"])
        for key in (
            "td3_bias_location",
            "td3_raw_bias_rows",
            "td3_effective_bias_rows",
        ):
            previous_guidance.pop(key, None)
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "PREVIOUS_GUIDANCE": previous_guidance,
            "PREVIOUS_RAW_TD3_BIAS": previous["_previous_raw_td3_bias"],
            "OUTPUT_JSON_SCHEMA": output_schema_v3(),
            "VALIDATION_FEEDBACK_OR_EMPTY": validation_feedback,
        }

    def _parse_guidance_response(self, content, sigma_episode_start):
        return parse_guidance_response_v3(content, sigma_episode_start)

    def _guidance_cache_content(self, td3_bias, routing_prior):
        return _compact_json(
            {
                "td3_bias": np.asarray(td3_bias).tolist(),
                "routing_prior": routing_matrix_to_named(routing_prior),
            }
        )

    def _prompt_budget(self, **kwargs):
        budget = super()._prompt_budget(**kwargs)
        budget["input_hard_limit_tokens"] = int(
            self.config.input_hard_limit_tokens
        )
        budget["target_input_tokens"] = LLM_EXPLORER_V3_TARGET_INPUT_TOKENS
        budget["within_target_input_tokens"] = bool(
            int(budget["input_tokens"])
            <= LLM_EXPLORER_V3_TARGET_INPUT_TOKENS
        )
        budget["fits_input_hard_limit"] = bool(
            int(budget["input_tokens"])
            <= int(self.config.input_hard_limit_tokens)
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
                    reason=f"v3 token budget artifact: {type(exc).__name__}: {exc}",
                )
        if int(budget["input_tokens"]) > int(self.config.input_hard_limit_tokens):
            reason = (
                f"{kwargs['stage']} input requires {budget['input_tokens']} tokens; "
                f"v3 hard limit is {self.config.input_hard_limit_tokens}"
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

    @staticmethod
    def _disabled(reason: str, *, weight: float, alpha: float) -> dict[str, Any]:
        value = LLMExplorerController._unguided(reason)
        value.update(
            td3_fallback=False,
            routing_fallback=False,
            annealing_weight=float(weight),
            routing_alpha=float(alpha),
            td3_bias_raw=np.zeros((NUM_UAV, 3), dtype=np.float32),
            td3_bias_effective=np.zeros((NUM_UAV, 3), dtype=np.float32),
            routing_prior_named=None,
        )
        return value

    def begin_episode(self, *, episode_index, total_joint_transitions, sigma_episode_start, **kwargs):
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
        alpha = ROUTING_PRIOR_WEIGHT * weight
        self._current_weight = weight
        self._current_alpha = alpha
        if self.evaluation:
            self.active_guidance = self._disabled(
                "evaluation_disabled", weight=0.0, alpha=0.0
            )
            return self.active_guidance
        if episode >= self.config.annealing_end_episode:
            self.active_guidance = self._disabled(
                "annealing_finished", weight=0.0, alpha=0.0
            )
            return self.active_guidance
        if int(total_joint_transitions) < int(self.config.warmup_joint_transitions):
            self.active_guidance = self._disabled("warmup", weight=0.0, alpha=0.0)
            return self.active_guidance
        if weight <= 0.0:
            self.active_guidance = self._disabled(
                "annealing_finished", weight=0.0, alpha=0.0
            )
            return self.active_guidance
        result = super().begin_episode(
            episode_index=episode_index,
            total_joint_transitions=total_joint_transitions,
            sigma_episode_start=sigma_episode_start,
            **kwargs,
        )
        raw = np.asarray(result["td3_bias"], dtype=np.float32).reshape(NUM_UAV, 3)
        effective = (float(weight) * raw).astype(np.float32)
        result.update(
            annealing_weight=float(weight),
            routing_alpha=float(alpha),
            sigma_episode_start=float(sigma_episode_start),
            td3_bias_raw=raw,
            td3_bias_effective=effective,
            routing_prior_named=(
                None
                if result.get("routing_prior") is None
                else routing_matrix_to_named(result["routing_prior"])
            ),
        )
        self.active_guidance = result
        named_sender_count = len(result.get("routing_prior_named") or {})
        print(
            f"[LLM explorer v3] episode={episode} w={weight:.6f} alpha={alpha:.6f} "
            f"enabled={str(bool(result['enabled'])).lower()} "
            f"reason={result.get('td3_reason') or 'active'} "
            f"raw_bias_abs_max={float(np.max(np.abs(raw))):.6f} "
            f"effective_bias_abs_max={float(np.max(np.abs(effective))):.6f} "
            f"named_routing_senders={named_sender_count}",
            flush=True,
        )
        return result

    def complete_episode(self, *, episode_index: int, metrics: dict[str, Any]) -> None:
        movement_summary, movement_cases = movement_episode_summary(
            self.movement_records, self.active_guidance
        )
        routing_summary, routing_cases = routing_episode_summary(
            self.routing_records,
            maximum_cases=5,
            episode=int(episode_index) + 1,
        )
        cases = select_representative_cases(
            movement_cases,
            routing_cases,
            movement_maximum=3,
            routing_maximum=5,
        )
        samples = stratified_routing_samples(self.routing_records, maximum=100)
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
            "routing_alpha": float(self.active_guidance.get("routing_alpha", 0.0)),
            "guidance_enabled": bool(self.active_guidance.get("enabled", False)),
            "td3_fallback": bool(self.active_guidance.get("td3_fallback", False)),
            "routing_fallback": bool(
                self.active_guidance.get("routing_fallback", False)
            ),
            "disabled_reason": self.active_guidance.get("td3_reason"),
        }
        summary_feedback = {
            "schema_version": LLM_EXPLORER_V3_VERSION,
            "summary_version": LLM_EXPLORER_V3_SUMMARY_VERSION,
            "case_selection_version": LLM_EXPLORER_V3_CASE_VERSION,
            "metrics": verified_metrics,
            "movement_summaries": movement_summary,
            "routing_summaries": routing_summary,
            "representative_cases": cases,
            "case_semantics": "illustrative only; full summaries are authoritative for frequency",
            "routing_samples": samples,
            "applied_guidance": self._serializable_guidance(self.active_guidance),
        }
        self.previous_episode_feedback = _json_clone(summary_feedback)
        if self.root is not None:
            episode_root = self.root / f"episode_{episode_index + 1:06d}"
            _atomic_json(
                episode_root / "feedback.json",
                self._feedback_artifact(
                    summary_feedback, include_slot_service_records=False
                ),
            )
            _atomic_json(self.root / "state.json", self.state_dict())
            self._compact_success_artifacts(episode_root)
        print(
            f"[LLM explorer v3] episode={episode_index + 1} "
            f"w={self.active_guidance.get('annealing_weight', 0.0):.6f} "
            f"alpha={self.active_guidance.get('routing_alpha', 0.0):.6f} "
            f"enabled={str(bool(self.active_guidance.get('enabled', False))).lower()} "
            f"reason={self.active_guidance.get('td3_reason') or 'active'} "
            f"movement_decisions={len(self.movement_records)} "
            f"routing_decisions={len(self.routing_records)} cases={len(cases)} "
            f"requests_total={self.statistics['requests_total']} "
            f"tokens_total={self.statistics['token_usage']['total_tokens']} "
            f"analysis_tokens={self.statistics['token_usage_by_stage']['analysis']['total_tokens']} "
            f"guidance_tokens={self.statistics['token_usage_by_stage']['guidance']['total_tokens']} "
            f"analysis_wait_s={self.statistics['wait_seconds_by_stage']['analysis']:.3f} "
            f"guidance_wait_s={self.statistics['wait_seconds_by_stage']['guidance']:.3f} "
            f"failed_attempts={self.statistics['failed_attempts_total']} "
            f"fallback_episodes={self.statistics['fallback_episodes_total']}",
            flush=True,
        )


def explorer_metadata_v3(config: LLMExplorerV3Config) -> dict[str, Any]:
    return {
        "schema_version": LLM_EXPLORER_V3_VERSION,
        "prompt_version": config.prompt_version,
        "summary_version": LLM_EXPLORER_V3_SUMMARY_VERSION,
        "case_selection_version": LLM_EXPLORER_V3_CASE_VERSION,
        "guidance_context_version": LLM_EXPLORER_V3_GUIDANCE_CONTEXT_VERSION,
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "episode_joint_transitions": config.episode_joint_transitions,
        "first_eligible_episode": first_eligible_episode(
            config.warmup_joint_transitions, config.episode_joint_transitions
        ),
        "annealing_end_episode": config.annealing_end_episode,
        "target_input_tokens": LLM_EXPLORER_V3_TARGET_INPUT_TOKENS,
        "input_hard_limit_tokens": config.input_hard_limit_tokens,
        "routing_alpha": "0.8 * w",
        "td3_intervention": "episode-fixed w * raw bias exactly once before projection",
        "evaluation_calls_llm": False,
    }
