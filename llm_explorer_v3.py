"""Compact, annealed LLM exploration guidance for the isolated v3 method."""

from __future__ import annotations

import copy
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
    _semantic_guidance,
    environment_constants_and_definitions,
    prompt_token_budget,
    render_prompt,
    stratified_routing_samples,
    validate_td3_bias,
)


LLM_EXPLORER_V3_METHOD_ID = "td3_dinkelbach_llm_explorer_v3"
LLM_EXPLORER_V3_VERSION = "uav-hrl-llm-explorer-v3"
LLM_EXPLORER_V3_PROMPT_VERSION = "uav-hrl-llm-explorer-prompts-v3"
LLM_EXPLORER_V3_SUMMARY_VERSION = "verified-episode-summary-v1"
LLM_EXPLORER_V3_CASE_VERSION = "deterministic-representative-cases-v1"
LLM_EXPLORER_V3_ANNEAL_END_EPISODE = 600
LLM_EXPLORER_V3_MAX_INPUT_TOKENS = 20_000


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
                        and len(cases) < 8
                    ):
                        cases.append(
                            {
                                "kind": "movement_speed_saturation_with_positive_bias",
                                "episode": int(record.get("episode_index", -1)) + 1,
                                "time_s": record.get("time_seconds"),
                                "uav_id": uav_id,
                                "tasks": item.get("task_types", []),
                                "movement_mask": True,
                                "action_source": record.get("action_source"),
                                "actor_speed_mps": round(speed_mps, 4),
                                "raw_speed_bias": round(float(raw_bias[uav_id, 0]), 6),
                                "interpretation": "illustrative only; it does not prove a causal or optimal action",
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
    cases: list[dict[str, Any]] = []
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
            if (
                record.get("decision_branch") == "epsilon_exploration"
                and selected == sender
                and float(record.get("sender_total_backlog_bits", 0.0)) > 0.0
                and any(action not in (sender,) for action in np.flatnonzero(mask))
                and len(cases) < max(maximum_cases - 1, 0)
            ):
                distribution = record.get("exploration_distribution")
                wait_probability = (
                    None if distribution is None else float(distribution[sender])
                )
                if wait_probability is None or wait_probability >= 0.5:
                    cases.append(
                        {
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
                            "interpretation": "illustrative only; another legal action is not necessarily better than WAIT",
                        }
                    )
            if (
                bool(record.get("zero_legal_prior_mass_fallback", False))
                and len(cases) < max(maximum_cases - 1, 0)
            ):
                cases.append(
                    {
                        "kind": "masked_prior_zero_mass",
                        "episode": episode,
                        "time_s": record.get("time_seconds"),
                        "sender": f"UAV_{sender}",
                        "selected_action": routing_action_name(sender, selected),
                        "legal_actions": [
                            routing_action_name(sender, int(action))
                            for action in np.flatnonzero(mask)
                        ],
                        "interpretation": "the host used legal-uniform exploration for this decision",
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
    if len(cases) < maximum_cases:
        for record in records:
            sender = int(record["sender_uav_id"])
            if bool(record.get("zero_legal_prior_mass_fallback", False)):
                continue
            cases.append(
                {
                    "kind": "ordinary_recorded_routing_decision",
                    "episode": episode,
                    "time_s": record.get("time_seconds"),
                    "sender": f"UAV_{sender}",
                    "branch": record.get("decision_branch"),
                    "selected_action": routing_action_name(
                        sender, int(record["selected_action"])
                    ),
                    "interpretation": "illustrative only; no optimality claim",
                }
            )
            break
    return by_sender, cases[:maximum_cases]


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
    sender_properties = {
        f"UAV_{sender}": {
            "type": "object",
            "description": (
                "Sparse probabilities over GS, WAIT, and UAV_k except the sender's own UAV_k name; omitted actions are zero"
            ),
            "properties": {
                name: {"type": "number", "minimum": 0, "maximum": 1}
                for name in (
                    ["GS", "WAIT"]
                    + [
                        f"UAV_{target}"
                        for target in range(NUM_UAV)
                        if target != sender
                    ]
                )
            },
            "additionalProperties": False,
            "minProperties": 1,
        }
        for sender in range(NUM_UAV)
    }
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
                "required": list(sender_properties),
                "additionalProperties": False,
                "properties": sender_properties,
            },
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
        return identity

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        return {
            "metrics": _json_clone(previous.get("metrics", {})),
            "movement_summaries": _json_clone(previous.get("movement_summaries", {})),
            "routing_summaries": _json_clone(previous.get("routing_summaries", {})),
            "representative_cases": _json_clone(
                previous.get("representative_cases", [])
            ),
            "applied_guidance": _semantic_guidance(
                previous.get("applied_guidance", {})
            ),
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
        cases = list(previous.get("representative_cases", []))
        last_prompt = None
        candidates = []
        for keep in (min(8, len(cases)), 4, 2, 0):
            candidate = dict(previous)
            candidate["representative_cases"] = cases[: min(keep, len(cases))]
            candidates.append(candidate)
        condensed = copy.deepcopy(candidates[-1])
        for sender in condensed.get("routing_summaries", {}).values():
            for action_summary in sender.get(
                "legal_action_observable_summaries", {}
            ).values():
                for field, summary in list(action_summary.items()):
                    if isinstance(summary, dict):
                        action_summary[field] = {
                            "count": summary.get("count"),
                            "mean": summary.get("mean"),
                        }
        condensed["metrics"] = copy.deepcopy(condensed.get("metrics", {}))
        condensed["metrics"]["input_budget_reduction"] = (
            "representative cases removed; per-action observable ranges reduced to count and mean"
        )
        candidates.append(condensed)
        for candidate in candidates:
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
                return prompt
        return last_prompt

    def _guidance_replacements(self, *, analysis, previous, current_observable_initial_state, settings, validation_feedback, **_):
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "TRAINING_PHASE_AND_SETTINGS": settings,
            "PREVIOUS_GUIDANCE": previous["applied_guidance"],
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
            self.routing_records, episode=int(episode_index) + 1
        )
        cases = movement_cases[:3] + routing_cases[:5]
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
                {
                    **summary_feedback,
                    "movement_decisions": self.movement_records,
                    "routing_decisions": self.routing_records,
                },
            )
            _atomic_json(self.root / "state.json", self.state_dict())
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
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "episode_joint_transitions": config.episode_joint_transitions,
        "first_eligible_episode": first_eligible_episode(
            config.warmup_joint_transitions, config.episode_joint_transitions
        ),
        "annealing_end_episode": config.annealing_end_episode,
        "input_hard_limit_tokens": config.input_hard_limit_tokens,
        "routing_alpha": "0.8 * w",
        "td3_intervention": "episode-fixed w * raw bias exactly once before projection",
        "evaluation_calls_llm": False,
    }
