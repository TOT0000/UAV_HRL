"""Episode-level LLM exploration guidance for centralized TD3 and safe-DDQN.

This module is deliberately independent from the older ``llm_enabled``
state/reward-artifact path.  It never changes observations or rewards.  The
only products consumed by training are a fixed TD3 mean bias and a fixed
pre-mask safe-DDQN routing prior for one episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


LLM_EXPLORER_METHOD_ID = "td3_dinkelbach_llm_explorer"
LLM_EXPLORER_MODEL = "gpt-4o-mini"
LLM_EXPLORER_VERSION = "uav-hrl-llm-explorer-firstgen-v3"
LLM_EXPLORER_PROMPT_VERSION = "uav-hrl-llm-explorer-prompts-v5"
LLM_EXPLORER_SUMMARY_VERSION = "complete-core-feedback-v2"
LLM_EXPLORER_CASE_VERSION = "deterministic-diverse-cases-v1"
LLM_EXPLORER_CACHE_VERSION = "episode-fixed-dual-guidance-cache-v2"
LLM_EXPLORER_COMPRESSION_VERSION = LLM_EXPLORER_SUMMARY_VERSION
LLM_EXPLORER_WARMUP_TRANSITIONS = 10_000
LLM_EXPLORER_UPDATE_INTERVAL_EPISODES = 1
LLM_EXPLORER_MAX_ATTEMPTS = 2
# Retained for the separately versioned annealed v3 method.
ROUTING_PRIOR_WEIGHT = 0.8
FIRST_GEN_ROUTING_PRIOR_WEIGHT = 1.0
NUM_UAV = 16
ROUTING_ACTION_DIM = 17
LLM_EXPLORER_CONTEXT_TOKENS = 128_000
LLM_EXPLORER_SAFETY_MARGIN_TOKENS = 4_096
LLM_EXPLORER_ENCODING = "o200k_base"
WINDOWS_CONSERVATIVE_PATH_LIMIT = 240


class ExplorerValidationError(ValueError):
    """A response or persisted explorer artifact failed validation."""


class PromptBudgetError(ExplorerValidationError):
    """A complete prompt cannot fit in the configured model context."""


@dataclass(frozen=True)
class LLMExplorerConfig:
    model: str = LLM_EXPLORER_MODEL
    request_timeout_seconds: float = 60.0
    max_attempts: int = LLM_EXPLORER_MAX_ATTEMPTS
    update_interval_episodes: int = LLM_EXPLORER_UPDATE_INTERVAL_EPISODES
    warmup_joint_transitions: int = LLM_EXPLORER_WARMUP_TRANSITIONS
    analysis_max_output_tokens: int = 1400
    guidance_max_output_tokens: int = 5000
    temperature: float = 0.2
    prompt_version: str = LLM_EXPLORER_PROMPT_VERSION
    context_window_tokens: int = LLM_EXPLORER_CONTEXT_TOKENS
    safety_margin_tokens: int = LLM_EXPLORER_SAFETY_MARGIN_TOKENS
    tokenizer_encoding: str = LLM_EXPLORER_ENCODING

    def __post_init__(self) -> None:
        if self.model != LLM_EXPLORER_MODEL:
            # The model is configurable for explicit experiments, but never
            # silently substituted.  Empty values are therefore invalid.
            if not str(self.model).strip():
                raise ValueError("LLM explorer model must be non-empty")
        if not math.isfinite(float(self.request_timeout_seconds)) or float(
            self.request_timeout_seconds
        ) <= 0.0:
            raise ValueError("LLM explorer timeout must be finite and positive")
        if int(self.max_attempts) != 2:
            raise ValueError("LLM explorer permits exactly two attempts per stage")
        if int(self.update_interval_episodes) != 1:
            raise ValueError("LLM explorer update interval K must remain 1")
        if int(self.warmup_joint_transitions) != 10_000:
            raise ValueError("LLM explorer warmup threshold must remain 10,000")
        if int(self.context_window_tokens) != LLM_EXPLORER_CONTEXT_TOKENS:
            raise ValueError("gpt-4o-mini context window must remain 128,000 tokens")
        if int(self.safety_margin_tokens) < 1024:
            raise ValueError("LLM explorer token safety margin is too small")
        if str(self.tokenizer_encoding) != LLM_EXPLORER_ENCODING:
            raise ValueError("LLM explorer must use the o200k_base tokenizer")


def _json_clone(value: Any) -> Any:
    """Detach mutable simulator objects while enforcing JSON-safe values."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _token_encoder(encoding_name: str = LLM_EXPLORER_ENCODING):
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised by deployment preflight
        raise RuntimeError(
            "td3_dinkelbach_llm_explorer requires tiktoken; install "
            "requirements-llm-explorer.txt before training"
        ) from exc
    return tiktoken.get_encoding(str(encoding_name))


def prompt_token_budget(
    system_prompt: str,
    user_prompt: str,
    *,
    reserved_output_tokens: int,
    context_window_tokens: int = LLM_EXPLORER_CONTEXT_TOKENS,
    safety_margin_tokens: int = LLM_EXPLORER_SAFETY_MARGIN_TOKENS,
    encoding_name: str = LLM_EXPLORER_ENCODING,
) -> dict[str, Any]:
    """Count the exact message text with the model tokenizer plus chat overhead."""

    encoder = _token_encoder(encoding_name)
    system_tokens = len(encoder.encode(str(system_prompt)))
    user_tokens = len(encoder.encode(str(user_prompt)))
    # OpenAI chat framing is not exposed as text.  Three tokens per message and
    # three assistant-priming tokens is the documented conservative accounting
    # convention for current chat models; the separate 4K margin absorbs future
    # framing changes.
    message_overhead_tokens = 3 * 2 + 3
    total = (
        system_tokens
        + user_tokens
        + message_overhead_tokens
        + int(reserved_output_tokens)
        + int(safety_margin_tokens)
    )
    return {
        "compression_version": LLM_EXPLORER_COMPRESSION_VERSION,
        "tokenizer_encoding": str(encoding_name),
        "system_tokens": int(system_tokens),
        "user_tokens": int(user_tokens),
        "message_overhead_tokens": int(message_overhead_tokens),
        "input_tokens": int(system_tokens + user_tokens + message_overhead_tokens),
        "reserved_output_tokens": int(reserved_output_tokens),
        "safety_margin_tokens": int(safety_margin_tokens),
        "context_window_tokens": int(context_window_tokens),
        "total_budgeted_tokens": int(total),
        "fits": bool(total <= int(context_window_tokens)),
    }


def _round_optional(value: Any, digits: int) -> Any:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("LLM compact input contains a non-finite number")
    rounded = round(numeric, int(digits))
    return 0.0 if rounded == 0.0 else rounded


def _semantic_guidance(guidance: dict[str, Any]) -> dict[str, Any]:
    """Remove cache/execution provenance that must not perturb later prompts."""

    result = {
        key: value
        for key, value in guidance.items()
        if key not in {"reused", "cache_source", "request_ids"}
    }
    return LLMExplorerController._serializable_guidance(result)


def _compact_movement_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_source_codes = {
        "warmup_random": "W",
        "actor_plus_gaussian_exploration": "A",
        "actor_plus_llm_bias_plus_gaussian_exploration": "L",
        "random_method": "R",
        "evaluation_deterministic_policy": "E",
        "evaluation_policy_with_sampling_noise": "N",
    }
    task_bits = {"Search": 1, "FOV": 2, "COM": 4, "Hovering": 8}
    target_codes = {"roi": "R", "sr": "S"}
    decision_columns = [
        "episode_index", "interval_index", "time_s", "action_source",
        "behavior_noise_std", "known_roi_ids",
    ]
    uav_columns = [
        "decision_row", "uav_id", "position_x_m", "position_y_m", "position_z_m",
        "energy_j", "task_encoding", "observable_target_rows", "movement_mask",
        "actor_action_norm", "llm_bias_norm", "executed_action_norm",
    ]
    decision_rows: list[list[Any]] = []
    uav_rows: list[list[Any]] = []
    for decision_index, record in enumerate(records):
        decision_rows.append(
            [
                int(record.get("episode_index", -1)),
                int(record.get("interval_index", decision_index)),
                _round_optional(record.get("time_seconds"), 3),
                action_source_codes.get(
                    str(record.get("action_source", "unknown")),
                    str(record.get("action_source", "unknown")),
                ),
                _round_optional(record.get("behavior_noise_std"), 5),
                [int(value) for value in record.get("known_roi_ids_at_sample_time", [])],
            ]
        )
        for uav in record.get("uavs", []):
            targets = []
            for target in uav.get("observable_targets", []):
                targets.append(
                    [
                        target_codes.get(
                            str(target.get("kind", "unknown")),
                            str(target.get("kind", "unknown")),
                        ),
                        int(target["id"]),
                        *[
                            _round_optional(v, 1)
                            for v in target.get("relative_position_m", [])
                        ],
                    ]
                )
            def vec(name: str, digits: int) -> Any:
                value = uav.get(name)
                return None if value is None else [_round_optional(v, digits) for v in value]
            position = vec("position_m", 1)
            if position is not None and len(position) != 3:
                raise ValueError("movement position must contain exactly three values")
            raw_tasks = [str(value) for value in uav.get("task_types", [])]
            task_encoding = [
                sum(task_bits.get(value, 0) for value in raw_tasks),
                [value for value in raw_tasks if value not in task_bits],
            ]
            uav_rows.append(
                [
                    decision_index,
                    int(uav["uav_id"]),
                    *(position if position is not None else [None, None, None]),
                    _round_optional(uav.get("remaining_energy_j"), 1),
                    task_encoding,
                    targets,
                    bool(uav.get("movement_mask", False)),
                    vec("actor_normalized_action", 3),
                    vec("applied_llm_bias", 4),
                    vec("executed_normalized_action", 3),
                ]
            )
    return {
        "schema": {
            "decision_columns": decision_columns,
            "uav_columns": uav_columns,
            "units": {
                "time_s": "seconds",
                "position_m_xyz": "metres",
                "energy_j": "joules",
                "*_action_norm": "normalized action units",
                "null": "unknown/not-applicable; numeric 0 remains a valid observed zero",
            },
            "action_source_codes": action_source_codes,
            "task_encoding": {
                "known_task_bitmask": task_bits,
                "value": "[known_bitmask, unknown_task_names]; [0,[]] means observed unassigned",
            },
            "target_kind_codes": target_codes,
        },
        "decision_rows": decision_rows,
        "uav_rows": uav_rows,
    }


def _routing_state_named_values(state: Any) -> dict[str, float]:
    from observation_strategy import routing_state_feature_names

    values = np.asarray(state, dtype=np.float64).reshape(-1)
    names = routing_state_feature_names()
    if values.shape != (len(names),):
        raise ValueError(
            f"routing state has shape {values.shape}; expected ({len(names)},)"
        )
    if not np.isfinite(values).all():
        raise ValueError("routing state contains a non-finite value")
    return dict(zip(names, values.tolist()))


def compact_routing_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert the authoritative 143-D state into a compact named table."""

    named = _routing_state_named_values(record["compact_routing_state"])
    sender = int(record["sender_uav_id"])
    masks = [int(v) for v in record["legal_action_mask"]]
    if len(masks) != ROUTING_ACTION_DIM:
        raise ValueError("routing legal mask must contain 17 entries")
    if any(value not in (0, 1) for value in masks):
        raise ValueError("routing legal mask must contain only 0/1 values")
    legal_actions = [action for action in range(ROUTING_ACTION_DIM) if masks[action]]
    action_feature_columns = [
        legal_actions,
        [_round_optional(named[f"link_capacity[{action}]"], 3) for action in legal_actions],
        [_round_optional(named[f"next_hop_backlog[{action}]"], 3) for action in legal_actions],
        [_round_optional(named[f"gs_progress_norm[{action}]"], 3) for action in legal_actions],
        [bool(round(named[f"next_hop_assigned_fov[{action}]" ])) for action in legal_actions],
    ]
    sender_row = [
        _round_optional(named["energy"], 5),
        _round_optional(named["backlog"], 5),
        [
            bool(round(named[f"assigned_task_{task}"]))
            for task in ("search", "fov", "com", "hovering")
        ],
        bool(round(named["assigned_source_uav"])),
        [
            _round_optional(named["position_x"], 2),
            _round_optional(named["position_y"], 2),
            _round_optional(named["position_z"], 5),
        ],
        _round_optional(named["distance_to_ground_station"], 5),
        _round_optional(named["eta_to_ground_station"], 5),
        _round_optional(named["dinkelbach_lambda"], 5),
        [
            _round_optional(named["hol_assigned_task_fov"], 3),
            _round_optional(named["hol_assigned_task_com"], 3),
            _round_optional(named["hol_deadline_remaining"], 5),
            _round_optional(named["hol_packet_fraction_remaining"], 5),
        ],
    ]
    return {
        "decision": [
            _round_optional(record.get("time_seconds"), 3),
            sender,
            int(record["selected_action"]),
            str(record.get("decision_branch", "unknown")),
            _round_optional(record.get("epsilon"), 6),
            _round_optional(record.get("sender_total_backlog_bits"), 1),
            int(record["hol_packet_id"]) if record.get("hol_packet_id") is not None else None,
            str(record.get("hol_task_type")) if record.get("hol_task_type") is not None else None,
            _round_optional(record.get("hol_remaining_bits"), 1),
            _round_optional(record.get("hol_remaining_deadline_seconds"), 4),
            bool(record.get("zero_legal_prior_mass_fallback", False)),
        ],
        "sender": sender_row,
        "legal_action_indices": legal_actions,
        "legal_action_features": action_feature_columns,
    }


def _compact_routing_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": {
            "decision_columns": [
                "time_s", "sender_uav_id", "selected_action", "decision_branch",
                "epsilon", "sender_backlog_bits", "hol_packet_id", "hol_task_type",
                "hol_remaining_bits", "hol_deadline_s", "zero_prior_mass_fallback",
            ],
            "sender_columns": [
                "energy_norm", "backlog_norm", "task_flags_search_fov_com_hover",
                "is_source", "position_x_m_y_m_z_norm", "distance_to_gs_norm",
                "eta_to_gs_norm", "dinkelbach_lambda_norm",
                "hol_flags_fov_com_deadline_norm_fraction_remaining",
            ],
            "legal_action_feature_arrays": [
                "action_indices", "capacity_norm",
                "next_hop_backlog_log_norm", "gs_progress_norm", "next_hop_is_fov",
            ],
            "actions": "0..15 are UAV next hops; 16 is GS; sender index is WAIT",
            "feature_alignment": "legal_action_indices is the lossless sparse form of the 17-entry mask; parallel feature arrays use the same order",
            "normalization": {
                "capacity_norm": "nominal link Mbps / action-specific reference Mbps, clipped [0,1]",
                "next_hop_backlog_log_norm": "log1p(backlog_bits)/log1p(B_MAX)",
                "gs_progress_norm": "action-wise normalized geometric progress toward GS",
                "hol_deadline_norm": "remaining deadline / task deadline, clipped [0,1]",
                "null": "unknown/not-applicable; numeric 0 means observed valid zero",
            },
        },
        "rows": [compact_routing_record(record) for record in records],
    }


def compact_feedback_for_llm(previous: dict[str, Any]) -> dict[str, Any]:
    """Create an LLM-only copy without changing stored training diagnostics."""

    if previous.get("summary_version") == LLM_EXPLORER_SUMMARY_VERSION:
        return {
            "compression_version": LLM_EXPLORER_COMPRESSION_VERSION,
            "episode_outcomes": _json_clone(previous.get("episode_outcomes", {})),
            "previous_complete_guidance": _json_clone(
                previous.get("previous_complete_guidance", {})
            ),
            "previous_application_status": _json_clone(
                previous.get("previous_application_status", {})
            ),
            "uav_core_summary": _json_clone(previous.get("uav_core_summary", {})),
            "routing_core_summary": _json_clone(
                previous.get("routing_core_summary", {})
            ),
            "actual_slot_service_summary": _json_clone(
                previous.get("actual_slot_service_summary", {})
            ),
            "representative_cases": _json_clone(
                previous.get("representative_cases", [])
            ),
        }
    return {
        "compression_version": LLM_EXPLORER_COMPRESSION_VERSION,
        "metrics": _json_clone(previous.get("metrics", {})),
        "movement": _compact_movement_records(previous.get("movement_decisions", [])),
        "routing": _compact_routing_records(previous.get("routing_samples", [])),
        "full_episode_statistics": _json_clone(
            previous.get("full_episode_statistics", {})
        ),
        "applied_guidance": _semantic_guidance(
            previous.get("applied_guidance", {})
        ),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _propagate_wrapped_keyboard_interrupt(exc: BaseException) -> None:
    """Do not let transport adapters turn user cancellation into a retry."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, KeyboardInterrupt):
            raise current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def masked_routing_exploration_distribution(
    prior: Any,
    legal_mask: Any,
    *,
    prior_weight: float = ROUTING_PRIOR_WEIGHT,
) -> tuple[np.ndarray, bool]:
    """Mask/normalize a routing prior, then mix it with legal uniform mass.

    Returns ``(distribution, used_zero_legal_mass_fallback)``.  This helper
    performs no RNG draws and is safe to use only after the epsilon branch has
    already been selected.
    """

    weights = np.asarray(prior, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=bool)
    if weights.shape != (ROUTING_ACTION_DIM,):
        raise ValueError(
            f"routing prior row must have shape ({ROUTING_ACTION_DIM},)"
        )
    if mask.shape != weights.shape:
        raise ValueError("routing legal mask and prior shapes differ")
    if not mask.any():
        raise ValueError("routing action mask has no legal action")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("routing prior must contain finite non-negative values")
    if not math.isfinite(float(prior_weight)) or not 0.0 <= float(
        prior_weight
    ) <= 1.0:
        raise ValueError("routing prior mixture weight must be in [0,1]")

    legal_indices = np.flatnonzero(mask)
    uniform = np.zeros_like(weights)
    uniform[legal_indices] = 1.0 / float(legal_indices.size)
    masked = np.where(mask, weights, 0.0)
    legal_mass = float(masked.sum())
    zero_mass = not math.isfinite(legal_mass) or legal_mass <= 0.0
    llm_distribution = uniform if zero_mass else masked / legal_mass
    mixed = float(prior_weight) * llm_distribution + (
        1.0 - float(prior_weight)
    ) * uniform
    mixed[~mask] = 0.0
    mixed /= float(mixed.sum())
    return mixed, zero_mass


def validate_td3_bias(value: Any, sigma_episode_start: float) -> np.ndarray:
    if _contains_boolean(value):
        raise ExplorerValidationError("td3_bias must contain JSON numbers, not booleans")
    bias = np.asarray(value, dtype=np.float64)
    sigma = float(sigma_episode_start)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ExplorerValidationError("sigma_episode_start must be finite and non-negative")
    if bias.shape != (NUM_UAV, 3):
        raise ExplorerValidationError("td3_bias must have shape [16,3]")
    if not np.isfinite(bias).all():
        raise ExplorerValidationError("td3_bias contains NaN or Infinity")
    tolerance = max(
        1e-7,
        4.0 * np.finfo(np.float32).eps * max(1.0, abs(sigma)),
    )
    if np.any(np.abs(bias) > sigma + tolerance):
        raise ExplorerValidationError(
            "td3_bias exceeds the episode-start behavior-noise sigma bound"
        )
    return bias.astype(np.float32)


def validate_routing_prior(value: Any) -> np.ndarray:
    if _contains_boolean(value):
        raise ExplorerValidationError(
            "routing_prior must contain JSON numbers, not booleans"
        )
    prior = np.asarray(value, dtype=np.float64)
    if prior.shape != (NUM_UAV, ROUTING_ACTION_DIM):
        raise ExplorerValidationError("routing_prior must have shape [16,17]")
    if not np.isfinite(prior).all():
        raise ExplorerValidationError("routing_prior contains NaN or Infinity")
    if np.any(prior < 0.0) or np.any(prior > 1.0):
        raise ExplorerValidationError("routing_prior probabilities must be in [0,1]")
    sums = prior.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        raise ExplorerValidationError("each routing_prior row must sum to 1")
    # Normalize only harmless floating-point row-sum drift accepted above.
    return prior / sums[:, None]


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def parse_guidance_response(
    content: str, sigma_episode_start: float
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, str]]:
    """Strict JSON parse with independent TD3 and routing validation."""

    errors: dict[str, str] = {}
    try:
        value = json.loads(str(content))
    except (TypeError, json.JSONDecodeError) as exc:
        reason = f"invalid JSON: {type(exc).__name__}: {exc}"
        return None, None, {"td3": reason, "routing": reason}
    if not isinstance(value, dict):
        return None, None, {
            "td3": "response root is not an object",
            "routing": "response root is not an object",
        }
    extra = sorted(set(value).difference({"td3_bias", "routing_prior"}))
    if extra:
        reason = f"unexpected response fields: {extra}"
        return None, None, {"td3": reason, "routing": reason}
    try:
        td3_bias = validate_td3_bias(value.get("td3_bias"), sigma_episode_start)
    except (ExplorerValidationError, TypeError, ValueError) as exc:
        td3_bias = None
        errors["td3"] = str(exc)
    try:
        routing_prior = validate_routing_prior(value.get("routing_prior"))
    except (ExplorerValidationError, TypeError, ValueError) as exc:
        routing_prior = None
        errors["routing"] = str(exc)
    return td3_bias, routing_prior, errors


def stratified_routing_samples(
    records: list[dict[str, Any]], maximum: int = 100
) -> list[dict[str, Any]]:
    """Allocate sender quotas, then sample each sender uniformly over time."""

    maximum = int(maximum)
    if maximum <= 0 or not records:
        return []
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        sender = int(record["sender_uav_id"])
        groups.setdefault(sender, []).append(_json_clone(record))
    for sender in groups:
        groups[sender].sort(
            key=lambda item: (
                float(item.get("time_seconds", 0.0)),
                int(item.get("decision_index", 0)),
            )
        )
    senders = sorted(groups)
    capacity = min(maximum, len(records))
    quotas = {sender: 0 for sender in senders}
    # Give every actual sender representation before redistributing shortages.
    for sender in senders[:capacity]:
        quotas[sender] = 1
    remaining = capacity - sum(quotas.values())
    while remaining:
        progressed = False
        for sender in senders:
            if quotas[sender] < len(groups[sender]):
                quotas[sender] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    selected: list[dict[str, Any]] = []
    for sender in senders:
        values = groups[sender]
        quota = quotas[sender]
        if quota == 0:
            continue
        indices = (
            np.arange(len(values), dtype=int)
            if quota >= len(values)
            else np.rint(np.linspace(0, len(values) - 1, quota)).astype(int)
        )
        selected.extend(values[int(index)] for index in indices)
    selected.sort(
        key=lambda item: (
            float(item.get("time_seconds", 0.0)),
            int(item.get("sender_uav_id", -1)),
            int(item.get("decision_index", 0)),
        )
    )
    return selected[:capacity]


def routing_action_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sender: dict[str, Any] = {}
    exploration = greedy = 0
    for record in records:
        sender = str(int(record["sender_uav_id"]))
        action = str(int(record["selected_action"]))
        entry = by_sender.setdefault(
            sender,
            {"decision_count": 0, "action_counts": {}, "wait_count": 0},
        )
        entry["decision_count"] += 1
        entry["action_counts"][action] = entry["action_counts"].get(action, 0) + 1
        if int(action) == int(sender):
            entry["wait_count"] += 1
        if record.get("decision_branch") == "epsilon_exploration":
            exploration += 1
        elif record.get("decision_branch") == "safe_q_greedy":
            greedy += 1
    return {
        "total_real_decisions": len(records),
        "epsilon_exploration_decisions": exploration,
        "safe_q_greedy_decisions": greedy,
        "by_sender": by_sender,
    }


def _numeric_summary(values: list[float], digits: int = 6) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values], dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"valid_sample_count": 0, "mean": None, "min": None, "max": None}
    return {
        "valid_sample_count": int(clean.size),
        "mean": round(float(clean.mean()), digits),
        "min": round(float(clean.min()), digits),
        "max": round(float(clean.max()), digits),
    }


def _component_summary(rows: list[list[float]]) -> dict[str, Any]:
    columns = ("speed", "heading", "vertical")
    if not rows:
        return {name: _numeric_summary([]) for name in columns}
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return {
        name: _numeric_summary(values[:, index].tolist())
        for index, name in enumerate(columns)
    }


def _circular_summary(radians: list[float]) -> dict[str, Any]:
    clean = np.asarray(radians, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "valid_sample_count": 0,
            "circular_mean_rad": None,
            "resultant_length": None,
            "circular_variance": None,
            "mean_direction_status": "unknown_no_valid_samples",
        }
    resultant = np.mean(np.exp(1j * clean))
    length = float(abs(resultant))
    representative = length >= 0.1
    return {
        "valid_sample_count": int(clean.size),
        "circular_mean_rad": (
            round(float(np.angle(resultant)), 6) if representative else None
        ),
        "resultant_length": round(length, 6),
        "circular_variance": round(1.0 - length, 6),
        "mean_direction_status": (
            "representative" if representative else "dispersed_mean_not_representative"
        ),
    }


def _select_representative_cases(
    cases: list[dict[str, Any]], maximum: int = 8
) -> list[dict[str, Any]]:
    """Choose deterministic domain/kind/UAV/time-diverse real observations."""

    candidates = [_json_clone(case) for case in cases if isinstance(case, dict)]
    if not candidates or int(maximum) <= 0:
        return []
    times = [
        float(case["time_s"])
        for case in candidates
        if isinstance(case.get("time_s"), (int, float))
        and math.isfinite(float(case["time_s"]))
    ]
    low = min(times) if times else None
    high = max(times) if times else None

    def segment(case: Mapping[str, Any]) -> int | None:
        if low is None or high is None or not isinstance(
            case.get("time_s"), (int, float)
        ):
            return None
        value = float(case["time_s"])
        if not math.isfinite(value):
            return None
        return 0 if high == low else min(2, int(3.0 * (value - low) / (high - low)))

    def entity(case: Mapping[str, Any]) -> str:
        value = case.get("uav_id", case.get("sender_uav_id", "unknown"))
        return str(value)

    def canonical(index: int) -> tuple[Any, ...]:
        case = candidates[index]
        return (
            str(case.get("domain", "")),
            str(case.get("kind", "")),
            entity(case),
            float(case.get("time_s", math.inf)),
            _canonical_json(case),
        )

    remaining = set(range(len(candidates)))
    selected: list[int] = []
    seen_domains: set[str] = set()
    seen_kinds: set[str] = set()
    seen_entities: set[str] = set()
    seen_segments: set[int] = set()
    while remaining and len(selected) < int(maximum):
        def rank(index: int) -> tuple[int, ...]:
            case = candidates[index]
            time_segment = segment(case)
            return (
                int(str(case.get("domain")) not in seen_domains),
                int(str(case.get("kind")) not in seen_kinds),
                int(entity(case) not in seen_entities),
                int(time_segment is not None and time_segment not in seen_segments),
            )

        best = max(rank(index) for index in remaining)
        chosen = min(
            (index for index in remaining if rank(index) == best), key=canonical
        )
        selected.append(chosen)
        remaining.remove(chosen)
        case = candidates[chosen]
        seen_domains.add(str(case.get("domain")))
        seen_kinds.add(str(case.get("kind")))
        seen_entities.add(entity(case))
        time_segment = segment(case)
        if time_segment is not None:
            seen_segments.add(time_segment)
    return [candidates[index] for index in selected]


def first_gen_movement_summary(
    records: list[dict[str, Any]], guidance: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize every UAV and retain directional service evidence."""

    requested = None
    if bool(guidance.get("enabled")) and not bool(guidance.get("td3_fallback", True)):
        value = np.asarray(guidance.get("td3_bias"), dtype=np.float64)
        if value.shape == (NUM_UAV, 3) and np.isfinite(value).all():
            requested = value
    by_uav: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    totals = Counter()
    for uav_id in range(NUM_UAV):
        roles: Counter[str] = Counter()
        actor_rows: list[list[float]] = []
        projected_rows: list[list[float]] = []
        executed_rows: list[list[float]] = []
        applied_rows: list[list[float]] = []
        energy_all: list[float] = []
        energy_td3: list[float] = []
        energy_other: list[float] = []
        target_distances: dict[str, list[float]] = {"FOV": [], "COM": []}
        target_bearings: dict[str, list[float]] = {"FOV": [], "COM": []}
        queue_backlog: list[float] = []
        hol_deadlines: list[float] = []
        coverage: list[float] = []
        image_quantity: list[float] = []
        sr_backlog: list[float] = []
        sr_hol_deadlines: list[float] = []
        s2u_capacity: list[float] = []
        counts = Counter()
        for record in records:
            rows = record.get("uavs", [])
            if uav_id >= len(rows) or not isinstance(rows[uav_id], Mapping):
                counts["missing_uav_record"] += 1
                continue
            item = rows[uav_id]
            tasks = [str(task) for task in item.get("task_types", [])]
            roles.update(tasks or ["unassigned"])
            controlled = bool(item.get("movement_mask", False))
            counts["recorded_decisions"] += 1
            counts["td3_controlled_decisions"] += int(controlled)
            counts["other_controller_decisions"] += int(not controlled)
            for field, destination in (
                ("actor_normalized_action", actor_rows),
                ("policy_projected_normalized_action", projected_rows),
                ("executed_normalized_action", executed_rows),
                ("applied_llm_bias", applied_rows),
            ):
                value = item.get(field)
                if value is not None:
                    array = np.asarray(value, dtype=np.float64)
                    if array.shape == (3,) and np.isfinite(array).all():
                        destination.append(array.tolist())
            pre = item.get("pre_projection_normalized_action")
            projected = item.get("policy_projected_normalized_action")
            if pre is not None and projected is not None:
                pre_array = np.asarray(pre, dtype=np.float64)
                projected_array = np.asarray(projected, dtype=np.float64)
                if pre_array.shape == projected_array.shape == (3,):
                    counts["host_action_projection_decisions"] += int(
                        not np.allclose(pre_array, projected_array, atol=1e-8)
                    )
            final = item.get("executed_normalized_action")
            if projected is not None and final is not None:
                projected_array = np.asarray(projected, dtype=np.float64)
                final_array = np.asarray(final, dtype=np.float64)
                if projected_array.shape == final_array.shape == (3,):
                    counts["environment_adjusted_decisions"] += int(
                        not np.allclose(projected_array, final_array, atol=1e-8)
                    )
            if item.get("movement_energy_j") is not None:
                value = float(item["movement_energy_j"])
                if math.isfinite(value):
                    energy_all.append(value)
                    (energy_td3 if controlled else energy_other).append(value)
            for target in item.get("observable_targets", []):
                relative = np.asarray(target.get("relative_position_m", []), dtype=float)
                if relative.shape != (3,) or not np.isfinite(relative).all():
                    continue
                kind = "FOV" if str(target.get("kind")) == "roi" else "COM"
                target_distances[kind].append(float(np.linalg.norm(relative)))
                target_bearings[kind].append(float(math.atan2(relative[1], relative[0])))
            observation = item.get("service_observation")
            if isinstance(observation, Mapping) and observation.get("snapshot_available"):
                counts["service_snapshot_available_decisions"] += 1
                queue = observation.get("uav_queue")
                if isinstance(queue, Mapping) and queue.get("valid"):
                    counts["queue_valid_decisions"] += 1
                    queue_backlog.append(float(queue.get("backlog_bits", 0.0)))
                    counts["queue_empty_decisions"] += int(bool(queue.get("empty")))
                    if queue.get("hol_present"):
                        counts["hol_present_decisions"] += 1
                        if queue.get("hol_remaining_deadline_s") is not None:
                            hol_deadlines.append(float(queue["hol_remaining_deadline_s"]))
                    else:
                        counts["no_hol_decisions"] += 1
                fov = observation.get("fov")
                if isinstance(fov, Mapping) and fov.get("pair_valid"):
                    counts["fov_geometry_valid_decisions"] += 1
                    coverage.append(float(fov.get("coverage_ratio", 0.0)))
                    image_quantity.append(float(fov.get("image_quantity", 0.0)))
                    counts["fov_capture_valid_decisions"] += int(
                        bool(fov.get("capture_valid"))
                    )
                com = observation.get("com")
                if isinstance(com, Mapping) and com.get("target_valid"):
                    counts["com_target_valid_decisions"] += 1
                    if com.get("sr_queue_valid"):
                        counts["com_sr_queue_valid_decisions"] += 1
                        sr_backlog.append(float(com.get("sr_backlog_bits", 0.0)))
                        counts["com_sr_queue_empty_decisions"] += int(
                            bool(com.get("sr_queue_empty"))
                        )
                        if com.get("sr_hol_remaining_deadline_s") is not None:
                            sr_hol_deadlines.append(
                                float(com["sr_hol_remaining_deadline_s"])
                            )
                    if com.get("s2u_link_valid"):
                        counts["s2u_link_valid_decisions"] += 1
                        counts["s2u_in_range_decisions"] += int(
                            bool(com.get("s2u_in_range"))
                        )
                        s2u_capacity.append(
                            float(com.get("s2u_reference_capacity_mbps", 0.0))
                        )
            role = (
                "dual_fov_com"
                if {"FOV", "COM"}.issubset(set(tasks))
                else "fov"
                if "FOV" in tasks
                else "com"
                if "COM" in tasks
                else "other"
            )
            cases.append(
                {
                    "domain": "movement",
                    "kind": f"movement_{role}_{'td3' if controlled else 'other_controller'}",
                    "time_s": record.get("time_seconds"),
                    "uav_id": uav_id,
                    "roles": tasks,
                    "movement_mask": controlled,
                    "actor_action": item.get("actor_normalized_action"),
                    "applied_bias": item.get("applied_llm_bias"),
                    "executed_action": item.get("executed_normalized_action"),
                    "observable_targets": item.get("observable_targets", []),
                    "queue_and_sensing": observation,
                }
            )
        totals.update(counts)
        by_uav[f"UAV_{uav_id}"] = {
            "role_decision_counts": dict(sorted(roles.items())),
            "recorded_decision_count": int(counts["recorded_decisions"]),
            "td3_controlled_decision_count": int(counts["td3_controlled_decisions"]),
            "other_controller_decision_count": int(counts["other_controller_decisions"]),
            "actor_action_normalized": _component_summary(actor_rows),
            "actor_heading_circular_rad": _circular_summary(
                [math.pi * row[1] for row in actor_rows]
            ),
            "policy_projected_action_normalized": _component_summary(projected_rows),
            "actual_executed_action_normalized": _component_summary(executed_rows),
            "actual_executed_heading_circular_rad": _circular_summary(
                [math.pi * row[1] for row in executed_rows]
            ),
            "host_action_projection_decision_count": int(counts["host_action_projection_decisions"]),
            "environment_adjusted_decision_count": int(counts["environment_adjusted_decisions"]),
            "requested_episode_fixed_raw_bias_normalized": (
                None if requested is None else requested[uav_id].round(7).tolist()
            ),
            "actually_added_bias_normalized": _component_summary(applied_rows),
            "bias_application_count": len(applied_rows),
            "guidance_fallback": bool(guidance.get("td3_fallback", True)),
            "movement_energy_j": {
                "all_controllers": _numeric_summary(energy_all, 4),
                "td3_controlled_intervals": _numeric_summary(energy_td3, 4),
                "other_controller_intervals": _numeric_summary(energy_other, 4),
            },
            "service_target_geometry": {
                kind: {
                    "distance_3d_m": _numeric_summary(target_distances[kind], 4),
                    "bearing_xy_rad": _circular_summary(target_bearings[kind]),
                }
                for kind in ("FOV", "COM")
            },
            "uav_queue": {
                "valid_observation_count": int(counts["queue_valid_decisions"]),
                "empty_observation_count": int(counts["queue_empty_decisions"]),
                "backlog_bits": _numeric_summary(queue_backlog, 2),
                "hol_present_count": int(counts["hol_present_decisions"]),
                "no_hol_count": int(counts["no_hol_decisions"]),
                "hol_remaining_deadline_s": _numeric_summary(hol_deadlines, 6),
            },
            "fov_sensing": {
                "valid_geometry_observation_count": int(counts["fov_geometry_valid_decisions"]),
                "capture_valid_count": int(counts["fov_capture_valid_decisions"]),
                "coverage_ratio": _numeric_summary(coverage),
                "image_quantity": _numeric_summary(image_quantity),
            },
            "com_access": {
                "target_valid_observation_count": int(
                    counts["com_target_valid_decisions"]
                ),
                "sr_queue_valid_observation_count": int(
                    counts["com_sr_queue_valid_decisions"]
                ),
                "sr_queue_empty_observation_count": int(
                    counts["com_sr_queue_empty_decisions"]
                ),
                "sr_backlog_bits": _numeric_summary(sr_backlog, 2),
                "sr_hol_remaining_deadline_s": _numeric_summary(
                    sr_hol_deadlines, 6
                ),
                "s2u_link_valid_observation_count": int(
                    counts["s2u_link_valid_decisions"]
                ),
                "s2u_in_range_observation_count": int(
                    counts["s2u_in_range_decisions"]
                ),
                "s2u_reference_capacity_mbps": _numeric_summary(
                    s2u_capacity, 6
                ),
            },
        }
    return {
        "scope": "complete episode; every recorded one-second movement boundary; all 16 UAVs",
        "units_and_validity": {
            "actions_and_bias": "normalized action units",
            "direction": "radians with circular statistics; null mean can indicate dispersed evidence",
            "energy": "joules; all-controller and TD3-only scopes are separate",
            "unknown": "null with zero valid_sample_count; observed numeric zero remains zero",
        },
        "totals": dict(totals),
        "by_uav": by_uav,
    }, _select_representative_cases(cases, maximum=4)


def first_gen_slot_service_record(
    slot_result: Mapping[str, Any],
    *,
    pre_slot_violations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    current_time: float,
    slot_seconds: float,
    preselected_actions_by_sender: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Retain authoritative positive-bit service and endpoint completions."""

    from llm_state_rule_explorer import state_rule_slot_service_record

    record = state_rule_slot_service_record(
        slot_result,
        pre_slot_violations=pre_slot_violations,
        current_time=current_time,
    )
    completed_hops: list[dict[str, Any]] = []
    for packet in slot_result.get("s2u_arrivals", ()):
        if not isinstance(packet, Mapping):
            continue
        hops = packet.get("per_hop", [])
        hop = hops[-1] if hops and isinstance(hops[-1], Mapping) else {}
        completed_hops.append(
            {
                "link_type": "S2U",
                "sender_id": hop.get("from"),
                "receiver_id": hop.get("to"),
                "packet_id": packet.get("id"),
            }
        )
    for arrival in slot_result.get("relay_arrivals", ()):
        if isinstance(arrival, Mapping):
            completed_hops.append(
                {
                    "link_type": "U2U",
                    "sender_id": arrival.get("sender"),
                    "receiver_id": arrival.get("receiver"),
                    "packet_id": arrival.get("packet_id"),
                }
            )
    delivered_packet_ids: set[int] = set()
    for outcome in slot_result.get("outcomes", ()):
        if not isinstance(outcome, Mapping) or not isinstance(
            outcome.get("packet"), Mapping
        ):
            continue
        packet = outcome["packet"]
        terminal = str(packet.get("terminal_outcome", ""))
        if terminal not in {"on_time_delivered", "late_delivered"}:
            continue
        packet_id = int(outcome.get("packet_id", packet.get("id")))
        if packet_id in delivered_packet_ids:
            continue
        delivered_packet_ids.add(packet_id)
        hops = packet.get("per_hop", [])
        hop = hops[-1] if hops and isinstance(hops[-1], Mapping) else {}
        completed_hops.append(
            {
                "link_type": "U2G",
                "sender_id": hop.get("from", outcome.get("attributed_sender")),
                "receiver_id": hop.get("to", 16),
                "packet_id": packet_id,
                "terminal_outcome": terminal,
            }
        )
    record.update(
        slot_seconds=float(slot_seconds),
        completed_hops=completed_hops,
        packet_decision_uses=[
            {
                "packet_id": int(item["packet_id"]),
                "sender_uav_id": int(item["sender_uav_id"]),
                "selected_action": (
                    int(
                        preselected_actions_by_sender[
                            int(item["sender_uav_id"])
                        ]
                    )
                    if preselected_actions_by_sender is not None
                    and int(item["sender_uav_id"])
                    in preselected_actions_by_sender
                    else None
                ),
            }
            for item in slot_result.get("packet_decision_uses", ())
            if isinstance(item, Mapping)
            and item.get("packet_id") is not None
            and item.get("sender_uav_id") is not None
        ],
        global_routing_reward=float(slot_result.get("global_routing_reward", 0.0)),
        shared_reward_row_sum=float(slot_result.get("shared_reward_row_sum", 0.0)),
    )
    return _json_clone(record)


def first_gen_routing_summary(
    records: list[dict[str, Any]],
    slot_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize all senders, branches, legal opportunities, and service."""

    by_uav_service: dict[int, dict[int, Counter[str]]] = {
        sender: {} for sender in range(NUM_UAV)
    }
    access_received: dict[int, Counter[str]] = {
        sender: Counter() for sender in range(NUM_UAV)
    }
    packet_use_counts: Counter[int] = Counter()
    packet_use_counts_by_action: dict[int, Counter[int]] = {
        sender: Counter() for sender in range(NUM_UAV)
    }
    service_cases: list[dict[str, Any]] = []
    slot_seconds_values: list[float] = []
    for slot in slot_records:
        slot_seconds = float(slot.get("slot_seconds", 0.25))
        slot_seconds_values.append(slot_seconds)
        for use in slot.get("packet_decision_uses", []):
            if isinstance(use, Mapping):
                sender = int(use["sender_uav_id"])
                packet_use_counts[sender] += 1
                action = use.get("selected_action")
                if action is not None and 0 <= int(action) < ROUTING_ACTION_DIM:
                    packet_use_counts_by_action[sender][int(action)] += 1
        for link in slot.get("active_links", []):
            if not isinstance(link, Mapping):
                continue
            link_type = str(link.get("link_type", "unknown"))
            sender = int(link.get("sender_id", -1))
            receiver = int(link.get("receiver_id", -1))
            bits = float(link.get("transmitted_bits", 0.0))
            active_seconds = float(link.get("active_seconds", 0.0))
            if link_type == "S2U" and 0 <= receiver < NUM_UAV:
                access_received[receiver]["bits"] += bits
                access_received[receiver]["active_seconds"] += active_seconds
                access_received[receiver]["link_observations"] += 1
            elif link_type in {"U2U", "U2G"} and 0 <= sender < NUM_UAV:
                action = 16 if link_type == "U2G" else receiver
                counters = by_uav_service[sender].setdefault(action, Counter())
                counters["transmitted_bits"] += bits
                counters["active_seconds"] += active_seconds
                counters["active_link_observations"] += 1
                service_cases.append(
                    {
                        "domain": "routing",
                        "kind": "positive_actual_service",
                        "time_s": slot.get("time_seconds"),
                        "sender_uav_id": sender,
                        "selected_action": action,
                        "link_type": link_type,
                        "transmitted_bits": bits,
                        "active_seconds": active_seconds,
                    }
                )
        for hop in slot.get("completed_hops", []):
            if not isinstance(hop, Mapping):
                continue
            link_type = str(hop.get("link_type"))
            if link_type == "S2U":
                receiver = hop.get("receiver_id")
                if isinstance(receiver, (int, float)) and 0 <= int(receiver) < NUM_UAV:
                    access_received[int(receiver)]["completed_packets"] += 1
                continue
            sender = hop.get("sender_id")
            receiver = hop.get("receiver_id")
            if not isinstance(sender, (int, float)) or not 0 <= int(sender) < NUM_UAV:
                continue
            action = 16 if link_type == "U2G" else int(receiver)
            by_uav_service[int(sender)].setdefault(action, Counter())[
                "completed_hops"
            ] += 1

    default_slot_seconds = (
        float(np.median(slot_seconds_values)) if slot_seconds_values else 0.25
    )
    by_sender: dict[str, Any] = {}
    decision_cases: list[dict[str, Any]] = []
    for sender in range(NUM_UAV):
        sender_records = [
            record
            for record in records
            if int(record.get("sender_uav_id", -1)) == sender
        ]
        branch_counts = {
            branch: [0] * ROUTING_ACTION_DIM
            for branch in ("epsilon_exploration", "safe_q_greedy")
        }
        legal_counts = [0] * ROUTING_ACTION_DIM
        reference_capacity: dict[int, list[float]] = {
            action: [] for action in range(ROUTING_ACTION_DIM)
        }
        neighbor_backlog: dict[int, list[float]] = {
            action: [] for action in range(ROUTING_ACTION_DIM)
        }
        sender_backlogs: list[float] = []
        hol_deadlines: list[float] = []
        counts = Counter()
        for record in sender_records:
            action = int(record["selected_action"])
            branch = str(record.get("decision_branch", "unknown"))
            if branch in branch_counts:
                branch_counts[branch][action] += 1
            mask = np.asarray(record.get("legal_action_mask", []), dtype=bool)
            if mask.shape == (ROUTING_ACTION_DIM,):
                for legal_action in np.flatnonzero(mask):
                    legal_counts[int(legal_action)] += 1
            backlog = record.get("sender_total_backlog_bits")
            if backlog is not None and math.isfinite(float(backlog)):
                sender_backlogs.append(float(backlog))
                counts["slot_start_backlog_positive"] += int(float(backlog) > 0.0)
                counts["slot_start_backlog_zero"] += int(float(backlog) == 0.0)
            else:
                counts["slot_start_backlog_unknown"] += 1
            if "hol_packet_id" not in record:
                counts["hol_presence_unknown"] += 1
            elif record.get("hol_packet_id") is None:
                counts["no_hol"] += 1
            else:
                counts["hol_present"] += 1
                if record.get("hol_remaining_deadline_seconds") is not None:
                    hol_deadlines.append(
                        float(record["hol_remaining_deadline_seconds"])
                    )
                if action == sender:
                    counts["hol_wait_action"] += 1
            counts["zero_prior_mass_fallback"] += int(
                bool(record.get("zero_legal_prior_mass_fallback", False))
            )
            raw_capacities = record.get("reference_capacity_bps_by_legal_action")
            if isinstance(raw_capacities, Mapping):
                for key, value in raw_capacities.items():
                    index = int(key)
                    if 0 <= index < ROUTING_ACTION_DIM and math.isfinite(float(value)):
                        reference_capacity[index].append(float(value))
            try:
                named = _routing_state_named_values(record["compact_routing_state"])
            except (KeyError, TypeError, ValueError):
                named = {}
            for legal_action in np.flatnonzero(mask) if mask.shape == (ROUTING_ACTION_DIM,) else ():
                key = f"next_hop_backlog[{int(legal_action)}]"
                if key in named:
                    neighbor_backlog[int(legal_action)].append(float(named[key]))
            if bool(record.get("zero_legal_prior_mass_fallback", False)) or (
                record.get("hol_packet_id") is not None and action == sender
            ):
                decision_cases.append(
                    {
                        "domain": "routing",
                        "kind": (
                            "zero_legal_prior_mass_fallback"
                            if record.get("zero_legal_prior_mass_fallback")
                            else "wait_with_hol"
                        ),
                        "time_s": record.get("time_seconds"),
                        "sender_uav_id": sender,
                        "branch": branch,
                        "selected_action": action,
                        "sender_backlog_bits": backlog,
                        "hol_remaining_deadline_s": record.get(
                            "hol_remaining_deadline_seconds"
                        ),
                    }
                )
        service = by_uav_service[sender]
        reasons: dict[int, list[str]] = {}
        if legal_counts[16] > 0:
            reasons.setdefault(16, []).append("observable_GS_link")
        positive_actions = [
            action
            for action, values in service.items()
            if float(values["transmitted_bits"]) > 0.0
        ]
        if positive_actions:
            action = max(
                positive_actions,
                key=lambda value: (float(service[value]["transmitted_bits"]), -value),
            )
            reasons.setdefault(action, []).append("highest_actual_service_bits")
        alternatives = [
            action
            for action in range(ROUTING_ACTION_DIM)
            if legal_counts[action] > 0 and action not in reasons
        ]
        if alternatives:
            action = max(
                alternatives,
                key=lambda value: (
                    legal_counts[value],
                    np.mean(reference_capacity[value])
                    if reference_capacity[value]
                    else -1.0,
                    -value,
                ),
            )
            reasons.setdefault(action, []).append("most_observed_legal_alternative")
        representative_links = []
        for action, selection_reasons in list(reasons.items())[:3]:
            actual = service.get(action, Counter())
            representative_links.append(
                {
                    "action_index": action,
                    "endpoint": "WAIT" if action == sender else "GS" if action == 16 else f"UAV_{action}",
                    "selection_reasons": selection_reasons,
                    "legal_opportunity_count": int(legal_counts[action]),
                    "reference_capacity_bps": _numeric_summary(reference_capacity[action], 2),
                    "neighbor_backlog_log_normalized": _numeric_summary(neighbor_backlog[action]),
                    "actual_transmitted_bits": float(actual["transmitted_bits"]),
                    "actual_active_service_seconds": float(actual["active_seconds"]),
                    "completed_hop_count": int(actual["completed_hops"]),
                    "active_link_observation_count": int(actual["active_link_observations"]),
                }
            )
        by_sender[f"UAV_{sender}"] = {
            "action_index_semantics": "0..15 UAV actions, sender index WAIT, 16 GS",
            "decision_count": len(sender_records),
            "epsilon_exploration_action_counts_17": branch_counts["epsilon_exploration"],
            "safe_q_greedy_action_counts_17": branch_counts["safe_q_greedy"],
            "legal_action_opportunity_counts_17": legal_counts,
            "slot_start_sender_backlog_bits": _numeric_summary(sender_backlogs, 2),
            "slot_start_backlog_positive_count": int(counts["slot_start_backlog_positive"]),
            "slot_start_backlog_zero_count": int(counts["slot_start_backlog_zero"]),
            "slot_start_backlog_unknown_count": int(counts["slot_start_backlog_unknown"]),
            "hol_present_count": int(counts["hol_present"]),
            "no_hol_count": int(counts["no_hol"]),
            "hol_presence_unknown_count": int(counts["hol_presence_unknown"]),
            "hol_remaining_deadline_s": _numeric_summary(hol_deadlines, 6),
            "hol_present_observed_seconds": round(float(counts["hol_present"]) * default_slot_seconds, 6),
            "hol_wait_action_seconds": round(float(counts["hol_wait_action"]) * default_slot_seconds, 6),
            "zero_legal_prior_mass_fallback_count": int(counts["zero_prior_mass_fallback"]),
            "preselected_action_packet_use_count": int(packet_use_counts[sender]),
            "preselected_action_packet_use_counts_17": [
                int(packet_use_counts_by_action[sender][action])
                for action in range(ROUTING_ACTION_DIM)
            ],
            "actual_uav_service": {
                "transmitted_bits": float(sum(values["transmitted_bits"] for values in service.values())),
                "active_service_seconds": float(sum(values["active_seconds"] for values in service.values())),
                "completed_hops": int(sum(values["completed_hops"] for values in service.values())),
            },
            "s2u_access_received": {
                "transmitted_bits": float(access_received[sender]["bits"]),
                "active_service_seconds": float(
                    access_received[sender]["active_seconds"]
                ),
                "completed_packets": int(
                    access_received[sender]["completed_packets"]
                ),
                "active_link_observation_count": int(
                    access_received[sender]["link_observations"]
                ),
            },
            "representative_links_maximum_3": representative_links,
        }
    return {
        "scope": "complete episode routing decisions and authoritative PacketEngine event-time service",
        "slot_duration_seconds": default_slot_seconds,
        "global_shared_routing_reward": {
            "episode_sum_counting_each_slot_once": float(
                math.fsum(
                    float(slot.get("global_routing_reward", 0.0))
                    for slot in slot_records
                )
            ),
            "duplicated_reward_row_sum_diagnostic": float(
                math.fsum(
                    float(slot.get("shared_reward_row_sum", 0.0))
                    for slot in slot_records
                )
            ),
            "aggregation_rule": "the episode learning-signal total uses the global slot reward once, not the per-UAV row sum",
        },
        "service_semantics": {
            "preselection": "selected action at slot start",
            "packet_use": "a packet was joined to the sender decision",
            "actual_transmission": "positive authoritative transmitted bits, including partial hops",
            "completed_hop": "full S2U, U2U, or U2G hop completion",
            "gs_delivery": "terminal GS outcome, separate from relay service",
        },
        "by_sender": by_sender,
    }, _select_representative_cases(decision_cases + service_cases, maximum=4)


def _complete_previous_guidance(guidance: Mapping[str, Any]) -> dict[str, Any]:
    td3_known = bool(guidance.get("enabled")) and not bool(
        guidance.get("td3_fallback", True)
    )
    routing_known = bool(guidance.get("enabled")) and not bool(
        guidance.get("routing_fallback", True)
    )
    return {
        "td3_bias": (
            {
                "status": "known_applied_episode_guidance",
                "shape": [NUM_UAV, 3],
                "value": np.asarray(guidance["td3_bias"], dtype=float).tolist(),
            }
            if td3_known and guidance.get("td3_bias") is not None
            else {
                "status": "unknown_no_applied_guidance",
                "reason": guidance.get("td3_reason"),
                "value": None,
            }
        ),
        "routing_prior": (
            {
                "status": "known_applied_episode_guidance",
                "shape": [NUM_UAV, ROUTING_ACTION_DIM],
                "value": np.asarray(guidance["routing_prior"], dtype=float).tolist(),
            }
            if routing_known and guidance.get("routing_prior") is not None
            else {
                "status": "unknown_no_applied_guidance",
                "reason": guidance.get("routing_reason"),
                "value": None,
            }
        ),
    }


def assemble_first_gen_feedback(
    *,
    metrics: Mapping[str, Any],
    movement_records: list[dict[str, Any]],
    routing_records: list[dict[str, Any]],
    slot_service_records: list[dict[str, Any]],
    applied_guidance: Mapping[str, Any],
) -> dict[str, Any]:
    movement, movement_cases = first_gen_movement_summary(
        movement_records, applied_guidance
    )
    routing, routing_cases = first_gen_routing_summary(
        routing_records, slot_service_records
    )
    from llm_state_rule_explorer import slot_service_summary

    service = slot_service_summary(slot_service_records, metrics)
    application = {
        "guidance_enabled": bool(applied_guidance.get("enabled", False)),
        "td3_fallback": bool(applied_guidance.get("td3_fallback", True)),
        "routing_fallback": bool(applied_guidance.get("routing_fallback", True)),
        "td3_reason": applied_guidance.get("td3_reason"),
        "routing_reason": applied_guidance.get("routing_reason"),
        "movement_action_source_counts": dict(
            sorted(
                Counter(
                    str(record.get("action_source", "unknown"))
                    for record in movement_records
                ).items()
            )
        ),
        "td3_bias_application_count": int(
            sum(
                item.get("applied_llm_bias") is not None
                for record in movement_records
                for item in record.get("uavs", [])
            )
        ),
        "routing_zero_legal_prior_mass_fallback_count": int(
            sum(
                bool(record.get("zero_legal_prior_mass_fallback", False))
                for record in routing_records
            )
        ),
    }
    return {
        "schema_version": LLM_EXPLORER_VERSION,
        "summary_version": LLM_EXPLORER_SUMMARY_VERSION,
        "case_selection_version": LLM_EXPLORER_CASE_VERSION,
        "episode_outcomes": _json_clone(metrics),
        "previous_complete_guidance": _complete_previous_guidance(
            applied_guidance
        ),
        "previous_application_status": application,
        "uav_core_summary": movement,
        "routing_core_summary": routing,
        "actual_slot_service_summary": service,
        "representative_cases": _select_representative_cases(
            movement_cases + routing_cases, maximum=8
        ),
        "case_semantics": "deterministic illustrations only; complete summaries provide denominators",
        "applied_guidance": _semantic_guidance(dict(applied_guidance)),
    }


def build_observable_initial_state(env: Any, packet_engine: Any) -> dict[str, Any]:
    """Build episode-start input without serializing hidden RoIs or SRs."""

    known_rois = {
        int(gt.id): gt for gt in env.gts if bool(getattr(gt, "is_found", False))
    }
    observable_srs = {
        int(sr.id): sr
        for sr in env.SR_teams
        if getattr(sr, "assigned_gt_id", None) is not None
        and int(sr.assigned_gt_id) in known_rois
    }

    def safe_tasks(uav_id: int) -> list[dict[str, Any]]:
        result = []
        for raw in env.multi_tasks.get(uav_id, ()):
            task_type = str(raw.get("task_type", "Hovering"))
            target = raw.get("target_obj_id")
            if task_type == "FOV":
                target = int(target) if target is not None and int(target) in known_rois else None
            elif task_type == "COM":
                target = int(target) if target is not None and int(target) in observable_srs else None
            else:
                target = None
            result.append(
                {
                    "task_type": task_type,
                    "target_obj_id": target,
                    "target_status": "assigned_observable" if target is not None else "unassigned_or_unobservable",
                }
            )
        return result

    backlog = getattr(packet_engine, "backlog_bits", {})
    value = {
        "observation_time": "episode_start_before_first_movement_or_routing_action",
        "hidden_roi_count_disclosed": False,
        "configured_roi_count_range": [2, 8],
        "known_rois": [
            {
                "roi_id": roi_id,
                "position_m": [float(gt.x), float(gt.y), float(gt.z)],
                "radius_m": float(gt.radius),
                "discovered": True,
            }
            for roi_id, gt in sorted(known_rois.items())
        ],
        "observable_sr_teams": [
            {
                "sr_id": sr_id,
                "position_m": [float(sr.x), float(sr.y), float(sr.z)],
                "assigned_roi_id": int(sr.assigned_gt_id),
                "arrived": bool(getattr(sr, "arrived", False)),
            }
            for sr_id, sr in sorted(observable_srs.items())
        ],
        "uavs": [
            {
                "uav_id": int(uav_id),
                "position_m": [float(uav.x_u), float(uav.y_u), float(uav.z_u)],
                "remaining_energy_j": float(uav.energy),
                "battery_percent": float(uav.battery),
                "tasks": safe_tasks(int(uav_id)),
                "queue_backlog_bits": float(backlog.get(uav_id, 0.0)),
            }
            for uav_id, uav in sorted(env.uav_dict.items())
        ],
        "ground_station": {
            "action_index": 16,
            "position_m": [float(value) for value in env.GS_pos],
        },
    }
    return _json_clone(value)


def build_movement_decision_record(
    env: Any,
    *,
    episode_index: int,
    interval_index: int,
    movement_mask: Any,
    action_source: str,
    executed_action: Any,
    actor_action: Any = None,
    applied_bias: Any = None,
    noise_std: float | None = None,
    auxiliary_snapshot: Mapping[str, Any] | None = None,
    action_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture only information observable at this exact decision time."""

    known_rois = {
        int(gt.id): gt for gt in env.gts if bool(getattr(gt, "is_found", False))
    }
    uavs = []
    mask = np.asarray(movement_mask, dtype=bool)
    actions = np.asarray(executed_action, dtype=float).reshape(NUM_UAV, 3)
    actors = None if actor_action is None else np.asarray(actor_action, dtype=float).reshape(NUM_UAV, 3)
    biases = None if applied_bias is None else np.asarray(applied_bias, dtype=float).reshape(NUM_UAV, 3)
    pre_projection = None
    if isinstance(action_diagnostics, Mapping) and action_diagnostics.get(
        "pre_projection_action"
    ) is not None:
        pre_projection = np.asarray(
            action_diagnostics["pre_projection_action"], dtype=float
        ).reshape(NUM_UAV, 3)

    snapshot_valid = bool(
        isinstance(auxiliary_snapshot, Mapping)
        and np.asarray(
            auxiliary_snapshot.get("snapshot_valid", [False]), dtype=bool
        ).reshape(-1)[0]
    )

    def snapshot_value(name: str, index: int, *, default: Any = None) -> Any:
        if not snapshot_valid or name not in auxiliary_snapshot:
            return default
        values = np.asarray(auxiliary_snapshot[name])
        if values.ndim == 0 or index >= len(values):
            return default
        value = values[index]
        if np.asarray(value).ndim:
            return np.asarray(value).tolist()
        return value.item() if hasattr(value, "item") else value

    def snapshot_matrix_value(
        name: str, row: int, column: int, *, default: Any = None
    ) -> Any:
        if not snapshot_valid or name not in auxiliary_snapshot:
            return default
        values = np.asarray(auxiliary_snapshot[name])
        if values.ndim < 2 or row >= values.shape[0] or column >= values.shape[1]:
            return default
        value = values[row, column]
        return value.item() if hasattr(value, "item") else value

    for uav_id, uav in sorted(env.uav_dict.items()):
        targets = []
        task_types = []
        for task in env.multi_tasks.get(uav_id, ()):
            task_type = str(task.get("task_type", "Hovering"))
            task_types.append(task_type)
            if task_type == "FOV" and task.get("target_obj_id") in known_rois:
                target = known_rois[int(task["target_obj_id"])]
                targets.append(
                    {
                        "kind": "roi",
                        "id": int(target.id),
                        "relative_position_m": [
                            float(target.x - uav.x_u),
                            float(target.y - uav.y_u),
                            float(target.z - uav.z_u),
                        ],
                    }
                )
            elif task_type == "COM":
                sr_id = task.get("target_obj_id")
                if sr_id is not None and 0 <= int(sr_id) < len(env.SR_teams):
                    sr = env.SR_teams[int(sr_id)]
                    if getattr(sr, "assigned_gt_id", None) in known_rois:
                        targets.append(
                            {
                                "kind": "sr",
                                "id": int(sr.id),
                                "relative_position_m": [
                                    float(sr.x - uav.x_u),
                                    float(sr.y - uav.y_u),
                                    float(sr.z - uav.z_u),
                                ],
                            }
                        )
        service_observation = None
        if snapshot_valid:
            queue_valid = bool(snapshot_value("uav_queue_valid", uav_id, default=False))
            hol_valid = bool(snapshot_value("uav_hol_valid", uav_id, default=False))
            fov_valid = bool(snapshot_value("vs_pair_valid", uav_id, default=False))
            com_target_ids = [
                int(target["id"])
                for target in targets
                if str(target.get("kind")) == "sr"
            ]
            sr_row = None
            if com_target_ids:
                sr_ids = np.asarray(auxiliary_snapshot.get("sr_id", []), dtype=int)
                matches = np.flatnonzero(sr_ids == com_target_ids[0])
                if matches.size:
                    candidate = int(matches[0])
                    if bool(snapshot_value("sr_observable", candidate, default=False)):
                        sr_row = candidate
            sr_queue_valid = bool(
                sr_row is not None
                and snapshot_value("sr_queue_valid", sr_row, default=False)
            )
            s2u_valid = bool(
                sr_row is not None
                and snapshot_matrix_value(
                    "s2u_link_valid", sr_row, uav_id, default=False
                )
            )
            service_observation = {
                "snapshot_available": True,
                "uav_queue": {
                    "valid": queue_valid,
                    "empty": (
                        bool(snapshot_value("uav_queue_empty", uav_id))
                        if queue_valid
                        else None
                    ),
                    "backlog_bits": (
                        float(snapshot_value("uav_backlog_bits", uav_id))
                        if queue_valid
                        else None
                    ),
                    "fov_backlog_bits": (
                        float(snapshot_value("uav_vs_backlog_bits", uav_id))
                        if queue_valid
                        else None
                    ),
                    "com_backlog_bits": (
                        float(snapshot_value("uav_com_backlog_bits", uav_id))
                        if queue_valid
                        else None
                    ),
                    "hol_present": hol_valid,
                    "hol_type_code": (
                        int(snapshot_value("uav_hol_type", uav_id))
                        if hol_valid
                        else None
                    ),
                    "hol_remaining_deadline_s": (
                        float(
                            snapshot_value(
                                "uav_hol_remaining_deadline_s", uav_id
                            )
                        )
                        if hol_valid
                        else None
                    ),
                },
                "fov": {
                    "pair_valid": fov_valid,
                    "coverage_ratio": (
                        float(snapshot_value("vs_coverage_ratio", uav_id))
                        if fov_valid
                        else None
                    ),
                    "image_quantity": (
                        float(snapshot_value("vs_image_quantity", uav_id))
                        if fov_valid
                        else None
                    ),
                    "c9_margin_m": (
                        float(snapshot_value("vs_c9_margin_m", uav_id))
                        if fov_valid
                        else None
                    ),
                    "c10_geometry_valid": (
                        bool(snapshot_value("vs_c10_geometry_valid", uav_id))
                        if fov_valid
                        else None
                    ),
                    "c10_margin_m": (
                        float(snapshot_value("vs_c10_margin_m", uav_id))
                        if fov_valid
                        and bool(
                            snapshot_value(
                                "vs_c10_geometry_valid", uav_id, default=False
                            )
                        )
                        else None
                    ),
                    "capture_valid": (
                        bool(snapshot_value("vs_capture_valid", uav_id))
                        if fov_valid
                        else None
                    ),
                },
                "com": {
                    "target_valid": sr_row is not None,
                    "sr_id": None if sr_row is None else com_target_ids[0],
                    "sr_queue_valid": sr_queue_valid,
                    "sr_queue_empty": (
                        bool(snapshot_value("sr_queue_empty", sr_row))
                        if sr_queue_valid
                        else None
                    ),
                    "sr_backlog_bits": (
                        float(snapshot_value("sr_backlog_bits", sr_row))
                        if sr_queue_valid
                        else None
                    ),
                    "sr_packet_count": (
                        int(snapshot_value("sr_packet_count", sr_row))
                        if sr_queue_valid
                        else None
                    ),
                    "sr_hol_remaining_deadline_s": (
                        float(
                            snapshot_value(
                                "sr_hol_remaining_deadline_s", sr_row
                            )
                        )
                        if sr_queue_valid
                        and bool(
                            snapshot_value(
                                "sr_hol_deadline_valid", sr_row, default=False
                            )
                        )
                        else None
                    ),
                    "s2u_link_valid": s2u_valid,
                    "s2u_in_range": (
                        bool(
                            snapshot_matrix_value(
                                "s2u_in_range", sr_row, uav_id
                            )
                        )
                        if s2u_valid
                        else None
                    ),
                    "s2u_reference_capacity_mbps": (
                        float(
                            snapshot_matrix_value(
                                "s2u_reference_capacity_mbps", sr_row, uav_id
                            )
                        )
                        if s2u_valid
                        else None
                    ),
                },
            }
        uavs.append(
            {
                "uav_id": int(uav_id),
                "position_m": [float(uav.x_u), float(uav.y_u), float(uav.z_u)],
                "remaining_energy_j": float(uav.energy),
                "task_types": task_types,
                "observable_targets": targets,
                "movement_mask": bool(mask[uav_id]),
                "pre_projection_normalized_action": (
                    None
                    if pre_projection is None
                    else pre_projection[uav_id].tolist()
                ),
                "policy_projected_normalized_action": actions[uav_id].tolist(),
                "executed_normalized_action": actions[uav_id].tolist(),
                "actor_normalized_action": None if actors is None else actors[uav_id].tolist(),
                "applied_llm_bias": None if biases is None else biases[uav_id].tolist(),
                "service_observation": service_observation,
            }
        )
    return _json_clone(
        {
            "episode_index": int(episode_index),
            "interval_index": int(interval_index),
            "time_seconds": float(interval_index),
            "action_source": str(action_source),
            "behavior_noise_std": None if noise_std is None else float(noise_std),
            "known_roi_ids_at_sample_time": sorted(known_rois),
            "uavs": uavs,
        }
    )


def environment_constants_and_definitions(
    *,
    episode_seconds: int,
    routing_slot_seconds: float,
    environment_width_m: float = 1000.0,
    environment_height_m: float = 1000.0,
    task_deadlines_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    from experiment_config import (
        COM_PACKET_RATE_PER_SECOND,
        COM_PACKET_SIZE_BITS,
        COMMUNICATION_RANGE_M,
        MOVEMENT_CONSTRAINT_WEIGHT_C10,
        MOVEMENT_CONSTRAINT_WEIGHT_C9,
        MOVEMENT_CONSTRAINT_WEIGHT_COM,
        PROPULSION_MODEL_ID,
        PROPULSION_PARAMETERS,
        REFERENCE_COM_BANDWIDTH_HZ,
        ROUTING_REWARD_BITS_PER_UNIT,
        SAFE_DDQN_ETA_C,
        SAFE_DDQN_QOS_TARGET_PROBABILITY,
        TOTAL_COMMUNICATION_BANDWIDTH_HZ,
    )
    from Energy_model import EnergyConsumptionModel
    from visual_sensing import visual_sensing_metadata

    visual = visual_sensing_metadata()
    energy_model = EnergyConsumptionModel(E_max=1.0, N_u=NUM_UAV)
    propulsion_references = {
        "hover_0_0_0_w": energy_model.propulsion_power((0.0, 0.0, 0.0)),
        "horizontal_5_mps_w": energy_model.propulsion_power((5.0, 0.0, 0.0)),
        "horizontal_10_mps_w": energy_model.propulsion_power((10.0, 0.0, 0.0)),
        "vertical_up_2_mps_w": energy_model.propulsion_power((0.0, 0.0, 2.0)),
        "vertical_down_2_mps_w": energy_model.propulsion_power((0.0, 0.0, -2.0)),
    }
    return {
        "schema_version": LLM_EXPLORER_VERSION,
        "num_uav": NUM_UAV,
        "uav_id_range_inclusive": [0, 15],
        "ground_station_action_index": 16,
        "configured_roi_count_range": [2, 8],
        "operating_area_m": {
            "width": float(environment_width_m),
            "height": float(environment_height_m),
        },
        "uav_altitude_bounds_m": [50.0, 150.0],
        "episode_seconds": int(episode_seconds),
        "movement_interval_seconds": 1.0,
        "routing_slot_seconds": float(routing_slot_seconds),
        "routing_slots_per_movement_interval": int(round(1.0 / routing_slot_seconds)),
        "communication_range_3d_m_inclusive": float(COMMUNICATION_RANGE_M),
        "task_deadlines_seconds": {
            str(key): float(value)
            for key, value in (task_deadlines_seconds or {"FOV": 0.015, "COM": 0.010}).items()
        },
        "packet_generation_model": "fixed intervals from effective configured rates; events are invisible before generation",
        "routing_decision_model": (
            "all UAVs preselect one physically legal next hop at each "
            f"{float(routing_slot_seconds):g} s slot start and hold it for the slot"
        ),
        "packet_service_model": "event-time full-packet store-and-forward; completed packets may traverse multiple hops in one slot",
        "bandwidth_model": "dynamic equal-FDMA over currently serviceable S2U/U2U/U2G links; empty preselection reserves no bandwidth",
        "fading_model": "5 ms gain blocks; service completion and next-hop start may be finer than 5 ms",
        "traffic_defaults_packets_per_second": {
            "FOV": 5.0,
            "COM": float(COM_PACKET_RATE_PER_SECOND),
        },
        "com_packet_bits": float(COM_PACKET_SIZE_BITS),
        "fov_packet_bits": visual["packet_size"],
        "total_dynamic_fdma_bandwidth_hz": float(
            TOTAL_COMMUNICATION_BANDWIDTH_HZ
        ),
        "s2u_reference_bandwidth_hz": float(REFERENCE_COM_BANDWIDTH_HZ),
        "action_decode": {
            "horizontal_speed_mps": "5 * (speed_scalar + 1)",
            "absolute_heading_rad": "pi * heading_scalar",
            "vertical_speed_mps": "2 * vertical_scalar",
            "speed_and_vertical_domain": [-1.0, 1.0],
            "heading_domain": "periodic [-1,1)",
        },
        "movement_reward": {
            "formula": "timely_Mbit - lambda_EE_Mbit_per_J * movement_J - P_C9 - P_C10 - P_COM",
            "constraint_weights": {
                "c9": float(MOVEMENT_CONSTRAINT_WEIGHT_C9),
                "c10": float(MOVEMENT_CONSTRAINT_WEIGHT_C10),
                "com_range": float(MOVEMENT_CONSTRAINT_WEIGHT_COM),
            },
            "penalty_denominators": (
                "each penalty is averaged over its own currently applicable "
                "service-pair set; an empty set contributes zero"
            ),
        },
        "routing_reward": {
            "formula": "slot timely useful bits delivered to GS / 10000",
            "bits_per_reward_unit": float(ROUTING_REWARD_BITS_PER_UNIT),
            "unit": "1 reward per decimal 10 kbit",
            "sharing": "the same global slot reward is stored in every UAV routing transition; episode routing reward sums each slot once",
            "zero_and_bounds": "zero when no timely useful GS delivery; no clipping or dynamic normalization",
        },
        "safe_ddqn": {
            "score": "Q_reward - lambda_cost * Q_cost",
            "qos_target_probability": float(SAFE_DDQN_QOS_TARGET_PROBABILITY),
            "lambda_cost_step_size": float(SAFE_DDQN_ETA_C),
            "lambda_cost_update": (
                "episode-end update from global eligible and violated packet "
                "counts; target is the configured QoS probability"
            ),
        },
        "propulsion": {
            "model_id": PROPULSION_MODEL_ID,
            "formula_scope": "P(actual projected 3-D velocity); E=P*delta_t",
            "parameters": dict(PROPULSION_PARAMETERS),
            "reference_power_w": propulsion_references,
            "reference_duration_s": 1.0,
        },
        "feedback_dictionary": {
            "timely_useful_delivery_bits": (
                "authoritative EE numerator: complete on-time GS delivery; "
                "FOV physical bits weighted by capture coverage and COM bits full"
            ),
            "generated_and_terminal_packet_counts": (
                "FOV and COM are separate; eligible/violated counts retain the "
                "PacketEngine terminal-outcome definitions"
            ),
            "movement_energy_j": "all UAVs and all movement controllers, including hover",
            "routing_reward_episode_sum": "global shared reward counted once per slot",
            "branch_action_counts": "17-entry arrays in action-index order; sender index is WAIT",
            "legal_action_opportunities": "decision-time effective-mask counts, not service counts",
            "actual_service": "positive PacketEngine transmission bits and completion events",
            "circular_direction": (
                "bearing means use atan2 and circular resultant length; a null "
                "mean denotes absent or directionally dispersed samples"
            ),
            "representative_links": (
                "at most three deterministic profiles per sender; omitted links "
                "are not implied illegal or unused"
            ),
            "representative_cases": "at most eight deterministic illustrations; not frequency estimates",
        },
        "visual_sensing": visual,
        "validity": {
            "unknown": "not observed or not valid at the recorded time",
            "unassigned": "no established service target",
            "zero": "observed valid numeric zero",
            "empty_queue": "observed valid queue with zero backlog",
        },
    }


def render_prompt(template: str, replacements: dict[str, Any]) -> str:
    """Literal one-pass placeholder substitution; JSON braces are never parsed."""

    rendered = str(template)
    unresolved_template = str(template)
    for name, value in replacements.items():
        token = "{{" + str(name) + "}}"
        if token not in unresolved_template:
            raise ValueError(f"prompt template lacks placeholder {token}")
        replacement = value if isinstance(value, str) else _compact_json(value)
        rendered = rendered.replace(token, replacement)
        unresolved_template = unresolved_template.replace(token, "")
    # Inspect only the original template remainder.  Inserted JSON or model
    # analysis can legitimately contain double braces and must never be
    # reinterpreted as template syntax.
    missing_start = unresolved_template.find("{{")
    if missing_start >= 0:
        missing_end = unresolved_template.find("}}", missing_start + 2)
        if missing_end >= 0:
            raise ValueError(
                "unfilled prompt placeholder "
                f"{unresolved_template[missing_start:missing_end + 2]}"
            )
    return rendered


def llm_explorer_path_preflight(
    run_directory: str | Path,
    *,
    maximum_episode: int,
    windows_limit: int = WINDOWS_CONSERVATIVE_PATH_LIMIT,
    artifact_directory: str = "llm_explorer",
    named_routing: bool = False,
    td3_only: bool = False,
    state_rule: bool = False,
) -> dict[str, Any]:
    """Validate every filename produced by the explorer and streaming client.

    The conservative limit leaves headroom below classic MAX_PATH for Windows
    API expansion.  It is applied on every platform so a run directory created
    on Linux cannot later be mistaken for native Windows validation.
    """

    run_root = Path(run_directory).resolve()
    episode = f"episode_{max(1, int(maximum_episode)):06d}"
    stages = ("analysis", "guidance")
    stage_files = (
        "system_prompt.txt", "user_prompt.txt", "identity.json",
        "completed.json", "token_budget.json", ".completed.json.tmp",
        ".token_budget.json.tmp",
    )
    attempt_files = (
        "prompt.txt", "request.json", "response.txt", "response_metadata.json",
        "validation.json", "token_budget.json", ".token_budget.json.tmp",
        "stream_events.jsonl", "response_content.partial.txt",
        "reasoning.partial.txt", "stream_status.json", ".response_metadata.json.tmp",
        ".validation.json.tmp", ".stream_status.json.tmp",
    )
    artifact_root = Path(str(artifact_directory))
    if artifact_root.name != str(artifact_directory) or str(artifact_directory) in {"", ".", ".."}:
        raise ValueError("artifact_directory must be one short path component")
    relative_paths = [
        artifact_root / "events.jsonl",
        artifact_root / "state.json",
        artifact_root / ".state.json.tmp",
        artifact_root / episode / "feedback.json",
        artifact_root / episode / ".feedback.json.tmp",
        artifact_root / episode / "episode_result.json",
        artifact_root / episode / ".episode_result.json.tmp",
    ]
    for stage in stages:
        relative_paths.extend(
            artifact_root / episode / stage / name
            for name in stage_files
        )
        for attempt in (1, LLM_EXPLORER_MAX_ATTEMPTS):
            relative_paths.extend(
                artifact_root / episode / stage / f"attempt_{attempt:02d}" / name
                for name in attempt_files
            )
    checks = [
        {
            "relative_path": str(relative),
            "absolute_path": str(run_root / relative),
            "characters": len(str(run_root / relative)),
        }
        for relative in relative_paths
    ]
    longest = max(checks, key=lambda item: item["characters"])
    encoder = _token_encoder()
    if state_rule:
        # Exercise the output reserve with a near-maximum legal AST: each
        # balanced 17-leaf component has 33 nodes, for 99 nodes in total.
        def representative_component() -> Any:
            nodes: list[Any] = [round(index / 17.0, 8) for index in range(17)]
            while len(nodes) > 1:
                next_nodes: list[Any] = []
                for index in range(0, len(nodes), 2):
                    next_nodes.append(
                        nodes[index]
                        if index + 1 == len(nodes)
                        else {"op": "add", "args": [nodes[index], nodes[index + 1]]}
                    )
                nodes = next_nodes
            return nodes[0]

        representative_guidance_value = {
            "schema_version": "td3-state-bias-rule-v1",
            "speed": representative_component(),
            "heading": representative_component(),
            "vertical": representative_component(),
        }
        representative_ast_nodes = 99
        guidance_output_reserve = 4096
    else:
        representative_guidance_value = {
            "td3_bias": [[-0.123456, 0.123456, 0.0] for _ in range(NUM_UAV)]
        }
        representative_ast_nodes = None
        guidance_output_reserve = LLMExplorerConfig().guidance_max_output_tokens
    if not td3_only and not state_rule:
        representative_guidance_value["routing_prior"] = (
            {
                f"UAV_{sender}": {
                    ("WAIT" if action == sender else "GS" if action == NUM_UAV else f"UAV_{action}"): round(1.0 / ROUTING_ACTION_DIM, 8)
                    for action in range(ROUTING_ACTION_DIM)
                }
                for sender in range(NUM_UAV)
            }
            if named_routing
            else [
                [round(1.0 / ROUTING_ACTION_DIM, 8) for _ in range(ROUTING_ACTION_DIM)]
                for _ in range(NUM_UAV)
            ]
        )
    representative_guidance = _compact_json(representative_guidance_value)
    representative_output_tokens = len(encoder.encode(representative_guidance))
    if representative_output_tokens >= guidance_output_reserve:
        raise ValueError(
            "configured guidance output reserve cannot hold the complete "
            + (
                "state-rule AST JSON response"
                if state_rule
                else "[16,3] JSON response"
                if td3_only
                else "[16,3] and [16,17] JSON response"
            )
        )
    report = {
        "strategy": "conservative_windows_character_limit",
        "native_windows_verified": os.name == "nt",
        "windows_limit_characters": int(windows_limit),
        "maximum_episode": int(maximum_episode),
        "artifact_directory": str(artifact_root),
        "named_routing": bool(named_routing),
        "td3_only": bool(td3_only),
        "state_rule": bool(state_rule),
        "checked_path_count": len(checks),
        "longest": longest,
        "paths": checks,
        "tokenizer_encoding": LLM_EXPLORER_ENCODING,
        "representative_full_guidance_output_tokens": representative_output_tokens,
        "representative_state_rule_ast_nodes": representative_ast_nodes,
        "guidance_reserved_output_tokens": guidance_output_reserve,
    }
    if int(longest["characters"]) > int(windows_limit):
        raise ValueError(
            "LLM explorer artifact path is too deep for the conservative Windows "
            f"limit ({longest['characters']} > {windows_limit} characters): "
            f"{longest['absolute_path']}. Choose a shorter --output-root; existing "
            "runs are not moved automatically."
        )
    return report


class DeterministicMockExplorerClient:
    """Offline client used by smoke tests; it performs no network requests."""

    def chat(self, *, model: str, prompt: str, system_prompt: str, **_: Any) -> dict[str, Any]:
        del system_prompt
        if "[Requested analysis]" in prompt:
            content = (
                "1. Priority issues\nNo reliable issue in offline mock data.\n"
                "2. TD3 recommendations\nUse neutral raw biases.\n"
                "3. Routing recommendations\nUse neutral priors.\n"
                "4. Expected effects and limitations\nNo live model inference."
            )
        else:
            content = json.dumps(
                {
                    "td3_bias": [[0.0, 0.0, 0.0] for _ in range(NUM_UAV)],
                    "routing_prior": [
                        [1.0 / ROUTING_ACTION_DIM for _ in range(ROUTING_ACTION_DIM)]
                        for _ in range(NUM_UAV)
                    ],
                },
                separators=(",", ":"),
            )
        return {
            "content": content,
            "actual_model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "stop",
        }


class LLMExplorerController:
    """Own the two-stage episode protocol, cache, diagnostics, and sampling."""

    schema_version = LLM_EXPLORER_VERSION
    artifact_directory_name = "llm_explorer"

    def __init__(
        self,
        *,
        run_directory: str | Path | None,
        run_identity: dict[str, Any],
        config: LLMExplorerConfig | None = None,
        client: Any = None,
        evaluation: bool = False,
        mock: bool = False,
        rng_seed: int = 0,
        lazy_client: bool = False,
    ) -> None:
        self.config = config or LLMExplorerConfig()
        self.run_identity = _json_clone(run_identity)
        self.evaluation = bool(evaluation)
        self.root = (
            None
            if run_directory is None
            else Path(run_directory) / self.artifact_directory_name
        )
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._client_override = client
        self._mock_client = bool(mock)
        self.client = None
        if not self.evaluation and not lazy_client:
            self._ensure_client()
        self.rng = np.random.default_rng(int(rng_seed))
        self.previous_episode_feedback: dict[str, Any] | None = None
        self.active_guidance = self._unguided("not_started")
        self.movement_records: list[dict[str, Any]] = []
        self.routing_records: list[dict[str, Any]] = []
        self.slot_service_records: list[dict[str, Any]] = []
        self._request_records: dict[str, dict[str, Any]] = {}
        self._fallback_records: dict[str, dict[str, Any]] = {}
        self._local_failure_records: dict[str, dict[str, Any]] = {}
        self.statistics = self._empty_statistics()
        self._reconcile_durable_events()

    def _ensure_client(self) -> None:
        """Construct a client on demand; v3 uses this to avoid post-cutoff setup."""

        if self.evaluation or self.client is not None:
            return
        if self._client_override is not None:
            self.client = self._client_override
        elif self._mock_client:
            self.client = DeterministicMockExplorerClient()
        else:
            from llm_design import OpenAIClient

            self.client = OpenAIClient(
                timeout=float(self.config.request_timeout_seconds),
                connect_timeout=min(10.0, float(self.config.request_timeout_seconds)),
                total_timeout=float(self.config.request_timeout_seconds),
                retries=0,
            )

    @staticmethod
    def _empty_statistics() -> dict[str, Any]:
        return {
            "requests_total": 0,
            "failed_attempts_total": 0,
            "retry_success_total": 0,
            "fallback_episodes_total": 0,
            "td3_fallback_episodes_total": 0,
            "routing_fallback_episodes_total": 0,
            "routing_zero_legal_prior_mass_total": 0,
            "wait_seconds_by_stage": {"analysis": 0.0, "guidance": 0.0},
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "unknown_attempts": 0},
            "actual_models": {},
            "failure_counts": {
                "api_or_transport": 0,
                "response_incomplete": 0,
                "content_validation": 0,
                "local_preflight": 0,
                "artifact_write": 0,
            },
        }

    @staticmethod
    def _unguided(reason: str) -> dict[str, Any]:
        return {
            "enabled": False,
            "td3_bias": np.zeros((NUM_UAV, 3), dtype=np.float32),
            "routing_prior": None,
            "td3_fallback": True,
            "routing_fallback": True,
            "td3_reason": str(reason),
            "routing_reason": str(reason),
            "analysis_attempt": None,
            "td3_guidance_attempt": None,
            "routing_guidance_attempt": None,
            "reused": False,
        }

    @property
    def event_path(self) -> Path | None:
        return None if self.root is None else self.root / "events.jsonl"

    def _event(self, kind: str, **fields: Any) -> None:
        event = {
            "schema_version": self.schema_version,
            "event": str(kind),
            "wall_time_unix": time.time(),
            **_json_clone(fields),
        }
        if self.event_path is not None:
            try:
                _append_jsonl(self.event_path, event)
            except OSError as exc:
                failure_id = _sha256_text(
                    f"event_journal|{kind}|{type(exc).__name__}|{exc}"
                )
                self._local_failure_records[failure_id] = {
                    "failure_id": failure_id,
                    "failure_category": "artifact_write",
                    "reason": f"event journal: {type(exc).__name__}: {exc}",
                }
                self._rebuild_statistics()
                print(
                    f"[LLM explorer] artifact_write failed reason={exc}",
                    flush=True,
                )

    def _reconcile_durable_events(self) -> None:
        if self.event_path is None or not self.event_path.is_file():
            return
        for line in self.event_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "request_attempt":
                request_id = event.get("request_id")
                if request_id:
                    self._request_records[str(request_id)] = event
            elif kind == "episode_fallback":
                episode = event.get("episode")
                if episode is not None:
                    self._fallback_records[str(int(episode))] = event
            elif kind in {"local_preflight_failure", "artifact_write_failure"}:
                failure_id = event.get("failure_id")
                if failure_id:
                    self._local_failure_records[str(failure_id)] = event
        self._rebuild_statistics()

    def _rebuild_statistics(self) -> None:
        zero_mass = int(self.statistics.get("routing_zero_legal_prior_mass_total", 0))
        stats = self._empty_statistics()
        stats["routing_zero_legal_prior_mass_total"] = zero_mass
        for event in self._request_records.values():
            stats["requests_total"] += 1
            if not bool(event.get("success")):
                stats["failed_attempts_total"] += 1
            if bool(event.get("retry_success")):
                stats["retry_success_total"] += 1
            stage = str(event.get("stage"))
            if stage in stats["wait_seconds_by_stage"]:
                stats["wait_seconds_by_stage"][stage] += float(
                    event.get("elapsed_seconds", 0.0)
                )
            category = str(event.get("failure_category") or "")
            if category in stats["failure_counts"]:
                stats["failure_counts"][category] += int(not bool(event.get("success")))
            usage = event.get("usage")
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if isinstance(usage.get(key), (int, float)):
                        stats["token_usage"][key] += int(usage[key])
            else:
                stats["token_usage"]["unknown_attempts"] += 1
            model = str(event.get("actual_model") or "unknown")
            stats["actual_models"][model] = stats["actual_models"].get(model, 0) + 1
        for event in self._fallback_records.values():
            stats["fallback_episodes_total"] += 1
            stats["td3_fallback_episodes_total"] += int(bool(event.get("td3")))
            stats["routing_fallback_episodes_total"] += int(bool(event.get("routing")))
        for event in self._local_failure_records.values():
            category = str(event.get("failure_category") or "")
            if category in stats["failure_counts"]:
                stats["failure_counts"][category] += 1
        self.statistics = stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": asdict(self.config),
            "run_identity": copy.deepcopy(self.run_identity),
            "previous_episode_feedback": copy.deepcopy(self.previous_episode_feedback),
            "active_guidance": self._serializable_guidance(self.active_guidance),
            "statistics": copy.deepcopy(self.statistics),
            "request_records": copy.deepcopy(self._request_records),
            "fallback_records": copy.deepcopy(self._fallback_records),
            "local_failure_records": copy.deepcopy(self._local_failure_records),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != self.schema_version:
            raise ExplorerValidationError("LLM explorer checkpoint version is incompatible")
        if state.get("config") != asdict(self.config):
            raise ExplorerValidationError("LLM explorer checkpoint config is incompatible")
        if state.get("run_identity") != self.run_identity:
            raise ExplorerValidationError("LLM explorer checkpoint run identity is incompatible")
        self.previous_episode_feedback = copy.deepcopy(state.get("previous_episode_feedback"))
        self.active_guidance = self._restore_guidance(state.get("active_guidance") or {})
        self._request_records.update(copy.deepcopy(state.get("request_records") or {}))
        self._fallback_records.update(copy.deepcopy(state.get("fallback_records") or {}))
        self._local_failure_records.update(
            copy.deepcopy(state.get("local_failure_records") or {})
        )
        saved_zero_mass = int(
            (state.get("statistics") or {}).get(
                "routing_zero_legal_prior_mass_total", 0
            )
        )
        self.statistics["routing_zero_legal_prior_mass_total"] = max(
            int(self.statistics.get("routing_zero_legal_prior_mass_total", 0)),
            saved_zero_mass,
        )
        self._rebuild_statistics()
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    @staticmethod
    def _serializable_guidance(guidance: dict[str, Any]) -> dict[str, Any]:
        result = dict(guidance)
        for key in (
            "td3_bias",
            "routing_prior",
            "td3_bias_raw",
            "td3_bias_effective",
        ):
            if isinstance(result.get(key), np.ndarray):
                result[key] = result[key].tolist()
        return _json_clone(result)

    @staticmethod
    def _restore_guidance(value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        bias = np.asarray(
            result.get("td3_bias", np.zeros((NUM_UAV, 3))), dtype=np.float64
        )
        if bias.shape != (NUM_UAV, 3) or not np.isfinite(bias).all():
            raise ExplorerValidationError("checkpoint td3_bias is invalid")
        result["td3_bias"] = bias.astype(np.float32)
        for key in ("td3_bias_raw", "td3_bias_effective"):
            if key in result:
                restored = np.asarray(result[key], dtype=np.float64)
                if restored.shape != (NUM_UAV, 3) or not np.isfinite(restored).all():
                    raise ExplorerValidationError(f"checkpoint {key} is invalid")
                result[key] = restored.astype(np.float32)
        if result.get("routing_prior") is not None:
            result["routing_prior"] = validate_routing_prior(result["routing_prior"])
        return result

    def _templates(self) -> tuple[str, str, str]:
        prompt_root = Path(__file__).resolve().parent / "prompts"
        return tuple(
            (prompt_root / name).read_text(encoding="utf-8")
            for name in (
                "llm_explorer_system.txt",
                "llm_explorer_analysis.txt",
                "llm_explorer_guidance.txt",
            )
        )

    def _render_system_prompt(
        self, template: str, environment_constants: dict[str, Any]
    ) -> str:
        return render_prompt(
            template,
            {"SETTINGS_AND_DATA_DEFINITIONS": environment_constants},
        )

    def _compact_feedback(self, previous: dict[str, Any]) -> dict[str, Any]:
        return compact_feedback_for_llm(previous)

    def _exploration_settings(
        self, *, sigma_episode_start: float, routing_epsilon: float
    ) -> dict[str, Any]:
        return {
            "upcoming_episode": getattr(self, "_upcoming_episode", None),
            "total_joint_transitions": getattr(
                self, "_total_joint_transitions", None
            ),
            "warmup_joint_transitions": self.config.warmup_joint_transitions,
            "activation_uses_actual_joint_transition_count": True,
            "sigma_episode_start": float(sigma_episode_start),
            "td3_bias_bounds": [
                -float(sigma_episode_start),
                float(sigma_episode_start),
            ],
            "routing_epsilon_at_episode_start": float(routing_epsilon),
            "routing_epsilon_schedule": "unchanged host schedule",
            "routing_epsilon_branch": (
                "100% masked and renormalized LLM prior; legal-uniform only "
                "when legal prior mass is zero"
            ),
            "bias_units": "normalized action units",
            "guidance_fixed_for_episode": True,
            "guidance_weight": 1.0,
            "annealing_end_episode": None,
            "episode_600_cutoff": False,
            "update_interval_episodes": 1,
        }

    def _analysis_replacements(
        self,
        *,
        previous: dict[str, Any],
        environment_constants: dict[str, Any],
        current_observable_initial_state: dict[str, Any],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        del environment_constants
        return {
            "PHASE_AND_EXPLORATION_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "EPISODE_OUTCOMES": previous["episode_outcomes"],
            "PREVIOUS_COMPLETE_GUIDANCE": previous[
                "previous_complete_guidance"
            ],
            "PREVIOUS_APPLICATION_STATUS": previous[
                "previous_application_status"
            ],
            "UAV_CORE_SUMMARY": previous["uav_core_summary"],
            "ROUTING_CORE_SUMMARY": {
                "routing_decisions": previous["routing_core_summary"],
                "actual_slot_service": previous[
                    "actual_slot_service_summary"
                ],
            },
            "REPRESENTATIVE_CASES": previous["representative_cases"],
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
        prompt = render_prompt(
            analysis_template,
            self._analysis_replacements(
                previous=previous,
                environment_constants=environment_constants,
                current_observable_initial_state=current_observable_initial_state,
                settings=settings,
            ),
        )
        # The complete core summaries are already compact aggregates.  Cases
        # and representative links have finite caps, so no content is silently
        # truncated here; the normal prompt preflight owns any overflow.
        del system_prompt
        return prompt

    def _guidance_replacements(
        self,
        *,
        analysis: str,
        previous: dict[str, Any],
        environment_constants: dict[str, Any],
        current_observable_initial_state: dict[str, Any],
        settings: dict[str, Any],
        validation_feedback: str,
    ) -> dict[str, Any]:
        guidance = previous["previous_complete_guidance"]
        return {
            "ANALYSIS_STAGE_OUTPUT": analysis,
            "PHASE_AND_EXPLORATION_SETTINGS": settings,
            "CURRENT_OBSERVABLE_INITIAL_STATE": current_observable_initial_state,
            "PREVIOUS_COMPLETE_TD3_BIAS": guidance["td3_bias"],
            "PREVIOUS_COMPLETE_ROUTING_PRIOR": guidance["routing_prior"],
            "PREVIOUS_APPLICATION_STATUS": previous[
                "previous_application_status"
            ],
            "CONCISE_OUTCOMES_AND_CASES": {
                "episode_outcomes": previous["episode_outcomes"],
                "actual_slot_service": previous[
                    "actual_slot_service_summary"
                ],
                "representative_cases": previous["representative_cases"],
            },
            "VALIDATION_FEEDBACK_OR_EMPTY": (
                validation_feedback
                or "No local format-validation feedback is present."
            ),
        }

    def _parse_guidance_response(
        self, content: str, sigma_episode_start: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, str]]:
        return parse_guidance_response(content, sigma_episode_start)

    def _guidance_cache_content(
        self, td3_bias: np.ndarray, routing_prior: np.ndarray
    ) -> str:
        return _compact_json(
            {"td3_bias": td3_bias.tolist(), "routing_prior": routing_prior.tolist()}
        )

    def _routing_guidance_required(self) -> bool:
        """Whether the numerical response contract includes a routing layer."""

        return True

    def _cache_identity(
        self, *, episode_index: int, stage: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        identity = {
            "run_identity": self.run_identity,
            "episode_index": int(episode_index),
            "stage": str(stage),
            "complete_input_sha256": _sha256_text(system_prompt + "\n" + user_prompt),
            "model": self.config.model,
            "prompt_version": self.config.prompt_version,
            "config_sha256": _sha256_text(_canonical_json(asdict(self.config))),
        }
        if self.schema_version == LLM_EXPLORER_VERSION:
            identity.update(
                summary_version=LLM_EXPLORER_SUMMARY_VERSION,
                cache_version=LLM_EXPLORER_CACHE_VERSION,
                routing_epsilon_branch="100_percent_masked_llm_prior",
            )
        return identity

    def _stable_seed(
        self, *, episode_index: int, stage: str, attempt: int, input_sha256: str
    ) -> int:
        material = _canonical_json(
            {
                "run_identity": self.run_identity,
                "episode_index": int(episode_index),
                "stage": str(stage),
                "attempt": int(attempt),
                "input_sha256": str(input_sha256),
                "prompt_version": self.config.prompt_version,
            }
        )
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**31 - 1)

    def _stage_display_name(self, stage: str) -> str:
        return str(stage)

    def _before_request(self, *, episode_index: int, stage: str, attempt: int) -> None:
        """Optional method-specific progress reporting before an API request."""

        del episode_index, stage, attempt

    def _prompt_budget(
        self,
        *,
        episode_index: int,
        stage: str,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        attempt_root: Path | None,
    ) -> dict[str, Any]:
        budget = prompt_token_budget(
            system_prompt,
            user_prompt,
            reserved_output_tokens=max_output_tokens,
            context_window_tokens=self.config.context_window_tokens,
            safety_margin_tokens=self.config.safety_margin_tokens,
            encoding_name=self.config.tokenizer_encoding,
        )
        budget.update(
            episode=episode_index + 1,
            stage=str(stage),
            attempt=int(attempt),
            complete_input_sha256=_sha256_text(system_prompt + "\n" + user_prompt),
        )
        if attempt_root is not None:
            try:
                _atomic_json(attempt_root / "token_budget.json", budget)
            except OSError as exc:
                self._record_local_failure(
                    episode_index=episode_index,
                    stage=stage,
                    attempt=attempt,
                    category="artifact_write",
                    reason=f"token budget artifact: {type(exc).__name__}: {exc}",
                )
        if not budget["fits"]:
            self._record_local_failure(
                episode_index=episode_index,
                stage=stage,
                attempt=attempt,
                category="local_preflight",
                reason=(
                    f"prompt requires {budget['total_budgeted_tokens']} tokens including "
                    f"reserved output and margin; context limit is "
                    f"{budget['context_window_tokens']}"
                ),
            )
            raise PromptBudgetError(
                f"{stage} prompt exceeds context window: "
                f"{budget['total_budgeted_tokens']} > {budget['context_window_tokens']}"
            )
        return budget

    def _record_local_failure(
        self,
        *,
        episode_index: int,
        stage: str,
        attempt: int,
        category: str,
        reason: str,
    ) -> None:
        failure_id = _sha256_text(
            f"{episode_index}|{stage}|{attempt}|{category}|{reason}"
        )
        event = {
            "failure_id": failure_id,
            "episode": episode_index + 1,
            "stage": str(stage),
            "attempt": int(attempt),
            "failure_category": str(category),
            "reason": str(reason),
        }
        self._local_failure_records[failure_id] = event
        self._event(
            "local_preflight_failure" if category == "local_preflight" else "artifact_write_failure",
            **event,
        )
        self._rebuild_statistics()

    @staticmethod
    def _response_completion_error(response: dict[str, Any]) -> str | None:
        finish_reason = response.get("finish_reason")
        if finish_reason != "stop":
            return f"finish_reason={finish_reason!r}, expected 'stop'"
        if response.get("refusal") not in (None, "", False):
            return "model response was refused"
        for key in ("transport_completed", "done_received", "terminal_chunk_received"):
            if key in response and not bool(response.get(key)):
                return f"stream completion flag {key}=false"
        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            return "response content is missing or empty"
        return None

    def _episode_result_path(self, episode_index: int) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"episode_{episode_index + 1:06d}" / "episode_result.json"

    def _load_episode_result(
        self, episode_index: int, identity: dict[str, Any], sigma: float
    ) -> dict[str, Any] | None:
        path = self._episode_result_path(episode_index)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("identity") != identity:
                return None
            guidance = self._restore_guidance(value["guidance"])
            if not guidance.get("td3_fallback"):
                guidance["td3_bias"] = validate_td3_bias(
                    guidance["td3_bias"], sigma
                )
            if not guidance.get("routing_fallback"):
                guidance["routing_prior"] = validate_routing_prior(
                    guidance["routing_prior"]
                )
            guidance["reused"] = True
            guidance["cache_source"] = "episode_result"
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
        try:
            _atomic_json(
                path,
                {
                    "schema_version": self.schema_version,
                    "identity": identity,
                    "guidance": _semantic_guidance(guidance),
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

    def _stage_root(self, episode_index: int, stage: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"episode_{episode_index + 1:06d}" / stage

    def _load_cached(
        self, *, episode_index: int, stage: str, identity: dict[str, Any]
    ) -> dict[str, Any] | None:
        stage_root = self._stage_root(episode_index, stage)
        cache_path = None if stage_root is None else stage_root / "completed.json"
        if cache_path is None or not cache_path.is_file():
            return None
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cached if cached.get("identity") == identity else None

    def _request(
        self,
        *,
        episode_index: int,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        validate: Callable[[str], tuple[bool, Any, str | None]],
    ) -> tuple[Any | None, int | None, bool]:
        identity = self._cache_identity(
            episode_index=episode_index,
            stage=stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        cached = self._load_cached(episode_index=episode_index, stage=stage, identity=identity)
        if cached is not None:
            ok, value, _ = validate(cached["content"])
            if ok:
                return value, int(cached["attempt"]), True

        stage_root = self._stage_root(episode_index, stage)
        if stage_root is not None:
            try:
                stage_root.mkdir(parents=True, exist_ok=True)
                (stage_root / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
                (stage_root / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
                _atomic_json(stage_root / "identity.json", identity)
            except OSError as exc:
                self._record_local_failure(
                    episode_index=episode_index,
                    stage=stage,
                    attempt=0,
                    category="artifact_write",
                    reason=f"stage setup: {type(exc).__name__}: {exc}",
                )
                return None, None, False
        last_reason = "unknown failure"
        for attempt in range(1, int(self.config.max_attempts) + 1):
            attempt_root = None if stage_root is None else stage_root / f"attempt_{attempt:02d}"
            if attempt_root is not None:
                try:
                    attempt_root.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    self._record_local_failure(
                        episode_index=episode_index,
                        stage=stage,
                        attempt=attempt,
                        category="artifact_write",
                        reason=f"attempt directory: {type(exc).__name__}: {exc}",
                    )
                    return None, None, False
            try:
                self._prompt_budget(
                    episode_index=episode_index,
                    stage=stage,
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    attempt_root=attempt_root,
                )
            except PromptBudgetError as exc:
                print(
                    f"[LLM explorer] episode={episode_index + 1} "
                    f"stage={self._stage_display_name(stage)} "
                    f"local_preflight_failed reason={exc} retry=false "
                    f"requests_total={self.statistics['requests_total']} "
                    f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                    f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                    flush=True,
                )
                return None, None, False
            started = time.monotonic()
            response: dict[str, Any] = {}
            input_sha = identity["complete_input_sha256"]
            request_id = (
                f"ep{episode_index + 1}:{stage}:a{attempt}:{input_sha[:16]}:"
                f"{uuid.uuid4().hex}"
            )
            try:
                self._ensure_client()
                self._before_request(
                    episode_index=episode_index, stage=stage, attempt=attempt
                )
                response = self.client.chat(
                    model=self.config.model,
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    temperature=float(self.config.temperature),
                    max_output_tokens=int(max_output_tokens),
                    seed=self._stable_seed(
                        episode_index=episode_index,
                        stage=stage,
                        attempt=attempt,
                        input_sha256=input_sha,
                    ),
                    attempt_directory=(attempt_root or Path(".")),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                _propagate_wrapped_keyboard_interrupt(exc)
                elapsed = time.monotonic() - started
                last_reason = f"{type(exc).__name__}: {exc}"
                self._record_attempt(
                    episode_index, stage, attempt, False, last_reason, elapsed, {},
                    retry_success=False, request_id=request_id,
                    failure_category="api_or_transport",
                )
                will_retry = attempt < int(self.config.max_attempts)
                print(
                    f"[LLM explorer] episode={episode_index + 1} "
                    f"stage={self._stage_display_name(stage)} "
                    f"attempt={attempt} failed reason={last_reason} "
                    f"retry={str(will_retry).lower()} "
                    f"requests_total={self.statistics['requests_total']} "
                    f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                    f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                    flush=True,
                )
                continue

            elapsed = time.monotonic() - started
            completion_error = self._response_completion_error(response)
            content = response.get("content")
            if completion_error is not None:
                ok, value = False, None
                last_reason = completion_error
                failure_category = "response_incomplete"
            else:
                ok, value, validation_reason = validate(str(content))
                last_reason = validation_reason or "response validation failed"
                failure_category = None if ok else "content_validation"
            self._record_attempt(
                episode_index,
                stage,
                attempt,
                bool(ok),
                None if ok else last_reason,
                elapsed,
                response,
                retry_success=bool(ok and attempt > 1),
                request_id=request_id,
                failure_category=failure_category,
            )
            if attempt_root is not None:
                try:
                    (attempt_root / "response.txt").write_text(
                        str(content) if content is not None else "", encoding="utf-8"
                    )
                    _atomic_json(
                        attempt_root / "response_metadata.json",
                        self._safe_response_metadata(response),
                    )
                    if ok:
                        _atomic_json(
                            stage_root / "completed.json",
                            {
                                "identity": identity,
                                "attempt": attempt,
                                "content": content,
                                "response_metadata": self._safe_response_metadata(response),
                            },
                        )
                except OSError as exc:
                    self._record_local_failure(
                        episode_index=episode_index,
                        stage=stage,
                        attempt=attempt,
                        category="artifact_write",
                        reason=f"response artifacts: {type(exc).__name__}: {exc}",
                    )
            if ok:
                return value, attempt, False
            will_retry = attempt < int(self.config.max_attempts)
            print(
                f"[LLM explorer] episode={episode_index + 1} "
                f"stage={self._stage_display_name(stage)} "
                f"attempt={attempt} failed category={failure_category} "
                f"reason={last_reason} retry={str(will_retry).lower()} "
                f"requests_total={self.statistics['requests_total']} "
                f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                flush=True,
            )
        return None, None, False

    @staticmethod
    def _safe_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
        return _json_clone(
            {
                "actual_model": response.get("actual_model"),
                "usage": response.get("usage"),
                "finish_reason": response.get("finish_reason"),
                "refusal": response.get("refusal"),
                "transport_completed": response.get("transport_completed"),
                "done_received": response.get("done_received"),
                "terminal_chunk_received": response.get("terminal_chunk_received"),
            }
        )

    def _record_attempt(
        self,
        episode_index: int,
        stage: str,
        attempt: int,
        success: bool,
        reason: str | None,
        elapsed: float,
        response: dict[str, Any],
        *,
        retry_success: bool,
        request_id: str,
        failure_category: str | None,
    ) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        actual_model = response.get("actual_model") if isinstance(response, dict) else None
        record = {
            "request_id": str(request_id),
            "episode": episode_index + 1,
            "stage": stage,
            "attempt": attempt,
            "success": bool(success),
            "retry_success": bool(retry_success),
            "reason": reason,
            "failure_category": failure_category,
            "elapsed_seconds": float(elapsed),
            "actual_model": actual_model,
            "usage": usage if usage is not None else "unknown",
        }
        self._request_records[str(request_id)] = _json_clone(record)
        self._event("request_attempt", **record)
        self._rebuild_statistics()

    def begin_episode(
        self,
        *,
        episode_index: int,
        total_joint_transitions: int,
        sigma_episode_start: float,
        routing_epsilon: float,
        current_observable_initial_state: dict[str, Any],
        environment_constants: dict[str, Any],
    ) -> dict[str, Any]:
        self.movement_records = []
        self.routing_records = []
        self.slot_service_records = []
        self._upcoming_episode = int(episode_index) + 1
        self._total_joint_transitions = int(total_joint_transitions)
        routing_guidance_required = self._routing_guidance_required()
        if self.evaluation:
            self.active_guidance = self._unguided("evaluation_disabled")
            return self.active_guidance
        if int(total_joint_transitions) < int(self.config.warmup_joint_transitions):
            self.active_guidance = self._unguided("warmup")
            return self.active_guidance
        if self.previous_episode_feedback is None:
            self.active_guidance = self._unguided("no_previous_episode_feedback")
            return self.active_guidance

        system_template, analysis_template, guidance_template = self._templates()
        system_template = self._render_system_prompt(
            system_template, environment_constants
        )
        settings = self._exploration_settings(
            sigma_episode_start=sigma_episode_start,
            routing_epsilon=routing_epsilon,
        )
        previous = self._compact_feedback(self.previous_episode_feedback)
        analysis_prompt = self._render_analysis_prompt(
            system_prompt=system_template,
            analysis_template=analysis_template,
            previous=previous,
            environment_constants=environment_constants,
            current_observable_initial_state=current_observable_initial_state,
            settings=settings,
        )
        episode_identity = self._cache_identity(
            episode_index=episode_index,
            stage="episode_result",
            system_prompt=system_template,
            user_prompt=analysis_prompt,
        )
        replayed = self._load_episode_result(
            episode_index, episode_identity, sigma_episode_start
        )
        if replayed is not None:
            self.active_guidance = replayed
            return self.active_guidance

        def analysis_validator(content: str) -> tuple[bool, Any, str | None]:
            clean = str(content).strip()
            if not clean:
                return False, None, "analysis response is empty"
            return True, clean, None

        analysis, analysis_attempt, analysis_reused = self._request(
            episode_index=episode_index,
            stage="analysis",
            system_prompt=system_template,
            user_prompt=analysis_prompt,
            max_output_tokens=self.config.analysis_max_output_tokens,
            validate=analysis_validator,
        )
        if analysis is None:
            self.active_guidance = self._unguided("analysis_stage_failed")
            self._record_fallback_episode(
                episode_index, td3=True, routing=routing_guidance_required
            )
            self._save_episode_result(
                episode_index=episode_index,
                identity=episode_identity,
                guidance=self.active_guidance,
            )
            return self.active_guidance

        feedback = ""
        guidance_prompt = render_prompt(
            guidance_template,
            self._guidance_replacements(
                analysis=analysis,
                previous=previous,
                environment_constants=environment_constants,
                current_observable_initial_state=current_observable_initial_state,
                settings=settings,
                validation_feedback=feedback,
            ),
        )

        # Guidance validation stays here so a valid layer from attempt one
        # survives a retry for only the missing layer.
        accepted_td3 = accepted_routing = None
        td3_attempt = routing_attempt = None
        stage_root = self._stage_root(episode_index, "guidance")
        identity = self._cache_identity(
            episode_index=episode_index,
            stage="guidance",
            system_prompt=system_template,
            user_prompt=guidance_prompt,
        )
        cached = self._load_cached(episode_index=episode_index, stage="guidance", identity=identity)
        guidance_reused = False
        if cached is not None:
            td3, routing, errors = self._parse_guidance_response(
                cached["content"], sigma_episode_start
            )
            if (
                not errors
                and td3 is not None
                and (not routing_guidance_required or routing is not None)
            ):
                accepted_td3, accepted_routing = td3, routing
                td3_attempt = int(
                    cached.get("td3_source_attempt", cached["attempt"])
                )
                if routing_guidance_required:
                    routing_attempt = int(
                        cached.get("routing_source_attempt", cached["attempt"])
                    )
                guidance_reused = True
        if accepted_td3 is None or (
            routing_guidance_required and accepted_routing is None
        ):
            if stage_root is not None:
                try:
                    stage_root.mkdir(parents=True, exist_ok=True)
                    (stage_root / "system_prompt.txt").write_text(system_template, encoding="utf-8")
                    (stage_root / "user_prompt.txt").write_text(guidance_prompt, encoding="utf-8")
                    _atomic_json(stage_root / "identity.json", identity)
                except OSError as exc:
                    self._record_local_failure(
                        episode_index=episode_index,
                        stage="guidance",
                        attempt=0,
                        category="artifact_write",
                        reason=f"stage setup: {type(exc).__name__}: {exc}",
                    )
            had_failure = False
            for attempt in range(1, self.config.max_attempts + 1):
                if stage_root is not None and not stage_root.is_dir():
                    break
                attempt_root = None if stage_root is None else stage_root / f"attempt_{attempt:02d}"
                if attempt_root is not None:
                    try:
                        attempt_root.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        self._record_local_failure(
                            episode_index=episode_index,
                            stage="guidance",
                            attempt=attempt,
                            category="artifact_write",
                            reason=f"attempt directory: {type(exc).__name__}: {exc}",
                        )
                        break
                attempt_prompt = guidance_prompt
                if attempt > 1 and feedback:
                    attempt_prompt = render_prompt(
                        guidance_template,
                        self._guidance_replacements(
                            analysis=analysis,
                            previous=previous,
                            environment_constants=environment_constants,
                            current_observable_initial_state=current_observable_initial_state,
                            settings=settings,
                            validation_feedback=feedback,
                        ),
                    )
                try:
                    self._prompt_budget(
                        episode_index=episode_index,
                        stage="guidance",
                        attempt=attempt,
                        system_prompt=system_template,
                        user_prompt=attempt_prompt,
                        max_output_tokens=self.config.guidance_max_output_tokens,
                        attempt_root=attempt_root,
                    )
                except PromptBudgetError as exc:
                    feedback = str(exc)
                    print(
                        f"[LLM explorer] episode={episode_index + 1} "
                        f"stage={self._stage_display_name('guidance')} "
                        f"local_preflight_failed reason={exc} retry=false "
                        f"requests_total={self.statistics['requests_total']} "
                        f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                        f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                        flush=True,
                    )
                    break
                started = time.monotonic()
                response: dict[str, Any] = {}
                input_sha = _sha256_text(system_template + "\n" + attempt_prompt)
                request_id = (
                    f"ep{episode_index + 1}:guidance:a{attempt}:{input_sha[:16]}:"
                    f"{uuid.uuid4().hex}"
                )
                try:
                    self._ensure_client()
                    self._before_request(
                        episode_index=episode_index,
                        stage="guidance",
                        attempt=attempt,
                    )
                    response = self.client.chat(
                        model=self.config.model,
                        system_prompt=system_template,
                        prompt=attempt_prompt,
                        temperature=float(self.config.temperature),
                        max_output_tokens=int(self.config.guidance_max_output_tokens),
                        seed=self._stable_seed(
                            episode_index=episode_index,
                            stage="guidance",
                            attempt=attempt,
                            input_sha256=input_sha,
                        ),
                        attempt_directory=(attempt_root or Path(".")),
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    _propagate_wrapped_keyboard_interrupt(exc)
                    had_failure = True
                    reason = f"{type(exc).__name__}: {exc}"
                    self._record_attempt(
                        episode_index,
                        "guidance",
                        attempt,
                        False,
                        reason,
                        time.monotonic() - started,
                        {},
                        retry_success=False,
                        request_id=request_id,
                        failure_category="api_or_transport",
                    )
                    feedback = reason
                    print(
                        f"[LLM explorer] episode={episode_index + 1} "
                        f"stage={self._stage_display_name('guidance')} "
                        f"attempt={attempt} failed reason={reason} "
                        f"retry={str(attempt < self.config.max_attempts).lower()} "
                        f"requests_total={self.statistics['requests_total']} "
                        f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                        f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                        flush=True,
                    )
                    continue

                completion_error = self._response_completion_error(response)
                content = response.get("content")
                if completion_error is None:
                    td3, routing, errors = self._parse_guidance_response(
                        str(content), sigma_episode_start
                    )
                    if accepted_td3 is None and td3 is not None:
                        accepted_td3, td3_attempt = td3, attempt
                    if (
                        routing_guidance_required
                        and accepted_routing is None
                        and routing is not None
                    ):
                        accepted_routing, routing_attempt = routing, attempt
                    success = accepted_td3 is not None and (
                        not routing_guidance_required
                        or accepted_routing is not None
                    )
                    missing_errors = {
                        layer: message
                        for layer, message in errors.items()
                        if (layer == "td3" and accepted_td3 is None)
                        or (
                            routing_guidance_required
                            and layer == "routing"
                            and accepted_routing is None
                        )
                    }
                    reason = None if success else "; ".join(
                        f"{layer}: {message}"
                        for layer, message in sorted(missing_errors.items())
                    ) or "one or more guidance layers remain invalid"
                    category = None if success else "content_validation"
                else:
                    td3 = routing = None
                    errors = {"response": completion_error}
                    success = False
                    reason = completion_error
                    category = "response_incomplete"
                self._record_attempt(
                    episode_index,
                    "guidance",
                    attempt,
                    success,
                    reason,
                    time.monotonic() - started,
                    response,
                    retry_success=success and had_failure,
                    request_id=request_id,
                    failure_category=category,
                )
                if not success:
                    had_failure = True
                if attempt_root is not None:
                    try:
                        (attempt_root / "prompt.txt").write_text(attempt_prompt, encoding="utf-8")
                        (attempt_root / "response.txt").write_text(
                            str(content) if content is not None else "", encoding="utf-8"
                        )
                        _atomic_json(
                            attempt_root / "validation.json",
                            {
                                "errors": errors,
                                "accepted_td3_this_attempt": td3 is not None,
                                "routing_guidance_required": routing_guidance_required,
                                "accepted_routing_this_attempt": (
                                    routing is not None
                                    if routing_guidance_required
                                    else None
                                ),
                            },
                        )
                        _atomic_json(
                            attempt_root / "response_metadata.json",
                            self._safe_response_metadata(response),
                        )
                        if success:
                            _atomic_json(
                                stage_root / "completed.json",
                                {
                                    "identity": identity,
                                    "attempt": attempt,
                                    "content": self._guidance_cache_content(
                                        accepted_td3, accepted_routing
                                    ),
                                    "td3_source_attempt": td3_attempt,
                                    "routing_source_attempt": routing_attempt,
                                    "response_metadata": self._safe_response_metadata(response),
                                },
                            )
                    except OSError as exc:
                        self._record_local_failure(
                            episode_index=episode_index,
                            stage="guidance",
                            attempt=attempt,
                            category="artifact_write",
                            reason=f"response artifacts: {type(exc).__name__}: {exc}",
                        )
                if success:
                    break
                feedback = reason or "unknown validation error"
                print(
                    f"[LLM explorer] episode={episode_index + 1} "
                    f"stage={self._stage_display_name('guidance')} "
                    f"attempt={attempt} failed category={category} reason={feedback} "
                    f"retry={str(attempt < self.config.max_attempts).lower()} "
                    f"requests_total={self.statistics['requests_total']} "
                    f"failed_attempts_total={self.statistics['failed_attempts_total']} "
                    f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
                    flush=True,
                )

        td3_fallback = accepted_td3 is None
        routing_fallback = bool(
            routing_guidance_required and accepted_routing is None
        )
        self.active_guidance = {
            "enabled": True,
            "td3_bias": (
                np.zeros((NUM_UAV, 3), dtype=np.float32)
                if td3_fallback
                else accepted_td3
            ),
            "routing_prior": (
                None
                if not routing_guidance_required or routing_fallback
                else accepted_routing
            ),
            "td3_fallback": td3_fallback,
            "routing_fallback": routing_fallback,
            "td3_reason": "guidance_invalid_or_failed" if td3_fallback else None,
            "routing_reason": (
                "guidance_invalid_or_failed"
                if routing_fallback
                else "routing_guidance_disabled"
                if not routing_guidance_required
                else None
            ),
            "analysis_attempt": analysis_attempt,
            "td3_guidance_attempt": td3_attempt,
            "routing_guidance_attempt": routing_attempt,
            "reused": bool(analysis_reused and guidance_reused),
        }
        if td3_fallback or routing_fallback:
            self._record_fallback_episode(
                episode_index, td3=td3_fallback, routing=routing_fallback
            )
        self._save_episode_result(
            episode_index=episode_index,
            identity=episode_identity,
            guidance=self.active_guidance,
        )
        print(
            f"[LLM explorer] episode={episode_index + 1} update_complete "
            f"analysis_attempt={analysis_attempt} td3_attempt={td3_attempt} "
            f"routing_attempt={routing_attempt} td3_fallback={td3_fallback} "
            f"routing_fallback={routing_fallback} requests_total={self.statistics['requests_total']}",
            flush=True,
        )
        return self.active_guidance

    def _record_fallback_episode(self, episode_index: int, *, td3: bool, routing: bool) -> None:
        key = str(episode_index + 1)
        record = {
            "episode": episode_index + 1,
            "td3": bool(td3),
            "routing": bool(routing),
        }
        existing = self._fallback_records.get(key)
        if existing is not None:
            record["td3"] = bool(existing.get("td3")) or bool(td3)
            record["routing"] = bool(existing.get("routing")) or bool(routing)
        self._fallback_records[key] = record
        if existing != record:
            self._event("episode_fallback", **record)
        self._rebuild_statistics()

    def record_zero_legal_prior_mass(self) -> None:
        self.statistics["routing_zero_legal_prior_mass_total"] += 1

    def record_movement(self, record: dict[str, Any]) -> None:
        self.movement_records.append(_json_clone(record))

    def record_routing(self, record: dict[str, Any]) -> None:
        copied = _json_clone(record)
        copied.setdefault("decision_index", len(self.routing_records))
        self.routing_records.append(copied)

    def record_slot_service(self, record: dict[str, Any]) -> None:
        self.slot_service_records.append(_json_clone(record))

    def complete_episode(self, *, episode_index: int, metrics: dict[str, Any]) -> None:
        feedback = assemble_first_gen_feedback(
            metrics=metrics,
            movement_records=self.movement_records,
            routing_records=self.routing_records,
            slot_service_records=self.slot_service_records,
            applied_guidance=self._serializable_guidance(self.active_guidance),
        )
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
            f"[LLM explorer] episode={episode_index + 1} summary "
            f"movement_samples={len(self.movement_records)} routing_decisions={len(self.routing_records)} "
            f"routing_slots={len(self.slot_service_records)} cases={len(feedback['representative_cases'])} "
            f"requests_total={self.statistics['requests_total']} "
            f"failed_attempts_total={self.statistics['failed_attempts_total']} "
            f"fallback_episodes_total={self.statistics['fallback_episodes_total']}",
            flush=True,
        )


def explorer_metadata(config: LLMExplorerConfig) -> dict[str, Any]:
    return {
        "schema_version": LLM_EXPLORER_VERSION,
        "method_id": LLM_EXPLORER_METHOD_ID,
        "prompt_version": config.prompt_version,
        "summary_version": LLM_EXPLORER_SUMMARY_VERSION,
        "case_selection_version": LLM_EXPLORER_CASE_VERSION,
        "cache_version": LLM_EXPLORER_CACHE_VERSION,
        "model": config.model,
        "warmup_joint_transitions": config.warmup_joint_transitions,
        "update_interval_episodes": config.update_interval_episodes,
        "max_attempts_per_stage": config.max_attempts,
        "request_timeout_seconds": config.request_timeout_seconds,
        "context_window_tokens": config.context_window_tokens,
        "safety_margin_tokens": config.safety_margin_tokens,
        "td3_intervention": "episode_fixed_raw_bias_added_once_with_original_gaussian_before_projection",
        "routing_intervention": "epsilon_branch_100_percent_masked_renormalized_prior; legal_uniform_only_on_zero_legal_mass",
        "guidance_weight": 1.0,
        "annealing_end_episode": None,
        "evaluation_calls_llm": False,
    }
