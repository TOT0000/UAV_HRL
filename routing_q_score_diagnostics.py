"""Read-only safe-DDQN Q-score diagnostics for paper evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION = (
    "uav-hrl-routing-q-score-diagnostics-v1"
)
ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS = {
    "routing_q_r_definition": "raw reward critic output Q_r(s,a)",
    "routing_q_c_definition": (
        "raw unconstrained cost critic output Q_c(s,a), which may be negative; "
        "no absolute value, activation, clipping, or clamp is applied"
    ),
    "routing_q_safe_definition": (
        "Q_safe(s,a) = Q_r(s,a) - lambda_cost_used * Q_c(s,a)"
    ),
    "routing_q_wait_definition": "Wait is action index == sender UAV ID",
    "routing_q_voluntary_wait_definition": (
        "selected Wait while at least one effective-mask legal non-Wait action exists"
    ),
    "routing_q_reward_argmax_definition": (
        "NumPy deterministic argmax of raw Q_r over effective-mask legal actions"
    ),
    "routing_q_safe_argmax_definition": (
        "NumPy deterministic argmax of Q_safe over effective-mask legal actions"
    ),
    "routing_q_cost_induced_flip_definition": (
        "reward argmax is non-Wait while safe argmax is Wait"
    ),
    "routing_q_negative_qc_definition": "raw learned Q_c value < 0",
}

VOLUNTARY_WAIT_SCORE_FIELDS = (
    "q_r_wait",
    "q_c_wait",
    "q_safe_wait",
    "q_r_best_safe_forward",
    "q_c_best_safe_forward",
    "q_safe_best_safe_forward",
    "q_safe_wait_minus_best_forward",
    "q_r_wait_minus_best_forward",
    "cost_contribution_margin_against_best_safe_forward",
)
VOLUNTARY_WAIT_EVENT_FIELDS = (
    "scenario_id",
    "episode_index",
    "slot_index",
    "time_seconds",
    "sender_uav_id",
    "hol_task_type",
    "lambda_cost_used",
    "selected_action",
    "reward_argmax_action",
    "safe_argmax_action",
    "legal_action_count",
    "legal_nonwait_action_count",
    "q_r_wait",
    "q_c_wait",
    "q_safe_wait",
    "cost_term_wait",
    "q_c_wait_is_negative",
    "best_qr_forward_action",
    "q_r_best_qr_forward",
    "q_c_best_qr_forward",
    "q_safe_best_qr_forward",
    "best_safe_forward_action",
    "q_r_best_safe_forward",
    "q_c_best_safe_forward",
    "q_safe_best_safe_forward",
    "reward_also_prefers_wait",
    "cost_induced_forward_to_wait_flip",
    "q_safe_wait_minus_best_forward",
    "q_r_wait_minus_best_forward",
    "cost_contribution_margin_against_best_safe_forward",
)
_DISTRIBUTION_STATISTICS = (
    "mean",
    "std",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
)


def _masked_argmax(values, legal_mask):
    masked = np.full(values.shape, -np.inf, dtype=np.float64)
    masked[legal_mask] = values[legal_mask]
    return int(np.argmax(masked))


def _empty_group():
    return {
        "total_routing_q_decisions": 0,
        "selected_wait_count": 0,
        "selected_forward_count": 0,
        "selected_action_qc_negative_count": 0,
        "legal_qc_value_count": 0,
        "legal_qc_negative_value_count": 0,
        "reward_argmax_wait_count": 0,
        "safe_argmax_wait_count": 0,
        "reward_argmax_forward_but_safe_argmax_wait_count": 0,
        "voluntary_wait_count": 0,
        "voluntary_wait_qc_wait_negative_count": 0,
        "voluntary_wait_reward_also_prefers_wait_count": 0,
        "voluntary_wait_cost_induced_flip_count": 0,
        "cost_induced_flip_and_qc_wait_negative_count": 0,
        "samples": {field: [] for field in VOLUNTARY_WAIT_SCORE_FIELDS},
    }


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _distribution(values):
    if not values:
        return {statistic: None for statistic in _DISTRIBUTION_STATISTICS}
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("routing Q-score diagnostic distribution is non-finite")
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
    }


class RoutingQScoreDiagnosticAccumulator:
    """Aggregate all decisions and retain only voluntary-Wait scalar rows."""

    def __init__(self):
        self._groups = {name: _empty_group() for name in ("ALL", "COM", "FOV")}
        self.voluntary_wait_events = []

    def add_decision(
        self,
        inspection,
        *,
        selected_action,
        sender_uav_id,
        hol_task_type,
        scenario_id,
        episode_index,
        slot_index,
        time_seconds,
    ):
        task_type = str(hol_task_type).upper()
        if task_type not in {"COM", "FOV"}:
            raise ValueError("routing Q-score HOL task type must be COM or FOV")
        q_r_native = np.asarray(inspection["q_r"])
        q_c_native = np.asarray(inspection["q_c"])
        q_safe_native = np.asarray(inspection["q_safe"])
        legal_mask = np.asarray(inspection["legal_mask"], dtype=bool)
        if q_r_native.ndim != 1 or not (
            q_r_native.shape
            == q_c_native.shape
            == q_safe_native.shape
            == legal_mask.shape
        ):
            raise ValueError("routing Q-score arrays must be aligned one-dimensional values")
        for name, values in (
            ("Q_r", q_r_native),
            ("Q_c", q_c_native),
            ("Q_safe", q_safe_native),
        ):
            if not np.issubdtype(values.dtype, np.floating):
                raise TypeError(f"routing {name} values must have a floating dtype")
        if not legal_mask.any():
            raise ValueError("routing Q-score diagnostic has no legal action")
        for name, values in (
            ("Q_r", q_r_native),
            ("Q_c", q_c_native),
            ("Q_safe", q_safe_native),
        ):
            if not np.all(np.isfinite(values[legal_mask])):
                raise ValueError(f"legal routing {name} values must be finite")

        lambda_cost = float(inspection["lambda_cost_used"])
        if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
            raise ValueError("routing Q-score lambda_cost_used must be finite and non-negative")
        q_r_legal_native = q_r_native[legal_mask]
        q_c_legal_native = q_c_native[legal_mask]
        q_safe_legal_native = q_safe_native[legal_mask]
        cost_term_legal_native = lambda_cost * q_c_legal_native
        expected_safe_legal_native = q_r_legal_native - cost_term_legal_native
        if expected_safe_legal_native.dtype != q_safe_legal_native.dtype:
            raise TypeError(
                "routing Q_safe dtype differs from native Q_r - lambda_cost * Q_c"
            )
        if not np.all(np.isfinite(expected_safe_legal_native)):
            raise ValueError("expected legal routing Q_safe values must be finite")
        # Bound native multiply/subtract roundoff by machine epsilon and the
        # largest operand/result magnitude; 32 eps is deliberately conservative.
        scale = max(
            1.0,
            float(np.max(np.abs(q_r_legal_native).astype(np.float64))),
            float(np.max(np.abs(cost_term_legal_native).astype(np.float64))),
            float(np.max(np.abs(q_safe_legal_native).astype(np.float64))),
        )
        native_atol = 32.0 * float(np.finfo(q_safe_native.dtype).eps) * scale
        if not np.allclose(
            q_safe_legal_native,
            expected_safe_legal_native,
            rtol=0.0,
            atol=native_atol,
        ):
            raise AssertionError("routing Q_safe does not equal Q_r - lambda_cost * Q_c")

        q_r = q_r_native.astype(np.float64, copy=False)
        q_c = q_c_native.astype(np.float64, copy=False)
        q_safe = q_safe_native.astype(np.float64, copy=False)

        sender = int(sender_uav_id)
        selected = int(selected_action)
        if not 0 <= sender < q_r.size or not bool(legal_mask[sender]):
            raise ValueError("routing Q-score Wait action must be legal")
        if not 0 <= selected < q_r.size or not bool(legal_mask[selected]):
            raise ValueError("routing Q-score selected action must be legal")
        reward_argmax = _masked_argmax(q_r, legal_mask)
        safe_argmax = _masked_argmax(q_safe, legal_mask)
        if int(inspection["reward_argmax_action"]) != reward_argmax:
            raise AssertionError("diagnostic reward argmax is inconsistent")
        if int(inspection["safe_argmax_action"]) != safe_argmax:
            raise AssertionError("diagnostic safe argmax is inconsistent")
        if safe_argmax != selected:
            raise AssertionError(
                "evaluation diagnostic safe argmax differs from production action"
            )

        forward_mask = legal_mask.copy()
        forward_mask[sender] = False
        legal_count = int(np.count_nonzero(legal_mask))
        legal_forward_count = int(np.count_nonzero(forward_mask))
        global_cost_flip = reward_argmax != sender and safe_argmax == sender
        selected_qc_negative = bool(q_c[selected] < 0.0)

        for group_name in ("ALL", task_type):
            group = self._groups[group_name]
            group["total_routing_q_decisions"] += 1
            group[
                "selected_wait_count" if selected == sender else "selected_forward_count"
            ] += 1
            group["selected_action_qc_negative_count"] += int(selected_qc_negative)
            group["legal_qc_value_count"] += legal_count
            group["legal_qc_negative_value_count"] += int(
                np.count_nonzero(q_c[legal_mask] < 0.0)
            )
            group["reward_argmax_wait_count"] += int(reward_argmax == sender)
            group["safe_argmax_wait_count"] += int(safe_argmax == sender)
            group[
                "reward_argmax_forward_but_safe_argmax_wait_count"
            ] += int(global_cost_flip)

        if selected != sender or legal_forward_count == 0:
            return None

        best_qr_forward = _masked_argmax(q_r, forward_mask)
        best_safe_forward = _masked_argmax(q_safe, forward_mask)
        q_c_wait_negative = bool(q_c[sender] < 0.0)
        reward_prefers_wait = reward_argmax == sender
        cost_flip = reward_argmax != sender and safe_argmax == sender
        event = {
            "scenario_id": scenario_id,
            "episode_index": int(episode_index),
            "slot_index": int(slot_index),
            "time_seconds": float(time_seconds),
            "sender_uav_id": sender,
            "hol_task_type": task_type,
            "lambda_cost_used": lambda_cost,
            "selected_action": selected,
            "reward_argmax_action": reward_argmax,
            "safe_argmax_action": safe_argmax,
            "legal_action_count": legal_count,
            "legal_nonwait_action_count": legal_forward_count,
            "q_r_wait": float(q_r[sender]),
            "q_c_wait": float(q_c[sender]),
            "q_safe_wait": float(q_safe[sender]),
            "cost_term_wait": float(-lambda_cost * q_c[sender]),
            "q_c_wait_is_negative": q_c_wait_negative,
            "best_qr_forward_action": best_qr_forward,
            "q_r_best_qr_forward": float(q_r[best_qr_forward]),
            "q_c_best_qr_forward": float(q_c[best_qr_forward]),
            "q_safe_best_qr_forward": float(q_safe[best_qr_forward]),
            "best_safe_forward_action": best_safe_forward,
            "q_r_best_safe_forward": float(q_r[best_safe_forward]),
            "q_c_best_safe_forward": float(q_c[best_safe_forward]),
            "q_safe_best_safe_forward": float(q_safe[best_safe_forward]),
            "reward_also_prefers_wait": reward_prefers_wait,
            "cost_induced_forward_to_wait_flip": cost_flip,
            "q_safe_wait_minus_best_forward": float(
                q_safe[sender] - q_safe[best_safe_forward]
            ),
            "q_r_wait_minus_best_forward": float(
                q_r[sender] - q_r[best_qr_forward]
            ),
            "cost_contribution_margin_against_best_safe_forward": float(
                -lambda_cost * q_c[sender]
                - (-lambda_cost * q_c[best_safe_forward])
            ),
        }
        if tuple(event) != VOLUNTARY_WAIT_EVENT_FIELDS:
            raise AssertionError("voluntary-Wait event schema is inconsistent")
        if any(
            isinstance(event[field], float) and not math.isfinite(event[field])
            for field in event
        ):
            raise ValueError("voluntary-Wait event contains a non-finite scalar")
        self.voluntary_wait_events.append(event)

        for group_name in ("ALL", task_type):
            group = self._groups[group_name]
            group["voluntary_wait_count"] += 1
            group["voluntary_wait_qc_wait_negative_count"] += int(
                q_c_wait_negative
            )
            group["voluntary_wait_reward_also_prefers_wait_count"] += int(
                reward_prefers_wait
            )
            group["voluntary_wait_cost_induced_flip_count"] += int(cost_flip)
            group["cost_induced_flip_and_qc_wait_negative_count"] += int(
                cost_flip and q_c_wait_negative
            )
            for field in VOLUNTARY_WAIT_SCORE_FIELDS:
                group["samples"][field].append(float(event[field]))
        return event

    def summary(self):
        groups = {}
        for name, raw in self._groups.items():
            decisions = raw["total_routing_q_decisions"]
            voluntary = raw["voluntary_wait_count"]
            flips = raw["voluntary_wait_cost_induced_flip_count"]
            group = {
                "task_type": name,
                "total_routing_q_decisions": decisions,
                "selected_wait_count": raw["selected_wait_count"],
                "selected_wait_fraction": _ratio(
                    raw["selected_wait_count"], decisions
                ),
                "selected_forward_count": raw["selected_forward_count"],
                "selected_forward_fraction": _ratio(
                    raw["selected_forward_count"], decisions
                ),
                "selected_action_qc_negative_count": raw[
                    "selected_action_qc_negative_count"
                ],
                "selected_action_qc_negative_fraction": _ratio(
                    raw["selected_action_qc_negative_count"], decisions
                ),
                "legal_qc_value_count": raw["legal_qc_value_count"],
                "legal_qc_negative_value_count": raw[
                    "legal_qc_negative_value_count"
                ],
                "legal_qc_negative_fraction": _ratio(
                    raw["legal_qc_negative_value_count"],
                    raw["legal_qc_value_count"],
                ),
                "reward_argmax_wait_count": raw["reward_argmax_wait_count"],
                "reward_argmax_wait_fraction": _ratio(
                    raw["reward_argmax_wait_count"], decisions
                ),
                "safe_argmax_wait_count": raw["safe_argmax_wait_count"],
                "safe_argmax_wait_fraction": _ratio(
                    raw["safe_argmax_wait_count"], decisions
                ),
                "reward_argmax_forward_but_safe_argmax_wait_count": raw[
                    "reward_argmax_forward_but_safe_argmax_wait_count"
                ],
                "reward_argmax_forward_but_safe_argmax_wait_fraction": _ratio(
                    raw["reward_argmax_forward_but_safe_argmax_wait_count"],
                    decisions,
                ),
                "voluntary_wait_count": voluntary,
                "voluntary_wait_fraction": _ratio(voluntary, decisions),
                "voluntary_wait_qc_wait_negative_count": raw[
                    "voluntary_wait_qc_wait_negative_count"
                ],
                "voluntary_wait_qc_wait_negative_fraction": _ratio(
                    raw["voluntary_wait_qc_wait_negative_count"], voluntary
                ),
                "voluntary_wait_reward_also_prefers_wait_count": raw[
                    "voluntary_wait_reward_also_prefers_wait_count"
                ],
                "voluntary_wait_reward_also_prefers_wait_fraction": _ratio(
                    raw["voluntary_wait_reward_also_prefers_wait_count"],
                    voluntary,
                ),
                "voluntary_wait_cost_induced_flip_count": flips,
                "voluntary_wait_cost_induced_flip_fraction": _ratio(
                    flips, voluntary
                ),
                "cost_induced_flip_and_qc_wait_negative_count": raw[
                    "cost_induced_flip_and_qc_wait_negative_count"
                ],
                "cost_induced_flip_and_qc_wait_negative_fraction_of_flips": _ratio(
                    raw["cost_induced_flip_and_qc_wait_negative_count"], flips
                ),
            }
            if group["selected_wait_count"] + group["selected_forward_count"] != decisions:
                raise AssertionError("routing Q-score selected-action conservation failed")
            for field in VOLUNTARY_WAIT_SCORE_FIELDS:
                statistics = _distribution(raw["samples"][field])
                group.update(
                    {
                        f"{field}_{statistic}": value
                        for statistic, value in statistics.items()
                    }
                )
            groups[name] = group
        return {
            "routing_q_score_diagnostic_contract_version": (
                ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION
            ),
            **ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS,
            "groups": groups,
        }


def write_routing_q_score_diagnostic_artifacts(
    output_directory, diagnostics, voluntary_wait_events
):
    output_directory = Path(output_directory)
    json_path = output_directory / "routing_q_score_diagnostics.json"
    csv_path = output_directory / "routing_q_score_diagnostics.csv"
    voluntary_path = output_directory / "routing_q_score_voluntary_waits.csv"
    json_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    groups = diagnostics["groups"]
    rows = [groups[name] for name in ("ALL", "COM", "FOV")]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with voluntary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VOLUNTARY_WAIT_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(voluntary_wait_events)
    return {
        "routing_q_score_diagnostics_json": json_path.resolve(),
        "routing_q_score_diagnostics_csv": csv_path.resolve(),
        "routing_q_score_voluntary_waits_csv": voluntary_path.resolve(),
    }
