"""Read-only safe-DDQN Q-score diagnostics for paper evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION = (
    "uav-hrl-routing-q-score-diagnostics-v2"
)
ROUTING_TRAINING_DIAGNOSTIC_CONTRACT_VERSION = (
    "uav-hrl-routing-training-diagnostics-v1"
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
    "count",
    "mean",
    "min",
    "max",
    "std",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
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
        return {
            statistic: (0 if statistic == "count" else None)
            for statistic in _DISTRIBUTION_STATISTICS
        }
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("routing Q-score diagnostic distribution is non-finite")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
    }


class RoutingTrainingEpisodeDiagnostics:
    """Bounded, RNG-free scalar diagnostics for one training episode."""

    _SAMPLE_FIELDS = (
        "legal_q_r",
        "legal_q_c",
        "legal_cost_term",
        "legal_q_safe",
        "reward_argmax_minus_safe_argmax_q_r",
        "reward_argmax_minus_safe_argmax_cost_term",
        "safe_argmax_minus_reward_argmax_q_safe",
        "reward_bellman_target",
        "reward_td_error",
        "cost_bellman_target",
        "cost_td_error",
        "reward_loss",
        "cost_loss",
    )

    def __init__(self):
        self._samples = {name: [] for name in self._SAMPLE_FIELDS}
        self.slot_timely_useful_bits = []
        self.slot_rewards = []
        self.decision_count = 0
        self.empty_hol_decision_count = 0
        self.epsilon_decision_count = 0
        self.greedy_decision_count = 0
        self.reward_safe_argmax_difference_count = 0
        self.greedy_selected_differs_from_reward_argmax_count = 0
        self.legal_qc_negative_count = 0
        self.legal_qc_above_one_count = 0
        self.legal_qc_count = 0
        self.representative_cases = []
        self.optimizer_event_count = 0

    def add_slot(self, timely_useful_bits, reward):
        bits = float(timely_useful_bits)
        reward = float(reward)
        if not math.isfinite(bits) or bits < 0.0 or not math.isfinite(reward):
            raise ValueError("routing slot diagnostics require finite non-negative data")
        self.slot_timely_useful_bits.append(bits)
        self.slot_rewards.append(reward)

    def add_decision(
        self,
        diagnostics,
        *,
        selected_action,
        sender_uav_id,
        hol_task_type,
        slot_index,
    ):
        q_r = np.asarray(diagnostics["q_r"], dtype=np.float64)
        q_c = np.asarray(diagnostics["q_c"], dtype=np.float64)
        q_safe = np.asarray(diagnostics["q_safe"], dtype=np.float64)
        legal = np.asarray(diagnostics["legal_mask"], dtype=bool)
        if q_r.ndim != 1 or not (
            q_r.shape == q_c.shape == q_safe.shape == legal.shape
        ) or not legal.any():
            raise ValueError("training Q diagnostics require aligned legal scores")
        if not all(np.all(np.isfinite(values[legal])) for values in (q_r, q_c, q_safe)):
            raise ValueError("training Q diagnostics contain non-finite legal values")
        lambda_cost = float(diagnostics["lambda_cost_used"])
        cost_term = lambda_cost * q_c
        if not np.allclose(
            q_safe[legal],
            (q_r - cost_term)[legal],
            rtol=0.0,
            atol=32.0 * np.finfo(np.float32).eps * max(
                1.0,
                float(np.max(np.abs(q_r[legal]))),
                float(np.max(np.abs(cost_term[legal]))),
            ),
        ):
            raise AssertionError("training Q_safe is inconsistent with Q_r-lambda*Q_c")
        reward_argmax = _masked_argmax(q_r, legal)
        safe_argmax = _masked_argmax(q_safe, legal)
        branch = str(diagnostics.get("decision_branch", "unknown"))
        is_epsilon = branch == "epsilon_exploration"
        is_greedy = branch == "safe_q_greedy"
        self.decision_count += 1
        self.empty_hol_decision_count += int(hol_task_type is None)
        self.epsilon_decision_count += int(is_epsilon)
        self.greedy_decision_count += int(is_greedy)
        differs = reward_argmax != safe_argmax
        self.reward_safe_argmax_difference_count += int(differs)
        self.greedy_selected_differs_from_reward_argmax_count += int(
            is_greedy and int(selected_action) != reward_argmax
        )
        legal_qc = q_c[legal]
        self.legal_qc_count += int(legal_qc.size)
        self.legal_qc_negative_count += int(np.count_nonzero(legal_qc < 0.0))
        self.legal_qc_above_one_count += int(np.count_nonzero(legal_qc > 1.0))
        self._samples["legal_q_r"].extend(q_r[legal].tolist())
        self._samples["legal_q_c"].extend(legal_qc.tolist())
        self._samples["legal_cost_term"].extend(cost_term[legal].tolist())
        self._samples["legal_q_safe"].extend(q_safe[legal].tolist())
        self._samples["reward_argmax_minus_safe_argmax_q_r"].append(
            float(q_r[reward_argmax] - q_r[safe_argmax])
        )
        self._samples[
            "reward_argmax_minus_safe_argmax_cost_term"
        ].append(float(cost_term[reward_argmax] - cost_term[safe_argmax]))
        self._samples["safe_argmax_minus_reward_argmax_q_safe"].append(
            float(q_safe[safe_argmax] - q_safe[reward_argmax])
        )
        case = {
            "slot_index": int(slot_index),
            "sender_uav_id": int(sender_uav_id),
            "hol_task_type": (
                "EMPTY" if hol_task_type is None else str(hol_task_type).upper()
            ),
            "decision_branch": branch,
            "selected_action": int(selected_action),
            "reward_argmax_action": reward_argmax,
            "safe_argmax_action": safe_argmax,
            "lambda_cost_used": lambda_cost,
            "legal_action_count": int(np.count_nonzero(legal)),
        }
        # Deterministic bounded selection: retain flips first, then the first
        # remaining decisions. No random sampling or extra inference occurs.
        if differs:
            flip_count = sum(
                row["reward_argmax_action"] != row["safe_argmax_action"]
                for row in self.representative_cases
            )
            if flip_count < 4:
                self.representative_cases.insert(flip_count, case)
            elif len(self.representative_cases) < 8:
                self.representative_cases.append(case)
        elif len(self.representative_cases) < 8:
            self.representative_cases.append(case)

    def add_optimizer_events(self, events):
        for event in events:
            self.optimizer_event_count += 1
            for field in (
                "reward_bellman_target",
                "reward_td_error",
                "cost_bellman_target",
                "cost_td_error",
            ):
                self._samples[field].extend(float(value) for value in event[field])
            self._samples["reward_loss"].append(float(event["reward_loss"]))
            self._samples["cost_loss"].append(float(event["cost_loss"]))

    def summary(self, **lifecycle):
        slot_bits = _distribution(self.slot_timely_useful_bits)
        slot_kbits = _distribution(
            [value / 1000.0 for value in self.slot_timely_useful_bits]
        )
        slot_reward = _distribution(self.slot_rewards)
        slot_count = len(self.slot_rewards)
        return {
            "schema_version": ROUTING_TRAINING_DIAGNOSTIC_CONTRACT_VERSION,
            "slot_timely_useful_bits": slot_bits,
            "slot_timely_useful_kbits": slot_kbits,
            "slot_global_reward": slot_reward,
            "slot_zero_reward_count": int(
                np.count_nonzero(np.asarray(self.slot_rewards) == 0.0)
            ),
            "slot_zero_reward_fraction": _ratio(
                int(np.count_nonzero(np.asarray(self.slot_rewards) == 0.0)),
                slot_count,
            ),
            "decision_count": self.decision_count,
            "empty_hol_decision_count": self.empty_hol_decision_count,
            "epsilon_decision_count": self.epsilon_decision_count,
            "greedy_decision_count": self.greedy_decision_count,
            "other_decision_branch_count": (
                self.decision_count
                - self.epsilon_decision_count
                - self.greedy_decision_count
            ),
            "reward_safe_argmax_difference_count": (
                self.reward_safe_argmax_difference_count
            ),
            "reward_safe_argmax_difference_fraction": _ratio(
                self.reward_safe_argmax_difference_count, self.decision_count
            ),
            "greedy_selected_differs_from_reward_argmax_count": (
                self.greedy_selected_differs_from_reward_argmax_count
            ),
            "greedy_selected_differs_from_reward_argmax_fraction": _ratio(
                self.greedy_selected_differs_from_reward_argmax_count,
                self.greedy_decision_count,
            ),
            "legal_qc_negative_fraction": _ratio(
                self.legal_qc_negative_count, self.legal_qc_count
            ),
            "legal_qc_above_one_fraction": _ratio(
                self.legal_qc_above_one_count, self.legal_qc_count
            ),
            "score_distributions": {
                field: _distribution(self._samples[field])
                for field in self._SAMPLE_FIELDS
            },
            "optimizer_event_count": self.optimizer_event_count,
            "representative_cases": list(self.representative_cases[:8]),
            **lifecycle,
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
