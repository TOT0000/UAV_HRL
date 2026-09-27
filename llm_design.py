"""Offline LM Studio candidate generation, validation, and fixed-pair scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable
import urllib.error
import urllib.request
import uuid

import numpy as np

from llm_baseline import estimate_empirical_lipschitz, reconstruct_baseline_rewards
from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    parse_candidate_json,
    save_approved_artifact,
    validate_candidate,
)
from llm_design_contract import (
    DESIGN_RUN_SCHEMA_VERSION,
    PROMPT_VERSION,
    build_obs_arrays,
    candidate_schema,
    estimate_token_budget,
    load_design_inputs,
    render_prompt,
)


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_CONTEXT_LENGTH = 20_000
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED = 20260927
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BETA = 1.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_API_TIMEOUT_SECONDS = 600.0
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0
# Backward-compatible module name; CLI --timeout is the API timeout.
DEFAULT_TIMEOUT_SECONDS = DEFAULT_API_TIMEOUT_SECONDS
DEFAULT_ABSOLUTE_TOLERANCE = 1e-12
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_OUTPUT_ROOT = Path("results") / "llm_designs"
DEFAULT_FAILED_CONTENT_LIMIT = 12_000


class APIError(RuntimeError):
    pass


class ContextBudgetError(ValueError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return result[:48] or "model"


def allocate_design_directory(
    model: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        result = Path(output_dir).resolve()
        result.parent.mkdir(parents=True, exist_ok=True)
        result.mkdir()
        return result
    root = Path(output_root).resolve() / _slug(model)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        result = root / f"design-{stamp}-{uuid.uuid4().hex[:8]}"
        try:
            result.mkdir()
            return result
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique directory below {root}")


def _native_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    return root.rstrip("/") + "/api/v1/models"


class LMStudioClient:
    """Small stdlib client for the documented LM Studio HTTP interfaces."""

    def __init__(self, base_url=DEFAULT_BASE_URL, *, timeout=DEFAULT_API_TIMEOUT_SECONDS, retries=2):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.token = os.environ.get("LM_STUDIO_API_TOKEN")

    def _request(self, method: str, url: str, payload: dict | None = None) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_error = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise APIError("LM Studio response is not a JSON object")
                return value
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error = APIError(f"LM Studio HTTP {exc.code}: {body}")
                if 400 <= exc.code < 500:
                    raise error from exc
                last_error = error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = APIError(f"LM Studio request failed: {type(exc).__name__}: {exc}")
            if attempt < self.retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))
        raise last_error or APIError("LM Studio request failed")

    def list_models(self) -> dict[str, Any]:
        openai = self._request("GET", self.base_url + "/models")
        try:
            native = self._request("GET", _native_models_url(self.base_url))
            native_error = None
        except APIError as exc:
            native = None
            native_error = str(exc)
        return {"openai": openai, "native": native, "native_error": native_error}

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        seed: int,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        base_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
            "seed": int(seed),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "uav_hrl_llm_candidate",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        payload = dict(base_payload)
        fallbacks = []
        try:
            raw = self._request("POST", self.base_url + "/chat/completions", payload)
        except APIError as first:
            message = str(first).lower()
            if "response_format" in message or "json_schema" in message or "structured" in message:
                payload.pop("response_format", None)
                fallbacks.append("structured_output_not_supported; retried with strict text parsing")
                raw = self._request("POST", self.base_url + "/chat/completions", payload)
            elif "seed" in message:
                payload.pop("seed", None)
                fallbacks.append("seed_not_supported; retried without seed")
                raw = self._request("POST", self.base_url + "/chat/completions", payload)
            else:
                raise
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise APIError("chat completion has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        adapter = model_adapter(model)
        content, inline_reasoning = adapter_separate_reasoning(adapter, content)
        if inline_reasoning:
            reasoning = inline_reasoning if reasoning is None else f"{reasoning}\n{inline_reasoning}"
        return {
            "request": payload,
            "raw": raw,
            "content": content,
            "reasoning": reasoning,
            "finish_reason": choice.get("finish_reason"),
            "actual_model": raw.get("model"),
            "usage": raw.get("usage"),
            "adapter": adapter,
            "fallbacks": fallbacks,
            "seed_sent": "seed" in payload,
            "structured_output_sent": "response_format" in payload,
        }


def model_adapter(model: str) -> str:
    lowered = model.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return "openai-compatible-generic"


def adapter_separate_reasoning(adapter: str, content: Any) -> tuple[str | None, str | None]:
    if content is None:
        return None, None
    content = str(content)
    if adapter == "qwen":
        match = re.match(r"\s*<think>(.*?)</think>\s*(.*)\Z", content, re.DOTALL)
        if match:
            return match.group(2), match.group(1)
    return content, None


def model_inventory_summary(inventory: dict[str, Any], requested: str) -> dict[str, Any]:
    openai_data = (inventory.get("openai") or {}).get("data") or []
    visible_ids = sorted(
        str(item.get("id")) for item in openai_data if item.get("id") is not None
    )
    if requested not in visible_ids:
        raise APIError(
            f"requested model {requested!r} is not visible at /v1/models; "
            f"available model IDs: {visible_ids}"
        )
    native_models = (inventory.get("native") or {}).get("models") or []
    native = next(
        (
            item
            for item in native_models
            if item.get("key") == requested
            or any(instance.get("id") == requested for instance in item.get("loaded_instances", []))
        ),
        None,
    )
    loaded_context = None
    if native is not None:
        for instance in native.get("loaded_instances", []):
            if instance.get("id") == requested or native.get("key") == requested:
                loaded_context = (instance.get("config") or {}).get("context_length")
                if loaded_context is not None:
                    break
    return {
        "requested_api_identifier": requested,
        "visible_model_ids": visible_ids,
        "native_model_information": native,
        "model_supported_max_context_length": (
            None if native is None else native.get("max_context_length")
        ),
        "actual_loaded_context_length": loaded_context,
        "native_inventory_error": inventory.get("native_error"),
    }


def _numeric_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None, "mean": None}
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _pair_rank(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _pair_blocks(n: int, batch_size: int):
    for left_start in range(0, n, batch_size):
        left_stop = min(n, left_start + batch_size)
        for right_start in range(left_start, n, batch_size):
            right_stop = min(n, right_start + batch_size)
            if left_start == right_start:
                local_i, local_j = np.triu_indices(left_stop - left_start, k=1)
                pair_i = local_i + left_start
                pair_j = local_j + right_start
            else:
                pair_i = np.repeat(
                    np.arange(left_start, left_stop, dtype=np.int64),
                    right_stop - right_start,
                )
                pair_j = np.tile(
                    np.arange(right_start, right_stop, dtype=np.int64),
                    left_stop - left_start,
                )
            if pair_i.size:
                yield pair_i, pair_j


@dataclass
class EvaluationContext:
    original_state: np.ndarray
    original_distances: np.ndarray
    primary_mask: np.ndarray
    zero_mask: np.ndarray
    near_mask: np.ndarray
    base_rewards: dict[str, np.ndarray]
    lambdas: tuple[float, ...]
    baseline_values: dict[str, float]
    distance_epsilon: float
    reward_epsilon: float
    fixed_metadata: dict[str, Any]
    fixed_arrays: dict[str, np.ndarray]
    pair_hash: str


def prepare_evaluation_context(
    fixed_arrays: dict[str, np.ndarray],
    fixed_metadata: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    batch_size: int,
) -> EvaluationContext:
    lambdas = tuple(
        float(value)
        for value in baseline_report["parameters"]["lambda_values_mbit_per_joule"]
    )
    rewards = reconstruct_baseline_rewards(fixed_arrays, lambdas)
    distance_epsilon = float(baseline_report["parameters"]["distance_epsilon"])
    reward_epsilon = float(
        baseline_report["parameters"]["reward_difference_tolerance"]
    )
    verified = estimate_empirical_lipschitz(
        fixed_arrays["state"],
        rewards,
        arrays=fixed_arrays,
        sample_metadata=fixed_metadata,
        batch_size=batch_size,
        distance_epsilon=distance_epsilon,
        reward_epsilon=reward_epsilon,
    )
    saved = baseline_report["lipschitz"]
    for field in (
        "all_pair_count",
        "primary_pair_count",
        "zero_distance_pair_count",
        "near_zero_distance_pair_count",
        "primary_pair_set_sha256",
    ):
        if verified[field] != saved[field]:
            raise ValueError(f"recomputed baseline pair field differs: {field}")
    baseline_values = {}
    for value in lambdas:
        key = format(value, ".17g")
        actual = verified["estimates_by_lambda_mbit_per_joule"][key]["l_hat"]
        expected = saved["estimates_by_lambda_mbit_per_joule"][key]["l_hat"]
        if actual is None or expected is None or not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"recomputed baseline L_hat differs for lambda {key}")
        baseline_values[key] = float(expected)
    states = np.asarray(fixed_arrays["state"], dtype=np.float64)
    n = states.shape[0]
    distances = np.empty(n * (n - 1) // 2, dtype=np.float64)
    for pair_i, pair_j in _pair_blocks(n, int(batch_size)):
        difference = states[pair_i] - states[pair_j]
        distances[_pair_rank(n, pair_i, pair_j)] = np.sqrt(
            np.einsum("ij,ij->i", difference, difference)
        )
    primary = distances > distance_epsilon
    zero = distances == 0.0
    near = (distances > 0.0) & (distances <= distance_epsilon)
    return EvaluationContext(
        original_state=states,
        original_distances=distances,
        primary_mask=primary,
        zero_mask=zero,
        near_mask=near,
        base_rewards=rewards,
        lambdas=lambdas,
        baseline_values=baseline_values,
        distance_epsilon=distance_epsilon,
        reward_epsilon=reward_epsilon,
        fixed_metadata=fixed_metadata,
        fixed_arrays=fixed_arrays,
        pair_hash=verified["primary_pair_set_sha256"],
    )


def _sample_trace(metadata: dict, index: int) -> dict:
    return (metadata.get("selection") or [])[index]


def evaluate_candidate(
    context: EvaluationContext,
    candidate: dict[str, Any],
    extra_state: np.ndarray,
    reward_terms: np.ndarray,
    *,
    beta: float,
    batch_size: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    extra = np.asarray(extra_state, dtype=np.float64)
    terms = np.asarray(reward_terms, dtype=np.float64)
    weights = np.asarray(
        [item["weight"] for item in candidate["reward_terms"]], dtype=np.float64
    )
    reward_extra = terms @ weights
    candidate_rewards = {
        key: base + float(beta) * reward_extra
        for key, base in context.base_rewards.items()
    }
    n = context.original_state.shape[0]
    pair_count = context.original_distances.size
    candidate_distances = np.empty(pair_count, dtype=np.float64)
    amplification = np.empty(int(np.count_nonzero(context.primary_mask)), dtype=np.float64)
    amplification_offset = 0
    direct_max_error = 0.0
    direct_checked = 0
    trackers = {
        key: {"ratio": -math.inf, "rank": None, "pair": None}
        for key in candidate_rewards
    }
    excluded_trackers = {
        key: {
            "zero_max_delta": -math.inf,
            "zero_pair": None,
            "near_max_ratio": -math.inf,
            "near_pair": None,
        }
        for key in candidate_rewards
    }
    zero_candidate_nonzero = 0
    zero_candidate_distances = []
    for pair_i, pair_j in _pair_blocks(n, int(batch_size)):
        ranks = _pair_rank(n, pair_i, pair_j)
        original_distance = context.original_distances[ranks]
        extra_difference = extra[pair_i] - extra[pair_j]
        extra_squared = np.einsum("ij,ij->i", extra_difference, extra_difference)
        distance = np.sqrt(original_distance * original_distance + extra_squared)
        candidate_distances[ranks] = distance
        primary = context.primary_mask[ranks]
        if np.any(primary):
            values = distance[primary] / original_distance[primary]
            amplification[
                amplification_offset : amplification_offset + values.size
            ] = values
            amplification_offset += values.size
        zero = context.zero_mask[ranks]
        if np.any(zero):
            zero_values = distance[zero]
            zero_candidate_nonzero += int(np.count_nonzero(zero_values > 0.0))
            zero_candidate_distances.append(zero_values)
        if direct_checked < 4096:
            take = min(4096 - direct_checked, pair_i.size)
            direct = np.linalg.norm(
                np.concatenate(
                    (
                        context.original_state[pair_i[:take]]
                        - context.original_state[pair_j[:take]],
                        extra[pair_i[:take]] - extra[pair_j[:take]],
                    ),
                    axis=1,
                ),
                axis=1,
            )
            direct_max_error = max(
                direct_max_error,
                float(np.max(np.abs(direct - distance[:take]))),
            )
            direct_checked += take
        for key, reward in candidate_rewards.items():
            delta = np.abs(reward[pair_i] - reward[pair_j])
            zero_positions = np.flatnonzero(context.zero_mask[ranks])
            if zero_positions.size:
                local = int(np.argmax(delta[zero_positions]))
                position = int(zero_positions[local])
                value = float(delta[position])
                if value > excluded_trackers[key]["zero_max_delta"]:
                    excluded_trackers[key]["zero_max_delta"] = value
                    excluded_trackers[key]["zero_pair"] = (
                        int(pair_i[position]),
                        int(pair_j[position]),
                        float(distance[position]),
                    )
            near_positions = np.flatnonzero(context.near_mask[ranks])
            if near_positions.size:
                ratios = delta[near_positions] / distance[near_positions]
                local = int(np.argmax(ratios))
                position = int(near_positions[local])
                value = float(ratios[local])
                if value > excluded_trackers[key]["near_max_ratio"]:
                    excluded_trackers[key]["near_max_ratio"] = value
                    excluded_trackers[key]["near_pair"] = (
                        int(pair_i[position]),
                        int(pair_j[position]),
                        float(distance[position]),
                        float(delta[position]),
                    )
            if not np.any(primary):
                continue
            primary_positions = np.flatnonzero(primary)
            ratios = (
                np.abs(reward[pair_i[primary]] - reward[pair_j[primary]])
                / distance[primary]
            )
            local = int(np.argmax(ratios))
            position = int(primary_positions[local])
            rank = int(ranks[position])
            value = float(ratios[local])
            tracker = trackers[key]
            if value > tracker["ratio"] or (
                value == tracker["ratio"]
                and (tracker["rank"] is None or rank < tracker["rank"])
            ):
                tracker["ratio"] = value
                tracker["rank"] = rank
                tracker["pair"] = (int(pair_i[position]), int(pair_j[position]))
    if amplification_offset != amplification.size:
        raise RuntimeError("candidate pair traversal did not cover the fixed primary set")
    by_lambda = {}
    all_pass = True
    for value in context.lambdas:
        key = format(value, ".17g")
        baseline = context.baseline_values[key]
        candidate_value = float(trackers[key]["ratio"])
        margin = max(
            float(absolute_tolerance), float(relative_tolerance) * abs(baseline)
        )
        passed = candidate_value < baseline - margin
        all_pass &= passed
        i, j = trackers[key]["pair"]
        reward = candidate_rewards[key]
        by_lambda[key] = {
            "lambda_mbit_per_joule": value,
            "baseline_l_hat": baseline,
            "candidate_l_hat": candidate_value,
            "improvement": baseline - candidate_value,
            "required_margin": margin,
            "passed": bool(passed),
            "maximum_pair": {
                "i": i,
                "j": j,
                "sample_i": _sample_trace(context.fixed_metadata, i),
                "sample_j": _sample_trace(context.fixed_metadata, j),
                "original_distance": float(
                    context.original_distances[trackers[key]["rank"]]
                ),
                "candidate_distance": float(
                    candidate_distances[trackers[key]["rank"]]
                ),
                "base_reward_i": float(context.base_rewards[key][i]),
                "base_reward_j": float(context.base_rewards[key][j]),
                "extra_reward_i": float(reward_extra[i]),
                "extra_reward_j": float(reward_extra[j]),
                "candidate_reward_i": float(reward[i]),
                "candidate_reward_j": float(reward[j]),
                "candidate_reward_difference": float(abs(reward[i] - reward[j])),
            },
        }
    excluded_by_lambda = {}
    for key, tracker in excluded_trackers.items():
        zero_pair = tracker["zero_pair"]
        near_pair = tracker["near_pair"]
        excluded_by_lambda[key] = {
            "maximum_zero_original_distance_reward_difference": (
                None
                if zero_pair is None
                else {
                    "i": zero_pair[0],
                    "j": zero_pair[1],
                    "candidate_distance": zero_pair[2],
                    "reward_difference": tracker["zero_max_delta"],
                    "ratio": (
                        None
                        if zero_pair[2] == 0.0
                        else tracker["zero_max_delta"] / zero_pair[2]
                    ),
                }
            ),
            "maximum_near_original_distance_ratio": (
                None
                if near_pair is None
                else {
                    "i": near_pair[0],
                    "j": near_pair[1],
                    "candidate_distance": near_pair[2],
                    "reward_difference": near_pair[3],
                    "ratio": tracker["near_max_ratio"],
                }
            ),
        }
    zero_distances = (
        np.concatenate(zero_candidate_distances)
        if zero_candidate_distances
        else np.empty(0, dtype=np.float64)
    )
    return {
        "status": "passed" if all_pass else "not_improved_for_all_lambdas",
        "passed": bool(all_pass),
        "empirical_scope": "fixed samples and baseline-defined original-state primary pair set only",
        "fixed_primary_pair_set_sha256": context.pair_hash,
        "fixed_primary_pair_count": int(np.count_nonzero(context.primary_mask)),
        "candidate_pair_reselection": False,
        "beta": float(beta),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "by_lambda": by_lambda,
        "extra_reward_distribution": _numeric_distribution(reward_extra),
        "original_primary_distance_distribution": _numeric_distribution(
            context.original_distances[context.primary_mask]
        ),
        "candidate_primary_distance_distribution": _numeric_distribution(
            candidate_distances[context.primary_mask]
        ),
        "distance_amplification_distribution": _numeric_distribution(amplification),
        "direct_concat_distance_check": {
            "checked_pair_count": direct_checked,
            "maximum_absolute_difference": direct_max_error,
            "passed": bool(direct_max_error <= 1e-12),
        },
        "baseline_excluded_pairs": {
            "zero_distance_pair_count": int(np.count_nonzero(context.zero_mask)),
            "near_zero_distance_pair_count": int(np.count_nonzero(context.near_mask)),
            "zero_pairs_with_nonzero_augmented_distance": zero_candidate_nonzero,
            "augmented_zero_pair_distance_distribution": _numeric_distribution(
                zero_distances
            ),
            "diagnostics_by_lambda": excluded_by_lambda,
            "included_in_primary_candidate_estimate": False,
        },
    }


def _round_request(
    attempt: int,
    max_attempts: int,
    previous_attempt: dict[str, Any] | None,
    *,
    failed_content_limit: int | None = None,
) -> str:
    if attempt == 1:
        return "Generate the first candidate."
    if previous_attempt is None:
        raise ValueError("revision round requires the immediately preceding attempt")
    parsed = previous_attempt.get("parsed_candidate")
    raw_content = previous_attempt.get("raw_final_content")
    feedback = previous_attempt.get("feedback")
    if parsed is not None:
        previous = (
            "Previous complete parsed candidate:\n"
            + json.dumps(parsed, indent=2, ensure_ascii=False, allow_nan=False)
        )
    elif isinstance(raw_content, str) and raw_content:
        previous = (
            "Previous failed raw final content (not a valid candidate JSON):\n"
            + _failed_content_excerpt(raw_content, feedback, failed_content_limit)
        )
    else:
        prior_kind = (
            "The previous response contained reasoning, but no final JSON content. "
            "Reasoning is intentionally omitted from this revision request."
            if previous_attempt.get("reasoning_present")
            else "The previous response had no final content."
        )
        previous = "Previous failed raw final content: <missing>\n" + prior_kind
    return (
        f"Revision round {attempt} of {max_attempts}. Return a complete replacement JSON object, not a patch.\n\n"
        + previous
        + "\n\nLatest validation/evaluation feedback for that same output:\n"
        + json.dumps(feedback, indent=2, ensure_ascii=False, allow_nan=False)
    )


def _failed_content_excerpt(
    content: str,
    feedback: dict[str, Any] | None,
    limit: int | None,
) -> str:
    if limit is None or len(content) <= int(limit):
        return content
    limit = max(0, int(limit))
    error = "" if feedback is None else str(feedback.get("error", ""))
    match = re.search(r"(?:char|position)\s+(\d+)", error, flags=re.IGNORECASE)
    center = int(match.group(1)) if match else len(content) // 2
    center = min(max(center, 0), len(content))
    if limit == 0:
        return f"[TRUNCATED: omitted all {len(content)} raw characters to fit the context budget]"
    start = max(0, center - limit // 2)
    stop = min(len(content), start + limit)
    start = max(0, stop - limit)
    location = f" near parser character {center}" if match else ""
    return (
        f"[TRUNCATED: showing raw characters {start}:{stop} of {len(content)}{location}]\n"
        + content[start:stop]
        + "\n[END TRUNCATED RAW CONTENT]"
    )


def _feedback_from_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": "candidate_not_improved_for_all_lambdas",
        "status": report["status"],
        "by_lambda": report["by_lambda"],
        "feature_and_reward_diagnostics": report.get("candidate_numeric_diagnostics"),
        "distance_amplification_distribution": report[
            "distance_amplification_distribution"
        ],
        "instruction": "Revise the design and return a complete candidate JSON using the unchanged interface and evaluation rules.",
    }


def _summarize_error(category: str, exc: BaseException) -> dict[str, Any]:
    return {
        "category": category,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "instruction": "Return a complete corrected candidate JSON using the unchanged interface and evaluation rules.",
    }


def _effective_context_budget(
    requested: int, model_info: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    requested = int(requested)
    loaded = None if model_info is None else model_info.get("actual_loaded_context_length")
    supported = None if model_info is None else model_info.get("model_supported_max_context_length")
    effective = requested
    constraints = [requested]
    if loaded is not None:
        constraints.append(int(loaded))
    if supported is not None:
        constraints.append(int(supported))
    effective = min(constraints)
    return effective, {
        "requested_client_budget": requested,
        "model_supported_maximum": supported,
        "actual_loaded_context_length": loaded,
        "effective_budget": effective,
        "server_setting_changed_by_client": False,
    }


def run_design(
    *,
    fixed_sample: str | Path,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    beta: float = DEFAULT_BETA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    worker_timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    client: Any | None = None,
) -> dict[str, Any]:
    if not model:
        raise ValueError("model API identifier is required")
    if int(context_length) <= 0 or int(max_output_tokens) <= 0:
        raise ValueError("context and output token budgets must be positive")
    if int(max_attempts) <= 0 or int(batch_size) <= 0:
        raise ValueError("max_attempts and batch_size must be positive")
    for label, value in (
        ("temperature", temperature),
        ("beta", beta),
        ("timeout", timeout),
        ("worker_timeout", worker_timeout),
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")

    arrays, fixed_metadata, baseline, constants_metadata = load_design_inputs(
        fixed_sample
    )
    output = allocate_design_directory(
        model, output_root=output_root, output_dir=output_dir
    )
    model_info = None
    api_client = client or LMStudioClient(base_url, timeout=timeout)
    if not dry_run:
        inventory = api_client.list_models()
        model_info = model_inventory_summary(inventory, model)
    effective_context, context_info = _effective_context_budget(
        context_length, model_info
    )
    metadata = {
        "schema_version": DESIGN_RUN_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "prompt_version": PROMPT_VERSION,
        "candidate_schema": candidate_schema(),
        "model": {
            "requested_api_identifier": model,
            "adapter": model_adapter(model),
            "base_url": str(base_url),
            "inventory": model_info,
            "quantization_or_load_information": (
                "unknown in dry-run; no API request was sent"
                if dry_run
                else model_info.get("native_model_information")
            ),
        },
        "generation": {
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
            "seed_requested": int(seed),
            "seed_support": "requested when generation runs; deterministic reproduction is not guaranteed by this client",
            "max_attempts": int(max_attempts),
            "api_timeout_seconds_per_request": float(timeout),
            "api_retry_count": int(getattr(api_client, "retries", 0)),
            "api_total_wait_note": "timeout applies per request; retries can make total waiting time longer",
            "worker_timeout_seconds": float(worker_timeout),
            "structured_output": "requested first; explicit recorded fallback to strict text parsing only when rejected by API",
        },
        "context": context_info,
        "fixed_sample": {
            "directory": str(Path(fixed_sample).resolve()),
            "sample_content_sha256": fixed_metadata["sample_content_sha256"],
            "primary_pair_set_sha256": baseline["lipschitz"][
                "primary_pair_set_sha256"
            ],
            "sample_count": fixed_metadata["sample_count"],
            "primary_pair_count": baseline["lipschitz"]["primary_pair_count"],
            "baseline_schema_version": baseline["schema_version"],
        },
        "evaluation": {
            "beta": float(beta),
            "batch_size": int(batch_size),
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
            "all_lambdas_must_pass": True,
            "candidate_pair_reselection": False,
        },
        "observation_interface_version": "uav-hrl-llm-current-observation-v1",
        "dry_run": bool(dry_run),
    }
    _write_json(output / "constants.json", constants_metadata)
    first_prompt = render_prompt(
        fixed_metadata=fixed_metadata,
        baseline_report=baseline,
        constants_metadata=constants_metadata,
        beta=beta,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        round_request="Generate the first candidate.",
    )
    first_budget = estimate_token_budget(
        first_prompt,
        context_length=effective_context,
        max_output_tokens=max_output_tokens,
    )
    metadata["first_prompt_token_budget"] = first_budget
    (output / "prompt_attempt_01.txt").write_text(first_prompt, encoding="utf-8")
    _write_json(
        output / "request_attempt_01.json",
        {
            "model": model,
            "messages": [{"role": "user", "content": first_prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
            "seed": int(seed),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "uav_hrl_llm_candidate",
                    "strict": True,
                    "schema": candidate_schema(),
                },
            },
            "context_length_note": "client budget only; not sent to alter LM Studio load configuration",
        },
    )
    if not first_budget["fits_client_budget"]:
        metadata.update(
            {
                "status": "failed_context_budget",
                "failure": (
                    "estimated prompt upper bound plus reserved output exceeds the "
                    "effective context budget"
                ),
            }
        )
        _write_json(output / "run_metadata.json", metadata)
        raise ContextBudgetError(metadata["failure"])
    if dry_run:
        metadata["status"] = "dry_run_complete"
        metadata["generation_requests_sent"] = 0
        _write_json(output / "run_metadata.json", metadata)
        return {"status": metadata["status"], "output_directory": str(output), "metadata": metadata}

    context = prepare_evaluation_context(
        arrays, fixed_metadata, baseline, batch_size=batch_size
    )
    if context.pair_hash != fixed_metadata["pair_contract"][
        "primary_pair_set_sha256"
    ]:
        raise ValueError("recomputed primary pair hash differs from fixed artifact")
    obs_arrays = build_obs_arrays(arrays)
    previous_attempt = None
    feedback = None
    history = []
    approved = None
    actual_models = []
    adapter_fallbacks = []
    seed_sent_values = []
    for attempt in range(1, int(max_attempts) + 1):
        round_directory = output / f"attempt_{attempt:02d}"
        round_directory.mkdir()
        raw_failure = (
            previous_attempt is not None
            and previous_attempt.get("parsed_candidate") is None
            and isinstance(previous_attempt.get("raw_final_content"), str)
            and bool(previous_attempt.get("raw_final_content"))
        )
        content_limits = (
            [
                None,
                DEFAULT_FAILED_CONTENT_LIMIT,
                8_000,
                4_000,
                2_000,
                1_000,
                500,
                0,
            ]
            if raw_failure
            else [None]
        )
        prompt = None
        budget = None
        for content_limit in content_limits:
            request_text = _round_request(
                attempt,
                int(max_attempts),
                previous_attempt,
                failed_content_limit=content_limit,
            )
            candidate_prompt = render_prompt(
                fixed_metadata=fixed_metadata,
                baseline_report=baseline,
                constants_metadata=constants_metadata,
                beta=beta,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                round_request=request_text,
            )
            candidate_budget = estimate_token_budget(
                candidate_prompt,
                context_length=effective_context,
                max_output_tokens=max_output_tokens,
            )
            prompt, budget = candidate_prompt, candidate_budget
            if candidate_budget["fits_client_budget"]:
                break
        assert prompt is not None and budget is not None
        (round_directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        _write_json(round_directory / "token_budget.json", budget)
        _write_json(
            round_directory / "request_planned.json",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
                "max_tokens": int(max_output_tokens),
                "seed": int(seed),
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "uav_hrl_llm_candidate",
                        "strict": True,
                        "schema": candidate_schema(),
                    },
                },
                "context_length_note": "budget only; not a server load-setting request",
            },
        )
        if not budget["fits_client_budget"]:
            feedback = {
                "category": "context_budget_exceeded",
                "token_budget": budget,
            }
            history.append({"attempt": attempt, "status": "context_budget_exceeded"})
            break
        attempt_output = {
            "raw_final_content": None,
            "parsed_candidate": None,
            "reasoning_present": False,
            "feedback": None,
        }
        try:
            response = api_client.chat(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
                schema=candidate_schema(),
            )
            _write_json(round_directory / "request.json", response["request"])
            _write_json(round_directory / "raw_response.json", response["raw"])
            _write_json(
                round_directory / "response_metadata.json",
                {
                    key: response.get(key)
                    for key in (
                        "finish_reason",
                        "actual_model",
                        "usage",
                        "adapter",
                        "fallbacks",
                        "seed_sent",
                        "structured_output_sent",
                    )
                },
            )
            if response.get("actual_model") is not None:
                actual_models.append(str(response["actual_model"]))
            adapter_fallbacks.extend(str(value) for value in response.get("fallbacks", []))
            seed_sent_values.append(bool(response.get("seed_sent")))
            if response.get("actual_model") not in {None, model}:
                raise APIError(
                    "LM Studio returned a different model identifier: "
                    f"requested={model!r}, actual={response.get('actual_model')!r}"
                )
            if response.get("reasoning") is not None:
                (round_directory / "reasoning.txt").write_text(
                    str(response["reasoning"]), encoding="utf-8"
                )
            content = response.get("content")
            attempt_output["raw_final_content"] = (
                None if content is None else str(content)
            )
            attempt_output["reasoning_present"] = bool(response.get("reasoning"))
            (round_directory / "response_content.txt").write_text(
                "" if content is None else str(content), encoding="utf-8"
            )
            if response.get("finish_reason") == "length":
                raise CandidateError("model output was truncated (finish_reason=length)")
            if content is None or not str(content).strip():
                reason = "model returned reasoning but no final JSON" if response.get("reasoning") else "model returned no final content"
                raise CandidateError(reason)
            candidate = parse_candidate_json(str(content))
            attempt_output["parsed_candidate"] = candidate
            static = validate_candidate(candidate, constants_metadata)
            _write_json(round_directory / "candidate.json", candidate)
            (round_directory / "candidate.py").write_text(
                candidate["code"], encoding="utf-8"
            )
            extra, terms, execution = execute_candidate_isolated(
                candidate,
                obs_arrays,
                constants_metadata,
                timeout=worker_timeout,
            )
            validation_report = {
                "status": "passed",
                "static": static,
                "execution": execution,
                "numeric_tolerance": 1e-6,
            }
            _write_json(round_directory / "validation_report.json", validation_report)
            evaluation = evaluate_candidate(
                context,
                candidate,
                extra,
                terms,
                beta=beta,
                batch_size=batch_size,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            evaluation["candidate_numeric_diagnostics"] = candidate_numeric_diagnostics(
                arrays["state"], extra, terms, candidate
            )
            _write_json(round_directory / "evaluation_report.json", evaluation)
            if evaluation["passed"]:
                approved = save_approved_artifact(
                    output,
                    candidate=candidate,
                    constants_metadata=constants_metadata,
                    validation_report=validation_report,
                    evaluation_report=evaluation,
                    provenance={
                        "run_directory": str(output),
                        "attempt": attempt,
                        "model_requested": model,
                        "model_actual": response.get("actual_model"),
                        "fixed_sample_content_sha256": fixed_metadata[
                            "sample_content_sha256"
                        ],
                        "primary_pair_set_sha256": context.pair_hash,
                        "beta": float(beta),
                        "git_sha": metadata["git_sha"],
                    },
                )
                history.append({"attempt": attempt, "status": "approved"})
                break
            feedback = _feedback_from_evaluation(evaluation)
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            _write_json(round_directory / "feedback.json", feedback)
            history.append({"attempt": attempt, "status": evaluation["status"]})
        except APIError as exc:
            feedback = _summarize_error("api_failure", exc)
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": "api_failure"})
            break
        except CandidateExecutionError as exc:
            feedback = _summarize_error("candidate_execution_failure", exc)
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": "candidate_execution_failure"})
        except CandidateError as exc:
            feedback = _summarize_error("candidate_format_or_validation_failure", exc)
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": "candidate_format_or_validation_failure"})
        if feedback is not None:
            _write_json(round_directory / "feedback.json", feedback)
    metadata["attempt_history"] = history
    metadata["attempts_completed"] = len(history)
    metadata["model"]["actual_api_identifiers_reported"] = sorted(
        set(actual_models)
    )
    metadata["model"]["adapter_fallbacks_used"] = adapter_fallbacks
    metadata["generation"]["seed_sent_for_all_generation_requests"] = (
        bool(seed_sent_values) and all(seed_sent_values)
    )
    if approved is not None:
        metadata["status"] = "approved"
        metadata["approved_artifact"] = str(approved)
    else:
        metadata["status"] = "failed_no_approved_candidate"
        metadata["final_feedback"] = feedback
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "run_metadata.json", metadata)
    return {
        "status": metadata["status"],
        "output_directory": str(output),
        "approved_artifact": None if approved is None else str(approved),
        "metadata": metadata,
    }
