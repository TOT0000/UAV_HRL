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
    candidate_semantic_fingerprint,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    feature_reward,
    parse_candidate_json,
    save_approved_artifact,
    validate_candidate,
    validate_candidate_staged,
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
from llm_streaming import (
    StreamTransportError,
    capture_chat_stream,
    open_http_stream,
    read_response_body_bounded,
)


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_CONTEXT_LENGTH = 40_000
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED = 20260927
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BETA = 1.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_API_TIMEOUT_SECONDS = 600.0
DEFAULT_API_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_API_TOTAL_TIMEOUT_SECONDS = 1800.0
DEFAULT_API_PROGRESS_INTERVAL_SECONDS = 10.0
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0
# Backward-compatible module name; CLI --timeout is the stream idle timeout.
DEFAULT_TIMEOUT_SECONDS = DEFAULT_API_TIMEOUT_SECONDS
DEFAULT_ABSOLUTE_TOLERANCE = 1e-12
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_OUTPUT_ROOT = Path("results") / "llm_designs"
DEFAULT_FAILED_CONTENT_LIMIT = 12_000


class APIError(RuntimeError):
    def __init__(self, message: str, *, category: str = "api_failure"):
        super().__init__(message)
        self.category = str(category)


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

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        *,
        timeout=DEFAULT_API_TIMEOUT_SECONDS,
        connect_timeout=DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        total_timeout=DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        progress_interval=DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
        retries=2,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.total_timeout = float(total_timeout)
        self.progress_interval = float(progress_interval)
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
                with urllib.request.urlopen(
                    request, timeout=self.connect_timeout
                ) as response:
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

    def _open_stream(
        self,
        request,
        *,
        connect_timeout,
        header_timeout,
        total_timeout,
    ):
        return open_http_stream(
            request,
            connect_timeout=connect_timeout,
            header_timeout=header_timeout,
            total_timeout=total_timeout,
        )

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        seed: int,
        attempt_directory: str | Path,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
            "seed": int(seed),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        attempt_directory = Path(attempt_directory)
        _write_json(attempt_directory / "request.json", payload)
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            method="POST",
            headers=headers,
        )
        fallbacks = []
        request_started = time.monotonic()
        try:
            response = self._open_stream(
                request,
                connect_timeout=min(self.connect_timeout, self.total_timeout),
                header_timeout=min(self.timeout, self.total_timeout),
                total_timeout=self.total_timeout,
            )
        except KeyboardInterrupt as exc:
            _write_json(
                attempt_directory / "stream_status.json",
                {
                    "status": "cancelled",
                    "failure": "request cancelled by user during connection establishment",
                },
            )
            raise APIError(
                "request cancelled by user during connection establishment",
                category="stream_cancelled",
            ) from exc
        except StreamTransportError as exc:
            _write_json(
                attempt_directory / "stream_status.json",
                {
                    "status": exc.category,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise APIError(str(exc), category=exc.category) from exc
        if getattr(response, "status", 200) >= 400:
            remaining_total = max(
                0.0, self.total_timeout - (time.monotonic() - request_started)
            )
            body_bytes, body_status = read_response_body_bounded(
                response,
                idle_timeout=self.timeout,
                total_timeout=remaining_total,
            )
            body = body_bytes.decode("utf-8", errors="replace")
            status = {
                **body_status,
                "http_status": int(response.status),
                "http_reason": getattr(response, "reason", ""),
                "body": body,
            }
            _write_json(attempt_directory / "stream_status.json", status)
            if body_status["status"] != "complete":
                raise APIError(
                    body_status["failure"] or "HTTP error body read failed",
                    category=body_status["status"],
                )
            raise APIError(
                f"LM Studio HTTP {response.status}: {body}", category="http_error"
            )
        remaining_total = self.total_timeout - (time.monotonic() - request_started)
        if remaining_total <= 0.0:
            abort = getattr(response, "abort", None)
            if callable(abort):
                abort()
            raise APIError(
                f"request exceeded total timeout of {self.total_timeout:g} seconds during connection establishment",
                category="stream_total_timeout",
            )
        try:
            streamed = capture_chat_stream(
                response,
                directory=attempt_directory,
                idle_timeout=self.timeout,
                total_timeout=remaining_total,
                progress_interval=self.progress_interval,
            )
        except StreamTransportError as exc:
            raise APIError(str(exc), category=exc.category) from exc
        content = streamed["content"]
        reasoning = streamed["reasoning"]
        adapter = model_adapter(model)
        content, inline_reasoning = adapter_separate_reasoning(adapter, content)
        if inline_reasoning:
            reasoning = inline_reasoning if not reasoning else f"{reasoning}\n{inline_reasoning}"
        raw = {
            "reconstructed_from_stream": True,
            "model": streamed.get("actual_model"),
            "choices": [
                {
                    "finish_reason": streamed.get("finish_reason"),
                    "message": {"content": content, "reasoning_content": reasoning},
                }
            ],
            "usage": streamed.get("usage"),
        }
        return {
            "request": payload,
            "raw": raw,
            "content": content,
            "reasoning": reasoning,
            "finish_reason": streamed.get("finish_reason"),
            "actual_model": streamed.get("actual_model"),
            "usage": streamed.get("usage"),
            "adapter": adapter,
            "fallbacks": fallbacks,
            "seed_sent": True,
            "structured_output_sent": False,
            "transport_completed": streamed["transport_completed"],
            "done_received": streamed["done_received"],
            "terminal_chunk_received": streamed["terminal_chunk_received"],
            "tool_calls_seen": streamed["tool_calls_seen"],
            "stream_elapsed_seconds": streamed["elapsed_seconds"],
            "request_elapsed_seconds": time.monotonic() - request_started,
            "stream_event_count": streamed["event_count"],
            "stream_received_bytes": streamed["received_bytes"],
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
    *,
    beta: float,
    batch_size: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    extra = np.asarray(extra_state, dtype=np.float64)
    reward_extra = feature_reward(extra, candidate)
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
    feedback_override: dict[str, Any] | None = None,
) -> str:
    if attempt == 1:
        return "Generate the first candidate."
    if previous_attempt is None:
        raise ValueError("revision round requires the immediately preceding attempt")
    parsed = previous_attempt.get("parsed_candidate")
    raw_content = previous_attempt.get("raw_final_content")
    feedback = (
        previous_attempt.get("feedback")
        if feedback_override is None
        else feedback_override
    )
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
    confirmed = [] if feedback is None else feedback.get("confirmed_errors") or []
    reported_character = next(
        (
            item.get("character")
            for item in confirmed
            if isinstance(item, dict) and isinstance(item.get("character"), int)
        ),
        None,
    )
    center = (
        int(match.group(1))
        if match
        else int(reported_character)
        if reported_character is not None
        else len(content) // 2
    )
    center = min(max(center, 0), len(content))
    if limit == 0:
        return f"[TRUNCATED: omitted all {len(content)} raw characters to fit the context budget]"
    start = max(0, center - limit // 2)
    stop = min(len(content), start + limit)
    start = max(0, stop - limit)
    location = (
        f" near parser character {center}"
        if match or reported_character is not None
        else ""
    )
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
        "shared_feature_and_weighted_reward_diagnostics": report.get(
            "candidate_numeric_diagnostics"
        ),
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


def _json_failure_report(exc: CandidateError) -> dict[str, Any]:
    cause = exc.__cause__
    location = "$"
    details: dict[str, Any] = {}
    if isinstance(cause, json.JSONDecodeError):
        location = f"line {cause.lineno}, column {cause.colno}"
        details = {
            "line": int(cause.lineno),
            "column": int(cause.colno),
            "character": int(cause.pos),
        }
    issue = {
        "code": "JSON_PARSE_ERROR",
        "stage": "json",
        "location": location,
        "problem": str(exc),
        "requirement": "Return one complete valid JSON object with no Markdown or surrounding text.",
        **details,
    }
    return {
        "status": "failed",
        "can_execute": False,
        "errors": [issue],
        "checks": {
            "json": {"status": "failed", "completed": True, "error_count": 1},
            "schema": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "JSON could not be parsed",
            },
            "static": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "JSON/code could not be parsed",
            },
            "execution": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "JSON/schema/static prerequisites failed",
            },
        },
    }


def _feedback_from_validation(report: dict[str, Any], category: str) -> dict[str, Any]:
    errors = list(report.get("errors") or [])
    compact_checks = {
        name: {
            key: value
            for key, value in check.items()
            if key in {"status", "skipped_reason"} and value is not None
        }
        for name, check in (report.get("checks") or {}).items()
        if isinstance(check, dict)
    }
    return {
        "category": category,
        "status": report.get("status", "failed"),
        "confirmed_errors": errors,
        "confirmed_error_count": len(errors),
        "checks": compact_checks,
        "instruction": (
            "Correct every confirmed error above and return one complete replacement "
            "candidate JSON object using the unchanged interface and evaluation rules."
        ),
    }


def _shorten_feedback_text(value: Any, limit: int = 600) -> Any:
    if not isinstance(value, str) or len(value) <= int(limit):
        return value
    kept = max(0, int(limit) - 56)
    return value[:kept] + f"...[omitted {len(value) - kept} characters]"


def _feedback_root_key(issue: dict[str, Any]) -> tuple[str, ...]:
    """Stable, value-insensitive key for one independently actionable problem."""

    source_field = str(issue.get("source_field", ""))
    feature_index = issue.get("feature_index")
    feature_identity = (
        f"feature:{int(feature_index)}"
        if isinstance(feature_index, int) and not isinstance(feature_index, bool)
        else f"mapping:{issue.get('feature_mapping', '')}"
    )
    if source_field:
        return (
            str(issue.get("code", "UNKNOWN")),
            feature_identity,
            source_field,
        )
    return (
        str(issue.get("code", "UNKNOWN")),
        str(issue.get("exception_type", "")),
        str(issue.get("candidate_function", "")),
        str(issue.get("candidate_line", issue.get("location", ""))),
    )


def _compact_feedback_issue(issue: dict[str, Any]) -> dict[str, Any]:
    retained = (
        "code",
        "stage",
        "location",
        "exception_type",
        "candidate_function",
        "candidate_line",
        "problem",
        "requirement",
        "occurrence_count",
        "line",
        "column",
        "character",
        "feature_index",
        "source_field",
        "source_locations",
        "feature_mapping",
        "matched_attempt",
        "carried_from_attempt",
        "representative_locations",
    )
    result = {
        key: _shorten_feedback_text(issue[key])
        for key in retained
        if key in issue and issue[key] is not None
    }
    if isinstance(result.get("source_locations"), list):
        locations = result["source_locations"]
        result["source_locations"] = locations[:3]
        if len(locations) > 3:
            result["source_locations_omitted"] = len(locations) - 3
    examples = issue.get("representative_samples")
    if isinstance(examples, list) and examples:
        result["representative_samples"] = examples[:1]
        if len(examples) > 1:
            result["representative_samples_omitted"] = len(examples) - 1
    return result


def _actionable_feedback_issues(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flat, independently actionable issues for duplicate revisions."""

    confirmed = feedback.get("confirmed_errors")
    if isinstance(confirmed, list) and confirmed:
        actionable = [
            dict(issue)
            for issue in confirmed
            if isinstance(issue, dict)
            and issue.get("code") != "DUPLICATE_FAILED_CANDIDATE"
        ]
        if actionable:
            return actionable
    by_lambda = feedback.get("by_lambda")
    if isinstance(by_lambda, dict):
        issues: list[dict[str, Any]] = []
        for lambda_value, result in by_lambda.items():
            if not isinstance(result, dict) or bool(result.get("passed")):
                continue
            issues.append(
                {
                    "code": "EVALUATION_NOT_IMPROVED",
                    "stage": "evaluation",
                    "location": f"lambda={lambda_value}",
                    "problem": (
                        "the candidate did not exceed the required Lipschitz improvement "
                        f"at lambda={lambda_value}; baseline={result.get('baseline_l_hat')}, "
                        f"candidate={result.get('candidate_l_hat')}"
                    ),
                    "requirement": (
                        "Revise the feature computations or reward weights so this lambda "
                        "passes the unchanged per-lambda improvement threshold."
                    ),
                    "lambda_value": str(lambda_value),
                }
            )
        if issues:
            return issues
    return [
        {
            "code": str(feedback.get("category", "CANDIDATE_FAILURE")).upper(),
            "stage": "revision",
            "location": "candidate",
            "problem": str(feedback.get("error", feedback.get("status", "candidate failed"))),
            "requirement": str(
                feedback.get(
                    "instruction",
                    "Correct the candidate and return one complete replacement JSON object.",
                )
            ),
        }
    ]


def _merge_feedback_roots(
    errors: list[dict[str, Any]], *, representative_limit: int = 3
) -> list[dict[str, Any]]:
    """Merge repeated occurrences without merging distinct actionable roots."""

    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for raw_issue in errors:
        issue = dict(raw_issue)
        key = _feedback_root_key(issue)
        count = int(issue.get("occurrence_count", 1))
        if key not in merged:
            issue["occurrence_count"] = max(1, count)
            issue["representative_locations"] = []
            merged[key] = issue
        else:
            merged[key]["occurrence_count"] += max(1, count)
        destination = merged[key]["representative_locations"]
        locations = issue.get("source_locations")
        if not isinstance(locations, list) or not locations:
            locations = [issue.get("location")]
        for location in locations:
            if (
                location is not None
                and location not in destination
                and len(destination) < int(representative_limit)
            ):
                destination.append(location)
        samples = issue.get("representative_samples")
        if isinstance(samples, list) and samples:
            saved = merged[key].setdefault("representative_samples", [])
            for sample in samples:
                if sample not in saved and len(saved) < int(representative_limit):
                    saved.append(sample)
    return list(merged.values())


def _select_feedback_representatives(
    errors: list[dict[str, Any]], maximum: int | None
) -> tuple[list[dict[str, Any]], int]:
    # Collapse repeated occurrences while retaining distinct structured roots.
    # Then select one representative from each broad code before taking
    # additional roots. This prevents a flood of one class from hiding another.
    unique = _merge_feedback_roots(errors)
    if maximum is None or len(unique) <= int(maximum):
        return unique, 0
    maximum = max(1, int(maximum))
    has_duplicate = any(
        issue.get("code") == "DUPLICATE_FAILED_CANDIDATE" for issue in unique
    )
    has_actionable = any(
        issue.get("code") != "DUPLICATE_FAILED_CANDIDATE" for issue in unique
    )
    if has_duplicate and has_actionable:
        # A duplicate warning without an original fix is not actionable.
        maximum = max(2, maximum)
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for issue in unique:
        code = str(issue.get("code", "UNKNOWN"))
        if code not in seen_codes and len(chosen) < maximum:
            chosen.append(issue)
            seen_codes.add(code)
        else:
            deferred.append(issue)
    for issue in deferred:
        if len(chosen) >= maximum:
            break
        chosen.append(issue)
    return chosen, len(unique) - len(chosen)


def _feedback_prompt_variants(
    feedback: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Return deterministic full-to-minimal prompt feedback alternatives."""

    if not isinstance(feedback, dict):
        return [("full", feedback)]
    errors = feedback.get("confirmed_errors")
    if not isinstance(errors, list) or not errors:
        return [("full", feedback)]
    variants: list[tuple[str, dict[str, Any]]] = [("full", feedback)]
    merged_roots = _merge_feedback_roots(errors)
    original_count = len(merged_roots)
    original_occurrence_count = sum(
        int(issue.get("occurrence_count", 1)) for issue in merged_roots
    )
    for maximum in (None, 64, 32, 16, 8, 4, 2, 1):
        selected, omitted = _select_feedback_representatives(errors, maximum)
        compact = {
            key: value
            for key, value in feedback.items()
            if key not in {"confirmed_errors", "confirmed_error_count"}
        }
        compact["confirmed_errors"] = [
            _compact_feedback_issue(issue) for issue in selected
        ]
        compact["confirmed_error_count"] = original_count
        compact["prompt_feedback_summary"] = {
            "summary_applied": True,
            "original_error_count": original_count,
            "included_error_count": len(selected),
            "omitted_error_count": omitted,
            "original_error_occurrence_count": original_occurrence_count,
            "selection_rule": (
                "stable first root per error code, then additional distinct roots; "
                "long prose and repeated sample cases are shortened"
            ),
            "full_report_location": "the preceding attempt's validation_report.json",
        }
        label = "compact_all_roots" if maximum is None else f"representatives_{maximum}"
        if variants[-1][1] != compact:
            variants.append((label, compact))
    return variants


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
    connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    progress_interval: float = DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
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
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    for label, value in (
        ("timeout", timeout),
        ("connect_timeout", connect_timeout),
        ("total_timeout", total_timeout),
        ("progress_interval", progress_interval),
        ("worker_timeout", worker_timeout),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive")

    arrays, fixed_metadata, baseline, constants_metadata = load_design_inputs(
        fixed_sample
    )
    output = allocate_design_directory(
        model, output_root=output_root, output_dir=output_dir
    )
    model_info = None
    api_client = client or LMStudioClient(
        base_url,
        timeout=timeout,
        connect_timeout=connect_timeout,
        total_timeout=total_timeout,
        progress_interval=progress_interval,
    )
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
            "api_connect_timeout_seconds": float(connect_timeout),
            "api_stream_idle_timeout_seconds": float(timeout),
            "api_total_timeout_seconds_per_request": float(total_timeout),
            # Kept for readers of design metadata written before streaming v1.
            "api_timeout_seconds_per_request": float(timeout),
            "api_retry_count": int(getattr(api_client, "retries", 0)),
            "api_stream_generation_retry_count": 0,
            "api_progress_interval_seconds": float(progress_interval),
            "api_total_wait_note": (
                "connect, stream-idle, and total limits apply independently to each "
                "request; generation transport failures are not retried"
            ),
            "worker_timeout_seconds": float(worker_timeout),
            "structured_output": "disabled; strict JSON parsing and local validation remain mandatory",
            "streaming": True,
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
            "stream": True,
            "stream_options": {"include_usage": True},
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
    failed_candidates: dict[str, dict[str, Any]] = {}
    for attempt in range(1, int(max_attempts) + 1):
        round_directory = output / f"attempt_{attempt:02d}"
        round_directory.mkdir()
        current_candidate_fingerprint = None
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
                250,
                100,
                0,
            ]
            if raw_failure
            else [None]
        )
        feedback_variants = (
            _feedback_prompt_variants(previous_attempt.get("feedback"))
            if previous_attempt is not None
            and previous_attempt.get("parsed_candidate") is not None
            else [("full", None)]
        )
        prompt = None
        budget = None
        selected_feedback = None
        selected_feedback_strategy = "full"
        selected_content_limit = None
        for feedback_strategy, prompt_feedback in feedback_variants:
            for content_limit in content_limits:
                request_text = _round_request(
                    attempt,
                    int(max_attempts),
                    previous_attempt,
                    failed_content_limit=content_limit,
                    feedback_override=prompt_feedback,
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
                selected_feedback = prompt_feedback
                selected_feedback_strategy = feedback_strategy
                selected_content_limit = content_limit
                if candidate_budget["fits_client_budget"]:
                    break
            if budget is not None and budget["fits_client_budget"]:
                break
        assert prompt is not None and budget is not None
        (round_directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        _write_json(round_directory / "token_budget.json", budget)
        _write_json(
            round_directory / "prompt_feedback.json",
            {
                "strategy": selected_feedback_strategy,
                "failed_raw_content_limit": selected_content_limit,
                "feedback": selected_feedback,
            },
        )
        _write_json(
            round_directory / "request_planned.json",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
                "max_tokens": int(max_output_tokens),
                "seed": int(seed),
                "stream": True,
                "stream_options": {"include_usage": True},
                "context_length_note": "budget only; not a server load-setting request",
            },
        )
        if not budget["fits_client_budget"]:
            feedback = {
                "category": "context_budget_exceeded",
                "token_budget": budget,
                "prompt_feedback_strategy": selected_feedback_strategy,
                "reason": (
                    "The complete previous candidate plus the common interface, schema, "
                    "output reservation, and minimum representative feedback do not fit. "
                    "The candidate was not truncated or rewritten."
                ),
            }
            history.append({"attempt": attempt, "status": "context_budget_exceeded"})
            break
        attempt_output = {
            "raw_final_content": None,
            "parsed_candidate": None,
            "reasoning_present": False,
            "feedback": None,
        }
        staged = None
        try:
            response = api_client.chat(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
                attempt_directory=round_directory,
            )
            # Real clients save this before opening the connection.  This write
            # also keeps injected test clients and older adapters auditable.
            if not (round_directory / "request.json").is_file():
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
                        "transport_completed",
                        "done_received",
                        "terminal_chunk_received",
                        "tool_calls_seen",
                        "stream_elapsed_seconds",
                        "request_elapsed_seconds",
                        "stream_event_count",
                        "stream_received_bytes",
                    )
                },
            )
            if response.get("actual_model") is not None:
                actual_models.append(str(response["actual_model"]))
            adapter_fallbacks.extend(str(value) for value in response.get("fallbacks", []))
            seed_sent_values.append(bool(response.get("seed_sent")))
            if not response.get("transport_completed", False):
                raise APIError(
                    "generation response was not confirmed as a complete stream",
                    category="stream_incomplete",
                )
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
            if response.get("finish_reason") != "stop":
                raise CandidateError(
                    "model output did not finish normally "
                    f"(finish_reason={response.get('finish_reason')!r})"
                )
            if response.get("tool_calls_seen"):
                raise CandidateError(
                    "model returned tool calls; one plain candidate JSON object is required"
                )
            if content is None or not str(content).strip():
                reason = "model returned reasoning but no final JSON" if response.get("reasoning") else "model returned no final content"
                raise CandidateError(reason)
            try:
                candidate = parse_candidate_json(str(content))
            except CandidateError as exc:
                validation_report = _json_failure_report(exc)
                _write_json(
                    round_directory / "validation_report.json", validation_report
                )
                feedback = _feedback_from_validation(
                    validation_report, "candidate_json_failure"
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                _write_json(round_directory / "failure.json", feedback)
                _write_json(round_directory / "feedback.json", feedback)
                history.append({"attempt": attempt, "status": "candidate_json_failure"})
                continue
            attempt_output["parsed_candidate"] = candidate
            _write_json(round_directory / "candidate.json", candidate)
            current_candidate_fingerprint = candidate_semantic_fingerprint(candidate)
            prior_failure = failed_candidates.get(current_candidate_fingerprint)
            if prior_failure is not None:
                unresolved = [
                    {**dict(issue), "carried_from_attempt": int(prior_failure["attempt"])}
                    for issue in prior_failure["unresolved_issues"]
                ]
                duplicate_report = {
                    "status": "failed",
                    "can_execute": False,
                    "errors": [
                        {
                            "code": "DUPLICATE_FAILED_CANDIDATE",
                            "stage": "revision",
                            "location": "$.features and $.code",
                            "problem": (
                                "this candidate is semantically unchanged from failed "
                                f"attempt {prior_failure['attempt']} after ignoring candidate_name, "
                                "code comments, and code formatting"
                            ),
                            "requirement": (
                                "Make a substantive correction to the concrete unresolved issues "
                                "listed below; renaming or reformatting is not a revision."
                            ),
                            "matched_attempt": int(prior_failure["attempt"]),
                        }
                    ] + unresolved,
                    "checks": {
                        "duplicate_failed_candidate": {
                            "status": "failed",
                            "matched_attempt": int(prior_failure["attempt"]),
                            "semantic_fingerprint": current_candidate_fingerprint,
                        }
                    },
                }
                _write_json(round_directory / "validation_report.json", duplicate_report)
                feedback = _feedback_from_validation(
                    duplicate_report, "duplicate_failed_candidate"
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                _write_json(round_directory / "failure.json", feedback)
                _write_json(round_directory / "feedback.json", feedback)
                history.append({"attempt": attempt, "status": "duplicate_failed_candidate"})
                continue
            staged = validate_candidate_staged(candidate, constants_metadata)
            if isinstance(candidate.get("code"), str):
                (round_directory / "candidate.py").write_text(
                    candidate["code"], encoding="utf-8"
                )
            if not staged["can_execute"]:
                _write_json(round_directory / "validation_report.json", staged)
                feedback = _feedback_from_validation(
                    staged, "candidate_schema_or_static_failure"
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "feedback": feedback,
                    "unresolved_issues": _actionable_feedback_issues(feedback),
                }
                _write_json(round_directory / "failure.json", feedback)
                _write_json(round_directory / "feedback.json", feedback)
                history.append(
                    {"attempt": attempt, "status": "candidate_schema_or_static_failure"}
                )
                continue
            static = validate_candidate(candidate, constants_metadata)
            extra, worker_reward, execution = execute_candidate_isolated(
                candidate,
                obs_arrays,
                constants_metadata,
                timeout=worker_timeout,
            )
            validation_report = {
                "status": "passed",
                "errors": [],
                "checks": {
                    **staged["checks"],
                    "execution": {"status": "passed", "completed": True},
                },
                "static": static,
                "execution": execution,
                "numeric_tolerance": 1e-6,
            }
            _write_json(round_directory / "validation_report.json", validation_report)
            evaluation = evaluate_candidate(
                context,
                candidate,
                extra,
                beta=beta,
                batch_size=batch_size,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            evaluation["candidate_numeric_diagnostics"] = candidate_numeric_diagnostics(
                arrays["state"], extra, candidate
            )
            evaluation["shared_feature_reward_consistency"] = {
                "maximum_absolute_difference": float(
                    np.max(np.abs(feature_reward(extra, candidate) - worker_reward))
                ),
                "passed": bool(
                    np.allclose(
                        feature_reward(extra, candidate),
                        worker_reward,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
            }
            if not evaluation["shared_feature_reward_consistency"]["passed"]:
                raise CandidateError(
                    "isolated worker reward does not match the authoritative host-side "
                    "dot(feature_reward_weights, extra_state) calculation"
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
            failed_candidates[current_candidate_fingerprint] = {
                "attempt": attempt,
                "feedback": feedback,
                "unresolved_issues": _actionable_feedback_issues(feedback),
            }
            _write_json(round_directory / "feedback.json", feedback)
            history.append({"attempt": attempt, "status": evaluation["status"]})
        except APIError as exc:
            category = getattr(exc, "category", "api_failure")
            feedback = _summarize_error(category, exc)
            feedback["retry_as_candidate_revision"] = False
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": category})
            break
        except CandidateExecutionError as exc:
            worker_report = exc.report or {
                "status": "failed",
                "errors": [
                    {
                        "code": "RUNTIME_WORKER_FAILURE",
                        "stage": "execution",
                        "location": "candidate worker",
                        "problem": str(exc),
                        "requirement": "Return functions that pass all isolated execution checks.",
                    }
                ],
                "checks": {"execution": {"status": "failed", "completed": False}},
            }
            execution_report = {
                "status": "failed",
                "can_execute": False,
                "errors": list(worker_report.get("errors") or []),
                "checks": {
                    **(
                        staged.get("checks", {})
                        if isinstance(staged, dict)
                        else {}
                    ),
                    **(worker_report.get("checks") or {}),
                },
            }
            _write_json(
                round_directory / "validation_report.json", execution_report
            )
            feedback = _feedback_from_validation(
                execution_report, "candidate_execution_failure"
            )
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            if current_candidate_fingerprint is not None:
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "feedback": feedback,
                    "unresolved_issues": _actionable_feedback_issues(feedback),
                }
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": "candidate_execution_failure"})
        except CandidateError as exc:
            feedback = _summarize_error("candidate_format_or_validation_failure", exc)
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            if current_candidate_fingerprint is not None:
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "feedback": feedback,
                    "unresolved_issues": _actionable_feedback_issues(feedback),
                }
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
