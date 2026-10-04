"""Offline provider candidate generation, validation, and fixed-pair scoring."""

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
    ModelCandidateSchemaError,
    candidate_semantic_fingerprint,
    candidate_numeric_diagnostics,
    normalize_candidate_submission,
    execute_candidate_isolated,
    feature_reward,
    parse_candidate_json_envelope,
    save_approved_artifact,
    validate_candidate,
    validate_candidate_staged,
)
from llm_design_contract import (
    DESIGN_RUN_SCHEMA_VERSION,
    OBS_INTERFACE_VERSION,
    NAMED_STATE_FIELD_SPECS,
    PROMPT_VERSION,
    build_obs_arrays,
    candidate_schema,
    model_candidate_schema,
    estimate_token_budget,
    load_design_inputs,
    render_prompt,
    runtime_diagnostic_contract,
)
from llm_streaming import (
    StreamTransportError,
    capture_chat_stream,
    open_http_stream,
    read_response_body_bounded,
)
from replay_auxiliary import SNAPSHOT_FIELD_SPECS


DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Backward-compatible public name used by existing LM Studio callers.
DEFAULT_BASE_URL = DEFAULT_LMSTUDIO_BASE_URL
SUPPORTED_PROVIDERS = {"lmstudio", "openai"}
GPT4O_SNAPSHOTS = {
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
}
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
EVALUATION_DIAGNOSTICS_VERSION = "uav-hrl-lipschitz-evaluation-diagnostics-v2"
EVALUATION_FINDING_ABSOLUTE_TOLERANCE = 1e-6
PAIRWISE_DIAGNOSTICS_VERSION = "uav-hrl-single-feature-pairwise-diagnostics-v1"
PAIRWISE_CLASSIFICATION_ABSOLUTE_TOLERANCE = 1e-12
PAIRWISE_CLASSIFICATION_RELATIVE_TOLERANCE = 1e-6

EVALUATION_REVISION_INSTRUCTION = """Your candidate passed the implementation checks but did not meet
the Lipschitz improvement criterion.

Evaluation diagnostics:
{evaluation_diagnostics}

{single_feature_pairwise_feedback}

Review the diagnostics before revising:

1. For the sample pairs determining the largest ratios, compare
   feature differences, weighted reward contributions, and state
   distances. Identify what limits improvement.

2. Use the relevant inputs and validity flags to check whether
   feature selection, filtering, aggregation, normalization, or
   weighting explains the result. Distinguish supported conclusions
   from hypotheses; state when the supplied evidence is insufficient.

3. Retain useful features and revise the computations or weights
   responsible for the limitations. Consider additional observable
   information when the current features miss task-relevant differences.
   Keep the design applicable beyond the diagnostic samples.

Perform this review internally. Briefly explain the reasons for your
changes in the existing feature descriptions, and return the complete
revised candidate JSON under the unchanged interface and evaluation rules."""

SINGLE_FEATURE_PAIRWISE_GUIDANCE = """The table evaluates each feature separately: it adds only that feature to the original state and adds its weighted contribution to the baseline reward. The percentages report pairwise ratios that improve, remain unchanged, or worsen relative to the baseline. The full-candidate row uses all proposed features and their current weights.

Use the task objective, single-feature results, and critical-pair diagnostics to revise the design:

- Features with higher improvement percentages may be retained or considered for increased weight.
- For features with higher worsening percentages, review the formula, data selection, normalization, and the sign and magnitude of the weight.
- Features unsuitable for directly contributing to the reward may be removed.

Check that the behavior encouraged by each formula and weight matches its description and the task objective.

These results apply to the current weights. Increasing a weight or combining features still requires re-evaluating the complete candidate. The existing maximum pairwise-ratio acceptance criterion remains unchanged."""


class APIError(RuntimeError):
    def __init__(self, message: str, *, category: str = "api_failure"):
        super().__init__(message)
        self.category = str(category)


class ContextBudgetError(ValueError):
    pass


def _redact_text(value: Any, secrets: Iterable[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[REDACTED]")
    return re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        text,
    )


def redact_provider_secrets(value: Any) -> str:
    return _redact_text(
        value,
        tuple(
            secret
            for secret in (
                os.environ.get("OPENAI_API_KEY"),
                os.environ.get("LM_STUDIO_API_TOKEN"),
            )
            if secret
        ),
    )


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
        self.provider = "lmstudio"

    def validate_configuration(self) -> None:
        return None

    def _safe(self, value: Any) -> str:
        return _redact_text(value, (self.token,) if self.token else ())

    def _payload(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        seed: int,
        reasoning_effort: str | None = None,
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
        if reasoning_effort is not None:
            payload["reasoning_effort"] = str(reasoning_effort)
        return payload

    def _http_error_category(self, status: int) -> str:
        return "http_error"

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
                error = APIError(f"LM Studio HTTP {exc.code}: {self._safe(body)}")
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
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        self.validate_configuration()
        payload = self._payload(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed,
            reasoning_effort=reasoning_effort,
        )
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
                    "error": self._safe(exc),
                },
            )
            raise APIError(self._safe(exc), category=exc.category) from exc
        if getattr(response, "status", 200) >= 300:
            remaining_total = max(
                0.0, self.total_timeout - (time.monotonic() - request_started)
            )
            body_bytes, body_status = read_response_body_bounded(
                response,
                idle_timeout=self.timeout,
                total_timeout=remaining_total,
                redact=self._safe,
            )
            body = self._safe(body_bytes.decode("utf-8", errors="replace"))
            status = {
                **body_status,
                "http_status": int(response.status),
                "http_reason": getattr(response, "reason", ""),
                "body": body,
            }
            _write_json(attempt_directory / "stream_status.json", status)
            if body_status["status"] != "complete":
                raise APIError(
                    self._safe(body_status["failure"] or "HTTP error body read failed"),
                    category=body_status["status"],
                )
            raise APIError(
                f"{self.provider} HTTP {response.status}: {body}",
                category=self._http_error_category(int(response.status)),
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
                redact=self._safe,
            )
        except StreamTransportError as exc:
            raise APIError(self._safe(exc), category=exc.category) from exc
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
            "refusal": streamed.get("refusal"),
        }


class OpenAIClient(LMStudioClient):
    """OpenAI Chat Completions client without SDK retries or model inventory calls."""

    def __init__(self, base_url=DEFAULT_OPENAI_BASE_URL, **kwargs):
        retries = int(kwargs.pop("retries", 0))
        if retries != 0:
            raise ValueError("OpenAI paid generation retries must remain disabled")
        normalized = str(base_url).rstrip("/")
        if normalized != DEFAULT_OPENAI_BASE_URL:
            raise ValueError(
                "OpenAI provider only permits the official base URL "
                f"{DEFAULT_OPENAI_BASE_URL!r}; custom endpoints are not supported"
            )
        super().__init__(normalized, retries=0, **kwargs)
        self.provider = "openai"
        self.token = os.environ.get("OPENAI_API_KEY")

    def validate_configuration(self) -> None:
        if not self.token:
            raise APIError(
                "OPENAI_API_KEY is required for OpenAI generation; set it in the "
                "environment before starting the process",
                category="authentication_error",
            )

    def list_models(self) -> dict[str, Any]:
        raise RuntimeError("OpenAI provider does not use model inventory discovery")

    def _payload(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        seed: int,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_completion_tokens": int(max_output_tokens),
            "seed": int(seed),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = str(reasoning_effort)
        return payload

    def _http_error_category(self, status: int) -> str:
        if status in {401, 403}:
            return "authentication_error"
        if status == 429:
            return "rate_limit_or_quota_error"
        return "http_error"


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


def provider_base_url(provider: str, base_url: str | None) -> str:
    provider = str(provider).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"provider must be one of {sorted(SUPPORTED_PROVIDERS)}, got {provider!r}"
        )
    if base_url is not None:
        return str(base_url).rstrip("/")
    return (
        DEFAULT_OPENAI_BASE_URL
        if provider == "openai"
        else DEFAULT_LMSTUDIO_BASE_URL
    )


def planned_chat_request(
    provider: str,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    seed: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    token_field = "max_completion_tokens" if provider == "openai" else "max_tokens"
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        token_field: int(max_output_tokens),
        "seed": int(seed),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = str(reasoning_effort)
    return request


def response_model_matches(provider: str, requested: str, returned: str) -> bool:
    """Apply provider-specific, non-prefix model identity checks."""

    if provider != "openai" or requested != "gpt-4o":
        return returned == requested
    return returned == requested or returned in GPT4O_SNAPSHOTS


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


def _pairwise_classification_counts(
    baseline_ratios: np.ndarray,
    candidate_ratios: np.ndarray,
) -> np.ndarray:
    """Count improved/unchanged/worsened pairs for one or more candidates."""

    baseline = np.asarray(baseline_ratios, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_ratios, dtype=np.float64)
    if candidate.ndim == 1:
        candidate = candidate[:, np.newaxis]
    if candidate.ndim != 2 or candidate.shape[0] != baseline.size:
        raise ValueError("pairwise ratio arrays have incompatible shapes")
    tolerance = np.maximum(
        PAIRWISE_CLASSIFICATION_ABSOLUTE_TOLERANCE,
        PAIRWISE_CLASSIFICATION_RELATIVE_TOLERANCE * np.abs(baseline),
    )[:, np.newaxis]
    reference = baseline[:, np.newaxis]
    improved = candidate < reference - tolerance
    worsened = candidate > reference + tolerance
    unchanged = ~(improved | worsened)
    return np.stack(
        (
            np.count_nonzero(improved, axis=0),
            np.count_nonzero(unchanged, axis=0),
            np.count_nonzero(worsened, axis=0),
        ),
        axis=1,
    ).astype(np.int64, copy=False)


def _pairwise_percentage(count: int, total: int) -> float | None:
    return None if total == 0 else float(count) * 100.0 / float(total)


def _single_feature_pairwise_report(
    *,
    context: "EvaluationContext",
    candidate: dict[str, Any],
    counts_by_lambda: dict[str, dict[str, np.ndarray]],
    beta: float,
) -> dict[str, Any]:
    pair_count = int(np.count_nonzero(context.primary_mask))
    result_by_lambda: dict[str, Any] = {}
    for lambda_value in context.lambdas:
        key = format(lambda_value, ".17g")
        counts = counts_by_lambda[key]
        rows = []
        for feature_index, definition in enumerate(candidate["features"]):
            improved, unchanged, worsened = map(
                int, counts["features"][feature_index]
            )
            if improved + unchanged + worsened != pair_count:
                raise RuntimeError("single-feature pair classification is incomplete")
            rows.append(
                {
                    "row_kind": "single_feature",
                    "feature_index": int(feature_index),
                    "feature_name": str(definition["name"]),
                    "current_reward_weight": float(definition["reward_weight"]),
                    "pair_count": pair_count,
                    "improved_count": improved,
                    "unchanged_count": unchanged,
                    "worsened_count": worsened,
                    "improved_percent": _pairwise_percentage(improved, pair_count),
                    "unchanged_percent": _pairwise_percentage(unchanged, pair_count),
                    "worsened_percent": _pairwise_percentage(worsened, pair_count),
                }
            )
        improved, unchanged, worsened = map(int, counts["full_candidate"])
        if improved + unchanged + worsened != pair_count:
            raise RuntimeError("full-candidate pair classification is incomplete")
        rows.append(
            {
                "row_kind": "full_candidate",
                "feature_index": None,
                "feature_name": "full candidate",
                "current_reward_weight": "all current weights",
                "pair_count": pair_count,
                "improved_count": improved,
                "unchanged_count": unchanged,
                "worsened_count": worsened,
                "improved_percent": _pairwise_percentage(improved, pair_count),
                "unchanged_percent": _pairwise_percentage(unchanged, pair_count),
                "worsened_percent": _pairwise_percentage(worsened, pair_count),
            }
        )
        result_by_lambda[key] = {
            "lambda_mbit_per_joule": float(lambda_value),
            "primary_pair_count": pair_count,
            "rows": rows,
        }
    return {
        "schema_version": PAIRWISE_DIAGNOSTICS_VERSION,
        "scope": (
            "each single-feature row independently augments the original state with "
            "only that feature and augments baseline reward with beta times its current "
            "weight and value; the full-candidate row uses all current features and weights"
        ),
        "comparison_baseline": (
            "original state and reconstructed baseline reward on the unchanged fixed "
            "primary pair set"
        ),
        "primary_pair_set_sha256": context.pair_hash,
        "primary_pair_count": pair_count,
        "beta": float(beta),
        "formulas": {
            "baseline_ratio": "abs(r_base_i-r_base_j)/norm(s_i-s_j,2)",
            "single_feature_state": "concat(s_i,[f_i_k])",
            "single_feature_reward": "r_base_i + beta*w_k*f_i_k",
            "full_candidate": "all proposed features and all current reward weights",
            "pair_tolerance": "max(1e-12,1e-6*abs(baseline_pair_ratio))",
        },
        "classification": {
            "improved": "candidate_ratio < baseline_ratio - pair_tolerance",
            "unchanged": "otherwise within the inclusive tolerance band",
            "worsened": "candidate_ratio > baseline_ratio + pair_tolerance",
            "absolute_tolerance": PAIRWISE_CLASSIFICATION_ABSOLUTE_TOLERANCE,
            "relative_tolerance": PAIRWISE_CLASSIFICATION_RELATIVE_TOLERANCE,
            "formal_acceptance_criterion_changed": False,
        },
        "percent_scale": "0-100",
        "percentages_are_unrounded_in_report": True,
        "candidate_pair_reselection": False,
        "by_lambda": result_by_lambda,
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


def _json_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _source_field_contract(
    source_field: str,
    *,
    context: EvaluationContext,
    constants_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if source_field == "obs.state":
        checkpoint = (context.fixed_metadata.get("compatibility_contract") or {}).get(
            "source_checkpoint_contract"
        ) or {}
        schema = checkpoint.get("movement_state_feature_schema") or {}
        return {
            "timing": "current-only; available before the movement action",
            "shape": [int(context.original_state.shape[1])],
            "dtype": "float32",
            "ordering_and_normalization": _json_native(schema),
            "validity": "always present in the fixed observation adapter",
        }
    if source_field == "obs.movement_mask":
        return {
            "timing": "current-only; available before the movement action",
            "shape": (
                None
                if constants_metadata is None
                else [int(constants_metadata["num_uav"]["value"])]
            ),
            "dtype": "bool",
            "semantics": "true exactly where centralized movement control owns the UAV",
            "axis_mapping": "row index equals UAV id",
            "validity": "always present; an all-false mask is a valid empty selection",
        }
    if source_field.startswith("obs."):
        name = source_field[4:]
        named = NAMED_STATE_FIELD_SPECS.get(name)
        if named is not None:
            return {
                "timing": "current-only; deterministic view of obs.state",
                "shape": list(named["shape"]),
                "dtype": np.dtype(named["dtype"]).name,
                "unit": named["unit"],
                "semantics": named["semantics"],
                "normalization": named["normalization"],
                "validity_field": (
                    None
                    if named["validity"] == "always"
                    else f"obs.{named['validity']}"
                ),
                "missing_data_rule": named["missing"],
            }
        spec = SNAPSHOT_FIELD_SPECS.get(name)
        if spec is None:
            return {"limitation": "field is not in the current observation contract"}
        return {
            "timing": "current-only; available before the movement action",
            "shape": list(spec["shape"]),
            "dtype": np.dtype(spec["dtype"]).name,
            "unit": spec["unit"],
            "semantics": spec["semantics"],
            "validity_field": (
                None if spec.get("mask") is None else f"obs.{spec['mask']}"
            ),
        }
    if source_field.startswith("constants."):
        name = source_field[10:]
        item = None if constants_metadata is None else constants_metadata.get(name)
        if not isinstance(item, dict):
            return {"limitation": "constant metadata was unavailable"}
        return {
            key: _json_native(item.get(key))
            for key in ("dtype", "unit", "meaning", "source")
        }
    return {"limitation": "unrecognized source field"}


def _candidate_diagnostic_dependencies(
    candidate: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str], list[str], dict[str, list[str]]]:
    by_feature: dict[str, list[str]] = {}
    declared: set[str] = set()
    for index, feature in enumerate(candidate["features"]):
        fields = sorted(
            value
            for value in feature.get("source_fields", [])
            if isinstance(value, str)
        )
        by_feature[str(index)] = fields
        declared.update(fields)
    supporting: set[str] = set()
    reasons: dict[str, set[str]] = {}

    def add_support(field: str, reason: str) -> None:
        supporting.add(field)
        reasons.setdefault(field, set()).add(reason)

    add_support(
        "obs.movement_mask",
        "distinguishes the current centralized-control set and its empty-set case",
    )
    for field in declared:
        if not field.startswith("obs."):
            continue
        name = field[4:]
        named = NAMED_STATE_FIELD_SPECS.get(name)
        if named is not None and named["validity"] != "always":
            validity_field = f"obs.{named['validity']}"
            add_support(
                validity_field,
                f"is the authoritative validity flag for {field}",
            )
        spec = SNAPSHOT_FIELD_SPECS.get(name)
        if spec is not None:
            if field != "obs.snapshot_valid":
                add_support(
                    "obs.snapshot_valid",
                    f"establishes whether the snapshot containing {field} was recorded",
                )
            if spec.get("mask"):
                add_support(
                    f"obs.{spec['mask']}",
                    f"is the authoritative validity mask for {field}",
                )
        if name.startswith("sr_") or name.startswith("s2u_"):
            add_support("obs.sr_id", f"maps compact SR rows used by {field} to SR ids")
            add_support(
                "obs.sr_observable",
                f"marks which compact SR rows used by {field} are observable",
            )
        if name.startswith("roi_") or name.startswith("vs_"):
            add_support("obs.roi_id", f"maps compact RoI rows relevant to {field}")
            add_support(
                "obs.roi_observable",
                f"marks which compact RoI rows relevant to {field} are observable",
            )
        if name.startswith("sr_roi_"):
            add_support(
                "obs.sr_roi_id", f"provides the observable SR-to-RoI id mapping for {field}"
            )
            add_support(
                "obs.sr_roi_mapping_valid",
                f"validates the SR-to-RoI id mapping used to interpret {field}",
            )
        if name.startswith("task_"):
            for support_name, purpose in (
                ("task_type", "identifies whether a target id denotes an RoI or SR"),
                ("task_target_id", "provides the assigned service-target id"),
                ("task_pair_valid", "marks populated task-assignment slots"),
                ("task_target_valid", "marks decision-visible service targets"),
                ("sr_id", "maps COM target ids to observable SR rows"),
                ("sr_observable", "marks observable SR rows"),
                ("roi_id", "maps VS target ids to discovered RoI rows"),
                ("roi_observable", "marks observable RoI rows"),
            ):
                add_support(f"obs.{support_name}", f"{purpose} for {field}")
    support_only = sorted(supporting.difference(declared))
    return (
        by_feature,
        sorted(declared),
        support_only,
        {field: sorted(reasons.get(field, set())) for field in support_only},
    )


def _sample_input_diagnostic(
    sample_index: int,
    fields: Iterable[str],
    *,
    obs_arrays: dict[str, np.ndarray] | None,
    constants_metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    limitations: list[str] = []
    for source_field in fields:
        if source_field.startswith("constants."):
            name = source_field[10:]
            item = None if constants_metadata is None else constants_metadata.get(name)
            if not isinstance(item, dict) or "value" not in item:
                limitations.append(f"{source_field}: value unavailable")
                continue
            value = item["value"]
        elif source_field.startswith("obs."):
            name = source_field[4:]
            if obs_arrays is None or name not in obs_arrays:
                limitations.append(f"{source_field}: adapter value unavailable")
                continue
            array = np.asarray(obs_arrays[name])
            if sample_index >= array.shape[0]:
                limitations.append(f"{source_field}: sample index is out of range")
                continue
            value = array[sample_index]
        else:
            limitations.append(f"{source_field}: source namespace is unsupported")
            continue
        array = np.asarray(value)
        entry = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "value": _json_native(array),
            "empty_array": bool(array.size == 0),
        }
        if array.size:
            if np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, np.bool_
            ):
                entry["zero_element_count"] = int(np.count_nonzero(array == 0))
            if np.issubdtype(array.dtype, np.bool_):
                entry["true_element_count"] = int(np.count_nonzero(array))
        values[source_field] = entry
    return values, limitations


def _pair_numeric_findings(
    *,
    pair_records: dict[str, Any],
    lambda_records: dict[str, Any],
    extra: np.ndarray,
    reward_extra: np.ndarray,
    obs_arrays: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    pair_lambdas: dict[str, list[str]] = {}
    for lambda_key, record in lambda_records.items():
        pair_lambdas.setdefault(str(record["maximum_pair_ref"]), []).append(lambda_key)
    findings: list[dict[str, Any]] = []
    tolerance = float(EVALUATION_FINDING_ABSOLUTE_TOLERANCE)
    movement = None if obs_arrays is None else obs_arrays.get("movement_mask")
    movement = None if movement is None else np.asarray(movement, dtype=bool)
    for pair_ref, pair in pair_records.items():
        i = int(str(pair["sample_i_ref"]).split("_", 1)[1])
        j = int(str(pair["sample_j_ref"]).split("_", 1)[1])
        difference = np.asarray(extra[i] - extra[j], dtype=np.float64)
        maximum_absolute_difference = (
            0.0 if difference.size == 0 else float(np.max(np.abs(difference)))
        )
        difference_norm = float(np.linalg.norm(difference))
        exactly_equal = bool(np.array_equal(extra[i], extra[j]))
        within_tolerance = bool(
            np.allclose(extra[i], extra[j], rtol=0.0, atol=tolerance)
        )
        extra_reward_difference = float(reward_extra[i] - reward_extra[j])
        common = {
            "pair_ref": pair_ref,
            "lambda_refs": sorted(pair_lambdas.get(pair_ref, [])),
            "evidence": {
                "feature_vector_comparison": (
                    "exactly_equal"
                    if exactly_equal
                    else "different_but_within_tolerance"
                    if within_tolerance
                    else "different_beyond_tolerance"
                ),
                "feature_difference_l2_norm": difference_norm,
                "maximum_absolute_feature_difference": maximum_absolute_difference,
                "extra_reward_difference_i_minus_j": extra_reward_difference,
            },
        }
        if exactly_equal:
            findings.append(
                {
                    "code": "IDENTICAL_ADDED_FEATURE_VECTOR_ON_MAXIMUM_PAIR",
                    **common,
                    "message": (
                        "The added feature vectors are identical on this pair. "
                        "Consequently, the added state distance and the extra-reward "
                        "difference are zero. Changing only the fixed weights while "
                        "preserving these feature outputs cannot change this pair's ratio."
                    ),
                    "revision_direction": (
                        "Review feature selection and computation for this pair rather "
                        "than changing only the fixed weights."
                    ),
                    "scope_note": (
                        "This conclusion applies only to this pair and these exact "
                        "feature outputs; it is not a claim about all fixed samples."
                    ),
                }
            )
        elif extra_reward_difference == 0.0:
            findings.append(
                {
                    "code": "EQUAL_EXTRA_REWARD_WITH_DIFFERENT_FEATURE_VECTOR",
                    **common,
                    "message": (
                        "The added feature vectors differ, but the current fixed weights "
                        "produce equal extra reward on this pair. The current extra reward "
                        "does not change the pair's reward difference, while the added "
                        "features can still change its state distance."
                    ),
                    "revision_direction": (
                        "Inspect the per-feature weighted contributions for cancellation "
                        "or missing task-relevant distinctions."
                    ),
                    "scope_note": (
                        "Equal extra reward under the current weights does not imply that "
                        "the added features have no effect."
                    ),
                }
            )
        if not exactly_equal and within_tolerance:
            findings.append(
                {
                    "code": "NEAR_BUT_NOT_IDENTICAL_ADDED_FEATURE_VECTOR",
                    **common,
                    "message": (
                        "The added feature vectors are not exactly identical, although "
                        "their differences are within the recorded diagnostic tolerance."
                    ),
                    "revision_direction": (
                        "Treat the nonzero differences as measured evidence; do not infer "
                        "that changing weights is mathematically unable to affect this pair."
                    ),
                    "scope_note": "No exact-equality conclusion is made for this pair.",
                }
            )
        if (
            movement is not None
            and i < movement.shape[0]
            and j < movement.shape[0]
            and not np.any(movement[i])
            and not np.any(movement[j])
        ):
            findings.append(
                {
                    "code": "EMPTY_MOVEMENT_MASK_ON_BOTH_PAIR_OBSERVATIONS",
                    "pair_ref": pair_ref,
                    "lambda_refs": sorted(pair_lambdas.get(pair_ref, [])),
                    "evidence": {
                        "sample_i_true_indices": [],
                        "sample_j_true_indices": [],
                        "movement_mask_shape": list(movement[i].shape),
                    },
                    "message": (
                        "Both observations have an empty movement_mask. Inspect whether "
                        "empty-selection handling limits feature discrimination. An empty "
                        "control mask does not imply that all observable information is invalid."
                    ),
                    "revision_direction": (
                        "Check empty-selection branches while retaining other valid "
                        "current-only observable inputs."
                    ),
                    "scope_note": (
                        "This is an observed condition, not evidence that the movement "
                        "mask caused every feature output."
                    ),
                }
            )
    return findings


def _build_lipschitz_evaluation_diagnostics(
    *,
    context: EvaluationContext,
    candidate: dict[str, Any],
    extra: np.ndarray,
    reward_extra: np.ndarray,
    candidate_rewards: dict[str, np.ndarray],
    candidate_distances: np.ndarray,
    trackers: dict[str, dict[str, Any]],
    by_lambda: dict[str, Any],
    beta: float,
    obs_arrays: dict[str, np.ndarray] | None,
    constants_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    host_managed_dependencies = all(
        str(item.get("formula", "")).startswith(
            "Host-managed executable definition:"
        )
        for item in candidate["features"]
    )
    by_feature, declared_fields, supporting_fields, supporting_field_reasons = (
        _candidate_diagnostic_dependencies(candidate)
    )
    all_input_fields = sorted(set(declared_fields).union(supporting_fields))
    observation_fields = [
        field for field in all_input_fields if field.startswith("obs.")
    ]
    constant_fields = [
        field for field in all_input_fields if field.startswith("constants.")
    ]
    unique_pairs = sorted(
        {tuple(tracker["pair"]) for tracker in trackers.values()}
    )
    sample_indices = sorted({index for pair in unique_pairs for index in pair})
    samples: dict[str, Any] = {}
    limitations: list[str] = []
    for index in sample_indices:
        values, sample_limitations = _sample_input_diagnostic(
            index,
            observation_fields,
            obs_arrays=obs_arrays,
            constants_metadata=constants_metadata,
        )
        samples[f"sample_{index}"] = {
            "sample_index": index,
            "trace": _sample_trace(context.fixed_metadata, index),
            "current_only_inputs": values,
        }
        limitations.extend(sample_limitations)
    constant_values, constant_limitations = _sample_input_diagnostic(
        0,
        constant_fields,
        obs_arrays=None,
        constants_metadata=constants_metadata,
    )
    limitations.extend(constant_limitations)

    pair_records: dict[str, Any] = {}
    lambda_records: dict[str, Any] = {}
    for key, result in by_lambda.items():
        i, j = trackers[key]["pair"]
        rank = int(trackers[key]["rank"])
        pair_ref = f"pair_{i}_{j}"
        if pair_ref not in pair_records:
            features = []
            for index, definition in enumerate(candidate["features"]):
                weight = float(definition["reward_weight"])
                value_i = float(extra[i, index])
                value_j = float(extra[j, index])
                contribution_i = weight * value_i
                contribution_j = weight * value_j
                features.append(
                    {
                        "index": index,
                        "name": definition["name"],
                        "reward_weight": weight,
                        "feature_i": value_i,
                        "feature_j": value_j,
                        "feature_difference_i_minus_j": value_i - value_j,
                        "weighted_contribution_excludes_beta_i": contribution_i,
                        "weighted_contribution_excludes_beta_j": contribution_j,
                        "weighted_contribution_difference_i_minus_j_excludes_beta": (
                            contribution_i - contribution_j
                        ),
                        "source_fields": by_feature[str(index)],
                        "source_field_mapping": (
                            "aggregate literal subscriptions for the candidate; "
                            "not asserted as exact for this output"
                            if host_managed_dependencies
                            else "legacy model-declared per-feature fields"
                        ),
                    }
                )
            pair_records[pair_ref] = {
                "sample_i_ref": f"sample_{i}",
                "sample_j_ref": f"sample_{j}",
                "difference_direction": "all signed differences are sample i minus sample j",
                "original_state_distance": float(context.original_distances[rank]),
                "augmented_state_distance": float(candidate_distances[rank]),
                "features": features,
                "sum_weighted_contributions_excludes_beta_i": float(
                    sum(
                        item["weighted_contribution_excludes_beta_i"]
                        for item in features
                    )
                ),
                "sum_weighted_contributions_excludes_beta_j": float(
                    sum(
                        item["weighted_contribution_excludes_beta_j"]
                        for item in features
                    )
                ),
                "extra_reward_i": float(reward_extra[i]),
                "extra_reward_j": float(reward_extra[j]),
                "extra_reward_difference_i_minus_j": float(
                    reward_extra[i] - reward_extra[j]
                ),
                "extra_reward_identity": (
                    "sum_k(reward_weight_k * feature_k); feature contributions above "
                    "exclude beta, which is applied only when forming total reward"
                ),
                "weighted_sum_matches_extra_reward": bool(
                    math.isclose(
                        sum(
                            item["weighted_contribution_excludes_beta_i"]
                            for item in features
                        ),
                        float(reward_extra[i]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        sum(
                            item["weighted_contribution_excludes_beta_j"]
                            for item in features
                        ),
                        float(reward_extra[j]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
            }
        base = context.base_rewards[key]
        total = candidate_rewards[key]
        original_distance = float(context.original_distances[rank])
        augmented_distance = float(candidate_distances[rank])
        base_difference = float(base[i] - base[j])
        total_difference = float(total[i] - total[j])
        lambda_records[key] = {
            "lambda_mbit_per_joule": result["lambda_mbit_per_joule"],
            "maximum_pair_ref": pair_ref,
            "global_baseline_l_hat_over_all_primary_pairs": result["baseline_l_hat"],
            "global_candidate_l_hat_over_all_primary_pairs": result["candidate_l_hat"],
            "improvement": result["improvement"],
            "required_margin": result["required_margin"],
            "passed": result["passed"],
            "reward_outcomes_after_action": {
                "base_reward_i": float(base[i]),
                "base_reward_j": float(base[j]),
                "base_reward_difference_i_minus_j": base_difference,
                "extra_reward_i": float(reward_extra[i]),
                "extra_reward_j": float(reward_extra[j]),
                "extra_reward_difference_i_minus_j": float(
                    reward_extra[i] - reward_extra[j]
                ),
                "total_reward_i": float(total[i]),
                "total_reward_j": float(total[j]),
                "total_reward_difference_i_minus_j": total_difference,
                "total_reward_identity": "r_total = r_base + beta * r_extra",
                "beta": float(beta),
            },
            "ratios_on_this_candidate_maximum_pair": {
                "baseline_ratio_abs_base_difference_over_original_distance": (
                    abs(base_difference) / original_distance
                ),
                "candidate_ratio_abs_total_difference_over_augmented_distance": (
                    abs(total_difference) / augmented_distance
                ),
                "note": (
                    "the baseline ratio on this pair is not the global baseline "
                    "maximum unless the baseline happens to maximize on the same pair"
                ),
            },
        }

    contracts = {
        field: _source_field_contract(
            field, context=context, constants_metadata=constants_metadata
        )
        for field in all_input_fields
    }
    findings = _pair_numeric_findings(
        pair_records=pair_records,
        lambda_records=lambda_records,
        extra=extra,
        reward_extra=reward_extra,
        obs_arrays=obs_arrays,
    )
    finding_definitions: dict[str, dict[str, str]] = {}
    for finding in findings:
        code = str(finding["code"])
        finding_definitions.setdefault(
            code,
            {
                key: str(finding.pop(key))
                for key in ("message", "revision_direction", "scope_note")
            },
        )
    observed_names = {
        field[4:] for field in all_input_fields if field.startswith("obs.")
    }
    axis_and_id_mapping: dict[str, str] = {}
    if any(
        name == "state"
        or name == "movement_mask"
        or name.startswith(("uav_", "u2u_", "u2g_", "s2u_", "task_", "vs_"))
        for name in observed_names
    ):
        axis_and_id_mapping["uav"] = "UAV array row index equals UAV id"
    if any(
        name.startswith(("sr_", "roi_", "s2u_", "task_", "vs_"))
        for name in observed_names
    ):
        axis_and_id_mapping["roi_and_sr"] = (
            "RoI/SR arrays use compact padded rows; roi_id/sr_id must be matched "
            "under observability/validity flags and are not row indices"
        )
        axis_and_id_mapping["missingness"] = (
            "invalid/padded IDs use -1 and numeric padding uses zero; validity flags "
            "distinguish missing/invalid data from valid zero values and empty queues"
        )
    if any(name.startswith(("u2u_", "u2g_", "s2u_")) for name in observed_names):
        axis_and_id_mapping["links"] = (
            "U2U=[sender_uav,receiver_uav], U2G=[sender_uav], "
            "S2U=[compact_sr_row,receiver_uav]"
        )
    return {
        "schema_version": EVALUATION_DIAGNOSTICS_VERSION,
        "diagnostic_scope": (
            "candidate maximum-ratio pairs from the unchanged fixed primary pair set; "
            "this is finite-sample evidence, not a training guarantee"
        ),
        "input_timing": (
            "candidate inputs are action-pre current-only obs/constants; reward outcomes "
            "below are evaluation results and are not candidate inputs"
        ),
        "dependency_selection": {
            "basis": (
                "host-recorded aggregate literal field subscriptions, plus their "
                "validity masks and movement_mask"
                if host_managed_dependencies
                else "legacy candidate source_fields accepted by static validation, "
                "plus their declared validity masks and movement_mask"
            ),
            "feature_source_fields": by_feature,
            "declared_source_fields": declared_fields,
            "supporting_fields_added_for_interpretation": supporting_fields,
            "supporting_field_reasons": supporting_field_reasons,
            "limitation": (
                "host-recorded source_fields are aggregate candidate dependencies and "
                "are not mapped to individual outputs; the analysis is not a general "
                "runtime tracer and infers no unverified expression or state slice"
                if host_managed_dependencies
                else "legacy source_fields identify allowed dependencies but are not a "
                "general runtime tracer; no unverified expression or state slice is inferred"
            ),
        },
        "axis_and_id_mapping": axis_and_id_mapping,
        "source_field_contracts": contracts,
        "constants": constant_values,
        "samples": samples,
        "pairs": pair_records,
        "by_lambda": lambda_records,
        "findings": findings,
        "finding_definitions": finding_definitions,
        "finding_contract": {
            "exact_equality": "np.array_equal over the complete added feature vectors",
            "near_equality": (
                "np.allclose with relative_tolerance=0 and the recorded absolute tolerance"
            ),
            "absolute_tolerance": float(EVALUATION_FINDING_ABSOLUTE_TOLERANCE),
            "findings_are_rejection_conditions": False,
        },
        "deduplication": {
            "unique_pair_count": len(pair_records),
            "unique_sample_count": len(samples),
            "lambda_count": len(lambda_records),
            "pairs_and_samples_are_referenced_instead_of_repeated": True,
        },
        "limitations": sorted(set(limitations)),
    }


def evaluate_candidate(
    context: EvaluationContext,
    candidate: dict[str, Any],
    extra_state: np.ndarray,
    *,
    beta: float,
    batch_size: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    obs_arrays: dict[str, np.ndarray] | None = None,
    constants_metadata: dict[str, Any] | None = None,
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
    feature_weights = np.asarray(
        [float(item["reward_weight"]) for item in candidate["features"]],
        dtype=np.float64,
    )
    pairwise_counts = {
        key: {
            "features": np.zeros((extra.shape[1], 3), dtype=np.int64),
            "full_candidate": np.zeros(3, dtype=np.int64),
        }
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
            primary_extra_difference = extra_difference[primary]
            single_feature_distances = np.sqrt(
                original_distance[primary, np.newaxis] ** 2
                + primary_extra_difference**2
            )
            primary_pair_i = pair_i[primary]
            primary_pair_j = pair_j[primary]
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
            base_reward = context.base_rewards[key]
            signed_base_difference = (
                base_reward[primary_pair_i] - base_reward[primary_pair_j]
            )
            baseline_ratios = (
                np.abs(signed_base_difference) / original_distance[primary]
            )
            single_feature_reward_differences = (
                signed_base_difference[:, np.newaxis]
                + float(beta)
                * feature_weights[np.newaxis, :]
                * primary_extra_difference
            )
            single_feature_ratios = (
                np.abs(single_feature_reward_differences)
                / single_feature_distances
            )
            pairwise_counts[key]["features"] += _pairwise_classification_counts(
                baseline_ratios, single_feature_ratios
            )
            pairwise_counts[key]["full_candidate"] += (
                _pairwise_classification_counts(baseline_ratios, ratios)[0]
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
    evaluation_diagnostics = _build_lipschitz_evaluation_diagnostics(
        context=context,
        candidate=candidate,
        extra=extra,
        reward_extra=reward_extra,
        candidate_rewards=candidate_rewards,
        candidate_distances=candidate_distances,
        trackers=trackers,
        by_lambda=by_lambda,
        beta=beta,
        obs_arrays=obs_arrays,
        constants_metadata=constants_metadata,
    )
    single_feature_pairwise = _single_feature_pairwise_report(
        context=context,
        candidate=candidate,
        counts_by_lambda=pairwise_counts,
        beta=beta,
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
        "single_feature_pairwise_comparison": single_feature_pairwise,
        "evaluation_diagnostics": evaluation_diagnostics,
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


def _format_pairwise_percentage(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _single_feature_pairwise_feedback_table(comparison: Any) -> str:
    if not isinstance(comparison, dict):
        return ""
    by_lambda = comparison.get("by_lambda")
    if not isinstance(by_lambda, dict) or not by_lambda:
        return ""
    sections = ["Single-feature pairwise comparison:"]
    for key, record in by_lambda.items():
        if not isinstance(record, dict):
            continue
        sections.extend(
            [
                "",
                f"lambda = {key}",
                "| feature | current weight | improved (%) | unchanged (%) | worsened (%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in record.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("row_kind") == "full_candidate":
                label = "full candidate"
                weight = "all current weights"
            else:
                name = str(row.get("feature_name", "")).replace("|", "\\|")
                label = f"{row.get('feature_index')}: {name}"
                weight = format(float(row.get("current_reward_weight", 0.0)), ".17g")
            sections.append(
                "| "
                + " | ".join(
                    (
                        label,
                        weight,
                        _format_pairwise_percentage(row.get("improved_percent")),
                        _format_pairwise_percentage(row.get("unchanged_percent")),
                        _format_pairwise_percentage(row.get("worsened_percent")),
                    )
                )
                + " |"
            )
    sections.extend(["", SINGLE_FEATURE_PAIRWISE_GUIDANCE])
    return "\n".join(sections)


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
    parsed = previous_attempt.get("parsed_model_candidate")
    if parsed is None:
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
    if (
        isinstance(feedback, dict)
        and feedback.get("category") == "candidate_not_improved_for_all_lambdas"
    ):
        diagnostics = feedback.get("evaluation_diagnostics")
        rendered_feedback = EVALUATION_REVISION_INSTRUCTION.format(
            evaluation_diagnostics=json.dumps(
                diagnostics,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            single_feature_pairwise_feedback=(
                _single_feature_pairwise_feedback_table(
                    feedback.get("single_feature_pairwise_comparison")
                )
            ),
        )
        historical = feedback.get("historical_unverified_errors")
        if isinstance(historical, list) and historical:
            rendered_feedback += (
                "\n\nPrior issues not revalidated in this attempt:\n"
                + json.dumps(
                    historical, indent=2, ensure_ascii=False, allow_nan=False
                )
            )
    else:
        rendered_feedback = json.dumps(
            feedback, indent=2, ensure_ascii=False, allow_nan=False
        )
    return (
        f"Revision round {attempt} of {max_attempts}. Return a complete replacement JSON object, not a patch.\n\n"
        + previous
        + "\n\nLatest validation/evaluation feedback for that same output, followed "
        "by any separately marked prior issues that were not revalidated in this "
        "attempt:\n"
        + rendered_feedback
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
        "feedback_contract_version": EVALUATION_DIAGNOSTICS_VERSION,
        "status": report["status"],
        "by_lambda": report["by_lambda"],
        "single_feature_pairwise_comparison": report[
            "single_feature_pairwise_comparison"
        ],
        "evaluation_diagnostics": report["evaluation_diagnostics"],
        "instruction": (
            "Review the supplied evaluation diagnostics internally, briefly explain "
            "changes in the existing feature descriptions, and return one complete "
            "replacement candidate JSON under the unchanged interface and evaluation rules."
        ),
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
        "requirement": (
            "Return one complete valid JSON object, either plain or inside exactly one "
            "complete `json`/unlabelled Markdown fence. Explanatory text may surround "
            "that unique fence, but do not provide additional fences or candidate objects."
        ),
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


def _model_schema_failure_report(exc: ModelCandidateSchemaError) -> dict[str, Any]:
    return {
        "status": "failed",
        "can_execute": False,
        "errors": list(exc.issues),
        "checks": {
            "json": {"status": "passed", "completed": True, "error_count": 0},
            "schema": {
                "status": "failed",
                "completed": True,
                "error_count": len(exc.issues),
            },
            "static": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "simplified candidate schema validation failed",
            },
            "execution": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "schema/static prerequisites failed",
            },
        },
    }


def _feedback_from_validation(
    report: dict[str, Any],
    category: str,
    historical_unverified: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors = []
    tracked_historical = list(historical_unverified or [])
    report_historical = []
    for raw_issue in report.get("errors") or []:
        issue = dict(raw_issue)
        if issue.get("issue_status") == "not_revalidated" or issue.get(
            "carried_from_attempt"
        ) is not None:
            report_historical.append(issue)
        else:
            issue.setdefault("issue_status", "confirmed_current")
            errors.append(issue)
    # A duplicate report contains the most specific carry provenance for this
    # attempt. Keep it ahead of the tracker copy when both describe one root.
    historical = _merge_feedback_roots(report_historical + tracked_historical)
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
        "confirmed_error_status": "confirmed_current",
        "historical_unverified_errors": historical,
        "historical_unverified_error_count": len(historical),
        "historical_error_status": "not_revalidated",
        "checks": compact_checks,
        "instruction": (
            "Correct every confirmed_current error above, and also address each prior "
            "issue marked not_revalidated. Return one complete replacement candidate "
            "JSON object using the unchanged interface and evaluation rules."
        ),
    }


def _tracked_issue_key(issue: dict[str, Any]) -> tuple[str, ...]:
    """Use a stable AST operation when available; otherwise preserve location."""

    if str(issue.get("stage", "")) == "execution":
        operation = str(issue.get("operation_fingerprint", ""))
        return (
            str(issue.get("code", "UNKNOWN")),
            str(issue.get("exception_type", "")),
            str(issue.get("candidate_function", "")),
            str(issue.get("problem_signature", issue.get("problem", ""))),
            (
                f"operation:{operation}"
                if operation
                else "location:"
                + str(issue.get("candidate_line", issue.get("location", "")))
            ),
        )
    return _feedback_root_key(issue)


def _issue_stage(issue: dict[str, Any]) -> str:
    stage = str(issue.get("stage", ""))
    return {
        "json": "json",
        "schema": "schema",
        "static": "static",
        "execution": "execution",
        "evaluation": "evaluation",
    }.get(stage, stage or "unknown")


class _IssueTracker:
    """Track validation evidence without treating an unrun check as a fix."""

    def __init__(self):
        self._records: dict[tuple[str, ...], dict[str, Any]] = {}

    @staticmethod
    def _identity(candidate_identity: dict[str, Any] | None) -> dict[str, Any]:
        return dict(candidate_identity or {"kind": "unparsed_response"})

    def apply(
        self,
        report: dict[str, Any],
        *,
        attempt: int,
        candidate_identity: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        checks = report.get("checks") or {}
        completed_checks = {
            str(name)
            for name, check in checks.items()
            if isinstance(check, dict) and check.get("status") == "passed"
        }
        for stage, check in checks.items():
            if not isinstance(check, dict):
                continue
            for name, subcheck in (check.get("subchecks") or {}).items():
                if isinstance(subcheck, dict) and subcheck.get("completed") is True:
                    completed_checks.add(f"{stage}.{name}")
        evidence = {
            "attempt": int(attempt),
            "candidate": self._identity(candidate_identity),
            "completed_checks": sorted(completed_checks),
            "passed_checks": sorted(
                str(name)
                for name, check in checks.items()
                if isinstance(check, dict) and check.get("status") == "passed"
            ),
        }

        current_keys = set()
        for raw_issue in report.get("errors") or []:
            if not isinstance(raw_issue, dict):
                continue
            issue = dict(raw_issue)
            if issue.get("carried_from_attempt") is not None:
                continue
            if issue.get("code") == "DUPLICATE_FAILED_CANDIDATE":
                continue
            key = _tracked_issue_key(issue)
            current_keys.add(key)
            issue_id = hashlib.sha256(
                json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            if key not in self._records:
                self._records[key] = {
                    "issue_id": issue_id,
                    "stage": _issue_stage(issue),
                    "validation_check": issue.get(
                        "validation_check", _issue_stage(issue)
                    ),
                    "status": "confirmed_current",
                    "first_detected_attempt": int(attempt),
                    "latest_confirmed_attempt": int(attempt),
                    "source_candidate": self._identity(candidate_identity),
                    "source_location": issue.get("location"),
                    "issue": issue,
                    "history": [],
                }
            record = self._records[key]
            record.pop("resolved_attempt", None)
            record.pop("resolution_evidence", None)
            record.update(
                {
                    "status": "confirmed_current",
                    "latest_confirmed_attempt": int(attempt),
                    "latest_candidate": self._identity(candidate_identity),
                    "latest_location": issue.get("location"),
                    "latest_validation_attempt": int(attempt),
                    "issue": issue,
                }
            )
            record["history"].append(
                {
                    "attempt": int(attempt),
                    "status": "confirmed_current",
                    "candidate": self._identity(candidate_identity),
                    "location": issue.get("location"),
                }
            )

        carried = []
        for key, record in self._records.items():
            if key in current_keys:
                continue
            validation_check = str(
                record.get("validation_check", record.get("stage", "unknown"))
            )
            if validation_check in completed_checks:
                if record["status"] != "resolved":
                    record["status"] = "resolved"
                    record["resolved_attempt"] = int(attempt)
                    record["latest_validation_attempt"] = int(attempt)
                    record["resolution_evidence"] = {
                        **evidence,
                        "validation_check": validation_check,
                        "reason": (
                            f"the {validation_check} check completed over its configured "
                            "scope and did not report this issue"
                        ),
                    }
                    record["history"].append(
                        {
                            "attempt": int(attempt),
                            "status": "resolved",
                            "evidence": record["resolution_evidence"],
                        }
                    )
                continue
            if record["status"] == "resolved":
                continue
            record["status"] = "not_revalidated"
            record["latest_validation_attempt"] = int(attempt)
            carry = dict(record["issue"])
            carry.update(
                {
                    "issue_status": "not_revalidated",
                    "tracked_issue_id": record["issue_id"],
                    "source_attempt": int(record["first_detected_attempt"]),
                    "source_candidate": record["source_candidate"],
                    "source_location": record.get("source_location"),
                    "latest_confirmed_attempt": int(
                        record["latest_confirmed_attempt"]
                    ),
                    "latest_confirmed_candidate": record.get(
                        "latest_candidate", record["source_candidate"]
                    ),
                    "latest_confirmed_location": record.get(
                        "latest_location", record.get("source_location")
                    ),
                    "latest_validation_attempt": int(attempt),
                    "feedback_carried_forward_by_attempt": int(attempt),
                    "status_note": (
                        "Previously confirmed, but this attempt did not complete the "
                        f"{validation_check} check. The source and latest-confirmed "
                        "locations belong to their recorded candidates and are not asserted "
                        "as lines in the current candidate."
                    ),
                }
            )
            carried.append(carry)
            record["history"].append(
                {
                    "attempt": int(attempt),
                    "status": "not_revalidated",
                    "evidence": evidence,
                }
            )
        return carried

    def annotate_confirmed(
        self, errors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        annotated = []
        for raw_issue in errors:
            issue = dict(raw_issue)
            if (
                issue.get("carried_from_attempt") is None
                and issue.get("code") != "DUPLICATE_FAILED_CANDIDATE"
            ):
                record = self._records.get(_tracked_issue_key(issue))
                if record is not None:
                    issue.update(
                        {
                            "issue_status": "confirmed_current",
                            "tracked_issue_id": record["issue_id"],
                            "source_attempt": int(record["first_detected_attempt"]),
                            "source_candidate": record["source_candidate"],
                            "source_location": record.get("source_location"),
                            "latest_confirmed_attempt": int(
                                record["latest_confirmed_attempt"]
                            ),
                            "latest_confirmed_candidate": record.get(
                                "latest_candidate", record["source_candidate"]
                            ),
                            "latest_confirmed_location": record.get(
                                "latest_location", record.get("source_location")
                            ),
                        }
                    )
            annotated.append(issue)
        return annotated

    def snapshot(self) -> dict[str, Any]:
        records = sorted(
            (dict(value) for value in self._records.values()),
            key=lambda item: (int(item["first_detected_attempt"]), item["issue_id"]),
        )
        return {
            "schema_version": "uav-hrl-llm-issue-tracker-v2",
            "records": records,
            "pending_count": sum(item["status"] != "resolved" for item in records),
            "resolved_count": sum(item["status"] == "resolved" for item in records),
        }


def _generated_feedback(
    feedback: dict[str, Any], *, attempt: int, max_attempts: int
) -> dict[str, Any]:
    feedback = dict(feedback)
    feedback["feedback_provenance"] = {
        "record_role": "validation_feedback_generated_from_response",
        "source_attempt": int(attempt),
        "intended_next_attempt": (
            int(attempt) + 1 if int(attempt) < int(max_attempts) else None
        ),
    }
    return feedback


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
    operation = str(issue.get("operation_fingerprint", ""))
    return (
        str(issue.get("code", "UNKNOWN")),
        str(issue.get("exception_type", "")),
        str(issue.get("candidate_function", "")),
        (
            f"operation:{operation}"
            if operation
            else "location:"
            + str(issue.get("candidate_line", issue.get("location", "")))
        ),
    )


def _compact_feedback_issue(issue: dict[str, Any]) -> dict[str, Any]:
    retained = (
        "code",
        "stage",
        "location",
        "json_path",
        "exception_type",
        "candidate_function",
        "candidate_line",
        "candidate_source_line",
        "problem",
        "requirement",
        "occurrence_count",
        "line",
        "column",
        "character",
        "feature_index",
        "feature_name",
        "observed_value",
        "allowed_range",
        "observed_shape",
        "observed_dtype",
        "source_field",
        "source_locations",
        "feature_mapping",
        "matched_attempt",
        "duplicate_matched_attempt",
        "carried_from_attempt",
        "feedback_carried_from_attempt",
        "feedback_carried_forward_by_attempt",
        "reused_validation_report",
        "reused_validation_report_from_attempt",
        "issue_status",
        "tracked_issue_id",
        "source_attempt",
        "source_candidate",
        "source_location",
        "latest_confirmed_attempt",
        "latest_confirmed_candidate",
        "latest_confirmed_location",
        "latest_validation_attempt",
        "status_note",
        "validation_check",
        "write_target",
        "write_kind",
        "target_sources",
        "runtime_diagnostics",
    )
    result = {
        key: _shorten_feedback_text(issue[key])
        for key in retained
        if key in issue and issue[key] is not None
    }
    diagnostics = result.get("runtime_diagnostics")
    if isinstance(diagnostics, dict):
        # Full diagnostics remain in validation_report.json.  The revision
        # prompt keeps the failing operation, nearby source, relevant shapes,
        # and interface facts without allowing one runtime root to crowd out
        # other independently actionable roots.
        line = issue.get("candidate_line")
        excerpt = diagnostics.get("candidate_source_excerpt")
        if isinstance(excerpt, list):
            nearby = [
                item
                for item in excerpt
                if isinstance(item, dict)
                and (
                    not isinstance(line, int)
                    or abs(int(item.get("line", -1000000)) - line) <= 1
                )
            ]
        else:
            nearby = None
        source_analysis = diagnostics.get("index_source_analysis")
        relevant_names: set[str] = set()
        origin_line = None
        if isinstance(source_analysis, dict):
            for key in ("indexed_expression", "mask_expression"):
                value = source_analysis.get(key)
                if isinstance(value, str) and value.isidentifier():
                    relevant_names.add(value)
            origin = source_analysis.get("state_slice_origin")
            if isinstance(origin, dict) and isinstance(origin.get("line"), int):
                origin_line = int(origin["line"])
        if isinstance(excerpt, list) and origin_line is not None:
            for item in excerpt:
                if (
                    isinstance(item, dict)
                    and item.get("line") == origin_line
                    and item not in nearby
                ):
                    nearby.insert(0, item)
        local_summaries = diagnostics.get("local_array_summaries")
        if isinstance(local_summaries, dict) and relevant_names:
            local_summaries = {
                name: value
                for name, value in local_summaries.items()
                if name in relevant_names
            }
        compact_diagnostics = {
            key: diagnostics[key]
            for key in (
                "diagnostic_status",
                "related_interface",
                "diagnostic_summary",
                "targeted_requirement",
                "diagnostic_error_type",
                "diagnostic_error",
            )
            if key in diagnostics
        }
        operation = diagnostics.get("traceback_operation")
        if isinstance(operation, dict):
            compact_diagnostics["traceback_operation"] = {
                key: operation[key]
                for key in ("node_type", "source")
                if key in operation
            }
        if isinstance(source_analysis, dict):
            compact_diagnostics["index_source_analysis"] = {
                key: source_analysis[key]
                for key in (
                    "status",
                    "indexed_expression",
                    "mask_expression",
                    "reason",
                    "state_slice_origin",
                )
                if key in source_analysis
            }
        if nearby:
            compact_diagnostics["candidate_source_excerpt"] = nearby
        if local_summaries:
            compact_diagnostics["local_array_summaries"] = local_summaries
        result["runtime_diagnostics"] = compact_diagnostics
    if result.get("latest_confirmed_candidate") == result.get("source_candidate"):
        result.pop("latest_confirmed_candidate", None)
    if result.get("latest_confirmed_location") == result.get("source_location"):
        result.pop("latest_confirmed_location", None)
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
    historical = feedback.get("historical_unverified_errors")
    combined = []
    if isinstance(confirmed, list):
        combined.extend(confirmed)
    if isinstance(historical, list):
        combined.extend(historical)
    if combined:
        actionable = [
            dict(issue)
            for issue in combined
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
    feedback_groups = {
        str(issue.get("_feedback_group"))
        for issue in unique
        if issue.get("_feedback_group") is not None
    }
    if len(feedback_groups) > 1:
        # A current format error without at least one pending historical root
        # recreates the loss of actionable runtime feedback this summary exists
        # to prevent.
        maximum = max(len(feedback_groups), maximum)
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


def _coordinate_lists(mask: np.ndarray) -> list[list[int]]:
    return [list(map(int, coordinate)) for coordinate in np.argwhere(mask)]


def _lossless_boolean_diagnostic(
    entry: dict[str, Any], array: np.ndarray
) -> tuple[dict[str, Any], int, int]:
    result = dict(entry)
    true_coordinates = _coordinate_lists(np.asarray(array, dtype=bool))
    result["value"] = {
        "representation": "lossless_true_coordinates",
        "original_shape": list(array.shape),
        "coordinate_order": "row-major axes in the original array order",
        "true_coordinates": true_coordinates,
        "all_unlisted_positions_are_false": True,
        "true_element_count": len(true_coordinates),
        "empty_true_set": len(true_coordinates) == 0,
        "complete_value_in_prompt": True,
    }
    result.pop("zero_element_count", None)
    result.pop("true_element_count", None)
    result.pop("shape", None)
    result.pop("empty_array", None)
    return result, int(array.size), 0


def _valid_data_coordinates(
    array: np.ndarray, validity: np.ndarray
) -> list[tuple[int, ...]] | None:
    if array.shape == validity.shape:
        return [tuple(map(int, coordinate)) for coordinate in np.argwhere(validity)]
    if (
        array.ndim >= validity.ndim
        and tuple(array.shape[: validity.ndim]) == tuple(validity.shape)
    ):
        suffix_shape = array.shape[validity.ndim :]
        suffixes = list(np.ndindex(suffix_shape)) if suffix_shape else [()]
        return [
            tuple(map(int, prefix)) + tuple(map(int, suffix))
            for prefix in np.argwhere(validity)
            for suffix in suffixes
        ]
    return None


def _validity_aligned_diagnostic(
    entry: dict[str, Any],
    array: np.ndarray,
    validity: np.ndarray,
    validity_field: str,
) -> tuple[dict[str, Any], int, int] | None:
    coordinates = _valid_data_coordinates(array, validity)
    if coordinates is None:
        return None
    selected = [
        {"coordinate": list(coordinate), "value": _json_native(array[coordinate])}
        for coordinate in coordinates
    ]
    result = dict(entry)
    result["value"] = {
        "representation": "validity_aligned_original_coordinates",
        "original_shape": list(array.shape),
        "coordinate_order": "original array axes; coordinates are not reindexed",
        "validity_field": validity_field,
        "validity_shape": list(validity.shape),
        "validity_true_coordinates": _coordinate_lists(validity),
        "selected_elements": selected,
        "selection_method": (
            "all values whose authoritative validity coordinate is true; trailing "
            "data axes are retained in full"
        ),
        "valid_mask_true_count": int(np.count_nonzero(validity)),
        "retained_data_element_count": len(selected),
        "omitted_invalid_data_element_count": int(array.size) - len(selected),
        "empty_valid_set": not bool(np.any(validity)),
        "complete_valid_values_in_prompt": True,
        "complete_value_in_prompt": len(selected) == int(array.size),
    }
    result.pop("zero_element_count", None)
    result.pop("true_element_count", None)
    result.pop("shape", None)
    result.pop("empty_array", None)
    return result, len(selected), int(array.size) - len(selected)


def _unresolved_large_array_diagnostic(
    entry: dict[str, Any],
    array: np.ndarray,
    *,
    field: str,
    validity_field: str | None = None,
    validity_always: bool = False,
    validity_alignment_issue: str | None = None,
) -> tuple[dict[str, Any], int, int]:
    result = dict(entry)
    is_state = field == "obs.state"
    if validity_alignment_issue is not None:
        selection_method = (
            f"no partial selection because {validity_alignment_issue}; "
            "no valid coordinates were guessed"
        )
    elif validity_always:
        selection_method = (
            "the field is valid at every position and has no validity-mask field; "
            "no partial selection was inferred for this oversized value"
        )
    else:
        selection_method = (
            "no partial selection; the field has no authoritative validity mask"
        )
    result["value"] = {
        "representation": (
            "state_dependency_indices_unresolved"
            if is_state
            else "unmasked_large_value_not_partially_selected"
        ),
        "original_shape": list(array.shape),
        "original_element_count": int(array.size),
        "selected_elements": [],
        "selected_coordinates": [],
        "selection_method": (
            "no state indices selected because verified per-feature state dependency "
            "indices are unavailable"
            if is_state
            else selection_method
        ),
        "omitted_element_count": int(array.size),
        "complete_value_in_prompt": False,
        "dependency_indices_known": False if is_state else None,
    }
    if not is_state:
        result["value"]["validity_field"] = validity_field
        result["value"]["validity_is_always"] = bool(validity_always)
        result["value"]["validity_alignment_issue"] = validity_alignment_issue
    if array.size and np.issubdtype(array.dtype, np.number):
        finite = array.astype(np.float64)
        if np.all(np.isfinite(finite)):
            result["value"]["minimum"] = float(np.min(finite))
            result["value"]["maximum"] = float(np.max(finite))
    result.pop("zero_element_count", None)
    result.pop("true_element_count", None)
    result.pop("shape", None)
    result.pop("empty_array", None)
    return result, 0, int(array.size)


def _diagnostic_validity_contract(field: str) -> tuple[str | None, bool]:
    """Return the authoritative mask field, or whether validity is unconditional."""

    if not field.startswith("obs."):
        return None, False
    name = field[4:]
    named = NAMED_STATE_FIELD_SPECS.get(name)
    if named is not None:
        validity = named.get("validity")
        if validity == "always":
            return None, True
        if isinstance(validity, str) and validity:
            return f"obs.{validity}", False
        return None, False
    snapshot = SNAPSHOT_FIELD_SPECS.get(name)
    mask = None if snapshot is None else snapshot.get("mask")
    return (None if not mask else f"obs.{mask}"), False


def _compact_diagnostic_inputs(
    inputs: dict[str, Any], maximum_items: int
) -> tuple[dict[str, Any], int, int, int]:
    original = json.loads(json.dumps(inputs, ensure_ascii=False, allow_nan=False))
    compact: dict[str, Any] = {}
    included = omitted = summarized = 0
    for field, entry in original.items():
        if not isinstance(entry, dict) or "value" not in entry:
            compact[field] = entry
            continue
        try:
            array = np.asarray(entry["value"])
        except (TypeError, ValueError):
            compact[field] = entry
            continue
        if np.issubdtype(array.dtype, np.bool_):
            result, kept, removed = _lossless_boolean_diagnostic(entry, array)
        else:
            validity_field, validity_always = _diagnostic_validity_contract(field)
            aligned = None
            validity_alignment_issue = None
            if validity_field:
                validity_entry = original.get(validity_field)
                if not isinstance(validity_entry, dict) or "value" not in validity_entry:
                    validity_alignment_issue = (
                        f"authoritative validity field {validity_field} is missing"
                    )
                else:
                    try:
                        validity = np.asarray(validity_entry["value"], dtype=bool)
                    except (TypeError, ValueError) as exc:
                        validity_alignment_issue = (
                            f"authoritative validity field {validity_field} could not "
                            f"be read as boolean ({type(exc).__name__})"
                        )
                    else:
                        aligned = _validity_aligned_diagnostic(
                            entry, array, validity, validity_field
                        )
                        if aligned is None:
                            validity_alignment_issue = (
                                f"authoritative validity field {validity_field} has shape "
                                f"{list(validity.shape)}, incompatible with data shape "
                                f"{list(array.shape)}"
                            )
            if aligned is not None:
                result, kept, removed = aligned
            elif int(array.size) <= int(maximum_items):
                result, kept, removed = entry, int(array.size), 0
            else:
                result, kept, removed = _unresolved_large_array_diagnostic(
                    entry,
                    array,
                    field=field,
                    validity_field=validity_field,
                    validity_always=validity_always,
                    validity_alignment_issue=validity_alignment_issue,
                )
        compact[field] = result
        included += kept
        omitted += removed
        summarized += int(removed > 0 or result.get("value") != entry.get("value"))
    return compact, included, omitted, summarized


def _compact_evaluation_diagnostics(
    diagnostics: dict[str, Any], maximum_array_items: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact = json.loads(
        json.dumps(diagnostics, ensure_ascii=False, allow_nan=False)
    )
    included = 0
    omitted = 0
    summarized_fields = 0
    for sample in (compact.get("samples") or {}).values():
        inputs = sample.get("current_only_inputs") if isinstance(sample, dict) else None
        trace = sample.get("trace") if isinstance(sample, dict) else None
        if isinstance(trace, dict):
            trace_keys = (
                ("fixed_index",)
                if int(maximum_array_items) == 0
                else ("fixed_index", "episode_id", "td3_step", "scenario_index")
            )
            sample["trace"] = {key: trace[key] for key in trace_keys if key in trace}
        if not isinstance(inputs, dict):
            continue
        compact_inputs, kept, removed, summarized = _compact_diagnostic_inputs(
            inputs, maximum_array_items
        )
        sample["current_only_inputs"] = compact_inputs
        included += kept
        omitted += removed
        summarized_fields += summarized
    constants = compact.get("constants") or {}
    if isinstance(constants, dict):
        compact_constants, kept, removed, summarized = _compact_diagnostic_inputs(
            constants, maximum_array_items
        )
        compact["constants"] = compact_constants
        included += kept
        omitted += removed
        summarized_fields += summarized
    if int(maximum_array_items) <= 4:
        for pair in (compact.get("pairs") or {}).values():
            if not isinstance(pair, dict):
                continue
            pair.pop("sum_weighted_contributions_excludes_beta_i", None)
            pair.pop("sum_weighted_contributions_excludes_beta_j", None)
            pair.pop("weighted_sum_matches_extra_reward", None)
            for feature in pair.get("features") or []:
                if isinstance(feature, dict):
                    feature.pop("source_fields", None)
    if int(maximum_array_items) == 0:
        compact.pop("diagnostic_scope", None)
        compact.pop("input_timing", None)
        dependency = compact.get("dependency_selection")
        if isinstance(dependency, dict):
            dependency.pop("basis", None)
            dependency.pop("limitation", None)
        for field, contract in list(
            (compact.get("source_field_contracts") or {}).items()
        ):
            if not isinstance(contract, dict):
                continue
            compact["source_field_contracts"][field] = {
                "contract_reference": (
                    "the current-only environment interface in this same prompt"
                ),
                **{
                    key: contract[key]
                    for key in ("shape", "dtype", "unit", "validity_field")
                    if key in contract and contract[key] is not None
                },
            }
        for pair in (compact.get("pairs") or {}).values():
            if isinstance(pair, dict):
                pair.pop("difference_direction", None)
                pair.pop("extra_reward_identity", None)
        for record in (compact.get("by_lambda") or {}).values():
            if not isinstance(record, dict):
                continue
            outcomes = record.get("reward_outcomes_after_action")
            if isinstance(outcomes, dict):
                outcomes.pop("total_reward_identity", None)
            ratios = record.get("ratios_on_this_candidate_maximum_pair")
            if isinstance(ratios, dict):
                ratios.pop("note", None)

    state_contract = (compact.get("source_field_contracts") or {}).get("obs.state")
    if isinstance(state_contract, dict):
        schema = state_contract.get("ordering_and_normalization")
        features = schema.get("features") if isinstance(schema, dict) else None
        if isinstance(features, list):
            kept = features[: max(0, min(int(maximum_array_items), len(features)))]
            summarized_schema = {
                key: schema[key]
                for key in ("schema_version", "dimension", "ordering")
                if key in schema
            }
            if kept:
                summarized_schema["features"] = {
                    "representation": "partial_ordered_feature_schema",
                    "selected_features": kept,
                    "selected_indices": [item.get("index") for item in kept],
                    "selection_method": "first N authoritative schema entries",
                    "total_feature_count": len(features),
                    "omitted_feature_count": len(features) - len(kept),
                    "complete_schema_in_prompt": False,
                }
            else:
                summarized_schema["features"] = {
                    "representation": "schema_entries_omitted",
                    "selected_indices": [],
                    "selection_method": "none at the minimum prompt budget",
                    "total_feature_count": len(features),
                    "omitted_feature_count": len(features),
                    "complete_schema_in_prompt": False,
                }
            for index_kind in ("continuous_indices", "discrete_indices"):
                values = schema.get(index_kind)
                if isinstance(values, list):
                    summarized_schema[f"{index_kind}_count"] = len(values)
            state_contract["ordering_and_normalization"] = summarized_schema
            included += len(kept)
            omitted += len(features) - len(kept)
            summarized_fields += int(len(features) > len(kept))
    if int(maximum_array_items) == 0:
        for sample in (compact.get("samples") or {}).values():
            if isinstance(sample, dict) and "current_only_inputs" in sample:
                sample["current_only_inputs"] = {
                    "details_omitted_from_prompt": True,
                    "contract_reference": (
                        "the named current-only input interface in this same prompt"
                    ),
                    "full_values_location": (
                        "the preceding attempt's evaluation_report.json and feedback.json"
                    ),
                }
        if "constants" in compact:
            compact["constants"] = {
                "details_omitted_from_prompt": True,
                "contract_reference": (
                    "the constants table in the current-only interface in this same prompt"
                ),
                "full_values_location": (
                    "the preceding attempt's evaluation_report.json and feedback.json"
                ),
            }
        if "source_field_contracts" in compact:
            compact["source_field_contracts"] = {
                "details_omitted_from_prompt": True,
                "contract_reference": (
                    "the named current-only input interface in this same prompt"
                ),
            }
        compact.pop("axis_and_id_mapping", None)
    summary = {
        "summary_applied": True,
        "array_item_limit_per_field": int(maximum_array_items),
        "included_array_or_schema_elements": included,
        "omitted_array_or_schema_elements": omitted,
        "summarized_field_count": summarized_fields,
        "selection_is_stable": True,
        "full_diagnostics_location": (
            "the preceding attempt's evaluation_report.json and feedback.json"
        ),
        "candidate_code_truncated_or_rewritten": False,
    }
    return compact, summary


def _evaluation_diagnostic_element_count(diagnostics: dict[str, Any]) -> int:
    total = 0
    for sample in (diagnostics.get("samples") or {}).values():
        inputs = sample.get("current_only_inputs") if isinstance(sample, dict) else None
        if not isinstance(inputs, dict):
            continue
        for entry in inputs.values():
            if isinstance(entry, dict) and "value" in entry:
                try:
                    total += int(np.asarray(entry["value"]).size)
                except (TypeError, ValueError):
                    pass
    for entry in (diagnostics.get("constants") or {}).values():
        if isinstance(entry, dict) and "value" in entry:
            try:
                total += int(np.asarray(entry["value"]).size)
            except (TypeError, ValueError):
                pass
    state_contract = (diagnostics.get("source_field_contracts") or {}).get(
        "obs.state"
    )
    if isinstance(state_contract, dict):
        schema = state_contract.get("ordering_and_normalization")
        features = schema.get("features") if isinstance(schema, dict) else None
        if isinstance(features, list):
            total += len(features)
    return total


def _feedback_prompt_variants(
    feedback: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Return deterministic full-to-minimal prompt feedback alternatives."""

    if not isinstance(feedback, dict):
        return [("full", feedback)]
    if feedback.get("category") == "candidate_not_improved_for_all_lambdas":
        diagnostics = feedback.get("evaluation_diagnostics")
        if not isinstance(diagnostics, dict):
            return [("full", feedback)]
        original_elements = _evaluation_diagnostic_element_count(diagnostics)
        full = {
            **feedback,
            "prompt_feedback_summary": {
                "summary_applied": False,
                "included_array_or_schema_elements": original_elements,
                "omitted_array_or_schema_elements": 0,
                "full_diagnostics_location": (
                    "the preceding attempt's evaluation_report.json and feedback.json"
                ),
                "candidate_code_truncated_or_rewritten": False,
            },
        }
        variants: list[tuple[str, dict[str, Any]]] = [("full", full)]
        for maximum in (128, 64, 32, 16, 8, 4, 0):
            compact_diagnostics, summary = _compact_evaluation_diagnostics(
                diagnostics, maximum
            )
            compact = {
                **feedback,
                "evaluation_diagnostics": compact_diagnostics,
                "prompt_feedback_summary": summary,
            }
            label = f"evaluation_diagnostics_{maximum}_items"
            if variants[-1][1] != compact:
                variants.append((label, compact))
        return variants
    current_errors = feedback.get("confirmed_errors")
    historical_errors = feedback.get("historical_unverified_errors")
    current_errors = current_errors if isinstance(current_errors, list) else []
    historical_errors = historical_errors if isinstance(historical_errors, list) else []
    if not current_errors and not historical_errors:
        return [("full", feedback)]
    variants: list[tuple[str, dict[str, Any]]] = [("full", feedback)]
    errors = [
        {**dict(issue), "_feedback_group": "confirmed_current"}
        for issue in current_errors
        if isinstance(issue, dict)
    ] + [
        {**dict(issue), "_feedback_group": "historical_unverified"}
        for issue in historical_errors
        if isinstance(issue, dict)
    ]
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
            if key
            not in {
                "confirmed_errors",
                "confirmed_error_count",
                "historical_unverified_errors",
                "historical_unverified_error_count",
            }
        }
        selected_current = []
        selected_historical = []
        for issue in selected:
            destination = (
                selected_historical
                if issue.get("_feedback_group") == "historical_unverified"
                else selected_current
            )
            clean = {key: value for key, value in issue.items() if key != "_feedback_group"}
            destination.append(_compact_feedback_issue(clean))
        compact["confirmed_errors"] = selected_current
        compact["confirmed_error_count"] = len(current_errors)
        compact["historical_unverified_errors"] = selected_historical
        compact["historical_unverified_error_count"] = len(historical_errors)
        compact["prompt_feedback_summary"] = {
            "summary_applied": True,
            "original_error_count": original_count,
            "included_error_count": len(selected),
            "omitted_error_count": omitted,
            "original_error_occurrence_count": original_occurrence_count,
            "confirmed_current_original_count": len(current_errors),
            "confirmed_current_included_count": len(selected_current),
            "historical_unverified_original_count": len(historical_errors),
            "historical_unverified_included_count": len(selected_historical),
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
    provider: str = "lmstudio",
    base_url: str | None = None,
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
    provider = str(provider).lower()
    resolved_base_url = provider_base_url(provider, base_url)
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
    if client is not None:
        api_client = client
    else:
        client_type = OpenAIClient if provider == "openai" else LMStudioClient
        api_client = client_type(
            resolved_base_url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            total_timeout=total_timeout,
            progress_interval=progress_interval,
        )
    if not dry_run:
        validate_configuration = getattr(api_client, "validate_configuration", None)
        if callable(validate_configuration):
            validate_configuration()
        if provider == "lmstudio":
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
        "model_candidate_schema": model_candidate_schema(),
        "internal_candidate_schema": candidate_schema(),
        "model": {
            "provider": provider,
            "requested_api_identifier": model,
            "adapter": model_adapter(model),
            "base_url": resolved_base_url,
            "inventory": model_info,
            "quantization_or_load_information": (
                "not applicable to OpenAI provider"
                if provider == "openai"
                else "unknown in dry-run; no API request was sent"
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
            "provider": provider,
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
            "diagnostics_version": EVALUATION_DIAGNOSTICS_VERSION,
            "beta": float(beta),
            "batch_size": int(batch_size),
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
            "all_lambdas_must_pass": True,
            "candidate_pair_reselection": False,
        },
        "observation_interface_version": OBS_INTERFACE_VERSION,
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
            **planned_chat_request(
                provider,
                model=model,
                prompt=first_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
            ),
            "context_length_note": (
                "client budget only; not sent to alter provider model capacity"
            ),
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
    worker_diagnostic_contract = runtime_diagnostic_contract(
        fixed_metadata, constants_metadata
    )
    previous_attempt = None
    feedback = None
    history = []
    approved = None
    actual_models = []
    adapter_fallbacks = []
    seed_sent_values = []
    failed_candidates: dict[str, dict[str, Any]] = {}
    issue_tracker = _IssueTracker()

    def tracked_validation_feedback(
        report: dict[str, Any],
        category: str,
        *,
        attempt_number: int,
        candidate_identity: dict[str, Any] | None,
        directory: Path,
    ) -> dict[str, Any]:
        historical_unverified = issue_tracker.apply(
            report,
            attempt=attempt_number,
            candidate_identity=candidate_identity,
        )
        report_root_keys = {
            _tracked_issue_key(issue)
            for issue in report.get("errors") or []
            if isinstance(issue, dict)
        }
        historical_unverified = [
            issue
            for issue in historical_unverified
            if _tracked_issue_key(issue) not in report_root_keys
        ]
        snapshot = issue_tracker.snapshot()
        _write_json(directory / "issue_tracker.json", snapshot)
        _write_json(output / "issue_tracker.json", snapshot)
        annotated_report = dict(report)
        annotated_report["errors"] = issue_tracker.annotate_confirmed(
            list(report.get("errors") or [])
        )
        return _generated_feedback(
            _feedback_from_validation(
                annotated_report,
                category,
                historical_unverified=historical_unverified,
            ),
            attempt=attempt_number,
            max_attempts=max_attempts,
        )

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
            else [
                (
                    "full",
                    None
                    if previous_attempt is None
                    else previous_attempt.get("feedback"),
                )
            ]
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
                "record_role": "incoming_feedback_used_to_build_request",
                "target_attempt": attempt,
                "source_attempt": None if previous_attempt is None else attempt - 1,
                "source_candidate_identity": (
                    None
                    if previous_attempt is None
                    else previous_attempt.get("candidate_identity")
                ),
                "strategy": selected_feedback_strategy,
                "failed_raw_content_limit": selected_content_limit,
                "feedback": selected_feedback,
            },
        )
        _write_json(
            round_directory / "request_planned.json",
            {
                **planned_chat_request(
                    provider,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    seed=seed,
                ),
                "context_length_note": "budget only; not a server capacity-setting request",
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
            "parsed_model_candidate": None,
            "reasoning_present": False,
            "feedback": None,
            "candidate_identity": None,
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
                        "refusal",
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
            if response.get("actual_model") is not None and not response_model_matches(
                provider, model, str(response.get("actual_model"))
            ):
                raise APIError(
                    "provider returned a different model identifier: "
                    f"provider={provider!r}, requested={model!r}, "
                    f"actual={response.get('actual_model')!r}"
                )
            if response.get("refusal"):
                raise APIError(
                    f"model refused the generation request: {response.get('refusal')}",
                    category="model_refusal",
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
                submitted_candidate, parse_metadata = parse_candidate_json_envelope(str(content))
            except CandidateError as exc:
                validation_report = _json_failure_report(exc)
                _write_json(
                    round_directory / "validation_report.json", validation_report
                )
                raw_identity = {
                    "kind": "unparsed_response",
                    "raw_final_content_sha256": hashlib.sha256(
                        str(content).encode("utf-8")
                    ).hexdigest(),
                }
                attempt_output["candidate_identity"] = raw_identity
                feedback = tracked_validation_feedback(
                    validation_report,
                    "candidate_json_failure",
                    attempt_number=attempt,
                    candidate_identity=raw_identity,
                    directory=round_directory,
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                _write_json(round_directory / "failure.json", feedback)
                _write_json(round_directory / "feedback.json", feedback)
                history.append({"attempt": attempt, "status": "candidate_json_failure"})
                continue
            attempt_output["parsed_model_candidate"] = submitted_candidate
            _write_json(
                round_directory / "candidate_parse_metadata.json", parse_metadata
            )
            _write_json(
                round_directory / "parsed_model_candidate.json", submitted_candidate
            )
            try:
                model_candidate, candidate, transformation = normalize_candidate_submission(
                    submitted_candidate, constants_metadata
                )
                parse_metadata["host_transformation"] = transformation
                _write_json(
                    round_directory / "candidate_parse_metadata.json", parse_metadata
                )
                _write_json(round_directory / "model_candidate.json", model_candidate)
            except ModelCandidateSchemaError as exc:
                validation_report = _model_schema_failure_report(exc)
                _write_json(
                    round_directory / "validation_report.json", validation_report
                )
                serialized = json.dumps(
                    submitted_candidate,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                candidate_identity = {
                    "kind": "parsed_model_candidate_schema_failure",
                    "model_candidate_sha256": hashlib.sha256(
                        serialized.encode("utf-8")
                    ).hexdigest(),
                }
                attempt_output["candidate_identity"] = candidate_identity
                feedback = tracked_validation_feedback(
                    validation_report,
                    "candidate_model_schema_failure",
                    attempt_number=attempt,
                    candidate_identity=candidate_identity,
                    directory=round_directory,
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                _write_json(round_directory / "failure.json", feedback)
                _write_json(round_directory / "feedback.json", feedback)
                history.append(
                    {"attempt": attempt, "status": "candidate_model_schema_failure"}
                )
                continue
            attempt_output["parsed_model_candidate"] = model_candidate
            attempt_output["parsed_candidate"] = candidate
            _write_json(round_directory / "candidate.json", candidate)
            current_candidate_fingerprint = candidate_semantic_fingerprint(candidate)
            candidate_identity = {
                "kind": "parsed_candidate",
                "semantic_fingerprint": current_candidate_fingerprint,
                "candidate_name": candidate.get("candidate_name"),
            }
            attempt_output["candidate_identity"] = candidate_identity
            # Parsing success is independent evidence that an earlier JSON-envelope
            # problem is fixed, even if duplicate detection stops later checks.
            issue_tracker.apply(
                {
                    "errors": [],
                    "checks": {"json": {"status": "passed", "completed": True}},
                },
                attempt=attempt,
                candidate_identity=candidate_identity,
            )
            prior_failure = failed_candidates.get(current_candidate_fingerprint)
            if prior_failure is not None:
                unresolved = []
                for raw_issue in prior_failure["unresolved_issues"]:
                    issue = dict(raw_issue)
                    original_source_attempt = int(
                        issue.get("source_attempt", prior_failure["attempt"])
                    )
                    original_source_candidate = issue.get(
                        "source_candidate",
                        prior_failure.get(
                            "candidate_identity",
                            {
                                "kind": "parsed_candidate",
                                "semantic_fingerprint": current_candidate_fingerprint,
                            },
                        ),
                    )
                    original_source_location = issue.get(
                        "source_location", issue.get("location")
                    )
                    issue.update(
                        {
                            "carried_from_attempt": int(attempt - 1),
                            "feedback_carried_from_attempt": int(attempt - 1),
                            "duplicate_matched_attempt": int(
                                prior_failure["attempt"]
                            ),
                            "reused_validation_report": True,
                            "reused_validation_report_from_attempt": int(
                                prior_failure["attempt"]
                            ),
                            "issue_status": "not_revalidated",
                            "source_attempt": original_source_attempt,
                            "source_candidate": original_source_candidate,
                            "source_location": original_source_location,
                            "latest_confirmed_attempt": int(
                                issue.get(
                                    "latest_confirmed_attempt",
                                    prior_failure["attempt"],
                                )
                            ),
                            "latest_confirmed_candidate": issue.get(
                                "latest_confirmed_candidate",
                                prior_failure.get("candidate_identity"),
                            ),
                            "latest_confirmed_location": issue.get(
                                "latest_confirmed_location", issue.get("location")
                            ),
                            "status_note": (
                                "The duplicate check reused the validation report from "
                                f"attempt {prior_failure['attempt']}; the current candidate "
                                "was not revalidated. Source fields identify the original "
                                "candidate, while latest-confirmed fields identify the most "
                                "recent actual confirmation."
                            ),
                        }
                    )
                    unresolved.append(issue)
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
                            "reused_validation_report": True,
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
                feedback = tracked_validation_feedback(
                    duplicate_report,
                    "duplicate_failed_candidate",
                    attempt_number=attempt,
                    candidate_identity=candidate_identity,
                    directory=round_directory,
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
                feedback = tracked_validation_feedback(
                    staged,
                    "candidate_schema_or_static_failure",
                    attempt_number=attempt,
                    candidate_identity=candidate_identity,
                    directory=round_directory,
                )
                attempt_output["feedback"] = feedback
                previous_attempt = attempt_output
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "candidate_identity": candidate_identity,
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
                diagnostic_contract=worker_diagnostic_contract,
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
            issue_tracker.apply(
                validation_report,
                attempt=attempt,
                candidate_identity=candidate_identity,
            )
            tracker_snapshot = issue_tracker.snapshot()
            _write_json(round_directory / "issue_tracker.json", tracker_snapshot)
            _write_json(output / "issue_tracker.json", tracker_snapshot)
            evaluation = evaluate_candidate(
                context,
                candidate,
                extra,
                beta=beta,
                batch_size=batch_size,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                obs_arrays=obs_arrays,
                constants_metadata=constants_metadata,
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
                        "provider": provider,
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
            feedback = _generated_feedback(
                _feedback_from_evaluation(evaluation),
                attempt=attempt,
                max_attempts=max_attempts,
            )
            feedback["feedback_provenance"]["source_candidate_identity"] = (
                candidate_identity
            )
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            failed_candidates[current_candidate_fingerprint] = {
                "attempt": attempt,
                "candidate_identity": candidate_identity,
                "feedback": feedback,
                "unresolved_issues": _actionable_feedback_issues(feedback),
            }
            _write_json(round_directory / "feedback.json", feedback)
            history.append({"attempt": attempt, "status": evaluation["status"]})
        except APIError as exc:
            category = getattr(exc, "category", "api_failure")
            feedback = _generated_feedback(
                _summarize_error(category, exc),
                attempt=attempt,
                max_attempts=max_attempts,
            )
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
            candidate_identity = (
                {
                    "kind": "parsed_candidate",
                    "semantic_fingerprint": current_candidate_fingerprint,
                    "candidate_name": (
                        attempt_output.get("parsed_candidate") or {}
                    ).get("candidate_name"),
                }
                if current_candidate_fingerprint is not None
                else {"kind": "candidate_execution_without_fingerprint"}
            )
            feedback = tracked_validation_feedback(
                execution_report,
                "candidate_execution_failure",
                attempt_number=attempt,
                candidate_identity=candidate_identity,
                directory=round_directory,
            )
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            if current_candidate_fingerprint is not None:
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "candidate_identity": candidate_identity,
                    "feedback": feedback,
                    "unresolved_issues": _actionable_feedback_issues(feedback),
                }
            _write_json(round_directory / "failure.json", feedback)
            history.append({"attempt": attempt, "status": "candidate_execution_failure"})
        except CandidateError as exc:
            feedback = _generated_feedback(
                _summarize_error("candidate_format_or_validation_failure", exc),
                attempt=attempt,
                max_attempts=max_attempts,
            )
            attempt_output["feedback"] = feedback
            previous_attempt = attempt_output
            if current_candidate_fingerprint is not None:
                failed_candidates[current_candidate_fingerprint] = {
                    "attempt": attempt,
                    "candidate_identity": candidate_identity,
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
