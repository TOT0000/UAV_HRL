"""Autonomous, tool-driven offline LLM feature design using smolagents.

This module is intentionally optional.  Core training and ``run_llm_design`` do
not import it, so an environment without smolagents keeps the legacy behavior.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as distribution_version
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable
import urllib.request

import numpy as np

try:
    from smolagents import (
        ChatMessage,
        ChatMessageToolCall,
        MessageRole,
        Model,
        TokenUsage,
        Tool,
        ToolCall,
        ToolCallingAgent,
        ToolOutput,
    )
    from smolagents.agents import ActionOutput
    from smolagents.memory import TaskStep
    from smolagents.models import ChatMessageToolCallFunction, get_tool_json_schema
except ImportError as exc:  # pragma: no cover - exercised in a clean subprocess
    raise ImportError(
        "run_llm_agent.py requires the optional agent dependency; install "
        "requirements-llm-agent.txt. The legacy run_llm_design.py entry point "
        "does not require it."
    ) from exc

from llm_candidate import (
    CandidateExecutionError,
    ModelCandidateSchemaError,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    feature_reward,
    normalize_candidate_submission,
    save_approved_artifact,
    validate_candidate,
    validate_candidate_staged,
)
from llm_design import (
    APIError,
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    EVALUATION_DIAGNOSTICS_VERSION,
    LMStudioClient,
    OpenAIClient,
    _compact_evaluation_diagnostics,
    _effective_context_budget,
    _git_sha,
    _json_native,
    _slug,
    _write_json,
    adapter_separate_reasoning,
    evaluate_candidate,
    model_adapter,
    model_inventory_summary,
    prepare_evaluation_context,
    provider_base_url,
    response_model_matches,
)
from llm_design_contract import (
    OBS_INTERFACE_VERSION,
    PROMPT_VERSION,
    SUPPORTED_OPERATIONS,
    baseline_lines,
    build_obs_arrays,
    estimate_token_budget,
    format_schema_and_example,
    load_design_inputs,
    model_candidate_schema,
    render_environment_interface,
    runtime_diagnostic_contract,
)
from llm_streaming import (
    StreamTransportError,
    capture_chat_stream,
    read_response_body_bounded,
)


AGENT_RUN_SCHEMA_VERSION = "uav-hrl-llm-feature-agent-run-v1"
AGENT_PROMPT_VERSION = "uav-hrl-llm-feature-agent-prompt-v6"
AGENT_TOOL_CONTRACT_VERSION = "uav-hrl-llm-feature-agent-tools-v8"
DEFAULT_MAX_MODEL_CALLS = 20
DEFAULT_AGENT_OUTPUT_ROOT = Path("results") / "llm_agents"
DEFAULT_AGENT_PREVIEW_ROOT = Path("results") / "llm_agent_previews"
WORK_SUMMARY_MAX_ISSUE_GROUPS = 16
WORK_SUMMARY_MAX_EVALUATION_INDEXES = 12
WORK_SUMMARY_MAX_QUERY_INDEXES = 8
WORK_SUMMARY_MAX_CRITICAL_PAIRS = 3
WORK_SUMMARY_MAX_PAIR_FEATURES = 6
WORK_SUMMARY_MAX_FINDINGS = 6
MODEL_TOOL_RESULT_MAX_CHARS = 6_000
PAGED_TOOL_PAYLOAD_MAX_CHARS = 5_000
REPORT_TEXT_SEGMENT_MAX_CHARS = 512
INTERFACE_PAGE_MAX_LINES = 80
AGENT_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompts") / "llm_agent_prompt.txt"
SMOLAGENTS_VERSION = distribution_version("smolagents")

HOST_CONTROLLED_SYSTEM_PROMPT = """You are an expert assistant solving an offline design task through tool calls.

Each tool call is an action. Read its observation before deciding the next action, and do not repeat a completed call with identical arguments.

Only the host can complete this task: completion requires a candidate to pass formal_evaluate and an approved artifact to be saved. The final_answer tool is only a request to stop. Before host approval it will return completion_rejected and you must continue using the available tools. Plain text without a tool call is treated the same way. Never infer completion from your own prose.

Available tools:
{%- for tool in tools.values() %}
- {{ tool.to_tool_calling_prompt() }}
{%- endfor %}

{%- if custom_instructions %}
{{ custom_instructions }}
{%- endif %}

Always issue a tool call with arguments matching its schema. Use the concrete values returned by earlier tools. Begin now.
"""


def _model_candidate_view(record: dict[str, Any]) -> dict[str, Any]:
    """Return only model-editable fields, projecting legacy records if needed."""

    stored = record.get("model_candidate")
    if isinstance(stored, dict) and set(stored) == {"features", "code"}:
        return copy.deepcopy(stored)
    internal = record.get("candidate") if "candidate" in record else record
    if not isinstance(internal, dict):
        return {"features": [], "code": None}
    features = internal.get("features")
    return {
        "features": [
            {
                "name": item.get("name"),
                "description": item.get("description"),
                "reward_weight": item.get("reward_weight"),
            }
            for item in (features if isinstance(features, list) else [])
            if isinstance(item, dict)
        ],
        "code": internal.get("code"),
    }


class AgentBudgetError(RuntimeError):
    pass


class AgentContextBudgetError(RuntimeError):
    pass


def _request_token_budget(
    model: Model,
    messages: list[ChatMessage],
    tools: Iterable[Tool],
    *,
    context_length: int,
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and budget the same message/tool payload used for generation."""
    completion = model._prepare_completion_kwargs(
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=list(tools),
        tool_choice="required",
    )
    serialized = json.dumps(
        completion,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return (
        estimate_token_budget(
            serialized,
            context_length=int(context_length),
            max_output_tokens=int(max_output_tokens),
        ),
        completion,
    )


def _json_characters(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        )
    )


def _message_characters(messages: list[ChatMessage]) -> int:
    return _json_characters([message.dict() for message in messages])


def _budget_component(characters: int) -> dict[str, Any]:
    """Report a transparent character-based range, never an exact token count."""

    characters = max(0, int(characters))
    return {
        "serialized_characters": characters,
        "token_estimate": {
            "exact": False,
            "method": "character-ratio uncertainty range; no tokenizer used",
            "lower": math.ceil(characters / 4.5),
            "central": math.ceil(characters / 3.5),
            "upper": math.ceil(characters / 2.5),
        },
    }


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _evaluation_history_summary(
    *, candidate_id: str, report_id: str, report: dict[str, Any]
) -> dict[str, Any]:
    """Build a bounded, lossless-for-scores evaluation history entry."""

    by_lambda: dict[str, Any] = {}
    for lambda_key, source in (report.get("by_lambda") or {}).items():
        source = source or {}
        maximum_pair = source.get("maximum_pair") or {}
        entry = {
            key: copy.deepcopy(source.get(key))
            for key in (
                "lambda_mbit_per_joule",
                "baseline_l_hat",
                "candidate_l_hat",
                "improvement",
                "required_margin",
                "passed",
            )
            if key in source
        }
        if "i" in maximum_pair or "j" in maximum_pair:
            entry["maximum_pair_indices"] = {
                key: copy.deepcopy(maximum_pair.get(key))
                for key in ("i", "j")
                if key in maximum_pair
            }
        by_lambda[str(lambda_key)] = entry
    return {
        "candidate_id": candidate_id,
        "path_id": report_id,
        "report_id": report_id,
        "status": report.get("status"),
        "passed": report.get("passed"),
        "by_lambda": by_lambda,
        "details_query": {
            "tool": "get_history",
            "arguments": {
                "candidate_id": candidate_id,
                "record_type": "report",
                "report_id": report_id,
                "start": 0,
                "limit": 50,
            },
        },
    }


def _bounded_mapping(value: Any, *, maximum_items: int = 8) -> Any:
    """Keep small scalar evidence without re-expanding report-sized payloads."""

    if isinstance(value, dict):
        keys = list(value)[:maximum_items]
        return {
            key: _bounded_mapping(value[key], maximum_items=maximum_items)
            for key in keys
        }
    if isinstance(value, list):
        return [
            _bounded_mapping(item, maximum_items=maximum_items)
            for item in value[:maximum_items]
        ]
    if isinstance(value, str) and len(value) > 256:
        return {
            "prefix": value[:256],
            "original_character_count": len(value),
            "omitted_character_count": len(value) - 256,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return copy.deepcopy(value)
    return str(value)


def _compact_tool_arguments(tool: str, arguments: Any) -> Any:
    """Keep executable paging/candidate arguments while bounding invalid noise."""

    if not isinstance(arguments, dict):
        return _bounded_mapping(arguments)
    if tool == "submit_candidate":
        # Immutable candidate source is subject to the request budget, never a
        # lossy per-field truncation.
        return copy.deepcopy(arguments)
    allowed_by_tool = {
        "query_samples": (
            "fields",
            "start",
            "limit",
            "condition_field",
            "condition",
        ),
        "inspect_interface": ("section", "start", "limit"),
        "get_history": (
            "candidate_id",
            "record_type",
            "start",
            "limit",
            "report_id",
        ),
        "test_candidate": ("candidate_id",),
        "formal_evaluate": ("candidate_id",),
        "final_answer": ("answer",),
    }
    selected = {
        key: copy.deepcopy(arguments[key])
        for key in allowed_by_tool.get(tool, tuple(arguments))
        if key in arguments
    }
    shortened = []
    for key, value in list(selected.items()):
        if isinstance(value, str) and len(value) > 256:
            selected[key] = {
                "prefix": value[:256],
                "original_character_count": len(value),
                "omitted_character_count": len(value) - 256,
                "complete_value_available_in": "get_history(record_type='tool_calls')",
            }
            shortened.append(key)
        elif isinstance(value, list) and len(value) > 20:
            selected[key] = copy.deepcopy(value[:20])
            shortened.append(key)
    if shortened:
        selected["summary_omissions"] = {
            "shortened_argument_fields": shortened,
            "full_arguments_preserved": True,
        }
    return selected


def _evaluation_model_summary(
    *,
    candidate_id: str,
    report_id: str,
    report: dict[str, Any],
    maximum_pairs: int = WORK_SUMMARY_MAX_CRITICAL_PAIRS,
    maximum_features_per_pair: int = WORK_SUMMARY_MAX_PAIR_FEATURES,
    maximum_findings: int = WORK_SUMMARY_MAX_FINDINGS,
) -> dict[str, Any]:
    """Summarize one formal evaluation without repeating pair diagnostics."""

    diagnostics = report.get("evaluation_diagnostics") or {}
    diagnostic_by_lambda = diagnostics.get("by_lambda") or {}
    all_pairs = diagnostics.get("pairs") or {}
    pair_references: list[str] = []
    by_lambda: dict[str, Any] = {}
    for lambda_key, source in (report.get("by_lambda") or {}).items():
        source = source or {}
        diagnostic = diagnostic_by_lambda.get(lambda_key) or {}
        pair_ref = diagnostic.get("maximum_pair_ref")
        if pair_ref is not None and str(pair_ref) not in pair_references:
            pair_references.append(str(pair_ref))
        maximum_pair = source.get("maximum_pair") or {}
        entry = {
            key: copy.deepcopy(source.get(key))
            for key in (
                "lambda_mbit_per_joule",
                "baseline_l_hat",
                "candidate_l_hat",
                "improvement",
                "required_margin",
                "passed",
            )
            if key in source
        }
        if pair_ref is not None:
            entry["maximum_pair_ref"] = str(pair_ref)
        if "i" in maximum_pair or "j" in maximum_pair:
            entry["maximum_pair_indices"] = {
                key: copy.deepcopy(maximum_pair.get(key))
                for key in ("i", "j")
                if key in maximum_pair
            }
        reward_outcomes = diagnostic.get("reward_outcomes_after_action") or {}
        if reward_outcomes:
            entry["reward_differences_i_minus_j"] = {
                key: copy.deepcopy(reward_outcomes.get(key))
                for key in (
                    "base_reward_difference_i_minus_j",
                    "extra_reward_difference_i_minus_j",
                    "total_reward_difference_i_minus_j",
                )
                if key in reward_outcomes
            }
        by_lambda[str(lambda_key)] = entry

    selected_pair_refs = pair_references[: max(0, int(maximum_pairs))]
    pair_summaries: dict[str, Any] = {}
    for pair_ref in selected_pair_refs:
        source_pair = all_pairs.get(pair_ref) or {}
        features = list(source_pair.get("features") or [])

        def feature_rank(item: tuple[int, Any]) -> tuple[float, int]:
            position, feature = item
            if not isinstance(feature, dict):
                return (0.0, position)
            difference = feature.get("feature_difference_i_minus_j")
            try:
                magnitude = abs(float(difference))
            except (TypeError, ValueError):
                magnitude = 0.0
            return (-magnitude, position)

        selected_features = sorted(enumerate(features), key=feature_rank)[
            : max(0, int(maximum_features_per_pair))
        ]
        feature_summaries = []
        for _, feature in selected_features:
            if not isinstance(feature, dict):
                continue
            feature_summaries.append(
                {
                    key: copy.deepcopy(feature.get(key))
                    for key in (
                        "index",
                        "name",
                        "reward_weight",
                        "feature_i",
                        "feature_j",
                        "feature_difference_i_minus_j",
                        "weighted_contribution_difference_i_minus_j_excludes_beta",
                    )
                    if key in feature
                }
            )
        pair_summaries[pair_ref] = {
            key: copy.deepcopy(source_pair.get(key))
            for key in (
                "sample_i_ref",
                "sample_j_ref",
                "original_state_distance",
                "augmented_state_distance",
                "extra_reward_i",
                "extra_reward_j",
                "extra_reward_difference_i_minus_j",
            )
            if key in source_pair
        } | {
            "feature_difference_summary": {
                "total_feature_count": len(features),
                "included_feature_count": len(feature_summaries),
                "omitted_feature_count": max(0, len(features) - len(feature_summaries)),
                "selection_rule": (
                    "largest absolute feature_difference_i_minus_j, then original "
                    "feature order"
                ),
                "features": feature_summaries,
            }
        }

    findings = []
    for finding in list(diagnostics.get("findings") or [])[: max(0, int(maximum_findings))]:
        if not isinstance(finding, dict):
            continue
        findings.append(
            {
                key: _bounded_mapping(finding.get(key))
                for key in ("code", "pair_ref", "lambda_refs", "evidence")
                if key in finding
            }
        )
    details_query = {
        "tool": "get_history",
        "arguments": {
            "candidate_id": candidate_id,
            "record_type": "report",
            "report_id": report_id,
            "start": 0,
            "limit": 50,
        },
    }
    return {
        "candidate_id": candidate_id,
        "report_id": report_id,
        "status": "approved" if report.get("passed") else report.get("status"),
        "passed": bool(report.get("passed")),
        "by_lambda": by_lambda,
        "critical_pairs": pair_summaries,
        "critical_pair_summary": {
            "total_referenced_pair_count": len(pair_references),
            "included_pair_count": len(pair_summaries),
            "omitted_pair_count": max(0, len(pair_references) - len(pair_summaries)),
            "deduplication": "each maximum_pair_ref is expanded at most once",
        },
        "findings": findings,
        "finding_count": len(diagnostics.get("findings") or []),
        "omitted_finding_count": max(
            0, len(diagnostics.get("findings") or []) - len(findings)
        ),
        "details_query": details_query,
        "summary_contract": {
            "level": "evaluation_failure_diagnostics",
            "full_report_preserved": True,
            "samples_traces_contracts_and_finding_definitions_omitted": True,
        },
    }


def _minimal_evaluation_history_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain a usable report lookup if even the score summary cannot fit."""

    return {
        key: copy.deepcopy(summary.get(key))
        for key in (
            "candidate_id",
            "path_id",
            "report_id",
            "status",
            "passed",
            "details_query",
        )
    } | {
        "summary_omitted": True,
        "summary_omitted_reason": (
            "The per-lambda summary exceeds the model-facing page budget. "
            "Use details_query to read the registered report losslessly."
        ),
    }


def _report_text_segments(text: str) -> list[dict[str, Any]]:
    """Split canonical report JSON without losing text or source positions."""

    segments: list[dict[str, Any]] = []
    absolute_start = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        for line_start in range(0, len(line), REPORT_TEXT_SEGMENT_MAX_CHARS):
            chunk = line[line_start : line_start + REPORT_TEXT_SEGMENT_MAX_CHARS]
            segments.append(
                {
                    "segment_index": len(segments),
                    "line_number": line_number,
                    "line_character_start": line_start,
                    "line_character_end": line_start + len(chunk),
                    "absolute_character_start": absolute_start + line_start,
                    "absolute_character_end": absolute_start + line_start + len(chunk),
                    "completes_line": line_start + len(chunk) == len(line),
                    "text": chunk,
                }
            )
        absolute_start += len(line)
    if not segments:
        segments.append(
            {
                "segment_index": 0,
                "line_number": 1,
                "line_character_start": 0,
                "line_character_end": 0,
                "absolute_character_start": 0,
                "absolute_character_end": 0,
                "completes_line": True,
                "text": "",
            }
        )
    return segments


def _candidate_tool_input_schema() -> dict[str, Any]:
    """Embed the canonical candidate schema as one tool argument.

    The source schema uses root-relative ``#/$defs`` references. Resolve those
    while copying because the candidate schema is nested below the tool's
    ``candidate`` property. This avoids maintaining a second schema.
    """

    schema = model_candidate_schema()
    definitions = schema.get("$defs", {})

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                if name not in definitions:
                    raise ValueError(
                        f"candidate schema contains an unknown local reference: {reference}"
                    )
                resolved = resolve(copy.deepcopy(definitions[name]))
                siblings = {key: item for key, item in value.items() if key != "$ref"}
                if siblings:
                    resolved.update(resolve(siblings))
                return resolved
            result = {
                key: resolve(item)
                for key, item in value.items()
                if key not in {"$schema", "$id", "$defs", "title"}
            }
            if "const" in result and "type" not in result:
                const_type = {
                    str: "string",
                    bool: "boolean",
                    int: "integer",
                    float: "number",
                }.get(type(result["const"]))
                if const_type is not None:
                    result["type"] = const_type
            return result
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return copy.deepcopy(value)

    resolved = resolve(schema)
    resolved["description"] = (
        "Complete simplified feature candidate as a JSON object. Pass the object "
        "directly; do not serialize it into a JSON string."
    )
    return resolved


def _agent_directory(
    model: str,
    *,
    output_root: str | Path,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        path = Path(output_dir).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        return path
    root = Path(output_root).resolve() / _slug(model)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for counter in range(100):
        suffix = hashlib.sha256(f"{time.time_ns()}-{counter}".encode()).hexdigest()[:8]
        path = root / f"agent-{stamp}-{suffix}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique agent run below {root}")


def _resume_preview_directory(
    source_directory: Path,
    *,
    output_root: str | Path,
    output_dir: str | Path | None,
) -> Path:
    """Allocate dry-run output away from the immutable source run."""
    if output_dir is not None:
        path = Path(output_dir).resolve()
        if path == source_directory.resolve() or source_directory.resolve() in path.parents:
            raise ValueError("resume dry-run output must be outside the source run")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        return path
    root = Path(output_root).resolve().parent / DEFAULT_AGENT_PREVIEW_ROOT.name
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for counter in range(100):
        suffix = hashlib.sha256(
            f"{source_directory}-{time.time_ns()}-{counter}".encode()
        ).hexdigest()[:8]
        path = root / f"{source_directory.name}-preview-{stamp}-{suffix}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a resume preview below {root}")


def _render_agent_prompt(
    *,
    fixed_metadata: dict[str, Any],
    baseline: dict[str, Any],
    constants_metadata: dict[str, Any],
    beta: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    current_work_state: dict[str, Any],
) -> str:
    template = AGENT_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    candidate_contract = (
        format_schema_and_example(model_candidate_schema())
        + "\n\nSubmission interface:\n"
        + "- Call submit_candidate with the complete example-shaped object in the candidate argument.\n"
        + "- Do not serialize that object into a candidate_json string. The host receives candidate.code as decoded Python source and handles persistence serialization.\n"
        + "\n\n"
        + SUPPORTED_OPERATIONS
    )
    evaluation = (
        f"beta={float(beta):.17g}.\n"
        f"margin(lambda)=max({float(absolute_tolerance):.17g}, "
        f"{float(relative_tolerance):.17g}*abs(L_baseline(lambda))); require "
        "L_candidate < L_baseline - margin for every lambda.\n"
        + baseline_lines(baseline)
    )
    replacements = {
        "{{ENVIRONMENT_AND_INPUT_CONTRACT}}": render_environment_interface(
            fixed_metadata, constants_metadata
        ),
        "{{CANDIDATE_CONTRACT_AND_EXAMPLE}}": candidate_contract,
        "{{EVALUATION_SETTINGS_AND_BASELINE}}": evaluation,
        "{{CURRENT_WORK_STATE}}": json.dumps(
            current_work_state,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
    }
    for placeholder, replacement in replacements.items():
        template = template.replace(placeholder, replacement)
    missing = [placeholder for placeholder in replacements if placeholder in template]
    if missing:
        raise RuntimeError(f"agent task prompt contains unfilled placeholders: {missing}")
    return template


class ModelCallBudget:
    def __init__(self, maximum: int, used: int = 0):
        if int(maximum) <= 0 or int(used) < 0 or int(used) > int(maximum):
            raise ValueError("model call budget is invalid")
        self.maximum = int(maximum)
        self.used = int(used)

    @property
    def remaining(self) -> int:
        return self.maximum - self.used

    def consume(self) -> int:
        if self.used >= self.maximum:
            raise AgentBudgetError(
                f"model call budget exhausted ({self.used}/{self.maximum})"
            )
        self.used += 1
        return self.used


class AgentWorkspace:
    """Durable state and the only implementation behind agent tools."""

    def __init__(
        self,
        *,
        directory: Path,
        fixed_sample: str | Path,
        provider: str,
        model: str,
        beta: float,
        batch_size: int,
        worker_timeout: float,
        absolute_tolerance: float,
        relative_tolerance: float,
        max_model_calls: int,
        resume_state: dict[str, Any] | None = None,
        read_only: bool = False,
    ):
        self.directory = Path(directory).resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.directory.is_dir():
                raise FileNotFoundError(
                    f"read-only agent workspace does not exist: {self.directory}"
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.arrays, self.fixed_metadata, self.baseline, self.constants = (
            load_design_inputs(fixed_sample)
        )
        self.obs_arrays = build_obs_arrays(self.arrays)
        self.context = prepare_evaluation_context(
            self.arrays,
            self.fixed_metadata,
            self.baseline,
            batch_size=int(batch_size),
        )
        self.diagnostic_contract = runtime_diagnostic_contract(
            self.fixed_metadata, self.constants
        )
        self.provider = str(provider)
        self.model = str(model)
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.worker_timeout = float(worker_timeout)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self._lock = threading.RLock()
        self._prepared_delivery_record_ids: list[str] = []
        if resume_state is None:
            self.state = {
                "schema_version": AGENT_RUN_SCHEMA_VERSION,
                "status": "running",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider,
                "model": self.model,
                "git_sha": _git_sha(),
                "contracts": {
                    "prompt": AGENT_PROMPT_VERSION,
                    "tools": AGENT_TOOL_CONTRACT_VERSION,
                    "observation_interface": OBS_INTERFACE_VERSION,
                    "evaluation_diagnostics": EVALUATION_DIAGNOSTICS_VERSION,
                },
                "fixed_sample": {
                    "directory": str(Path(fixed_sample).resolve()),
                    "sample_content_sha256": self.fixed_metadata["sample_content_sha256"],
                    "primary_pair_set_sha256": self.context.pair_hash,
                },
                "settings": {
                    "beta": self.beta,
                    "batch_size": self.batch_size,
                    "worker_timeout": self.worker_timeout,
                    "absolute_tolerance": self.absolute_tolerance,
                    "relative_tolerance": self.relative_tolerance,
                    "max_model_calls": int(max_model_calls),
                },
                "model_calls_used": 0,
                "budget_extension_events": [],
                "tool_operations": [],
                "framework_tool_calls": [],
                "tool_result_deliveries": [],
                "completion_rejections": [],
                "candidates": {},
                "candidate_order": [],
                "current_candidate_id": None,
                "evaluation_cache": {},
                "approved_candidate_id": None,
                "approved_artifact": None,
                "stop_reason": None,
                "context_compactions": [],
            }
        else:
            self.state = copy.deepcopy(resume_state)
            if not self.read_only:
                for operation in self.state.get("tool_operations", []):
                    if operation.get("status") == "running":
                        operation["status"] = "interrupted"
                        operation["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                        operation["result_summary"] = {
                            "status": "interrupted",
                            "reusable_as_pass": False,
                        }
        self._validate_compatibility(max_model_calls)
        self._save()

    def _validate_compatibility(self, max_model_calls: int) -> None:
        expected_contracts = {
            "prompt": AGENT_PROMPT_VERSION,
            "tools": AGENT_TOOL_CONTRACT_VERSION,
            "observation_interface": OBS_INTERFACE_VERSION,
            "evaluation_diagnostics": EVALUATION_DIAGNOSTICS_VERSION,
        }
        if self.state.get("contracts") != expected_contracts:
            raise ValueError("agent resume contracts are incompatible")
        if self.state.get("git_sha") != _git_sha():
            raise ValueError("agent resume git revision is incompatible")
        fixed = self.state["fixed_sample"]
        if fixed["sample_content_sha256"] != self.fixed_metadata["sample_content_sha256"]:
            raise ValueError("resume fixed-sample content hash is incompatible")
        if fixed["primary_pair_set_sha256"] != self.context.pair_hash:
            raise ValueError("resume primary-pair hash is incompatible")
        settings = self.state["settings"]
        expected = {
            "beta": self.beta,
            "batch_size": self.batch_size,
            "worker_timeout": self.worker_timeout,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }
        for key, value in expected.items():
            if settings.get(key) != value:
                raise ValueError(f"resume setting is incompatible: {key}")
        if int(max_model_calls) != int(settings["max_model_calls"]):
            raise ValueError("resume model-call budget must match the original run")

    @property
    def approved(self) -> bool:
        return self.state.get("approved_candidate_id") is not None

    def _save(self) -> None:
        if self.read_only:
            return
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.directory / "agent_state.json", self.state)

    @staticmethod
    def _issue_with_defaults(
        issue: dict[str, Any],
        *,
        candidate_id: str,
        status: str = "open",
        check_stage: str | None = None,
    ) -> dict[str, Any]:
        validation_check = str(issue.get("validation_check") or "")
        if check_stage is None:
            if "runtime" in validation_check or "worker" in validation_check:
                check_stage = "runtime_validation"
            else:
                check_stage = "static_validation"
        return {
            **issue,
            "status": issue.get("status", status),
            "check_stage": issue.get("check_stage", check_stage),
            "source_candidate_id": issue.get("source_candidate_id", candidate_id),
        }

    def _latest_evaluation_summary(
        self,
        candidate_id: str,
        record: dict[str, Any],
        *,
        summary_level: str = "standard",
    ) -> dict[str, Any] | None:
        paths = list((record.get("evaluations") or {}).values())
        if not paths:
            return None
        relative_path = paths[-1]
        report_path = self.directory / relative_path
        if not report_path.is_file():
            return {
                "candidate_id": candidate_id,
                "report_id": relative_path,
                "status": "report_missing",
            }
        report = json.loads(report_path.read_text(encoding="utf-8"))
        minimal = summary_level == "minimal"
        summary = _evaluation_model_summary(
            candidate_id=candidate_id,
            report_id=relative_path,
            report=report,
            maximum_pairs=1 if minimal else WORK_SUMMARY_MAX_CRITICAL_PAIRS,
            maximum_features_per_pair=(
                2 if minimal else WORK_SUMMARY_MAX_PAIR_FEATURES
            ),
            maximum_findings=2 if minimal else WORK_SUMMARY_MAX_FINDINGS,
        )
        summary["source_is_exact_candidate"] = True
        summary["work_summary_level"] = summary_level
        return summary

    @staticmethod
    def _work_issue_key(issue: dict[str, Any]) -> str:
        """Group only issues that have the same actionable root."""
        identity_fields = (
            "check_stage",
            "validation_check",
            "code",
            "exception_type",
            "candidate_function",
            "candidate_line",
            "location",
            "feature_index",
            "source_field",
            "lambda",
        )
        identity = {
            key: issue.get(key)
            for key in identity_fields
            if issue.get(key) is not None
        }
        location_fields = identity_fields[3:]
        if not any(issue.get(key) is not None for key in location_fields):
            identity["problem"] = issue.get("problem")
        return json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)

    def _compact_unresolved_issues(
        self,
        current_id: str,
        current: dict[str, Any],
        *,
        maximum_groups: int = WORK_SUMMARY_MAX_ISSUE_GROUPS,
    ) -> dict[str, Any]:
        all_issues = (
            list(current.get("issues", []))
            + list(current.get("formal_evaluation_issues", []))
            + [
                issue
                for issue in current.get("inherited_issue_context", [])
                if issue.get("status") != "resolved"
            ]
        )
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for issue in all_issues:
            key = self._work_issue_key(issue)
            if key not in groups:
                representative = {
                    field: issue.get(field)
                    for field in (
                        "code",
                        "problem",
                        "requirement",
                        "fix",
                        "check_stage",
                        "validation_check",
                        "exception_type",
                        "candidate_function",
                        "candidate_line",
                        "location",
                        "feature_index",
                        "source_field",
                        "lambda",
                        "baseline_l_hat",
                        "candidate_l_hat",
                        "required_margin",
                        "maximum_pair_ref",
                    )
                    if issue.get(field) is not None
                }
                groups[key] = {
                    **representative,
                    "status": "not_revalidated",
                    "occurrence_count": 0,
                    "source_candidate_ids": [],
                    "history_index": {
                        "candidate_id": current_id,
                        "record_type": "issues",
                    },
                }
                order.append(key)
            group = groups[key]
            group["occurrence_count"] += int(issue.get("occurrence_count", 1))
            source = issue.get("source_candidate_id") or current_id
            if source not in group["source_candidate_ids"]:
                group["source_candidate_ids"].append(source)
            if source == current_id and issue.get("status", "open") != "resolved":
                group["status"] = "confirmed_current"
        for group in groups.values():
            sources = list(group["source_candidate_ids"])
            group["source_candidate_count"] = len(sources)
            group["source_candidate_ids"] = sources[-8:]
            group["omitted_source_candidate_count"] = max(0, len(sources) - 8)
        compact = [groups[key] for key in order[:maximum_groups]]
        omitted = [groups[key] for key in order[maximum_groups:]]
        omitted_by_kind: dict[str, int] = {}
        for issue in omitted:
            kind = f"{issue.get('check_stage', 'unknown')}:{issue.get('code', 'UNKNOWN')}"
            omitted_by_kind[kind] = omitted_by_kind.get(kind, 0) + 1
        return {
            "total_distinct_unresolved": len(order),
            "included_distinct_unresolved": len(compact),
            "omitted_distinct_unresolved": len(omitted),
            "issues": compact,
            "omitted_by_kind": omitted_by_kind,
            "omitted_details_history_index": (
                {"candidate_id": current_id, "record_type": "issues"}
                if omitted
                else None
            ),
            "selection_rule": (
                "stable first occurrence after grouping by stage, code, location, "
                "field, lambda, and problem; resolved issues are excluded"
            ),
        }

    def _selected_evaluation_summary(
        self,
        current_id: str,
        current: dict[str, Any],
        *,
        summary_level: str = "standard",
    ) -> dict[str, Any] | None:
        candidate_id: str | None = current_id
        record = current
        while candidate_id:
            summary = self._latest_evaluation_summary(
                candidate_id, record, summary_level=summary_level
            )
            if summary is not None:
                summary["is_current_candidate"] = candidate_id == current_id
                summary["history_index"] = {
                    "candidate_id": candidate_id,
                    "record_type": "evaluation",
                }
                if candidate_id != current_id:
                    summary["relationship_to_current"] = (
                        "nearest evaluated ancestor; the current candidate has not yet "
                        "completed formal evaluation"
                    )
                return summary
            candidate_id = record.get("parent_candidate_id")
            record = self.state["candidates"].get(candidate_id) if candidate_id else None
            if record is None:
                break
        return None

    @staticmethod
    def _latest_test_summary(
        candidate_id: str, record: dict[str, Any]
    ) -> dict[str, Any] | None:
        tests = list((record.get("tests") or {}).values())
        if not tests:
            return None
        latest = tests[-1]
        result = latest.get("tool_result") or {}
        numeric = result.get("numeric_diagnostics") or {}
        features = list(numeric.get("features") or [])
        return {
            key: copy.deepcopy(result.get(key))
            for key in (
                "status",
                "candidate_id",
                "test_scope",
                "fixed_sample_count",
                "tested_sample_count",
                "successful_output_sample_count",
                "complete_fixed_sample",
                "reward_consistency",
                "checks_not_run",
                "error_type",
                "error",
            )
            if key in result
        } | {
            "numeric_feature_count": len(features),
            "numeric_feature_summaries": copy.deepcopy(features[:8]),
            "omitted_numeric_feature_summaries": max(0, len(features) - 8),
            "history_index": {
                "candidate_id": candidate_id,
                "record_type": "tests",
                "start": 0,
                "limit": 20,
            },
        }

    def _evaluation_history_index(self, current_id: str) -> dict[str, Any]:
        evaluated = [
            candidate_id
            for candidate_id in self.state["candidate_order"]
            if self.state["candidates"][candidate_id].get("evaluations")
        ]
        recent = evaluated[-WORK_SUMMARY_MAX_EVALUATION_INDEXES:]
        return {
            "total_evaluated_candidates": len(evaluated),
            "recent": [
                {"candidate_id": candidate_id, "record_type": "evaluation"}
                for candidate_id in recent
            ],
            "omitted_count": len(evaluated) - len(recent),
            "catalog_history_index": {
                "candidate_id": None,
                "record_type": "run",
                "start": 0,
                "limit": 20,
            },
            "note": (
                "Older evaluation details are not expanded automatically; use "
                "get_history with the indexed candidate id."
            ),
        }

    def work_state(
        self,
        *,
        include_candidate: bool = True,
        model_calls_maximum: int | None = None,
        include_pending_payloads: bool = False,
        pending_payload_record_ids: set[str] | None = None,
        summary_level: str = "standard",
    ) -> dict[str, Any]:
        if summary_level not in {"standard", "minimal"}:
            raise ValueError("summary_level must be standard or minimal")
        minimal = summary_level == "minimal"
        current_id = self.state.get("current_candidate_id")
        current = self.state["candidates"].get(current_id) if current_id else None
        maximum = int(
            self.state["settings"]["max_model_calls"]
            if model_calls_maximum is None
            else model_calls_maximum
        )
        completion_events = self.state.get("completion_rejections", [])
        completion_summaries = []
        for event in completion_events[-(2 if minimal else 4):]:
            answer = (event.get("arguments") or {}).get("answer")
            answer_text = answer if isinstance(answer, str) else json.dumps(
                answer, ensure_ascii=False, default=str
            )
            completion_summaries.append(
                {
                    "created_at_utc": event.get("created_at_utc"),
                    "tool_call_id": event.get("tool_call_id"),
                    "source": event.get("source"),
                    "requested_answer_preview": answer_text[:500],
                    "requested_answer_truncated": len(answer_text) > 500,
                    "result": event.get("result"),
                }
            )
        framework_calls = self.state.get("framework_tool_calls", [])
        recent_queries = []
        for record in reversed(framework_calls):
            if record.get("name") not in {
                "query_samples",
                "inspect_interface",
                "get_history",
            }:
                continue
            result = record.get("model_result")
            result_summary = {}
            if isinstance(result, dict):
                for key in (
                    "status",
                    "section",
                    "matched_sample_count",
                    "requested_start",
                    "requested_limit",
                    "returned_sample_count",
                    "returned_sample_refs",
                    "remaining_sample_count",
                    "has_more",
                    "next_start",
                    "next_query_arguments",
                    "record_type",
                    "report_id",
                    "page_start",
                    "returned_count",
                    "remaining_count",
                    "returned_line_count",
                    "remaining_line_count",
                    "returned_segment_count",
                    "remaining_segment_count",
                    "total_segment_count",
                ):
                    if key in result:
                        result_summary[key] = (
                            _compact_tool_arguments(
                                str(record.get("name")), result[key]
                            )
                            if key == "next_query_arguments"
                            else copy.deepcopy(result[key])
                        )
            recent_queries.append(
                {
                    "record_id": record.get("record_id"),
                    "tool_call_id": record.get("tool_call_id"),
                    "tool": record.get("name"),
                    "arguments": _compact_tool_arguments(
                        str(record.get("name")), record.get("arguments")
                    ),
                    "execution_status": record.get("status"),
                    "delivery_status": record.get("delivery_status"),
                    "result_summary": result_summary,
                }
            )
            if len(recent_queries) >= (4 if minimal else WORK_SUMMARY_MAX_QUERY_INDEXES):
                break
        recent_queries.reverse()
        pending_payload_filter = pending_payload_record_ids
        pending_results = []
        for record in self.pending_delivery_records():
            item = {
                "record_id": record.get("record_id"),
                "tool_call_id": record.get("tool_call_id"),
                "tool": record.get("name"),
                "arguments": _compact_tool_arguments(
                    str(record.get("name")), record.get("arguments")
                ),
                "execution_status": record.get("status"),
                "delivery_status": "pending",
                "delivery_note": (
                    "Not yet included in a request that produced a complete accepted "
                    "model response; completion does not imply understanding."
                ),
            }
            include_payload = include_pending_payloads and (
                pending_payload_filter is None
                or str(record.get("record_id")) in pending_payload_filter
            )
            if include_payload:
                item["result"] = copy.deepcopy(record.get("model_result"))
            else:
                item["result_location"] = "paired tool-call/result memory step"
            pending_results.append(item)
        result = {
            "status": self.state["status"],
            "current_candidate_id": current_id,
            "candidate_count": len(self.state["candidate_order"]),
            "approved_candidate_id": self.state.get("approved_candidate_id"),
            "model_calls_used": self.state["model_calls_used"],
            "model_calls_maximum": maximum,
            "model_calls_remaining": maximum - int(self.state["model_calls_used"]),
            "work_summary_level": summary_level,
            "budget_extension_events": self.state.get("budget_extension_events", []),
            "completion_rejections": {
                "count": len(completion_events),
                "recent": completion_summaries,
                "full_records": "agent_state.json",
            },
            "recent_completed_queries": {
                "included_count": len(recent_queries),
                "total_count": sum(
                    record.get("name")
                    in {"query_samples", "inspect_interface", "get_history"}
                    for record in framework_calls
                ),
                "items": recent_queries,
                "selection_rule": "most recent bounded query/interface/history calls",
                "older_history_index": {
                    "candidate_id": None,
                    "record_type": "tool_calls",
                    "start": 0,
                    "limit": 20,
                },
            },
            "pending_tool_result_delivery": {
                "count": len(pending_results),
                "items": pending_results,
                "status_meaning": (
                    "pending means not yet included in a request with a complete "
                    "accepted response; it does not mean the model understood it"
                ),
            },
            "recent_operations": [
                {
                    key: operation.get(key)
                    for key in ("operation_id", "tool", "candidate_id", "status", "result_summary")
                }
                for operation in self.state["tool_operations"][-(4 if minimal else 8):]
            ],
        }
        if current is not None:
            result["current_candidate"] = {
                "candidate_id": current_id,
                "content_sha256": current["content_sha256"],
                "parent_candidate_id": current.get("parent_candidate_id"),
                "validation_status": current.get("validation_status"),
                "evaluation_status": current.get("evaluation_status"),
                "unresolved_issue_summary": self._compact_unresolved_issues(
                    current_id,
                    current,
                    maximum_groups=8 if minimal else WORK_SUMMARY_MAX_ISSUE_GROUPS,
                ),
                "selected_formal_evaluation": self._selected_evaluation_summary(
                    current_id, current, summary_level=summary_level
                ),
                "latest_complete_test": self._latest_test_summary(
                    current_id, current
                ),
                "evaluation_history_index": self._evaluation_history_index(current_id),
                "history_indexes": {
                    record_type: {
                        "candidate_id": current_id,
                        "record_type": record_type,
                    }
                    for record_type in ("candidate", "issues", "tests", "evaluation")
                },
            }
            if include_candidate:
                result["current_candidate"]["candidate"] = _model_candidate_view(current)
        return result

    def reject_completion(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        model_calls_maximum: int,
        source: str,
    ) -> dict[str, Any]:
        """Persist and explain a model completion request that lacks approval."""

        current_id = self.state.get("current_candidate_id")
        current = self.state["candidates"].get(current_id) if current_id else None
        unresolved = (
            self._compact_unresolved_issues(current_id, current)
            if current_id and current is not None
            else {
                "total_distinct_unresolved": 0,
                "included_distinct_unresolved": 0,
                "omitted_distinct_unresolved": 0,
                "issues": [],
            }
        )
        used = int(self.state.get("model_calls_used", 0))
        result = {
            "status": "completion_rejected",
            "reason": "no_host_approved_artifact",
            "source": str(source),
            "current_candidate_id": current_id,
            "validation_status": (
                current.get("validation_status") if current is not None else None
            ),
            "evaluation_status": (
                current.get("evaluation_status") if current is not None else None
            ),
            "unresolved_issue_summary": unresolved,
            "model_calls_used": used,
            "model_calls_maximum": int(model_calls_maximum),
            "model_calls_remaining": max(0, int(model_calls_maximum) - used),
            "required_next_action": (
                "The task is unfinished. Continue with query_samples, "
                "submit_candidate, test_candidate, formal_evaluate, or get_history."
            ),
        }
        event = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool_call_id": str(tool_call_id),
            "source": str(source),
            "arguments": copy.deepcopy(arguments),
            "result": copy.deepcopy(result),
        }
        self.state.setdefault("completion_rejections", []).append(event)
        self._save()
        return result

    def begin_operation(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            operation = {
                "operation_id": f"operation-{len(self.state['tool_operations']) + 1:04d}",
                "tool": str(tool),
                "arguments": arguments,
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.state["tool_operations"].append(operation)
            self._save()
            return operation

    def finish_operation(
        self, operation: dict[str, Any], *, status: str, result: dict[str, Any]
    ) -> None:
        with self._lock:
            operation["status"] = status
            operation["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            operation["candidate_id"] = result.get("candidate_id")
            operation["result_summary"] = {
                key: result.get(key)
                for key in (
                    "status",
                    "candidate_id",
                    "cache_hit",
                    "passed",
                    "approved_artifact",
                    "error",
                    "test_scope",
                    "fixed_sample_count",
                    "tested_sample_count",
                    "successful_output_sample_count",
                    "complete_fixed_sample",
                )
                if key in result
            }
            self._save()

    @staticmethod
    def _decoded_tool_result(output: Any) -> Any:
        if isinstance(output, str):
            try:
                return json.loads(output)
            except (TypeError, ValueError):
                return output
        return copy.deepcopy(output)

    @staticmethod
    def _serialized_size(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                default=str,
            )
        )

    def _compact_model_tool_result(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        """Return a bounded, truthful model-facing view of a persisted result."""

        if self._serialized_size(result) <= MODEL_TOOL_RESULT_MAX_CHARS:
            return result
        if (
            tool == "get_history"
            and isinstance(result, dict)
            and str(result.get("record_type") or arguments.get("record_type"))
            == "candidate"
        ):
            # Candidate code is never silently truncated. The request budget
            # check will stop explicitly if the complete immutable candidate
            # cannot be delivered.
            complete_candidate = copy.deepcopy(result)
            complete_candidate["model_delivery"] = {
                "summary_level": "complete_candidate_size_limit_exception",
                "general_character_limit_applied": False,
                "reason": "candidate code must not be truncated",
                "serialized_characters": self._serialized_size(result),
            }
            return complete_candidate
        if not isinstance(result, dict):
            return {
                "status": "result_too_large",
                "tool": tool,
                "error": "The structured result exceeds the delivery budget.",
                "correction": "Request a narrower interface section or history page.",
            }
        common_keys = (
            "status",
            "candidate_id",
            "passed",
            "cache_hit",
            "approved_artifact",
            "test_scope",
            "fixed_sample_count",
            "tested_sample_count",
            "successful_output_sample_count",
            "complete_fixed_sample",
            "reward_consistency",
            "checks_not_run",
            "error_type",
            "error",
            "report_id",
        )
        compact = {
            key: copy.deepcopy(result[key])
            for key in common_keys
            if key in result
        }
        compact.update(
            {
                "tool_result_compacted": True,
                "original_serialized_characters": self._serialized_size(result),
                "delivery_character_limit": MODEL_TOOL_RESULT_MAX_CHARS,
                "omission_does_not_change_status": True,
            }
        )
        candidate_id = result.get("candidate_id")
        if tool == "formal_evaluate":
            evaluation_summary = _evaluation_model_summary(
                candidate_id=str(candidate_id),
                report_id=str(result.get("report_id") or ""),
                report=result,
            )
            compact.update(evaluation_summary)
            compact["cache_hit"] = bool(result.get("cache_hit"))
            compact["approved_artifact"] = result.get("approved_artifact")
            compact["approval_is_host_determined"] = bool(
                result.get("approval_is_host_determined", True)
            )
            compact["model_delivery"] = {
                "summary_level": "bounded_formal_evaluation",
                "original_serialized_characters": self._serialized_size(result),
                "delivery_character_limit": MODEL_TOOL_RESULT_MAX_CHARS,
                "full_result_preserved": True,
            }
            if self._serialized_size(compact) > MODEL_TOOL_RESULT_MAX_CHARS:
                compact_summary = _evaluation_model_summary(
                    candidate_id=str(candidate_id),
                    report_id=str(result.get("report_id") or ""),
                    report=result,
                    maximum_pairs=1,
                    maximum_features_per_pair=2,
                    maximum_findings=2,
                )
                compact = {
                    **{
                        key: copy.deepcopy(result.get(key))
                        for key in ("cache_hit", "approved_artifact")
                        if key in result
                    },
                    **compact_summary,
                    "approval_is_host_determined": bool(
                        result.get("approval_is_host_determined", True)
                    ),
                    "model_delivery": {
                        "summary_level": "minimal_formal_evaluation",
                        "original_serialized_characters": self._serialized_size(result),
                        "delivery_character_limit": MODEL_TOOL_RESULT_MAX_CHARS,
                        "full_result_preserved": True,
                    },
                }
            if self._serialized_size(compact) > MODEL_TOOL_RESULT_MAX_CHARS:
                score_summary = _evaluation_history_summary(
                    candidate_id=str(candidate_id),
                    report_id=str(result.get("report_id") or ""),
                    report=result,
                )
                compact = {
                    **score_summary,
                    "model_delivery": {
                        "summary_level": "scores_only_formal_evaluation",
                        "original_serialized_characters": self._serialized_size(result),
                        "delivery_character_limit": MODEL_TOOL_RESULT_MAX_CHARS,
                        "full_result_preserved": True,
                        "diagnostics_omitted": True,
                    },
                }
        elif tool == "test_candidate":
            numeric = copy.deepcopy(result.get("numeric_diagnostics") or {})
            features = list(numeric.get("features") or [])
            numeric["features"] = features[:8]
            numeric["feature_count"] = len(features)
            numeric["omitted_feature_count"] = max(0, len(features) - 8)
            compact["numeric_diagnostics"] = numeric
            compact["representative_outputs"] = copy.deepcopy(
                (result.get("representative_outputs") or [])[:4]
            )
            compact["details_query"] = {
                "tool": "get_history",
                "arguments": {
                    "candidate_id": candidate_id,
                    "record_type": "report",
                    "start": 0,
                    "limit": 50,
                    "report_id": result.get("report_id"),
                },
            }
        elif tool == "submit_candidate":
            errors = list(result.get("errors") or [])
            compact["errors"] = copy.deepcopy(errors[:8])
            compact["error_count"] = len(errors)
            compact["omitted_error_count"] = max(0, len(errors) - 8)
            if candidate_id:
                compact["details_query"] = {
                    "tool": "get_history",
                    "arguments": {
                        "candidate_id": candidate_id,
                        "record_type": "issues",
                        "start": 0,
                        "limit": 20,
                    },
                }
        elif tool == "get_history":
            record_type = str(result.get("record_type") or arguments.get("record_type"))
            start = int(result.get("page_start", arguments.get("start", 0)))
            limit = int(result.get("page_limit", arguments.get("limit", 20)))
            compact.update(
                {
                    "record_type": record_type,
                    "page_start": start,
                    "page_limit": limit,
                    "total_record_count": result.get("total_record_count"),
                }
            )
            if record_type == "run":
                source_items = list(result.get("candidate_history_page") or [])
                key = "candidate_history_page"
                compact["work_state"] = copy.deepcopy(result.get("work_state"))
            elif record_type == "tool_calls":
                source_items = list(result.get("tool_call_records") or [])
                key = "tool_call_records"
            elif record_type == "report":
                # Report pages are already bounded lossless text-segment views.
                return result
            elif record_type == "tool_result":
                compact.update(
                    {
                        key: copy.deepcopy(result.get(key))
                        for key in (
                            "record_id",
                            "tool_call_id",
                            "tool",
                            "arguments",
                            "execution_status",
                            "delivery_status",
                            "result",
                            "raw_result_sha256",
                            "raw_result_serialized_characters",
                            "model_result_serialized_characters",
                            "model_result_summary_level",
                        )
                        if key in result
                    }
                )
                if self._serialized_size(compact) <= MODEL_TOOL_RESULT_MAX_CHARS:
                    return compact
                nested_result = copy.deepcopy(result.get("result"))
                minimum = {
                    "status": result.get("status"),
                    "record_type": "tool_result",
                    "record_id": result.get("record_id"),
                    "tool_call_id": result.get("tool_call_id"),
                    "tool": result.get("tool"),
                    "result": nested_result,
                    "wrapper_fields_omitted": True,
                }
                if self._serialized_size(minimum) <= MODEL_TOOL_RESULT_MAX_CHARS:
                    return minimum
                details_query = (
                    nested_result.get("details_query")
                    if isinstance(nested_result, dict)
                    else None
                )
                return {
                    "status": "result_delivery_budget_exceeded",
                    "record_type": "tool_result",
                    "record_id": result.get("record_id"),
                    "tool_call_id": result.get("tool_call_id"),
                    "tool": result.get("tool"),
                    "original_result_status": (
                        nested_result.get("status")
                        if isinstance(nested_result, dict)
                        else None
                    ),
                    "details_query": copy.deepcopy(details_query),
                    "error": (
                        "The indexed model-facing result plus its history wrapper "
                        "does not fit the delivery limit. Use details_query when "
                        "available; the original indexed result remains preserved."
                    ),
                }
            elif record_type == "issues":
                source_items = list(result.get("issue_records") or [])
                key = "issue_records"
            elif record_type == "tests":
                source_items = [
                    {
                        "test_key": key,
                        "status": value.get("status"),
                        "test_scope": value.get("test_scope"),
                        "fixed_sample_count": value.get("fixed_sample_count"),
                        "report_id": value.get("report"),
                        "result_summary": self._compact_model_tool_result(
                            tool="test_candidate",
                            arguments={"candidate_id": candidate_id},
                            result=value.get("tool_result") or {},
                        ),
                    }
                    for key, value in (result.get("tests") or {}).items()
                ]
                key = "test_records"
            elif record_type == "evaluation":
                source_items = list(result.get("evaluations") or [])
                key = "evaluations"
            else:
                source_items = []
                key = "records"
            selected = []
            for item in source_items:
                trial = {**compact, key: selected + [copy.deepcopy(item)]}
                if self._serialized_size(trial) > MODEL_TOOL_RESULT_MAX_CHARS:
                    break
                selected.append(copy.deepcopy(item))
            compact[key] = selected
            total = result.get("total_record_count")
            if total is not None:
                next_start = start + len(selected)
                has_more = next_start < int(total)
                compact.update(
                    {
                        "returned_count": len(selected),
                        "remaining_count": max(0, int(total) - next_start),
                        "has_more": has_more,
                        "next_start": next_start if has_more else None,
                        "next_query_arguments": (
                            {
                                "candidate_id": arguments.get("candidate_id"),
                                "record_type": record_type,
                                "start": next_start,
                                "limit": limit,
                            }
                            if has_more and selected
                            else None
                        ),
                    }
                )
            if not selected and source_items:
                if record_type == "evaluation":
                    minimal = _minimal_evaluation_history_summary(source_items[0])
                    compact[key] = [minimal]
                    selected = [minimal]
                    total = int(result.get("total_record_count") or len(source_items))
                    next_start = start + 1
                    has_more = next_start < total
                    compact.update(
                        {
                            "status": "summary_too_large",
                            "returned_count": 1,
                            "remaining_count": max(0, total - next_start),
                            "has_more": has_more,
                            "next_start": next_start if has_more else None,
                            "next_query_arguments": (
                                {
                                    "candidate_id": arguments.get("candidate_id"),
                                    "record_type": record_type,
                                    "start": next_start,
                                    "limit": limit,
                                }
                                if has_more
                                else None
                            ),
                        }
                    )
                else:
                    compact["status"] = "page_too_large"
                    compact["error"] = (
                        "One complete history record exceeds the model delivery budget; "
                        "the host preserved it but cannot silently truncate it."
                    )
        else:
            compact["correction"] = (
                "Request a narrower page with the same filters; the complete result "
                "is preserved by the host."
            )
            compact["original_arguments"] = copy.deepcopy(arguments)
        if self._serialized_size(compact) > MODEL_TOOL_RESULT_MAX_CHARS:
            # Last-resort structured reduction.  Status and scoring facts remain;
            # verbose diagnostics stay available through the declared history page.
            if tool == "test_candidate":
                compact.pop("representative_outputs", None)
            elif tool == "get_history" and compact.get("record_type") == "run":
                compact["work_state"] = {
                    key: copy.deepcopy((compact.get("work_state") or {}).get(key))
                    for key in (
                        "status",
                        "current_candidate_id",
                        "candidate_count",
                        "model_calls_used",
                        "model_calls_remaining",
                        "unresolved_or_unverified_issues",
                    )
                    if key in (compact.get("work_state") or {})
                }
        if self._serialized_size(compact) > MODEL_TOOL_RESULT_MAX_CHARS:
            details_query = compact.get("details_query")
            compact = {
                key: copy.deepcopy(compact.get(key))
                for key in ("status", "candidate_id", "passed", "report_id")
                if key in compact
            } | {
                "status": "result_delivery_budget_exceeded",
                "original_status": compact.get("status"),
                "tool": tool,
                "details_query": copy.deepcopy(details_query),
                "model_delivery": {
                    "summary_level": "minimum_result_budget_failure",
                    "original_serialized_characters": self._serialized_size(result),
                    "delivery_character_limit": MODEL_TOOL_RESULT_MAX_CHARS,
                    "full_result_preserved": True,
                },
                "error": (
                    "The minimum structured result does not fit the model-facing "
                    "tool-result limit; the complete result remains indexed."
                ),
            }
        if isinstance(compact.get("model_delivery"), dict):
            for _ in range(3):
                compact["model_delivery"]["serialized_characters"] = (
                    self._serialized_size(compact)
                )
        return compact

    def _framework_record(self, record_id: str) -> dict[str, Any] | None:
        for record in self.state.get("framework_tool_calls", []):
            if record.get("record_id") == str(record_id):
                return record
        return None

    def pending_delivery_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self.state.get("framework_tool_calls", [])
            if record.get("delivery_status") == "pending"
        ]

    def prepare_result_delivery(self, record_ids: Iterable[str]) -> None:
        ordered = []
        for record_id in record_ids:
            record = self._framework_record(str(record_id))
            if (
                record is not None
                and record.get("delivery_status") == "pending"
                and record_id not in ordered
            ):
                ordered.append(str(record_id))
        self._prepared_delivery_record_ids = ordered

    def mark_prepared_results_delivered(self, *, model_call_number: int) -> None:
        with self._lock:
            delivered = []
            for record_id in self._prepared_delivery_record_ids:
                record = self._framework_record(record_id)
                if record is None or record.get("delivery_status") != "pending":
                    continue
                record["delivery_status"] = "delivered"
                record["delivered_model_call_number"] = int(model_call_number)
                record["delivered_at_utc"] = datetime.now(timezone.utc).isoformat()
                delivered.append(record_id)
            if delivered:
                self.state.setdefault("tool_result_deliveries", []).append(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model_call_number": int(model_call_number),
                        "record_ids": delivered,
                        "meaning": (
                            "included in a request that produced a complete accepted "
                            "model response; this does not claim model understanding"
                        ),
                    }
                )
                self._save()
            self._prepared_delivery_record_ids = []

    def pending_record_ids_for_call_ids(
        self, tool_call_ids: Iterable[str]
    ) -> list[str]:
        ids = {str(value) for value in tool_call_ids}
        return [
            str(record["record_id"])
            for record in self.pending_delivery_records()
            if str(record.get("tool_call_id")) in ids
        ]

    def invoke(self, tool: str, arguments: dict[str, Any], function) -> str:
        operation = self.begin_operation(tool, arguments)
        try:
            result = function()
            if not isinstance(result, dict):
                raise TypeError("agent tool implementation did not return an object")
            operation_status = (
                "failed"
                if result.get("status") in {"invalid_arguments", "tool_error"}
                else "completed"
            )
            self.finish_operation(operation, status=operation_status, result=result)
        except Exception as exc:
            result = {
                "status": "tool_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.finish_operation(operation, status="failed", result=result)
        return json.dumps(result, ensure_ascii=False, allow_nan=False)

    def register_framework_calls(self, calls: Iterable[Any]) -> None:
        with self._lock:
            for call in calls:
                record_id = (
                    f"framework-call-{len(self.state['framework_tool_calls']) + 1:06d}"
                )
                self.state["framework_tool_calls"].append(
                    {
                        "record_id": record_id,
                        "tool_call_id": str(call.id),
                        "name": str(call.function.name),
                        "arguments": copy.deepcopy(call.function.arguments),
                        "status": "requested",
                        "delivery_status": "not_available",
                    }
                )
            self._save()

    def complete_framework_call(self, call_id: str, output: Any) -> None:
        self.finish_framework_call(call_id, status="completed", output=output)

    def finish_framework_call(
        self, call_id: str, *, status: str, output: Any
    ) -> Any:
        with self._lock:
            model_result = self._decoded_tool_result(output)
            for record in reversed(self.state["framework_tool_calls"]):
                if record["tool_call_id"] == str(call_id):
                    decoded = self._decoded_tool_result(output)
                    result_directory = self.directory / "tool_results"
                    result_directory.mkdir(parents=True, exist_ok=True)
                    result_path = result_directory / f"{record['record_id']}.json"
                    _write_json(
                        result_path,
                        {
                            "record_id": record["record_id"],
                            "tool_call_id": record["tool_call_id"],
                            "tool": record["name"],
                            "arguments": record.get("arguments"),
                            "execution_status": str(status),
                            "result": decoded,
                        },
                    )
                    model_result = self._compact_model_tool_result(
                        tool=str(record["name"]),
                        arguments=record.get("arguments") or {},
                        result=decoded,
                    )
                    record["status"] = str(status)
                    record["output"] = json.dumps(
                        model_result,
                        ensure_ascii=False,
                        allow_nan=False,
                        default=str,
                    )
                    record["model_result"] = model_result
                    record["raw_result_serialized_characters"] = self._serialized_size(
                        decoded
                    )
                    record["model_result_serialized_characters"] = (
                        self._serialized_size(model_result)
                    )
                    delivery_metadata = (
                        model_result.get("model_delivery")
                        if isinstance(model_result, dict)
                        else None
                    )
                    record["model_result_summary_level"] = (
                        delivery_metadata.get("summary_level")
                        if isinstance(delivery_metadata, dict)
                        else "complete"
                    )
                    record["raw_result_path"] = str(
                        result_path.relative_to(self.directory)
                    )
                    record["raw_result_sha256"] = hashlib.sha256(
                        json.dumps(
                            decoded,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    record["delivery_status"] = "pending"
                    record["delivery_created_at_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    break
            self._save()
            return copy.deepcopy(model_result)

    def skip_framework_call(self, call_id: str, *, reason: str) -> Any:
        return self.skip_framework_call_with_status(
            call_id, status="skipped_after_approval", reason=reason
        )

    def skip_framework_call_with_status(
        self, call_id: str, *, status: str, reason: str
    ) -> Any:
        return self.finish_framework_call(
            call_id,
            status=status,
            output={"status": str(status), "reason": str(reason)},
        )

    def inspect_interface(
        self,
        section: str,
        *,
        start: int = 0,
        limit: int = INTERFACE_PAGE_MAX_LINES,
    ) -> dict[str, Any]:
        section = str(section or "overview")
        if section == "overview":
            value = render_environment_interface(self.fixed_metadata, self.constants)
        elif section == "candidate":
            value = {
                "schema": model_candidate_schema(),
                "submit_candidate_arguments": {
                    "candidate": "direct object matching schema",
                    "parent_candidate_id": "optional immutable candidate id or null",
                },
                "serialization_note": (
                    "Pass candidate as an object, not a JSON-encoded string. "
                    "The host serializes submissions for persistence."
                ),
                "supported_operations": SUPPORTED_OPERATIONS,
            }
        elif section == "evaluation":
            value = {
                "beta": self.beta,
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
                "baseline_by_lambda": baseline_lines(self.baseline),
                "sample_count": self.fixed_metadata["sample_count"],
                "primary_pair_count": self.baseline["lipschitz"]["primary_pair_count"],
                "candidate_test_scope": (
                    "test_candidate always validates the complete loaded fixed-sample "
                    "artifact; query_samples is inspection only"
                ),
                "completion_control": (
                    "Only a successful host formal evaluation and saved approved "
                    "artifact complete the task; an earlier final_answer is rejected"
                ),
                "diagnostic_only_note": (
                    "baseline rewards, sample ids, and evaluation results may be used "
                    "for analysis but are not candidate feature inputs"
                ),
            }
        elif section == "fields":
            value = render_environment_interface(self.fixed_metadata, self.constants)
        elif section == "constants":
            value = self.constants
        else:
            return {
                "status": "invalid_arguments",
                "error": "section must be overview, fields, constants, candidate, or evaluation",
            }
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                default=str,
            )
        )
        lines = rendered.splitlines() or [""]
        page_start = max(0, int(start))
        page_limit = min(
            INTERFACE_PAGE_MAX_LINES,
            max(1, int(limit)),
        )
        selected = lines[page_start : page_start + page_limit]

        def interface_page(page_lines: list[str]) -> dict[str, Any]:
            next_start = (
                page_start + len(page_lines)
                if page_start + len(page_lines) < len(lines)
                else None
            )
            return {
                "status": "ok",
                "contract_version": AGENT_TOOL_CONTRACT_VERSION,
                "section": section,
                "content_format": "complete UTF-8 lines in original order",
                "total_line_count": len(lines),
                "page_start": page_start,
                "page_limit": page_limit,
                "returned_line_count": len(page_lines),
                "remaining_line_count": max(
                    0, len(lines) - page_start - len(page_lines)
                ),
                "has_more": next_start is not None,
                "next_start": next_start,
                "next_query_arguments": (
                    {
                        "section": section,
                        "start": next_start,
                        "limit": page_limit,
                    }
                    if next_start is not None
                    else None
                ),
                "data_lines": page_lines,
            }

        while (
            len(selected) > 1
            and self._serialized_size(interface_page(selected))
            > PAGED_TOOL_PAYLOAD_MAX_CHARS
        ):
            selected.pop()
        return interface_page(selected)

    def query_samples(
        self,
        *,
        fields: list[str],
        start: int,
        limit: int,
        condition_field: str | None,
        condition: str | None,
    ) -> dict[str, Any]:
        if not fields or len(fields) > 12:
            return {"status": "invalid_arguments", "error": "request 1..12 fields"}
        unknown = [field for field in fields if not field.startswith("obs.") or field[4:] not in self.obs_arrays]
        if unknown:
            return {"status": "invalid_arguments", "error": f"unavailable current-only fields: {unknown}"}
        indices = list(range(int(self.fixed_metadata["sample_count"])))
        if condition_field:
            if not condition_field.startswith("obs.") or condition_field[4:] not in self.obs_arrays:
                return {"status": "invalid_arguments", "error": "condition field is unavailable"}
            condition_array = np.asarray(self.obs_arrays[condition_field[4:]])
            predicate = str(condition or "any_true")
            if predicate not in {"any_true", "all_false"}:
                return {"status": "invalid_arguments", "error": "condition must be any_true or all_false"}
            indices = [
                index
                for index in indices
                if (bool(np.any(condition_array[index])) if predicate == "any_true" else not bool(np.any(condition_array[index])))
            ]
        requested_start = int(start)
        requested_limit = int(limit)
        start = max(0, requested_start)
        limit = min(20, max(1, requested_limit))
        selected = indices[start : start + limit]
        samples = []
        from llm_design import _compact_diagnostic_inputs, _sample_input_diagnostic

        def page_result(page_samples: list[dict[str, Any]]) -> dict[str, Any]:
            returned = len(page_samples)
            next_start = (
                start + returned if start + returned < len(indices) else None
            )
            next_arguments = (
                {
                    "fields": list(fields),
                    "start": next_start,
                    "limit": limit,
                    "condition_field": condition_field,
                    "condition": condition,
                }
                if next_start is not None
                else None
            )
            return {
                "status": "ok",
                "fields": list(fields),
                "feature_input_timing": "action-pre current-only",
                "matched_sample_count": len(indices),
                "requested_start": requested_start,
                "requested_limit": requested_limit,
                "effective_page_limit": limit,
                "returned_sample_count": returned,
                "returned_sample_refs": [
                    sample["sample_ref"] for sample in page_samples
                ],
                "returned_fixed_indices": [
                    sample["fixed_index"] for sample in page_samples
                ],
                "remaining_sample_count": max(
                    0, len(indices) - (start + returned)
                ),
                "has_more": next_start is not None,
                "page_start": start,
                "page_limit": limit,
                "next_start": next_start,
                "next_query_arguments": next_arguments,
                "selection_contract": (
                    "Stable fixed-sample order after applying the declared condition; "
                    "the next page preserves fields and filters."
                ),
                "samples": page_samples,
            }

        for index in selected:
            values, limitations = _sample_input_diagnostic(
                index,
                fields,
                obs_arrays=self.obs_arrays,
                constants_metadata=self.constants,
            )
            compact, _, _, _ = _compact_diagnostic_inputs(values, 32)
            trace = (self.fixed_metadata.get("selection") or [])[index]
            sample = (
                {
                    "sample_ref": f"sample_{index}",
                    "fixed_index": trace.get("fixed_index", index),
                    "feature_input_data": compact,
                    "limitations": limitations,
                }
            )
            candidate_samples = samples + [sample]
            if self._serialized_size(page_result(candidate_samples)) > PAGED_TOOL_PAYLOAD_MAX_CHARS:
                break
            samples = candidate_samples
        if selected and not samples:
            narrower_fields = list(fields[: max(1, len(fields) // 2)])
            can_narrow = len(narrower_fields) < len(fields)
            return {
                "status": "page_too_large",
                "fields": list(fields),
                "matched_sample_count": len(indices),
                "requested_start": requested_start,
                "requested_limit": requested_limit,
                "returned_sample_count": 0,
                "remaining_sample_count": max(0, len(indices) - start),
                "has_more": can_narrow,
                "next_start": None,
                "next_query_arguments": (
                    {
                        "fields": narrower_fields,
                        "start": start,
                        "limit": 1,
                        "condition_field": condition_field,
                        "condition": condition,
                    }
                    if can_narrow
                    else None
                ),
                "error": (
                    "One complete sample with the requested fields exceeds the model "
                    "delivery budget. Retry with the provided narrower field set."
                    if can_narrow
                    else "One compacted value for the single requested field exceeds "
                    "the model delivery budget; no unchanged zero-progress next page "
                    "is offered. Inspect the field contract and request a different, "
                    "narrower observable field."
                ),
            }
        return page_result(samples)

    def submit_candidate(
        self, candidate: Any, parent_candidate_id: str | None
    ) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            type_name = type(candidate).__name__
            report = {
                "status": "invalid_arguments",
                "error_type": "CandidateTypeError",
                "error": (
                    "candidate must be a JSON object, not a JSON-encoded string"
                    if isinstance(candidate, str)
                    else f"candidate must be a JSON object, not {type_name}"
                ),
                "candidate_created": False,
            }
            raw_hash = hashlib.sha256(
                json.dumps(
                    {"type": type_name, "value": candidate},
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            path = self.directory / "submissions" / f"submission-{raw_hash[:12]}"
            path.mkdir(parents=True, exist_ok=True)
            _write_json(path / "submission_report.json", report)
            return report
        try:
            submitted_candidate = copy.deepcopy(candidate)
            if {
                "schema_version", "candidate_name", "reward_input_mode", "features", "code"
            } == set(submitted_candidate):
                # Programmatic/backward compatibility for saved pre-v6 agent
                # candidates. The model-facing tool schema no longer exposes
                # this verbose form.
                candidate = submitted_candidate
                model_candidate = {
                    "features": [
                        {
                            "name": item.get("name"),
                            "description": item.get("description"),
                            "reward_weight": item.get("reward_weight"),
                        }
                        for item in candidate.get("features", [])
                        if isinstance(item, dict)
                    ],
                    "code": candidate.get("code"),
                }
                transformation = {
                    "mode": "legacy_internal_candidate_passthrough",
                    "note": "accepted for persisted/programmatic compatibility only",
                }
            else:
                model_candidate, candidate, transformation = normalize_candidate_submission(
                    submitted_candidate, self.constants
                )
            serialized = json.dumps(
                submitted_candidate,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            submitted_candidate = json.loads(serialized)
        except ModelCandidateSchemaError as exc:
            report = {
                "status": "model_schema_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "errors": copy.deepcopy(exc.issues),
                "checks": {
                    "json": {"status": "passed", "completed": True},
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
                "candidate_created": False,
            }
            raw_serialized = json.dumps(
                candidate,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            raw_hash = hashlib.sha256(raw_serialized.encode("utf-8")).hexdigest()
            path = self.directory / "submissions" / f"submission-{raw_hash[:12]}"
            path.mkdir(parents=True, exist_ok=True)
            try:
                _write_json(path / "parsed_model_candidate.json", submitted_candidate)
            except (TypeError, ValueError):
                (path / "parsed_model_candidate_unserializable.txt").write_text(
                    raw_serialized, encoding="utf-8"
                )
            _write_json(path / "submission_report.json", report)
            return report
        except (TypeError, ValueError) as exc:
            report = {
                "status": "invalid_arguments",
                "error_type": type(exc).__name__,
                "error": f"candidate must be a finite JSON object: {exc}",
                "candidate_created": False,
            }
            raw_hash = hashlib.sha256(
                f"{type(exc).__name__}:{exc}".encode("utf-8")
            ).hexdigest()
            path = self.directory / "submissions" / f"submission-{raw_hash[:12]}"
            path.mkdir(parents=True, exist_ok=True)
            _write_json(path / "submission_report.json", report)
            return report
        envelope = {
            "format": "direct_tool_object",
            "host_serialized": True,
            "host_transformation": transformation,
        }
        digest = _content_hash(candidate)
        candidate_id = f"candidate-{digest[:12]}"
        existing = self.state["candidates"].get(candidate_id)
        if existing is not None:
            self.state["current_candidate_id"] = candidate_id
            self._save()
            return {
                "status": "duplicate_candidate",
                "candidate_id": candidate_id,
                "content_sha256": digest,
                "cache_hit": True,
                "validation_status": existing.get("validation_status"),
                "evaluation_status": existing.get("evaluation_status"),
                "issues": existing.get("issues", []),
            }
        if parent_candidate_id and parent_candidate_id not in self.state["candidates"]:
            return {"status": "invalid_arguments", "error": "parent candidate id is unknown"}
        staged = validate_candidate_staged(candidate, self.constants)
        inherited = []
        if parent_candidate_id:
            parent = self.state["candidates"][parent_candidate_id]
            parent_issues = (
                list(parent.get("issues", []))
                + list(parent.get("inherited_issue_context", []))
                + list(parent.get("formal_evaluation_issues", []))
            )
            for issue in parent_issues:
                normalized = self._issue_with_defaults(
                    issue,
                    candidate_id=parent_candidate_id,
                )
                inherited.append(
                    {
                        **normalized,
                        "status": "not_revalidated",
                        "carried_from_candidate_id": parent_candidate_id,
                        "note": "A new candidate version has not yet completed the relevant check.",
                    }
                )
            if staged["can_execute"]:
                for issue in inherited:
                    if issue.get("check_stage") == "static_validation":
                        issue["status"] = "resolved"
                        issue["resolution_evidence"] = (
                            "the revised candidate completed staged static validation"
                        )
        current_issues = [
            self._issue_with_defaults(issue, candidate_id=candidate_id)
            for issue in (staged.get("errors") or [])
        ]
        record = {
            "candidate_id": candidate_id,
            "content_sha256": digest,
            "parent_candidate_id": parent_candidate_id,
            "candidate": candidate,
            "model_candidate": model_candidate,
            "envelope": envelope,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_status": "static_passed" if staged["can_execute"] else "static_failed",
            "evaluation_status": "not_run",
            "issues": current_issues,
            "inherited_issue_context": inherited,
            "formal_evaluation_issues": [],
            "tests": {},
            "evaluations": {},
        }
        self.state["candidates"][candidate_id] = record
        self.state["candidate_order"].append(candidate_id)
        self.state["current_candidate_id"] = candidate_id
        candidate_dir = self.directory / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "model_candidate.json", model_candidate)
        _write_json(candidate_dir / "candidate.json", candidate)
        if isinstance(candidate.get("code"), str):
            (candidate_dir / "candidate.py").write_text(
                candidate["code"], encoding="utf-8"
            )
        _write_json(candidate_dir / "staged_validation.json", staged)
        self._save()
        return {
            "status": record["validation_status"],
            "candidate_id": candidate_id,
            "content_sha256": digest,
            "parent_candidate_id": parent_candidate_id,
            "can_execute": staged["can_execute"],
            "errors": staged.get("errors") or [],
            "candidate_created": True,
        }

    def _candidate_record(self, candidate_id: str) -> dict[str, Any]:
        record = self.state["candidates"].get(str(candidate_id))
        if record is None:
            raise ValueError(f"unknown candidate id: {candidate_id}")
        return record

    def test_candidate(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:
        record = self._candidate_record(candidate_id)
        total = int(self.fixed_metadata["sample_count"])
        scope = "full_fixed_samples"
        if record["validation_status"] == "static_failed":
            return {
                "status": "prerequisite_failed",
                "candidate_id": candidate_id,
                "test_scope": scope,
                "fixed_sample_count": total,
                "tested_sample_count": 0,
                "successful_output_sample_count": 0,
                "complete_fixed_sample": False,
                "errors": record["issues"],
                "checks_not_run": [
                    "isolated runtime validation",
                    "full fixed-sample numeric diagnostics",
                ],
            }
        key = _content_hash(
            {
                "scope": scope,
                "candidate": record["content_sha256"],
                "fixed_sample": self.fixed_metadata["sample_content_sha256"],
                "sample_count": total,
                "worker_contract": self.diagnostic_contract,
            }
        )
        cached = record["tests"].get(key)
        if (
            cached is not None
            and cached.get("status") in {"passed", "failed"}
            and cached.get("test_scope") == scope
            and cached.get("fixed_sample_count") == total
        ):
            return {**cached["tool_result"], "cache_hit": True}
        obs = self.obs_arrays
        try:
            static = validate_candidate(record["candidate"], self.constants)
            extra, worker_reward, execution = execute_candidate_isolated(
                record["candidate"],
                obs,
                self.constants,
                timeout=self.worker_timeout,
                diagnostic_contract=self.diagnostic_contract,
            )
            numeric = candidate_numeric_diagnostics(
                np.asarray(self.arrays["state"]), extra, record["candidate"]
            )
            expected = feature_reward(extra, record["candidate"])
            consistency = bool(np.allclose(expected, worker_reward, rtol=0.0, atol=1e-12))
            report = {
                "status": "passed" if consistency else "failed",
                "candidate_id": candidate_id,
                "test_scope": scope,
                "fixed_sample_count": total,
                "tested_sample_count": total,
                "successful_output_sample_count": total,
                "complete_fixed_sample": True,
                "static": static,
                "execution": execution,
                "numeric_diagnostics": numeric,
                "reward_consistency": consistency,
                "extra_state_shape": list(extra.shape),
                "extra_reward_minimum": float(np.min(expected)),
                "extra_reward_maximum": float(np.max(expected)),
                "representative_outputs": [
                    {
                        "sample_ref": f"sample_{index}",
                        "features": _json_native(extra[index]),
                        "extra_reward": float(expected[index]),
                    }
                    for index in range(min(8, total))
                ],
                "checks_not_run": ["formal Lipschitz evaluation"],
                "empty_probe_statistics_included": False,
            }
        except CandidateExecutionError as exc:
            attempted = (
                ((exc.report or {}).get("checks") or {})
                .get("execution", {})
                .get("sample_count_attempted")
            )
            report = {
                "status": "failed",
                "candidate_id": candidate_id,
                "test_scope": scope,
                "fixed_sample_count": total,
                "tested_sample_count": (
                    int(attempted) if attempted is not None else 0
                ),
                "successful_output_sample_count": 0,
                "complete_fixed_sample": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_report": exc.report,
                "checks_not_run": [
                    "complete fixed-sample output validation",
                    "full fixed-sample numeric diagnostics",
                    "formal Lipschitz evaluation",
                ],
            }
        candidate_dir = self.directory / "candidates" / candidate_id
        report_path = candidate_dir / f"test_full_{key[:12]}.json"
        _write_json(report_path, report)
        tool_result = {
            key_name: report.get(key_name)
            for key_name in (
                "status",
                "candidate_id",
                "test_scope",
                "fixed_sample_count",
                "tested_sample_count",
                "successful_output_sample_count",
                "complete_fixed_sample",
                "numeric_diagnostics",
                "reward_consistency",
                "representative_outputs",
                "checks_not_run",
                "empty_probe_statistics_included",
                "error_type",
                "error",
                "worker_report",
            )
            if key_name in report
        }
        tool_result["cache_hit"] = False
        tool_result["report_id"] = str(report_path.relative_to(self.directory))
        record["tests"][key] = {
            "status": report["status"],
            "test_scope": scope,
            "fixed_sample_count": total,
            "report": str(report_path.relative_to(self.directory)),
            "tool_result": tool_result,
        }
        if report["status"] == "passed":
            record["validation_status"] = "full_passed"
            record["issues"] = []
            updated = []
            for issue in record.get("inherited_issue_context", []):
                if issue.get("check_stage") == "runtime_validation":
                    issue = {
                        **issue,
                        "status": "resolved",
                        "resolution_evidence": str(
                            report_path.relative_to(self.directory)
                        ),
                    }
                updated.append(issue)
            record["inherited_issue_context"] = updated
        else:
            record["validation_status"] = "full_failed"
            errors = (report.get("worker_report") or {}).get("errors") or []
            record["issues"] = [
                self._issue_with_defaults(
                    issue,
                    candidate_id=candidate_id,
                    check_stage="runtime_validation",
                )
                for issue in (
                    errors
                    or [
                        {
                            "code": report.get("error_type", "EXECUTION_ERROR"),
                            "problem": report.get("error"),
                        }
                    ]
                )
            ]
        self._save()
        return tool_result

    def _evaluation_cache_key(self, record: dict[str, Any]) -> str:
        return _content_hash(
            {
                "candidate": record["content_sha256"],
                "sample": self.fixed_metadata["sample_content_sha256"],
                "pairs": self.context.pair_hash,
                "beta": self.beta,
                "batch_size": self.batch_size,
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
                "observation_interface": OBS_INTERFACE_VERSION,
                "evaluation_diagnostics": EVALUATION_DIAGNOSTICS_VERSION,
                "git_sha": _git_sha(),
            }
        )

    def formal_evaluate(self, candidate_id: str) -> dict[str, Any]:
        record = self._candidate_record(candidate_id)
        if record["validation_status"] == "static_failed":
            return {"status": "prerequisite_failed", "candidate_id": candidate_id, "errors": record["issues"]}
        cache_key = self._evaluation_cache_key(record)
        cached = self.state["evaluation_cache"].get(cache_key)
        if cached is not None:
            report = json.loads((self.directory / cached["report"]).read_text(encoding="utf-8"))
            return self._evaluation_tool_result(
                candidate_id,
                report,
                cache_hit=True,
                report_id=cached["report"],
            )
        validation_result = self.test_candidate(candidate_id)
        if validation_result["status"] != "passed":
            return {
                "status": "validation_failed",
                "candidate_id": candidate_id,
                "validation": validation_result,
                "passed": False,
            }
        extra, worker_reward, execution = execute_candidate_isolated(
            record["candidate"],
            self.obs_arrays,
            self.constants,
            timeout=self.worker_timeout,
            diagnostic_contract=self.diagnostic_contract,
        )
        report = evaluate_candidate(
            self.context,
            record["candidate"],
            extra,
            beta=self.beta,
            batch_size=self.batch_size,
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
            obs_arrays=self.obs_arrays,
            constants_metadata=self.constants,
        )
        report["candidate_numeric_diagnostics"] = candidate_numeric_diagnostics(
            self.arrays["state"], extra, record["candidate"]
        )
        report["shared_feature_reward_consistency"] = {
            "maximum_absolute_difference": float(
                np.max(np.abs(feature_reward(extra, record["candidate"]) - worker_reward))
            ),
            "passed": bool(
                np.allclose(
                    feature_reward(extra, record["candidate"]),
                    worker_reward,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
        }
        report["validation_execution"] = execution
        candidate_dir = self.directory / "candidates" / candidate_id
        report_path = candidate_dir / f"evaluation_{cache_key[:12]}.json"
        _write_json(report_path, report)
        relative_report = str(report_path.relative_to(self.directory))
        self.state["evaluation_cache"][cache_key] = {
            "candidate_id": candidate_id,
            "report": relative_report,
        }
        record["evaluations"][cache_key] = relative_report
        record["evaluation_status"] = "passed" if report["passed"] else "failed"
        if report["passed"]:
            record["formal_evaluation_issues"] = []
            record["inherited_issue_context"] = [
                {
                    **issue,
                    "status": "resolved",
                    "resolution_evidence": relative_report,
                }
                if issue.get("check_stage") == "formal_evaluation"
                else issue
                for issue in record.get("inherited_issue_context", [])
            ]
            artifact = save_approved_artifact(
                self.directory,
                candidate=record["candidate"],
                constants_metadata=self.constants,
                validation_report={
                    "status": "passed",
                    "candidate_id": candidate_id,
                    "full_test": validation_result,
                },
                evaluation_report=report,
                provenance={
                    "run_directory": str(self.directory),
                    "agent_run_schema_version": AGENT_RUN_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "candidate_content_sha256": record["content_sha256"],
                    "provider": self.provider,
                    "model_requested": self.model,
                    "fixed_sample_content_sha256": self.fixed_metadata["sample_content_sha256"],
                    "primary_pair_set_sha256": self.context.pair_hash,
                    "beta": self.beta,
                    "git_sha": _git_sha(),
                },
            )
            self.state["approved_candidate_id"] = candidate_id
            self.state["approved_artifact"] = str(artifact)
            self.state["status"] = "approved"
            self.state["stop_reason"] = "host-approved formal evaluation"
        else:
            evaluation_issues = []
            diagnostic_by_lambda = (
                (report.get("evaluation_diagnostics") or {}).get("by_lambda") or {}
            )
            for lambda_value, result in (report.get("by_lambda") or {}).items():
                if result.get("passed"):
                    continue
                diagnostic = diagnostic_by_lambda.get(lambda_value) or {}
                evaluation_issues.append(
                    {
                        "code": "LIPSCHITZ_NOT_IMPROVED",
                        "problem": (
                            "Candidate L_hat did not improve over the baseline by "
                            "the required margin for this lambda."
                        ),
                        "check_stage": "formal_evaluation",
                        "status": "open",
                        "source_candidate_id": candidate_id,
                        "lambda": lambda_value,
                        "baseline_l_hat": result.get("baseline_l_hat"),
                        "candidate_l_hat": result.get("candidate_l_hat"),
                        "required_margin": result.get("required_margin"),
                        "maximum_pair_ref": diagnostic.get("maximum_pair_ref"),
                        "report_id": relative_report,
                    }
                )
            record["formal_evaluation_issues"] = evaluation_issues
        self._save()
        return self._evaluation_tool_result(
            candidate_id,
            report,
            cache_hit=False,
            report_id=relative_report,
        )

    def _evaluation_tool_result(
        self,
        candidate_id: str,
        report: dict[str, Any],
        *,
        cache_hit: bool,
        report_id: str,
    ) -> dict[str, Any]:
        diagnostics, summary = _compact_evaluation_diagnostics(
            report.get("evaluation_diagnostics") or {}, 8
        )
        return {
            "status": "approved" if report.get("passed") else report.get("status", "failed"),
            "candidate_id": candidate_id,
            "passed": bool(report.get("passed")),
            "cache_hit": bool(cache_hit),
            "report_id": report_id,
            "by_lambda": report.get("by_lambda"),
            "candidate_numeric_diagnostics": report.get("candidate_numeric_diagnostics"),
            "evaluation_diagnostics": diagnostics,
            "diagnostic_compaction": summary,
            "approved_artifact": self.state.get("approved_artifact") if report.get("passed") else None,
            "approval_is_host_determined": True,
        }

    def get_history(
        self,
        candidate_id: str | None,
        record_type: str,
        *,
        start: int = 0,
        limit: int = 20,
        report_id: str | None = None,
    ) -> dict[str, Any]:
        start = max(0, int(start))
        limit = min(50, max(1, int(limit)))

        def pagination(total: int, returned: int) -> dict[str, Any]:
            next_start = start + returned if start + returned < total else None
            return {
                "page_start": start,
                "page_limit": limit,
                "total_record_count": total,
                "returned_count": returned,
                "remaining_count": max(0, total - start - returned),
                "has_more": next_start is not None,
                "next_start": next_start,
                "next_query_arguments": (
                    {
                        "candidate_id": candidate_id,
                        "record_type": record_type,
                        "start": next_start,
                        "limit": limit,
                        "report_id": report_id,
                    }
                    if next_start is not None
                    else None
                ),
            }

        if record_type == "run":
            candidate_ids = list(self.state["candidate_order"])
            selected = candidate_ids[start : start + limit]
            return {
                "status": "ok",
                "work_state": self.work_state(include_candidate=False),
                "candidate_history_page": [
                    {
                        "candidate_id": item,
                        "parent_candidate_id": self.state["candidates"][item].get(
                            "parent_candidate_id"
                        ),
                        "validation_status": self.state["candidates"][item].get(
                            "validation_status"
                        ),
                        "evaluation_status": self.state["candidates"][item].get(
                            "evaluation_status"
                        ),
                        "has_evaluation": bool(
                            self.state["candidates"][item].get("evaluations")
                        ),
                    }
                    for item in selected
                ],
                "total_candidate_count": len(candidate_ids),
                **pagination(len(candidate_ids), len(selected)),
            }
        if record_type == "tool_calls":
            records = list(self.state.get("framework_tool_calls", []))
            selected = records[start : start + limit]
            return {
                "status": "ok",
                "record_type": "tool_calls",
                "tool_call_records": [
                    {
                        key: copy.deepcopy(item.get(key))
                        for key in (
                            "record_id",
                            "tool_call_id",
                            "name",
                            "arguments",
                            "status",
                            "delivery_status",
                            "delivered_model_call_number",
                            "raw_result_path",
                            "raw_result_sha256",
                            "raw_result_serialized_characters",
                            "model_result_serialized_characters",
                            "model_result_summary_level",
                        )
                        if item.get(key) is not None
                    }
                    | {
                        "result_history_query": {
                            "candidate_id": None,
                            "record_type": "tool_result",
                            "start": 0,
                            "limit": 1,
                            "report_id": item.get("record_id"),
                        }
                    }
                    for item in selected
                ],
                **pagination(len(records), len(selected)),
            }
        if record_type == "tool_result":
            framework_record = self._framework_record(str(report_id or ""))
            if framework_record is None:
                return {
                    "status": "invalid_arguments",
                    "error": "report_id must identify an indexed framework tool result",
                }
            return {
                "status": "ok",
                "record_type": "tool_result",
                "record_id": framework_record.get("record_id"),
                "tool_call_id": framework_record.get("tool_call_id"),
                "tool": framework_record.get("name"),
                "arguments": copy.deepcopy(framework_record.get("arguments")),
                "execution_status": framework_record.get("status"),
                "delivery_status": framework_record.get("delivery_status"),
                "result": copy.deepcopy(framework_record.get("model_result")),
                "raw_result_sha256": framework_record.get("raw_result_sha256"),
                "raw_result_serialized_characters": framework_record.get(
                    "raw_result_serialized_characters"
                ),
                "model_result_serialized_characters": framework_record.get(
                    "model_result_serialized_characters"
                ),
                "model_result_summary_level": framework_record.get(
                    "model_result_summary_level"
                ),
            }
        if not candidate_id:
            return {"status": "invalid_arguments", "error": "candidate_id is required for this record type"}
        record = self._candidate_record(candidate_id)
        if record_type == "report":
            allowed_reports = {
                str(item.get("report"))
                for item in (record.get("tests") or {}).values()
                if item.get("report")
            } | {
                str(path) for path in (record.get("evaluations") or {}).values()
            }
            if not report_id or str(report_id) not in allowed_reports:
                return {
                    "status": "invalid_arguments",
                    "error": "report_id must identify a report indexed for this candidate",
                }
            report_path = self.directory / str(report_id)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            serialized_report = json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                default=str,
            )
            segments = _report_text_segments(serialized_report)
            selected = segments[start : start + limit]

            def report_page(page_segments: list[dict[str, Any]]) -> dict[str, Any]:
                next_start = (
                    start + len(page_segments)
                    if start + len(page_segments) < len(segments)
                    else None
                )
                return {
                    "status": "ok",
                    "candidate_id": candidate_id,
                    "record_type": "report",
                    "report_id": report_id,
                    "content_format": (
                        "lossless ordered segments of canonical UTF-8 JSON; "
                        "concatenate each data_segments[].text without separators"
                    ),
                    "report_sha256": hashlib.sha256(
                        serialized_report.encode("utf-8")
                    ).hexdigest(),
                    "page_start": start,
                    "page_limit": limit,
                    "total_character_count": len(serialized_report),
                    "total_segment_count": len(segments),
                    "returned_segment_count": len(page_segments),
                    "remaining_segment_count": max(
                        0, len(segments) - start - len(page_segments)
                    ),
                    "has_more": next_start is not None,
                    "next_start": next_start,
                    "next_query_arguments": (
                        {
                            "candidate_id": candidate_id,
                            "record_type": "report",
                            "start": next_start,
                            "limit": limit,
                            "report_id": report_id,
                        }
                        if next_start is not None
                        else None
                    ),
                    "data_segments": page_segments,
                }

            while (
                len(selected) > 1
                and self._serialized_size(report_page(selected))
                > PAGED_TOOL_PAYLOAD_MAX_CHARS
            ):
                selected.pop()
            return report_page(selected)
        if record_type == "candidate":
            return {
                "status": "ok",
                "candidate_id": candidate_id,
                "candidate": _model_candidate_view(record),
                "parent_candidate_id": record.get("parent_candidate_id"),
                "validation_status": record.get("validation_status"),
                "evaluation_status": record.get("evaluation_status"),
            }
        if record_type == "issues":
            entries = [
                {"issue_group": "current", "issue": item}
                for item in record.get("issues", [])
            ] + [
                {"issue_group": "inherited", "issue": item}
                for item in record.get("inherited_issue_context", [])
            ] + [
                {"issue_group": "formal_evaluation", "issue": item}
                for item in record.get("formal_evaluation_issues", [])
            ]
            selected = entries[start : start + limit]
            return {
                "status": "ok",
                "candidate_id": candidate_id,
                "record_type": "issues",
                "issue_records": selected,
                "issues": [
                    item["issue"]
                    for item in selected
                    if item["issue_group"] == "current"
                ],
                "inherited_issue_context": [
                    item["issue"]
                    for item in selected
                    if item["issue_group"] == "inherited"
                ],
                "formal_evaluation_issues": [
                    item["issue"]
                    for item in selected
                    if item["issue_group"] == "formal_evaluation"
                ],
                **pagination(len(entries), len(selected)),
            }
        if record_type == "tests":
            entries = list((record.get("tests") or {}).items())
            selected = entries[start : start + limit]
            return {
                "status": "ok",
                "candidate_id": candidate_id,
                "record_type": "tests",
                "tests": {key: value for key, value in selected},
                **pagination(len(entries), len(selected)),
            }
        if record_type == "evaluation":
            paths = list((record.get("evaluations") or {}).values())
            reports: list[dict[str, Any]] = []
            for path in paths[start : start + limit]:
                report = json.loads((self.directory / path).read_text(encoding="utf-8"))
                summary = _evaluation_history_summary(
                    candidate_id=candidate_id,
                    report_id=path,
                    report=report,
                )
                trial_reports = reports + [summary]
                trial = {
                    "status": "ok",
                    "candidate_id": candidate_id,
                    "record_type": "evaluation",
                    "evaluations": trial_reports,
                    **pagination(len(paths), len(trial_reports)),
                }
                if self._serialized_size(trial) > PAGED_TOOL_PAYLOAD_MAX_CHARS:
                    if reports:
                        break
                    minimal = _minimal_evaluation_history_summary(summary)
                    reports.append(minimal)
                    break
                reports.append(summary)
            result = {
                "status": "ok",
                "candidate_id": candidate_id,
                "record_type": "evaluation",
                "evaluations": reports,
                **pagination(len(paths), len(reports)),
            }
            if reports and reports[0].get("summary_omitted"):
                result["status"] = "summary_too_large"
            return result
        return {"status": "invalid_arguments", "error": "record_type must be run, tool_calls, tool_result, candidate, issues, tests, evaluation, or report"}


class _WorkspaceTool(Tool):
    workspace: AgentWorkspace

    def __init__(self, workspace: AgentWorkspace):
        self.workspace = workspace
        super().__init__()


class InspectInterfaceTool(_WorkspaceTool):
    name = "inspect_interface"
    description = "Read a stable line page from one controlled section of the authoritative current-only input, candidate, or evaluation contract. Follow next_query_arguments when has_more is true."
    inputs = {
        "section": {"type": "string", "description": "overview, fields, constants, candidate, or evaluation"},
        "start": {"type": "integer", "description": "optional zero-based line offset", "nullable": True},
        "limit": {"type": "integer", "description": f"optional line count, capped at {INTERFACE_PAGE_MAX_LINES}", "nullable": True},
    }
    output_type = "string"

    def forward(
        self,
        section: str,
        start: int | None = None,
        limit: int | None = None,
    ) -> str:
        arguments = {
            "section": section,
            "start": 0 if start is None else start,
            "limit": INTERFACE_PAGE_MAX_LINES if limit is None else limit,
        }
        return self.workspace.invoke(
            self.name,
            arguments,
            lambda: self.workspace.inspect_interface(**arguments),
        )


class QuerySamplesTool(_WorkspaceTool):
    name = "query_samples"
    description = "Read a bounded, complete page of current-only fixed-sample fields. The host may return fewer samples than requested to fit delivery; follow next_query_arguments without changing fields or filters. IDs and evaluation outcomes are diagnostics, never candidate inputs."
    inputs = {
        "fields": {"type": "array", "description": "1..12 obs.* current-only field names", "items": {"type": "string"}},
        "start": {"type": "integer", "description": "zero-based page offset"},
        "limit": {"type": "integer", "description": "page size, capped at 20"},
        "condition_field": {"type": "string", "description": "optional obs.* Boolean field", "nullable": True},
        "condition": {"type": "string", "description": "optional any_true or all_false", "nullable": True},
    }
    output_type = "string"

    def forward(self, fields: list[str], start: int, limit: int, condition_field: str | None = None, condition: str | None = None) -> str:
        arguments = {"fields": fields, "start": start, "limit": limit, "condition_field": condition_field, "condition": condition}
        return self.workspace.invoke(self.name, arguments, lambda: self.workspace.query_samples(**arguments))


class SubmitCandidateTool(_WorkspaceTool):
    name = "submit_candidate"
    description = "Submit one simplified feature candidate object containing only features(name, description, reward_weight) and code. Pass the object directly, never a JSON-encoded string. The host adds artifact metadata deterministically and never repairs code."
    inputs = {
        "candidate": _candidate_tool_input_schema(),
        "parent_candidate_id": {"type": "string", "description": "optional candidate id this revision derives from", "nullable": True},
    }
    output_type = "string"

    def forward(self, candidate: dict[str, Any], parent_candidate_id: str | None = None) -> str:
        arguments = {"candidate": candidate, "parent_candidate_id": parent_candidate_id}
        return self.workspace.invoke(
            self.name,
            arguments,
            lambda: self.workspace.submit_candidate(candidate, parent_candidate_id),
        )


class TestCandidateTool(_WorkspaceTool):
    name = "test_candidate"
    description = "Run isolated candidate validation on every sample in the loaded fixed-sample artifact. A complete test pass is not approval and does not run formal Lipschitz scoring."
    inputs = {
        "candidate_id": {"type": "string", "description": "immutable candidate id returned by submit_candidate"},
    }
    output_type = "string"

    def forward(self, candidate_id: str) -> str:
        return self.workspace.invoke(
            self.name,
            {"candidate_id": candidate_id},
            lambda: self.workspace.test_candidate(candidate_id),
        )


class FormalEvaluateTool(_WorkspaceTool):
    name = "formal_evaluate"
    description = "Fully validate and score one exact candidate on the unchanged fixed pair set and all lambdas. Only a host-approved result creates an artifact."
    inputs = {"candidate_id": {"type": "string", "description": "immutable candidate id to validate and score"}}
    output_type = "string"

    def forward(self, candidate_id: str) -> str:
        return self.workspace.invoke(self.name, {"candidate_id": candidate_id}, lambda: self.workspace.formal_evaluate(candidate_id))


class GetHistoryTool(_WorkspaceTool):
    name = "get_history"
    description = "Retrieve controlled run, tool-call, candidate, issue, test, or evaluation history. Read an indexed model-facing tool result with record_type=tool_result and its record ID in report_id. Indexed test/evaluation reports use record_type=report and their returned report_id. Arbitrary paths are not accepted."
    inputs = {
        "candidate_id": {"type": "string", "description": "candidate id, or null only for run history", "nullable": True},
        "record_type": {"type": "string", "description": "run, tool_calls, tool_result, candidate, issues, tests, evaluation, or report"},
        "start": {"type": "integer", "description": "run-history page offset", "nullable": True},
        "limit": {"type": "integer", "description": "run-history page size, capped at 50", "nullable": True},
        "report_id": {"type": "string", "description": "optional host-indexed tool-result record id or test/evaluation report id", "nullable": True},
    }
    output_type = "string"

    def forward(
        self,
        record_type: str,
        candidate_id: str | None = None,
        start: int | None = None,
        limit: int | None = None,
        report_id: str | None = None,
    ) -> str:
        arguments = {
            "candidate_id": candidate_id,
            "record_type": record_type,
            "start": 0 if start is None else start,
            "limit": 20 if limit is None else limit,
            "report_id": report_id,
        }
        return self.workspace.invoke(
            self.name,
            arguments,
            lambda: self.workspace.get_history(
                candidate_id,
                record_type,
                start=arguments["start"],
                limit=arguments["limit"],
                report_id=arguments["report_id"],
            ),
        )


def build_agent_tools(workspace: AgentWorkspace) -> list[Tool]:
    return [
        InspectInterfaceTool(workspace),
        QuerySamplesTool(workspace),
        SubmitCandidateTool(workspace),
        TestCandidateTool(workspace),
        FormalEvaluateTool(workspace),
        GetHistoryTool(workspace),
    ]


def _normalize_plain_text_completion_request(
    message: ChatMessage, *, call_id: str
) -> ChatMessage:
    """Turn a complete plain-text reply into a host-rejectable stop request."""

    if message.tool_calls or message.content is None:
        return message
    if isinstance(message.content, str) and not message.content.strip():
        return message
    message.tool_calls = [
        ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments={"answer": message.content},
            ),
            id=f"plain-text-{call_id}",
            type="function",
        )
    ]
    return message


class StreamingProviderModel(Model):
    """smolagents model adapter using the project's bounded SSE transport."""

    def __init__(
        self,
        *,
        workspace: AgentWorkspace,
        client: LMStudioClient,
        provider: str,
        model_id: str,
        budget: ModelCallBudget,
        context_length: int,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str | None,
    ):
        super().__init__(model_id=model_id)
        self.workspace = workspace
        self.client = client
        self.provider = str(provider)
        self.budget = budget
        self.context_length = int(context_length)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.reasoning_effort = reasoning_effort

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        if response_format is not None:
            raise ValueError("agent mode does not use response_format")
        token_budget, completion = _request_token_budget(
            self,
            messages,
            tools_to_call_from or [],
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        prospective_call_number = self.budget.used + 1
        call_dir = self.workspace.directory / f"model_call_{prospective_call_number:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        _write_json(call_dir / "token_budget.json", token_budget)
        if not token_budget["fits_client_budget"]:
            raise AgentContextBudgetError("agent messages plus output reservation exceed the client context budget")
        payload = {
            "model": self.model_id,
            **completion,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            if "gpt-oss" not in str(self.model_id).lower():
                raise ValueError("--reasoning-effort is currently supported only for GPT-OSS in agent mode")
            payload["reasoning_effort"] = self.reasoning_effort
        _write_json(call_dir / "request.json", payload)
        started = time.monotonic()
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if self.client.token:
            headers["Authorization"] = f"Bearer {self.client.token}"
        request = urllib.request.Request(
            self.client.base_url + "/chat/completions",
            data=data,
            method="POST",
            headers=headers,
        )
        # Count a generation only when the fully validated request is about to
        # be sent.  Local parameter/context failures above consume no budget.
        call_number = self.budget.consume()
        self.workspace.state["model_calls_used"] = self.budget.used
        self.workspace._save()
        try:
            response = self.client._open_stream(
                request,
                connect_timeout=min(self.client.connect_timeout, self.client.total_timeout),
                header_timeout=min(self.client.timeout, self.client.total_timeout),
                total_timeout=self.client.total_timeout,
            )
            if getattr(response, "status", 200) >= 300:
                remaining = max(0.0, self.client.total_timeout - (time.monotonic() - started))
                body, status = read_response_body_bounded(
                    response,
                    idle_timeout=self.client.timeout,
                    total_timeout=remaining,
                    redact=self.client._safe,
                )
                raise APIError(
                    f"{self.provider} HTTP {response.status}: {self.client._safe(body.decode('utf-8', errors='replace'))}",
                    category=status.get("status", "http_error"),
                )
            remaining = self.client.total_timeout - (time.monotonic() - started)
            streamed = capture_chat_stream(
                response,
                directory=call_dir,
                idle_timeout=self.client.timeout,
                total_timeout=remaining,
                progress_interval=self.client.progress_interval,
                redact=self.client._safe,
            )
        except StreamTransportError as exc:
            raise APIError(self.client._safe(exc), category=exc.category) from exc
        if not streamed["transport_completed"]:
            raise APIError("agent generation stream did not complete", category="stream_incomplete")
        actual = streamed.get("actual_model")
        content, inline_reasoning = adapter_separate_reasoning(
            model_adapter(str(self.model_id)), streamed.get("content")
        )
        reasoning = streamed.get("reasoning")
        if inline_reasoning:
            reasoning = inline_reasoning if not reasoning else f"{reasoning}\n{inline_reasoning}"
        raw_record = {
            "actual_model": actual,
            "finish_reason": streamed.get("finish_reason"),
            "content": content,
            "reasoning": reasoning,
            "refusal": streamed.get("refusal"),
            "done_received": streamed.get("done_received"),
            "terminal_chunk_received": streamed.get("terminal_chunk_received"),
            "tool_calls": streamed.get("tool_calls") or [],
            "usage": streamed.get("usage") or {},
        }
        _write_json(call_dir / "response.json", raw_record)
        if actual is not None and not response_model_matches(
            self.provider, str(self.model_id), str(actual)
        ):
            raise APIError(
                "provider returned a different model: "
                f"requested={self.model_id!r}, actual={actual!r}"
            )
        refusal = streamed.get("refusal")
        if refusal:
            raise APIError(
                "model refused the agent request; no tool call was executed",
                category="model_refusal",
            )
        finish_reason = streamed.get("finish_reason")
        raw_tool_calls = streamed.get("tool_calls") or []
        if finish_reason == "length":
            raise APIError(
                "agent output was truncated at the generation limit; no tool call was executed",
                category="output_truncated",
            )
        if finish_reason == "content_filter":
            raise APIError(
                "agent output was stopped by content filtering; no tool call was executed",
                category="content_filter",
            )
        normal_tool_completion = finish_reason == "tool_calls"
        lmstudio_stop_tool_completion = (
            self.provider == "lmstudio"
            and finish_reason == "stop"
            and bool(raw_tool_calls)
        )
        normal_text_completion = (
            finish_reason == "stop"
            and not raw_tool_calls
            and content is not None
            and (not isinstance(content, str) or bool(content.strip()))
        )
        if not (
            normal_tool_completion
            or lmstudio_stop_tool_completion
            or normal_text_completion
        ):
            raise APIError(
                "agent response did not end with a recognized complete tool-call signal "
                f"(finish_reason={finish_reason!r}); no tool call was executed",
                category="invalid_finish_reason",
            )
        if normal_text_completion:
            raw_tool_calls = [
                {
                    "index": 0,
                    "id": f"plain-text-{call_number}",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps(
                            {"answer": content},
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    },
                }
            ]
        tool_calls = []
        for index, raw in enumerate(raw_tool_calls):
            function = raw.get("function") or {}
            arguments = function.get("arguments", "")
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise APIError(f"tool call arguments are incomplete or invalid JSON: {exc}", category="tool_call_parse_error") from exc
            if not isinstance(parsed_arguments, dict):
                raise APIError("tool call arguments must decode to an object", category="tool_call_parse_error")
            tool_calls.append(
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=str(function.get("name", "")), arguments=parsed_arguments
                    ),
                    id=str(raw.get("id") or f"call-{call_number}-{index}"),
                    type=str(raw.get("type") or "function"),
                )
            )
        if not tool_calls:
            raise APIError(
                "agent response completed without a tool call",
                category="missing_tool_call",
            )
        usage = streamed.get("usage") or {}
        token_usage = None
        if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
            token_usage = TokenUsage(
                input_tokens=int(usage["prompt_tokens"]),
                output_tokens=int(usage["completion_tokens"]),
            )
        self.workspace.mark_prepared_results_delivered(
            model_call_number=call_number
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls or None,
            raw=raw_record,
            token_usage=token_usage,
        )


class BudgetedModelProxy(Model):
    """Apply the same exact call budget to injected/mock framework models."""

    def __init__(self, delegate: Model, budget: ModelCallBudget, workspace: AgentWorkspace):
        super().__init__(model_id=getattr(delegate, "model_id", None))
        self.delegate = delegate
        self.budget = budget
        self.workspace = workspace

    def generate(self, *args, **kwargs):
        self.budget.consume()
        self.workspace.state["model_calls_used"] = self.budget.used
        self.workspace._save()
        message = self.delegate.generate(*args, **kwargs)
        normalized = _normalize_plain_text_completion_request(
            message, call_id=str(self.budget.used)
        )
        self.workspace.mark_prepared_results_delivered(
            model_call_number=self.budget.used
        )
        return normalized


class ControlledToolCallingAgent(ToolCallingAgent):
    """Framework agent with sequential tools, bounded calls, and host approval stop."""

    def __init__(
        self,
        *args,
        workspace: AgentWorkspace,
        context_length: int,
        max_output_tokens: int,
        task_renderer=None,
        model_calls_maximum: int | None = None,
        **kwargs,
    ):
        self.workspace = workspace
        self.context_length = int(context_length)
        self.max_output_tokens = int(max_output_tokens)
        self.task_renderer = task_renderer
        self.model_calls_maximum = model_calls_maximum
        super().__init__(*args, max_tool_threads=1, planning_interval=None, **kwargs)
        self.prompt_templates["system_prompt"] = HOST_CONTROLLED_SYSTEM_PROMPT
        final_tool = self.tools.get("final_answer")
        if final_tool is not None:
            final_tool.description = (
                "Request task completion. The host rejects this request unless a "
                "formal evaluation has already saved an approved artifact."
            )

    def process_tool_calls(self, chat_message, memory_step):
        chat_calls = list(chat_message.tool_calls or [])
        self.workspace.register_framework_calls(chat_calls)
        memory_calls: list[ToolCall] = []
        observations: list[str] = []
        approval_reason = "a prior tool in this model response received host approval"
        prior_tool_error: dict[str, Any] | None = None

        def sync_memory() -> None:
            memory_step.tool_calls = list(memory_calls)
            memory_step.observations = (
                "\n".join(observations) if observations else None
            )

        def append_observation(tool_call: ToolCall, status: str, value: Any) -> str:
            payload = {
                "tool_call_id": tool_call.id,
                "tool": tool_call.name,
                "status": status,
                "result": value,
            }
            observation = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, default=str
            )
            observations.append(observation)
            sync_memory()
            return observation

        for chat_call in chat_calls:
            tool_call = ToolCall(
                name=chat_call.function.name,
                arguments=chat_call.function.arguments,
                id=chat_call.id,
            )
            memory_calls.append(tool_call)
            sync_memory()
            yield tool_call
            if self.workspace.approved:
                skipped = {
                    "status": "skipped_after_approval",
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "reason": approval_reason,
                }
                skipped = self.workspace.skip_framework_call(
                    tool_call.id,
                    reason=approval_reason,
                )
                observation = append_observation(
                    tool_call, "skipped_after_approval", skipped
                )
                yield ToolOutput(
                    id=tool_call.id,
                    output=skipped,
                    is_final_answer=False,
                    observation=observation,
                    tool_call=tool_call,
                )
                continue
            if prior_tool_error is not None:
                skipped = {
                    "status": "skipped_after_tool_error",
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "reason": (
                        "not executed because an earlier tool call in this model "
                        "response failed; correct the failed call in the next turn"
                    ),
                    "earlier_failed_tool_call_id": prior_tool_error["tool_call_id"],
                }
                skipped = self.workspace.skip_framework_call_with_status(
                    tool_call.id,
                    status="skipped_after_tool_error",
                    reason=skipped["reason"],
                )
                observation = append_observation(
                    tool_call, "skipped_after_tool_error", skipped
                )
                yield ToolOutput(
                    id=tool_call.id,
                    output=skipped,
                    is_final_answer=False,
                    observation=observation,
                    tool_call=tool_call,
                )
                continue
            if tool_call.name == "final_answer":
                arguments = tool_call.arguments or {}
                result = self.workspace.reject_completion(
                    tool_call_id=tool_call.id,
                    arguments=arguments,
                    model_calls_maximum=int(
                        self.model_calls_maximum
                        or self.workspace.state["settings"]["max_model_calls"]
                    ),
                    source=(
                        "plain_text_response"
                        if str(tool_call.id).startswith("plain-text-")
                        else "final_answer_tool"
                    ),
                )
                result = self.workspace.finish_framework_call(
                    tool_call.id,
                    status="completion_rejected",
                    output=result,
                )
                observation = append_observation(
                    tool_call, "completion_rejected", result
                )
                yield ToolOutput(
                    id=tool_call.id,
                    output=result,
                    is_final_answer=False,
                    observation=observation,
                    tool_call=tool_call,
                )
                continue
            try:
                result = self.execute_tool_call(
                    tool_call.name, tool_call.arguments or {}
                )
                parsed_result = None
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                    except (TypeError, ValueError):
                        pass
                result_status = (
                    str(parsed_result.get("status"))
                    if isinstance(parsed_result, dict)
                    else "completed"
                )
                correctable_failure = result_status in {
                    "invalid_arguments",
                    "tool_error",
                }
                framework_status = "failed" if correctable_failure else "completed"
                result = self.workspace.finish_framework_call(
                    tool_call.id, status=framework_status, output=result
                )
            except Exception as exc:
                available = sorted(
                    str(getattr(tool, "name", tool))
                    for tool in self.tools_and_managed_agents
                )
                result_status = "tool_error"
                framework_status = "failed"
                result = {
                    "status": result_status,
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "correction": (
                        "Use one of the available tool names and match its declared "
                        "argument schema before retrying. Previously completed calls "
                        "will not be replayed automatically."
                    ),
                    "available_tools": available,
                }
                result = self.workspace.finish_framework_call(
                    tool_call.id, status=framework_status, output=result
                )
                correctable_failure = True
            observation = append_observation(tool_call, framework_status, result)
            yield ToolOutput(
                id=tool_call.id,
                output=result,
                is_final_answer=False,
                observation=observation,
                tool_call=tool_call,
            )
            if correctable_failure:
                prior_tool_error = {
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "status": result_status,
                }
        sync_memory()

    def _step_stream(self, memory_step):
        for output in super()._step_stream(memory_step):
            yield output
        if self.workspace.approved:
            yield ActionOutput(
                output=json.dumps(
                    {
                        "status": "approved",
                        "candidate_id": self.workspace.state[
                            "approved_candidate_id"
                        ],
                        "approved_artifact": self.workspace.state[
                            "approved_artifact"
                        ],
                    },
                    ensure_ascii=False,
                ),
                is_final_answer=True,
            )

    def write_memory_to_messages(self, summary_mode: bool = False):
        messages = super().write_memory_to_messages(summary_mode=summary_mode)
        all_pending_records = self.workspace.pending_delivery_records()
        all_pending_ids = [
            str(record["record_id"]) for record in all_pending_records
        ]
        budget, _ = _request_token_budget(
            self.model,
            messages,
            self.tools_and_managed_agents,
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        if budget["fits_client_budget"]:
            self.workspace.prepare_result_delivery(all_pending_ids)
            return messages
        # Keep whole memory-step message groups. Pending results are protected;
        # only older groups whose results were already delivered are removable.
        system_messages = self.memory.system_prompt.to_messages(summary_mode=False)
        group_records = []
        represented_pending_ids: set[str] = set()
        for step in self.memory.steps[1:]:
            calls = list(getattr(step, "tool_calls", None) or [])
            call_ids = [str(call.id) for call in calls]
            pending_ids = self.workspace.pending_record_ids_for_call_ids(call_ids)
            represented_pending_ids.update(pending_ids)
            group_records.append(
                {
                    "messages": step.to_messages(summary_mode=False),
                    "pending_record_ids": pending_ids,
                }
            )
        pending_without_memory = set(all_pending_ids) - represented_pending_ids
        current_candidate_id = self.workspace.state.get("current_candidate_id")
        pending_supplies_current_candidate = any(
            record.get("name") == "get_history"
            and (record.get("arguments") or {}).get("record_type") == "candidate"
            and (record.get("arguments") or {}).get("candidate_id")
            == current_candidate_id
            for record in all_pending_records
        )
        kept_indexes = list(range(len(group_records)))

        def compact_context(summary_level: str):
            work_state = self.workspace.work_state(
                include_candidate=not pending_supplies_current_candidate,
                model_calls_maximum=self.model_calls_maximum,
                include_pending_payloads=bool(pending_without_memory),
                pending_payload_record_ids=pending_without_memory,
                summary_level=summary_level,
            )
            if self.task_renderer is None:  # Defensive for direct construction.
                task = (
                    "Host-generated compact work-state summary. Structured facts only; "
                    "older full messages remain on disk:\n"
                    + json.dumps(
                        work_state,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            else:
                # Rebuild the common task around exactly one fresh work-state block.
                # This avoids retaining the stale initial summary and appending a
                # second, ever-growing copy during context compaction or resume.
                task = self.task_renderer(work_state)
            task_messages = TaskStep(task=task).to_messages(summary_mode=False)
            return work_state, task, task_messages

        def request_messages(
            task_messages: list[ChatMessage], indexes: list[int]
        ) -> list[ChatMessage]:
            return system_messages + task_messages + [
                message
                for index in indexes
                for message in group_records[index]["messages"]
            ]

        def budget_components(
            work_state: dict[str, Any],
            task: str,
            task_messages: list[ChatMessage],
            indexes: list[int],
            *,
            summary_level: str,
        ) -> dict[str, Any]:
            current = work_state.get("current_candidate") or {}
            work_state_characters = _json_characters(work_state)
            candidate_characters = _json_characters(current.get("candidate") or {})
            evaluation_characters = _json_characters(
                current.get("selected_formal_evaluation") or {}
            )
            issue_characters = _json_characters(
                current.get("unresolved_issue_summary") or {}
            )
            query_characters = _json_characters(
                work_state.get("recent_completed_queries") or {}
            )
            pending_characters = sum(
                _json_characters(record.get("model_result"))
                for record in all_pending_records
            )
            retained_history_characters = sum(
                _message_characters(group_records[index]["messages"])
                for index in indexes
            )
            tool_characters = _json_characters(
                [get_tool_json_schema(tool) for tool in self.tools_and_managed_agents]
            )
            work_known = (
                candidate_characters
                + evaluation_characters
                + issue_characters
                + query_characters
            )
            return {
                "summary_level": summary_level,
                "components_are_diagnostic_not_additive": True,
                "system": _budget_component(_message_characters(system_messages)),
                "task_and_interface_excluding_work_state": _budget_component(
                    max(0, len(task) - work_state_characters)
                ),
                "tool_definitions": _budget_component(tool_characters),
                "candidate": _budget_component(candidate_characters),
                "work_summary_evaluation": _budget_component(
                    evaluation_characters
                ),
                "work_summary_issues": _budget_component(issue_characters),
                "work_summary_queries": _budget_component(query_characters),
                "work_summary_other": _budget_component(
                    max(0, work_state_characters - work_known)
                ),
                "pending_tool_results": _budget_component(pending_characters),
                "other_retained_history": _budget_component(
                    retained_history_characters
                ),
                "compact_task_messages": _budget_component(
                    _message_characters(task_messages)
                ),
                "output_reservation": {
                    "tokens": self.max_output_tokens,
                    "exact_for_client_setting": True,
                },
            }

        summary_levels = ("standard", "minimal")
        last_context = None
        for summary_level in summary_levels:
            compact_work_state, compact_task, compact_task_messages = compact_context(
                summary_level
            )
            while True:
                candidate = request_messages(compact_task_messages, kept_indexes)
                estimate, _ = _request_token_budget(
                    self.model,
                    candidate,
                    self.tools_and_managed_agents,
                    context_length=self.context_length,
                    max_output_tokens=self.max_output_tokens,
                )
                components = budget_components(
                    compact_work_state,
                    compact_task,
                    compact_task_messages,
                    kept_indexes,
                    summary_level=summary_level,
                )
                last_context = (
                    compact_work_state,
                    compact_task,
                    compact_task_messages,
                    estimate,
                    components,
                )
                if estimate["fits_client_budget"]:
                    omitted = len(group_records) - len(kept_indexes)
                    protected = sum(
                        bool(group_records[index]["pending_record_ids"])
                        for index in kept_indexes
                    )
                    self.workspace.state["context_compactions"].append(
                        {
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "omitted_complete_step_groups": omitted,
                            "retained_complete_step_groups": len(kept_indexes),
                            "retained_pending_result_groups": protected,
                            "pending_result_record_ids": all_pending_ids,
                            "pending_results_embedded_in_work_state": sorted(
                                pending_without_memory
                            ),
                            "candidate_code_truncated": False,
                            "tool_call_result_pairs_split": False,
                            "work_summary_level": summary_level,
                            "token_budget": estimate,
                            "budget_components": components,
                        }
                    )
                    self.workspace._save()
                    self.workspace.prepare_result_delivery(all_pending_ids)
                    return candidate
                removable = next(
                    (
                        index
                        for index in kept_indexes
                        if not group_records[index]["pending_record_ids"]
                    ),
                    None,
                )
                if removable is None:
                    break
                kept_indexes.remove(removable)

        assert last_context is not None
        (
            compact_work_state,
            compact_task,
            compact_task_messages,
            estimate,
            components,
        ) = last_context
        minimum = request_messages(compact_task_messages, kept_indexes)
        estimate, _ = _request_token_budget(
            self.model,
            minimum,
            self.tools_and_managed_agents,
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        self.workspace.state["context_compactions"].append(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "context_budget_exceeded",
                "omitted_complete_step_groups": len(group_records)
                - len(kept_indexes),
                "retained_complete_step_groups": len(kept_indexes),
                "retained_pending_result_groups": sum(
                    bool(group_records[index]["pending_record_ids"])
                    for index in kept_indexes
                ),
                "pending_result_record_ids": all_pending_ids,
                "pending_results_embedded_in_work_state": sorted(
                    pending_without_memory
                ),
                "candidate_code_truncated": False,
                "tool_call_result_pairs_split": False,
                "work_summary_level": "minimal",
                "token_budget": estimate,
                "budget_components": components,
            }
        )
        self.workspace._save()
        self.workspace.prepare_result_delivery([])
        raise AgentContextBudgetError(
            "system/task prompt, complete current candidate, tool definitions, "
            "output reservation, and the minimum pending tool-result delivery do "
            "not fit; pending results were preserved and no incomplete request was sent"
        )

    def _handle_max_steps_reached(self, task: str) -> Any:
        # smolagents normally makes one extra model call here.  Returning a
        # deterministic host result keeps --max-model-calls an exact bound.
        self.workspace.state["status"] = "paused_budget_exhausted"
        self.workspace.state["stop_reason"] = (
            "offline design is unfinished because the model call budget was exhausted"
        )
        self.workspace._save()
        return json.dumps(
            {
                "status": "not_approved",
                "reason": self.workspace.state["stop_reason"],
                "current_candidate_id": self.workspace.state.get("current_candidate_id"),
            }
        )


def _load_resume_state(directory: Path) -> dict[str, Any]:
    path = directory / "agent_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"agent resume state is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != AGENT_RUN_SCHEMA_VERSION:
        raise ValueError("agent resume state schema is incompatible")
    if state.get("status") == "approved":
        raise ValueError("approved agent runs are complete and cannot be resumed")
    return state


def run_agent(
    *,
    fixed_sample: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    reasoning_effort: str | None = None,
    beta: float = DEFAULT_BETA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    progress_interval: float = DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    worker_timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    output_root: str | Path = DEFAULT_AGENT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    resume: str | Path | None = None,
    additional_model_calls: int | None = None,
    model_backend: Model | None = None,
    client: LMStudioClient | None = None,
) -> dict[str, Any]:
    if additional_model_calls is not None:
        if resume is None:
            raise ValueError("--additional-model-calls is valid only with --resume")
        if int(additional_model_calls) <= 0:
            raise ValueError("--additional-model-calls must be positive")
    resume_state = None
    source_directory: Path | None = None
    original_model_call_maximum = int(max_model_calls)
    if resume is not None:
        source_directory = Path(resume).resolve()
        directory = source_directory
        resume_state = _load_resume_state(source_directory)
        fixed_sample = resume_state["fixed_sample"]["directory"]
        provider = resume_state["provider"]
        model = resume_state["model"]
        settings = resume_state["settings"]
        beta = settings["beta"]
        batch_size = settings["batch_size"]
        worker_timeout = settings["worker_timeout"]
        absolute_tolerance = settings["absolute_tolerance"]
        relative_tolerance = settings["relative_tolerance"]
        original_model_call_maximum = int(settings["max_model_calls"])
        max_model_calls = original_model_call_maximum
        request_settings = resume_state.get("request_settings")
        if not isinstance(request_settings, dict):
            raise ValueError("agent resume state is missing request settings")
        base_url = request_settings["base_url"]
        context_length = request_settings["context_length"]
        max_output_tokens = request_settings["max_output_tokens"]
        temperature = request_settings["temperature"]
        seed = request_settings["seed"]
        reasoning_effort = request_settings["reasoning_effort"]
        timeout = request_settings["timeout"]
        connect_timeout = request_settings["connect_timeout"]
        total_timeout = request_settings["total_timeout"]
        progress_interval = request_settings["progress_interval"]
    else:
        if fixed_sample is None or provider is None or model is None:
            raise ValueError("new agent runs require --fixed-sample, --provider, and --model")
        directory = _agent_directory(
            str(model), output_root=output_root, output_dir=output_dir
        )
        original_model_call_maximum = int(max_model_calls)
    provider = str(provider).lower()
    if provider not in {"lmstudio", "openai"}:
        raise ValueError("provider must be lmstudio or openai")
    if int(max_model_calls) <= 0:
        raise ValueError("max_model_calls must be positive")
    resolved_base = provider_base_url(provider, base_url)
    workspace = AgentWorkspace(
        directory=directory,
        fixed_sample=str(fixed_sample),
        provider=provider,
        model=str(model),
        beta=beta,
        batch_size=batch_size,
        worker_timeout=worker_timeout,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        max_model_calls=max_model_calls,
        resume_state=resume_state,
        read_only=bool(resume is not None and dry_run),
    )
    request_settings = {
        "base_url": resolved_base,
        "context_length": int(context_length),
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "seed": int(seed),
        "reasoning_effort": reasoning_effort,
        "timeout": float(timeout),
        "connect_timeout": float(connect_timeout),
        "total_timeout": float(total_timeout),
        "progress_interval": float(progress_interval),
    }
    if resume_state is None:
        workspace.state["request_settings"] = request_settings
        workspace._save()
    elif workspace.state.get("request_settings") != request_settings:
        raise ValueError("agent resume request settings are incompatible")
    if resume is not None and dry_run:
        assert source_directory is not None
        artifact_directory = _resume_preview_directory(
            source_directory,
            output_root=output_root,
            output_dir=output_dir,
        )
    else:
        artifact_directory = directory
    effective_model_call_maximum = original_model_call_maximum + int(
        additional_model_calls or 0
    )
    budget = ModelCallBudget(
        effective_model_call_maximum,
        workspace.state["model_calls_used"],
    )
    model_info = None
    if client is None and model_backend is None:
        client_type = OpenAIClient if provider == "openai" else LMStudioClient
        client = client_type(
            resolved_base,
            timeout=timeout,
            connect_timeout=connect_timeout,
            total_timeout=total_timeout,
            progress_interval=progress_interval,
            retries=0,
        )
    if not dry_run and model_backend is None:
        assert client is not None
        client.validate_configuration()
        if provider == "lmstudio":
            model_info = model_inventory_summary(client.list_models(), str(model))
    effective_context, context_info = _effective_context_budget(context_length, model_info)
    tools = build_agent_tools(workspace)
    def render_task(current_work_state: dict[str, Any]) -> str:
        return _render_agent_prompt(
            fixed_metadata=workspace.fixed_metadata,
            baseline=workspace.baseline,
            constants_metadata=workspace.constants,
            beta=workspace.beta,
            absolute_tolerance=workspace.absolute_tolerance,
            relative_tolerance=workspace.relative_tolerance,
            current_work_state=current_work_state,
        )

    resume_pending = workspace.pending_delivery_records() if resume is not None else []
    current_candidate_id = workspace.state.get("current_candidate_id")
    pending_supplies_current_candidate = any(
        record.get("name") == "get_history"
        and (record.get("arguments") or {}).get("record_type") == "candidate"
        and (record.get("arguments") or {}).get("candidate_id")
        == current_candidate_id
        for record in resume_pending
    )
    task = render_task(
        workspace.work_state(
            include_candidate=not pending_supplies_current_candidate,
            model_calls_maximum=effective_model_call_maximum,
            include_pending_payloads=resume is not None,
        )
    )
    if model_backend is None:
        assert client is not None
        model_backend = StreamingProviderModel(
            workspace=workspace,
            client=client,
            provider=provider,
            model_id=str(model),
            budget=budget,
            context_length=effective_context,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
        )
    else:
        model_backend = BudgetedModelProxy(model_backend, budget, workspace)
    available_model_calls = budget.remaining
    agent = ControlledToolCallingAgent(
        tools=tools,
        model=model_backend,
        workspace=workspace,
        context_length=effective_context,
        max_output_tokens=max_output_tokens,
        task_renderer=render_task,
        model_calls_maximum=effective_model_call_maximum,
        max_steps=max(1, available_model_calls),
        add_base_tools=False,
        stream_outputs=False,
    )
    initial_messages = (
        agent.memory.system_prompt.to_messages(summary_mode=False)
        + TaskStep(task=task).to_messages(summary_mode=False)
    )
    prompt_budget, initial_completion = _request_token_budget(
        agent.model,
        initial_messages,
        agent.tools_and_managed_agents,
        context_length=effective_context,
        max_output_tokens=max_output_tokens,
    )
    (artifact_directory / "agent_task_prompt.txt").write_text(task, encoding="utf-8")
    (artifact_directory / "framework_system_prompt.txt").write_text(agent.system_prompt, encoding="utf-8")
    _write_json(
        artifact_directory / "tool_schemas.json",
        [get_tool_json_schema(tool) for tool in tools],
    )
    _write_json(artifact_directory / "initial_request_shape.json", initial_completion)
    metadata = {
        "schema_version": AGENT_RUN_SCHEMA_VERSION,
        "agent_framework": {
            "name": "smolagents",
            "version": SMOLAGENTS_VERSION,
            "agent": "ToolCallingAgent",
        },
        "prompt_version": AGENT_PROMPT_VERSION,
        "tool_contract_version": AGENT_TOOL_CONTRACT_VERSION,
        "provider": provider,
        "model": str(model),
        "base_url": resolved_base,
        "model_inventory": model_info,
        "generation": {
            "original_max_model_calls": original_model_call_maximum,
            "additional_model_calls": int(additional_model_calls or 0),
            "effective_max_model_calls": effective_model_call_maximum,
            "temperature": temperature,
            "seed": seed,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "planning_disabled": True,
            "automatic_generation_retries": 0,
            "smolagents_max_steps_extra_final_call_disabled": True,
        },
        "context": context_info,
        "initial_context_budget": prompt_budget,
        "dry_run": bool(dry_run),
        "resume": resume is not None,
        "source_run_directory": (
            str(source_directory) if source_directory is not None else None
        ),
        "resume_source_mutated": False if resume is not None and dry_run else None,
        "git_sha": _git_sha(),
    }
    if resume is None:
        metadata_path = artifact_directory / "run_metadata.json"
    elif dry_run:
        metadata_path = artifact_directory / "resume_preview_metadata.json"
    else:
        resume_number = len(workspace.state.get("resume_events", [])) + 1
        metadata_path = artifact_directory / f"resume_metadata_{resume_number:03d}.json"
    _write_json(metadata_path, metadata)
    if not prompt_budget["fits_client_budget"]:
        workspace.state["status"] = "failed_context_budget"
        workspace.state["stop_reason"] = "initial framework system/task/tool prompt exceeds context budget"
        workspace._save()
        if resume is not None and dry_run:
            _write_json(
                artifact_directory / "failure.json",
                {
                    "error_type": "AgentContextBudgetError",
                    "category": "context_budget_exceeded",
                    "error": workspace.state["stop_reason"],
                    "source_run_mutated": False,
                },
            )
        error_message = workspace.state["stop_reason"]
        if resume is not None and dry_run:
            error_message += f"; resume dry-run preview: {artifact_directory}"
        raise AgentContextBudgetError(error_message)
    if dry_run:
        if resume is None:
            workspace.state["status"] = "dry_run_complete"
            workspace.state["stop_reason"] = "dry run; no generation request sent"
            workspace._save()
        return {
            "status": "dry_run_complete",
            "output_directory": str(artifact_directory),
            "source_run_directory": (
                str(source_directory) if source_directory is not None else None
            ),
            "metadata": metadata,
            "model_calls_used": budget.used,
        }
    if resume is not None:
        if additional_model_calls:
            workspace.state.setdefault("budget_extension_events", []).append(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "additional_model_calls": int(additional_model_calls),
                    "previous_max_model_calls": original_model_call_maximum,
                    "new_max_model_calls": effective_model_call_maximum,
                    "model_calls_used": budget.used,
                }
            )
            workspace.state["settings"]["max_model_calls"] = (
                effective_model_call_maximum
            )
        workspace.state.setdefault("resume_events", []).append(
            {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "previous_status": workspace.state.get("status"),
                "model_calls_used": budget.used,
            }
        )
        workspace.state["status"] = "running"
        workspace.state["stop_reason"] = None
        workspace._save()
    if available_model_calls == 0:
        workspace.state["status"] = "paused_budget_exhausted"
        workspace.state["stop_reason"] = (
            "offline design remains unfinished; the saved model call budget is exhausted"
        )
        workspace._save()
        return {
            "status": workspace.state["status"],
            "output_directory": str(directory),
            "approved_artifact": workspace.state.get("approved_artifact"),
            "model_calls_used": budget.used,
            "tool_operations": len(workspace.state["tool_operations"]),
            "stop_reason": workspace.state["stop_reason"],
            "metadata": metadata,
        }
    try:
        result = agent.run(
            task,
            max_steps=available_model_calls,
            return_full_result=True,
        )
        (artifact_directory / "framework_final_output.txt").write_text(str(result.output), encoding="utf-8")
        _write_json(artifact_directory / "framework_steps.json", result.steps)
        if workspace.approved:
            workspace.state["status"] = "approved"
        elif workspace.state["status"] == "running":
            workspace.state["status"] = "stopped_without_approval"
            workspace.state["stop_reason"] = workspace.state.get("stop_reason") or "framework final answer without host approval"
    except KeyboardInterrupt:
        workspace.state["status"] = "interrupted"
        workspace.state["stop_reason"] = "cancelled by user; no unknown request is retried"
    except AgentContextBudgetError as exc:
        workspace.state["status"] = "failed_context_budget"
        workspace.state["stop_reason"] = f"context_budget_exceeded: {exc}"
        _write_json(
            artifact_directory / "failure.json",
            {
                "error_type": type(exc).__name__,
                "category": "context_budget_exceeded",
                "error": str(exc),
            },
        )
    except Exception as exc:
        workspace.state["status"] = "failed"
        workspace.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        _write_json(artifact_directory / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        workspace.state["model_calls_used"] = budget.used
        workspace._save()
    return {
        "status": workspace.state["status"],
        "output_directory": str(artifact_directory),
        "approved_artifact": workspace.state.get("approved_artifact"),
        "model_calls_used": budget.used,
        "tool_operations": len(workspace.state["tool_operations"]),
        "stop_reason": workspace.state.get("stop_reason"),
        "metadata": metadata,
    }
