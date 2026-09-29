"""Autonomous, tool-driven offline LLM feature design using smolagents.

This module is intentionally optional.  Core training and ``run_llm_design`` do
not import it, so an environment without smolagents keeps the legacy behavior.
"""

from __future__ import annotations

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
        ToolCallingAgent,
        ToolOutput,
    )
    from smolagents.models import ChatMessageToolCallFunction, get_tool_json_schema
except ImportError as exc:  # pragma: no cover - exercised in a clean subprocess
    raise ImportError(
        "run_llm_agent.py requires the optional agent dependency; install "
        "requirements-llm-agent.txt. The legacy run_llm_design.py entry point "
        "does not require it."
    ) from exc

from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    candidate_numeric_diagnostics,
    execute_candidate_isolated,
    feature_reward,
    parse_candidate_json_envelope,
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
    candidate_schema,
    estimate_token_budget,
    format_schema_and_example,
    load_design_inputs,
    render_environment_interface,
    runtime_diagnostic_contract,
)
from llm_streaming import (
    StreamTransportError,
    capture_chat_stream,
    read_response_body_bounded,
)


AGENT_RUN_SCHEMA_VERSION = "uav-hrl-llm-feature-agent-run-v1"
AGENT_PROMPT_VERSION = "uav-hrl-llm-feature-agent-prompt-v1"
AGENT_TOOL_CONTRACT_VERSION = "uav-hrl-llm-feature-agent-tools-v1"
DEFAULT_MAX_MODEL_CALLS = 20
DEFAULT_AGENT_OUTPUT_ROOT = Path("results") / "llm_agents"
AGENT_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompts") / "llm_agent_prompt.txt"
SMOLAGENTS_VERSION = distribution_version("smolagents")


class AgentBudgetError(RuntimeError):
    pass


class AgentContextBudgetError(RuntimeError):
    pass


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
        format_schema_and_example(candidate_schema())
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
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
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
    ):
        self.directory = Path(directory).resolve()
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
                "tool_operations": [],
                "framework_tool_calls": [],
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
            self.state = resume_state
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
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.directory / "agent_state.json", self.state)

    def work_state(self, *, include_candidate: bool = True) -> dict[str, Any]:
        current_id = self.state.get("current_candidate_id")
        current = self.state["candidates"].get(current_id) if current_id else None
        result = {
            "status": self.state["status"],
            "current_candidate_id": current_id,
            "candidate_count": len(self.state["candidate_order"]),
            "approved_candidate_id": self.state.get("approved_candidate_id"),
            "model_calls_used": self.state["model_calls_used"],
            "model_calls_maximum": self.state["settings"]["max_model_calls"],
            "model_calls_remaining": (
                int(self.state["settings"]["max_model_calls"])
                - int(self.state["model_calls_used"])
            ),
            "recent_operations": [
                {
                    key: operation.get(key)
                    for key in ("operation_id", "tool", "candidate_id", "status", "result_summary")
                }
                for operation in self.state["tool_operations"][-8:]
            ],
        }
        if current is not None:
            result["current_candidate"] = {
                "candidate_id": current_id,
                "content_sha256": current["content_sha256"],
                "parent_candidate_id": current.get("parent_candidate_id"),
                "validation_status": current.get("validation_status"),
                "evaluation_status": current.get("evaluation_status"),
                "issues": current.get("issues", []),
            }
            if include_candidate:
                result["current_candidate"]["candidate"] = current.get("candidate")
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
                )
                if key in result
            }
            self._save()

    def invoke(self, tool: str, arguments: dict[str, Any], function) -> str:
        operation = self.begin_operation(tool, arguments)
        try:
            result = function()
            if not isinstance(result, dict):
                raise TypeError("agent tool implementation did not return an object")
            self.finish_operation(operation, status="completed", result=result)
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
                self.state["framework_tool_calls"].append(
                    {
                        "tool_call_id": str(call.id),
                        "name": str(call.function.name),
                        "arguments": call.function.arguments,
                        "status": "requested",
                    }
                )
            self._save()

    def complete_framework_call(self, call_id: str, output: Any) -> None:
        with self._lock:
            for record in reversed(self.state["framework_tool_calls"]):
                if record["tool_call_id"] == str(call_id):
                    record["status"] = "completed"
                    record["output"] = str(output)
                    break
            self._save()

    def inspect_interface(self, section: str) -> dict[str, Any]:
        section = str(section or "overview")
        if section == "overview":
            value = render_environment_interface(self.fixed_metadata, self.constants)
        elif section == "candidate":
            value = {
                "schema": candidate_schema(),
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
        return {
            "status": "ok",
            "contract_version": AGENT_TOOL_CONTRACT_VERSION,
            "section": section,
            "data": value,
        }

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
        start = max(0, int(start))
        limit = min(20, max(1, int(limit)))
        selected = indices[start : start + limit]
        samples = []
        from llm_design import _compact_diagnostic_inputs, _sample_input_diagnostic

        for index in selected:
            values, limitations = _sample_input_diagnostic(
                index,
                fields,
                obs_arrays=self.obs_arrays,
                constants_metadata=self.constants,
            )
            compact, _, _, _ = _compact_diagnostic_inputs(values, 32)
            trace = (self.fixed_metadata.get("selection") or [])[index]
            samples.append(
                {
                    "sample_ref": f"sample_{index}",
                    "fixed_index": trace.get("fixed_index", index),
                    "feature_input_data": compact,
                    "limitations": limitations,
                }
            )
        return {
            "status": "ok",
            "fields": fields,
            "feature_input_timing": "action-pre current-only",
            "matched_sample_count": len(indices),
            "page_start": start,
            "page_limit": limit,
            "next_start": start + len(selected) if start + len(selected) < len(indices) else None,
            "samples": samples,
        }

    def submit_candidate(
        self, candidate_json: str, parent_candidate_id: str | None
    ) -> dict[str, Any]:
        raw_hash = hashlib.sha256(str(candidate_json).encode("utf-8")).hexdigest()
        try:
            candidate, envelope = parse_candidate_json_envelope(str(candidate_json))
        except CandidateError as exc:
            submission_id = f"submission-{raw_hash[:12]}"
            path = self.directory / "submissions" / submission_id
            path.mkdir(parents=True, exist_ok=True)
            (path / "raw.txt").write_text(str(candidate_json), encoding="utf-8")
            report = {
                "status": "failed",
                "candidate_id": submission_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "candidate_created": False,
            }
            _write_json(path / "submission_report.json", report)
            return report
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
            for issue in self.state["candidates"][parent_candidate_id].get("issues", []):
                inherited.append(
                    {
                        **issue,
                        "status": "not_revalidated",
                        "source_candidate_id": parent_candidate_id,
                        "note": "A new candidate version has not yet completed the relevant check.",
                    }
                )
        record = {
            "candidate_id": candidate_id,
            "content_sha256": digest,
            "parent_candidate_id": parent_candidate_id,
            "candidate": candidate,
            "envelope": envelope,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_status": "static_passed" if staged["can_execute"] else "static_failed",
            "evaluation_status": "not_run",
            "issues": list(staged.get("errors") or []),
            "inherited_issue_context": inherited,
            "tests": {},
            "evaluations": {},
        }
        self.state["candidates"][candidate_id] = record
        self.state["candidate_order"].append(candidate_id)
        self.state["current_candidate_id"] = candidate_id
        candidate_dir = self.directory / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "candidate.json", candidate)
        (candidate_dir / "candidate.py").write_text(candidate["code"], encoding="utf-8")
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
        *,
        mode: str,
        sample_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        record = self._candidate_record(candidate_id)
        if mode not in {"small", "full"}:
            return {"status": "invalid_arguments", "error": "mode must be small or full", "candidate_id": candidate_id}
        if record["validation_status"] == "static_failed":
            return {
                "status": "prerequisite_failed",
                "candidate_id": candidate_id,
                "errors": record["issues"],
            }
        total = int(self.fixed_metadata["sample_count"])
        if mode == "full":
            indices = list(range(total))
        elif sample_indices:
            indices = sorted(set(int(value) for value in sample_indices))
            if not indices or indices[0] < 0 or indices[-1] >= total or len(indices) > 32:
                return {"status": "invalid_arguments", "candidate_id": candidate_id, "error": "small-test sample indices must be 1..32 valid fixed indices"}
        else:
            indices = sorted(set(np.linspace(0, total - 1, min(8, total), dtype=int).tolist()))
        key = _content_hash({"mode": mode, "indices": indices, "worker_contract": self.diagnostic_contract})
        cached = record["tests"].get(key)
        if cached is not None and cached.get("status") in {"passed", "failed"}:
            return {**cached["tool_result"], "cache_hit": True}
        obs = {name: np.asarray(value)[indices] for name, value in self.obs_arrays.items()}
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
                np.asarray(self.arrays["state"])[indices], extra, record["candidate"]
            )
            expected = feature_reward(extra, record["candidate"])
            consistency = bool(np.allclose(expected, worker_reward, rtol=0.0, atol=1e-12))
            report = {
                "status": "passed" if consistency else "failed",
                "candidate_id": candidate_id,
                "mode": mode,
                "sample_indices": indices,
                "sample_count": len(indices),
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
                        "features": _json_native(extra[position]),
                        "extra_reward": float(expected[position]),
                    }
                    for position, index in enumerate(indices[:8])
                ],
                "checks_not_run": [] if mode == "full" else ["remaining fixed samples", "formal Lipschitz evaluation"],
            }
        except CandidateExecutionError as exc:
            report = {
                "status": "failed",
                "candidate_id": candidate_id,
                "mode": mode,
                "sample_indices": indices,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_report": exc.report,
            }
        candidate_dir = self.directory / "candidates" / candidate_id
        report_path = candidate_dir / f"test_{mode}_{key[:12]}.json"
        _write_json(report_path, report)
        tool_result = {
            key_name: report.get(key_name)
            for key_name in (
                "status",
                "candidate_id",
                "mode",
                "sample_count",
                "numeric_diagnostics",
                "reward_consistency",
                "representative_outputs",
                "checks_not_run",
                "error_type",
                "error",
                "worker_report",
            )
            if key_name in report
        }
        tool_result["cache_hit"] = False
        record["tests"][key] = {
            "status": report["status"],
            "mode": mode,
            "report": str(report_path.relative_to(self.directory)),
            "tool_result": tool_result,
        }
        if report["status"] == "passed":
            record["validation_status"] = f"{mode}_passed"
            if mode == "full":
                record["issues"] = []
                record["inherited_issue_context"] = [
                    {**issue, "status": "resolved", "resolution_evidence": str(report_path.relative_to(self.directory))}
                    for issue in record.get("inherited_issue_context", [])
                ]
        else:
            record["validation_status"] = f"{mode}_failed"
            errors = (report.get("worker_report") or {}).get("errors") or []
            record["issues"] = errors or [{"code": report.get("error_type", "EXECUTION_ERROR"), "problem": report.get("error")}]
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
            return self._evaluation_tool_result(candidate_id, report, cache_hit=True)
        validation_result = self.test_candidate(candidate_id, mode="full")
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
        self._save()
        return self._evaluation_tool_result(candidate_id, report, cache_hit=False)

    def _evaluation_tool_result(
        self, candidate_id: str, report: dict[str, Any], *, cache_hit: bool
    ) -> dict[str, Any]:
        diagnostics, summary = _compact_evaluation_diagnostics(
            report.get("evaluation_diagnostics") or {}, 8
        )
        return {
            "status": "approved" if report.get("passed") else report.get("status", "failed"),
            "candidate_id": candidate_id,
            "passed": bool(report.get("passed")),
            "cache_hit": bool(cache_hit),
            "by_lambda": report.get("by_lambda"),
            "candidate_numeric_diagnostics": report.get("candidate_numeric_diagnostics"),
            "evaluation_diagnostics": diagnostics,
            "diagnostic_compaction": summary,
            "approved_artifact": self.state.get("approved_artifact") if report.get("passed") else None,
            "approval_is_host_determined": True,
        }

    def get_history(
        self, candidate_id: str | None, record_type: str
    ) -> dict[str, Any]:
        if record_type == "run":
            return {"status": "ok", "work_state": self.work_state(include_candidate=True)}
        if not candidate_id:
            return {"status": "invalid_arguments", "error": "candidate_id is required for this record type"}
        record = self._candidate_record(candidate_id)
        if record_type == "candidate":
            return {"status": "ok", "candidate_id": candidate_id, "candidate": record["candidate"], "parent_candidate_id": record.get("parent_candidate_id")}
        if record_type == "issues":
            return {"status": "ok", "candidate_id": candidate_id, "issues": record.get("issues", []), "inherited_issue_context": record.get("inherited_issue_context", [])}
        if record_type == "tests":
            return {"status": "ok", "candidate_id": candidate_id, "tests": record.get("tests", {})}
        if record_type == "evaluation":
            reports = []
            for path in record.get("evaluations", {}).values():
                report = json.loads((self.directory / path).read_text(encoding="utf-8"))
                compact, summary = _compact_evaluation_diagnostics(report.get("evaluation_diagnostics") or {}, 8)
                reports.append({"path_id": path, "status": report.get("status"), "passed": report.get("passed"), "by_lambda": report.get("by_lambda"), "evaluation_diagnostics": compact, "compaction": summary})
            return {"status": "ok", "candidate_id": candidate_id, "evaluations": reports}
        return {"status": "invalid_arguments", "error": "record_type must be run, candidate, issues, tests, or evaluation"}


class _WorkspaceTool(Tool):
    workspace: AgentWorkspace

    def __init__(self, workspace: AgentWorkspace):
        self.workspace = workspace
        super().__init__()


class InspectInterfaceTool(_WorkspaceTool):
    name = "inspect_interface"
    description = "Read one controlled section of the authoritative current-only input, candidate, or evaluation contract."
    inputs = {"section": {"type": "string", "description": "overview, fields, constants, candidate, or evaluation"}}
    output_type = "string"

    def forward(self, section: str) -> str:
        return self.workspace.invoke(self.name, {"section": section}, lambda: self.workspace.inspect_interface(section))


class QuerySamplesTool(_WorkspaceTool):
    name = "query_samples"
    description = "Read a bounded page of current-only fixed-sample fields. IDs and evaluation outcomes are diagnostics, never candidate inputs."
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
    description = "Submit one complete shared-feature candidate JSON. Every distinct content hash creates an immutable version; the host never repairs it."
    inputs = {
        "candidate_json": {"type": "string", "description": "complete JSON object matching the supplied candidate schema"},
        "parent_candidate_id": {"type": "string", "description": "optional candidate id this revision derives from", "nullable": True},
    }
    output_type = "string"

    def forward(self, candidate_json: str, parent_candidate_id: str | None = None) -> str:
        return self.workspace.invoke(self.name, {"candidate_json": candidate_json, "parent_candidate_id": parent_candidate_id}, lambda: self.workspace.submit_candidate(candidate_json, parent_candidate_id))


class TestCandidateTool(_WorkspaceTool):
    name = "test_candidate"
    description = "Run isolated candidate validation in small or full mode. Small success is not approval and does not run formal scoring."
    inputs = {
        "candidate_id": {"type": "string", "description": "immutable candidate id returned by submit_candidate"},
        "mode": {"type": "string", "description": "small or full"},
        "sample_indices": {"type": "array", "description": "optional fixed indices for small mode", "items": {"type": "integer"}, "nullable": True},
    }
    output_type = "string"

    def forward(self, candidate_id: str, mode: str, sample_indices: list[int] | None = None) -> str:
        return self.workspace.invoke(self.name, {"candidate_id": candidate_id, "mode": mode, "sample_indices": sample_indices}, lambda: self.workspace.test_candidate(candidate_id, mode=mode, sample_indices=sample_indices))


class FormalEvaluateTool(_WorkspaceTool):
    name = "formal_evaluate"
    description = "Fully validate and score one exact candidate on the unchanged fixed pair set and all lambdas. Only a host-approved result creates an artifact."
    inputs = {"candidate_id": {"type": "string", "description": "immutable candidate id to validate and score"}}
    output_type = "string"

    def forward(self, candidate_id: str) -> str:
        return self.workspace.invoke(self.name, {"candidate_id": candidate_id}, lambda: self.workspace.formal_evaluate(candidate_id))


class GetHistoryTool(_WorkspaceTool):
    name = "get_history"
    description = "Retrieve controlled run, candidate, issue, test, or evaluation history by immutable candidate id; arbitrary paths are not accepted."
    inputs = {
        "candidate_id": {"type": "string", "description": "candidate id, or null only for run history", "nullable": True},
        "record_type": {"type": "string", "description": "run, candidate, issues, tests, or evaluation"},
    }
    output_type = "string"

    def forward(self, record_type: str, candidate_id: str | None = None) -> str:
        return self.workspace.invoke(self.name, {"candidate_id": candidate_id, "record_type": record_type}, lambda: self.workspace.get_history(candidate_id, record_type))


def build_agent_tools(workspace: AgentWorkspace) -> list[Tool]:
    return [
        InspectInterfaceTool(workspace),
        QuerySamplesTool(workspace),
        SubmitCandidateTool(workspace),
        TestCandidateTool(workspace),
        FormalEvaluateTool(workspace),
        GetHistoryTool(workspace),
    ]


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
        completion = self._prepare_completion_kwargs(
            messages,
            stop_sequences=None,
            response_format=None,
            tools_to_call_from=tools_to_call_from,
            tool_choice="required",
        )
        serialized = json.dumps(completion, ensure_ascii=False, allow_nan=False)
        token_budget = estimate_token_budget(
            serialized,
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        prospective_call_number = self.budget.used + 1
        call_dir = self.workspace.directory / f"model_call_{prospective_call_number:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        _write_json(call_dir / "token_budget.json", token_budget)
        if not token_budget["fits_client_budget"]:
            raise AgentContextBudgetError("agent messages plus output reservation exceed the client context budget")
        call_number = self.budget.consume()
        self.workspace.state["model_calls_used"] = self.budget.used
        self.workspace._save()
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
        if actual is not None and not response_model_matches(self.provider, str(self.model_id), str(actual)):
            raise APIError(f"provider returned a different model: requested={self.model_id!r}, actual={actual!r}")
        content, inline_reasoning = adapter_separate_reasoning(
            model_adapter(str(self.model_id)), streamed.get("content")
        )
        reasoning = streamed.get("reasoning")
        if inline_reasoning:
            reasoning = inline_reasoning if not reasoning else f"{reasoning}\n{inline_reasoning}"
        tool_calls = []
        for index, raw in enumerate(streamed.get("tool_calls") or []):
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
        usage = streamed.get("usage") or {}
        token_usage = None
        if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
            token_usage = TokenUsage(
                input_tokens=int(usage["prompt_tokens"]),
                output_tokens=int(usage["completion_tokens"]),
            )
        raw_record = {
            "actual_model": actual,
            "finish_reason": streamed.get("finish_reason"),
            "content": content,
            "reasoning": reasoning,
            "tool_calls": streamed.get("tool_calls") or [],
            "usage": usage,
        }
        _write_json(call_dir / "response.json", raw_record)
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
        return self.delegate.generate(*args, **kwargs)


class ControlledToolCallingAgent(ToolCallingAgent):
    """Framework agent with sequential tools, bounded calls, and host approval stop."""

    def __init__(self, *args, workspace: AgentWorkspace, context_length: int, max_output_tokens: int, **kwargs):
        self.workspace = workspace
        self.context_length = int(context_length)
        self.max_output_tokens = int(max_output_tokens)
        super().__init__(*args, max_tool_threads=1, planning_interval=None, **kwargs)

    def process_tool_calls(self, chat_message, memory_step):
        self.workspace.register_framework_calls(chat_message.tool_calls or [])
        for output in super().process_tool_calls(chat_message, memory_step):
            if isinstance(output, ToolOutput):
                self.workspace.complete_framework_call(output.id, output.output)
                if self.workspace.approved:
                    output.is_final_answer = True
                    output.output = json.dumps(
                        {
                            "status": "approved",
                            "candidate_id": self.workspace.state["approved_candidate_id"],
                            "approved_artifact": self.workspace.state["approved_artifact"],
                        }
                    )
            yield output

    def write_memory_to_messages(self, summary_mode: bool = False):
        messages = super().write_memory_to_messages(summary_mode=summary_mode)
        serialized = json.dumps(
            [message.dict() if hasattr(message, "dict") else str(message) for message in messages],
            default=str,
            ensure_ascii=False,
        )
        budget = estimate_token_budget(
            serialized,
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        if budget["fits_client_budget"]:
            return messages
        # Keep whole memory-step message groups, never split tool calls from results.
        system_messages = self.memory.system_prompt.to_messages(summary_mode=False)
        task_messages = self.memory.steps[0].to_messages(summary_mode=False) if self.memory.steps else []
        groups = [step.to_messages(summary_mode=False) for step in self.memory.steps[1:]]
        summary_message = ChatMessage(
            role=MessageRole.USER,
            content=(
                "Host-generated compact work-state summary. Structured facts only; "
                "older full messages remain on disk:\n"
                + json.dumps(self.workspace.work_state(include_candidate=True), ensure_ascii=False, allow_nan=False)
            ),
        )
        kept = list(groups)
        while kept:
            candidate = system_messages + task_messages + [summary_message] + [message for group in kept for message in group]
            estimate = estimate_token_budget(
                json.dumps([str(message) for message in candidate], ensure_ascii=False),
                context_length=self.context_length,
                max_output_tokens=self.max_output_tokens,
            )
            if estimate["fits_client_budget"]:
                self.workspace.state["context_compactions"].append(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "omitted_complete_step_groups": len(groups) - len(kept),
                        "retained_complete_step_groups": len(kept),
                        "candidate_code_truncated": False,
                        "tool_call_result_pairs_split": False,
                        "token_budget": estimate,
                    }
                )
                self.workspace._save()
                return candidate
            kept.pop(0)
        minimum = system_messages + task_messages + [summary_message]
        estimate = estimate_token_budget(
            json.dumps([str(message) for message in minimum], ensure_ascii=False),
            context_length=self.context_length,
            max_output_tokens=self.max_output_tokens,
        )
        if not estimate["fits_client_budget"]:
            raise AgentContextBudgetError(
                "system/task prompt, complete current candidate, unresolved state, and output reservation do not fit"
            )
        return minimum

    def _handle_max_steps_reached(self, task: str) -> Any:
        # smolagents normally makes one extra model call here.  Returning a
        # deterministic host result keeps --max-model-calls an exact bound.
        self.workspace.state["stop_reason"] = "model call/step budget exhausted without host approval"
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
    model_backend: Model | None = None,
    client: LMStudioClient | None = None,
) -> dict[str, Any]:
    resume_state = None
    if resume is not None:
        directory = Path(resume).resolve()
        resume_state = _load_resume_state(directory)
        fixed_sample = resume_state["fixed_sample"]["directory"]
        provider = resume_state["provider"]
        model = resume_state["model"]
        settings = resume_state["settings"]
        beta = settings["beta"]
        batch_size = settings["batch_size"]
        worker_timeout = settings["worker_timeout"]
        absolute_tolerance = settings["absolute_tolerance"]
        relative_tolerance = settings["relative_tolerance"]
        max_model_calls = settings["max_model_calls"]
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
    budget = ModelCallBudget(max_model_calls, workspace.state["model_calls_used"])
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
    task = _render_agent_prompt(
        fixed_metadata=workspace.fixed_metadata,
        baseline=workspace.baseline,
        constants_metadata=workspace.constants,
        beta=workspace.beta,
        absolute_tolerance=workspace.absolute_tolerance,
        relative_tolerance=workspace.relative_tolerance,
        current_work_state=workspace.work_state(include_candidate=True),
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
        max_steps=max(1, available_model_calls),
        add_base_tools=False,
        stream_outputs=False,
    )
    combined_prompt = agent.system_prompt + "\n\n" + task
    prompt_budget = estimate_token_budget(
        combined_prompt,
        context_length=effective_context,
        max_output_tokens=max_output_tokens,
    )
    (directory / "agent_task_prompt.txt").write_text(task, encoding="utf-8")
    (directory / "framework_system_prompt.txt").write_text(agent.system_prompt, encoding="utf-8")
    _write_json(
        directory / "tool_schemas.json",
        [get_tool_json_schema(tool) for tool in tools],
    )
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
            "max_model_calls": max_model_calls,
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
        "git_sha": _git_sha(),
    }
    if resume is None:
        metadata_path = directory / "run_metadata.json"
    else:
        resume_number = len(workspace.state.get("resume_events", [])) + 1
        metadata_path = directory / f"resume_metadata_{resume_number:03d}.json"
    _write_json(metadata_path, metadata)
    if not prompt_budget["fits_client_budget"]:
        workspace.state["status"] = "failed_context_budget"
        workspace.state["stop_reason"] = "initial framework system/task/tool prompt exceeds context budget"
        workspace._save()
        raise AgentContextBudgetError(workspace.state["stop_reason"])
    if dry_run:
        workspace.state["status"] = "dry_run_complete"
        workspace.state["stop_reason"] = "dry run; no generation request sent"
        workspace._save()
        return {"status": "dry_run_complete", "output_directory": str(directory), "metadata": metadata, "model_calls_used": budget.used}
    if resume is not None:
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
        workspace.state["status"] = "stopped_without_approval"
        workspace.state["stop_reason"] = (
            "saved model call budget is already exhausted without host approval"
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
        (directory / "framework_final_output.txt").write_text(str(result.output), encoding="utf-8")
        _write_json(directory / "framework_steps.json", result.steps)
        if workspace.approved:
            workspace.state["status"] = "approved"
        elif workspace.state["status"] == "running":
            workspace.state["status"] = "stopped_without_approval"
            workspace.state["stop_reason"] = workspace.state.get("stop_reason") or "framework final answer without host approval"
    except KeyboardInterrupt:
        workspace.state["status"] = "interrupted"
        workspace.state["stop_reason"] = "cancelled by user; no unknown request is retried"
    except Exception as exc:
        workspace.state["status"] = "failed"
        workspace.state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        _write_json(directory / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        workspace.state["model_calls_used"] = budget.used
        workspace._save()
    return {
        "status": workspace.state["status"],
        "output_directory": str(directory),
        "approved_artifact": workspace.state.get("approved_artifact"),
        "model_calls_used": budget.used,
        "tool_operations": len(workspace.state["tool_operations"]),
        "stop_reason": workspace.state.get("stop_reason"),
        "metadata": metadata,
    }
