"""Two-model offline feature design with host validation and model review."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from llm_candidate import (
    CandidateError,
    CandidateExecutionError,
    ModelCandidateSchemaError,
    candidate_numeric_diagnostics,
    candidate_semantic_fingerprint,
    execute_candidate_isolated,
    feature_reward,
    normalize_candidate_submission,
    parse_candidate_json_envelope,
    save_approved_artifact,
    validate_candidate,
    validate_candidate_staged,
)
from llm_baseline import load_fixed_samples
from llm_design import (
    APIError,
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    LMStudioClient,
    OpenAIClient,
    _effective_context_budget,
    _git_sha,
    _slug,
    _write_json,
    allocate_design_directory,
    model_inventory_summary,
    planned_chat_request,
    provider_base_url,
    response_model_matches,
)
from llm_design_contract import (
    OBS_INTERFACE_VERSION,
    SUPPORTED_OPERATIONS,
    build_obs_arrays,
    build_constants,
    estimate_token_budget,
    minimal_model_candidate_example,
    movement_and_energy_spec,
    render_environment_interface,
    runtime_diagnostic_contract,
    visual_sensing_spec,
    communication_spec,
)


REVIEW_RUN_SCHEMA_VERSION = "uav-hrl-llm-review-design-run-v2"
LEGACY_REVIEW_RUN_SCHEMA_VERSION = "uav-hrl-llm-review-design-run-v1"
LEGACY_REVIEW_COMPATIBLE_GIT_REVISIONS = {
    "a7aa991078f63e4ee039fed1e05709d10bf46595",
}
REVIEW_PROMPT_VERSION = "uav-hrl-llm-review-design-prompt-v1"
REVIEW_APPROVAL_VERSION = "uav-hrl-model-review-approval-v1"
DEFAULT_REVIEW_OUTPUT_ROOT = Path("results") / "llm_review_designs"
DEFAULT_MAX_REVIEW_ROUNDS = 5
DEFAULT_MAX_CODE_REPAIRS = 5
ROOT = Path(__file__).resolve().parent
COMMON_TEMPLATE = ROOT / "prompts" / "llm_review_common.txt"
PROPOSER_TEMPLATE = ROOT / "prompts" / "llm_review_proposer.txt"
REVIEWER_TEMPLATE = ROOT / "prompts" / "llm_review_reviewer.txt"


class ReviewDesignError(RuntimeError):
    pass


class CandidateVersionMismatch(ReviewDesignError):
    pass


def _load_review_inputs(fixed_directory: str | Path):
    """Load fixed observations without loading or evaluating the baseline report."""

    arrays, fixed_metadata = load_fixed_samples(Path(fixed_directory).resolve())
    return arrays, fixed_metadata, build_constants(fixed_metadata)


def _read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_TEMPLATE_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def _replace_all(template: str, replacements: dict[str, str]) -> str:
    """Replace placeholders found in the template, never in inserted values."""

    supplied = {}
    for key, value in replacements.items():
        name = key[2:-2] if key.startswith("{{") and key.endswith("}}") else key
        supplied[name] = str(value)
    required = list(dict.fromkeys(_TEMPLATE_PLACEHOLDER.findall(template)))
    missing = [name for name in required if name not in supplied]
    if missing:
        raise ReviewDesignError(
            "review-design prompt is missing replacements for: "
            + ", ".join(missing)
        )
    return _TEMPLATE_PLACEHOLDER.sub(
        lambda match: supplied[match.group(1)],
        template,
    )


def candidate_full_content_sha256(candidate: dict[str, Any]) -> str:
    """Hash every candidate JSON value using one stable serialization."""

    encoded = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_common_context(
    *, fixed_metadata: dict[str, Any], constants_metadata: dict[str, Any], beta: float
) -> str:
    return _replace_all(
        _read_template(COMMON_TEMPLATE),
        {
            "{{MOVEMENT_AND_ENERGY_SPEC}}": movement_and_energy_spec(constants_metadata),
            "{{VISUAL_SENSING_SPEC}}": visual_sensing_spec(constants_metadata),
            "{{COMMUNICATION_SPEC}}": communication_spec(constants_metadata),
            "{{INPUT_FIELD_TABLE}}": render_environment_interface(
                fixed_metadata,
                constants_metadata,
                include_system_semantics=False,
                include_constants=True,
            ),
            "{{BETA}}": format(float(beta), ".17g"),
            "{{SUPPORTED_OPERATIONS}}": SUPPORTED_OPERATIONS,
        },
    )


def render_proposer_prompt(*, common_context: str, round_context: str) -> str:
    return _replace_all(
        _read_template(PROPOSER_TEMPLATE),
        {
            "{{COMMON_CONTEXT}}": common_context,
            "{{MINIMAL_EXECUTABLE_EXAMPLE}}": json.dumps(
                minimal_model_candidate_example(),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "{{ROUND_CONTEXT}}": round_context,
        },
    )


def render_reviewer_prompt(
    *,
    common_context: str,
    candidate: dict[str, Any],
    validation_report: dict[str, Any],
    review_history: list[dict[str, Any]],
) -> str:
    compact_history = [
        {
            "review_round": item.get("review_round"),
            "candidate_id": item.get("candidate_id"),
            "review": item.get("review"),
        }
        for item in review_history[-3:]
    ]
    return _replace_all(
        _read_template(REVIEWER_TEMPLATE),
        {
            "{{COMMON_CONTEXT}}": common_context,
            "{{CANDIDATE}}": json.dumps(
                candidate, indent=2, ensure_ascii=False, allow_nan=False
            ),
            "{{VALIDATION_REPORT}}": json.dumps(
                _validation_for_prompt(validation_report),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "{{REVIEW_HISTORY}}": (
                json.dumps(
                    compact_history, indent=2, ensure_ascii=False, allow_nan=False
                )
                if compact_history
                else "No previous model reviews."
            ),
        },
    )


def _validation_for_prompt(report: dict[str, Any]) -> dict[str, Any]:
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    scope = copy.deepcopy(report.get("scope"))
    if isinstance(scope, dict):
        scope.pop("lipschitz_evaluation_performed", None)
    return {
        "status": report.get("status"),
        "scope": scope,
        "checks": report.get("checks"),
        "error_count": len(errors),
        "errors": errors[:20],
        "omitted_error_count": max(0, len(errors) - 20),
        "candidate_numeric_diagnostics": report.get("candidate_numeric_diagnostics"),
        "shared_feature_reward_consistency": report.get(
            "shared_feature_reward_consistency"
        ),
        "empirical_limit": (
            "Passing applies to the complete saved fixed sample set and probes; it is "
            "not a mathematical guarantee for every possible environment state."
        ),
    }


def _proposer_round_context(state: dict[str, Any]) -> str:
    if state.get("current_submission") is None and state.get("pending_review") is None:
        return (
            "Generate the initial complete candidate. There is no previous candidate, "
            "program-validation feedback, or model review."
        )
    blocks = [
        f"Target proposal cycle: {state['proposal_cycle']}. Candidate version under correction: "
        f"{state.get('current_candidate_id') or 'unparsed submission'}."
    ]
    if state.get("current_submission") is not None:
        blocks.extend(
            (
                "\nComplete current proposer output:\n"
                + (
                    json.dumps(
                        state["current_submission"],
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    if not isinstance(state["current_submission"], str)
                    else state["current_submission"]
                ),
                "\nProgram-validation feedback:\n"
                + json.dumps(
                    _validation_for_prompt(state.get("current_validation") or {}),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
        )
    if state.get("pending_review") is not None:
        blocks.append(
            "\nUnresolved reviewer suggestions for the preceding validated candidate. "
            "These remain requirements while repairing code:\n"
            + json.dumps(
                state["pending_review"],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    blocks.append(
        "Return a complete replacement candidate. Do not return a patch or omit unchanged features."
    )
    return "\n".join(blocks)


def _strict_review(content: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, metadata = parse_candidate_json_envelope(content)
    if not isinstance(value, dict):
        raise CandidateError("review response must be one JSON object")
    expected = {"needs_revision", "summary", "suggestions"}
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise CandidateError(
            f"review fields are incompatible; missing={missing}, extra={extra}"
        )
    if not isinstance(value["needs_revision"], bool):
        raise CandidateError("review needs_revision must be boolean")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise CandidateError("review summary must be non-empty text")
    suggestions = value["suggestions"]
    if not isinstance(suggestions, list):
        raise CandidateError("review suggestions must be an array")
    required = {"target", "reason", "suggested_change"}
    for index, item in enumerate(suggestions):
        if not isinstance(item, dict) or set(item) != required:
            raise CandidateError(
                f"review suggestions[{index}] must contain exactly {sorted(required)}"
            )
        for name in required:
            if not isinstance(item[name], str) or not item[name].strip():
                raise CandidateError(
                    f"review suggestions[{index}].{name} must be non-empty text"
                )
    if value["needs_revision"] and not suggestions:
        raise CandidateError("needs_revision=true requires actionable suggestions")
    if not value["needs_revision"] and suggestions:
        raise CandidateError("needs_revision=false requires an empty suggestions list")
    return value, metadata


def _candidate_id(candidate: dict[str, Any]) -> str:
    return "candidate-" + candidate_semantic_fingerprint(candidate)[:12]


def _candidate_id_matches_full_content(
    candidate_id: str,
    candidate: dict[str, Any],
    full_content_sha256: str,
) -> bool:
    base = _candidate_id(candidate)
    return candidate_id in {base, f"{base}-{full_content_sha256[:12]}"}


def _failure_report(stage: str, exc: BaseException) -> dict[str, Any]:
    issues = copy.deepcopy(getattr(exc, "issues", None) or [])
    if not issues and isinstance(exc, CandidateExecutionError):
        worker = exc.report or {}
        issues = copy.deepcopy(worker.get("errors") or [])
    if not issues:
        issues = [
            {
                "code": f"{stage.upper()}_ERROR",
                "stage": stage,
                "location": "$" if stage != "execution" else "compute_extra_state",
                "problem": str(exc),
                "requirement": "Correct this issue and return the complete candidate JSON.",
            }
        ]
    checks = {
        "json": {"status": "not_run", "completed": False},
        "schema": {"status": "not_run", "completed": False},
        "static": {"status": "not_run", "completed": False},
        "execution": {"status": "not_run", "completed": False},
    }
    order = ("json", "schema", "static", "execution")
    if stage in order:
        index = order.index(stage)
        for preceding in order[:index]:
            checks[preceding] = {"status": "passed", "completed": True}
        checks[stage] = {"status": "failed", "completed": True}
    return {"status": "failed", "can_execute": False, "errors": issues, "checks": checks}


def validate_submission(
    content: str,
    *,
    arrays: dict[str, np.ndarray],
    obs_arrays: dict[str, np.ndarray],
    constants_metadata: dict[str, Any],
    diagnostic_contract: dict[str, Any],
    worker_timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    details: dict[str, Any] = {"raw_content": content}
    try:
        submitted, parse_metadata = parse_candidate_json_envelope(content)
        details["parse_metadata"] = parse_metadata
    except CandidateError as exc:
        return None, _failure_report("json", exc), details
    details["submitted_candidate"] = submitted
    try:
        model_candidate, candidate, transformation = normalize_candidate_submission(
            submitted, constants_metadata
        )
        details.update(
            {
                "model_candidate": model_candidate,
                "candidate": candidate,
                "host_transformation": transformation,
            }
        )
    except ModelCandidateSchemaError as exc:
        return None, _failure_report("schema", exc), details
    staged = validate_candidate_staged(candidate, constants_metadata)
    if not staged["can_execute"]:
        return candidate, staged, details
    try:
        static = validate_candidate(candidate, constants_metadata)
        extra, worker_reward, execution = execute_candidate_isolated(
            candidate,
            obs_arrays,
            constants_metadata,
            timeout=worker_timeout,
            diagnostic_contract=diagnostic_contract,
        )
    except (CandidateError, CandidateExecutionError) as exc:
        report = _failure_report("execution", exc)
        report["checks"]["static"] = {"status": "passed", "completed": True}
        return candidate, report, details
    try:
        expected_reward = feature_reward(extra, candidate)
    except CandidateError as exc:
        report = _failure_report("execution", exc)
        report["checks"]["static"] = {"status": "passed", "completed": True}
        return candidate, report, details
    maximum_error = float(np.max(np.abs(expected_reward - worker_reward)))
    consistency = {
        "maximum_absolute_difference": maximum_error,
        "passed": bool(np.allclose(expected_reward, worker_reward, rtol=0.0, atol=1e-12)),
    }
    if not consistency["passed"]:
        report = _failure_report(
            "execution", CandidateError("worker reward differs from host weighted reward")
        )
        return candidate, report, details
    report = {
        "status": "passed",
        "can_execute": True,
        "errors": [],
        "scope": {
            "fixed_sample_count": int(extra.shape[0]),
            "uses_complete_fixed_sample_set": True,
            "includes_empty_probe": True,
            "lipschitz_evaluation_performed": False,
        },
        "checks": {
            **staged["checks"],
            "execution": {"status": "passed", "completed": True},
        },
        "static": static,
        "execution": execution,
        "candidate_numeric_diagnostics": candidate_numeric_diagnostics(
            arrays["state"], extra, candidate
        ),
        "shared_feature_reward_consistency": consistency,
    }
    details["extra_state"] = extra
    details["extra_reward"] = worker_reward
    return candidate, report, details


def _role_config(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    context_length: int,
    max_output_tokens: int,
    temperature: float,
    seed: int,
    reasoning_effort: str | None,
    timeout: float,
    connect_timeout: float,
    total_timeout: float,
    progress_interval: float,
) -> dict[str, Any]:
    normalized_provider = str(provider).lower()
    if normalized_provider not in {"lmstudio", "openai"}:
        raise ValueError(f"unsupported provider: {provider!r}")
    if not str(model).strip():
        raise ValueError("model must be non-empty")
    if int(context_length) <= 0 or int(max_output_tokens) <= 0:
        raise ValueError("context length and max output tokens must be positive")
    if int(max_output_tokens) >= int(context_length):
        raise ValueError("max output tokens must be smaller than context length")
    if not np.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    for name, value in (
        ("timeout", timeout),
        ("connect_timeout", connect_timeout),
        ("total_timeout", total_timeout),
        ("progress_interval", progress_interval),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return {
        "provider": normalized_provider,
        "model": str(model),
        "base_url": provider_base_url(provider, base_url),
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


def _make_client(config: dict[str, Any]):
    cls = OpenAIClient if config["provider"] == "openai" else LMStudioClient
    return cls(
        config["base_url"],
        timeout=config["timeout"],
        connect_timeout=config["connect_timeout"],
        total_timeout=config["total_timeout"],
        progress_interval=config["progress_interval"],
    )


def _role_inventory(config: dict[str, Any], client: Any, dry_run: bool):
    if dry_run or config["provider"] == "openai":
        return None
    return model_inventory_summary(client.list_models(), config["model"])


def _call_role(
    *,
    role: str,
    prompt: str,
    config: dict[str, Any],
    client: Any,
    output: Path,
    state: dict[str, Any],
    effective_context: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    counter_name = f"{role}_calls"
    call_number = int(state["counters"][counter_name]) + 1
    directory = output / f"{role}_call_{call_number:03d}"
    directory.mkdir(exist_ok=True)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    budget = estimate_token_budget(
        prompt,
        context_length=effective_context,
        max_output_tokens=config["max_output_tokens"],
    )
    _write_json(directory / "token_budget.json", budget)
    _write_json(
        directory / "planned_request.json",
        planned_chat_request(
            config["provider"],
            model=config["model"],
            prompt=prompt,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            seed=config["seed"],
            reasoning_effort=config["reasoning_effort"],
        ),
    )
    if not budget["fits_client_budget"]:
        return None, {
            "status": "paused_context_budget",
            "role": role,
            "budget": budget,
            "directory": str(directory),
        }
    state["counters"][counter_name] = call_number
    state["request_in_flight"] = {"role": role, "call": call_number}
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "state.json", state)
    try:
        response = client.chat(
            model=config["model"],
            prompt=prompt,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            seed=config["seed"],
            attempt_directory=directory,
            reasoning_effort=config["reasoning_effort"],
        )
    except (APIError, OSError, RuntimeError) as exc:
        state["request_in_flight"] = None
        return None, {
            "status": "paused_transport_failure",
            "role": role,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "directory": str(directory),
        }
    state["request_in_flight"] = None
    _write_json(directory / "response.json", response)
    if not response.get("transport_completed"):
        return None, {"status": "paused_incomplete_response", "role": role}
    if response.get("finish_reason") == "length":
        return None, {"status": "paused_truncated_response", "role": role}
    if response.get("finish_reason") != "stop" or response.get("tool_calls_seen"):
        return None, {
            "status": "paused_incomplete_response",
            "role": role,
            "finish_reason": response.get("finish_reason"),
        }
    actual_model = response.get("actual_model")
    if actual_model and not response_model_matches(
        config["provider"], config["model"], str(actual_model)
    ):
        return None, {
            "status": "paused_model_mismatch",
            "role": role,
            "requested": config["model"],
            "actual": actual_model,
        }
    content = response.get("content")
    if content is None or not str(content).strip():
        return None, {"status": "paused_missing_final_content", "role": role}
    return response, {"status": "complete", "role": role, "directory": str(directory)}


def _new_state(
    *,
    output: Path,
    fixed_sample: Path,
    fixed_metadata: dict[str, Any],
    proposer: dict[str, Any],
    reviewer: dict[str, Any],
    beta: float,
    worker_timeout: float,
    max_review_rounds: int,
    max_code_repairs: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": REVIEW_RUN_SCHEMA_VERSION,
        "status": "running",
        "phase": "proposer_needed",
        "created_at_utc": now,
        "updated_at_utc": now,
        "git_sha": _git_sha(),
        "prompt_version": REVIEW_PROMPT_VERSION,
        "output_directory": str(output),
        "fixed_sample": {
            "directory": str(fixed_sample.resolve()),
            "sample_content_sha256": fixed_metadata["sample_content_sha256"],
            "sample_count": int(fixed_metadata["sample_count"]),
        },
        "settings": {
            "proposer": proposer,
            "reviewer": reviewer,
            "beta": float(beta),
            "worker_timeout": float(worker_timeout),
            "max_review_rounds": int(max_review_rounds),
            "max_code_repairs": int(max_code_repairs),
        },
        "counters": {
            "proposer_calls": 0,
            "reviewer_calls": 0,
            "review_rounds_completed": 0,
            "proposal_outputs_in_cycle": 0,
            "code_repairs_used_in_cycle": 0,
        },
        "proposal_cycle": 1,
        "current_candidate_id": None,
        "current_candidate_full_content_sha256": None,
        "current_submission": None,
        "current_validation": None,
        "revision_parent_candidate_id": None,
        "pending_review": None,
        "review_history": [],
        "candidate_order": [],
        "candidates": {},
        "request_in_flight": None,
        "interrupted_requests": [],
        "approved_artifact": None,
        "stop_reason": None,
        "lipschitz_evaluation_performed": False,
    }


def _read_candidate_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateVersionMismatch(
            f"candidate file cannot be read as JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CandidateVersionMismatch(f"candidate file is not a JSON object: {path}")
    return value


def _bind_validation_hashes(
    report: dict[str, Any],
    *,
    candidate_id: str,
    semantic_fingerprint: str,
    full_content_sha256: str,
    legacy: bool,
) -> None:
    if report.get("candidate_id") not in (None, candidate_id):
        raise CandidateVersionMismatch(
            f"validation report candidate ID differs from {candidate_id}"
        )
    if legacy:
        legacy_value = report.get("candidate_content_sha256")
        if legacy_value not in (None, semantic_fingerprint):
            raise CandidateVersionMismatch(
                "legacy validation semantic fingerprint does not match the saved candidate"
            )
        if legacy_value is not None:
            report["legacy_candidate_semantic_fingerprint"] = legacy_value
        report.pop("candidate_content_sha256", None)
    report["candidate_id"] = candidate_id
    report["candidate_semantic_fingerprint"] = semantic_fingerprint
    report["candidate_full_content_sha256"] = full_content_sha256


def _migrate_legacy_state(
    output: Path,
    state: dict[str, Any],
    constants_metadata: dict[str, Any],
    *,
    persist: bool,
) -> None:
    if state.pop("_legacy_schema_version", None) is None:
        return
    source_git_sha = state.get("git_sha")
    updated_reports: list[tuple[Path, dict[str, Any]]] = []
    for candidate_id in state.get("candidate_order", []):
        record = state.get("candidates", {}).get(candidate_id)
        if not isinstance(record, dict):
            raise ReviewDesignError(
                f"legacy run cannot be safely migrated: missing record for {candidate_id}"
            )
        attempts = record.get("submission_attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ReviewDesignError(
                f"legacy run cannot be safely migrated: {candidate_id} has no saved submission attempt"
            )
        proposer_call = attempts[-1].get("proposer_call")
        if not isinstance(proposer_call, int) or proposer_call <= 0:
            raise ReviewDesignError(
                "legacy run cannot be safely migrated: "
                f"{candidate_id} has no valid saved proposer call"
            )
        submitted_path = (
            output
            / f"proposer_call_{int(proposer_call):03d}"
            / "submitted_candidate.json"
        )
        if not submitted_path.is_file():
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because the original submitted "
                f"candidate is missing: {submitted_path}"
            )
        submitted = _read_candidate_json(submitted_path)
        try:
            reconstructed = normalize_candidate_submission(
                submitted, constants_metadata
            )[1]
        except CandidateError as exc:
            raise ReviewDesignError(
                f"legacy run candidate cannot be reconstructed safely: {candidate_id}: {exc}"
            ) from exc
        candidate_path = output / "candidates" / candidate_id / "candidate.json"
        saved = _read_candidate_json(candidate_path)
        reconstructed_hash = candidate_full_content_sha256(reconstructed)
        saved_hash = candidate_full_content_sha256(saved)
        if saved_hash != reconstructed_hash:
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because candidate.json differs "
                f"from the original submitted candidate: {candidate_path}"
            )
        semantic = candidate_semantic_fingerprint(saved)
        legacy_semantic = record.get("content_sha256")
        if legacy_semantic not in (None, semantic):
            raise ReviewDesignError(
                f"legacy run semantic fingerprint is inconsistent for {candidate_id}"
            )
        if _candidate_id(saved) != candidate_id:
            raise ReviewDesignError(
                f"legacy run candidate ID is inconsistent for {candidate_path}"
            )
        code_path = candidate_path.with_name("candidate.py")
        if (
            not code_path.is_file()
            or code_path.read_text(encoding="utf-8") != saved["code"]
        ):
            raise ReviewDesignError(
                f"legacy run candidate.py differs from candidate.json: {code_path}"
            )
        record["legacy_semantic_content_sha256"] = legacy_semantic
        record.pop("content_sha256", None)
        record["semantic_fingerprint"] = semantic
        record["full_content_sha256"] = saved_hash
        validation_path = candidate_path.with_name("validation_report.json")
        if not validation_path.is_file():
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because its candidate validation "
                f"report is missing: {validation_path}"
            )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        _bind_validation_hashes(
            validation,
            candidate_id=candidate_id,
            semantic_fingerprint=semantic,
            full_content_sha256=saved_hash,
            legacy=True,
        )
        updated_reports.append((validation_path, validation))
        call_validation_path = submitted_path.with_name("validation_report.json")
        if not call_validation_path.is_file():
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because its proposer-call validation "
                f"report is missing: {call_validation_path}"
            )
        call_validation = json.loads(
            call_validation_path.read_text(encoding="utf-8")
        )
        _bind_validation_hashes(
            call_validation,
            candidate_id=candidate_id,
            semantic_fingerprint=semantic,
            full_content_sha256=saved_hash,
            legacy=True,
        )
        updated_reports.append((call_validation_path, call_validation))

    current_id = state.get("current_candidate_id")
    if current_id is not None:
        record = state["candidates"].get(current_id)
        if not isinstance(record, dict):
            raise ReviewDesignError(
                "legacy run cannot be safely migrated: current candidate record is missing"
            )
        current_validation = state.get("current_validation")
        if not isinstance(current_validation, dict):
            raise ReviewDesignError(
                "legacy run cannot be safely migrated: current validation report is missing"
            )
        _bind_validation_hashes(
            current_validation,
            candidate_id=current_id,
            semantic_fingerprint=record["semantic_fingerprint"],
            full_content_sha256=record["full_content_sha256"],
            legacy=True,
        )
        state["current_candidate_full_content_sha256"] = record[
            "full_content_sha256"
        ]
    else:
        state["current_candidate_full_content_sha256"] = None
    for review in state.get("review_history", []):
        review_candidate_id = review.get("candidate_id")
        review_record = state.get("candidates", {}).get(review_candidate_id)
        reviewer_call = review.get("reviewer_call")
        if not isinstance(review_record, dict) or not isinstance(reviewer_call, int):
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because a review lacks candidate provenance"
            )
        review_candidate_path = (
            output / "candidates" / review_candidate_id / "candidate.json"
        )
        review_candidate = _read_candidate_json(review_candidate_path)
        prompt_path = output / f"reviewer_call_{reviewer_call:03d}" / "prompt.txt"
        if not prompt_path.is_file():
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because its saved reviewer prompt "
                f"is missing: {prompt_path}"
            )
        candidate_text = json.dumps(
            review_candidate, indent=2, ensure_ascii=False, allow_nan=False
        )
        if candidate_text not in prompt_path.read_text(encoding="utf-8"):
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because the reviewer prompt does "
                f"not contain the saved candidate: {prompt_path}"
            )
        review["candidate_binding_status"] = "legacy_reviewer_prompt_verified"
        review["candidate_full_content_sha256"] = review_record[
            "full_content_sha256"
        ]
    if state.get("status") == "approved":
        approved_path = Path(state.get("approved_artifact") or output / "approved")
        approved_candidate_path = approved_path / "candidate.json"
        approved_candidate = _read_candidate_json(approved_candidate_path)
        if (
            current_id is None
            or candidate_full_content_sha256(approved_candidate)
            != state["candidates"][current_id]["full_content_sha256"]
        ):
            raise ReviewDesignError(
                "legacy run cannot be safely migrated because its approved artifact does "
                "not match the validated current candidate"
            )
    state["schema_version"] = REVIEW_RUN_SCHEMA_VERSION
    state["git_sha"] = _git_sha()
    state["migration"] = {
        "from_schema_version": LEGACY_REVIEW_RUN_SCHEMA_VERSION,
        "to_schema_version": REVIEW_RUN_SCHEMA_VERSION,
        "basis": "candidate reconstructed from the saved proposer submission and matched to candidate.json",
        "from_git_sha": source_git_sha,
        "to_git_sha": state["git_sha"],
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if persist:
        for path, report in updated_reports:
            _write_json(path, report)
        _write_json(output / "state.json", state)


def _verified_candidate_for_review(
    output: Path,
    state: dict[str, Any],
    candidate_id: str,
) -> tuple[dict[str, Any], str]:
    record = state.get("candidates", {}).get(candidate_id)
    validation = state.get("current_validation")
    expected = state.get("current_candidate_full_content_sha256")
    if not isinstance(record, dict) or not isinstance(validation, dict) or not expected:
        raise CandidateVersionMismatch(
            "candidate state lacks the full-content binding required for review"
        )
    record_hash = record.get("full_content_sha256")
    validation_hash = validation.get("candidate_full_content_sha256")
    if record_hash != expected or validation_hash != expected:
        raise CandidateVersionMismatch(
            "candidate state, candidate record, and validation report have different full-content hashes"
        )
    if validation.get("candidate_id") != candidate_id:
        raise CandidateVersionMismatch(
            "validation report is bound to a different candidate ID"
        )
    candidate_path = output / "candidates" / candidate_id / "candidate.json"
    candidate = _read_candidate_json(candidate_path)
    actual_hash = candidate_full_content_sha256(candidate)
    if actual_hash != expected:
        raise CandidateVersionMismatch(
            f"candidate file changed after validation: {candidate_path}; "
            f"expected {expected}, found {actual_hash}"
        )
    semantic = candidate_semantic_fingerprint(candidate)
    if semantic != record.get(
        "semantic_fingerprint"
    ) or not _candidate_id_matches_full_content(candidate_id, candidate, actual_hash):
        raise CandidateVersionMismatch(
            f"candidate semantic identity changed after validation: {candidate_path}"
        )
    code_path = candidate_path.with_name("candidate.py")
    if (
        not code_path.is_file()
        or code_path.read_text(encoding="utf-8") != candidate["code"]
    ):
        raise CandidateVersionMismatch(
            f"candidate.py differs from the validated candidate JSON: {code_path}"
        )
    validation_path = candidate_path.with_name("validation_report.json")
    try:
        saved_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateVersionMismatch(
            f"saved validation report cannot be read: {validation_path}: {exc}"
        ) from exc
    for field, expected_value in (
        ("candidate_id", candidate_id),
        ("candidate_semantic_fingerprint", semantic),
        ("candidate_full_content_sha256", expected),
        ("status", "passed"),
    ):
        if saved_validation.get(field) != expected_value:
            raise CandidateVersionMismatch(
                f"saved validation report has inconsistent {field}: {validation_path}"
            )
    if saved_validation != validation:
        raise CandidateVersionMismatch(
            f"saved validation report differs from current state: {validation_path}"
        )
    if validation.get("status") != "passed":
        raise CandidateVersionMismatch("candidate validation is not passing")
    return candidate, actual_hash


def _restore_state(
    resume: str | Path,
    *,
    max_review_rounds: int | None,
    max_code_repairs: int | None,
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    output = Path(resume).resolve()
    state_path = output / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"review-design state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    schema_version = state.get("schema_version")
    if schema_version not in {
        REVIEW_RUN_SCHEMA_VERSION,
        LEGACY_REVIEW_RUN_SCHEMA_VERSION,
    }:
        raise ReviewDesignError("review-design resume contract is incompatible")
    if schema_version == LEGACY_REVIEW_RUN_SCHEMA_VERSION:
        state["_legacy_schema_version"] = schema_version
    if state.get("git_sha") != _git_sha():
        if not (
            schema_version == LEGACY_REVIEW_RUN_SCHEMA_VERSION
            and state.get("git_sha") in LEGACY_REVIEW_COMPATIBLE_GIT_REVISIONS
        ):
            raise ReviewDesignError("review-design resume git revision is incompatible")
    settings = state["settings"]
    requested_review = (
        settings["max_review_rounds"]
        if max_review_rounds is None
        else int(max_review_rounds)
    )
    requested_repairs = (
        settings["max_code_repairs"]
        if max_code_repairs is None
        else int(max_code_repairs)
    )
    if requested_review < int(settings["max_review_rounds"]):
        raise ReviewDesignError("resume cannot reduce max-review-rounds")
    if requested_repairs < int(settings["max_code_repairs"]):
        raise ReviewDesignError("resume cannot reduce max-code-repairs")
    if not dry_run:
        in_flight = state.get("request_in_flight")
        if in_flight is not None:
            state.setdefault("interrupted_requests", []).append(
                {
                    **copy.deepcopy(in_flight),
                    "status": "outcome_unknown_not_completed",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            state["request_in_flight"] = None
        settings["max_review_rounds"] = requested_review
        settings["max_code_repairs"] = requested_repairs
        if state["status"] != "approved":
            state["status"] = "running"
            state["stop_reason"] = None
    return output, state


def _result(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state["status"],
        "output_directory": state["output_directory"],
        "approved_artifact": state.get("approved_artifact"),
        "stop_reason": state.get("stop_reason"),
        "proposer_calls": state["counters"]["proposer_calls"],
        "reviewer_calls": state["counters"]["reviewer_calls"],
        "review_rounds_completed": state["counters"]["review_rounds_completed"],
        "code_repairs_used_in_cycle": state["counters"][
            "code_repairs_used_in_cycle"
        ],
    }


def run_review_design(
    *,
    fixed_sample: str | Path | None = None,
    proposer_provider: str | None = None,
    proposer_model: str | None = None,
    proposer_base_url: str | None = None,
    proposer_context_length: int = DEFAULT_CONTEXT_LENGTH,
    proposer_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    proposer_temperature: float = DEFAULT_TEMPERATURE,
    proposer_seed: int = DEFAULT_SEED,
    proposer_reasoning_effort: str | None = None,
    proposer_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    proposer_connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    proposer_total_timeout: float = DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    proposer_progress_interval: float = DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    reviewer_provider: str | None = None,
    reviewer_model: str | None = None,
    reviewer_base_url: str | None = None,
    reviewer_context_length: int = DEFAULT_CONTEXT_LENGTH,
    reviewer_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reviewer_temperature: float = DEFAULT_TEMPERATURE,
    reviewer_seed: int = DEFAULT_SEED,
    reviewer_reasoning_effort: str | None = None,
    reviewer_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    reviewer_connect_timeout: float = DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    reviewer_total_timeout: float = DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    reviewer_progress_interval: float = DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    beta: float = DEFAULT_BETA,
    worker_timeout: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    max_review_rounds: int | None = None,
    max_code_repairs: int | None = None,
    output_root: str | Path = DEFAULT_REVIEW_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    resume: str | Path | None = None,
    proposer_client: Any | None = None,
    reviewer_client: Any | None = None,
) -> dict[str, Any]:
    if resume is not None:
        output, state = _restore_state(
            resume,
            max_review_rounds=max_review_rounds,
            max_code_repairs=max_code_repairs,
            dry_run=dry_run,
        )
        settings = state["settings"]
        fixed_sample = state["fixed_sample"]["directory"]
        proposer = settings["proposer"]
        reviewer = settings["reviewer"]
        beta = float(settings["beta"])
        worker_timeout = float(settings["worker_timeout"])
    else:
        if not fixed_sample or not proposer_provider or not proposer_model:
            raise ValueError("new review-design runs require fixed sample and proposer settings")
        if not reviewer_provider or not reviewer_model:
            raise ValueError("new review-design runs require reviewer settings")
        maximum_reviews = (
            DEFAULT_MAX_REVIEW_ROUNDS
            if max_review_rounds is None
            else int(max_review_rounds)
        )
        maximum_repairs = (
            DEFAULT_MAX_CODE_REPAIRS
            if max_code_repairs is None
            else int(max_code_repairs)
        )
        if maximum_reviews <= 0 or maximum_repairs < 0:
            raise ValueError("review rounds must be positive and code repairs non-negative")
        proposer = _role_config(
            provider=proposer_provider,
            model=proposer_model,
            base_url=proposer_base_url,
            context_length=proposer_context_length,
            max_output_tokens=proposer_max_output_tokens,
            temperature=proposer_temperature,
            seed=proposer_seed,
            reasoning_effort=proposer_reasoning_effort,
            timeout=proposer_timeout,
            connect_timeout=proposer_connect_timeout,
            total_timeout=proposer_total_timeout,
            progress_interval=proposer_progress_interval,
        )
        reviewer = _role_config(
            provider=reviewer_provider,
            model=reviewer_model,
            base_url=reviewer_base_url,
            context_length=reviewer_context_length,
            max_output_tokens=reviewer_max_output_tokens,
            temperature=reviewer_temperature,
            seed=reviewer_seed,
            reasoning_effort=reviewer_reasoning_effort,
            timeout=reviewer_timeout,
            connect_timeout=reviewer_connect_timeout,
            total_timeout=reviewer_total_timeout,
            progress_interval=reviewer_progress_interval,
        )
        output = allocate_design_directory(
            _slug(proposer_model) + "--" + _slug(reviewer_model),
            output_root=output_root,
            output_dir=output_dir,
        )
        arrays, fixed_metadata, constants_metadata = _load_review_inputs(fixed_sample)
        state = _new_state(
            output=output,
            fixed_sample=Path(fixed_sample),
            fixed_metadata=fixed_metadata,
            proposer=proposer,
            reviewer=reviewer,
            beta=beta,
            worker_timeout=worker_timeout,
            max_review_rounds=maximum_reviews,
            max_code_repairs=maximum_repairs,
        )
        _write_json(output / "state.json", state)

    arrays, fixed_metadata, constants_metadata = _load_review_inputs(fixed_sample)
    if fixed_metadata["sample_content_sha256"] != state["fixed_sample"][
        "sample_content_sha256"
    ]:
        raise ReviewDesignError("resume fixed-sample content hash is incompatible")
    _migrate_legacy_state(
        output,
        state,
        constants_metadata,
        persist=resume is not None and not dry_run,
    )
    if state["status"] == "approved":
        return _result(state)
    if resume is not None and not dry_run:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "state.json", state)
    if (
        not dry_run
        and state.get("pending_review") is not None
        and state["counters"]["review_rounds_completed"]
        >= int(state["settings"]["max_review_rounds"])
    ):
        state["status"] = "paused_review_rounds_exhausted"
        state["stop_reason"] = (
            "increase max-review-rounds explicitly before generating the requested revision"
        )
        _write_json(output / "state.json", state)
        return _result(state)
    obs_arrays = build_obs_arrays(arrays)
    diagnostic_contract = runtime_diagnostic_contract(
        fixed_metadata, constants_metadata
    )
    common = render_common_context(
        fixed_metadata=fixed_metadata,
        constants_metadata=constants_metadata,
        beta=beta,
    )
    pclient = proposer_client or _make_client(proposer)
    rclient = reviewer_client or _make_client(reviewer)
    proposer_inventory = _role_inventory(proposer, pclient, dry_run)
    reviewer_inventory = _role_inventory(reviewer, rclient, dry_run)
    proposer_effective, proposer_context = _effective_context_budget(
        proposer["context_length"], proposer_inventory
    )
    reviewer_effective, reviewer_context = _effective_context_budget(
        reviewer["context_length"], reviewer_inventory
    )
    state["model_inventory"] = {
        "proposer": proposer_inventory,
        "reviewer": reviewer_inventory,
    }
    state["context"] = {
        "proposer": proposer_context,
        "reviewer": reviewer_context,
    }

    if dry_run:
        proposer_prompt = render_proposer_prompt(
            common_context=common,
            round_context=_proposer_round_context(state),
        )
        reviewer_prompt = render_reviewer_prompt(
            common_context=common,
            candidate=normalize_candidate_submission(
                minimal_model_candidate_example(), constants_metadata
            )[1],
            validation_report={
                "status": "dry_run_example",
                "scope": {
                    "fixed_sample_count": fixed_metadata["sample_count"],
                },
                "checks": {},
                "errors": [],
            },
            review_history=[],
        )
        preview = {
            "status": "dry_run_complete",
            "proposer": {
                "settings": proposer,
                "context": proposer_context,
                "token_budget": estimate_token_budget(
                    proposer_prompt,
                    context_length=proposer_effective,
                    max_output_tokens=proposer["max_output_tokens"],
                ),
                "planned_request": planned_chat_request(
                    proposer["provider"],
                    model=proposer["model"],
                    prompt=proposer_prompt,
                    temperature=proposer["temperature"],
                    max_output_tokens=proposer["max_output_tokens"],
                    seed=proposer["seed"],
                    reasoning_effort=proposer["reasoning_effort"],
                ),
            },
            "reviewer": {
                "settings": reviewer,
                "context": reviewer_context,
                "token_budget": estimate_token_budget(
                    reviewer_prompt,
                    context_length=reviewer_effective,
                    max_output_tokens=reviewer["max_output_tokens"],
                ),
                "planned_request": planned_chat_request(
                    reviewer["provider"],
                    model=reviewer["model"],
                    prompt=reviewer_prompt,
                    temperature=reviewer["temperature"],
                    max_output_tokens=reviewer["max_output_tokens"],
                    seed=reviewer["seed"],
                    reasoning_effort=reviewer["reasoning_effort"],
                ),
            },
            "fixed_sample": state["fixed_sample"],
            "limits": {
                "max_review_rounds": state["settings"]["max_review_rounds"],
                "max_code_repairs": state["settings"]["max_code_repairs"],
            },
            "generation_requests_sent": 0,
        }
        if resume is None:
            (output / "proposer_dry_run_prompt.txt").write_text(
                proposer_prompt, encoding="utf-8"
            )
            (output / "reviewer_dry_run_prompt.txt").write_text(
                reviewer_prompt, encoding="utf-8"
            )
            _write_json(output / "dry_run.json", preview)
            state["status"] = "dry_run_complete"
            _write_json(output / "state.json", state)
        return {**_result({**state, "status": "dry_run_complete"}), "dry_run": preview}

    while True:
        state["status"] = "running"
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "state.json", state)
        if state["phase"] == "proposer_needed":
            maximum_outputs = int(state["settings"]["max_code_repairs"]) + 1
            if state["counters"]["proposal_outputs_in_cycle"] >= maximum_outputs:
                state["status"] = "paused_code_repairs_exhausted"
                state["stop_reason"] = "initial proposal plus code-repair allowance exhausted"
                _write_json(output / "state.json", state)
                return _result(state)
            prompt = render_proposer_prompt(
                common_context=common,
                round_context=_proposer_round_context(state),
            )
            response, call_status = _call_role(
                role="proposer",
                prompt=prompt,
                config=proposer,
                client=pclient,
                output=output,
                state=state,
                effective_context=proposer_effective,
            )
            if response is None:
                state["status"] = call_status["status"]
                state["stop_reason"] = call_status
                _write_json(output / "state.json", state)
                return _result(state)
            state["counters"]["proposal_outputs_in_cycle"] += 1
            state["counters"]["code_repairs_used_in_cycle"] = max(
                0, state["counters"]["proposal_outputs_in_cycle"] - 1
            )
            content = str(response["content"])
            candidate, validation, details = validate_submission(
                content,
                arrays=arrays,
                obs_arrays=obs_arrays,
                constants_metadata=constants_metadata,
                diagnostic_contract=diagnostic_contract,
                worker_timeout=worker_timeout,
            )
            call_dir = Path(call_status["directory"])
            _write_json(call_dir / "validation_report.json", validation)
            if details.get("parse_metadata") is not None:
                _write_json(call_dir / "candidate_parse_metadata.json", details["parse_metadata"])
            if details.get("submitted_candidate") is not None:
                _write_json(call_dir / "submitted_candidate.json", details["submitted_candidate"])
            state["current_submission"] = details.get("submitted_candidate", content)
            state["current_validation"] = validation
            state["current_candidate_id"] = None
            state["current_candidate_full_content_sha256"] = None
            if candidate is not None:
                semantic_fingerprint = candidate_semantic_fingerprint(candidate)
                full_content_sha256 = candidate_full_content_sha256(candidate)
                identifier = _candidate_id(candidate)
                existing = state["candidates"].get(identifier)
                if (
                    isinstance(existing, dict)
                    and existing.get("full_content_sha256") != full_content_sha256
                ):
                    identifier = f"{identifier}-{full_content_sha256[:12]}"
                validation["candidate_id"] = identifier
                validation["candidate_semantic_fingerprint"] = semantic_fingerprint
                validation["candidate_full_content_sha256"] = full_content_sha256
                _write_json(call_dir / "validation_report.json", validation)
                state["current_candidate_id"] = identifier
                state["current_candidate_full_content_sha256"] = full_content_sha256
                candidate_dir = output / "candidates" / identifier
                candidate_dir.mkdir(parents=True, exist_ok=True)
                _write_json(candidate_dir / "candidate.json", candidate)
                (candidate_dir / "candidate.py").write_text(
                    candidate["code"], encoding="utf-8"
                )
                _write_json(candidate_dir / "validation_report.json", validation)
                attempt = {
                    "proposal_cycle": state["proposal_cycle"],
                    "proposer_call": state["counters"]["proposer_calls"],
                    "actual_model": response.get("actual_model") or proposer["model"],
                    "validation_status": validation["status"],
                }
                if identifier not in state["candidates"]:
                    state["candidate_order"].append(identifier)
                    state["candidates"][identifier] = {
                        "candidate_id": identifier,
                        "semantic_fingerprint": semantic_fingerprint,
                        "full_content_sha256": full_content_sha256,
                        "parent_candidate_id": state.get(
                            "revision_parent_candidate_id"
                        ),
                        "directory": str(candidate_dir),
                        "submission_attempts": [],
                    }
                state["candidates"][identifier]["submission_attempts"].append(attempt)
            if validation["status"] != "passed":
                _write_json(output / "state.json", state)
                continue
            state["phase"] = "reviewer_needed"
            _write_json(output / "state.json", state)
            continue

        if state["phase"] != "reviewer_needed":
            raise ReviewDesignError(f"unknown review-design phase: {state['phase']!r}")
        identifier = state["current_candidate_id"]
        if not identifier or state["current_validation"].get("status") != "passed":
            raise ReviewDesignError("reviewer phase lacks a validated candidate")
        try:
            candidate, reviewed_candidate_hash = _verified_candidate_for_review(
                output, state, identifier
            )
        except CandidateVersionMismatch as exc:
            state["status"] = "paused_candidate_version_mismatch"
            state["stop_reason"] = {
                "status": state["status"],
                "phase": "before_reviewer_request",
                "candidate_id": identifier,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output / "state.json", state)
            return _result(state)
        prompt = render_reviewer_prompt(
            common_context=common,
            candidate=candidate,
            validation_report=state["current_validation"],
            review_history=state["review_history"],
        )
        response, call_status = _call_role(
            role="reviewer",
            prompt=prompt,
            config=reviewer,
            client=rclient,
            output=output,
            state=state,
            effective_context=reviewer_effective,
        )
        if response is None:
            state["status"] = call_status["status"]
            state["stop_reason"] = call_status
            _write_json(output / "state.json", state)
            return _result(state)
        try:
            review, parse_metadata = _strict_review(str(response["content"]))
        except CandidateError as exc:
            state["status"] = "paused_invalid_review"
            state["stop_reason"] = {
                "status": state["status"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output / "state.json", state)
            return _result(state)
        call_dir = Path(call_status["directory"])
        _write_json(call_dir / "review.json", review)
        _write_json(call_dir / "review_parse_metadata.json", parse_metadata)
        review_binding = {
            "candidate_id": identifier,
            "candidate_full_content_sha256": reviewed_candidate_hash,
            "status": "response_received_pending_integrity_check",
        }
        _write_json(call_dir / "candidate_binding.json", review_binding)
        try:
            current_candidate, current_candidate_hash = _verified_candidate_for_review(
                output, state, identifier
            )
            if (
                current_candidate_hash != reviewed_candidate_hash
                or candidate_full_content_sha256(candidate) != reviewed_candidate_hash
                or current_candidate != candidate
            ):
                raise CandidateVersionMismatch(
                    "candidate content changed while the reviewer request was in progress"
                )
        except CandidateVersionMismatch as exc:
            review_binding["status"] = "candidate_version_mismatch"
            review_binding["error"] = str(exc)
            _write_json(call_dir / "candidate_binding.json", review_binding)
            state["status"] = "paused_candidate_version_mismatch"
            state["stop_reason"] = {
                "status": state["status"],
                "phase": "after_reviewer_response",
                "candidate_id": identifier,
                "reviewer_call": state["counters"]["reviewer_calls"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output / "state.json", state)
            return _result(state)
        review_binding["status"] = "verified"
        _write_json(call_dir / "candidate_binding.json", review_binding)
        state["counters"]["review_rounds_completed"] += 1
        review_record = {
            "review_round": state["counters"]["review_rounds_completed"],
            "candidate_id": identifier,
            "reviewer_call": state["counters"]["reviewer_calls"],
            "actual_model": response.get("actual_model") or reviewer["model"],
            "candidate_full_content_sha256": reviewed_candidate_hash,
            "candidate_binding_status": "verified",
            "review": review,
        }
        state["review_history"].append(review_record)
        if review["needs_revision"]:
            state["pending_review"] = review_record
            state["revision_parent_candidate_id"] = identifier
            if state["counters"]["review_rounds_completed"] >= int(
                state["settings"]["max_review_rounds"]
            ):
                # Prepare the exact next phase before pausing. A later explicit
                # limit increase resumes by generating the requested revision;
                # it must not review the already-rejected version again.
                state["proposal_cycle"] += 1
                state["counters"]["proposal_outputs_in_cycle"] = 0
                state["counters"]["code_repairs_used_in_cycle"] = 0
                state["phase"] = "proposer_needed"
                state["status"] = "paused_review_rounds_exhausted"
                state["stop_reason"] = (
                    "last permitted review requested changes; no unsendable proposal was generated"
                )
                _write_json(output / "state.json", state)
                return _result(state)
            state["proposal_cycle"] += 1
            state["counters"]["proposal_outputs_in_cycle"] = 0
            state["counters"]["code_repairs_used_in_cycle"] = 0
            state["phase"] = "proposer_needed"
            _write_json(output / "state.json", state)
            continue

        if state["current_validation"].get("candidate_id") != identifier:
            raise ReviewDesignError(
                "review approval candidate does not match the validated candidate"
            )
        try:
            approval_candidate, approval_candidate_hash = _verified_candidate_for_review(
                output, state, identifier
            )
            if (
                approval_candidate_hash != reviewed_candidate_hash
                or review_record["candidate_full_content_sha256"]
                != approval_candidate_hash
                or approval_candidate != candidate
            ):
                raise CandidateVersionMismatch(
                    "validation, reviewer input, review result, and approval candidate do not match"
                )
        except CandidateVersionMismatch as exc:
            state["status"] = "paused_candidate_version_mismatch"
            state["stop_reason"] = {
                "status": state["status"],
                "phase": "before_approval",
                "candidate_id": identifier,
                "reviewer_call": state["counters"]["reviewer_calls"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output / "state.json", state)
            return _result(state)
        candidate_attempt = state["candidates"][identifier]["submission_attempts"][-1]
        approval = {
            "schema_version": REVIEW_APPROVAL_VERSION,
            "status": "approved",
            "passed": True,
            "approval_method": "model_review",
            "candidate_id": identifier,
            "candidate_semantic_fingerprint": state["candidates"][identifier][
                "semantic_fingerprint"
            ],
            "candidate_full_content_sha256": approval_candidate_hash,
            "validation_status": state["current_validation"]["status"],
            "review_round": state["counters"]["review_rounds_completed"],
            "review": review,
            "lipschitz_evaluation_performed": False,
            "training_performance_verified": False,
        }
        approved = save_approved_artifact(
            output,
            candidate=approval_candidate,
            constants_metadata=constants_metadata,
            validation_report=state["current_validation"],
            evaluation_report=approval,
            provenance={
                "approval_method": "model_review",
                "run_directory": str(output),
                "beta": float(beta),
                "model_requested": proposer["model"],
                "model_actual": candidate_attempt["actual_model"],
                "proposer": {
                    "provider": proposer["provider"],
                    "model": proposer["model"],
                    "actual_model": candidate_attempt["actual_model"],
                },
                "reviewer": {
                    "provider": reviewer["provider"],
                    "model": reviewer["model"],
                    "actual_model": review_record["actual_model"],
                },
                "candidate_id": identifier,
                "candidate_semantic_fingerprint": state["candidates"][identifier][
                    "semantic_fingerprint"
                ],
                "candidate_full_content_sha256": approval_candidate_hash,
                "review_round": state["counters"]["review_rounds_completed"],
            },
            observation_interface_version=OBS_INTERFACE_VERSION,
        )
        state["status"] = "approved"
        state["phase"] = "approved"
        state["approved_artifact"] = str(approved)
        state["stop_reason"] = "validated candidate accepted by reviewer"
        state["pending_review"] = None
        state["revision_parent_candidate_id"] = None
        _write_json(output / "state.json", state)
        return _result(state)
