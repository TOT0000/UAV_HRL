"""Isolated subprocess for executing an already statically checked candidate."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from llm_candidate import feature_reward, validate_candidate
from llm_numeric_operations import SAFE_BUILTIN_CALLS
from llm_design_contract import (
    LEGACY_OBS_KEYS,
    OBS_KEYS,
)


SAFE_BUILTINS = {
    name: getattr(builtins, name) for name in sorted(SAFE_BUILTIN_CALLS)
}
# Several ndarray methods lazily import NumPy's own implementation helpers from
# the caller frame.  Static validation still rejects Import/ImportFrom and every
# direct or indirect candidate call outside the explicit allow-list; exposing
# the interpreter hook here only lets already-approved NumPy methods finish.
SAFE_BUILTINS["__import__"] = builtins.__import__


class CandidateOutputValidationError(ValueError):
    """Structured raw-output violations found before host float32 conversion."""

    def __init__(self, issues):
        self.issues = list(issues)
        summary = "; ".join(str(item["problem"]) for item in self.issues[:3])
        if len(self.issues) > 3:
            summary += f"; and {len(self.issues) - 3} more"
        super().__init__(summary)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _empty_probe(obs):
    result = {}
    id_fields = {"sr_id", "sr_roi_id", "roi_id", "task_target_id", "vs_uav_id", "vs_roi_id"}
    for name, value in obs.items():
        array = np.zeros_like(value)
        if name in id_fields:
            array.fill(-1)
        result[name] = array
    result["snapshot_valid"][0] = True
    return result


def _run_function(function, obs, constants, expected_size, label):
    before = {name: value.copy() for name, value in obs.items()}
    constants_before = json.dumps(
        constants, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    first = function(obs, constants)
    second = function(obs, constants)
    if any(not np.array_equal(value, before[name]) for name, value in obs.items()):
        raise ValueError(f"{label} modified obs")
    if json.dumps(
        constants, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) != constants_before:
        raise ValueError(f"{label} modified constants")
    def checked(value, run_name):
        violations = []
        if not isinstance(value, (list, np.ndarray)):
            return None, [
                {
                    "code": "RUNTIME_OUTPUT_TYPE",
                    "location": label,
                    "problem": (
                        f"{run_name} returned {type(value).__name__}; expected a "
                        "one-dimensional numeric list or NumPy array"
                    ),
                    "requirement": (
                        "Return a one-dimensional numeric list or NumPy array. The host "
                        "validates it and then converts it to float32."
                    ),
                    "execution_run": run_name,
                }
            ]
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            return None, [
                {
                    "code": "RUNTIME_OUTPUT_TYPE",
                    "location": label,
                    "problem": f"{run_name} cannot be interpreted as a numeric array: {exc}",
                    "requirement": "Return a rectangular one-dimensional numeric list or NumPy array.",
                    "execution_run": run_name,
                }
            ]
        if raw.ndim != 1:
            return None, [
                {
                    "code": "RUNTIME_SHAPE",
                    "location": label,
                    "problem": f"{run_name} must be one-dimensional; returned shape {raw.shape}, expected ({expected_size},)",
                    "requirement": f"Return exactly {expected_size} values in one dimension.",
                    "execution_run": run_name,
                    "observed_shape": list(raw.shape),
                }
            ]
        if raw.dtype.kind not in "iuf":
            return None, [
                {
                    "code": "RUNTIME_OUTPUT_TYPE",
                    "location": label,
                    "problem": f"{run_name} returned non-numeric dtype {raw.dtype}",
                    "requirement": "Return numeric values in a list or NumPy array.",
                    "execution_run": run_name,
                    "observed_dtype": str(raw.dtype),
                }
            ]
        if raw.shape != (expected_size,):
            return None, [
                {
                    "code": "RUNTIME_SHAPE",
                    "location": label,
                    "problem": f"{run_name} returned shape {raw.shape}; expected ({expected_size},)",
                    "requirement": f"Return exactly {expected_size} values in feature order.",
                    "execution_run": run_name,
                    "observed_shape": list(raw.shape),
                }
            ]
        # Validate in float64 before the host's required float32 conversion so
        # conversion cannot hide NaN/Inf or an out-of-range raw value.
        raw64 = raw.astype(np.float64, copy=False)
        for index, numeric in enumerate(raw64):
            value_number = float(numeric)
            if not np.isfinite(value_number):
                violations.append(
                    {
                        "code": "RUNTIME_NONFINITE",
                        "location": f"{label} feature[{index}]",
                        "problem": f"{run_name} feature[{index}] returned NaN or Infinity ({value_number})",
                        "requirement": "Every feature value must be finite before float32 conversion.",
                        "execution_run": run_name,
                        "feature_index": index,
                        "observed_value": str(value_number),
                    }
                )
            elif value_number < -1e-6 or value_number > 1.0 + 1e-6:
                violations.append(
                    {
                        "code": "RUNTIME_FEATURE_BOUNDS",
                        "location": f"{label} feature[{index}]",
                        "problem": f"{run_name} feature[{index}] returned {value_number}, outside [0,1]",
                        "requirement": "Every shared feature value must remain within [0,1]; do not rely on host clipping.",
                        "execution_run": run_name,
                        "feature_index": index,
                        "observed_value": value_number,
                        "allowed_range": {"minimum": 0.0, "maximum": 1.0},
                    }
                )
        if violations:
            return None, violations
        converted = raw64.astype(np.float32)
        for index, numeric in enumerate(converted):
            if not np.isfinite(float(numeric)):
                violations.append(
                    {
                        "code": "RUNTIME_NONFINITE_AFTER_CONVERSION",
                        "location": f"{label} feature[{index}]",
                        "problem": f"{run_name} feature[{index}] became non-finite after float32 conversion",
                        "requirement": "Return values representable as finite float32 numbers.",
                        "execution_run": run_name,
                        "feature_index": index,
                    }
                )
        return converted, violations

    first, first_issues = checked(first, "first result")
    second, second_issues = checked(second, "second result")
    if first_issues or second_issues:
        raise CandidateOutputValidationError(first_issues + second_issues)
    if not np.array_equal(first, second):
        raise ValueError(f"{label} is not deterministic for identical input")
    return first


def _validate_output_ranges(extra, candidate, label):
    tolerance = 1e-6
    extra = np.asarray(extra)
    if extra.ndim == 1:
        extra = extra[None, :]
    for index, definition in enumerate(candidate["features"]):
        minimum = float(definition["range"]["minimum"])
        maximum = float(definition["range"]["maximum"])
        observed_minimum = float(np.min(extra[:, index]))
        observed_maximum = float(np.max(extra[:, index]))
        if observed_minimum < -tolerance or observed_maximum > 1.0 + tolerance:
            raise ValueError(
                f"compute_extra_state({label}) feature[{index}] is outside [0,1]: "
                f"observed [{observed_minimum},{observed_maximum}]"
            )
        if observed_minimum < minimum - tolerance or observed_maximum > maximum + tolerance:
            raise ValueError(
                f"compute_extra_state({label}) feature[{index}] violates its declared "
                f"range [{minimum},{maximum}]: observed "
                f"[{observed_minimum},{observed_maximum}]"
            )
    feature_reward(extra, candidate)


class _IssueAccumulator:
    def __init__(self, example_limit=3):
        self.example_limit = int(example_limit)
        self._items = {}

    def add(
        self,
        code,
        location,
        problem,
        requirement,
        sample,
        *,
        exception_type=None,
        candidate_function=None,
        candidate_line=None,
        problem_signature=None,
        diagnostics=None,
        operation_fingerprint=None,
        operation_identity=None,
        structured_details=None,
    ):
        key = (
            str(code),
            str(location),
            str(requirement),
            "" if exception_type is None else str(exception_type),
            "" if candidate_function is None else str(candidate_function),
            "" if candidate_line is None else str(candidate_line),
            "" if problem_signature is None else str(problem_signature),
            "" if operation_fingerprint is None else str(operation_fingerprint),
        )
        details = {}
        if exception_type is not None:
            details["exception_type"] = str(exception_type)
        if candidate_function is not None:
            details["candidate_function"] = str(candidate_function)
        if candidate_line is not None:
            details["candidate_line"] = int(candidate_line)
        if problem_signature is not None:
            details["problem_signature"] = str(problem_signature)
        if diagnostics:
            details["runtime_diagnostics"] = diagnostics
        if operation_fingerprint is not None:
            details["operation_fingerprint"] = str(operation_fingerprint)
        if operation_identity:
            details["operation_identity"] = operation_identity
        if structured_details:
            details.update(structured_details)
        item = self._items.setdefault(
            key,
            {
                "code": str(code),
                "stage": "execution",
                "location": str(location),
                "problem": str(problem),
                "requirement": str(requirement),
                "occurrence_count": 0,
                "representative_samples": [],
                **details,
            },
        )
        item["occurrence_count"] += 1
        if len(item["representative_samples"]) < self.example_limit:
            item["representative_samples"].append(sample)

    @property
    def errors(self):
        return list(self._items.values())


def _runtime_error_code(message):
    lowered = str(message).lower()
    if "shape" in lowered or "one-dimensional" in lowered:
        return "RUNTIME_SHAPE"
    if "dtype" in lowered:
        return "RUNTIME_DTYPE"
    if "nan" in lowered or "infinity" in lowered or "finite" in lowered:
        return "RUNTIME_NONFINITE"
    if "deterministic" in lowered:
        return "RUNTIME_NONDETERMINISTIC"
    if "modified" in lowered or "read-only" in lowered:
        return "RUNTIME_INPUT_MUTATION"
    return "RUNTIME_FUNCTION_ERROR"


def _candidate_exception_details(exc):
    candidate_frame = None
    candidate_summary = None
    current = exc.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        if code.co_filename == "<approved-candidate>":
            candidate_frame = current.tb_frame
            candidate_summary = traceback.extract_tb(current, limit=1)[0]
        current = current.tb_next
    exception_type = type(exc).__name__
    if candidate_frame is None:
        function = None
        line = None
    else:
        function = candidate_frame.f_code.co_name
        line = int(candidate_summary.lineno)
    # Numeric values and object addresses frequently differ by sample without
    # changing the faulty operation.  Field names and operation text remain,
    # so distinct accesses on the same source line can still stay separate.
    signature = re.sub(r"0x[0-9a-fA-F]+", "<address>", str(exc))
    signature = re.sub(
        r"(?<![A-Za-z_])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        "<number>",
        signature,
    )
    return exception_type, function, line, signature, candidate_frame


def _candidate_source_excerpt(source, line, radius=4):
    if not isinstance(source, str) or not isinstance(line, int):
        return []
    lines = source.splitlines()
    start = max(1, line - int(radius))
    stop = min(len(lines), line + int(radius))
    return [
        {"line": number, "code": lines[number - 1]}
        for number in range(start, stop + 1)
    ]


def _safe_local_array_summaries(frame):
    if frame is None:
        return {}
    summaries = {}
    for name, value in frame.f_locals.items():
        if isinstance(value, np.ndarray):
            summaries[str(name)] = {
                "type": "ndarray",
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
        elif isinstance(value, np.generic):
            summaries[str(name)] = {
                "type": "numpy_scalar",
                "dtype": str(value.dtype),
            }
    return summaries


def _iter_nodes_with_paths(node, path=()):
    yield node, path
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from _iter_nodes_with_paths(value, path + (field,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield from _iter_nodes_with_paths(
                        item, path + (f"{field}[{index}]",)
                    )


def _candidate_function_ast(source, function_name):
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return None
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )


def _node_contains_line(node, line):
    start = getattr(node, "lineno", None)
    stop = getattr(node, "end_lineno", start)
    return isinstance(start, int) and start <= line <= stop


def _local_array_summary(expression, summaries):
    if isinstance(expression, ast.Name):
        return summaries.get(expression.id)
    return None


def _traceback_subscript(function, line, summaries):
    candidates = []
    for node, path in _iter_nodes_with_paths(function):
        if not isinstance(node, ast.Subscript) or not _node_contains_line(node, line):
            continue
        indexed = _local_array_summary(node.value, summaries)
        mask = _local_array_summary(node.slice, summaries)
        score = 0
        if mask and mask.get("dtype") == "bool":
            score += 4
        if indexed and indexed.get("dtype") != "bool":
            score += 2
        if getattr(node, "lineno", None) == line:
            score += 1
        candidates.append((score, len(path), node, path))
    if not candidates:
        return None, None
    _, _, node, path = max(candidates, key=lambda item: (item[0], item[1]))
    return node, path


def _operation_identity(source, function, line, summaries, exception_type):
    if function is None or not isinstance(line, int):
        return None
    node, path = (
        _traceback_subscript(function, line, summaries)
        if exception_type == "IndexError"
        else (None, None)
    )
    if node is None:
        eligible = (
            ast.Subscript,
            ast.BinOp,
            ast.Call,
            ast.Compare,
            ast.BoolOp,
            ast.UnaryOp,
        )
        candidates = [
            (candidate, candidate_path)
            for candidate, candidate_path in _iter_nodes_with_paths(function)
            if isinstance(candidate, eligible)
            and _node_contains_line(candidate, line)
        ]
        if exception_type == "ZeroDivisionError":
            division_candidates = [
                item
                for item in candidates
                if isinstance(item[0], ast.BinOp)
                and isinstance(item[0].op, (ast.Div, ast.FloorDiv, ast.Mod))
            ]
            if division_candidates:
                candidates = division_candidates
        if candidates:
            node, path = max(candidates, key=lambda item: len(item[1]))
    if node is None:
        return None
    normalized_ast = ast.dump(node, annotate_fields=True, include_attributes=False)
    identity = {
        "node_type": type(node).__name__,
        "function_structural_path": list(path),
        "normalized_ast": normalized_ast,
        "source": ast.get_source_segment(source, node),
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return identity


def _assigned_names(node):
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, (ast.Name, ast.arg))
        and isinstance(getattr(item, "ctx", None), ast.Store)
    }


def _bindings_before_line(function, line):
    """Track only straight-line simple assignments before the traceback line."""

    bindings = {}
    for statement in function.body:
        start = getattr(statement, "lineno", 0)
        stop = getattr(statement, "end_lineno", start)
        if start <= line <= stop:
            break
        if stop >= line:
            break
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                bindings[target.id] = statement.value
            else:
                for name in _assigned_names(target):
                    bindings.pop(name, None)
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.value is None:
                bindings.pop(statement.target.id, None)
            else:
                bindings[statement.target.id] = statement.value
        elif isinstance(statement, ast.AugAssign):
            for name in _assigned_names(statement.target):
                bindings.pop(name, None)
        else:
            # Branches, loops, destructuring, and other side effects are outside
            # this intentionally small data-flow analysis. Invalidate every name
            # they may assign instead of reusing a stale source binding.
            for name in _assigned_names(statement):
                bindings.pop(name, None)
    return bindings


def _resolve_binding(expression, bindings, seen=None):
    seen = set() if seen is None else set(seen)
    while isinstance(expression, ast.Name) and expression.id in bindings:
        if expression.id in seen:
            return None
        seen.add(expression.id)
        expression = bindings[expression.id]
    return expression


def _is_obs_field(expression, field, bindings):
    expression = _resolve_binding(expression, bindings)
    return (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "obs"
        and isinstance(expression.slice, ast.Constant)
        and expression.slice.value == field
    )


def _state_stride_origin(expression, bindings, local_dimension, source):
    expression = _resolve_binding(expression, bindings)
    if not isinstance(expression, ast.Subscript) or not isinstance(
        expression.slice, ast.Slice
    ):
        return None
    state_slice = expression.slice
    if state_slice.upper is not None:
        return None
    step = state_slice.step
    if (
        not isinstance(step, ast.Constant)
        or not isinstance(step.value, int)
        or int(step.value) != int(local_dimension)
        or not _is_obs_field(expression.value, "state", bindings)
    ):
        return None
    lower = state_slice.lower
    if not isinstance(lower, ast.Constant) or not isinstance(lower.value, int):
        return None
    if not 0 <= int(lower.value) < int(local_dimension):
        return None
    return {
        "line": int(getattr(expression, "lineno", 0)),
        "start": int(lower.value),
        "step": int(step.value),
        "expression": ast.get_source_segment(source, expression),
    }


def _traceback_index_analysis(
    source, function, line, summaries, local_dimension
):
    if function is None or not isinstance(line, int) or local_dimension is None:
        return {"status": "unresolved", "reason": "traceback operation unavailable"}
    operation, _ = _traceback_subscript(function, line, summaries)
    if operation is None:
        return {
            "status": "unresolved",
            "reason": "no boolean-index subscript was identified at the traceback line",
        }
    bindings = _bindings_before_line(function, line)
    indexed_summary = _local_array_summary(operation.value, summaries)
    mask_summary = _local_array_summary(operation.slice, summaries)
    result = {
        "status": "unresolved",
        "indexed_expression": ast.get_source_segment(source, operation.value),
        "mask_expression": ast.get_source_segment(source, operation.slice),
        "indexed_array_summary": indexed_summary,
        "mask_array_summary": mask_summary,
    }
    if not (mask_summary and mask_summary.get("dtype") == "bool"):
        result["reason"] = "the traceback index was not linked to a local boolean array"
        return result
    if not _is_obs_field(operation.slice, "movement_mask", bindings):
        result["reason"] = "the traceback boolean index was not linked to obs.movement_mask"
        return result
    origin = _state_stride_origin(
        operation.value, bindings, local_dimension, source
    )
    if origin is None:
        result["reason"] = (
            "the traceback indexed array was not reliably linked to an unbounded "
            "stride slice of obs.state"
        )
        return result
    result.update({"status": "confirmed_state_stride", "state_slice_origin": origin})
    return result


def _runtime_diagnostics(
    exc,
    frame,
    candidate_source,
    candidate_function,
    candidate_line,
    contract,
):
    diagnostics = {
        "candidate_source_excerpt": _candidate_source_excerpt(
            candidate_source, candidate_line
        ),
        "local_array_summaries": _safe_local_array_summaries(frame),
    }
    message = str(exc)
    state_contract = contract.get("state", {}) if isinstance(contract, dict) else {}
    block = state_contract.get("uav_block", {}) if isinstance(state_contract, dict) else {}
    mask_contract = contract.get("movement_mask", {}) if isinstance(contract, dict) else {}
    expected_mask = (mask_contract.get("shape") or [None])[0]
    local_dimension = block.get("features_per_uav")
    array_summaries = diagnostics["local_array_summaries"]
    function = _candidate_function_ast(candidate_source, candidate_function)
    operation_identity = _operation_identity(
        candidate_source,
        function,
        candidate_line,
        array_summaries,
        type(exc).__name__,
    )
    if operation_identity is not None:
        diagnostics["traceback_operation"] = operation_identity
    index_analysis = (
        _traceback_index_analysis(
            candidate_source,
            function,
            candidate_line,
            array_summaries,
            local_dimension,
        )
        if type(exc).__name__ == "IndexError"
        else {
            "status": "not_applicable",
            "reason": "the runtime exception is not an IndexError",
        }
    )
    diagnostics["index_source_analysis"] = index_analysis
    related_interface = {}
    if index_analysis.get("mask_expression") is not None and mask_contract:
        related_interface["obs.movement_mask"] = mask_contract
    if index_analysis.get("status") == "confirmed_state_stride" and state_contract:
        related_interface["obs.state"] = state_contract
    if related_interface:
        diagnostics["related_interface"] = related_interface
    confident_uav_slice = (
        type(exc).__name__ == "IndexError"
        and "boolean index" in message.lower()
        and index_analysis.get("status") == "confirmed_state_stride"
        and (index_analysis.get("mask_array_summary") or {}).get("shape")
        == [expected_mask]
    )
    if confident_uav_slice:
        start = int(block["start"])
        stop = int(block["stop_exclusive"])
        num_uav = int(block["num_uav"])
        per_uav = int(block["features_per_uav"])
        indexed_shape = (index_analysis.get("indexed_array_summary") or {}).get(
            "shape"
        )
        observed_vectors = indexed_shape or ["reported by IndexError"]
        diagnostics["confirmed_state_slice_origin"] = index_analysis[
            "state_slice_origin"
        ]
        diagnostics["diagnostic_summary"] = (
            f"The authoritative UAV state block is indices {start}..{stop - 1}: "
            f"{num_uav} UAV groups with {per_uav} values each. The candidate produced "
            f"one-dimensional non-boolean local lengths {observed_vectors}, which cannot "
            f"be indexed by the length-{expected_mask} movement mask."
        )
        diagnostics["targeted_requirement"] = (
            f"Limit each per-UAV state slice to the authoritative half-open UAV block "
            f"[{start}:{stop}) before applying the length-{expected_mask} movement_mask, "
            "and update the corresponding feature formula/source description to match."
        )
    return diagnostics


def _check_function(
    function,
    obs,
    constants,
    expected_size,
    label,
    sample,
    issues,
    *,
    candidate_source,
    diagnostic_contract,
    feature_definitions=None,
):
    try:
        return _run_function(function, obs, constants, expected_size, label)
    except CandidateOutputValidationError as exc:
        for violation in exc.issues:
            index = violation.get("feature_index")
            feature_name = None
            if (
                isinstance(index, int)
                and isinstance(feature_definitions, list)
                and 0 <= index < len(feature_definitions)
            ):
                feature_name = feature_definitions[index].get("name")
            representative = dict(sample)
            representative["execution_run"] = violation.get("execution_run")
            if "observed_value" in violation:
                representative["observed_value"] = violation["observed_value"]
            details = {
                key: violation[key]
                for key in (
                    "feature_index",
                    "observed_value",
                    "allowed_range",
                    "observed_shape",
                    "observed_dtype",
                )
                if key in violation
            }
            if feature_name is not None:
                details["feature_name"] = feature_name
            issues.add(
                violation["code"],
                violation["location"],
                violation["problem"],
                violation["requirement"],
                representative,
                structured_details=details,
            )
        return None
    except BaseException as exc:
        exception_type, candidate_function, candidate_line, signature, frame = (
            _candidate_exception_details(exc)
        )
        try:
            diagnostics = _runtime_diagnostics(
                exc,
                frame,
                candidate_source,
                candidate_function,
                candidate_line,
                diagnostic_contract,
            )
        except BaseException as diagnostic_exc:
            # Diagnostics are strictly supplemental. Never replace the original
            # candidate exception with an analysis failure.
            diagnostics = {
                "candidate_source_excerpt": _candidate_source_excerpt(
                    candidate_source, candidate_line
                ),
                "local_array_summaries": _safe_local_array_summaries(frame),
                "diagnostic_status": "failed_without_affecting_runtime_error",
                "diagnostic_error_type": type(diagnostic_exc).__name__,
            }
        operation_identity = diagnostics.get("traceback_operation") or {}
        location = label
        if candidate_function is not None and candidate_line is not None:
            location = f"{candidate_function} at candidate line {candidate_line}"
        issues.add(
            _runtime_error_code(str(exc)),
            location,
            f"{exception_type}: {exc}",
            diagnostics.get(
                "targeted_requirement",
                "Return a deterministic, side-effect-free one-dimensional numeric "
                "list or NumPy array of the declared length with finite values; the "
                "host converts validated output to float32.",
            ),
            sample,
            exception_type=exception_type,
            candidate_function=candidate_function,
            candidate_line=candidate_line,
            problem_signature=signature,
            diagnostics=diagnostics,
            operation_fingerprint=operation_identity.get("fingerprint"),
            operation_identity=operation_identity or None,
        )
        return None


def _check_ranges(extra, candidate, label, sample, issues):
    tolerance = 1e-6
    if extra is not None:
        for index, definition in enumerate(candidate["features"]):
            value = float(extra[index])
            minimum = float(definition["range"]["minimum"])
            maximum = float(definition["range"]["maximum"])
            if value < -tolerance or value > 1.0 + tolerance:
                issues.add(
                    "RUNTIME_FEATURE_BOUNDS",
                    f"compute_extra_state({label}) feature[{index}]",
                    f"observed {value} outside [0,1]",
                    "Every shared feature value must remain within [0,1].",
                    sample,
                )
            if value < minimum - tolerance or value > maximum + tolerance:
                issues.add(
                    "RUNTIME_DECLARED_FEATURE_RANGE",
                    f"compute_extra_state({label}) feature[{index}]",
                    f"observed {value} outside declared [{minimum},{maximum}]",
                    "Make the implementation respect the feature's declared fixed range.",
                    sample,
                )
    if extra is not None:
        feature_reward(extra, candidate)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--constants", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--diagnostic-contract")
    args = parser.parse_args(argv)
    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        constants = json.loads(Path(args.constants).read_text(encoding="utf-8"))
        diagnostic_contract = (
            json.loads(Path(args.diagnostic_contract).read_text(encoding="utf-8"))
            if args.diagnostic_contract
            else {}
        )
        constants_metadata = {
            key: {"value": value, "dtype": "runtime", "unit": None, "meaning": "runtime", "source": "approved contract"}
            for key, value in constants.items()
        }
        validate_candidate(candidate, constants_metadata)
        with np.load(args.input, allow_pickle=False) as archive:
            field_set = set(archive.files)
            if field_set == set(OBS_KEYS):
                observation_keys = OBS_KEYS
            elif field_set == set(LEGACY_OBS_KEYS):
                observation_keys = LEGACY_OBS_KEYS
            else:
                raise ValueError("worker input exposes an unexpected observation field set")
            arrays = {key: np.asarray(archive[key]) for key in observation_keys}
        rows = arrays["state"].shape[0]
        if rows <= 0 or any(value.shape[0] != rows for value in arrays.values()):
            raise ValueError("worker observation batch has inconsistent rows")
        for value in arrays.values():
            value.setflags(write=False)
        namespace = {"__builtins__": SAFE_BUILTINS, "np": np}
        exec(compile(candidate["code"], "<approved-candidate>", "exec"), namespace, namespace)
        extra_function = namespace["compute_extra_state"]
        feature_count = len(candidate["features"])
        extra = np.empty((rows, feature_count), dtype=np.float32)
        issues = _IssueAccumulator()
        for row in range(rows):
            obs = {name: array[row] for name, array in arrays.items()}
            row_extra = _check_function(
                extra_function,
                obs,
                constants,
                feature_count,
                "compute_extra_state",
                {"fixed_sample_index": row},
                issues,
                candidate_source=candidate["code"],
                diagnostic_contract=diagnostic_contract,
                feature_definitions=candidate["features"],
            )
            _check_ranges(
                row_extra,
                candidate,
                "fixed sample",
                {"fixed_sample_index": row},
                issues,
            )
            if row_extra is not None:
                extra[row] = row_extra
        probe = _empty_probe({name: array[0] for name, array in arrays.items()})
        probe_extra = _check_function(
            extra_function,
            probe,
            constants,
            feature_count,
            "compute_extra_state(empty probe)",
            {"probe": "empty"},
            issues,
            candidate_source=candidate["code"],
            diagnostic_contract=diagnostic_contract,
            feature_definitions=candidate["features"],
        )
        _check_ranges(
            probe_extra,
            candidate,
            "empty probe",
            {"probe": "empty"},
            issues,
        )
        if issues.errors:
            output_contract_failed = any(
                item.get("code")
                in {
                    "RUNTIME_OUTPUT_TYPE",
                    "RUNTIME_SHAPE",
                    "RUNTIME_NONFINITE",
                    "RUNTIME_NONFINITE_AFTER_CONVERSION",
                    "RUNTIME_FEATURE_BOUNDS",
                    "RUNTIME_DECLARED_FEATURE_RANGE",
                }
                for item in issues.errors
            )
            _write_json(
                args.report,
                {
                    "status": "failed",
                    "errors": issues.errors,
                    "error_count": len(issues.errors),
                    "checks": {
                        "execution": {
                            "status": "failed",
                            "completed": True,
                            "sample_count_attempted": rows,
                            "empty_probe_attempted": True,
                            "subchecks": {
                                "raw_output_contract": {
                                    "status": "failed" if output_contract_failed else "not_run",
                                    "completed": output_contract_failed,
                                    "skipped_reason": (
                                        None
                                        if output_contract_failed
                                        else "candidate execution failed before a complete output was available"
                                    ),
                                },
                                "host_float32_outputs_and_reward": {
                                    "status": "not_run",
                                    "completed": False,
                                    "skipped_reason": "candidate execution or raw-output validation failed",
                                },
                            },
                        }
                    },
                },
            )
            return 1
        np.savez_compressed(
            args.output,
            extra_state=extra,
            extra_reward=feature_reward(extra, candidate),
        )
        _write_json(
            args.report,
            {
                "status": "passed",
                "sample_count": rows,
                "feature_count": feature_count,
                "feature_reward_source": "host dot(feature_reward_weights, extra_state)",
                "determinism_check": "all fixed samples plus empty probe",
                "input_mutation_check": "passed with read-only arrays and before/after comparison",
                "empty_probe_check": "passed",
                "numeric_tolerance": 1e-6,
            },
        )
        return 0
    except BaseException as exc:
        _write_json(
            args.report,
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
