"""Strict candidate parsing, static checks, isolated execution, and artifacts."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tokenize
from typing import Any

import numpy as np

from llm_design_contract import (
    APPROVED_ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    OBS_INTERFACE_VERSION,
    allowed_source_fields,
    build_obs_arrays,
    candidate_schema,
    runtime_constants,
)


NUMERIC_TOLERANCE = 1e-6
TOP_FIELDS = {
    "schema_version",
    "candidate_name",
    "reward_input_mode",
    "features",
    "code",
}
ITEM_FIELDS = {
    "index",
    "name",
    "dtype",
    "description",
    "range",
    "source_fields",
    "formula",
    "missing_data_rule",
    "reward_weight",
}
SAFE_BUILTIN_CALLS = {
    "abs",
    "bool",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "sum",
    "tuple",
    "zip",
}
SAFE_NUMPY_CALLS = {
    "np.abs",
    "np.all",
    "np.any",
    "np.arange",
    "np.array",
    "np.asarray",
    "np.bool_",
    "np.clip",
    "np.concatenate",
    "np.count_nonzero",
    "np.exp",
    "np.float32",
    "np.float64",
    "np.int32",
    "np.int64",
    "np.isfinite",
    "np.log",
    "np.log1p",
    "np.maximum",
    "np.mean",
    "np.minimum",
    "np.ones",
    "np.sqrt",
    "np.stack",
    "np.sum",
    "np.where",
    "np.zeros",
    "np.linalg.norm",
}
SAFE_NUMPY_ATTRIBUTES = {
    "np.float32",
    "np.float64",
    "np.int32",
    "np.int64",
    "np.bool_",
}
DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Lambda,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)


class CandidateError(ValueError):
    pass


class CandidateExecutionError(RuntimeError):
    def __init__(self, message: str, *, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def _reject_duplicate_key(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_candidate_json_envelope(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse a strict object, optionally inside one whole-response JSON fence."""

    if not isinstance(text, str) or not text.strip():
        raise CandidateError("candidate response is empty")
    stripped = text.strip()
    candidate_text = stripped
    envelope_removed = False

    def strict_loads(source: str) -> Any:
        return json.loads(
            source,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CandidateError(f"non-finite JSON number: {token}")
            ),
        )

    try:
        value = strict_loads(stripped)
    except CandidateError:
        raise
    except json.JSONDecodeError as plain_error:
        if "```" not in stripped:
            raise CandidateError(f"invalid JSON: {plain_error}") from plain_error
        match = re.fullmatch(
            r"```(?P<label>json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
            stripped,
        )
        if match is None:
            fence_count = stripped.count("```")
            opening = re.match(r"```([^\r\n]*)", stripped)
            if fence_count > 2:
                reason = "candidate response contains multiple Markdown code fences"
            elif stripped.startswith("```") and fence_count < 2:
                reason = "candidate response contains an incomplete Markdown code fence"
            elif opening is not None and opening.group(1).strip() not in {"", "json"}:
                reason = (
                    "candidate response uses unsupported Markdown code-fence label "
                    f"{opening.group(1).strip()!r}; only 'json' or no label is allowed"
                )
            elif not stripped.startswith("```") or not stripped.endswith("```"):
                reason = (
                    "candidate response contains explanatory text outside the single "
                    "JSON Markdown code fence"
                )
            else:
                reason = "candidate response has an invalid Markdown code-fence envelope"
            raise CandidateError(reason)
        candidate_text = match.group("body").strip()
        envelope_removed = True
        if not candidate_text:
            raise CandidateError("candidate Markdown code fence is empty")
        if "```" in candidate_text:
            raise CandidateError("candidate response contains multiple Markdown code fences")
        try:
            value = strict_loads(candidate_text)
        except CandidateError:
            raise
        except json.JSONDecodeError as exc:
            raise CandidateError(f"invalid JSON inside Markdown code fence: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateError("candidate response must be one JSON object")
    return value, {
        "markdown_envelope_removed": bool(envelope_removed),
        "accepted_envelope": "single_json_fence" if envelope_removed else "plain_json",
        "raw_character_count": len(text),
        "parsed_character_count": len(candidate_text),
    }


def parse_candidate_json(text: str) -> dict[str, Any]:
    value, _ = parse_candidate_json_envelope(text)
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise CandidateError(f"{label} must be finite")
    return value


def _validate_items(
    items: Any,
    *,
    allowed_fields: set[str],
) -> None:
    label = "features"
    if not isinstance(items, list) or not items:
        raise CandidateError(f"{label} must contain at least one item")
    names = []
    formulas = []
    weights = []
    exact_fields = ITEM_FIELDS
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != exact_fields:
            raise CandidateError(
                f"{label}[{index}] fields must be exactly {sorted(exact_fields)}"
            )
        if item["index"] != index or isinstance(item["index"], bool):
            raise CandidateError(f"{label} indices must be consecutive from zero")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise CandidateError(f"{label}[{index}].name must be non-empty")
        names.append(item["name"])
        if item["dtype"] != "float32":
            raise CandidateError(f"{label}[{index}].dtype must be float32")
        for text_field in ("description", "formula", "missing_data_rule"):
            if not isinstance(item[text_field], str) or not item[text_field].strip():
                raise CandidateError(f"{label}[{index}].{text_field} must be non-empty")
        formulas.append("".join(item["formula"].split()).lower())
        declared_range = item["range"]
        if not isinstance(declared_range, dict) or set(declared_range) != {
            "minimum",
            "maximum",
        }:
            raise CandidateError(f"{label}[{index}].range is invalid")
        minimum = _finite_number(declared_range["minimum"], f"{label}[{index}].minimum")
        maximum = _finite_number(declared_range["maximum"], f"{label}[{index}].maximum")
        if minimum > maximum:
            raise CandidateError(f"{label}[{index}] range is reversed")
        if minimum < 0.0 or maximum > 1.0:
            raise CandidateError("feature declared ranges must lie within [0,1]")
        source_fields = item["source_fields"]
        if (
            not isinstance(source_fields, list)
            or not source_fields
            or any(not isinstance(value, str) for value in source_fields)
            or len(set(source_fields)) != len(source_fields)
        ):
            raise CandidateError(f"{label}[{index}].source_fields is invalid")
        unknown = sorted(set(source_fields).difference(allowed_fields))
        if unknown:
            raise CandidateError(f"{label}[{index}] declares forbidden fields: {unknown}")
        weights.append(
            _finite_number(
                item["reward_weight"], f"{label}[{index}].reward_weight"
            )
        )
    if len(set(names)) != len(names):
        raise CandidateError(f"{label} names must be unique")
    duplicate_formulas = sorted(
        formula for formula in set(formulas) if formulas.count(formula) > 1
    )
    if duplicate_formulas:
        raise CandidateError(f"{label} contains duplicate formulas")
    if math.fsum(abs(value) for value in weights) > 1.0 + 1e-12:
        raise CandidateError("sum(abs(feature reward weights)) must be <= 1")


def validate_candidate_schema(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> None:
    if set(candidate) != TOP_FIELDS:
        raise CandidateError(f"top-level fields must be exactly {sorted(TOP_FIELDS)}")
    if candidate["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise CandidateError("candidate schema_version is incompatible")
    if candidate["reward_input_mode"] != "current_only":
        raise CandidateError("reward_input_mode must be current_only")
    if not isinstance(candidate["candidate_name"], str) or not candidate[
        "candidate_name"
    ].strip():
        raise CandidateError("candidate_name must be non-empty")
    if not isinstance(candidate["code"], str) or not candidate["code"].strip():
        raise CandidateError("code must be a non-empty string")
    allowed = allowed_source_fields(constants_metadata)
    _validate_items(candidate["features"], allowed_fields=allowed)


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


class _CandidateVisitor(ast.NodeVisitor):
    def __init__(self):
        self.accessed_fields: set[str] = set()

    def generic_visit(self, node):
        if isinstance(node, DISALLOWED_NODES):
            raise CandidateError(f"disallowed syntax: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Attribute(self, node):
        dotted = _dotted_name(node)
        if dotted not in SAFE_NUMPY_CALLS and dotted not in SAFE_NUMPY_ATTRIBUTES and dotted != "np.linalg":
            raise CandidateError(f"attribute access is not allowed: {dotted or ast.dump(node)}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTIN_CALLS:
                raise CandidateError(f"function call is not allowed: {node.func.id}")
        else:
            dotted = _dotted_name(node.func)
            if dotted not in SAFE_NUMPY_CALLS:
                raise CandidateError(f"function call is not allowed: {dotted}")
        for keyword in node.keywords:
            if keyword.arg in {"out", "like"}:
                raise CandidateError(f"NumPy keyword {keyword.arg!r} is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in {"obs", "constants"}:
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if not isinstance(key, str):
                raise CandidateError(
                    f"{node.value.id} keys must be literal strings"
                )
            self.accessed_fields.add(f"{node.value.id}.{key}")
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if _root_name(target) in {"obs", "constants"} or isinstance(
                target, (ast.Subscript, ast.Attribute)
            ):
                raise CandidateError("assignment may target local names only")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if _root_name(node.target) in {"obs", "constants"} or isinstance(
            node.target, (ast.Subscript, ast.Attribute)
        ):
            raise CandidateError("augmented assignment may target local names only")
        self.generic_visit(node)


class _LocalNameResolver(ast.NodeTransformer):
    """Resolve straight-line local aliases without attempting general data flow."""

    def __init__(self, bindings: dict[str, ast.AST], resolving: set[str] | None = None):
        self.bindings = bindings
        self.resolving = set() if resolving is None else set(resolving)

    def visit_Name(self, node):
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in self.bindings
            and node.id not in self.resolving
        ):
            return _LocalNameResolver(
                self.bindings, self.resolving | {node.id}
            ).visit(copy.deepcopy(self.bindings[node.id]))
        return node


def _resolved_expression(node: ast.AST, bindings: dict[str, ast.AST]) -> ast.AST:
    return ast.fix_missing_locations(
        _LocalNameResolver(bindings).visit(copy.deepcopy(node))
    )


def _returned_feature_expressions(function: ast.FunctionDef) -> list[ast.AST] | None:
    """Extract common list/tuple outputs with simple straight-line alias tracking."""

    bindings: dict[str, ast.AST] = {}
    returned = None
    for statement in function.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            bindings[statement.targets[0].id] = _resolved_expression(
                statement.value, bindings
            )
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ) and statement.value is not None:
            bindings[statement.target.id] = _resolved_expression(
                statement.value, bindings
            )
        elif isinstance(statement, ast.AugAssign) and isinstance(
            statement.target, ast.Name
        ):
            # AugAssign reads the previous value and then replaces the binding.
            # Resolve both operands against the pre-update environment so aliases
            # such as ``y = x; x *= x`` retain their actual, distinct meanings.
            previous = bindings.get(statement.target.id)
            if previous is None:
                return None
            bindings[statement.target.id] = ast.BinOp(
                left=_resolved_expression(previous, bindings),
                op=copy.deepcopy(statement.op),
                right=_resolved_expression(statement.value, bindings),
            )
        elif isinstance(statement, ast.Return):
            if statement.value is None:
                return None
            returned = _resolved_expression(statement.value, bindings)
            break
        elif isinstance(
            statement,
            (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match),
        ):
            # This intentionally is not a general data-flow engine.  A binding
            # modified through control flow is path-dependent, so retaining any
            # pre-branch alias can create a false duplicate verdict.  Report the
            # function as unresolved instead of pretending those expressions are
            # known.
            return None
    if returned is None:
        return None
    if isinstance(returned, ast.Call) and _dotted_name(returned.func) in {
        "np.asarray",
        "np.array",
        "np.stack",
    }:
        if not returned.args:
            return None
        returned = _resolved_expression(returned.args[0], bindings)
    if isinstance(returned, (ast.List, ast.Tuple)):
        return [_resolved_expression(value, bindings) for value in returned.elts]
    return None


def _direct_original_state_index(node: ast.AST) -> int | None:
    """Recognize a direct scalar state copy, optionally through identity casts."""

    while isinstance(node, ast.Call) and _dotted_name(node.func) in {
        "float",
        "np.float32",
        "np.float64",
        "np.asarray",
    } and len(node.args) == 1:
        node = node.args[0]
    if not isinstance(node, ast.Subscript):
        return None
    index = node.slice.value if isinstance(node.slice, ast.Constant) else None
    source = node.value
    if (
        isinstance(index, int)
        and not isinstance(index, bool)
        and isinstance(source, ast.Subscript)
        and isinstance(source.value, ast.Name)
        and source.value.id == "obs"
        and isinstance(source.slice, ast.Constant)
        and source.slice.value == "state"
    ):
        return int(index)
    return None


def _validate_explicit_feature_redundancy(
    function: ast.FunctionDef, expected_count: int
) -> dict[str, Any]:
    expressions = _returned_feature_expressions(function)
    result = {
        "status": "limited_static_check_passed",
        "coverage": "common list/tuple returns with straight-line local aliases",
        "analyzed_feature_indices": [],
    }
    if expressions is None or len(expressions) != expected_count:
        result["status"] = "not_statically_resolved"
        return result
    result["analyzed_feature_indices"] = list(range(len(expressions)))
    normalized: dict[str, int] = {}
    for index, expression in enumerate(expressions):
        state_index = _direct_original_state_index(expression)
        if state_index is not None:
            raise CandidateError(
                f"feature[{index}] is an explicit direct copy of obs['state'][{state_index}]"
            )
        key = ast.dump(expression, annotate_fields=True, include_attributes=False)
        if key in normalized:
            raise CandidateError(
                f"features[{normalized[key]}] and [{index}] use the same statically normalized output expression"
            )
        normalized[key] = index
    return result


def validate_candidate_code(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    try:
        tree = ast.parse(candidate["code"], mode="exec")
    except SyntaxError as exc:
        raise CandidateError(f"candidate code has invalid syntax: {exc}") from exc
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_functions) != 1 or any(
        not isinstance(node, ast.FunctionDef) for node in tree.body
    ):
        raise CandidateError("code must contain exactly one top-level function")
    expected = {"compute_extra_state"}
    if {node.name for node in top_functions} != expected:
        raise CandidateError(f"code functions must be exactly {sorted(expected)}")
    for function in top_functions:
        args = function.args
        if (
            [argument.arg for argument in args.args] != ["obs", "constants"]
            or args.vararg is not None
            or args.kwarg is not None
            or args.defaults
            or args.kwonlyargs
        ):
            raise CandidateError(
                f"{function.name} must have signature (obs, constants)"
            )
    visitor = _CandidateVisitor()
    visitor.visit(tree)
    allowed = allowed_source_fields(constants_metadata)
    forbidden = sorted(visitor.accessed_fields.difference(allowed))
    if forbidden:
        raise CandidateError(f"code accesses forbidden input fields: {forbidden}")
    extra_function = next(
        function for function in top_functions if function.name == "compute_extra_state"
    )
    declared = {
        value for item in candidate["features"] for value in item["source_fields"]
    }
    expressions = _returned_feature_expressions(extra_function)
    if expressions is not None and len(expressions) == len(candidate["features"]):
        for index, expression in enumerate(expressions):
            expression_visitor = _CandidateVisitor()
            expression_visitor.visit(expression)
            missing = sorted(
                expression_visitor.accessed_fields.difference(
                    candidate["features"][index]["source_fields"]
                )
            )
            if missing:
                raise CandidateError(
                    f"feature[{index}] uses source fields absent from its metadata: {missing}"
                )
    else:
        undeclared = sorted(visitor.accessed_fields.difference(declared))
        if undeclared:
            raise CandidateError(
                "code uses source fields absent from metadata and output indices "
                f"cannot be reliably resolved: {undeclared}"
            )
    redundancy = _validate_explicit_feature_redundancy(
        extra_function, len(candidate["features"])
    )
    return {
        "accessed_source_fields": sorted(visitor.accessed_fields),
        "declared_source_fields": sorted(declared),
        "ast_status": "passed",
        "explicit_feature_redundancy_check": redundancy,
    }


def validate_candidate(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    validate_candidate_schema(candidate, constants_metadata)
    return validate_candidate_code(candidate, constants_metadata)


def candidate_reward_weights(candidate: dict[str, Any]) -> np.ndarray:
    """Return the single authoritative feature-to-reward weight vector."""

    weights = np.asarray(
        [item["reward_weight"] for item in candidate["features"]], dtype=np.float64
    )
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise CandidateError("feature reward weights must be a finite vector")
    if float(np.sum(np.abs(weights), dtype=np.float64)) > 1.0 + 1e-12:
        raise CandidateError("sum(abs(feature reward weights)) must be <= 1")
    return weights


def feature_reward(extra_state: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    """Apply approved weights to the exact feature array used by the state."""

    values = np.asarray(extra_state, dtype=np.float64)
    weights = candidate_reward_weights(candidate)
    if values.ndim not in (1, 2) or values.shape[-1] != weights.size:
        raise CandidateError("extra-state values and feature reward weights do not align")
    reward = values @ weights
    if not np.all(np.isfinite(reward)):
        raise CandidateError("weighted extra reward is non-finite")
    if np.any(reward < -1.0 - NUMERIC_TOLERANCE) or np.any(
        reward > 1.0 + NUMERIC_TOLERANCE
    ):
        raise CandidateError("weighted extra reward is outside [-1,1]")
    return np.asarray(reward, dtype=np.float64)


def candidate_semantic_fingerprint(candidate: dict[str, Any]) -> str:
    """Hash meaning-bearing candidate content, ignoring name/comments/formatting."""

    payload = {
        key: value
        for key, value in candidate.items()
        if key not in {"candidate_name", "code"}
    }
    code = candidate.get("code")
    if isinstance(code, str):
        try:
            payload["code_ast"] = ast.dump(
                ast.parse(code, mode="exec"),
                annotate_fields=True,
                include_attributes=False,
            )
        except SyntaxError:
            try:
                ignored = {
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                }
                payload["code_tokens"] = [
                    (token.type, token.string)
                    for token in tokenize.generate_tokens(io.StringIO(code).readline)
                    if token.type not in ignored
                ]
            except (IndentationError, tokenize.TokenError):
                payload["code_text"] = code
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_issue(
    code: str,
    stage: str,
    location: str,
    problem: str,
    requirement: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "code": str(code),
        "stage": str(stage),
        "location": str(location),
        "problem": str(problem),
        "requirement": str(requirement),
        **details,
    }


def _schema_validation_issues(
    candidate: Any, constants_metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(code, location, problem, requirement):
        issues.append(
            _validation_issue(code, "schema", location, problem, requirement)
        )

    if not isinstance(candidate, dict):
        add(
            "SCHEMA_TOP_LEVEL_TYPE",
            "$",
            f"top-level value has type {type(candidate).__name__}",
            "Return one JSON object.",
        )
        return issues
    missing = sorted(TOP_FIELDS.difference(candidate))
    extra = sorted(set(candidate).difference(TOP_FIELDS))
    for name in missing:
        add(
            "SCHEMA_MISSING_FIELD",
            f"$.{name}",
            "required field is missing",
            f"Include the required top-level field {name!r}.",
        )
    for name in extra:
        add(
            "SCHEMA_EXTRA_FIELD",
            f"$.{name}",
            "additional top-level field is not allowed",
            "Return only the fields defined by the candidate schema.",
        )
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        add(
            "SCHEMA_VERSION",
            "$.schema_version",
            f"received {candidate.get('schema_version')!r}",
            f"Use {CANDIDATE_SCHEMA_VERSION!r}.",
        )
    if candidate.get("reward_input_mode") != "current_only":
        add(
            "SCHEMA_REWARD_INPUT_MODE",
            "$.reward_input_mode",
            f"received {candidate.get('reward_input_mode')!r}",
            "Use exactly 'current_only'.",
        )
    for name in ("candidate_name", "code"):
        value = candidate.get(name)
        if not isinstance(value, str) or not value.strip():
            add(
                "SCHEMA_NONEMPTY_TEXT",
                f"$.{name}",
                "value is not non-empty text",
                f"Provide a non-empty string for {name}.",
            )

    allowed = allowed_source_fields(constants_metadata)
    all_weights: list[float] = []
    for group_name in ("features",):
        items = candidate.get(group_name)
        if not isinstance(items, list) or not items:
            add(
                "SCHEMA_NONEMPTY_ARRAY",
                f"$.{group_name}",
                "value is not a non-empty array",
                f"Provide at least one {group_name} item.",
            )
            continue
        names: list[str] = []
        formulas: list[str] = []
        exact = ITEM_FIELDS
        for index, item in enumerate(items):
            location = f"$.{group_name}[{index}]"
            if not isinstance(item, dict):
                add(
                    "SCHEMA_ITEM_TYPE",
                    location,
                    f"item has type {type(item).__name__}",
                    "Each item must be an object.",
                )
                continue
            for name in sorted(exact.difference(item)):
                add(
                    "SCHEMA_MISSING_FIELD",
                    f"{location}.{name}",
                    "required field is missing",
                    f"Include {name!r} for every {group_name} item.",
                )
            for name in sorted(set(item).difference(exact)):
                add(
                    "SCHEMA_EXTRA_FIELD",
                    f"{location}.{name}",
                    "additional item field is not allowed",
                    "Use only fields defined by the candidate schema.",
                )
            if item.get("index") != index or isinstance(item.get("index"), bool):
                add(
                    "SCHEMA_INDEX",
                    f"{location}.index",
                    f"received {item.get('index')!r}",
                    "Indices must be consecutive integers starting at zero.",
                )
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                add(
                    "SCHEMA_NONEMPTY_TEXT",
                    f"{location}.name",
                    "name is not non-empty text",
                    "Provide a unique non-empty name.",
                )
            else:
                names.append(name)
            if item.get("dtype") != "float32":
                add(
                    "SCHEMA_DTYPE",
                    f"{location}.dtype",
                    f"received {item.get('dtype')!r}",
                    "Declare dtype as exactly 'float32'.",
                )
            for field in ("description", "formula", "missing_data_rule"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    add(
                        "SCHEMA_NONEMPTY_TEXT",
                        f"{location}.{field}",
                        "value is not non-empty text",
                        f"Provide a non-empty {field}.",
                    )
            formula = item.get("formula")
            if isinstance(formula, str) and formula.strip():
                formulas.append("".join(formula.split()).lower())
            declared_range = item.get("range")
            if not isinstance(declared_range, dict) or set(declared_range) != {
                "minimum",
                "maximum",
            }:
                add(
                    "SCHEMA_RANGE",
                    f"{location}.range",
                    "range must contain exactly minimum and maximum",
                    "Declare a finite output range using minimum and maximum.",
                )
            else:
                numeric: list[float] = []
                for bound in ("minimum", "maximum"):
                    value = declared_range.get(bound)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        add(
                            "SCHEMA_RANGE",
                            f"{location}.range.{bound}",
                            f"received non-finite/non-numeric value {value!r}",
                            "Range bounds must be finite numbers.",
                        )
                    else:
                        numeric.append(float(value))
                if len(numeric) == 2:
                    minimum, maximum = numeric
                    if minimum > maximum:
                        add(
                            "SCHEMA_RANGE_REVERSED",
                            f"{location}.range",
                            f"minimum {minimum} exceeds maximum {maximum}",
                            "Use minimum <= maximum.",
                        )
                    if minimum < 0.0 or maximum > 1.0:
                        add(
                            "SCHEMA_FEATURE_RANGE",
                            f"{location}.range",
                            f"declared [{minimum}, {maximum}]",
                            "Feature ranges must lie within [0,1].",
                        )
            sources = item.get("source_fields")
            if not isinstance(sources, list) or not sources or any(
                not isinstance(value, str) for value in sources
            ) or len(set(sources)) != len(sources):
                add(
                    "SCHEMA_SOURCE_FIELDS",
                    f"{location}.source_fields",
                    "source_fields is not a unique non-empty string array",
                    "List the actual allowed obs/constants fields used by this item.",
                )
            else:
                unknown = sorted(set(sources).difference(allowed))
                if unknown:
                    add(
                        "SCHEMA_FORBIDDEN_SOURCE_FIELD",
                        f"{location}.source_fields",
                        f"unknown fields: {unknown}",
                        "Use only fields from the supplied current-only interface.",
                    )
            weight = item.get("reward_weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                add(
                    "SCHEMA_WEIGHT",
                    f"{location}.reward_weight",
                    f"received {weight!r}",
                    "Each reward_weight must be a finite signed number; booleans are invalid.",
                )
            else:
                all_weights.append(float(weight))
        duplicate_names = sorted({value for value in names if names.count(value) > 1})
        for value in duplicate_names:
            add(
                "SCHEMA_DUPLICATE_NAME",
                f"$.{group_name}",
                f"name {value!r} is repeated",
                "Names must be unique within the group.",
            )
        duplicate_formulas = sorted(
            {value for value in formulas if formulas.count(value) > 1}
        )
        for value in duplicate_formulas:
            add(
                "SCHEMA_DUPLICATE_FORMULA",
                f"$.{group_name}",
                f"normalized formula {value!r} is repeated",
                "Do not declare duplicate formulas.",
            )
    if math.fsum(abs(value) for value in all_weights) > 1.0 + 1e-12:
        add(
            "SCHEMA_WEIGHT_L1",
            "$.features",
            f"sum(abs(weight)) is {math.fsum(abs(value) for value in all_weights):.17g}",
            "Keep sum(abs(reward_weight)) <= 1.",
        )
    return issues


class _CollectingCandidateVisitor(ast.NodeVisitor):
    def __init__(self):
        self.accessed_fields: set[str] = set()
        self.field_locations: dict[str, list[str]] = {}
        self.issues: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int, int, str]] = set()

    def add(self, code: str, node: ast.AST, problem: str, requirement: str):
        line = int(getattr(node, "lineno", 0))
        column = int(getattr(node, "col_offset", 0))
        key = (code, line, column, problem)
        if key not in self._seen:
            self._seen.add(key)
            self.issues.append(
                _validation_issue(
                    code,
                    "static",
                    f"code:{line}:{column}",
                    problem,
                    requirement,
                )
            )

    def generic_visit(self, node):
        if isinstance(node, DISALLOWED_NODES):
            self.add(
                "STATIC_DISALLOWED_SYNTAX",
                node,
                f"{type(node).__name__} is not allowed",
                "Use only the documented deterministic expression subset; imports are not allowed and np is pre-provided.",
            )
            return
        return super().generic_visit(node)

    def visit_Attribute(self, node):
        dotted = _dotted_name(node)
        if dotted not in SAFE_NUMPY_CALLS and dotted not in SAFE_NUMPY_ATTRIBUTES and dotted != "np.linalg":
            self.add(
                "STATIC_ATTRIBUTE",
                node,
                f"attribute access {dotted or ast.dump(node)} is not allowed",
                "Use only documented np operations.",
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTIN_CALLS:
                self.add(
                    "STATIC_FUNCTION_CALL",
                    node,
                    f"function call {node.func.id!r} is not allowed",
                    "Use only documented builtins and np operations.",
                )
        else:
            dotted = _dotted_name(node.func)
            if dotted not in SAFE_NUMPY_CALLS:
                self.add(
                    "STATIC_NUMPY_CALL",
                    node,
                    f"function call {dotted!r} is not allowed",
                    "Use only documented np operations.",
                )
        for keyword in node.keywords:
            if keyword.arg in {"out", "like"}:
                self.add(
                    "STATIC_NUMPY_KEYWORD",
                    keyword,
                    f"NumPy keyword {keyword.arg!r} is not allowed",
                    "Do not use output-buffer or like= mutation hooks.",
                )
        # The callable attribute is already classified above. Visit only its
        # arguments so one unsupported np call does not become a second,
        # derivative attribute error.
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in {"obs", "constants"}:
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if not isinstance(key, str):
                self.add(
                    "STATIC_DYNAMIC_FIELD",
                    node,
                    f"{node.value.id} key is not a literal string",
                    "Use literal field names from the supplied interface.",
                )
            else:
                field = f"{node.value.id}.{key}"
                self.accessed_fields.add(field)
                self.field_locations.setdefault(field, []).append(
                    f"code:{int(getattr(node, 'lineno', 0))}:{int(getattr(node, 'col_offset', 0))}"
                )
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if _root_name(target) in {"obs", "constants"} or isinstance(
                target, (ast.Subscript, ast.Attribute)
            ):
                self.add(
                    "STATIC_INPUT_MUTATION",
                    target,
                    "assignment may modify an input or non-local target",
                    "Assign only to local variable names; functions must be side-effect free.",
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if _root_name(node.target) in {"obs", "constants"} or isinstance(
            node.target, (ast.Subscript, ast.Attribute)
        ):
            self.add(
                "STATIC_INPUT_MUTATION",
                node.target,
                "augmented assignment may modify an input or non-local target",
                "Apply augmented assignment only to local variable names.",
            )
        self.generic_visit(node)


def _static_validation_issues(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    code = candidate.get("code")
    if not isinstance(code, str) or not code.strip():
        return [], None
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [
            _validation_issue(
                "STATIC_PYTHON_SYNTAX",
                "static",
                f"code:{exc.lineno or 0}:{exc.offset or 0}",
                exc.msg,
                "Return syntactically valid Python defining compute_extra_state only.",
            )
        ], None
    issues: list[dict[str, Any]] = []
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    other_top = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
    if len(top_functions) != 1 or other_top:
        issues.append(
            _validation_issue(
                "STATIC_TOP_LEVEL",
                "static",
                "$.code",
                "code does not contain exactly one top-level function definition",
                "Define only compute_extra_state at top level.",
            )
        )
    expected = {"compute_extra_state"}
    names = {node.name for node in top_functions}
    if names != expected:
        issues.append(
            _validation_issue(
                "STATIC_FUNCTION_SET",
                "static",
                "$.code",
                f"found function names {sorted(names)}",
                f"Define exactly {sorted(expected)}.",
            )
        )
    for function in top_functions:
        args = function.args
        if (
            [argument.arg for argument in args.args] != ["obs", "constants"]
            or args.vararg is not None
            or args.kwarg is not None
            or args.defaults
            or args.kwonlyargs
        ):
            issues.append(
                _validation_issue(
                    "STATIC_FUNCTION_SIGNATURE",
                    "static",
                    f"code:{function.lineno}:0 {function.name}",
                    "function signature is incompatible",
                    f"Define {function.name}(obs, constants) with no other parameters.",
                )
            )
    visitor = _CollectingCandidateVisitor()
    visitor.visit(tree)
    issues.extend(visitor.issues)
    allowed = allowed_source_fields(constants_metadata)
    for field in sorted(visitor.accessed_fields.difference(allowed)):
        issues.append(
            _validation_issue(
                "STATIC_FORBIDDEN_FIELD",
                "static",
                f"$.code field {field}",
                "code accesses a field outside the supplied interface",
                "Use only current-only obs/constants fields listed in the prompt.",
            )
        )
    features = candidate.get("features")
    declared: set[str] = set()
    if isinstance(features, list):
        for item in features:
            if isinstance(item, dict) and isinstance(item.get("source_fields"), list):
                declared.update(value for value in item["source_fields"] if isinstance(value, str))
    redundancy = None
    extra_function = next(
        (node for node in top_functions if node.name == "compute_extra_state"), None
    )
    if extra_function is not None and isinstance(features, list) and features:
        expressions = _returned_feature_expressions(extra_function)
        if expressions is not None and len(expressions) == len(features):
            for index, expression in enumerate(expressions):
                expression_visitor = _CollectingCandidateVisitor()
                expression_visitor.visit(expression)
                item = features[index]
                item_fields = set(item.get("source_fields", [])) if isinstance(item, dict) else set()
                for field in sorted(expression_visitor.accessed_fields.difference(item_fields)):
                    locations = expression_visitor.field_locations.get(field) or visitor.field_locations.get(field) or []
                    issues.append(
                        _validation_issue(
                            "STATIC_UNDECLARED_FEATURE_SOURCE",
                            "static",
                            f"$.features[{index}].source_fields",
                            f"feature[{index}] uses {field!r} at {locations or ['unknown code location']} but does not declare it",
                            f"Add {field!r} to features[{index}].source_fields; include fields used by masks, conditions, normalization, and missing-data handling.",
                            feature_index=int(index),
                            source_field=str(field),
                            source_locations=list(locations),
                            feature_mapping="resolved",
                        )
                    )
        else:
            for field in sorted(visitor.accessed_fields.difference(declared)):
                locations = visitor.field_locations.get(field, [])
                issues.append(
                    _validation_issue(
                        "STATIC_UNDECLARED_FIELD_UNRESOLVED_FEATURE",
                        "static",
                        f"$.code field {field}",
                        f"code uses {field!r} at {locations}; control/data flow prevents reliable mapping to one feature",
                        f"Declare {field!r} in every feature whose computation, mask, condition, normalization, or missing-data path uses it. No feature index is inferred here.",
                        source_field=str(field),
                        source_locations=list(locations),
                        feature_mapping="unresolved",
                    )
                )
        try:
            redundancy = _validate_explicit_feature_redundancy(
                extra_function, len(features)
            )
        except CandidateError as exc:
            issues.append(
                _validation_issue(
                    "STATIC_EXPLICIT_FEATURE_REDUNDANCY",
                    "static",
                    "compute_extra_state return",
                    str(exc),
                    "Do not directly copy an original state dimension or return the same statically confirmed expression twice.",
                )
            )
    return issues, redundancy


def validate_candidate_staged(
    candidate: Any, constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Collect independent schema/static findings without executing code."""

    schema_issues = _schema_validation_issues(candidate, constants_metadata)
    static_issues: list[dict[str, Any]] = []
    redundancy = None
    static_completed = False
    if isinstance(candidate, dict) and isinstance(candidate.get("code"), str):
        static_issues, redundancy = _static_validation_issues(
            candidate, constants_metadata
        )
        static_completed = True
    issues = schema_issues + static_issues
    return {
        "status": "passed" if not issues else "failed",
        "can_execute": not issues,
        "errors": issues,
        "checks": {
            "json": {"status": "passed", "completed": True},
            "schema": {
                "status": "passed" if not schema_issues else "failed",
                "completed": True,
                "error_count": len(schema_issues),
            },
            "static": {
                "status": (
                    "passed"
                    if static_completed and not static_issues
                    else "failed" if static_completed else "not_run"
                ),
                "completed": static_completed,
                "error_count": len(static_issues),
                "skipped_reason": (
                    None if static_completed else "code is unavailable or not text"
                ),
                "explicit_feature_redundancy_check": redundancy,
            },
            "execution": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": (
                    None if not issues else "schema/static prerequisites failed"
                ),
            },
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def execute_candidate_isolated(
    candidate: dict[str, Any],
    obs_arrays: dict[str, np.ndarray],
    constants_metadata: dict[str, Any],
    *,
    timeout: float,
    diagnostic_contract: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    static = validate_candidate(candidate, constants_metadata)
    with tempfile.TemporaryDirectory(prefix="uav-hrl-candidate-") as raw:
        directory = Path(raw)
        candidate_path = directory / "candidate.json"
        constants_path = directory / "constants.json"
        input_path = directory / "obs.npz"
        output_path = directory / "outputs.npz"
        report_path = directory / "report.json"
        diagnostic_path = directory / "diagnostic_contract.json"
        _write_json(candidate_path, candidate)
        _write_json(constants_path, runtime_constants(constants_metadata))
        np.savez_compressed(input_path, **obs_arrays)
        if diagnostic_contract is not None:
            _write_json(diagnostic_path, diagnostic_contract)
        command = [
            sys.executable,
            str(Path(__file__).with_name("llm_candidate_worker.py")),
            "--candidate",
            str(candidate_path),
            "--constants",
            str(constants_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
        if diagnostic_contract is not None:
            command.extend(("--diagnostic-contract", str(diagnostic_path)))
        try:
            worker_environment = dict(os.environ)
            for name in ("OPENAI_API_KEY", "LM_STUDIO_API_TOKEN"):
                worker_environment.pop(name, None)
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
                env=worker_environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise CandidateExecutionError(
                f"candidate worker exceeded {float(timeout):g} seconds"
            ) from exc
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        if completed.returncode != 0:
            errors = report.get("errors") or []
            message = (
                report.get("error")
                or (
                    f"{errors[0].get('location')}: {errors[0].get('problem')}"
                    if errors
                    else None
                )
                or completed.stderr.strip()
                or completed.stdout.strip()
            )
            raise CandidateExecutionError(
                f"candidate worker failed with exit code {completed.returncode}: {message}",
                report=report,
            )
        with np.load(output_path, allow_pickle=False) as archive:
            extra = np.asarray(archive["extra_state"])
            reward = np.asarray(archive["extra_reward"])
        return extra, reward, {**static, **report}


def candidate_numeric_diagnostics(
    original_state: np.ndarray,
    extra_state: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result = {"features": [], "warnings": []}
    for index, definition in enumerate(candidate["features"]):
        column = np.asarray(extra_state[:, index], dtype=np.float64)
        item = {
            "index": index,
            "name": definition["name"],
            "reward_weight": float(definition["reward_weight"]),
            "minimum": float(np.min(column)),
            "maximum": float(np.max(column)),
            "mean": float(np.mean(column)),
            "standard_deviation": float(np.std(column)),
            "constant_on_fixed_samples": bool(np.ptp(column) <= NUMERIC_TOLERANCE),
        }
        result["features"].append(item)
        if item["constant_on_fixed_samples"]:
            result["warnings"].append(
                f"features[{index}] is constant on the fixed samples; this is diagnostic, not proof of semantic uselessness"
            )
    state64 = np.asarray(original_state, dtype=np.float64)
    for index in range(extra_state.shape[1]):
        matches = np.flatnonzero(
            np.all(
                np.isclose(
                    state64,
                    np.asarray(extra_state[:, index], dtype=np.float64)[:, None],
                    rtol=0.0,
                    atol=NUMERIC_TOLERANCE,
                ),
                axis=0,
            )
        )
        if matches.size:
            result["warnings"].append(
                f"feature[{index}] numerically duplicates original state indices {matches.tolist()} on fixed samples"
            )
    for left in range(extra_state.shape[1]):
        for right in range(left + 1, extra_state.shape[1]):
            if np.allclose(
                extra_state[:, left],
                extra_state[:, right],
                rtol=0.0,
                atol=NUMERIC_TOLERANCE,
            ):
                result["warnings"].append(
                    f"features[{left}] and [{right}] are numerically identical on fixed samples"
                )
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_approved_artifact(
    run_directory: str | Path,
    *,
    candidate: dict[str, Any],
    constants_metadata: dict[str, Any],
    validation_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    approved = Path(run_directory) / "approved"
    approved.mkdir()
    candidate_path = approved / "candidate.json"
    code_path = approved / "candidate.py"
    constants_path = approved / "constants.json"
    validation_path = approved / "validation_report.json"
    evaluation_path = approved / "evaluation_report.json"
    _write_json(candidate_path, candidate)
    code_path.write_text(candidate["code"], encoding="utf-8")
    _write_json(constants_path, constants_metadata)
    _write_json(validation_path, validation_report)
    _write_json(evaluation_path, evaluation_report)
    files = {
        path.name: _file_hash(path)
        for path in (
            candidate_path,
            code_path,
            constants_path,
            validation_path,
            evaluation_path,
        )
    }
    artifact = {
        "schema_version": APPROVED_ARTIFACT_SCHEMA_VERSION,
        "status": "approved",
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "observation_interface_version": OBS_INTERFACE_VERSION,
        "candidate_name": candidate["candidate_name"],
        "feature_count": len(candidate["features"]),
        "feature_order": [item["name"] for item in candidate["features"]],
        "feature_reward_weights": [
            float(item["reward_weight"]) for item in candidate["features"]
        ],
        "files_sha256": files,
        "provenance": provenance,
    }
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(approved / "artifact.json", artifact)
    return approved


@dataclass(frozen=True)
class ApprovedDesign:
    directory: Path
    artifact: dict[str, Any]
    candidate: dict[str, Any]
    constants_metadata: dict[str, Any]

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.artifact["feature_reward_weights"], dtype=np.float64)

    def evaluate_obs_arrays(
        self, obs_arrays: dict[str, np.ndarray], *, timeout: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray]:
        extra, reward, _ = execute_candidate_isolated(
            self.candidate,
            obs_arrays,
            self.constants_metadata,
            timeout=timeout,
        )
        return extra, reward

    def evaluate_fixed_samples(
        self, fixed_arrays: dict[str, np.ndarray], *, timeout: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate_obs_arrays(build_obs_arrays(fixed_arrays), timeout=timeout)


def load_approved_design(directory: str | Path) -> ApprovedDesign:
    directory = Path(directory).resolve()
    artifact_path = directory / "artifact.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"approved artifact metadata is missing: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_version = artifact.get("schema_version")
    if artifact_version != APPROVED_ARTIFACT_SCHEMA_VERSION:
        raise CandidateError(
            "approved artifact contract is incompatible with the shared-feature "
            f"runtime: received {artifact_version!r}, required "
            f"{APPROVED_ARTIFACT_SCHEMA_VERSION!r}; redesign and retrain the LLM method"
        )
    if artifact.get("status") != "approved":
        raise CandidateError("artifact status is not approved")
    content_hash = artifact.pop("content_sha256", None)
    expected = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    artifact["content_sha256"] = content_hash
    if content_hash != expected:
        raise CandidateError("approved artifact metadata content hash changed")
    for name, digest in artifact.get("files_sha256", {}).items():
        path = directory / name
        if not path.is_file() or _file_hash(path) != digest:
            raise CandidateError(f"approved artifact file changed: {name}")
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    constants = json.loads((directory / "constants.json").read_text(encoding="utf-8"))
    validate_candidate(candidate, constants)
    if int(artifact.get("feature_count", -1)) != len(candidate["features"]):
        raise CandidateError("approved artifact feature count is inconsistent")
    if not np.array_equal(
        np.asarray(artifact.get("feature_reward_weights"), dtype=np.float64),
        candidate_reward_weights(candidate),
    ):
        raise CandidateError("approved artifact feature reward weights are inconsistent")
    return ApprovedDesign(directory, artifact, candidate, constants)
