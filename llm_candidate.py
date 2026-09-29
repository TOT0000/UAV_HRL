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


_SOURCE_INPUT = "input_shared"
_SOURCE_OWNED = "local_owned"
_SOURCE_IMMUTABLE = "immutable_value"
_SOURCE_CONTAINER = "local_container"
_SOURCE_UNKNOWN = "unknown"
_INPUT_SOURCES = frozenset({_SOURCE_INPUT})
_OWNED_SOURCES = frozenset({_SOURCE_OWNED})
_IMMUTABLE_SOURCES = frozenset({_SOURCE_IMMUTABLE})
_CONTAINER_SOURCES = frozenset({_SOURCE_CONTAINER})
_UNKNOWN_SOURCES = frozenset({_SOURCE_UNKNOWN})
_SAFE_AUGMENTED_NAME_SOURCES = frozenset(
    {_SOURCE_OWNED, _SOURCE_IMMUTABLE}
)
_LOCAL_ALLOCATING_NUMPY_CALLS = {
    "np.abs",
    "np.arange",
    "np.array",
    "np.clip",
    "np.concatenate",
    "np.exp",
    "np.log",
    "np.log1p",
    "np.maximum",
    "np.minimum",
    "np.ones",
    "np.sqrt",
    "np.stack",
    "np.where",
    "np.zeros",
}
_LOCAL_NUMPY_VALUE_OR_ARRAY_CALLS = {
    "np.all",
    "np.any",
    "np.bool_",
    "np.count_nonzero",
    "np.float32",
    "np.float64",
    "np.int32",
    "np.int64",
    "np.isfinite",
    "np.linalg.norm",
    "np.mean",
    "np.sum",
}


def _operation_identities(tree: ast.AST, source: str) -> dict[int, dict[str, Any]]:
    """Return line-insensitive structural identities for candidate operations."""

    source_lines = source.splitlines()
    identities: dict[int, dict[str, Any]] = {}

    def walk(node: ast.AST, path: tuple[str, ...], function_name: str | None):
        current_function = (
            node.name if isinstance(node, ast.FunctionDef) else function_name
        )
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        payload = {
            "candidate_function": current_function,
            "function_structural_path": list(path),
            "normalized_ast": normalized,
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        line = int(getattr(node, "lineno", 0))
        payload["source"] = (
            source_lines[line - 1].strip()
            if 0 < line <= len(source_lines)
            else None
        )
        identities[id(node)] = payload
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                walk(value, path + (field,), current_function)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        walk(child, path + (f"{field}[{index}]",), current_function)

    walk(tree, (), None)
    return identities


def _merge_source_environments(
    *environments: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    names = set().union(*(environment.keys() for environment in environments))
    merged = {}
    for name in names:
        sources: set[str] = set()
        for environment in environments:
            sources.update(environment.get(name, _UNKNOWN_SOURCES))
        merged[name] = frozenset(sources)
    return merged


class _MutationSourceAnalyzer:
    """Conservative, function-local ownership analysis for write targets."""

    def __init__(
        self,
        source: str,
        identities: dict[int, dict[str, Any]],
    ):
        self.source = source
        self.source_lines = source.splitlines()
        self.identities = identities
        self.issues: list[dict[str, Any]] = []
        self._issue_indices: dict[tuple[str, str, str], int] = {}

    @staticmethod
    def _new_numeric_result(
        operand_sources: list[frozenset[str]],
    ) -> frozenset[str]:
        combined = set().union(*operand_sources) if operand_sources else set()
        if _SOURCE_UNKNOWN in combined or _SOURCE_CONTAINER in combined:
            return _UNKNOWN_SOURCES
        if _SOURCE_INPUT in combined or _SOURCE_OWNED in combined:
            return _OWNED_SOURCES
        if combined and combined <= {_SOURCE_IMMUTABLE}:
            return _IMMUTABLE_SOURCES
        return _UNKNOWN_SOURCES

    def expression_sources(
        self, node: ast.AST | None, environment: dict[str, frozenset[str]]
    ) -> frozenset[str]:
        if node is None:
            return _UNKNOWN_SOURCES
        if isinstance(node, ast.Name):
            return environment.get(node.id, _UNKNOWN_SOURCES)
        if isinstance(node, ast.Constant):
            return _IMMUTABLE_SOURCES
        if isinstance(node, ast.Subscript):
            return self.expression_sources(node.value, environment)
        if isinstance(node, ast.Attribute):
            return self.expression_sources(node.value, environment)
        if isinstance(node, ast.IfExp):
            return frozenset(
                set(self.expression_sources(node.body, environment))
                | set(self.expression_sources(node.orelse, environment))
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            children = []
            if isinstance(node, ast.Dict):
                children = [*node.keys, *node.values]
            else:
                children = list(node.elts)
            sources = {_SOURCE_CONTAINER}
            for child in children:
                child_sources = self.expression_sources(child, environment)
                if _SOURCE_INPUT in child_sources:
                    sources.add(_SOURCE_INPUT)
                if _SOURCE_UNKNOWN in child_sources:
                    sources.add(_SOURCE_UNKNOWN)
            return frozenset(sources)
        if isinstance(node, ast.BoolOp):
            # Python and/or return one operand rather than a fresh bool.
            return frozenset(
                set().union(
                    *(
                        self.expression_sources(value, environment)
                        for value in node.values
                    )
                )
            )
        if isinstance(node, ast.BinOp):
            return self._new_numeric_result(
                [
                    self.expression_sources(node.left, environment),
                    self.expression_sources(node.right, environment),
                ]
            )
        if isinstance(node, ast.UnaryOp):
            return self._new_numeric_result(
                [self.expression_sources(node.operand, environment)]
            )
        if isinstance(node, ast.Compare):
            return self._new_numeric_result(
                [
                    self.expression_sources(node.left, environment),
                    *(
                        self.expression_sources(value, environment)
                        for value in node.comparators
                    ),
                ]
            )
        if isinstance(node, ast.JoinedStr):
            return _IMMUTABLE_SOURCES
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted == "np.asarray":
                return (
                    self.expression_sources(node.args[0], environment)
                    if node.args
                    else _UNKNOWN_SOURCES
                )
            if dotted == "np.array":
                copy_keyword = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "copy"),
                    None,
                )
                if isinstance(copy_keyword, ast.Constant) and copy_keyword.value is False:
                    return (
                        self.expression_sources(node.args[0], environment)
                        if node.args
                        else _UNKNOWN_SOURCES
                    )
                if copy_keyword is not None and not (
                    isinstance(copy_keyword, ast.Constant)
                    and copy_keyword.value is True
                ):
                    return _UNKNOWN_SOURCES
                return _OWNED_SOURCES
            if dotted in _LOCAL_ALLOCATING_NUMPY_CALLS:
                return _OWNED_SOURCES
            if dotted in _LOCAL_NUMPY_VALUE_OR_ARRAY_CALLS:
                return frozenset({_SOURCE_IMMUTABLE, _SOURCE_OWNED})
            if isinstance(node.func, ast.Name) and node.func.id == "list":
                return _CONTAINER_SOURCES
            if isinstance(node.func, ast.Name) and node.func.id in {
                "bool",
                "float",
                "int",
                "len",
            }:
                return _IMMUTABLE_SOURCES
            if isinstance(node.func, ast.Name) and node.func.id in {"abs", "sum"}:
                return self._new_numeric_result(
                    [
                        self.expression_sources(argument, environment)
                        for argument in node.args
                    ]
                )
            if isinstance(node.func, ast.Name) and node.func.id in {"max", "min"}:
                sources = set()
                for argument in node.args:
                    sources.update(self.expression_sources(argument, environment))
                return frozenset(sources or {_SOURCE_UNKNOWN})
            if isinstance(node.func, ast.Name) and node.func.id in {
                "tuple",
                "enumerate",
                "zip",
            }:
                sources = {_SOURCE_CONTAINER}
                for argument in node.args:
                    argument_sources = self.expression_sources(argument, environment)
                    if _SOURCE_INPUT in argument_sources:
                        sources.add(_SOURCE_INPUT)
                    if _SOURCE_UNKNOWN in argument_sources:
                        sources.add(_SOURCE_UNKNOWN)
                return frozenset(sources)
            if isinstance(node.func, ast.Name) and node.func.id == "range":
                return _IMMUTABLE_SOURCES
            return _UNKNOWN_SOURCES
        return _UNKNOWN_SOURCES

    def _issue(
        self,
        target: ast.AST,
        sources: frozenset[str],
        *,
        augmented: bool,
        code_override: str | None = None,
        problem_override: str | None = None,
        requirement_override: str | None = None,
    ):
        identity = self.identities.get(id(target), {})
        line = int(getattr(target, "lineno", 0))
        code_line = (
            self.source_lines[line - 1].strip()
            if 0 < line <= len(self.source_lines)
            else None
        )
        target_text = ast.unparse(target)
        if code_override is not None:
            code = code_override
            problem = str(problem_override)
            requirement = str(requirement_override)
        elif isinstance(target, ast.Attribute):
            code = "STATIC_ATTRIBUTE_MUTATION"
            problem = f"attribute assignment to {target_text!r} is not allowed"
            requirement = (
                "Keep object attributes unchanged. Use a local array allocated by an "
                "allowed NumPy constructor and write only its indexed elements."
            )
        elif _SOURCE_INPUT in sources:
            code = "STATIC_INPUT_MUTATION"
            problem = (
                f"{'augmented assignment' if augmented else 'indexed assignment'} "
                f"target {target_text!r} may share data with obs or constants"
            )
            requirement = (
                "Do not write to obs, constants, their aliases or slices, or an "
                "np.asarray view. Allocate an independent buffer with np.zeros, "
                "np.ones, np.arange, or np.array(...), then write to that buffer."
            )
        else:
            code = "STATIC_UNCONFIRMED_WRITE_TARGET"
            problem = (
                "the validator cannot confirm that "
                f"{'augmented assignment' if augmented else 'indexed assignment'} "
                f"target {target_text!r} "
                "is independent of obs and constants"
            )
            requirement = (
                "Bind the target directly, on every control-flow path, to a supported "
                "independent allocation such as np.zeros, np.ones, np.arange, or "
                "np.array(...) before writing through it."
            )
        issue = _validation_issue(
            code,
            "static",
            f"code:{line}:{int(getattr(target, 'col_offset', 0))}",
            problem,
            requirement,
            validation_check="static.mutation",
            candidate_function=identity.get("candidate_function"),
            candidate_line=line,
            candidate_source_line=code_line,
            write_target=target_text,
            write_kind="augmented_assignment" if augmented else "assignment",
            target_sources=sorted(sources),
            operation_fingerprint=identity.get("fingerprint"),
            operation_identity=identity or None,
        )
        issue_key = (
            str(identity.get("fingerprint", f"line:{line}")),
            "augmented" if augmented else "assignment",
            code,
        )
        if issue_key in self._issue_indices:
            self.issues[self._issue_indices[issue_key]] = issue
        else:
            self._issue_indices[issue_key] = len(self.issues)
            self.issues.append(issue)

    def validate_write(
        self,
        target: ast.AST,
        environment: dict[str, frozenset[str]],
        *,
        augmented: bool,
        value_sources: frozenset[str] | None = None,
    ):
        if isinstance(target, ast.Name) and not augmented:
            return
        if isinstance(target, ast.Name):
            sources = environment.get(target.id, _UNKNOWN_SOURCES)
        elif isinstance(target, ast.Attribute):
            sources = self.expression_sources(target.value, environment)
        elif isinstance(target, ast.Subscript):
            sources = self.expression_sources(target.value, environment)
        else:
            sources = _UNKNOWN_SOURCES
        if isinstance(target, ast.Name) and augmented:
            right_sources = value_sources or _UNKNOWN_SOURCES
            if (
                sources
                and sources <= _SAFE_AUGMENTED_NAME_SOURCES
                and _SOURCE_UNKNOWN not in right_sources
                and _SOURCE_CONTAINER not in right_sources
            ):
                return
        elif (
            isinstance(target, ast.Subscript)
            and sources == _CONTAINER_SOURCES
            and value_sources is not None
        ):
            if value_sources == _IMMUTABLE_SOURCES:
                return
            self._issue(
                target,
                sources,
                augmented=augmented,
                code_override="STATIC_UNCONFIRMED_CONTAINER_REFERENCE",
                problem_override=(
                    f"assignment through local container target {ast.unparse(target)!r} "
                    "may retain a mutable or input-backed reference"
                ),
                requirement_override=(
                    "Only store proven immutable scalar values through local container "
                    "indices. Use a directly allocated NumPy buffer for numeric copies; "
                    "nested container-reference mutation is not statically supported."
                ),
            )
            return
        if isinstance(target, ast.Attribute) or sources != _OWNED_SOURCES:
            self._issue(target, sources, augmented=augmented)

    @staticmethod
    def _bind_target(
        target: ast.AST,
        sources: frozenset[str],
        environment: dict[str, frozenset[str]],
    ):
        if isinstance(target, ast.Name):
            environment[target.id] = sources
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                _MutationSourceAnalyzer._bind_target(
                    element, sources, environment
                )

    def _record_container_contents(
        self,
        target: ast.AST,
        value_sources: frozenset[str],
        environment: dict[str, frozenset[str]],
    ):
        if not isinstance(target, ast.Subscript):
            return
        root = _root_name(target)
        if root is None:
            return
        existing = environment.get(root, _UNKNOWN_SOURCES)
        if _SOURCE_CONTAINER in existing:
            risky_contents = set(value_sources).difference({_SOURCE_IMMUTABLE})
            if risky_contents:
                environment[root] = frozenset(set(existing) | risky_contents)

    @staticmethod
    def _augmented_result_sources(
        left_sources: frozenset[str], right_sources: frozenset[str]
    ) -> frozenset[str]:
        if _SOURCE_INPUT in left_sources:
            return left_sources
        if _SOURCE_UNKNOWN in left_sources or _SOURCE_UNKNOWN in right_sources:
            return _UNKNOWN_SOURCES
        result = set(left_sources)
        if _SOURCE_INPUT in right_sources or _SOURCE_OWNED in right_sources:
            result.add(_SOURCE_OWNED)
        return frozenset(result or {_SOURCE_UNKNOWN})

    def _loop_target_sources(
        self, iterator: ast.AST, environment: dict[str, frozenset[str]]
    ) -> frozenset[str]:
        if isinstance(iterator, ast.Call) and _dotted_name(iterator.func) == "range":
            return _IMMUTABLE_SOURCES
        sources = self.expression_sources(iterator, environment)
        if _SOURCE_INPUT in sources:
            return frozenset({_SOURCE_INPUT, _SOURCE_IMMUTABLE})
        return sources

    def _analyze_loop(
        self,
        statement: ast.For | ast.While,
        incoming: dict[str, frozenset[str]],
    ) -> dict[str, frozenset[str]]:
        # The finite source lattice grows monotonically at the loop head. This
        # covers zero iterations and every back-edge without unrolling runtime
        # iteration counts.
        loop_head = dict(incoming)
        maximum_iterations = max(
            2,
            5 * (len(set(loop_head)) + len(list(ast.walk(statement))) + 1),
        )
        for _ in range(maximum_iterations):
            iteration_environment = dict(loop_head)
            if isinstance(statement, ast.For):
                self._bind_target(
                    statement.target,
                    self._loop_target_sources(statement.iter, iteration_environment),
                    iteration_environment,
                )
            body_exit = self.analyze_statements(
                statement.body, iteration_environment
            )
            next_head = _merge_source_environments(incoming, body_exit)
            if next_head == loop_head:
                loop_head = next_head
                break
            loop_head = next_head
        else:
            # Defensive termination fallback: uncertainty is safer than using
            # a stale pre-loop binding.
            loop_head = {
                name: frozenset(set(sources) | {_SOURCE_UNKNOWN})
                for name, sources in loop_head.items()
            }
        if statement.orelse:
            return _merge_source_environments(
                loop_head,
                self.analyze_statements(statement.orelse, dict(loop_head)),
            )
        return loop_head

    def analyze_statements(
        self,
        statements: list[ast.stmt],
        incoming: dict[str, frozenset[str]],
    ) -> dict[str, frozenset[str]]:
        environment = dict(incoming)
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                sources = self.expression_sources(value, environment)
                for target in targets:
                    self.validate_write(
                        target,
                        environment,
                        augmented=False,
                        value_sources=sources,
                    )
                    self._record_container_contents(target, sources, environment)
                    self._bind_target(target, sources, environment)
            elif isinstance(statement, ast.AugAssign):
                right_sources = self.expression_sources(statement.value, environment)
                left_sources = (
                    environment.get(statement.target.id, _UNKNOWN_SOURCES)
                    if isinstance(statement.target, ast.Name)
                    else self.expression_sources(statement.target, environment)
                )
                self.validate_write(
                    statement.target,
                    environment,
                    augmented=True,
                    value_sources=right_sources,
                )
                if isinstance(statement.target, ast.Name):
                    environment[statement.target.id] = self._augmented_result_sources(
                        left_sources, right_sources
                    )
            elif isinstance(statement, ast.If):
                body = self.analyze_statements(statement.body, dict(environment))
                orelse = self.analyze_statements(statement.orelse, dict(environment))
                environment = _merge_source_environments(body, orelse)
            elif isinstance(statement, (ast.For, ast.While)):
                environment = self._analyze_loop(statement, environment)
            elif isinstance(statement, ast.FunctionDef):
                function_environment = {
                    "obs": _INPUT_SOURCES,
                    "constants": _INPUT_SOURCES,
                }
                self.analyze_statements(statement.body, function_environment)
        return environment


def _mutation_validation_issues(
    tree: ast.AST, source: str, identities: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    analyzer = _MutationSourceAnalyzer(source, identities)
    analyzer.analyze_statements(list(getattr(tree, "body", [])), {})
    return analyzer.issues


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
    identities = _operation_identities(tree, candidate["code"])
    mutation_issues = _mutation_validation_issues(
        tree, candidate["code"], identities
    )
    if mutation_issues:
        issue = mutation_issues[0]
        raise CandidateError(
            f"{issue['location']}: {issue['problem']}; {issue['requirement']}"
        )
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
    def __init__(
        self,
        source: str = "",
        identities: dict[int, dict[str, Any]] | None = None,
    ):
        self.accessed_fields: set[str] = set()
        self.field_locations: dict[str, list[str]] = {}
        self.issues: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int, int, str]] = set()
        self.source_lines = source.splitlines()
        self.identities = identities or {}
        self.current_function: str | None = None

    def visit_FunctionDef(self, node):
        previous = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = previous

    def add(self, code: str, node: ast.AST, problem: str, requirement: str):
        line = int(getattr(node, "lineno", 0))
        column = int(getattr(node, "col_offset", 0))
        key = (code, line, column, problem)
        if key not in self._seen:
            self._seen.add(key)
            identity = self.identities.get(id(node), {})
            validation_check = (
                "static.source_fields"
                if code in {"STATIC_DYNAMIC_FIELD", "STATIC_FORBIDDEN_FIELD"}
                else "static.allowed_operations"
            )
            self.issues.append(
                _validation_issue(
                    code,
                    "static",
                    f"code:{line}:{column}",
                    problem,
                    requirement,
                    validation_check=validation_check,
                    candidate_function=(
                        identity.get("candidate_function") or self.current_function
                    ),
                    candidate_line=line,
                    candidate_source_line=(
                        self.source_lines[line - 1].strip()
                        if 0 < line <= len(self.source_lines)
                        else None
                    ),
                    operation_fingerprint=identity.get("fingerprint"),
                    operation_identity=identity or None,
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
                if dotted and dotted.endswith(".append"):
                    requirement = (
                        "Array/list append methods are outside the candidate operation "
                        "whitelist. Preallocate an independently owned local NumPy array "
                        "with np.zeros, np.ones, np.arange, or np.array, then fill it by "
                        "indexed assignment. Do not write through obs, constants, or their "
                        "aliases/views."
                    )
                else:
                    requirement = "Use only documented builtins and np operations."
                self.add(
                    "STATIC_NUMPY_CALL",
                    node,
                    f"function call {dotted!r} is not allowed",
                    requirement,
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

def _static_validation_issues(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, dict[str, Any]],
]:
    code = candidate.get("code")
    if not isinstance(code, str) or not code.strip():
        return [], None, {}
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
                validation_check="static.syntax",
                candidate_line=int(exc.lineno or 0),
                candidate_source_line=(
                    code.splitlines()[int(exc.lineno) - 1].strip()
                    if exc.lineno and int(exc.lineno) <= len(code.splitlines())
                    else None
                ),
            )
        ], None, {
            "syntax": {"status": "failed", "completed": True},
            "structure": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
            "allowed_operations": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
            "mutation": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
            "source_fields": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
            "feature_sources": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
            "redundancy": {
                "status": "not_run",
                "completed": False,
                "skipped_reason": "Python syntax is invalid",
            },
        }
    identities = _operation_identities(tree, code)
    issues: list[dict[str, Any]] = []
    subchecks: dict[str, dict[str, Any]] = {
        name: {
            "status": "passed" if completed else "not_run",
            "completed": completed,
            "error_count": 0,
        }
        for name, completed in {
            "syntax": True,
            "structure": True,
            "allowed_operations": True,
            "mutation": True,
            "source_fields": True,
            "feature_sources": False,
            "redundancy": False,
        }.items()
    }
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
                validation_check="static.structure",
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
                validation_check="static.structure",
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
                    validation_check="static.structure",
                    candidate_function=function.name,
                    candidate_line=int(function.lineno),
                    candidate_source_line=code.splitlines()[function.lineno - 1].strip(),
                )
            )
    visitor = _CollectingCandidateVisitor(code, identities)
    visitor.visit(tree)
    issues.extend(visitor.issues)
    issues.extend(_mutation_validation_issues(tree, code, identities))
    allowed = allowed_source_fields(constants_metadata)
    for field in sorted(visitor.accessed_fields.difference(allowed)):
        issues.append(
            _validation_issue(
                "STATIC_FORBIDDEN_FIELD",
                "static",
                f"$.code field {field}",
                "code accesses a field outside the supplied interface",
                "Use only current-only obs/constants fields listed in the prompt.",
                validation_check="static.source_fields",
                source_field=str(field),
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
            subchecks["feature_sources"].update(
                {"status": "passed", "completed": True}
            )
            for index, expression in enumerate(expressions):
                expression_visitor = _CollectingCandidateVisitor(code, identities)
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
                            validation_check="static.feature_sources",
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
                        validation_check="static.feature_sources",
                    )
                )
        try:
            redundancy = _validate_explicit_feature_redundancy(
                extra_function, len(features)
            )
            if redundancy.get("status") != "not_statically_resolved":
                subchecks["redundancy"].update(
                    {"status": "passed", "completed": True}
                )
        except CandidateError as exc:
            subchecks["redundancy"].update(
                {"status": "failed", "completed": True}
            )
            issues.append(
                _validation_issue(
                    "STATIC_EXPLICIT_FEATURE_REDUNDANCY",
                    "static",
                    "compute_extra_state return",
                    str(exc),
                    "Do not directly copy an original state dimension or return the same statically confirmed expression twice.",
                    validation_check="static.redundancy",
                )
            )
    for name, check in subchecks.items():
        check_id = f"static.{name}"
        error_count = sum(
            1 for issue in issues if issue.get("validation_check") == check_id
        )
        check["error_count"] = error_count
        if error_count:
            check["status"] = "failed"
    return issues, redundancy, subchecks


def validate_candidate_staged(
    candidate: Any, constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Collect independent schema/static findings without executing code."""

    schema_issues = _schema_validation_issues(candidate, constants_metadata)
    static_issues: list[dict[str, Any]] = []
    redundancy = None
    static_completed = False
    if isinstance(candidate, dict) and isinstance(candidate.get("code"), str):
        static_issues, redundancy, static_subchecks = _static_validation_issues(
            candidate, constants_metadata
        )
        static_completed = True
    else:
        static_subchecks = {}
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
                "subchecks": static_subchecks,
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
