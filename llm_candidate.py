"""Strict candidate parsing, static checks, isolated execution, and artifacts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
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
    "reward_terms",
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
}
REWARD_ITEM_FIELDS = ITEM_FIELDS | {"weight"}
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
    pass


def _reject_duplicate_key(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_candidate_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise CandidateError("candidate response is empty")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CandidateError(f"non-finite JSON number: {token}")
            ),
        )
    except CandidateError:
        raise
    except json.JSONDecodeError as exc:
        raise CandidateError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateError("candidate response must be one JSON object")
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
    reward: bool,
    allowed_fields: set[str],
) -> None:
    label = "reward_terms" if reward else "features"
    if not isinstance(items, list) or not items:
        raise CandidateError(f"{label} must contain at least one item")
    names = []
    formulas = []
    weights = []
    exact_fields = REWARD_ITEM_FIELDS if reward else ITEM_FIELDS
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
        if reward:
            if minimum != 0.0 or maximum != 1.0:
                raise CandidateError("every reward term must declare range [0,1]")
        elif minimum < -1.0 or maximum > 1.0:
            raise CandidateError("feature declared ranges must lie within [-1,1]")
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
        if reward:
            weights.append(_finite_number(item["weight"], f"{label}[{index}].weight"))
    if len(set(names)) != len(names):
        raise CandidateError(f"{label} names must be unique")
    duplicate_formulas = sorted(
        formula for formula in set(formulas) if formulas.count(formula) > 1
    )
    if duplicate_formulas:
        raise CandidateError(f"{label} contains duplicate formulas")
    if reward and math.fsum(abs(value) for value in weights) > 1.0 + 1e-12:
        raise CandidateError("sum(abs(reward term weights)) must be <= 1")


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
    _validate_items(candidate["features"], reward=False, allowed_fields=allowed)
    _validate_items(candidate["reward_terms"], reward=True, allowed_fields=allowed)


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


def validate_candidate_code(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    try:
        tree = ast.parse(candidate["code"], mode="exec")
    except SyntaxError as exc:
        raise CandidateError(f"candidate code has invalid syntax: {exc}") from exc
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_functions) != 2 or any(
        not isinstance(node, ast.FunctionDef) for node in tree.body
    ):
        raise CandidateError("code must contain exactly the two top-level functions")
    expected = {"compute_extra_state", "compute_reward_terms"}
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
    declared = {
        value
        for group in (candidate["features"], candidate["reward_terms"])
        for item in group
        for value in item["source_fields"]
    }
    undeclared = sorted(visitor.accessed_fields.difference(declared))
    if undeclared:
        raise CandidateError(f"code uses source fields absent from metadata: {undeclared}")
    return {
        "accessed_source_fields": sorted(visitor.accessed_fields),
        "declared_source_fields": sorted(declared),
        "ast_status": "passed",
    }


def validate_candidate(
    candidate: dict[str, Any], constants_metadata: dict[str, Any]
) -> dict[str, Any]:
    validate_candidate_schema(candidate, constants_metadata)
    return validate_candidate_code(candidate, constants_metadata)


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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    static = validate_candidate(candidate, constants_metadata)
    with tempfile.TemporaryDirectory(prefix="uav-hrl-candidate-") as raw:
        directory = Path(raw)
        candidate_path = directory / "candidate.json"
        constants_path = directory / "constants.json"
        input_path = directory / "obs.npz"
        output_path = directory / "outputs.npz"
        report_path = directory / "report.json"
        _write_json(candidate_path, candidate)
        _write_json(constants_path, runtime_constants(constants_metadata))
        np.savez_compressed(input_path, **obs_arrays)
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
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
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
            message = report.get("error") or completed.stderr.strip() or completed.stdout.strip()
            raise CandidateExecutionError(
                f"candidate worker failed with exit code {completed.returncode}: {message}"
            )
        with np.load(output_path, allow_pickle=False) as archive:
            extra = np.asarray(archive["extra_state"])
            terms = np.asarray(archive["reward_terms"])
        return extra, terms, {**static, **report}


def candidate_numeric_diagnostics(
    original_state: np.ndarray,
    extra_state: np.ndarray,
    reward_terms: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result = {"features": [], "reward_terms": [], "warnings": []}
    for kind, values, definitions in (
        ("features", extra_state, candidate["features"]),
        ("reward_terms", reward_terms, candidate["reward_terms"]),
    ):
        for index, definition in enumerate(definitions):
            column = np.asarray(values[:, index], dtype=np.float64)
            item = {
                "index": index,
                "name": definition["name"],
                "minimum": float(np.min(column)),
                "maximum": float(np.max(column)),
                "mean": float(np.mean(column)),
                "standard_deviation": float(np.std(column)),
                "constant_on_fixed_samples": bool(np.ptp(column) <= NUMERIC_TOLERANCE),
            }
            result[kind].append(item)
            if item["constant_on_fixed_samples"]:
                result["warnings"].append(
                    f"{kind}[{index}] is constant on the fixed samples; this is diagnostic, not proof of semantic uselessness"
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
        "reward_term_count": len(candidate["reward_terms"]),
        "reward_term_order": [item["name"] for item in candidate["reward_terms"]],
        "reward_weights": [float(item["weight"]) for item in candidate["reward_terms"]],
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
        return np.asarray(self.artifact["reward_weights"], dtype=np.float64)

    def evaluate_obs_arrays(
        self, obs_arrays: dict[str, np.ndarray], *, timeout: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        extra, terms, _ = execute_candidate_isolated(
            self.candidate,
            obs_arrays,
            self.constants_metadata,
            timeout=timeout,
        )
        reward = np.asarray(terms, dtype=np.float64) @ self.weights
        return extra, terms, reward

    def evaluate_fixed_samples(
        self, fixed_arrays: dict[str, np.ndarray], *, timeout: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.evaluate_obs_arrays(build_obs_arrays(fixed_arrays), timeout=timeout)


def load_approved_design(directory: str | Path) -> ApprovedDesign:
    directory = Path(directory).resolve()
    artifact_path = directory / "artifact.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"approved artifact metadata is missing: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema_version") != APPROVED_ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "approved"
    ):
        raise CandidateError("artifact is not an approved compatible design")
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
    return ApprovedDesign(directory, artifact, candidate, constants)
