"""Validated LLM potential artifacts and potential-based shaping runtime.

This module deliberately uses a distinct artifact contract from the older
current-state LLM feature/reward method.  Candidate code sees one observation
snapshot at a time; the host owns weighting, temporal differencing, terminal
handling, and reward composition.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from llm_candidate import (
    CandidateError,
    execute_candidate_isolated,
    normalize_candidate_submission,
)
from llm_candidate_worker import SAFE_BUILTINS, _run_function, _validate_output_ranges
from llm_design_contract import OBS_INTERFACE_VERSION, obs_keys_for_interface, runtime_constants
from llm_runtime import build_online_obs, runtime_constants_metadata


POTENTIAL_CANDIDATE_SCHEMA_VERSION = "uav-hrl-llm-potential-candidate-v1"
POTENTIAL_ARTIFACT_SCHEMA_VERSION = "uav-hrl-approved-potential-design-v1"
POTENTIAL_RUNTIME_CONTRACT_VERSION = "uav-hrl-llm-potential-runtime-v1"
POTENTIAL_INTERFACE_VERSION = "uav-hrl-llm-potential-components-v1"
RUN_ARTIFACT_DIRECTORY_NAME = "potential_artifact"
DEFAULT_BETA = 1.0


class PotentialError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copytree(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()

    def extended(path: Path) -> str:
        value = str(path)
        if os.name != "nt" or value.startswith("\\\\?\\"):
            return value
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value[2:]
        return "\\\\?\\" + value

    shutil.copytree(extended(source), extended(destination))


def _rename_candidate_function(code: str, source: str, destination: str) -> str:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CandidateError(f"candidate code has invalid syntax: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or any(
        not isinstance(node, ast.FunctionDef) for node in tree.body
    ):
        raise CandidateError("code must contain exactly one top-level function")
    function = functions[0]
    if function.name != source:
        raise CandidateError(f"code function must be exactly {source}")
    function.name = destination
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def validate_submission_schema(submission: Any, *, expected_id: str | None = None) -> None:
    required = {"candidate_id", "design_summary", "components", "code"}
    if not isinstance(submission, dict) or set(submission) != required:
        raise CandidateError(f"candidate fields must be exactly {sorted(required)}")
    if expected_id is not None and submission["candidate_id"] != expected_id:
        raise CandidateError(
            f"candidate_id must be {expected_id!r}, received {submission['candidate_id']!r}"
        )
    if not isinstance(submission["candidate_id"], str) or not submission["candidate_id"].strip():
        raise CandidateError("candidate_id must be a non-empty string")
    if not isinstance(submission["design_summary"], str) or not submission["design_summary"].strip():
        raise CandidateError("design_summary must be a non-empty string")
    components = submission["components"]
    if not isinstance(components, list) or not components:
        raise CandidateError("components must contain at least one item")
    names = []
    for index, item in enumerate(components):
        if not isinstance(item, dict) or set(item) != {"name", "description", "weight"}:
            raise CandidateError(
                f"components[{index}] fields must be name, description, and weight"
            )
        for field in ("name", "description"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise CandidateError(f"components[{index}].{field} must be non-empty")
        weight = item["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise CandidateError(f"components[{index}].weight must be numeric")
        if not math.isfinite(float(weight)):
            raise CandidateError(f"components[{index}].weight must be finite")
        names.append(item["name"])
    if len(set(names)) != len(names):
        raise CandidateError("component names must be unique")
    if not isinstance(submission["code"], str) or not submission["code"].strip():
        raise CandidateError("code must be a non-empty string")


def _adapt_submission(
    submission: dict[str, Any], constants_metadata: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt only for reuse of the established sandbox/static validator."""

    validate_submission_schema(submission)
    model_form = {
        "features": [
            {
                "name": item["name"],
                "description": item["description"],
                "reward_weight": float(item["weight"]),
            }
            for item in submission["components"]
        ],
        "code": _rename_candidate_function(
            submission["code"], "compute_potential_components", "compute_extra_state"
        ),
    }
    _model, internal, metadata = normalize_candidate_submission(
        model_form, constants_metadata
    )
    return internal, metadata


def validate_potential_candidate(
    submission: dict[str, Any],
    *,
    constants_metadata: dict[str, Any],
    obs_arrays: dict[str, np.ndarray] | None = None,
    timeout: float = 60.0,
    expected_id: str | None = None,
) -> dict[str, Any]:
    """Validate schema, safe operations, determinism, shape, finiteness and range."""

    validate_submission_schema(submission, expected_id=expected_id)
    internal, adapter = _adapt_submission(submission, constants_metadata)
    report: dict[str, Any] = {
        "status": "passed",
        "candidate_schema_version": POTENTIAL_CANDIDATE_SCHEMA_VERSION,
        "potential_interface_version": POTENTIAL_INTERFACE_VERSION,
        "component_count": len(submission["components"]),
        "weights": [float(item["weight"]) for item in submission["components"]],
        "adapter": adapter,
    }
    if obs_arrays is not None:
        components, _weighted_sum, execution = execute_candidate_isolated(
            internal,
            obs_arrays,
            constants_metadata,
            timeout=float(timeout),
        )
        values = np.asarray(components, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(submission["components"]):
            raise CandidateError("potential component output has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise CandidateError("potential component output contains NaN or Inf")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise CandidateError("potential component output is outside [0,1]")
        report["execution"] = execution
        report["numeric_probe"] = {
            "sample_count": int(values.shape[0]),
            "minimum": values.min(axis=0).tolist(),
            "maximum": values.max(axis=0).tolist(),
        }
    return report


def candidate_fingerprint(submission: dict[str, Any]) -> str:
    """Hash normalized executable code, component order, and fixed weights."""

    validate_submission_schema(submission)
    tree = ast.parse(submission["code"], mode="exec")
    payload = {
        "code_ast": ast.dump(tree, annotate_fields=True, include_attributes=False),
        "weights": [float(item["weight"]) for item in submission["components"]],
        "component_count": len(submission["components"]),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PotentialDesign:
    directory: Path
    artifact: dict[str, Any]
    candidate: dict[str, Any]
    constants_metadata: dict[str, Any]

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(
            [item["weight"] for item in self.candidate["components"]],
            dtype=np.float64,
        )


def save_potential_artifact(
    parent: str | Path,
    *,
    candidate: dict[str, Any],
    constants_metadata: dict[str, Any],
    validation_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    provenance: dict[str, Any],
    gamma: float,
    beta: float = DEFAULT_BETA,
) -> Path:
    validate_submission_schema(candidate)
    gamma = float(gamma)
    beta = float(beta)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise PotentialError("potential artifact gamma must be finite and non-negative")
    if beta != DEFAULT_BETA:
        raise PotentialError("potential artifact beta is fixed at 1")
    approved = Path(parent).resolve() / "approved"
    approved.mkdir(parents=True, exist_ok=False)
    paths = {
        "candidate.json": candidate,
        "constants.json": constants_metadata,
        "validation_report.json": validation_report,
        "evaluation_report.json": evaluation_report,
    }
    for name, value in paths.items():
        _write_json(approved / name, value)
    (approved / "candidate.py").write_text(candidate["code"], encoding="utf-8")
    files = {
        name: _file_hash(approved / name)
        for name in (*paths, "candidate.py")
    }
    artifact = {
        "schema_version": POTENTIAL_ARTIFACT_SCHEMA_VERSION,
        "status": "approved",
        "candidate_schema_version": POTENTIAL_CANDIDATE_SCHEMA_VERSION,
        "potential_interface_version": POTENTIAL_INTERFACE_VERSION,
        "runtime_contract_version": POTENTIAL_RUNTIME_CONTRACT_VERSION,
        "observation_interface_version": OBS_INTERFACE_VERSION,
        "candidate_id": candidate["candidate_id"],
        "component_count": len(candidate["components"]),
        "component_order": [item["name"] for item in candidate["components"]],
        "weights": [float(item["weight"]) for item in candidate["components"]],
        "beta": beta,
        "gamma": gamma,
        "terminal_next_potential": "zero_for_true_terminal",
        "truncation_next_potential": "actual_next_observation_when_bootstrap_is_retained",
        "fingerprint": candidate_fingerprint(candidate),
        "files_sha256": files,
        "provenance": copy.deepcopy(provenance),
    }
    artifact["content_sha256"] = hashlib.sha256(
        _canonical_json(artifact).encode("utf-8")
    ).hexdigest()
    _write_json(approved / "artifact.json", artifact)
    return approved


def load_potential_artifact(directory: str | Path) -> PotentialDesign:
    directory = Path(directory).resolve()
    artifact_path = directory / "artifact.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"potential artifact metadata is missing: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != POTENTIAL_ARTIFACT_SCHEMA_VERSION:
        raise PotentialError(
            "artifact is not an approved potential design; current-state LLM reward "
            "artifacts are intentionally incompatible"
        )
    content_hash = artifact.pop("content_sha256", None)
    expected = hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()
    artifact["content_sha256"] = content_hash
    if content_hash != expected:
        raise PotentialError("potential artifact metadata content hash changed")
    for name, digest in artifact.get("files_sha256", {}).items():
        path = directory / name
        if not path.is_file() or _file_hash(path) != digest:
            raise PotentialError(f"potential artifact file changed: {name}")
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    constants = json.loads((directory / "constants.json").read_text(encoding="utf-8"))
    validate_submission_schema(candidate, expected_id=artifact.get("candidate_id"))
    internal, _adapter = _adapt_submission(candidate, constants)
    # Static validation occurs inside the established validator even without probes.
    from llm_candidate import validate_candidate

    validate_candidate(internal, constants)
    if artifact.get("fingerprint") != candidate_fingerprint(candidate):
        raise PotentialError("potential artifact candidate fingerprint changed")
    if float(artifact.get("beta", math.nan)) != DEFAULT_BETA:
        raise PotentialError("potential artifact beta must equal 1")
    return PotentialDesign(directory, artifact, candidate, constants)


def artifact_identity(design: PotentialDesign) -> dict[str, Any]:
    artifact = design.artifact
    return {
        "schema_version": POTENTIAL_ARTIFACT_SCHEMA_VERSION,
        "runtime_contract_version": POTENTIAL_RUNTIME_CONTRACT_VERSION,
        "artifact_content_sha256": artifact["content_sha256"],
        "candidate_id": artifact["candidate_id"],
        "component_count": int(artifact["component_count"]),
        "component_order": list(artifact["component_order"]),
        "weights": [float(value) for value in artifact["weights"]],
        "beta": float(artifact["beta"]),
        "gamma": float(artifact["gamma"]),
        "observation_interface_version": artifact["observation_interface_version"],
        "fingerprint": artifact["fingerprint"],
    }


def copy_potential_artifact(source: str | Path, run_directory: str | Path) -> PotentialDesign:
    source_design = load_potential_artifact(source)
    destination = Path(run_directory).resolve() / RUN_ARTIFACT_DIRECTORY_NAME
    if destination.exists():
        raise FileExistsError(f"run potential artifact already exists: {destination}")
    _copytree(source_design.directory, destination)
    copied = load_potential_artifact(destination)
    if artifact_identity(copied) != artifact_identity(source_design):
        raise PotentialError("copied potential artifact identity changed")
    return copied


def load_run_artifact(
    run_directory: str | Path, expected_identity: dict[str, Any] | None = None
) -> PotentialDesign:
    design = load_potential_artifact(
        Path(run_directory).resolve() / RUN_ARTIFACT_DIRECTORY_NAME
    )
    identity = artifact_identity(design)
    if expected_identity is not None and identity != expected_identity:
        raise PotentialError("run potential artifact differs from the recorded design")
    return design


def _worker_main(connection, candidate, constants, observation_interface_version):
    try:
        namespace = {"__builtins__": SAFE_BUILTINS, "np": np}
        exec(compile(candidate["code"], "<potential-runtime>", "exec"), namespace, namespace)
        function = namespace["compute_potential_components"]
        adapted, _ = _adapt_submission(candidate, {
            name: {"value": value} for name, value in constants.items()
        })
        # _adapt_submission metadata is not used by execution; supply the original
        # component ranges to the common output validator.
        runtime_candidate = copy.deepcopy(adapted)
        connection.send({"status": "ready"})
        while True:
            request = connection.recv()
            if request is None:
                return
            obs = request["obs"]
            if set(obs) != set(obs_keys_for_interface(observation_interface_version)):
                raise ValueError("runtime exposes an unexpected observation field set")
            for value in obs.values():
                value.setflags(write=False)
            values = _run_function(
                function,
                obs,
                constants,
                len(candidate["components"]),
                "compute_potential_components",
            )
            _validate_output_ranges(values, runtime_candidate, "online observation")
            connection.send({"status": "ok", "components": values})
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                {"status": "error", "type": type(exc).__name__, "message": str(exc)}
            )
        except BaseException:
            pass
    finally:
        connection.close()


@dataclass
class PotentialRuntime:
    design: PotentialDesign
    constants_metadata: dict[str, Any]
    timeout: float = 60.0

    def __post_init__(self):
        if not math.isfinite(float(self.timeout)) or float(self.timeout) <= 0.0:
            raise ValueError("potential worker timeout must be finite positive")
        self.identity = artifact_identity(self.design)
        self.weights = self.design.weights
        self.beta = float(self.identity["beta"])
        self.gamma = float(self.identity["gamma"])
        self.observation_interface_version = self.identity["observation_interface_version"]
        self.constants = runtime_constants(self.constants_metadata)
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(child, self.design.candidate, self.constants, self.observation_interface_version),
            daemon=True,
        )
        self._process.start()
        child.close()
        response = self._receive("initialization")
        if response.get("status") != "ready":
            self.close()
            raise PotentialError(f"potential runtime initialization failed: {response}")

    @property
    def component_count(self) -> int:
        return int(self.identity["component_count"])

    def _receive(self, phase: str) -> dict[str, Any]:
        if not self._connection.poll(float(self.timeout)):
            self.close()
            raise PotentialError(f"potential worker timed out during {phase}")
        try:
            return self._connection.recv()
        except EOFError as exc:
            self.close()
            raise PotentialError(f"potential worker exited during {phase}") from exc

    def evaluate(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if not self._process.is_alive():
            raise PotentialError("potential candidate worker is not running")
        self._connection.send({"obs": obs})
        response = self._receive("online evaluation")
        if response.get("status") != "ok":
            raise PotentialError(
                "potential candidate evaluation failed: "
                f"{response.get('type')}: {response.get('message')}"
            )
        values = np.asarray(response["components"], dtype=np.float32)
        if values.shape != (self.component_count,) or not np.all(np.isfinite(values)):
            raise PotentialError("potential candidate returned an invalid component vector")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise PotentialError("potential candidate returned a component outside [0,1]")
        return values

    def close(self) -> None:
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if connection is not None:
            connection.close()


def compute_shaping(
    current_components,
    next_components,
    weights,
    *,
    gamma: float,
    beta: float = DEFAULT_BETA,
    terminated: bool,
    bootstrap_allowed: bool,
) -> dict[str, Any]:
    """Compute host-owned potential shaping with explicit terminal semantics."""

    current = np.asarray(current_components, dtype=np.float64)
    following = np.asarray(next_components, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if current.ndim != 1 or following.shape != current.shape or weight.shape != current.shape:
        raise PotentialError("potential component and weight vectors must be aligned and 1-D")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(following)):
        raise PotentialError("potential components must be finite")
    if np.any(current < 0.0) or np.any(current > 1.0):
        raise PotentialError("current potential components must lie in [0,1]")
    if np.any(following < 0.0) or np.any(following > 1.0):
        raise PotentialError("next potential components must lie in [0,1]")
    gamma = float(gamma)
    beta = float(beta)
    if not math.isfinite(gamma) or gamma < 0.0 or not math.isfinite(beta):
        raise PotentialError("potential gamma and beta must be finite; gamma non-negative")
    if bool(terminated) and bool(bootstrap_allowed):
        raise PotentialError("a true terminal transition cannot retain bootstrap")
    effective_next = np.zeros_like(following) if terminated else following
    if not terminated and not bootstrap_allowed:
        # A non-terminal row with a disabled bootstrap mask is semantically
        # ambiguous and could silently use reset data.  Refuse it explicitly.
        raise PotentialError(
            "non-terminal potential shaping requires the actual next observation and bootstrap"
        )
    weighted_current = weight * current
    weighted_next = weight * effective_next
    component_shaping = beta * (gamma * weighted_next - weighted_current)
    current_phi = float(np.sum(weighted_current, dtype=np.float64))
    next_phi = float(np.sum(weighted_next, dtype=np.float64))
    shaping = float(np.sum(component_shaping, dtype=np.float64))
    if not all(map(math.isfinite, (current_phi, next_phi, shaping))):
        raise PotentialError("potential or shaping is non-finite")
    return {
        "current_components": current,
        "effective_next_components": effective_next,
        "weighted_current_components": weighted_current,
        "component_shaping": component_shaping,
        "current_potential": current_phi,
        "effective_next_potential": next_phi,
        "shaping": shaping,
        "terminated": bool(terminated),
        "bootstrap_allowed": bool(bootstrap_allowed),
        "gamma": gamma,
        "beta": beta,
    }


@dataclass
class MergeableMoments:
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    total_absolute: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise PotentialError("statistics cannot record NaN or Inf")
        self.count += 1
        self.total += value
        self.total_squares += value * value
        self.total_absolute += abs(value)

    def merge(self, other: "MergeableMoments") -> None:
        self.count += int(other.count)
        self.total += float(other.total)
        self.total_squares += float(other.total_squares)
        self.total_absolute += float(other.total_absolute)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "count": int(self.count),
            "sum": float(self.total),
            "sum_of_squares": float(self.total_squares),
            "sum_of_absolute_values": float(self.total_absolute),
        }
        if not self.count:
            return {**result, "missing": True, "mean": None, "standard_deviation": None, "mean_absolute": None}
        mean = self.total / self.count
        variance = max(self.total_squares / self.count - mean * mean, 0.0)
        return {
            **result,
            "missing": False,
            "mean": float(mean),
            "standard_deviation": float(math.sqrt(variance)),
            "mean_absolute": float(self.total_absolute / self.count),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MergeableMoments":
        return cls(
            count=int(value.get("count", 0)),
            total=float(value.get("sum", 0.0)),
            total_squares=float(value.get("sum_of_squares", 0.0)),
            total_absolute=float(value.get("sum_of_absolute_values", 0.0)),
        )


def summarize_component_records(
    rows: list[dict[str, Any]], *, component_names: list[str]
) -> list[dict[str, Any]]:
    """Merge per-episode sufficient statistics without averaging deviations."""

    result = []
    for index, name in enumerate(component_names):
        potential = MergeableMoments()
        ordinary = MergeableMoments()
        terminal = MergeableMoments()
        for row in rows:
            components = row.get("potential_components") or row.get("components") or []
            if index >= len(components):
                raise PotentialError("episode component statistics are incomplete")
            item = components[index]
            potential.merge(MergeableMoments.from_dict(item["weighted_potential"]))
            ordinary.merge(MergeableMoments.from_dict(item["ordinary_shaping"]))
            terminal.merge(MergeableMoments.from_dict(item["terminal_shaping"]))
        result.append(
            {
                "index": index,
                "name": name,
                "weighted_potential": potential.to_dict(),
                "ordinary_shaping": ordinary.to_dict(),
                "terminal_shaping": terminal.to_dict(),
            }
        )
    return result


__all__ = [
    "DEFAULT_BETA",
    "MergeableMoments",
    "POTENTIAL_ARTIFACT_SCHEMA_VERSION",
    "POTENTIAL_INTERFACE_VERSION",
    "PotentialDesign",
    "PotentialError",
    "PotentialRuntime",
    "artifact_identity",
    "build_online_obs",
    "candidate_fingerprint",
    "compute_shaping",
    "copy_potential_artifact",
    "load_potential_artifact",
    "load_run_artifact",
    "runtime_constants_metadata",
    "save_potential_artifact",
    "summarize_component_records",
    "validate_potential_candidate",
    "validate_submission_schema",
]
