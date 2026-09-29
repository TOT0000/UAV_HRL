"""Isolated subprocess for executing an already statically checked candidate."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from llm_candidate import feature_reward, validate_candidate
from llm_design_contract import OBS_KEYS


SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


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
    if not isinstance(first, np.ndarray) or first.ndim != 1:
        raise ValueError(f"{label} must return a one-dimensional NumPy array")
    if first.dtype != np.float32:
        raise ValueError(f"{label} must return dtype float32")
    if first.shape != (expected_size,):
        raise ValueError(
            f"{label} returned shape {first.shape}, expected ({expected_size},)"
        )
    if not np.isfinite(first).all():
        raise ValueError(f"{label} returned NaN or Infinity")
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
    weighted = feature_reward(extra, candidate)
    if np.any(weighted < -1.0 - tolerance) or np.any(weighted > 1.0 + tolerance):
        raise ValueError(
            f"weighted extra reward({label}) is outside [-1,1]: observed "
            f"[{float(np.min(weighted))},{float(np.max(weighted))}]"
        )


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
    ):
        key = (
            str(code),
            str(location),
            str(requirement),
            "" if exception_type is None else str(exception_type),
            "" if candidate_function is None else str(candidate_function),
            "" if candidate_line is None else str(candidate_line),
            "" if problem_signature is None else str(problem_signature),
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


def _state_aliases_and_open_slices(source):
    """Return statically evident unbounded slices of obs['state'] aliases."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return []
    aliases = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == "obs"
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == "state"
        ):
            aliases.add(target.id)
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
            continue
        base_is_state = isinstance(node.value, ast.Name) and node.value.id in aliases
        direct_state = (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "obs"
            and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value == "state"
        )
        if not (base_is_state or direct_state) or node.slice.upper is not None:
            continue
        step = node.slice.step
        if not isinstance(step, ast.Constant) or not isinstance(step.value, int):
            continue
        lower = node.slice.lower
        results.append(
            {
                "line": int(getattr(node, "lineno", 0)),
                "start": (
                    int(lower.value)
                    if isinstance(lower, ast.Constant) and isinstance(lower.value, int)
                    else None
                ),
                "step": int(step.value),
                "expression": ast.get_source_segment(source, node),
            }
        )
    return results


def _runtime_diagnostics(exc, frame, candidate_source, candidate_line, contract):
    diagnostics = {
        "candidate_source_excerpt": _candidate_source_excerpt(
            candidate_source, candidate_line
        ),
        "local_array_summaries": _safe_local_array_summaries(frame),
    }
    excerpt_text = "\n".join(
        str(item.get("code", "")) for item in diagnostics["candidate_source_excerpt"]
    )
    relevant_interface = {}
    if "state" in excerpt_text and isinstance(contract, dict) and contract.get("state"):
        relevant_interface["obs.state"] = contract["state"]
    if (
        "movement_mask" in excerpt_text
        and isinstance(contract, dict)
        and contract.get("movement_mask")
    ):
        relevant_interface["obs.movement_mask"] = contract["movement_mask"]
    if relevant_interface:
        diagnostics["related_interface"] = relevant_interface

    message = str(exc)
    state_contract = contract.get("state", {}) if isinstance(contract, dict) else {}
    block = state_contract.get("uav_block", {}) if isinstance(state_contract, dict) else {}
    mask_contract = contract.get("movement_mask", {}) if isinstance(contract, dict) else {}
    expected_mask = (mask_contract.get("shape") or [None])[0]
    local_dimension = block.get("features_per_uav")
    array_summaries = diagnostics["local_array_summaries"]
    bool_lengths = {
        value["shape"][0]
        for value in array_summaries.values()
        if value.get("dtype") == "bool" and len(value.get("shape", [])) == 1
    }
    vector_lengths = {
        value["shape"][0]
        for value in array_summaries.values()
        if value.get("dtype") != "bool" and len(value.get("shape", [])) == 1
    }
    slices = _state_aliases_and_open_slices(candidate_source)
    suspect_slices = [
        item
        for item in slices
        if local_dimension is not None and item.get("step") == int(local_dimension)
    ]
    confident_uav_slice = (
        type(exc).__name__ == "IndexError"
        and "boolean index" in message.lower()
        and expected_mask in bool_lengths
        and any(length != expected_mask for length in vector_lengths)
        and bool(suspect_slices)
        and all(
            item.get("start") is None or 0 <= int(item["start"]) < int(local_dimension)
            for item in suspect_slices
        )
    )
    if confident_uav_slice:
        start = int(block["start"])
        stop = int(block["stop_exclusive"])
        num_uav = int(block["num_uav"])
        per_uav = int(block["features_per_uav"])
        observed_vectors = sorted(
            length
            for length in vector_lengths
            if length != state_contract.get("shape", [None])[0]
        )
        diagnostics["suspected_state_slices"] = suspect_slices
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
):
    try:
        return _run_function(function, obs, constants, expected_size, label)
    except BaseException as exc:
        exception_type, candidate_function, candidate_line, signature, frame = (
            _candidate_exception_details(exc)
        )
        diagnostics = _runtime_diagnostics(
            exc,
            frame,
            candidate_source,
            candidate_line,
            diagnostic_contract,
        )
        location = label
        if candidate_function is not None and candidate_line is not None:
            location = f"{candidate_function} at candidate line {candidate_line}"
        issues.add(
            _runtime_error_code(str(exc)),
            location,
            f"{exception_type}: {exc}",
            diagnostics.get(
                "targeted_requirement",
                "Return a deterministic, side-effect-free one-dimensional float32 "
                "array of the declared length with finite values.",
            ),
            sample,
            exception_type=exception_type,
            candidate_function=candidate_function,
            candidate_line=candidate_line,
            problem_signature=signature,
            diagnostics=diagnostics,
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
        weighted = float(feature_reward(extra, candidate))
        if weighted < -1.0 - tolerance or weighted > 1.0 + tolerance:
            issues.add(
                "RUNTIME_WEIGHTED_REWARD_BOUNDS",
                "weighted extra reward",
                f"observed {weighted} outside [-1,1]",
                "Keep feature values and reward weights within the declared constraints so the weighted sum lies in [-1,1].",
                sample,
            )


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
            if set(archive.files) != set(OBS_KEYS):
                raise ValueError("worker input exposes an unexpected observation field set")
            arrays = {key: np.asarray(archive[key]) for key in OBS_KEYS}
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
        )
        _check_ranges(
            probe_extra,
            candidate,
            "empty probe",
            {"probe": "empty"},
            issues,
        )
        if issues.errors:
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
