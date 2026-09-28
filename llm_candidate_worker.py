"""Isolated subprocess for executing an already statically checked candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from llm_candidate import validate_candidate
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


def _validate_output_ranges(extra, terms, candidate, label):
    tolerance = 1e-6
    extra = np.asarray(extra)
    terms = np.asarray(terms)
    if extra.ndim == 1:
        extra = extra[None, :]
    if terms.ndim == 1:
        terms = terms[None, :]
    for index, definition in enumerate(candidate["features"]):
        minimum = float(definition["range"]["minimum"])
        maximum = float(definition["range"]["maximum"])
        observed_minimum = float(np.min(extra[:, index]))
        observed_maximum = float(np.max(extra[:, index]))
        if observed_minimum < -1.0 - tolerance or observed_maximum > 1.0 + tolerance:
            raise ValueError(
                f"compute_extra_state({label}) feature[{index}] is outside [-1,1]: "
                f"observed [{observed_minimum},{observed_maximum}]"
            )
        if observed_minimum < minimum - tolerance or observed_maximum > maximum + tolerance:
            raise ValueError(
                f"compute_extra_state({label}) feature[{index}] violates its declared "
                f"range [{minimum},{maximum}]: observed "
                f"[{observed_minimum},{observed_maximum}]"
            )
    for index in range(terms.shape[1]):
        observed_minimum = float(np.min(terms[:, index]))
        observed_maximum = float(np.max(terms[:, index]))
        if observed_minimum < -tolerance or observed_maximum > 1.0 + tolerance:
            raise ValueError(
                f"compute_reward_terms({label}) reward_terms[{index}] is outside [0,1]: "
                f"observed [{observed_minimum},{observed_maximum}]"
            )
    weights = np.asarray(
        [item["weight"] for item in candidate["reward_terms"]], dtype=np.float64
    )
    weighted = np.asarray(terms, dtype=np.float64) @ weights
    if np.any(weighted < -1.0 - tolerance) or np.any(weighted > 1.0 + tolerance):
        raise ValueError(
            f"weighted extra reward({label}) is outside [-1,1]: observed "
            f"[{float(np.min(weighted))},{float(np.max(weighted))}]"
        )


class _IssueAccumulator:
    def __init__(self, example_limit=3):
        self.example_limit = int(example_limit)
        self._items = {}

    def add(self, code, location, problem, requirement, sample):
        key = (str(code), str(location), str(requirement))
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


def _check_function(function, obs, constants, expected_size, label, sample, issues):
    try:
        return _run_function(function, obs, constants, expected_size, label)
    except BaseException as exc:
        issues.add(
            _runtime_error_code(str(exc)),
            label,
            f"{type(exc).__name__}: {exc}",
            "Return a deterministic, side-effect-free one-dimensional float32 array of the declared length with finite values.",
            sample,
        )
        return None


def _check_ranges(extra, terms, candidate, label, sample, issues):
    tolerance = 1e-6
    if extra is not None:
        for index, definition in enumerate(candidate["features"]):
            value = float(extra[index])
            minimum = float(definition["range"]["minimum"])
            maximum = float(definition["range"]["maximum"])
            if value < -1.0 - tolerance or value > 1.0 + tolerance:
                issues.add(
                    "RUNTIME_FEATURE_BOUNDS",
                    f"compute_extra_state({label}) feature[{index}]",
                    f"observed {value} outside [-1,1]",
                    "Every extra-state value must remain within [-1,1].",
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
    if terms is not None:
        for index, value in enumerate(np.asarray(terms, dtype=np.float64)):
            if value < -tolerance or value > 1.0 + tolerance:
                issues.add(
                    "RUNTIME_REWARD_TERM_BOUNDS",
                    f"compute_reward_terms({label}) reward_terms[{index}]",
                    f"observed {float(value)} outside [0,1]",
                    "Every unweighted reward term must remain within [0,1].",
                    sample,
                )
        weights = np.asarray(
            [item["weight"] for item in candidate["reward_terms"]], dtype=np.float64
        )
        weighted = float(np.asarray(terms, dtype=np.float64) @ weights)
        if weighted < -1.0 - tolerance or weighted > 1.0 + tolerance:
            issues.add(
                "RUNTIME_WEIGHTED_REWARD_BOUNDS",
                "weighted extra reward",
                f"observed {weighted} outside [-1,1]",
                "Keep term values and weights within the declared constraints so the weighted sum lies in [-1,1].",
                sample,
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--constants", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        constants = json.loads(Path(args.constants).read_text(encoding="utf-8"))
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
        reward_function = namespace["compute_reward_terms"]
        feature_count = len(candidate["features"])
        term_count = len(candidate["reward_terms"])
        extra = np.empty((rows, feature_count), dtype=np.float32)
        terms = np.empty((rows, term_count), dtype=np.float32)
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
            )
            row_terms = _check_function(
                reward_function,
                obs,
                constants,
                term_count,
                "compute_reward_terms",
                {"fixed_sample_index": row},
                issues,
            )
            _check_ranges(
                row_extra,
                row_terms,
                candidate,
                "fixed sample",
                {"fixed_sample_index": row},
                issues,
            )
            if row_extra is not None:
                extra[row] = row_extra
            if row_terms is not None:
                terms[row] = row_terms
        probe = _empty_probe({name: array[0] for name, array in arrays.items()})
        probe_extra = _check_function(
            extra_function,
            probe,
            constants,
            feature_count,
            "compute_extra_state(empty probe)",
            {"probe": "empty"},
            issues,
        )
        probe_terms = _check_function(
            reward_function,
            probe,
            constants,
            term_count,
            "compute_reward_terms(empty probe)",
            {"probe": "empty"},
            issues,
        )
        _check_ranges(
            probe_extra,
            probe_terms,
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
        np.savez_compressed(args.output, extra_state=extra, reward_terms=terms)
        _write_json(
            args.report,
            {
                "status": "passed",
                "sample_count": rows,
                "feature_count": feature_count,
                "reward_term_count": term_count,
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
