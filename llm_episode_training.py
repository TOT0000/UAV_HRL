"""Subprocess adapters for formal episode-search training and evaluation."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import numpy as np

from evaluation_selection import resolve_training_run_checkpoint
from experiment_paths import read_run_status
from llm_candidate import load_approved_design
from llm_runtime import artifact_identity, load_run_artifact
from scenario_manifest import ScenarioManifest
from training_checkpoint import CHECKPOINT_PROVENANCE_FIELDS


class EpisodeTrainingError(RuntimeError):
    def __init__(
        self, message, *, command=None, output=None, output_directory=None,
        path_preflight=None,
    ):
        super().__init__(message)
        self.command = list(command) if command is not None else None
        self.output = output
        self.output_directory = (
            str(Path(output_directory).resolve())
            if output_directory is not None
            else None
        )
        self.path_preflight = path_preflight


SEARCH_TRAINING_STATE_SCHEMA_VERSION = (
    "uav-hrl-llm-episode-search-training-state-v1"
)
SEARCH_TRAINING_STATE_FILENAME = "episode_search_training_state.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_rows(path: str | Path, *, expected_count: int) -> list[dict[str, Any]]:
    if path is None:
        raise EpisodeTrainingError("per-episode evaluation output path is missing")
    path = Path(path).resolve()
    if not path.is_file():
        raise EpisodeTrainingError(f"per-episode evaluation output is missing: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeTrainingError(
                f"per-episode evaluation output has invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise EpisodeTrainingError(
                f"per-episode evaluation output row {line_number} is not an object"
            )
        rows.append(row)
    if len(rows) != int(expected_count):
        raise EpisodeTrainingError(
            f"per-episode evaluation output is incomplete: expected {expected_count}, received {len(rows)}"
        )
    scenario_ids = [row.get("scenario_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in scenario_ids):
        raise EpisodeTrainingError("per-episode evaluation rows lack scenario IDs")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise EpisodeTrainingError("per-episode evaluation rows contain duplicate scenarios")
    return rows


def _evaluation_metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_values(name: str) -> np.ndarray:
        try:
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeTrainingError(f"evaluation metric {name!r} is unavailable") from exc
        if values.shape != (len(rows),) or np.any(~np.isfinite(values)):
            raise EpisodeTrainingError(f"evaluation metric {name!r} is non-finite")
        return values

    ee = finite_values("energy_efficiency_mbit_per_j")
    timely = finite_values("total_timely_useful_mbits")
    energy = finite_values("total_mobility_energy_j")
    result = {
        "episode_count": len(rows),
        "episode_energy_efficiency_mbit_per_j": {
            "mean": float(np.mean(ee)),
            "std": float(np.std(ee)),
            "aggregation": "arithmetic mean and population standard deviation across episodes",
        },
        "timely_useful_delivery_mbit": {
            "total": float(np.sum(timely)),
            "mean_per_episode": float(np.mean(timely)),
            "aggregation": "sum and arithmetic mean of per-episode timely useful Mbit",
        },
        "movement_energy_j": {
            "total": float(np.sum(energy)),
            "mean_per_episode": float(np.mean(energy)),
            "aggregation": "sum and arithmetic mean of per-episode movement joules",
        },
        "end_to_end_delay_violation_probability": {},
    }
    for label, prefix in (("VS", "fov"), ("COM", "com")):
        try:
            eligible = sum(int(row[f"{prefix}_eligible_packets"]) for row in rows)
            violations = sum(int(row[f"{prefix}_violation_packets"]) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeTrainingError(
                f"evaluation lacks canonical {label} eligible/violation counts"
            ) from exc
        if eligible < 0 or violations < 0 or violations > eligible:
            raise EpisodeTrainingError(f"evaluation has invalid {label} QoS counts")
        result["end_to_end_delay_violation_probability"][label] = {
            "eligible_packets": eligible,
            "violated_packets": violations,
            "value": float(violations / eligible) if eligible else None,
            "aggregation": "pooled violated packets / pooled eligible packets",
            "missing": eligible == 0,
        }
    return result


def _last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise EpisodeTrainingError("subprocess did not emit a JSON result")


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line.rstrip(), flush=True)
    returncode = process.wait()
    output = "".join(lines)
    if returncode:
        raise EpisodeTrainingError(
            f"command failed ({returncode}): {' '.join(command)}\n"
            f"combined output:\n{output[-8000:]}",
            command=command,
            output=output,
        )
    return _last_json_object(output)


WINDOWS_PORTABLE_PATH_LIMIT = 240
EVALUATION_OUTPUT_FILENAMES = (
    "scenario_manifest.json",
    "packet_outcomes.jsonl",
    "packet_routing_diagnostics.json",
    "packet_routing_diagnostics.csv",
    "terminal_uav_distribution.csv",
    "per_episode.csv",
    "per_episode.jsonl",
    "per_training_seed_summary.csv",
    "per_training_seed_summary.json",
    "canonical_per_seed_aggregation.json",
    "canonical_cross_seed_aggregation.json",
    "run_metadata.json",
    "aggregated_plot_data.json",
    "aggregated_plot_data.csv",
    "paper_evaluation_metadata.json",
)


def preflight_candidate_evaluation_output(
    output_directory: str | Path,
    *,
    portable_path_limit: int = WINDOWS_PORTABLE_PATH_LIMIT,
) -> dict[str, Any]:
    """Validate the flattened fixed-RoI output paths before simulation."""

    output = Path(output_directory).resolve()
    paths = [output / name for name in EVALUATION_OUTPUT_FILENAMES]
    longest = max(paths, key=lambda item: len(str(item)))
    result = {
        "schema_version": "uav-hrl-episode-search-evaluation-path-preflight-v1",
        "status": "passed",
        "output_directory": str(output),
        "portable_path_limit": int(portable_path_limit),
        "paths": [str(path) for path in paths],
        # The fixed-RoI evaluator's writers are direct writes, not atomic temp writes.
        "temporary_paths": [],
        "longest_path": str(longest),
        "longest_path_length": len(str(longest)),
    }
    if result["longest_path_length"] > int(portable_path_limit):
        result["status"] = "failed"
        raise EpisodeTrainingError(
            "candidate evaluation output exceeds the portable Windows path "
            f"limit before simulation: {result['longest_path_length']} > "
            f"{int(portable_path_limit)}: {longest}",
            output_directory=output,
            path_preflight=result,
        )
    return result


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_run_directories(root: Path, method_id: str) -> list[Path]:
    method_root = Path(root) / str(method_id)
    if not method_root.is_dir():
        return []
    return sorted(path.resolve() for path in method_root.iterdir() if path.is_dir())


def _nonempty_jsonl(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return True


def _full_checkpoint_directories(run_directory: Path) -> list[Path]:
    root = Path(run_directory) / "checkpoints" / "full"
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.iterdir() if path.is_dir())


def _training_run_evidence(run_directory: Path) -> dict[str, Any]:
    """Classify persisted lifecycle evidence without treating existence as resume."""

    run_directory = Path(run_directory).resolve()
    try:
        status = read_run_status(run_directory)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "classification": "invalid_lifecycle",
            "run_directory": str(run_directory),
            "error": f"{type(exc).__name__}: {exc}",
            "has_training_progress": True,
            "full_checkpoint_directories": [],
        }
    transitions = [item.get("state") for item in (status or {}).get("transitions", [])]
    checkpoint_directories = _full_checkpoint_directories(run_directory)
    progress_files = [
        name
        for name in ("training_history.jsonl", "llm_training_episode_metrics.jsonl")
        if _nonempty_jsonl(run_directory / name)
    ]
    has_running_transition = any(
        value in {"RUNNING", "RESUMING", "COMPLETED"} for value in transitions
    )
    has_progress = bool(has_running_transition or progress_files or checkpoint_directories)
    resolved_path = run_directory / "resolved_config.json"
    resolved = None
    resolved_error = None
    if resolved_path.is_file():
        try:
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            if not isinstance(resolved, dict):
                raise ValueError("resolved_config.json is not an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            resolved_error = f"{type(exc).__name__}: {exc}"
    if (
        status is not None
        and status.get("state") == "COMPLETED"
        and isinstance(resolved, dict)
        and resolved.get("status") == "COMPLETED"
    ):
        classification = "completed"
    elif checkpoint_directories:
        # run_experiment resume performs the authoritative compatibility and
        # payload validation.  If that rejects the checkpoint, this adapter
        # records the failure and never falls back to fresh training.
        classification = "resume_checkpoint_present"
    elif has_progress:
        classification = "progress_without_checkpoint"
    elif status is None or status.get("state") in {"PREPARING", "FAILED", "INTERRUPTED"}:
        classification = "initialization_incomplete"
    else:
        classification = "empty_shell"
    return {
        "classification": classification,
        "run_directory": str(run_directory),
        "lifecycle_state": (status or {}).get("state"),
        "lifecycle_transitions": transitions,
        "lifecycle_exception": (status or {}).get("exception"),
        "has_training_progress": has_progress,
        "progress_files": progress_files,
        "full_checkpoint_directories": [str(path) for path in checkpoint_directories],
        "resolved_config_present": resolved_path.is_file(),
        "resolved_config_status": (
            resolved.get("status") if isinstance(resolved, dict) else None
        ),
        "resolved_config_episodes": (
            resolved.get("episodes") if isinstance(resolved, dict) else None
        ),
        "resolved_config_error": resolved_error,
        "partial_artifact_present": (run_directory / "llm_artifact").exists(),
    }


def _new_training_state(
    *,
    method_id: str,
    artifact_identity_record: dict[str, Any],
    episodes: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SEARCH_TRAINING_STATE_SCHEMA_VERSION,
        "status": "ready",
        "created_at_utc": now,
        "updated_at_utc": now,
        "method_id": str(method_id),
        "artifact_identity": artifact_identity_record,
        "episodes": int(episodes),
        "seed": int(seed),
        "output_root": str(output),
        "active_run_directory": None,
        "attempts": [],
    }


def _load_training_state(
    path: Path,
    *,
    method_id: str,
    artifact_identity_record: dict[str, Any],
    episodes: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return _new_training_state(
            method_id=method_id,
            artifact_identity_record=artifact_identity_record,
            episodes=episodes,
            seed=seed,
            output=output,
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeTrainingError("training recovery state is unreadable") from exc
    expected = {
        "schema_version": SEARCH_TRAINING_STATE_SCHEMA_VERSION,
        "method_id": str(method_id),
        "artifact_identity": artifact_identity_record,
        "episodes": int(episodes),
        "seed": int(seed),
        "output_root": str(output),
    }
    mismatches = {
        key: {"saved": state.get(key), "expected": value}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise EpisodeTrainingError(
            f"training recovery state is incompatible: {mismatches}"
        )
    if not isinstance(state.get("attempts"), list):
        raise EpisodeTrainingError("training recovery attempt history is invalid")
    return state


def _record_attempt(
    state: dict[str, Any],
    *,
    origin: str,
    evidence: dict[str, Any],
    result: str,
    error: str | None = None,
) -> dict[str, Any]:
    run_directory = evidence.get("run_directory")
    existing = next(
        (
            item
            for item in state["attempts"]
            if item.get("run_directory") == run_directory and run_directory is not None
        ),
        None,
    )
    record = existing if existing is not None else {}
    if existing is None:
        state["attempts"].append(record)
    record.update(
        {
            "attempt": state["attempts"].index(record) + 1,
            "origin": origin,
            "run_directory": run_directory,
            "classification": evidence.get("classification"),
            "result": result,
            "evidence": evidence,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if error is not None:
        record["error"] = error
    return record


def run_candidate_training(
    *,
    method_id: str,
    artifact: str | Path,
    episodes: int,
    seed: int,
    output_directory: str | Path,
    resume_record: dict[str, Any] | None,
    legacy_output_directory: str | Path | None = None,
    prior_output_directories: list[str | Path] | tuple[str | Path, ...] = (),
    restart_from_scratch: bool = False,
    restart_authorization_id: str | None = None,
    restart_failed_run_directory: str | Path | None = None,
    summary_block_size: int = 100,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_design = load_approved_design(artifact)
    identity = artifact_identity(source_design)
    state_path = output / SEARCH_TRAINING_STATE_FILENAME
    state = _load_training_state(
        state_path,
        method_id=method_id,
        artifact_identity_record=identity,
        episodes=episodes,
        seed=seed,
        output=output,
    )
    if restart_from_scratch and (
        not restart_authorization_id or restart_failed_run_directory is None
    ):
        raise EpisodeTrainingError(
            "explicit restart requires its persisted authorization identity and failed run"
        )
    if bool(restart_authorization_id) != bool(restart_failed_run_directory):
        raise EpisodeTrainingError(
            "checkpoint restart authorization identity and failed run must be provided together"
        )
    if restart_authorization_id:
        restart_binding = {
            "operation_id": str(restart_authorization_id),
            "failed_run_directory": str(
                Path(restart_failed_run_directory).resolve()
            ),
        }
        saved_binding = state.get("checkpoint_restart_authorization")
        if saved_binding is not None and any(
            saved_binding.get(key) != value
            for key, value in restart_binding.items()
        ):
            raise EpisodeTrainingError(
                "training recovery state belongs to a different checkpoint restart authorization"
            )
        if saved_binding is None:
            state["checkpoint_restart_authorization"] = restart_binding

    known_by_directory: dict[Path, str] = {}

    def discover(origin: str, root_directory: str | Path | None) -> None:
        if root_directory is None:
            return
        root_path = Path(root_directory).resolve()
        for path in _training_run_directories(root_path, method_id):
            known_by_directory.setdefault(path, origin)

    discover("short_output", output)
    discover("legacy_nested_output", legacy_output_directory)
    for prior in prior_output_directories:
        discover("prior_training_output", prior)
    for origin, value in (
        (
            "saved_search_state",
            (resume_record or {}).get("run_directory"),
        ),
        ("training_recovery_state", state.get("active_run_directory")),
    ):
        if value:
            path = Path(value).resolve()
            known_by_directory.setdefault(path, origin)

    evidence_by_directory: dict[Path, tuple[str, dict[str, Any]]] = {}
    for directory, origin in known_by_directory.items():
        if not directory.is_dir():
            raise EpisodeTrainingError(
                f"recorded training directory is missing: {directory}"
            )
        evidence = _training_run_evidence(directory)
        if evidence["has_training_progress"] or evidence["classification"] == "completed":
            try:
                load_run_artifact(directory, identity)
            except Exception as exc:
                evidence["classification"] = "incompatible_or_incomplete_run_artifact"
                evidence["artifact_error"] = f"{type(exc).__name__}: {exc}"
        if (
            evidence["classification"] == "completed"
            and int(evidence.get("resolved_config_episodes", -1)) != int(episodes)
        ):
            evidence["classification"] = "incompatible_completed_horizon"
        evidence_by_directory[directory] = (origin, evidence)

    previously_abandoned = {
        Path(item["run_directory"]).resolve()
        for item in state["attempts"]
        if item.get("run_directory")
        and item.get("result")
        in {
            "abandoned_initialization_failure",
            "abandoned_for_explicit_restart",
        }
    }
    if restart_authorization_id:
        authorized_failed = Path(restart_failed_run_directory).resolve()
        replacement_shells = [
            (directory, origin, evidence)
            for directory, (origin, evidence) in evidence_by_directory.items()
            if directory != authorized_failed
            and directory not in previously_abandoned
            and origin in {"short_output", "training_recovery_state"}
            and evidence["classification"] in {"initialization_incomplete", "empty_shell"}
        ]
        if replacement_shells:
            for _, origin, evidence in replacement_shells:
                _record_attempt(
                    state,
                    origin=origin,
                    evidence=evidence,
                    result="blocked_replacement_initialization_incomplete",
                )
            state["status"] = "blocked_replacement_initialization_incomplete"
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(state_path, state)
            raise EpisodeTrainingError(
                "checkpoint restart replacement run was already created but has no "
                "resumable checkpoint; refusing to create another run"
            )

    for origin, evidence in evidence_by_directory.values():
        if evidence["classification"] in {"initialization_incomplete", "empty_shell"}:
            _record_attempt(
                state,
                origin=origin,
                evidence=evidence,
                result="abandoned_initialization_failure",
            )

    abandoned = {
        Path(item["run_directory"]).resolve()
        for item in state["attempts"]
        if item.get("run_directory")
        and item.get("result")
        in {
            "abandoned_initialization_failure",
            "abandoned_for_explicit_restart",
        }
    }
    candidates = [
        (directory, origin, evidence)
        for directory, (origin, evidence) in evidence_by_directory.items()
        if directory not in abandoned
        and evidence["classification"]
        in {
            "completed",
            "resume_checkpoint_present",
            "progress_without_checkpoint",
            "invalid_lifecycle",
            "incompatible_or_incomplete_run_artifact",
            "incompatible_completed_horizon",
        }
    ]
    safely_resumable = [
        item
        for item in candidates
        if item[2]["classification"] in {"completed", "resume_checkpoint_present"}
    ]
    unsafe = [item for item in candidates if item not in safely_resumable]
    if len(safely_resumable) > 1 or (safely_resumable and unsafe):
        raise EpisodeTrainingError(
            "multiple training runs contain conflicting progress; refusing to guess"
        )
    if safely_resumable:
        _, origin, evidence = safely_resumable[0]
        selected = (origin, evidence)
    elif restart_from_scratch:
        authorized_failed = Path(restart_failed_run_directory).resolve()
        pending_saved_launch = bool(
            not unsafe
            and authorized_failed in abandoned
            and (state.get("checkpoint_restart_authorization") or {}).get(
                "launch_status"
            )
            == "authorized_launch_pending"
        )
        authorized_failed_is_current = bool(
            len(unsafe) == 1
            and unsafe[0][2]["classification"]
            == "progress_without_checkpoint"
            and Path(unsafe[0][2]["run_directory"]).resolve()
            == authorized_failed
        )
        if not authorized_failed_is_current and not pending_saved_launch:
            raise EpisodeTrainingError(
                "explicit restart requires exactly its authorized progress-without-checkpoint run"
            )
        if authorized_failed_is_current:
            _, origin, evidence = unsafe[0]
            _record_attempt(
                state,
                origin=origin,
                evidence=evidence,
                result="abandoned_for_explicit_restart",
            )
            state.setdefault("explicit_restart_history", []).append(
                {
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_directory": evidence["run_directory"],
                    "reason": "checkpoint_path_failure_without_full_resume_checkpoint",
                    "restart_episode": 1,
                    "operation_id": str(restart_authorization_id),
                }
            )
        selected = None
    elif len(unsafe) == 1:
        _, origin, evidence = unsafe[0]
        selected = (origin, evidence)
    elif len(unsafe) > 1:
        raise EpisodeTrainingError(
            "multiple training runs contain conflicting progress; refusing to guess"
        )
    else:
        selected = None

    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(state_path, state)

    if selected is not None:
        origin, evidence = selected
        resume_directory = Path(evidence["run_directory"])
        state["active_run_directory"] = str(resume_directory)
        if restart_authorization_id:
            state["checkpoint_restart_authorization"][
                "replacement_run_directory"
            ] = str(resume_directory.resolve())
        classification = evidence["classification"]
        if classification == "completed":
            result = {"run_directory": str(resume_directory), "status": "COMPLETED"}
        elif classification == "resume_checkpoint_present":
            state["status"] = "resuming"
            _record_attempt(
                state, origin=origin, evidence=evidence, result="resume_requested"
            )
            _write_json_atomic(state_path, state)
            try:
                result = _run(
                    [
                        sys.executable,
                        str(root / "run_experiment.py"),
                        "resume",
                        str(resume_directory),
                        "--target-episodes",
                        str(int(episodes)),
                    ],
                    cwd=root,
                )
            except EpisodeTrainingError as exc:
                state["status"] = "resume_failed"
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _record_attempt(
                    state,
                    origin=origin,
                    evidence=_training_run_evidence(resume_directory),
                    result="resume_failed_no_retraining",
                    error=f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(state_path, state)
                return {
                    "status": "incomplete",
                    "reason": str(exc),
                    "run_directory": str(resume_directory),
                    "episodes": int(episodes),
                    "method_id": method_id,
                    "training_state_record": str(state_path),
                    "recovery_action": "resume_failed_no_retraining",
                    "summaries": [],
                }
        else:
            state["status"] = "blocked_progress_without_valid_checkpoint"
            _record_attempt(
                state,
                origin=origin,
                evidence=evidence,
                result="blocked_no_safe_restart",
            )
            _write_json_atomic(state_path, state)
            raise EpisodeTrainingError(
                "training progress exists but no resumable checkpoint was found; "
                f"refusing to restart: {resume_directory}"
            )
    else:
        before = set(_training_run_directories(output, method_id))
        state["status"] = "starting_fresh"
        state["active_run_directory"] = None
        if restart_authorization_id:
            state["checkpoint_restart_authorization"]["launch_status"] = (
                "authorized_launch_pending"
            )
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(state_path, state)
        try:
            result = _run(
                [
                    sys.executable,
                    str(root / "run_experiment.py"),
                    str(method_id),
                    "--llm-artifact",
                    str(source_design.directory),
                    "--episodes",
                    str(int(episodes)),
                    "--seed",
                    str(int(seed)),
                    "--output-root",
                    str(output),
                ],
                cwd=root,
            )
        except EpisodeTrainingError as exc:
            after = set(_training_run_directories(output, method_id))
            created = sorted(after.difference(before))
            run_directory = created[0] if len(created) == 1 else None
            evidence = (
                _training_run_evidence(run_directory)
                if run_directory is not None
                else {
                    "classification": "failed_before_run_directory",
                    "run_directory": None,
                    "has_training_progress": False,
                    "full_checkpoint_directories": [],
                }
            )
            state["status"] = "initialization_failed"
            state["active_run_directory"] = (
                str(run_directory) if run_directory is not None else None
            )
            if restart_authorization_id and run_directory is not None:
                state["checkpoint_restart_authorization"][
                    "replacement_run_directory"
                ] = str(run_directory.resolve())
                state["checkpoint_restart_authorization"]["launch_status"] = (
                    "replacement_run_created"
                )
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _record_attempt(
                state,
                origin="short_output",
                evidence=evidence,
                result="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json_atomic(state_path, state)
            return {
                "status": "incomplete",
                "reason": str(exc),
                "run_directory": (
                    str(run_directory) if run_directory is not None else None
                ),
                "episodes": int(episodes),
                "method_id": method_id,
                "training_state_record": str(state_path),
                "recovery_action": "fresh_initialization_failed",
                "summaries": [],
            }
    run_directory = result.get("run_directory")
    if not run_directory:
        raise EpisodeTrainingError("training subprocess did not identify its run directory")
    final_evidence = _training_run_evidence(Path(run_directory))
    if result.get("status") != "COMPLETED" or final_evidence["classification"] != "completed":
        raise EpisodeTrainingError("training subprocess did not produce a validated completed run")
    load_run_artifact(Path(run_directory), identity)
    state["status"] = "completed"
    state["active_run_directory"] = str(Path(run_directory).resolve())
    if restart_authorization_id:
        state["checkpoint_restart_authorization"]["replacement_run_directory"] = str(
            Path(run_directory).resolve()
        )
        state["checkpoint_restart_authorization"]["launch_status"] = "completed"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _record_attempt(
        state,
        origin=(selected[0] if selected is not None else "short_output"),
        evidence=final_evidence,
        result="completed",
    )
    _write_json_atomic(state_path, state)
    return {
        "status": "complete",
        "run_directory": str(Path(run_directory).resolve()),
        "episodes": int(episodes),
        "method_id": method_id,
        "training_state_record": str(state_path),
        "output_root": str(output),
        "summaries": (
            _training_summaries(
                Path(run_directory),
                expected_episodes=int(episodes),
                block_size=int(summary_block_size),
            )
            if run_directory
            else []
        ),
        "raw_result": result,
    }


def _training_summaries(
    run_directory: Path, *, expected_episodes: int, block_size: int = 100
) -> list[dict[str, Any]]:
    path = run_directory / "llm_training_episode_metrics.jsonl"
    if not path.is_file():
        raise EpisodeTrainingError(
            "LLM training episode metrics are missing; an older interrupted run "
            "cannot provide a complete reward-component summary"
        )
    from llm_episode_search import summarize_training_blocks

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return summarize_training_blocks(
            rows, block_size=int(block_size), expected_episodes=expected_episodes
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise EpisodeTrainingError(
            "LLM training episode metrics are incomplete or incompatible: "
            f"{exc}"
        ) from exc


def validate_baseline_preflight(
    *,
    baseline_run: str | Path,
    baseline_evaluation: str | Path,
    manifest: str | Path,
    checkpoint_episode: int = 1500,
    episodes: int = 100,
    roi_count: int = 8,
    environment_size_m: tuple[float, float] = (1000.0, 1000.0),
    episode_seconds: int = 60,
) -> dict[str, Any]:
    """Bind the explicit baseline run, checkpoint, manifest, rows, and metrics."""

    context = resolve_training_run_checkpoint(
        baseline_run,
        int(checkpoint_episode),
        expected_method="td3_dinkelbach",
    )
    baseline_resolved = json.loads(
        (Path(context["run_dir"]) / "resolved_config.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = Path(manifest).resolve()
    scenario_manifest = ScenarioManifest.load(manifest_path)
    if scenario_manifest.split != "test":
        raise EpisodeTrainingError("baseline comparison manifest must use the test split")
    if scenario_manifest.episode_count != int(episodes):
        raise EpisodeTrainingError("baseline comparison manifest episode count is incompatible")
    if int(scenario_manifest.manifest_seed) != int(baseline_resolved["seed"]):
        raise EpisodeTrainingError(
            "baseline evaluation manifest seed differs from the baseline training seed"
        )
    if scenario_manifest.generation_profile.get("fixed_num_gt") != int(roi_count):
        raise EpisodeTrainingError("baseline comparison manifest is not fixed at 8 RoIs")
    if tuple(map(float, environment_size_m)) != (1000.0, 1000.0):
        raise EpisodeTrainingError("baseline comparison requires a 1000x1000 m environment")
    if (
        int(scenario_manifest.environment_width_m) != 1000
        or int(scenario_manifest.environment_height_m) != 1000
        or scenario_manifest.environment_size_m not in (None, 1000)
    ):
        raise EpisodeTrainingError("baseline manifest environment size is incompatible")

    baseline_path = Path(baseline_evaluation).resolve()
    metadata_path = (
        baseline_path / "paper_evaluation_metadata.json"
        if baseline_path.is_dir()
        else baseline_path
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeTrainingError("baseline evaluation metadata is missing or invalid") from exc
    if metadata.get("semantic_suite") != "fixed_roi":
        raise EpisodeTrainingError("baseline evaluation must use the fixed_roi suite")
    points = metadata.get("points") or []
    if len(points) != 1:
        raise EpisodeTrainingError("baseline evaluation must contain exactly one point")
    point = points[0]
    if metadata.get("method_id") != "td3_dinkelbach":
        raise EpisodeTrainingError("baseline evaluation method is not td3_dinkelbach")
    if point.get("training_run_id") != context["training_run_id"]:
        raise EpisodeTrainingError("baseline evaluation belongs to a different training run")
    if int(point.get("checkpoint_episode", -1)) != int(checkpoint_episode):
        raise EpisodeTrainingError(
            f"baseline evaluation did not use checkpoint episode {int(checkpoint_episode)}"
        )
    for field in CHECKPOINT_PROVENANCE_FIELDS:
        expected = context["checkpoint_artifact_provenance"][field]
        if point.get(field) != expected:
            raise EpisodeTrainingError(
                f"baseline evaluation checkpoint provenance mismatch: {field}"
            )
    if int(point.get("evaluation_episode_count", -1)) != int(episodes):
        raise EpisodeTrainingError("baseline evaluation episode count is incompatible")
    if int(point.get("fixed_num_gt", point.get("roi_count", -1))) != int(roi_count):
        raise EpisodeTrainingError("baseline evaluation RoI count is incompatible")
    if (
        float(point.get("evaluation_environment_width_m", -1.0)) != 1000.0
        or float(point.get("evaluation_environment_height_m", -1.0)) != 1000.0
    ):
        raise EpisodeTrainingError("baseline evaluation environment size is incompatible")
    if int(point.get("evaluation_episode_horizon_s", -1)) != int(episode_seconds):
        raise EpisodeTrainingError("baseline evaluation episode horizon is incompatible")
    runtime = point.get("evaluation_runtime_provenance") or {}
    resolved_evaluation = runtime.get("resolved_evaluation_config") or {}
    if resolved_evaluation.get("learning_state_frozen") is not True:
        raise EpisodeTrainingError("baseline evaluation did not freeze learning state")
    if resolved_evaluation.get("new_training_started") is not False:
        raise EpisodeTrainingError("baseline evaluation started new training")
    if runtime.get("lambda_cost_source") != "checkpoint_frozen":
        raise EpisodeTrainingError("baseline evaluation did not keep checkpoint lambda fixed")
    if (
        baseline_resolved.get("exploration_schedule_configuration", {}).get(
            "evaluation_exploration_mode"
        )
        != "disabled"
    ):
        raise EpisodeTrainingError("baseline evaluation exploration contract is not disabled")
    if point.get("scenario_manifest_hash") != scenario_manifest.content_hash:
        raise EpisodeTrainingError("baseline evaluation manifest hash is incompatible")
    if list(point.get("scenario_ids") or ()) != [
        str(item["scenario_id"]) for item in scenario_manifest.episodes
    ]:
        raise EpisodeTrainingError("baseline evaluation scenario IDs are incompatible")
    outputs = point.get("outputs") or {}
    rows = _episode_rows(outputs.get("per_episode_jsonl"), expected_count=episodes)
    if [row["scenario_id"] for row in rows] != [
        str(item["scenario_id"]) for item in scenario_manifest.episodes
    ]:
        raise EpisodeTrainingError("baseline per-episode rows do not match the manifest order")
    metrics = _evaluation_metrics_summary(rows)
    return {
        "schema_version": "uav-hrl-episode-search-baseline-preflight-v1",
        "status": "passed",
        "baseline_run": str(context["run_dir"]),
        "baseline_training_run_id": context["training_run_id"],
        "checkpoint_episode": int(checkpoint_episode),
        "checkpoint_path": str(context["checkpoint"]),
        "checkpoint_provenance": context["checkpoint_artifact_provenance"],
        "baseline_evaluation_metadata": str(metadata_path),
        "baseline_evaluation_metadata_sha256": _file_sha256(metadata_path),
        "per_episode_jsonl": str(Path(outputs["per_episode_jsonl"]).resolve()),
        "per_episode_jsonl_sha256": _file_sha256(Path(outputs["per_episode_jsonl"])),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_content_hash": scenario_manifest.content_hash,
        "scenario_ids": [str(item["scenario_id"]) for item in scenario_manifest.episodes],
        "evaluation_contract": {
            "episodes": int(episodes),
            "roi_count": int(roi_count),
            "environment_size_m": [1000.0, 1000.0],
            "episode_seconds": int(episode_seconds),
            "exploration": "disabled",
            "learning_updates": "disabled",
            "lambda": "checkpoint_frozen",
        },
        "metrics": metrics,
    }


def run_candidate_evaluation(
    *,
    method_id: str,
    run_directory: str | Path,
    checkpoint_episode: int,
    episodes: int,
    roi_count: int,
    environment_size_m: tuple[float, float],
    manifest: str | Path,
    baseline_run: str | Path,
    baseline_evaluation: str | Path,
    baseline_preflight: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    if tuple(map(float, environment_size_m)) != (1000.0, 1000.0):
        raise EpisodeTrainingError("formal search evaluation requires a 1000x1000 m environment")
    root = Path(__file__).resolve().parent
    output = Path(output_directory).resolve()
    path_preflight = preflight_candidate_evaluation_output(output)
    command = [
        sys.executable,
        str(root / "run_paper_evaluation.py"),
        method_id,
        "--run-dir",
        str(Path(run_directory).resolve()),
        "--suite",
        "fixed_roi",
        "--roi-count",
        str(int(roi_count)),
        "--episodes",
        str(int(episodes)),
        "--checkpoint-episode",
        str(int(checkpoint_episode)),
        "--manifest",
        str(Path(manifest).resolve()),
        "--output-directory",
        str(output),
    ]
    try:
        result = _run(command, cwd=root)
    except EpisodeTrainingError as exc:
        if exc.output_directory is None:
            exc.output_directory = str(output)
        if exc.command is None:
            exc.command = command
        if exc.path_preflight is None:
            exc.path_preflight = path_preflight
        raise
    point = result["points"][0]
    if int(point.get("checkpoint_episode", -1)) != int(checkpoint_episode):
        raise EpisodeTrainingError("candidate evaluation used the wrong checkpoint")
    if int(point.get("evaluation_episode_count", -1)) != int(episodes):
        raise EpisodeTrainingError("candidate evaluation episode count is incompatible")
    if int(point.get("fixed_num_gt", point.get("roi_count", -1))) != int(roi_count):
        raise EpisodeTrainingError("candidate evaluation RoI count is incompatible")
    if int(point.get("evaluation_episode_horizon_s", -1)) != int(
        baseline_preflight["evaluation_contract"]["episode_seconds"]
    ):
        raise EpisodeTrainingError("candidate evaluation episode horizon is incompatible")
    if (
        float(point.get("evaluation_environment_width_m", -1.0)) != 1000.0
        or float(point.get("evaluation_environment_height_m", -1.0)) != 1000.0
    ):
        raise EpisodeTrainingError("candidate evaluation did not use a 1000x1000 m environment")
    per_episode = Path(point["outputs"]["per_episode_jsonl"])
    rows = _episode_rows(per_episode, expected_count=episodes)
    if [row["scenario_id"] for row in rows] != baseline_preflight.get(
        "scenario_ids"
    ):
        raise EpisodeTrainingError(
            "candidate per-episode rows differ from the baseline scenario order"
        )
    ee = np.asarray([row["energy_efficiency_mbit_per_j"] for row in rows], dtype=np.float64)
    if ee.size != int(episodes) or np.any(~np.isfinite(ee)):
        raise EpisodeTrainingError("evaluation per-episode EE output is incomplete")
    if result.get("points", [{}])[0].get("scenario_manifest_hash") != baseline_preflight.get(
        "manifest_content_hash"
    ):
        raise EpisodeTrainingError(
            "candidate evaluation manifest differs from baseline preflight"
        )
    if result.get("points", [{}])[0].get("scenario_ids") != baseline_preflight.get(
        "scenario_ids"
    ):
        raise EpisodeTrainingError(
            "candidate evaluation scenarios differ from baseline preflight"
        )
    baseline_path = Path(baseline_evaluation).resolve()
    baseline_run_path = Path(baseline_run).resolve()
    if baseline_run_path != Path(baseline_preflight["baseline_run"]).resolve():
        raise EpisodeTrainingError("baseline run differs from the preflight binding")
    if (
        baseline_path / "paper_evaluation_metadata.json"
        if baseline_path.is_dir()
        else baseline_path
    ).resolve() != Path(
        baseline_preflight["baseline_evaluation_metadata"]
    ).resolve():
        raise EpisodeTrainingError("baseline evaluation differs from the preflight binding")
    if not baseline_path.exists() or not baseline_run_path.exists():
        raise EpisodeTrainingError("explicit baseline run/evaluation inputs are missing")
    baseline_config = json.loads(
        (baseline_run_path / "resolved_config.json").read_text(encoding="utf-8")
    )
    if (baseline_config.get("method_spec") or {}).get("method_id") != "td3_dinkelbach":
        raise EpisodeTrainingError("baseline run must use td3_dinkelbach")
    baseline_metadata_path = (
        baseline_path / "paper_evaluation_metadata.json"
        if baseline_path.is_dir()
        else baseline_path
    )
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    baseline_points = baseline_metadata.get("points") or []
    if len(baseline_points) != 1:
        raise EpisodeTrainingError("baseline evaluation must contain one matching fixed-RoI point")
    baseline_point = baseline_points[0]
    if int(baseline_point.get("checkpoint_episode", -1)) != int(
        baseline_preflight["checkpoint_episode"]
    ):
        raise EpisodeTrainingError(
            "baseline evaluation checkpoint differs from the preflight binding"
        )
    if _file_sha256(baseline_metadata_path) != baseline_preflight.get(
        "baseline_evaluation_metadata_sha256"
    ):
        raise EpisodeTrainingError("baseline evaluation metadata changed after preflight")
    if baseline_point.get("scenario_manifest_hash") != point["scenario_manifest_hash"]:
        raise EpisodeTrainingError("baseline and candidate evaluations do not share a manifest")
    if int(baseline_point.get("evaluation_episode_count", -1)) != int(episodes):
        raise EpisodeTrainingError("baseline evaluation episode count is incompatible")
    baseline_rows_path = Path(baseline_point["outputs"]["per_episode_jsonl"])
    if _file_sha256(baseline_rows_path) != baseline_preflight.get(
        "per_episode_jsonl_sha256"
    ):
        raise EpisodeTrainingError("baseline per-episode rows changed after preflight")
    baseline_rows = _episode_rows(baseline_rows_path, expected_count=episodes)
    baseline_ee = np.asarray(
        [row["energy_efficiency_mbit_per_j"] for row in baseline_rows],
        dtype=np.float64,
    )
    if baseline_ee.size != int(episodes) or np.any(~np.isfinite(baseline_ee)):
        raise EpisodeTrainingError("baseline per-episode EE output is incomplete")
    candidate_metrics = _evaluation_metrics_summary(rows)
    baseline_metrics = _evaluation_metrics_summary(baseline_rows)
    if baseline_metrics != baseline_preflight.get("metrics"):
        raise EpisodeTrainingError("baseline metrics changed after preflight")
    candidate_mean = float(np.mean(ee))
    baseline_mean = float(np.mean(baseline_ee))
    absolute_gap = candidate_mean - baseline_mean
    percent_gap = (
        100.0 * absolute_gap / baseline_mean
        if baseline_mean != 0.0
        else None
    )
    return {
        "status": "complete",
        "method_id": method_id,
        "run_directory": str(Path(run_directory).resolve()),
        "checkpoint_episode": int(checkpoint_episode),
        "evaluation_directory": result["output_directory"],
        "scenario_manifest_hash": point["scenario_manifest_hash"],
        "scenario_manifest_path": point["scenario_manifest_path"],
        "episode_count": len(rows),
        "roi_count": int(roi_count),
        "environment_size_m": [1000.0, 1000.0],
        "mean_episode_energy_efficiency_mbit_per_j": candidate_mean,
        "std_episode_energy_efficiency_mbit_per_j": float(np.std(ee)),
        "baseline_run": str(baseline_run_path),
        "baseline_evaluation": str(baseline_path),
        "baseline_mean_episode_energy_efficiency_mbit_per_j": baseline_mean,
        "baseline_absolute_gap_mbit_per_j": absolute_gap,
        "baseline_percent_gap": percent_gap,
        "improves_over_baseline_mean_episode_ee": bool(
            candidate_mean > baseline_mean
        ),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "path_preflight": path_preflight,
        "command": command,
        "raw_result": result,
    }


def validate_saved_candidate_evaluation(
    result: dict[str, Any],
    *,
    method_id: str,
    run_directory: str | Path,
    checkpoint_episode: int,
    episodes: int,
    roi_count: int,
    baseline_preflight: dict[str, Any],
) -> None:
    """Reject incomplete or mismatched saved evaluation results before reuse."""

    if not isinstance(result, dict) or result.get("status") != "complete":
        raise EpisodeTrainingError("saved candidate evaluation is not complete")
    if result.get("method_id") != method_id:
        raise EpisodeTrainingError("saved candidate evaluation method changed")
    if Path(result.get("run_directory", "")).resolve() != Path(run_directory).resolve():
        raise EpisodeTrainingError("saved candidate evaluation training run changed")
    if int(result.get("checkpoint_episode", -1)) != int(checkpoint_episode):
        raise EpisodeTrainingError("saved candidate evaluation checkpoint changed")
    if int(result.get("episode_count", -1)) != int(episodes):
        raise EpisodeTrainingError("saved candidate evaluation episode count changed")
    if int(result.get("roi_count", -1)) != int(roi_count):
        raise EpisodeTrainingError("saved candidate evaluation RoI count changed")
    if result.get("scenario_manifest_hash") != baseline_preflight.get(
        "manifest_content_hash"
    ):
        raise EpisodeTrainingError("saved candidate evaluation manifest changed")
    raw = result.get("raw_result") or {}
    points = raw.get("points") or []
    if len(points) != 1:
        raise EpisodeTrainingError("saved candidate evaluation lacks one complete point")
    rows = _episode_rows(
        (points[0].get("outputs") or {}).get("per_episode_jsonl"),
        expected_count=episodes,
    )
    if [row["scenario_id"] for row in rows] != baseline_preflight.get("scenario_ids"):
        raise EpisodeTrainingError("saved candidate evaluation scenario order changed")
    directory = Path(result.get("evaluation_directory", ""))
    required = [directory / name for name in EVALUATION_OUTPUT_FILENAMES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EpisodeTrainingError(
            "saved candidate evaluation is missing required outputs: "
            + ", ".join(missing[:3])
        )
