"""Subprocess adapters for formal episode-search training and evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from evaluation_selection import resolve_training_run_checkpoint
from scenario_manifest import ScenarioManifest
from training_checkpoint import CHECKPOINT_PROVENANCE_FIELDS


class EpisodeTrainingError(RuntimeError):
    pass


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
            f"combined output:\n{output[-8000:]}"
        )
    return _last_json_object(output)


def run_candidate_training(
    *,
    method_id: str,
    artifact: str | Path,
    episodes: int,
    seed: int,
    output_directory: str | Path,
    resume_record: dict[str, Any] | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    discovered = []
    method_root = output / str(method_id)
    if method_root.is_dir():
        discovered = sorted(path for path in method_root.iterdir() if path.is_dir())
    resume_directory = (
        Path(resume_record["run_directory"])
        if resume_record and resume_record.get("run_directory")
        else discovered[0]
        if len(discovered) == 1
        else None
    )
    if len(discovered) > 1 and resume_directory is None:
        raise EpisodeTrainingError("multiple incomplete training runs require explicit resume state")
    if resume_directory is not None:
        resolved_path = resume_directory / "resolved_config.json"
        resolved = (
            json.loads(resolved_path.read_text(encoding="utf-8"))
            if resolved_path.is_file()
            else {}
        )
        if resolved.get("status") == "COMPLETED" and int(
            resolved.get("episodes", -1)
        ) == int(episodes):
            result = {"run_directory": str(resume_directory), "status": "COMPLETED"}
        else:
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
    else:
        result = _run(
            [
                sys.executable,
                str(root / "run_experiment.py"),
                str(method_id),
                "--llm-artifact",
                str(Path(artifact).resolve()),
                "--episodes",
                str(int(episodes)),
                "--seed",
                str(int(seed)),
                "--output-root",
                str(output),
            ],
            cwd=root,
        )
    run_directory = result.get("run_directory")
    return {
        "status": "complete" if result.get("status") == "COMPLETED" else "incomplete",
        "run_directory": run_directory,
        "episodes": int(episodes),
        "method_id": method_id,
        "summaries": (
            _training_summaries(Path(run_directory), expected_episodes=int(episodes))
            if run_directory
            else []
        ),
        "raw_result": result,
    }


def _training_summaries(
    run_directory: Path, *, expected_episodes: int
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
            rows, block_size=100, expected_episodes=expected_episodes
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
    manifest_path = Path(manifest).resolve()
    scenario_manifest = ScenarioManifest.load(manifest_path)
    if scenario_manifest.split != "test":
        raise EpisodeTrainingError("baseline comparison manifest must use the test split")
    if scenario_manifest.episode_count != int(episodes):
        raise EpisodeTrainingError("baseline comparison manifest episode count is incompatible")
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
        raise EpisodeTrainingError("baseline evaluation did not use checkpoint episode 1500")
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
    result = _run(
        [
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
            "--output-root",
            str(output),
        ],
        cwd=root,
    )
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
    if int(baseline_point.get("checkpoint_episode", -1)) != 1500:
        raise EpisodeTrainingError("baseline evaluation must use the episode-1500 checkpoint")
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
    return {
        "status": "complete",
        "method_id": method_id,
        "evaluation_directory": result["output_directory"],
        "scenario_manifest_hash": point["scenario_manifest_hash"],
        "scenario_manifest_path": point["scenario_manifest_path"],
        "episode_count": len(rows),
        "roi_count": int(roi_count),
        "environment_size_m": [1000.0, 1000.0],
        "mean_episode_energy_efficiency_mbit_per_j": float(np.mean(ee)),
        "std_episode_energy_efficiency_mbit_per_j": float(np.std(ee)),
        "baseline_run": str(baseline_run_path),
        "baseline_evaluation": str(baseline_path),
        "baseline_mean_episode_energy_efficiency_mbit_per_j": float(
            np.mean(baseline_ee)
        ),
        "improves_over_baseline_mean_episode_ee": bool(
            float(np.mean(ee)) > float(np.mean(baseline_ee))
        ),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "raw_result": result,
    }
