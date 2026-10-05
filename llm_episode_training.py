"""Subprocess adapters for formal episode-search training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


class EpisodeTrainingError(RuntimeError):
    pass


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
        "summaries": _training_summaries(Path(run_directory)) if run_directory else [],
        "raw_result": result,
    }


def _training_summaries(run_directory: Path) -> list[dict[str, Any]]:
    path = run_directory / "llm_training_episode_metrics.jsonl"
    if not path.is_file():
        return []
    from llm_episode_search import summarize_training_blocks

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return summarize_training_blocks(rows, block_size=100)


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
    if (
        float(point.get("evaluation_environment_width_m", -1.0)) != 1000.0
        or float(point.get("evaluation_environment_height_m", -1.0)) != 1000.0
    ):
        raise EpisodeTrainingError("candidate evaluation did not use a 1000x1000 m environment")
    per_episode = Path(point["outputs"]["per_episode_jsonl"])
    rows = [json.loads(line) for line in per_episode.read_text(encoding="utf-8").splitlines() if line.strip()]
    ee = np.asarray([row["energy_efficiency_mbit_per_j"] for row in rows], dtype=np.float64)
    if ee.size != int(episodes) or np.any(~np.isfinite(ee)):
        raise EpisodeTrainingError("evaluation per-episode EE output is incomplete")
    baseline_path = Path(baseline_evaluation).resolve()
    baseline_run_path = Path(baseline_run).resolve()
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
    if int(baseline_point.get("checkpoint_completed_episodes", -1)) != 1500:
        raise EpisodeTrainingError("baseline evaluation must use the episode-1500 checkpoint")
    if baseline_point.get("scenario_manifest_hash") != point["scenario_manifest_hash"]:
        raise EpisodeTrainingError("baseline and candidate evaluations do not share a manifest")
    if int(baseline_point.get("evaluation_episode_count", -1)) != int(episodes):
        raise EpisodeTrainingError("baseline evaluation episode count is incompatible")
    baseline_rows_path = Path(baseline_point["outputs"]["per_episode_jsonl"])
    baseline_rows = [
        json.loads(line)
        for line in baseline_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_ee = np.asarray(
        [row["energy_efficiency_mbit_per_j"] for row in baseline_rows],
        dtype=np.float64,
    )
    if baseline_ee.size != int(episodes) or np.any(~np.isfinite(baseline_ee)):
        raise EpisodeTrainingError("baseline per-episode EE output is incomplete")
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
        "raw_result": result,
    }
