"""Preflight and sequential execution for checkpoint-by-RoI evaluations."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid

from evaluation_selection import (
    resolve_checkpoint_episodes,
    resolve_roi_counts,
    resolve_training_run_checkpoint,
    validate_fixed_roi_manifest,
)
from experiment_config import (
    DEFAULT_TRAINING_SEED,
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
)
from paper_evaluation import run_paper_evaluation
from paper_metrics import validate_canonical_aggregate_rows
from scenario_manifest import generate_manifest
from training_checkpoint import CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS


CHECKPOINT_ROI_SWEEP_SCHEMA_VERSION = "uav-hrl-checkpoint-roi-sweep-v1"
SUMMARY_FIELDS = (
    "method_id",
    "training_run_id",
    "checkpoint_episode",
    "checkpoint_planned_total_episodes",
    "current_training_run_total_episodes",
    "horizon_extension_compatible",
    "allowed_horizon_differences",
    "checkpoint_training_manifest_hash",
    "current_training_manifest_hash",
    "manifest_prefix_compatible",
    "training_total_episodes",
    "roi_count",
    "evaluation_episode_count",
    "manifest_seed",
    "manifest_hash",
    "energy_efficiency_bit_per_joule",
    "timely_delivered_bits",
    "timely_goodput_bps",
    "propulsion_energy_j",
    "mobility_energy_j",
    "coverage",
    "eligible_packet_count",
    "delay_violation_count",
    "delay_violation_probability",
    "sr_admission_drop_count",
    "result_directory",
    "status",
)
METRIC_FIELD_MAPPING = {
    "energy_efficiency_bit_per_joule": (
        "canonical energy_efficiency_mbit_per_j row value multiplied by 1e6"
    ),
    "timely_delivered_bits": (
        "canonical energy-efficiency numerator timely_goodput_mbits multiplied by 1e6"
    ),
    "timely_goodput_bps": (
        "timely_delivered_bits / (evaluation_episode_count * episode_seconds)"
    ),
    "propulsion_energy_j": (
        "canonical total_mobility_energy_j sum; mobility energy is propulsion-only"
    ),
    "mobility_energy_j": "canonical energy-efficiency denominator",
    "coverage": "arithmetic mean of canonical per-episode coverage",
    "eligible_packet_count": "canonical ALL violation denominator",
    "delay_violation_count": "canonical ALL violation numerator",
    "delay_violation_probability": (
        "canonical ALL violation value; null when eligible_packet_count is zero"
    ),
    "sr_admission_drop_count": (
        "sum of canonical per-episode sr_admission_drop_count"
    ),
}


class SweepPreflightError(RuntimeError):
    def __init__(self, errors):
        self.errors = tuple(str(error) for error in errors)
        super().__init__(
            "checkpoint RoI sweep preflight failed:\n- "
            + "\n- ".join(self.errors)
        )


class SweepExecutionError(RuntimeError):
    pass


def _git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _batch_directory(output_root, git_sha, batch_id=None):
    root = Path(output_root).resolve()
    identity = batch_id
    if identity is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        identity = f"{stamp}_{git_sha[:12]}_{uuid.uuid4().hex[:8]}"
    identity = str(identity)
    if not identity or any(char in identity for char in '<>:"/\\|?*'):
        raise ValueError("checkpoint RoI sweep batch identity is invalid")
    return root / identity


def _public_point(point):
    return {
        key: value
        for key, value in point.items()
        if not key.startswith("_")
    }


def public_sweep_plan(plan):
    return {
        "schema_version": CHECKPOINT_ROI_SWEEP_SCHEMA_VERSION,
        "batch_output_directory": str(plan["batch_output_directory"]),
        "training_run_count": len(plan["training_run_directories"]),
        "checkpoint_episodes": list(plan["checkpoint_episodes"]),
        "roi_counts": list(plan["roi_counts"]),
        "evaluation_episode_count": plan["evaluation_episode_count"],
        "episode_seconds": plan["episode_seconds"],
        "manifest_seed": plan["manifest_seed"],
        "total_evaluation_points": len(plan["points"]),
        "shared_manifests": {
            str(roi): {
                "path": str(plan["manifest_paths"][roi]),
                "manifest_hash": manifest.content_hash,
                "scenario_ids": [
                    entry["scenario_id"] for entry in manifest.episodes
                ],
            }
            for roi, manifest in plan["manifests"].items()
        },
        "points": [_public_point(point) for point in plan["points"]],
    }


def build_checkpoint_roi_sweep_plan(
    training_run_directories,
    *,
    checkpoint_episode=None,
    checkpoint_episodes=None,
    roi_count=None,
    roi_counts=None,
    evaluation_episodes,
    episode_seconds,
    manifest_seed,
    output_root,
    batch_id=None,
):
    """Validate the entire matrix before allocating any output directory."""

    run_values = tuple(training_run_directories or ())
    if not run_values:
        raise SweepPreflightError(("at least one training run is required",))
    resolved_checkpoints = resolve_checkpoint_episodes(
        checkpoint_episode, checkpoint_episodes
    )
    resolved_rois = resolve_roi_counts(roi_count, roi_counts)
    evaluation_episodes = int(
        FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
        if evaluation_episodes is None
        else evaluation_episodes
    )
    episode_seconds = int(
        FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"]
        if episode_seconds is None
        else episode_seconds
    )
    manifest_seed = int(
        DEFAULT_TRAINING_SEED if manifest_seed is None else manifest_seed
    )
    if evaluation_episodes <= 0 or episode_seconds <= 0:
        raise SweepPreflightError(
            ("evaluation episodes and episode seconds must be positive",)
        )
    git_sha = _git_sha()
    batch_dir = _batch_directory(output_root, git_sha, batch_id=batch_id)
    errors = []
    if batch_dir.exists():
        errors.append(f"batch output directory already exists: {batch_dir}")

    canonical_run_dirs = [Path(value).resolve() for value in run_values]
    duplicates = sorted(
        {
            str(path)
            for path in canonical_run_dirs
            if canonical_run_dirs.count(path) > 1
        }
    )
    if duplicates:
        errors.append(f"duplicate training run directories: {duplicates}")

    manifests = {
        roi: generate_manifest(
            "test", manifest_seed, evaluation_episodes, num_gt=roi
        )
        for roi in resolved_rois
    }
    for roi, manifest in manifests.items():
        validate_fixed_roi_manifest(
            manifest, roi, evaluation_episodes, manifest_seed
        )
    manifest_paths = {
        roi: batch_dir / "manifests" / f"roi_{roi}.json"
        for roi in resolved_rois
    }

    contexts = {}
    for run_dir in canonical_run_dirs:
        for selected_checkpoint in resolved_checkpoints:
            try:
                context = resolve_training_run_checkpoint(
                    run_dir,
                    selected_checkpoint,
                    require_run_metadata=True,
                )
                contexts[(run_dir, selected_checkpoint)] = context
            except Exception as exc:
                errors.append(
                    f"run={run_dir}, checkpoint={selected_checkpoint}: {exc}"
                )

    points = []
    result_directories = set()
    for run_dir in canonical_run_dirs:
        for selected_checkpoint in resolved_checkpoints:
            context = contexts.get((run_dir, selected_checkpoint))
            if context is None:
                continue
            for roi in resolved_rois:
                result_dir = (
                    batch_dir
                    / context["method"].method_id
                    / context["training_run_id"]
                    / f"ep_{selected_checkpoint:04d}"
                    / f"roi_{roi}"
                )
                if result_dir in result_directories:
                    errors.append(
                        f"evaluation output collision: {result_dir}"
                    )
                    continue
                result_directories.add(result_dir)
                is_formal = selected_checkpoint == FORMAL_CHECKPOINT_EPISODE
                points.append(
                    {
                        "method_id": context["method"].method_id,
                        "training_run_id": context["training_run_id"],
                        "training_run_directory": str(context["run_dir"]),
                        "training_total_episodes": context[
                            "training_total_episodes"
                        ],
                        "checkpoint_episode": selected_checkpoint,
                        "checkpoint_path": str(context["checkpoint"]),
                        **{
                            field: context.get(field)
                            for field in CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS
                        },
                        **context["checkpoint_artifact_provenance"],
                        "roi_count": roi,
                        "evaluation_episode_count": evaluation_episodes,
                        "episode_seconds": episode_seconds,
                        "manifest_seed": manifest_seed,
                        "manifest_hash": manifests[roi].content_hash,
                        "scenario_ids": [
                            entry["scenario_id"]
                            for entry in manifests[roi].episodes
                        ],
                        "manifest_path": str(manifest_paths[roi]),
                        "result_directory": str(result_dir),
                        "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
                        "is_formal_checkpoint": is_formal,
                        "evaluation_purpose": (
                            "formal_checkpoint_evaluation"
                            if is_formal
                            else "diagnostic_checkpoint_progress_evaluation"
                        ),
                        "status": "PENDING",
                        "_context": context,
                        "_manifest": manifests[roi],
                    }
                )
    expected_point_count = (
        len(canonical_run_dirs)
        * len(resolved_checkpoints)
        * len(resolved_rois)
    )
    if not errors and len(points) != expected_point_count:
        errors.append(
            "evaluation point count is inconsistent: "
            f"planned={len(points)}, expected={expected_point_count}"
        )
    if errors:
        raise SweepPreflightError(errors)
    return {
        "git_sha": git_sha,
        "batch_output_directory": batch_dir,
        "training_run_directories": tuple(canonical_run_dirs),
        "checkpoint_episodes": resolved_checkpoints,
        "roi_counts": resolved_rois,
        "evaluation_episode_count": evaluation_episodes,
        "episode_seconds": episode_seconds,
        "manifest_seed": manifest_seed,
        "manifests": manifests,
        "manifest_paths": manifest_paths,
        "points": points,
    }


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_summary_files(batch_dir, rows):
    csv_path = batch_dir / "checkpoint_roi_sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in SUMMARY_FIELDS}
            for row in rows
        )
    _write_json(batch_dir / "checkpoint_roi_sweep_summary.json", rows)


def _read_json_value(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"evaluation output is unreadable: {path}") from exc


def _summary_identity(point, *, status, result_directory=None):
    return {
        "method_id": point["method_id"],
        "training_run_id": point["training_run_id"],
        "checkpoint_episode": point["checkpoint_episode"],
        **{
            field: point.get(field)
            for field in CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS
        },
        "training_total_episodes": point["training_total_episodes"],
        "roi_count": point["roi_count"],
        "evaluation_episode_count": point["evaluation_episode_count"],
        "manifest_seed": point["manifest_seed"],
        "manifest_hash": point["manifest_hash"],
        "result_directory": result_directory or point["result_directory"],
        "status": status,
    }


def _completed_summary(point, result):
    result_points = result.get("points") or []
    if len(result_points) != 1:
        raise RuntimeError("checkpoint RoI evaluator returned an invalid point count")
    evaluated = result_points[0]
    if (
        evaluated.get("manifest_hash") != point["manifest_hash"]
        or list(evaluated.get("scenario_ids", ())) != point["scenario_ids"]
        or int(evaluated.get("roi_count", -1)) != point["roi_count"]
    ):
        raise RuntimeError(
            "evaluation point did not use its canonical shared RoI manifest"
        )
    per_episode_path = evaluated.get("outputs", {}).get("per_episode_jsonl")
    rows = [
        json.loads(line)
        for line in Path(per_episode_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != point["evaluation_episode_count"]:
        raise RuntimeError("evaluation per-episode output count is incompatible")
    if any(int(row["num_GT"]) != point["roi_count"] for row in rows):
        raise RuntimeError("evaluation output has an incompatible actual RoI count")
    aggregates = _read_json_value(evaluated["aggregated_plot_data"])
    validate_canonical_aggregate_rows(
        aggregates, point["method_id"], f"roi_{point['roi_count']}"
    )
    energy = next(
        row
        for row in aggregates
        if row["metric"] == "energy_efficiency_mbit_per_j"
    )
    violation = next(
        row
        for row in aggregates
        if row["metric"] == "violation_probability"
        and row["task_type"] == "ALL"
    )
    timely_bits = float(energy["numerator"]) * 1e6
    mobility_joules = float(energy["denominator"])
    horizon_seconds = (
        point["evaluation_episode_count"] * point["episode_seconds"]
    )
    return {
        **_summary_identity(
            point,
            status="COMPLETED",
            result_directory=evaluated["output_directory"],
        ),
        "energy_efficiency_bit_per_joule": float(energy["value"]) * 1e6,
        "timely_delivered_bits": timely_bits,
        "timely_goodput_bps": timely_bits / horizon_seconds,
        "propulsion_energy_j": mobility_joules,
        "mobility_energy_j": mobility_joules,
        "coverage": sum(float(row["coverage"]) for row in rows) / len(rows),
        "eligible_packet_count": int(violation["denominator"]),
        "delay_violation_count": int(violation["numerator"]),
        "delay_violation_probability": violation["value"],
        "sr_admission_drop_count": sum(
            int(row["sr_admission_drop_count"]) for row in rows
        ),
    }


def _metadata(plan, point_statuses, *, status, failure=None):
    public = public_sweep_plan(plan)
    points = []
    for index, point in enumerate(public["points"]):
        points.append({**point, "status": point_statuses[index]})
    return {
        **public,
        "status": status,
        "completed_point_count": sum(
            value == "COMPLETED" for value in point_statuses
        ),
        "metric_field_mapping": METRIC_FIELD_MAPPING,
        "failure": failure,
        "points": points,
    }


def execute_checkpoint_roi_sweep(plan, *, evaluator=run_paper_evaluation):
    """Run one GPU-safe evaluation point at a time and stop on first failure."""

    batch_dir = Path(plan["batch_output_directory"])
    batch_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "manifests").mkdir()
    for roi, manifest in plan["manifests"].items():
        validate_fixed_roi_manifest(
            manifest,
            roi,
            plan["evaluation_episode_count"],
            plan["manifest_seed"],
        )
        manifest.save(plan["manifest_paths"][roi])

    statuses = ["PENDING"] * len(plan["points"])
    summaries = []
    metadata_path = batch_dir / "checkpoint_roi_sweep_metadata.json"
    _write_json(metadata_path, _metadata(plan, statuses, status="RUNNING"))
    _write_summary_files(batch_dir, summaries)

    for index, point in enumerate(plan["points"]):
        try:
            manifest = validate_fixed_roi_manifest(
                point["_manifest"],
                point["roi_count"],
                point["evaluation_episode_count"],
                point["manifest_seed"],
            )
            if manifest.content_hash != point["manifest_hash"]:
                raise RuntimeError("shared manifest changed after preflight")
            result = evaluator(
                point["method_id"],
                run_directory=point["training_run_directory"],
                suite="fixed_roi",
                manifest_seed=point["manifest_seed"],
                episodes=point["evaluation_episode_count"],
                episode_seconds=point["episode_seconds"],
                checkpoint_episode=point["checkpoint_episode"],
                roi_counts=(point["roi_count"],),
                output_directory=point["result_directory"],
                fixed_roi_manifests={point["roi_count"]: manifest},
                allow_registered_fixed_roi_method=True,
                flatten_single_point=True,
            )
            summary = _completed_summary(point, result)
        except Exception as exc:
            statuses[index] = "FAILED"
            failed = {
                **_summary_identity(point, status="FAILED"),
                **{
                    field: None
                    for field in SUMMARY_FIELDS
                    if field not in _summary_identity(point, status="FAILED")
                },
            }
            summaries.append(failed)
            failure = {
                "method_id": point["method_id"],
                "training_run_directory": point["training_run_directory"],
                "checkpoint_episode": point["checkpoint_episode"],
                "roi_count": point["roi_count"],
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _write_summary_files(batch_dir, summaries)
            _write_json(
                metadata_path,
                _metadata(plan, statuses, status="FAILED", failure=failure),
            )
            raise SweepExecutionError(
                "checkpoint RoI evaluation failed: "
                f"method={point['method_id']}, "
                f"run={point['training_run_directory']}, "
                f"checkpoint={point['checkpoint_episode']}, "
                f"roi={point['roi_count']}: {exc}"
            ) from exc
        statuses[index] = "COMPLETED"
        summaries.append(summary)
        _write_summary_files(batch_dir, summaries)
        _write_json(metadata_path, _metadata(plan, statuses, status="RUNNING"))

    final_metadata = _metadata(plan, statuses, status="COMPLETED")
    _write_json(metadata_path, final_metadata)
    return {
        "output_directory": str(batch_dir),
        "metadata": final_metadata,
        "summary_rows": summaries,
    }
