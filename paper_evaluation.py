"""Semantic paper evaluation suites over the shared production simulator."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluation_metrics import write_evaluation_outputs
from experiment_config import (
    DEFAULT_TRAINING_SEED,
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    NUM_UAV,
)
from HRL_task_aware import TrainingConfig, train
from Packet_scheduler_v1 import TASK_DEADLINE_SECONDS
from paper_figure_registry import FIGURE_REGISTRY, PAPER_METHOD_MAPPINGS
from paper_metrics import (
    aggregate_paper_point_metrics,
    normalize_episode_ee,
    validate_canonical_aggregate_rows,
)
from scenario_manifest import ScenarioManifest, generate_manifest
from training_checkpoint import CHECKPOINT_PROVENANCE_FIELDS


TRAJECTORY_SNAPSHOT_SECONDS = (5.0, 10.0, 15.0, 25.0)
FIXED_ROI_VALUES = tuple(range(2, 9))
ARRIVAL_RATE_SWEEPS = {
    "COM": {"values": (50.0, 100.0, 150.0, 200.0), "fixed": {"FOV": 5.0}},
    "FOV": {"values": (10.0, 20.0, 30.0, 40.0), "fixed": {"COM": 50.0}},
}
DEADLINE_SWEEP_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS = 57.0

_FIXED_ROI_METHODS = tuple(
    dict.fromkeys(
        (
            *PAPER_METHOD_MAPPINGS["task_assignment_ee_vs_number_of_rois"].values(),
            *PAPER_METHOD_MAPPINGS["trajectory_design_ee_vs_number_of_rois"],
            *PAPER_METHOD_MAPPINGS["hierarchical_architecture_ee_vs_number_of_rois"],
            *PAPER_METHOD_MAPPINGS["task_type_delay_vs_number_of_rois"],
        )
    )
)

PAPER_EVALUATION_SUITES = {
    "training_ee_vs_episode": {
        "methods": tuple(FIGURE_REGISTRY["training_ee_vs_episode"]["methods"]),
        "kind": "training_history",
    },
    "uav_trajectory_snapshots": {
        "methods": ("td3_dinkelbach",),
        "kind": "trajectory",
        "requires_manifest": True,
    },
    "task_type_delay_vs_arrival_rate": {
        "methods": tuple(
            PAPER_METHOD_MAPPINGS["task_type_delay_vs_arrival_rate"].values()
        ),
        "kind": "arrival",
        "requires_manifest": True,
    },
    "task_type_delay_violation_vs_target_delay": {
        "methods": tuple(
            PAPER_METHOD_MAPPINGS[
                "task_type_delay_violation_vs_target_delay"
            ]
        ),
        "kind": "deadline",
        "requires_manifest": True,
    },
    "fixed_roi": {"methods": _FIXED_ROI_METHODS, "kind": "fixed_roi"},
}

DEPRECATED_SUITE_ALIASES = {
    "fig2_convergence": "training_ee_vs_episode",
    "fig3_trajectory": "uav_trajectory_snapshots",
    "fig5_arrival": "task_type_delay_vs_arrival_rate",
    "fig6_deadline": "task_type_delay_violation_vs_target_delay",
    "fig7_fixed_roi": "fixed_roi",
}


def resolve_evaluation_suite(suite):
    key = str(suite).strip().lower()
    key = DEPRECATED_SUITE_ALIASES.get(key, key)
    if key not in PAPER_EVALUATION_SUITES:
        raise ValueError(f"unknown paper evaluation suite: {suite}")
    return key


def validate_production_deadlines():
    expected = {"FOV": 1.5, "COM": 1.0}
    actual = {key: float(value) for key, value in TASK_DEADLINE_SECONDS.items()}
    if actual != expected:
        raise RuntimeError(
            f"production task deadlines differ from the paper contract: {actual}"
        )
    return actual


def evaluation_sweep_points(suite):
    suite = resolve_evaluation_suite(suite)
    kind = PAPER_EVALUATION_SUITES[suite]["kind"]
    if kind == "training_history":
        return ({"point_id": "training_history", "overrides": {}},)
    if kind == "trajectory":
        return (
            {
                "point_id": "trajectory",
                "overrides": {},
                "snapshot_times_seconds": TRAJECTORY_SNAPSHOT_SECONDS,
            },
        )
    if kind == "arrival":
        points = []
        for task_type, sweep in ARRIVAL_RATE_SWEEPS.items():
            for value in sweep["values"]:
                rates = {task_type: value, **sweep["fixed"]}
                points.append(
                    {
                        "point_id": f"{task_type.lower()}_rate_{value:g}",
                        "overrides": {
                            "fov_rate_packets_per_second": rates["FOV"],
                            "com_rate_packets_per_second": rates["COM"],
                        },
                        "swept_task": task_type,
                        "display_task": "VS" if task_type == "FOV" else "COM",
                        "x_value": value,
                        "x_unit": "packets/s",
                    }
                )
        return tuple(points)
    if kind == "deadline":
        defaults = validate_production_deadlines()
        points = []
        for task_type in ("COM", "FOV"):
            for value in DEADLINE_SWEEP_SECONDS:
                deadlines = dict(defaults)
                deadlines[task_type] = value
                points.append(
                    {
                        "point_id": f"{task_type.lower()}_deadline_{value:g}s",
                        "overrides": {
                            "fov_deadline_seconds": deadlines["FOV"],
                            "com_deadline_seconds": deadlines["COM"],
                            "packet_injection_cutoff_seconds": (
                                DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS
                            ),
                        },
                        "swept_task": task_type,
                        "display_task": "VS" if task_type == "FOV" else "COM",
                        "x_value": value,
                        "x_unit": "seconds",
                    }
                )
        return tuple(points)
    if kind == "fixed_roi":
        return tuple(
            {
                "point_id": f"roi_{num_gt}",
                "overrides": {},
                "fixed_num_gt": num_gt,
                "x_value": num_gt,
                "x_unit": "RoIs",
            }
            for num_gt in FIXED_ROI_VALUES
        )
    raise RuntimeError(f"unsupported paper suite kind: {kind}")


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _unique_directory(root, git_sha):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt:02d}"
        candidate = root / f"{stamp}_{git_sha[:12]}{suffix}"
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique directory below {root}")


def _load_training_run(run_directory, expected_method):
    if run_directory is None:
        raise ValueError(f"{expected_method} requires --run-dir")
    run_dir = Path(run_directory).resolve()
    resolved = _read_json(run_dir / "resolved_config.json")
    method = MethodSpec.parse(expected_method)
    if resolved.get("method") != method.method_id:
        raise RuntimeError(
            f"paper method/run mismatch: requested={method.method_id}, run={resolved.get('method')}"
        )
    if resolved.get("method_spec") != method.to_dict():
        raise RuntimeError("training run method metadata is incompatible")
    if resolved.get("status") != "COMPLETED":
        raise RuntimeError("paper evaluation requires a completed training run")
    checkpoint = run_dir / "checkpoints" / "models" / f"ep_{FORMAL_CHECKPOINT_EPISODE:04d}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"formal ep_{FORMAL_CHECKPOINT_EPISODE} checkpoint is missing: {checkpoint}"
        )
    return {
        "run_dir": run_dir,
        "resolved": resolved,
        "method": method,
        "training_seed": int(resolved["seed"]),
        "checkpoint": checkpoint.resolve(),
        "expected_training_config": dict(resolved["training_config"]),
        "checkpoint_required": True,
    }


def _no_checkpoint_context(method, evaluation_seed, run_directory):
    if run_directory is not None:
        raise ValueError(
            f"{method.method_id} is a pure-random baseline and must not receive --run-dir"
        )
    return {
        "run_dir": None,
        "resolved": None,
        "method": method,
        "training_seed": int(evaluation_seed),
        "checkpoint": None,
        "expected_training_config": None,
        "checkpoint_required": False,
    }


def _evaluation_config(episodes, episode_seconds, seed):
    return TrainingConfig(
        total_episodes=int(episodes),
        mode="custom",
        episode_seconds=int(episode_seconds),
        routing_slot_seconds=FORMAL_EXPERIMENT_DEFAULTS["routing_slot_seconds"],
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=int(seed),
    )


def _manifest_for_point(point, *, base_manifest, manifest_seed, episodes):
    if "fixed_num_gt" in point:
        return generate_manifest(
            "test", int(manifest_seed), int(episodes), num_gt=int(point["fixed_num_gt"])
        )
    if base_manifest is None:
        raise ValueError("this paper suite requires an explicit common manifest")
    if base_manifest.episode_count != int(episodes):
        raise ValueError(
            "paper evaluation manifest episode_count must equal requested episodes"
        )
    return base_manifest


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty paper aggregate")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_paper_evaluation(
    method_id,
    *,
    run_directory=None,
    suite,
    manifest_path=None,
    manifest_seed=None,
    episodes=None,
    episode_seconds=None,
    target_uav_id=None,
    output_root="results/paper_evaluations",
):
    validate_production_deadlines()
    suite = resolve_evaluation_suite(suite)
    definition = PAPER_EVALUATION_SUITES[suite]
    method = MethodSpec.parse(method_id)
    if method.method_id not in definition["methods"]:
        raise ValueError(f"{method.method_id} is not part of {suite}: {definition['methods']}")

    requested_manifest_seed = int(
        DEFAULT_TRAINING_SEED if manifest_seed is None else manifest_seed
    )
    checkpoint_required = bool(method.learns_movement or method.learns_routing)
    context = (
        _load_training_run(run_directory, method.method_id)
        if checkpoint_required
        else _no_checkpoint_context(method, requested_manifest_seed, run_directory)
    )
    git_sha = _git_sha()
    output_dir = _unique_directory(Path(output_root) / suite / method.method_id, git_sha)

    if definition["kind"] == "training_history":
        history_path = context["run_dir"] / "training_history.jsonl"
        rows = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        normalized = normalize_episode_ee(method.method_id, rows)
        _write_json(output_dir / "normalized_plot_data.json", normalized)
        metadata = {
            "semantic_suite": suite,
            "method_id": method.method_id,
            "training_run": str(context["run_dir"]),
            "training_history": str(history_path.resolve()),
            "checkpoint_required": True,
            "git_sha": git_sha,
            "training_history_only": True,
            "new_training_started": False,
        }
        _write_json(output_dir / "paper_evaluation_metadata.json", metadata)
        return {"output_directory": str(output_dir), **metadata}

    base_manifest = ScenarioManifest.load(manifest_path) if manifest_path is not None else None
    if definition.get("requires_manifest") and base_manifest is None:
        raise ValueError(f"{suite} requires --manifest for shared scenarios")
    default_episodes = (
        1
        if definition["kind"] == "trajectory"
        else (
            base_manifest.episode_count
            if base_manifest is not None
            else FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
        )
    )
    resolved_episodes = int(default_episodes if episodes is None else episodes)
    resolved_seconds = int(
        FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"]
        if episode_seconds is None
        else episode_seconds
    )
    if definition["kind"] == "trajectory" and target_uav_id is None:
        raise ValueError("uav_trajectory_snapshots requires --target-uav-id")

    point_results = []
    all_aggregates = []
    for point in evaluation_sweep_points(suite):
        manifest = _manifest_for_point(
            point,
            base_manifest=base_manifest,
            manifest_seed=requested_manifest_seed,
            episodes=resolved_episodes,
        )
        point_dir = output_dir / point["point_id"]
        point_dir.mkdir()
        manifest.save(point_dir / "scenario_manifest.json")
        result = train(
            _evaluation_config(resolved_episodes, resolved_seconds, context["training_seed"]),
            scenario_manifest=manifest,
            method_spec=method,
            evaluation=True,
            checkpoint_dir=context["checkpoint"],
            expected_checkpoint_episodes=(
                FORMAL_CHECKPOINT_EPISODE if checkpoint_required else None
            ),
            expected_checkpoint_formal_config=context["expected_training_config"],
            evaluation_overrides=point.get("overrides"),
            trajectory_snapshot_times=point.get("snapshot_times_seconds"),
            trajectory_target_uav_id=(
                int(target_uav_id) if definition["kind"] == "trajectory" else None
            ),
        )
        run_metadata = {
            **result["run_metadata"],
            "semantic_suite": suite,
            "paper_sweep_point": point,
            "git_sha": git_sha,
            "checkpoint_required": checkpoint_required,
            "checkpoint_path": (
                str(context["checkpoint"]) if checkpoint_required else None
            ),
            "scenario_manifest": str((point_dir / "scenario_manifest.json").resolve()),
        }
        outputs = write_evaluation_outputs(point_dir, result["episode_metrics"], run_metadata)
        _write_json(point_dir / "packet_outcomes.json", result["packet_outcome_artifacts"])
        trajectories = [
            {
                **artifact,
                "semantic_suite": suite,
                "method_id": method.method_id,
                "method_spec": method.to_dict(),
                "git_sha": git_sha,
            }
            for artifact in result["trajectory_artifacts"]
        ]
        if trajectories:
            _write_json(point_dir / "trajectory_artifacts.json", trajectories)
        aggregates = aggregate_paper_point_metrics(
            method.method_id, suite, point, result["episode_metrics"]
        )
        validate_canonical_aggregate_rows(
            aggregates, method.method_id, point["point_id"]
        )
        _write_json(point_dir / "aggregated_plot_data.json", aggregates)
        _write_csv(point_dir / "aggregated_plot_data.csv", aggregates)
        all_aggregates.extend(aggregates)
        point_results.append(
            {
                **point,
                "scenario_manifest_path": str(
                    (point_dir / "scenario_manifest.json").resolve()
                ),
                "scenario_manifest_hash": manifest.content_hash,
                "manifest_hash": manifest.content_hash,
                "scenario_ids": list(result["scenario_ids"]),
                "evaluation_episode_count": resolved_episodes,
                "evaluation_horizon_seconds": resolved_seconds,
                "evaluation_seed": context["training_seed"],
                "manifest_seed": manifest.manifest_seed,
                "num_uav": NUM_UAV,
                "resolved_overrides": result["run_metadata"].get(
                    "evaluation_overrides"
                ),
                "checkpoint_required": checkpoint_required,
                "checkpoint_path": (
                    str(context["checkpoint"]) if checkpoint_required else None
                ),
                **{
                    field: result["run_metadata"].get(field)
                    for field in CHECKPOINT_PROVENANCE_FIELDS
                },
                "output_directory": str(point_dir.resolve()),
                "outputs": {key: str(value) for key, value in outputs.items()},
                "aggregated_plot_data": str(
                    (point_dir / "aggregated_plot_data.json").resolve()
                ),
            }
        )
    _write_json(output_dir / "aggregated_plot_data.json", all_aggregates)
    _write_csv(output_dir / "aggregated_plot_data.csv", all_aggregates)
    metadata = {
        "semantic_suite": suite,
        "method_id": method.method_id,
        "method_spec": method.to_dict(),
        "training_run": str(context["run_dir"]) if context["run_dir"] else None,
        "checkpoint_required": checkpoint_required,
        "checkpoint_path": str(context["checkpoint"]) if context["checkpoint"] else None,
        "checkpoint_episode": FORMAL_CHECKPOINT_EPISODE if checkpoint_required else None,
        **{
            field: point_results[0].get(field) if point_results else None
            for field in CHECKPOINT_PROVENANCE_FIELDS
        },
        "formal_training_config": context["expected_training_config"],
        "no_checkpoint_reason": (
            None
            if checkpoint_required
            else "pure-random movement and routing have no neural checkpoint state"
        ),
        "training_seed": context["training_seed"] if checkpoint_required else None,
        "evaluation_seed": context["training_seed"],
        "manifest_seed": requested_manifest_seed,
        "evaluation_episodes_per_point": resolved_episodes,
        "evaluation_horizon_seconds": resolved_seconds,
        "target_uav_id": int(target_uav_id) if target_uav_id is not None else None,
        "git_sha": git_sha,
        "new_training_started": False,
        "aggregation": {
            "delay": "sum delivered E2E delay / sum delivered packets",
            "violation_probability": "sum violations / sum generated packets",
            "energy_efficiency": "sum timely delivered Mbit / max(sum mobility J, epsilon)",
            "zero_delivered_delay": "null with missing=true",
        },
        "points": point_results,
    }
    _write_json(output_dir / "paper_evaluation_metadata.json", metadata)
    return {"output_directory": str(output_dir), **metadata}
