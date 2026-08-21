"""One-method-at-a-time formal paper evaluation over shared production flows."""

from __future__ import annotations

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
)
from HRL_task_aware import TrainingConfig, train
from Packet_scheduler_v1 import TASK_DEADLINE_SECONDS
from paper_figure_registry import FIGURE_REGISTRY, PAPER_METHOD_MAPPINGS
from paper_figures import normalize_episode_ee
from scenario_manifest import ScenarioManifest, generate_manifest


TRAJECTORY_SNAPSHOT_SECONDS = (5.0, 10.0, 15.0, 25.0)
FIXED_ROI_VALUES = tuple(range(2, 9))
ARRIVAL_RATE_SWEEPS = {
    "COM": {"values": (50.0, 100.0, 150.0, 200.0), "fixed": {"FOV": 5.0}},
    "FOV": {"values": (10.0, 20.0, 30.0, 40.0), "fixed": {"COM": 50.0}},
}
DEADLINE_SWEEP_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

PAPER_EVALUATION_SUITES = {
    "fig2_convergence": {
        "methods": tuple(FIGURE_REGISTRY["fig2"]["methods"]),
        "kind": "training_history",
    },
    "fig3_trajectory": {
        "methods": ("td3_dinkelbach",),
        "kind": "trajectory",
        "requires_manifest": True,
    },
    "fig5_arrival": {
        "methods": tuple(PAPER_METHOD_MAPPINGS["fig5"].values()),
        "kind": "arrival",
        "requires_manifest": True,
    },
    "fig6_deadline": {
        "methods": tuple(PAPER_METHOD_MAPPINGS["fig6"]),
        "kind": "deadline",
        "requires_manifest": True,
    },
    "fig7_fixed_roi": {
        "methods": tuple(PAPER_METHOD_MAPPINGS["fig7"]),
        "kind": "fixed_roi",
    },
}


def validate_production_deadlines():
    expected = {"FOV": 1.5, "COM": 1.0}
    actual = {key: float(value) for key, value in TASK_DEADLINE_SECONDS.items()}
    if actual != expected:
        raise RuntimeError(
            f"production task deadlines differ from the paper contract: {actual}"
        )
    return actual


def evaluation_sweep_points(suite):
    suite = str(suite)
    definition = PAPER_EVALUATION_SUITES.get(suite)
    if definition is None:
        raise ValueError(f"unknown paper evaluation suite: {suite}")
    kind = definition["kind"]
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
                        },
                        "swept_task": task_type,
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
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
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
    run_dir = Path(run_directory).resolve()
    resolved = _read_json(run_dir / "resolved_config.json")
    method = MethodSpec.parse(expected_method)
    if resolved.get("method") != method.method_id:
        raise RuntimeError(
            f"paper method/run mismatch: requested={method.method_id}, "
            f"run={resolved.get('method')}"
        )
    if resolved.get("method_spec") != method.to_dict():
        raise RuntimeError("training run method metadata is incompatible")
    if resolved.get("status") != "COMPLETED":
        raise RuntimeError("paper evaluation requires a completed training run")
    checkpoint = (
        run_dir
        / "checkpoints"
        / "models"
        / f"ep_{FORMAL_CHECKPOINT_EPISODE:04d}"
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"formal ep_{FORMAL_CHECKPOINT_EPISODE} checkpoint is missing: {checkpoint}"
        )
    training_config = dict(resolved["training_config"])
    return {
        "run_dir": run_dir,
        "resolved": resolved,
        "method": method,
        "training_seed": int(resolved["seed"]),
        "checkpoint": checkpoint.resolve(),
        "expected_training_config": training_config,
    }


def _evaluation_config(episodes, episode_seconds, training_seed):
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
        random_seed=int(training_seed),
    )


def _manifest_for_point(point, *, base_manifest, manifest_seed, episodes):
    if "fixed_num_gt" in point:
        return generate_manifest(
            "test",
            manifest_seed=int(manifest_seed),
            episode_count=int(episodes),
            num_gt=int(point["fixed_num_gt"]),
        )
    if base_manifest is None:
        raise ValueError("this paper suite requires an explicit common manifest")
    if base_manifest.episode_count < int(episodes):
        raise ValueError("common manifest has fewer entries than requested episodes")
    if base_manifest.episode_count == int(episodes):
        return base_manifest
    # Preserve the exact first N scenario entries and recompute no identities by
    # requiring callers to provide a manifest with the intended formal count.
    raise ValueError(
        "paper evaluation manifest episode_count must equal requested episodes"
    )


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_paper_evaluation(
    method_id,
    *,
    run_directory,
    suite,
    manifest_path=None,
    manifest_seed=None,
    episodes=None,
    episode_seconds=None,
    output_root="results/paper_evaluations",
):
    validate_production_deadlines()
    definition = PAPER_EVALUATION_SUITES.get(str(suite))
    if definition is None:
        raise ValueError(f"unknown paper evaluation suite: {suite}")
    method = MethodSpec.parse(method_id)
    if method.method_id not in definition["methods"]:
        raise ValueError(
            f"{method.method_id} is not part of {suite}: {definition['methods']}"
        )
    context = _load_training_run(run_directory, method.method_id)
    git_sha = _git_sha()
    output_dir = _unique_directory(
        Path(output_root) / str(suite) / method.method_id,
        git_sha,
    )
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
            "suite": suite,
            "method_id": method.method_id,
            "training_run": str(context["run_dir"]),
            "training_history": str(history_path.resolve()),
            "git_sha": git_sha,
            "training_history_only": True,
            "new_training_started": False,
        }
        _write_json(output_dir / "paper_evaluation_metadata.json", metadata)
        return {"output_directory": str(output_dir), **metadata}

    base_manifest = (
        ScenarioManifest.load(manifest_path) if manifest_path is not None else None
    )
    if definition.get("requires_manifest") and base_manifest is None:
        raise ValueError(f"{suite} requires --manifest for shared scenarios")
    resolved_episodes = int(
        episodes
        if episodes is not None
        else (
            base_manifest.episode_count
            if base_manifest is not None
            else FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
        )
    )
    resolved_seconds = int(
        episode_seconds
        if episode_seconds is not None
        else FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"]
    )
    resolved_manifest_seed = int(
        manifest_seed
        if manifest_seed is not None
        else DEFAULT_TRAINING_SEED
    )
    point_results = []
    for point in evaluation_sweep_points(suite):
        manifest = _manifest_for_point(
            point,
            base_manifest=base_manifest,
            manifest_seed=resolved_manifest_seed,
            episodes=resolved_episodes,
        )
        point_dir = output_dir / point["point_id"]
        point_dir.mkdir()
        manifest.save(point_dir / "scenario_manifest.json")
        result = train(
            _evaluation_config(
                resolved_episodes,
                resolved_seconds,
                context["training_seed"],
            ),
            scenario_manifest=manifest,
            method_spec=method,
            evaluation=True,
            checkpoint_dir=context["checkpoint"],
            expected_checkpoint_episodes=FORMAL_CHECKPOINT_EPISODE,
            expected_checkpoint_formal_config=context["expected_training_config"],
            evaluation_overrides=point.get("overrides"),
            trajectory_snapshot_times=point.get("snapshot_times_seconds"),
        )
        outputs = write_evaluation_outputs(
            point_dir,
            result["episode_metrics"],
            {
                **result["run_metadata"],
                "paper_suite": suite,
                "paper_sweep_point": point,
                "git_sha": git_sha,
                "checkpoint_path": str(context["checkpoint"]),
                "scenario_manifest": str(
                    (point_dir / "scenario_manifest.json").resolve()
                ),
            },
        )
        _write_json(
            point_dir / "packet_outcomes.json",
            result["packet_outcome_artifacts"],
        )
        trajectories = [
            {**artifact, "git_sha": git_sha}
            for artifact in result["trajectory_artifacts"]
        ]
        if trajectories:
            _write_json(point_dir / "trajectory_artifacts.json", trajectories)
        point_results.append(
            {
                **point,
                "manifest_hash": manifest.content_hash,
                "checkpoint_metadata_fingerprint": result["run_metadata"].get(
                    "checkpoint_metadata_fingerprint"
                ),
                "output_directory": str(point_dir.resolve()),
                "outputs": {key: str(value) for key, value in outputs.items()},
            }
        )
    metadata = {
        "suite": suite,
        "method_id": method.method_id,
        "method_spec": method.to_dict(),
        "training_run": str(context["run_dir"]),
        "checkpoint_path": str(context["checkpoint"]),
        "checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
        "training_seed": context["training_seed"],
        "manifest_seed": resolved_manifest_seed,
        "evaluation_episodes_per_point": resolved_episodes,
        "evaluation_horizon_seconds": resolved_seconds,
        "git_sha": git_sha,
        "new_training_started": False,
        "points": point_results,
    }
    _write_json(output_dir / "paper_evaluation_metadata.json", metadata)
    return {"output_directory": str(output_dir), **metadata}
