"""Semantic paper evaluation suites over the shared production simulator."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from evaluation_metrics import write_evaluation_outputs
from evaluation_selection import (
    DEFAULT_FIXED_ROI_COUNTS,
    resolve_checkpoint_episodes,
    resolve_roi_counts,
    resolve_training_run_checkpoint,
    validate_fixed_roi_manifest,
)
from experiment_config import (
    DEFAULT_TRAINING_SEED,
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    NUM_UAV,
    PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS,
    PRODUCTION_TASK_DEADLINE_SECONDS,
    comparison_method_configuration,
)
from HRL_task_aware import TrainingConfig, train
from Packet_scheduler_v1 import TASK_DEADLINE_SECONDS
from packet_outcome_artifacts import (
    PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
    PACKET_OUTCOME_MODE_DISABLED,
    PACKET_OUTCOME_MODE_STREAMING,
    PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
    PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS,
    PacketOutcomeJsonlWriter,
    write_packet_routing_diagnostic_artifacts,
)
from routing_q_score_diagnostics import (
    ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION,
    ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS,
    write_routing_q_score_diagnostic_artifacts,
)
from paper_figure_registry import FIGURE_REGISTRY, PAPER_METHOD_MAPPINGS
from paper_metrics import (
    PAPER_AGGREGATE_SCHEMA_VERSION,
    aggregate_paper_point_metrics,
    normalize_episode_ee,
    validate_canonical_aggregate_rows,
)
from scenario_manifest import (
    ScenarioManifest,
    generate_manifest,
    validate_manifest_initial_topologies,
)
from training_checkpoint import (
    CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS,
    CHECKPOINT_PROVENANCE_FIELDS,
    EVALUATION_PROVENANCE_FIELDS,
)


TRAJECTORY_SNAPSHOT_SECONDS = (5.0, 10.0, 15.0, 25.0)
FIXED_ROI_VALUES = DEFAULT_FIXED_ROI_COUNTS
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
    expected = {
        key: float(value)
        for key, value in PRODUCTION_TASK_DEADLINE_SECONDS.items()
    }
    actual = {key: float(value) for key, value in TASK_DEADLINE_SECONDS.items()}
    if actual != expected:
        raise RuntimeError(
            f"production task deadlines differ from the paper contract: {actual}"
        )
    horizon = float(FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"])
    if not math.isclose(
        float(PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS)
        + max(actual.values()),
        horizon,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("production packet cutoff does not preserve a full deadline")
    return actual


def _validate_deadline_seconds(deadline_seconds):
    try:
        values = tuple(float(value) for value in deadline_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadline_seconds values must be floats") from exc
    if not values:
        raise ValueError("deadline_seconds must contain at least one value")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("deadline_seconds values must all be finite")
    if any(value <= 0.0 for value in values):
        raise ValueError("deadline_seconds values must all be greater than zero")
    if len(set(values)) != len(values):
        raise ValueError("deadline_seconds values must not contain duplicates")
    return values


def evaluation_sweep_points(
    suite,
    roi_counts=None,
    deadline_seconds=None,
    episode_seconds=None,
):
    suite = resolve_evaluation_suite(suite)
    kind = PAPER_EVALUATION_SUITES[suite]["kind"]
    if roi_counts is not None and kind != "fixed_roi":
        raise ValueError("RoI selectors are available only for the fixed_roi suite")
    if deadline_seconds is not None and kind != "deadline":
        raise ValueError(
            "deadline_seconds is available only for the "
            "task_type_delay_violation_vs_target_delay suite"
        )
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
        if deadline_seconds is None:
            sweep_seconds = DEADLINE_SWEEP_SECONDS
            injection_cutoff_seconds = (
                DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS
            )
        else:
            sweep_seconds = _validate_deadline_seconds(deadline_seconds)
            if episode_seconds is None:
                raise ValueError(
                    "episode_seconds is required for custom deadline_seconds"
                )
            horizon_seconds = float(episode_seconds)
            if not math.isfinite(horizon_seconds) or horizon_seconds <= 0.0:
                raise ValueError(
                    "episode_seconds must be finite and greater than zero for "
                    "custom deadline_seconds"
                )
            max_required_deadline = max(
                max(sweep_seconds),
                defaults["FOV"],
                defaults["COM"],
            )
            if max_required_deadline >= horizon_seconds:
                raise ValueError(
                    "maximum required deadline must be less than episode_seconds: "
                    f"deadline={max_required_deadline:g}, "
                    f"episode_seconds={horizon_seconds:g}"
                )
            injection_cutoff_seconds = (
                horizon_seconds - max_required_deadline
            )
        points = []
        for task_type in ("COM", "FOV"):
            for value in sweep_seconds:
                deadlines = dict(defaults)
                deadlines[task_type] = value
                points.append(
                    {
                        "point_id": f"{task_type.lower()}_deadline_{value:g}s",
                        "overrides": {
                            "fov_deadline_seconds": deadlines["FOV"],
                            "com_deadline_seconds": deadlines["COM"],
                            "packet_injection_cutoff_seconds": injection_cutoff_seconds,
                        },
                        "swept_task": task_type,
                        "display_task": "VS" if task_type == "FOV" else "COM",
                        "x_value": value,
                        "x_unit": "seconds",
                    }
                )
        return tuple(points)
    if kind == "fixed_roi":
        resolved_roi_counts = (
            FIXED_ROI_VALUES
            if roi_counts is None
            else resolve_roi_counts(roi_counts=roi_counts)
        )
        return tuple(
            {
                "point_id": f"roi_{num_gt}",
                "overrides": {},
                "fixed_num_gt": num_gt,
                "x_value": num_gt,
                "x_unit": "RoIs",
            }
            for num_gt in resolved_roi_counts
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


def _load_training_run(
    run_directory,
    expected_method,
    checkpoint_episode=FORMAL_CHECKPOINT_EPISODE,
):
    return resolve_training_run_checkpoint(
        run_directory,
        checkpoint_episode,
        expected_method=expected_method,
    )


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
        "checkpoint_episode": None,
        "training_run_id": None,
        "training_total_episodes": None,
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
        collect_packet_outcomes=False,
        packet_outcome_artifact_mode=PACKET_OUTCOME_MODE_STREAMING,
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


def _write_routing_q_score_outputs(output_directory, method, result):
    """Write independent Q-score artifacts only for safe-DDQN evaluation."""

    if method.routing != "safe_ddqn":
        return {}
    diagnostics = result.get("routing_q_score_diagnostics")
    voluntary_waits = result.get("routing_q_score_voluntary_waits")
    if diagnostics is None or voluntary_waits is None:
        raise RuntimeError("safe-DDQN evaluation lacks routing Q-score diagnostics")
    return write_routing_q_score_diagnostic_artifacts(
        output_directory, diagnostics, voluntary_waits
    )


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
    checkpoint_episode=None,
    roi_counts=None,
    deadline_seconds=None,
    output_directory=None,
    fixed_roi_manifests=None,
    allow_registered_fixed_roi_method=False,
    flatten_single_point=False,
):
    validate_production_deadlines()
    suite = resolve_evaluation_suite(suite)
    definition = PAPER_EVALUATION_SUITES[suite]
    if deadline_seconds is not None and definition["kind"] != "deadline":
        raise ValueError(
            "deadline_seconds is available only for the "
            "task_type_delay_violation_vs_target_delay suite"
        )
    if deadline_seconds is not None:
        deadline_seconds = _validate_deadline_seconds(deadline_seconds)
        evaluation_sweep_points(
            suite,
            deadline_seconds=deadline_seconds,
            episode_seconds=int(
                FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"]
                if episode_seconds is None
                else episode_seconds
            ),
        )
    method = MethodSpec.parse(method_id)
    registry_fixed_roi = bool(
        suite == "fixed_roi"
        and allow_registered_fixed_roi_method
        and (method.learns_movement or method.learns_routing)
    )
    if method.method_id not in definition["methods"] and not registry_fixed_roi:
        raise ValueError(f"{method.method_id} is not part of {suite}: {definition['methods']}")

    selected_checkpoint_episode = resolve_checkpoint_episodes(
        checkpoint_episode=checkpoint_episode
    )[0]
    selected_roi_counts = (
        resolve_roi_counts(roi_counts=roi_counts)
        if suite == "fixed_roi"
        else None
    )
    if suite != "fixed_roi" and (
        roi_counts is not None or fixed_roi_manifests is not None
    ):
        raise ValueError("RoI selectors are available only for the fixed_roi suite")

    requested_manifest_seed = int(
        DEFAULT_TRAINING_SEED if manifest_seed is None else manifest_seed
    )
    checkpoint_required = bool(method.learns_movement or method.learns_routing)
    context = (
        _load_training_run(
            run_directory,
            method.method_id,
            selected_checkpoint_episode,
        )
        if checkpoint_required
        else _no_checkpoint_context(method, requested_manifest_seed, run_directory)
    )
    is_formal_checkpoint = bool(
        checkpoint_required
        and context["checkpoint_episode"] == FORMAL_CHECKPOINT_EPISODE
    )
    evaluation_purpose = (
        "formal_checkpoint_evaluation"
        if is_formal_checkpoint
        else (
            "diagnostic_checkpoint_progress_evaluation"
            if checkpoint_required
            else "random_policy_baseline_evaluation"
        )
    )
    git_sha = _git_sha()

    if definition["kind"] == "training_history":
        if output_directory is None:
            output_dir = _unique_directory(
                Path(output_root) / suite / method.method_id, git_sha
            )
        else:
            output_dir = Path(output_directory).resolve()
            output_dir.mkdir(parents=True, exist_ok=False)
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
            "checkpoint_episode": context["checkpoint_episode"],
            "checkpoint_path": str(context["checkpoint"]),
            "training_run_id": context["training_run_id"],
            "training_total_episodes": context["training_total_episodes"],
            "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
            "is_formal_checkpoint": bool(
                context["checkpoint_episode"] == FORMAL_CHECKPOINT_EPISODE
            ),
            "evaluation_purpose": (
                "formal_checkpoint_evaluation"
                if context["checkpoint_episode"] == FORMAL_CHECKPOINT_EPISODE
                else "diagnostic_checkpoint_progress_evaluation"
            ),
            "git_sha": git_sha,
            "training_history_only": True,
            "new_training_started": False,
            "collect_packet_outcomes": False,
            "packet_outcome_artifact_mode": PACKET_OUTCOME_MODE_DISABLED,
            "packet_outcome_artifact_schema_version": None,
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

    points = evaluation_sweep_points(
        suite,
        selected_roi_counts,
        deadline_seconds=deadline_seconds,
        episode_seconds=resolved_seconds,
    )
    if flatten_single_point and len(points) != 1:
        raise ValueError("flatten_single_point requires exactly one evaluation point")
    shared_manifests = dict(fixed_roi_manifests or {})
    if shared_manifests:
        expected_keys = {int(point["fixed_num_gt"]) for point in points}
        actual_keys = {int(key) for key in shared_manifests}
        if actual_keys != expected_keys:
            raise ValueError(
                "shared fixed-RoI manifest keys disagree with selected RoIs: "
                f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
            )
        shared_manifests = {
            int(key): validate_fixed_roi_manifest(
                manifest,
                int(key),
                resolved_episodes,
                requested_manifest_seed,
            )
            for key, manifest in shared_manifests.items()
        }

    resolved_point_manifests = []
    for point in points:
        fixed_num_gt = point.get("fixed_num_gt")
        manifest = (
            shared_manifests[int(fixed_num_gt)]
            if fixed_num_gt is not None and shared_manifests
            else _manifest_for_point(
                point,
                base_manifest=base_manifest,
                manifest_seed=requested_manifest_seed,
                episodes=resolved_episodes,
            )
        )
        if fixed_num_gt is not None:
            validate_fixed_roi_manifest(
                manifest,
                int(fixed_num_gt),
                resolved_episodes,
                requested_manifest_seed,
            )
        validate_manifest_initial_topologies(
            manifest, episode_count=resolved_episodes
        )
        resolved_point_manifests.append((point, manifest))

    if output_directory is None:
        output_dir = _unique_directory(
            Path(output_root) / suite / method.method_id, git_sha
        )
    else:
        output_dir = Path(output_directory).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

    point_results = []
    all_aggregates = []
    for point, manifest in resolved_point_manifests:
        fixed_num_gt = point.get("fixed_num_gt")
        point_dir = (
            output_dir
            if flatten_single_point
            else output_dir / point["point_id"]
        )
        if not flatten_single_point:
            point_dir.mkdir()
        manifest.save(point_dir / "scenario_manifest.json")
        packet_outcomes_path = point_dir / "packet_outcomes.jsonl"
        with PacketOutcomeJsonlWriter(packet_outcomes_path) as outcome_writer:
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
                expected_checkpoint_episodes=(
                    context["checkpoint_episode"] if checkpoint_required else None
                ),
                expected_checkpoint_formal_config=(
                    context["expected_training_config"]
                ),
                expected_checkpoint_training_manifest=context.get(
                    "training_manifest"
                ),
                evaluation_overrides=point.get("overrides"),
                trajectory_snapshot_times=point.get("snapshot_times_seconds"),
                trajectory_target_uav_id=(
                    int(target_uav_id)
                    if definition["kind"] == "trajectory"
                    else None
                ),
                packet_outcome_sink=outcome_writer.write_episode,
                collect_routing_q_score_diagnostics=(
                    method.routing == "safe_ddqn"
                ),
            )
        diagnostic_outputs = write_packet_routing_diagnostic_artifacts(
            point_dir,
            outcome_writer.routing_diagnostics(),
        )
        routing_q_score_outputs = _write_routing_q_score_outputs(
            point_dir, method, result
        )
        if fixed_num_gt is not None and any(
            int(row["num_GT"]) != int(fixed_num_gt)
            for row in result["episode_metrics"]
        ):
            raise RuntimeError(
                "evaluation episode actual RoI count differs from the selected value"
            )
        if outcome_writer.episode_count != resolved_episodes:
            raise RuntimeError(
                "paper packet outcome stream episode count mismatch: "
                f"written={outcome_writer.episode_count}, "
                f"expected={resolved_episodes}"
            )
        run_metadata = {
            **result["run_metadata"],
            "semantic_suite": suite,
            "paper_sweep_point": point,
            "git_sha": git_sha,
            "training_run_id": context["training_run_id"],
            "training_run_directory": (
                str(context["run_dir"]) if context["run_dir"] else None
            ),
            "training_total_episodes": context["training_total_episodes"],
            "checkpoint_required": checkpoint_required,
            "checkpoint_episode": context["checkpoint_episode"],
            "checkpoint_path": (
                str(context["checkpoint"]) if checkpoint_required else None
            ),
            "roi_count": (
                int(fixed_num_gt) if fixed_num_gt is not None else None
            ),
            "evaluation_episode_count": resolved_episodes,
            "episode_seconds": resolved_seconds,
            "manifest_seed": int(manifest.manifest_seed),
            "manifest_hash": manifest.content_hash,
            "scenario_ids": list(result["scenario_ids"]),
            "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
            "is_formal_checkpoint": is_formal_checkpoint,
            "evaluation_purpose": evaluation_purpose,
            **{
                field: context.get(field)
                for field in CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS
            },
            "scenario_manifest": str((point_dir / "scenario_manifest.json").resolve()),
        }
        outputs = write_evaluation_outputs(point_dir, result["episode_metrics"], run_metadata)
        outputs["packet_outcomes_jsonl"] = packet_outcomes_path.resolve()
        outputs.update(diagnostic_outputs)
        outputs.update(routing_q_score_outputs)
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
                "collect_packet_outcomes": False,
                "packet_outcome_artifact_mode": PACKET_OUTCOME_MODE_STREAMING,
                "packet_outcome_artifact_schema_version": (
                    PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION
                ),
                "packet_outcome_streamed_episode_count": (
                    outcome_writer.episode_count
                ),
                "packet_routing_diagnostic_contract_version": (
                    PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION
                ),
                "routing_q_score_diagnostics_enabled": (
                    method.routing == "safe_ddqn"
                ),
                "routing_q_score_diagnostic_contract_version": (
                    ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION
                    if method.routing == "safe_ddqn"
                    else None
                ),
                "resolved_overrides": result["run_metadata"].get(
                    "evaluation_overrides"
                ),
                "checkpoint_required": checkpoint_required,
                "training_run_id": context["training_run_id"],
                "training_run_directory": (
                    str(context["run_dir"]) if context["run_dir"] else None
                ),
                "training_total_episodes": context[
                    "training_total_episodes"
                ],
                "checkpoint_episode": context["checkpoint_episode"],
                "checkpoint_path": (
                    str(context["checkpoint"]) if checkpoint_required else None
                ),
                "roi_count": (
                    int(fixed_num_gt) if fixed_num_gt is not None else None
                ),
                "episode_seconds": resolved_seconds,
                "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
                "is_formal_checkpoint": is_formal_checkpoint,
                "evaluation_purpose": evaluation_purpose,
                **{
                    field: context.get(field)
                    for field in CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS
                },
                **{
                    field: result["run_metadata"].get(field)
                    for field in (
                        *CHECKPOINT_PROVENANCE_FIELDS,
                        *EVALUATION_PROVENANCE_FIELDS,
                    )
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
    method_contracts = comparison_method_configuration(method)
    metadata = {
        "aggregate_schema_version": PAPER_AGGREGATE_SCHEMA_VERSION,
        **{
            field: method_contracts[field]
            for field in (
                "packet_qos_contract_version",
                "fov_packet_generation_contract_version",
                "timely_useful_goodput_contract_version",
                "gs_gateway_contract_version",
                "permanent_gs_gateway_uav_id",
                "packet_routing_causality_contract_version",
                "routing_reward_contract_version",
                "qos_aggregate_contract_version",
            )
        },
        "semantic_suite": suite,
        "method_id": method.method_id,
        "method_spec": method.to_dict(),
        "training_run": str(context["run_dir"]) if context["run_dir"] else None,
        "training_run_id": context["training_run_id"],
        "training_run_directory": (
            str(context["run_dir"]) if context["run_dir"] else None
        ),
        "training_total_episodes": context["training_total_episodes"],
        "checkpoint_required": checkpoint_required,
        "checkpoint_path": str(context["checkpoint"]) if context["checkpoint"] else None,
        "checkpoint_episode": context["checkpoint_episode"],
        "formal_checkpoint_episode": FORMAL_CHECKPOINT_EPISODE,
        "is_formal_checkpoint": is_formal_checkpoint,
        "evaluation_purpose": evaluation_purpose,
        **{
            field: context.get(field)
            for field in CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS
        },
        **{
            field: point_results[0].get(field) if point_results else None
            for field in (
                *CHECKPOINT_PROVENANCE_FIELDS,
                *EVALUATION_PROVENANCE_FIELDS,
            )
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
        "collect_packet_outcomes": False,
        "packet_outcome_artifact_mode": PACKET_OUTCOME_MODE_STREAMING,
        "packet_outcome_artifact_schema_version": (
            PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION
        ),
        "packet_routing_diagnostic_contract_version": (
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION
        ),
        **PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS,
        "routing_q_score_diagnostics_enabled": (
            method.routing == "safe_ddqn"
        ),
        "routing_q_score_diagnostic_contract_version": (
            ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION
            if method.routing == "safe_ddqn"
            else None
        ),
        **{
            field: definition if method.routing == "safe_ddqn" else None
            for field, definition in ROUTING_Q_SCORE_DIAGNOSTIC_DEFINITIONS.items()
        },
        "target_uav_id": int(target_uav_id) if target_uav_id is not None else None,
        "git_sha": git_sha,
        "new_training_started": False,
        "aggregation": {
            "delay": "sum delivered E2E delay / sum delivered packets",
            "violation_probability": (
                "task diagnostics and ALL each pool raw canonical violations / "
                "raw eligible packets across episodes; ALL pools FOV+COM"
            ),
            "energy_efficiency": (
                "per-seed sum timely useful Mbit / sum mobility J; "
                "zero denominator is missing; valid seed values are equally weighted"
            ),
            "fov_coverage_snapshot_timing": "packet generation/capture time",
            "zero_delivered_delay": "null with missing=true",
        },
        "points": point_results,
    }
    _write_json(output_dir / "paper_evaluation_metadata.json", metadata)
    return {"output_directory": str(output_dir), **metadata}
