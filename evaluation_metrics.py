"""Common evaluation metrics, artifact writers, and seed-level aggregation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import t as student_t

from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from com_capacity_calibration import load_com_capacity_reference
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    effective_training_config,
)
from experiment_paths import (
    evaluation_run_directory,
    evaluation_run_identity,
    prepare_run_directory,
    validate_run_directory_preflight,
    write_run_status,
)
from scenario_manifest import ScenarioManifest
from training_checkpoint import inspect_model_checkpoint
from evaluation_aggregation import (
    EVALUATION_AGGREGATION_SCHEMA_VERSION,
    aggregate_episode_rows_by_seed,
    aggregate_seed_rows,
)


IDENTITY_COLUMNS = (
    "method_id",
    "training_seed",
    "evaluation_split",
    "scenario_id",
    "evaluation_manifest_hash",
    "training_manifest_hash",
    "checkpoint_completed_episodes",
    "checkpoint_metadata_fingerprint",
)

METRIC_COLUMNS = (
    "num_GT",
    "timely_goodput_mbits",
    "raw_final_hop_mbits",
    "total_mobility_energy_j",
    "energy_efficiency_mbit_per_j",
    "fov_timely_delivered_packets",
    "com_timely_delivered_packets",
    "fov_deadline_violations",
    "com_deadline_violations",
    "total_deadline_violations",
    "eligible_packet_count",
    "delay_violation_probability",
    "sr_admission_drop_count",
    "coverage",
    "found_GT_ratio",
    "routing_wait_count",
    "partial_transmission_count",
    "slot_budget_violation_count",
    "unattributed_transition_violation_count",
)

OPTIONAL_METRIC_COLUMNS = {"delay_violation_probability"}

SEED_CANONICAL_METRIC_COLUMNS = (
    "fov_average_e2e_delay_seconds",
    "com_average_e2e_delay_seconds",
    "fov_violation_probability",
    "com_violation_probability",
    "all_violation_probability",
)
SEED_SUMMARY_METRIC_COLUMNS = (*METRIC_COLUMNS, *SEED_CANONICAL_METRIC_COLUMNS)

PACKET_METRIC_COLUMNS = tuple(
    f"{task}_{field}"
    for task in ("fov", "com")
    for field in (
        "generated_packets",
        "source_generated_packets",
        "eligible_packets",
        "sr_admission_drop_packets",
        "on_time_delivered_packets",
        "late_delivered_packets",
        "expired_dropped_packets",
        "delivered_packets",
        "delivered_e2e_delay_sum_seconds",
        "average_e2e_delay_seconds",
        "violation_packets",
        "violation_probability",
    )
) + (
    "fov_rate_packets_per_second",
    "com_rate_packets_per_second",
    "fov_deadline_seconds",
    "com_deadline_seconds",
)

CANONICAL_AGGREGATION_INPUT_COLUMNS = tuple(
    f"{task}_{field}"
    for task in ("fov", "com")
    for field in (
        "eligible_packets",
        "delivered_packets",
        "delivered_e2e_delay_sum_seconds",
        "violation_packets",
    )
)

EPISODE_COLUMNS = (*IDENTITY_COLUMNS, *METRIC_COLUMNS, *PACKET_METRIC_COLUMNS)

SEED_SUMMARY_IDENTITY_COLUMNS = (
    "method_id",
    "evaluation_split",
    "evaluation_manifest_hash",
    "training_manifest_hash",
    "training_seed",
    "checkpoint_completed_episodes",
    "checkpoint_metadata_fingerprint",
    "evaluation_episode_count",
)


def aggregation_metadata(seed_count):
    degrees_of_freedom = max(int(seed_count) - 1, 0)
    critical = (
        float(student_t.ppf(0.975, df=degrees_of_freedom))
        if degrees_of_freedom > 0
        else 0.0
    )
    return {
        "evaluation_unit": "episode",
        "uncertainty_unit": "trained-policy seed mean",
        "standard_deviation": "sample (n-1)",
        "confidence_interval": "Student-t",
        "confidence_level": 0.95,
        "degrees_of_freedom": degrees_of_freedom,
        "t_critical_975": critical,
        "pooled_episode_inference": False,
        "aggregation_schema_version": EVALUATION_AGGREGATION_SCHEMA_VERSION,
        "within_seed_rule": "ratio of episode numerator sums / denominator sums",
        "cross_seed_rule": "equal weight across valid trained-policy seed values",
        "zero_denominator": "missing",
    }


def safe_energy_efficiency(timely_goodput_mbits, mobility_energy_j):
    numerator = float(timely_goodput_mbits)
    denominator = float(mobility_energy_j)
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _optional_int(value):
    return None if value is None or value == "" else int(value)


def _optional_text(value):
    return None if value is None or value == "" else str(value)


def summarize_training_seeds(episode_rows):
    grouped = {}
    for row in episode_rows:
        key = (
            str(row["method_id"]),
            str(row["evaluation_split"]),
            str(row["evaluation_manifest_hash"]),
            _optional_text(row["training_manifest_hash"]),
            int(row["training_seed"]),
            _optional_int(row["checkpoint_completed_episodes"]),
            _optional_text(row["checkpoint_metadata_fingerprint"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, rows in sorted(grouped.items()):
        (
            method_id,
            evaluation_split,
            evaluation_manifest_hash,
            training_manifest_hash,
            training_seed,
            checkpoint_completed_episodes,
            checkpoint_metadata_fingerprint,
        ) = key
        summary = {
            "method_id": method_id,
            "evaluation_split": evaluation_split,
            "evaluation_manifest_hash": evaluation_manifest_hash,
            "training_manifest_hash": training_manifest_hash,
            "training_seed": training_seed,
            "checkpoint_completed_episodes": checkpoint_completed_episodes,
            "checkpoint_metadata_fingerprint": checkpoint_metadata_fingerprint,
            "evaluation_episode_count": len(rows),
        }
        for metric in METRIC_COLUMNS:
            values = [
                float(row[metric])
                for row in rows
                if row[metric] is not None
            ]
            summary[metric] = fmean(values) if values else None
        canonical_rows = aggregate_episode_rows_by_seed(rows)
        canonical = {
            (row["metric"], row.get("task_type")): row["value"]
            for row in canonical_rows
        }
        summary["energy_efficiency_mbit_per_j"] = canonical[
            ("energy_efficiency_mbit_per_j", None)
        ]
        summary["fov_average_e2e_delay_seconds"] = canonical[
            ("average_e2e_delay_seconds", "FOV")
        ]
        summary["com_average_e2e_delay_seconds"] = canonical[
            ("average_e2e_delay_seconds", "COM")
        ]
        summary["fov_violation_probability"] = canonical[
            ("violation_probability", "FOV")
        ]
        summary["com_violation_probability"] = canonical[
            ("violation_probability", "COM")
        ]
        summary["all_violation_probability"] = canonical[
            ("violation_probability", "ALL")
        ]
        summary["delay_violation_probability"] = summary[
            "all_violation_probability"
        ]
        summaries.append(summary)
    return summaries


def aggregate_seed_means(seed_summaries):
    """Aggregate seed means; episodes are never treated as independent seeds."""

    grouped = {}
    for row in seed_summaries:
        key = (
            str(row["method_id"]),
            str(row["evaluation_split"]),
            str(row["evaluation_manifest_hash"]),
            _optional_text(row["training_manifest_hash"]),
            _optional_int(row["checkpoint_completed_episodes"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for key, rows in sorted(grouped.items()):
        (
            method_id,
            evaluation_split,
            evaluation_manifest_hash,
            training_manifest_hash,
            checkpoint_completed_episodes,
        ) = key
        for metric in SEED_SUMMARY_METRIC_COLUMNS:
            values = [
                float(row[metric])
                for row in rows
                if row[metric] is not None
            ]
            mean_value = fmean(values) if values else None
            sample_stddev = (
                stdev(values) if len(values) > 1 else (0.0 if values else None)
            )
            degrees_of_freedom = max(len(values) - 1, 0)
            t_critical = (
                float(student_t.ppf(0.975, df=degrees_of_freedom))
                if degrees_of_freedom > 0
                else 0.0
            )
            ci95_half_width = (
                t_critical * sample_stddev / math.sqrt(len(values))
                if values
                else None
            )
            aggregates.append(
                {
                    "method_id": method_id,
                    "evaluation_split": evaluation_split,
                    "evaluation_manifest_hash": evaluation_manifest_hash,
                    "training_manifest_hash": training_manifest_hash,
                    "checkpoint_completed_episodes": (
                        checkpoint_completed_episodes
                    ),
                    "metric": metric,
                    "training_seed_count": len(values),
                    "degrees_of_freedom": degrees_of_freedom,
                    "t_critical_975": t_critical,
                    "mean": mean_value,
                    "sample_stddev": sample_stddev,
                    "ci95_half_width": ci95_half_width,
                    "ci95_lower": (
                        mean_value - ci95_half_width if values else None
                    ),
                    "ci95_upper": (
                        mean_value + ci95_half_width if values else None
                    ),
                }
            )
    return aggregates


def validate_formal_aggregation_rows(
    episode_rows,
    *,
    expected_method_id,
    expected_split,
    expected_seed_count=FORMAL_EXPERIMENT_DEFAULTS["training_seed_count"],
    expected_episodes_per_seed=FORMAL_EXPERIMENT_DEFAULTS[
        "evaluation_episodes_per_trained_seed"
    ],
    expected_completed_episodes=FORMAL_EXPERIMENT_DEFAULTS[
        "formal_checkpoint_episode"
    ],
):
    if int(expected_seed_count) <= 0 or int(expected_episodes_per_seed) <= 0:
        raise ValueError("expected seed and episode counts must be positive")
    missing = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            *METRIC_COLUMNS,
            *CANONICAL_AGGREGATION_INPUT_COLUMNS,
        )
        if any(column not in row for row in episode_rows)
    ]
    if missing:
        raise ValueError(f"evaluation rows are missing columns: {sorted(missing)}")
    required_text_columns = (
        "method_id",
        "evaluation_split",
        "scenario_id",
        "evaluation_manifest_hash",
        "training_manifest_hash",
        "checkpoint_metadata_fingerprint",
    )
    for column in required_text_columns:
        if any(
            row[column] is None or not str(row[column]).strip()
            for row in episode_rows
        ):
            raise ValueError(f"identity column {column} must be non-empty")

    methods = {str(row["method_id"]) for row in episode_rows}
    if methods != {str(expected_method_id)}:
        raise ValueError(f"aggregation requires one expected method_id: {methods}")
    splits = {str(row["evaluation_split"]) for row in episode_rows}
    if splits != {str(expected_split)} or str(expected_split) not in {
        "validation",
        "test",
    }:
        raise ValueError(f"aggregation evaluation split mismatch: {splits}")

    evaluation_hashes = {
        str(row["evaluation_manifest_hash"]) for row in episode_rows
    }
    if len(evaluation_hashes) != 1 or "" in evaluation_hashes:
        raise ValueError(
            "aggregation requires exactly one evaluation manifest hash"
        )
    training_hashes = {
        str(row["training_manifest_hash"]) for row in episode_rows
    }
    if len(training_hashes) != 1 or "" in training_hashes:
        raise ValueError("aggregation requires exactly one training manifest hash")

    for row in episode_rows:
        for metric in METRIC_COLUMNS:
            if row[metric] is None and metric in OPTIONAL_METRIC_COLUMNS:
                continue
            try:
                value = float(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metric {metric} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"metric {metric} must be finite")

    identities = [
        (
            str(row["method_id"]),
            int(row["training_seed"]),
            str(row["evaluation_manifest_hash"]),
            str(row["scenario_id"]),
        )
        for row in episode_rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate evaluation row or seed rerun detected")

    by_seed = {}
    for row in episode_rows:
        seed = int(row["training_seed"])
        by_seed.setdefault(seed, []).append(row)
    if len(by_seed) != int(expected_seed_count):
        raise ValueError(
            "training seed count mismatch: "
            f"found={len(by_seed)}, expected={int(expected_seed_count)}"
        )

    reference_scenarios = None
    for seed, rows in sorted(by_seed.items()):
        if len(rows) != int(expected_episodes_per_seed):
            raise ValueError(
                f"evaluation episode count mismatch for seed {seed}: "
                f"found={len(rows)}, expected={int(expected_episodes_per_seed)}"
            )
        scenarios = {str(row["scenario_id"]) for row in rows}
        if len(scenarios) != len(rows):
            raise ValueError(f"scenario IDs are not unique for seed {seed}")
        if reference_scenarios is None:
            reference_scenarios = scenarios
        elif scenarios != reference_scenarios:
            raise ValueError("training seeds do not share the same scenario ID set")

        completed = {int(row["checkpoint_completed_episodes"]) for row in rows}
        if completed != {int(expected_completed_episodes)}:
            raise ValueError(
                f"checkpoint completed episodes mismatch for seed {seed}: {completed}"
            )
        checkpoint_fingerprints = {
            str(row["checkpoint_metadata_fingerprint"]) for row in rows
        }
        if len(checkpoint_fingerprints) != 1 or "" in checkpoint_fingerprints:
            raise ValueError(
                f"seed {seed} must use exactly one checkpoint fingerprint"
            )
    return episode_rows


def write_evaluation_outputs(output_dir, episode_rows, run_metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {column: row.get(column) for column in EPISODE_COLUMNS}
        for row in episode_rows
    ]
    seed_summaries = summarize_training_seeds(normalized_rows)
    seed_fields = (*SEED_SUMMARY_IDENTITY_COLUMNS, *SEED_SUMMARY_METRIC_COLUMNS)

    per_episode_csv = _write_csv(
        output_dir / "per_episode.csv", normalized_rows, EPISODE_COLUMNS
    )
    per_episode_jsonl = output_dir / "per_episode.jsonl"
    per_episode_jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in normalized_rows
        ),
        encoding="utf-8",
    )
    seed_csv = _write_csv(
        output_dir / "per_training_seed_summary.csv", seed_summaries, seed_fields
    )
    seed_json = output_dir / "per_training_seed_summary.json"
    seed_json.write_text(
        json.dumps(seed_summaries, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    canonical_seed_rows = aggregate_episode_rows_by_seed(normalized_rows)
    canonical_cross_seed_rows = aggregate_seed_rows(canonical_seed_rows)
    canonical_seed_json = output_dir / "canonical_per_seed_aggregation.json"
    canonical_seed_json.write_text(
        json.dumps(canonical_seed_rows, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    canonical_cross_seed_json = output_dir / "canonical_cross_seed_aggregation.json"
    canonical_cross_seed_json.write_text(
        json.dumps(
            canonical_cross_seed_rows,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "per_episode_csv": per_episode_csv,
        "per_episode_jsonl": per_episode_jsonl,
        "per_training_seed_csv": seed_csv,
        "per_training_seed_json": seed_json,
        "canonical_per_seed_aggregation_json": canonical_seed_json,
        "canonical_cross_seed_aggregation_json": canonical_cross_seed_json,
        "run_metadata": metadata_path,
    }


def _read_episode_csvs(input_dir):
    rows = []
    for path in sorted(Path(input_dir).glob("**/per_episode.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"no per_episode.csv files found under {Path(input_dir)}"
        )
    return rows


def run_evaluation_command(args):
    preflight = _evaluation_preflight(args)
    run_dir = prepare_run_directory(
        preflight["run_directory"], preflight["identity"]
    )
    try:
        write_run_status(run_dir, "RUNNING")
        result = preflight["train"](
            preflight["config"],
            scenario_manifest=preflight["manifest"],
            method_spec=preflight["method"],
            evaluation=True,
            checkpoint_dir=args.checkpoint,
            expected_checkpoint_episodes=FORMAL_EXPERIMENT_DEFAULTS[
                "training_episodes_per_seed"
            ],
            expected_checkpoint_formal_config=preflight[
                "expected_training_config"
            ],
        )
        metadata = {
            **result["run_metadata"],
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "evaluation_invariants": result["evaluation_invariants"],
            "run_directory": str(run_dir),
            "run_identity": preflight["identity"],
            "run_status_file": str(run_dir / "run_status.json"),
        }
        paths = write_evaluation_outputs(
            run_dir, result["episode_metrics"], metadata
        )
        write_run_status(run_dir, "COMPLETED")
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()},
                ensure_ascii=False,
            )
        )
    except BaseException as exc:
        try:
            write_run_status(run_dir, "FAILED", exception=exc)
        except BaseException:
            pass
        raise
    return 0


def _evaluation_preflight(args):
    """Validate evaluation inputs and checkpoint metadata without weights."""

    from HRL_task_aware import (
        ROUTING_STATE_DIM,
        TrainingConfig,
        formal_training_config,
        train,
    )

    if args.resume is not None:
        raise ValueError("evaluation accepts --checkpoint, not --resume")
    method = args.method
    MethodSpec(**{
        key: value
        for key, value in method.to_dict().items()
        if key != "method_id"
    })
    manifest = ScenarioManifest.load(args.manifest)
    if manifest.split != args.split:
        raise ValueError(
            f"manifest split mismatch: manifest={manifest.split}, requested={args.split}"
        )
    episode_count = (
        int(args.episodes)
        if args.episodes is not None
        else FORMAL_EXPERIMENT_DEFAULTS["evaluation_episodes_per_trained_seed"]
    )
    if manifest.episode_count < episode_count:
        raise ValueError(
            "scenario manifest has fewer entries than requested episodes"
        )
    identity = evaluation_run_identity(method, manifest, args.training_seed)
    run_dir = evaluation_run_directory(
        args.output_dir, method, manifest, args.training_seed
    )
    config = TrainingConfig(
        total_episodes=episode_count,
        mode="custom",
        episode_seconds=FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"],
        routing_slot_seconds=FORMAL_EXPERIMENT_DEFAULTS["routing_slot_seconds"],
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=args.training_seed,
    )
    expected_training_config = effective_training_config(
        formal_training_config(
            FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"],
            random_seed=args.training_seed,
        ),
        method,
    )
    _, calibration = load_com_capacity_reference()
    inspect_model_checkpoint(
        args.checkpoint,
        movement_state_dim=MOVEMENT_STATE_DIM,
        joint_action_dim=JOINT_ACTION_DIM,
        routing_state_dim=ROUTING_STATE_DIM,
        td3_gamma=1.0,
        ddqn_gamma=0.99,
        calibration=calibration,
        expected_experiment_metadata={
            "method_spec_fingerprint": method.compatible_fingerprints,
            "training_seed": int(args.training_seed),
        },
        expected_completed_episodes=FORMAL_EXPERIMENT_DEFAULTS[
            "training_episodes_per_seed"
        ],
        expected_formal_config=expected_training_config,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
    )
    validate_run_directory_preflight(run_dir, identity)
    return {
        "method": method,
        "manifest": manifest,
        "identity": identity,
        "run_directory": Path(run_dir).resolve(),
        "config": config,
        "expected_training_config": expected_training_config,
        "train": train,
    }


def run_aggregate_command(args):
    episode_rows = _read_episode_csvs(args.input_dir)
    method_id = args.method.method_id
    validate_formal_aggregation_rows(
        episode_rows,
        expected_method_id=method_id,
        expected_split=args.split,
        expected_seed_count=args.expected_seed_count,
        expected_episodes_per_seed=args.expected_episodes_per_seed,
        expected_completed_episodes=FORMAL_EXPERIMENT_DEFAULTS[
            "training_episodes_per_seed"
        ],
    )
    if args.manifest is not None:
        manifest = ScenarioManifest.load(args.manifest)
        evaluation_hashes = {
            str(row["evaluation_manifest_hash"]) for row in episode_rows
        }
        if manifest.split != args.split or evaluation_hashes != {
            manifest.content_hash
        }:
            raise ValueError("aggregate manifest does not match evaluation rows")
    seed_summaries = summarize_training_seeds(episode_rows)
    aggregate_rows = aggregate_seed_means(seed_summaries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_fields = (*SEED_SUMMARY_IDENTITY_COLUMNS, *SEED_SUMMARY_METRIC_COLUMNS)
    aggregate_fields = tuple(aggregate_rows[0].keys())
    _write_csv(output_dir / "per_training_seed_summary.csv", seed_summaries, seed_fields)
    _write_csv(output_dir / "cross_seed_summary.csv", aggregate_rows, aggregate_fields)
    canonical_seed_rows = aggregate_episode_rows_by_seed(episode_rows)
    canonical_cross_seed_rows = aggregate_seed_rows(canonical_seed_rows)
    (output_dir / "canonical_per_seed_aggregation.json").write_text(
        json.dumps(canonical_seed_rows, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "canonical_cross_seed_aggregation.json").write_text(
        json.dumps(
            canonical_cross_seed_rows,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "cross_seed_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregation_metadata.json").write_text(
        json.dumps(
            aggregation_metadata(args.expected_seed_count),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0
