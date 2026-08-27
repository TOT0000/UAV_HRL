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
from scenario_manifest import (
    ScenarioManifest,
    validate_manifest_initial_topologies,
)
from training_checkpoint import inspect_model_checkpoint
from evaluation_aggregation import (
    CANONICAL_METRICS,
    EVALUATION_AGGREGATION_SCHEMA_VERSION,
    canonical_aggregation,
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
    "unattributed_pre_routing_violation_count",
    "system_qos_violation_count",
    "system_qos_eligible_packets",
    "routing_cost_sum",
    "routing_cost_eligible_packets",
    "routing_credit_violation_count",
    "replay_attributed_violation_cost_count",
)

OPTIONAL_METRIC_COLUMNS = {"delay_violation_probability"}

GENERIC_CROSS_SEED_ARTIFACT_SCHEMA_VERSION = (
    "uav-hrl-generic-cross-seed-aggregate-v2"
)

# Every per-episode metric must opt in to exactly one generic aggregation path.
# The canonical and alias entries are never handled by the descriptive helper.
DESCRIPTIVE_EPISODE_METRIC_COLUMNS = (
    "num_GT",
    "timely_goodput_mbits",
    "raw_final_hop_mbits",
    "total_mobility_energy_j",
    "fov_timely_delivered_packets",
    "com_timely_delivered_packets",
    "fov_deadline_violations",
    "com_deadline_violations",
    "total_deadline_violations",
    "eligible_packet_count",
    "sr_admission_drop_count",
    "coverage",
    "found_GT_ratio",
    "routing_wait_count",
    "partial_transmission_count",
    "slot_budget_violation_count",
    "unattributed_transition_violation_count",
    "unattributed_pre_routing_violation_count",
    "system_qos_violation_count",
    "system_qos_eligible_packets",
    "routing_cost_sum",
    "routing_cost_eligible_packets",
    "routing_credit_violation_count",
    "replay_attributed_violation_cost_count",
)

METRIC_AGGREGATION_REGISTRY = {
    **{
        metric: {"aggregation_kind": "descriptive_seed_mean"}
        for metric in DESCRIPTIVE_EPISODE_METRIC_COLUMNS
    },
    "energy_efficiency_mbit_per_j": {
        "aggregation_kind": "canonical_ratio",
        "canonical_metric": "energy_efficiency_mbit_per_j",
        "canonical_task_type": None,
    },
    "delay_violation_probability": {
        "aggregation_kind": "canonical_alias",
        "alias_of_metric": "violation_probability",
        "alias_of_task_type": "ALL",
    },
}

if set(METRIC_AGGREGATION_REGISTRY) != set(METRIC_COLUMNS):
    raise RuntimeError(
        "every METRIC_COLUMNS entry must declare its generic aggregation kind"
    )

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
    return {
        "generic_cross_seed_artifact_schema_version": (
            GENERIC_CROSS_SEED_ARTIFACT_SCHEMA_VERSION
        ),
        "evaluation_unit": "episode",
        "uncertainty_unit": "trained-policy seed mean",
        "configured_training_seed_count": int(seed_count),
        "standard_deviation": "sample (n-1)",
        "confidence_interval": "Student-t",
        "confidence_level": 0.95,
        "result_specific_degrees_of_freedom": "serialized in every result row",
        "pooled_episode_inference": False,
        "aggregation_schema_version": EVALUATION_AGGREGATION_SCHEMA_VERSION,
        "within_seed_rule": "ratio of episode numerator sums / denominator sums",
        "canonical_within_seed_rule": (
            "ratio of episode numerator sums / denominator sums"
        ),
        "descriptive_within_seed_rule": "episode arithmetic mean",
        "cross_seed_rule": "equal weight across valid trained-policy seed values",
        "zero_denominator": "missing",
        "aggregation_kinds": {
            "canonical_ratio": "six metrics from canonical_aggregation()",
            "canonical_alias": (
                "delay_violation_probability copies canonical ALL violation"
            ),
            "descriptive_seed_mean": (
                "episode mean within seed, then equal-weight seed inference"
            ),
        },
        "metric_aggregation_registry": METRIC_AGGREGATION_REGISTRY,
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
                if row[metric] is not None and row[metric] != ""
            ]
            summary[metric] = fmean(values) if values else None
        canonical_rows, _canonical_cross_seed_rows = canonical_aggregation(
            rows
        )
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


def _cross_seed_identity(rows, *, source_name):
    rows = list(rows)
    if not rows:
        raise ValueError(f"{source_name} requires input rows")

    def unique(field, transform=lambda value: value):
        values = {transform(row.get(field)) for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"{source_name} rows disagree on identity field: {field}"
            )
        return next(iter(values))

    checkpoint_by_seed = {}
    for row in rows:
        seed = int(row["training_seed"])
        fingerprint = _optional_text(row.get("checkpoint_metadata_fingerprint"))
        existing = checkpoint_by_seed.setdefault(seed, fingerprint)
        if existing != fingerprint:
            raise ValueError(
                f"{source_name} rows disagree on checkpoint identity for seed {seed}"
            )

    return {
        "generic_cross_seed_artifact_schema_version": (
            GENERIC_CROSS_SEED_ARTIFACT_SCHEMA_VERSION
        ),
        "method_id": unique("method_id", str),
        "evaluation_split": unique("evaluation_split", str),
        "evaluation_manifest_hash": unique("evaluation_manifest_hash", str),
        "training_manifest_hash": unique(
            "training_manifest_hash", _optional_text
        ),
        "checkpoint_completed_episodes": unique(
            "checkpoint_completed_episodes", _optional_int
        ),
        "training_seeds": sorted(checkpoint_by_seed),
        "checkpoint_identities": [
            {
                "training_seed": seed,
                "checkpoint_metadata_fingerprint": checkpoint_by_seed[seed],
            }
            for seed in sorted(checkpoint_by_seed)
        ],
    }


def canonical_cross_seed_artifact_rows(canonical_rows, episode_rows):
    """Attach generic identity fields without recomputing canonical statistics."""

    rows = list(canonical_rows)
    identities = [(row["metric"], row.get("task_type")) for row in rows]
    if len(rows) != len(CANONICAL_METRICS) or set(identities) != set(
        CANONICAL_METRICS
    ):
        raise ValueError("generic canonical artifact requires exactly six metrics")
    identity = _cross_seed_identity(
        episode_rows, source_name="canonical artifact serialization"
    )
    return [
        {
            **identity,
            "aggregation_kind": "canonical_ratio",
            **dict(row),
        }
        for row in rows
    ]


def aggregate_descriptive_seed_metrics(seed_summaries):
    """Aggregate only descriptive metrics from episode means within each seed."""

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
    for _key, rows in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        rows = sorted(rows, key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("descriptive aggregation requires one row per seed")
        identity = _cross_seed_identity(
            rows, source_name="descriptive aggregation"
        )
        for metric in DESCRIPTIVE_EPISODE_METRIC_COLUMNS:
            per_seed_values = []
            for row in rows:
                raw_value = row.get(metric)
                if raw_value is None or raw_value == "":
                    per_seed_values.append(None)
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"descriptive metric {metric} must be finite when present"
                    )
                per_seed_values.append(value)
            valid_pairs = [
                (seed, value)
                for seed, value in zip(seeds, per_seed_values)
                if value is not None
            ]
            values = [value for _seed, value in valid_pairs]
            count = len(values)
            mean = fmean(values) if values else None
            sample_sd = stdev(values) if count > 1 else (0.0 if count else None)
            degrees_of_freedom = max(count - 1, 0)
            critical = (
                float(student_t.ppf(0.975, df=degrees_of_freedom))
                if degrees_of_freedom
                else 0.0
            )
            half_width = (
                critical * sample_sd / math.sqrt(count) if count else None
            )
            aggregates.append(
                {
                    **identity,
                    "aggregation_kind": "descriptive_seed_mean",
                    "aggregation_level": "cross_seed",
                    "aggregation_rule": (
                        "equal_weight_valid_seed_episode_means"
                    ),
                    "metric": metric,
                    "task_type": None,
                    "valid_training_seed_count": count,
                    "missing_training_seed_count": len(rows) - count,
                    "training_seed_count": len(rows),
                    "valid_training_seeds": [seed for seed, _value in valid_pairs],
                    "missing_training_seeds": [
                        seed
                        for seed, value in zip(seeds, per_seed_values)
                        if value is None
                    ],
                    "per_seed_values": per_seed_values,
                    "per_seed_episode_counts": [
                        int(row["evaluation_episode_count"]) for row in rows
                    ],
                    "mean": mean,
                    "sample_stddev": sample_sd,
                    "degrees_of_freedom": degrees_of_freedom,
                    "confidence_interval_method": "Student-t",
                    "confidence_level": 0.95,
                    "t_critical_975": critical,
                    "ci95_half_width": half_width,
                    "ci95_lower": mean - half_width if count else None,
                    "ci95_upper": mean + half_width if count else None,
                    "missing": count == 0,
                }
            )
    return aggregates


def build_generic_cross_seed_artifact(
    canonical_cross_seed_rows,
    descriptive_cross_seed_rows,
    episode_rows,
):
    """Combine canonical serialization, one canonical alias, and diagnostics."""

    canonical_rows = canonical_cross_seed_artifact_rows(
        canonical_cross_seed_rows, episode_rows
    )
    all_violation = next(
        row
        for row in canonical_rows
        if (row["metric"], row.get("task_type"))
        == ("violation_probability", "ALL")
    )
    alias_row = {
        **all_violation,
        "aggregation_kind": "canonical_alias",
        "metric": "delay_violation_probability",
        "task_type": None,
        "alias_of_metric": "violation_probability",
        "alias_of_task_type": "ALL",
    }
    descriptive_rows = [dict(row) for row in descriptive_cross_seed_rows]
    if (
        len(descriptive_rows) != len(DESCRIPTIVE_EPISODE_METRIC_COLUMNS)
        or {row.get("metric") for row in descriptive_rows}
        != set(DESCRIPTIVE_EPISODE_METRIC_COLUMNS)
        or any(
            row.get("aggregation_kind") != "descriptive_seed_mean"
            for row in descriptive_rows
        )
    ):
        raise ValueError(
            "generic artifact requires exactly one row for every descriptive metric"
        )
    identity_fields = (
        "generic_cross_seed_artifact_schema_version",
        "method_id",
        "evaluation_split",
        "evaluation_manifest_hash",
        "training_manifest_hash",
        "checkpoint_completed_episodes",
        "training_seeds",
        "checkpoint_identities",
    )
    if any(
        any(
            row.get(field) != canonical_rows[0].get(field)
            for field in identity_fields
        )
        for row in descriptive_rows
    ):
        raise ValueError("canonical and descriptive artifact identities disagree")
    return [*canonical_rows, alias_row, *descriptive_rows]


def stable_cross_seed_csv_fields(artifact_rows):
    """Return a deterministic union schema without performing aggregation."""

    fields = []
    seen = set()
    for row in artifact_rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return tuple(fields)


def canonical_cross_seed_csv_rows(artifact_rows, fieldnames=None):
    """Project the stable union schema and JSON containers; calculate nothing."""

    fieldnames = tuple(fieldnames or stable_cross_seed_csv_fields(artifact_rows))
    return [
        {
            field: (
                json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (list, dict))
                else value
            )
            for field in fieldnames
            for value in (row.get(field),)
        }
        for row in artifact_rows
    ]


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
            if row[metric] is None or row[metric] == "":
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
    canonical_seed_rows, canonical_cross_seed_rows = canonical_aggregation(
        normalized_rows
    )
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
    validate_manifest_initial_topologies(
        manifest, episode_count=episode_count
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
    canonical_seed_rows, canonical_cross_seed_rows = canonical_aggregation(
        episode_rows
    )
    descriptive_cross_seed_rows = aggregate_descriptive_seed_metrics(
        seed_summaries
    )
    aggregate_rows = build_generic_cross_seed_artifact(
        canonical_cross_seed_rows,
        descriptive_cross_seed_rows,
        episode_rows,
    )
    aggregate_fields = stable_cross_seed_csv_fields(aggregate_rows)
    aggregate_csv_rows = canonical_cross_seed_csv_rows(
        aggregate_rows, aggregate_fields
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_fields = (*SEED_SUMMARY_IDENTITY_COLUMNS, *SEED_SUMMARY_METRIC_COLUMNS)
    _write_csv(output_dir / "per_training_seed_summary.csv", seed_summaries, seed_fields)
    _write_csv(
        output_dir / "cross_seed_summary.csv",
        aggregate_csv_rows,
        aggregate_fields,
    )
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
