"""Common evaluation metrics, artifact writers, and seed-level aggregation."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from experiment_config import FORMAL_EXPERIMENT_DEFAULTS
from scenario_manifest import ScenarioManifest


EPISODE_COLUMNS = (
    "method_id",
    "training_seed",
    "scenario_id",
    "manifest_hash",
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
    "coverage",
    "found_GT_ratio",
    "routing_wait_count",
    "partial_transmission_count",
    "slot_budget_violation_count",
)

METRIC_COLUMNS = EPISODE_COLUMNS[4:]

AGGREGATION_METADATA = {
    "evaluation_unit": "episode",
    "uncertainty_unit": "trained-policy seed mean",
    "standard_deviation": "sample (n-1)",
    "confidence_interval": "mean +/- 1.96 * sample_stddev / sqrt(seed_count)",
    "pooled_episode_inference": False,
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


def summarize_training_seeds(episode_rows):
    grouped = {}
    for row in episode_rows:
        key = (
            str(row["method_id"]),
            str(row["manifest_hash"]),
            int(row["training_seed"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (method_id, manifest_hash, training_seed), rows in sorted(grouped.items()):
        summary = {
            "method_id": method_id,
            "manifest_hash": manifest_hash,
            "training_seed": training_seed,
            "evaluation_episode_count": len(rows),
        }
        for metric in METRIC_COLUMNS:
            summary[metric] = fmean(float(row[metric]) for row in rows)
        summaries.append(summary)
    return summaries


def aggregate_seed_means(seed_summaries):
    """Aggregate seed means; episodes are never treated as independent seeds."""

    grouped = {}
    for row in seed_summaries:
        key = (str(row["method_id"]), str(row["manifest_hash"]))
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (method_id, manifest_hash), rows in sorted(grouped.items()):
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in rows]
            mean_value = fmean(values)
            sample_stddev = stdev(values) if len(values) > 1 else 0.0
            ci95_half_width = 1.96 * sample_stddev / math.sqrt(len(values))
            aggregates.append(
                {
                    "method_id": method_id,
                    "manifest_hash": manifest_hash,
                    "metric": metric,
                    "training_seed_count": len(values),
                    "mean": mean_value,
                    "sample_stddev": sample_stddev,
                    "ci95_half_width": ci95_half_width,
                    "ci95_lower": mean_value - ci95_half_width,
                    "ci95_upper": mean_value + ci95_half_width,
                }
            )
    return aggregates


def write_evaluation_outputs(output_dir, episode_rows, run_metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {column: row[column] for column in EPISODE_COLUMNS}
        for row in episode_rows
    ]
    seed_summaries = summarize_training_seeds(normalized_rows)
    seed_fields = (
        "method_id",
        "manifest_hash",
        "training_seed",
        "evaluation_episode_count",
        *METRIC_COLUMNS,
    )

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
    from HRL_task_aware import TrainingConfig, formal_training_config, train

    if args.resume is not None:
        raise ValueError("evaluation accepts --checkpoint, not --resume")
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
    expected_training_config = asdict(
        formal_training_config(
            FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"],
            random_seed=args.training_seed,
        )
    )
    result = train(
        config,
        scenario_manifest=manifest,
        method_spec=args.method,
        evaluation=True,
        checkpoint_dir=args.checkpoint,
        expected_checkpoint_episodes=FORMAL_EXPERIMENT_DEFAULTS[
            "training_episodes_per_seed"
        ],
        expected_checkpoint_formal_config=expected_training_config,
    )
    metadata = {
        **result["run_metadata"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "evaluation_invariants": result["evaluation_invariants"],
    }
    paths = write_evaluation_outputs(
        args.output_dir, result["episode_metrics"], metadata
    )
    print(
        json.dumps(
            {name: str(path) for name, path in paths.items()}, ensure_ascii=False
        )
    )
    return 0


def run_aggregate_command(args):
    episode_rows = _read_episode_csvs(args.input_dir)
    method_id = args.method.method_id
    episode_rows = [row for row in episode_rows if row["method_id"] == method_id]
    if not episode_rows:
        raise ValueError(f"no evaluation rows found for method {method_id}")
    seed_summaries = summarize_training_seeds(episode_rows)
    aggregate_rows = aggregate_seed_means(seed_summaries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_fields = (
        "method_id",
        "manifest_hash",
        "training_seed",
        "evaluation_episode_count",
        *METRIC_COLUMNS,
    )
    aggregate_fields = tuple(aggregate_rows[0].keys())
    _write_csv(output_dir / "per_training_seed_summary.csv", seed_summaries, seed_fields)
    _write_csv(output_dir / "cross_seed_summary.csv", aggregate_rows, aggregate_fields)
    (output_dir / "cross_seed_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregation_metadata.json").write_text(
        json.dumps(
            AGGREGATION_METADATA,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0
