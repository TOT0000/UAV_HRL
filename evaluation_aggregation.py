"""Canonical ratio-of-sums evaluation aggregation for generic and paper paths."""

from __future__ import annotations

import math
from statistics import fmean, stdev

from scipy.stats import t as student_t


EVALUATION_AGGREGATION_SCHEMA_VERSION = (
    "canonical-useful-goodput-single-artifact-source-v3"
)

CANONICAL_METRICS = (
    ("energy_efficiency_mbit_per_j", None),
    ("average_e2e_delay_seconds", "FOV"),
    ("average_e2e_delay_seconds", "COM"),
    ("violation_probability", "FOV"),
    ("violation_probability", "COM"),
    ("violation_probability", "ALL"),
)


def _row_number(row, key):
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"canonical aggregation input is missing: {key}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"canonical aggregation input must be finite and non-negative: {key}"
        )
    return number


def _row_count(row, key):
    number = _row_number(row, key)
    if not number.is_integer():
        raise ValueError(f"canonical aggregation packet count is not integral: {key}")
    return int(number)


def _ratio_row(metric, task_type, numerator, denominator, numerator_unit, denominator_unit, value_unit):
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or numerator < 0.0:
        raise ValueError(f"{metric} numerator must be finite and non-negative")
    if not math.isfinite(denominator) or denominator < 0.0:
        raise ValueError(f"{metric} denominator must be finite and non-negative")
    missing = denominator == 0.0
    return {
        "aggregation_schema_version": EVALUATION_AGGREGATION_SCHEMA_VERSION,
        "aggregation_level": "training_seed",
        "aggregation_rule": "ratio_of_episode_sums",
        "metric": metric,
        "task_type": task_type,
        "numerator": numerator,
        "numerator_unit": numerator_unit,
        "denominator": denominator,
        "denominator_unit": denominator_unit,
        "value": None if missing else numerator / denominator,
        "value_unit": value_unit,
        "missing": missing,
    }


def aggregate_episode_rows_by_seed(episode_rows, *, seed_field="training_seed"):
    """Compute one ratio-of-sums value per trained-policy seed."""

    grouped = {}
    for row in episode_rows:
        seed = int(row.get(seed_field, 0))
        grouped.setdefault(seed, []).append(row)
    output = []
    for seed, rows in sorted(grouped.items()):
        timely = 0.0
        for row in rows:
            alias = _row_number(row, "timely_goodput_mbits")
            useful = (
                alias
                if row.get("total_timely_useful_mbits") in (None, "")
                else _row_number(row, "total_timely_useful_mbits")
            )
            if not math.isclose(useful, alias, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "timely_goodput_mbits must alias total_timely_useful_mbits"
                )
            timely += useful
        energy = sum(_row_number(row, "total_mobility_energy_j") for row in rows)
        seed_rows = [
            _ratio_row(
                "energy_efficiency_mbit_per_j", None,
                timely, energy, "Mbit", "J", "Mbit/J",
            )
        ]
        violation_sums = {}
        for task_type in ("FOV", "COM"):
            prefix = task_type.lower()
            delivered = sum(
                _row_count(row, f"{prefix}_delivered_packets")
                for row in rows
            )
            delay_sum = sum(
                _row_number(
                    row, f"{prefix}_delivered_e2e_delay_sum_seconds"
                )
                for row in rows
            )
            violations = sum(
                _row_count(row, f"{prefix}_violation_packets")
                for row in rows
            )
            eligible = sum(
                _row_count(row, f"{prefix}_eligible_packets")
                for row in rows
            )
            if violations > eligible:
                raise ValueError(f"{task_type} violations exceed eligible packets")
            violation_sums[task_type] = (violations, eligible)
            seed_rows.extend(
                (
                    _ratio_row(
                        "average_e2e_delay_seconds", task_type,
                        delay_sum, delivered, "seconds", "delivered_packets", "seconds",
                    ),
                    _ratio_row(
                        "violation_probability", task_type,
                        violations, eligible, "violation_packets", "eligible_packets", "probability",
                    ),
                )
            )
        seed_rows.append(
            _ratio_row(
                "violation_probability", "ALL",
                sum(value[0] for value in violation_sums.values()),
                sum(value[1] for value in violation_sums.values()),
                "violation_packets", "eligible_packets", "probability",
            )
        )
        for row in seed_rows:
            row["training_seed"] = seed
            row["evaluation_episode_count"] = len(rows)
        output.extend(seed_rows)
    return output


def aggregate_seed_rows(seed_rows):
    """Equal-weight valid seed values with sample SD and Student-t 95% CI."""

    grouped = {}
    for row in seed_rows:
        grouped.setdefault((row["metric"], row.get("task_type")), []).append(row)
    output = []
    for metric_task in CANONICAL_METRICS:
        rows = sorted(
            grouped.get(metric_task, []), key=lambda row: int(row["training_seed"])
        )
        valid = [row for row in rows if not row["missing"]]
        values = [float(row["value"]) for row in valid]
        count = len(values)
        mean = fmean(values) if values else None
        sample_sd = stdev(values) if count > 1 else (0.0 if count else None)
        df = max(count - 1, 0)
        critical = float(student_t.ppf(0.975, df=df)) if df else 0.0
        half_width = (
            critical * sample_sd / math.sqrt(count) if count else None
        )
        template = rows[0] if rows else {}
        output.append(
            {
                "aggregation_schema_version": EVALUATION_AGGREGATION_SCHEMA_VERSION,
                "aggregation_level": "cross_seed",
                "aggregation_rule": "equal_weight_valid_training_seed_values",
                "metric": metric_task[0],
                "task_type": metric_task[1],
                "numerator_unit": template.get("numerator_unit"),
                "denominator_unit": template.get("denominator_unit"),
                "value_unit": template.get("value_unit"),
                "valid_training_seed_count": count,
                "missing_training_seed_count": len(rows) - count,
                "training_seed_count": len(rows),
                "valid_training_seeds": [int(row["training_seed"]) for row in valid],
                "per_seed_numerators": [float(row["numerator"]) for row in rows],
                "per_seed_denominators": [float(row["denominator"]) for row in rows],
                "per_seed_values": [row["value"] for row in rows],
                "pooled_numerator": sum(float(row["numerator"]) for row in rows),
                "pooled_denominator": sum(float(row["denominator"]) for row in rows),
                "mean": mean,
                "sample_stddev": sample_sd,
                "degrees_of_freedom": df,
                "confidence_interval_method": "Student-t",
                "confidence_level": 0.95,
                "t_critical_975": critical,
                "ci95_half_width": half_width,
                "ci95_lower": mean - half_width if count else None,
                "ci95_upper": mean + half_width if count else None,
                "missing": count == 0,
            }
        )
    return output


def canonical_aggregation(episode_rows, *, seed_field="training_seed"):
    per_seed = aggregate_episode_rows_by_seed(
        episode_rows, seed_field=seed_field
    )
    return per_seed, aggregate_seed_rows(per_seed)
