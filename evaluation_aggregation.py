"""Canonical ratio-of-sums evaluation aggregation for generic and paper paths."""

from __future__ import annotations

import math
from statistics import fmean, stdev

from scipy.stats import t as student_t


EVALUATION_AGGREGATION_SCHEMA_VERSION = (
    "canonical-complete-useful-goodput-service-only-v5"
)

LEGACY_PACKET_DEADLINE_POLICY = "drop_at_target_delay_v1"
CONTINUED_PACKET_DEADLINE_POLICY = "continue_until_episode_end_v1"
TIMELY_EE_DEFINITION = "timely_useful_bits_per_mobility_energy_v1"
COMPLETE_EE_DEFINITION = "complete_useful_bits_per_mobility_energy_v1"
EVALUATION_SEMANTIC_FIELDS = (
    "packet_deadline_policy",
    "primary_energy_efficiency_definition_id",
)
_COMPLETE_USEFUL_FIELDS = (
    "total_late_useful_mbits",
    "total_all_useful_mbits",
    "total_late_useful_bits",
    "total_all_useful_bits",
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


def _source_name(row):
    return str(row.get("_source_path") or "<in-memory episode row>")


def _inferred_semantics(row):
    policy = row.get("packet_deadline_policy")
    ee_definition = row.get("primary_energy_efficiency_definition_id")
    present = [row.get(field) not in (None, "") for field in _COMPLETE_USEFUL_FIELDS]
    if policy in (None, "") and ee_definition in (None, ""):
        if all(present):
            return CONTINUED_PACKET_DEADLINE_POLICY, COMPLETE_EE_DEFINITION
        if not any(present):
            return LEGACY_PACKET_DEADLINE_POLICY, TIMELY_EE_DEFINITION
        raise ValueError(
            f"{_source_name(row)}: evaluation packet policy cannot be identified; "
            "complete-useful fields are only partially present"
        )
    if policy in (None, "") or ee_definition in (None, ""):
        raise ValueError(
            f"{_source_name(row)}: evaluation packet policy and primary EE "
            "definition must both be identified"
        )
    return str(policy), str(ee_definition)


def _validate_complete_useful_conservation(row):
    source = _source_name(row)
    missing = [field for field in _COMPLETE_USEFUL_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"{source}: continued-delivery evaluation row is missing fields: {missing}"
        )
    for suffix, abs_tol in (("mbits", 1e-12), ("bits", 1e-6)):
        timely = _row_number(row, f"total_timely_useful_{suffix}")
        late = _row_number(row, f"total_late_useful_{suffix}")
        complete = _row_number(row, f"total_all_useful_{suffix}")
        if not math.isclose(
            complete, timely + late, rel_tol=1e-9, abs_tol=abs_tol
        ):
            raise ValueError(
                f"{source}: complete useful {suffix} is not conserved: "
                f"all={complete}, timely={timely}, late={late}"
            )


def resolve_episode_evaluation_semantics(episode_rows):
    """Validate and return the one packet-policy/EE identity in a batch."""

    rows = list(episode_rows)
    if not rows:
        raise ValueError("evaluation aggregation requires at least one episode row")
    identities = {}
    for row in rows:
        policy, ee_definition = _inferred_semantics(row)
        expected = {
            LEGACY_PACKET_DEADLINE_POLICY: TIMELY_EE_DEFINITION,
            CONTINUED_PACKET_DEADLINE_POLICY: COMPLETE_EE_DEFINITION,
        }.get(policy)
        if expected is None or ee_definition != expected:
            raise ValueError(
                f"{_source_name(row)}: unsupported or inconsistent evaluation "
                f"semantics: packet_policy={policy!r}, ee_definition={ee_definition!r}"
            )
        if policy == CONTINUED_PACKET_DEADLINE_POLICY:
            _validate_complete_useful_conservation(row)
        row["packet_deadline_policy"] = policy
        row["primary_energy_efficiency_definition_id"] = ee_definition
        identities.setdefault((policy, ee_definition), set()).add(_source_name(row))
    if len(identities) != 1:
        conflicts = "; ".join(
            f"policy={policy}, ee_definition={definition}, "
            f"paths={sorted(paths)}"
            for (policy, definition), paths in sorted(identities.items())
        )
        raise ValueError(f"mixed evaluation policies cannot be aggregated: {conflicts}")
    return next(iter(identities))


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

    episode_rows = list(episode_rows)
    packet_policy, ee_definition = resolve_episode_evaluation_semantics(episode_rows)
    grouped = {}
    for row in episode_rows:
        seed = int(row.get(seed_field, 0))
        grouped.setdefault(seed, []).append(row)
    output = []
    for seed, rows in sorted(grouped.items()):
        primary_useful = 0.0
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
            if ee_definition == COMPLETE_EE_DEFINITION:
                primary_useful += _row_number(row, "total_all_useful_mbits")
            else:
                primary_useful += useful
        energy = sum(_row_number(row, "total_mobility_energy_j") for row in rows)
        seed_rows = [
            _ratio_row(
                "energy_efficiency_mbit_per_j", None,
                primary_useful, energy, "Mbit", "J", "Mbit/J",
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
            row["packet_deadline_policy"] = packet_policy
            row["primary_energy_efficiency_definition_id"] = ee_definition
        output.extend(seed_rows)
    return output


def aggregate_seed_metric_rows(seed_rows, metric_tasks):
    """Equal-weight seed metrics with the canonical Student-t statistics."""

    metric_tasks = tuple(metric_tasks)
    if len(metric_tasks) != len(set(metric_tasks)):
        raise ValueError("aggregate metric identities must be unique")
    grouped = {}
    for row in seed_rows:
        grouped.setdefault((row["metric"], row.get("task_type")), []).append(row)
    output = []
    for metric_task in metric_tasks:
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


def aggregate_seed_rows(seed_rows):
    """Equal-weight canonical seed values with sample SD and Student-t 95% CI."""

    return aggregate_seed_metric_rows(seed_rows, CANONICAL_METRICS)


def canonical_aggregation(episode_rows, *, seed_field="training_seed"):
    per_seed = aggregate_episode_rows_by_seed(
        episode_rows, seed_field=seed_field
    )
    return per_seed, aggregate_seed_rows(per_seed)
