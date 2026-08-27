"""Canonical paper metrics shared by evaluation writers and figure validators."""

from __future__ import annotations

import math

from evaluation_aggregation import (
    EVALUATION_AGGREGATION_SCHEMA_VERSION,
    canonical_aggregation,
)


PAPER_EE_EPSILON_J = 1e-12
AGGREGATE_REL_TOL = 1e-12
AGGREGATE_ABS_TOL = 1e-15
PAPER_AGGREGATE_SCHEMA_VERSION = "uav-hrl-paper-aggregate-v3"
CANONICAL_AGGREGATE_ROWS = (
    ("energy_efficiency_mbit_per_j", None),
    ("average_e2e_delay_seconds", "FOV"),
    ("average_e2e_delay_seconds", "COM"),
    ("violation_probability", "FOV"),
    ("violation_probability", "COM"),
    ("violation_probability", "ALL"),
)
AGGREGATE_COMPARE_FIELDS = (
    "aggregate_schema_version",
    "semantic_suite",
    "method_id",
    "point_id",
    "x_value",
    "x_unit",
    "fixed_num_gt",
    "swept_task",
    "evaluation_episode_count",
    "metric",
    "task_type",
    "display_task_type",
    "numerator",
    "numerator_unit",
    "denominator",
    "denominator_unit",
    "value",
    "value_unit",
    "missing",
    "canonical_aggregation_schema_version",
    "aggregation_rule",
    "valid_training_seed_count",
    "sample_stddev",
    "confidence_interval_method",
    "confidence_level",
    "ci95_half_width",
    "ci95_lower",
    "ci95_upper",
)


def causal_trailing_average(values, window=50):
    window = int(window)
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    result = []
    running = 0.0
    finite_values = []
    for index, value in enumerate(values):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("energy-efficiency series must be finite")
        finite_values.append(value)
        running += value
        if index >= window:
            running -= finite_values[index - window]
        result.append(running / min(index + 1, window))
    return result


def paper_energy_efficiency(timely_goodput_mbits, mobility_energy_j):
    """Production paper adapter: timely delivered Mbit per mobility joule."""

    numerator = float(timely_goodput_mbits)
    denominator = float(mobility_energy_j)
    if not math.isfinite(numerator) or numerator < 0.0:
        raise ValueError("timely goodput must be finite and non-negative")
    if not math.isfinite(denominator) or denominator < 0.0:
        raise ValueError("mobility energy must be finite and non-negative")
    value = numerator / max(denominator, PAPER_EE_EPSILON_J)
    if not math.isfinite(value):
        raise ValueError("paper energy efficiency is non-finite")
    return value


def normalize_episode_ee(method_id, history_rows, window=50):
    ordered = sorted(history_rows, key=lambda row: int(row["episode"]))
    episodes = [int(row["episode"]) for row in ordered]
    if not episodes or episodes != list(range(1, len(episodes) + 1)):
        raise ValueError(
            f"{method_id} training history must contain contiguous episodes from 1"
        )
    raw_mbit = []
    for row in ordered:
        try:
            timely_mbits = row["timely_goodput_mbits"]
            mobility_joules = row["mobility_energy_j"]
        except KeyError as exc:
            raise ValueError(
                f"{method_id} training history lacks production EE inputs"
            ) from exc
        raw_mbit.append(paper_energy_efficiency(timely_mbits, mobility_joules))
    raw_bits = [value * 1e6 for value in raw_mbit]
    averaged = causal_trailing_average(raw_bits, window=window)
    return [
        {
            "method_id": str(method_id),
            "episode": episode,
            "timely_goodput_mbits": float(source["timely_goodput_mbits"]),
            "mobility_energy_j": float(source["mobility_energy_j"]),
            "raw_energy_efficiency_bit_per_j": raw_value,
            "trailing_50_energy_efficiency_bit_per_j": average,
        }
        for episode, source, raw_value, average in zip(
            episodes, ordered, raw_bits, averaged
        )
    ]


def aggregate_paper_point_metrics(method_id, suite, point, episode_rows):
    """Adapt the shared seed-ratio aggregation into paper plot rows."""

    rows = list(episode_rows)
    if not rows:
        raise ValueError(
            f"paper evaluation point has no episode rows: method={method_id}, "
            f"point={point.get('point_id')}"
        )
    common = {
        "aggregate_schema_version": PAPER_AGGREGATE_SCHEMA_VERSION,
        "semantic_suite": str(suite),
        "method_id": str(method_id),
        "point_id": str(point["point_id"]),
        "x_value": point.get("x_value"),
        "x_unit": point.get("x_unit"),
        "fixed_num_gt": point.get("fixed_num_gt"),
        "swept_task": point.get("swept_task"),
        "evaluation_episode_count": len(rows),
    }
    per_seed, cross_seed = canonical_aggregation(rows)
    result = []
    for aggregate in cross_seed:
        task_type = aggregate["task_type"]
        result.append(
            {
                **common,
                "canonical_aggregation_schema_version": (
                    EVALUATION_AGGREGATION_SCHEMA_VERSION
                ),
                "aggregation_rule": aggregate["aggregation_rule"],
                "metric": aggregate["metric"],
                "task_type": task_type,
                "display_task_type": (
                    "VS" if task_type == "FOV" else task_type
                ),
                "numerator": aggregate["pooled_numerator"],
                "numerator_unit": aggregate["numerator_unit"],
                "denominator": aggregate["pooled_denominator"],
                "denominator_unit": aggregate["denominator_unit"],
                "value": aggregate["mean"],
                "value_unit": aggregate["value_unit"],
                "missing": aggregate["missing"],
                "valid_training_seed_count": aggregate[
                    "valid_training_seed_count"
                ],
                "missing_training_seed_count": aggregate[
                    "missing_training_seed_count"
                ],
                "training_seed_count": aggregate["training_seed_count"],
                "valid_training_seeds": aggregate["valid_training_seeds"],
                "per_seed_numerators": aggregate["per_seed_numerators"],
                "per_seed_denominators": aggregate["per_seed_denominators"],
                "per_seed_values": aggregate["per_seed_values"],
                "sample_stddev": aggregate["sample_stddev"],
                "degrees_of_freedom": aggregate["degrees_of_freedom"],
                "confidence_interval_method": aggregate[
                    "confidence_interval_method"
                ],
                "confidence_level": aggregate["confidence_level"],
                "t_critical_975": aggregate["t_critical_975"],
                "ci95_half_width": aggregate["ci95_half_width"],
                "ci95_lower": aggregate["ci95_lower"],
                "ci95_upper": aggregate["ci95_upper"],
                "per_seed_aggregation": [
                    row
                    for row in per_seed
                    if row["metric"] == aggregate["metric"]
                    and row.get("task_type") == task_type
                ],
            }
        )
    validate_canonical_aggregate_rows(result, method_id, point["point_id"])
    return result


def aggregate_row_key(row):
    return (
        str(row.get("method_id")),
        str(row.get("point_id")),
        row.get("metric"),
        row.get("task_type"),
    )


def _identity(method_id, point_id, metric, task_type):
    return (
        f"method={method_id}, point={point_id}, metric={metric}, "
        f"task_type={task_type}"
    )


def _finite_nonnegative(value, field, identity, *, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{identity}: {field} is not numeric: actual={value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{identity}: {field} must be finite and non-negative: actual={value!r}"
        )
    if integer and not number.is_integer():
        raise ValueError(f"{identity}: {field} must be an integer: actual={value!r}")
    return int(number) if integer else number


def _same_number(expected, actual):
    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(expected_number) and math.isfinite(actual_number) and math.isclose(
        expected_number,
        actual_number,
        rel_tol=AGGREGATE_REL_TOL,
        abs_tol=AGGREGATE_ABS_TOL,
    )


def _require_value(expected, actual, field, identity):
    matches = (
        expected is None and actual is None
        or isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and _same_number(expected, actual)
        or expected == actual
    )
    if not matches:
        raise ValueError(
            f"{identity}: {field} mismatch: expected={expected!r}, actual={actual!r}"
        )


def validate_canonical_aggregate_rows(rows, method_id, point_id):
    rows = list(rows)
    expected_pairs = set(CANONICAL_AGGREGATE_ROWS)
    seen = {}
    for row in rows:
        metric = row.get("metric")
        task_type = row.get("task_type")
        identity = _identity(method_id, point_id, metric, task_type)
        _require_value(
            PAPER_AGGREGATE_SCHEMA_VERSION,
            row.get("aggregate_schema_version"),
            "aggregate_schema_version",
            identity,
        )
        _require_value(str(method_id), row.get("method_id"), "method_id", identity)
        _require_value(str(point_id), row.get("point_id"), "point_id", identity)
        pair = (metric, task_type)
        if pair not in expected_pairs:
            raise ValueError(
                f"{identity}: unexpected canonical aggregate row: "
                f"expected_one_of={sorted(expected_pairs, key=str)}, actual={pair}"
            )
        if pair in seen:
            raise ValueError(f"{identity}: duplicate canonical aggregate row")
        seen[pair] = row
    missing = expected_pairs.difference(seen)
    if missing:
        raise ValueError(
            f"method={method_id}, point={point_id}: missing canonical aggregate rows: "
            f"expected={sorted(missing, key=str)}, actual={sorted(seen, key=str)}"
        )
    if len(rows) != len(CANONICAL_AGGREGATE_ROWS):
        raise ValueError(
            f"method={method_id}, point={point_id}: aggregate row count mismatch: "
            f"expected={len(CANONICAL_AGGREGATE_ROWS)}, actual={len(rows)}"
        )
    combined = seen[("violation_probability", "ALL")]
    task_rows = [
        seen[("violation_probability", task_type)]
        for task_type in ("FOV", "COM")
    ]
    for task_row in task_rows:
        task_identity = _identity(
            method_id,
            point_id,
            "violation_probability",
            task_row["task_type"],
        )
        task_numerator = _finite_nonnegative(
            task_row.get("numerator"), "numerator", task_identity, integer=True
        )
        task_denominator = _finite_nonnegative(
            task_row.get("denominator"), "denominator", task_identity, integer=True
        )
        if task_numerator > task_denominator:
            raise ValueError(
                f"{task_identity}: violation numerator exceeds denominator: "
                f"actual={task_numerator}>{task_denominator}"
            )
    combined_identity = _identity(
        method_id, point_id, "violation_probability", "ALL"
    )
    _require_value(
        sum(int(row["numerator"]) for row in task_rows),
        combined["numerator"],
        "pooled_task_numerator",
        combined_identity,
    )
    _require_value(
        sum(int(row["denominator"]) for row in task_rows),
        combined["denominator"],
        "pooled_task_denominator",
        combined_identity,
    )
    for metric, task_type in CANONICAL_AGGREGATE_ROWS:
        row = seen[(metric, task_type)]
        identity = _identity(method_id, point_id, metric, task_type)
        if type(row.get("missing")) is not bool:
            raise ValueError(
                f"{identity}: missing must be boolean: actual={row.get('missing')!r}"
            )
        _require_value(
            EVALUATION_AGGREGATION_SCHEMA_VERSION,
            row.get("canonical_aggregation_schema_version"),
            "canonical_aggregation_schema_version",
            identity,
        )
        _require_value(
            "equal_weight_valid_training_seed_values",
            row.get("aggregation_rule"),
            "aggregation_rule",
            identity,
        )
        per_seed_values = row.get("per_seed_values")
        if not isinstance(per_seed_values, list):
            raise ValueError(f"{identity}: per_seed_values must be a list")
        valid_values = [
            float(value) for value in per_seed_values if value is not None
        ]
        expected_value = (
            sum(valid_values) / len(valid_values) if valid_values else None
        )
        expected_missing = not valid_values
        if task_type != "ALL":
            _require_value(
                sum(float(value) for value in row.get("per_seed_numerators", [])),
                row.get("numerator"),
                "per_seed_numerator_sum",
                identity,
            )
            _require_value(
                sum(float(value) for value in row.get("per_seed_denominators", [])),
                row.get("denominator"),
                "per_seed_denominator_sum",
                identity,
            )
        _require_value(
            len(valid_values),
            row.get("valid_training_seed_count"),
            "valid_training_seed_count",
            identity,
        )
        _require_value("Student-t", row.get("confidence_interval_method"), "confidence_interval_method", identity)
        _require_value(0.95, row.get("confidence_level"), "confidence_level", identity)
        if metric == "energy_efficiency_mbit_per_j":
            numerator = _finite_nonnegative(row.get("numerator"), "numerator", identity)
            denominator = _finite_nonnegative(row.get("denominator"), "denominator", identity)
            expected_units = ("Mbit", "J", "Mbit/J")
        elif metric == "average_e2e_delay_seconds":
            numerator = _finite_nonnegative(row.get("numerator"), "numerator", identity)
            denominator = _finite_nonnegative(
                row.get("denominator"), "denominator", identity, integer=True
            )
            expected_units = ("seconds", "delivered_packets", "seconds")
        else:
            numerator = _finite_nonnegative(
                row.get("numerator"), "numerator", identity, integer=True
            )
            denominator = _finite_nonnegative(
                row.get("denominator"), "denominator", identity, integer=True
            )
            if numerator > denominator:
                raise ValueError(
                    f"{identity}: violation numerator exceeds denominator: "
                    f"expected=0<=numerator<=denominator, actual={numerator}>{denominator}"
                )
            expected_units = (
                "violation_packets",
                "eligible_packets",
                "probability",
            )
        for field, expected in zip(
            ("numerator_unit", "denominator_unit", "value_unit"), expected_units
        ):
            _require_value(expected, row.get(field), field, identity)
        _require_value(expected_missing, row.get("missing"), "missing", identity)
        _require_value(expected_value, row.get("value"), "value", identity)
        if metric == "violation_probability" and expected_value is not None and not 0.0 <= expected_value <= 1.0:
            raise ValueError(
                f"{identity}: violation probability outside [0,1]: actual={expected_value}"
            )
    return rows


def validate_aggregate_collection(rows, method_id, point_ids):
    rows = list(rows)
    expected_points = tuple(str(point_id) for point_id in point_ids)
    grouped = {point_id: [] for point_id in expected_points}
    for row in rows:
        point_id = str(row.get("point_id"))
        if point_id not in grouped:
            raise ValueError(
                f"method={method_id}: unexpected aggregate point: "
                f"expected={expected_points}, actual={point_id}"
            )
        grouped[point_id].append(row)
    for point_id in expected_points:
        validate_canonical_aggregate_rows(grouped[point_id], method_id, point_id)
    return rows


def compare_aggregate_collections(expected_rows, actual_rows, *, context):
    expected_rows = list(expected_rows)
    actual_rows = list(actual_rows)
    expected = {aggregate_row_key(row): row for row in expected_rows}
    actual = {aggregate_row_key(row): row for row in actual_rows}
    if len(expected) != len(expected_rows) or len(actual) != len(actual_rows):
        raise ValueError(f"{context}: aggregate comparison received duplicate keys")
    if set(expected) != set(actual):
        raise ValueError(
            f"{context}: aggregate key mismatch: "
            f"expected={sorted(expected, key=str)}, actual={sorted(actual, key=str)}"
        )
    numeric_fields = {
        "x_value",
        "evaluation_episode_count",
        "numerator",
        "denominator",
        "value",
    }
    for key in sorted(expected, key=str):
        identity = _identity(key[0], key[1], key[2], key[3])
        for field in AGGREGATE_COMPARE_FIELDS:
            expected_value = expected[key].get(field)
            actual_value = actual[key].get(field)
            if field in numeric_fields and expected_value is not None:
                matches = _same_number(expected_value, actual_value)
            else:
                matches = expected_value == actual_value
            if not matches:
                raise ValueError(
                    f"{context}; {identity}: {field} mismatch: "
                    f"expected={expected_value!r}, actual={actual_value!r}"
                )
    return actual_rows
