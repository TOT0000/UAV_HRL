from copy import deepcopy
import math
import unittest

from paper_metrics import (
    aggregate_paper_point_metrics,
    validate_canonical_aggregate_rows,
)


class CanonicalPaperAggregateValidationTest(unittest.TestCase):
    method = "td3_dinkelbach"
    point = {"point_id": "roi_2", "x_value": 2, "x_unit": "RoIs", "fixed_num_gt": 2}

    @staticmethod
    def _episode_rows(*, fov_delivered=2, fov_eligible=4, com_delivered=1, com_eligible=3):
        return [
            {
                "timely_goodput_mbits": 3.0,
                "total_mobility_energy_j": 6.0,
                "fov_delivered_packets": fov_delivered,
                "fov_delivered_e2e_delay_sum_seconds": 0.5 if fov_delivered else 0.0,
                "fov_generated_packets": fov_eligible + 2,
                "fov_eligible_packets": fov_eligible,
                "fov_violation_packets": min(1, fov_eligible),
                "com_delivered_packets": com_delivered,
                "com_delivered_e2e_delay_sum_seconds": 0.25 if com_delivered else 0.0,
                "com_generated_packets": com_eligible + 3,
                "com_eligible_packets": com_eligible,
                "com_violation_packets": min(2, com_eligible),
            }
        ]

    def _rows(self, **kwargs):
        return aggregate_paper_point_metrics(
            self.method,
            "fixed_roi",
            self.point,
            self._episode_rows(**kwargs),
        )

    def test_exact_six_row_cartesian_is_required_before_figure_filtering(self):
        rows = self._rows()
        self.assertEqual(len(rows), 6)
        unused = next(
            row for row in rows
            if row["metric"] == "violation_probability" and row["task_type"] == "COM"
        )
        cases = {
            "missing_unused": [row for row in rows if row is not unused],
            "duplicate_unused": [*rows, deepcopy(unused)],
            "extra": [*rows, {**deepcopy(unused), "task_type": "UNKNOWN"}],
        }
        for name, candidate in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "missing|duplicate|unexpected"
            ):
                validate_canonical_aggregate_rows(
                    candidate, self.method, self.point["point_id"]
                )

    def test_numerator_denominator_value_unit_and_missing_mutations_fail(self):
        rows = self._rows()
        base = next(
            row for row in rows
            if row["metric"] == "average_e2e_delay_seconds" and row["task_type"] == "FOV"
        )
        mutations = {
            "numerator": {"numerator": base["numerator"] + 1.0},
            "denominator": {"denominator": base["denominator"] + 1},
            "value": {"value": base["value"] + 0.1},
            "unit": {"value_unit": "milliseconds"},
            "missing": {"missing": True},
            "nonfinite": {"numerator": math.inf},
            "negative": {"denominator": -1},
        }
        for name, updates in mutations.items():
            candidate = deepcopy(rows)
            target = next(
                row for row in candidate
                if row["metric"] == base["metric"] and row["task_type"] == base["task_type"]
            )
            target.update(updates)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_canonical_aggregate_rows(
                    candidate, self.method, self.point["point_id"]
                )

    def test_violation_bounds_and_zero_denominator_missing_semantics(self):
        rows = self._rows(fov_delivered=0, fov_eligible=0, com_eligible=1)
        delay = next(
            row for row in rows
            if row["metric"] == "average_e2e_delay_seconds" and row["task_type"] == "FOV"
        )
        violation = next(
            row for row in rows
            if row["metric"] == "violation_probability" and row["task_type"] == "FOV"
        )
        self.assertIsNone(delay["value"])
        self.assertTrue(delay["missing"])
        self.assertIsNone(violation["value"])
        self.assertTrue(violation["missing"])

        for row in (delay, violation):
            candidate = deepcopy(rows)
            target = next(
                value for value in candidate
                if value["metric"] == row["metric"] and value["task_type"] == "FOV"
            )
            target["value"] = 0.0
            target["missing"] = False
            with self.assertRaises(ValueError):
                validate_canonical_aggregate_rows(
                    candidate, self.method, self.point["point_id"]
                )

        candidate = self._rows(com_eligible=1)
        com_violation = next(
            row for row in candidate
            if row["metric"] == "violation_probability" and row["task_type"] == "COM"
        )
        com_violation["numerator"] = 2
        com_violation["value"] = 2.0
        with self.assertRaisesRegex(ValueError, "exceeds denominator"):
            validate_canonical_aggregate_rows(
                candidate, self.method, self.point["point_id"]
            )

    def test_all_violation_probability_pools_raw_fov_and_com_counts(self):
        rows = self._rows(fov_eligible=4, com_eligible=6)
        combined = next(
            row
            for row in rows
            if row["metric"] == "violation_probability"
            and row["task_type"] == "ALL"
        )
        self.assertEqual(combined["numerator"], 3)
        self.assertEqual(combined["denominator"], 10)
        self.assertEqual(combined["denominator_unit"], "eligible_packets")
        self.assertAlmostEqual(combined["value"], 0.3)

        combined["numerator"] = 4
        combined["value"] = 0.4
        with self.assertRaisesRegex(ValueError, "pooled_task_numerator"):
            validate_canonical_aggregate_rows(
                rows, self.method, self.point["point_id"]
            )

    def test_all_zero_eligible_is_missing_not_zero(self):
        rows = self._rows(fov_eligible=0, com_eligible=0)
        combined = next(
            row
            for row in rows
            if row["metric"] == "violation_probability"
            and row["task_type"] == "ALL"
        )
        self.assertEqual(combined["denominator"], 0)
        self.assertIsNone(combined["value"])
        self.assertTrue(combined["missing"])

    def test_combined_constraint_is_point_nine_not_task_diagnostic(self):
        episode = self._episode_rows(fov_eligible=1, com_eligible=9)[0]
        episode["fov_violation_packets"] = 0
        episode["com_violation_packets"] = 9
        rows = aggregate_paper_point_metrics(
            self.method, "fixed_roi", self.point, [episode]
        )
        probabilities = {
            row["task_type"]: row["value"]
            for row in rows
            if row["metric"] == "violation_probability"
        }
        self.assertEqual(probabilities["FOV"], 0.0)
        self.assertEqual(probabilities["COM"], 1.0)
        self.assertEqual(probabilities["ALL"], 0.9)


if __name__ == "__main__":
    unittest.main()
