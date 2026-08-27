from copy import deepcopy
import math
import unittest

from evaluation_metrics import METRIC_COLUMNS, validate_formal_aggregation_rows
from experiment_config import MethodSpec


class FormalAggregationValidationTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for seed in (101, 202):
            for scenario in ("s0", "s1", "s2"):
                row = {
                    "method_id": MethodSpec().method_id,
                    "training_seed": seed,
                    "evaluation_split": "test",
                    "scenario_id": scenario,
                    "evaluation_manifest_hash": "evaluation-hash",
                    "training_manifest_hash": "training-hash",
                    "checkpoint_completed_episodes": 1500,
                    "checkpoint_metadata_fingerprint": f"checkpoint-{seed}",
                }
                row.update({metric: 1.0 for metric in METRIC_COLUMNS})
                row.update(
                    {
                        "com_delivered_e2e_delay_sum_seconds": 1.0,
                        "com_delivered_packets": 1,
                        "com_eligible_packets": 1,
                        "com_violation_packets": 1,
                        "fov_delivered_e2e_delay_sum_seconds": 1.0,
                        "fov_delivered_packets": 1,
                        "fov_eligible_packets": 1,
                        "fov_violation_packets": 1,
                    }
                )
                rows.append(row)
        return rows

    def _validate(self, rows):
        return validate_formal_aggregation_rows(
            rows,
            expected_method_id=MethodSpec().method_id,
            expected_split="test",
            expected_seed_count=2,
            expected_episodes_per_seed=3,
        )

    def test_complete_fixture_is_accepted(self):
        rows = self._rows()

        self.assertIs(self._validate(rows), rows)

    def test_missing_seed_or_episode_is_rejected(self):
        rows = self._rows()
        with self.assertRaisesRegex(ValueError, "seed count"):
            self._validate(rows[:3])

        with self.assertRaisesRegex(ValueError, "episode count"):
            self._validate(rows[:-1])

    def test_duplicate_or_rerun_row_is_rejected(self):
        rows = self._rows()
        rows.append(deepcopy(rows[0]))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._validate(rows)

    def test_different_scenario_sets_are_rejected(self):
        rows = self._rows()
        rows[-1]["scenario_id"] = "different"

        with self.assertRaisesRegex(ValueError, "same scenario ID set"):
            self._validate(rows)

    def test_mixed_split_or_manifest_is_rejected(self):
        rows = self._rows()
        rows[-1]["evaluation_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split"):
            self._validate(rows)

        rows = self._rows()
        rows[-1]["evaluation_manifest_hash"] = "other-evaluation-hash"
        with self.assertRaisesRegex(ValueError, "evaluation manifest"):
            self._validate(rows)

        rows = self._rows()
        rows[-1]["training_manifest_hash"] = "other-training-hash"
        with self.assertRaisesRegex(ValueError, "training manifest"):
            self._validate(rows)

    def test_non_finite_metric_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                rows = self._rows()
                rows[0]["coverage"] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    self._validate(rows)


if __name__ == "__main__":
    unittest.main()
