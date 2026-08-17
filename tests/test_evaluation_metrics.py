import math
import tempfile
import unittest
from pathlib import Path

from evaluation_metrics import (
    METRIC_COLUMNS,
    aggregate_seed_means,
    safe_energy_efficiency,
    summarize_training_seeds,
    write_evaluation_outputs,
)
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from scenario_manifest import generate_manifest


class EvaluationMetricTest(unittest.TestCase):
    def _row(self, training_seed, scenario_id, value):
        row = {
            "method_id": MethodSpec().method_id,
            "training_seed": training_seed,
            "scenario_id": scenario_id,
            "manifest_hash": "manifest-hash",
        }
        row.update({metric: float(value) for metric in METRIC_COLUMNS})
        return row

    def test_zero_energy_efficiency_is_finite(self):
        value = safe_energy_efficiency(12.5, 0.0)

        self.assertEqual(value, 0.0)
        self.assertTrue(math.isfinite(value))

    def test_aggregation_uses_seed_means_for_uncertainty(self):
        episodes = [
            self._row(11, "a", 1.0),
            self._row(11, "b", 3.0),
            self._row(22, "a", 3.0),
            self._row(22, "b", 5.0),
        ]

        seed_summaries = summarize_training_seeds(episodes)
        aggregate = aggregate_seed_means(seed_summaries)
        ee = next(
            row
            for row in aggregate
            if row["metric"] == "energy_efficiency_mbit_per_j"
        )

        self.assertEqual(len(seed_summaries), 2)
        self.assertEqual(ee["training_seed_count"], 2)
        self.assertAlmostEqual(ee["mean"], 3.0)
        self.assertAlmostEqual(ee["sample_stddev"], math.sqrt(2.0))
        self.assertAlmostEqual(ee["ci95_half_width"], 1.96)

    def test_evaluation_outputs_are_structured_and_finite(self):
        rows = [self._row(11, "a", 0.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_evaluation_outputs(temp_dir, rows, {"evaluation": True})

            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            self.assertNotIn("NaN", Path(paths["per_episode_jsonl"]).read_text())
            self.assertNotIn("Infinity", Path(paths["per_episode_jsonl"]).read_text())


class EvaluationModeIntegrationTest(unittest.TestCase):
    def test_evaluation_disables_exploration_and_preserves_learning_state(self):
        method = MethodSpec()
        training_seed = 31415
        train_manifest = generate_manifest("train", 800, 1)
        evaluation_manifest = generate_manifest("validation", 801, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "checkpoints"
            training_config = TrainingConfig(
                total_episodes=1,
                mode="custom",
                episode_seconds=1,
                routing_slot_seconds=0.25,
                warmup_joint_transitions=0,
                batch_size=1,
                model_checkpoint_every=1,
                checkpoint_root=str(checkpoint_root),
                enable_model_checkpoints=True,
                enable_full_resume=False,
                enable_plots=False,
                enable_csv=False,
                random_seed=training_seed,
            )
            train(
                training_config,
                scenario_manifest=train_manifest,
                method_spec=method,
            )
            checkpoint_dir = checkpoint_root / "models" / "ep_0001"

            evaluation_config = TrainingConfig(
                total_episodes=1,
                mode="custom",
                episode_seconds=1,
                routing_slot_seconds=0.25,
                warmup_joint_transitions=0,
                batch_size=1,
                enable_model_checkpoints=False,
                enable_full_resume=False,
                enable_plots=False,
                enable_csv=False,
                random_seed=training_seed,
            )
            result = train(
                evaluation_config,
                scenario_manifest=evaluation_manifest,
                method_spec=method,
                evaluation=True,
                checkpoint_dir=checkpoint_dir,
            )

        self.assertTrue(all(result["evaluation_invariants"].values()))
        self.assertEqual(result["td3_noise_log"], [])
        self.assertEqual(result["routing_epsilon_log"], [0.0] * 4)
        self.assertEqual(result["joint_replay_size"], 0)
        self.assertEqual(result["routing_replay_size"], 0)
        self.assertEqual(len(result["episode_metrics"]), 1)
        self.assertEqual(
            result["episode_metrics"][0]["scenario_id"],
            evaluation_manifest.episodes[0]["scenario_id"],
        )
        metadata = result["run_metadata"]
        self.assertEqual(
            metadata["training_manifest_hash"], train_manifest.content_hash
        )
        self.assertEqual(
            metadata["evaluation_manifest_hash"],
            evaluation_manifest.content_hash,
        )
        self.assertEqual(metadata["checkpoint_completed_episodes"], 1)
        self.assertEqual(metadata["checkpoint_training_seed"], training_seed)
        self.assertEqual(
            metadata["checkpoint_method_spec_fingerprint"], method.fingerprint
        )
        self.assertTrue(metadata["checkpoint_metadata_path"].endswith("metadata.json"))
        self.assertEqual(len(metadata["checkpoint_metadata_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
