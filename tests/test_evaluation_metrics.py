import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation_metrics import (
    METRIC_COLUMNS,
    canonical_cross_seed_artifact_rows,
    safe_energy_efficiency,
    summarize_training_seeds,
    write_evaluation_outputs,
)
from evaluation_aggregation import canonical_aggregation
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from scenario_manifest import generate_manifest


class EvaluationMetricTest(unittest.TestCase):
    def _row(self, training_seed, scenario_id, value):
        row = {
            "method_id": MethodSpec().method_id,
            "training_seed": training_seed,
            "evaluation_split": "test",
            "scenario_id": scenario_id,
            "evaluation_manifest_hash": "evaluation-manifest-hash",
            "training_manifest_hash": "training-manifest-hash",
            "checkpoint_completed_episodes": 1500,
            "checkpoint_metadata_fingerprint": f"checkpoint-{training_seed}",
        }
        row.update({metric: float(value) for metric in METRIC_COLUMNS})
        # Canonical EE is computed from raw per-seed numerator/denominator sums.
        row["timely_goodput_mbits"] = float(value)
        row["total_mobility_energy_j"] = 1.0
        for prefix in ("fov", "com"):
            row[f"{prefix}_delivered_packets"] = 1
            row[f"{prefix}_delivered_e2e_delay_sum_seconds"] = float(value)
            row[f"{prefix}_violation_packets"] = 0
            row[f"{prefix}_eligible_packets"] = 1
        return row

    def test_zero_energy_efficiency_is_finite(self):
        value = safe_energy_efficiency(12.5, 0.0)

        self.assertEqual(value, 0.0)
        self.assertTrue(math.isfinite(value))

    def test_aggregation_uses_seed_means_for_uncertainty(self):
        episodes = []
        for training_seed, value in enumerate(range(1, 6), start=11):
            episodes.extend(
                (
                    self._row(training_seed, "a", value),
                    self._row(training_seed, "b", value),
                )
            )

        seed_summaries = summarize_training_seeds(episodes)
        _per_seed, aggregate = canonical_aggregation(episodes)
        ee = next(
            row
            for row in aggregate
            if row["metric"] == "energy_efficiency_mbit_per_j"
        )
        artifact_rows = canonical_cross_seed_artifact_rows(
            aggregate, episodes
        )
        artifact_ee = next(
            row
            for row in artifact_rows
            if row["metric"] == "energy_efficiency_mbit_per_j"
        )

        self.assertEqual(len(seed_summaries), 5)
        self.assertEqual(ee["valid_training_seed_count"], 5)
        self.assertAlmostEqual(ee["mean"], 3.0)
        self.assertAlmostEqual(ee["sample_stddev"], math.sqrt(2.5))
        self.assertAlmostEqual(ee["t_critical_975"], 2.7764451051977987)
        self.assertAlmostEqual(
            ee["ci95_half_width"],
            ee["t_critical_975"] * math.sqrt(2.5) / math.sqrt(5),
        )
        self.assertEqual(artifact_ee["mean"], ee["mean"])
        self.assertEqual(
            artifact_ee["per_seed_numerators"], ee["per_seed_numerators"]
        )

    def test_evaluation_outputs_are_structured_and_finite(self):
        rows = [self._row(11, "a", 0.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_evaluation_outputs(temp_dir, rows, {"evaluation": True})

            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            self.assertNotIn("NaN", Path(paths["per_episode_jsonl"]).read_text())
            self.assertNotIn("Infinity", Path(paths["per_episode_jsonl"]).read_text())

    def test_no_checkpoint_baseline_preserves_null_provenance(self):
        row = self._row(11, "a", 0.0)
        row["training_manifest_hash"] = None
        row["checkpoint_completed_episodes"] = None
        row["checkpoint_metadata_fingerprint"] = None
        summaries = summarize_training_seeds([row])
        self.assertIsNone(summaries[0]["training_manifest_hash"])
        self.assertIsNone(summaries[0]["checkpoint_completed_episodes"])
        self.assertIsNone(summaries[0]["checkpoint_metadata_fingerprint"])


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
            checkpoint_metadata = json.loads(
                (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
            )
            checkpoint_dinkelbach_state = checkpoint_metadata["experiment"][
                "dinkelbach_state"
            ]
            checkpoint_constraint_state = checkpoint_metadata[
                "routing_agent_configuration"
            ]

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
        self.assertEqual(result["routing_training_global_slot_count"], 0)
        self.assertEqual(result["movement_post_warmup_transition_count"], 0)
        self.assertEqual(result["dinkelbach_update_count"], 0)
        self.assertEqual(result["dinkelbach_state"], checkpoint_dinkelbach_state)
        for field, value in result["safe_ddqn_constraint_state"].items():
            self.assertEqual(value, checkpoint_constraint_state[field])
        self.assertEqual(
            result["lambda_cost_used_log"],
            [checkpoint_constraint_state["lambda_cost"]],
        )
        self.assertEqual(
            result["lambda_cost_after_episode_log"],
            [checkpoint_constraint_state["lambda_cost"]],
        )
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
        self.assertEqual(
            metadata["checkpoint_dinkelbach_config"],
            {
                field: checkpoint_metadata["experiment"]["formal_config"][field]
                for field in metadata["checkpoint_dinkelbach_config"]
            },
        )
        self.assertEqual(
            metadata["checkpoint_dinkelbach_state"],
            checkpoint_dinkelbach_state,
        )
        self.assertTrue(metadata["checkpoint_metadata_path"].endswith("metadata.json"))
        self.assertEqual(len(metadata["checkpoint_metadata_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
