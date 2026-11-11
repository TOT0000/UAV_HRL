import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation_metrics import (
    METRIC_COLUMNS,
    _read_episode_csvs,
    canonical_cross_seed_artifact_rows,
    safe_energy_efficiency,
    summarize_training_seeds,
    write_evaluation_outputs,
)
from evaluation_aggregation import canonical_aggregation
from evaluation_aggregation import (
    COMPLETE_EE_DEFINITION,
    CONTINUED_PACKET_DEADLINE_POLICY,
    LEGACY_PACKET_DEADLINE_POLICY,
    TIMELY_EE_DEFINITION,
)
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from Packet_scheduler_v1 import (
    PACKET_DEADLINE_POLICY_CONTINUE,
)
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
        row.update(
            {
                "packet_deadline_policy": CONTINUED_PACKET_DEADLINE_POLICY,
                "primary_energy_efficiency_definition_id": COMPLETE_EE_DEFINITION,
                "total_timely_useful_mbits": float(value),
                "total_late_useful_mbits": 0.0,
                "total_all_useful_mbits": float(value),
                "total_timely_useful_bits": float(value) * 1e6,
                "total_late_useful_bits": 0.0,
                "total_all_useful_bits": float(value) * 1e6,
            }
        )
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

    def test_primary_ee_uses_complete_timely_plus_late_bits(self):
        row = self._row(11, "a", 1.0)
        row["total_timely_useful_mbits"] = 1.0
        row["timely_goodput_mbits"] = 1.0
        row["total_late_useful_mbits"] = 2.0
        row["total_all_useful_mbits"] = 3.0
        row["total_timely_useful_bits"] = 1e6
        row["total_late_useful_bits"] = 2e6
        row["total_all_useful_bits"] = 3e6
        row["total_mobility_energy_j"] = 2.0

        per_seed, cross_seed = canonical_aggregation([row])
        energy = next(
            item
            for item in per_seed
            if item["metric"] == "energy_efficiency_mbit_per_j"
        )
        summary = summarize_training_seeds([row])[0]

        self.assertEqual(energy["numerator"], 3.0)
        self.assertEqual(energy["value"], 1.5)
        self.assertEqual(summary["timely_energy_efficiency_mbit_per_j"], 0.5)
        self.assertEqual(summary["all_energy_efficiency_mbit_per_j"], 1.5)
        self.assertEqual(cross_seed[0]["per_seed_numerators"], [3.0])

    def test_old_rows_without_late_fields_keep_timely_ee_semantics(self):
        row = self._row(11, "legacy", 2.0)
        for field in (
            "total_late_useful_mbits",
            "total_all_useful_mbits",
            "total_late_useful_bits",
            "total_all_useful_bits",
        ):
            row.pop(field)
        row["packet_deadline_policy"] = LEGACY_PACKET_DEADLINE_POLICY
        row["primary_energy_efficiency_definition_id"] = TIMELY_EE_DEFINITION
        per_seed, _cross_seed = canonical_aggregation([row])
        energy = next(
            item
            for item in per_seed
            if item["metric"] == "energy_efficiency_mbit_per_j"
        )
        self.assertEqual(energy["numerator"], 2.0)

    def test_mixed_old_and_new_evaluation_policies_are_rejected_with_paths(self):
        current = self._row(11, "current", 2.0)
        current["_source_path"] = "new/per_episode.csv"
        legacy = self._row(12, "legacy", 2.0)
        legacy["packet_deadline_policy"] = LEGACY_PACKET_DEADLINE_POLICY
        legacy["primary_energy_efficiency_definition_id"] = TIMELY_EE_DEFINITION
        legacy["_source_path"] = "old/per_episode.csv"

        with self.assertRaisesRegex(
            ValueError, "mixed evaluation policies.*new/per_episode.csv.*old/per_episode.csv"
        ):
            canonical_aggregation([current, legacy])

    def test_complete_useful_bit_conservation_is_required(self):
        row = self._row(11, "bad", 1.0)
        row["_source_path"] = "bad/per_episode.csv"
        row["total_all_useful_bits"] = 2.0

        with self.assertRaisesRegex(
            ValueError, "bad/per_episode.csv.*complete useful bits is not conserved"
        ):
            canonical_aggregation([row])

    def test_csv_reader_preserves_legacy_identity_and_rejects_mixed_paths(self):
        current = self._row(11, "current", 2.0)
        legacy = self._row(12, "legacy", 2.0)
        for field in (
            "packet_deadline_policy",
            "primary_energy_efficiency_definition_id",
            "total_late_useful_mbits",
            "total_all_useful_mbits",
            "total_late_useful_bits",
            "total_all_useful_bits",
        ):
            legacy.pop(field)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, row in (("old", legacy), ("new", current)):
                directory = root / name
                directory.mkdir()
                with (directory / "per_episode.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(row))
                    writer.writeheader()
                    writer.writerow(row)

            old_rows = _read_episode_csvs(root / "old")
            self.assertEqual(
                old_rows[0]["packet_deadline_policy"],
                LEGACY_PACKET_DEADLINE_POLICY,
            )
            old_ee = next(
                item
                for item in canonical_aggregation(old_rows)[0]
                if item["metric"] == "energy_efficiency_mbit_per_j"
            )
            self.assertEqual(old_ee["numerator"], 2.0)

            with self.assertRaisesRegex(
                ValueError, "mixed evaluation policies.*new.*per_episode.csv.*old.*per_episode.csv|mixed evaluation policies.*old.*per_episode.csv.*new.*per_episode.csv"
            ):
                _read_episode_csvs(root)

    def test_csv_reader_recognizes_pre_fix_current_metadata_without_ids(self):
        row = self._row(11, "current", 2.0)
        row.pop("packet_deadline_policy")
        row.pop("primary_energy_efficiency_definition_id")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "per_episode.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            (root / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "packet_deadline_policy": CONTINUED_PACKET_DEADLINE_POLICY,
                        "primary_energy_efficiency_definition": (
                            "(timely useful bits + late complete useful bits) / "
                            "UAV movement energy"
                        ),
                    }
                ),
                encoding="utf-8",
            )

            loaded = _read_episode_csvs(root)

        self.assertEqual(
            loaded[0]["primary_energy_efficiency_definition_id"],
            COMPLETE_EE_DEFINITION,
        )

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
            self.assertEqual(
                checkpoint_metadata["experiment"]["packet_deadline_policy"],
                PACKET_DEADLINE_POLICY_CONTINUE,
            )
            self.assertEqual(
                checkpoint_metadata["experiment"][
                    "primary_energy_efficiency_definition_id"
                ],
                COMPLETE_EE_DEFINITION,
            )
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
                formal_evaluation=True,
                checkpoint_dir=checkpoint_dir,
                evaluation_overrides={
                    "total_communication_bandwidth_hz": 5.5e6,
                    "s2u_tx_power_dbm": -2.5,
                    "uav_tx_power_dbm": 18.25,
                },
            )
            implicit_default_result = train(
                evaluation_config,
                scenario_manifest=evaluation_manifest,
                method_spec=method,
                evaluation=True,
                formal_evaluation=True,
                checkpoint_dir=checkpoint_dir,
            )
            explicit_default_result = train(
                evaluation_config,
                scenario_manifest=evaluation_manifest,
                method_spec=method,
                evaluation=True,
                formal_evaluation=True,
                checkpoint_dir=checkpoint_dir,
                evaluation_overrides={
                    "total_communication_bandwidth_hz": 10e6,
                    "s2u_tx_power_dbm": 23.0,
                    "uav_tx_power_dbm": 30.0,
                },
            )

        self.assertTrue(all(result["evaluation_invariants"].values()))
        self.assertTrue(
            all(implicit_default_result["evaluation_invariants"].values())
        )
        self.assertTrue(
            all(explicit_default_result["evaluation_invariants"].values())
        )
        self.assertEqual(
            implicit_default_result["episode_metrics"],
            explicit_default_result["episode_metrics"],
        )
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
            metadata["evaluation_overrides"]["total_communication_bandwidth_hz"],
            5.5e6,
        )
        self.assertEqual(
            metadata["evaluation_overrides"]["s2u_tx_power_dbm"], -2.5
        )
        self.assertEqual(
            metadata["evaluation_overrides"]["uav_tx_power_dbm"], 18.25
        )
        self.assertTrue(metadata["formal_evaluation"])
        self.assertEqual(
            metadata["packet_deadline_policy"], PACKET_DEADLINE_POLICY_CONTINUE
        )
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

    def test_formal_evaluation_uses_explicit_continue_policy_for_random_baseline(self):
        method = MethodSpec.parse("kkm_random_action_random_routing")
        manifest = generate_manifest("test", 9917, 1)
        result = train(
            TrainingConfig(
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
                random_seed=9917,
            ),
            scenario_manifest=manifest,
            method_spec=method,
            evaluation=True,
            formal_evaluation=True,
        )

        metadata = result["run_metadata"]
        self.assertTrue(metadata["formal_evaluation"])
        self.assertEqual(
            metadata["packet_deadline_policy"],
            PACKET_DEADLINE_POLICY_CONTINUE,
        )
        row = result["episode_metrics"][0]
        self.assertAlmostEqual(
            row["total_all_useful_bits"],
            row["total_timely_useful_bits"] + row["total_late_useful_bits"],
        )
        self.assertEqual(
            row["energy_efficiency_mbit_per_j"],
            row["all_energy_efficiency_mbit_per_j"],
        )


if __name__ == "__main__":
    unittest.main()
