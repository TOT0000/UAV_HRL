from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from DDQN import DDQN
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    movement_state_feature_schema,
    projected_joint_action_schema,
)
from com_capacity_calibration import load_com_capacity_reference
from comparison_experiment import main as comparison_main
from design_dataset import (
    ARRAY_NAMES,
    DESIGN_EPISODES_CSV,
    DESIGN_METADATA_FILENAME,
    DESIGN_TRANSITIONS_FILENAME,
    _atomic_write_dataset,
    read_reference_episode_csv,
    reconstruct_reward,
    validate_reference_metrics,
    validate_reference_identity,
)
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from evaluation_metrics import EPISODE_COLUMNS, write_evaluation_outputs
from experiment_config import MethodSpec
from experiment_paths import design_run_directory, read_run_status
from HRL_task_aware import TrainingConfig, formal_training_config, train
from scenario_manifest import generate_manifest
from td3 import TD3
from training_checkpoint import save_model_checkpoint


class DesignSchemaTest(unittest.TestCase):
    def test_state_and_action_schema_cover_every_dimension_once(self):
        state = movement_state_feature_schema()
        action = projected_joint_action_schema()

        continuous = set(state["continuous_indices"])
        discrete = set(state["discrete_indices"])
        self.assertFalse(continuous.intersection(discrete))
        self.assertEqual(continuous | discrete, set(range(532)))
        self.assertEqual([item["index"] for item in state["features"]], list(range(532)))
        self.assertEqual(len(action["features"]), 48)
        self.assertEqual(
            {item["index"] for item in action["features"]}, set(range(48))
        )
        self.assertEqual(action["range"], [-1.0, 1.0])


class DesignDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.method = MethodSpec()
        cls.training_seed = 20260818
        cls.manifest = generate_manifest("validation", 8811, 2)
        cls.manifest_path = cls.root / "validation.json"
        cls.manifest.save(cls.manifest_path)
        cls.formal_config = asdict(
            formal_training_config(2500, random_seed=cls.training_seed)
        )
        cls.dinkelbach_state = DinkelbachBlockState.from_config(cls.formal_config)
        for _ in range(2500):
            cls.dinkelbach_state.record_episode(1.0, 2.0)

        td3 = TD3(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_action=1.0, gamma=1.0)
        ddqn = DDQN(126, 17)
        _, calibration = load_com_capacity_reference()
        cls.checkpoint = cls.root / "checkpoints" / "models" / "ep_2500"
        save_model_checkpoint(
            cls.checkpoint,
            episode=2499,
            td3=td3,
            ddqn=ddqn,
            movement_state_dim=532,
            joint_action_dim=48,
            routing_state_dim=126,
            calibration=calibration,
            experiment_metadata={
                "method_id": cls.method.method_id,
                "method_spec": cls.method.to_dict(),
                "method_spec_fingerprint": cls.method.fingerprint,
                "manifest_hash": "a" * 64,
                "manifest_split": "train",
                "training_seed": cls.training_seed,
                **dinkelbach_config_metadata(cls.formal_config),
                "lambda_ee": cls.dinkelbach_state.current_lambda,
                "dinkelbach_state": cls.dinkelbach_state.training_state(),
                "formal_config": cls.formal_config,
            },
        )

        evaluation_config = TrainingConfig(
            total_episodes=2,
            mode="custom",
            episode_seconds=60,
            routing_slot_seconds=0.25,
            warmup_joint_transitions=0,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=cls.training_seed,
        )
        cls.ordinary = train(
            evaluation_config,
            scenario_manifest=cls.manifest,
            method_spec=cls.method,
            evaluation=True,
            checkpoint_dir=cls.checkpoint,
            expected_checkpoint_episodes=2500,
            expected_checkpoint_formal_config=cls.formal_config,
        )
        cls.reference_dir = cls.root / "reference"
        reference_paths = write_evaluation_outputs(
            cls.reference_dir, cls.ordinary["episode_metrics"], {"reference": True}
        )
        cls.reference_csv = Path(reference_paths["per_episode_csv"])

        cls.output_one = cls.root / "design-one"
        cls.output_two = cls.root / "design-two"
        cls._run_collection(cls.output_one)
        cls._run_collection(cls.output_two)
        cls.run_one = design_run_directory(
            cls.output_one, cls.method, cls.manifest, cls.training_seed
        )
        cls.run_two = design_run_directory(
            cls.output_two, cls.method, cls.manifest, cls.training_seed
        )
        with np.load(cls.run_one / DESIGN_TRANSITIONS_FILENAME, allow_pickle=False) as data:
            cls.arrays_one = {name: data[name].copy() for name in data.files}
        with np.load(cls.run_two / DESIGN_TRANSITIONS_FILENAME, allow_pickle=False) as data:
            cls.arrays_two = {name: data[name].copy() for name in data.files}
        cls.metadata_one = json.loads(
            (cls.run_one / DESIGN_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        cls.metadata_two = json.loads(
            (cls.run_two / DESIGN_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        cls.design_rows = read_reference_episode_csv(
            cls.run_one / DESIGN_EPISODES_CSV
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @classmethod
    def _run_collection(cls, output_root):
        result = comparison_main(
            [
                "collect-design-dataset",
                "--split",
                "validation",
                "--manifest",
                str(cls.manifest_path),
                "--training-seed",
                str(cls.training_seed),
                "--episodes",
                "2",
                "--checkpoint",
                str(cls.checkpoint),
                "--output-dir",
                str(output_root),
                "--reference-per-episode",
                str(cls.reference_csv),
            ]
        )
        if result != 0:
            raise AssertionError(f"collector command failed with {result}")

    def test_two_episodes_produce_120_ordered_joint_transitions(self):
        arrays = self.arrays_one
        self.assertEqual(set(arrays), set(ARRAY_NAMES))
        self.assertEqual(arrays["state"].shape, (120, 532))
        self.assertEqual(arrays["projected_joint_action"].shape, (120, 48))
        self.assertEqual(arrays["next_state"].shape, (120, 532))
        self.assertEqual(int(arrays["done"].sum()), 2)
        self.assertEqual(self.metadata_one["centralized_actor_calls"], 120)
        np.testing.assert_array_equal(arrays["global_transition_index"], np.arange(120))
        np.testing.assert_array_equal(arrays["episode_index"], np.repeat([0, 1], 60))
        np.testing.assert_array_equal(arrays["movement_step"], np.tile(np.arange(60), 2))
        self.assertTrue(arrays["done"][59])
        self.assertTrue(arrays["done"][119])
        self.assertFalse(arrays["done"][58])
        self.assertFalse(arrays["done"][60])

    def test_reward_reconstructs_at_checkpoint_lambda(self):
        reward = reconstruct_reward(
            self.arrays_one,
            beta_search=self.formal_config["beta_search"],
            beta_vs=self.formal_config["beta_vs"],
            beta_com=self.formal_config["beta_com"],
        )
        np.testing.assert_allclose(
            reward,
            self.arrays_one["reward_at_checkpoint_lambda"],
            rtol=0.0,
            atol=1e-12,
        )
        for terminal in (59, 119):
            self.assertEqual(self.arrays_one["phi_search_t1"][terminal], 0.0)
            self.assertEqual(self.arrays_one["phi_vs_t1"][terminal], 0.0)
            self.assertEqual(self.arrays_one["phi_com_t1"][terminal], 0.0)

    def test_repeated_collection_is_bitwise_deterministic(self):
        for name in ARRAY_NAMES:
            self.assertTrue(
                np.array_equal(self.arrays_one[name], self.arrays_two[name]), name
            )
            self.assertEqual(
                self.metadata_one["arrays"][name]["sha256"],
                self.metadata_two["arrays"][name]["sha256"],
            )
        self.assertEqual(
            self.metadata_one["deterministic_content_fingerprint"],
            self.metadata_two["deterministic_content_fingerprint"],
        )

    def test_collection_is_non_mutating_and_exploration_free(self):
        invariants = self.metadata_one["evaluation_invariants"]
        self.assertTrue(all(invariants.values()))
        fingerprints = self.metadata_one["learning_state_fingerprints"]
        self.assertEqual(fingerprints["before"], fingerprints["after"])
        self.assertEqual(
            self.metadata_one["exploration"],
            {"td3_noise": 0.0, "ddqn_epsilon": 0.0, "ddqn_logits_noise": 0.0},
        )
        self.assertEqual(self.metadata_one["action_perturbation"], "disabled")

    def test_collector_metrics_equal_ordinary_evaluation_and_reference(self):
        validate_reference_metrics(
            read_reference_episode_csv(self.reference_csv), self.ordinary["episode_metrics"]
        )
        validate_reference_metrics(self.design_rows, self.ordinary["episode_metrics"])
        self.assertEqual(
            self.metadata_one["ordinary_evaluation_metrics_fingerprint"],
            self.metadata_one["reference_metrics_fingerprint"],
        )

    def test_tampered_reference_is_rejected_at_first_metric(self):
        reference = read_reference_episode_csv(self.reference_csv)
        reference[0] = dict(reference[0])
        reference[0]["coverage"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "field=coverage"):
            validate_reference_metrics(reference, self.ordinary["episode_metrics"])

    def test_partial_collection_selects_requested_reference_scenarios(self):
        reference = read_reference_episode_csv(self.reference_csv)
        duplicated_full_reference = [*reference, {**reference[1], "scenario_id": "extra"}]
        selected = validate_reference_identity(
            duplicated_full_reference,
            method_id=self.method.method_id,
            training_seed=self.training_seed,
            split="validation",
            evaluation_manifest_hash=self.manifest.content_hash,
            training_manifest_hash="a" * 64,
            checkpoint_completed_episodes=2500,
            checkpoint_fingerprint=reference[0]["checkpoint_metadata_fingerprint"],
            expected_scenario_ids=[reference[0]["scenario_id"]],
        )
        self.assertEqual(selected, reference[:1])

    def test_output_lifecycle_and_collision_protection(self):
        self.assertEqual(read_run_status(self.run_one)["state"], "COMPLETED")
        self.assertTrue((self.run_one / "run_identity.json").is_file())
        with mock.patch("HRL_task_aware.Simulator") as simulator:
            with self.assertRaisesRegex(FileExistsError, "already contains output"):
                self._run_collection(self.output_one)
            simulator.assert_not_called()

    def test_invalid_checkpoint_preflight_creates_no_output_or_simulator(self):
        output = self.root / "invalid-output"
        args = [
            "collect-design-dataset",
            "--manifest",
            str(self.manifest_path),
            "--training-seed",
            str(self.training_seed),
            "--episodes",
            "2",
            "--checkpoint",
            str(self.root / "missing"),
            "--output-dir",
            str(output),
        ]
        with mock.patch("HRL_task_aware.Simulator") as simulator:
            with self.assertRaises(FileNotFoundError):
                comparison_main(args)
            simulator.assert_not_called()
        self.assertFalse(output.exists())

    def test_invalid_manifest_split_creates_no_output_or_simulator(self):
        output = self.root / "invalid-manifest-output"
        manifest_path = self.root / "train-manifest.json"
        generate_manifest("train", 8812, 2).save(manifest_path)
        args = [
            "collect-design-dataset",
            "--split",
            "validation",
            "--manifest",
            str(manifest_path),
            "--training-seed",
            str(self.training_seed),
            "--episodes",
            "2",
            "--checkpoint",
            str(self.checkpoint),
            "--output-dir",
            str(output),
        ]
        with mock.patch("HRL_task_aware.Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "manifest split mismatch"):
                comparison_main(args)
            simulator.assert_not_called()
        self.assertFalse(output.exists())

    def test_atomic_write_failure_leaves_no_complete_artifact(self):
        run_dir = self.root / "write-failure"
        run_dir.mkdir()
        rows = self.ordinary["episode_metrics"]
        with mock.patch("design_dataset.np.savez_compressed", side_effect=OSError("disk")):
            with self.assertRaisesRegex(OSError, "disk"):
                _atomic_write_dataset(
                    run_dir,
                    self.arrays_one,
                    {"schema_version": 1},
                    rows,
                )
        self.assertFalse((run_dir / DESIGN_TRANSITIONS_FILENAME).exists())
        self.assertFalse((run_dir / DESIGN_METADATA_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
