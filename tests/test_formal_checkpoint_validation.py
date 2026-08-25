from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from DDQN import DDQN
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from HRL_task_aware import formal_training_config
from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from com_capacity_calibration import load_com_capacity_reference
from comparison_experiment import main as comparison_main
from experiment_config import MethodSpec, effective_training_config
from routing_lifecycle import RoutingLearnerLifecycle
from scenario_manifest import extend_training_manifest, generate_manifest
from td3 import TD3
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    calibration_fingerprint,
    load_model_checkpoint,
    save_model_checkpoint,
    checkpoint_run_compatibility_from_metadata,
    validate_model_checkpoint_metadata,
)


ROUTING_STATE_DIM = 90
ROUTING_ACTION_DIM = 11


class FormalCheckpointMetadataTest(unittest.TestCase):
    def setUp(self):
        self.method = MethodSpec()
        self.training_seed = 17
        self.calibration = {"seed": 8, "c_ref_com": 12.5}
        self.formal_config = effective_training_config(
            formal_training_config(1500, random_seed=self.training_seed),
            self.method,
        )
        dinkelbach_state = DinkelbachBlockState.from_config(self.formal_config)
        for _ in range(1500):
            dinkelbach_state.record_episode(1.0, 2.0)
        self.metadata = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 1499,
            "movement_state_dim": MOVEMENT_STATE_DIM,
            "joint_action_dim": JOINT_ACTION_DIM,
            "routing_state_dim": ROUTING_STATE_DIM,
            "movement_agent_kind": "td3",
            "movement_agent_gamma": 1.0,
            "movement_agent_configuration": deepcopy(
                self.formal_config["movement_agent_configuration"]
            ),
            "centralized_td3_gamma": 1.0,
            "routing_ddqn_gamma": 0.99,
            "routing_agent_kind": "safe_ddqn",
            "routing_agent_configuration": {
                "lambda_cost": 0.0,
                "initial_lambda_cost": 0.0,
                "eta_c": 0.01,
                "qos_target_probability": 0.1,
                "lambda_update_scope": "episode_end",
                "cost_denominator": "eligible_packets",
                "mid_episode_checkpoint_supported": False,
            },
            "com_calibration_fingerprint": calibration_fingerprint(
                self.calibration
            ),
            "experiment": {
                "method_id": self.method.method_id,
                "method_spec": self.method.to_dict(),
                "method_spec_fingerprint": self.method.fingerprint,
                "training_seed": self.training_seed,
                "git_sha": "synthetic-training-sha",
                "manifest_hash": "training-manifest",
                "formal_config": self.formal_config,
                **dinkelbach_config_metadata(self.formal_config),
                "lambda_ee": dinkelbach_state.current_lambda,
                "dinkelbach_state": dinkelbach_state.training_state(),
            },
        }
        lifecycle = RoutingLearnerLifecycle(
            global_slot_count=240000,
            optimizer_update_count=17,
            target_update_count=17,
            epsilon_decay_start_slot=63,
            last_optimizer_update_slot=240000,
        ).state_dict()
        resolved = deepcopy(self.formal_config)
        resolved.update(
            {
                "method_key": self.method.method_id,
                "method_id": self.method.method_id,
                "method_spec": self.method.to_dict(),
                "method_spec_fingerprint": self.method.fingerprint,
                "training_episode_count": 1500,
                "training_seed": self.training_seed,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            }
        )
        self.metadata["training_provenance"] = {
            "training_episode_count": 1500,
            "training_git_sha": "synthetic-training-sha",
            "resolved_training_config": resolved,
            "routing_lifecycle": lifecycle,
            "safe_ddqn_constraint_state": deepcopy(
                self.metadata["routing_agent_configuration"]
            ),
            "provenance_complete": True,
        }

    def _validate(self, metadata=None):
        return validate_model_checkpoint_metadata(
            metadata or self.metadata,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            td3_gamma=1.0,
            ddqn_gamma=0.99,
            calibration=self.calibration,
            expected_experiment_metadata={
                "method_spec_fingerprint": self.method.fingerprint,
                "training_seed": self.training_seed,
            },
            expected_completed_episodes=1500,
            expected_formal_config=self.formal_config,
        )

    def _metadata_at_episode(self, completed_episodes, formal_config=None):
        formal_config = deepcopy(formal_config or self.formal_config)
        metadata = deepcopy(self.metadata)
        metadata["episode"] = int(completed_episodes) - 1
        metadata["experiment"]["formal_config"] = formal_config
        metadata["movement_agent_configuration"] = deepcopy(
            formal_config["movement_agent_configuration"]
        )
        state = DinkelbachBlockState.from_config(formal_config)
        for _ in range(int(completed_episodes)):
            state.record_episode(1.0, 2.0)
        metadata["experiment"].update(
            lambda_ee=state.current_lambda,
            dinkelbach_state=state.training_state(),
            **dinkelbach_config_metadata(formal_config),
        )
        resolved = metadata["training_provenance"]["resolved_training_config"]
        resolved.update(formal_config)
        resolved["training_episode_count"] = int(completed_episodes)
        metadata["training_provenance"]["training_episode_count"] = int(
            completed_episodes
        )
        return metadata

    def test_synthetic_1500_episode_metadata_is_accepted(self):
        validated = self._validate()

        self.assertIs(validated, self.metadata)
        self.assertEqual(validated["episode"] + 1, 1500)

    def test_1500_checkpoint_metadata_accepts_only_monotonic_3000_horizon(self):
        previous_manifest = generate_manifest("train", self.training_seed, 1500)
        extended_manifest, _ = extend_training_manifest(previous_manifest, 3000)
        current_config = deepcopy(self.formal_config)
        current_config["total_episodes"] = 3000

        for completed in (1000, 1500):
            with self.subTest(completed=completed):
                metadata = self._metadata_at_episode(completed)
                metadata["experiment"]["manifest_hash"] = (
                    previous_manifest.content_hash
                )
                validated = validate_model_checkpoint_metadata(
                    metadata,
                    expected_experiment_metadata={
                        "method_spec_fingerprint": self.method.fingerprint,
                        "training_seed": self.training_seed,
                        "manifest_hash": extended_manifest.content_hash,
                    },
                    expected_completed_episodes=completed,
                    expected_formal_config=current_config,
                    current_training_manifest=extended_manifest,
                )
                self.assertIs(validated, metadata)
                compatibility = checkpoint_run_compatibility_from_metadata(
                    metadata,
                    current_config,
                    current_training_manifest=extended_manifest,
                )
                self.assertTrue(
                    compatibility["horizon_extension_compatible"]
                )
                self.assertEqual(
                    compatibility["allowed_horizon_differences"],
                    ["total_episodes"],
                )
                self.assertTrue(compatibility["manifest_prefix_compatible"])
                self.assertEqual(
                    compatibility["checkpoint_planned_total_episodes"], 1500
                )
                self.assertEqual(
                    compatibility["current_training_run_total_episodes"], 3000
                )

    def test_native_3000_checkpoint_has_no_extension_difference(self):
        native_config = effective_training_config(
            formal_training_config(3000, random_seed=self.training_seed),
            self.method,
        )
        native_manifest = generate_manifest("train", self.training_seed, 3000)
        metadata = self._metadata_at_episode(1000, native_config)
        metadata["experiment"]["manifest_hash"] = native_manifest.content_hash
        compatibility = checkpoint_run_compatibility_from_metadata(
            metadata,
            native_config,
            current_training_manifest=native_manifest,
        )
        self.assertFalse(compatibility["horizon_extension_compatible"])
        self.assertEqual(compatibility["allowed_horizon_differences"], [])
        self.assertTrue(compatibility["manifest_prefix_compatible"])

    def test_horizon_policy_rejects_every_non_administrative_change(self):
        previous_manifest = generate_manifest("train", self.training_seed, 1500)
        extended_manifest, _ = extend_training_manifest(previous_manifest, 3000)
        current_config = deepcopy(self.formal_config)
        current_config["total_episodes"] = 3000
        cases = {
            "reward": ("movement_objective", "wrong-reward"),
            "energy": ("propulsion_model_id", "wrong-energy"),
            "qos": ("packet_qos_contract_version", "wrong-qos"),
            "warmup": ("warmup_joint_transitions", 999),
            "exploration": ("movement_exploration_decay_episodes", 999),
        }
        for label, (field, value) in cases.items():
            with self.subTest(label=label):
                metadata = self._metadata_at_episode(1500)
                metadata["experiment"]["manifest_hash"] = (
                    previous_manifest.content_hash
                )
                metadata["experiment"]["formal_config"][field] = value
                with self.assertRaisesRegex(RuntimeError, field):
                    validate_model_checkpoint_metadata(
                        metadata,
                        expected_completed_episodes=1500,
                        expected_formal_config=current_config,
                        current_training_manifest=extended_manifest,
                    )

        for key, value in (
            ("training_seed", self.training_seed + 1),
            ("method_spec_fingerprint", "wrong-method"),
        ):
            metadata = self._metadata_at_episode(1500)
            metadata["experiment"]["manifest_hash"] = previous_manifest.content_hash
            with self.assertRaisesRegex(RuntimeError, key):
                validate_model_checkpoint_metadata(
                    metadata,
                    expected_experiment_metadata={key: value},
                    expected_completed_episodes=1500,
                    expected_formal_config=current_config,
                    current_training_manifest=extended_manifest,
                )

    def test_training_and_evaluation_manifest_hashes_may_differ(self):
        metadata = deepcopy(self.metadata)
        metadata["experiment"]["manifest_hash"] = "train-hash"

        self._validate(metadata)

    def test_dimension_gamma_calibration_seed_and_method_mismatches_fail(self):
        cases = (
            (("movement_state_dim",), 531, "movement_state_dim"),
            (("joint_action_dim",), 47, "joint_action_dim"),
            (("routing_state_dim",), 125, "routing_state_dim"),
            (("centralized_td3_gamma",), 0.9, "centralized_td3_gamma"),
            (("routing_ddqn_gamma",), 0.9, "routing_ddqn_gamma"),
            (("com_calibration_fingerprint",), "wrong", "calibration"),
            (("experiment", "training_seed"), 18, "training_seed"),
            (
                ("experiment", "method_spec_fingerprint"),
                "wrong",
                "method_spec_fingerprint",
            ),
        )
        for path, value, message in cases:
            with self.subTest(path=path):
                metadata = deepcopy(self.metadata)
                target = metadata
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    self._validate(metadata)

    def test_completed_episode_and_formal_core_config_mismatches_fail(self):
        too_short = deepcopy(self.metadata)
        too_short["episode"] = 0
        with self.assertRaisesRegex(RuntimeError, "completed training episodes"):
            self._validate(too_short)

        wrong_config = deepcopy(self.metadata)
        wrong_config["experiment"]["formal_config"]["batch_size"] = 32
        with self.assertRaisesRegex(RuntimeError, "batch_size"):
            self._validate(wrong_config)

    def test_dinkelbach_state_and_provenance_mismatches_fail(self):
        missing_state = deepcopy(self.metadata)
        del missing_state["experiment"]["dinkelbach_state"]
        with self.assertRaisesRegex(RuntimeError, "missing Dinkelbach block state"):
            self._validate(missing_state)

        wrong_lambda = deepcopy(self.metadata)
        wrong_lambda["experiment"]["lambda_ee"] = 123.0
        with self.assertRaisesRegex(RuntimeError, "lambda metadata"):
            self._validate(wrong_lambda)

        wrong_provenance = deepcopy(self.metadata)
        wrong_provenance["experiment"][
            "dinkelbach_update_interval_episodes"
        ] = 25
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            self._validate(wrong_provenance)

    def test_old_checkpoint_schema_is_rejected(self):
        old = deepcopy(self.metadata)
        old["checkpoint_schema_version"] = 1
        with self.assertRaisesRegex(RuntimeError, "checkpoint_schema_version"):
            self._validate(old)


class FormalCheckpointLoadOrderTest(unittest.TestCase):
    def _models(self):
        return (
            TD3(
                MOVEMENT_STATE_DIM,
                JOINT_ACTION_DIM,
                max_action=1.0,
                gamma=1.0,
                policy_delay=2,
            ),
            DDQN(ROUTING_STATE_DIM, ROUTING_ACTION_DIM),
        )

    def test_invalid_metadata_is_rejected_before_torch_load_or_mutation(self):
        td3, ddqn = self._models()
        actor_before = {
            key: value.detach().clone()
            for key, value in td3.actor.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "checkpoint_type": MODEL_CHECKPOINT_TYPE,
                        "movement_state_dim": 531,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("training_checkpoint.torch.load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "movement_state_dim"):
                    load_model_checkpoint(
                        checkpoint_dir,
                        td3,
                        ddqn,
                        movement_state_dim=MOVEMENT_STATE_DIM,
                    )
                torch_load.assert_not_called()

        for key, expected in actor_before.items():
            self.assertTrue(torch.equal(td3.actor.state_dict()[key], expected))

    def test_formal_cli_rejects_one_episode_checkpoint_before_weight_load(self):
        training_seed = 41
        method = MethodSpec()
        _, calibration = load_com_capacity_reference()
        td3, ddqn = self._models()
        formal_config = effective_training_config(
            formal_training_config(1500, random_seed=training_seed), method
        )
        dinkelbach_state = DinkelbachBlockState.from_config(formal_config)
        dinkelbach_state.record_episode(1.0, 2.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoint"
            save_model_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
                experiment_metadata={
                    "method_id": method.method_id,
                    "method_spec": method.to_dict(),
                    "method_spec_fingerprint": method.fingerprint,
                    "training_seed": training_seed,
                    "git_sha": "fixture-training-sha",
                    "manifest_hash": "training-manifest",
                    "formal_config": formal_config,
                    **dinkelbach_config_metadata(formal_config),
                    "lambda_ee": dinkelbach_state.current_lambda,
                    "dinkelbach_state": dinkelbach_state.training_state(),
                },
                routing_lifecycle_state=RoutingLearnerLifecycle().state_dict(),
            )
            manifest_path = root / "validation.json"
            generate_manifest("validation", 902, 1).save(manifest_path)

            with mock.patch("training_checkpoint.torch.load") as torch_load:
                with self.assertRaisesRegex(
                    RuntimeError, "completed training episodes"
                ):
                    comparison_main(
                        [
                            "evaluate",
                            "--split",
                            "validation",
                            "--manifest",
                            str(manifest_path),
                            "--training-seed",
                            str(training_seed),
                            "--episodes",
                            "1",
                            "--checkpoint",
                            str(checkpoint_dir),
                            "--output-dir",
                            str(root / "output"),
                        ]
                    )
                torch_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
