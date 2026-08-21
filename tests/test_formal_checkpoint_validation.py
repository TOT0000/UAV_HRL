from copy import deepcopy
from dataclasses import asdict
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
from experiment_config import MethodSpec
from scenario_manifest import generate_manifest
from td3 import TD3
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    calibration_fingerprint,
    load_model_checkpoint,
    save_model_checkpoint,
    validate_model_checkpoint_metadata,
)


ROUTING_STATE_DIM = 126
ROUTING_ACTION_DIM = 17


class FormalCheckpointMetadataTest(unittest.TestCase):
    def setUp(self):
        self.method = MethodSpec()
        self.training_seed = 17
        self.calibration = {"seed": 8, "c_ref_com": 12.5}
        self.formal_config = asdict(
            formal_training_config(2500, random_seed=self.training_seed)
        )
        dinkelbach_state = DinkelbachBlockState.from_config(self.formal_config)
        for _ in range(2500):
            dinkelbach_state.record_episode(1.0, 2.0)
        self.metadata = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 2499,
            "movement_state_dim": MOVEMENT_STATE_DIM,
            "joint_action_dim": JOINT_ACTION_DIM,
            "routing_state_dim": ROUTING_STATE_DIM,
            "centralized_td3_gamma": 1.0,
            "routing_ddqn_gamma": 0.99,
            "com_calibration_fingerprint": calibration_fingerprint(
                self.calibration
            ),
            "experiment": {
                "method_spec_fingerprint": self.method.fingerprint,
                "training_seed": self.training_seed,
                "manifest_hash": "training-manifest",
                "formal_config": self.formal_config,
                **dinkelbach_config_metadata(self.formal_config),
                "lambda_ee": dinkelbach_state.current_lambda,
                "dinkelbach_state": dinkelbach_state.training_state(),
            },
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
            expected_completed_episodes=2500,
            expected_formal_config=self.formal_config,
        )

    def test_synthetic_2500_episode_metadata_is_accepted(self):
        validated = self._validate()

        self.assertIs(validated, self.metadata)
        self.assertEqual(validated["episode"] + 1, 2500)

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
        formal_config = asdict(
            formal_training_config(2500, random_seed=training_seed)
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
                    "method_spec_fingerprint": method.fingerprint,
                    "training_seed": training_seed,
                    "manifest_hash": "training-manifest",
                    "formal_config": formal_config,
                    **dinkelbach_config_metadata(formal_config),
                    "lambda_ee": dinkelbach_state.current_lambda,
                    "dinkelbach_state": dinkelbach_state.training_state(),
                },
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
