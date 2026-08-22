import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from DDQN import DDQN
from centralized_movement import (
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    MOVEMENT_STATE_DIM,
    movement_mask_from_state,
    project_joint_action,
)
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import MethodSpec, effective_training_config
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    _full_training_state,
    formal_training_config,
)
from observation_strategy import apply_observation_strategy
from td3 import TD3
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION,
    load_full_resume_checkpoint,
    load_model_checkpoint,
    save_full_resume_checkpoint,
    save_model_checkpoint,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint
from fov_ema_fixtures import initialized_fov_ema_state


class MovementMaskCheckpointTest(unittest.TestCase):
    calibration = {"fixture": "movement-projection-mask"}

    @staticmethod
    def _components():
        return (
            TD3(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, 1.0, gamma=1.0),
            DDQN(ROUTING_STATE_DIM, 17, hidden_dim=16),
            ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=4),
            ReplayBufferDiscrete(
                ROUTING_STATE_DIM, 17, max_size=4, n_step=1, gamma=0.99
            ),
        )

    def _save_checkpoint(self, root, method_key):
        method = MethodSpec.parse(method_key)
        config = formal_training_config(
            1,
            random_seed=20260817,
            enable_plots=False,
            enable_csv=False,
        )
        formal_config = effective_training_config(config, method)
        movement, routing_agent, joint, routing = self._components()
        physical_state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
        physical_next_state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
        for uav_id in (0, 4):
            physical_state[uav_id * LOCAL_MOVEMENT_DIM] = 1.0
        for uav_id in (1, 4, 9):
            physical_next_state[uav_id * LOCAL_MOVEMENT_DIM + 1] = 1.0
        current_mask = movement_mask_from_state(physical_state)
        next_mask = movement_mask_from_state(physical_next_state)
        state = apply_observation_strategy(
            physical_state, method.task_observation, "movement"
        )
        next_state = apply_observation_strategy(
            physical_next_state, method.task_observation, "movement"
        )
        action = project_joint_action(
            np.full(JOINT_ACTION_DIM, 0.25, dtype=np.float32),
            movement_mask=current_mask,
        )
        joint.add(
            state,
            action,
            next_state,
            done=True,
            delivered_mbits=1.0,
            total_mobility_energy=2.0,
            phi_search_t=0.0,
            phi_search_t1=0.0,
            phi_vs_t=0.0,
            phi_vs_t1=0.0,
            phi_com_t=0.0,
            phi_com_t1=0.0,
            current_movement_mask=current_mask,
            next_movement_mask=next_mask,
        )
        dinkelbach = DinkelbachBlockState.from_config(config)
        event = dinkelbach.record_episode(1.0, 2.0)
        training_state = _full_training_state(
            episode=0,
            dinkelbach_state=dinkelbach,
            reward_log=[0.0],
            delivered_log=[1.0],
            energy_log=[2.0],
            lambda_used_log=[event["dinkelbach_lambda_used"]],
            lambda_after_episode_log=[event["dinkelbach_lambda_after_episode"]],
            total_joint_transitions=1,
            routing_slots_executed=4,
            td3_noise_log=[],
            routing_epsilon_log=[1.0] * 4,
            warmup_joint_transitions=config.warmup_joint_transitions,
            training_history_rows=[],
            fov_ema_state=initialized_fov_ema_state(),
        )
        checkpoint = Path(root) / method_key / "ep_0001"
        experiment = {
            "method_id": method.method_id,
            "method_spec": method.to_dict(),
            "method_spec_fingerprint": method.fingerprint,
            "training_seed": config.random_seed,
            "formal_config": formal_config,
            "dinkelbach_state": dinkelbach.training_state(),
            "lambda_ee": dinkelbach.current_lambda,
            **dinkelbach_config_metadata(config),
        }
        save_full_resume_checkpoint(
            checkpoint,
            episode=0,
            td3=movement,
            ddqn=routing_agent,
            joint_replay=joint,
            routing_replay=routing,
            training_state=training_state,
            formal_config=formal_config,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=self.calibration,
            experiment_metadata=experiment,
        )
        return {
            "checkpoint": checkpoint,
            "method": method,
            "formal_config": formal_config,
            "current_mask": current_mask,
            "next_mask": next_mask,
            "movement": movement,
        }

    @staticmethod
    def _downgrade_to_pre_mask_schema(checkpoint):
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_schema_version"] = (
            PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        replay_path = checkpoint / "joint_replay.npz"
        with np.load(replay_path, allow_pickle=False) as saved:
            arrays = {
                key: saved[key]
                for key in saved.files
                if key
                not in {
                    "current_movement_mask",
                    "next_movement_mask",
                    "movement_mask_valid",
                }
            }
        np.savez_compressed(replay_path, **arrays)

    def _load(self, saved):
        movement, routing_agent, joint, routing = self._components()
        result = load_full_resume_checkpoint(
            saved["checkpoint"],
            td3=movement,
            ddqn=routing_agent,
            joint_replay=joint,
            routing_replay=routing,
            movement_state_dim=MOVEMENT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            routing_state_dim=ROUTING_STATE_DIM,
            calibration=self.calibration,
            expected_experiment_metadata={
                "method_spec_fingerprint": saved["method"].compatible_fingerprints,
            },
            expected_formal_config=saved["formal_config"],
        )
        return movement, joint, result

    def test_current_schema_round_trip_preserves_masks_value_for_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saved = self._save_checkpoint(temp_dir, "td3_dinkelbach_wo_ta")
            metadata = json.loads(
                (saved["checkpoint"] / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION)
            with np.load(saved["checkpoint"] / "joint_replay.npz") as replay:
                np.testing.assert_array_equal(
                    replay["current_movement_mask"][0], saved["current_mask"]
                )
                np.testing.assert_array_equal(
                    replay["next_movement_mask"][0], saved["next_mask"]
                )
                self.assertTrue(replay["movement_mask_valid"].all())
            _, restored, _ = self._load(saved)
        np.testing.assert_array_equal(
            restored.current_movement_mask[0], saved["current_mask"]
        )
        np.testing.assert_array_equal(
            restored.next_movement_mask[0], saved["next_mask"]
        )
        self.assertTrue(restored.movement_mask_valid[0, 0])

    def test_pre_adaptive_safe_checkpoint_is_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saved = self._save_checkpoint(temp_dir, "td3_dinkelbach")
            self._downgrade_to_pre_mask_schema(saved["checkpoint"])
            with self.assertRaisesRegex(RuntimeError, "legacy safe-DDQN"):
                self._load(saved)

    def test_legacy_masked_checkpoint_is_rejected_before_network_restore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saved = self._save_checkpoint(temp_dir, "td3_dinkelbach_wo_ta")
            self._downgrade_to_pre_mask_schema(saved["checkpoint"])
            movement, routing_agent, joint, routing = self._components()
            actor_before = {
                key: value.detach().clone()
                for key, value in movement.actor.state_dict().items()
            }
            with self.assertRaisesRegex(
                RuntimeError,
                "legacy safe-DDQN",
            ):
                load_full_resume_checkpoint(
                    saved["checkpoint"],
                    td3=movement,
                    ddqn=routing_agent,
                    joint_replay=joint,
                    routing_replay=routing,
                    movement_state_dim=MOVEMENT_STATE_DIM,
                    joint_action_dim=JOINT_ACTION_DIM,
                    routing_state_dim=ROUTING_STATE_DIM,
                    calibration=self.calibration,
                    expected_experiment_metadata={
                        "method_spec_fingerprint": saved[
                            "method"
                        ].compatible_fingerprints,
                    },
                    expected_formal_config=saved["formal_config"],
                )
            for key, expected in actor_before.items():
                self.assertTrue(torch.equal(movement.actor.state_dict()[key], expected))
            self.assertEqual(joint.size, 0)

    def test_pre_adaptive_model_only_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            movement, routing_agent, _, _ = self._components()
            checkpoint = Path(temp_dir) / "model" / "ep_0001"
            save_model_checkpoint(
                checkpoint,
                episode=0,
                td3=movement,
                ddqn=routing_agent,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=self.calibration,
            )
            metadata_path = checkpoint / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema_version"] = (
                PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            restored_movement, restored_routing, _, _ = self._components()
            with self.assertRaisesRegex(RuntimeError, "legacy safe-DDQN"):
                load_model_checkpoint(
                    checkpoint,
                    restored_movement,
                    restored_routing,
                    movement_state_dim=MOVEMENT_STATE_DIM,
                    joint_action_dim=JOINT_ACTION_DIM,
                    routing_state_dim=ROUTING_STATE_DIM,
                    calibration=self.calibration,
                    expected_completed_episodes=1,
                )


if __name__ == "__main__":
    unittest.main()
