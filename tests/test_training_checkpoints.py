import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from DDQN import DDQN
from HRL_task_aware import (
    _seed_training_rng,
    _validate_resume_config,
    _uses_warmup_random_action,
    formal_training_config,
    parse_training_config,
)
from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from td3 import TD3
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_full_resume_checkpoint,
    load_model_checkpoint,
    save_full_resume_checkpoint,
    save_model_checkpoint,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint


ROUTING_STATE_DIM = 126
ROUTING_ACTION_DIM = 17


def _assert_module_equal(testcase, expected, actual):
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    testcase.assertEqual(expected_state.keys(), actual_state.keys())
    for key in expected_state:
        testcase.assertTrue(torch.equal(expected_state[key], actual_state[key]), key)


class TrainingSeedTest(unittest.TestCase):
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

    def test_same_seed_reproduces_all_td3_and_ddqn_networks(self):
        _seed_training_rng(20260817)
        first_td3, first_ddqn = self._models()
        _seed_training_rng(20260817)
        second_td3, second_ddqn = self._models()

        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_1_target",
            "critic_2",
            "critic_2_target",
        ):
            _assert_module_equal(
                self, getattr(first_td3, name), getattr(second_td3, name)
            )
        for name in (
            "q_network",
            "target_q_network",
            "cost_network",
            "target_cost_network",
        ):
            _assert_module_equal(
                self, getattr(first_ddqn, name), getattr(second_ddqn, name)
            )

    def test_same_seed_reproduces_python_numpy_torch_and_cuda_rng(self):
        def draw_values():
            values = {
                "python": random.random(),
                "numpy": np.random.random(),
                "torch_cpu": torch.rand(4),
            }
            if torch.cuda.is_available():
                values["torch_cuda"] = torch.rand(4, device="cuda")
            return values

        _seed_training_rng(20260817)
        first = draw_values()
        _seed_training_rng(20260817)
        second = draw_values()

        self.assertEqual(first["python"], second["python"])
        self.assertEqual(first["numpy"], second["numpy"])
        self.assertTrue(torch.equal(first["torch_cpu"], second["torch_cpu"]))
        if torch.cuda.is_available():
            self.assertTrue(torch.equal(first["torch_cuda"], second["torch_cuda"]))


class FullResumeCheckpointTest(unittest.TestCase):
    def _components(self):
        td3 = TD3(
            MOVEMENT_STATE_DIM,
            JOINT_ACTION_DIM,
            max_action=1.0,
            gamma=1.0,
            policy_delay=2,
        )
        ddqn = DDQN(ROUTING_STATE_DIM, ROUTING_ACTION_DIM, hidden_dim=16)
        joint = ReplayBufferJoint(
            MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=8
        )
        routing = ReplayBufferDiscrete(
            ROUTING_STATE_DIM,
            ROUTING_ACTION_DIM,
            max_size=8,
            n_step=1,
            gamma=0.99,
        )
        return td3, ddqn, joint, routing

    def _populate_and_train(self, td3, ddqn, joint, routing):
        state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
        state[0] = 1.0
        state[531] = 0.5
        action = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        for index, done in enumerate((False, True)):
            joint.add(
                state,
                action,
                state,
                done=done,
                delivered_mbits=1.0 + index,
                total_mobility_energy=2.0,
                phi_search_t=0.1,
                phi_search_t1=0.2,
                phi_vs_t=0.3,
                phi_vs_t1=0.4,
                phi_com_t=0.5,
                phi_com_t1=0.6,
            )
        td3.update_joint(joint, current_lambda=0.1, batch_size=1)
        td3.update_joint(joint, current_lambda=0.1, batch_size=1)

        routing_state = np.zeros(ROUTING_STATE_DIM, dtype=np.float32)
        routing_state[0] = 1.0
        routing_state[24:41] = 1.0
        routing_state[24] = 0.0
        routing.add(
            routing_state,
            1,
            routing_state,
            reward=3.0,
            cost=0.25,
            done=False,
            tag_gt=4,
        )
        routing.add(
            routing_state,
            2,
            routing_state,
            reward=4.0,
            cost=0.5,
            done=True,
            tag_gt=4,
        )
        ddqn.train(routing, batch_size=1)

    def test_full_resume_round_trip_restores_exact_training_state_and_rng(self):
        td3, ddqn, joint, routing = self._components()
        self._populate_and_train(td3, ddqn, joint, routing)
        calibration = {"seed": 20260817, "c_ref_com": 32.5, "unit": "Mbps"}
        training_state = {
            "completed_episode_index": 6,
            "next_episode_index": 7,
            "lambda_EE_global": 0.123,
            "reward_log": [1.0, 2.0],
            "delivered_log": [3.0, 4.0],
            "energy_log": [5.0, 6.0],
            "lambda_log": [0.2, 0.123],
            "total_joint_transitions": 37,
            "global_routing_slot": 148,
            "td3_post_warmup_transition": 0,
            "ddqn_schedule_slot": 148,
            "td3_noise_log": [0.2],
            "routing_epsilon_log": [1.0, 0.99],
        }
        formal_config = {
            "mode": "train",
            "total_episodes": 100,
            "episode_seconds": 60,
            "warmup_joint_transitions": 1000,
        }

        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(789)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=6,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )

            with np.load(checkpoint_dir / "joint_replay.npz") as saved_joint:
                self.assertEqual(saved_joint["state"].shape[0], joint.size)
            with np.load(checkpoint_dir / "routing_replay.npz") as saved_routing:
                self.assertEqual(saved_routing["state"].shape[0], routing.size)

            expected_python = random.random()
            expected_numpy = np.random.random()
            probe_state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
            expected_action = td3.select_action(probe_state, noise_std=0.1)
            expected_batch = joint.sample(1, current_lambda=0.123, gamma=1.0)
            expected_torch = torch.rand(3)

            restored_td3, restored_ddqn, restored_joint, restored_routing = (
                self._components()
            )
            restored = load_full_resume_checkpoint(
                checkpoint_dir,
                td3=restored_td3,
                ddqn=restored_ddqn,
                joint_replay=restored_joint,
                routing_replay=restored_routing,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )

            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            np.testing.assert_array_equal(
                restored_td3.select_action(probe_state, noise_std=0.1),
                expected_action,
            )
            restored_batch = restored_joint.sample(
                1, current_lambda=0.123, gamma=1.0
            )
            for expected, actual in zip(expected_batch, restored_batch):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))

        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_1_target",
            "critic_2",
            "critic_2_target",
        ):
            _assert_module_equal(
                self, getattr(td3, name), getattr(restored_td3, name)
            )
        for name in (
            "q_network",
            "target_q_network",
            "cost_network",
            "target_cost_network",
        ):
            _assert_module_equal(
                self, getattr(ddqn, name), getattr(restored_ddqn, name)
            )
        self.assertTrue(restored_td3.actor_optimizer.state_dict()["state"])
        self.assertTrue(restored_td3.critic_1_optimizer.state_dict()["state"])
        self.assertTrue(restored_ddqn.optimizer.state_dict()["state"])
        self.assertTrue(restored_ddqn.cost_optimizer.state_dict()["state"])
        self.assertEqual(restored_td3.num_training, td3.num_training)
        self.assertEqual(restored_ddqn.num_training, ddqn.num_training)
        self.assertEqual(restored_td3.gamma, td3.gamma)
        self.assertEqual(restored_td3.policy_delay, td3.policy_delay)
        self.assertEqual(restored_ddqn.eta, ddqn.eta)
        self.assertEqual(restored_ddqn.loss_log, ddqn.loss_log)
        self.assertEqual(restored_ddqn.cost_loss_log, ddqn.cost_loss_log)
        self.assertEqual(restored_joint.ptr, joint.ptr)
        self.assertEqual(restored_joint.size, joint.size)
        self.assertEqual(restored_routing.ptr, routing.ptr)
        self.assertEqual(restored_routing.size, routing.size)
        self.assertEqual(restored_joint.max_size, joint.max_size)
        self.assertEqual(restored_routing.max_size, routing.max_size)
        self.assertEqual(restored_routing.n_step_buffer, routing.n_step_buffer)
        np.testing.assert_array_equal(
            restored_joint.delivered_mbits[: joint.size],
            joint.delivered_mbits[: joint.size],
        )
        np.testing.assert_array_equal(
            restored_routing.cost[: routing.size], routing.cost[: routing.size]
        )
        self.assertEqual(restored["training_state"], training_state)
        self.assertEqual(restored["formal_config"], formal_config)
        self.assertEqual(restored["training_state"]["next_episode_index"], 7)
        self.assertLess(
            restored["training_state"]["total_joint_transitions"],
            formal_config["warmup_joint_transitions"],
        )
        self.assertTrue(_uses_warmup_random_action(37, 1000))
        self.assertFalse(_uses_warmup_random_action(1000, 1000))
        self.assertFalse(_uses_warmup_random_action(1200, 1000))

    def test_model_only_is_complete_for_evaluation_but_refused_for_resume(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "model"
            save_model_checkpoint(
                checkpoint_dir,
                episode=1,
                td3=td3,
                ddqn=ddqn,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )
            metadata = json.loads(
                (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["checkpoint_type"], "model-only")
            models = torch.load(
                checkpoint_dir / "models.pt", weights_only=False
            )
            self.assertEqual(
                set(models["ddqn"]),
                {
                    "q_network",
                    "target_q_network",
                    "cost_network",
                    "target_cost_network",
                },
            )
            original_actor = {
                key: value.detach().clone()
                for key, value in td3.actor.state_dict().items()
            }
            with torch.no_grad():
                for parameter in td3.actor.parameters():
                    parameter.zero_()
            load_model_checkpoint(checkpoint_dir, td3, ddqn)
            for key, expected in original_actor.items():
                self.assertTrue(torch.equal(td3.actor.state_dict()[key], expected))
            with self.assertRaisesRegex(RuntimeError, "evaluation, not exact resume"):
                load_full_resume_checkpoint(
                    checkpoint_dir,
                    td3=td3,
                    ddqn=ddqn,
                    joint_replay=joint,
                    routing_replay=routing,
                    movement_state_dim=MOVEMENT_STATE_DIM,
                    joint_action_dim=JOINT_ACTION_DIM,
                    routing_state_dim=ROUTING_STATE_DIM,
                    calibration=calibration,
                )

    def test_full_resume_rejects_dimension_gamma_schema_and_calibration_mismatch(self):
        td3, ddqn, joint, routing = self._components()
        calibration = {"seed": 1, "c_ref_com": 10.0}
        training_state = {
            "next_episode_index": 1,
            "total_joint_transitions": 0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "full"
            save_full_resume_checkpoint(
                checkpoint_dir,
                episode=0,
                td3=td3,
                ddqn=ddqn,
                joint_replay=joint,
                routing_replay=routing,
                training_state=training_state,
                formal_config={"mode": "train"},
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration=calibration,
            )

            common = {
                "checkpoint_dir": checkpoint_dir,
                "td3": td3,
                "ddqn": ddqn,
                "joint_replay": joint,
                "routing_replay": routing,
                "movement_state_dim": MOVEMENT_STATE_DIM,
                "joint_action_dim": JOINT_ACTION_DIM,
                "routing_state_dim": ROUTING_STATE_DIM,
                "calibration": calibration,
            }
            for field, value in (
                ("movement_state_dim", 531),
                ("joint_action_dim", JOINT_ACTION_DIM - 1),
                ("routing_state_dim", 122),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, field
                ):
                    load_full_resume_checkpoint(**(common | {field: value}))

            wrong_td3 = TD3(
                MOVEMENT_STATE_DIM,
                JOINT_ACTION_DIM,
                max_action=1.0,
                gamma=0.99,
            )
            with self.assertRaisesRegex(RuntimeError, "centralized_td3_gamma"):
                load_full_resume_checkpoint(**(common | {"td3": wrong_td3}))
            with self.assertRaisesRegex(RuntimeError, "calibration fingerprint"):
                load_full_resume_checkpoint(
                    **(common | {"calibration": {"different": True}})
                )

            metadata_path = checkpoint_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_schema_version"):
                load_full_resume_checkpoint(**common)


class TrainingCliTest(unittest.TestCase):
    def test_formal_mode_requires_explicit_episodes(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "train", "--seed", "20260817"]
                )
        self.assertIn("formal train mode requires --episodes", stderr.getvalue())

    def test_formal_mode_requires_explicit_seed(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "train", "--episodes", "1500"]
                )
        self.assertIn("formal train mode requires --seed", stderr.getvalue())

    def test_cli_builds_smoke_and_formal_configs(self):
        smoke = parse_training_config(["--mode", "smoke"])
        self.assertEqual(smoke.total_episodes, 1)
        self.assertEqual(smoke.random_seed, 20260817)
        self.assertFalse(smoke.enable_full_resume)
        formal = parse_training_config(
            [
                "--mode",
                "train",
                "--episodes",
                "1500",
                "--seed",
                "20260817",
            ]
        )
        self.assertEqual(formal.mode, "train")
        self.assertEqual(formal.total_episodes, 1500)
        self.assertEqual(formal.random_seed, 20260817)
        self.assertEqual(formal.warmup_joint_transitions, 1000)
        self.assertEqual(formal.model_checkpoint_every, 2)
        self.assertEqual(formal.full_resume_every, 50)

    def test_smoke_mode_rejects_seed_override(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_training_config(
                    ["--mode", "smoke", "--seed", "123"]
                )
        self.assertIn(
            "smoke mode fixes --seed to 20260817; do not pass --seed",
            stderr.getvalue(),
        )

    def test_resume_config_requires_matching_seed(self):
        stored = vars(
            formal_training_config(1500, random_seed=20260817)
        ).copy()
        matching = formal_training_config(1500, random_seed=20260817)
        _validate_resume_config(stored, matching)

        different = formal_training_config(1500, random_seed=123)
        with self.assertRaisesRegex(RuntimeError, "random_seed"):
            _validate_resume_config(stored, different)

    def test_checkpoint_schema_is_explicit(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
