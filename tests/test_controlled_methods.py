from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from centralized_ddpg import CentralizedDDPG
from centralized_movement import JOINT_ACTION_DIM, MOVEMENT_STATE_DIM
from DDQN import DDQN
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    METHOD_REGISTRY,
    MethodSpec,
    NUM_UAV,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    TrainingConfig,
    _full_training_state,
    _interval_reward,
    formal_training_config,
    train,
)
from run_experiment import build_parser, create_unique_run_directory
from scenario_manifest import generate_manifest
from Simulator import Simulator
from training_checkpoint import (
    load_full_resume_checkpoint,
    save_full_resume_checkpoint,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint


EXPECTED_METHODS = (
    "td3_dinkelbach",
    "ddpg_dinkelbach",
    "td3_ratio",
    "ddpg_ratio",
    "random_action",
    "td3_dinkelbach_no_task_potential",
    "ddpg_dinkelbach_no_task_potential",
)


class ControlledMethodRegistryTest(unittest.TestCase):
    def test_registry_contains_and_parses_exactly_seven_methods(self):
        self.assertEqual(tuple(METHOD_REGISTRY), EXPECTED_METHODS)
        self.assertEqual(
            [MethodSpec.parse(key).method_key for key in EXPECTED_METHODS],
            list(EXPECTED_METHODS),
        )
        self.assertEqual(MethodSpec.parse("random_action").label, "Random selected")

    def test_shared_environment_contract_is_kkm_16_uav_and_roi_2_through_8(self):
        for key in EXPECTED_METHODS:
            spec = MethodSpec.parse(key)
            self.assertEqual(spec.assignment, "current_k_km")
            self.assertEqual(spec.routing, "safe_ddqn")
        self.assertEqual(NUM_UAV, 16)
        self.assertEqual((ROI_COUNT_MIN, ROI_COUNT_MAX), (2, 8))
        environment = Simulator(num_UAV=NUM_UAV)
        environment.num_GT = ROI_COUNT_MAX + 1
        with self.assertRaisesRegex(ValueError, r"\[2, 8\]"):
            environment.reset_environment()

    def test_runner_accepts_exactly_one_positional_method(self):
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["ddpg_ratio", "--smoke"]).method.method_key,
            "ddpg_ratio",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["td3_ratio", "ddpg_ratio"])

    def test_formal_defaults_are_centralized(self):
        self.assertEqual(FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"], 2500)
        self.assertEqual(FORMAL_EXPERIMENT_DEFAULTS["formal_checkpoint_episode"], 2500)
        self.assertEqual(FORMAL_EXPERIMENT_DEFAULTS["training_seed_count"], 1)
        self.assertEqual(
            FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["replay_size"],
            200_000,
        )

    def test_one_transition_smoke_uses_shared_flow_for_all_methods(self):
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)
        for key in EXPECTED_METHODS:
            with self.subTest(method=key):
                spec = MethodSpec.parse(key)
                result = train(
                    TrainingConfig(
                        total_episodes=1,
                        mode="smoke",
                        episode_seconds=1,
                        warmup_joint_transitions=0,
                        batch_size=1,
                        enable_model_checkpoints=False,
                        enable_full_resume=False,
                        enable_plots=False,
                        enable_csv=False,
                        random_seed=20260817,
                    ),
                    scenario_manifest=manifest,
                    method_spec=spec,
                )
                self.assertEqual(result["movement_agent_kind"], spec.agent)
                self.assertEqual(result["movement_state_dim"], 532)
                self.assertEqual(result["joint_action_dim"], 48)
                self.assertEqual(result["proposal_batches"], 1)
                if spec.agent == "random":
                    self.assertEqual(result["joint_replay_size"], 0)
                    self.assertEqual(result["critic_updates"], 0)
                    self.assertEqual(result["actor_updates"], 0)
                if spec.reward_mode == "ratio":
                    self.assertEqual(result["dinkelbach_update_count"], 0)
                    self.assertIsNone(result["lambda"])


class ControlledRewardTest(unittest.TestCase):
    def setUp(self):
        self.config = formal_training_config(1)
        self.potential_t = (0.2, 0.3, 0.4)
        self.potential_t1 = (0.5, 0.6, 0.7)

    def test_task_potential_flag_changes_only_shaping_term(self):
        shaped = _interval_reward(
            4.0, 2.0, 0.25, 1.0, self.potential_t,
            self.potential_t1, False, self.config,
            reward_mode="dinkelbach", task_potential_enabled=True,
        )
        unshaped = _interval_reward(
            4.0, 2.0, 0.25, 1.0, self.potential_t,
            self.potential_t1, False, self.config,
            reward_mode="dinkelbach", task_potential_enabled=False,
        )
        expected_shaping = sum(b - a for a, b in zip(self.potential_t, self.potential_t1))
        self.assertAlmostEqual(shaped - unshaped, expected_shaping)
        self.assertAlmostEqual(unshaped, 3.5)

    def test_ratio_reward_is_safe_and_does_not_use_lambda(self):
        first = _interval_reward(
            4.0, 2.0, 0.0, 1.0, self.potential_t, self.potential_t1,
            False, self.config, reward_mode="ratio", task_potential_enabled=False,
        )
        second = _interval_reward(
            4.0, 2.0, 999.0, 1.0, self.potential_t, self.potential_t1,
            False, self.config, reward_mode="ratio", task_potential_enabled=False,
        )
        zero_energy = _interval_reward(
            4.0, 0.0, 999.0, 1.0, self.potential_t, self.potential_t1,
            False, self.config, reward_mode="ratio", task_potential_enabled=False,
        )
        self.assertEqual((first, second, zero_energy), (2.0, 2.0, 0.0))


class ControlledDDPGCheckpointTest(unittest.TestCase):
    def _agent(self):
        return CentralizedDDPG(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, 1.0)

    def test_ddpg_is_single_critic_with_controlled_hyperparameters(self):
        agent = self._agent()
        self.assertFalse(hasattr(agent, "critic_2"))
        self.assertEqual(agent.gamma, 1.0)
        self.assertEqual(agent.tau, 0.005)
        self.assertEqual(agent.actor_optimizer.param_groups[0]["lr"], 6e-5)
        self.assertEqual(agent.critic_optimizer.param_groups[0]["lr"], 2e-4)
        actor_linears = [module for module in agent.actor.modules() if isinstance(module, torch.nn.Linear)]
        critic_linears = [module for module in agent.critic.modules() if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(actor_linears), 5)
        self.assertEqual(len(critic_linears), 5)

    def test_ddpg_full_checkpoint_round_trip_has_no_second_critic(self):
        config = formal_training_config(1, enable_plots=False, enable_csv=False)
        formal_config = asdict(config)
        agent = self._agent()
        ddqn = DDQN(ROUTING_STATE_DIM, NUM_UAV + 1)
        joint = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=8)
        routing = ReplayBufferDiscrete(
            ROUTING_STATE_DIM, NUM_UAV + 1, max_size=8, n_step=1, gamma=0.99
        )
        dinkelbach = DinkelbachBlockState.from_config(config)
        event = dinkelbach.record_episode(1.0, 2.0)
        training_state = _full_training_state(
            episode=0,
            dinkelbach_state=dinkelbach,
            reward_log=[0.0], delivered_log=[1.0], energy_log=[2.0],
            lambda_used_log=[event["dinkelbach_lambda_used"]],
            lambda_after_episode_log=[event["dinkelbach_lambda_after_episode"]],
            total_joint_transitions=1, routing_slots_executed=4,
            td3_noise_log=[], routing_epsilon_log=[1.0] * 4,
            warmup_joint_transitions=config.warmup_joint_transitions,
            training_history_rows=[],
        )
        expected_actor = {
            key: value.detach().cpu().clone()
            for key, value in agent.actor.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ep_0001"
            save_full_resume_checkpoint(
                checkpoint, episode=0, td3=agent, ddqn=ddqn,
                joint_replay=joint, routing_replay=routing,
                training_state=training_state, formal_config=formal_config,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration={"fixture": "controlled"},
                experiment_metadata={
                    **dinkelbach_config_metadata(config),
                    "formal_config": formal_config,
                    "dinkelbach_state": dinkelbach.training_state(),
                    "lambda_ee": dinkelbach.current_lambda,
                },
            )
            payload = torch.load(
                checkpoint / "training_state.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["networks"]["movement_agent"]["kind"], "ddpg")
            self.assertNotIn("critic_2", payload["networks"]["movement_agent"])

            restored = self._agent()
            restored_ddqn = DDQN(ROUTING_STATE_DIM, NUM_UAV + 1)
            restored_joint = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=8)
            restored_routing = ReplayBufferDiscrete(
                ROUTING_STATE_DIM, NUM_UAV + 1, max_size=8, n_step=1, gamma=0.99
            )
            load_full_resume_checkpoint(
                checkpoint, td3=restored, ddqn=restored_ddqn,
                joint_replay=restored_joint, routing_replay=restored_routing,
                movement_state_dim=MOVEMENT_STATE_DIM,
                joint_action_dim=JOINT_ACTION_DIM,
                routing_state_dim=ROUTING_STATE_DIM,
                calibration={"fixture": "controlled"},
                expected_formal_config=formal_config,
            )
        for key, expected in expected_actor.items():
            self.assertTrue(torch.equal(restored.actor.state_dict()[key].cpu(), expected))


class UniqueRunDirectoryTest(unittest.TestCase):
    def test_repeated_requests_create_distinct_empty_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = create_unique_run_directory(temp_dir, "td3_ratio", 7, "abc123")
            second = create_unique_run_directory(temp_dir, "td3_ratio", 7, "abc123")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertEqual(list(first.iterdir()), [])
            self.assertEqual(list(second.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
