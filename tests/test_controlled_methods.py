import json
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
    FOV_COM_PAIR_MAX_DISTANCE_M,
    FORMAL_EXPERIMENT_DEFAULTS,
    METHOD_REGISTRY,
    MethodSpec,
    NUM_UAV,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    SEARCH_COVERAGE_THRESHOLD,
    effective_training_config,
    exploration_schedule_configuration,
    movement_agent_configuration,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    TrainingConfig,
    _full_training_state,
    _interval_reward,
    formal_training_config,
    terminal_ratio_objective,
    train,
)
from run_experiment import build_parser, create_unique_run_directory
from routing_lifecycle import RoutingLearnerLifecycle
from scenario_manifest import generate_manifest
from Simulator import Simulator
from training_checkpoint import (
    load_full_resume_checkpoint,
    save_full_resume_checkpoint,
    validate_model_checkpoint_metadata,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint
from fov_ema_fixtures import initialized_fov_ema_state


EXISTING_METHODS = (
    "td3_dinkelbach",
    "ddpg_dinkelbach",
    "td3_ratio",
    "ddpg_ratio",
    "random_action",
    "td3_dinkelbach_no_task_potential",
    "ddpg_dinkelbach_no_task_potential",
)
NEW_METHODS = (
    "td3_dinkelbach_wo_ta",
    "td3_dinkelbach_dqn",
    "kkm_random_action_random_routing",
    "km_td3_dinkelbach",
    "random_assignment_td3_dinkelbach",
    "km_ddpg_dinkelbach",
    "ddpg_dinkelbach_wo_ta",
    "td3_dinkelbach_random_routing",
    "td3_dinkelbach_dqn_wo_ta",
)
EXPECTED_METHODS = EXISTING_METHODS + NEW_METHODS


class ControlledMethodRegistryTest(unittest.TestCase):
    def test_registry_contains_and_parses_all_sixteen_methods(self):
        self.assertEqual(tuple(METHOD_REGISTRY), EXPECTED_METHODS)
        self.assertEqual(
            [MethodSpec.parse(key).method_key for key in EXPECTED_METHODS],
            list(EXPECTED_METHODS),
        )
        self.assertEqual(MethodSpec.parse("random_action").label, "Random selected")

    def test_shared_environment_contract_is_kkm_16_uav_and_roi_2_through_8(self):
        for key in EXISTING_METHODS:
            spec = MethodSpec.parse(key)
            self.assertEqual(spec.assignment, "k_km")
            self.assertEqual(spec.routing, "safe_ddqn")
        self.assertEqual(NUM_UAV, 10)
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
                        routing_warmup_transitions=1,
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
                self.assertEqual(result["movement_state_dim"], 429)
                self.assertEqual(result["joint_action_dim"], 30)
                self.assertEqual(result["proposal_batches"], 1)
                metadata = result["run_metadata"]
                self.assertEqual(metadata["assignment_strategy"], spec.assignment)
                self.assertEqual(metadata["assignment_rounds"], spec.assignment_rounds)
                self.assertEqual(metadata["movement_policy"], spec.agent)
                self.assertEqual(metadata["movement_objective"], spec.reward_mode)
                self.assertEqual(metadata["routing_policy"], spec.routing)
                self.assertEqual(
                    metadata["task_observation_mode"], spec.task_observation
                )
                self.assertEqual(
                    metadata["exploration_schedule_version"],
                    "linear_v2_1000ep",
                )
                self.assertEqual(metadata["evaluation_exploration_mode"], "disabled")
                self.assertEqual(
                    metadata["routing_optimizer_update_scope"],
                    "every_4_routing_slots" if spec.learns_routing else None,
                )
                self.assertEqual(
                    metadata["routing_warmup_counter_source"],
                    "routing_replay_size" if spec.learns_routing else None,
                )
                self.assertEqual(metadata["routing_slots_per_episode"], 4)
                self.assertEqual(
                    metadata["fov_com_pair_max_distance_m"],
                    FOV_COM_PAIR_MAX_DISTANCE_M,
                )
                self.assertTrue(metadata["service_assignment_only"])
                self.assertEqual(
                    metadata["search_coverage_threshold"],
                    SEARCH_COVERAGE_THRESHOLD,
                )
                self.assertFalse(metadata["hover_assignment_candidate"])
                self.assertEqual(
                    metadata["masked_state_fields"] is not None,
                    spec.task_observation == "masked",
                )
                if spec.agent == "random":
                    self.assertEqual(result["joint_replay_size"], 0)
                    self.assertEqual(result["critic_updates"], 0)
                    self.assertEqual(result["actor_updates"], 0)
                if spec.routing == "random":
                    self.assertEqual(result["routing_replay_size"], 0)
                    self.assertEqual(result["ddqn_training_updates"], 0)
                    self.assertEqual(result["routing_target_update_count"], 0)
                else:
                    self.assertGreaterEqual(result["routing_replay_size"], 1)
                    self.assertEqual(result["ddqn_training_updates"], 1)
                    self.assertEqual(result["routing_target_update_count"], 1)
                if spec.reward_mode == "ratio":
                    self.assertEqual(result["dinkelbach_update_count"], 0)
                    self.assertIsNone(result["lambda"])

    def test_new_methods_resolve_orthogonal_strategies(self):
        expected = {
            "td3_dinkelbach_wo_ta": (
                "k_km", "td3", "dinkelbach", "masked", "safe_ddqn", 2
            ),
            "td3_dinkelbach_dqn": (
                "k_km", "td3", "dinkelbach", "full", "dqn", 2
            ),
            "kkm_random_action_random_routing": (
                "k_km", "random", "ratio", "full", "random", 2
            ),
            "km_td3_dinkelbach": (
                "km", "td3", "dinkelbach", "full", "safe_ddqn", 1
            ),
            "random_assignment_td3_dinkelbach": (
                "random_one_to_one", "td3", "dinkelbach", "full", "safe_ddqn", 1
            ),
        }
        for method_key, values in expected.items():
            method = MethodSpec.parse(method_key)
            self.assertEqual(
                (
                    method.assignment,
                    method.agent,
                    method.reward_mode,
                    method.task_observation,
                    method.routing,
                    method.assignment_rounds,
                ),
                values,
            )


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
            ratio_objective_reward=0.0,
        )
        second = _interval_reward(
            4.0, 2.0, 999.0, 1.0, self.potential_t, self.potential_t1,
            True, self.config, reward_mode="ratio", task_potential_enabled=False,
            ratio_objective_reward=2.5,
        )
        zero_energy = _interval_reward(
            4.0, 0.0, 999.0, 1.0, self.potential_t, self.potential_t1,
            True, self.config, reward_mode="ratio", task_potential_enabled=False,
            ratio_objective_reward=0.0,
        )
        self.assertEqual((first, second, zero_energy), (0.0, 2.5, 0.0))

    def test_ratio_of_episode_sums_is_stored_only_on_terminal_transition(self):
        replay = ReplayBufferJoint(1, 1, max_size=4)
        transitions = (
            (1.0, 1.0, False, 1.25, 0.0),
            (9.0, 3.0, True, -0.75, 0.0),
        )
        cumulative_delivered = 0.0
        cumulative_energy = 0.0
        objectives = []
        online_rewards = []
        for delivered, energy, done, phi_t, phi_t1 in transitions:
            cumulative_delivered += delivered
            cumulative_energy += energy
            objective = terminal_ratio_objective(
                "ratio",
                done,
                cumulative_delivered,
                cumulative_energy,
            )
            objectives.append(objective)
            replay.add(
                [0.0], [0.0], [0.0], done=done,
                delivered_mbits=delivered,
                total_mobility_energy=energy,
                ratio_objective_reward=objective,
                phi_search_t=phi_t,
                phi_search_t1=phi_t1,
                phi_vs_t=0.0, phi_vs_t1=0.0,
                phi_com_t=0.0, phi_com_t1=0.0,
            )
            online_rewards.append(
                _interval_reward(
                    delivered,
                    energy,
                    current_lambda=999.0,
                    gamma=1.0,
                    potentials_t=(phi_t, 0.0, 0.0),
                    potentials_t1=(phi_t1, 0.0, 0.0),
                    done=done,
                    config=self.config,
                    reward_mode="ratio",
                    task_potential_enabled=True,
                    ratio_objective_reward=objective,
                )
            )
        self.assertEqual(objectives, [0.0, 2_500_000.0])
        self.assertAlmostEqual(
            objectives[-1],
            (1.0 + 9.0) * 1e6 / (1.0 + 3.0),
        )
        unshaped = replay._reward_numpy(
            np.asarray([0, 1]), current_lambda=0.0, gamma=1.0,
            reward_mode="ratio", task_potential_enabled=False,
        ).ravel()
        self.assertTrue(np.allclose(unshaped, [0.0, 2_500_000.0]))
        self.assertAlmostEqual(float(unshaped.sum()), 2_500_000.0)
        self.assertNotAlmostEqual(float(unshaped.sum()), 1.0 / 1.0 + 9.0 / 3.0)

        shaped = replay._reward_numpy(
            np.asarray([0, 1]), current_lambda=999.0, gamma=1.0,
            reward_mode="ratio", task_potential_enabled=True,
        ).ravel()
        self.assertTrue(np.allclose(shaped, [-1.25, 2_500_000.75]))
        self.assertTrue(np.allclose(online_rewards, shaped))
        self.assertAlmostEqual(
            sum(online_rewards),
            objectives[-1] + (-1.25 + 0.75),
        )
        same_ratio = replay._reward_numpy(
            np.asarray([0, 1]), current_lambda=-123.0, gamma=1.0,
            reward_mode="ratio", task_potential_enabled=False,
        ).ravel()
        self.assertTrue(np.array_equal(unshaped, same_ratio))

        dinkelbach = replay._reward_numpy(
            np.asarray([0, 1]), current_lambda=2.0, gamma=1.0,
            reward_mode="dinkelbach", task_potential_enabled=False,
        ).ravel()
        self.assertTrue(np.allclose(dinkelbach, [-1.0, 3.0]))


class EffectiveMovementConfigurationTest(unittest.TestCase):
    def test_effective_algorithm_settings_are_method_specific(self):
        expected = {
            "td3_dinkelbach": (2, 0.20, 0.50, True),
            "ddpg_dinkelbach": (1, None, None, False),
            "random_action": (None, None, None, False),
        }
        for method_key, values in expected.items():
            with self.subTest(method=method_key):
                method = MethodSpec.parse(method_key)
                effective = movement_agent_configuration(method)
                self.assertEqual(
                    (
                        effective["policy_delay"],
                        effective["target_policy_noise"],
                        effective["target_noise_clip"],
                        effective["twin_critics"],
                    ),
                    values,
                )
                serialized = effective_training_config(
                    formal_training_config(1), method
                )
                self.assertEqual(serialized["policy_delay"], values[0])
                self.assertEqual(
                    serialized["movement_agent_configuration"], effective
                )


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
        formal_config = effective_training_config(
            config, MethodSpec.parse("ddpg_dinkelbach")
        )
        agent = self._agent()
        ddqn = DDQN(ROUTING_STATE_DIM, NUM_UAV + 1)
        joint = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=8)
        joint.add(
            np.zeros(MOVEMENT_STATE_DIM),
            np.zeros(JOINT_ACTION_DIM),
            np.zeros(MOVEMENT_STATE_DIM),
            done=True,
            delivered_mbits=10.0,
            total_mobility_energy=4.0,
            phi_search_t=0.0, phi_search_t1=0.0,
            phi_vs_t=0.0, phi_vs_t1=0.0,
            phi_com_t=0.0, phi_com_t1=0.0,
            ratio_objective_reward=2.5,
        )
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
            fov_ema_state=initialized_fov_ema_state(),
            routing_lifecycle_state=RoutingLearnerLifecycle(
                global_slot_count=4
            ).state_dict(),
            exploration_state=exploration_schedule_configuration(
                config, MethodSpec.parse("ddpg_dinkelbach")
            ),
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
            metadata = json.loads(
                (checkpoint / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["networks"]["movement_agent"]["kind"], "ddpg")
            self.assertNotIn("critic_2", payload["networks"]["movement_agent"])
            self.assertNotIn("centralized_td3_gamma", metadata)
            self.assertEqual(metadata["movement_agent_kind"], "ddpg")
            self.assertEqual(metadata["movement_agent_gamma"], 1.0)

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
        self.assertEqual(restored_joint.ratio_objective_reward[0, 0], 2.5)

    def test_legacy_ratio_checkpoint_is_rejected_without_fake_migration(self):
        legacy = {
            "checkpoint_schema_version": 2,
            "checkpoint_type": "model-only",
            "experiment": {"reward_mode": "ratio"},
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "incompatible.*must be retrained",
        ):
            validate_model_checkpoint_metadata(legacy)


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
