import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import td3 as td3_module

from centralized_ddpg import RandomMovementController
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    get_global_movement_state,
    project_joint_action,
)
from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import (
    MethodSpec,
    effective_training_config,
    exploration_schedule_configuration,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    TrainingConfig,
    _full_training_state,
    _run_routing_slot,
    formal_training_config,
    train,
)
from movement_agents import create_movement_agent, sample_random_joint_action
from observation_strategy import (
    MOVEMENT_TASK_ASSIGNMENT_INDICES,
    ROUTING_TASK_ASSIGNMENT_INDICES,
    apply_observation_strategy,
)
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from routing_agents import ControlledDQN, RandomRoutingController
from routing_lifecycle import RoutingLearnerLifecycle
from scenario_manifest import generate_manifest
from Simulator import Simulator
from td3 import TD3
from training_checkpoint import (
    load_full_resume_checkpoint,
    save_full_resume_checkpoint,
    validate_checkpoint_experiment_metadata,
)
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint
from fov_ema_fixtures import initialized_fov_ema_state
from channel_fixtures import initialized_channel_lifecycle_state


class FixedNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values[: inputs.shape[0]]


class MaskedTaskObservationTest(unittest.TestCase):
    def test_named_masks_preserve_dimensions_and_non_task_fields(self):
        movement = np.arange(MOVEMENT_STATE_DIM, dtype=np.float32) + 1.0
        routing = np.arange(ROUTING_STATE_DIM, dtype=np.float32) + 1.0
        masked_movement = apply_observation_strategy(
            movement, "masked", "movement"
        )
        masked_routing = apply_observation_strategy(routing, "masked", "routing")
        self.assertEqual(masked_movement.shape, (519,))
        self.assertEqual(masked_routing.shape, (101,))
        np.testing.assert_array_equal(
            masked_movement[list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        np.testing.assert_array_equal(
            masked_routing[list(ROUTING_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        movement_keep = np.ones(519, dtype=bool)
        movement_keep[list(MOVEMENT_TASK_ASSIGNMENT_INDICES)] = False
        routing_keep = np.ones(ROUTING_STATE_DIM, dtype=bool)
        routing_keep[list(ROUTING_TASK_ASSIGNMENT_INDICES)] = False
        np.testing.assert_array_equal(masked_movement[movement_keep], movement[movement_keep])
        np.testing.assert_array_equal(masked_routing[routing_keep], routing[routing_keep])
        np.testing.assert_array_equal(masked_routing[24:41], routing[24:41])

    def test_production_replays_are_masked_but_projection_sees_true_assignment(self):
        method = MethodSpec.parse("td3_dinkelbach_wo_ta")
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "checkpoints"
            config = TrainingConfig(
                total_episodes=1,
                mode="train",
                episode_seconds=1,
                warmup_joint_transitions=1000,
                batch_size=64,
                model_checkpoint_every=1,
                full_resume_every=1,
                checkpoint_root=str(root),
                enable_model_checkpoints=False,
                enable_full_resume=True,
                enable_plots=False,
                enable_csv=False,
                random_seed=20260817,
            )
            train(config, scenario_manifest=manifest, method_spec=method)
            checkpoint = root / "full" / "ep_0001"
            with np.load(checkpoint / "joint_replay.npz", allow_pickle=False) as replay:
                self.assertEqual(replay["state"].shape, (1, 519))
                np.testing.assert_array_equal(
                    replay["state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
                )
                np.testing.assert_array_equal(
                    replay["next_state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
                )
                self.assertEqual(replay["current_movement_mask"].shape, (1, 10))
                self.assertEqual(replay["next_movement_mask"].shape, (1, 10))
                self.assertTrue(replay["movement_mask_valid"].all())
                self.assertTrue(replay["current_movement_mask"].any())
            with np.load(checkpoint / "routing_replay.npz", allow_pickle=False) as replay:
                # A one-second episode may end before the random scenario admits
                # any packet to the routing layer.  When routing transitions do
                # exist, both current and next observations must stay masked.
                self.assertEqual(replay["state"].shape[1], ROUTING_STATE_DIM)
                np.testing.assert_array_equal(
                    replay["state"][:, list(ROUTING_TASK_ASSIGNMENT_INDICES)], 0.0
                )
                np.testing.assert_array_equal(
                    replay["next_state"][:, list(ROUTING_TASK_ASSIGNMENT_INDICES)], 0.0
                )
                if replay["state"].shape[0]:
                    self.assertTrue(np.any(replay["state"][:, 24:41] > 0.0))
            metadata = json.loads(
                (checkpoint / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["experiment"]["task_observation_mode"], "masked"
            )
            with self.assertRaisesRegex(RuntimeError, "method_spec_fingerprint"):
                validate_checkpoint_experiment_metadata(
                    metadata,
                    {
                        "method_spec_fingerprint": MethodSpec.parse(
                            "td3_dinkelbach"
                        ).compatible_fingerprints
                    },
                )

    def test_two_transition_smoke_really_updates_actor_with_true_masks(self):
        method = MethodSpec.parse("td3_dinkelbach_wo_ta")
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)
        config = TrainingConfig(
            total_episodes=1,
            mode="smoke",
            episode_seconds=2,
            warmup_joint_transitions=0,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
        )
        agent = create_movement_agent(
            method, MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, config
        )
        actor_before = [
            parameter.detach().clone() for parameter in agent.actor.parameters()
        ]
        environment_masks = []
        training_masks = []
        original_projection = project_joint_action

        def environment_projection(raw_action, movement_state=None, *, movement_mask=None):
            self.assertIsNone(movement_state)
            environment_masks.append(np.asarray(movement_mask, dtype=bool).copy())
            return original_projection(raw_action, movement_mask=movement_mask)

        def training_projection(raw_action, movement_state=None, *, movement_mask=None):
            self.assertIsNone(movement_state)
            training_masks.append(movement_mask.detach().cpu().numpy().copy())
            return original_projection(raw_action, movement_mask=movement_mask)

        with (
            mock.patch("HRL_task_aware.create_movement_agent", return_value=agent),
            mock.patch(
                "HRL_task_aware.project_joint_action",
                side_effect=environment_projection,
            ),
            mock.patch.object(
                td3_module,
                "project_joint_action",
                side_effect=training_projection,
            ),
        ):
            result = train(config, scenario_manifest=manifest, method_spec=method)

        self.assertEqual(result["critic_updates"], 2)
        self.assertEqual(result["actor_updates"], 1)
        self.assertEqual(len(environment_masks), 2)
        self.assertGreaterEqual(len(training_masks), 5)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(actor_before, agent.actor.parameters())
            )
        )
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in agent.actor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_total, 0.0)
        update = agent.last_joint_update
        np.testing.assert_array_equal(
            update["state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        sampled_current = update["current_movement_mask"][0].numpy()
        sampled_next = update["next_movement_mask"][0].numpy()
        self.assertTrue(
            any(np.array_equal(sampled_current, mask) for mask in environment_masks)
        )
        self.assertTrue(
            any(np.array_equal(sampled_next, mask[0]) for mask in training_masks)
        )

class ControlledDQNTest(unittest.TestCase):
    def test_effective_mask_is_recomputed_in_every_routing_slot(self):
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)
        active_slot = [None]
        effective_mask_calls = {}
        physical_mask_calls = {}
        original_effective_mask = PacketEngine.get_effective_action_mask
        original_physical_mask = Simulator.get_routing_action_mask

        def track_slot(*args, **kwargs):
            active_slot[0] = float(kwargs["current_time"])
            try:
                return _run_routing_slot(*args, **kwargs)
            finally:
                active_slot[0] = None

        def track_effective_mask(engine, env, uav_id, physical_mask=None):
            if active_slot[0] is not None:
                effective_mask_calls[active_slot[0]] = (
                    effective_mask_calls.get(active_slot[0], 0) + 1
                )
            return original_effective_mask(engine, env, uav_id, physical_mask)

        def track_physical_mask(env, uav_id):
            if active_slot[0] is not None:
                physical_mask_calls[active_slot[0]] = (
                    physical_mask_calls.get(active_slot[0], 0) + 1
                )
            return original_physical_mask(env, uav_id)

        config = TrainingConfig(
            total_episodes=1,
            mode="smoke",
            episode_seconds=1,
            warmup_joint_transitions=1000,
            batch_size=64,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
        )
        with (
            mock.patch("HRL_task_aware._run_routing_slot", side_effect=track_slot),
            mock.patch.object(
                PacketEngine,
                "get_effective_action_mask",
                autospec=True,
                side_effect=track_effective_mask,
            ),
            mock.patch.object(
                Simulator,
                "get_routing_action_mask",
                autospec=True,
                side_effect=track_physical_mask,
            ),
        ):
            train(
                config,
                scenario_manifest=manifest,
                method_spec=MethodSpec.parse("td3_dinkelbach_dqn"),
            )
        # This short random scenario may not discover/assign a source.  For
        # every slot that does have a routing decision, both masks must be
        # recomputed in that same slot (terminal bookkeeping happens outside
        # the tracked routing-slot wrapper).
        self.assertEqual(set(effective_mask_calls), set(physical_mask_calls))
        self.assertTrue(
            set(effective_mask_calls).issubset({0.0, 0.25, 0.5, 0.75})
        )
        self.assertTrue(all(count > 0 for count in effective_mask_calls.values()))
        self.assertTrue(all(count > 0 for count in physical_mask_calls.values()))

    def test_has_reward_network_only_and_uses_standard_masked_target_maximum(self):
        agent = ControlledDQN.__new__(ControlledDQN)
        agent.action_dim = 3
        agent.gamma = 0.9
        agent.q_network = FixedNetwork([[100.0, 0.0, 1.0]])
        agent.target_q_network = FixedNetwork([[1.0, 999.0, 8.0]])
        self.assertFalse(hasattr(agent, "cost_network"))
        self.assertFalse(hasattr(agent, "cost_optimizer"))
        self.assertFalse(hasattr(agent, "eta"))
        next_state = torch.zeros((1, 42), dtype=torch.float32)
        next_state[0, 10:13] = torch.tensor([1.0, 0.0, 1.0])
        target = agent._standard_targets(
            next_state,
            reward=torch.tensor([[2.0]]),
            not_done=torch.tensor([[1.0]]),
        )
        self.assertAlmostEqual(target.item(), 2.0 + 0.9 * 8.0, places=6)

    def test_epsilon_and_greedy_actions_respect_current_mask(self):
        agent = ControlledDQN(state_dim=42, action_dim=3, hidden_dim=8)
        agent.q_network = FixedNetwork([[100.0, 2.0, 3.0]])
        state = np.zeros(42, dtype=np.float32)
        mask = np.asarray([False, True, False])
        for epsilon in (0.0, 1.0):
            for _ in range(20):
                self.assertEqual(
                    agent.select_action(state, 0, mask=mask, epsilon=epsilon),
                    1,
                )

    def test_full_checkpoint_round_trip_has_no_cost_network(self):
        method = MethodSpec.parse("td3_dinkelbach_dqn")
        config = formal_training_config(
            1,
            enable_plots=False,
            enable_csv=False,
            routing_warmup_transitions=1,
        )
        formal_config = effective_training_config(config, method)
        movement = TD3(4, 2, 1.0, gamma=1.0)
        routing = ControlledDQN(101, 3, hidden_dim=8)
        routing.num_training = 1
        routing.target_update_count = 1
        routing.reward_optimizer_update_count = 1
        routing.reward_target_update_count = 1
        routing.loss_log = [1.25]
        joint = ReplayBufferJoint(4, 2, max_size=8)
        joint.add(
            np.zeros(4), np.zeros(2), np.ones(4), True,
            delivered_mbits=1.0, total_mobility_energy=2.0,
            phi_search_t=0.0, phi_search_t1=0.0,
            phi_vs_t=0.0, phi_vs_t1=0.0,
            phi_com_t=0.0, phi_com_t1=0.0,
        )
        routing_replay = ReplayBufferDiscrete(101, 3, max_size=8, n_step=1)
        routing_state = np.zeros(101, dtype=np.float32)
        routing_state[10:13] = [1.0, 0.0, 1.0]
        routing_replay.add(
            routing_state, 0, routing_state, reward=1.0, cost=3.0, done=False
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
                warmup_transitions=1,
                global_slot_count=4,
                optimizer_update_count=1,
                target_update_count=1,
                epsilon_decay_start_slot=1,
                last_optimizer_update_slot=4,
            ).state_dict(),
            exploration_state=exploration_schedule_configuration(config, method),
            channel_lifecycle_state=initialized_channel_lifecycle_state(),
        )
        experiment = {
            "method_spec_fingerprint": method.fingerprint,
            "method_spec": method.to_dict(),
            "formal_config": formal_config,
            "dinkelbach_state": dinkelbach.training_state(),
            "lambda_ee": dinkelbach.current_lambda,
            **dinkelbach_config_metadata(config),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ep_0001"
            save_full_resume_checkpoint(
                checkpoint,
                episode=0,
                td3=movement,
                ddqn=routing,
                joint_replay=joint,
                routing_replay=routing_replay,
                training_state=training_state,
                formal_config=formal_config,
                movement_state_dim=4,
                joint_action_dim=2,
                routing_state_dim=101,
                calibration={"fixture": "dqn"},
                experiment_metadata=experiment,
            )
            payload = torch.load(
                checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["networks"]["routing_agent"]["kind"], "dqn")
            self.assertNotIn("cost_network", payload["networks"]["routing_agent"])
            self.assertNotIn("ddqn_optimizers", payload)

            restored_movement = TD3(4, 2, 1.0, gamma=1.0)
            restored_routing = ControlledDQN(101, 3, hidden_dim=8)
            restored_joint = ReplayBufferJoint(4, 2, max_size=8)
            restored_replay = ReplayBufferDiscrete(101, 3, max_size=8, n_step=1)
            load_full_resume_checkpoint(
                checkpoint,
                td3=restored_movement,
                ddqn=restored_routing,
                joint_replay=restored_joint,
                routing_replay=restored_replay,
                movement_state_dim=4,
                joint_action_dim=2,
                routing_state_dim=101,
                calibration={"fixture": "dqn"},
                expected_experiment_metadata={
                    "method_spec_fingerprint": method.compatible_fingerprints
                },
                expected_formal_config=formal_config,
            )
        self.assertEqual(restored_routing.num_training, 1)
        self.assertEqual(restored_routing.target_update_count, 1)
        self.assertEqual(restored_routing.loss_log, [1.25])
        self.assertEqual(restored_replay.size, 1)


class RandomMovementRoutingTest(unittest.TestCase):
    def test_seeded_random_actions_use_shared_domain_projection_and_mask(self):
        np.random.seed(55)
        first = sample_random_joint_action(JOINT_ACTION_DIM)
        np.random.seed(55)
        second = sample_random_joint_action(JOINT_ACTION_DIM)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= -1.0))
        self.assertTrue(np.all(first <= 1.0))

        env = Simulator(num_UAV=10)
        env.num_GT = 2
        env.reset_environment()
        packet_engine = PacketEngine(num_uav=10, step_time=0.25)
        state = get_global_movement_state(
            env, packet_engine, packet_engine.backlog_bits, 1.0, 1.0
        )
        projected = project_joint_action(first, state)
        self.assertTrue(np.all(projected >= -1.0))
        self.assertTrue(np.all(projected <= 1.0))

        routing = RandomRoutingController()
        mask = np.asarray([False, True, False, True])
        np.random.seed(9)
        selected = {
            routing.select_action(np.zeros(1), 0, mask=mask) for _ in range(100)
        }
        self.assertEqual(selected, {1, 3})

    def test_training_updates_no_network_replay_or_dinkelbach_state(self):
        method = MethodSpec.parse("kkm_random_action_random_routing")
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)
        config = TrainingConfig(
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
        )
        first = train(config, scenario_manifest=manifest, method_spec=method)
        second = train(config, scenario_manifest=manifest, method_spec=method)
        self.assertEqual(first["reward_log"], second["reward_log"])
        self.assertEqual(first["joint_replay_size"], 0)
        self.assertEqual(first["routing_replay_size"], 0)
        self.assertEqual(first["critic_updates"], 0)
        self.assertEqual(first["actor_updates"], 0)
        self.assertEqual(first["ddqn_training_updates"], 0)
        self.assertEqual(first["routing_target_update_count"], 0)
        self.assertEqual(first["dinkelbach_update_count"], 0)
        self.assertIsNone(first["lambda"])

    def test_pure_random_training_does_not_create_a_fake_checkpoint(self):
        method = MethodSpec.parse("kkm_random_action_random_routing")
        manifest = generate_manifest("train", 20260817, 1, num_gt=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                total_episodes=1,
                mode="train",
                episode_seconds=1,
                warmup_joint_transitions=0,
                batch_size=1,
                model_checkpoint_every=1,
                full_resume_every=1,
                checkpoint_root=str(Path(temp_dir) / "checkpoints"),
                enable_model_checkpoints=True,
                enable_full_resume=True,
                enable_plots=False,
                enable_csv=False,
                random_seed=20260817,
            )
            train(config, scenario_manifest=manifest, method_spec=method)
            self.assertFalse(Path(config.checkpoint_root, "models").exists())
            self.assertFalse(Path(config.checkpoint_root, "full").exists())

    def test_pure_random_evaluation_runs_without_checkpoint_provenance(self):
        method = MethodSpec.parse("kkm_random_action_random_routing")
        manifest = generate_manifest("test", 20260817, 1, num_gt=2)
        result = train(
            TrainingConfig(
                total_episodes=1,
                mode="custom",
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
            method_spec=method,
            evaluation=True,
            checkpoint_dir=None,
        )
        metadata = result["run_metadata"]
        self.assertFalse(metadata["checkpoint_required"])
        self.assertIsNone(metadata["checkpoint_metadata_path"])
        self.assertIsNone(metadata["checkpoint_metadata_fingerprint"])
        self.assertIn("no neural state", metadata["no_checkpoint_reason"])


if __name__ == "__main__":
    unittest.main()
