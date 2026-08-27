import copy
import random
import unittest

import numpy as np
import torch

from centralized_ddpg import CentralizedDDPG
from centralized_movement import (
    JOINT_ACTION_DIM,
    project_joint_action,
    project_local_action,
)
from experiment_config import METHOD_REGISTRY, MethodSpec, movement_agent_configuration
from HRL_task_aware import PRODUCTION_WARMUP_TRANSITIONS, _uses_warmup_random_action
from rng_contract import NamedRNGStreams, RNG_CONTRACT_VERSION, RNG_STREAM_IDS
from routing_agents import create_routing_agent
from td3 import TD3
from utils_update_v2 import ReplayBufferJoint


def _module_equal(test, left, right):
    for left_value, right_value in zip(left.state_dict().values(), right.state_dict().values()):
        test.assertTrue(torch.equal(left_value, right_value))


def _add_transition(replay, value=0.0):
    mask = np.asarray([True, False] + [True] * 8, dtype=bool)
    state = np.full(4, value, dtype=np.float32)
    action = np.linspace(-1.2, 1.2, JOINT_ACTION_DIM, dtype=np.float32)
    replay.add(
        state,
        action,
        state + 0.1,
        done=False,
        delivered_mbits=1.0,
        total_mobility_energy=2.0,
        phi_search_t=0.0,
        phi_search_t1=0.1,
        phi_vs_t=0.0,
        phi_vs_t1=0.1,
        phi_com_t=0.0,
        phi_com_t1=0.1,
        current_movement_mask=mask,
        next_movement_mask=mask,
    )


class NamedRngIsolationTest(unittest.TestCase):
    def test_extra_draws_do_not_cross_subsystem_or_train_eval_boundaries(self):
        baseline = NamedRNGStreams(1234)
        perturbed = NamedRNGStreams(1234)
        perturbed.numpy("movement_exploration").normal(size=100)
        perturbed.numpy("random_assignment").integers(0, 100, size=100)

        np.testing.assert_array_equal(
            baseline.numpy("movement_replay_sampling").integers(0, 1000, 16),
            perturbed.numpy("movement_replay_sampling").integers(0, 1000, 16),
        )
        np.testing.assert_array_equal(
            baseline.numpy("evaluation_random_assignment").integers(0, 1000, 16),
            perturbed.numpy("evaluation_random_assignment").integers(0, 1000, 16),
        )

    def test_state_round_trip_restores_exact_next_numpy_and_torch_draws(self):
        streams = NamedRNGStreams(9876)
        for name in RNG_STREAM_IDS:
            streams.numpy(name).random()
        torch.randn(1, generator=streams.torch("td3_target_policy_noise"))
        state = copy.deepcopy(streams.state_dict())
        expected_numpy = {
            name: streams.numpy(name).integers(0, 2**31, size=4)
            for name in RNG_STREAM_IDS
        }
        expected_torch = torch.randn(
            4, generator=streams.torch("td3_target_policy_noise")
        )

        restored = NamedRNGStreams(9876)
        restored.load_state_dict(state)
        self.assertEqual(state["rng_contract_version"], RNG_CONTRACT_VERSION)
        for name, expected in expected_numpy.items():
            np.testing.assert_array_equal(
                restored.numpy(name).integers(0, 2**31, size=4), expected
            )
        self.assertTrue(
            torch.equal(
                torch.randn(
                    4, generator=restored.torch("td3_target_policy_noise")
                ),
                expected_torch,
            )
        )

    def test_agent_construction_is_order_independent_and_global_rng_clean(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        td3 = TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=NamedRNGStreams(55))
        ddpg = CentralizedDDPG(
            4, JOINT_ACTION_DIM, 1.0, rng_streams=NamedRNGStreams(55)
        )
        _module_equal(self, td3.actor, ddpg.actor)
        _module_equal(self, td3.critic_1, ddpg.critic)
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_td3_extra_critic_does_not_change_routing_init_and_seeds_differ(self):
        method = MethodSpec.parse("td3_dinkelbach")
        with_td3 = NamedRNGStreams(71)
        TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=with_td3)
        routed_after_td3 = create_routing_agent(method, 90, 11, with_td3)
        routed_direct = create_routing_agent(
            method, 90, 11, NamedRNGStreams(71)
        )
        _module_equal(self, routed_after_td3.q_network, routed_direct.q_network)
        _module_equal(self, routed_after_td3.cost_network, routed_direct.cost_network)

        different = TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=NamedRNGStreams(72))
        same = TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=NamedRNGStreams(71))
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(
                    same.actor.state_dict().values(),
                    different.actor.state_dict().values(),
                )
            )
        )

    def test_random_baseline_streams_are_reproducible_and_mutually_isolated(self):
        first = NamedRNGStreams(818)
        second = NamedRNGStreams(818)
        for name in ("random_assignment", "random_movement", "random_routing"):
            np.testing.assert_array_equal(
                first.numpy(name).integers(0, 1000, size=12),
                second.numpy(name).integers(0, 1000, size=12),
            )
        first.numpy("random_assignment").random(20)
        np.testing.assert_array_equal(
            first.numpy("random_movement").integers(0, 1000, size=12),
            second.numpy("random_movement").integers(0, 1000, size=12),
        )


class ProjectionSmoothingReplayTest(unittest.TestCase):
    def test_projection_clamps_speed_vertical_wraps_heading_then_masks(self):
        raw = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        raw[:9] = [2.0, 1.0, -2.0, -2.0, 3.25, 2.0, 0.5, -3.25, 0.5]
        mask = np.asarray([True, False, True] + [False] * 7, dtype=bool)
        expected = np.asarray([1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.5, 0.75, 0.5])
        projected = project_joint_action(raw, movement_mask=mask)
        np.testing.assert_allclose(projected[:9], expected)
        torch_projected = project_joint_action(
            torch.from_numpy(raw), movement_mask=torch.from_numpy(mask)
        )
        np.testing.assert_allclose(torch_projected.numpy(), projected)
        self.assertAlmostEqual(float(project_local_action([0.0, 1.08, 0.0])[1]), -0.92, places=6)
        self.assertAlmostEqual(float(project_local_action([0.0, -1.08, 0.0])[1]), 0.92, places=6)
        equivalent = CentralizedDDPG.decode_action([0.0, 0.2, 0.0])
        wrapped_equivalent = CentralizedDDPG.decode_action([0.0, 2.2, 0.0])
        np.testing.assert_allclose(equivalent, wrapped_equivalent, atol=1e-6)

    def test_td3_target_noise_is_independent_clipped_and_masked(self):
        first_streams = NamedRNGStreams(44)
        second_streams = NamedRNGStreams(44)
        first = TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=first_streams)
        second = TD3(4, JOINT_ACTION_DIM, 1.0, rng_streams=second_streams)
        first_replay = ReplayBufferJoint(
            4, JOINT_ACTION_DIM, max_size=4,
            rng=first_streams.numpy("movement_replay_sampling"),
        )
        second_replay = ReplayBufferJoint(
            4, JOINT_ACTION_DIM, max_size=4,
            rng=second_streams.numpy("movement_replay_sampling"),
        )
        _add_transition(first_replay)
        _add_transition(second_replay)
        first.select_action(np.zeros(4, dtype=np.float32), noise_std=0.2)

        first.update_joint(first_replay, current_lambda=0.0, batch_size=1)
        second.update_joint(second_replay, current_lambda=0.0, batch_size=1)
        first_noise = first.last_joint_update["target_policy_noise"].numpy()
        second_noise = second.last_joint_update["target_policy_noise"].numpy()
        np.testing.assert_array_equal(first_noise, second_noise)
        self.assertLessEqual(float(np.abs(first_noise).max()), 0.25)
        np.testing.assert_array_equal(first_noise.reshape(1, 10, 3)[:, 1], 0.0)
        smoothed = first.last_joint_update["target_smoothed_action"].numpy().reshape(1, 10, 3)
        self.assertTrue(np.all(smoothed[..., 1] >= -1.0))
        self.assertTrue(np.all(smoothed[..., 1] < 1.0))

    def test_replay_wrap_diagnostics_and_executed_action_storage(self):
        replay = ReplayBufferJoint(4, JOINT_ACTION_DIM, max_size=3)
        for value in range(5):
            _add_transition(replay, float(value))
        self.assertEqual(
            replay.diagnostics(),
            {
                "capacity": 3,
                "size": 3,
                "write_pointer": 2,
                "total_added": 5,
                "wrapped": True,
                "oldest_physical_index": 2,
                "newest_physical_index": 1,
                "oldest_age": 2,
                "newest_age": 0,
            },
        )
        inactive = replay.action[: replay.size].reshape(-1, 10, 3)[:, 1]
        np.testing.assert_array_equal(
            inactive,
            np.tile(np.asarray([-1.0, 0.0, 0.0]), (replay.size, 1)),
        )
        self.assertTrue(np.all(replay.action[: replay.size, 0::3] <= 1.0))
        self.assertTrue(np.all(replay.action[: replay.size, 0::3] >= -1.0))

    def test_replay_diagnostics_do_not_consume_sampler_and_unfilled_is_stable(self):
        first_streams = NamedRNGStreams(313)
        second_streams = NamedRNGStreams(313)
        first = ReplayBufferJoint(
            4, JOINT_ACTION_DIM, max_size=5,
            rng=first_streams.numpy("movement_replay_sampling"),
        )
        second = ReplayBufferJoint(
            4, JOINT_ACTION_DIM, max_size=5,
            rng=second_streams.numpy("movement_replay_sampling"),
        )
        for value in range(3):
            _add_transition(first, value)
            _add_transition(second, value)
        np.testing.assert_array_equal(first.state[:3, 0], [0.0, 1.0, 2.0])
        before = (first.size, first.ptr)
        first.diagnostics()
        self.assertEqual((first.size, first.ptr), before)
        first_batch = first.sample(3, 0.0, 1.0)
        second_batch = second.sample(3, 0.0, 1.0)
        for left, right in zip(first_batch, second_batch):
            self.assertTrue(torch.equal(left, right))

        size_ptr = (first.size, first.ptr)
        reward_zero = first._reward_numpy(np.asarray([0]), 0.0, 1.0)
        reward_nonzero = first._reward_numpy(np.asarray([0]), 5.0, 1.0)
        self.assertEqual((first.size, first.ptr), size_ptr)
        self.assertFalse(np.array_equal(reward_zero, reward_nonzero))


class RegistryContractTest(unittest.TestCase):
    def test_registry_capabilities_and_fixed_learned_contract(self):
        self.assertEqual(PRODUCTION_WARMUP_TRANSITIONS, 10_000)
        self.assertTrue(_uses_warmup_random_action(9_999, 10_000))
        self.assertFalse(_uses_warmup_random_action(10_000, 10_000))
        for method_id in METHOD_REGISTRY:
            method = MethodSpec.parse(method_id)
            resolved = movement_agent_configuration(method)
            with self.subTest(method=method_id):
                if method.learns_movement:
                    self.assertEqual(resolved["replay_capacity"], 50_000)
                    self.assertEqual(resolved["warmup_joint_transitions"], 10_000)
                    self.assertEqual(
                        resolved["replay_action_semantics"],
                        "executed_projected_joint_action",
                    )
                else:
                    self.assertIsNone(resolved["replay_capacity"])
                    self.assertIsNone(resolved["warmup_joint_transitions"])
                if method.agent == "td3":
                    self.assertEqual(resolved["target_policy_noise"], 0.10)
                    self.assertEqual(resolved["target_noise_clip"], 0.25)
                else:
                    self.assertIsNone(resolved["target_policy_noise"])
                    self.assertIsNone(resolved["target_noise_clip"])

    def test_replay_capacity_is_independent_of_formal_episode_horizon(self):
        method = MethodSpec.parse("td3_dinkelbach")
        for episodes in (1500, 2500, 3000):
            with self.subTest(episodes=episodes):
                resolved = movement_agent_configuration(
                    method,
                    {
                        "total_episodes": episodes,
                        "replay_max_size": 50_000,
                        "warmup_joint_transitions": 10_000,
                    },
                )
                self.assertEqual(resolved["replay_capacity"], 50_000)


if __name__ == "__main__":
    unittest.main()
