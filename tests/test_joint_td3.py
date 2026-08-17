import unittest
from unittest import mock

import numpy as np
import torch

from Simulator import Simulator
from centralized_movement import (
    HOVER_ACTION,
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    MOVEMENT_STATE_DIM,
    apply_joint_movement_proposals,
    build_joint_movement_proposals,
    project_joint_action,
)
from td3 import TD3, _bellman_target
from utils_update_v2 import ReplayBufferJoint


def movement_state(movable_uavs):
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    for uav_id in movable_uavs:
        state[uav_id * LOCAL_MOVEMENT_DIM] = 1.0
    return state


class JointActionProjectionTest(unittest.TestCase):
    def test_shape_block_order_and_hover_projection(self):
        state = movement_state({0, 3, 15})
        raw = np.linspace(-1.0, 1.0, JOINT_ACTION_DIM, dtype=np.float32)
        projected = project_joint_action(raw, state)
        self.assertEqual(projected.shape, (48,))
        blocks = projected.reshape(16, 3)
        raw_blocks = raw.reshape(16, 3)
        for uav_id in range(16):
            if uav_id in {0, 3, 15}:
                np.testing.assert_array_equal(blocks[uav_id], raw_blocks[uav_id])
            else:
                np.testing.assert_array_equal(blocks[uav_id], HOVER_ACTION)

    def test_torch_projection_preserves_gradient_only_for_movable_blocks(self):
        state = torch.from_numpy(movement_state({2})).reshape(1, -1)
        raw = torch.zeros((1, JOINT_ACTION_DIM), requires_grad=True)
        projected = project_joint_action(raw, state)
        projected.sum().backward()
        gradients = raw.grad.reshape(16, 3)
        self.assertTrue(torch.all(gradients[2] == 1.0))
        self.assertTrue(torch.all(gradients[:2] == 0.0))
        self.assertTrue(torch.all(gradients[3:] == 0.0))

    def test_all_proposals_precede_mutation_hover_is_stationary_and_energy_positive(self):
        env = Simulator(num_UAV=16)
        env.num_GT = 2
        env.reset_environment()
        model = TD3(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, 1.0)
        state = movement_state({15})
        raw = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        raw.reshape(16, 3)[15] = np.array([1.0, 0.0, 0.0])
        projected = project_joint_action(raw, state)
        positions_before = [uav.get_position() for uav in env.UAVs]

        proposals = build_joint_movement_proposals(env, model, projected)
        self.assertEqual(len(proposals), 16)
        self.assertEqual(positions_before, [uav.get_position() for uav in env.UAVs])

        with mock.patch.object(
            env.energy_model,
            "compute_mobility_energy",
            wraps=env.energy_model.compute_mobility_energy,
        ) as energy_call:
            energies = apply_joint_movement_proposals(env, proposals)
        self.assertEqual(energy_call.call_count, 16)
        self.assertTrue(np.all(energies > 0.0))
        for uav_id in range(15):
            self.assertEqual(env.uav_dict[uav_id].get_position(), positions_before[uav_id])
        self.assertNotEqual(env.uav_dict[15].get_position(), positions_before[15])


class JointReplayAndLearnerTest(unittest.TestCase):
    def _add(self, replay, done):
        state = movement_state({0})
        action = project_joint_action(np.zeros(JOINT_ACTION_DIM, dtype=np.float32), state)
        replay.add(
            state,
            action,
            state,
            done=done,
            delivered_mbits=5.0,
            total_mobility_energy=2.0,
            phi_search_t=1.0,
            phi_search_t1=2.0,
            phi_vs_t=1.0,
            phi_vs_t1=2.0,
            phi_com_t=1.0,
            phi_com_t1=2.0,
        )

    def test_current_lambda_reward_terminal_potential_and_replay_size(self):
        replay = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=4)
        self._add(replay, done=False)
        self._add(replay, done=True)
        size_before = replay.size
        reward_lambda_3 = replay._reward_numpy(
            np.array([0, 1]), current_lambda=3.0, gamma=0.9
        ).ravel()
        reward_lambda_4 = replay._reward_numpy(
            np.array([0, 1]), current_lambda=4.0, gamma=0.9
        ).ravel()
        self.assertAlmostEqual(reward_lambda_3[0], 1.4, places=5)
        self.assertAlmostEqual(reward_lambda_3[1], -4.0, places=5)
        np.testing.assert_allclose(reward_lambda_4, reward_lambda_3 - 2.0)
        self.assertEqual(replay.size, size_before)
        np.testing.assert_array_equal(replay.not_done[:2, 0], [1.0, 0.0])

        target = _bellman_target(
            torch.tensor([[5.0], [5.0]]),
            torch.tensor([[7.0], [7.0]]),
            torch.from_numpy(replay.not_done[:2]),
            0.9,
        )
        self.assertTrue(torch.allclose(target, torch.tensor([[11.3], [5.0]])))

    def test_target_smoothing_and_actor_update_share_hover_projection(self):
        replay = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=4)
        self._add(replay, done=False)
        self._add(replay, done=False)
        model = TD3(
            MOVEMENT_STATE_DIM,
            JOINT_ACTION_DIM,
            1.0,
            policy_delay=2,
        )
        self.assertFalse(model.update_joint(replay, current_lambda=0.1, batch_size=1))
        self.assertTrue(model.update_joint(replay, current_lambda=0.1, batch_size=1))
        self.assertEqual(model.num_critic_update_iteration, 2)
        self.assertEqual(model.num_actor_update_iteration, 1)

        for name in ("target_actor_action", "target_smoothed_action", "actor_action"):
            blocks = model.last_joint_update[name].reshape(-1, 16, 3)
            expected = torch.tensor(HOVER_ACTION, dtype=blocks.dtype)
            self.assertTrue(torch.allclose(blocks[:, 1:, :], expected.expand_as(blocks[:, 1:, :])))


if __name__ == "__main__":
    unittest.main()
