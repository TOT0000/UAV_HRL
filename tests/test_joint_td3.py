import unittest
from unittest import mock

import numpy as np
import torch

from Simulator import Simulator
from centralized_ddpg import CentralizedDDPG
from centralized_movement import (
    HOVER_ACTION,
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    MOVEMENT_STATE_DIM,
    apply_joint_movement_proposals,
    build_joint_movement_proposals,
    movement_mask_from_state,
    project_joint_action,
)
from observation_strategy import (
    MOVEMENT_TASK_ASSIGNMENT_INDICES,
    apply_observation_strategy,
)
from td3 import TD3, _bellman_target
from utils_update_v2 import ReplayBufferJoint


def movement_state(movable_uavs):
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    for uav_id in movable_uavs:
        state[uav_id * LOCAL_MOVEMENT_DIM] = 1.0
    return state


class JointActionProjectionTest(unittest.TestCase):
    def test_explicit_single_and_batch_masks_match_full_state_projection(self):
        full_current = movement_state({0, 3, 9})
        full_next = movement_state({1, 7})
        masked = apply_observation_strategy(full_current, "masked", "movement")
        np.testing.assert_array_equal(
            masked[list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        current_mask = movement_mask_from_state(full_current)
        next_mask = movement_mask_from_state(full_next)
        raw = np.linspace(-0.8, 0.8, JOINT_ACTION_DIM, dtype=np.float32)

        derived = project_joint_action(raw, full_current)
        explicit = project_joint_action(raw, movement_mask=current_mask)
        np.testing.assert_array_equal(explicit, derived)
        blocks = explicit.reshape(16, 3)
        raw_blocks = raw.reshape(16, 3)
        for uav_id in range(16):
            expected = raw_blocks[uav_id] if current_mask[uav_id] else HOVER_ACTION
            np.testing.assert_array_equal(blocks[uav_id], expected)

        batch_raw = np.stack([raw, -raw])
        batch_mask = np.stack([current_mask, next_mask])
        batch_projected = project_joint_action(
            batch_raw, movement_mask=batch_mask
        )
        np.testing.assert_array_equal(batch_projected[0], explicit)
        np.testing.assert_array_equal(
            batch_projected[1],
            project_joint_action(-raw, movement_mask=next_mask),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            project_joint_action(raw, full_current, movement_mask=current_mask)
        with self.assertRaisesRegex(ValueError, "shape"):
            project_joint_action(raw, movement_mask=np.ones(9, dtype=bool))
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            project_joint_action(raw, movement_mask=np.full(16, 0.5))

    def test_shape_block_order_and_hover_projection(self):
        state = movement_state({0, 3, 9})
        raw = np.linspace(-1.0, 1.0, JOINT_ACTION_DIM, dtype=np.float32)
        projected = project_joint_action(raw, state)
        self.assertEqual(projected.shape, (48,))
        blocks = projected.reshape(16, 3)
        raw_blocks = raw.reshape(16, 3)
        for uav_id in range(16):
            if uav_id in {0, 3, 9}:
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
        state = movement_state({9})
        raw = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        raw.reshape(16, 3)[9] = np.array([1.0, 0.0, 0.0])
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
        for uav_id in set(range(16)) - {9}:
            self.assertEqual(env.uav_dict[uav_id].get_position(), positions_before[uav_id])
        self.assertNotEqual(env.uav_dict[9].get_position(), positions_before[9])


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

    def test_masked_replay_keeps_ordered_true_current_and_next_masks(self):
        replay = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=2)
        full_current = movement_state({0, 4})
        full_next = movement_state({1, 4, 9})
        masked_current = apply_observation_strategy(full_current, "masked", "movement")
        masked_next = apply_observation_strategy(full_next, "masked", "movement")
        current_mask = movement_mask_from_state(full_current)
        next_mask = movement_mask_from_state(full_next)
        action = project_joint_action(
            np.zeros(JOINT_ACTION_DIM, dtype=np.float32),
            movement_mask=current_mask,
        )
        replay.add(
            masked_current,
            action,
            masked_next,
            done=False,
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
        batch = replay.sample(
            1, current_lambda=0.0, gamma=1.0, include_movement_masks=True
        )
        state, _, next_state, _, _, sampled_current, sampled_next = batch
        self.assertEqual(tuple(sampled_current.shape), (1, 16))
        self.assertEqual(tuple(sampled_next.shape), (1, 16))
        np.testing.assert_array_equal(sampled_current.cpu().numpy()[0], current_mask)
        np.testing.assert_array_equal(sampled_next.cpu().numpy()[0], next_mask)
        np.testing.assert_array_equal(
            state.cpu().numpy()[:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        np.testing.assert_array_equal(
            next_state.cpu().numpy()[:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )

    def _masked_update_replay(self):
        replay = ReplayBufferJoint(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, max_size=4)
        full_current = movement_state({0, 3})
        full_next = movement_state({1, 3, 8})
        masked_current = apply_observation_strategy(full_current, "masked", "movement")
        masked_next = apply_observation_strategy(full_next, "masked", "movement")
        current_mask = movement_mask_from_state(full_current)
        next_mask = movement_mask_from_state(full_next)
        action = project_joint_action(
            np.full(JOINT_ACTION_DIM, 0.25, dtype=np.float32),
            movement_mask=current_mask,
        )
        for delivered in (1.0, 2.0):
            replay.add(
                masked_current,
                action,
                masked_next,
                done=False,
                delivered_mbits=delivered,
                total_mobility_energy=0.5,
                phi_search_t=0.0,
                phi_search_t1=0.0,
                phi_vs_t=0.0,
                phi_vs_t1=0.0,
                phi_com_t=0.0,
                phi_com_t1=0.0,
                current_movement_mask=current_mask,
                next_movement_mask=next_mask,
            )
        return replay, current_mask, next_mask

    def test_masked_td3_actor_and_target_updates_use_true_masks(self):
        torch.manual_seed(17)
        replay, current_mask, next_mask = self._masked_update_replay()
        model = TD3(
            MOVEMENT_STATE_DIM,
            JOINT_ACTION_DIM,
            1.0,
            policy_noise=0.0,
            policy_delay=2,
        )
        self.assertFalse(model.update_joint(replay, current_lambda=0.0, batch_size=1))
        before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        self.assertTrue(model.update_joint(replay, current_lambda=0.0, batch_size=1))
        self.assertEqual(model.num_actor_update_iteration, 1)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(before, model.actor.parameters())
            )
        )
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.actor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_total, 0.0)
        update = model.last_joint_update
        np.testing.assert_array_equal(update["current_movement_mask"][0], current_mask)
        np.testing.assert_array_equal(update["next_movement_mask"][0], next_mask)
        np.testing.assert_array_equal(
            update["state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)], 0.0
        )
        for action_name, mask in (
            ("actor_action", current_mask),
            ("target_actor_action", next_mask),
            ("target_smoothed_action", next_mask),
        ):
            blocks = update[action_name].reshape(-1, 16, 3)[0]
            hover = torch.tensor(HOVER_ACTION, dtype=blocks.dtype)
            self.assertTrue(torch.all(blocks[~torch.from_numpy(mask)] == hover))
            self.assertTrue(torch.any(blocks[torch.from_numpy(mask)] != hover))

    def test_ddpg_uses_the_same_explicit_mask_dataflow(self):
        torch.manual_seed(23)
        replay, current_mask, next_mask = self._masked_update_replay()
        model = CentralizedDDPG(MOVEMENT_STATE_DIM, JOINT_ACTION_DIM, 1.0)
        before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        self.assertTrue(model.update_joint(replay, current_lambda=0.0, batch_size=1))
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(before, model.actor.parameters())
            )
        )
        update = model.last_joint_update
        np.testing.assert_array_equal(update["current_movement_mask"][0], current_mask)
        np.testing.assert_array_equal(update["next_movement_mask"][0], next_mask)
        for action_name, mask in (
            ("actor_action", current_mask),
            ("target_actor_action", next_mask),
        ):
            blocks = update[action_name].reshape(-1, 16, 3)[0]
            hover = torch.tensor(HOVER_ACTION, dtype=blocks.dtype)
            self.assertTrue(torch.all(blocks[~torch.from_numpy(mask)] == hover))
            self.assertTrue(torch.any(blocks[torch.from_numpy(mask)] != hover))


if __name__ == "__main__":
    unittest.main()
