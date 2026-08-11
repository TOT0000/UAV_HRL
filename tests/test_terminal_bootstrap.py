import unittest

import numpy as np
import torch

from DDQN import _bellman_target as ddqn_bellman_target
from td3 import _bellman_target as td3_bellman_target
from utils_update_v2 import ReplayBufferContinuous, ReplayBufferDiscrete


class TerminalBootstrapTest(unittest.TestCase):
    def test_td3_replay_flag_and_target_semantics(self):
        for done, expected_not_done, expected_target in (
            (False, 1.0, 1.0 + 0.9 * 5.0),
            (True, 0.0, 1.0),
        ):
            with self.subTest(done=done):
                replay = ReplayBufferContinuous(
                    state_dim=2, action_dim=1, max_size=4, n_step=1, gamma=0.9
                )
                replay.device = torch.device("cpu")
                replay.add(
                    np.array([0.0, 0.0]),
                    np.array([0.0]),
                    np.array([1.0, 1.0]),
                    reward=1.0,
                    done=done,
                    tag_gt=0,
                )

                _, _, _, reward, not_done, _ = replay.sample_by_tag(
                    batch_size=1, curr_tag=0
                )
                target = td3_bellman_target(
                    reward, torch.tensor([[5.0]]), not_done, discount=0.9
                )

                self.assertEqual(not_done.item(), expected_not_done)
                self.assertAlmostEqual(target.item(), expected_target)

    def test_safe_ddqn_replay_flag_and_reward_cost_targets(self):
        for done, expected_not_done, expected_reward, expected_cost in (
            (False, 1.0, 2.0 + 0.9 * 5.0, 3.0 + 0.9 * 7.0),
            (True, 0.0, 2.0, 3.0),
        ):
            with self.subTest(done=done):
                replay = ReplayBufferDiscrete(
                    state_dim=2, action_dim=2, max_size=4, n_step=1, gamma=0.9
                )
                replay.device = torch.device("cpu")
                replay.add(
                    np.array([0.0, 0.0]),
                    0,
                    np.array([1.0, 1.0]),
                    reward=2.0,
                    cost=3.0,
                    done=done,
                )

                _, _, _, reward, cost, not_done = replay.sample(batch_size=1)
                reward_target = ddqn_bellman_target(
                    reward.squeeze(1),
                    torch.tensor([5.0]),
                    not_done.squeeze(1),
                    discount=0.9,
                )
                cost_target = ddqn_bellman_target(
                    cost.squeeze(1),
                    torch.tensor([7.0]),
                    not_done.squeeze(1),
                    discount=0.9,
                )

                self.assertEqual(not_done.item(), expected_not_done)
                self.assertAlmostEqual(reward_target.item(), expected_reward, delta=1e-6)
                self.assertAlmostEqual(cost_target.item(), expected_cost, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
