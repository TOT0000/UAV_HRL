import unittest

import numpy as np
import torch
from torch import nn

from DDQN import DDQN
from utils_update_v2 import ReplayBufferDiscrete


class FixedNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values[: inputs.shape[0]]


class SafeDDQNTargetTest(unittest.TestCase):
    def test_negative_cost_gets_no_bonus_and_matches_execution_scoring(self):
        model = DDQN.__new__(DDQN)
        model.action_dim = 3
        model.gamma = 0.9
        model.eta = 1.0

        model.q_network = FixedNetwork([[0.0, 1.0, 2.0]])
        model.cost_network = FixedNetwork([[0.0, -10.0, 0.0]])
        model.target_q_network = FixedNetwork([[10.0, 20.0, 30.0]])
        model.target_cost_network = FixedNetwork([[40.0, 50.0, 60.0]])

        next_state = torch.zeros((1, 42), dtype=torch.float32)
        next_state[0, 0] = 1.0
        next_state[0, 10:13] = torch.tensor([0.0, 1.0, 1.0])

        execution_action = model.select_action(
            next_state[0].numpy(),
            uav_id=0,
            mask=np.array([False, True, True]),
            epsilon=0.0,
            logits_noise_std=0.0,
        )
        target_q, target_c, target_actions = model._safe_targets(
            next_state,
            reward=torch.tensor([[2.0]]),
            cost=torch.tensor([[-4.0]]),
            not_done=torch.tensor([[0.0]]),
        )

        # Without clamping, action 1 would receive a +10 bonus from negative cost.
        raw_cost_scores = torch.tensor([0.0, 1.0, 2.0]) - torch.tensor(
            [0.0, -10.0, 0.0]
        )
        self.assertEqual(raw_cost_scores.argmax().item(), 1)
        self.assertEqual(execution_action, 2)
        self.assertEqual(target_actions.item(), 2)
        self.assertEqual(target_q.item(), 2.0)
        self.assertEqual(target_c.item(), -4.0)

    def test_masked_online_selection_is_shared_by_both_targets(self):
        model = DDQN.__new__(DDQN)
        model.action_dim = 3
        model.gamma = 0.9
        model.eta = 1.0

        model.q_network = FixedNetwork([[1.0, 100.0, 8.0], [1.0, 100.0, 8.0]])
        model.cost_network = FixedNetwork([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        model.target_q_network = FixedNetwork(
            [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]
        )
        model.target_cost_network = FixedNetwork(
            [[40.0, 99.0, 60.0], [40.0, 99.0, 60.0]]
        )

        # Task-aware state layout: 6N + 30, with N=2 and mask starting at N+8.
        next_state = torch.zeros((2, 42), dtype=torch.float32)
        next_state[0, 0] = 1.0
        next_state[1, 1] = 1.0
        next_state[:, 10:13] = torch.tensor([1.0, 0.0, 1.0])

        reward = torch.tensor([[2.0], [2.0]])
        cost = torch.tensor([[3.0], [3.0]])
        not_done = torch.tensor([[1.0], [0.0]])

        target_q, target_c, next_actions = model._safe_targets(
            next_state, reward, cost, not_done
        )

        # Action 1 has the highest safe score but is illegal, so action 2 is shared.
        self.assertEqual(next_actions.tolist(), [[2], [2]])
        self.assertTrue(torch.allclose(target_q, torch.tensor([29.0, 2.0])))
        self.assertTrue(torch.allclose(target_c, torch.tensor([57.0, 3.0])))

    def test_wait_is_legal_and_empty_target_mask_is_rejected(self):
        model = DDQN.__new__(DDQN)
        model.action_dim = 3
        model.gamma = 0.9
        model.eta = 1.0
        model.q_network = FixedNetwork([[100.0, 1.0, 200.0]])
        model.cost_network = FixedNetwork([[0.0, 0.0, 0.0]])
        model.target_q_network = FixedNetwork([[10.0, 20.0, 30.0]])
        model.target_cost_network = FixedNetwork([[40.0, 50.0, 60.0]])

        next_state = torch.zeros((1, 42), dtype=torch.float32)
        next_state[0, 1] = 1.0
        next_state[0, 10:13] = torch.tensor([0.0, 1.0, 0.0])
        action_mask = model._routing_action_mask(next_state)
        self.assertEqual(action_mask.tolist(), [[False, True, False]])
        self.assertEqual(
            model.select_action(
                next_state[0].numpy(),
                uav_id=1,
                mask=np.array([False, True, False]),
                epsilon=0.0,
                logits_noise_std=0.0,
            ),
            1,
        )
        _, _, next_actions = model._safe_targets(
            next_state,
            reward=torch.zeros((1, 1)),
            cost=torch.zeros((1, 1)),
            not_done=torch.ones((1, 1)),
        )
        self.assertEqual(next_actions.item(), 1)

        next_state[0, 10:13] = 0.0
        with self.assertRaisesRegex(ValueError, "no legal action"):
            model._routing_action_mask(next_state)

    def test_single_training_step_smoke(self):
        model = DDQN(state_dim=42, action_dim=3, hidden_dim=8)
        replay = ReplayBufferDiscrete(
            state_dim=42, action_dim=3, max_size=4, n_step=1
        )

        state = np.zeros(42, dtype=np.float32)
        state[0] = 1.0
        next_state = state.copy()
        next_state[10:13] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        replay.add(state, 0, next_state, reward=1.0, cost=0.5, done=False)

        model.train(replay, batch_size=1)

        self.assertEqual(len(model.loss_log), 1)
        self.assertEqual(len(model.cost_loss_log), 1)


if __name__ == "__main__":
    unittest.main()
