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
    def test_exact_unclamped_lagrangian_score_matches_execution_and_target(self):
        model = DDQN.__new__(DDQN)
        model.action_dim = 3
        model.gamma = 0.9
        model.lambda_cost = 1.0

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

        # The required score is exactly Q - lambda*C, including learned negatives.
        raw_cost_scores = torch.tensor([0.0, 1.0, 2.0]) - torch.tensor(
            [0.0, -10.0, 0.0]
        )
        self.assertEqual(raw_cost_scores.argmax().item(), 1)
        self.assertEqual(execution_action, 1)
        self.assertEqual(target_actions.item(), 1)
        self.assertEqual(target_q.item(), 2.0)
        self.assertEqual(target_c.item(), -4.0)

    def test_masked_online_selection_is_shared_by_both_targets(self):
        model = DDQN.__new__(DDQN)
        model.action_dim = 3
        model.gamma = 0.9
        model.lambda_cost = 1.0

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
        model.lambda_cost = 1.0
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

    def test_constraint_defaults_and_episode_end_update_formula(self):
        model = DDQN(state_dim=42, action_dim=3, hidden_dim=8)
        self.assertEqual(model.qos_cost_budget, 12.0)
        self.assertEqual(model.lambda_cost, 0.0)
        self.assertEqual(model.eta_c, 0.01)

        increased = model.update_cost_multiplier(
            episode_cost_sum=20.0 * 240,
            episode_slot_steps=240,
        )
        self.assertAlmostEqual(increased, 0.08)
        self.assertEqual(model.cost_multiplier_update_count, 1)
        decreased = model.update_cost_multiplier(
            episode_cost_sum=0.0,
            episode_slot_steps=240,
        )
        self.assertEqual(decreased, 0.0)
        self.assertEqual(model.cost_multiplier_update_count, 2)
        state = model.constraint_state()
        self.assertEqual(state["lambda_update_scope"], "episode_end")
        self.assertEqual(state["cost_denominator"], "network_routing_slot_steps")
        self.assertFalse(state["mid_episode_checkpoint_supported"])


if __name__ == "__main__":
    unittest.main()
