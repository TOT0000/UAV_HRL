import unittest

import numpy as np
import torch
from torch import nn

from DDQN import DDQN
from paper_metrics import aggregate_paper_point_metrics
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
        self.assertEqual(model.qos_target_probability, 0.1)
        self.assertEqual(model.lambda_cost, 0.0)
        self.assertEqual(model.eta_c, 0.01)

        unchanged = model.update_cost_multiplier(
            episode_violation_count=2,
            episode_eligible_packet_count=20,
        )
        self.assertEqual(unchanged, 0.0)
        self.assertEqual(model.cost_multiplier_update_count, 1)
        increased = model.update_cost_multiplier(
            episode_violation_count=3,
            episode_eligible_packet_count=20,
        )
        self.assertAlmostEqual(increased, 0.01 * 0.05)
        self.assertEqual(model.cost_multiplier_update_count, 2)
        decreased = model.update_cost_multiplier(
            episode_violation_count=0,
            episode_eligible_packet_count=20,
        )
        self.assertEqual(decreased, 0.0)
        self.assertEqual(model.cost_multiplier_update_count, 3)
        zero_eligible = model.update_cost_multiplier(
            episode_violation_count=0,
            episode_eligible_packet_count=0,
        )
        self.assertEqual(zero_eligible, 0.0)
        self.assertEqual(model.cost_multiplier_update_count, 3)
        state = model.constraint_state()
        self.assertEqual(state["lambda_update_scope"], "episode_end")
        self.assertEqual(state["cost_denominator"], "eligible_packets")
        self.assertEqual(state["last_episode_eligible_packet_count"], 0)
        self.assertIsNone(state["last_episode_violation_probability"])
        self.assertFalse(state["mid_episode_checkpoint_supported"])

    def test_paper_all_row_matches_episode_safe_ddqn_inputs(self):
        episode = {
            "timely_goodput_mbits": 1.0,
            "total_mobility_energy_j": 2.0,
            "fov_delivered_packets": 3,
            "fov_delivered_e2e_delay_sum_seconds": 0.3,
            "fov_generated_packets": 7,
            "fov_eligible_packets": 4,
            "fov_violation_packets": 1,
            "com_delivered_packets": 4,
            "com_delivered_e2e_delay_sum_seconds": 0.4,
            "com_generated_packets": 9,
            "com_eligible_packets": 6,
            "com_violation_packets": 2,
            "total_deadline_violations": 3,
            "eligible_packet_count": 10,
            "delay_violation_probability": 0.3,
        }
        rows = aggregate_paper_point_metrics(
            "td3_dinkelbach",
            "fixed_roi",
            {"point_id": "roi_2", "x_value": 2},
            [episode],
        )
        combined = next(
            row
            for row in rows
            if row["metric"] == "violation_probability"
            and row["task_type"] == "ALL"
        )
        self.assertEqual(combined["numerator"], episode["total_deadline_violations"])
        self.assertEqual(combined["denominator"], episode["eligible_packet_count"])
        self.assertEqual(combined["value"], episode["delay_violation_probability"])

        model = DDQN(state_dim=42, action_dim=3, hidden_dim=8)
        model.update_cost_multiplier(
            combined["numerator"], combined["denominator"]
        )
        state = model.constraint_state()
        self.assertEqual(
            state["last_episode_violation_probability"], combined["value"]
        )


if __name__ == "__main__":
    unittest.main()
