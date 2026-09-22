import unittest

import numpy as np
import torch
from torch import nn

from DDQN import device
from experiment_config import (
    MethodSpec,
    routing_agent_configuration,
)
from paper_evaluation import evaluation_method_supported
from paper_figure_registry import METHOD_DISPLAY_NAMES, PAPER_METHOD_MAPPINGS, PLOT_STYLES
from routing_agents import ControlledDDQN, ControlledDQN, create_routing_agent
from training_checkpoint import (
    _restore_routing_training_payload,
    _routing_training_payload,
)


class FixedQNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, state):
        return self.values.unsqueeze(0).expand(state.shape[0], -1)


def routing_state_batch(masks):
    """Build legacy-layout routing states with the supplied effective masks."""

    states = torch.zeros((len(masks), 30), dtype=torch.float32)
    states[:, 9:12] = torch.tensor(masks, dtype=torch.float32)
    return states.to(device)


class RoutingTargetTest(unittest.TestCase):
    def test_dqn_and_ddqn_use_distinct_targets_and_terminal_cutoff(self):
        dqn = ControlledDQN(30, 3, gamma=0.9)
        ddqn = ControlledDDQN(30, 3, gamma=0.9)
        for agent in (dqn, ddqn):
            agent.q_network = FixedQNetwork([1.0, 10.0, 5.0]).to(device)
            agent.target_q_network = FixedQNetwork([7.0, 2.0, 3.0]).to(device)

        next_state = routing_state_batch([[1, 1, 1], [1, 1, 1]])
        reward = torch.tensor([[1.0], [4.0]], device=device)
        not_done = torch.tensor([[1.0], [0.0]], device=device)

        dqn_target = dqn._standard_targets(next_state, reward, not_done)
        ddqn_target = ddqn._standard_targets(next_state, reward, not_done)

        self.assertTrue(
            torch.allclose(dqn_target, torch.tensor([7.3, 4.0], device=device))
        )
        self.assertTrue(
            torch.allclose(ddqn_target, torch.tensor([2.8, 4.0], device=device))
        )
        self.assertNotEqual(float(dqn_target[0]), float(ddqn_target[0]))

    def test_ddqn_online_argmax_uses_the_baseline_action_mask(self):
        dqn = ControlledDQN(30, 3, gamma=0.5)
        ddqn = ControlledDDQN(30, 3, gamma=0.5)
        for agent in (dqn, ddqn):
            agent.q_network = FixedQNetwork([1.0, 100.0, 5.0]).to(device)
            agent.target_q_network = FixedQNetwork([7.0, 200.0, 3.0]).to(device)

        next_state = routing_state_batch([[1, 0, 1]])
        reward = torch.tensor([[0.0]], device=device)
        not_done = torch.tensor([[1.0]], device=device)

        self.assertEqual(
            float(dqn._standard_targets(next_state, reward, not_done)[0]), 3.5
        )
        self.assertEqual(
            float(ddqn._standard_targets(next_state, reward, not_done)[0]), 1.5
        )
        state = np.zeros(30, dtype=np.float32)
        mask = np.array([True, False, True])
        self.assertEqual(dqn.select_action(state, 0, mask=mask, epsilon=0.0), 2)
        self.assertEqual(ddqn.select_action(state, 0, mask=mask, epsilon=0.0), 2)
        self.assertFalse(hasattr(ddqn, "cost_network"))
        self.assertFalse(hasattr(ddqn, "lambda_cost"))

    def test_ddqn_can_complete_a_replay_and_target_update(self):
        class Replay:
            def sample(self, batch_size):
                self.batch_size = batch_size
                state = routing_state_batch([[1, 1, 1], [1, 0, 1]])
                return (
                    state,
                    torch.tensor([0, 2], dtype=torch.long, device=device),
                    state.clone(),
                    torch.tensor([[1.0], [2.0]], device=device),
                    torch.zeros((2, 1), device=device),
                    torch.tensor([[1.0], [0.0]], device=device),
                )

        replay = Replay()
        agent = ControlledDDQN(30, 3)
        agent.train(replay, batch_size=2)
        agent.update_target()
        self.assertEqual(replay.batch_size, 2)
        self.assertEqual(agent.num_training, 1)
        self.assertEqual(agent.target_update_count, 1)
        self.assertEqual(agent.reward_optimizer_update_count, 1)
        self.assertEqual(agent.reward_target_update_count, 1)


class DDQNMethodIsolationTest(unittest.TestCase):
    def test_dqn_ddqn_and_safe_ddqn_are_separate_configurations(self):
        dqn = MethodSpec.parse("td3_dinkelbach_dqn")
        ddqn = MethodSpec.parse("td3_dinkelbach_ddqn")
        safe = MethodSpec.parse("td3_dinkelbach")
        self.assertEqual((dqn.routing, ddqn.routing, safe.routing), (
            "dqn", "ddqn", "safe_ddqn"
        ))

        dqn_spec = dqn.to_dict()
        ddqn_spec = ddqn.to_dict()
        excluded = {"method_id", "method_key", "routing", "label"}
        self.assertEqual(
            {key: value for key, value in dqn_spec.items() if key not in excluded},
            {key: value for key, value in ddqn_spec.items() if key not in excluded},
        )
        dqn_config = routing_agent_configuration(dqn)
        ddqn_config = routing_agent_configuration(ddqn)
        self.assertEqual(
            {key: value for key, value in dqn_config.items() if key != "routing_agent_kind"},
            {key: value for key, value in ddqn_config.items() if key != "routing_agent_kind"},
        )
        self.assertNotEqual(dqn.fingerprint, ddqn.fingerprint)

        dqn_agent = create_routing_agent(dqn, 30, 3)
        ddqn_agent = create_routing_agent(ddqn, 30, 3)
        safe_agent = create_routing_agent(safe, 30, 3)
        self.assertIsInstance(dqn_agent, ControlledDQN)
        self.assertNotIsInstance(dqn_agent, ControlledDDQN)
        self.assertIsInstance(ddqn_agent, ControlledDDQN)
        self.assertTrue(hasattr(safe_agent, "cost_network"))

    def test_resume_payload_rejects_dqn_ddqn_cross_loading(self):
        dqn = create_routing_agent(MethodSpec.parse("td3_dinkelbach_dqn"), 30, 3)
        ddqn = create_routing_agent(MethodSpec.parse("td3_dinkelbach_ddqn"), 30, 3)
        with self.assertRaisesRegex(RuntimeError, "kind is incompatible"):
            _restore_routing_training_payload(ddqn, _routing_training_payload(dqn))
        with self.assertRaisesRegex(RuntimeError, "kind is incompatible"):
            _restore_routing_training_payload(dqn, _routing_training_payload(ddqn))

    def test_evaluation_and_plot_registration_do_not_change_defaults(self):
        method_id = "td3_dinkelbach_ddqn"
        self.assertTrue(
            evaluation_method_supported(method_id, "task_type_delay_vs_arrival_rate")
        )
        self.assertTrue(
            evaluation_method_supported(
                method_id, "task_type_delay_violation_vs_target_delay"
            )
        )
        self.assertTrue(evaluation_method_supported(method_id, "environment_size"))
        self.assertFalse(
            evaluation_method_supported(method_id, "uav_trajectory_snapshots")
        )
        self.assertNotIn(
            method_id,
            PAPER_METHOD_MAPPINGS["task_type_delay_vs_arrival_rate"].values(),
        )
        self.assertEqual(METHOD_DISPLAY_NAMES[method_id], "K-KM + TD3 + DDQN")
        self.assertEqual(
            PLOT_STYLES["task_type_delay_vs_arrival_rate"][method_id]["color"],
            "#D35400",
        )


if __name__ == "__main__":
    unittest.main()
