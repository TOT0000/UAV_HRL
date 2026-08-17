import unittest

import numpy as np

from DDQN import DDQN
from HRL_task_aware import _select_routing_actions
from exploration_schedules import (
    ddqn_decay_steps,
    ddqn_epsilon,
    evaluation_exploration_settings,
    td3_behavior_noise,
    td3_decay_steps,
)
from td3 import TD3


class ExplorationScheduleTest(unittest.TestCase):
    def test_td3_noise_start_mid_end_overrun_and_evaluation(self):
        self.assertAlmostEqual(td3_behavior_noise(0, 100), 0.20)
        self.assertAlmostEqual(td3_behavior_noise(50, 100), 0.125)
        self.assertAlmostEqual(td3_behavior_noise(100, 100), 0.05)
        self.assertAlmostEqual(td3_behavior_noise(150, 100), 0.05)
        self.assertEqual(td3_behavior_noise(0, 100, evaluation=True), 0.0)

    def test_ddqn_epsilon_start_mid_end_overrun_resume_and_evaluation(self):
        self.assertAlmostEqual(ddqn_epsilon(0, 100), 1.0)
        self.assertAlmostEqual(ddqn_epsilon(50, 100), 0.525)
        self.assertAlmostEqual(ddqn_epsilon(100, 100), 0.05)
        self.assertAlmostEqual(ddqn_epsilon(150, 100), 0.05)
        uninterrupted = [ddqn_epsilon(step, 100) for step in range(101)]
        resumed = [ddqn_epsilon(step, 100) for step in range(37, 101)]
        self.assertEqual(resumed, uninterrupted[37:])
        self.assertEqual(ddqn_epsilon(0, 100, evaluation=True), 0.0)

    def test_default_decay_horizons_cover_eighty_percent_of_formal_steps(self):
        self.assertEqual(td3_decay_steps(100, 60, 1000), 4000)
        self.assertEqual(ddqn_decay_steps(100, 60, 4), 19200)

    def test_same_routing_slot_uses_one_epsilon_and_zero_logits_noise(self):
        calls = []

        class RecordingDDQN:
            def select_action(self, state, uav_id, mask, **kwargs):
                calls.append((uav_id, kwargs["epsilon"], kwargs["logits_noise_std"]))
                return int(np.flatnonzero(mask)[0])

        states = {2: np.zeros(4), 0: np.zeros(4), 1: np.zeros(4)}
        masks = {uid: np.array([False, False, False, True]) for uid in states}
        actions = _select_routing_actions(
            RecordingDDQN(), states, masks, epsilon=0.375
        )

        self.assertEqual(actions, {0: 3, 1: 3, 2: 3})
        self.assertEqual(calls, [(0, 0.375, 0.0), (1, 0.375, 0.0), (2, 0.375, 0.0)])

    def test_td3_zero_noise_is_deterministic(self):
        model = TD3(state_dim=3, action_dim=2, max_action=1.0)
        state = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        first = model.select_action(state, noise_std=0.0)
        second = model.select_action(state, noise_std=0.0)
        np.testing.assert_array_equal(first, second)

    def test_ddqn_evaluation_is_deterministic(self):
        model = DDQN(state_dim=38, action_dim=3, hidden_dim=8)
        state = np.zeros(38, dtype=np.float32)
        mask = np.array([False, True, True])
        epsilon = ddqn_epsilon(0, 100, evaluation=True)
        first = model.select_action(
            state,
            uav_id=0,
            mask=mask,
            epsilon=epsilon,
            logits_noise_std=0.0,
        )
        second = model.select_action(
            state,
            uav_id=0,
            mask=mask,
            epsilon=epsilon,
            logits_noise_std=0.0,
        )
        self.assertEqual(first, second)

    def test_evaluation_settings_disable_all_behavior_exploration(self):
        self.assertEqual(
            evaluation_exploration_settings(),
            {
                "td3_behavior_noise": 0.0,
                "ddqn_epsilon": 0.0,
                "ddqn_logits_noise_std": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
