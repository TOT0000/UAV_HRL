import inspect
import math
import unittest

from HRL_task_aware import (
    MOVEMENT_CONTROL_INTERVAL,
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_POLICY_DELAY,
    PRODUCTION_WARMUP_TRANSITIONS,
    ROUTING_STATE_DIM,
    TrainingConfig,
    _dinkelbach_update,
    _interval_reward,
    train,
)


class CentralizedTrainingFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train(
            TrainingConfig(
                total_episodes=1,
                episode_seconds=2,
                warmup_joint_transitions=0,
                batch_size=1,
                enable_checkpoints=False,
                enable_plots=False,
                enable_csv=False,
                random_seed=2026,
            )
        )

    def test_production_defaults_and_state_separation(self):
        defaults = TrainingConfig(total_episodes=6)
        self.assertEqual(defaults.episode_seconds, 60)
        self.assertEqual(defaults.warmup_joint_transitions, PRODUCTION_WARMUP_TRANSITIONS)
        self.assertEqual(defaults.batch_size, PRODUCTION_BATCH_SIZE)
        self.assertEqual(defaults.policy_delay, PRODUCTION_POLICY_DELAY)
        self.assertEqual(ROUTING_STATE_DIM, 122)
        self.assertEqual(self.result["movement_state_dim"], 532)
        self.assertEqual(self.result["routing_state_dim"], 122)
        self.assertEqual(self.result["joint_action_dim"], 48)
        self.assertEqual(self.result["centralized_td3_gamma"], 1.0)
        self.assertEqual(self.result["routing_ddqn_gamma"], 0.99)

    def test_one_actor_call_and_one_transition_per_interval(self):
        self.assertEqual(self.result["environment_actor_calls"], 2)
        self.assertEqual(self.result["proposal_batches"], 2)
        self.assertEqual(self.result["joint_transitions"], 2)
        self.assertEqual(self.result["joint_replay_size"], 2)
        self.assertEqual(
            self.result["routing_slots_executed"],
            2 * MOVEMENT_CONTROL_INTERVAL,
        )

    def test_energy_terminal_and_update_counts(self):
        self.assertEqual(self.result["energy_evaluations"], 2 * 16)
        self.assertEqual(self.result["terminal_joint_transitions"], 1)
        self.assertEqual(self.result["critic_updates"], 2)
        self.assertEqual(self.result["actor_updates"], 1)
        self.assertGreater(self.result["energy_log"][0], 0.0)
        self.assertEqual(self.result["duplicate_target_assertions"], 0)

    def test_smoke_outputs_are_finite(self):
        for key in ("lambda",):
            self.assertTrue(math.isfinite(self.result[key]))
        for series in ("reward_log", "delivered_log", "energy_log"):
            self.assertTrue(all(math.isfinite(value) for value in self.result[series]))

    def test_old_task_specific_td3_is_not_in_active_train(self):
        source = inspect.getsource(train)
        self.assertNotIn("Model_TD3_search", source)
        self.assertNotIn("Model_TD3_fov", source)
        self.assertNotIn("ReplayBufferContinuous", source)

    def test_zero_energy_dinkelbach_protection_and_unclipped_ratio(self):
        self.assertEqual(_dinkelbach_update(10.0, 0.0, 0.25), 0.25)
        self.assertEqual(_dinkelbach_update(10.0, 2.0, 0.25), 5.0)

    def test_finite_horizon_potential_shaping_uses_unit_discount(self):
        config = TrainingConfig(total_episodes=1)
        reward = _interval_reward(
            delivered_mbits=0.0,
            energy=0.0,
            current_lambda=0.0,
            gamma=1.0,
            potentials_t=(0.1, 0.2, 0.3),
            potentials_t1=(0.4, 0.5, 0.6),
            done=False,
            config=config,
        )
        self.assertAlmostEqual(reward, 0.9)


if __name__ == "__main__":
    unittest.main()
