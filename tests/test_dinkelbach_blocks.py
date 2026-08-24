import math
import unittest

from dinkelbach_blocks import (
    DINKELBACH_INITIAL_LAMBDA,
    DINKELBACH_UPDATE_INTERVAL_EPISODES,
    DinkelbachBlockState,
    dinkelbach_config_metadata,
    dinkelbach_full_block_count,
)


class DinkelbachBlockUpdateTest(unittest.TestCase):
    def test_initial_lambda_and_formal_schedule(self):
        state = DinkelbachBlockState()

        self.assertEqual(state.current_lambda, 0.0)
        self.assertEqual(DINKELBACH_INITIAL_LAMBDA, 0.0)
        self.assertEqual(DINKELBACH_UPDATE_INTERVAL_EPISODES, 50)
        self.assertEqual(dinkelbach_full_block_count(1500, 50), 30)

    def test_episodes_1_through_49_hold_lambda_and_50_updates_once(self):
        state = DinkelbachBlockState()
        for episode in range(1, 50):
            event = state.record_episode(2.0, 4.0)
            self.assertEqual(event["dinkelbach_lambda_used"], 0.0)
            self.assertEqual(event["dinkelbach_lambda_after_episode"], 0.0)
            self.assertFalse(event["dinkelbach_lambda_updated"])
            self.assertEqual(event["dinkelbach_block_episode"], episode)

        boundary = state.record_episode(2.0, 4.0)

        self.assertTrue(boundary["dinkelbach_lambda_updated"])
        self.assertEqual(boundary["dinkelbach_update_status"], "updated")
        self.assertEqual(boundary["dinkelbach_lambda_used"], 0.0)
        self.assertEqual(boundary["dinkelbach_lambda_after_episode"], 0.5)
        self.assertEqual(state.current_lambda, 0.5)
        self.assertEqual(state.update_count, 1)
        self.assertEqual(state.block_index, 2)
        self.assertEqual(state.block_completed_episodes, 0)

        episode_51 = state.record_episode(3.0, 6.0)
        self.assertEqual(episode_51["dinkelbach_lambda_used"], 0.5)
        self.assertEqual(episode_51["dinkelbach_block_index"], 2)
        self.assertEqual(episode_51["dinkelbach_block_episode"], 1)
        self.assertFalse(episode_51["dinkelbach_lambda_updated"])

    def test_episode_100_performs_the_second_update(self):
        state = DinkelbachBlockState()
        for _ in range(50):
            state.record_episode(1.0, 2.0)
        for _ in range(49):
            event = state.record_episode(4.0, 2.0)
            self.assertFalse(event["dinkelbach_lambda_updated"])

        boundary = state.record_episode(4.0, 2.0)

        self.assertEqual(boundary["dinkelbach_block_index"], 2)
        self.assertEqual(boundary["dinkelbach_block_episode"], 50)
        self.assertEqual(boundary["dinkelbach_lambda_used"], 0.5)
        self.assertEqual(boundary["dinkelbach_lambda_after_episode"], 2.0)
        self.assertEqual(state.update_count, 2)

    def test_update_is_ratio_of_sums_not_mean_of_episode_ratios(self):
        state = DinkelbachBlockState(update_interval_episodes=2)

        first = state.record_episode(10.0, 1.0)
        second = state.record_episode(0.0, 9.0)

        self.assertFalse(first["dinkelbach_lambda_updated"])
        self.assertEqual(second["dinkelbach_lambda_after_episode"], 1.0)
        self.assertNotEqual(second["dinkelbach_lambda_after_episode"], 5.0)
        self.assertEqual(second["dinkelbach_block_timely_mbits_so_far"], 10.0)
        self.assertEqual(second["dinkelbach_block_energy_joules_so_far"], 10.0)

    def test_invalid_energy_never_produces_non_finite_lambda(self):
        for invalid_energy, expected_status in (
            (0.0, "invalid_denominator"),
            (-1.0, "invalid_block_inputs"),
            (float("nan"), "invalid_block_inputs"),
            (float("inf"), "invalid_block_inputs"),
        ):
            with self.subTest(energy=invalid_energy):
                state = DinkelbachBlockState(
                    current_lambda=0.25,
                    initial_lambda=0.25,
                    update_interval_episodes=1,
                )

                event = state.record_episode(10.0, invalid_energy)

                self.assertEqual(state.current_lambda, 0.25)
                self.assertTrue(math.isfinite(state.current_lambda))
                self.assertFalse(event["dinkelbach_lambda_updated"])
                self.assertEqual(event["dinkelbach_update_status"], expected_status)
                self.assertTrue(
                    math.isfinite(
                        event["dinkelbach_block_energy_joules_so_far"]
                    )
                )

    def test_smoke_and_75_episode_partial_block_are_not_forced(self):
        smoke = DinkelbachBlockState()
        smoke_event = smoke.record_episode(1.0, 2.0)
        self.assertFalse(smoke_event["dinkelbach_lambda_updated"])
        self.assertEqual(smoke.update_count, 0)

        state = DinkelbachBlockState()
        events = [state.record_episode(1.0, 2.0) for _ in range(75)]
        self.assertEqual(sum(event["dinkelbach_lambda_updated"] for event in events), 1)
        self.assertEqual(state.update_count, 1)
        self.assertEqual(state.block_index, 2)
        self.assertEqual(state.block_completed_episodes, 25)
        self.assertEqual(state.block_timely_delivered_mbits, 25.0)
        self.assertEqual(state.block_mobility_energy_joules, 50.0)


class DinkelbachBlockStateSerializationTest(unittest.TestCase):
    def _config(self, interval=50, initial=0.0):
        config = dinkelbach_config_metadata()
        config["dinkelbach_update_interval_episodes"] = interval
        config["dinkelbach_initial_lambda"] = initial
        return config

    def test_mid_block_resume_matches_uninterrupted_state(self):
        uninterrupted = DinkelbachBlockState()
        for episode in range(25):
            uninterrupted.record_episode(episode + 1.0, episode + 2.0)
        restored = DinkelbachBlockState.from_training_state(
            uninterrupted.training_state(),
            self._config(),
            expected_completed_episodes=25,
        )

        for episode in range(25, 75):
            sample = (episode + 1.0, episode + 2.0)
            uninterrupted_event = uninterrupted.record_episode(*sample)
            restored_event = restored.record_episode(*sample)
            self.assertEqual(restored_event, uninterrupted_event)

        self.assertEqual(restored.training_state(), uninterrupted.training_state())

    def test_boundary_resume_does_not_repeat_update(self):
        state = DinkelbachBlockState()
        for _ in range(50):
            state.record_episode(2.0, 4.0)
        self.assertEqual(state.update_count, 1)
        restored = DinkelbachBlockState.from_training_state(
            state.training_state(),
            self._config(),
            expected_completed_episodes=50,
        )

        episode_51 = restored.record_episode(2.0, 4.0)

        self.assertFalse(episode_51["dinkelbach_lambda_updated"])
        self.assertEqual(episode_51["dinkelbach_lambda_used"], 0.5)
        self.assertEqual(restored.update_count, 1)
        self.assertEqual(restored.block_index, 2)
        self.assertEqual(restored.block_completed_episodes, 1)

    def test_old_state_without_block_fields_is_explicitly_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing Dinkelbach block state"):
            DinkelbachBlockState.from_training_state(
                {"lambda_EE_global": 0.1}, self._config()
            )


if __name__ == "__main__":
    unittest.main()
