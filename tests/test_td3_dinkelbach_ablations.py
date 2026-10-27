import unittest

import numpy as np

from dinkelbach_blocks import DinkelbachBlockState
from evaluation_metrics import METRIC_COLUMNS
from experiment_config import (
    MethodSpec,
    effective_training_config,
    movement_agent_configuration,
    resolve_method_training_config,
)
from HRL_task_aware import _interval_reward, formal_training_config
from paper_evaluation import evaluation_method_supported
from run_experiment import build_parser
from utils_update_v2 import ReplayBufferJoint


BASE_METHOD = "td3_dinkelbach"
NO_C10_METHOD = "td3_dinkelbach_wo_c10_penalty"
INTERVAL_200_METHOD = "td3_dinkelbach_interval_200"


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add(prefix + (key,))
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], prefix + (key,))
                )
        return differences
    return set() if left == right else {prefix}


class AblationRegistryAndConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = formal_training_config(1500)
        self.base = MethodSpec.parse(BASE_METHOD)
        self.no_c10 = MethodSpec.parse(NO_C10_METHOD)
        self.interval_200 = MethodSpec.parse(INTERVAL_200_METHOD)

    def test_registry_cli_and_evaluation_accept_both_methods(self):
        parser = build_parser()
        for method_id, expected_label in (
            (NO_C10_METHOD, "TD3-Dinkelbach w/o C10 Penalty"),
            (
                INTERVAL_200_METHOD,
                "TD3-Dinkelbach (200-Episode Lambda Update)",
            ),
        ):
            with self.subTest(method=method_id):
                spec = MethodSpec.parse(method_id)
                self.assertEqual(spec.label, expected_label)
                self.assertEqual(
                    parser.parse_args([method_id, "--smoke"]).method,
                    spec,
                )
                for suite in (
                    "uav_trajectory_snapshots",
                    "task_type_delay_vs_arrival_rate",
                    "task_type_delay_violation_vs_target_delay",
                ):
                    self.assertTrue(evaluation_method_supported(method_id, suite))

    def test_resolved_configurations_are_single_factor_ablations(self):
        baseline = effective_training_config(self.config, self.base)
        no_c10 = effective_training_config(self.config, self.no_c10)
        interval_200 = effective_training_config(self.config, self.interval_200)

        self.assertEqual(
            _leaf_differences(baseline, no_c10),
            {
                ("c10_penalty_enabled",),
                ("effective_movement_constraint_penalty_weights", "c10"),
                ("task_potential_configuration", "reward_form"),
                (
                    "task_potential_configuration",
                    "c10",
                    "reward_penalty_enabled",
                ),
            },
        )
        self.assertEqual(
            _leaf_differences(baseline, interval_200),
            {("dinkelbach_update_interval_episodes",)},
        )
        self.assertEqual(no_c10["dinkelbach_update_interval_episodes"], 50)
        self.assertFalse(no_c10["c10_penalty_enabled"])
        self.assertEqual(
            no_c10["effective_movement_constraint_penalty_weights"],
            {"c9": 1.0, "c10": 0.0, "com_range": 1.0},
        )
        self.assertEqual(interval_200["dinkelbach_update_interval_episodes"], 200)
        self.assertEqual(
            interval_200["effective_movement_constraint_penalty_weights"]["c10"],
            1.0,
        )
        self.assertEqual(
            movement_agent_configuration(self.base),
            movement_agent_configuration(self.no_c10),
        )
        self.assertEqual(
            movement_agent_configuration(self.base),
            movement_agent_configuration(self.interval_200),
        )

    def test_method_runtime_config_and_checkpoint_identity_are_distinct(self):
        baseline = resolve_method_training_config(self.config, self.base)
        no_c10 = resolve_method_training_config(self.config, self.no_c10)
        interval_200 = resolve_method_training_config(self.config, self.interval_200)

        self.assertEqual(baseline.dinkelbach_update_interval_episodes, 50)
        self.assertEqual(no_c10.dinkelbach_update_interval_episodes, 50)
        self.assertEqual(interval_200.dinkelbach_update_interval_episodes, 200)
        self.assertEqual(len({
            self.base.fingerprint,
            self.no_c10.fingerprint,
            self.interval_200.fingerprint,
        }), 3)


class C10PenaltyAblationRewardTest(unittest.TestCase):
    def setUp(self):
        self.delivered = 7.0
        self.energy = 4.0
        self.current_lambda = 0.5
        self.penalties = {
            "c9_penalty_mean": 0.25,
            "c10_penalty_mean": 0.75,
            "com_range_penalty_mean": 0.5,
        }

    def _live_reward(self, c10_penalty_enabled):
        return _interval_reward(
            self.delivered,
            self.energy,
            self.current_lambda,
            self.penalties,
            c10_penalty_enabled=c10_penalty_enabled,
        )

    def test_live_and_replay_rewards_remove_only_nonzero_c10_penalty(self):
        baseline_live = self._live_reward(True)
        no_c10_live = self._live_reward(False)
        self.assertAlmostEqual(baseline_live, 3.5)
        self.assertAlmostEqual(no_c10_live, 4.25)
        self.assertAlmostEqual(no_c10_live - baseline_live, 0.75)

        replay = ReplayBufferJoint(1, 1, max_size=2)
        replay.add(
            [0.0],
            [0.0],
            [1.0],
            done=False,
            delivered_mbits=self.delivered,
            total_mobility_energy=self.energy,
            c9_penalty=self.penalties["c9_penalty_mean"],
            c10_penalty=self.penalties["c10_penalty_mean"],
            com_range_penalty=self.penalties["com_range_penalty_mean"],
        )
        baseline_replay = replay._reward_numpy(
            np.asarray([0]),
            current_lambda=self.current_lambda,
            gamma=1.0,
            c10_penalty_enabled=True,
        )[0, 0]
        no_c10_replay = replay._reward_numpy(
            np.asarray([0]),
            current_lambda=self.current_lambda,
            gamma=1.0,
            c10_penalty_enabled=False,
        )[0, 0]

        self.assertAlmostEqual(float(baseline_replay), baseline_live)
        self.assertAlmostEqual(float(no_c10_replay), no_c10_live)
        self.assertAlmostEqual(float(replay.c10_penalty[0, 0]), 0.75)
        self.assertGreater(float(replay.c10_penalty[0, 0]), 0.0)

    def test_c10_observation_auxiliary_and_metric_contracts_remain_present(self):
        spec = MethodSpec.parse(NO_C10_METHOD)
        resolved = effective_training_config(formal_training_config(1), spec)
        self.assertEqual(spec.task_observation, "full")
        self.assertIn("c10", resolved["task_potential_configuration"])
        self.assertIn("c10_penalty", ReplayBufferJoint(1, 1).all_fields)
        for metric in (
            "c10_penalty_sum",
            "c10_penalty_mean",
            "c10_penalty_sample_count",
            "c10_satisfied_pair_count",
            "c10_violated_pair_count",
        ):
            self.assertIn(metric, METRIC_COLUMNS)


class DinkelbachIntervalAblationTest(unittest.TestCase):
    def _config(self, method_id):
        return effective_training_config(
            formal_training_config(1500), MethodSpec.parse(method_id)
        )

    def test_baseline_50_episode_schedule_is_unchanged(self):
        state = DinkelbachBlockState.from_config(self._config(BASE_METHOD))
        updates = []
        for episode in range(1, 201):
            event = state.record_episode(1.0, 2.0)
            if event["dinkelbach_lambda_updated"]:
                updates.append(episode)
        self.assertEqual(updates, [50, 100, 150, 200])

    def test_200_episode_schedule_ratio_of_sums_and_final_partial_block(self):
        state = DinkelbachBlockState.from_config(self._config(INTERVAL_200_METHOD))
        updates = []
        first_boundary = None
        for episode in range(1, 1501):
            if episode == 1:
                timely, energy = 10.0, 1.0
            elif episode == 2:
                timely, energy = 0.0, 9.0
            else:
                timely, energy = 1.0, 2.0
            event = state.record_episode(timely, energy)
            if event["dinkelbach_lambda_updated"]:
                updates.append(episode)
                if episode == 200:
                    first_boundary = event

        self.assertEqual(updates, [200, 400, 600, 800, 1000, 1200, 1400])
        self.assertIsNotNone(first_boundary)
        self.assertAlmostEqual(
            first_boundary["dinkelbach_lambda_after_episode"],
            (10.0 + 198.0) / (1.0 + 9.0 + 396.0),
        )
        self.assertEqual(state.update_count, 7)
        self.assertEqual(state.block_index, 8)
        self.assertEqual(state.block_completed_episodes, 100)
        self.assertEqual(state.block_timely_delivered_mbits, 100.0)
        self.assertEqual(state.block_mobility_energy_joules, 200.0)

    def test_partial_block_resume_preserves_state_and_rejects_interval_mismatch(self):
        config_200 = self._config(INTERVAL_200_METHOD)
        config_50 = self._config(BASE_METHOD)
        for completed in (150, 350):
            with self.subTest(completed=completed):
                uninterrupted = DinkelbachBlockState.from_config(config_200)
                for episode in range(1, completed + 1):
                    uninterrupted.record_episode(episode / 10.0, episode + 1.0)
                saved = uninterrupted.training_state()
                restored = DinkelbachBlockState.from_training_state(
                    saved,
                    config_200,
                    expected_completed_episodes=completed,
                )
                self.assertEqual(restored.training_state(), saved)

                next_update = ((completed // 200) + 1) * 200
                for episode in range(completed + 1, next_update + 1):
                    sample = (episode / 10.0, episode + 1.0)
                    expected_event = uninterrupted.record_episode(*sample)
                    restored_event = restored.record_episode(*sample)
                    self.assertEqual(restored_event, expected_event)
                    self.assertEqual(
                        restored_event["dinkelbach_lambda_updated"],
                        episode == next_update,
                    )

                with self.assertRaisesRegex(RuntimeError, "update interval"):
                    DinkelbachBlockState.from_training_state(
                        saved,
                        config_50,
                        expected_completed_episodes=completed,
                    )


if __name__ == "__main__":
    unittest.main()
