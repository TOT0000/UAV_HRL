import unittest

import numpy as np

from HRL_task_aware import _interval_reward
from Simulator import Simulator
from centralized_movement import fov_task_geometry, movement_constraint_penalties
from experiment_config import (
    METHOD_REGISTRY,
    COM_PENALTY_REFERENCE_DISTANCE_M,
    MOVEMENT_CONSTRAINT_WEIGHT_COM,
    S2U_COMMUNICATION_RANGE_M,
    MethodSpec,
    comparison_method_configuration,
)
from utils_update_v2 import ReplayBufferJoint
from visual_sensing import VS_CAMERA, vs_c10_edge_distances


class MovementConstraintPenaltyTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.multi_tasks = {uid: [] for uid in range(self.env.num_UAV)}
        gt = self.env.gts[0]
        gt.x, gt.y, gt.z, gt.radius = 500.0, 500.0, 0.0, 80.0
        self.fov_task = {
            "task_type": "FOV",
            "target_id": 0,
            "target_obj_id": 0,
            "target_pos": gt.get_position(),
        }

    def _uav_position(self, uav_id, x, y, z):
        uav = self.env.uav_dict[uav_id]
        uav.x_u, uav.y_u, uav.z_u = float(x), float(y), float(z)

    def test_c9_boundary_and_c10_gating(self):
        self.env.multi_tasks[1] = [self.fov_task]
        limit = VS_CAMERA.b1 * 100.0
        for offset, expected_violations in ((-1.0, 0), (0.0, 0), (1.0, 1)):
            with self.subTest(offset=offset):
                self._uav_position(1, 500.0 + limit + offset, 500.0, 100.0)
                result = movement_constraint_penalties(self.env)
                self.assertEqual(result["c9_sample_count"], 1)
                self.assertEqual(
                    result["c9_violated_pair_count"], expected_violations
                )
                if expected_violations:
                    self.assertGreater(result["c9_penalty_mean"], 0.0)
                    self.assertEqual(result["c10_sample_count"], 0)
                    self.assertEqual(result["c10_penalty_mean"], 0.0)

    def test_c10_uses_finite_geometry_after_c9(self):
        self.env.multi_tasks[1] = [self.fov_task]
        self._uav_position(1, 500.0, 500.0, 100.0)
        result = movement_constraint_penalties(self.env)
        expected_factor = min(100.0 / VS_CAMERA.b1 / 80.0, 1.0)
        self.assertEqual(result["c9_penalty_mean"], 0.0)
        self.assertEqual(result["c10_sample_count"], 1)
        self.assertAlmostEqual(
            result["c10_penalty_mean"], 1.0 - expected_factor
        )
        self.assertEqual(result["c10_violated_pair_count"], 1)

    def test_c9_boundary_has_finite_c10_geometry_and_is_sampled(self):
        self.env.multi_tasks[1] = [self.fov_task]
        limit = VS_CAMERA.b1 * 100.0
        self._uav_position(1, 500.0 + limit, 500.0, 100.0)

        geometry = fov_task_geometry(self.env, 1, self.fov_task)
        edge_distances = vs_c10_edge_distances(geometry)
        result = movement_constraint_penalties(self.env)

        self.assertTrue(geometry.model_range_valid)
        self.assertFalse(geometry.sensing_valid_now)
        self.assertIsNotNone(edge_distances)
        self.assertTrue(np.isfinite(edge_distances).all())
        self.assertEqual(result["c10_sample_count"], 1)

    def test_com_range_boundaries_and_mean_aggregation(self):
        for sr_id, distance, uav_id in ((0, 200.0, 1), (1, 800.0, 2)):
            sr = self.env.SR_teams[sr_id]
            sr.x, sr.y, sr.z = 100.0, 100.0, 0.0
            self._uav_position(uav_id, 100.0 + distance, 100.0, 0.0)
            self.env.multi_tasks[uav_id] = [{
                "task_type": "COM",
                "target_id": sr_id,
                "target_obj_id": sr_id,
                "target_pos": sr.get_position(),
            }]
        result = movement_constraint_penalties(self.env)
        self.assertEqual(result["com_range_sample_count"], 2)
        self.assertAlmostEqual(result["com_range_penalty_sum"], 0.525)
        self.assertAlmostEqual(result["com_range_penalty_mean"], 0.2625)
        self.assertAlmostEqual(
            result["com_range_weighted_penalty"], 0.525
        )
        self.assertEqual(result["com_in_range_pair_count"], 1)
        self.assertEqual(result["com_out_of_range_pair_count"], 1)

        sr = self.env.SR_teams[0]
        self._uav_position(
            1, sr.x + S2U_COMMUNICATION_RANGE_M, sr.y, sr.z
        )
        boundary = movement_constraint_penalties(self.env)
        self.assertAlmostEqual(boundary["com_range_penalty_mean"], 0.2875)

        self._uav_position(
            1, sr.x + COM_PENALTY_REFERENCE_DISTANCE_M, sr.y, sr.z
        )
        penalty_boundary = movement_constraint_penalties(self.env)
        self.assertAlmostEqual(
            penalty_boundary["com_range_penalty_mean"], 0.2625
        )

    def test_fov_com_pair_contributes_to_both_penalty_types(self):
        sr = self.env.SR_teams[0]
        sr.x, sr.y, sr.z = 500.0, 500.0, 0.0
        self._uav_position(1, 500.0, 500.0, 100.0)
        self.env.multi_tasks[1] = [
            self.fov_task,
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            },
        ]
        result = movement_constraint_penalties(self.env)
        self.assertEqual(result["c9_sample_count"], 1)
        self.assertEqual(result["c10_sample_count"], 1)
        self.assertEqual(result["com_range_sample_count"], 1)

    def test_empty_assignments_are_finite_zero(self):
        result = movement_constraint_penalties(self.env)
        for name in (
            "c9_penalty_mean", "c10_penalty_mean", "com_range_penalty_mean"
        ):
            self.assertEqual(result[name], 0.0)
            self.assertTrue(np.isfinite(result[name]))

    def test_final_reward_is_immediate_and_no_task_variant_disables_penalties(self):
        penalties = {
            "c9_penalty_mean": 0.1,
            "c10_penalty_mean": 0.2,
            "com_range_penalty_mean": 0.3,
        }
        enabled = _interval_reward(
            5.0, 2.0, 1.0, penalties,
            reward_mode="dinkelbach", task_potential_enabled=True,
        )
        disabled = _interval_reward(
            5.0, 2.0, 1.0, penalties,
            reward_mode="dinkelbach", task_potential_enabled=False,
        )
        self.assertAlmostEqual(enabled, 2.3)
        self.assertAlmostEqual(disabled, 3.0)

    def test_online_and_replay_reward_apply_com_weight_once_with_c10_switch(self):
        penalties = {
            "c9_penalty_mean": 0.1,
            "c10_penalty_mean": 0.2,
            "com_range_penalty_mean": 0.3,
        }
        replay = ReplayBufferJoint(state_dim=1, action_dim=1, max_size=1)
        replay.delivered_mbits[0, 0] = 5.0
        replay.total_mobility_energy[0, 0] = 2.0
        replay.c9_penalty[0, 0] = penalties["c9_penalty_mean"]
        replay.c10_penalty[0, 0] = penalties["c10_penalty_mean"]
        replay.com_range_penalty[0, 0] = penalties["com_range_penalty_mean"]
        for c10_enabled in (False, True):
            with self.subTest(c10_enabled=c10_enabled):
                online = _interval_reward(
                    5.0,
                    2.0,
                    1.0,
                    penalties,
                    reward_mode="dinkelbach",
                    task_potential_enabled=True,
                    c10_penalty_enabled=c10_enabled,
                )
                rebuilt = replay._reward_numpy(
                    np.asarray([0]),
                    current_lambda=1.0,
                    gamma=0.99,
                    c10_penalty_enabled=c10_enabled,
                )
                self.assertAlmostEqual(float(rebuilt[0, 0]), online, places=6)

        disabled_online = _interval_reward(
            5.0,
            2.0,
            1.0,
            penalties,
            task_potential_enabled=False,
            c10_penalty_enabled=True,
        )
        disabled_replay = replay._reward_numpy(
            np.asarray([0]),
            current_lambda=1.0,
            gamma=0.99,
            task_potential_enabled=False,
            c10_penalty_enabled=True,
        )
        self.assertAlmostEqual(
            float(disabled_replay[0, 0]), disabled_online, places=6
        )

    def test_all_methods_publish_fixed_penalty_weights(self):
        for method_id in METHOD_REGISTRY:
            method = MethodSpec.parse(method_id)
            config = comparison_method_configuration(method)
            expected = {
                "c9": 1.0 if method.task_potential_enabled else 0.0,
                "c10": (
                    1.0
                    if method.task_potential_enabled and method.c10_penalty_enabled
                    else 0.0
                ),
                "com_range": (
                    MOVEMENT_CONSTRAINT_WEIGHT_COM
                    if method.task_potential_enabled
                    else 0.0
                ),
            }
            self.assertEqual(
                set(config["movement_constraint_penalty_weights"].values()),
                {1.0, MOVEMENT_CONSTRAINT_WEIGHT_COM},
            )
            self.assertEqual(
                config["effective_movement_constraint_penalty_weights"],
                expected,
            )


if __name__ == "__main__":
    unittest.main()
