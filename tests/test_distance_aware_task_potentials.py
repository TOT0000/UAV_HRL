import math
import unittest
from unittest import mock

import numpy as np

from HRL_task_aware import TrainingConfig, _interval_reward
from Simulator import Simulator
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    blended_com_progress,
    blended_vs_progress,
    calculate_movement_potentials,
    normalized_horizontal_target_proximity,
    normalized_s2u_range_gap_proximity,
)
from experiment_config import (
    A2G_COMMUNICATION_RANGE_M,
    COM_CAPACITY_POTENTIAL_WEIGHT,
    COM_DISTANCE_POTENTIAL_WEIGHT,
    METHOD_REGISTRY,
    TASK_POTENTIAL_CONTRACT_VERSION,
    VS_DISTANCE_POTENTIAL_WEIGHT,
    VS_SENSING_POTENTIAL_WEIGHT,
    MethodSpec,
    comparison_method_configuration,
    task_potential_contract_metadata,
    validate_task_potential_weights,
)
from observation_strategy import ROUTING_STATE_DIM
from scenario_manifest import SCENARIO_SCHEMA_VERSION


class DistanceProgressHelperTest(unittest.TestCase):
    def test_vs_horizontal_boundaries_monotonicity_and_altitude_invariance(
        self,
    ):
        width = 1000.0
        height = 1000.0
        self.assertEqual(
            normalized_horizontal_target_proximity(
                (400.0, 500.0, 80.0),
                (400.0, 500.0, 0.0),
                width,
                height,
            ),
            1.0,
        )
        self.assertEqual(
            normalized_horizontal_target_proximity(
                (0.0, 0.0, 80.0),
                (1000.0, 1000.0, 0.0),
                width,
                height,
            ),
            0.0,
        )
        distances = (1000.0, 500.0, 100.0)
        progresses = [
            normalized_horizontal_target_proximity(
                (distance, 0.0, 100.0),
                (0.0, 0.0, 0.0),
                width,
                height,
            )
            for distance in distances
        ]
        self.assertLess(progresses[0], progresses[1])
        self.assertLess(progresses[1], progresses[2])
        low = normalized_horizontal_target_proximity(
            (300.0, 400.0, 50.0), (0.0, 0.0, 0.0), width, height
        )
        high = normalized_horizontal_target_proximity(
            (300.0, 400.0, 150.0), (0.0, 0.0, 0.0), width, height
        )
        self.assertEqual(low, high)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in progresses))

    def test_vs_blend_rewards_nearer_target_at_fixed_sensing_progress(self):
        far = blended_vs_progress(0.4, 0.2)
        near = blended_vs_progress(0.4, 0.8)
        self.assertLess(far, near)
        self.assertTrue(0.0 <= far <= 1.0)
        self.assertTrue(0.0 <= near <= 1.0)

    def test_com_range_gap_boundaries_continuity_and_monotonicity(self):
        helper = normalized_s2u_range_gap_proximity
        common = ((0.0, 0.0, 0.0), 1000.0, 1000.0)
        within = helper((100.0, 0.0, 0.0), *common)
        boundary = helper((200.0, 0.0, 0.0), *common)
        just_outside = helper((200.001, 0.0, 0.0), *common)
        farther = helper((600.0, 0.0, 0.0), *common)
        farthest = helper((1000.0, 1000.0, 150.0), *common)
        self.assertEqual(within, 1.0)
        self.assertEqual(boundary, 1.0)
        self.assertLess(just_outside, 1.0)
        self.assertGreater(just_outside, farther)
        self.assertGreater(farther, farthest)
        self.assertEqual(farthest, 0.0)
        self.assertTrue(
            all(
                0.0 <= value <= 1.0
                for value in (within, boundary, just_outside, farther, farthest)
            )
        )

    def test_com_blend_rewards_smaller_range_gap_at_fixed_capacity(self):
        far = blended_com_progress(0.3, 0.25)
        near = blended_com_progress(0.3, 0.75)
        in_range = blended_com_progress(0.3, 1.0)
        self.assertLess(far, near)
        self.assertLess(near, in_range)
        self.assertTrue(0.0 <= far <= 1.0)
        self.assertTrue(0.0 <= in_range <= 1.0)

    def test_weights_and_contract_metadata_are_authoritative(self):
        groups = validate_task_potential_weights()
        self.assertEqual(
            groups["VS"],
            (VS_SENSING_POTENTIAL_WEIGHT, VS_DISTANCE_POTENTIAL_WEIGHT),
        )
        self.assertEqual(
            groups["COM"],
            (COM_CAPACITY_POTENTIAL_WEIGHT, COM_DISTANCE_POTENTIAL_WEIGHT),
        )
        self.assertTrue(
            all(math.isclose(sum(weights), 1.0) for weights in groups.values())
        )
        metadata = task_potential_contract_metadata()
        self.assertEqual(
            metadata["contract_version"], TASK_POTENTIAL_CONTRACT_VERSION
        )
        self.assertEqual(metadata["vs"]["distance_dimensionality"], "horizontal_2d")
        self.assertEqual(
            metadata["com"]["distance_dimensionality"],
            "three_dimensional_3d",
        )
        self.assertEqual(metadata["com"]["s2u_range_m"], A2G_COMMUNICATION_RANGE_M)
        self.assertTrue(metadata["search"]["unchanged"])
        self.assertFalse(metadata["lifecycle"]["delivery_or_connectivity_potential"])

        with mock.patch(
            "experiment_config.VS_DISTANCE_POTENTIAL_WEIGHT", 0.6
        ), self.assertRaisesRegex(ValueError, "must sum to 1"):
            validate_task_potential_weights()
        with mock.patch(
            "experiment_config.COM_CAPACITY_POTENTIAL_WEIGHT", float("nan")
        ), self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate_task_potential_weights()


class DistanceAwarePotentialLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(10)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.multi_tasks = {uav_id: [] for uav_id in range(10)}
        self.config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
        )

    def _fov_task(self, target_id=0):
        target = self.env.gts[target_id]
        return {
            "task_type": "FOV",
            "target_id": target_id,
            "target_obj_id": target_id,
            "target_pos": target.get_position(),
        }

    def _com_task(self, sr_id=0):
        sr = self.env.SR_teams[sr_id]
        return {
            "task_type": "COM",
            "target_id": sr_id,
            "target_obj_id": sr_id,
            "target_pos": sr.get_position(),
        }

    def _shaping(self, current, following, *, done=False, enabled=True):
        return _interval_reward(
            0.0,
            0.0,
            0.0,
            1.0,
            current,
            following,
            done,
            self.config,
            reward_mode="dinkelbach",
            task_potential_enabled=enabled,
        )

    def test_invalid_sensing_geometry_keeps_finite_distance_progress(self):
        self.env.multi_tasks[0] = [self._fov_task()]
        target = self.env.gts[0]
        self.env.uav_dict[0].x_u = target.x + 100.0
        self.env.uav_dict[0].y_u = target.y
        with mock.patch(
            "centralized_movement.fov_task_metrics",
            return_value=(0.9, 0.8, False),
        ):
            _, phi_vs, _ = calculate_movement_potentials(self.env, 1.0)
        distance = normalized_horizontal_target_proximity(
            self.env.uav_dict[0].get_position(),
            target.get_position(),
            self.env.env_width,
            self.env.env_height,
        )
        self.assertAlmostEqual(phi_vs, blended_vs_progress(0.0, distance))
        self.assertTrue(math.isfinite(phi_vs))
        self.assertTrue(0.0 <= phi_vs <= 1.0)

    def test_vs_approach_unchanged_and_retreat_have_signed_differences(self):
        self.env.multi_tasks[0] = [self._fov_task()]
        target = self.env.gts[0]
        with mock.patch(
            "centralized_movement.fov_task_metrics",
            return_value=(0.4, 0.5, True),
        ):
            self.env.uav_dict[0].x_u = 0.0
            self.env.uav_dict[0].y_u = 0.0
            far = calculate_movement_potentials(self.env, 1.0)
            self.env.uav_dict[0].x_u = target.x
            self.env.uav_dict[0].y_u = target.y
            near = calculate_movement_potentials(self.env, 1.0)
        self.assertEqual(self._shaping(far, far), 0.0)
        self.assertGreater(self._shaping(far, near), 0.0)
        self.assertLess(self._shaping(near, far), 0.0)
        self.assertAlmostEqual(self._shaping(near, far, done=True), -sum(near))

    def test_com_approach_unchanged_and_retreat_have_signed_differences(self):
        self.env.multi_tasks[0] = [self._com_task()]
        sr = self.env.SR_teams[0]
        uav = self.env.uav_dict[0]
        with mock.patch.object(
            self.env,
            "get_sr_uav_normalized_utility",
            return_value=0.4,
        ):
            uav.x_u, uav.y_u, uav.z_u = 1000.0, 1000.0, 150.0
            far = calculate_movement_potentials(self.env, 1.0)
            uav.x_u, uav.y_u, uav.z_u = sr.x, sr.y, sr.z + 100.0
            near = calculate_movement_potentials(self.env, 1.0)
        self.assertEqual(self._shaping(far, far), 0.0)
        self.assertGreater(self._shaping(far, near), 0.0)
        self.assertLess(self._shaping(near, far), 0.0)

    def test_no_tasks_are_zero_and_multiple_tasks_use_arithmetic_mean(self):
        self.env.visited_bitmap[:] = False
        self.assertEqual(
            calculate_movement_potentials(self.env, 1.0),
            (0.0, 0.0, 0.0),
        )

        self.env.multi_tasks[0] = [self._fov_task(0)]
        self.env.multi_tasks[1] = [self._fov_task(1)]
        self.env.uav_dict[0].x_u, self.env.uav_dict[0].y_u = 0.0, 0.0
        self.env.uav_dict[1].x_u, self.env.uav_dict[1].y_u = 500.0, 500.0
        with mock.patch(
            "centralized_movement.fov_task_metrics",
            return_value=(0.0, 0.0, True),
        ):
            _, phi_vs, _ = calculate_movement_potentials(self.env, 1.0)
        per_task = [
            blended_vs_progress(
                0.0,
                normalized_horizontal_target_proximity(
                    self.env.uav_dict[uav_id].get_position(),
                    task["target_pos"],
                    self.env.env_width,
                    self.env.env_height,
                ),
            )
            for uav_id, task in (
                (0, self.env.multi_tasks[0][0]),
                (1, self.env.multi_tasks[1][0]),
            )
        ]
        self.assertAlmostEqual(phi_vs, float(np.mean(per_task)))

    def test_search_potential_remains_global_coverage_mean(self):
        self.env.visited_bitmap[:] = False
        self.env.visited_bitmap[5:123, 17:301] = True
        expected = float(self.env.visited_bitmap.mean())
        self.env.multi_tasks[0] = [self._fov_task()]
        self.env.multi_tasks[1] = [self._com_task()]
        phi_search, _, _ = calculate_movement_potentials(self.env, 1.0)
        self.assertEqual(phi_search, expected)

    def test_multiple_com_tasks_use_arithmetic_mean(self):
        self.env.multi_tasks[0] = [self._com_task(0)]
        self.env.multi_tasks[1] = [self._com_task(1)]
        self.env.uav_dict[0].x_u, self.env.uav_dict[0].y_u = 0.0, 0.0
        self.env.uav_dict[1].x_u, self.env.uav_dict[1].y_u = 900.0, 900.0
        with mock.patch.object(
            self.env,
            "get_sr_uav_normalized_utility",
            return_value=0.25,
        ):
            _, _, phi_com = calculate_movement_potentials(self.env, 1.0)
        per_task = [
            blended_com_progress(
                0.25,
                normalized_s2u_range_gap_proximity(
                    self.env.uav_dict[uav_id].get_position(),
                    self.env.SR_teams[sr_id].get_position(),
                    self.env.env_width,
                    self.env.env_height,
                ),
            )
            for uav_id, sr_id in ((0, 0), (1, 1))
        ]
        self.assertAlmostEqual(phi_com, float(np.mean(per_task)))

    def test_no_task_potential_disables_all_new_shaping(self):
        current = (0.2, 0.3, 0.4)
        following = (0.4, 0.7, 0.8)
        self.assertGreater(self._shaping(current, following, enabled=True), 0.0)
        self.assertEqual(self._shaping(current, following, enabled=False), 0.0)


class TaskPotentialMethodContractTest(unittest.TestCase):
    def test_all_methods_publish_one_contract_without_dimension_changes(self):
        shared = None
        for method_id in METHOD_REGISTRY:
            with self.subTest(method=method_id):
                method = MethodSpec.parse(method_id)
                config = comparison_method_configuration(method)
                self.assertEqual(
                    config["task_potential_contract_version"],
                    TASK_POTENTIAL_CONTRACT_VERSION,
                )
                self.assertEqual(
                    config["task_potential_enabled"],
                    method.task_potential_enabled,
                )
                if shared is None:
                    shared = config["task_potential_configuration"]
                self.assertEqual(config["task_potential_configuration"], shared)
        self.assertEqual(MOVEMENT_STATE_DIM, 429)
        self.assertEqual(JOINT_ACTION_DIM, 30)
        self.assertEqual(ROUTING_STATE_DIM, 90)
        self.assertEqual(SCENARIO_SCHEMA_VERSION, "uav-hrl-scenario-v4")


if __name__ == "__main__":
    unittest.main()
