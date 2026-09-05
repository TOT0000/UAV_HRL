import inspect
import unittest
from unittest.mock import patch

import numpy as np

from Channel_model import a2g_expected_capacity_mbps
from Packet_scheduler_v1 import PacketEngine, final_hop_delivered_bits
from Simulator import Simulator
from Task_assignment import Task, UAVAssigner
from centralized_movement import (
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    blended_com_progress,
    calculate_movement_potentials,
    get_global_movement_state,
    movement_state_feature_schema,
    normalized_s2u_range_gap_proximity,
    project_joint_action,
    vs_data_valid,
)
from com_capacity_calibration import calibrate_com_capacity


class VisualSensingPacketGenerationTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.packet_engine = PacketEngine(num_uav=10, step_time=0.25)
        self.uav = self.env.uav_dict[0]
        self.target = self.env.gts[0]
        self.uav.x_u = self.target.x
        self.uav.y_u = self.target.y
        self.uav.z_u = 80.0
        self.task = {
            "task_type": "FOV",
            "target_id": 0,
            "target_obj_id": 0,
            "target_pos": self.target.get_position(),
        }
        self.env.multi_tasks = {uid: [] for uid in range(10)}
        self.env.multi_tasks[0] = [self.task]
        self.env.source_uavs = {0}

    def test_assignment_generates_for_valid_and_zero_coverage_geometry(self):
        self.assertTrue(vs_data_valid(self.env, 0, self.task))
        self.packet_engine.inject_packets(
            self.env,
            delay_bound_steps=20,
            current_time=0.0,
            step_time=0.25,
            base_fov_rate=4,
        )
        self.assertEqual(self.packet_engine.active_count(), 1)
        old_packet = self.packet_engine.get_active_packets()[0]

        self.uav.x_u = 0.0 if self.target.x > 300.0 else 1000.0
        self.uav.y_u = 0.0 if self.target.y > 300.0 else 1000.0
        self.assertFalse(vs_data_valid(self.env, 0, self.task))
        self.packet_engine.inject_packets(
            self.env,
            delay_bound_steps=20,
            current_time=0.25,
            step_time=0.25,
            base_fov_rate=4,
        )
        self.assertEqual(self.packet_engine.active_count(), 2)
        new_packet = self.packet_engine.get_active_packets()[1]
        self.assertEqual(new_packet["capture_coverage_ratio"], 0.0)
        self.assertEqual(self.packet_engine.eligible_packet_counts["FOV"], 2)

        # Capture quality is frozen per packet and does not change physical FIFO.
        result = self.packet_engine.serve_active_links(
            self.env,
            actions={0: self.env.GS_ID},
            capacities={(0, self.env.GS_ID): 100.0},
            current_time=0.25,
        )
        self.assertGreater(result["timely_goodput_bits"], 0.0)
        self.assertTrue(old_packet["done"])
        self.assertEqual(old_packet["current"], self.env.GS_ID)

    def test_vs_potential_is_continuous_and_positive_for_full_coverage(self):
        _, phi_vs, _, _ = calculate_movement_potentials(self.env, c_ref_com=1.0)
        self.assertGreater(phi_vs, 0.0)
        self.assertLessEqual(phi_vs, 1.0)

    def test_full_coverage_still_requires_finite_bounded_image_score(self):
        for image_score in (0.0, -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(image_score=image_score), patch(
                "centralized_movement.fov_task_metrics",
                return_value=(1.0, image_score, True),
            ):
                self.assertFalse(vs_data_valid(self.env, 0, self.task))

        with patch(
            "centralized_movement.fov_task_metrics",
            return_value=(float("nan"), 0.5, True),
        ):
            self.assertFalse(vs_data_valid(self.env, 0, self.task))

    def test_image_score_outside_old_validity_gate_still_injects(self):
        with patch(
            "Packet_scheduler_v1.fov_task_metrics",
            return_value=(1.0, 1.5, True),
        ):
            self.packet_engine.inject_packets(
                self.env,
                delay_bound_steps=20,
                current_time=0.0,
                step_time=0.25,
                base_fov_rate=4,
            )
        self.assertEqual(self.packet_engine.active_count(), 1)
        packet = self.packet_engine.get_active_packets()[0]
        self.assertEqual(packet["capture_coverage_ratio"], 1.0)
        self.assertGreaterEqual(packet["size_bits"], 0.0)


class ComStateCalibrationAndDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_com_only_uav_is_projected_as_movable(self):
        sr = self.env.SR_teams[0]
        self.env.multi_tasks = {uid: [] for uid in range(10)}
        self.env.multi_tasks[5] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet_engine = PacketEngine(num_uav=10, step_time=0.25)
        state = get_global_movement_state(
            self.env,
            packet_engine,
            packet_engine.backlog_bits,
            c_ref_com=1.0,
            remaining_time=0.5,
        )
        com_flag_index = 5 * LOCAL_MOVEMENT_DIM + 2
        self.assertEqual(state[com_flag_index], 1.0)
        raw = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        raw.reshape(10, 3)[5] = [0.75, 0.25, -0.5]
        projected = project_joint_action(raw, state).reshape(10, 3)
        np.testing.assert_array_equal(projected[5], raw.reshape(10, 3)[5])

    def test_calibration_is_reproducible_and_restores_geometry(self):
        original_uav = self.env.uav_dict[0].get_position()
        original_sr = self.env.SR_teams[0].get_position()
        first = calibrate_com_capacity(self.env, seed=1234, sample_count=1000)
        second = calibrate_com_capacity(self.env, seed=1234, sample_count=1000)
        self.assertEqual(first, second)
        self.assertEqual(
            first["schema"], "fixed-s2u-los-rician-expected-maximum-v2"
        )
        self.assertEqual(first["reference_bandwidth_denominator"], 18)
        self.assertAlmostEqual(first["reference_bandwidth_hz"], 10e6 / 18)
        self.assertEqual(first["offered_rate_bps"], 50 * 256)
        self.assertGreater(first["reference_s2u_max_capacity_mbps"], 0.0)
        self.assertEqual(
            first["c_ref_com"], first["reference_s2u_max_capacity_mbps"]
        )
        self.assertEqual(self.env.uav_dict[0].get_position(), original_uav)
        self.assertEqual(self.env.SR_teams[0].get_position(), original_sr)

    def test_canonical_sr_uav_capacity_uses_declared_units_and_range(self):
        uav = self.env.uav_dict[0]
        sr = self.env.SR_teams[0]
        sr.x, sr.y, sr.z = 100.0, 100.0, 0.0
        uav.x_u, uav.y_u, uav.z_u = 100.0, 100.0, 100.0

        snr = self.env.get_snr(0, 0)
        capacity = self.env.get_sr_uav_capacity_mbps(0, 0)
        expected = float(
            a2g_expected_capacity_mbps(
                uav.get_position(),
                sr.get_position(),
                10e6 / 18,
                23.0,
                self.env.channel.a2g_state("S2U", 0, 0),
            )
        )
        self.assertGreater(snr, 0.0)
        self.assertAlmostEqual(capacity, expected)
        self.assertGreater(capacity, 0.0)

        uav.x_u = sr.x + 10_000.0
        uav.y_u = sr.y
        uav.z_u = sr.z
        self.assertGreater(self.env.get_snr(0, 0), 0.0)
        self.assertGreater(self.env.get_sr_uav_capacity_mbps(0, 0), 0.0)

    def test_nearer_feasible_link_capacity_is_not_lower(self):
        uav = self.env.uav_dict[0]
        sr = self.env.SR_teams[0]
        sr.x, sr.y, sr.z = 500.0, 500.0, 0.0
        uav.x_u, uav.y_u, uav.z_u = 500.0, 500.0, 50.0
        near = self.env.get_sr_uav_capacity_mbps(0, 0)
        uav.x_u = 500.0 + np.sqrt(199.0 ** 2 - 50.0 ** 2)
        far = self.env.get_sr_uav_capacity_mbps(0, 0)
        self.assertGreaterEqual(near, far)
        self.assertGreater(far, 0.0)

    def test_state_potential_and_assignment_share_helper_mbps_scale(self):
        uav = self.env.uav_dict[0]
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        task_dict = {
            "task_type": "COM",
            "target_id": 0,
            "target_obj_id": 0,
            "target_pos": sr.get_position(),
        }
        self.env.multi_tasks = {uid: [] for uid in range(10)}
        self.env.multi_tasks[0] = [task_dict]
        packet_engine = PacketEngine(num_uav=10, step_time=0.25)
        with patch.object(
            self.env, "get_sr_uav_normalized_utility", return_value=0.5
        ) as utility_helper:
            state = get_global_movement_state(
                self.env,
                packet_engine,
                packet_engine.backlog_bits,
                c_ref_com=24.0,
                remaining_time=1.0,
            )
            _, _, phi_com, _ = calculate_movement_potentials(
                self.env, c_ref_com=24.0
            )
            assigner = UAVAssigner(self.env)
            problem = assigner.build_problem(
                [0],
                [
                    Task(
                        task_id=0,
                        task_type="COM",
                        target_obj=sr,
                        target_obj_id=0,
                    )
                ],
            )
            assignment = assigner.assign_uav_tasks_k_times(
                [0],
                [
                    Task(
                        task_id=0,
                        task_type="COM",
                        target_obj=sr,
                        target_obj_id=0,
                    )
                ],
                K=1,
            )

        com_capacity_index = next(
            feature["index"]
            for feature in movement_state_feature_schema()["features"]
            if feature["name"] == "uav_0.com_capacity"
        )
        self.assertEqual(state[com_capacity_index], 0.5)
        expected_distance = normalized_s2u_range_gap_proximity(
            self.env.uav_dict[0].get_position(),
            sr.get_position(),
            self.env.env_width,
            self.env.env_height,
        )
        self.assertEqual(phi_com, blended_com_progress(0.5, expected_distance))
        self.assertEqual(problem.raw_com_utility[0, 0], 0.5)
        self.assertEqual(assignment[0][0][2], 0.5)
        self.assertGreaterEqual(utility_helper.call_count, 3)
        source = inspect.getsource(UAVAssigner.assign_uav_tasks_k_times)
        self.assertNotIn("capacity / 1e6", source)

    def test_delivered_bits_only_count_final_gs_hop(self):
        self.assertEqual(final_hop_delivered_bits(3, 10, 123.0), 0.0)
        self.assertEqual(final_hop_delivered_bits(10, 10, 123.0), 123.0)


if __name__ == "__main__":
    unittest.main()
