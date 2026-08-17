import unittest

import numpy as np

from Channel_model import ChannelModel
from Packet_scheduler_v1 import PacketEngine, final_hop_delivered_bits
from Simulator import Simulator
from centralized_movement import (
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    calculate_movement_potentials,
    get_global_movement_state,
    project_joint_action,
    vs_data_valid,
)
from com_capacity_calibration import calibrate_com_capacity


class VisualSensingPacketGateTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.packet_engine = PacketEngine(num_uav=16, step_time=0.25)
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
        self.env.multi_tasks = {uid: [] for uid in range(16)}
        self.env.multi_tasks[0] = [self.task]
        self.env.source_uavs = {0}

    def test_valid_geometry_generates_and_invalid_geometry_does_not(self):
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
        self.assertEqual(self.packet_engine.active_count(), 1)
        self.assertIs(self.packet_engine.get_active_packets()[0], old_packet)

    def test_vs_potential_is_continuous_and_positive_for_full_coverage(self):
        _, phi_vs, _ = calculate_movement_potentials(self.env, c_ref_com=1.0)
        self.assertGreater(phi_vs, 0.0)
        self.assertLessEqual(phi_vs, 1.0)


class ComStateCalibrationAndDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_com_only_uav_is_projected_as_movable(self):
        sr = self.env.SR_teams[0]
        self.env.multi_tasks = {uid: [] for uid in range(16)}
        self.env.multi_tasks[5] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet_engine = PacketEngine(num_uav=16, step_time=0.25)
        state = get_global_movement_state(
            self.env, packet_engine, packet_engine.backlog_bits, c_ref_com=1.0
        )
        com_flag_index = 5 * LOCAL_MOVEMENT_DIM + 2
        self.assertEqual(state[com_flag_index], 1.0)
        raw = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
        raw.reshape(16, 3)[5] = [0.75, 0.25, -0.5]
        projected = project_joint_action(raw, state).reshape(16, 3)
        np.testing.assert_array_equal(projected[5], raw.reshape(16, 3)[5])

    def test_calibration_is_reproducible_and_restores_geometry(self):
        original_uav = self.env.uav_dict[0].get_position()
        original_sr = self.env.SR_teams[0].get_position()
        first = calibrate_com_capacity(self.env, seed=1234, sample_count=1000)
        second = calibrate_com_capacity(self.env, seed=1234, sample_count=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["capacity_unit"], "Mbps")
        self.assertTrue(first["feasible_only"])
        self.assertEqual(first["carrier_frequency_ghz"], 2.0)
        self.assertEqual(first["bandwidth_hz"], 2e6)
        self.assertGreaterEqual(first["c_ref_com"], 0.1)
        self.assertLessEqual(first["c_ref_com"], 1000.0)
        self.assertLessEqual(
            first["sampled_distance_range_m"][1],
            self.env.SR_UAV_MAX_RANGE_M,
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
            ChannelModel.C_ug(self.env.SR_UAV_BANDWIDTH_HZ, snr)
        )
        self.assertGreater(snr, 0.0)
        self.assertAlmostEqual(capacity, expected)
        self.assertGreater(capacity, 0.1)

        uav.x_u = sr.x + self.env.SR_UAV_MAX_RANGE_M + 1.0
        uav.y_u = sr.y
        uav.z_u = sr.z
        self.assertEqual(self.env.get_snr(0, 0), 0.0)
        self.assertEqual(self.env.get_sr_uav_capacity_mbps(0, 0), 0.0)

    def test_delivered_bits_only_count_final_gs_hop(self):
        self.assertEqual(final_hop_delivered_bits(3, 16, 123.0), 0.0)
        self.assertEqual(final_hop_delivered_bits(16, 16, 123.0), 123.0)


if __name__ == "__main__":
    unittest.main()
