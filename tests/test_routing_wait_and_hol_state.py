import unittest

import numpy as np

from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator


class RoutingWaitAndHolStateTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.current_time = 0.5
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        self.engine = PacketEngine(num_uav=16, step_time=0.25)

    def test_empty_physical_connectivity_exposes_only_wait(self):
        self.env.Capacity_matrix[:] = 0.0
        self.env.gs_capacity[:] = 0.0

        mask = self.env.get_routing_action_mask(3).astype(bool)

        self.assertEqual(np.flatnonzero(mask).tolist(), [3])

    def test_partial_hop_mask_contains_only_lock_and_wait(self):
        packet = self.engine.create_packet(0, "COM", 100.0, 0.0)
        self.engine.record_hop_transmission(packet, 0, 2, 25.0)
        physical = np.zeros(17, dtype=bool)
        physical[[2, 3]] = True

        mask = self.engine.get_effective_action_mask(
            self.env, 0, physical
        )
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 2])

        physical[2] = False
        mask = self.engine.get_effective_action_mask(
            self.env, 0, physical
        )
        self.assertEqual(np.flatnonzero(mask).tolist(), [0])

    def test_empty_queue_effective_mask_contains_only_wait(self):
        physical = np.ones(17, dtype=bool)

        mask = self.engine.get_effective_action_mask(self.env, 4, physical)

        self.assertEqual(np.flatnonzero(mask).tolist(), [4])

    def test_candidate_link_delays_use_hol_bits_and_nominal_capacity(self):
        packet = self.engine.create_packet(0, "COM", 6e6, 0.0)
        self.env.Capacity_matrix[0, 1] = 2.0
        self.env.Capacity_matrix[0, 2] = 4.0
        self.env.gs_capacity[0] = 3.0
        physical = np.zeros(17, dtype=bool)
        physical[[0, 1, 2, self.env.GS_ID]] = True

        before_e2e = packet["e2e_delay_ms"]
        state = self.engine.get_state_ta(
            self.env,
            0,
            backlog_bits=self.engine.backlog_bits,
            action_mask=physical,
        )
        delay_start = 16 + 8 + 17
        delays = state[delay_start:delay_start + 17]

        self.assertEqual(state.shape, (143,))
        self.assertAlmostEqual(delays[1], 1.0)
        self.assertAlmostEqual(delays[2], 0.5)
        self.assertAlmostEqual(delays[self.env.GS_ID], 2.0 / 3.0)
        self.assertEqual(delays[0], 0.0)
        self.assertEqual(delays[3], 0.0)

        self.engine.get_state_ta(
            self.env,
            0,
            backlog_bits=self.engine.backlog_bits,
            action_mask=physical,
        )
        self.assertEqual(packet["e2e_delay_ms"], before_e2e)

    def test_relay_hol_context_uses_packet_identity(self):
        packet = self.engine.create_packet(0, "COM", 100.0, 0.0)
        packet["deadline_abs"] = 1.0
        self.engine.record_hop_transmission(packet, 0, 5, 100.0)
        arrival = self.engine.detach_completed_hop(
            packet, 0, 5, completion_time=0.25
        )
        self.engine.enqueue_relay_arrivals([arrival])

        # The relay's assignment is intentionally FOV; HOL identity stays COM.
        self.env.uav_dict[5].task_type = "FOV"
        state = self.engine.get_state_ta(
            self.env,
            5,
            backlog_bits=self.engine.backlog_bits,
            action_mask=self.env.get_routing_action_mask(5),
        )

        self.assertEqual(state.shape, (143,))
        np.testing.assert_allclose(state[-4:], [0.0, 1.0, 0.25, 1.0])
        self.engine.get_state_ta(
            self.env,
            5,
            backlog_bits=self.engine.backlog_bits,
            action_mask=self.env.get_routing_action_mask(5),
        )
        self.assertEqual(packet["e2e_delay_ms"], 0.0)

        packet["rem_bits"] = 50.0
        self.engine._sync_backlog(5)
        state = self.engine.get_state_ta(
            self.env,
            5,
            backlog_bits=self.engine.backlog_bits,
            action_mask=self.env.get_routing_action_mask(5),
        )
        self.assertEqual(state[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
