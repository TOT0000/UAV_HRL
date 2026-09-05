import unittest

import numpy as np

from Channel_model import (
    U2U_U2G_TX_POWER_DBM,
    block_capacity_profile_mbps,
)
from Simulator import Simulator


class ActiveLinkBandwidthTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.PL_uu_cache = np.full((16, 16), 100.0, dtype=float)
        np.fill_diagonal(self.env.PL_uu_cache, 0.0)
        self.env.PL_ug_cache = np.full(16, 105.0, dtype=float)
        positions = np.zeros((16, 3), dtype=float)
        positions[:, 2] = 100.0
        self.env.channel.reset_episode(
            uav_positions=positions,
            sr_positions=np.zeros((0, 3), dtype=float),
            gs_position=np.zeros(3, dtype=float),
            episode_identity="bandwidth-test",
        )
        self.env.prepare_channel_routing_slot(0)

    def test_two_active_u2u_links_each_receive_five_mhz(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: 1, 2: 3}
        )

        self.assertEqual(bandwidths[(0, 1)], 5e6)
        self.assertEqual(bandwidths[(2, 3)], 5e6)
        gains = self.env.channel.gain_profile("U2U", 0, 1)
        expected_profile = block_capacity_profile_mbps(
            100.0, 5e6, U2U_U2G_TX_POWER_DBM, gains
        )
        np.testing.assert_allclose(
            self.env.active_link_capacity_profiles_mbps[(0, 1)],
            expected_profile,
        )
        self.assertAlmostEqual(capacities[(0, 1)], float(expected_profile.mean()))

        full_profile = block_capacity_profile_mbps(
            100.0, 10e6, U2U_U2G_TX_POWER_DBM, gains
        )
        self.assertNotAlmostEqual(capacities[(0, 1)], float(full_profile.mean()) / 2.0)

    def test_three_active_u2g_links_each_receive_one_third_pool(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: self.env.GS_ID, 1: self.env.GS_ID, 2: self.env.GS_ID}
        )

        allocated = 10e6 / 3.0
        for sender in range(3):
            link = (sender, self.env.GS_ID)
            self.assertAlmostEqual(bandwidths[link], allocated)
            gains = self.env.channel.gain_profile("U2G", sender, self.env.GS_ID)
            expected_profile = block_capacity_profile_mbps(
                105.0, allocated, U2U_U2G_TX_POWER_DBM, gains
            )
            self.assertAlmostEqual(capacities[link], float(expected_profile.mean()))

    def test_wait_action_is_excluded_from_active_bandwidth(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: 0, 1: 2}
        )

        self.assertNotIn((0, 0), capacities)
        self.assertNotIn((0, 0), bandwidths)
        self.assertEqual(bandwidths[(1, 2)], 10e6)


if __name__ == "__main__":
    unittest.main()
