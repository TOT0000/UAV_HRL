import unittest

import numpy as np

from Channel_model import ChannelModel
from Simulator import Simulator


class ActiveLinkBandwidthTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.PL_uu_cache = np.full((10, 10), 100.0, dtype=float)
        np.fill_diagonal(self.env.PL_uu_cache, 0.0)
        self.env.PL_ug_cache = np.full(10, 105.0, dtype=float)

    def test_two_active_u2u_links_each_receive_five_mhz(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: 1, 2: 3}
        )

        self.assertEqual(bandwidths[(0, 1)], 5e6)
        self.assertEqual(bandwidths[(2, 3)], 5e6)
        expected_snr = ChannelModel.SNR_uu(30.0, -169.0, 100.0, 5e6)
        expected_capacity = float(ChannelModel.C_uu(5e6, expected_snr))
        self.assertAlmostEqual(capacities[(0, 1)], expected_capacity)

        full_snr = ChannelModel.SNR_uu(30.0, -169.0, 100.0, 10e6)
        full_capacity = float(ChannelModel.C_uu(10e6, full_snr))
        self.assertNotAlmostEqual(capacities[(0, 1)], full_capacity / 2.0)

    def test_three_active_u2g_links_each_receive_one_third_pool(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: self.env.GS_ID, 1: self.env.GS_ID, 2: self.env.GS_ID}
        )

        allocated = 10e6 / 3.0
        for sender in range(3):
            link = (sender, self.env.GS_ID)
            self.assertAlmostEqual(bandwidths[link], allocated)
            expected_snr = ChannelModel.SNR_ug(
                30.0, -169.0, 105.0, allocated
            )
            expected_capacity = float(
                ChannelModel.C_ug(allocated, expected_snr)
            )
            self.assertAlmostEqual(capacities[link], expected_capacity)

    def test_wait_action_is_excluded_from_active_bandwidth(self):
        capacities, bandwidths = self.env.allocate_active_link_capacities(
            {0: 0, 1: 2}
        )

        self.assertNotIn((0, 0), capacities)
        self.assertNotIn((0, 0), bandwidths)
        self.assertEqual(bandwidths[(1, 2)], 10e6)


if __name__ == "__main__":
    unittest.main()
