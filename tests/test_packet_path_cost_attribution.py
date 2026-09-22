import unittest

import numpy as np
import torch

from DDQN import _bellman_target
from routing_transition_ledger import RoutingTransitionLedger
from utils_update_v2 import ReplayBufferDiscrete


class PacketPathCostAttributionTest(unittest.TestCase):
    def setUp(self):
        self.replay = ReplayBufferDiscrete(6, 4, max_size=32, n_step=1)
        self.ledger = RoutingTransitionLedger()
        self.ledger.begin_episode()

    def _decision(self, packet_id, agent_id, action, value):
        state = np.full(6, value, dtype=np.float32)
        transition_id = self.ledger.create(
            packet_id=packet_id,
            agent_id=agent_id,
            state=state,
            action=action,
            tag_gt=2,
        )
        self.ledger.set_reward(transition_id, 0.0)
        self.ledger.finalize_causality(
            {agent_id: state}, {agent_id: {"id": packet_id}}
        )
        self.ledger.commit_ready(self.replay)
        return transition_id

    def test_cross_uav_expiry_chain_is_zero_zero_one(self):
        ids = [
            self._decision(11, 0, 1, 1.0),
            self._decision(11, 1, 2, 2.0),
            self._decision(11, 2, 3, 3.0),
        ]
        self.assertTrue(self.ledger.finalize_packet(11, violated=True))
        self.ledger.commit_ready(self.replay)
        rows = [
            int(np.flatnonzero(self.replay.transition_id == transition_id)[0])
            for transition_id in ids
        ]
        np.testing.assert_array_equal(self.replay.cost[rows, 0], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(
            self.replay.cost_not_done[rows, 0], [1.0, 1.0, 0.0]
        )
        backed_up = _bellman_target(
            torch.tensor([0.0]), torch.tensor([1.0]),
            torch.tensor([1.0]), 0.99,
        )
        self.assertAlmostEqual(float(backed_up.item()), 0.99)

    def test_timely_delivery_has_zero_terminal_cost(self):
        self._decision(12, 0, 1, 1.0)
        final_id = self._decision(12, 1, 3, 2.0)
        self.assertTrue(self.ledger.finalize_packet(12, violated=False))
        self.ledger.commit_ready(self.replay)
        row = int(np.flatnonzero(self.replay.transition_id == final_id)[0])
        self.assertEqual(self.replay.cost[row, 0], 0.0)
        self.assertEqual(self.replay.cost_not_done[row, 0], 0.0)

    def test_loop_and_hol_wait_are_preserved_without_uav_deduplication(self):
        ids = [
            self._decision(13, 0, 0, 1.0),  # HOL WAIT
            self._decision(13, 1, 0, 2.0),
            self._decision(13, 0, 1, 3.0),  # loop back to UAV 0
        ]
        self.ledger.finalize_packet(13, violated=True)
        self.ledger.commit_ready(self.replay)
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.ledger.episode_decision_count, 3)
        self.assertEqual(self.ledger.episode_cost_transition_count, 3)

    def test_non_hol_wait_creates_nothing_and_pre_routing_has_no_fake_row(self):
        self.assertFalse(self.ledger.finalize_packet(14, violated=True))
        self.assertEqual(self.replay.size, 0)
        self.assertEqual(self.ledger.episode_decision_count, 0)
        self.assertEqual(self.ledger.episode_terminal_without_decision_count, 1)
        # A queued non-HOL packet is represented by the absence of create().
        self.assertNotIn(15, self.ledger.packet_pending)

    def test_duplicate_terminal_outcome_is_ignored(self):
        self._decision(16, 0, 1, 1.0)
        self.assertTrue(self.ledger.finalize_packet(16, violated=True))
        self.assertFalse(self.ledger.finalize_packet(16, violated=True))
        self.assertEqual(self.ledger.episode_terminal_cost_sum, 1.0)


if __name__ == "__main__":
    unittest.main()
