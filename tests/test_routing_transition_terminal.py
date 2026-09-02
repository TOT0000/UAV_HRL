import unittest

import numpy as np

from HRL_task_aware import _run_routing_slot
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from utils_update_v2 import ReplayBufferDiscrete


class FixedRoutingPolicy:
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, state, uav_id, mask=None, **kwargs):
        return self.actions[int(uav_id)]


class RoutingTransitionTerminalTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.source_uavs = set()
        self.env.current_time = 0.0
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        self.engine = PacketEngine(num_uav=10, step_time=0.25)
        self.masks = {
            uid: np.ones(11, dtype=bool) for uid in range(10)
        }
        self.stats = {
            "FOV": {
                "timely_delivered_packets": 0,
                "deadline_violated_packets": 0,
            },
            "COM": {
                "timely_delivered_packets": 0,
                "deadline_violated_packets": 0,
            },
        }

    def run_slot(
        self, actions, capacities, episode_done=False, current_time=0.0
    ):
        self.env.allocate_active_link_capacities = (
            lambda proposed, s2u_links=None: (
                {
                    (sender, receiver): capacities[(sender, receiver)]
                    for sender, receiver in proposed.items()
                },
                {},
            )
        )
        replay = ReplayBufferDiscrete(90, 11, max_size=16, n_step=1)
        _run_routing_slot(
            self.env,
            self.engine,
            FixedRoutingPolicy(actions),
            replay,
            self.masks,
            current_time=current_time,
            done=episode_done,
            delay_bound_steps=20,
            violation_stats=self.stats,
            epsilon=0.0,
            traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
        )
        return replay

    def test_only_packet_delivered_to_gs_is_terminal(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)

        replay = self.run_slot(
            {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0004}
        )

        self.assertEqual(replay.size, 1)
        self.assertEqual(replay.not_done[0, 0], 0.0)
        self.assertEqual(self.stats["COM"]["timely_delivered_packets"], 1)
        self.assertEqual(self.stats["COM"]["deadline_violated_packets"], 0)

    def test_deadline_outcome_only_increments_violation_stat(self):
        packet = self.engine.create_packet(0, "COM", 100.0, 0.0)
        packet["deadline_abs"] = 0.25

        replay = self.run_slot(
            {0: 1}, {(0, 1): 0.001}
        )

        self.assertEqual(replay.cost[0, 0], 1.0)
        self.assertEqual(self.stats["COM"]["timely_delivered_packets"], 0)
        self.assertEqual(self.stats["COM"]["deadline_violated_packets"], 1)

    def test_partial_packet_and_next_queued_packet_keep_bootstrap(self):
        with self.subTest(case="partial"):
            self.engine.create_packet(0, "COM", 100.0, 0.0)
            replay = self.run_slot(
                {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0002}
            )
            self.assertEqual(replay.not_done[0, 0], 1.0)

        self.engine.reset_packet_state()
        with self.subTest(case="next packet"):
            self.engine.create_packet(0, "COM", 100.0, 0.0)
            self.engine.create_packet(0, "COM", 50.0, 0.0)
            replay = self.run_slot(
                {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0004}
            )
            self.assertEqual(replay.not_done[0, 0], 1.0)

    def test_same_slot_relay_arrival_keeps_sender_bootstrap(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)
        relayed = self.engine.create_packet(1, "FOV", 100.0, 0.0)

        replay = self.run_slot(
            {0: self.env.GS_ID, 1: 0},
            {(0, self.env.GS_ID): 0.0004, (1, 0): 0.0004},
        )

        self.assertIs(self.engine.get_hol_packet(0), relayed)
        self.assertEqual(replay.not_done[0, 0], 1.0)
        self.assertEqual(replay.not_done[1, 0], 0.0)

    def test_episode_end_terminates_every_existing_transition(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)

        replay = self.run_slot(
            {0: 0}, {}, episode_done=True
        )

        self.assertEqual(replay.not_done[0, 0], 0.0)

    def test_empty_sender_at_slot_start_creates_no_transition(self):
        replay = self.run_slot({}, {})

        self.assertEqual(replay.size, 0)

    def test_s2u_receiver_gets_no_same_slot_decision_or_replay(self):
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        receiver = 1
        self.env.multi_tasks[receiver] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet = self.engine.create_sr_packet(0, 100.0, generation_time=0.0)
        self.env.active_s2u_capacities = {(0, receiver): 0.001}

        replay = self.run_slot({}, {})

        self.assertEqual(replay.size, 0)
        self.assertIs(self.engine.get_hol_packet(receiver), packet)
        self.assertIsNone(packet["last_routing_sender"])
        self.assertEqual(packet["routing_eligible_time"], 0.1)
        self.assertEqual(packet["rem_bits"], packet["size_bits"])

        self.env.active_s2u_capacities = {}
        next_replay = self.run_slot(
            {receiver: self.env.GS_ID},
            {(receiver, self.env.GS_ID): 0.001},
            current_time=0.25,
        )
        self.assertEqual(next_replay.size, 1)
        self.assertEqual(next_replay.latest_index_by_agent, {receiver: 0})
        self.assertEqual(packet["last_routing_sender"], receiver)
        self.assertTrue(packet["done"])


if __name__ == "__main__":
    unittest.main()
