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
        return self.actions.get(int(uav_id), int(uav_id))


class RoutingTransitionTerminalTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.source_uavs = set()
        self.env.current_time = 0.0
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        self.engine = PacketEngine(
            num_uav=16,
            step_time=0.25,
            task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
        )
        self.masks = {
            uid: np.ones(17, dtype=bool) for uid in range(16)
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
        def allocate(proposed, s2u_links=None):
            active = {
                (sender, receiver): capacities.get((sender, receiver), 0.0)
                for sender, receiver in proposed.items()
            }
            s2u = {
                (sr, receiver): capacities.get(("S2U", sr, receiver), 0.0)
                for sr, receiver in dict(s2u_links or {}).items()
            }
            self.env.active_link_capacity_profiles_mbps = {
                link: np.full(50, value, dtype=float)
                for link, value in active.items()
            }
            self.env.active_s2u_capacity_profiles_mbps = {
                link: np.full(50, value, dtype=float)
                for link, value in s2u.items()
            }
            self.env.active_s2u_capacities = s2u
            return active, {}

        self.env.allocate_active_link_capacities = allocate
        replay = ReplayBufferDiscrete(143, 17, max_size=64, n_step=1)
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

    def test_delivery_does_not_cut_next_slot_bootstrap(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)

        replay = self.run_slot(
            {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0004}
        )

        self.assertEqual(replay.size, self.env.num_UAV)
        self.assertTrue(np.all(replay.not_done[: replay.size, 0] == 1.0))
        self.assertEqual(self.stats["COM"]["timely_delivered_packets"], 1)
        self.assertEqual(self.stats["COM"]["deadline_violated_packets"], 0)

    def test_deadline_outcome_only_increments_violation_stat(self):
        packet = self.engine.create_packet(0, "COM", 100.0, 0.0)
        self.engine._set_packet_deadline(packet, 0.25)

        replay = self.run_slot(
            {0: 1}, {(0, 1): 0.001}
        )

        self.assertTrue(np.all(replay.cost[: replay.size, 0] == 0.0))
        self.assertEqual(self.stats["COM"]["timely_delivered_packets"], 0)
        self.assertEqual(self.stats["COM"]["deadline_violated_packets"], 1)

    def test_partial_packet_and_next_queued_packet_keep_bootstrap(self):
        with self.subTest(case="partial"):
            self.engine.create_packet(0, "COM", 100.0, 0.0)
            replay = self.run_slot(
                {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0002}
            )
            self.assertTrue(np.all(replay.not_done[: replay.size, 0] == 1.0))

        self.engine.reset_packet_state()
        with self.subTest(case="next packet"):
            self.engine.create_packet(0, "COM", 100.0, 0.0)
            self.engine.create_packet(0, "COM", 50.0, 0.0)
            replay = self.run_slot(
                {0: self.env.GS_ID}, {(0, self.env.GS_ID): 0.0004}
            )
            self.assertTrue(np.all(replay.not_done[: replay.size, 0] == 1.0))

    def test_same_slot_relay_arrival_keeps_sender_bootstrap(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)
        relayed = self.engine.create_packet(1, "FOV", 100.0, 0.0)

        replay = self.run_slot(
            {0: self.env.GS_ID, 1: 0},
            {(0, self.env.GS_ID): 0.0008, (1, 0): 0.0008},
        )

        self.assertTrue(relayed["done"])
        self.assertIsNone(self.engine.get_hol_packet(0))
        self.assertTrue(np.all(replay.not_done[: replay.size, 0] == 1.0))

    def test_episode_end_terminates_every_existing_transition(self):
        self.engine.create_packet(0, "COM", 100.0, 0.0)

        replay = self.run_slot(
            {0: 0}, {}, episode_done=True
        )

        self.assertEqual(replay.size, self.env.num_UAV)
        self.assertTrue(np.all(replay.not_done[: replay.size, 0] == 0.0))

    def test_empty_senders_preselect_and_create_neutral_reward_transitions(self):
        replay = self.run_slot({}, {})

        self.assertEqual(replay.size, self.env.num_UAV)
        self.assertTrue(np.all(replay.reward[: replay.size, 0] == 0.0))
        self.assertEqual(self.engine.system_qos_counts(), (0, 0))

    def test_s2u_receiver_uses_its_preselected_action_in_same_slot(self):
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
        replay = self.run_slot(
            {receiver: self.env.GS_ID},
            {
                ("S2U", 0, receiver): 0.001,
                (receiver, self.env.GS_ID): 0.001,
            },
        )

        self.assertEqual(replay.size, self.env.num_UAV)
        self.assertEqual(packet["last_routing_sender"], receiver)
        self.assertAlmostEqual(packet["routing_eligible_time"], 0.1)
        self.assertTrue(packet["done"])


if __name__ == "__main__":
    unittest.main()
