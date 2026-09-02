import unittest

import numpy as np

from HRL_task_aware import _run_routing_slot
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from utils_update_v2 import ReplayBufferDiscrete


class FixedRoutingPolicy:
    def __init__(self, actions):
        self.actions = {int(key): int(value) for key, value in actions.items()}

    def select_action(self, state, uav_id, mask=None, **kwargs):
        return self.actions[int(uav_id)]


def violation_stats():
    return {
        task: {
            "timely_delivered_packets": 0,
            "deadline_violated_packets": 0,
        }
        for task in ("FOV", "COM")
    }


class RoutingImmediateCostContractTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.source_uavs = set()
        self.env.current_time = 0.0
        self.engine = PacketEngine(num_uav=10, step_time=0.25)

    def run_slot(
        self,
        actions,
        capacities=None,
        *,
        current_time=0.0,
        done=False,
        replay=None,
    ):
        capacities = dict(capacities or {})
        self.env.allocate_active_link_capacities = (
            lambda proposed, s2u_links=None: (
                {
                    (sender, receiver): capacities.get((sender, receiver), 0.0)
                    for sender, receiver in proposed.items()
                },
                {},
            )
        )
        replay = replay or ReplayBufferDiscrete(90, 11, max_size=32, n_step=1)
        _run_routing_slot(
            self.env,
            self.engine,
            FixedRoutingPolicy(actions),
            replay,
            None,
            current_time=current_time,
            done=done,
            delay_bound_steps=20,
            violation_stats=violation_stats(),
            epsilon=0.0,
            traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
        )
        return replay

    def test_a_pre_action_expiry_is_system_outcome_not_current_cost(self):
        expired = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        expired["deadline_abs"] = 0.0
        live = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        live["deadline_abs"] = 10.0

        replay = self.run_slot({0: 0})

        self.assertEqual(replay.size, 1)
        self.assertEqual(replay.cost[0, 0], 0.0)
        self.assertEqual(self.engine.system_qos_counts(), (1, 2))
        self.assertEqual(self.engine.routing_constraint_counts(), (1, 2))

    def _behind_hol_expiry(self, receiver, capacities):
        hol = self.engine.create_packet(0, "FOV", 1e9, 0.0)
        hol["deadline_abs"] = 10.0
        for _ in range(2):
            packet = self.engine.create_packet(0, "FOV", 100.0, 0.0)
            packet["deadline_abs"] = 0.25
        replay = self.run_slot({0: receiver}, capacities)
        return replay

    def test_b_wait_counts_behind_hol_expiry(self):
        replay = self._behind_hol_expiry(0, {})
        self.assertEqual(replay.cost[0, 0], 2.0)

    def test_c_forward_counts_same_behind_hol_population(self):
        replay = self._behind_hol_expiry(1, {(0, 1): 1e-9})
        self.assertEqual(replay.cost[0, 0], 2.0)

    def test_d_and_e_post_action_relay_arrival_has_no_current_or_retroactive_cost(self):
        relayed = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        relayed["deadline_abs"] = 1.0
        resident = self.engine.create_packet(1, "FOV", 100.0, 0.0)
        resident["deadline_abs"] = 10.0
        replay = ReplayBufferDiscrete(90, 11, max_size=32, n_step=1)

        self.run_slot(
            {0: 1, 1: 1},
            {(0, 1): 0.001},
            replay=replay,
        )
        first_costs = replay.cost[: replay.size, 0].copy()
        self.assertTrue(np.all(first_costs == 0.0))
        self.assertEqual(relayed["current"], 1)

        relayed["deadline_abs"] = 0.25
        self.run_slot({1: 1}, current_time=0.25, replay=replay)

        np.testing.assert_array_equal(replay.cost[:2, 0], first_costs)
        self.assertEqual(replay.cost[2, 0], 0.0)
        self.assertEqual(self.engine.routing_constraint_counts(), (1, 2))

    def test_f_three_frozen_packets_store_raw_cost_three(self):
        for _ in range(3):
            packet = self.engine.create_packet(0, "FOV", 100.0, 0.0)
            packet["deadline_abs"] = 0.25

        replay = self.run_slot({0: 0})

        self.assertEqual(replay.size, 1)
        self.assertEqual(replay.cost[0, 0], 3.0)
        self.assertEqual(self.engine.routing_immediate_cost_sum, 3.0)

    def test_g_com_pre_s2u_violation_is_system_only(self):
        packet = self.engine.create_sr_packet(0, 100.0, 0.0)
        packet["deadline_abs"] = 0.0

        events = self.engine.expire_packets(0.0, inclusive=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(self.engine.system_qos_counts(), (1, 1))
        self.assertEqual(self.engine.routing_constraint_counts(), (0, 0))
        self.assertEqual(self.engine.pre_routing_violation_count, 1)

    def test_h_com_becomes_routing_eligible_only_after_s2u_enqueue(self):
        sr = self.env.SR_teams[0]
        self.env.multi_tasks[1] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet = self.engine.create_sr_packet(0, 100.0, 0.0)
        self.assertEqual(self.engine.routing_constraint_counts(), (0, 0))

        result = self.engine.serve_s2u_links(
            self.env,
            {(0, 1): 0.001},
            current_time=0.0,
        )

        self.assertEqual(result["arrivals"], [packet])
        self.assertEqual(self.engine.routing_constraint_counts(), (0, 1))
        packet["deadline_abs"] = 0.25
        self.engine.expire_packets(0.25, inclusive=True)
        self.assertEqual(self.engine.routing_constraint_counts(), (1, 1))
        self.assertTrue(self.engine.assert_violation_credit_conservation())

    def test_i_fov_is_routing_eligible_on_enqueue_even_before_hol(self):
        first = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        second = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        self.assertIs(self.engine.get_hol_packet(0), first)
        self.assertEqual(self.engine.routing_constraint_counts(), (0, 2))

        second["deadline_abs"] = 0.25
        self.engine.expire_packets(0.25, inclusive=True)

        self.assertEqual(self.engine.routing_constraint_counts(), (1, 2))

    def test_terminal_cost_only_uses_final_frozen_snapshot(self):
        frozen = self.engine.create_packet(0, "FOV", 100.0, 0.0)
        frozen["deadline_abs"] = 10.0

        replay = self.run_slot({0: 0}, done=True)

        self.assertEqual(replay.cost[0, 0], 1.0)
        self.assertEqual(self.engine.system_qos_counts(), (0, 1))
        self.engine.finalize_episode(0.25)
        self.assertEqual(self.engine.system_qos_counts(), (1, 1))
        self.assertEqual(self.engine.routing_constraint_counts(), (1, 1))


if __name__ == "__main__":
    unittest.main()
