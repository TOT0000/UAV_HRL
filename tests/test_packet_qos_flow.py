import unittest
from types import SimpleNamespace

from Packet_scheduler_v1 import PacketEngine


class DummyUav:
    def __init__(self, x):
        self.x_u = float(x)
        self.y_u = 0.0


def routing_env(num_uav=3):
    return SimpleNamespace(
        GS_ID=num_uav,
        GS_pos=(0.0, 0.0, 0.0),
        uav_dict={uav_id: DummyUav(100.0 - 10.0 * uav_id)
                  for uav_id in range(num_uav)},
    )


class PacketQosFlowTest(unittest.TestCase):
    def test_fifo_packets_share_one_link_slot_budget(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packets = [
            engine.create_packet(0, "COM", size, 0.0)
            for size in (30.0, 40.0, 50.0)
        ]

        result = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0003},
            current_time=0.0,
        )

        budget = 0.0003 * 1e6 * 0.25
        self.assertAlmostEqual(
            result["transmitted_bits_by_link"][(0, env.GS_ID)], budget
        )
        self.assertLessEqual(
            result["transmitted_bits_by_link"][(0, env.GS_ID)], budget + 1e-9
        )
        self.assertTrue(packets[0]["done"])
        self.assertTrue(packets[1]["done"])
        self.assertFalse(packets[2]["done"])
        self.assertEqual(packets[2]["rem_bits"], 45.0)
        self.assertEqual(engine.get_hol_packet(0), packets[2])

    def test_wait_does_not_transmit_or_consume_a_link_budget(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)

        result = engine.serve_active_links(
            env, actions={0: 0}, capacities={}, current_time=0.0
        )

        self.assertEqual(result["transmitted_bits_by_link"], {})
        self.assertEqual(packet["rem_bits"], 100.0)
        self.assertEqual(engine.wait_actions, 1)

    def test_actual_e2e_uses_gs_completion_time_and_is_not_state_estimate(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)

        engine.log_hop_delay(
            env,
            packet,
            current_node=0,
            next_hop=env.GS_ID,
            link_capacity_mbps=1.0,
            current_time=0.5,
            pkt_bits=100.0,
            backlog_bits=1000.0,
        )
        engine.log_hop_delay(
            env,
            packet,
            current_node=0,
            next_hop=env.GS_ID,
            link_capacity_mbps=1.0,
            current_time=0.5,
            pkt_bits=100.0,
            backlog_bits=1000.0,
        )
        self.assertEqual(packet["e2e_delay_ms"], 0.0)

        engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.001},
            current_time=0.5,
        )
        self.assertAlmostEqual(packet["finish_time"], 0.6)
        self.assertAlmostEqual(packet["e2e_delay_ms"], 600.0)

    def test_deadlines_exact_completion_and_expired_partial_cost(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        com = engine.create_packet(0, "COM", 100.0, 0.0)
        fov = engine.create_packet(1, "FOV", 100.0, 0.0)
        self.assertEqual(com["deadline_abs"], 1.0)
        self.assertEqual(fov["deadline_abs"], 1.5)

        timely = engine.serve_active_links(
            env,
            actions={0: env.GS_ID, 1: 1},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.75,
        )
        self.assertEqual(timely["timely_goodput_bits"], 100.0)
        self.assertEqual(timely["cost_by_sender"][0], 0.0)

        expired_engine = PacketEngine(num_uav=3, step_time=0.25)
        expired = expired_engine.create_packet(0, "COM", 100.0, 0.0)
        missed = expired_engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0002},
            current_time=0.75,
        )
        self.assertTrue(expired["done"])
        self.assertEqual(expired["reason"], "deadline")
        self.assertEqual(missed["cost_by_sender"][0], 1.0)
        self.assertEqual(missed["timely_goodput_bits"], 0.0)
        self.assertEqual(missed["raw_final_hop_bits"], 50.0)
        self.assertEqual(expired_engine.deadline_drops, 1)
        self.assertEqual(expired_engine.expire_packets(1.25), [])
        self.assertEqual(expired_engine.deadline_drops, 1)

    def test_injection_cutoff_is_strictly_before_58_5_seconds(self):
        env = SimpleNamespace(
            source_uavs={0},
            multi_tasks={0: [{"task_type": "COM"}]},
            load_factor=1.0,
        )
        engine = PacketEngine(num_uav=1, step_time=0.25)
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=58.25,
            step_time=0.25,
            base_ctrl_rate=4,
        )
        self.assertEqual(engine.active_count(), 1)
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=58.5,
            step_time=0.25,
            base_ctrl_rate=4,
        )
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=59.0,
            step_time=0.25,
            base_ctrl_rate=4,
        )
        self.assertEqual(engine.active_count(), 1)

    def test_partial_final_hop_is_raw_only_and_timely_goodput_counts_once(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)

        partial = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0002},
            current_time=0.0,
        )
        self.assertEqual(partial["raw_final_hop_bits"], 50.0)
        self.assertEqual(partial["timely_goodput_bits"], 0.0)
        self.assertEqual(packet["final_hop_accum_bits"], 50.0)

        completed = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0002},
            current_time=0.25,
        )
        self.assertEqual(completed["raw_final_hop_bits"], 50.0)
        self.assertEqual(completed["timely_goodput_bits"], 100.0)
        self.assertEqual(engine.raw_final_hop_bits, 100.0)
        self.assertEqual(engine.timely_goodput_bits, 100.0)
        self.assertEqual(packet["final_hop_accum_bits"], 100.0)

        repeated = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0002},
            current_time=0.5,
        )
        self.assertEqual(repeated["timely_goodput_bits"], 0.0)
        self.assertEqual(engine.timely_goodput_bits, 100.0)
        self.assertTrue(packet["timely_goodput_counted"])

    def test_full_relay_arrival_cannot_forward_again_in_same_slot(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)

        result = engine.serve_active_links(
            env,
            actions={0: 2, 2: env.GS_ID},
            capacities={(0, 2): 0.001, (2, env.GS_ID): 0.001},
            current_time=0.0,
        )

        self.assertEqual(result["transmitted_bits_by_link"][(0, 2)], 100.0)
        self.assertEqual(
            result["transmitted_bits_by_link"][(2, env.GS_ID)], 0.0
        )
        self.assertIs(engine.get_hol_packet(2), packet)
        self.assertEqual(packet["current"], 2)
        self.assertFalse(packet["done"])


if __name__ == "__main__":
    unittest.main()
