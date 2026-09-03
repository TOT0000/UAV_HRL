import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Packet_scheduler_v1 import PacketEngine


class DummyUav:
    def __init__(self, x):
        self.x_u = float(x)
        self.y_u = 0.0
        self.z_u = 0.0

    def get_position(self):
        return self.x_u, self.y_u, self.z_u


def routing_env(num_uav=3):
    return SimpleNamespace(
        GS_ID=num_uav,
        GS_pos=(0.0, 0.0, 0.0),
        uav_dict={
            uav_id: DummyUav(100.0 - 10.0 * uav_id)
            for uav_id in range(num_uav)
        },
    )


class IncrementalBacklogTest(unittest.TestCase):
    def assert_backlog_invariant(self, engine, uav_id):
        self.assertGreaterEqual(engine.backlog_bits[uav_id], 0.0)
        self.assertAlmostEqual(
            engine.backlog_bits[uav_id],
            engine.recompute_backlog_for_assertion(uav_id),
        )

    def test_source_enqueue_updates_incremental_backlog(self):
        engine = PacketEngine(num_uav=3)

        engine.create_packet(0, "COM", 100.0, 0.0)
        engine.create_packet(0, "FOV", 25.0, 0.0)

        self.assertEqual(engine.backlog_bits[0], 125.0)
        self.assert_backlog_invariant(engine, 0)

    def test_partial_transmission_decrements_incremental_backlog(self):
        engine = PacketEngine(num_uav=3)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)

        engine.record_hop_transmission(packet, 0, 2, 40.0)

        self.assertEqual(engine.backlog_bits[0], 60.0)
        self.assert_backlog_invariant(engine, 0)

    def test_relay_completion_moves_full_size_to_receiver_backlog(self):
        engine = PacketEngine(num_uav=3)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)

        engine.record_hop_transmission(packet, 0, 2, 100.0)
        arrival = engine.detach_completed_hop(packet, 0, 2, 0.1)
        self.assertEqual(engine.backlog_bits[0], 0.0)
        engine.enqueue_relay_arrivals([arrival])

        self.assertEqual(engine.backlog_bits[2], 100.0)
        self.assert_backlog_invariant(engine, 0)
        self.assert_backlog_invariant(engine, 2)

    def test_timely_gs_delivery_does_not_double_decrement(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)
        engine.create_packet(0, "COM", 100.0, 0.0)

        engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.0,
        )

        self.assertEqual(engine.backlog_bits[0], 0.0)
        self.assertEqual(engine.active_count(), 0)
        self.assert_backlog_invariant(engine, 0)

    def test_deadline_drop_decrements_only_remaining_bits(self):
        engine = PacketEngine(num_uav=3)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)
        engine.record_hop_transmission(packet, 0, 2, 40.0)

        violations = engine.expire_packets(2.0)

        self.assertEqual(len(violations), 1)
        self.assertEqual(engine.backlog_bits[0], 0.0)
        self.assert_backlog_invariant(engine, 0)

    def test_multiple_non_hol_expirations_filter_queue_once(self):
        engine = PacketEngine(num_uav=3)
        survivor = engine.create_packet(0, "FOV", 100.0, 0.75)
        expired_a = engine.create_packet(0, "COM", 30.0, 0.0)
        expired_b = engine.create_packet(0, "COM", 20.0, 0.0)

        violations = engine.expire_packets(2.0)

        self.assertEqual(len(violations), 2)
        self.assertIs(engine.get_hol_packet(0), survivor)
        self.assertEqual(engine.get_queue_packets(0), [survivor])
        self.assertTrue(expired_a["done"])
        self.assertTrue(expired_b["done"])
        self.assertEqual(engine.backlog_bits[0], 100.0)
        self.assert_backlog_invariant(engine, 0)

    def test_hot_path_never_calls_full_backlog_recomputation(self):
        env = routing_env()
        engine = PacketEngine(num_uav=3, step_time=0.25)

        with patch.object(
            engine,
            "recompute_backlog_for_assertion",
            side_effect=AssertionError("full backlog scan entered hot path"),
        ):
            packet = engine.create_packet(0, "COM", 100.0, 0.0)
            self.assertIs(engine.get_hol_packet(0), packet)
            engine.record_hop_transmission(packet, 0, 2, 40.0)
            engine.record_hop_transmission(packet, 0, 2, 60.0)
            arrival = engine.detach_completed_hop(packet, 0, 2, 0.1)
            engine.enqueue_relay_arrivals([arrival])
            engine.serve_active_links(
                env,
                actions={2: env.GS_ID},
                capacities={(2, env.GS_ID): 0.0004},
                current_time=0.25,
            )

        self.assertEqual(engine.backlog_bits[0], 0.0)
        self.assertEqual(engine.backlog_bits[2], 0.0)

    def test_ten_thousand_packet_expiration_uses_set_and_no_queue_removal(self):
        engine = PacketEngine(num_uav=1)
        for _ in range(10_000):
            engine.create_packet(0, "COM", 1.0, 0.0)

        self.assertIsInstance(engine._active_idx, set)
        source = inspect.getsource(PacketEngine.expire_packets)
        self.assertNotIn(".remove(", source)
        with patch.object(
            engine,
            "_remove_from_queue",
            side_effect=AssertionError("batch expiration used per-packet removal"),
        ) as remove_spy:
            violations = engine.expire_packets(2.0)

        self.assertEqual(remove_spy.call_count, 0)
        self.assertEqual(len(violations), 10_000)
        self.assertEqual(engine.deadline_drops, 10_000)
        self.assertEqual(engine.backlog_bits[0], 0.0)
        self.assertEqual(len(engine.uav_queues[0]), 0)
        self.assertEqual(engine.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
