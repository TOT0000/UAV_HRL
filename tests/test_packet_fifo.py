import unittest

from Packet_scheduler_v1 import PacketEngine


class PerUavFifoTest(unittest.TestCase):
    def test_relay_arrival_uses_local_queue_entry_order_not_generation_time(self):
        engine = PacketEngine(num_uav=4, step_time=0.25)
        older_remote = engine.create_packet(0, "FOV", 100.0, 0.0)
        local_packet = engine.create_packet(2, "COM", 50.0, 1.0)

        self.assertTrue(
            engine.record_hop_transmission(older_remote, 0, 2, 100.0)
        )
        arrival = engine.detach_completed_hop(
            older_remote, 0, 2, completion_time=2.0
        )

        # A full relay arrival is not routable until the slot finishes.
        self.assertIs(engine.get_hol_packet(2), local_packet)
        engine.enqueue_relay_arrivals([arrival])

        self.assertEqual(
            [pkt["id"] for pkt in engine.get_queue_packets(2)],
            [local_packet["id"], older_remote["id"]],
        )
        self.assertEqual(engine.backlog_bits[2], 150.0)

    def test_partial_hop_is_receiver_locked_until_full_arrival(self):
        engine = PacketEngine(num_uav=4, step_time=0.25)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)

        self.assertFalse(engine.record_hop_transmission(packet, 0, 2, 40.0))
        self.assertEqual(packet["hop_receiver"], 2)
        self.assertEqual(packet["current"], 0)
        self.assertEqual(engine.backlog_bits[0], 60.0)
        self.assertIsNone(engine.get_hol_packet(2))

        with self.assertRaisesRegex(AssertionError, "locked"):
            engine.record_hop_transmission(packet, 0, 1, 10.0)

        self.assertTrue(engine.record_hop_transmission(packet, 0, 2, 60.0))
        arrival = engine.detach_completed_hop(
            packet, 0, 2, completion_time=0.5
        )
        self.assertIsNone(engine.get_hol_packet(2))
        engine.enqueue_relay_arrivals([arrival])

        self.assertIs(engine.get_hol_packet(2), packet)
        self.assertEqual(packet["current"], 2)
        self.assertIsNone(packet["hop_receiver"])
        self.assertEqual(packet["rem_bits"], packet["size_bits"])

    def test_same_time_relay_arrivals_have_deterministic_sender_packet_order(self):
        engine = PacketEngine(num_uav=4, step_time=0.25)
        packets = [
            engine.create_packet(1, "COM", 10.0, 0.0),
            engine.create_packet(0, "COM", 10.0, 0.0),
        ]
        arrivals = []
        for sender, packet in ((1, packets[0]), (0, packets[1])):
            engine.record_hop_transmission(packet, sender, 2, 10.0)
            arrivals.append(
                engine.detach_completed_hop(
                    packet, sender, 2, completion_time=0.25
                )
            )

        ordered = engine.enqueue_relay_arrivals(arrivals)
        self.assertEqual([item["sender"] for item in ordered], [0, 1])
        self.assertEqual(
            [pkt["source"] for pkt in engine.get_queue_packets(2)], [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
