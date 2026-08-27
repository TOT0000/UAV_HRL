import unittest
from types import SimpleNamespace

from Packet_scheduler_v1 import (
    EPISODE_INJECTION_CUTOFF_SECONDS,
    PacketEngine,
    TASK_DEADLINE_SECONDS,
)


class PaperPacketMetricTest(unittest.TestCase):
    def test_terminal_outcomes_are_mutually_exclusive_and_conserved(self):
        engine = PacketEngine(num_uav=10, step_time=0.25)

        on_time = engine.create_packet(0, "FOV", 100.0, 0.0)
        engine.mark_packet_done(on_time, current_time=0.5, reason="delivered")

        late = engine.create_packet(1, "COM", 100.0, 0.0)
        engine._mark_deadline_violation(
            late, 1.2, sender=1, reason="late_delivered"
        )

        expired = engine.create_packet(2, "FOV", 100.0, 0.0)
        engine.expire_packets(1.5)

        dropped = engine.create_packet(3, "COM", 100.0, 0.0)
        engine._mark_deadline_violation(
            dropped, 0.75, sender=3, reason="max_hops"
        )

        engine.create_packet(4, "FOV", 100.0, 0.25)
        summary = engine.finalize_episode(1.0)

        self.assertEqual(len(engine.packet_outcomes), 5)
        self.assertEqual(len({row["packet_id"] for row in engine.packet_outcomes}), 5)
        self.assertEqual(summary["FOV"]["generated_packets"], 3)
        self.assertEqual(summary["FOV"]["on_time_delivered_packets"], 1)
        self.assertEqual(summary["FOV"]["expired_dropped_packets"], 2)
        self.assertAlmostEqual(summary["FOV"]["average_e2e_delay_seconds"], 0.5)
        self.assertAlmostEqual(summary["FOV"]["violation_probability"], 2 / 3)
        self.assertEqual(summary["COM"]["generated_packets"], 2)
        self.assertEqual(summary["COM"]["late_delivered_packets"], 1)
        self.assertEqual(summary["COM"]["expired_dropped_packets"], 1)
        self.assertAlmostEqual(summary["COM"]["average_e2e_delay_seconds"], 1.2)
        self.assertEqual(summary["COM"]["violation_probability"], 1.0)
        self.assertEqual(engine.total_violated, 4)
        self.assertEqual(engine.deadline_drops, 4)

    def test_no_delivered_packet_is_missing_not_fake_zero(self):
        engine = PacketEngine(num_uav=10)
        engine.create_packet(0, "COM", 100.0, 0.0)
        summary = engine.finalize_episode(0.5)
        self.assertIsNone(summary["COM"]["average_e2e_delay_seconds"])

    def test_deadline_override_is_instance_scoped(self):
        production = dict(TASK_DEADLINE_SECONDS)
        short = PacketEngine(
            num_uav=10,
            task_deadlines_seconds={"FOV": 0.5, "COM": 2.5},
        )
        normal = PacketEngine(num_uav=10)
        self.assertEqual(short.create_packet(0, "FOV", 1.0, 0.0)["deadline"], 0.5)
        self.assertEqual(normal.create_packet(0, "FOV", 1.0, 0.0)["deadline"], 1.5)
        self.assertEqual(TASK_DEADLINE_SECONDS, production)

    def test_rate_override_does_not_leak_to_the_next_injection(self):
        env = SimpleNamespace(
            source_uavs=set(),
            multi_tasks={
                0: [{"task_type": "COM", "target_obj_id": 0}]
            },
            SR_teams=[SimpleNamespace(id=0, assigned_gt_id=0, arrived=False)],
            load_factor=1.0,
            is_s2u_in_range=lambda sr_id, uav_id: True,
        )
        engine = PacketEngine(num_uav=1, step_time=0.25)
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=0.0,
            step_time=0.25,
            base_ctrl_rate=0.0,
            rate_overrides={"FOV": 0.0, "COM": 4.0},
        )
        self.assertEqual(engine.generated_packet_counts["COM"], 1)
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=0.25,
            step_time=0.25,
            base_ctrl_rate=0.0,
        )
        self.assertEqual(engine.generated_packet_counts["COM"], 1)

    def test_packet_injection_cutoff_override_is_instance_scoped(self):
        production = PacketEngine(num_uav=1)
        deadline_sweep = PacketEngine(
            num_uav=1, injection_cutoff_seconds=57.0
        )
        self.assertEqual(production.injection_cutoff_seconds, 58.5)
        self.assertEqual(deadline_sweep.injection_cutoff_seconds, 57.0)
        self.assertEqual(EPISODE_INJECTION_CUTOFF_SECONDS, 58.5)


if __name__ == "__main__":
    unittest.main()
