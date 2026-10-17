import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from Channel_model import FADING_BLOCKS_PER_ROUTING_SLOT
from Packet_scheduler_v1 import PacketEngine, event_time_tolerance_seconds


class DeadlineEnvironment:
    """Minimal deterministic event-service environment."""

    def __init__(self, num_uav):
        self.num_UAV = int(num_uav)
        self.GS_ID = self.num_UAV
        self.GS_pos = (100.0, 0.0, 0.0)
        self.uav_dict = {
            uid: SimpleNamespace(
                get_position=(lambda uid=uid: (float(uid), 0.0, 100.0))
            )
            for uid in range(self.num_UAV)
        }
        self.B_tot = 10e6
        self.current_time = 0.0
        self.source_uavs = set()
        self.multi_tasks = {uid: [] for uid in range(self.num_UAV)}
        self.SR_teams = []
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []

    def is_routing_link_in_range(self, sender, receiver):
        return int(sender) != int(receiver)

    def is_s2u_in_range(self, sr_id, receiver):
        del sr_id, receiver
        return True

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        uav_links = sorted(
            (int(sender), int(receiver))
            for sender, receiver in dict(proposed_links).items()
            if int(sender) != int(receiver)
        )
        source_links = sorted(
            (int(sr_id), int(receiver))
            for sr_id, receiver in dict(s2u_links or {}).items()
        )
        count = len(uav_links) + len(source_links)
        bandwidth = self.B_tot / count if count else 0.0
        # 100 kbit/s per active link with one link; enough to exercise exact
        # completion/deadline ordering without stochastic channel state.
        profile = np.full(
            FADING_BLOCKS_PER_ROUTING_SLOT,
            bandwidth * 0.01 / 1e6,
            dtype=np.float64,
        )
        self.active_link_capacity_profiles_mbps = {
            link: profile.copy() for link in uav_links
        }
        self.active_s2u_capacity_profiles_mbps = {
            link: profile.copy() for link in source_links
        }
        self.active_s2u_capacities = {
            link: float(np.mean(profile)) for link in source_links
        }
        capacities = {
            link: float(np.mean(profile)) for link in uav_links
        }
        bandwidths = {link: bandwidth for link in uav_links}
        bandwidths.update(
            {("S2U", *link): bandwidth for link in source_links}
        )
        return capacities, bandwidths


def _queue_ids(engine):
    return {
        "uav": {
            uid: [int(pkt["id"]) for pkt in queue]
            for uid, queue in engine.uav_queues.items()
        },
        "sr": {
            sr_id: [int(pkt["id"]) for pkt in queue]
            for sr_id, queue in engine.sr_queues.items()
        },
    }


def _engine_state(engine):
    return {
        "queues": _queue_ids(engine),
        "active_ids": [int(pkt["id"]) for pkt in engine.get_active_packets()],
        "uav_backlog": dict(engine.backlog_bits),
        "sr_backlog": dict(engine.s2u_backlog_bits),
        "outcomes": copy.deepcopy(engine.packet_outcomes),
        "total_violated": int(engine.total_violated),
        "deadline_drops": int(engine.deadline_drops),
        "routing_immediate_cost_sum": float(engine.routing_immediate_cost_sum),
    }


def test_empty_unexpired_and_simultaneous_deadlines_scan_only_when_due():
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 0.015, "COM": 0.010}
    )
    assert math.isinf(engine.next_packet_deadline(0.0))
    assert engine.expire_packets(0.0) == []

    first = engine.create_packet(0, "COM", 10.0, 0.0)
    second = engine.create_packet(1, "COM", 20.0, 0.0)
    assert engine.next_packet_deadline(0.0) == pytest.approx(0.010)
    assert engine.expire_packets(0.009) == []
    assert engine.deadline_index_statistics()["full_scans"] == 0

    violations = engine.expire_packets(0.010)
    assert [row["packet_id"] for row in violations] == [first["id"], second["id"]]
    assert [row["packet_id"] for row in engine.packet_outcomes] == [0, 1]
    stats = engine.deadline_index_statistics()
    assert stats["full_scans"] == 1
    assert stats["uav_queue_entries_visited"] == 2
    assert stats["live_entries"] == 0
    assert math.isinf(engine.next_packet_deadline(0.010))


def test_non_hol_and_mixed_task_deadlines_are_not_missed():
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 0.005, "COM": 0.015}
    )
    hol = engine.create_packet(0, "COM", 100.0, 0.0)
    non_hol = engine.create_packet(0, "FOV", 40.0, 0.0)

    violations = engine.expire_packets(0.005)
    assert [row["packet_id"] for row in violations] == [non_hol["id"]]
    assert list(engine.uav_queues[0]) == [hol]
    assert hol["_queued_uav"] == 0
    assert non_hol["_queued_uav"] is None
    assert engine.backlog_bits[0] == pytest.approx(100.0)
    assert engine.next_packet_deadline(0.005) == pytest.approx(0.015)


def test_completion_and_early_terminal_drop_invalidate_lazy_entries():
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 0.015, "COM": 0.010}
    )
    delivered = engine.create_packet(0, "COM", 10.0, 0.0)
    engine.mark_packet_done(delivered, current_time=0.001, reason="delivered")
    assert math.isinf(engine.next_packet_deadline(0.001))
    assert engine.expire_packets(0.010) == []

    dropped = engine.create_packet(0, "FOV", 10.0, 0.020)
    engine._mark_deadline_violation(
        dropped, 0.021, sender=0, reason="max_hops"
    )
    assert math.isinf(engine.next_packet_deadline(0.021))
    before = list(engine.packet_outcomes)
    assert engine.expire_packets(1.0) == []
    assert engine.packet_outcomes == before


def test_new_earlier_deadline_reset_id_reuse_and_rebuild():
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 0.005, "COM": 0.010}
    )
    engine.create_packet(0, "COM", 10.0, 1.0)
    assert engine.next_packet_deadline(1.0) == pytest.approx(1.010)
    earlier = engine.create_packet(0, "FOV", 10.0, 1.0)
    assert engine.next_packet_deadline(1.0) == pytest.approx(1.005)
    engine._set_packet_deadline(earlier, 1.003)
    assert engine.next_packet_deadline(1.0) == pytest.approx(1.003)

    engine._deadline_heap.clear()
    engine._deadline_entries.clear()
    engine._rebuild_deadline_index()
    assert engine.next_packet_deadline(1.0) == pytest.approx(1.003)

    engine.reset_packet_state()
    assert math.isinf(engine.next_packet_deadline(0.0))
    reused = engine.create_packet(0, "COM", 10.0, 0.0)
    assert reused["id"] == 0
    assert engine.next_packet_deadline(0.0) == pytest.approx(0.010)
    assert engine.deadline_index_statistics()["live_entries"] == 1


def test_deferred_done_queue_cleanup_is_not_skipped_by_fast_path():
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0}
    )
    packet = engine.create_packet(0, "COM", 25.0, 0.0)
    engine.mark_packet_done(
        packet, current_time=0.1, reason="delivered", remove_from_queue=False
    )
    assert list(engine.uav_queues[0]) == [packet]
    assert engine.backlog_bits[0] == pytest.approx(25.0)

    assert engine.expire_packets(0.2) == []
    assert list(engine.uav_queues[0]) == []
    assert packet["_queued_uav"] is None
    assert engine.backlog_bits[0] == 0.0
    assert engine.deadline_index_statistics()["full_scans"] == 1


def test_inclusive_exclusive_tolerance_matches_reference_predicate():
    deadline = 10.010
    tolerance = event_time_tolerance_seconds(deadline, deadline)

    inclusive = PacketEngine(1)
    inclusive.create_packet(0, "COM", 1.0, 10.0)
    assert inclusive.expire_packets(deadline - tolerance, inclusive=True)

    exclusive = PacketEngine(1)
    packet = exclusive.create_packet(0, "COM", 1.0, 10.0)
    assert exclusive.expire_packets(
        deadline + tolerance, inclusive=False
    ) == []
    assert packet["done"] is False
    assert exclusive.expire_packets(
        np.nextafter(deadline + tolerance, float("inf")), inclusive=False
    )


def _run_reference_sequence(index_enabled):
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 0.015, "COM": 0.010}
    )
    engine._deadline_index_enabled = bool(index_enabled)
    engine.create_packet(1, "FOV", 31.0, 0.0)
    engine.create_packet(0, "COM", 11.0, 0.0)
    engine.create_packet(0, "FOV", 21.0, -0.005)
    engine.create_sr_packet(4, 41.0, 0.0)
    engine.create_sr_packet(3, 51.0, 0.0)
    snapshots = []
    for moment in (0.004, 0.010, 0.015, 0.020):
        violations = engine.expire_packets(moment)
        snapshots.append(
            ([row["packet_id"] for row in violations], _engine_state(engine))
        )
    return snapshots


def test_indexed_expiration_preserves_full_scan_order_and_state_exactly():
    assert _run_reference_sequence(True) == _run_reference_sequence(False)


def _run_event_slot(index_enabled, size_bits):
    env = DeadlineEnvironment(1)
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 0.015, "COM": 0.001}
    )
    engine._deadline_index_enabled = bool(index_enabled)
    packet = engine.create_packet(0, "COM", size_bits, 0.0)
    result = engine.serve_event_driven_slot(
        env,
        {0: env.GS_ID},
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: size_bits},
    )
    return {
        "result": copy.deepcopy(result),
        "packet": copy.deepcopy(packet),
        "engine": _engine_state(engine),
    }


@pytest.mark.parametrize("size_bits", [100.0, 100.1])
def test_event_completion_deadline_order_is_exact_against_reference(size_bits):
    assert _run_event_slot(True, size_bits) == _run_event_slot(False, size_bits)


def test_runtime_index_is_not_checkpointed_and_terminal_checkpoint_is_compatible():
    engine = PacketEngine(1)
    packet = engine.create_packet(0, "COM", 10.0, 0.0)
    engine.mark_packet_done(packet, current_time=0.001, reason="delivered")
    state = engine.checkpoint_state()
    assert not any("deadline" in key for key in state)
    assert state["active_packets"] == []

    restored_episode = PacketEngine(1)
    restored_episode.reset_packet_state()
    assert restored_episode.deadline_index_statistics()["heap_entries"] == 0
    assert math.isinf(restored_episode.next_packet_deadline(0.0))
