from types import SimpleNamespace

import numpy as np

from Channel_model import FADING_BLOCKS_PER_ROUTING_SLOT
from Packet_scheduler_v1 import MAX_PACKET_HOPS, PacketEngine


class _NonIterableActiveIndex(set):
    """Permit point lookups, but fail if cleanup scans every active packet."""

    def __iter__(self):
        raise AssertionError("max-hop cleanup traversed the source-wide active index")


class _DeterministicEnvironment:
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
        link_count = len(uav_links) + len(source_links)
        bandwidth = self.B_tot / link_count if link_count else 0.0
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
        capacities = {link: float(np.mean(profile)) for link in uav_links}
        bandwidths = {link: bandwidth for link in uav_links}
        bandwidths.update(
            {("S2U", *link): bandwidth for link in source_links}
        )
        return capacities, bandwidths


def test_large_sr_backlog_is_not_traversed_by_max_hop_cleanup():
    engine = PacketEngine(1)
    for index in range(5_000):
        engine.create_sr_packet(0, 100.0, generation_time=float(index))

    assert len(engine._active_idx) == 5_000
    assert engine._routing_active_idx == set()
    engine._active_idx = _NonIterableActiveIndex(engine._active_idx)

    assert engine.drop_expired_packets(5_001.0) == []
    assert len(engine._active_idx) == 5_000
    assert engine._routing_active_idx == set()


def test_queued_and_detached_max_hop_packets_use_the_routing_index():
    engine = PacketEngine(2)
    below_limit = engine.create_packet(0, "FOV", 100.0, 0.0)
    at_limit = engine.create_packet(1, "COM", 200.0, 0.0)
    below_limit["hops"] = MAX_PACKET_HOPS - 1
    at_limit["hops"] = MAX_PACKET_HOPS

    violations = engine.drop_expired_packets(0.01)
    assert [row["packet_id"] for row in violations] == [at_limit["id"]]
    assert engine.packet_pool[at_limit["_pool_idx"]] is None
    assert below_limit["_pool_idx"] in engine._routing_active_idx

    assert engine._remove_from_queue(below_limit)
    below_limit["hops"] = MAX_PACKET_HOPS
    assert below_limit["_queued_uav"] is None
    assert below_limit["_pool_idx"] in engine._routing_active_idx

    violations = engine.drop_expired_packets(0.02)
    assert [row["packet_id"] for row in violations] == [below_limit["id"]]
    assert engine._active_idx == set()
    assert engine._routing_active_idx == set()


def test_delivery_s2g_terminal_finalize_and_reset_keep_index_consistent():
    env = _DeterministicEnvironment(1)
    engine = PacketEngine(1)

    routed = engine.create_packet(0, "COM", 100.0, 0.0)
    routed["hops"] = MAX_PACKET_HOPS - 1
    routed_index = routed["_pool_idx"]
    result = engine.serve_event_driven_slot(env, {0: env.GS_ID}, 0.0)
    assert [row["packet_id"] for row in result["outcomes"]] == [routed["id"]]
    assert routed["hops"] == MAX_PACKET_HOPS
    assert routed["reason"] == "delivered"
    assert engine.packet_pool[routed_index] is None
    assert routed_index not in engine._routing_active_idx

    direct = engine.create_sr_packet(0, 100.0, 0.01)
    direct_index = direct["_pool_idx"]
    assert direct_index not in engine._routing_active_idx
    direct_result = engine._event_result()
    engine._event_complete_s2g(
        direct,
        0,
        engine.num_UAV,
        0.011,
        direct_result,
    )
    assert direct["reason"] == "delivered"
    assert direct_index not in engine._routing_active_idx

    terminal_routing = engine.create_packet(0, "FOV", 50.0, 0.02)
    terminal_sr = engine.create_sr_packet(1, 75.0, 0.02)
    assert terminal_routing["_pool_idx"] in engine._routing_active_idx
    assert terminal_sr["_pool_idx"] not in engine._routing_active_idx
    engine.finalize_episode(0.03)
    assert engine._active_idx == set()
    assert engine._routing_active_idx == set()

    reset_packet = engine.create_packet(0, "FOV", 25.0, 0.04)
    assert reset_packet["_pool_idx"] in engine._routing_active_idx
    engine.reset_packet_state()
    assert engine.packet_pool == []
    assert engine._active_idx == set()
    assert engine._routing_active_idx == set()


def test_checkpoint_contract_remains_episode_boundary_and_index_is_derived():
    engine = PacketEngine(1)
    packet = engine.create_packet(0, "FOV", 100.0, 0.0)
    assert packet["_pool_idx"] in engine._routing_active_idx

    active_state = engine.checkpoint_state()
    assert active_state["mid_episode_checkpoint_supported"] is False
    assert active_state["checkpoint_scope"] == "episode_boundary_terminal_snapshot"
    assert "routing_active_idx" not in active_state

    engine.finalize_episode(0.01)
    terminal_state = engine.checkpoint_state()
    assert terminal_state["active_packets"] == []
    assert all(not ids for ids in terminal_state["uav_queue_packet_ids"].values())
    assert all(not ids for ids in terminal_state["sr_queue_packet_ids"].values())


def test_repeated_routing_entry_is_idempotent_for_index_and_qos_count():
    engine = PacketEngine(1)
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    pool_idx = packet["_pool_idx"]
    before = dict(engine.routing_stage_eligible_packet_counts)

    engine._routing_active_idx.clear()
    assert engine._mark_routing_stage_eligible(packet, 0.01) is False
    assert engine._routing_active_idx == {pool_idx}
    assert engine.routing_stage_eligible_packet_counts == before


def test_stale_routing_entries_are_cleaned_without_dropping_live_sr_packets():
    engine = PacketEngine(1)
    source_packet = engine.create_sr_packet(0, 100.0, 0.0)
    stale_index = 10_000
    engine._routing_active_idx.update(
        {source_packet["_pool_idx"], stale_index}
    )

    engine.drop_expired_packets(0.01)

    assert engine._routing_active_idx == set()
    assert source_packet["_pool_idx"] in engine._active_idx
    assert engine.packet_pool[source_packet["_pool_idx"]] is source_packet
    assert stale_index not in engine._active_idx
