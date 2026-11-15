import copy
import json
from types import SimpleNamespace

import numpy as np

from HRL_task_aware import _run_routing_slot
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from rng_contract import NamedRNGStreams
from routing_loop_diagnostics import (
    RoutingLoopDiagnosticsCollector,
    verify_episode_jsonl_prefix,
)
from scenario_manifest import generate_manifest, manifest_prefix
from utils_update_v2 import ReplayBufferDiscrete


class TopologyEnv:
    def __init__(self, edges):
        self.num_UAV = 3
        self.GS_ID = 3
        self.GS_pos = (0.0, 0.0, 0.0)
        self.uav_dict = {
            uid: SimpleNamespace(get_position=lambda uid=uid: (uid * 10.0, 0.0, 100.0))
            for uid in range(3)
        }
        self.edges = {tuple(sorted(edge)) for edge in edges}

    def is_routing_link_in_range(self, sender, receiver):
        return tuple(sorted((int(sender), int(receiver)))) in self.edges


def _masks(*legal_by_sender):
    result = {}
    for sender, legal in enumerate(legal_by_sender):
        mask = np.zeros(4, dtype=bool)
        mask[list(legal)] = True
        result[sender] = mask
    return result


def _slot(collector, env, *, slot=0, hol=None, masks=None, actions=None):
    masks = masks or _masks((0, 1), (0, 1), (2,))
    actions = actions or {0: 1, 1: 0, 2: 2}
    hol = hol or {0: None, 1: None, 2: None}
    collector.observe_slot(
        env=env,
        packet_engine=None,
        current_time=slot * 0.25,
        slot_index=slot,
        physical_masks=masks,
        effective_masks=masks,
        selection_masks=masks,
        selected_actions=actions,
        start_of_slot_hol_by_sender=hol,
    )


def _hop(collector, phase, packet_id, sender, receiver, number, completed_hops, **extra):
    event = {
        "phase": phase,
        "packet_id": packet_id,
        "task_type": "COM",
        "sender_uav_id": sender,
        "receiver_id": receiver,
        "link_type": "U2U",
        "service_start_time_seconds": extra.get("service_start", 0.0),
        "completion_time_seconds": extra.get("completion_time", 0.001),
        "cross_slot_transmission": extra.get("cross_slot", False),
        "hop_sequence_number": number,
        "completed_hops": completed_hops,
    }
    collector.observe_hop_event(event)


def test_foldback_uses_original_selection_mask_and_reports_alternatives(tmp_path):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=2, scenario_id="s", episode_index=0)
    env = TopologyEnv({(0, 1), (1, 2), (1, 3)})
    masks = _masks((0, 1), (0, 1, 2, 3), (1, 2))
    packet = {"id": 7, "task_type": "COM", "hop_receiver": None}
    _slot(collector, env, hol={0: packet, 1: None, 2: None}, masks=masks)
    first = [{"from": 0, "to": 1, "link_type": "U2U"}]
    _hop(collector, "started", 7, 0, 1, 1, ())
    _hop(collector, "completed", 7, 0, 1, 1, first)
    _hop(collector, "started", 7, 1, 0, 2, ())
    _hop(collector, "completed", 7, 1, 0, 2, first + [{"from": 1, "to": 0, "link_type": "U2U"}])
    collector.end_episode(
        [{"packet_id": 7, "task_type": "COM", "outcome": "expired_dropped", "completed_uav_hop_count": 2, "delivered_to_gs": False}]
    )
    collector.finalize()

    event = json.loads((tmp_path / "routing_loop_events.jsonl").read_text().strip())
    assert event["alternative_actual_legal_next_hops"] == [2, 3]
    assert event["alternative_actual_legal_next_hop_count"] == 2
    assert event["physical_gs_reachable_at_decision"] is True
    assert event["packet_decision_relation"] == "reused_empty_queue_preselected_action"
    assert event["reused_preselected_action"] is True


def test_only_foldback_target_and_disconnected_topology_are_distinct(tmp_path):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=3, scenario_id="disconnected", episode_index=0)
    env = TopologyEnv({(0, 1), (1, 2)})
    masks = _masks((0, 1), (0, 1), (1, 2))
    packet = {"id": 9, "task_type": "FOV", "hop_receiver": None}
    _slot(collector, env, hol={0: None, 1: packet, 2: None}, masks=masks)
    hops = [{"from": 0, "to": 1, "link_type": "U2U"}]
    _hop(collector, "started", 9, 1, 0, 2, ())
    _hop(collector, "completed", 9, 1, 0, 2, hops + [{"from": 1, "to": 0, "link_type": "U2U"}])
    collector.end_episode(
        [{"packet_id": 9, "task_type": "FOV", "outcome": "on_time_delivered", "completed_uav_hop_count": 2, "delivered_to_gs": True}]
    )
    collector.finalize()
    event = json.loads((tmp_path / "routing_loop_events.jsonl").read_text().strip())
    assert event["alternative_actual_legal_next_hops"] == []
    assert event["physical_gs_reachable_at_decision"] is False
    assert event["was_slot_start_hol"] is True


def test_cross_slot_origin_s2u_unfinished_and_twentieth_hop_delivery(tmp_path):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=4, scenario_id="cross-slot", episode_index=0)
    env = TopologyEnv({(0, 1), (1, 3)})
    packet = {"id": 12, "task_type": "COM", "hop_receiver": None}
    _slot(collector, env, hol={0: None, 1: packet, 2: None})
    _hop(collector, "started", 12, 1, 0, 2, (), service_start=0.0)
    # S2U and an unfinished start cannot create a foldback event.
    collector.observe_hop_event(
        {"phase": "completed", "packet_id": 99, "task_type": "COM", "sender_uav_id": 0, "receiver_id": 1, "link_type": "S2U", "hop_sequence_number": 1}
    )
    _hop(collector, "started", 98, 0, 1, 1, ())
    packet["hop_receiver"] = 0
    _slot(collector, env, slot=1, hol={0: None, 1: packet, 2: None})
    hops = [
        {"from": 0, "to": 1, "link_type": "U2U"},
        {"from": 1, "to": 0, "link_type": "U2U"},
    ]
    _hop(
        collector,
        "completed",
        12,
        1,
        0,
        2,
        hops,
        cross_slot=True,
        completion_time=0.251,
    )
    # A 20th completed U2G hop is delivery, never HopLimit.
    for number in range(1, 21):
        collector.observe_hop_event(
            {
                "phase": "completed",
                "packet_id": 20,
                "task_type": "COM",
                "sender_uav_id": 1,
                "receiver_id": 3,
                "link_type": "U2G",
                "completion_time_seconds": 0.252 + number * 1e-6,
                "cross_slot_transmission": False,
                "hop_sequence_number": number,
                "completed_hops": (),
            }
        )
    collector.end_episode(
        [
            {"packet_id": 12, "task_type": "COM", "outcome": "late_delivered", "completed_uav_hop_count": 2, "delivered_to_gs": True},
            {"packet_id": 20, "task_type": "COM", "outcome": "on_time_delivered", "completed_uav_hop_count": 20, "delivered_to_gs": True},
        ]
    )
    collector.finalize()
    events = [json.loads(line) for line in (tmp_path / "routing_loop_events.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["cross_slot_transmission"] is True
    assert events[0]["decision_slot_index"] == 0
    assert events[0]["completion_slot_index"] == 1
    assert events[0]["receiver_lock_at_completion_slot_start"] == 0
    assert events[0]["completion_slot_alternative_actual_legal_next_hops"] == []
    assert (
        events[0]["completion_slot_physical_alternative_blocked_by_mask_or_lock"]
        is True
    )
    summary = json.loads((tmp_path / "routing_loop_summary.json").read_text())
    all_row = next(row for row in summary["groups"] if row["group"] == "ALL")
    assert all_row["twenty_hop_delivered_to_gs"] == 1
    assert all_row["twenty_hop_hop_limit_not_delivered"] == 0
    assert all_row["twenty_hop_all_completed_in_one_slot_ratio"] == 1.0


class FixedRoutingPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.exploration_rng = np.random.default_rng(123)

    def select_action(self, state, uav_id, mask=None, **kwargs):
        del state, mask, kwargs
        return int(self.actions.get(int(uav_id), int(uav_id)))


def _nonintervention_run(tmp_path, enabled):
    streams = NamedRNGStreams(20261115)
    env = Simulator(num_UAV=16, rng_streams=streams)
    env.num_GT = 2
    env.reset_environment()
    env.source_uavs = set()
    engine = PacketEngine(16, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0})
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    agent = FixedRoutingPolicy({0: env.GS_ID})
    replay = ReplayBufferDiscrete(143, 17, max_size=32, n_step=1)
    collector = RoutingLoopDiagnosticsCollector(tmp_path) if enabled else None
    if collector is not None:
        collector.begin_episode(roi_count=2, scenario_id="same", episode_index=0)
    result = _run_routing_slot(
        env,
        engine,
        agent,
        replay,
        None,
        current_time=0.0,
        done=True,
        delay_bound_steps=20,
        violation_stats={
            task: {"timely_delivered_packets": 0, "deadline_violated_packets": 0}
            for task in ("FOV", "COM")
        },
        epsilon=0.0,
        traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
        routing_loop_diagnostics=collector,
    )
    engine.finalize_episode(0.25)
    if collector is not None:
        collector.end_episode(engine.packet_outcomes)
        collector.finalize()
    return {
        "result": copy.deepcopy(result),
        "path": copy.deepcopy(packet["path"]),
        "outcomes": copy.deepcopy(engine.packet_outcomes),
        "generated": copy.deepcopy(engine.generated_packet_counts),
        "delivered_bits": engine.total_all_useful_bits,
        "energy": np.asarray(engine.energy).tolist(),
        "reward": replay.reward[: replay.size].tolist(),
        "cost": replay.cost[: replay.size].tolist(),
        "actions": replay.action[: replay.size].tolist(),
        "simulation_time": env.current_time,
        "environment_rng": copy.deepcopy(streams.state_dict()),
        "agent_rng": copy.deepcopy(agent.exploration_rng.bit_generator.state),
    }


def test_diagnostics_on_off_are_nonintervening_including_rng(tmp_path):
    assert _nonintervention_run(tmp_path / "on", True) == _nonintervention_run(
        tmp_path / "off", False
    )


def test_manifest_prefix_preserves_exact_source_entries():
    source = generate_manifest("test", 20261115, 4, num_gt=2)
    prefix = manifest_prefix(source, 2)
    assert list(prefix.episodes) == list(source.episodes[:2])
    assert [row["scenario_id"] for row in prefix.episodes] == [
        row["scenario_id"] for row in source.episodes[:2]
    ]
    assert prefix.content_hash != source.content_hash


def test_episode_prefix_comparison_allows_only_manifest_hash_to_differ(tmp_path):
    source = tmp_path / "source.jsonl"
    actual = tmp_path / "actual.jsonl"
    source.write_text(
        json.dumps(
            {"scenario_id": "s", "evaluation_manifest_hash": "full", "reward": 1.25}
        )
        + "\n",
        encoding="utf-8",
    )
    actual.write_text(
        json.dumps(
            {"scenario_id": "s", "evaluation_manifest_hash": "prefix", "reward": 1.25}
        )
        + "\n",
        encoding="utf-8",
    )
    assert verify_episode_jsonl_prefix(actual, source, ["s"]) == 1
