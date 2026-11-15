import copy
import csv
import json
from types import SimpleNamespace

import numpy as np

from DDQN import forwarding_or_forced_wait_mask
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


class LoopPacketEnv:
    """Deterministic event-service environment with no fading RNG."""

    def __init__(self, bits_per_second_per_hz=0.01):
        self.num_UAV = 2
        self.GS_ID = 2
        self.GS_pos = (100.0, 0.0, 0.0)
        self.uav_dict = {
            uid: SimpleNamespace(
                get_position=lambda uid=uid: (float(uid), 0.0, 100.0)
            )
            for uid in range(2)
        }
        self.B_tot = 10e6
        self.bits_per_second_per_hz = float(bits_per_second_per_hz)
        self.current_time = 0.0
        self.source_uavs = set()
        self.multi_tasks = {0: [], 1: []}
        self.SR_teams = []
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []

    def is_routing_link_in_range(self, sender, receiver):
        return int(sender) != int(receiver)

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        links = [
            (int(sender), int(receiver))
            for sender, receiver in sorted(dict(proposed_links).items())
            if int(sender) != int(receiver)
        ]
        source_links = list(dict(s2u_links or {}).items())
        count = len(links) + len(source_links)
        bandwidth = self.B_tot / count if count else 0.0
        capacity_mbps = bandwidth * self.bits_per_second_per_hz / 1e6
        profile = np.full(50, capacity_mbps, dtype=float)
        self.active_link_capacity_profiles_mbps = {
            link: profile.copy() for link in links
        }
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []
        return (
            {link: capacity_mbps for link in links},
            {link: bandwidth for link in links},
        )


def _observe_two_uav_slot(collector, env, engine, slot, packet):
    effective = {0: np.ones(3, dtype=bool), 1: np.ones(3, dtype=bool)}
    selection = {
        uid: forwarding_or_forced_wait_mask(effective[uid], uid, 3)
        for uid in range(2)
    }
    collector.observe_slot(
        env=env,
        packet_engine=engine,
        current_time=slot * engine.step_time,
        slot_index=slot,
        physical_masks=effective,
        effective_masks=effective,
        selection_masks=selection,
        selected_actions={0: 1, 1: 0},
        start_of_slot_hol_by_sender={
            0: engine.get_hol_packet(0),
            1: engine.get_hol_packet(1),
        },
    )
    return effective


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


def test_packet_engine_twenty_hop_foldback_marks_all_19_as_reused(tmp_path):
    env = LoopPacketEnv()
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0}
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=2, scenario_id="engine-loop", episode_index=0)
    effective = _observe_two_uav_slot(collector, env, engine, 0, packet)
    engine.serve_event_driven_slot(
        env,
        {0: 1, 1: 0},
        0.0,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: 100.0, 1: 0.0},
        effective_masks_by_sender=effective,
        uav_hop_observer=collector.observe_hop_event,
    )
    assert packet["hops"] == 20
    assert packet["terminal_outcome"] == "expired_dropped"
    collector.end_episode(
        [{
            "packet_id": packet["id"],
            "task_type": "COM",
            "outcome": "expired_dropped",
            "completed_uav_hop_count": 20,
            "delivered_to_gs": False,
        }]
    )
    collector.finalize()

    events = [
        json.loads(line)
        for line in (tmp_path / "routing_loop_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 19
    assert all(row["reused_preselected_action"] for row in events)
    assert sum(row["reused_same_packet_preselected_action"] for row in events) == 18
    assert events[1]["hop_sequence_number"] == 3
    assert events[1]["was_original_decision_slot_start_hol"] is True
    assert events[1]["packet_decision_relation"] == (
        "reused_same_packet_preselected_action"
    )

    pair_rows = list(
        csv.DictReader(
            (tmp_path / "routing_loop_pairs.csv").open(encoding="utf-8-sig")
        )
    )
    pair_all = next(
        row
        for row in pair_rows
        if row["group"] == "ALL" and row["uav_a"] == "0" and row["uav_b"] == "1"
    )
    assert pair_all["twenty_hop_terminal_packet_count"] == "1"
    assert pair_all["twenty_hop_all_completed_in_one_slot"] == "1"
    assert pair_all["twenty_hop_all_completed_in_one_slot_ratio"] == "1.0"


def test_packet_engine_cross_slot_partial_preserves_origin_and_reuse(tmp_path):
    # 200 bit/s gives 50 bits in slot 0: hop 1 is partial.  It completes in
    # slot 1, then the packet returns to A and reuses A's slot-1 decision.
    env = LoopPacketEnv(bits_per_second_per_hz=0.00002)
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 2.0, "COM": 2.0}
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=3, scenario_id="partial", episode_index=0)
    effective = _observe_two_uav_slot(collector, env, engine, 0, packet)
    engine.serve_event_driven_slot(
        env, {0: 1, 1: 0}, 0.0,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: 100.0, 1: 0.0},
        effective_masks_by_sender=effective,
        uav_hop_observer=collector.observe_hop_event,
    )
    assert packet["hops"] == 0
    assert packet["hop_receiver"] == 1

    env.bits_per_second_per_hz = 0.01
    effective = _observe_two_uav_slot(collector, env, engine, 1, packet)
    engine.serve_event_driven_slot(
        env, {0: 1, 1: 0}, 0.25,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: 50.0, 1: 0.0},
        effective_masks_by_sender=effective,
        uav_hop_observer=collector.observe_hop_event,
    )
    collector.end_episode([{
        "packet_id": packet["id"], "task_type": "COM",
        "outcome": "expired_dropped", "completed_uav_hop_count": packet["hops"],
        "delivered_to_gs": False,
    }])
    collector.finalize()
    events = [
        json.loads(line)
        for line in (tmp_path / "routing_loop_events.jsonl").read_text().splitlines()
    ]
    hop2 = next(row for row in events if row["hop_sequence_number"] == 2)
    hop3 = next(row for row in events if row["hop_sequence_number"] == 3)
    assert hop2["cross_slot_transmission"] is False
    assert hop2["decision_slot_index"] == 1
    assert hop2["completion_slot_index"] == 1
    assert hop2["packet_decision_relation"] == "reused_empty_queue_preselected_action"
    assert hop3["decision_slot_index"] == 1
    assert hop3["was_original_decision_slot_start_hol"] is True
    assert hop3["reused_same_packet_preselected_action"] is True


def test_prior_departure_under_older_decision_does_not_fake_same_decision_reuse(
    tmp_path,
):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=3, scenario_id="fresh-slot", episode_index=0)
    env = TopologyEnv({(0, 1)})
    packet = {"id": 44, "task_type": "COM", "hop_receiver": None}
    _slot(collector, env, slot=0, hol={0: packet, 1: None, 2: None})
    path = []
    for number, sender, receiver in ((1, 0, 1), (2, 1, 0)):
        _hop(collector, "started", 44, sender, receiver, number, tuple(path))
        path.append({"from": sender, "to": receiver, "link_type": "U2U"})
        _hop(collector, "completed", 44, sender, receiver, number, tuple(path))

    # A new slot makes a new packet-specific decision while packet 44 is HOL.
    _slot(collector, env, slot=1, hol={0: packet, 1: None, 2: None})
    _hop(collector, "started", 44, 0, 1, 3, tuple(path))
    path.append({"from": 0, "to": 1, "link_type": "U2U"})
    _hop(collector, "completed", 44, 0, 1, 3, tuple(path))
    collector.end_episode([{
        "packet_id": 44, "task_type": "COM", "outcome": "expired_dropped",
        "completed_uav_hop_count": 3, "delivered_to_gs": False,
    }])
    collector.finalize()
    events = [
        json.loads(line)
        for line in (tmp_path / "routing_loop_events.jsonl").read_text().splitlines()
    ]
    hop3 = next(row for row in events if row["hop_sequence_number"] == 3)
    assert hop3["was_original_decision_slot_start_hol"] is True
    assert hop3["reused_same_packet_preselected_action"] is False
    assert hop3["packet_decision_relation"] == "first_departure_slot_start_hol"


def _cross_slot_loop_nonintervention_run(tmp_path, enabled):
    streams = NamedRNGStreams(314159)
    env = LoopPacketEnv(bits_per_second_per_hz=0.00006)
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 2.0, "COM": 2.0}
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    collector = RoutingLoopDiagnosticsCollector(tmp_path) if enabled else None
    if collector is not None:
        collector.begin_episode(roi_count=4, scenario_id="same", episode_index=0)
    results = []
    for slot, bps_per_hz in ((0, 0.00006), (1, 0.01)):
        env.bits_per_second_per_hz = bps_per_hz
        effective = {0: np.ones(3, dtype=bool), 1: np.ones(3, dtype=bool)}
        if collector is not None:
            _observe_two_uav_slot(collector, env, engine, slot, packet)
        results.append(
            engine.serve_event_driven_slot(
                env, {0: 1, 1: 0}, slot * engine.step_time,
                observed_hol_by_sender={
                    0: engine.get_hol_packet(0),
                    1: engine.get_hol_packet(1),
                },
                observed_backlog_bits_by_sender={
                    0: engine.get_backlog_total(0),
                    1: engine.get_backlog_total(1),
                },
                effective_masks_by_sender=effective,
                uav_hop_observer=(
                    None if collector is None else collector.observe_hop_event
                ),
            )
        )
    outcome = {
        "packet_id": packet["id"], "task_type": "COM",
        "outcome": packet["terminal_outcome"],
        "completed_uav_hop_count": packet["hops"],
        "delivered_to_gs": False,
    }
    if collector is not None:
        collector.end_episode([outcome])
        collector.finalize()
    return {
        "results": copy.deepcopy(results),
        "path": copy.deepcopy(packet["path"]),
        "per_hop": copy.deepcopy(packet["per_hop"]),
        "outcome": outcome,
        "generated": copy.deepcopy(engine.generated_packet_counts),
        "delivered_bits": engine.total_all_useful_bits,
        "energy": np.asarray(engine.energy).tolist(),
        "reward": sum(row["total_routing_reward"] for row in results),
        "cost": sum(
            sum(row["cost_by_sender"].values()) for row in results
        ),
        "simulation_time": env.current_time,
        "rng": copy.deepcopy(streams.state_dict()),
    }


def test_cross_slot_foldback_diagnostics_are_nonintervening(tmp_path):
    assert _cross_slot_loop_nonintervention_run(
        tmp_path / "on", True
    ) == _cross_slot_loop_nonintervention_run(tmp_path / "off", False)


def test_pair_twenty_hop_counts_are_distinct_per_pair_and_delivery(tmp_path):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=5, scenario_id="multi-pair", episode_index=0)
    env = TopologyEnv({(0, 1), (0, 2), (0, 3)})
    _slot(collector, env, hol={0: {"id": 31, "task_type": "COM"}, 1: None, 2: None})
    paths = [
        (2, 1, 0, [{"from": 0, "to": 1, "link_type": "U2U"}, {"from": 1, "to": 0, "link_type": "U2U"}]),
        (4, 2, 0, [{"from": 0, "to": 2, "link_type": "U2U"}, {"from": 2, "to": 0, "link_type": "U2U"}]),
        (6, 1, 0, [{"from": 0, "to": 1, "link_type": "U2U"}, {"from": 1, "to": 0, "link_type": "U2U"}]),
    ]
    for number, sender, receiver, hops in paths:
        _hop(collector, "started", 31, sender, receiver, number, ())
        _hop(collector, "completed", 31, sender, receiver, number, hops)
    collector.end_episode([{
        "packet_id": 31, "task_type": "COM", "outcome": "on_time_delivered",
        "completed_uav_hop_count": 20, "delivered_to_gs": True,
    }])
    collector.finalize()
    rows = list(
        csv.DictReader(
            (tmp_path / "routing_loop_pairs.csv").open(encoding="utf-8-sig")
        )
    )
    involved = [
        row for row in rows
        if row["group"] == "ALL" and row["twenty_hop_terminal_packet_count"] == "1"
    ]
    assert {(row["uav_a"], row["uav_b"]) for row in involved} == {("0", "1"), ("0", "2")}
    assert all(row["twenty_hop_delivered_to_gs"] == "1" for row in involved)
    assert all(row["twenty_hop_hop_limit_not_delivered"] == "0" for row in involved)


def test_group_matrix_is_complete_and_conserves_counts(tmp_path):
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    for episode, (roi, task) in enumerate(((2, "COM"), (3, "FOV"))):
        collector.begin_episode(roi_count=roi, scenario_id=f"s{roi}", episode_index=episode)
        env = TopologyEnv({(0, 1)})
        packet = {"id": roi, "task_type": task, "hop_receiver": None}
        _slot(collector, env, hol={0: packet, 1: None, 2: None})
        hops = [{"from": 1, "to": 0, "link_type": "U2U"}, {"from": 0, "to": 1, "link_type": "U2U"}]
        _hop(collector, "started", roi, 0, 1, 2, (), task_type=task)
        # _hop defaults COM, so exercise task grouping with the event directly.
        event = {
            "phase": "completed", "packet_id": roi, "task_type": task,
            "sender_uav_id": 0, "receiver_id": 1, "link_type": "U2U",
            "service_start_time_seconds": 0.0, "completion_time_seconds": 0.001,
            "cross_slot_transmission": False, "hop_sequence_number": 2,
            "completed_hops": hops,
        }
        collector.observe_hop_event(event)
        collector.end_episode([{
            "packet_id": roi, "task_type": task, "outcome": "expired_dropped",
            "completed_uav_hop_count": 2, "delivered_to_gs": False,
        }])
    collector.finalize()
    rows = {
        row["group"]: row
        for row in json.loads((tmp_path / "routing_loop_summary.json").read_text())["groups"]
    }
    assert set(rows) == {
        "ALL", "VS", "COM",
        "ROI_2_ALL", "ROI_2_VS", "ROI_2_COM",
        "ROI_3_ALL", "ROI_3_VS", "ROI_3_COM",
    }
    assert rows["ROI_2_VS"]["event_count"] == 0
    assert rows["ROI_3_COM"]["event_count"] == 0
    for roi in (2, 3):
        assert rows[f"ROI_{roi}_ALL"]["event_count"] == (
            rows[f"ROI_{roi}_VS"]["event_count"]
            + rows[f"ROI_{roi}_COM"]["event_count"]
        )
    for task in ("ALL", "VS", "COM"):
        assert rows[task]["event_count"] == sum(
            rows[f"ROI_{roi}_{task}"]["event_count"] for roi in (2, 3)
        )
    assert rows["ROI_2_VS"]["with_alternative_actual_legal_ratio"] is None


def test_formal_selection_mask_is_recorded_without_mutating_effective_mask(tmp_path):
    _nonintervention_run(tmp_path, True)
    snapshot = json.loads(
        (tmp_path / "routing_loop_slot_snapshots.jsonl").read_text().splitlines()[0]
    )
    for uid_text, effective_values in snapshot["effective_masks"].items():
        uid = int(uid_text)
        effective = np.asarray(effective_values, dtype=bool)
        expected = forwarding_or_forced_wait_mask(effective, uid, len(effective))
        assert snapshot["selection_masks"][uid_text] == expected.astype(int).tolist()
        if np.delete(effective, uid).any():
            assert snapshot["selection_masks"][uid_text][uid] == 0

    forced_wait_effective = np.asarray([False, True, False], dtype=bool)
    original = forced_wait_effective.copy()
    forced_wait_selection = forwarding_or_forced_wait_mask(
        forced_wait_effective, 1, 3
    )
    assert forced_wait_effective.tolist() == original.tolist()
    assert forced_wait_selection.tolist() == [False, True, False]


class FixedRoutingPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.exploration_rng = np.random.default_rng(123)

    def select_action(self, state, uav_id, mask=None, **kwargs):
        del state, kwargs
        selection = forwarding_or_forced_wait_mask(mask, uav_id, len(mask))
        preferred = int(self.actions.get(int(uav_id), int(uav_id)))
        return preferred if selection[preferred] else int(np.flatnonzero(selection)[0])


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
