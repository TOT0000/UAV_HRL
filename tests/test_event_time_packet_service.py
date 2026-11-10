from types import SimpleNamespace

import numpy as np
import pytest

import Packet_scheduler_v1 as packet_scheduler
from Channel_model import FADING_BLOCK_SECONDS, FADING_BLOCKS_PER_ROUTING_SLOT
from Packet_scheduler_v1 import (
    COM_PACKET_SIZE_BITS,
    PACKET_DEADLINE_POLICY_CONTINUE,
    PACKET_EPS,
    PacketEngine,
    TASK_DEADLINE_SECONDS,
)
from experiment_config import (
    METHOD_REGISTRY,
    PACKET_SERVICE_CONTRACT_VERSION,
    PRODUCTION_TASK_DEADLINE_SECONDS,
    comparison_method_configuration,
    MethodSpec,
)
from routing_transition_ledger import RoutingTransitionLedger
from utils_update_v2 import ReplayBufferDiscrete


class EventCapacityEnvironment:
    """Small deterministic equal-FDMA environment for hand-computable tests."""

    def __init__(self, num_uav, *, bits_per_second_per_hz=0.01, block_scale=None):
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
        self.bits_per_second_per_hz = float(bits_per_second_per_hz)
        self.block_scale = np.ones(FADING_BLOCKS_PER_ROUTING_SLOT)
        if block_scale is not None:
            self.block_scale[:] = np.asarray(block_scale, dtype=float)
        self.current_time = 0.0
        self.source_uavs = set()
        self.multi_tasks = {uid: [] for uid in range(self.num_UAV)}
        self.SR_teams = []
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []
        self.allocation_calls = []

    def is_routing_link_in_range(self, sender, receiver):
        return int(sender) != int(receiver)

    def is_s2u_in_range(self, sr_id, receiver):
        del sr_id, receiver
        return True

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        proposed = {
            int(sender): int(receiver)
            for sender, receiver in dict(proposed_links).items()
            if int(sender) != int(receiver)
        }
        s2u = {int(key): int(value) for key, value in dict(s2u_links or {}).items()}
        links = [(sender, receiver) for sender, receiver in sorted(proposed.items())]
        source_links = [(sr, receiver) for sr, receiver in sorted(s2u.items())]
        count = len(links) + len(source_links)
        bandwidth = self.B_tot / count if count else 0.0
        base_bps = bandwidth * self.bits_per_second_per_hz
        profile = self.block_scale * base_bps / 1e6
        self.active_link_capacity_profiles_mbps = {
            link: profile.copy() for link in links
        }
        self.active_s2u_capacity_profiles_mbps = {
            link: profile.copy() for link in source_links
        }
        capacities = {link: float(np.mean(profile)) for link in links}
        self.active_s2u_capacities = {
            link: float(np.mean(profile)) for link in source_links
        }
        bandwidths = {link: bandwidth for link in links}
        bandwidths.update({("S2U", *link): bandwidth for link in source_links})
        self.active_link_diagnostics = []
        self.allocation_calls.append(
            {
                "time": self.current_time,
                "uav_links": tuple(links),
                "s2u_links": tuple(source_links),
                "bandwidths": dict(bandwidths),
            }
        )
        return capacities, bandwidths


def _actions(*values):
    return {sender: int(receiver) for sender, receiver in enumerate(values)}


def test_more_than_100000_progressing_events_complete_the_slot(monkeypatch):
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(1)
    event_count = 100_005
    event_times = np.linspace(
        0.0,
        engine.step_time,
        event_count + 2,
        dtype=float,
    )[1:-1]
    events = [{"time_seconds": float(value)} for value in event_times]
    consumed = []

    def consume_event(_env, event):
        consumed.append(event["time_seconds"])
        return None

    monkeypatch.setattr(engine, "materialize_generation_event", consume_event)

    result = engine.serve_event_driven_slot(
        env,
        _actions(0),
        0.0,
        generation_events=events,
    )

    assert len(consumed) == event_count
    assert len(result["resource_segments"]) > 100_000
    assert env.current_time == pytest.approx(engine.step_time)


def test_repeated_no_progress_aborts_with_compact_context_and_hol(capsys, monkeypatch):
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
    )
    packet = engine.create_packet(0, "COM", 10.0, generation_time=0.0)
    engine._event_service_stall_guard_limit = 3
    monkeypatch.setattr(engine, "next_packet_deadline", lambda moment: float(moment))

    with pytest.raises(RuntimeError, match="3 consecutive iterations without progress"):
        engine.serve_event_driven_slot(
            env,
            _actions(0),
            0.0,
            stall_diagnostic_context={"episode": 312, "slot_index": 236},
        )

    diagnostic = capsys.readouterr().err
    assert diagnostic.count("[PacketEngine stall]") == 1
    for expected in (
        "slot_start=0.0",
        "slot_end=0.25",
        "current_time=0.0",
        "'deadline': 0.0",
        "consecutive_no_progress=3",
        "episode=312",
        "slot_index=236",
        "'packet_id': 0",
        "'remaining_bits': 10.0",
        "'hops': 0",
    ):
        assert expected in diagnostic
    assert packet["done"] is False


def test_nextafter_only_motion_without_service_state_change_is_stalled(
    capsys, monkeypatch
):
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
    )
    packet = engine.create_packet(
        0,
        "COM",
        PACKET_EPS / 10.0,
        generation_time=59.0,
    )
    engine._event_service_stall_guard_limit = 3
    monkeypatch.setattr(
        engine,
        "record_hop_transmission",
        lambda _packet, _sender, _receiver, _bits: False,
    )

    with pytest.raises(RuntimeError, match="3 consecutive iterations without progress"):
        engine.serve_event_driven_slot(
            env,
            _actions(env.GS_ID),
            59.0,
        )

    diagnostic = capsys.readouterr().err
    assert "current_time=59.00000000000002" in diagnostic
    assert "'completion': 59.00000000000002" in diagnostic
    assert packet["rem_bits"] == pytest.approx(PACKET_EPS / 10.0)
    assert packet["hops"] == 0


def test_nextafter_sized_service_and_same_time_events_are_real_progress(monkeypatch):
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
    )
    packet = engine.create_packet(
        0,
        "COM",
        np.nextafter(0.0, float("inf")),
        generation_time=0.0,
    )
    consumed = []
    monkeypatch.setattr(
        engine,
        "materialize_generation_event",
        lambda _env, event: consumed.append(event["sequence"]) or None,
    )

    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        generation_events=[
            {"time_seconds": 0.0, "sequence": 1},
            {"time_seconds": 0.0, "sequence": 2},
        ],
    )

    assert consumed == [1, 2]
    assert packet["terminal_outcome"] == "on_time_delivered"
    assert packet["hops"] == 1
    assert result["outcomes"][0]["packet_id"] == packet["id"]


def test_authoritative_millisecond_deadlines_reach_every_registered_method():
    assert PRODUCTION_TASK_DEADLINE_SECONDS == {"FOV": 0.015, "COM": 0.010}
    assert TASK_DEADLINE_SECONDS == PRODUCTION_TASK_DEADLINE_SECONDS
    for method_id in METHOD_REGISTRY:
        resolved = comparison_method_configuration(MethodSpec.parse(method_id))
        assert resolved["resolved_fov_deadline_seconds"] == 0.015
        assert resolved["resolved_com_deadline_seconds"] == 0.010
        assert resolved["packet_service_contract_version"] == PACKET_SERVICE_CONTRACT_VERSION
        assert resolved["routing_decision_scope"] == "all_uavs_once_per_250ms_slot"


def test_sr_uav_uav_gs_completes_three_real_hops_in_one_slot():
    env = EventCapacityEnvironment(2)
    sr = SimpleNamespace(id=0, assigned_gt_id=0)
    env.SR_teams = [sr]
    env.multi_tasks[0] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    engine = PacketEngine(
        2, task_deadlines_seconds={"FOV": 0.015, "COM": 0.003}
    )
    packet = engine.create_sr_packet(0, 100.0, generation_time=0.0)

    result = engine.serve_event_driven_slot(
        env,
        _actions(1, env.GS_ID),
        0.0,
        observed_hol_by_sender={0: None, 1: None},
        observed_backlog_bits_by_sender={0: 0.0, 1: 0.0},
        routing_transition_ids_by_sender={0: 10, 1: 11},
        effective_masks_by_sender={0: np.ones(3, bool), 1: np.ones(3, bool)},
    )

    assert packet["terminal_outcome"] == "on_time_delivered"
    assert packet["finish_time"] == pytest.approx(0.003, abs=2 * PACKET_EPS)
    assert packet["e2e_delay_ms"] == pytest.approx(3.0, abs=2e-6)
    assert [hop["link_type"] for hop in packet["per_hop"]] == [
        "S2U",
        "U2U",
        "U2G",
    ]
    assert [row["sender_uav_id"] for row in result["packet_decision_uses"]] == [0, 1]
    assert result["timely_goodput_bits"] == pytest.approx(100.0)
    assert result["global_routing_reward"] == pytest.approx(0.01)
    assert result["reward_by_sender"] == pytest.approx({0: 0.01, 1: 0.01})


def test_dynamic_fdma_reallocates_only_future_time_and_is_order_independent():
    def run(short_sender):
        env = EventCapacityEnvironment(2)
        engine = PacketEngine(2)
        packets = {}
        for sender in range(2):
            bits = 50.0 if sender == short_sender else 150.0
            packets[sender] = engine.create_packet(sender, "COM", bits, 0.0)
        result = engine.serve_event_driven_slot(
            env,
            _actions(env.GS_ID, env.GS_ID),
            0.0,
            observed_hol_by_sender=packets,
            observed_backlog_bits_by_sender={
                sender: packet["size_bits"] for sender, packet in packets.items()
            },
        )
        return packets, result

    first, first_result = run(0)
    second, second_result = run(1)
    assert sorted(packet["finish_time"] for packet in first.values()) == pytest.approx([0.001, 0.002])
    assert sorted(packet["finish_time"] for packet in second.values()) == pytest.approx([0.001, 0.002])
    assert first_result["resource_segments"][0]["active_link_count"] == 2
    assert any(row["active_link_count"] == 1 for row in first_result["resource_segments"])
    assert sum(first_result["transmitted_bits_by_link"].values()) == pytest.approx(200.0)
    assert sum(second_result["transmitted_bits_by_link"].values()) == pytest.approx(200.0)


def test_incremental_event_service_cache_is_exact_and_rechecks_only_dirty_nodes():
    def run(cache_enabled):
        env = EventCapacityEnvironment(3, bits_per_second_per_hz=0.004)
        engine = PacketEngine(
            3, task_deadlines_seconds={"FOV": 0.030, "COM": 0.025}
        )
        engine._event_service_cache_enabled = bool(cache_enabled)
        packets = [
            engine.create_packet(0, "COM", 60.0, 0.0),
            engine.create_packet(0, "COM", 90.0, 0.0),
            engine.create_packet(2, "FOV", 120.0, 0.0),
        ]
        result = engine.serve_event_driven_slot(
            env,
            _actions(1, env.GS_ID, 0),
            0.0,
            observed_hol_by_sender={0: packets[0], 1: None, 2: packets[2]},
            observed_backlog_bits_by_sender={0: 150.0, 1: 0.0, 2: 120.0},
            routing_transition_ids_by_sender={0: 10, 1: 11, 2: 12},
            effective_masks_by_sender={
                sender: np.ones(4, dtype=bool) for sender in range(3)
            },
        )
        return result, packets, engine.event_service_cache_statistics()

    cached_result, cached_packets, cached_stats = run(True)
    reference_result, reference_packets, _ = run(False)

    assert cached_result == reference_result
    assert cached_packets == reference_packets
    assert cached_stats["active_link_incremental_updates"] > 0
    assert cached_stats["active_link_nodes_rechecked"] > 0
    assert cached_stats["active_link_full_scans"] < cached_stats["active_link_queries"]
    # Each segment owns its diagnostic mapping even though the fixed strings
    # and ordering are prepared once per allocation.
    segments = cached_result["resource_segments"]
    if len(segments) >= 2:
        assert segments[0]["bandwidth_hz_by_link"] is not segments[1][
            "bandwidth_hz_by_link"
        ]


def test_completion_is_sub_5ms_and_cross_block_capacity_accumulates():
    scale = np.ones(FADING_BLOCKS_PER_ROUTING_SLOT)
    scale[0] = 0.1  # 10 kbit/s for 5 ms = 50 bits
    scale[1:] = 0.2  # then 20 kbit/s
    env = EventCapacityEnvironment(1, block_scale=scale)
    engine = PacketEngine(1)
    packet = engine.create_packet(0, "COM", 60.0, 0.0)
    engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 60.0},
    )
    assert packet["finish_time"] == pytest.approx(
        FADING_BLOCK_SECONDS + 0.0005, abs=2 * PACKET_EPS
    )


def test_fixed_interval_generation_is_not_visible_before_event_time():
    env = EventCapacityEnvironment(1)
    env.SR_teams = [SimpleNamespace(id=0, assigned_gt_id=0)]
    env.multi_tasks[0] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    engine = PacketEngine(1)
    events = engine.schedule_generation_events(
        env, 0.0, rate_overrides={"FOV": 0.0, "COM": 50.0}
    )
    assert engine.active_count() == 0
    assert [event["time_seconds"] for event in events] == pytest.approx(
        [index * 0.02 for index in range(1, 13)]
    )
    engine.materialize_generation_event(env, events[0])
    assert [pkt["generation_time"] for pkt in engine.get_active_packets()] == [
        pytest.approx(0.02)
    ]


def test_one_action_can_create_independent_cost_samples_for_multiple_packets():
    replay = ReplayBufferDiscrete(4, 3, max_size=32, n_step=1)
    ledger = RoutingTransitionLedger()
    ledger.begin_episode()
    state = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    transition = ledger.create(
        agent_id=0, state=state, action=2, tag_gt=1
    )
    ledger.set_reward(transition, 0.25)
    ledger.finalize_causality({0: state}, {0: None})
    assert ledger.attribute_packet(101, transition)
    assert ledger.attribute_packet(102, transition)
    assert ledger.finalize_packet(101, violated=False)
    assert ledger.finalize_packet(102, violated=True)
    ledger.commit_ready(replay)

    assert replay.size == 1
    assert replay.path_cost_size == 2
    assert replay.reward[0, 0] == pytest.approx(0.25)
    assert sorted(replay.path_cost_value[:2, 0]) == [0.0, 1.0]
    assert replay.path_cost_packet_id[:2].tolist() == [101, 102]


def test_exact_deadline_completion_wins_but_later_packet_expires_immediately():
    def run(size_bits):
        env = EventCapacityEnvironment(1)
        engine = PacketEngine(
            1, task_deadlines_seconds={"FOV": 0.015, "COM": 0.001}
        )
        packet = engine.create_packet(0, "COM", size_bits, 0.0)
        result = engine.serve_event_driven_slot(
            env,
            _actions(env.GS_ID),
            0.0,
            observed_hol_by_sender={0: packet},
            observed_backlog_bits_by_sender={0: size_bits},
        )
        return engine, packet, result

    _, exact, exact_result = run(100.0)
    assert exact["terminal_outcome"] == "on_time_delivered"
    assert exact["finish_time"] == pytest.approx(0.001, abs=2 * PACKET_EPS)
    assert len(exact_result["outcomes"]) == 1

    late_engine, late, late_result = run(100.1)
    assert late["terminal_outcome"] == "expired_dropped"
    assert late["finish_time"] == pytest.approx(0.001, abs=2 * PACKET_EPS)
    assert late["rem_bits"] == pytest.approx(0.1)
    assert late_engine.total_violated == 1
    assert len(late_result["outcomes"]) == 1

    # One representable arithmetic residual is treated consistently as the
    # theoretical boundary, while the materially late case above still fails.
    _, nextafter_boundary, _ = run(np.nextafter(100.0, float("inf")))
    assert nextafter_boundary["terminal_outcome"] == "on_time_delivered"
    assert nextafter_boundary["timely_goodput_counted"] is True


def test_formal_policy_counts_exact_deadline_as_timely_and_all_useful():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 0.015, "COM": 0.001},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)

    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 100.0},
    )

    assert packet["terminal_outcome"] == "on_time_delivered"
    assert result["total_timely_useful_bits"] == pytest.approx(100.0)
    assert result["total_late_useful_bits"] == 0.0
    assert result["total_all_useful_bits"] == pytest.approx(100.0)


def test_formal_policy_allows_late_multihop_delivery_and_counts_one_violation():
    env = EventCapacityEnvironment(2)
    env.SR_teams = [SimpleNamespace(id=0, assigned_gt_id=0)]
    env.multi_tasks[0] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    engine = PacketEngine(
        2,
        task_deadlines_seconds={"FOV": 0.015, "COM": 0.0015},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    packet = engine.create_sr_packet(0, 100.0, generation_time=0.0)

    result = engine.serve_event_driven_slot(
        env,
        _actions(1, env.GS_ID),
        0.0,
        observed_hol_by_sender={0: None, 1: None},
        observed_backlog_bits_by_sender={0: 0.0, 1: 0.0},
    )

    assert packet["terminal_outcome"] == "late_delivered"
    assert [hop["link_type"] for hop in packet["per_hop"]] == [
        "S2U",
        "U2U",
        "U2G",
    ]
    assert result["total_timely_useful_bits"] == 0.0
    assert result["total_late_useful_bits"] == pytest.approx(100.0)
    assert result["total_all_useful_bits"] == pytest.approx(100.0)
    assert engine.total_violated == 1
    assert engine.deadline_drops == 0
    assert engine.packet_metric_summary()["COM"]["violation_packets"] == 1


def test_formal_policy_counts_two_full_wait_slots_before_late_delivery_once():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        step_time=0.25,
        task_deadlines_seconds={"FOV": 0.015, "COM": 0.10},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
        enable_packet_diagnostic_artifacts=True,
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    mask = np.asarray([True, True], dtype=bool)

    for slot_start in (0.0, 0.25):
        result = engine.serve_event_driven_slot(
            env,
            _actions(0),
            slot_start,
            observed_hol_by_sender={0: packet},
            observed_backlog_bits_by_sender={0: packet["rem_bits"]},
            effective_masks_by_sender={0: mask},
        )
        assert result["total_all_useful_bits"] == 0.0

    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.50,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: packet["rem_bits"]},
        effective_masks_by_sender={0: mask},
    )

    assert packet["terminal_outcome"] == "late_delivered"
    assert packet["routing_wait_seconds"] == pytest.approx(0.50)
    assert engine.total_violated == 1
    assert engine.deadline_drops == 0
    assert result["total_timely_useful_bits"] == 0.0
    assert result["total_late_useful_bits"] == pytest.approx(100.0)
    assert engine.total_all_useful_bits == pytest.approx(100.0)


def test_formal_policy_partial_delivery_at_episode_end_adds_no_useful_bits():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 0.015, "COM": 0.001},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    packet = engine.create_packet(0, "COM", 100_000.0, 0.0)

    engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: packet["size_bits"]},
    )
    assert 0.0 < engine.raw_final_hop_bits < packet["size_bits"]
    assert engine.total_all_useful_bits == 0.0

    engine.finalize_episode(0.25)
    assert packet["terminal_outcome"] == "expired_dropped"
    assert engine.total_all_useful_bits == 0.0


def test_formal_policy_late_fov_keeps_capture_coverage_weight():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 0.0005, "COM": 0.010},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    packet = engine.create_packet(0, "FOV", 100.0, 0.0)
    packet["capture_coverage_ratio"] = 0.25

    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 100.0},
    )

    assert packet["terminal_outcome"] == "late_delivered"
    assert result["fov_late_delivered_raw_bits"] == pytest.approx(100.0)
    assert result["fov_late_useful_bits"] == pytest.approx(25.0)
    assert result["total_all_useful_bits"] == pytest.approx(25.0)


def test_formal_policy_keeps_fifo_order_after_head_packet_becomes_late():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 0.015, "COM": 0.001},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    first = engine.create_packet(0, "COM", 20_000.0, 0.0)
    second = engine.create_packet(0, "COM", 1_000.0, 0.0)

    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: first},
        observed_backlog_bits_by_sender={0: 21_000.0},
    )

    assert [row["packet_id"] for row in result["outcomes"]] == [
        first["id"],
        second["id"],
    ]
    assert first["finish_time"] < second["finish_time"]
    assert first["terminal_outcome"] == second["terminal_outcome"] == "late_delivered"


def test_empty_preselected_links_reserve_no_bandwidth():
    env = EventCapacityEnvironment(3)
    engine = PacketEngine(3)
    actions = _actions(1, 2, env.GS_ID)
    result = engine.serve_event_driven_slot(
        env,
        actions,
        0.0,
        observed_hol_by_sender={0: None, 1: None, 2: None},
        observed_backlog_bits_by_sender={0: 0.0, 1: 0.0, 2: 0.0},
    )

    assert result["packet_decision_uses"] == []
    assert result["cost_by_sender"] == {0: 0.0, 1: 0.0, 2: 0.0}
    assert result["global_routing_reward"] == 0.0
    assert result["reward_by_sender"] == {0: 0.0, 1: 0.0, 2: 0.0}
    assert result["active_links"] == []
    assert all(call["uav_links"] == () for call in env.allocation_calls)
    assert all(call["bandwidths"] == {} for call in env.allocation_calls)


def test_evaluation_wait_diagnostics_do_not_require_replay_transition_id():
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1,
        task_deadlines_seconds={"FOV": 1.0, "COM": 1.0},
        enable_packet_diagnostic_artifacts=True,
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    engine.serve_event_driven_slot(
        env,
        _actions(0),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 100.0},
        routing_transition_ids_by_sender={},
        effective_masks_by_sender={0: np.asarray([True, True])},
    )
    assert packet["routing_decision_slot_count"] == 1
    assert packet["routing_wait_slot_count"] == 1
    assert packet["routing_wait_seconds"] == pytest.approx(0.25)


def test_empty_uav_action_serves_mid_slot_arrival_and_multiple_packets(
    monkeypatch,
):
    env = EventCapacityEnvironment(1)
    task = {"task_type": "FOV", "target_obj_id": 7, "target_id": 9}
    env.source_uavs = {0}
    env.multi_tasks[0] = [task]
    geometry = SimpleNamespace(
        sensing_valid_now=True,
        image_quantity=0.01,
        coverage_ratio=0.75,
    )
    monkeypatch.setattr(packet_scheduler, "fov_task_geometry", lambda *args: geometry)
    monkeypatch.setattr(
        packet_scheduler, "fov_physical_packet_size_bits", lambda quantity: 50.0
    )
    engine = PacketEngine(1)
    events = [
        {
            "time_seconds": event_time,
            "kind": "FOV",
            "key": "FOV:0:7:9",
            "uav_id": 0,
            "task": dict(task),
        }
        for event_time in (0.002, 0.004)
    ]
    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        generation_events=events,
        observed_hol_by_sender={0: None},
        observed_backlog_bits_by_sender={0: 0.0},
        routing_transition_ids_by_sender={0: 77},
        effective_masks_by_sender={0: np.ones(2, dtype=bool)},
    )

    assert len(result["generated_packet_ids"]) == 2
    assert [row["packet_id"] for row in result["packet_decision_uses"]] == [0, 1]
    assert {row["transition_id"] for row in result["packet_decision_uses"]} == {77}
    assert [row["violated"] for row in result["outcomes"]] == [False, False]
    assert [row["packet"]["finish_time"] for row in result["outcomes"]] == pytest.approx(
        [0.0025, 0.0045]
    )
    # Reward is the frozen-coverage useful goodput (2 * 50 * .75 bits), even
    # though the UAV was EMPTY at action selection. Cost still needs real use.
    assert result["total_timely_useful_bits"] == pytest.approx(75.0)
    assert result["global_routing_reward"] == pytest.approx(0.0075)
    assert result["reward_by_sender"][0] == pytest.approx(0.0075)


def test_shared_global_reward_is_unclipped_and_not_multiplied_by_uav_count():
    env = EventCapacityEnvironment(16, bits_per_second_per_hz=2.0)
    engine = PacketEngine(16)
    packet = engine.create_packet(3, "COM", 50_000.0, 0.0)
    actions = {uid: uid for uid in range(16)}
    actions[3] = env.GS_ID
    result = engine.serve_event_driven_slot(
        env,
        actions,
        0.0,
        observed_hol_by_sender={uid: packet if uid == 3 else None for uid in range(16)},
        observed_backlog_bits_by_sender={uid: 50_000.0 if uid == 3 else 0.0 for uid in range(16)},
    )

    assert result["total_timely_useful_bits"] == pytest.approx(50_000.0)
    assert result["global_routing_reward"] == pytest.approx(5.0)
    assert set(result["reward_by_sender"]) == set(range(16))
    assert all(
        value == pytest.approx(5.0)
        for value in result["reward_by_sender"].values()
    )
    assert result["shared_reward_row_sum"] == pytest.approx(80.0)
    assert len(result["active_links"]) == 1
    active = result["active_links"][0]
    assert (active["sender_id"], active["receiver_id"], active["link_type"]) == (
        3,
        16,
        "U2G",
    )
    assert active["capacity_bits_per_second"] == pytest.approx(20_000_000.0)
    assert active["transmitted_bits"] == pytest.approx(50_000.0)
    assert active["active_seconds"] == pytest.approx(0.0025)

    env2 = EventCapacityEnvironment(1, bits_per_second_per_hz=2.0)
    engine2 = PacketEngine(1)
    large = engine2.create_packet(0, "COM", 150_000.0, 0.0)
    large_result = engine2.serve_event_driven_slot(
        env2,
        _actions(env2.GS_ID),
        0.0,
        observed_hol_by_sender={0: large},
        observed_backlog_bits_by_sender={0: 150_000.0},
    )
    assert large_result["global_routing_reward"] == pytest.approx(15.0)


def test_reward_is_zero_until_cross_slot_packet_reaches_gs_then_counted_once():
    env = EventCapacityEnvironment(1, bits_per_second_per_hz=0.01)
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 1.0, "COM": 1.0}
    )
    packet = engine.create_packet(0, "COM", 50_000.0, 0.0)
    first = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.0,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 50_000.0},
    )
    second = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.25,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 25_000.0},
    )
    third = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        0.50,
        observed_hol_by_sender={0: None},
        observed_backlog_bits_by_sender={0: 0.0},
    )
    assert first["global_routing_reward"] == 0.0
    assert second["global_routing_reward"] == pytest.approx(5.0)
    assert third["global_routing_reward"] == 0.0
    assert engine.total_timely_useful_bits == pytest.approx(50_000.0)


@pytest.mark.parametrize("start_time", [0.0, 10.0, 50.0])
def test_deadline_boundary_is_translation_invariant(start_time):
    env = EventCapacityEnvironment(1)
    engine = PacketEngine(
        1, task_deadlines_seconds={"FOV": 0.015, "COM": 0.010}
    )
    packet = engine.create_packet(0, "COM", 1000.0, start_time)
    result = engine.serve_event_driven_slot(
        env,
        _actions(env.GS_ID),
        start_time,
        observed_hol_by_sender={0: packet},
        observed_backlog_bits_by_sender={0: 1000.0},
    )
    assert packet["terminal_outcome"] == "on_time_delivered"
    assert result["global_routing_reward"] == pytest.approx(0.1)


def test_fixed_generation_clock_persists_across_slots_and_rate_change_resets_credit():
    env = EventCapacityEnvironment(1)
    sr = SimpleNamespace(id=0, assigned_gt_id=0)
    env.SR_teams = [sr]
    env.multi_tasks[0] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    engine = PacketEngine(1)

    first = engine.schedule_generation_events(
        env, 0.0, rate_overrides={"FOV": 0.0, "COM": 50.0}
    )
    second = engine.schedule_generation_events(
        env, 0.25, rate_overrides={"FOV": 0.0, "COM": 50.0}
    )
    changed = engine.schedule_generation_events(
        env, 0.50, rate_overrides={"FOV": 0.0, "COM": 25.0}
    )

    first_times = [row["time_seconds"] for row in first]
    second_times = [row["time_seconds"] for row in second]
    changed_times = [row["time_seconds"] for row in changed]
    assert first_times == pytest.approx([index * 0.02 for index in range(1, 13)])
    assert second_times == pytest.approx(
        [0.26 + index * 0.02 for index in range(13)]
    )
    assert changed_times == pytest.approx(
        [0.54 + index * 0.04 for index in range(6)]
    )
    assert len(set(first_times + second_times + changed_times)) == 31
    state = engine.checkpoint_state()
    assert state["generation_clocks"]["SR_0_COM"] == {
        "kind": "COM",
        "rate_packets_per_second": 25.0,
        "next_time_seconds": pytest.approx(0.78),
    }

    paused = engine.schedule_generation_events(
        env, 0.75, rate_overrides={"FOV": 0.0, "COM": 0.0}
    )
    assert paused == []
    assert "SR_0_COM" not in engine.generation_clocks
    resumed = engine.schedule_generation_events(
        env, 1.0, rate_overrides={"FOV": 0.0, "COM": 25.0}
    )
    assert resumed[0]["time_seconds"] == pytest.approx(1.04)

    sr.assigned_gt_id = None
    assert engine.schedule_generation_events(
        env, 1.25, rate_overrides={"FOV": 0.0, "COM": 25.0}
    ) == []
    assert "SR_0_COM" not in engine.generation_clocks


def test_generation_activation_load_factor_cutoff_and_actual_fov_geometry(monkeypatch):
    env = EventCapacityEnvironment(1)
    sr = SimpleNamespace(id=0, assigned_gt_id=None)
    env.SR_teams = [sr]
    env.multi_tasks[0] = []
    inactive = PacketEngine(1, injection_cutoff_seconds=0.055)
    assert inactive.schedule_generation_events(
        env, 0.0, rate_overrides={"FOV": 0.0, "COM": 50.0}
    ) == []
    assert "SR_0_COM" not in inactive.generation_clocks

    sr.assigned_gt_id = 0
    env.multi_tasks[0] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    env.load_factor = 2.0
    events = inactive.schedule_generation_events(
        env, 0.0, rate_overrides={"FOV": 0.0, "COM": 50.0}
    )
    assert [row["time_seconds"] for row in events] == pytest.approx(
        [0.01, 0.02, 0.03, 0.04, 0.05]
    )
    assert all(row["time_seconds"] <= 0.055 for row in events)

    fov_env = EventCapacityEnvironment(1)
    task = {"task_type": "FOV", "target_obj_id": 4, "target_id": 8}
    fov_env.source_uavs = {0}
    fov_env.multi_tasks[0] = [task]

    def geometry_at_event(environment, *_args):
        return SimpleNamespace(
            sensing_valid_now=True,
            image_quantity=0.2 if environment.current_time < 0.1 else 0.8,
            coverage_ratio=0.4 if environment.current_time < 0.1 else 0.9,
        )

    monkeypatch.setattr(packet_scheduler, "fov_task_geometry", geometry_at_event)
    monkeypatch.setattr(
        packet_scheduler,
        "fov_physical_packet_size_bits",
        lambda quantity: 1000.0 * quantity,
    )
    fov_engine = PacketEngine(1)
    fov_events = fov_engine.schedule_generation_events(
        fov_env, 0.0, rate_overrides={"FOV": 5.0, "COM": 0.0}
    )
    assert [row["time_seconds"] for row in fov_events] == [pytest.approx(0.2)]
    fov_env.current_time = 0.2
    packet = fov_engine.materialize_generation_event(fov_env, fov_events[0])
    assert packet["generation_time"] == pytest.approx(0.2)
    assert packet["size_bits"] == pytest.approx(800.0)
    assert packet["capture_coverage_ratio"] == pytest.approx(0.9)
    assert COM_PACKET_SIZE_BITS > 0.0
