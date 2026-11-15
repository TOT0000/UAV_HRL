import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from DDQN import forwarding_or_forced_wait_mask
from HRL_task_aware import _select_routing_actions, smoke_training_config, train
from Packet_scheduler_v1 import PacketEngine
from mutual_next_hop_guard import evaluation_routing_strategy_info
from paper_evaluation import _evaluation_config_fingerprint
from routing_guard_comparison import build_routing_guard_comparison
from routing_loop_diagnostics import RoutingLoopDiagnosticsCollector


def _mask(action_dim, *legal):
    result = np.zeros(action_dim, dtype=bool)
    result[list(legal)] = True
    return result


class PreferredLegalPolicy:
    def __init__(self, preferences):
        self.preferences = preferences
        self.calls = []
        self.exploration_rng = np.random.default_rng(731)

    def select_action(self, state, uav_id, mask=None, **kwargs):
        del state, kwargs
        formal = forwarding_or_forced_wait_mask(mask, uav_id, len(mask))
        self.calls.append((int(uav_id), np.asarray(mask, dtype=bool).copy()))
        for action in self.preferences.get(int(uav_id), ()):
            if formal[int(action)]:
                return int(action)
        return int(np.flatnonzero(formal)[0])


def _select(masks, preferences, slot=0):
    policy = PreferredLegalPolicy(preferences)
    original = {uid: value.copy() for uid, value in masks.items()}
    actions, context = _select_routing_actions(
        policy,
        {uid: np.zeros(143, dtype=np.float32) for uid in masks},
        masks,
        epsilon=0.0,
        mutual_next_hop_guard=True,
        slot_index=slot,
        return_selection_context=True,
    )
    assert all(np.array_equal(masks[uid], original[uid]) for uid in masks)
    return actions, context, policy


def test_mrv_selects_constrained_uav_first_and_excludes_mutual_candidate():
    masks = {
        0: _mask(4, 0, 1, 2),
        1: _mask(4, 0, 1),
        2: _mask(4, 0, 1, 2, 3),
    }
    actions, context, policy = _select(
        masks, {0: (1, 2), 1: (0,), 2: (3,)}, slot=0
    )
    assert context["selection_order"][0] == 1
    assert actions[1] == 0
    assert actions[0] == 2
    assert context["per_uav"][0]["excluded_candidates_by_selected_uav"] == [1]
    assert not bool(policy.calls[1][1][1])


def test_remaining_candidate_counts_are_recomputed_before_each_selection():
    masks = {
        0: _mask(5, 0, 1, 2, 3),
        1: _mask(5, 0, 1),
        2: _mask(5, 0, 2, 3),
        3: _mask(5, 0, 1, 2, 3, 4),
    }
    _, context, _ = _select(
        masks, {0: (2,), 1: (0,), 2: (3,), 3: (4,)}, slot=0
    )
    # Initially UAV 2 (two candidates) is ahead of UAV 0 (three).  UAV 1's
    # action removes candidate 1 from UAV 0, producing a tie that rotation
    # resolves in favour of UAV 0.
    assert context["selection_order"][:3] == [1, 0, 2]
    assert context["per_uav"][0]["filtered_non_wait_candidate_count"] == 2


def test_mask_update_checks_every_previously_selected_uav():
    masks = {
        0: _mask(5, 0, 3),
        1: _mask(5, 1, 3),
        2: _mask(5, 0, 2, 4),
        3: _mask(5, 0, 1, 2, 3),
    }
    actions, context, _ = _select(
        masks, {0: (3,), 1: (3,), 2: (4,), 3: (2,)}, slot=0
    )
    assert context["selection_order"][:3] == [0, 1, 3]
    assert context["per_uav"][3]["excluded_candidates_by_selected_uav"] == [0, 1]
    assert context["per_uav"][3]["filtered_non_wait_candidate_count"] == 1
    assert actions[3] == 2


def test_equal_counts_rotate_by_slot_without_consuming_rng():
    masks = {
        0: _mask(4, 0, 3),
        1: _mask(4, 1, 3),
        2: _mask(4, 2, 3),
    }
    policy = PreferredLegalPolicy({0: (3,), 1: (3,), 2: (3,)})
    before = copy.deepcopy(policy.exploration_rng.bit_generator.state)
    _, context = _select_routing_actions(
        policy,
        {uid: np.zeros(143) for uid in masks},
        masks,
        epsilon=0.0,
        mutual_next_hop_guard=True,
        slot_index=1,
        return_selection_context=True,
    )
    assert context["selection_order"] == [1, 2, 0]
    assert policy.exploration_rng.bit_generator.state == before
    assert all(
        context["per_uav"][uid]["original_non_wait_candidate_count"] == 1
        for uid in masks
    )  # self is excluded and the legal GS is included


def test_forced_wait_distinguishes_original_and_guard_induced_zero_candidates():
    masks = {
        0: _mask(4, 0, 1),
        1: _mask(4, 0, 1),
        2: _mask(4, 2),
    }
    actions, context, policy = _select(masks, {0: (1,), 1: (0,)}, slot=0)
    assert context["selection_order"][0] == 2
    assert actions[2] == 2
    assert context["per_uav"][2]["originally_no_forwarding_candidate"] is True
    assert context["per_uav"][2]["guard_induced_forced_wait"] is False
    assert actions[1] == 1
    assert context["per_uav"][1]["guard_induced_forced_wait"] is True
    assert [uid for uid, _ in policy.calls] == [0]


def test_three_node_cycle_is_allowed_and_each_uav_is_selected_once():
    masks = {
        0: _mask(4, 0, 1),
        1: _mask(4, 1, 2),
        2: _mask(4, 0, 2),
    }
    actions, context, policy = _select(
        masks, {0: (1,), 1: (2,), 2: (0,)}, slot=0
    )
    assert actions == {0: 1, 1: 2, 2: 0}
    assert context["selection_order"] == [0, 1, 2]
    assert [uid for uid, _ in policy.calls] == [0, 1, 2]


class PartialEnv:
    def __init__(self):
        self.num_UAV = 2
        self.GS_ID = 2
        self.GS_pos = (100.0, 0.0, 0.0)
        self.B_tot = 10e6
        self.bits_per_second_per_hz = 0.00002
        self.current_time = 0.0
        self.source_uavs = set()
        self.multi_tasks = {0: [], 1: []}
        self.SR_teams = []
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []
        self.uav_dict = {
            uid: SimpleNamespace(get_position=lambda uid=uid: (uid, 0.0, 100.0))
            for uid in range(2)
        }

    def is_routing_link_in_range(self, sender, receiver):
        return int(sender) != int(receiver)

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        links = [
            (int(left), int(right))
            for left, right in proposed_links.items()
            if int(left) != int(right)
        ]
        bandwidth = self.B_tot / max(len(links), 1)
        capacity = bandwidth * self.bits_per_second_per_hz / 1e6
        profile = np.full(50, capacity)
        self.active_link_capacity_profiles_mbps = {
            link: profile.copy() for link in links
        }
        return ({link: capacity for link in links}, {link: bandwidth for link in links})


def test_packet_engine_partial_hop_waits_without_losing_lock_or_progress_then_resumes():
    env = PartialEnv()
    engine = PacketEngine(2, task_deadlines_seconds={"FOV": 2.0, "COM": 2.0})
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    all_legal = {0: _mask(3, 0, 1), 1: _mask(3, 0, 1)}
    engine.serve_event_driven_slot(
        env, {0: 1, 1: 1}, 0.0,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: 100.0, 1: 0.0},
        effective_masks_by_sender=all_legal,
    )
    assert packet["hop_receiver"] == 1
    progress = float(packet["hop_bits_sent"])
    remaining = float(packet["rem_bits"])
    assert progress > 0.0

    locked_masks = {0: _mask(3, 0, 1), 1: _mask(3, 0, 1)}
    actions, context, _ = _select(locked_masks, {0: (1,), 1: (0,)}, slot=1)
    assert actions == {1: 0, 0: 0}
    assert context["per_uav"][0]["guard_induced_forced_wait"] is True
    engine.serve_event_driven_slot(
        env, actions, engine.step_time,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: remaining, 1: 0.0},
        effective_masks_by_sender=context["applied_effective_masks"],
    )
    assert packet["hop_receiver"] == 1
    assert packet["hop_bits_sent"] == progress
    assert packet["rem_bits"] == remaining

    actions, context, _ = _select(locked_masks, {0: (1,), 1: (0,)}, slot=2)
    assert actions[0] == 1
    engine.serve_event_driven_slot(
        env, actions, 2 * engine.step_time,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: remaining, 1: 0.0},
        effective_masks_by_sender=context["applied_effective_masks"],
    )
    assert packet["hop_bits_sent"] > progress or packet["hops"] > 0


def _guarded_packet_run(output, diagnostics_enabled):
    env = PartialEnv()
    engine = PacketEngine(2, task_deadlines_seconds={"FOV": 2.0, "COM": 2.0})
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    policy = PreferredLegalPolicy({0: (1,), 1: (0,)})
    collector = (
        RoutingLoopDiagnosticsCollector(output) if diagnostics_enabled else None
    )
    if collector is not None:
        collector.begin_episode(roi_count=2, scenario_id="guard-loop", episode_index=0)
    actions_seen = []
    rewards = []
    for slot, rate in enumerate((0.00002, 0.01, 0.01, 0.01)):
        env.bits_per_second_per_hz = rate
        masks = {0: _mask(3, 0, 1), 1: _mask(3, 0, 1)}
        actions, context = _select_routing_actions(
            policy,
            {0: np.zeros(143), 1: np.zeros(143)},
            masks,
            epsilon=0.0,
            mutual_next_hop_guard=True,
            slot_index=slot,
            return_selection_context=True,
        )
        hol = {0: engine.get_hol_packet(0), 1: engine.get_hol_packet(1)}
        if collector is not None:
            collector.observe_slot(
                env=env, packet_engine=engine,
                current_time=slot * engine.step_time, slot_index=slot,
                physical_masks=masks, effective_masks=masks,
                applied_effective_masks=context["applied_effective_masks"],
                selection_masks=context["selection_masks"],
                selected_actions=actions,
                start_of_slot_hol_by_sender=hol,
                selection_context=context,
            )
        result = engine.serve_event_driven_slot(
            env, actions, slot * engine.step_time,
            observed_hol_by_sender=hol,
            observed_backlog_bits_by_sender={
                0: engine.get_backlog_total(0), 1: engine.get_backlog_total(1)
            },
            effective_masks_by_sender=context["applied_effective_masks"],
            uav_hop_observer=(
                collector.observe_hop_event if collector is not None else None
            ),
            held_action_packet_observer=(
                collector.observe_held_action_packet
                if collector is not None
                else None
            ),
        )
        actions_seen.append(actions)
        rewards.append(float(result["total_routing_reward"]))
    outcome = {
        "packet_id": packet["id"], "task_type": "COM",
        "outcome": "expired_dropped", "completed_uav_hop_count": packet["hops"],
        "delivered_to_gs": False,
    }
    if collector is not None:
        collector.end_episode([outcome])
        collector.finalize()
    return {
        "actions": actions_seen,
        "path": copy.deepcopy(packet["path"]),
        "hops": int(packet["hops"]),
        "receiver_lock": packet["hop_receiver"],
        "remaining_bits": float(packet["rem_bits"]),
        "rewards": rewards,
        "generated": copy.deepcopy(engine.generated_packet_counts),
        "delivered_bits": float(engine.total_all_useful_bits),
        "energy": np.asarray(engine.energy).tolist(),
        "simulation_time": 4 * engine.step_time,
        "agent_rng": copy.deepcopy(policy.exploration_rng.bit_generator.state),
    }


def test_guarded_foldback_and_cross_slot_diagnostics_are_nonintervening(tmp_path):
    enabled = _guarded_packet_run(tmp_path / "on", True)
    disabled = _guarded_packet_run(tmp_path / "off", False)
    assert enabled == disabled
    assert enabled["hops"] >= 2
    events = [
        json.loads(line)
        for line in (tmp_path / "on" / "routing_loop_events.jsonl").read_text().splitlines()
    ]
    assert events


def test_off_path_preserves_fixed_order_and_masks():
    masks = {0: _mask(3, 0, 1), 1: _mask(3, 0, 1)}
    policy = PreferredLegalPolicy({0: (1,), 1: (0,)})
    actions, context = _select_routing_actions(
        policy, {0: np.zeros(143), 1: np.zeros(143)}, masks, epsilon=0.0,
        mutual_next_hop_guard=False, slot_index=1,
        return_selection_context=True,
    )
    assert actions == {0: 1, 1: 0}
    assert context["selection_order"] == [0, 1]
    assert all(np.array_equal(context["applied_effective_masks"][uid], masks[uid]) for uid in masks)


def test_training_rejects_evaluation_only_guard():
    with pytest.raises(ValueError, match="evaluation-only"):
        train(config=smoke_training_config(), mutual_next_hop_guard=True)


def test_off_on_comparison_accepts_changed_results_and_uses_ratio_of_sums(tmp_path):
    source = tmp_path / "off"
    guarded = tmp_path / "on"
    for root, useful, violation in ((source, 10.0, False), (guarded, 8.0, True)):
        point = root / "roi_2"
        point.mkdir(parents=True)
        episode = {
            "scenario_id": "s", "total_mobility_energy_j": 2.0,
            "total_all_useful_mbits": useful,
            "fov_timely_useful_bits": useful * 1e6,
            "fov_late_useful_bits": 0.0,
            "com_timely_delivered_bits": 0.0,
            "com_late_delivered_bits": 0.0,
        }
        (point / "per_episode.jsonl").write_text(json.dumps(episode) + "\n")
        packet = {
            "packet_id": 1, "task_type": "FOV", "qos_eligible": True,
            "outcome": "expired_dropped" if violation else "on_time_delivered",
            "delivered_to_gs": not violation, "terminal_node_type": "UAV",
            "completed_uav_hop_count": 20 if violation else 1,
            "per_hop": [], "routing_only_wait_no_available_link_slot_count": 1,
        }
        (point / "packet_outcomes.jsonl").write_text(
            json.dumps({"scenario_id": "s", "packet_outcomes": [packet]}) + "\n"
        )
    (guarded / "routing_loop_summary.json").write_text(json.dumps({
        "schema_version": "routing-loop-diagnostics-v4",
        "groups": [
            {
                "roi_count": roi, "task": task,
                "held_action_packet_guard_induced_forced_wait_count": (
                    1 if task in {"ALL", "VS"} else 0
                ),
                "slot_start_hol_guard_induced_forced_wait_decision_count": 0,
            }
            for roi in (None, 2) for task in ("ALL", "VS", "COM")
        ],
        "uav_preselection_decisions": [
            {
                "roi_count": roi,
                "uav_preselection_guard_induced_forced_wait_decision_count": 1,
            }
            for roi in (None, 2)
        ],
    }))
    (guarded / "paper_evaluation_metadata.json").write_text(json.dumps({
        "evaluation_routing_strategy": evaluation_routing_strategy_info(True)
    }))
    outputs = build_routing_guard_comparison(
        source_evaluation_dir=source, guarded_evaluation_dir=guarded,
        roi_counts=(2,), selected_scenario_ids_by_roi={"2": ["s"]},
    )
    payload = json.loads(open(outputs["json"], encoding="utf-8").read())
    row = next(item for item in payload["groups"] if item["group"] == "ALL")
    assert row["off_energy_efficiency_mbit_per_j"] == 5.0
    assert row["on_energy_efficiency_mbit_per_j"] == 4.0
    assert row["delta_violation_packet_count"] == 1
    assert row["on_guard_induced_forced_wait_count"] == 1
    assert row[
        "on_slot_start_hol_guard_induced_forced_wait_decision_count"
    ] == 0
    assert row[
        "on_uav_preselection_guard_induced_forced_wait_decision_count"
    ] == 1
    assert row["on_guard_induced_forced_wait_count"] <= row[
        "on_forced_wait_no_forwarding_candidate_count"
    ]


def test_packet_engine_counts_mid_slot_packet_using_guard_held_wait(tmp_path):
    env = PartialEnv()
    env.bits_per_second_per_hz = 0.01
    engine = PacketEngine(
        2,
        task_deadlines_seconds={"FOV": 2.0, "COM": 2.0},
        enable_packet_diagnostic_artifacts=True,
    )
    packet = engine.create_packet(0, "COM", 100.0, 0.0)
    masks = {0: _mask(3, 0, 1), 1: _mask(3, 0, 1)}
    policy = PreferredLegalPolicy({0: (1,), 1: (0,)})
    actions, context = _select_routing_actions(
        policy,
        {0: np.zeros(143), 1: np.zeros(143)},
        masks,
        epsilon=0.0,
        mutual_next_hop_guard=True,
        slot_index=0,
        return_selection_context=True,
    )
    assert actions == {0: 1, 1: 1}
    assert context["per_uav"][1]["guard_induced_forced_wait"] is True
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=2, scenario_id="mid-slot", episode_index=0)
    collector.observe_slot(
        env=env,
        packet_engine=engine,
        current_time=0.0,
        slot_index=0,
        physical_masks=masks,
        effective_masks=masks,
        applied_effective_masks=context["applied_effective_masks"],
        selection_masks=context["selection_masks"],
        selected_actions=actions,
        start_of_slot_hol_by_sender={0: packet, 1: None},
        selection_context=context,
    )
    engine.serve_event_driven_slot(
        env,
        actions,
        0.0,
        observed_hol_by_sender={0: packet, 1: None},
        observed_backlog_bits_by_sender={0: 100.0, 1: 0.0},
        effective_masks_by_sender=context["applied_effective_masks"],
        held_action_packet_observer=collector.observe_held_action_packet,
    )
    # Repeated observation of the same packet/sender/slot is idempotent.
    collector.observe_held_action_packet(
        {
            "packet_id": packet["id"],
            "task_type": "COM",
            "sender_uav_id": 1,
            "selected_action": 1,
        }
    )
    collector.end_episode([])
    collector.finalize()
    summary = json.loads((tmp_path / "routing_loop_summary.json").read_text())
    com = next(
        row for row in summary["groups"]
        if row["roi_count"] is None and row["task"] == "COM"
    )
    all_preselection = next(
        row for row in summary["uav_preselection_decisions"]
        if row["roi_count"] is None
    )
    assert all_preselection[
        "uav_preselection_guard_induced_forced_wait_decision_count"
    ] == 1
    assert com["slot_start_hol_guard_induced_forced_wait_decision_count"] == 0
    assert com["held_action_packet_forced_wait_count"] == 1
    assert com["held_action_packet_guard_induced_forced_wait_count"] == 1
    assert packet["routing_only_wait_no_available_link_slot_count"] == 1


def test_evaluation_config_fingerprint_distinguishes_guard_and_preserves_off():
    point = {
        "point_id": "roi_2",
        "environment_width_m": None,
        "environment_height_m": None,
        "evaluation_episode_horizon_s": 60,
        "overrides": {},
    }
    legacy_payload = {
        "method_id": "ours",
        "point_id": "roi_2",
        "episodes": 10,
        "evaluation_seed": 7,
        "environment_width_m": None,
        "environment_height_m": None,
        "evaluation_episode_horizon_s": 60,
        "overrides": {},
    }
    legacy = hashlib.sha256(
        json.dumps(
            legacy_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    manifest = SimpleNamespace(environment_size_m=None)
    off = _evaluation_config_fingerprint(
        "ours", point, 10, 7, manifest,
        evaluation_routing_strategy=evaluation_routing_strategy_info(False),
    )
    on_a = _evaluation_config_fingerprint(
        "ours", point, 10, 7, manifest,
        evaluation_routing_strategy=evaluation_routing_strategy_info(True),
    )
    on_b = _evaluation_config_fingerprint(
        "ours", point, 10, 7, manifest,
        evaluation_routing_strategy=evaluation_routing_strategy_info(True),
    )
    assert off == legacy
    assert on_a == on_b
    assert on_a != off


def test_guarded_comparison_rejects_missing_v4_packet_wait_diagnostics(tmp_path):
    source = tmp_path / "off"
    guarded = tmp_path / "on"
    for root in (source, guarded):
        point = root / "roi_2"
        point.mkdir(parents=True)
        (point / "per_episode.jsonl").write_text(json.dumps({
            "scenario_id": "s", "total_mobility_energy_j": 1.0,
            "total_all_useful_mbits": 0.0, "fov_timely_useful_bits": 0.0,
            "fov_late_useful_bits": 0.0, "com_timely_delivered_bits": 0.0,
            "com_late_delivered_bits": 0.0,
        }) + "\n")
        (point / "packet_outcomes.jsonl").write_text(
            json.dumps({"scenario_id": "s", "packet_outcomes": []}) + "\n"
        )
    (guarded / "routing_loop_summary.json").write_text(json.dumps({
        "schema_version": "routing-loop-diagnostics-v3", "groups": []
    }))
    (guarded / "paper_evaluation_metadata.json").write_text(json.dumps({
        "evaluation_routing_strategy": evaluation_routing_strategy_info(True)
    }))
    with pytest.raises(ValueError, match="diagnostics v4"):
        build_routing_guard_comparison(
            source_evaluation_dir=source,
            guarded_evaluation_dir=guarded,
            roi_counts=(2,),
            selected_scenario_ids_by_roi={"2": ["s"]},
        )


def test_held_wait_dedup_separates_sender_slot_and_original_no_candidate(tmp_path):
    env = PartialEnv()
    collector = RoutingLoopDiagnosticsCollector(tmp_path)
    collector.begin_episode(roi_count=3, scenario_id="partial-vs", episode_index=0)
    packet = {"id": 9, "task_type": "FOV", "hop_receiver": 1}
    masks = {0: _mask(3, 0), 1: _mask(3, 1)}
    for slot in (0, 1):
        actions, context = _select_routing_actions(
            PreferredLegalPolicy({}),
            {0: np.zeros(143), 1: np.zeros(143)},
            masks,
            epsilon=0.0,
            mutual_next_hop_guard=True,
            slot_index=slot,
            return_selection_context=True,
        )
        collector.observe_slot(
            env=env,
            packet_engine=None,
            current_time=slot * 0.25,
            slot_index=slot,
            physical_masks=masks,
            effective_masks=masks,
            applied_effective_masks=context["applied_effective_masks"],
            selection_masks=context["selection_masks"],
            selected_actions=actions,
            start_of_slot_hol_by_sender={0: packet, 1: None},
            selection_context=context,
        )
        event = {
            "packet_id": 9, "task_type": "FOV", "sender_uav_id": 0,
            "selected_action": 0, "hop_receiver_lock": 1,
        }
        collector.observe_held_action_packet(event)
        collector.observe_held_action_packet(event)
        if slot == 1:
            collector.observe_held_action_packet({
                **event, "sender_uav_id": 1, "selected_action": 1,
            })
    collector.end_episode([])
    collector.finalize()
    summary = json.loads((tmp_path / "routing_loop_summary.json").read_text())
    vs = next(
        row for row in summary["groups"]
        if row["roi_count"] is None and row["task"] == "VS"
    )
    assert vs["held_action_packet_use_count"] == 3
    assert vs["held_action_packet_forced_wait_count"] == 3
    assert vs["held_action_packet_guard_induced_forced_wait_count"] == 0
