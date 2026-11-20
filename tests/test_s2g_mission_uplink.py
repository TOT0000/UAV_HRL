import math
from types import SimpleNamespace

import numpy as np
import pytest

from Packet_scheduler_v1 import (
    EPISODE_INJECTION_CUTOFF_SECONDS,
    PACKET_DEADLINE_POLICY_CONTINUE,
    PacketEngine,
)
from Channel_model import s2g_ground_path_loss_db
from Simulator import Simulator
from Task_assignment import Task, UAVAssigner
from centralized_movement import (
    MOVEMENT_STATE_DIM,
    get_global_movement_state,
    movement_state_feature_schema,
    movement_constraint_penalties,
)
from observation_strategy import apply_observation_strategy
from experiment_config import PRODUCTION_EPISODE_HORIZON_SECONDS
from experiment_defaults import DEFAULT_SR_SPEED_MPS
from scenario_manifest import generate_manifest


def _environment(num_gt=2):
    entry = generate_manifest("test", 7701 + num_gt, 1, num_gt=num_gt).episodes[0]
    env = Simulator(16)
    env.apply_scenario_entry(entry)
    env.prepare_channel_routing_slot(0)
    return env


def _assign_sr_mission(env, sr_id=0, gt_id=0, assignment_time=0.0):
    sr = env.SR_teams[sr_id]
    sr.assigned_gt_id = int(gt_id)
    env.sr_movement_events[sr_id] = {
        "assigned_roi_id": int(gt_id),
        "assignment_time_seconds": float(assignment_time),
        "entered_roi_time_seconds": None,
        "arrived_center_time_seconds": None,
    }
    return sr


@pytest.mark.parametrize("num_gt", (2, 7, 8))
def test_sr_quarter_arc_layout_and_new_time_defaults(num_gt):
    manifest = generate_manifest("test", 441, 1, num_gt=num_gt)
    entry = manifest.episodes[0]
    positions = [tuple(row["position"]) for row in entry["sr_teams"]]

    assert len(positions) == num_gt
    assert len(set(positions)) == num_gt
    for index, (x, y, z) in enumerate(positions):
        theta = (index + 1) * math.pi / (2.0 * (num_gt + 1))
        assert (x, y, z) == pytest.approx(
            (80.0 * math.cos(theta), 80.0 * math.sin(theta), 0.0)
        )
        assert math.dist((x, y, z), (0.0, 0.0, 0.0)) == pytest.approx(80.0)
        assert 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0
        assert entry["sr_teams"][index]["movement_primitive"]["speed_mps"] == 1.5

    assert DEFAULT_SR_SPEED_MPS == 1.5
    assert PRODUCTION_EPISODE_HORIZON_SECONDS == 500.0
    assert EPISODE_INJECTION_CUTOFF_SECONDS == pytest.approx(499.985)


def test_mission_clock_generates_without_uav_or_connectivity_and_does_not_reset():
    sr = SimpleNamespace(id=0, assigned_gt_id=3, arrived=False)
    env = SimpleNamespace(
        source_uavs=set(),
        multi_tasks={},
        SR_teams=[sr],
        sr_movement_events={
            0: {"assigned_roi_id": 3, "assignment_time_seconds": 0.1}
        },
        load_factor=1.0,
    )
    engine = PacketEngine(1)

    first = engine.schedule_generation_events(env, 0.0)
    assert [row["time_seconds"] for row in first] == pytest.approx(
        [0.12 + 0.02 * index for index in range(7)]
    )
    # Connectivity/UAV assignment changes do not recreate the mission clock.
    env.multi_tasks = {
        0: [{"task_type": "COM", "target_obj_id": 0, "target_id": 3}]
    }
    second = engine.schedule_generation_events(env, 0.25)
    assert second[0]["time_seconds"] == pytest.approx(0.26)
    assert engine.com_sessions[0]["activation_time_seconds"] == pytest.approx(0.1)


def test_s2g_boundary_and_real_packet_service_without_com_uav_assignment():
    env = _environment()
    sr = _assign_sr_mission(env)
    sr.x, sr.y, sr.z = 100.0, 0.0, 0.0
    assert env.is_s2g_in_range(0)
    sr.x = 100.000001
    assert not env.is_s2g_in_range(0)
    sr.x = 80.0

    engine = PacketEngine(16)
    packet = engine.create_sr_packet(0, 256.0, 0.0)
    result = engine.serve_event_driven_slot(
        env,
        {uav_id: uav_id for uav_id in range(16)},
        0.0,
    )

    assert packet["done"]
    assert packet["delivery_path"] == "S2G"
    assert not packet["routing_eligible"]
    assert result["total_timely_useful_bits"] == pytest.approx(256.0)
    assert result["global_routing_reward"] > 0.0
    assert all(
        reward == pytest.approx(result["global_routing_reward"])
        for reward in result["reward_by_sender"].values()
    )
    assert result["progress_routing_reward"] == pytest.approx(0.0)
    assert engine.s2g_completed_packets == 1
    assert engine.s2u_completed_packets == 0
    outcome = engine.packet_outcomes[0]
    assert outcome["outcome"] == "on_time_delivered"
    assert outcome["delivery_path"] == "S2G"


def test_s2g_ground_path_loss_reference_points_and_other_ranges_remain_400m():
    gs = (0.0, 0.0, 0.0)
    assert s2g_ground_path_loss_db((0.1, 0.0, 0.0), gs) == pytest.approx(38.46)
    assert s2g_ground_path_loss_db((1.0, 0.0, 0.0), gs) == pytest.approx(38.46)
    assert s2g_ground_path_loss_db((10.0, 0.0, 0.0), gs) == pytest.approx(78.46)
    assert s2g_ground_path_loss_db((100.0, 0.0, 0.0), gs) == pytest.approx(118.46)
    env = _environment()
    sr = _assign_sr_mission(env)
    sr.x, sr.y, sr.z = 100.0, 0.0, 0.0
    assert env.is_s2g_in_range(0)
    sr.x = 100.000001
    assert not env.is_s2g_in_range(0)
    uav = env.uav_dict[1]
    sr.x, sr.y, sr.z = 0.0, 0.0, 0.0
    uav.x_u, uav.y_u, uav.z_u = 400.0, 0.0, 0.0
    assert env.is_s2u_in_range(0, 1)
    uav.x_u = 400.000001
    assert not env.is_s2u_in_range(0, 1)


def test_new_roi_creates_both_fov_and_com_even_while_sr_is_direct():
    env = _environment()
    env.is_visible = lambda _uid, gt, **_kwargs: int(gt.id) == 0
    env.update_visited_grid(0, observation_time_seconds=0.0)
    assert env.is_s2g_in_range(0)
    assert [task.task_type for task in env.task_list] == ["FOV", "COM"]
    assert env.need_reassign
    env.assign_tasks()
    assert any(
        task.get("task_type") == "COM" and int(task["target_obj_id"]) == 0
        for tasks in env.multi_tasks.values()
        for task in tasks
    )


def test_late_s2g_delivery_counts_complete_useful_bits_but_not_timely_reward():
    env = _environment()
    sr = _assign_sr_mission(env)
    sr.x, sr.y, sr.z = 80.0, 0.0, 0.0
    engine = PacketEngine(
        16,
        task_deadlines_seconds={"FOV": 0.015, "COM": 1e-9},
        packet_deadline_policy=PACKET_DEADLINE_POLICY_CONTINUE,
    )
    packet = engine.create_sr_packet(0, 256.0, 0.0)

    result = engine.serve_event_driven_slot(
        env,
        {uav_id: uav_id for uav_id in range(16)},
        0.0,
    )

    assert packet["done"]
    assert engine.packet_outcomes[0]["outcome"] == "late_delivered"
    assert result["total_timely_useful_bits"] == 0.0
    assert result["total_all_useful_bits"] == pytest.approx(256.0)
    assert result["global_routing_reward"] == 0.0
    assert engine.routing_stage_eligible_packet_counts["COM"] == 0
    assert engine.pre_routing_violation_count == 1


def test_s2g_shares_fdma_and_never_overlaps_s2u_for_one_sr():
    env = _environment()
    _assign_sr_mission(env)
    proposed = {}
    for sender in range(env.num_UAV):
        for receiver in range(env.num_UAV + 1):
            if sender != receiver and env.is_routing_link_in_range(sender, receiver):
                proposed[sender] = receiver
                break
        if len(proposed) == 3:
            break
    assert len(proposed) == 3

    env.allocate_active_link_capacities(proposed, s2g_links={0: env.GS_ID})
    assert len(env.active_link_bandwidths) == 4
    assert set(env.active_link_bandwidths.values()) == {2_500_000.0}
    assert sum(env.active_link_bandwidths.values()) == pytest.approx(10_000_000.0)
    assert env.active_s2g_capacities[(0, env.GS_ID)] > 0.0

    s2u_receiver = next(
        uid for uid in range(env.num_UAV) if env.is_s2u_in_range(0, uid)
    )
    with pytest.raises(AssertionError, match="cannot use S2G and S2U"):
        env.allocate_active_link_capacities(
            {},
            s2u_links={0: s2u_receiver},
            s2g_links={0: env.GS_ID},
        )


def test_s2g_allocation_reuses_slot_rayleigh_profile_when_fdma_changes():
    env = _environment()
    _assign_sr_mission(env)

    gains_before = env.channel.gain_profile("S2G", 0, env.GS_ID).copy()
    env.allocate_active_link_capacities({}, s2g_links={0: env.GS_ID})
    profile_full = env.active_s2g_capacity_profiles_mbps[(0, env.GS_ID)].copy()
    proposed = next(
        ({sender: receiver} for sender in range(env.num_UAV)
         for receiver in range(env.num_UAV + 1)
         if sender != receiver and env.is_routing_link_in_range(sender, receiver)),
        None,
    )
    env.allocate_active_link_capacities(proposed, s2g_links={0: env.GS_ID})
    profile_half = env.active_s2g_capacity_profiles_mbps[(0, env.GS_ID)]
    assert np.array_equal(
        gains_before, env.channel.gain_profile("S2G", 0, env.GS_ID)
    )
    assert np.std(gains_before) > 0.0
    assert np.all(profile_half < profile_full)


def test_direct_state_change_keeps_com_task_binding_and_does_not_reassign():
    env = _environment()
    sr = _assign_sr_mission(env, gt_id=1)
    mission = sr.assigned_gt_id
    initial_invocations = env.assignment_invocations
    assert env.sr_direct_to_gs[0]
    env.task_list.append(Task(len(env.task_list), "COM", sr, 0))
    env.assign_tasks()
    initial_invocations = env.assignment_invocations
    original_binding = next(
        uid for uid, tasks in env.multi_tasks.items()
        if any(task["task_type"] == "COM" and task["target_obj_id"] == 0 for task in tasks)
    )

    sr.x, sr.y, sr.z = 101.0, 0.0, 0.0
    sr.path = []
    sr.arrived = True
    assert not env.prepare_next_movement_interval(1)
    assert sr.assigned_gt_id == mission
    assert env.assignment_invocations == initial_invocations
    assert any(
        task.task_type == "COM" and task.target_obj_id == 0
        for task in env.task_list
    )

    sr.x, sr.y, sr.z = 80.0, 0.0, 0.0
    assert not env.prepare_next_movement_interval(2)
    assert sr.assigned_gt_id == mission
    assert env.assignment_invocations == initial_invocations
    assert next(
        uid for uid, tasks in env.multi_tasks.items()
        if any(task["task_type"] == "COM" and task["target_obj_id"] == 0 for task in tasks)
    ) == original_binding
    assert not any(event.get("reason") == "sr_direct_state_change" for event in env.pending_assignment_events)
    assert [row["direct_to_gs_after"] for row in env.sr_direct_state_history] == [
        False,
        True,
    ]


def test_first_hop_endpoint_change_restarts_packet_without_changing_identity():
    env = _environment()
    sr = _assign_sr_mission(env)
    sr.x, sr.y, sr.z = 80.0, 0.0, 0.0
    engine = PacketEngine(16)
    packet = engine.create_sr_packet(0, 256.0, 1.0)
    packet_id = packet["id"]
    deadline = packet["deadline_abs"]
    assert engine.synchronize_sr_first_hop(env, 0, packet) == ("S2G", env.GS_ID)

    packet["s2u_bits_sent"] = 100.0
    packet["rem_bits"] = 156.0
    packet["_s2u_service_terms"] = [100.0]
    engine.s2u_backlog_bits[0] = 156.0
    sr.x = 101.0
    env.multi_tasks[1] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]

    assert engine.synchronize_sr_first_hop(env, 0, packet) == ("S2U", 1)
    assert packet["id"] == packet_id
    assert packet["deadline_abs"] == deadline
    assert packet["rem_bits"] == 256.0
    assert packet["s2u_bits_sent"] == 0.0
    assert engine.s2u_backlog_bits[0] == 256.0
    assert engine.sr_first_hop_switch_count == 1
    assert engine.sr_first_hop_retransmitted_bits == 100.0


def test_partial_s2u_then_s2g_delay_decomposition_is_mutually_exclusive():
    env = _environment()
    sr = _assign_sr_mission(env)
    sr.x, sr.y, sr.z = 101.0, 0.0, 0.0
    env.multi_tasks[1] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    engine = PacketEngine(
        16,
        task_deadlines_seconds={"FOV": 0.015, "COM": 1.0},
        enable_packet_diagnostic_artifacts=True,
    )
    packet = engine.create_sr_packet(0, 256.0, 0.0)
    assert engine.synchronize_sr_first_hop(env, 0, packet) == ("S2U", 1)
    packet.update(
        s2u_bits_sent=100.0,
        rem_bits=156.0,
        _s2u_service_terms=[100.0],
        s2u_service_start_time=0.001,
        s2u_actual_tx_seconds=0.005,
        sr_first_hop_attempt_actual_tx_seconds=0.005,
        sr_first_hop_cumulative_actual_tx_seconds=0.005,
    )
    engine.s2u_backlog_bits[0] = 156.0
    sr.x = 80.0
    result = engine.serve_event_driven_slot(
        env, {uid: uid for uid in range(16)}, 0.02
    )
    assert packet["done"]
    assert packet["delivery_path"] == "S2G"
    assert packet["sr_first_hop_retransmitted_bits"] == pytest.approx(100.0)
    hop = packet["per_hop"][0]
    assert hop["link_type"] == "S2G"
    assert hop["retransmitted_actual_tx_s"] == pytest.approx(0.005)
    assert hop["queue_s"] + hop["actual_tx_s"] == pytest.approx(
        packet["finish_time"] - packet["generation_time"]
    )
    assert result["total_all_useful_bits"] == pytest.approx(256.0)


def test_out_of_range_com_assignment_utility_is_zero_but_still_feasible():
    env = _environment()
    sr = _assign_sr_mission(env)
    uav = env.uav_dict[1]
    sr.x, sr.y, sr.z = 0.0, 0.0, 0.0
    uav.x_u, uav.y_u, uav.z_u = 401.0, 0.0, 0.0
    task = Task(0, "COM", sr, 0)
    problem = UAVAssigner(env).build_problem(
        [1], [task], candidate_task_types=("COM",)
    )

    assert problem.feasible_mask[0, 0]
    assert problem.raw_com_utility[0, 0] == 0.0
    assert problem.utility_matrix[0, 0] == 0.0


def test_sr_observation_modes_backlog_padding_and_task_aware_mask():
    env = _environment(num_gt=2)
    sr = _assign_sr_mission(env, sr_id=0, gt_id=1)
    engine = PacketEngine(16)
    engine.create_sr_packet(0, 256.0, 0.0)
    state = get_global_movement_state(env, engine, None, 1.0, 1.0)
    assert state.shape == (MOVEMENT_STATE_DIM,) == (619,)
    schema = movement_state_feature_schema()["features"]
    indices = {row["name"]: row["index"] for row in schema}
    assert state[indices["sr_0.valid"]] == 1.0
    assert state[indices["sr_0.mode_direct"]] == 1.0
    assert state[indices["sr_0.fifo_backlog"]] > 0.0
    assert state[indices["sr_0.target_x"]] == pytest.approx(
        env.gts[1].x / env.env_width
    )
    # SR 1 exists in the exogenous scenario, but its RoI is still undiscovered;
    # the fixed slot must not leak either that RoI or the dormant SR.
    assert state[indices["sr_1.valid"]] == 0.0
    assert all(
        state[row["index"]] == 0.0
        for row in schema
        if row["name"].startswith("sr_1.")
    )
    assert state[indices["sr_2.valid"]] == 0.0

    sr.x, sr.y, sr.z = 101.0, 0.0, 0.0
    waiting = get_global_movement_state(env, engine, None, 1.0, 1.0)
    assert waiting[indices["sr_0.mode_waiting"]] == 1.0
    env.multi_tasks[1] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 1}
    ]
    supported = get_global_movement_state(env, engine, None, 1.0, 1.0)
    assert supported[indices["sr_0.mode_support"]] == 1.0
    masked = apply_observation_strategy(supported, "masked", "movement")
    assert all(
        masked[row["index"]] == 0.0
        for row in schema
        if row["name"].startswith("sr_")
    )


def test_direct_sr_com_pair_still_contributes_to_movement_penalty():
    env = _environment()
    sr = _assign_sr_mission(env)
    assert env.is_s2g_in_range(0)
    env.multi_tasks[1] = [
        {"task_type": "COM", "target_obj_id": 0, "target_id": 0}
    ]
    result = movement_constraint_penalties(env)
    assert result["com_range_sample_count"] == 1
    assert result["com_distance_mean_m"] == pytest.approx(
        math.dist(env.uav_dict[1].get_position(), sr.get_position())
    )
