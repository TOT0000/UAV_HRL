from types import SimpleNamespace

import numpy as np
import pytest

from object import SRTeam, straight_line_route, validate_sr_speed_mps
from HRL_task_aware import _mark_search_observations
from paper_evaluation import evaluation_sweep_points
from experiment_config import NUM_UAV
from scenario_manifest import generate_manifest
from Simulator import Simulator
from sr_speed_diagnostics import (
    aggregate_diagnostics,
    build_sr_episode_diagnostics,
    empty_dual_task_accumulator,
    observe_dual_task_service,
    summarize_dual_task_accumulator,
)


def test_sr_speed_sweep_builds_roi_speed_cartesian_with_unique_point_ids():
    points = evaluation_sweep_points(
        "sr_speed_sweep",
        roi_counts=(3, 5, 7),
        sr_speeds_mps=(0.125, 1.5),
    )
    assert [(point["fixed_num_gt"], point["sr_speed_mps"]) for point in points] == [
        (3, 0.125),
        (3, 1.5),
        (5, 0.125),
        (5, 1.5),
        (7, 0.125),
        (7, 1.5),
    ]
    assert len({point["point_id"] for point in points}) == len(points)


@pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_sr_speeds_are_rejected(value):
    with pytest.raises(ValueError, match="finite positive|requires"):
        evaluation_sweep_points("sr_speed_sweep", roi_counts=(3,), sr_speeds_mps=(value,))


def test_sr_speed_sweep_requires_explicit_speed_values():
    with pytest.raises(ValueError, match="requires.*sr-speeds-mps"):
        evaluation_sweep_points("sr_speed_sweep", roi_counts=(3,))


def test_fractional_speed_uses_update_duration_and_never_overshoots():
    route = straight_line_route((0.0, 0.0), (2.0, 0.0), 1.5, update_seconds=0.5)
    np.testing.assert_allclose(route, [(0.75, 0.0), (1.5, 0.0), (2.0, 0.0)])
    team = SRTeam(0)
    team.assign_mission(0, (2.0, 0.0), speed=1.5, update_seconds=1.0)
    team.step_forward()
    assert team.get_position()[:2] == pytest.approx((1.5, 0.0))
    # Four routing slots do not advance SR state; only the next movement boundary does.
    for _ in range(4):
        assert team.get_position()[:2] == pytest.approx((1.5, 0.0))
    team.step_forward()
    assert team.get_position()[:2] == pytest.approx((2.0, 0.0))
    assert team.arrived is True


def _movement_env(*, assigned=True, entered=None, arrived=None):
    team = SRTeam(0)
    team.x, team.y, team.z = (8.0, 0.0, 0.0)
    team.assigned_gt_id = 0 if assigned else None
    team.arrived = arrived is not None
    target = SimpleNamespace(x=10.0, y=0.0, z=0.0)
    target.get_position = lambda: (target.x, target.y, target.z)
    event = (
        {
            "assigned_roi_id": 0,
            "assignment_time_seconds": 4.0,
            "assignment_position_m": [0.0, 0.0, 0.0],
            "assignment_distance_to_roi_center_m": 10.0,
            "entered_roi_time_seconds": entered,
            "entered_roi_event_source": (
                "discrete_movement_update" if entered is not None else None
            ),
            "arrived_center_time_seconds": arrived,
            "arrived_center_event_source": (
                "discrete_movement_update" if arrived is not None else None
            ),
        }
        if assigned
        else None
    )
    return SimpleNamespace(
        SR_teams=[team],
        gts=[target],
        sr_speed_mps=1.25,
        dt=0.25,
        sr_update_seconds=1.0,
        sr_movement_events={} if event is None else {0: event},
    )


def test_entry_and_center_arrival_are_distinct_and_nulls_are_preserved():
    rows, summary = build_sr_episode_diagnostics(
        _movement_env(entered=7.0, arrived=10.0),
        episode_index=2,
        scenario_id="scenario-2",
        episode_end_s=12.0,
    )
    assert rows[0]["assignment_to_roi_entry_seconds"] == 3.0
    assert rows[0]["assignment_to_center_arrival_seconds"] == 6.0
    assert summary["entered_roi_ratio"] == 1.0
    assert summary["arrived_center_ratio"] == 1.0

    rows, summary = build_sr_episode_diagnostics(
        _movement_env(assigned=False),
        episode_index=0,
        scenario_id="unassigned",
        episode_end_s=60.0,
    )
    assert rows[0]["assigned_roi_id"] is None
    assert rows[0]["first_entered_roi_time_seconds"] is None
    assert rows[0]["first_arrived_center_time_seconds"] is None
    assert summary["assigned_sr_count"] == 0
    assert summary["entered_roi_ratio"] is None
    assert summary["mean_assignment_to_roi_entry_seconds"] is None
    assert summary["zero_assigned_denominator"] is True


def test_dual_task_uav_seconds_are_exclusive_and_grouped(monkeypatch):
    monkeypatch.setattr(
        "sr_speed_diagnostics.vs_geometry",
        lambda *args, **kwargs: SimpleNamespace(sensing_valid_now=True),
    )
    gt = SimpleNamespace(id=1, radius=80.0, get_position=lambda: (0.0, 0.0, 0.0))
    sr = SimpleNamespace(assigned_gt_id=1)
    uav = SimpleNamespace(get_position=lambda: (0.0, 0.0, 100.0))
    env = SimpleNamespace(
        num_UAV=1,
        multi_tasks={
            0: [
                {"task_type": "FOV", "target_obj_id": 1},
                {"task_type": "COM", "target_obj_id": 0},
            ]
        },
        SR_teams=[sr],
        gts=[SimpleNamespace(), gt],
        uav_dict={0: uav},
        is_s2u_in_range=lambda sr_id, uav_id: False,
    )
    accumulator = empty_dual_task_accumulator()
    observe_dual_task_service(env, accumulator, 0.25)
    rows = {row["roi_relationship"]: row for row in summarize_dual_task_accumulator(accumulator)}
    assert rows["all"]["dual_task_uav_seconds"] == 0.25
    assert rows["all"]["vs_only_uav_seconds"] == 0.25
    assert rows["same_roi"]["dual_task_uav_seconds"] == 0.25
    assert rows["different_roi"]["dual_task_uav_seconds"] == 0.0
    categories = sum(
        rows["all"][key]
        for key in (
            "both_valid_uav_seconds",
            "vs_only_uav_seconds",
            "com_only_uav_seconds",
            "neither_valid_uav_seconds",
        )
    )
    assert categories == rows["all"]["dual_task_uav_seconds"]


def test_speed_validator_preserves_fractional_values_and_rejects_bool():
    assert validate_sr_speed_mps("0.375") == 0.375
    with pytest.raises(ValueError):
        validate_sr_speed_mps(True)


def test_cross_episode_summary_pools_raw_counts_and_event_time_sums():
    assigned_entered = {
        "received_rescue_mission": True,
        "entered_roi": True,
        "arrived_center": True,
        "assignment_to_roi_entry_seconds": 2.0,
        "assignment_to_roi_entry_time_valid": True,
        "assignment_to_center_arrival_seconds": 5.0,
        "assignment_to_center_arrival_time_valid": True,
    }
    assigned_censored = {
        "received_rescue_mission": True,
        "entered_roi": False,
        "arrived_center": False,
        "assignment_to_roi_entry_seconds": None,
        "assignment_to_roi_entry_time_valid": False,
        "assignment_to_center_arrival_seconds": None,
        "assignment_to_center_arrival_time_valid": False,
    }
    assigned_unresolved = {
        "received_rescue_mission": True,
        "entered_roi": True,
        "arrived_center": False,
        "assignment_to_roi_entry_seconds": None,
        "assignment_to_roi_entry_time_valid": False,
        "assignment_to_center_arrival_seconds": None,
        "assignment_to_center_arrival_time_valid": False,
    }
    unassigned = {
        "received_rescue_mission": False,
        "entered_roi": False,
        "arrived_center": False,
        "assignment_to_roi_entry_seconds": None,
        "assignment_to_roi_entry_time_valid": False,
        "assignment_to_center_arrival_seconds": None,
        "assignment_to_center_arrival_time_valid": False,
    }
    dual = empty_dual_task_accumulator()
    dual["all"]["dual_task_uav_seconds"] = 2.0
    dual["all"]["both_valid_uav_seconds"] = 0.5
    dual_rows = summarize_dual_task_accumulator(dual, episode_index=0)
    summary = aggregate_diagnostics(
        [
            assigned_entered,
            assigned_censored,
            assigned_unresolved,
            unassigned,
        ],
        dual_rows,
    )
    sr = summary["sr_movement"]
    assert sr["assigned_sr_count"] == 3
    assert sr["entered_roi_count"] == 2
    assert sr["entered_roi_ratio"] == pytest.approx(2.0 / 3.0)
    assert sr["mean_assignment_to_roi_entry_seconds"] == 2.0
    assert sr["roi_entry_elapsed_seconds_sample_count"] == 1
    assert sr["roi_entry_elapsed_seconds_unresolved_event_count"] == 1
    all_dual = summary["dual_task_service"][0]
    assert all_dual["dual_task_uav_seconds"] == 2.0
    assert all_dual["simultaneously_valid_ratio"] == 0.25


def test_runtime_speed_override_does_not_mutate_manifest_or_other_simulator():
    manifest = generate_manifest(
        split="test",
        manifest_seed=2468,
        episode_count=1,
        num_gt=3,
    )
    original_hash = manifest.content_hash
    original_speed = manifest.episodes[0]["sr_teams"][0]["movement_primitive"][
        "speed_mps"
    ]
    fast = Simulator(num_UAV=NUM_UAV, evaluation=True, sr_speed_mps_override=2.75)
    default = Simulator(num_UAV=NUM_UAV, evaluation=True)
    fast.apply_scenario_entry(manifest.episodes[0])
    default.apply_scenario_entry(manifest.episodes[0])
    assert fast.sr_speed_mps == 2.75
    assert default.sr_speed_mps == original_speed == 1.0
    assert [sr.get_position() for sr in fast.SR_teams] == [
        sr.get_position() for sr in default.SR_teams
    ]
    assert manifest.content_hash == original_hash
    assert manifest.episodes[0]["sr_teams"][0]["movement_primitive"][
        "speed_mps"
    ] == original_speed


def test_assignment_inside_roi_records_entry_at_assignment_time():
    manifest = generate_manifest(
        split="test",
        manifest_seed=97531,
        episode_count=1,
        num_gt=3,
    )
    env = Simulator(num_UAV=NUM_UAV, evaluation=True, sr_speed_mps_override=0.75)
    env.apply_scenario_entry(manifest.episodes[0])
    sr = env.SR_teams[0]
    gt = env.gts[0]
    sr.x, sr.y = float(gt.x + gt.radius / 2.0), float(gt.y)
    selected = env.SR_team_gogo(gt, assignment_time_seconds=6.25)
    assert selected.id == sr.id
    event = env.sr_movement_events[sr.id]
    assert event["entered_roi_time_seconds"] == 6.25
    assert event["arrived_center_time_seconds"] is None


def _executed_assignment_and_update(*, assignment_time, start_distance, speed=5.0):
    manifest = generate_manifest(
        split="test",
        manifest_seed=8675309,
        episode_count=1,
        num_gt=3,
    )
    env = Simulator(
        num_UAV=NUM_UAV,
        evaluation=True,
        sr_speed_mps_override=speed,
    )
    env.apply_scenario_entry(manifest.episodes[0])
    # Formal execution performs an unassigned slot-0 SR boundary update first.
    env.advance_sr_teams(observation_time_seconds=0.0)
    gt = env.gts[0]
    gt.x, gt.y = 500.0, 500.0
    for other in env.gts[1:]:
        other.is_found = True
    for index, sr in enumerate(env.SR_teams):
        sr.x, sr.y, sr.z = (
            (gt.x - start_distance, gt.y, 0.0)
            if index == 0
            else (1000.0, 1000.0, 0.0)
        )
    env.is_visible = lambda uav_id, target, **kwargs: int(target.id) == 0
    # This is the production discovery -> assignment call, with the routing-slot
    # observation timestamp supplied by HRL_task_aware.
    _mark_search_observations(
        env,
        observation_time_seconds=assignment_time,
    )
    assigned = next(sr for sr in env.SR_teams if sr.assigned_gt_id == gt.id)
    env.advance_sr_teams(observation_time_seconds=1.0)
    rows, summary = build_sr_episode_diagnostics(
        env,
        episode_index=0,
        scenario_id="executed-order",
        episode_end_s=1.0,
    )
    return env, assigned, next(row for row in rows if row["sr_id"] == assigned.id), summary


def test_last_routing_slot_assignment_then_same_boundary_update_has_unresolved_elapsed():
    _, _, row, summary = _executed_assignment_and_update(
        assignment_time=1.0,
        start_distance=84.0,
    )
    assert row["mission_assignment_time_seconds"] == 1.0
    assert row["first_entered_roi_observation_time_seconds"] == 1.0
    assert row["entered_roi"] is True
    assert row["assignment_to_roi_entry_seconds"] is None
    assert row["assignment_to_roi_entry_time_valid"] is False
    assert row["assignment_to_roi_entry_time_reason"] == (
        "same_timestamp_discrete_movement_after_assignment"
    )
    assert row["movement_state_update_resolution_seconds"] == 1.0
    assert summary["entered_roi_count"] == 1
    assert summary["roi_entry_elapsed_seconds_sample_count"] == 0
    assert summary["roi_entry_elapsed_seconds_unresolved_event_count"] == 1
    assert summary["mean_assignment_to_roi_entry_seconds"] is None


def test_earlier_routing_slot_assignment_uses_later_boundary_observation_elapsed():
    _, _, row, summary = _executed_assignment_and_update(
        assignment_time=0.25,
        start_distance=84.0,
    )
    assert row["first_entered_roi_observation_time_seconds"] == 1.0
    assert row["assignment_to_roi_entry_seconds"] == 0.75
    assert row["assignment_to_roi_entry_time_valid"] is True
    assert row["assignment_to_roi_entry_time_reason"] == (
        "observed_at_later_discrete_movement_update"
    )
    assert summary["roi_entry_elapsed_seconds_sample_count"] == 1
    assert summary["roi_entry_elapsed_seconds_unresolved_event_count"] == 0


@pytest.mark.parametrize(
    ("start_distance", "expected_center"),
    [(40.0, False), (0.0, True)],
)
def test_assignment_state_satisfying_event_keeps_legal_zero_elapsed(
    start_distance, expected_center
):
    manifest = generate_manifest(
        split="test",
        manifest_seed=123123,
        episode_count=1,
        num_gt=3,
    )
    env = Simulator(num_UAV=NUM_UAV, evaluation=True, sr_speed_mps_override=5.0)
    env.apply_scenario_entry(manifest.episodes[0])
    gt = env.gts[0]
    sr = env.SR_teams[0]
    for other in env.SR_teams[1:]:
        other.x, other.y = 1000.0, 1000.0
    sr.x, sr.y = float(gt.x - start_distance), float(gt.y)
    selected = env.SR_team_gogo(gt, assignment_time_seconds=1.0)
    assert selected.id == sr.id
    rows, _ = build_sr_episode_diagnostics(
        env,
        episode_index=0,
        scenario_id="legal-zero",
        episode_end_s=1.0,
    )
    row = next(item for item in rows if item["sr_id"] == sr.id)
    assert row["assignment_to_roi_entry_seconds"] == 0.0
    assert row["assignment_to_roi_entry_time_valid"] is True
    assert row["assignment_to_roi_entry_time_reason"] == (
        "already_satisfied_at_assignment"
    )
    assert row["arrived_center"] is expected_center
    assert row["assignment_to_center_arrival_seconds"] == (
        0.0 if expected_center else None
    )


def test_executed_entry_and_center_arrival_remain_distinct_events():
    env, assigned, row, _ = _executed_assignment_and_update(
        assignment_time=0.25,
        start_distance=84.0,
    )
    assert row["entered_roi"] is True
    assert row["arrived_center"] is False
    for boundary in range(2, 18):
        env.advance_sr_teams(observation_time_seconds=float(boundary))
    rows, _ = build_sr_episode_diagnostics(
        env,
        episode_index=0,
        scenario_id="distinct-events",
        episode_end_s=17.0,
    )
    row = next(item for item in rows if item["sr_id"] == assigned.id)
    assert row["first_entered_roi_observation_time_seconds"] == 1.0
    assert row["first_arrived_center_observation_time_seconds"] == 17.0
    assert row["assignment_to_roi_entry_seconds"] == 0.75
    assert row["assignment_to_center_arrival_seconds"] == 16.75
    assert row["episode_end_remaining_distance_to_roi_center_m"] == pytest.approx(0.0)
