from types import SimpleNamespace

import pytest

from object import SRTeam, straight_line_route, validate_sr_speed_mps
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
    assert route == pytest.approx([(0.75, 0.0), (1.5, 0.0), (2.0, 0.0)])
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
            "arrived_center_time_seconds": arrived,
        }
        if assigned
        else None
    )
    return SimpleNamespace(
        SR_teams=[team],
        gts=[target],
        sr_speed_mps=1.25,
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
        "assignment_to_center_arrival_seconds": 5.0,
    }
    assigned_censored = {
        "received_rescue_mission": True,
        "entered_roi": False,
        "arrived_center": False,
        "assignment_to_roi_entry_seconds": None,
        "assignment_to_center_arrival_seconds": None,
    }
    unassigned = {
        "received_rescue_mission": False,
        "entered_roi": False,
        "arrived_center": False,
        "assignment_to_roi_entry_seconds": None,
        "assignment_to_center_arrival_seconds": None,
    }
    dual = empty_dual_task_accumulator()
    dual["all"]["dual_task_uav_seconds"] = 2.0
    dual["all"]["both_valid_uav_seconds"] = 0.5
    dual_rows = summarize_dual_task_accumulator(dual, episode_index=0)
    summary = aggregate_diagnostics(
        [assigned_entered, assigned_censored, unassigned], dual_rows
    )
    sr = summary["sr_movement"]
    assert sr["assigned_sr_count"] == 2
    assert sr["entered_roi_ratio"] == 0.5
    assert sr["mean_assignment_to_roi_entry_seconds"] == 2.0
    assert sr["roi_entry_elapsed_seconds_sample_count"] == 1
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
