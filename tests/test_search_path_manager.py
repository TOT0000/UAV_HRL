import math

import numpy as np
import pytest

from Simulator import Simulator
from centralized_movement import (
    HOVER_ACTION,
    JOINT_ACTION_DIM,
    LOCAL_MOVEMENT_DIM,
    MOVEMENT_STATE_DIM,
    movement_mask_from_state,
)
from experiment_config import (
    METHOD_REGISTRY,
    SEARCH_CONTROLLER_CONTRACT_VERSION,
    TASK_POTENTIAL_BETA_SEARCH,
    MethodSpec,
    comparison_method_configuration,
)
from search_path_manager import (
    SearchPathManager,
    build_search_regions,
    entropy_weights,
)
from visual_sensing import SearchFootprint, search_detection_overlap_ratio


@pytest.mark.parametrize(
    ("size", "axis_regions"),
    [(750, 8), (1000, 10), (1250, 13), (1500, 15), (1750, 18), (2000, 20)],
)
def test_region_grid_is_row_major_and_partitions_every_bitmap_cell(size, axis_regions):
    regions = build_search_regions(size, size, 2)
    assert len(regions) == axis_regions**2
    assert [region.region_id for region in regions] == list(range(len(regions)))
    assert regions[axis_regions].row == 1
    assert regions[axis_regions].column == 0
    ownership = np.zeros((size // 2, size // 2), dtype=np.uint8)
    for region in regions:
        ownership[region.x_start:region.x_stop, region.y_start:region.y_stop] += 1
    assert np.all(ownership == 1)
    assert sum(region.total_cells for region in regions) == ownership.size


def environment(width=1000):
    env = Simulator(16, environment_width_m=width, environment_height_m=width)
    env.num_GT = 2
    env.reset_environment()
    return env


def test_region_completion_is_exactly_99_percent():
    env = environment()
    manager = env.search_path_manager
    region = manager.regions[0]
    patch = env.visited_bitmap[region.x_start:region.x_stop, region.y_start:region.y_stop]
    required = math.ceil(0.99 * patch.size)
    patch.flat[: required - 1] = True
    assert not manager.region_completed(region)
    patch.flat[required - 1] = True
    assert manager.region_completed(region)


def test_entropy_weights_constant_columns_and_assignment_are_finite_deterministic():
    assert entropy_weights([1.0, 1.0], [2.0, 2.0]) == pytest.approx((0.5, 0.5))
    weights = entropy_weights([1.0, 2.0, 4.0], [3.0, 3.0, 3.0])
    assert np.isfinite(weights).all()
    assert weights == pytest.approx((1.0, 0.0))

    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[0] = [{"task_type": "Search"}]
    env.multi_tasks[1] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    manager.plan_interval(0)
    unfinished = manager.unfinished_regions()
    for region in unfinished:
        feasible = [uav_id for uav_id in (0, 1) if manager.region_feasible(uav_id, region)]
        if feasible:
            assert manager.owner_by_region[region.region_id] in feasible
    assert all(manager.owner_by_region[region_id] in (0, 1) for region_id in manager.owner_by_region)


def test_reallocation_only_on_events_and_preserves_active_region_lock():
    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[1] = [{"task_type": "Search"}]
    env.multi_tasks[2] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    manager.plan_interval(0)
    first_count = manager.reallocation_count
    locked_region = manager.active_region_by_uav[1]
    manager.plan_interval(1)
    assert manager.reallocation_count == first_count
    env.multi_tasks[2] = [{"task_type": "FOV", "target_obj_id": 0}]
    manager.plan_interval(2)
    assert manager.reallocation_count == first_count + 1
    assert manager.owner_by_region[locked_region] == 1
    assert all(owner != 2 for owner in manager.owner_by_region.values())


def test_frontier_waypoint_and_search_command_are_finite_and_speed_limited():
    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[1] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    region = manager.regions[0]
    env.visited_bitmap[region.x_start, region.y_start] = True
    candidates = manager._frontier_candidates(region)
    assert candidates
    for (x_index, y_index), point in candidates:
        assert not env.visited_bitmap[x_index, y_index]
        assert region.x_start <= x_index < region.x_stop
        assert region.y_start <= y_index < region.y_stop
        assert np.isfinite(point).all()
    env.uav_dict[1].z_u = 100.0
    command = manager.plan_interval(0)[1]
    assert np.linalg.norm(command[:2]) <= 10.0 + 1e-12
    assert command[2] == pytest.approx(2.0)
    env.uav_dict[1].z_u = 130.0
    command = manager.plan_interval(1)[1]
    assert command[2] == pytest.approx(-2.0)


def test_gateway_gets_only_wholly_reachable_regions_and_legal_waypoints():
    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[0] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    commands = manager.plan_interval(0)
    assert 0 in commands
    owned = [region for region in manager.regions if manager.owner_by_region.get(region.region_id) == 0]
    assert owned
    assert all(manager.region_feasible(0, region) for region in owned)
    active = manager.regions[manager.active_region_by_uav[0]]
    waypoint = manager._select_waypoint(0, active)
    assert math.dist((*waypoint, 120.0), env.GS_pos) <= 400.0 + 1e-12


def test_gateway_hovers_when_every_reachable_region_is_complete():
    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[0] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    for region in manager.regions:
        if manager.region_feasible(0, region):
            env.visited_bitmap[
                region.x_start:region.x_stop, region.y_start:region.y_stop
            ] = True
    command = manager.plan_interval(0)[0]
    np.testing.assert_array_equal(command, np.zeros(3))
    assert 0 not in manager.active_region_by_uav


def test_effective_visit_requires_new_cells_and_is_deduplicated_per_interval():
    env = environment()
    for uav_id in range(env.num_UAV):
        env.multi_tasks[uav_id] = [{"task_type": "Hovering"}]
    env.multi_tasks[1] = [{"task_type": "Search"}]
    manager = env.search_path_manager
    before = env.visited_bitmap.copy()
    transition = env.mark_search_coverage(1, visited_snapshot=before, commit=False)
    touched = [
        region.region_id
        for region in manager.regions
        if transition.current_footprint is not None
        and max(transition.current_footprint[0], region.x_start)
        < min(transition.current_footprint[1] + 1, region.x_stop)
        and max(transition.current_footprint[2], region.y_start)
        < min(transition.current_footprint[3] + 1, region.y_stop)
    ]
    manager.record_effective_visits((transition,), before, interval_index=0)
    manager.record_effective_visits((transition,), before, interval_index=0)
    assert all(manager.visit_counts[region_id] == 1 for region_id in touched)
    env.mark_search_coverage(1, commit=True)
    covered = env.visited_bitmap.copy()
    no_change = env.mark_search_coverage(1, visited_snapshot=covered, commit=False)
    manager.record_effective_visits((no_change,), covered, interval_index=1)
    assert all(manager.visit_counts[region_id] == 1 for region_id in touched)


def test_search_and_hover_masks_are_false_service_masks_true_and_replay_is_hover():
    state = np.zeros(MOVEMENT_STATE_DIM, dtype=np.float32)
    # Search, FOV, COM, Hovering task flags for UAVs 0..3.
    for uav_id, task_index in enumerate((0, 1, 2, 3)):
        state[uav_id * LOCAL_MOVEMENT_DIM + task_index] = 1.0
    mask = movement_mask_from_state(state)
    np.testing.assert_array_equal(mask[:4], [False, True, True, False])
    env = environment()
    executed = np.linspace(-1.0, 1.0, JOINT_ACTION_DIM, dtype=np.float32)
    replay = env.search_path_manager.replay_action(executed, mask, HOVER_ACTION).reshape(16, 3)
    np.testing.assert_array_equal(replay[0], HOVER_ACTION)
    np.testing.assert_array_equal(replay[3], HOVER_ACTION)
    np.testing.assert_array_equal(replay[1], executed.reshape(16, 3)[1])
    np.testing.assert_array_equal(replay[2], executed.reshape(16, 3)[2])


def test_single_footprint_discovery_uses_complete_roi_area_without_accumulation():
    radius = 1.0
    side_0999 = math.sqrt(0.0999 * math.pi)
    side_10 = math.sqrt(0.10 * math.pi)
    bounds = (-2.0, 2.0, -2.0, 2.0)
    below = SearchFootprint(-side_0999 / 2, side_0999 / 2, -side_0999 / 2, side_0999 / 2)
    exact = SearchFootprint(-side_10 / 2, side_10 / 2, -side_10 / 2, side_10 / 2)
    assert search_detection_overlap_ratio(below, (0, 0), radius, map_bounds=bounds) < 0.10
    assert search_detection_overlap_ratio(exact, (0, 0), radius, map_bounds=bounds) == pytest.approx(0.10)
    # Two sub-threshold samples remain two independent decisions.
    assert all(
        search_detection_overlap_ratio(below, (0, 0), radius, map_bounds=bounds) < 0.10
        for _ in range(2)
    )


def test_all_registered_methods_publish_the_same_external_search_contract():
    for method_id in METHOD_REGISTRY:
        metadata = comparison_method_configuration(MethodSpec.parse(method_id))
        assert metadata["search_controller"] == "deterministic_region_frontier_manager"
        assert metadata["search_controller_contract_version"] == SEARCH_CONTROLLER_CONTRACT_VERSION
        assert metadata["search_controlled_by_movement_policy"] is False
        assert metadata["search_potential_coefficient"] == 0
    assert TASK_POTENTIAL_BETA_SEARCH == 0
