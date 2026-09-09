"""Deterministic physical geometry and shared-production-path contract tests."""
import ast
import copy
from dataclasses import replace
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from visual_sensing import (
    CAMERA, VISUAL_SENSING_CONTRACT_VERSION, VS_PACKET_MAX_BITS,
    circle_polygon_intersection_area, search_footprint, vs_geometry,
    visual_sensing_metadata,
)
from centralized_movement import calculate_movement_potentials, fov_task_geometry, fov_task_metrics
from experiment_config import METHOD_REGISTRY, MethodSpec, effective_training_config
from HRL_task_aware import TrainingConfig, _mark_search_observations, _sensing_coverage
from Packet_scheduler_v1 import PacketEngine, fov_physical_packet_size_bits
from Simulator import Simulator
from Task_assignment import Task, UAVAssigner
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema


def environment():
    env = Simulator(num_UAV=16)
    env.num_GT = 2
    env.reset_environment()
    env.multi_tasks = {uid: [] for uid in range(16)}
    target = env.gts[0]
    target.x, target.y, target.z, target.radius = 500.0, 500.0, 0.0, 80.0
    uav = env.uav_dict[1]
    uav.x_u, uav.y_u, uav.z_u = 500.0, 500.0, 100.0
    descriptor = {"task_type": "FOV", "target_id": 0, "target_obj_id": 0,
                  "target_pos": target.get_position()}
    return env, target, uav, descriptor


def test_single_camera_source_and_nadir_dimensions():
    root = Path(__file__).resolve().parents[1]
    sources = []
    for file in root.glob('*.py'):
        if file.name.startswith('update_visual'):
            continue
        tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        if any(isinstance(node, ast.Constant) and node.value in (0.035, 0.0156, 0.0235)
               for node in ast.walk(tree)):
            sources.append(file.name)
    assert sources == ['visual_sensing.py']
    fp = search_footprint((0, 0, 100))
    assert fp.width == pytest.approx(44.5714285714)
    assert fp.height == pytest.approx(67.1428571429)


def test_search_boundary_outside_and_one_frozen_footprint():
    env, target, uav, _ = environment()
    env.multi_tasks[1] = [{"task_type": "Search"}]
    fp = env.search_footprint(1)
    target.x, target.y = fp.xmax, fp.ymax
    assert env.is_visible(1, target)
    target.x = fp.xmax + 1e-5
    assert not env.is_visible(1, target)
    target.x, target.y = uav.x_u + 450, uav.y_u + 450
    assert not env.is_visible(1, target)  # Old full-circle dL/dR gate could pass.
    target.x, target.y = fp.xmax, fp.ymax
    expected = np.zeros_like(env.visited_bitmap)
    indices = env.fov_footprint_indices(1, footprint=fp)
    x0, x1, y0, y1 = indices
    expected[x0:x1+1, y0:y1+1] = True
    env.visited_bitmap[:] = False
    with patch.object(env, 'search_footprint', wraps=env.search_footprint) as calls:
        transitions = _mark_search_observations(env)
    assert calls.call_count == 1
    assert target.is_found
    assert transitions[1].current_footprint == indices
    np.testing.assert_array_equal(env.visited_bitmap, expected)
    assert calculate_movement_potentials(env, 1)[0] == expected.mean()


@pytest.mark.parametrize('role', ['FOV', 'COM', 'Relay', 'Hovering'])
def test_nonsearch_cannot_discover_or_contribute(role):
    env, target, _, descriptor = environment()
    env.multi_tasks[1] = [{**descriptor, 'task_type': role}]
    env.visited_bitmap[:] = False
    env.update_visited_grid(1)
    transition = env.mark_search_coverage(1, coverage_contributor=True)
    assert not target.is_found
    assert not transition.coverage_contributor
    assert transition.current_footprint is None
    assert not env.visited_bitmap.any()
    env.multi_tasks[1] = [{'task_type': 'Search'}]
    env.update_visited_grid(1, coverage_contributor=False)
    assert not target.is_found
    assert env.mark_search_coverage(1, coverage_contributor=False).current_footprint is None


def test_nadir_vs_area_raw_quantity_and_relative_altitude():
    geometry = vs_geometry((0, 0, 125), (0, 0, 25), 80)
    fp = search_footprint((0, 0, 125), ground_z=25)
    assert geometry.sensing_valid_now
    assert geometry.relative_altitude == 100
    np.testing.assert_allclose(np.ptp(geometry.polygon, axis=0), [fp.width, fp.height])
    assert geometry.footprint_area == pytest.approx(fp.width*fp.height)
    assert geometry.image_quantity == pytest.approx(math.pi*80**2/geometry.footprint_area)
    assert geometry.image_quantity > 1
    assert geometry.coverage_ratio == pytest.approx(1/geometry.image_quantity)


def test_evaluation_exports_active_camera_pose_and_actual_roi_radius():
    env, target, uav, descriptor = environment()
    target.radius = 17.0
    uav.x_u -= 180
    env.multi_tasks[1] = [descriptor]
    geometry = vs_geometry(uav.get_position(),target.get_position(),17)
    exported = _sensing_coverage(env,1)[0]
    assert exported['geometry'] == 'oblique_ground_polygon'
    np.testing.assert_allclose(exported['polygon'],geometry.polygon)
    assert exported['image_quantity'] == geometry.image_quantity
    assert fov_task_metrics(env,1,descriptor)[1] == geometry.image_quantity
    env.multi_tasks[1] = [{'task_type':'Relay'}]
    assert _sensing_coverage(env,1) == []


@pytest.mark.parametrize('angle', [math.pi/2, math.pi/3, -math.pi/4, math.pi])
def test_oblique_rotation_area_and_original_image_quantity(angle):
    base = vs_geometry((-180, 0, 100), (0, 0, 0), 80)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    xy = rotation @ np.array([-180, 0])
    rotated = vs_geometry((*xy, 100), (0, 0, 0), 80)
    np.testing.assert_allclose(rotated.polygon, np.asarray(base.polygon) @ rotation.T, atol=1e-10)
    assert rotated.coverage_ratio == pytest.approx(base.coverage_ratio)
    assert rotated.footprint_area == pytest.approx(base.footprint_area)
    # Independent original model formula, used only as an analytical test oracle.
    z, d = 100, 180
    f, w, h = CAMERA.f_m, CAMERA.image_width_m, CAMERA.image_length_m
    expected_i = f*f*math.pi*80**2*(z*z-w*w*d*d/(4*f*f))**2/(w*h*(d*d+z*z)**1.5*z**3)
    assert rotated.image_quantity == pytest.approx(expected_i)


@pytest.mark.parametrize('polygon, expected', [
    ([(2,2),(3,2),(3,3),(2,3)], 0),
    ([(-2,-2),(2,-2),(2,2),(-2,2)], math.pi),
    ([(0,-2),(2,-2),(2,2),(0,2)], math.pi/2),
    ([(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)], 1),
    ([(1,-2),(2,-2),(2,2),(1,2)], 0),
])
def test_exact_circle_polygon_cases(polygon, expected):
    assert circle_polygon_intersection_area(polygon, (0,0), 1) == pytest.approx(expected, abs=1e-12)
    assert circle_polygon_intersection_area(polygon[::-1], (0,0), 1) == pytest.approx(expected, abs=1e-12)


def test_oblique_full_partial_and_finite_horizon_boundary():
    assert vs_geometry((0,0,100), (0,0,0), 5).coverage_ratio == pytest.approx(1)
    assert 0 < vs_geometry((0,0,100), (0,0,0), 80).coverage_ratio < 1
    for altitude in (1, 50, 100, 150):
        for fraction in (0, .1, .5, .9, .999, 1-1e-9, 1, 1+1e-9, 2):
            g = vs_geometry((CAMERA.b1*altitude*fraction,0,altitude), (0,0,0))
            assert np.isfinite([g.footprint_area, g.image_quantity, g.coverage_ratio, g.pair_score]).all()
            assert 0 <= g.coverage_ratio <= 1
            assert np.isfinite(g.polygon).all()
            if fraction >= 1:
                assert (g.image_quantity, g.coverage_ratio, g.quality) == (0,0,0)
    boundary = vs_geometry((CAMERA.b1*100,0,100), (0,0,0))
    assert boundary.model_range_valid and not boundary.sensing_valid_now
    assert boundary.proximity == 1


@pytest.mark.parametrize('position', [(0,0,0),(0,0,-1),(math.nan,0,100),(0,math.inf,100),(0,0,math.inf)])
def test_invalid_inputs_fail_closed(position):
    g = vs_geometry(position, (0,0,0))
    assert not g.sensing_valid_now
    assert (g.image_quantity,g.coverage_ratio,g.quality,g.proximity) == (0,0,0,0)


def test_assignment_outside_range_and_shared_quality_proximity_potential():
    env, target, _, descriptor = environment()
    target.is_found = True
    for uid, distance in ((1,900),(2,600)):
        uav = env.uav_dict[uid]
        uav.x_u, uav.y_u, uav.z_u = target.x+distance, target.y, 100
        env.multi_tasks[uid] = [descriptor]
    problem = UAVAssigner(env).build_problem([1,2], [Task(0,'FOV',target,0)])
    assert problem.feasible_mask.all()
    assert problem.raw_fov_utility[1,0] > problem.raw_fov_utility[0,0] > 0
    for uid in (1,2):
        g = fov_task_geometry(env,uid,descriptor)
        assert (g.image_quantity,g.coverage_ratio,g.quality) == (0,0,0)
        assert 0 < g.proximity < 1
    assert calculate_movement_potentials(env,1)[1] == pytest.approx(problem.raw_fov_utility.mean())
    for uid, distance in ((1,0),(2,180)):
        env.uav_dict[uid].x_u = target.x+distance
        g = fov_task_geometry(env,uid,descriptor)
        assert g.proximity == 1
        assert g.pair_score == pytest.approx(.8*g.coverage_ratio*min(g.image_quantity,1)+.2)
    problem = UAVAssigner(env).build_problem([1,2], [Task(0,'FOV',target,0)])
    qualities = [fov_task_geometry(env,uid,descriptor).quality for uid in (1,2)]
    assert np.argmax(problem.raw_fov_utility[:,0]) == np.argmax(qualities)


@pytest.mark.parametrize('quantity, expected', [(0,0),(.25,7900),(1,31600),(2,31600)])
def test_fixed_packet_size(quantity, expected):
    assert VS_PACKET_MAX_BITS == 31600
    assert fov_physical_packet_size_bits(quantity) == expected
    with patch('visual_sensing.CAMERA', replace(CAMERA, image_width_m=.5)):
        assert fov_physical_packet_size_bits(quantity) == expected


def test_invalid_injection_capture_freeze_and_useful_bits():
    env, target, uav, descriptor = environment()
    env.multi_tasks[1] = [descriptor]
    env.source_uavs = {1}
    engine = PacketEngine(num_uav=16, step_time=.25)
    coverage, quantity, _ = fov_task_metrics(env,1,descriptor)
    engine.inject_packets(env, delay_bound_steps=20, current_time=0, step_time=.25)
    first = engine.get_active_packets()[0]
    captured = (first['size_bits'], first['capture_coverage_ratio'])
    assert captured == pytest.approx((fov_physical_packet_size_bits(quantity),coverage))
    uav.x_u = target.x+1000
    for time in (.25,.5,.75):
        engine.inject_packets(env,delay_bound_steps=20,current_time=time,step_time=.25)
    assert engine.generated_packet_counts['FOV'] == 1
    assert engine.eligible_packet_counts['FOV'] == 1
    assert engine.active_count() == 1
    assert not engine.inject_buffer
    assert (first['size_bits'],first['capture_coverage_ratio']) == captured
    # Move into GS range; old capture stays unchanged during delivery.
    uav.x_u, uav.y_u, uav.z_u = env.GS_pos[0], env.GS_pos[1], 100
    engine.serve_active_links(env, actions={1:env.GS_ID}, capacities={(1,env.GS_ID):100}, current_time=1)
    assert engine.fov_timely_delivered_raw_bits == pytest.approx(captured[0])
    assert engine.fov_timely_useful_bits == pytest.approx(captured[0]*captured[1])


@pytest.mark.parametrize('method_id', list(METHOD_REGISTRY))
def test_all_method_assignments_share_visual_model(method_id):
    method = MethodSpec.parse(method_id)
    env, target, _, descriptor = environment()
    target.is_found = True
    task = Task(0,'FOV',target,0)
    assignment = UAVAssigner(env).assign_tasks([1,2], [task], strategy=method.assignment)
    selected = [uid for uid, entries in assignment.items() if entries]
    assert len(selected) == 1
    uid = selected[0]
    env.multi_tasks[uid] = [descriptor]
    g = fov_task_geometry(env,uid,descriptor)
    assert fov_task_metrics(env,uid,descriptor) == (g.coverage_ratio,g.image_quantity,g.sensing_valid_now)
    assert calculate_movement_potentials(env,1)[1] == pytest.approx(g.pair_score)
    cfg = effective_training_config(TrainingConfig(total_episodes=1),method)
    assert cfg['visual_sensing_configuration'] == visual_sensing_metadata()
    env.source_uavs = {uid}
    engine = PacketEngine(num_uav=16, step_time=.25)
    # Every registry method follows the same invalid -> valid scheduler gate.
    selected_uav = env.uav_dict[uid]
    selected_uav.x_u, selected_uav.y_u, selected_uav.z_u = target.x+1000, target.y, 100
    engine.inject_packets(env, delay_bound_steps=20, current_time=0, step_time=.25)
    assert not engine.packet_pool
    assert not engine.inject_buffer
    selected_uav.x_u = target.x
    g = fov_task_geometry(env,uid,descriptor)
    engine.inject_packets(env, delay_bound_steps=20, current_time=.25, step_time=.25)
    packet = engine.get_active_packets()[0]
    assert packet['size_bits'] == fov_physical_packet_size_bits(g.image_quantity)
    assert packet['capture_coverage_ratio'] == g.coverage_ratio
    from Simulator_KM import Simulator as KM
    from Simulator_Rand import Simulator as Rand
    assert KM is Rand is Simulator


def test_checkpoint_rejects_old_missing_or_changed_visual_contract():
    with pytest.raises(RuntimeError, match='retrained'):
        _validate_checkpoint_schema({'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION-1})
    current = {'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
               'visual_sensing_contract_version': VISUAL_SENSING_CONTRACT_VERSION,
               'visual_sensing_configuration': visual_sensing_metadata()}
    for key in ('visual_sensing_contract_version','visual_sensing_configuration'):
        bad = copy.deepcopy(current)
        del bad[key]
        with pytest.raises(RuntimeError, match='visual sensing'):
            _validate_checkpoint_schema(bad)
    current['visual_sensing_configuration']['packet_max_bits'] = 120500
    with pytest.raises(RuntimeError, match='visual sensing'):
        _validate_checkpoint_schema(current)
