"""Capture validity, assignment credit lifetime, and FIFO integration."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from centralized_movement import fov_task_geometry
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator


@pytest.fixture
def scene():
    env = Simulator(num_UAV=16)
    env.num_GT = 2
    env.reset_environment()
    env.multi_tasks = {uid: [] for uid in range(16)}
    for target in env.gts:
        target.x, target.y, target.z, target.radius = 500., 500., 0., 80.
    uav = env.uav_dict[1]
    uav.x_u, uav.y_u, uav.z_u = 500., 500., 100.
    task = {"task_type": "FOV", "target_obj_id": 0, "target_id": 10}
    env.multi_tasks[1] = [task]
    env.source_uavs = {1}
    # Explicitly leave COM inactive unless a test activates a session.
    for sr in env.SR_teams:
        sr.assigned_gt_id = None
    return env, uav, task, PacketEngine(num_uav=16, step_time=.25)


def inject(env, engine, slot):
    engine.inject_packets(env, 20, slot*.25, step_time=.25)


def credit(engine):
    return sum(value for key, value in engine.inject_buffer.items() if key.startswith('FOV:'))


def test_invalid_intervals_have_no_objects_counters_queue_or_credit(scene):
    env, uav, task, engine = scene
    uav.x_u = 1500.
    assert not fov_task_geometry(env, 1, task).sensing_valid_now
    with patch.object(engine, 'create_packet', wraps=engine.create_packet) as create:
        for slot in range(20):
            inject(env, engine, slot)
            assert engine.packet_pool == []
            assert engine.generated_packet_counts['FOV'] == 0
            assert engine.eligible_packet_counts['FOV'] == 0
            assert engine.fov_generated_raw_bits == 0
            assert engine.fov_capture_coverage_count == 0
            assert engine.fov_zero_coverage_packet_count == 0
            assert all(not queue for queue in engine.uav_queues.values())
            assert engine.active_count() == 0
            assert credit(engine) == 0
        create.assert_not_called()
    uav.x_u = 500.
    inject(env, engine, 20)
    assert engine.generated_packet_counts['FOV'] == 1
    assert credit(engine) == .25


def test_continuous_valid_partial_coverage_keeps_five_pps_cadence(scene):
    env, _, task, engine = scene
    geometry = fov_task_geometry(env, 1, task)
    assert geometry.sensing_valid_now and 0 < geometry.coverage_ratio < 1
    for slot, expected in enumerate((1, 2, 3, 5, 6, 7, 8, 10)):
        inject(env, engine, slot)
        assert engine.generated_packet_counts['FOV'] == expected
        assert engine.eligible_packet_counts['FOV'] == expected
        assert engine.fov_generated_raw_bits == pytest.approx(expected*31600*min(geometry.image_quantity, 1))


@pytest.mark.parametrize('transition', ['invalid', 'remove', 'remove_source', 'roi', 'task', 'reset'])
def test_fractional_credit_cannot_cross_assignment_or_validity_boundaries(scene, transition):
    env, uav, task, engine = scene
    for slot in range(3):
        inject(env, engine, slot)
    assert credit(engine) == .75
    old_keys = set(engine.inject_buffer)
    if transition == 'invalid':
        uav.x_u = 1500.
        inject(env, engine, 3)
        assert credit(engine) == 0
        uav.x_u = 500.
    elif transition == 'remove':
        env.multi_tasks[1] = []
        inject(env, engine, 3)
        assert credit(engine) == 0
        env.multi_tasks[1] = [task]
    elif transition == 'remove_source':
        env.source_uavs.clear()
        inject(env, engine, 3)
        assert credit(engine) == 0
        env.source_uavs = {1}
    elif transition == 'roi':
        task['target_obj_id'] = 1
    elif transition == 'task':
        task['target_id'] = 11
    else:
        engine.reset_packet_state()
        assert not engine.inject_buffer
    before = engine.generated_packet_counts['FOV']
    inject(env, engine, 4)
    assert engine.generated_packet_counts['FOV'] == before+1
    assert credit(engine) == .25
    if transition in ('roi', 'task'):
        assert old_keys.isdisjoint(engine.inject_buffer)
        # Switching back also starts a fresh segment, not the old .75 credit.
        task.update(target_obj_id=0, target_id=10)
        inject(env, engine, 5)
        assert credit(engine) == .25


def test_capture_quantity_and_identity_survive_movement_reassignment_and_delivery(scene):
    env, uav, task, engine = scene
    geometry = fov_task_geometry(env, 1, task)
    inject(env, engine, 0)
    packet = engine.packet_pool[0]
    fields = ('size_bits', 'capture_coverage_ratio', 'capture_image_quantity', 'capture_roi_id', 'capture_task_id')
    frozen = tuple(packet[name] for name in fields)
    assert frozen == (31600*min(geometry.image_quantity, 1), geometry.coverage_ratio, geometry.image_quantity, 0, 10)
    task.update(target_obj_id=1, target_id=11)
    uav.x_u = 1500.
    inject(env, engine, 1)
    assert tuple(packet[name] for name in fields) == frozen
    uav.x_u, uav.y_u, uav.z_u = env.GS_pos[0], env.GS_pos[1], 100.
    engine.serve_active_links(env, actions={1: env.GS_ID}, capacities={(1, env.GS_ID): 100.}, current_time=.5)
    assert packet['done']
    assert tuple(packet[name] for name in fields) == frozen
    assert engine.fov_timely_delivered_raw_bits == frozen[0]
    assert engine.fov_timely_useful_bits == pytest.approx(frozen[0]*frozen[1])


@pytest.mark.parametrize('quantity', [0., -1., float('nan'), float('inf')])
def test_inconsistent_valid_geometry_fails_before_credit_or_packet_creation(scene, quantity):
    env, _, task, engine = scene
    geometry = replace(fov_task_geometry(env, 1, task), image_quantity=quantity)
    with patch('Packet_scheduler_v1.fov_task_geometry', return_value=geometry):
        with pytest.raises(ValueError, match='positive finite packet size'):
            inject(env, engine, 0)
    assert not engine.packet_pool
    assert credit(engine) == 0


def test_invalid_vs_does_not_block_generated_com_or_change_com_credit(scene):
    env, uav, _, engine = scene
    uav.x_u = 1500.
    # First, an invalid capture leaves the UAV FIFO empty.
    inject(env, engine, 0)
    assert engine.get_hol_packet(1) is None
    sr = env.SR_teams[0]
    sr.assigned_gt_id = 0
    env.multi_tasks[1].append({'task_type': 'COM', 'target_obj_id': sr.id})
    # Deterministic link availability isolates generation and FIFO behavior.
    with patch.object(env, 'is_s2u_in_range', return_value=True):
        for slot in range(1, 5):
            engine.inject_packets(env, 20, slot*.25, rate_overrides={'COM': 5})
            assert engine.inject_buffer[f'SR_{sr.id}_COM'] == (slot*.25) % 1
        assert engine.generated_packet_counts['COM'] == 5
        assert engine.inject_buffer[f'SR_{sr.id}_COM'] == 0
        assert engine.generated_packet_counts['FOV'] == 0
        packets = list(engine.packet_pool)
        assert all(p['task_type'] == 'COM' for p in packets)
        engine.serve_s2u_links(env, {(sr.id, 1): 100.}, current_time=1.)
    uav.x_u, uav.y_u, uav.z_u = env.GS_pos[0], env.GS_pos[1], 100.
    engine.serve_active_links(env, actions={1: env.GS_ID}, capacities={(1, env.GS_ID): 100.}, current_time=1.25)
    assert all(p['done'] and p['current'] == env.GS_ID for p in packets)
    assert engine.get_hol_packet(1) is None
