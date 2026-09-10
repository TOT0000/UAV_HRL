"""Topology, resource allocation and movement contracts for snapshot Relay roles."""
import copy
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from Simulator import Simulator
from Task_assignment import Task, UAVAssigner
from relay_contract import (
    COUNT_RULE, bounded_minimax_center, connectivity_graph, initial_relay_count,
    movement_bounds, pair_relay_slots, plan_relays, refresh_relay_targets,
    relay_potential, relay_snapshot, reverse_bfs, valid_in_air_backlog, virtual_positions,
)


def environment(points, weights=None):
    uavs = {}
    for uid, point in enumerate(points):
        uav = SimpleNamespace(position=np.array(point, dtype=float), min_AGL=50., max_AGL=150.)
        uav.get_position = lambda uav=uav: tuple(uav.position)
        uavs[uid] = uav
    env = SimpleNamespace(num_UAV=len(points), uav_dict=uavs, GS_ID=len(points),
                          GS_pos=(0., 0., 0.), env_width=2000., env_height=2000.,
                          assignment_backlog_snapshot=weights or {},
                          assignment_rng=np.random.default_rng(72), multi_tasks={})
    env.is_u2g_position_in_range = lambda p: math.dist(p, env.GS_pos) <= 400.
    env.is_u2g_in_range = lambda uid: math.dist(uavs[uid].position, env.GS_pos) <= 400.
    return env


def install(env, plan, free):
    pair_relay_slots(env, plan, free)
    env.relay_plan = plan
    env.multi_tasks = {uid: [] for uid in range(env.num_UAV)}
    for slot in plan['slots']:
        env.multi_tasks[slot['assigned_uav_id']] = [
            {'task_type': 'Relay', 'target_id': slot['slot_id'], 'target_obj_id': None,
             'target_pos': tuple(slot['virtual_position']), 'relay_slot': slot}]


@pytest.mark.parametrize('distance, count', [(0,0),(399.99,0),(400,0),(400.01,1),
                                          (799.99,1),(800,1),(800.01,2),(1200,2)])
def test_initial_count_boundaries(distance, count):
    assert initial_relay_count(distance) == count


@pytest.mark.parametrize('dx, dz, expected', [(399.99,0,True),(400,0,True),(400.01,0,False),
                                           (400,1,False),(math.sqrt(400**2-50**2),50,True)])
def test_u2u_3d_range(dx, dz, expected):
    env = environment([(0,0,50),(dx,0,50+dz)])
    assert (1 in connectivity_graph(env)[0]) == expected


def test_connected_sources_no_relay_and_no_channel_access():
    env = environment([(0,0,100),(350,0,100),(700,0,100)], {2: 100})
    plan = plan_relays(env, [2], [1])
    assert plan['required_before_budget'] == 0
    assert plan['planning_source_ids'] == [2]
    assert plan['relay_count_rule'] == COUNT_RULE
    assert 'heuristic' in plan['optimality']


def test_component_pair_and_numeric_id_tie_break():
    env = environment([(100,100,100),(900,100,100),(950,100,100),(100,100,100)])
    plan = plan_relays(env, [2], [3])
    bridge = plan['source_bridges']['2']
    assert (bridge['L_s'], bridge['F_s'], bridge['endpoint_distance_m']) == (1,0,800.)
    assert plan['source_components']['2'] == [1,2]


def test_union_redundancy_shared_neighbors_and_deletion_minimal():
    env = environment([(100,100,100),(900,100,100),(900,150,100),(100,100,100)])
    plan = plan_relays(env, [1,2], [3])
    assert len(plan['initial_candidates']) == 2
    assert plan['required_before_budget'] == 1
    assert any(test['removed'] for test in plan['redundancy_tests'])
    slot = plan['slots'][0]
    assert slot['shared'] and len(slot['neighbor_ids']) >= 2
    assert slot['supported_source_ids'] == [1,2]
    assert plan['merge_validation'][-1]['valid']
    assert slot['witness_paths']['1']
    assert not all(s in reverse_bfs(connectivity_graph(env),env.GS_ID) for s in [1,2])


def test_shared_relay_bridges_two_disjoint_source_components_with_three_neighbors():
    env=environment([(100,100,100),(800,100,100),(600,450,100),(100,100,100)])
    plan=plan_relays(env,[1,2],[3])
    assert plan['source_components']['1']==[1]
    assert plan['source_components']['2']==[2]
    assert len(plan['initial_candidates'])==2 and len(plan['slots'])==1
    slot=plan['slots'][0]
    assert slot['shared'] and set(slot['neighbor_ids'])=={0,1,2}
    assert slot['supported_source_ids']==[1,2]
    np.testing.assert_allclose(slot['virtual_position'],[450,132.142857142857,100],atol=1e-4)
    assert plan['merge_validation'][-1]['valid']


def test_minimax_is_not_mean_and_fallback_is_deterministic():
    env = environment([(100,100,100)])
    points = [(0,0,100),(600,0,100),(50,0,100)]
    center, info = bounded_minimax_center(points, movement_bounds(env))
    np.testing.assert_allclose(center, [300,0,100], atol=1e-5)
    assert not np.allclose(center, np.mean(points,axis=0))
    fallback, details = bounded_minimax_center(points,movement_bounds(env),max_iter=0)
    repeated, repeated_details = bounded_minimax_center(points,movement_bounds(env),max_iter=0)
    assert details['fallback']
    np.testing.assert_allclose(fallback,center,atol=1e-5)
    np.testing.assert_array_equal(fallback,repeated)
    assert details == repeated_details


def test_failed_shared_validation_restores_candidate_and_rebuilds_witnesses():
    env = environment([(100,100,100),(900,100,100),(900,150,100),(100,100,100)])
    original = virtual_positions
    calls = []
    def fail_first(environment, slots):
        positions, status = original(environment,slots)
        if not calls:
            for item in status.values():
                item['feasible'] = False
        calls.append(len(slots))
        return positions,status
    with mock.patch('relay_contract.virtual_positions', side_effect=fail_first):
        plan = plan_relays(env,[1,2],[3])
    assert plan['merge_validation'][0]['valid'] is False
    assert plan['merge_validation'][0]['restored_candidate'] is not None
    assert plan['merge_validation'][-1]['valid']
    assert calls[:2] == [1,2]


def test_budget_recomputes_marginal_loss_and_prioritizes_actual_backlog():
    env = environment([(0,0,100),(1000,0,100),(0,1000,100),(0,0,100),(0,0,100)],{1:500})
    plan = plan_relays(env,[1,2],[3,4])
    assert plan['required_before_budget'] == 4
    assert plan['assigned_relay_count'] == 2
    assert plan['shortage'] == 2
    assert plan['unsupported_source_ids'] == [2]
    assert plan['unsupported_backlog'] == 0.
    assert plan['source_backlog_bits']['2'] == 0.
    for step in plan['budget_pruning']:
        assert step['backlog_loss'] == min(t['backlog_loss'] for t in step['candidate_tests'])
    assert len(plan['budget_pruning'][0]['candidate_tests']) == 4
    assert len(plan['budget_pruning'][1]['candidate_tests']) == 3
    assert plan['budget_pruning'][0]['lost_source_ids'] == [2]
    assert plan['budget_pruning'][1]['lost_source_ids'] == []


def test_greedy_3d_distance_ties_and_random_seed_pairing():
    env = environment([(0,0,100),(1000,0,100),(0,0,100),(0,0,100)])
    plan = plan_relays(env,[1],[2,3])
    first = copy.deepcopy(plan)
    pair_relay_slots(env,first,[3,2])
    assert first['slots'][0]['assigned_uav_id'] == 2
    for slot in first['slots']:
        assert slot['assignment_distance_m'] == math.dist(env.uav_dict[slot['assigned_uav_id']].position, slot['virtual_position'])
    a,b = copy.deepcopy(plan),copy.deepcopy(plan)
    env.assignment_rng = np.random.default_rng(19)
    pair_relay_slots(env,a,[2,3],random=True)
    env.assignment_rng = np.random.default_rng(19)
    pair_relay_slots(env,b,[3,2],random=True)
    assert a == b


def test_greedy_uses_altitude_and_actual_high_backlog_shortage():
    env = environment([(0,0,100),(1000,0,100),(500,100,150),(500,110,50)])
    plan = {'slots': [{'slot_id': 'q', 'virtual_position': [500,100,50]}], 'slot_to_uav': {}}
    pair_relay_slots(env,plan,[2,3])
    assert plan['slot_to_uav']['q'] == 3
    assert plan['slots'][0]['assignment_distance_m'] == 10.
    env = environment([(0,0,100),(1000,0,100),(0,1000,100),(0,0,100),(0,0,100)],{1:500,2:50})
    plan = plan_relays(env,[1,2],[3,4])
    assert plan['unsupported_source_ids'] == [2]
    assert plan['unsupported_backlog'] == 50.


def test_no_gs_component_records_unsupported_without_projection():
    env = environment([(900,900,100),(1000,1000,100)],{1:40})
    plan = plan_relays(env,[1],[0])
    assert plan['gs_component'] == []
    assert plan['unsupported_source_ids'] == [1]
    assert plan['unsupported_backlog'] == 40.
    assert plan['required_before_budget'] == 0


def test_virtual_gs_contract_and_minimax_ground_neighbor():
    env = environment([(100,100,100)])
    graph = connectivity_graph(env, {'q': [200.,0.,100.]})
    assert env.GS_ID in graph['q']
    center,info = bounded_minimax_center([(0,0,0),(0,0,80)],movement_bounds(env),max_iter=0)
    np.testing.assert_array_equal(center,[0,0,50])
    assert info['fallback']


def test_nonshared_chain_interpolation_clips_and_preserves_identity():
    env = environment([(0,0,100),(1000,0,100),(0,0,100)])
    plan = plan_relays(env,[1],[2])
    slot = plan['slots'][0]
    slot['shared'] = False
    slot['neighbor_ids'] = [0,1]
    env.uav_dict[1].position[0] = 1200
    positions,_ = virtual_positions(env,[slot])
    fraction=slot['chain_index']/(slot['chain_count']+1)
    expected=env.uav_dict[slot['L_s']].position[0]*(1-fraction)+env.uav_dict[slot['F_s']].position[0]*fraction
    assert positions[slot['slot_id']][0] == pytest.approx(expected)
    env.uav_dict[1].position[0]=10000
    _,status=virtual_positions(env,[slot])
    assert status[slot['slot_id']]['clipped']


def test_fixed_identity_target_motion_and_infeasible_diagnostics():
    env = environment([(100,100,100),(900,100,100),(100,100,100)])
    install(env,plan_relays(env,[1],[2]),[2])
    before = copy.deepcopy(env.relay_plan['slots'][0])
    env.uav_dict[1].position[0] = 1100.
    refresh_relay_targets(env)
    slot = env.relay_plan['slots'][0]
    for key in ('slot_id','assigned_uav_id','neighbor_ids','L_s','F_s'):
        assert slot[key] == before[key]
    assert slot['virtual_position'][0] == pytest.approx(600.,abs=1e-5)
    assert not env.relay_plan['position_status'][slot['slot_id']]['feasible']
    assert len(env.relay_plan['slots']) == 1


def test_potential_coefficients_reference_and_observational_diagnostics():
    env = environment([(100,100,100),(900,100,100),(100,100,100)])
    install(env,plan_relays(env,[1],[2]),[2])
    slot = env.relay_plan['slots'][0]
    env.uav_dict[2].position[:] = (500,100,100)
    from Channel_model import reference_u2u_max_capacity_mbps
    from experiment_config import TOTAL_COMMUNICATION_BANDWIDTH_HZ
    ref = reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
    with mock.patch('relay_contract.u2u_capacity_mbps', return_value=ref*.5):
        metrics = relay_potential(env,2,slot)
    assert metrics['P_pos'] == pytest.approx(1.)
    assert metrics['P_link'] == .5
    assert metrics['Phi_relay'] == pytest.approx(.65)
    before_plan = copy.deepcopy(env.relay_plan)
    before_rng = copy.deepcopy(env.assignment_rng.bit_generator.state)
    before_tasks = copy.deepcopy(env.multi_tasks)
    first,second = relay_snapshot(env),relay_snapshot(env)
    assert first == second
    assert env.relay_plan == before_plan
    assert env.multi_tasks == before_tasks
    assert env.assignment_rng.bit_generator.state == before_rng


def test_backlog_excludes_expired_ground_and_other_tasks_without_mutation():
    def packet(task,bits,deadline=2,done=False):
        return dict(task_type=task, rem_bits=bits,deadline_abs=deadline,done=done)
    engine = SimpleNamespace(num_UAV=2,uav_queues={0:[packet('FOV',10),packet('COM',20),
        packet('FOV',100,1),packet('COM',100,done=True),packet('Search',100)],1:[]},
        s2u_queues={0:[packet('COM',999)]})
    before = copy.deepcopy(engine.__dict__)
    assert valid_in_air_backlog(engine,1.) == {0:30.,1:0.}
    assert engine.__dict__ == before


@pytest.fixture
def production_env():
    env=Simulator(num_UAV=16)
    env.num_GT=2
    env.reset_environment()
    for uid,uav in env.uav_dict.items():
        uav.x_u,uav.y_u,uav.z_u=(100. if uid==0 else 900.),100.,100.
    with mock.patch.object(env,'is_visible',return_value=True):
        env.update_visited_grid(1)
    env.update_u2u_channels()
    env.update_u2g_channels()
    return env


@pytest.mark.parametrize('strategy,rounds',[('k_km',2),('km',1),('random_one_to_one',2)])
def test_service_first_and_relay_solver_spy(production_env,strategy,rounds):
    env=production_env
    env.assignment_strategy=strategy
    env.assignment_rounds=rounds
    import Task_assignment as module
    original_build=module.UAVAssigner.build_problem
    def limited_service(self,*args,**kwargs):
        problem=original_build(self,*args,**kwargs)
        for row,uid in enumerate(problem.uav_ids):
            if uid not in {1,2}:
                problem.feasible_mask[row,:]=False
        return problem
    original_pair=module.pair_relay_slots
    def spy_pair(*args,**kwargs):
        with mock.patch('Task_assignment.linear_sum_assignment',side_effect=AssertionError('Relay must not use Hungarian')):
            return original_pair(*args,**kwargs)
    with (mock.patch.object(module.UAVAssigner,'build_problem',new=limited_service),
          mock.patch('Task_assignment.pair_relay_slots',side_effect=spy_pair),
          mock.patch('Task_assignment.linear_sum_assignment',wraps=module.linear_sum_assignment) as solver):
        env.prepare_next_movement_interval(1)
    assert solver.call_count == (0 if strategy=='random_one_to_one' else rounds)
    assert env.relay_plan['assigned_relay_count'] > 0
    for uid,tasks in env.multi_tasks.items():
        if any(t['task_type']=='Relay' for t in tasks):
            assert len(tasks)==1
            assert uid not in env.relay_plan['prospective_source_ids']
            assert uid not in env.reserved_search_uav_ids
            assert uid != env.permanent_gs_gateway_uav_id
    problem=env.last_assignment.build_problem([1,2],env.task_list)
    assert all(t.task_type in {'FOV','COM'} for t in problem.tasks)
    assert not hasattr(problem,'raw_relay_utility')


def test_new_roi_boundary_only_and_normalized_masked_observation(production_env):
    env=production_env
    before=env.assignment_invocations
    assert env.prepare_next_movement_interval(1)
    assert env.assignment_invocations==before+1
    stable=copy.deepcopy(env.relay_plan['slots'])
    env.uav_dict[1].x_u=950.
    env.set_assignment_backlog_snapshot({1:500})
    assert not env.prepare_next_movement_interval(2)
    assert env.assignment_invocations==before+1
    assert [s['slot_id'] for s in stable]==[s['slot_id'] for s in env.relay_plan['slots']]
    assert [s['assigned_uav_id'] for s in stable]==[s['assigned_uav_id'] for s in env.relay_plan['slots']]
    from centralized_movement import get_global_movement_state,movement_state_feature_schema
    from observation_strategy import apply_observation_strategy
    from Packet_scheduler_v1 import PacketEngine
    state=get_global_movement_state(env,PacketEngine(16),{},1.,remaining_time=.5)
    assert state.shape==(595,) and np.isfinite(state).all()
    features={f['name']:f['index'] for f in movement_state_feature_schema()['features']}
    positions,_=virtual_positions(env,env.relay_plan['slots'])
    for uid in range(16):
        indices=[features[f'uav_{uid}.relay_target_d{a}'] for a in 'xyz']
        assigned=[s for s in env.relay_plan['slots'] if s['assigned_uav_id']==uid]
        if assigned:
            u=env.uav_dict[uid]
            expected=(positions[assigned[0]['slot_id']]-u.get_position())/[env.env_width,env.env_height,u.max_AGL-u.min_AGL]
            np.testing.assert_allclose(state[indices],np.clip(expected,-1,1),atol=1e-7)
        else:
            np.testing.assert_array_equal(state[indices],0.)
        np.testing.assert_array_equal(apply_observation_strategy(state,'masked','movement')[indices],0.)


def test_production_diagnostics_preserve_channel_rng_queues_and_environment(production_env):
    import pickle
    from relay_contract import relay_position_snapshot
    from Packet_scheduler_v1 import PacketEngine
    env=production_env
    env.prepare_next_movement_interval(1)
    engine=PacketEngine(16)
    engine.uav_queues[1].append({'task_type':'FOV','rem_bits':50.,'deadline_abs':2.})
    before=(pickle.dumps(env.channel.__dict__), pickle.dumps(env.multi_tasks),
            pickle.dumps(env.relay_plan),pickle.dumps(engine.__dict__),
            copy.deepcopy(env.assignment_rng.bit_generator.state))
    env.assignment_metadata()
    relay_snapshot(env)
    relay_position_snapshot(env)
    valid_in_air_backlog(engine,env.current_time)
    after=(pickle.dumps(env.channel.__dict__), pickle.dumps(env.multi_tasks),
           pickle.dumps(env.relay_plan),pickle.dumps(engine.__dict__),
           copy.deepcopy(env.assignment_rng.bit_generator.state))
    assert before==after


@pytest.mark.parametrize('method',['td3_dinkelbach','km_td3_dinkelbach','random_assignment_td3_dinkelbach'])
def test_evaluation_zero_relay_artifacts_all_assignment_methods(method,tmp_path):
    import json
    from HRL_task_aware import TrainingConfig,train
    from experiment_config import MethodSpec
    from scenario_manifest import generate_manifest
    from relay_diagnostics import write_relay_diagnostics, validate_relay_diagnostics
    checkpoint=tmp_path/'checkpoints'/'models'/'ep_0001'
    train(TrainingConfig(total_episodes=1,mode='custom',episode_seconds=1,
        warmup_joint_transitions=0,batch_size=1,enable_model_checkpoints=True,
        model_checkpoint_every=1,checkpoint_root=str(tmp_path/'checkpoints'),
        enable_full_resume=False,enable_csv=False,enable_plots=False,random_seed=72),
        scenario_manifest=generate_manifest('train',72,1,num_gt=2),method_spec=MethodSpec.parse(method))
    result=train(TrainingConfig(total_episodes=1, mode='custom', episode_seconds=1,
        warmup_joint_transitions=10000,batch_size=1,enable_model_checkpoints=False,
        enable_full_resume=False,enable_csv=False,enable_plots=False,random_seed=72),
        scenario_manifest=generate_manifest('test',72,1,num_gt=2),
        method_spec=MethodSpec.parse(method),evaluation=True,checkpoint_dir=checkpoint)
    diagnostics=result['relay_diagnostics']
    plan=diagnostics['episodes'][0]['assignment']['relay_planning']
    assert plan['required_before_budget']==plan['assigned_relay_count']==0
    assert plan['slots']==[] and plan['slot_to_uav']=={}
    assert diagnostics['planning_summary']['shortage_sum']==0
    _,path=write_relay_diagnostics(tmp_path,diagnostics)
    validate_relay_diagnostics(json.loads(path.read_text(encoding='utf-8')))


def test_relay_training_smoke_updates_actor_and_critic(production_env):
    from HRL_task_aware import TrainingConfig,train
    from experiment_config import MethodSpec
    from scenario_manifest import generate_manifest
    original=Simulator.assign_tasks
    def bridge(environment):
        if environment.count_found_targets():
            for uid,u in environment.uav_dict.items():
                u.x_u,u.y_u,u.z_u=(100. if uid==0 else 900.),100.,100.
            environment.update_u2u_channels()
            environment.update_u2g_channels()
        return original(environment)
    config=TrainingConfig(total_episodes=1,mode='custom',episode_seconds=3,
        warmup_joint_transitions=0,batch_size=1,enable_model_checkpoints=False,
        enable_full_resume=False,enable_csv=False,enable_plots=False,random_seed=73,
        beta_com=.6)
    assert config.beta_relay == .6
    with (mock.patch.object(Simulator,'is_visible',return_value=True),
          mock.patch.object(Simulator,'assign_tasks',new=bridge)):
        result=train(config,scenario_manifest=generate_manifest('train',73,1,num_gt=2),
                     method_spec=MethodSpec.parse('td3_ratio'))
    assert result['critic_updates'] > 0
    assert result['actor_updates'] > 0
    diagnostics=result['relay_diagnostics']
    assert diagnostics['episodes'][0]['assignment']['assigned_relay_count'] > 0
    assert diagnostics['planning_summary']['observed_movement_boundaries']==3
