import copy
import math
import unittest
from unittest import mock

import numpy as np

from HRL_task_aware import TrainingConfig, _interval_reward
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from Task_assignment import Task, UAVAssigner
from centralized_movement import (
    MOVEMENT_STATE_DIM,
    calculate_movement_potentials,
    get_global_movement_state,
    movement_mask_from_state,
    movement_state_feature_schema,
)
from observation_strategy import apply_observation_strategy, routing_state_feature_names
from relay_contract import relay_metrics, requested_relay_count
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema


def discover_all(env):
    with mock.patch.object(env, "is_visible", return_value=True):
        env.update_visited_grid(1)


class RelayCountAndAssignmentTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 4
        self.env.reset_environment()

    def test_roi_to_relay_count_mapping(self):
        self.assertEqual(
            [requested_relay_count(count) for count in range(9)],
            [0, 0, 1, 1, 2, 2, 3, 3, 4],
        )

    def test_multiple_discoveries_trigger_one_next_boundary_assignment(self):
        before = self.env.assignment_invocations
        discover_all(self.env)
        self.env.update_visited_grid(2)
        self.assertEqual(self.env.count_found_targets(), 4)
        self.assertTrue(self.env.need_reassign)
        self.assertEqual(self.env.assignment_invocations, before)
        self.assertTrue(self.env.prepare_next_movement_interval(1))
        self.assertEqual(self.env.assignment_invocations, before + 1)
        self.assertEqual(
            self.env.assignment_history[-1]["discovered_roi_count"], 4
        )

    def test_k_km_fills_zero_utility_relay_quota_and_locks_roles(self):
        for gt in self.env.gts:
            gt.is_found = True
        self.env.Capacity_matrix[:] = 0.0
        self.env.gs_capacity[:] = 0.0
        self.env.assign_tasks()
        relay_ids = self.env.last_assignment.selected_relay_uav_ids
        self.assertEqual(len(relay_ids), 2)
        self.assertEqual(relay_ids, [1, 2])
        self.assertNotIn(self.env.permanent_gs_gateway_uav_id, relay_ids)
        for uid in relay_ids:
            tasks = self.env.multi_tasks[uid]
            self.assertEqual([task["task_type"] for task in tasks], ["Relay"])
            self.assertNotIn("target_pos", tasks[0])
            self.assertIsNone(tasks[0]["target_obj_id"])

    def test_km_uses_one_joint_relay_fov_com_round(self):
        discover_all(self.env)
        self.env.assignment_strategy = "km"
        self.env.assignment_rounds = 1
        self.env.assign_tasks()
        problem = self.env.last_assignment.last_round_problems[0]
        service_task_count = sum(
            task.task_type in {"Relay", "FOV", "COM"}
            for task in self.env.task_list
        )
        self.assertEqual(problem[0].shape[1], service_task_count)
        self.assertEqual(len(self.env.last_assignment.last_round_problems), 1)
        self.assertTrue(
            all(
                sum(
                    task["task_type"] in {"Relay", "FOV", "COM"}
                    for task in tasks
                )
                <= 1
                for tasks in self.env.multi_tasks.values()
            )
        )

    def test_random_joint_assignment_uses_named_rng_deterministically(self):
        self.env.gts[0].is_found = True
        self.env.SR_teams[0].assigned_gt_id = 0
        tasks = [
            Task("relay-1", "Relay", None, None),
            Task(1, "FOV", self.env.gts[0], 0),
            Task(2, "COM", self.env.SR_teams[0], 0),
        ]
        first = UAVAssigner(self.env)
        second = UAVAssigner(self.env)
        self.env.assignment_rng = np.random.default_rng(44)
        first_result = copy.deepcopy(
            first.assign_tasks([1, 2, 3], tasks, strategy="random_one_to_one")
        )
        self.env.assignment_rng = np.random.default_rng(44)
        second_result = copy.deepcopy(
            second.assign_tasks([1, 2, 3], tasks, strategy="random_one_to_one")
        )
        self.assertEqual(first_result, second_result)
        assigned_types = {
            task_type
            for assignments in first_result.values()
            for _index, task_type, _utility in assignments
        }
        self.assertEqual(assigned_types, {"Relay", "FOV", "COM"})


class RelayUtilityAndStateTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()

    def _three_node_path(self):
        self.env.get_available_uav_ids = lambda: [0, 1, 2]
        self.env.uav_dict[0].x_u, self.env.uav_dict[0].y_u = 0.0, 0.0
        self.env.uav_dict[1].x_u, self.env.uav_dict[1].y_u = 700.0, 0.0
        self.env.uav_dict[2].x_u, self.env.uav_dict[2].y_u = 350.0, 0.0
        for uid in (0, 1, 2):
            self.env.uav_dict[uid].z_u = 100.0
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()

    def test_scores_uniform_fallback_and_nonrelay_shortest_path(self):
        self._three_node_path()
        metrics = relay_metrics(self.env, 1, backlog_bits={0: 0, 1: 0, 2: 0})
        self.assertTrue(metrics.zero_backlog_fallback)
        self.assertTrue(metrics.reachable)
        self.assertEqual(metrics.first_next_hop, 2)
        self.assertEqual(metrics.shortest_path[0], 1)
        self.assertEqual(metrics.shortest_path[-1], self.env.GS_ID)
        for value in (
            metrics.receive_score,
            metrics.forward_score,
            metrics.utility,
        ):
            self.assertTrue(math.isfinite(value))
            self.assertTrue(0.0 <= value <= 1.0)

    def test_empty_source_out_of_range_and_unreachable_are_zero(self):
        self.env.get_available_uav_ids = lambda: [1]
        self.env.uav_dict[1].x_u = 1000.0
        self.env.uav_dict[1].y_u = 1000.0
        self.env.uav_dict[1].z_u = 100.0
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        metrics = relay_metrics(self.env, 1, backlog_bits={1: 100.0})
        self.assertEqual(metrics.source_uav_ids, ())
        self.assertEqual(metrics.receive_score, 0.0)
        self.assertFalse(metrics.reachable)
        self.assertEqual(metrics.forward_score, 0.0)
        self.assertEqual(metrics.utility, 0.0)

    def test_relay_state_schema_mask_and_movement_mask(self):
        self.env.multi_tasks = {uid: [] for uid in range(self.env.num_UAV)}
        self.env.multi_tasks[1] = [
            {"task_type": "Relay", "target_id": "relay-2-1", "target_obj_id": None}
        ]
        packet_engine = PacketEngine(10)
        state = get_global_movement_state(
            self.env,
            packet_engine,
            {uid: 0.0 for uid in range(10)},
            c_ref_com=1.0,
            remaining_time=1.0,
        )
        self.assertEqual(state.shape, (MOVEMENT_STATE_DIM,))
        schema = movement_state_feature_schema()["features"]
        by_name = {feature["name"]: feature["index"] for feature in schema}
        self.assertEqual(state[by_name["uav_1.task_relay"]], 1.0)
        self.assertTrue(movement_mask_from_state(state)[1])
        relay_suffixes = (
            "task_relay",
            "relay_receive_score",
            "relay_forward_score",
            "relay_receive_dx",
            "relay_receive_dy",
            "relay_receive_dz",
            "relay_forward_dx",
            "relay_forward_dy",
            "relay_forward_dz",
        )
        for suffix in relay_suffixes:
            self.assertEqual(state[by_name[f"uav_2.{suffix}"]], 0.0)
        masked = apply_observation_strategy(state, "masked", "movement")
        for uid in range(10):
            for suffix in relay_suffixes:
                self.assertEqual(masked[by_name[f"uav_{uid}.{suffix}"]], 0.0)

    def test_relay_potential_uses_explicit_frozen_backlog(self):
        self._three_node_path()
        self.env.multi_tasks = {uid: [] for uid in range(self.env.num_UAV)}
        self.env.multi_tasks[1] = [{"task_type": "Relay"}]
        frozen = {0: 0.0, 1: 10_000.0, 2: 50_000.0}
        first = calculate_movement_potentials(
            self.env, 1.0, backlog_bits=frozen
        )[3]
        self.env.assignment_backlog_snapshot = {uid: 9e9 for uid in range(10)}
        second = calculate_movement_potentials(
            self.env, 1.0, backlog_bits=frozen
        )[3]
        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first))
        self.assertTrue(0.0 <= first <= 1.0)


class RelayRoutingCheckpointDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_reassignment_preserves_packet_queue_identity_and_ownership(self):
        engine = PacketEngine(10)
        packet = {"id": 7, "rem_bits": 128.0}
        engine.uav_queues[1].append(packet)
        engine.backlog_bits[1] = 128.0
        queue_id = id(engine.uav_queues[1])
        for gt in self.env.gts:
            gt.is_found = True
        self.env.assign_tasks()
        self.assertEqual(id(engine.uav_queues[1]), queue_id)
        self.assertIs(engine.uav_queues[1][0], packet)
        self.assertEqual(engine.backlog_bits[1], 128.0)

    def test_relay_role_does_not_change_routing_mask_or_schema(self):
        before = self.env.get_routing_action_mask(1).copy()
        self.env.multi_tasks[1] = [{"task_type": "Relay"}]
        after = self.env.get_routing_action_mask(1)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(len(routing_state_feature_names()), 101)
        self.assertFalse(
            any("next_hop_is_relay" in name for name in routing_state_feature_names())
        )

    def test_no_task_potential_disables_relay_shaping(self):
        config = TrainingConfig(total_episodes=1)
        reward = _interval_reward(
            0.0,
            0.0,
            0.0,
            1.0,
            (0.0, 0.0, 0.0, 0.1),
            (0.0, 0.0, 0.0, 0.9),
            False,
            config,
            task_potential_enabled=False,
        )
        self.assertEqual(reward, 0.0)

    def test_old_checkpoint_fails_before_loading(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 23)
        with self.assertRaisesRegex(RuntimeError, "Relay.*retrained"):
            _validate_checkpoint_schema({"checkpoint_schema_version": 22})

    def test_diagnostics_are_rng_observational_and_publish_forwarding_groups(self):
        state = copy.deepcopy(self.env.assignment_rng.bit_generator.state)
        self.env.assignment_metadata()
        self.assertEqual(state, self.env.assignment_rng.bit_generator.state)
        engine = PacketEngine(10, enable_packet_diagnostic_artifacts=True)
        engine._record_relay_forwarding_observation(
            self.env, 1, self.env.GS_ID, 64.0, True
        )
        summary = engine.relay_forwarding_summary()
        self.assertEqual(
            set(summary),
            {
                "assigned_relay_forwarding",
                "nonassigned_uav_forwarding",
                "traversed_assigned_relay",
            },
        )
        self.assertEqual(summary["nonassigned_uav_forwarding"]["bits"], 64.0)


if __name__ == "__main__":
    unittest.main()
