import copy
import math
import unittest

import numpy as np

from Channel_model import (
    A2G_LOS_A,
    A2G_LOS_B,
    A2G_LOS_EXCESS_DB,
    A2G_NLOS_EXCESS_DB,
    a2g_capacity_mbps,
    noise_power_dbm,
    u2u_path_loss_db,
)
from HRL_task_aware import (
    ROUTING_STATE_DIM,
    TrainingConfig,
    _interval_reward,
    terminal_ratio_objective,
)
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from Task_assignment import Task
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    calculate_movement_potentials,
    get_global_movement_state,
    movement_mask_from_state,
)
from experiment_config import (
    NUM_UAV,
    PERMANENT_GS_GATEWAY_UAV_ID,
    REFERENCE_COM_BANDWIDTH_HZ,
    RESERVED_SEARCH_UAV_IDS,
)
from observation_strategy import apply_observation_strategy
from scenario_manifest import ScenarioManifest, generate_manifest
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema


class Uav10ConfigurationContractTest(unittest.TestCase):
    def test_dimensions_and_manifest_layout_are_authoritative(self):
        self.assertEqual(NUM_UAV, 10)
        self.assertEqual(MOVEMENT_STATE_DIM, 519)
        self.assertEqual(JOINT_ACTION_DIM, 30)
        self.assertEqual(ROUTING_STATE_DIM, 101)
        self.assertEqual(RESERVED_SEARCH_UAV_IDS, (0, 9))
        self.assertEqual(PERMANENT_GS_GATEWAY_UAV_ID, 0)
        self.assertAlmostEqual(REFERENCE_COM_BANDWIDTH_HZ, 10e6 / 18.0)

        manifest = generate_manifest("test", 123, 1)
        entry = manifest.episodes[0]
        positions = [tuple(row["position"][:2]) for row in entry["uavs"]]
        self.assertEqual(
            positions,
            [
                (50.0, 50.0),
                (300.0, 250.0),
                (500.0, 250.0),
                (700.0, 250.0),
                (900.0, 250.0),
                (100.0, 750.0),
                (300.0, 750.0),
                (500.0, 750.0),
                (700.0, 750.0),
                (900.0, 750.0),
            ],
        )
        self.assertEqual(entry["exogenous_primitives"]["num_uav"], 10)
        self.assertEqual(
            entry["exogenous_primitives"]["reserved_search_uav_ids"], [0, 9]
        )

    def test_legacy_manifest_and_checkpoint_fail_fast(self):
        data = generate_manifest("test", 124, 1).to_dict()
        data["schema_version"] = "uav-hrl-scenario-v2"
        with self.assertRaisesRegex(ValueError, "16-UAV.*incompatible"):
            ScenarioManifest.from_dict(data)
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 23)
        with self.assertRaisesRegex(RuntimeError, "must be retrained"):
            _validate_checkpoint_schema({"checkpoint_schema_version": 9})


class StateAndAssignmentContractTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(NUM_UAV)
        self.env.apply_scenario_entry(generate_manifest("test", 200, 1, num_gt=2).episodes[0])
        self.engine = PacketEngine(NUM_UAV, step_time=0.25)
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()

    def movement_state(self):
        return get_global_movement_state(
            self.env, self.engine, self.engine.backlog_bits, 0.0128, 1.0
        )

    def test_state_does_not_observe_true_roi_count_and_uses_found_over_eight(self):
        before = self.movement_state()
        self.env.num_GT = 8
        after = self.movement_state()
        np.testing.assert_array_equal(before, after)

        self.env.gts[0].is_found = True
        discovered = self.movement_state()
        global_base = NUM_UAV * 26 + 16 * 16
        self.assertAlmostEqual(discovered[global_base + 1], 1.0 / 8.0)
        self.assertEqual(discovered.shape, (519,))

    def test_reserved_search_is_outside_solver_and_release_reassigns_all(self):
        gt = self.env.gts[0]
        sr = self.env.SR_teams[0]
        gt.is_found = True
        sr.assigned_gt_id = 0
        self.env.task_list.extend(
            [
                Task(0, "FOV", gt, gt.id),
                Task(1, "COM", sr, sr.id),
            ]
        )
        self.env.assign_tasks()
        for uav_id in RESERVED_SEARCH_UAV_IDS:
            self.assertEqual(
                [task["task_type"] for task in self.env.multi_tasks[uav_id]],
                ["Search"],
            )
        solver_ids = set(self.env.last_assignment.assignments)
        self.assertTrue(solver_ids.isdisjoint(RESERVED_SEARCH_UAV_IDS))

        previous_invocations = self.env.assignment_invocations
        self.env.visited_bitmap[:] = True
        self.env.current_time = 12.5
        self.env.convert_search_to_hovering()
        self.assertGreater(self.env.assignment_invocations, previous_invocations)
        self.assertEqual(self.env.search_release_time, 12.5)
        self.assertAlmostEqual(self.env.search_release_coverage, 1.0)
        self.assertFalse(
            any(
                task["task_type"] == "Search"
                for entries in self.env.multi_tasks.values()
                for task in entries
            )
        )
        self.assertEqual(
            set(self.env.last_assignment.assignments),
            set(range(NUM_UAV)) - {PERMANENT_GS_GATEWAY_UAV_ID},
        )
        self.assertEqual(
            [
                task["task_type"]
                for task in self.env.multi_tasks[PERMANENT_GS_GATEWAY_UAV_ID]
            ],
            ["Hovering"],
        )

    def test_fov_com_order_is_observation_potential_and_generation_invariant(self):
        uav_id = 1
        gt = self.env.gts[0]
        sr = self.env.SR_teams[0]
        gt.is_found = True
        sr.assigned_gt_id = 0
        uav = self.env.uav_dict[uav_id]
        uav.x_u, uav.y_u = gt.x, gt.y
        fov = {
            "task_type": "FOV",
            "target_id": 0,
            "target_obj_id": gt.id,
            "target_pos": gt.get_position(),
        }
        com = {
            "task_type": "COM",
            "target_id": 1,
            "target_obj_id": sr.id,
            "target_pos": sr.get_position(),
        }

        observations = []
        for tasks in ([fov, com], [com, fov]):
            self.env.multi_tasks[uav_id] = copy.deepcopy(tasks)
            self.env.update_source_uavs()
            movement = self.movement_state()
            routing = self.engine.get_state_ta(
                self.env,
                uav_id,
                backlog_bits=self.engine.backlog_bits,
                action_mask=self.env.get_routing_action_mask(uav_id),
            )
            potentials = calculate_movement_potentials(self.env, 0.0128)
            fresh = PacketEngine(NUM_UAV, step_time=0.25)
            fresh.inject_packets(
                self.env,
                20,
                current_time=0.0,
                step_time=0.25,
                rate_overrides={"FOV": 4.0, "COM": 4.0},
            )
            reward = _interval_reward(
                delivered_mbits=0.25,
                energy=100.0,
                current_lambda=0.001,
                gamma=1.0,
                potentials_t=potentials,
                potentials_t1=potentials,
                done=False,
                config=TrainingConfig(total_episodes=1),
            )
            observations.append(
                (
                    movement,
                    routing,
                    movement_mask_from_state(movement),
                    apply_observation_strategy(movement, "masked", "movement"),
                    apply_observation_strategy(routing, "masked", "routing"),
                    potentials,
                    fresh.generated_packet_counts.copy(),
                    reward,
                    self.env.assignment_metadata(),
                )
            )
        for index in range(6):
            np.testing.assert_array_equal(observations[0][index], observations[1][index])
        self.assertEqual(observations[0][6:], observations[1][6:])


class ChannelAndPacketContractTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(NUM_UAV)
        self.env.apply_scenario_entry(generate_manifest("test", 300, 1, num_gt=2).episodes[0])
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()

    def test_no_gs_projection_and_channel_closed_forms(self):
        uav = self.env.uav_dict[0]
        uav.x_u, uav.y_u, uav.z_u = 500.0, 500.0, 100.0
        proposal = uav.propose_movement(
            10.0,
            0.0,
            0.0,
            mobility_params=self.env.mobility_params,
            env_width=1000.0,
            env_height=1000.0,
        )
        self.assertGreater(math.dist(proposal["new_position"], self.env.GS_pos), 200.0)
        self.assertGreater(proposal["new_position"][0], 500.0)

        self.assertEqual((A2G_LOS_A, A2G_LOS_B), (11.95, 0.136))
        self.assertEqual((A2G_LOS_EXCESS_DB, A2G_NLOS_EXCESS_DB), (2.0, 20.0))
        self.assertAlmostEqual(float(noise_power_dbm(10e6)), -99.0, places=9)
        near = float(a2g_capacity_mbps((100, 0, 100), (0, 0, 0), 1e6, 30.0))
        far = float(a2g_capacity_mbps((800, 0, 100), (0, 0, 0), 1e6, 30.0))
        farther = float(a2g_capacity_mbps((800.001, 0, 100), (0, 0, 0), 1e6, 30.0))
        self.assertGreater(near, far)
        self.assertLess(abs(farther - far), 0.01)

        sender = np.array([0.0, 0.0, 100.0])
        receiver = np.array([500.0, 0.0, 120.0])
        expected = max(23.9 - 1.8 * math.log10(100.0), 20.0) * math.log10(
            np.linalg.norm(sender - receiver)
        ) + 20.0 * math.log10(40.0 * math.pi * 2.4 / 3.0)
        self.assertAlmostEqual(float(u2u_path_loss_db(sender, receiver)), expected)
        reverse = float(u2u_path_loss_db(receiver, sender))
        self.assertNotAlmostEqual(reverse, expected)

    def test_all_link_types_share_one_fdma_pool(self):
        positions = {
            0: (100.0, 0.0, 100.0),
            1: (0.0, 0.0, 100.0),
            2: (100.0, 0.0, 100.0),
            3: (0.0, 0.0, 100.0),
        }
        for uid, (x, y, z) in positions.items():
            self.env.uav_dict[uid].x_u = x
            self.env.uav_dict[uid].y_u = y
            self.env.uav_dict[uid].z_u = z
        self.env.SR_teams[0].x = 0.0
        self.env.SR_teams[0].y = 0.0
        self.env.SR_teams[0].z = 0.0
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        routing, bandwidths = self.env.allocate_active_link_capacities(
            {0: self.env.GS_ID, 1: 2}, s2u_links={0: 3}
        )
        self.assertEqual(len(routing), 2)
        self.assertEqual(len(bandwidths), 3)
        for bandwidth in bandwidths.values():
            self.assertAlmostEqual(bandwidth, 10e6 / 3.0)
        self.assertLessEqual(sum(bandwidths.values()), 10e6 + 1e-6)
        self.assertEqual(
            {row["link_type"] for row in self.env.active_link_diagnostics},
            {"S2U", "U2U", "U2G"},
        )

    def test_sr_fifo_partial_lock_next_slot_causality_and_e2e(self):
        engine = PacketEngine(NUM_UAV, step_time=0.25)
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        receiver = 1
        self.env.multi_tasks[receiver] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet = engine.create_sr_packet(0, 256.0, generation_time=0.0)
        next_packet = engine.create_sr_packet(0, 256.0, generation_time=0.1)

        self.env.active_s2u_capacities = {(0, receiver): 0.000512}
        engine.serve_active_links(self.env, {}, {}, current_time=0.0)
        self.assertIs(engine.get_sr_hol_packet(0), packet)
        self.assertEqual(packet["s2u_receiver"], receiver)
        self.assertIsNone(engine.get_hol_packet(receiver))

        self.env.multi_tasks[receiver] = []
        self.env.multi_tasks[2] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        self.assertEqual(engine.active_s2u_links(self.env), {0: receiver})
        self.env.active_s2u_capacities = {(0, receiver): 0.001024}
        arrival_slot = engine.serve_active_links(
            self.env, {}, {}, current_time=0.25
        )
        self.assertIs(engine.get_sr_hol_packet(0), next_packet)
        self.assertEqual(next_packet["rem_bits"], 256.0)
        self.assertIsNone(next_packet["s2u_receiver"])
        self.assertEqual(engine.active_s2u_links(self.env), {0: 2})
        self.assertIs(engine.get_hol_packet(receiver), packet)
        self.assertAlmostEqual(packet["routing_eligible_time"], 0.375)
        self.assertEqual(packet["generation_time"], 0.0)
        self.assertIsNone(packet["_queued_sr"])
        self.assertIsNone(packet["last_routing_sender"])
        self.assertEqual(arrival_slot["reward_by_sender"], {})
        self.assertEqual(arrival_slot["start_of_slot_routing_sender_ids"], ())

        engine.serve_active_links(
            self.env,
            {receiver: self.env.GS_ID},
            {(receiver, self.env.GS_ID): 1.0},
            current_time=0.5,
        )
        self.assertTrue(packet["done"])
        self.assertGreaterEqual(packet["e2e_delay_ms"], 500.0)
        self.assertEqual(engine.generated_packet_counts["COM"], 2)
        self.assertEqual(engine.total_delivered, 1)

    def test_s2u_arrival_appends_behind_frozen_hol_without_credit(self):
        engine = PacketEngine(NUM_UAV, step_time=0.25)
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        receiver = 1
        self.env.multi_tasks[receiver] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        resident = engine.create_packet(receiver, "FOV", 100.0, 0.0)
        arrival = engine.create_sr_packet(0, 100.0, generation_time=0.0)
        self.env.active_s2u_capacities = {(0, receiver): 0.001}

        result = engine.serve_active_links(
            self.env,
            {receiver: receiver},
            {},
            current_time=0.0,
        )

        self.assertIs(engine.get_hol_packet(receiver), resident)
        self.assertEqual(engine.get_queue_packets(receiver), [resident, arrival])
        self.assertEqual(resident["last_routing_sender"], receiver)
        self.assertIsNone(arrival["last_routing_sender"])
        self.assertEqual(result["reward_by_sender"][receiver], -0.5)
        self.assertEqual(result["cost_by_sender"], {receiver: 0.0})

    def test_s2u_completion_at_deadline_is_one_formal_violation(self):
        engine = PacketEngine(NUM_UAV, step_time=0.25)
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        receiver = 1
        self.env.multi_tasks[receiver] = [
            {
                "task_type": "COM",
                "target_id": 0,
                "target_obj_id": 0,
                "target_pos": sr.get_position(),
            }
        ]
        packet = engine.create_sr_packet(0, 256.0, generation_time=0.0)
        self.env.active_s2u_capacities = {(0, receiver): 0.001024}

        result = engine.serve_active_links(
            self.env, {}, {}, current_time=1.75
        )

        self.assertTrue(packet["done"])
        self.assertEqual(packet["reason"], "deadline")
        self.assertFalse(packet["routing_eligible"])
        self.assertIsNone(packet["routing_eligible_time"])
        self.assertIsNone(packet["last_routing_sender"])
        self.assertEqual(engine.eligible_packet_counts["COM"], 1)
        self.assertEqual(engine.total_violated, 1)
        self.assertEqual(len(result["outcomes"]), 1)
        self.assertTrue(result["outcomes"][0]["violated"])
        self.assertEqual(result["cost_by_sender"], {})
        self.assertEqual(engine.routing_constraint_counts(), (0, 0))
        self.assertEqual(engine.pre_routing_violation_count, 1)

    def test_direct_ratio_is_terminal_bit_per_j(self):
        self.assertEqual(terminal_ratio_objective("ratio", False, 1.0, 200000.0), 0.0)
        self.assertAlmostEqual(
            terminal_ratio_objective("ratio", True, 1.0, 200000.0), 5.0
        )


if __name__ == "__main__":
    unittest.main()
