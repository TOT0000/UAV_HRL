import copy
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Packet_scheduler_v1 import PacketEngine, sanitize_capture_coverage_ratio
from Simulator import Simulator
from Task_assignment import Task
from centralized_ddpg import CentralizedDDPG, RandomMovementController
from centralized_movement import (
    HOVER_ACTION,
    apply_joint_movement_proposals,
    build_joint_movement_proposals,
    executed_joint_action_from_displacement,
    gs_gateway_distance_m,
    project_gs_gateway_position,
)
from evaluation_aggregation import canonical_aggregation
from experiment_config import (
    COMMUNICATION_RANGE_CONTRACT_VERSION,
    GS_GATEWAY_HARD_RADIUS_M,
    GS_GATEWAY_PROJECTION_MODE,
    GS_GATEWAY_SOFT_RADIUS_M,
    METHOD_REGISTRY,
    NUM_UAV,
    PERMANENT_GS_GATEWAY_UAV_ID,
    RESERVED_SEARCH_UAV_IDS,
    MethodSpec,
    comparison_method_configuration,
)
from paper_metrics import aggregate_paper_point_metrics
from scenario_manifest import generate_manifest
from td3 import TD3


def _routing_env(num_uav=3):
    return SimpleNamespace(GS_ID=num_uav)


def _canonical_row(*, useful_mbits, energy_j, seed=1):
    row = {
        "training_seed": seed,
        "timely_goodput_mbits": useful_mbits,
        "total_timely_useful_mbits": useful_mbits,
        "total_mobility_energy_j": energy_j,
    }
    for task in ("fov", "com"):
        row.update(
            {
                f"{task}_delivered_packets": 1,
                f"{task}_delivered_e2e_delay_sum_seconds": 0.25,
                f"{task}_violation_packets": 0,
                f"{task}_eligible_packets": 1,
            }
        )
    return row


class PermanentGatewayProjectionTest(unittest.TestCase):
    def test_authoritative_constants_and_all_3d_boundaries(self):
        self.assertEqual(RESERVED_SEARCH_UAV_IDS, (0, 9))
        self.assertEqual(PERMANENT_GS_GATEWAY_UAV_ID, 0)
        self.assertEqual(GS_GATEWAY_SOFT_RADIUS_M, 360.0)
        self.assertEqual(GS_GATEWAY_HARD_RADIUS_M, 400.0)
        self.assertEqual(GS_GATEWAY_PROJECTION_MODE, "gs_only")

        direction = np.asarray((3.0, 4.0, 12.0)) / 13.0
        expected_distances = {
            359.0: 359.0,
            360.0: 360.0,
            380.0: 370.0,
            400.0: 400.0,
            500.0: 400.0,
        }
        for proposed_distance, expected_distance in expected_distances.items():
            with self.subTest(proposed_distance=proposed_distance):
                proposed = direction * proposed_distance
                projected = np.asarray(project_gs_gateway_position(proposed))
                self.assertTrue(
                    np.allclose(
                        projected,
                        direction * expected_distance,
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
                self.assertAlmostEqual(
                    gs_gateway_distance_m(projected), expected_distance
                )
        self.assertEqual(project_gs_gateway_position((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))

    def test_projection_respects_altitude_xy_and_height_dependent_horizontal_radius(self):
        for altitude in (50.0, 100.0, 150.0):
            boundary_x = math.sqrt(400.0**2 - altitude**2)
            with self.subTest(altitude=altitude, position="boundary"):
                projected = project_gs_gateway_position(
                    (boundary_x, 0.0, altitude),
                    min_altitude_m=50.0,
                    max_altitude_m=150.0,
                    env_width_m=1000.0,
                    env_height_m=1000.0,
                )
                np.testing.assert_allclose(
                    projected,
                    (boundary_x, 0.0, altitude),
                    rtol=0.0,
                    atol=1e-9,
                )
            with self.subTest(altitude=altitude, position="outside"):
                projected = project_gs_gateway_position(
                    (500.0, -20.0, altitude),
                    min_altitude_m=50.0,
                    max_altitude_m=150.0,
                    env_width_m=1000.0,
                    env_height_m=1000.0,
                )
                horizontal = math.hypot(projected[0], projected[1])
                horizontal_limit = math.sqrt(400.0**2 - projected[2] ** 2)
                self.assertGreaterEqual(projected[2], 50.0)
                self.assertLessEqual(projected[2], 150.0)
                self.assertGreaterEqual(projected[0], 0.0)
                self.assertLessEqual(projected[0], 1000.0)
                self.assertGreaterEqual(projected[1], 0.0)
                self.assertLessEqual(projected[1], 1000.0)
                self.assertLessEqual(horizontal, horizontal_limit + 1e-9)
                self.assertLessEqual(gs_gateway_distance_m(projected), 400.0 + 1e-9)

    def test_gateway_lifecycle_and_uav9_projection_exclusion(self):
        env = Simulator(num_UAV=NUM_UAV)
        env.num_GT = 2
        env.reset_environment()
        gt = env.gts[0]
        sr = env.SR_teams[0]
        gt.is_found = True
        sr.assigned_gt_id = gt.id
        env.task_list.extend(
            [Task(0, "FOV", gt, gt.id), Task(1, "COM", sr, sr.id)]
        )
        env.assign_tasks()
        self.assertEqual(
            [task["task_type"] for task in env.multi_tasks[0]], ["Search"]
        )
        self.assertTrue(env.multi_tasks[0][0]["permanent_gs_gateway"])

        env.visited_bitmap[:] = True
        env.convert_search_to_hovering()
        self.assertEqual(
            [task["task_type"] for task in env.multi_tasks[0]], ["Hovering"]
        )
        self.assertFalse(
            any(
                task["task_type"] in {"FOV", "COM"}
                for task in env.multi_tasks[0]
            )
        )

        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (390.0, 0.0, 50.0)
        env.uav_dict[9].x_u, env.uav_dict[9].y_u, env.uav_dict[9].z_u = (450.0, 0.0, 50.0)
        action = np.tile(np.asarray(HOVER_ACTION, dtype=np.float32), NUM_UAV)
        proposals = build_joint_movement_proposals(
            env, RandomMovementController(), action, step_time=0.25
        )
        self.assertLessEqual(gs_gateway_distance_m(proposals[0]["new_position"]), 400.0)
        self.assertEqual(proposals[9]["new_position"], (450.0, 0.0, 50.0))

    def test_random_td3_and_ddpg_share_gateway_execution_constraint(self):
        env = Simulator(num_UAV=NUM_UAV)
        env.num_GT = 2
        env.reset_environment()
        z = 50.0
        x = math.sqrt(399.0**2 - z**2)
        action = np.tile(np.asarray(HOVER_ACTION, dtype=np.float32), NUM_UAV)
        action[:3] = (1.0, 0.0, 0.0)
        controllers = (
            ("random", RandomMovementController()),
            ("td3", TD3.__new__(TD3)),
            ("ddpg", CentralizedDDPG),
        )
        for name, controller in controllers:
            with self.subTest(controller=name):
                env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
                    x,
                    0.0,
                    z,
                )
                proposals = build_joint_movement_proposals(
                    env, controller, action, step_time=1.0
                )
                apply_joint_movement_proposals(env, proposals, step_time=1.0)
                self.assertLessEqual(
                    gs_gateway_distance_m(env.uav_dict[0].get_position()),
                    GS_GATEWAY_HARD_RADIUS_M + 1e-9,
                )

    def test_executed_replay_action_matches_projected_displacement_and_energy(self):
        env = Simulator(num_UAV=NUM_UAV)
        env.num_GT = 2
        env.reset_environment()
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
            math.sqrt(359.0**2 - 50.0**2),
            0.0,
            50.0,
        )
        initial = np.asarray(
            [env.uav_dict[uid].get_position() for uid in range(NUM_UAV)], dtype=float
        )
        action = np.tile(np.asarray(HOVER_ACTION, dtype=np.float32), NUM_UAV)
        action[:3] = (1.0, 0.0, 0.0)
        proposals = build_joint_movement_proposals(
            env, RandomMovementController(), action, step_time=1.0
        )
        energies = apply_joint_movement_proposals(env, proposals, step_time=1.0)
        final = np.asarray(
            [env.uav_dict[uid].get_position() for uid in range(NUM_UAV)], dtype=float
        )
        replay_action = executed_joint_action_from_displacement(initial, final, 1.0)
        decoded = np.asarray(
            [
                CentralizedDDPG.decode_action(replay_action[uid * 3 : (uid + 1) * 3])
                for uid in range(NUM_UAV)
            ]
        )
        self.assertTrue(np.allclose(decoded, final - initial, atol=1e-5))
        self.assertGreaterEqual(float(energies[0]), 0.0)
        self.assertEqual(tuple(final[0]), env.uav_dict[0].get_position())

    def test_initial_gateway_validation_fails_before_reset_mutation(self):
        manifest = generate_manifest("test", 721, 1)
        valid = manifest.episodes[0]
        self.assertLessEqual(
            gs_gateway_distance_m(valid["uavs"][0]["position"]), 400.0
        )
        invalid = copy.deepcopy(valid)
        invalid["uavs"][0]["position"] = [450.0, 50.0, 100.0]
        env = Simulator(num_UAV=NUM_UAV)
        with self.assertRaisesRegex(ValueError, "permanent GS gateway.*hard radius"):
            env.reset_environment(invalid)

    def test_gateway_remains_u2u_receiver_and_u2g_sender(self):
        env = _routing_env()
        engine = PacketEngine(num_uav=3)
        packet = engine.create_packet(1, "COM", 100.0, 0.0)
        engine.serve_active_links(
            env, actions={1: 0}, capacities={(1, 0): 1.0}, current_time=0.0
        )
        self.assertIs(engine.get_hol_packet(0), packet)
        engine.serve_active_links(
            env, actions={0: env.GS_ID}, capacities={(0, env.GS_ID): 1.0}, current_time=0.25
        )
        self.assertTrue(packet["done"])
        self.assertEqual(packet["current"], env.GS_ID)


class CoverageWeightedUsefulGoodputTest(unittest.TestCase):
    def test_zero_and_invalid_coverage_still_generate_qos_packets(self):
        env = Simulator(num_UAV=NUM_UAV)
        env.num_GT = 2
        env.reset_environment()
        target = env.gts[0]
        env.multi_tasks = {uid: [] for uid in range(NUM_UAV)}
        env.multi_tasks[1] = [
            {
                "task_type": "FOV",
                "target_obj_id": target.id,
                "target_pos": target.get_position(),
            }
        ]
        env.source_uavs = {1}
        engine = PacketEngine(num_uav=NUM_UAV)
        with mock.patch(
            "Packet_scheduler_v1.fov_task_metrics",
            return_value=(float("nan"), 0.8, False),
        ):
            engine.inject_packets(
                env,
                delay_bound_steps=20,
                current_time=0.0,
                step_time=0.25,
                base_fov_rate=4,
            )
        packet = engine.get_active_packets()[0]
        self.assertEqual(packet["capture_coverage_ratio"], 0.0)
        self.assertGreater(packet["size_bits"], 0.0)
        self.assertEqual(engine.eligible_packet_counts["FOV"], 1)
        self.assertEqual(engine.fov_zero_coverage_packet_count, 1)
        for value in (float("nan"), float("inf"), -1.0, None):
            self.assertEqual(sanitize_capture_coverage_ratio(value), 0.0)

    def test_capture_snapshot_preserves_full_physical_service_and_useful_bits(self):
        env = _routing_env()
        engine = PacketEngine(num_uav=3)
        packet = engine.create_packet(
            0, "FOV", 100.0, 0.0, capture_coverage_ratio=0.8
        )
        partial = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0002},
            current_time=0.0,
        )
        self.assertEqual(partial["raw_final_hop_bits"], 50.0)
        self.assertEqual(packet["rem_bits"], 50.0)
        self.assertEqual(engine.backlog_bits[0], 50.0)
        # Delivery uses the frozen packet field; no geometry is consulted here.
        completed = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 1.0},
            current_time=0.25,
        )
        self.assertEqual(packet["capture_coverage_ratio"], 0.8)
        self.assertEqual(engine.fov_timely_delivered_raw_bits, 100.0)
        self.assertEqual(engine.fov_timely_useful_bits, 80.0)
        self.assertEqual(completed["timely_goodput_bits"], 80.0)
        self.assertEqual(engine.total_timely_useful_bits, 80.0)

    def test_full_zero_com_and_late_delivery_accounting(self):
        env = _routing_env()
        full = PacketEngine(num_uav=3)
        full.create_packet(0, "FOV", 100.0, 0.0, capture_coverage_ratio=1.0)
        full.serve_active_links(
            env, actions={0: env.GS_ID}, capacities={(0, env.GS_ID): 1.0}, current_time=0.0
        )
        self.assertEqual(full.fov_timely_delivered_raw_bits, 100.0)
        self.assertEqual(full.total_timely_useful_bits, 100.0)

        zero = PacketEngine(num_uav=3)
        zero.create_packet(0, "FOV", 100.0, 0.0, capture_coverage_ratio=0.0)
        zero.serve_active_links(
            env, actions={0: env.GS_ID}, capacities={(0, env.GS_ID): 1.0}, current_time=0.0
        )
        self.assertEqual(zero.fov_timely_delivered_raw_bits, 100.0)
        self.assertEqual(zero.total_timely_useful_bits, 0.0)
        self.assertEqual(zero.total_violated, 0)

        com = PacketEngine(num_uav=3)
        com.create_packet(0, "COM", 100.0, 0.0)
        com.serve_active_links(
            env, actions={0: env.GS_ID}, capacities={(0, env.GS_ID): 1.0}, current_time=0.0
        )
        self.assertEqual(com.com_timely_delivered_bits, 100.0)
        self.assertEqual(com.total_timely_useful_bits, 100.0)

        late = PacketEngine(
            num_uav=3, task_deadlines_seconds={"FOV": 0.1, "COM": 1.0}
        )
        late.create_packet(0, "FOV", 100.0, 0.0, capture_coverage_ratio=0.8)
        late.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.0,
        )
        self.assertEqual(late.fov_timely_useful_bits, 0.0)
        self.assertEqual(late.fov_timely_delivered_raw_bits, 0.0)
        self.assertEqual(late.total_violated, 1)

    def test_checkpoint_snapshot_keeps_coverage_integrator_and_counters(self):
        engine = PacketEngine(num_uav=3)
        engine.inject_buffer["0_FOV"] = 0.75
        packet = engine.create_packet(
            0, "FOV", 100.0, 0.0, capture_coverage_ratio=0.8
        )
        state = engine.checkpoint_state()
        saved = state["active_packets"][0]
        self.assertEqual(saved["id"], packet["id"])
        self.assertEqual(saved["capture_coverage_ratio"], 0.8)
        self.assertEqual(state["inject_buffer"], {"0_FOV": 0.75})
        self.assertEqual(state["fov_generated_raw_bits"], 100.0)
        self.assertEqual(state["fov_capture_coverage_sum"], 0.8)

    def test_generic_and_paper_share_useful_numerator(self):
        rows = [_canonical_row(useful_mbits=0.00008, energy_j=2.0)]
        _per_seed, generic = canonical_aggregation(rows)
        generic_ee = next(
            row for row in generic if row["metric"] == "energy_efficiency_mbit_per_j"
        )
        paper = aggregate_paper_point_metrics(
            "fixture",
            "fixed_roi",
            {
                "point_id": "roi_2",
                "x_value": 2,
                "x_unit": "RoI",
                "fixed_num_gt": 2,
                "swept_task": None,
            },
            rows,
        )
        paper_ee = next(
            row for row in paper if row["metric"] == "energy_efficiency_mbit_per_j"
        )
        self.assertEqual(generic_ee["pooled_numerator"], 0.00008)
        self.assertEqual(paper_ee["numerator"], generic_ee["pooled_numerator"])
        self.assertEqual(paper_ee["value"], generic_ee["mean"])

    def test_all_registry_methods_publish_identical_environment_contracts(self):
        fields = (
            "channel_environment_contract_version",
            "channel_fairness_contract_version",
            "communication_range_contract_version",
            "communication_range_boundary_rule",
            "maximum_3d_communication_distance_m",
            "s2u_communication_range_m",
            "u2g_communication_range_m",
            "u2u_communication_range_m",
            "routing_mask_scope",
            "com_session_lifecycle_version",
            "packet_service_contract_version",
            "evaluation_aggregation_schema_version",
            "permanent_gs_gateway_uav_id",
            "gs_gateway_contract_version",
            "fov_packet_generation_contract_version",
            "timely_useful_goodput_contract_version",
            "timely_goodput_definition",
            "fov_coverage_snapshot_timing",
        )
        contracts = [
            comparison_method_configuration(MethodSpec.parse(method_id))
            for method_id in METHOD_REGISTRY
        ]
        reference = {field: contracts[0][field] for field in fields}
        self.assertEqual(
            reference["communication_range_contract_version"],
            COMMUNICATION_RANGE_CONTRACT_VERSION,
        )
        self.assertEqual(reference["maximum_3d_communication_distance_m"], 400.0)
        self.assertEqual(reference["s2u_communication_range_m"], 400.0)
        self.assertEqual(reference["u2g_communication_range_m"], 400.0)
        self.assertEqual(reference["u2u_communication_range_m"], 400.0)
        self.assertTrue(
            all(
                {field: contract[field] for field in fields} == reference
                for contract in contracts
            )
        )


if __name__ == "__main__":
    unittest.main()
