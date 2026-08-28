import math
import unittest

import numpy as np

from Channel_model import (
    normalized_s2u_capacity_utility,
    reference_s2u_max_capacity_mbps,
    reference_u2g_max_capacity_mbps,
)
from DDQN import DDQN
from Energy_model import EnergyConsumptionModel
from Packet_scheduler_v1 import PacketEngine
from experiment_config import (
    FORMAL_CHECKPOINT_EPISODE,
    FORMAL_EXPERIMENT_DEFAULTS,
    FORMAL_TRAINING_EPISODES,
    METHOD_REGISTRY,
    MOVEMENT_CHANNEL_TIMING_VERSION,
    PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION,
    PROPULSION_MODEL_ID,
    QOS_AGGREGATE_CONTRACT_VERSION,
    REFERENCE_COM_BANDWIDTH_HZ,
    SAFE_DDQN_QOS_TARGET_PROBABILITY,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
    UTILITY_NORMALIZATION_MODE,
    MethodSpec,
    comparison_method_configuration,
)
from object import SRTeam, UAV
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, checkpoint_episode_schedule


class FormalContractTest(unittest.TestCase):
    def test_formal_horizon_and_checkpoint_schedule(self):
        self.assertEqual(FORMAL_TRAINING_EPISODES, 1500)
        self.assertEqual(FORMAL_CHECKPOINT_EPISODE, 1500)
        self.assertEqual(FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"], 1500)
        self.assertEqual(checkpoint_episode_schedule(1500, 50), list(range(50, 1501, 50)))
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 18)

    def test_all_methods_publish_the_same_physical_contracts(self):
        shared_fields = (
            "utility_normalization_mode",
            "reference_com_bandwidth_hz",
            "reference_s2u_max_capacity_mbps",
            "packet_qos_contract_version",
            "packet_routing_causality_contract_version",
            "qos_aggregate_contract_version",
            "routing_reward_contract_version",
            "reference_u2u_max_capacity_mbps",
            "reference_u2g_max_capacity_mbps",
            "propulsion_model_id",
            "propulsion_parameters",
            "movement_channel_timing_version",
            "movement_substeps_per_interval",
            "movement_substep_seconds",
        )
        configurations = [
            comparison_method_configuration(MethodSpec.parse(method_id))
            for method_id in METHOD_REGISTRY
        ]
        expected = {field: configurations[0][field] for field in shared_fields}
        for configuration in configurations:
            self.assertEqual(
                {field: configuration[field] for field in shared_fields}, expected
            )
        self.assertEqual(expected["utility_normalization_mode"], UTILITY_NORMALIZATION_MODE)
        self.assertEqual(expected["propulsion_model_id"], PROPULSION_MODEL_ID)
        self.assertEqual(
            expected["packet_routing_causality_contract_version"],
            PACKET_ROUTING_CAUSALITY_CONTRACT_VERSION,
        )
        self.assertEqual(
            expected["qos_aggregate_contract_version"],
            QOS_AGGREGATE_CONTRACT_VERSION,
        )
        self.assertEqual(
            expected["movement_channel_timing_version"],
            MOVEMENT_CHANNEL_TIMING_VERSION,
        )


class SRLifecycleContractTest(unittest.TestCase):
    def test_derived_lifecycle_is_read_only_and_never_contradictory(self):
        team = SRTeam(4)
        self.assertIsNone(team.assigned_gt_id)
        self.assertFalse(team.arrived)
        self.assertFalse(team.is_moving)
        self.assertFalse(team.com_source_enabled)
        self.assertFalse(team.active)
        for name in ("is_moving", "com_source_enabled", "active"):
            with self.assertRaises(AttributeError):
                setattr(team, name, True)

        team.x, team.y = 0.0, 0.0
        team.assign_mission(7, (1.0, 0.0, 0.0), speed=1.0)
        self.assertTrue(team.is_moving)
        self.assertTrue(team.com_source_enabled)
        team.step_forward()
        self.assertTrue(team.arrived)
        self.assertEqual(team.assigned_gt_id, 7)
        self.assertFalse(team.is_moving)
        self.assertTrue(team.com_source_enabled)
        with self.assertRaises(RuntimeError):
            team.assign_mission(8, (2.0, 0.0, 0.0))
        self.assertNotIn("active", team.route_state())
        self.assertNotIn("is_moving", team.route_state())
        self.assertNotIn("com_source_enabled", team.route_state())
        team.reset_lifecycle()
        self.assertIsNone(team.assigned_gt_id)
        self.assertFalse(team.arrived)


class COMUtilityContractTest(unittest.TestCase):
    def test_fixed_reference_geometry_and_monotonicity(self):
        best = normalized_s2u_capacity_utility(
            (0.0, 0.0, 50.0), (0.0, 0.0, 0.0), REFERENCE_COM_BANDWIDTH_HZ
        )
        near = normalized_s2u_capacity_utility(
            (20.0, 0.0, 50.0), (0.0, 0.0, 0.0), REFERENCE_COM_BANDWIDTH_HZ
        )
        far = normalized_s2u_capacity_utility(
            (300.0, 0.0, 50.0), (0.0, 0.0, 0.0), REFERENCE_COM_BANDWIDTH_HZ
        )
        self.assertAlmostEqual(best, 1.0, places=12)
        self.assertGreaterEqual(near, far)
        for value in (best, near, far):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreater(
            reference_s2u_max_capacity_mbps(REFERENCE_COM_BANDWIDTH_HZ), 0.0
        )

    def test_link_utility_is_independent_of_candidates_methods_and_rate(self):
        geometry = ((80.0, 30.0, 75.0), (10.0, 20.0, 0.0))
        baseline = normalized_s2u_capacity_utility(
            *geometry, REFERENCE_COM_BANDWIDTH_HZ
        )
        # Candidate population and offered rate are deliberately absent from
        # the canonical helper; repeat under every registry method and sweep.
        for method_id in METHOD_REGISTRY:
            configuration = comparison_method_configuration(
                MethodSpec.parse(method_id)
            )
            self.assertEqual(
                configuration["reference_com_bandwidth_hz"],
                REFERENCE_COM_BANDWIDTH_HZ,
            )
            for _candidate_count in (1, 3, 20):
                for _packets_per_second in (5.0, 50.0, 500.0):
                    self.assertEqual(
                        normalized_s2u_capacity_utility(
                            *geometry, REFERENCE_COM_BANDWIDTH_HZ
                        ),
                        baseline,
                    )


class PropulsionContractTest(unittest.TestCase):
    EXPECTED = (
        (0.0, 0.0, 172.387633),
        (1.0, 0.0, 171.683583),
        (5.0, 0.0, 156.409545),
        (10.0, 0.0, 124.976903),
        (0.0, 2.0, 209.154725),
        (0.0, -2.0, 130.076177),
        (5.0, 2.0, 194.205684),
        (5.0, -2.0, 114.047358),
        (10.0, 2.0, 164.531179),
        (10.0, -2.0, 83.261969),
    )

    def setUp(self):
        self.model = EnergyConsumptionModel(10_000.0, N_u=10)

    def test_regression_table_rotation_and_envelope(self):
        for horizontal, vertical, expected in self.EXPECTED:
            power = self.model.propulsion_power((horizontal, 0.0, vertical))
            self.assertAlmostEqual(power, expected, places=5)
            rotated = self.model.propulsion_power((0.0, horizontal, vertical))
            self.assertAlmostEqual(rotated, power, places=10)
        for horizontal in np.linspace(0.0, 10.0, 21):
            for vertical in np.linspace(-2.0, 2.0, 17):
                power = self.model.propulsion_power((horizontal, 0.0, vertical))
                self.assertTrue(math.isfinite(power))
                self.assertGreaterEqual(power, 0.0)

    def test_quarter_substeps_equal_one_second_and_boundary_uses_actual_velocity(self):
        power = self.model.propulsion_power((5.0, 0.0, -1.0))
        self.assertAlmostEqual(4.0 * power * 0.25, power * 1.0, places=12)

        uav = UAV(0, 999.0, 500.0, 50.0)
        proposal = uav.propose_movement(
            10.0, 0.0, -2.0, step_time=1.0, env_width=1000.0, env_height=1000.0
        )
        energy = uav.apply_movement_proposal(
            proposal, energy_model=self.model, step_time=1.0
        )
        self.assertEqual(uav.get_position(), (1000.0, 500.0, 50.0))
        self.assertAlmostEqual(
            energy, self.model.propulsion_power((1.0, 0.0, 0.0)), places=10
        )


class QoSAndRoutingRewardContractTest(unittest.TestCase):
    def test_multiplier_uses_probability_and_skips_empty_episode(self):
        agent = DDQN(4, 2)
        initial = agent.lambda_cost
        self.assertEqual(agent.qos_target_probability, SAFE_DDQN_QOS_TARGET_PROBABILITY)
        self.assertAlmostEqual(agent.update_cost_multiplier(2, 20), initial)
        increased = agent.update_cost_multiplier(3, 20)
        self.assertAlmostEqual(increased, initial + 0.01 * 0.05)
        agent.lambda_cost = 0.0
        self.assertEqual(agent.update_cost_multiplier(0, 20), 0.0)
        updates = agent.cost_multiplier_update_count
        self.assertEqual(agent.update_cost_multiplier(0, 0), 0.0)
        self.assertEqual(agent.cost_multiplier_update_count, updates)

    def test_activated_sr_packet_is_immediately_qos_eligible_and_terminal_violation(self):
        engine = PacketEngine(2)
        engine.create_sr_packet(9, 256.0, 0.0)
        summary = engine.finalize_episode(1.0)
        self.assertEqual(summary["COM"]["source_generated_packets"], 1)
        self.assertEqual(summary["COM"]["eligible_packets"], 1)
        self.assertEqual(summary["COM"]["sr_admission_drop_packets"], 0)
        self.assertEqual(summary["COM"]["violation_packets"], 1)
        self.assertEqual(summary["COM"]["violation_probability"], 1.0)
        self.assertEqual(engine.replay_attributed_violation_cost_count, 0.0)
        self.assertEqual(engine.unattributed_pre_routing_violation_count, 1)

    def test_eligible_violation_counts_once_and_cost_is_one(self):
        engine = PacketEngine(2)
        packet = engine.create_packet(0, "FOV", 512.0, 0.0)
        packet["last_routing_sender"] = 0
        first = engine.finalize_episode(2.0)
        second = engine.finalize_episode(2.0)
        self.assertEqual(first["FOV"]["eligible_packets"], 1)
        self.assertEqual(first["FOV"]["violation_packets"], 1)
        self.assertEqual(second["FOV"]["violation_packets"], 1)
        self.assertEqual(sum(engine.pending_terminal_cost_by_sender.values()), 1.0)

    def test_reward_is_exact_capacity_delay_formula_and_has_no_distance_term(self):
        class Environment:
            GS_ID = 2

        env = Environment()
        engine = PacketEngine(2, task_deadlines_seconds={"FOV": 2.0, "COM": 1.0})
        packet = engine.create_packet(0, "FOV", 1_000_000.0, 0.0)
        capacity = 10.0
        maximum = reference_u2g_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ)
        expected = capacity / maximum - 0.5 * ((0.1 / 2.0) + 0.0)
        first = engine.routing_local_reward(
            env,
            0,
            env.GS_ID,
            capacity,
            pkt=packet,
            current_time=0.0,
        )
        self.assertAlmostEqual(first, expected, places=12)
        # The reward API reads no GS geometry; changing a diagnostic distance
        # cannot reintroduce progress-to-GS shaping.
        env.gs_distance_m = 10_000.0
        second = engine.routing_local_reward(
            env,
            0,
            env.GS_ID,
            capacity,
            pkt=packet,
            current_time=0.0,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            engine.routing_local_reward(
                env, 0, 0, 0.0, pkt=packet, current_time=0.0
            ),
            -0.5,
        )
        self.assertEqual(
            engine.routing_local_reward(
                env, 0, 0, 0.0, pkt=packet, current_time=1.0
            ),
            -0.5,
        )

    def test_partial_hop_reward_keeps_service_start_queue_wait_fixed(self):
        class Environment:
            GS_ID = 2

        env = Environment()
        engine = PacketEngine(2, task_deadlines_seconds={"FOV": 2.0, "COM": 1.0})
        packet = engine.create_packet(0, "COM", 1_000_000.0, 0.0)
        packet["hop_service_start_time"] = 0.25
        first = engine.routing_local_reward(
            env, 0, 0, 0.0, pkt=packet, current_time=0.5
        )
        second = engine.routing_local_reward(
            env, 0, 0, 0.0, pkt=packet, current_time=0.75
        )
        self.assertEqual(first, -0.5)
        self.assertEqual(second, first)

    def test_next_hol_preserves_historical_wait_and_clips_at_deadline(self):
        class Environment:
            GS_ID = 2

        env = Environment()
        engine = PacketEngine(2, task_deadlines_seconds={"FOV": 2.0, "COM": 1.0})
        first_packet = engine.create_packet(0, "COM", 100.0, 0.0)
        second_packet = engine.create_packet(0, "COM", 100.0, 0.0)
        engine.serve_active_links(
            env,
            {0: env.GS_ID},
            {(0, env.GS_ID): 0.0004},
            current_time=0.5,
        )
        self.assertTrue(first_packet["done"])
        self.assertIs(engine.get_hol_packet(0), second_packet)
        self.assertEqual(second_packet["queue_enter_time"], 0.0)
        self.assertEqual(
            engine.routing_local_reward(
                env,
                0,
                0,
                0.0,
                pkt=second_packet,
                current_time=0.75,
            ),
            -0.5,
        )
        self.assertEqual(
            engine.routing_local_reward(
                env,
                0,
                0,
                0.0,
                pkt=second_packet,
                current_time=2.0,
            ),
            -0.5,
        )


if __name__ == "__main__":
    unittest.main()
