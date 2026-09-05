import math
import unittest
from types import SimpleNamespace

import numpy as np

from Packet_scheduler_v1 import (
    PacketEngine,
    action_wise_gs_progress,
    canonical_routing_gs_progress,
)
from communication_contract import normalized_gs_progress
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    MAX_3D_COMMUNICATION_DISTANCE_M,
    METHOD_REGISTRY,
    NUM_UAV,
    PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS,
    PRODUCTION_TASK_DEADLINE_SECONDS,
    ROUTING_ACTION_DIM,
    ROUTING_ACTION_FEATURE_GROUPS,
    ROUTING_REWARD_ALPHA_CAPACITY,
    ROUTING_REWARD_ALPHA_DELAY,
    ROUTING_REWARD_ALPHA_GS_PROGRESS,
    ROUTING_REWARD_CONTRACT_VERSION,
    ROUTING_STATE_DIM,
    ROUTING_STATE_SCHEMA_VERSION,
    MethodSpec,
    comparison_method_configuration,
)
from observation_strategy import apply_observation_strategy, routing_state_feature_names
from paper_evaluation import (
    DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS,
    DEADLINE_SWEEP_SECONDS,
    validate_production_deadlines,
)
from Simulator import Simulator
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema


def _point(position):
    return SimpleNamespace(get_position=lambda: tuple(position))


class GroundStationProgressHelperTest(unittest.TestCase):
    def test_three_dimensional_progress_direction_wait_direct_and_clipping(self):
        gs = (0.0, 0.0, 0.0)
        sender = (0.0, 0.0, 300.0)
        self.assertAlmostEqual(
            normalized_gs_progress(sender, (0.0, 0.0, 100.0), gs),
            0.5,
        )
        self.assertAlmostEqual(
            normalized_gs_progress(sender, (0.0, 0.0, 500.0), gs),
            -0.5,
        )
        self.assertEqual(normalized_gs_progress(sender, sender, gs), 0.0)
        self.assertEqual(
            normalized_gs_progress(sender, (1.0, 2.0, 3.0), gs, is_wait=True),
            0.0,
        )
        self.assertAlmostEqual(normalized_gs_progress(sender, None, gs), 0.75)
        self.assertEqual(
            normalized_gs_progress((0.0, 0.0, 1000.0), None, gs), 1.0
        )
        self.assertEqual(
            normalized_gs_progress(
                (0.0, 0.0, 0.0), (0.0, 0.0, 1000.0), gs
            ),
            -1.0,
        )
        self.assertEqual(MAX_3D_COMMUNICATION_DISTANCE_M, 400.0)

    def test_action_order_is_all_uavs_then_gs_and_ignores_link_legality(self):
        env = SimpleNamespace(
            num_UAV=3,
            GS_ID=3,
            GS_pos=(0.0, 0.0, 0.0),
            uav_dict={
                0: _point((300.0, 0.0, 0.0)),
                1: _point((100.0, 0.0, 0.0)),
                2: _point((500.0, 0.0, 0.0)),
            },
        )
        progress = action_wise_gs_progress(env, 0)
        np.testing.assert_allclose(progress, [0.0, 0.5, -0.5, 0.75])
        self.assertEqual(canonical_routing_gs_progress(env, 0, 0), 0.0)


class RoutingRewardAndStateContractTest(unittest.TestCase):
    def test_reward_prefers_equal_quality_link_that_is_closer_to_gs(self):
        env = SimpleNamespace(
            num_UAV=3,
            GS_ID=3,
            GS_pos=(0.0, 0.0, 0.0),
            uav_dict={
                0: _point((300.0, 0.0, 0.0)),
                1: _point((100.0, 0.0, 0.0)),
                2: _point((500.0, 0.0, 0.0)),
            },
        )
        engine = PacketEngine(3)
        packet = engine.create_packet(0, "COM", 1000.0, 0.0)
        closer = engine.routing_local_reward(
            env, 0, 1, 10.0, pkt=packet, current_time=0.0
        )
        farther = engine.routing_local_reward(
            env, 0, 2, 10.0, pkt=packet, current_time=0.0
        )
        self.assertGreater(closer, farther)
        self.assertAlmostEqual(closer - farther, 2.0, places=12)
        self.assertEqual(ROUTING_REWARD_ALPHA_CAPACITY, 1.0)
        self.assertEqual(ROUTING_REWARD_ALPHA_DELAY, 0.5)
        self.assertEqual(ROUTING_REWARD_ALPHA_GS_PROGRESS, 2.0)
        self.assertTrue(ROUTING_REWARD_CONTRACT_VERSION.endswith("v7"))

    def test_state_is_101d_and_progress_block_survives_task_masking(self):
        env = Simulator(num_UAV=NUM_UAV)
        env.num_GT = 2
        env.reset_environment()
        env.GS_pos = np.asarray((0.0, 0.0, 0.0), dtype=float)
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
            400.0,
            0.0,
            0.0,
        )
        env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (
            200.0,
            0.0,
            0.0,
        )
        engine = PacketEngine(NUM_UAV)
        engine.create_packet(0, "COM", 1000.0, 0.0)
        supplied_mask = np.zeros(ROUTING_ACTION_DIM, dtype=bool)
        supplied_mask[0] = True
        state = engine.get_state_ta(env, 0, action_mask=supplied_mask)
        names = routing_state_feature_names()
        progress_indices = [
            index
            for index, name in enumerate(names)
            if name.startswith("gs_progress_norm[")
        ]
        progress = state[progress_indices]

        self.assertEqual(state.shape, (ROUTING_STATE_DIM,))
        self.assertEqual(ROUTING_STATE_DIM, 7 * NUM_UAV + 31)
        self.assertEqual(len(progress_indices), ROUTING_ACTION_DIM)
        self.assertEqual(progress[0], 0.0)
        self.assertAlmostEqual(progress[1], 0.5)
        self.assertEqual(progress[env.GS_ID], 1.0)
        self.assertEqual(state[names.index("effective_action_mask[1]")], 0.0)

        masked = apply_observation_strategy(state, "masked", "routing")
        np.testing.assert_array_equal(masked[progress_indices], progress)

    def test_every_method_publishes_same_state_and_reward_environment_contract(self):
        configurations = [
            comparison_method_configuration(MethodSpec.parse(method_id))
            for method_id in METHOD_REGISTRY
        ]
        for configuration in configurations:
            self.assertEqual(configuration["routing_state_dim"], ROUTING_STATE_DIM)
            self.assertEqual(configuration["routing_action_dim"], ROUTING_ACTION_DIM)
            self.assertEqual(
                tuple(configuration["routing_action_feature_groups"]),
                ROUTING_ACTION_FEATURE_GROUPS,
            )
            self.assertEqual(
                configuration["routing_state_schema_version"],
                ROUTING_STATE_SCHEMA_VERSION,
            )
            self.assertEqual(
                configuration["routing_reward_contract_version"],
                ROUTING_REWARD_CONTRACT_VERSION,
            )
        random_methods = [
            MethodSpec.parse(method_id)
            for method_id in METHOD_REGISTRY
            if MethodSpec.parse(method_id).routing == "random"
        ]
        self.assertTrue(random_methods)
        self.assertTrue(all(not method.learns_routing for method in random_methods))


class QosAndCheckpointContractTest(unittest.TestCase):
    def test_production_deadlines_and_derived_cutoff_preserve_full_window(self):
        self.assertEqual(validate_production_deadlines(), {"FOV": 2.5, "COM": 2.0})
        self.assertEqual(PRODUCTION_TASK_DEADLINE_SECONDS, {"FOV": 2.5, "COM": 2.0})
        self.assertEqual(PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS, 57.5)
        self.assertTrue(
            math.isclose(
                PRODUCTION_PACKET_INJECTION_CUTOFF_SECONDS
                + max(PRODUCTION_TASK_DEADLINE_SECONDS.values()),
                FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"],
            )
        )
        self.assertEqual(DEADLINE_SWEEP_SECONDS, (0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        self.assertEqual(DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS, 57.0)

    def test_schema_21_and_90d_current_schema_are_rejected(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 24)
        with self.assertRaisesRegex(RuntimeError, "must be retrained"):
            _validate_checkpoint_schema({"checkpoint_schema_version": 21})
        with self.assertRaisesRegex(RuntimeError, "GS-progress"):
            _validate_checkpoint_schema(
                {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "routing_state_dim": 90,
                }
            )


if __name__ == "__main__":
    unittest.main()
