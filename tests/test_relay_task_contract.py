import copy
import math
import unittest
from unittest import mock

import numpy as np

from HRL_task_aware import TrainingConfig, _interval_reward, train
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
from relay_contract import (initial_relay_count, plan_relays, pair_relay_slots, connectivity_graph,
    reverse_bfs, bounded_minimax_center, movement_bounds, virtual_positions,
    refresh_relay_targets, relay_potential, relay_snapshot, valid_in_air_backlog, COUNT_RULE)
from scenario_manifest import generate_manifest
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema
from experiment_config import MethodSpec
from utils_update_v2 import ReplayBufferJoint


def discover_all(env):
    with mock.patch.object(env, "is_visible", return_value=True):
        env.update_visited_grid(1)


class RelayRoutingCheckpointDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_reassignment_preserves_packet_queue_identity_and_ownership(self):
        engine = PacketEngine(16)
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
        self.assertEqual(len(routing_state_feature_names()), 143)
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
            (0.2, 0.3, 0.4, 0.1),
            (0.8, 0.7, 0.6, 0.9),
            False,
            config,
            task_potential_enabled=False,
        )
        self.assertEqual(reward, 0.0)

    def test_new_roi_boundary_masks_only_relay_shaping_at_gamma_point_99(self):
        config = TrainingConfig(total_episodes=1)
        current = (0.2, 0.3, 0.4, 0.5)
        following = (0.8, 0.7, 0.6, 0.9)
        reward = _interval_reward(
            0.0, 0.0, 0.0, 0.99, current, following, False, config,
            relay_shaping_enabled=False,
        )
        expected_other = (
            config.beta_search * (0.99 * following[0] - current[0])
            + config.beta_vs * (0.99 * following[1] - current[1])
            + config.beta_com * (0.99 * following[2] - current[2])
        )
        self.assertEqual(reward, expected_other)
        normal = _interval_reward(
            0.0, 0.0, 0.0, 0.99, current, following, False, config,
            relay_shaping_enabled=True,
        )
        self.assertEqual(
            normal - reward,
            config.beta_relay * (0.99 * following[3] - current[3]),
        )

    def test_joint_replay_persists_and_applies_relay_shaping_mask(self):
        replay = ReplayBufferJoint(1, 1, max_size=2)
        for enabled in (False, True):
            replay.add(
                [0.0], [0.0], [0.0], done=False,
                delivered_mbits=0.0, total_mobility_energy=0.0,
                phi_search_t=0.0, phi_search_t1=0.0,
                phi_vs_t=0.0, phi_vs_t1=0.0,
                phi_com_t=0.0, phi_com_t1=0.0,
                phi_relay_t=0.5, phi_relay_t1=0.9,
                relay_shaping_enabled=enabled,
            )
        np.testing.assert_array_equal(
            replay.relay_shaping_enabled[:2, 0], [False, True]
        )
        reward = replay._reward_numpy(
            np.array([0, 1]), current_lambda=0.0, gamma=0.99,
            beta_relay=1.0,
        ).ravel()
        self.assertEqual(reward[0], 0.0)
        self.assertAlmostEqual(reward[1], 0.99 * 0.9 - 0.5, places=6)
        disabled_all = replay._reward_numpy(
            np.array([0, 1]), current_lambda=0.0, gamma=0.99,
            task_potential_enabled=False,
        ).ravel()
        np.testing.assert_array_equal(disabled_all, [0.0, 0.0])

    def test_replay_relay_potential_continuity_and_task_reset_mask(self):
        captured = []
        original_add = ReplayBufferJoint.add

        def capture_add(replay, state, action, next_state, **kwargs):
            captured.append(
                {
                    "state": np.asarray(state).copy(),
                    "next_state": np.asarray(next_state).copy(),
                    **copy.deepcopy(kwargs),
                }
            )
            return original_add(replay, state, action, next_state, **kwargs)

        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=5,
            warmup_joint_transitions=10_000,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
        )
        manifest = generate_manifest(
            "train", 20260817, 1, num_gt=2
        )
        # Exercise Relay boundary timing independently of whether this short
        # manifest happens to place two ROI centers in narrow Search footprints.
        # Discovery fires at the first real Search boundary; sensing/packets and
        # assignment still use the production canonical geometry.
        original_assign = Simulator.assign_tasks
        def disconnected_assign(environment):
            if environment.count_found_targets():
                for uid, uav in environment.uav_dict.items():
                    uav.x_u, uav.y_u, uav.z_u = (100. if uid == 0 else 900.), 100., 100.
                environment.update_u2u_channels()
                environment.update_u2g_channels()
            return original_assign(environment)
        with (
            mock.patch.object(ReplayBufferJoint, "add", new=capture_add),
            mock.patch.object(Simulator, "assign_tasks", new=disconnected_assign),
            mock.patch.object(Simulator, "is_visible", return_value=True),
        ):
            result = train(
                config,
                scenario_manifest=manifest,
                method_spec=MethodSpec.parse("td3_ratio"),
            )

        self.assertEqual(len(captured), 5)
        self.assertFalse(captured[0]["relay_shaping_enabled"])
        self.assertTrue(all(
            record["relay_shaping_enabled"] for record in captured[1:]
        ))
        schema = movement_state_feature_schema()["features"]
        backlog_indices = [
            feature["index"]
            for feature in schema
            if feature["name"].endswith(".backlog")
        ]
        relay_flag_indices = [
            feature["index"]
            for feature in schema
            if feature["name"].endswith(".task_relay")
        ]
        self.assertTrue(any(np.any(record["state"][relay_flag_indices]) for record in captured))

        self.assertGreater(
            result["relay_diagnostics"]["episodes"][0]["assignment"][
                "assigned_relay_count"
            ],
            0,
        )

        potential_names = ("search", "vs", "com", "relay")
        for current, following in zip(captured, captured[1:]):
            np.testing.assert_array_equal(
                current["next_state"], following["state"]
            )
            for name in potential_names:
                self.assertEqual(
                    current[f"phi_{name}_t1"], following[f"phi_{name}_t"]
                )
        for name in potential_names:
            self.assertEqual(captured[-1][f"phi_{name}_t1"], 0.0)

        self.assertEqual(captured[0]["phi_relay_t"], 0.0)
        shaping_history = result["relay_diagnostics"]["episodes"][0][
            "assignment"
        ]["relay_shaping_history"]
        self.assertTrue(shaping_history[0][
            "relay_assignment_changed_at_boundary"
        ])
        self.assertFalse(shaping_history[0]["relay_shaping_enabled"])
        self.assertEqual(shaping_history[0]["applied_relay_shaping"], 0.0)
        for entry in shaping_history[1:]:
            self.assertTrue(entry["relay_shaping_enabled"])
            self.assertAlmostEqual(
                entry["applied_relay_shaping"],
                config.beta_relay
                * entry["raw_relay_potential_difference"],
                places=12,
            )

    def test_old_checkpoint_fails_before_loading(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 29)
        for schema in (28, 23):
            with self.assertRaisesRegex(RuntimeError, "Relay.*retrained"):
                _validate_checkpoint_schema({"checkpoint_schema_version": schema})

    def test_diagnostics_are_rng_observational_and_publish_forwarding_groups(self):
        state = copy.deepcopy(self.env.assignment_rng.bit_generator.state)
        self.env.assignment_metadata()
        self.assertEqual(state, self.env.assignment_rng.bit_generator.state)
        engine = PacketEngine(16, enable_packet_diagnostic_artifacts=True)
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
