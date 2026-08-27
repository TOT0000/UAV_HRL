import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from Channel_model import (
    CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
    FADING_BLOCKS_PER_ROUTING_SLOT,
    a2g_los_probability,
)
from HRL_task_aware import TrainingConfig, _run_routing_slot, train
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from centralized_movement import LOCAL_MOVEMENT_DIM, movement_mask_from_state
from experiment_config import METHOD_REGISTRY, MethodSpec, comparison_method_configuration
from rng_contract import NamedRNGStreams
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION, _validate_checkpoint_schema
from utils_update_v2 import ReplayBufferDiscrete, ReplayBufferJoint


class AllNlosDraws:
    def random(self, size):
        return np.ones(size, dtype=np.float64)


class RecordingRoutingPolicy:
    def __init__(self, receiver):
        self.receiver = int(receiver)
        self.observations = []

    def select_action(self, state, uav_id, mask=None, **kwargs):
        self.observations.append(
            (int(uav_id), np.asarray(state).copy(), np.asarray(mask).copy())
        )
        return self.receiver


def violation_stats():
    return {
        task: {
            "timely_delivered_packets": 0,
            "deadline_violated_packets": 0,
        }
        for task in ("FOV", "COM")
    }


def deferred_environment(seed=20260817):
    env = Simulator(10, rng_streams=NamedRNGStreams(seed))
    env.defer_initial_channel_boundary = True
    env.num_GT = 2
    env.reset_environment()
    return env


class EpisodeAndBoundaryLifecycleTest(unittest.TestCase):
    def test_interval_zero_samples_post_sr_movement_geometry_once(self):
        env = deferred_environment(101)
        self.assertIsNone(env.channel.movement_interval_index)
        self.assertIsNone(env.channel._gain_matrix)
        before_draws = env.channel.large_scale_draw_count

        sr = env.SR_teams[0]
        sr.assign_mission(0, (sr.x + 3.0, sr.y), speed=1.0)
        position_before = sr.get_position()
        env.prepare_initial_movement_interval()
        decision_position = sr.get_position()

        self.assertNotEqual(position_before, decision_position)
        self.assertEqual(env.channel.movement_interval_index, 0)
        self.assertEqual(
            env.channel.large_scale_draw_count - before_draws,
            env.num_UAV * (len(env.SR_teams) + 1),
        )
        expected_probability = a2g_los_probability(
            np.asarray([u.get_position() for u in env.UAVs])[None, :, :],
            np.asarray([s.get_position() for s in env.SR_teams])[:, None, :],
        )
        np.testing.assert_allclose(
            env.channel.s2u_los_probability, expected_probability
        )
        self.assertEqual(
            env.last_assignment_metadata["channel_movement_interval_index"], 0
        )
        self.assertIsNone(env.channel._gain_matrix)
        self.assertFalse(env.begin_channel_movement_interval(0))
        self.assertEqual(
            env.channel.large_scale_draw_count - before_draws,
            env.num_UAV * (len(env.SR_teams) + 1),
        )

    def test_boundary_assignment_observes_next_interval_state(self):
        env = deferred_environment(102)
        env.prepare_initial_movement_interval()
        env.channel.u2g_los_state[:] = True
        env.channel.s2u_los_state[:] = True
        env.channel.large_scale_rng = AllNlosDraws()
        env.need_reassign = True
        observations = []
        original = env.assign_tasks

        def record_assignment():
            observations.append(
                (
                    env.channel.movement_interval_index,
                    env.channel.u2g_los_state.copy(),
                    env.channel.s2u_los_state.copy(),
                )
            )
            return original()

        env.assign_tasks = record_assignment
        self.assertTrue(env.prepare_next_movement_interval(1))

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0][0], 1)
        self.assertFalse(observations[0][1].any())
        self.assertFalse(observations[0][2].any())
        self.assertEqual(
            env.last_assignment_metadata["channel_movement_interval_index"], 1
        )

    def test_draw_counts_are_one_large_state_per_interval_and_one_profile_per_slot(self):
        env = deferred_environment(103)
        large_before = env.channel.large_scale_draw_count
        profile_before = env.channel.profile_generation_count
        env.prepare_initial_movement_interval()
        draws_per_interval = env.num_UAV * (len(env.SR_teams) + 1)
        self.assertEqual(
            env.channel.large_scale_draw_count - large_before, draws_per_interval
        )

        for slot in range(4):
            self.assertTrue(env.prepare_channel_routing_slot(slot))
            self.assertFalse(env.prepare_channel_routing_slot(slot))
        self.assertEqual(
            env.channel.profile_generation_count - profile_before, 4
        )
        self.assertEqual(
            env.channel.large_scale_draw_count - large_before, draws_per_interval
        )

        env.need_reassign = False
        env.prepare_next_movement_interval(1)
        self.assertIsNone(env.channel._gain_matrix)
        self.assertEqual(
            env.channel.large_scale_draw_count - large_before,
            2 * draws_per_interval,
        )
        for slot in range(4, 8):
            env.prepare_channel_routing_slot(slot)
        self.assertEqual(
            env.channel.profile_generation_count - profile_before, 8
        )
        self.assertEqual(
            env.channel._gain_matrix.shape[-1], FADING_BLOCKS_PER_ROUTING_SLOT
        )


class MovementBoundaryReplayTest(unittest.TestCase):
    def test_transition_next_state_mask_and_phi_match_next_policy_input(self):
        records = []
        boundary_states = []
        original_initial = Simulator.prepare_initial_movement_interval
        original_next = Simulator.prepare_next_movement_interval

        def force_initial_los(simulator):
            original_initial(simulator)
            simulator.channel.u2g_los_state[:] = True
            simulator.channel.s2u_los_state[:] = True
            simulator.update_u2g_channels()
            sr = simulator.SR_teams[0]
            simulator.multi_tasks[1] = [
                {
                    "task_type": "COM",
                    "target_id": 0,
                    "target_obj_id": 0,
                    "target_pos": sr.get_position(),
                }
            ]
            simulator.need_reassign = False
            boundary_states.append(simulator.channel.s2u_los_state.copy())

        def force_next_nlos(simulator, interval_index):
            simulator.channel.large_scale_rng = AllNlosDraws()
            result = original_next(simulator, interval_index)
            boundary_states.append(simulator.channel.s2u_los_state.copy())
            return result

        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=2,
            warmup_joint_transitions=10_000,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=104,
        )
        original_add = ReplayBufferJoint.add

        def record_transition(
            replay,
            state,
            action,
            next_state,
            **kwargs,
        ):
            records.append(
                {
                    "state": np.asarray(state).copy(),
                    "next_state": np.asarray(next_state).copy(),
                    "phi_com_t": float(kwargs["phi_com_t"]),
                    "phi_com_t1": float(kwargs["phi_com_t1"]),
                }
            )
            return original_add(replay, state, action, next_state, **kwargs)

        with mock.patch.object(
            Simulator, "prepare_initial_movement_interval", new=force_initial_los
        ), mock.patch.object(
            Simulator, "prepare_next_movement_interval", new=force_next_nlos
        ), mock.patch.object(
            ReplayBufferJoint, "add", new=record_transition
        ):
            train(config)

        self.assertEqual(len(records), 2)
        self.assertTrue(boundary_states[0].all())
        self.assertFalse(boundary_states[1].any())
        np.testing.assert_array_equal(records[0]["next_state"], records[1]["state"])
        np.testing.assert_array_equal(
            movement_mask_from_state(records[0]["next_state"]),
            movement_mask_from_state(records[1]["state"]),
        )
        self.assertEqual(records[0]["phi_com_t1"], records[1]["phi_com_t"])
        com_feature = LOCAL_MOVEMENT_DIM + 16
        self.assertNotEqual(
            float(records[0]["state"][com_feature]),
            float(records[0]["next_state"][com_feature]),
        )


class RoutingBoundaryReplayTest(unittest.TestCase):
    def _environment_and_engine(self):
        env = Simulator(10, rng_streams=NamedRNGStreams(105))
        env.num_GT = 2
        env.reset_environment()
        env.uav_dict[0].x_u = 100.0
        env.uav_dict[0].y_u = 0.0
        env.uav_dict[0].z_u = 100.0
        env.channel.u2g_los_state[:] = True
        env.channel.s2u_los_state[:] = True
        env.update_u2g_channels()
        env.need_reassign = False
        return env, PacketEngine(10)

    def test_pending_transition_uses_next_actual_observation_and_lock_mask(self):
        env, engine = self._environment_and_engine()
        packet = engine.create_packet(0, "COM", 1e12, 0.75)
        packet["deadline_abs"] = 10.0
        replay = ReplayBufferDiscrete(90, 11, max_size=16, n_step=1)
        pending = {}
        policy = RecordingRoutingPolicy(env.GS_ID)
        stats = violation_stats()

        _run_routing_slot(
            env,
            engine,
            policy,
            replay,
            None,
            current_time=0.75,
            done=False,
            delay_bound_steps=20,
            violation_stats=stats,
            epsilon=0.0,
            traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
            pending_routing_transitions=pending,
        )
        self.assertEqual(replay.size, 0)
        self.assertEqual(set(pending), {0})

        env.channel.large_scale_rng = AllNlosDraws()
        env.prepare_next_movement_interval(1)
        _run_routing_slot(
            env,
            engine,
            policy,
            replay,
            None,
            current_time=1.0,
            done=True,
            delay_bound_steps=20,
            violation_stats=stats,
            epsilon=0.0,
            traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
            pending_routing_transitions=pending,
        )

        self.assertEqual(replay.size, 2)
        self.assertEqual(replay.total_added, 2)
        self.assertEqual(pending, {})
        self.assertEqual(replay.not_done[0, 0], 1.0)
        np.testing.assert_array_equal(replay.next_state[0], policy.observations[1][1])
        self.assertFalse(np.array_equal(policy.observations[0][1], policy.observations[1][1]))
        self.assertEqual(
            set(np.flatnonzero(policy.observations[1][2])),
            {0, env.GS_ID},
        )

    def test_boundary_expiration_cost_attaches_to_pending_transition_before_truncation(self):
        env, engine = self._environment_and_engine()
        packet = engine.create_packet(0, "COM", 1e12, 0.75)
        packet["deadline_abs"] = 1.0
        replay = ReplayBufferDiscrete(90, 11, max_size=16, n_step=1)
        pending = {}
        policy = RecordingRoutingPolicy(env.GS_ID)
        stats = violation_stats()

        for current_time, done in ((0.75, False), (1.0, True)):
            if current_time == 1.0:
                env.channel.large_scale_rng = AllNlosDraws()
                env.prepare_next_movement_interval(1)
            _run_routing_slot(
                env,
                engine,
                policy,
                replay,
                None,
                current_time=current_time,
                done=done,
                delay_bound_steps=20,
                violation_stats=stats,
                epsilon=0.0,
                traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
                pending_routing_transitions=pending,
            )

        self.assertEqual(replay.size, 1)
        self.assertEqual(pending, {})
        self.assertEqual(replay.not_done[0, 0], 0.0)
        self.assertEqual(replay.cost[0, 0], 1.0)
        self.assertEqual(engine.replay_attributed_violation_cost_count, 1.0)
        self.assertEqual(stats["COM"]["deadline_violated_packets"], 1)


class CompatibilityAndRegistryTest(unittest.TestCase):
    @staticmethod
    def _assert_nested_exact(test_case, expected, actual):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
            return
        if isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                CompatibilityAndRegistryTest._assert_nested_exact(
                    test_case, expected[key], actual[key]
                )
            return
        if isinstance(expected, (list, tuple)):
            test_case.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                CompatibilityAndRegistryTest._assert_nested_exact(
                    test_case, expected_item, actual_item
                )
            return
        test_case.assertEqual(expected, actual)

    def test_full_resume_matches_uninterrupted_boundary_gain_and_transition(self):
        original_add = ReplayBufferJoint.add
        original_initial = Simulator.prepare_initial_movement_interval
        original_slot = Simulator.prepare_channel_routing_slot

        def run_with_capture(config):
            captured = {"boundaries": [], "profiles": [], "transitions": []}

            def capture_initial(simulator):
                result = original_initial(simulator)
                captured["boundaries"].append(simulator.channel_state_dict())
                return result

            def capture_slot(simulator, routing_slot_index):
                generated = original_slot(simulator, routing_slot_index)
                if generated:
                    captured["profiles"].append(
                        simulator.channel_state_dict()
                    )
                return generated

            def capture_transition(
                replay,
                state,
                action,
                next_state,
                **kwargs,
            ):
                captured["transitions"].append(
                    {
                        "state": np.asarray(state).copy(),
                        "action": np.asarray(action).copy(),
                        "next_state": np.asarray(next_state).copy(),
                        **{
                            key: (
                                np.asarray(value).copy()
                                if isinstance(value, np.ndarray)
                                else copy.deepcopy(value)
                            )
                            for key, value in kwargs.items()
                        },
                    }
                )
                return original_add(replay, state, action, next_state, **kwargs)

            with mock.patch.object(
                Simulator,
                "prepare_initial_movement_interval",
                new=capture_initial,
            ), mock.patch.object(
                Simulator,
                "prepare_channel_routing_slot",
                new=capture_slot,
            ), mock.patch.object(
                ReplayBufferJoint,
                "add",
                new=capture_transition,
            ):
                result = train(config)
            return result, captured

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uninterrupted_root = temp_path / "uninterrupted"
            base = dict(
                total_episodes=2,
                mode="custom",
                episode_seconds=1,
                warmup_joint_transitions=10_000,
                batch_size=1,
                model_checkpoint_every=1,
                full_resume_every=1,
                full_resume_keep_last=2,
                enable_model_checkpoints=False,
                enable_full_resume=True,
                enable_plots=False,
                enable_csv=False,
                random_seed=106,
            )
            uninterrupted, uninterrupted_capture = run_with_capture(
                TrainingConfig(
                    **base,
                    checkpoint_root=str(uninterrupted_root),
                )
            )
            resumed, resumed_capture = run_with_capture(
                TrainingConfig(
                    **base,
                    checkpoint_root=str(temp_path / "resumed"),
                    resume_dir=str(uninterrupted_root / "full" / "ep_0001"),
                )
            )

        self.assertEqual(len(uninterrupted_capture["boundaries"]), 2)
        self.assertEqual(len(uninterrupted_capture["profiles"]), 8)
        self.assertEqual(len(uninterrupted_capture["transitions"]), 2)
        self.assertEqual(len(resumed_capture["boundaries"]), 1)
        self.assertEqual(len(resumed_capture["profiles"]), 4)
        self.assertEqual(len(resumed_capture["transitions"]), 1)
        self._assert_nested_exact(
            self,
            uninterrupted_capture["boundaries"][1],
            resumed_capture["boundaries"][0],
        )
        self._assert_nested_exact(
            self,
            uninterrupted_capture["profiles"][4],
            resumed_capture["profiles"][0],
        )
        self._assert_nested_exact(
            self,
            uninterrupted_capture["transitions"][1],
            resumed_capture["transitions"][0],
        )
        self._assert_nested_exact(
            self,
            uninterrupted["channel_lifecycle_state"],
            resumed["channel_lifecycle_state"],
        )

    def test_old_boundary_checkpoint_is_rejected_before_restore(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 16)
        with self.assertRaisesRegex(RuntimeError, "must be retrained"):
            _validate_checkpoint_schema({"checkpoint_schema_version": 12})

    def test_all_methods_publish_boundary_aligned_channel_contract(self):
        for method_id in METHOD_REGISTRY:
            configuration = comparison_method_configuration(MethodSpec.parse(method_id))
            with self.subTest(method=method_id):
                self.assertEqual(
                    configuration["channel_environment_contract_version"],
                    CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
                )
                self.assertIn(
                    "boundary-aligned-next-state",
                    configuration["movement_replay_contract_version"],
                )
                self.assertIn(
                    "causality-credit-pending",
                    configuration["packet_routing_causality_contract_version"],
                )


if __name__ == "__main__":
    unittest.main()
