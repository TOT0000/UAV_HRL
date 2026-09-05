import copy
import random
import unittest

import numpy as np

from centralized_movement import fov_task_metrics
from experiment_config import FOV_EMA_LIFECYCLE_VERSION, MethodSpec
from fov_ema_lifecycle import validate_fov_ema_state
from HRL_task_aware import _mark_search_observations
from observation_strategy import (
    apply_observation_strategy,
    routing_state_feature_names,
)
from Packet_scheduler_v1 import PacketEngine
from scenario_manifest import generate_manifest
from Simulator import FovCoverageTransition, Simulator
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    FULL_RESUME_LOGGING_SCHEMA_VERSION,
    _validate_full_resume_logging_state,
)


class FovOverlapEmaLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = generate_manifest("test", 7319, 1).episodes[0]

    def _environment(self):
        env = Simulator(num_UAV=16)
        env.apply_scenario_entry(self.scenario)
        env._search_phase_over = False
        env.multi_tasks = {uav_id: [] for uav_id in range(env.num_UAV)}
        env.multi_tasks[0] = [{"task_type": "Search"}]
        for target in env.gts:
            target.is_found = True
        env.update_source_uavs()
        env.update_u2u_channels()
        env.update_u2g_channels()
        return env

    @staticmethod
    def _apply_production_transition(env, engine, marker):
        transitions = _mark_search_observations(env)
        updated = False
        if transitions:
            updated = engine.process_fov_transitions(
                env,
                marker,
                footprint_transitions=transitions,
            )
        return transitions, updated

    @staticmethod
    def _routing_state(env, engine):
        return engine.get_state_ta(
            env,
            0,
            backlog_bits=engine.backlog_bits,
            action_mask=env.get_routing_action_mask(0),
        )

    @staticmethod
    def _overlap(previous, current):
        bx_min, bx_max, by_min, by_max = current
        lbx_min, lbx_max, lby_min, lby_max = previous
        ix_min = max(bx_min, lbx_min)
        ix_max = min(bx_max, lbx_max)
        iy_min = max(by_min, lby_min)
        iy_max = min(by_max, lby_max)
        intersection = (
            (ix_max - ix_min + 1) * (iy_max - iy_min + 1)
            if ix_max >= ix_min and iy_max >= iy_min
            else 0
        )
        cells = (bx_max - bx_min + 1) * (by_max - by_min + 1)
        return intersection / float(cells)

    def test_production_order_preserves_disjoint_previous_footprint(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        uav = env.uav_dict[0]
        previous = (0, 4, 0, 4)
        uav.last_box_idx = previous
        uav.x_u = 900.0
        uav.y_u = 900.0

        transitions = _mark_search_observations(env)
        transition = next(item for item in transitions if item.uav_id == 0)
        self.assertTrue(transition.map_changed)
        self.assertEqual(transition.previous_footprint, previous)
        self.assertEqual(uav.last_box_idx, previous)

        self.assertTrue(
            engine.update_fov_ema(
                env,
                "episode=0,interval=0",
                footprint_transitions=transitions,
            )
        )
        self.assertEqual(engine.fov_ema[0]["overlap"], 0.0)
        self.assertEqual(uav.last_box_idx, transition.current_footprint)
        self.assertEqual(
            engine.fov_previous_footprints[0], transition.current_footprint
        )

    def test_overlap_geometry_and_first_observation_ema_values(self):
        alpha = 0.7
        cases = []

        env = self._environment()
        current = env.fov_footprint_indices(0)
        cases.append(("missing", env, None, current, 0.0))

        env = self._environment()
        current = env.fov_footprint_indices(0)
        cases.append(("identical", env, current, current, 1.0))

        env = self._environment()
        env.uav_dict[0].x_u = 900.0
        env.uav_dict[0].y_u = 900.0
        current = env.fov_footprint_indices(0)
        cases.append(("disjoint", env, (0, 4, 0, 4), current, 0.0))

        env = self._environment()
        env.uav_dict[0].x_u = 500.0
        env.uav_dict[0].y_u = 500.0
        current = env.fov_footprint_indices(0)
        bx_min, bx_max, by_min, by_max = current
        width = bx_max - bx_min + 1
        height = by_max - by_min + 1
        shift = max(1, width // 3)
        previous = (bx_min + shift, bx_max + shift, by_min, by_max)
        partial_overlap = ((width - shift) * height) / float(width * height)
        cases.append(("partial", env, previous, current, partial_overlap))

        for label, env, previous, current, expected_sample in cases:
            with self.subTest(label=label):
                engine = PacketEngine(num_uav=16)
                transition = FovCoverageTransition(
                    uav_id=0,
                    previous_footprint=previous,
                    current_footprint=current,
                    map_changed=True,
                )
                engine.update_fov_ema(
                    env,
                    f"geometry={label}",
                    footprint_transitions=(transition,),
                )
                self.assertAlmostEqual(
                    engine.fov_ema[0]["overlap"],
                    alpha * 0.0 + (1.0 - alpha) * expected_sample,
                )
                self.assertEqual(env.uav_dict[0].last_box_idx, current)

    def test_fov_assignment_geometry_is_unchanged_by_ema_transition(self):
        env = self._environment()
        target = env.gts[0]
        uav = env.uav_dict[0]
        uav.x_u, uav.y_u, uav.z_u = target.x, target.y, 100.0
        task = {
            "task_type": "FOV",
            "target_obj_id": target.id,
            "target_pos": [target.x, target.y, target.z],
        }
        before = fov_task_metrics(env, 0, task)
        engine = PacketEngine(num_uav=16)
        transitions, updated = self._apply_production_transition(
            env, engine, "utility-invariance"
        )
        self.assertTrue(transitions)
        self.assertTrue(updated)
        after = fov_task_metrics(env, 0, task)
        np.testing.assert_allclose(after[:2], before[:2])
        self.assertEqual(after[2], before[2])
        self.assertTrue(before[2])
        self.assertGreater(before[0], 0.0)

    def test_getters_are_pure_and_masking_preserves_the_real_ema(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        ema_indices = [
            routing_state_feature_names().index(name)
            for name in (
                "coverage_overlap_ema",
                "coverage_unvisited_ema",
                "coverage_frontier_ema",
            )
        ]
        ema_before = copy.deepcopy(engine.fov_ema_state())
        footprint_before = env.uav_dict[0].last_box_idx
        bitmap_before = env.visited_bitmap.copy()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()

        physical_first = self._routing_state(env, engine)
        physical_second = self._routing_state(env, engine)
        full = apply_observation_strategy(physical_first, "full", "routing")
        masked = apply_observation_strategy(physical_first, "masked", "routing")
        masked_again = apply_observation_strategy(
            self._routing_state(env, engine), "masked", "routing"
        )

        np.testing.assert_array_equal(physical_first, physical_second)
        np.testing.assert_array_equal(masked, masked_again)
        np.testing.assert_array_equal(full[ema_indices], masked[ema_indices])
        self.assertEqual(engine.fov_ema_state(), ema_before)
        self.assertEqual(env.uav_dict[0].last_box_idx, footprint_before)
        np.testing.assert_array_equal(env.visited_bitmap, bitmap_before)
        self.assertEqual(random.getstate(), python_rng_before)
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
        np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
        self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])

    def test_one_update_per_real_map_transition_and_exact_smoothing(self):
        env = self._environment()
        uav = env.uav_dict[0]
        uav.x_u, uav.y_u = 100.0, 100.0
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")

        _, updated = self._apply_production_transition(env, engine, "transition=1")
        self.assertTrue(updated)
        self.assertEqual(engine.fov_ema_update_count, 2)
        self.assertAlmostEqual(engine.fov_ema[0]["overlap"], 0.3)

        state_after_first = copy.deepcopy(engine.fov_ema_state())
        uav.x_u += 10.0
        footprint_b = env.fov_footprint_indices(0)
        bx_min, bx_max, by_min, by_max = footprint_b
        env.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True
        transitions, updated = self._apply_production_transition(
            env, engine, "transition=1-no-map-change"
        )
        self.assertTrue(transitions)
        self.assertFalse(
            any(
                item.map_changed
                for item in transitions
                if item.coverage_contributor
            )
        )
        self.assertFalse(updated)
        state_after_no_map_change = engine.fov_ema_state()
        self.assertEqual(
            state_after_no_map_change["values"], state_after_first["values"]
        )
        self.assertEqual(
            state_after_no_map_change["update_count"],
            state_after_first["update_count"],
        )
        self.assertEqual(
            state_after_no_map_change["transition_marker"],
            state_after_first["transition_marker"],
        )
        self.assertEqual(
            state_after_no_map_change["previous_footprints"]["0"],
            list(footprint_b),
        )
        self.assertEqual(uav.last_box_idx, footprint_b)
        self.assertNotEqual(
            state_after_no_map_change["footprint_transition_marker"],
            state_after_first["footprint_transition_marker"],
        )
        self.assertFalse(
            engine.process_fov_transitions(
                env,
                "transition=1-no-map-change",
                footprint_transitions=transitions,
            )
        )
        self.assertEqual(engine.fov_ema_state(), state_after_no_map_change)

        uav.x_u, uav.y_u = 900.0, 900.0
        _, updated = self._apply_production_transition(env, engine, "transition=2")
        self.assertTrue(updated)
        self.assertEqual(engine.fov_ema_update_count, 3)
        self.assertAlmostEqual(engine.fov_ema[0]["overlap"], 0.7 * 0.3)

    def test_mixed_multi_uav_interval_commits_all_search_footprints_once(self):
        env = self._environment()
        env.multi_tasks[1] = [{"task_type": "Search"}]
        uav0 = env.uav_dict[0]
        uav1 = env.uav_dict[1]
        uav0.x_u, uav0.y_u = 100.0, 100.0
        uav1.x_u, uav1.y_u = 800.0, 800.0
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        uav1.x_u -= 20.0
        uav1_current = env.fov_footprint_indices(1)
        bx_min, bx_max, by_min, by_max = uav1_current
        env.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True

        transitions, updated = self._apply_production_transition(
            env, engine, "mixed-map-change"
        )
        by_uav = {transition.uav_id: transition for transition in transitions}
        self.assertTrue(by_uav[0].map_changed)
        self.assertFalse(by_uav[1].map_changed)
        self.assertTrue(updated)
        self.assertEqual(engine.fov_ema_update_count, 2)
        self.assertEqual(
            engine.fov_previous_footprints[0], by_uav[0].current_footprint
        )
        self.assertEqual(
            engine.fov_previous_footprints[1], by_uav[1].current_footprint
        )
        self.assertEqual(uav0.last_box_idx, by_uav[0].current_footprint)
        self.assertEqual(uav1.last_box_idx, by_uav[1].current_footprint)

    def test_no_map_change_footprint_is_used_by_the_next_ema_sample(self):
        env = self._environment()
        uav = env.uav_dict[0]
        uav.x_u, uav.y_u = 300.0, 500.0
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        footprint_a = env.fov_footprint_indices(0)
        width = footprint_a[1] - footprint_a[0] + 1
        shift_cells = max(2, width // 4)

        uav.x_u += shift_cells * env.bit_resolution
        footprint_b = env.fov_footprint_indices(0)
        bx_min, bx_max, by_min, by_max = footprint_b
        env.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True
        transitions_b, updated_b = self._apply_production_transition(
            env, engine, "transition=B-no-map-change"
        )
        self.assertFalse(
            any(
                item.map_changed
                for item in transitions_b
                if item.coverage_contributor
            )
        )
        self.assertFalse(updated_b)

        uav.x_u += shift_cells * env.bit_resolution
        footprint_c = env.fov_footprint_indices(0)
        expected_bc = self._overlap(footprint_b, footprint_c)
        stale_ac = self._overlap(footprint_a, footprint_c)
        self.assertNotAlmostEqual(expected_bc, stale_ac)
        overlap_before = engine.fov_ema[0]["overlap"]
        transitions_c, updated_c = self._apply_production_transition(
            env, engine, "transition=C-map-change"
        )
        self.assertTrue(
            any(
                item.map_changed
                for item in transitions_c
                if item.coverage_contributor
            )
        )
        self.assertTrue(updated_c)
        alpha = engine.norm_cfg["ema_alpha"]
        actual_sample = (engine.fov_ema[0]["overlap"] - alpha * overlap_before) / (
            1.0 - alpha
        )
        self.assertAlmostEqual(actual_sample, expected_bc)

    def test_checkpoint_resume_matches_uninterrupted_next_transition(self):
        env_a = self._environment()
        env_b = self._environment()
        engine_a = PacketEngine(num_uav=16)
        engine_b = PacketEngine(num_uav=16)
        for env, engine in ((env_a, engine_a), (env_b, engine_b)):
            uav = env.uav_dict[0]
            uav.x_u, uav.y_u = 300.0, 500.0
            engine.update_fov_ema(env, "episode=0,map_reset")
            footprint_a = env.fov_footprint_indices(0)
            width = footprint_a[1] - footprint_a[0] + 1
            shift_cells = max(2, width // 4)
            uav.x_u += shift_cells * env.bit_resolution
            footprint_b = env.fov_footprint_indices(0)
            bx_min, bx_max, by_min, by_max = footprint_b
            env.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True
            transitions, updated = self._apply_production_transition(
                env, engine, "transition=2-no-map-change"
            )
            self.assertFalse(updated)
            self.assertFalse(
                any(
                    item.map_changed
                    for item in transitions
                    if item.coverage_contributor
                )
            )
            self.assertEqual(
                engine.fov_previous_footprints[0], footprint_b
            )

        saved = engine_b.fov_ema_state()
        expected_previous = saved["previous_footprints"]["0"]
        env_b.uav_dict[0].last_box_idx = None
        restored = PacketEngine(num_uav=16)
        restored.load_fov_ema_state(saved, env=env_b)
        self.assertEqual(
            list(env_b.uav_dict[0].last_box_idx), expected_previous
        )

        before_a = engine_a.fov_ema[0]["overlap"]
        before_b = restored.fov_ema[0]["overlap"]
        env_a.uav_dict[0].x_u += shift_cells * env_a.bit_resolution
        env_b.uav_dict[0].x_u += shift_cells * env_b.bit_resolution
        self._apply_production_transition(env_a, engine_a, "transition=3")
        self._apply_production_transition(env_b, restored, "transition=3")

        alpha = engine_a.norm_cfg["ema_alpha"]
        sample_a = (engine_a.fov_ema[0]["overlap"] - alpha * before_a) / (
            1.0 - alpha
        )
        sample_b = (restored.fov_ema[0]["overlap"] - alpha * before_b) / (
            1.0 - alpha
        )
        self.assertAlmostEqual(sample_a, sample_b)
        self.assertEqual(engine_a.fov_ema_state(), restored.fov_ema_state())
        env_a.update_u2u_channels()
        env_a.update_u2g_channels()
        env_b.update_u2u_channels()
        env_b.update_u2g_channels()
        np.testing.assert_array_equal(
            self._routing_state(env_a, engine_a),
            self._routing_state(env_b, restored),
        )

    def test_missing_checkpoint_previous_footprints_is_rejected(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        incomplete = engine.fov_ema_state()
        del incomplete["previous_footprints"]
        with self.assertRaisesRegex(RuntimeError, "previous.*footprint"):
            PacketEngine(num_uav=16).load_fov_ema_state(incomplete)

    def test_partial_checkpoint_previous_footprints_is_rejected(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        incomplete = engine.fov_ema_state()
        del incomplete["previous_footprints"]["9"]
        with self.assertRaisesRegex(RuntimeError, "previous.*footprint"):
            PacketEngine(num_uav=16).load_fov_ema_state(incomplete)

    def test_checkpoint_validation_rejects_malformed_lifecycle_states(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        valid = engine.fov_ema_state()
        invalid_states = {}

        state = copy.deepcopy(valid)
        state["previous_footprints"] = {}
        invalid_states["empty initialized footprints"] = state
        state = copy.deepcopy(valid)
        state["previous_footprints"]["0"] = [0, 1, 2]
        invalid_states["wrong footprint length"] = state
        state = copy.deepcopy(valid)
        state["previous_footprints"]["0"] = [2, 1, 0, 1]
        invalid_states["reversed footprint bounds"] = state
        state = copy.deepcopy(valid)
        state["previous_footprints"]["0"] = [0, 1.5, 0, 1]
        invalid_states["non-integer footprint"] = state
        state = copy.deepcopy(valid)
        state["previous_footprints"]["0"] = [0, 1, 0, float("inf")]
        invalid_states["non-finite footprint"] = state
        state = copy.deepcopy(valid)
        state["lifecycle_version"] = "incompatible"
        invalid_states["incompatible lifecycle"] = state
        state = copy.deepcopy(valid)
        state["previous_footprints"][0] = state["previous_footprints"]["0"]
        invalid_states["duplicate normalized UAV ID"] = state
        state = copy.deepcopy(valid)
        del state["values"]["9"]
        invalid_states["partial EMA values"] = state
        state = copy.deepcopy(valid)
        state["initialized_uav_ids"][-1] = 10
        invalid_states["out-of-range initialized UAV"] = state
        state = copy.deepcopy(valid)
        state["initialized_uav_ids"][-1] = 0
        invalid_states["duplicate initialized UAV"] = state
        state = copy.deepcopy(valid)
        state["values"] = []
        invalid_states["wrong values type"] = state

        for label, state in invalid_states.items():
            with self.subTest(label=label):
                with self.assertRaises(RuntimeError):
                    validate_fov_ema_state(state, num_uav=16)
                with self.assertRaises(RuntimeError):
                    PacketEngine(num_uav=16).load_fov_ema_state(state)

    def test_training_checkpoint_uses_the_same_strict_fov_validator(self):
        env = self._environment()
        engine = PacketEngine(num_uav=16)
        engine.update_fov_ema(env, "episode=0,map_reset")
        partial = engine.fov_ema_state()
        del partial["previous_footprints"]["9"]
        training_state = {
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [],
            "delivered_log": [],
            "energy_log": [],
            "lambda_used_log": [],
            "lambda_after_episode_log": [],
            "lambda_cost_used_log": [],
            "lambda_cost_after_episode_log": [],
            "fov_ema_state": partial,
            "sr_route_state": {},
        }
        with self.assertRaisesRegex(RuntimeError, "previous.*footprint"):
            _validate_full_resume_logging_state(
                training_state,
                completed_episode=0,
                routing_agent_kind="safe_ddqn",
                checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            )

    def test_completely_uninitialized_checkpoint_state_is_valid(self):
        empty = {
            "lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
            "values": {},
            "initialized_uav_ids": [],
            "previous_footprints": {},
            "transition_marker": None,
            "footprint_transition_marker": None,
            "update_count": 0,
        }
        validate_fov_ema_state(empty, num_uav=16)
        engine = PacketEngine(num_uav=16)
        engine.load_fov_ema_state(empty)
        self.assertEqual(engine.fov_ema_state(), empty)
        completed_training_state = {
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [0.0],
            "delivered_log": [0.0],
            "energy_log": [0.0],
            "lambda_used_log": [0.0],
            "lambda_after_episode_log": [0.0],
            "lambda_cost_used_log": [0.0],
            "lambda_cost_after_episode_log": [0.0],
            "fov_ema_state": empty,
            "sr_route_state": {},
        }
        with self.assertRaisesRegex(RuntimeError, "uninitialized"):
            _validate_full_resume_logging_state(
                completed_training_state,
                completed_episode=1,
                routing_agent_kind="safe_ddqn",
                checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            )

    def test_representative_routing_strategies_share_the_same_fov_state(self):
        cases = {
            "td3_dinkelbach": ("safe_ddqn", "full"),
            "td3_dinkelbach_wo_ta": ("safe_ddqn", "masked"),
            "td3_dinkelbach_dqn": ("dqn", "full"),
            "td3_dinkelbach_dqn_wo_ta": ("dqn", "masked"),
        }
        ema_indices = [
            routing_state_feature_names().index(name)
            for name in (
                "coverage_overlap_ema",
                "coverage_unvisited_ema",
                "coverage_frontier_ema",
            )
        ]
        for method_id, expected in cases.items():
            with self.subTest(method_id=method_id):
                spec = MethodSpec.parse(method_id)
                self.assertEqual((spec.routing, spec.task_observation), expected)
                env = self._environment()
                engine = PacketEngine(num_uav=16)
                engine.update_fov_ema(env, "episode=0,map_reset")
                physical = self._routing_state(env, engine)
                observed = apply_observation_strategy(
                    physical, spec.task_observation, "routing"
                )
                np.testing.assert_array_equal(
                    observed[ema_indices], physical[ema_indices]
                )
                self.assertEqual(observed.shape, (143,))


if __name__ == "__main__":
    unittest.main()
