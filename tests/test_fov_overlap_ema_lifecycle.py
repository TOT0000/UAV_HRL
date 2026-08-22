import copy
import random
import unittest

import numpy as np

from centralized_movement import fov_task_metrics
from experiment_config import MethodSpec
from HRL_task_aware import _mark_search_observations
from observation_strategy import (
    apply_observation_strategy,
    routing_state_feature_names,
)
from Packet_scheduler_v1 import PacketEngine
from scenario_manifest import generate_manifest
from Simulator import FovCoverageTransition, Simulator


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
        map_changed = any(transition.map_changed for transition in transitions)
        updated = False
        if map_changed:
            updated = engine.update_fov_ema(
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
        transitions, updated = self._apply_production_transition(
            env, engine, "transition=1-no-map-change"
        )
        self.assertTrue(transitions)
        self.assertFalse(any(item.map_changed for item in transitions))
        self.assertFalse(updated)
        self.assertEqual(engine.fov_ema_state(), state_after_first)
        self.assertFalse(
            engine.update_fov_ema(
                env,
                "transition=1",
                footprint_transitions=transitions,
            )
        )
        self.assertEqual(engine.fov_ema_state(), state_after_first)

        uav.x_u, uav.y_u = 900.0, 900.0
        _, updated = self._apply_production_transition(env, engine, "transition=2")
        self.assertTrue(updated)
        self.assertEqual(engine.fov_ema_update_count, 3)
        self.assertAlmostEqual(engine.fov_ema[0]["overlap"], 0.7 * 0.3)

    def test_checkpoint_resume_matches_uninterrupted_next_transition(self):
        env_a = self._environment()
        env_b = self._environment()
        engine_a = PacketEngine(num_uav=16)
        engine_b = PacketEngine(num_uav=16)
        for env, engine in ((env_a, engine_a), (env_b, engine_b)):
            env.uav_dict[0].x_u, env.uav_dict[0].y_u = 100.0, 100.0
            engine.update_fov_ema(env, "episode=0,map_reset")
            self._apply_production_transition(env, engine, "transition=1")
            env.uav_dict[0].x_u, env.uav_dict[0].y_u = 400.0, 100.0
            self._apply_production_transition(env, engine, "transition=2")

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
        env_a.uav_dict[0].x_u, env_a.uav_dict[0].y_u = 700.0, 100.0
        env_b.uav_dict[0].x_u, env_b.uav_dict[0].y_u = 700.0, 100.0
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
        with self.assertRaisesRegex(RuntimeError, "previous-footprint"):
            PacketEngine(num_uav=16).load_fov_ema_state(incomplete)

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
                self.assertEqual(observed.shape, (126,))


if __name__ == "__main__":
    unittest.main()
