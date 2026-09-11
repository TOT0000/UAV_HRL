from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from Simulator import Simulator
from Task_assignment import (
    AssignmentProblem,
    Task,
    UAVAssigner,
    fov_quality_transform,
    fov_com_pair_is_feasible,
    normalize_feasible_values,
    solve_assignment_plan_with_dummies,
    solve_assignment_with_dummies,
)


class AssignmentUtilityTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_type_specific_utility_normalization_and_hover_exclusion(self):
        gt = self.env.gts[0]
        gt.is_found = True
        sr = self.env.SR_teams[0]
        sr.assigned_gt_id = 0
        tasks = [
            Task(0, "FOV", gt, gt.id),
            Task(1, "COM", sr, sr.id),
            Task(2, "Search", self.env.uav_dict[0], 0),
            Task(3, "Hovering", self.env.uav_dict[1], 1),
        ]
        assigner = UAVAssigner(self.env)
        with (
            mock.patch(
                "Task_assignment.assignment_fov_pair_geometry",
                side_effect=[SimpleNamespace(coverage_ratio=0.25, image_quantity=1.0, pair_score=0.4),
                             SimpleNamespace(coverage_ratio=1.0, image_quantity=1.0, pair_score=1.0)],
            ),
            mock.patch.object(
                self.env,
                "get_sr_uav_normalized_utility",
                side_effect=[0.5, 1.0],
            ),
        ):
            problem = assigner.build_problem([0, 1], tasks)

        self.assertEqual([task.task_type for task in problem.tasks], ["FOV", "COM"])
        np.testing.assert_allclose(problem.utility_matrix[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(problem.utility_matrix[:, 1], [0.5, 1.0])
        self.assertTrue(problem.feasible_mask.all())
        np.testing.assert_allclose(problem.raw_fov_utility[:, 0], [0.4, 1.0])
        np.testing.assert_allclose(problem.raw_fov_coverage[:, 0], [0.25, 1.0])
        np.testing.assert_allclose(problem.raw_fov_image_quality[:, 0], [1.0, 1.0])
        self.assertTrue(np.isfinite(problem.utility_matrix).all())

    def test_fov_quality_transform_piecewise_policy_is_finite(self):
        cases = (
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 1.0),
            (float("nan"), 0.0),
            (float("inf"), 0.0),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                transformed = fov_quality_transform(value)
                self.assertTrue(np.isfinite(transformed))
                self.assertEqual(transformed, expected)
        self.assertGreaterEqual(fov_quality_transform(2.0), fov_quality_transform(1.0))

    def test_fov_coverage_multiplies_quality_and_i_above_one_remains_feasible(self):
        gt = self.env.gts[0]
        gt.is_found = True
        task = Task(0, "FOV", gt, gt.id)
        with mock.patch(
            "Task_assignment.assignment_fov_pair_geometry",
            return_value=SimpleNamespace(coverage_ratio=0.4, image_quantity=2.0, pair_score=0.52),
        ):
            problem = UAVAssigner(self.env).build_problem([0], [task])
        self.assertAlmostEqual(problem.raw_fov_utility[0, 0], 0.52)
        self.assertTrue(problem.feasible_mask[0, 0])
        self.assertAlmostEqual(problem.utility_matrix[0, 0], 0.5)

    def test_normalization_uses_only_feasible_values_and_equal_values_are_neutral(self):
        raw = np.asarray([[5.0, -999.0], [5.0, 999.0]])
        feasible = np.asarray([[True, False], [True, False]])
        normalized = normalize_feasible_values(raw, feasible)
        np.testing.assert_array_equal(normalized[:, 0], [0.5, 0.5])
        np.testing.assert_array_equal(normalized[:, 1], [0.0, 0.0])

    def test_dummy_prevents_an_infeasible_task_from_being_written(self):
        selected = solve_assignment_with_dummies(
            np.asarray([[100.0], [50.0]]),
            np.asarray([[False], [False]]),
        )
        self.assertEqual(selected, [])

    def test_dummy_ids_are_canonicalized_lexicographically(self):
        plan = solve_assignment_plan_with_dummies(
            np.zeros((3, 0), dtype=float),
            np.zeros((3, 0), dtype=bool),
        )
        self.assertEqual(
            [entry["dummy_id"] for entry in plan],
            ["dummy_1", "dummy_2", "dummy_3"],
        )

    def test_hover_and_search_are_excluded_after_threshold(self):
        self.env.visited_bitmap[:] = True
        tasks = [
            Task(0, "Search", self.env.uav_dict[0], 0),
            Task(1, "Hovering", self.env.uav_dict[1], 1),
        ]
        problem = UAVAssigner(self.env).build_problem([0], tasks)
        self.assertEqual(problem.tasks, ())
        self.assertEqual(problem.utility_matrix.shape, (1, 0))


class AssignmentCompatibilityTest(unittest.TestCase):
    @staticmethod
    def _task(task_id, task_type, x, y, target_id=0):
        target = SimpleNamespace(x=float(x), y=float(y))
        return Task(task_id, task_type, target, target_id)

    def test_fov_com_pairing_is_symmetric_distance_independent_and_roi_independent(self):
        fov = self._task(0, "FOV", 0.0, 0.0, target_id=7)
        for distance in (199.0, 200.0, 200.01, 900.0):
            with self.subTest(distance=distance):
                com = self._task(1, "COM", distance, 0.0, target_id=7)
                self.assertTrue(fov_com_pair_is_feasible(fov, com))
                self.assertTrue(fov_com_pair_is_feasible(com, fov))
        different_roi_com = self._task(2, "COM", 150.0, 0.0, target_id=99)
        self.assertTrue(fov_com_pair_is_feasible(fov, different_roi_com))

    def test_only_fov_com_is_compatible(self):
        tasks = {
            name: self._task(index, name, 0.0, 0.0)
            for index, name in enumerate(("FOV", "COM", "Search"))
        }
        self.assertTrue(fov_com_pair_is_feasible(tasks["FOV"], tasks["COM"]))
        for first, second in (
            ("FOV", "FOV"),
            ("COM", "COM"),
            ("Search", "Search"),
            ("Search", "FOV"),
            ("COM", "Search"),
        ):
            self.assertFalse(fov_com_pair_is_feasible(tasks[first], tasks[second]))

    def test_over_200m_pair_remains_feasible_and_utility_decides(self):
        env = SimpleNamespace()
        assigner = UAVAssigner(env)
        fov = self._task(0, "FOV", 0.0, 0.0)
        far = self._task(1, "COM", 201.0, 0.0)
        near = self._task(2, "COM", 199.0, 0.0)
        assigner._snapshot_tasks = [fov, far, near]
        utility = np.asarray([[1.0, 0.5]])
        problem = AssignmentProblem(
            uav_ids=(0,),
            tasks=(far, near),
            original_task_indices=(1, 2),
            utility_matrix=utility,
            feasible_mask=np.asarray([[True, True]]),
            raw_fov_utility=np.zeros((1, 2)),
            raw_com_utility=np.zeros((1, 2)),
        )
        feasible = assigner._round_feasible_mask(
            problem,
            {0: [(0, "FOV", 0.5)]},
            {1, 2},
            round_index=1,
        )
        np.testing.assert_array_equal(feasible, [[True, True]])
        np.testing.assert_array_equal(problem.utility_matrix, utility)
        self.assertEqual(solve_assignment_with_dummies(utility, feasible), [(0, 0)])

    def test_k_km_is_two_typed_stages_and_km_is_one_combined_stage(self):
        fov = self._task(0, "FOV", 0.0, 0.0)
        com = self._task(1, "COM", 100.0, 0.0)
        search = self._task(2, "Search", 0.0, 0.0)
        tasks = [fov, com, search]
        combined = AssignmentProblem(
            uav_ids=(0, 1),
            tasks=(fov, com),
            original_task_indices=(0, 1),
            utility_matrix=np.asarray([[1.0, 0.9], [0.8, 1.0]]),
            feasible_mask=np.ones((2, 2), dtype=bool),
            raw_fov_utility=np.zeros((2, 2)),
            raw_com_utility=np.zeros((2, 2)),
        )
        typed = {
            ("FOV",): AssignmentProblem(
                uav_ids=(0, 1), tasks=(fov,), original_task_indices=(0,),
                utility_matrix=np.asarray([[1.0], [0.0]]),
                feasible_mask=np.ones((2, 1), dtype=bool),
                raw_fov_utility=np.zeros((2, 1)), raw_com_utility=np.zeros((2, 1)),
            ),
            ("COM",): AssignmentProblem(
                uav_ids=(0, 1), tasks=(com,), original_task_indices=(1,),
                utility_matrix=np.asarray([[1.0], [0.0]]),
                feasible_mask=np.ones((2, 1), dtype=bool),
                raw_fov_utility=np.zeros((2, 1)), raw_com_utility=np.zeros((2, 1)),
            ),
        }

        kkm = UAVAssigner(SimpleNamespace())
        stages = []
        def typed_problem(*_args, candidate_task_types, **_kwargs):
            stages.append(tuple(candidate_task_types))
            return typed[tuple(candidate_task_types)]
        with mock.patch.object(kkm, "build_problem", side_effect=typed_problem):
            assignments = kkm.assign_tasks([0, 1], tasks, K=99, strategy="k_km")
        self.assertEqual(stages, [("FOV",), ("COM",)])
        self.assertEqual([item[1] for item in assignments[0]], ["FOV", "COM"])
        self.assertTrue(all(len(value) <= 2 for value in assignments.values()))

        km = UAVAssigner(SimpleNamespace())
        with mock.patch.object(km, "build_problem", return_value=combined) as build:
            assignments = km.assign_tasks([0, 1], tasks, K=99, strategy="km")
        self.assertEqual(build.call_args.kwargs["candidate_task_types"], ("FOV", "COM"))
        self.assertEqual(len(km.last_round_problems), 1)
        self.assertTrue(all(len(value) <= 1 for value in assignments.values()))
        selected = [item[0] for value in assignments.values() for item in value]
        self.assertEqual(len(selected), len(set(selected)))


class AssignmentLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_search_release_only_changes_fallback_roles(self):
        before = self.env.assignment_invocations
        for gt in self.env.gts:
            gt.is_found = True
        self.assertFalse(self.env._search_phase_over)
        self.assertLess(float(self.env.visited_bitmap.mean()), 0.99)

        self.env.visited_bitmap[:] = True
        self.env.convert_search_to_hovering()
        self.assertEqual(self.env.assignment_invocations, before)
        self.assertEqual(self.env.search_to_hover_conversions, 1)
        self.assertTrue(self.env._search_phase_over)
        task_types = [
            tasks[0]["task_type"] for tasks in self.env.multi_tasks.values()
        ]
        self.assertTrue(all(task_type == "Hovering" for task_type in task_types))
        self.assertFalse(any(task.task_type == "Search" for task in self.env.task_list))

        self.env.assign_tasks()
        self.assertEqual(self.env.assignment_invocations, before + 1)
        self.assertEqual(self.env.last_assignment.last_round_problems, [])

    def test_phase_fallback_is_search_below_threshold_and_hover_after(self):
        self.env.task_list = []
        self.env.visited_bitmap[:] = False
        self.env.assign_tasks()
        self.assertTrue(
            all(tasks[0]["task_type"] == "Search" for tasks in self.env.multi_tasks.values())
        )
        before = self.env.assignment_invocations
        self.env.visited_bitmap[:] = True
        self.env.convert_search_to_hovering()
        self.assertEqual(self.env.assignment_invocations, before)
        self.assertTrue(
            all(tasks[0]["task_type"] == "Hovering" for tasks in self.env.multi_tasks.values())
        )

    def _service_tasks(self):
        tasks = []
        for gt in self.env.gts:
            gt.is_found = True
            sr = self.env.SR_teams[gt.id]
            sr.assigned_gt_id = gt.id
            tasks.extend(
                [
                    Task(len(tasks), "FOV", gt, gt.id),
                    Task(len(tasks) + 1, "COM", sr, sr.id),
                ]
            )
        return tasks

    def test_random_assignment_round_count_and_typed_order(self):
        tasks = self._service_tasks()
        for rounds, expected_types in (
            (0, set()),
            (-1, set()),
            (1, {"FOV"}),
            (2, {"FOV", "COM"}),
            (99, {"FOV", "COM"}),
        ):
            with self.subTest(rounds=rounds):
                self.env.assignment_rng = np.random.default_rng(1234)
                assigner = UAVAssigner(self.env)
                assigned = assigner.assign_tasks(
                    [1, 2], tasks, K=rounds, strategy="random_one_to_one"
                )
                observed = {item[1] for entries in assigned.values() for item in entries}
                self.assertEqual(observed, expected_types)
                self.assertEqual(len(assigner.last_round_problems), min(max(rounds, 0), 2))

    def test_random_assignment_reuses_uav_by_type_and_preserves_uniqueness(self):
        tasks = self._service_tasks()
        assigned = UAVAssigner(self.env).assign_tasks(
            [1], tasks[:2], K=2, strategy="random_one_to_one"
        )
        self.assertEqual([item[1] for item in assigned[1]], ["FOV", "COM"])

        assigned = UAVAssigner(self.env).assign_tasks(
            [1, 2], tasks, K=2, strategy="random_one_to_one"
        )
        selected = [item[0] for entries in assigned.values() for item in entries]
        self.assertEqual(len(selected), len(set(selected)))
        for entries in assigned.values():
            types = [item[1] for item in entries]
            self.assertLessEqual(types.count("FOV"), 1)
            self.assertLessEqual(types.count("COM"), 1)

    def test_random_assignment_uses_named_rng_and_never_computes_utility(self):
        tasks = self._service_tasks()

        def run(seed):
            self.env.assignment_rng = np.random.default_rng(seed)
            assigner = UAVAssigner(self.env)
            with (
                mock.patch.object(
                    assigner,
                    "build_problem",
                    side_effect=AssertionError("Random must not build utility matrices"),
                ),
                mock.patch(
                    "Task_assignment.assignment_fov_pair_geometry",
                    side_effect=AssertionError("Random must not score FOV utility"),
                ),
                mock.patch.object(
                    self.env,
                    "get_sr_uav_normalized_utility",
                    side_effect=AssertionError("Random must not score COM utility"),
                ),
            ):
                return assigner.assign_tasks(
                    list(range(1, 8)), tasks, K=2, strategy="random_one_to_one"
                )

        first = run(1234)
        second = run(1234)
        third = run(4321)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_random_environment_excludes_gateway_reserved_and_fallback_tasks(self):
        self.env.assignment_strategy = "random_one_to_one"
        self.env.assignment_rounds = 2
        self.env.task_list = self._service_tasks() + [
            Task(4, "Search", self.env.uav_dict[3], 3),
            Task(5, "Hovering", self.env.uav_dict[4], 4),
        ]
        self.env.assign_tasks()
        for uid in (self.env.permanent_gs_gateway_uav_id, *self.env.reserved_search_uav_ids):
            self.assertFalse(
                any(task["task_type"] in {"FOV", "COM"} for task in self.env.multi_tasks[uid])
            )
        assigned_types = {
            task["task_type"]
            for tasks_by_uav in self.env.multi_tasks.values()
            for task in tasks_by_uav
            if task.get("target_id") is not None
        }
        self.assertEqual(assigned_types, {"FOV", "COM"})

    def test_crossing_distance_threshold_does_not_create_a_reassignment_event(self):
        before = self.env.assignment_invocations
        self.env.need_reassign = False
        sr = self.env.SR_teams[0]
        sr.x += 250.0
        self.assertEqual(self.env.assignment_invocations, before)
        self.assertFalse(self.env.need_reassign)
        self.env.assign_tasks()
        self.assertEqual(self.env.assignment_invocations, before + 1)


if __name__ == "__main__":
    unittest.main()
