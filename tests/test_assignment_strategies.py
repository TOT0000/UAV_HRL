from types import SimpleNamespace
import random
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
        self.env = Simulator(num_UAV=10)
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
                "Task_assignment.assignment_fov_pair_metrics",
                side_effect=[(0.25, 1.0, True), (1.0, 1.0, True)],
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
        np.testing.assert_allclose(problem.raw_fov_utility[:, 0], [0.25, 1.0])
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
            "Task_assignment.assignment_fov_pair_metrics",
            return_value=(0.4, 2.0, True),
        ):
            problem = UAVAssigner(self.env).build_problem([0], [task])
        self.assertAlmostEqual(problem.raw_fov_utility[0, 0], 0.4)
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

    def test_k_km_is_capped_at_two_rounds_and_km_at_one(self):
        fov = self._task(0, "FOV", 0.0, 0.0)
        com = self._task(1, "COM", 100.0, 0.0)
        search = self._task(2, "Search", 0.0, 0.0)
        tasks = [fov, com, search]
        problem = AssignmentProblem(
            uav_ids=(0, 1),
            tasks=tuple(tasks),
            original_task_indices=(0, 1, 2),
            utility_matrix=np.asarray([[1.0, 0.9, 0.1], [0.8, 1.0, 0.1]]),
            feasible_mask=np.ones((2, 3), dtype=bool),
            raw_fov_utility=np.zeros((2, 3)),
            raw_com_utility=np.zeros((2, 3)),
        )
        for strategy, expected_rounds, max_tasks in (
            ("k_km", 2, 2),
            ("km", 1, 1),
        ):
            assigner = UAVAssigner(SimpleNamespace())
            with mock.patch.object(assigner, "build_problem", return_value=problem):
                assignments = assigner.assign_tasks(
                    [0, 1], tasks, K=99, strategy=strategy
                )
            self.assertLessEqual(len(assigner.last_round_problems), expected_rounds)
            self.assertTrue(all(len(value) <= max_tasks for value in assignments.values()))
            selected = [item[0] for value in assignments.values() for item in value]
            self.assertEqual(len(selected), len(set(selected)))
            for value in assignments.values():
                if len(value) == 2:
                    self.assertEqual({value[0][1], value[1][1]}, {"FOV", "COM"})


class AssignmentLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_search_release_immediately_reassigns(self):
        before = self.env.assignment_invocations
        for gt in self.env.gts:
            gt.is_found = True
        self.assertFalse(self.env._search_phase_over)
        self.assertLess(float(self.env.visited_bitmap.mean()), 0.99)

        self.env.visited_bitmap[:] = True
        self.env.convert_search_to_hovering()
        self.assertEqual(self.env.assignment_invocations, before + 1)
        self.assertEqual(self.env.search_to_hover_conversions, 1)
        self.assertTrue(self.env._search_phase_over)
        task_types = [
            tasks[0]["task_type"] for tasks in self.env.multi_tasks.values()
        ]
        self.assertEqual(task_types.count("Relay"), 1)
        self.assertTrue(
            all(task_type in {"Relay", "Hovering"} for task_type in task_types)
        )
        self.assertFalse(any(task.task_type == "Search" for task in self.env.task_list))

        self.env.assign_tasks()
        self.assertEqual(self.env.assignment_invocations, before + 2)
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
        self.assertEqual(self.env.assignment_invocations, before + 1)
        self.assertTrue(
            all(tasks[0]["task_type"] == "Hovering" for tasks in self.env.multi_tasks.values())
        )

    def test_random_assignment_is_seeded_one_round_and_excludes_hover(self):
        self.env.assignment_strategy = "random_one_to_one"
        self.env.assignment_rounds = 1
        self.env.task_list.append(
            Task(99, "Hovering", self.env.uav_dict[0], 0)
        )
        random.seed(1234)
        with mock.patch(
            "Task_assignment.assignment_fov_pair_metrics",
            side_effect=AssertionError("random assignment must not score utilities"),
        ):
            self.env.assign_tasks()
        first = {
            uid: tuple(task["target_id"] for task in tasks)
            for uid, tasks in self.env.multi_tasks.items()
        }
        random.seed(1234)
        self.env.assign_tasks()
        second = {
            uid: tuple(task["target_id"] for task in tasks)
            for uid, tasks in self.env.multi_tasks.items()
        }
        self.assertEqual(first, second)
        self.assertTrue(all(len(tasks) == 1 for tasks in self.env.multi_tasks.values()))
        self.assertTrue(
            all(tasks[0]["task_type"] != "Hovering" for tasks in self.env.multi_tasks.values())
        )

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
