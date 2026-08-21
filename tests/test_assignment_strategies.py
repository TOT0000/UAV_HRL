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
    fov_com_pair_is_feasible,
    normalize_feasible_values,
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
        sr.active = True
        tasks = [
            Task(0, "FOV", gt, gt.id),
            Task(1, "COM", sr, sr.id),
            Task(2, "Search", self.env.uav_dict[0], 0),
            Task(3, "Hovering", self.env.uav_dict[1], 1),
        ]
        assigner = UAVAssigner(self.env)
        with (
            mock.patch(
                "Task_assignment.FovModel.calculate_fov_single",
                side_effect=[(2.0, None), (4.0, None)],
            ),
            mock.patch.object(
                self.env,
                "get_sr_uav_capacity_mbps",
                side_effect=[10.0, 30.0],
            ),
        ):
            problem = assigner.build_problem([0, 1], tasks)

        self.assertEqual([task.task_type for task in problem.tasks], ["FOV", "COM", "Search"])
        np.testing.assert_allclose(problem.utility_matrix[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(problem.utility_matrix[:, 1], [0.0, 1.0])
        np.testing.assert_allclose(problem.utility_matrix[:, 2], [0.05, 0.05])
        self.assertTrue(problem.feasible_mask.all())
        self.assertGreater(problem.raw_fov_utility[0, 0], 1.0)
        self.assertTrue(np.isfinite(problem.utility_matrix).all())

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

    def test_fov_com_distance_boundary_is_symmetric_and_roi_independent(self):
        fov = self._task(0, "FOV", 0.0, 0.0, target_id=7)
        for distance, expected in ((199.0, True), (200.0, True), (200.01, False)):
            with self.subTest(distance=distance):
                com = self._task(1, "COM", distance, 0.0, target_id=7)
                self.assertEqual(fov_com_pair_is_feasible(fov, com), expected)
                self.assertEqual(fov_com_pair_is_feasible(com, fov), expected)
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

    def test_over_distance_high_utility_is_masked_without_changing_utility(self):
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
            max_distance_m=200.0,
        )
        np.testing.assert_array_equal(feasible, [[False, True]])
        np.testing.assert_array_equal(problem.utility_matrix, utility)
        self.assertEqual(solve_assignment_with_dummies(utility, feasible), [(0, 1)])

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
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()

    def test_search_to_hover_is_threshold_only_and_does_not_reassign(self):
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
        self.assertTrue(
            all(
                tasks[0]["task_type"] == "Hovering"
                for tasks in self.env.multi_tasks.values()
            )
        )
        self.assertTrue(all(task.task_type == "Hovering" for task in self.env.task_list))

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

    def test_random_assignment_is_seeded_one_round_and_excludes_hover(self):
        self.env.assignment_strategy = "random_one_to_one"
        self.env.assignment_rounds = 1
        self.env.task_list.append(
            Task(99, "Hovering", self.env.uav_dict[0], 0)
        )
        random.seed(1234)
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
