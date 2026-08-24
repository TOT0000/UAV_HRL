"""Shared task-assignment strategies for K-KM, KM, and random baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from centralized_movement import fov_task_metrics
from experiment_config import (
    ASSIGNMENT_DUMMY_UTILITY,
    FOV_COM_PAIR_MAX_DISTANCE_M,
    RESERVED_SEARCH_UAV_IDS,
    SEARCH_COVERAGE_THRESHOLD,
)


ASSIGNMENT_TASK_TYPES = ("FOV", "COM")


@dataclass(frozen=True)
class AssignmentProblem:
    """Finite domain utilities plus the authoritative feasibility mask."""

    uav_ids: tuple[int, ...]
    tasks: tuple[object, ...]
    original_task_indices: tuple[int, ...]
    utility_matrix: np.ndarray
    feasible_mask: np.ndarray
    raw_fov_utility: np.ndarray
    raw_com_utility: np.ndarray
    raw_fov_coverage: np.ndarray | None = None
    raw_fov_image_quality: np.ndarray | None = None


def fov_quality_transform(image_quality):
    """Apply the assignment-only reciprocal quality policy to production I."""

    value = float(image_quality)
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return value if value <= 1.0 else 1.0 / value


def assignment_fov_pair_metrics(env, uav_id, task):
    """Reuse the movement path's ROI coverage and geometric image quantity."""

    target = task.target_obj
    descriptor = {
        "task_type": "FOV",
        "target_obj_id": int(task.target_obj_id),
        "target_pos": (
            float(target.x),
            float(target.y),
            float(getattr(target, "z", 0.0)),
        ),
    }
    coverage, image_quality, geometry_valid = fov_task_metrics(
        env, int(uav_id), descriptor
    )
    coverage = float(coverage)
    if not math.isfinite(coverage):
        coverage = 0.0
    coverage = min(max(coverage, 0.0), 1.0)
    return (
        coverage,
        float(image_quality),
        bool(geometry_valid),
    )


def normalize_feasible_values(raw_values, feasible_mask):
    """Globally min-max normalize one task type using feasible pairs only."""

    raw = np.asarray(raw_values, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if raw.shape != feasible.shape:
        raise ValueError("raw utility and feasible mask shapes differ")
    normalized = np.zeros(raw.shape, dtype=float)
    values = raw[feasible]
    if values.size == 0:
        return normalized
    if not np.isfinite(values).all():
        raise ValueError("feasible raw utility contains NaN or Inf")
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(minimum, maximum):
        normalized[feasible] = 0.5
    else:
        normalized[feasible] = (values - minimum) / (maximum - minimum)
    return normalized


def fov_com_distance_m(fov_task, com_task):
    """Return horizontal distance between the current FOV and COM targets."""

    fov_target = fov_task.target_obj
    com_target = com_task.target_obj
    return float(
        math.hypot(
            float(fov_target.x) - float(com_target.x),
            float(fov_target.y) - float(com_target.y),
        )
    )


def fov_com_pair_is_feasible(
    first_task,
    second_task,
    max_distance_m=FOV_COM_PAIR_MAX_DISTANCE_M,
):
    """Allow only symmetric FOV+COM pairs within the configured distance."""

    pair = {first_task.task_type, second_task.task_type}
    if pair != {"FOV", "COM"}:
        return False
    fov_task = first_task if first_task.task_type == "FOV" else second_task
    com_task = first_task if first_task.task_type == "COM" else second_task
    return fov_com_distance_m(fov_task, com_task) <= float(max_distance_m)


def solve_assignment_plan_with_dummies(
    utility_matrix,
    feasible_mask,
    dummy_utility=ASSIGNMENT_DUMMY_UTILITY,
):
    """Return the deterministic real/dummy plan for one assignment round."""

    utility = np.asarray(utility_matrix, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if utility.shape != feasible.shape or utility.ndim != 2:
        raise ValueError("assignment utility and feasible mask must be equal 2-D shapes")
    if not np.isfinite(utility).all():
        raise ValueError("assignment utility matrix must contain only finite values")
    row_count, task_count = utility.shape
    if row_count == 0:
        return []
    solver_cost = np.full((row_count, task_count + row_count), np.inf, dtype=float)
    solver_cost[:, :task_count][feasible] = -utility[feasible]
    solver_cost[:, task_count:] = -float(dummy_utility)
    rows, columns = linear_sum_assignment(solver_cost)
    assignments = list(zip(rows.tolist(), columns.tolist()))
    dummy_rows = sorted(row for row, column in assignments if column >= task_count)
    dummy_columns = sorted(
        column for _row, column in assignments if column >= task_count
    )
    canonical_dummy_by_row = dict(zip(dummy_rows, dummy_columns))
    plan = []
    for row, column in sorted(assignments):
        if column >= task_count:
            canonical_column = canonical_dummy_by_row[row]
            plan.append(
                {
                    "row": row,
                    "task_column": None,
                    "dummy_id": f"dummy_{canonical_column - task_count + 1}",
                }
            )
            continue
        if not feasible[row, column]:
            raise AssertionError("solver selected a task outside the feasible mask")
        plan.append(
            {"row": row, "task_column": column, "dummy_id": None}
        )
    return plan


def solve_assignment_with_dummies(
    utility_matrix,
    feasible_mask,
    dummy_utility=ASSIGNMENT_DUMMY_UTILITY,
):
    """Solve one round without exposing solver sentinels as domain utilities."""

    return [
        (entry["row"], entry["task_column"])
        for entry in solve_assignment_plan_with_dummies(
            utility_matrix,
            feasible_mask,
            dummy_utility=dummy_utility,
        )
        if entry["task_column"] is not None
    ]


class UAVAssigner:
    """Build and solve assignment rounds while preserving one environment flow."""

    def __init__(self, env):
        self.env = env
        self.assignments = {}
        self.last_round_problems = []

    def assign_tasks(
        self,
        uav_id_list,
        task_list,
        K=2,
        alpha=None,
        beta=None,
        mode="KM",
        strategy=None,
        max_distance_m=None,
        coverage_threshold=None,
    ):
        """Dispatch one centralized assignment strategy.

        ``alpha`` and ``beta`` remain accepted for call compatibility, but no
        weighted FOV/COM mixture is used.
        """

        del alpha, beta
        if strategy is None:
            strategy = "random_one_to_one" if mode == "Random" else "k_km"
        if strategy not in {"k_km", "km", "random_one_to_one"}:
            raise ValueError(f"unknown assignment strategy: {strategy}")
        self._snapshot_tasks = list(task_list)
        max_distance_m = float(
            FOV_COM_PAIR_MAX_DISTANCE_M
            if max_distance_m is None
            else max_distance_m
        )
        coverage_threshold = float(
            SEARCH_COVERAGE_THRESHOLD
            if coverage_threshold is None
            else coverage_threshold
        )
        if strategy == "random_one_to_one":
            return self.random_assign_tasks(
                uav_id_list,
                task_list,
                coverage_threshold=coverage_threshold,
            )
        rounds = 1 if strategy == "km" else min(int(K), 2)
        return self.assign_uav_tasks_k_times(
            uav_id_list,
            task_list,
            K=rounds,
            max_distance_m=max_distance_m,
            coverage_threshold=coverage_threshold,
        )

    def _candidate_tasks(self, task_list, coverage_threshold):
        candidates = []
        for index, task in enumerate(task_list):
            if task.task_type == "Hovering":
                continue
            if task.task_type not in ASSIGNMENT_TASK_TYPES:
                continue
            candidates.append((index, task))
        return candidates

    def build_problem(
        self,
        uav_ids,
        task_list,
        *,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
    ):
        candidates = self._candidate_tasks(task_list, coverage_threshold)
        uav_ids = tuple(int(uid) for uid in uav_ids)
        original_indices = tuple(index for index, _ in candidates)
        tasks = tuple(task for _, task in candidates)
        shape = (len(uav_ids), len(tasks))
        raw_fov = np.zeros(shape, dtype=float)
        raw_com = np.zeros(shape, dtype=float)
        raw_fov_coverage = np.zeros(shape, dtype=float)
        raw_fov_image_quality = np.zeros(shape, dtype=float)
        fov_feasible = np.zeros(shape, dtype=bool)
        com_feasible = np.zeros(shape, dtype=bool)

        for row, uav_id in enumerate(uav_ids):
            for column, task in enumerate(tasks):
                if task.task_type == "FOV":
                    gt = self.env.gts[int(task.target_obj_id)]
                    if not bool(gt.is_found):
                        continue
                    coverage, image_quality, geometry_valid = (
                        assignment_fov_pair_metrics(self.env, uav_id, task)
                    )
                    raw_fov_coverage[row, column] = coverage
                    raw_fov_image_quality[row, column] = image_quality
                    raw_fov[row, column] = (
                        coverage * fov_quality_transform(image_quality)
                    )
                    if geometry_valid:
                        fov_feasible[row, column] = True
                elif task.task_type == "COM":
                    sr = self.env.SR_teams[int(task.target_obj_id)]
                    if not bool(sr.active):
                        continue
                    capacity_bps = self.env.get_sr_uav_reference_capacity_mbps(
                        uav_id, int(task.target_obj_id)
                    ) * 1e6
                    required_bps = float(self.env.com_required_rate_bps)
                    raw = (
                        1.0
                        if required_bps <= 0.0
                        else min(max(float(capacity_bps), 0.0) / required_bps, 1.0)
                    )
                    if math.isfinite(raw):
                        raw_com[row, column] = raw
                        com_feasible[row, column] = True

        normalized_fov = normalize_feasible_values(raw_fov, fov_feasible)
        utility = np.zeros(shape, dtype=float)
        utility[fov_feasible] = normalized_fov[fov_feasible]
        utility[com_feasible] = raw_com[com_feasible]
        feasible = fov_feasible | com_feasible
        if not np.isfinite(utility).all():
            raise AssertionError("production assignment utility contains NaN or Inf")
        return AssignmentProblem(
            uav_ids=uav_ids,
            tasks=tasks,
            original_task_indices=original_indices,
            utility_matrix=utility,
            feasible_mask=feasible,
            raw_fov_utility=raw_fov,
            raw_com_utility=raw_com,
            raw_fov_coverage=raw_fov_coverage,
            raw_fov_image_quality=raw_fov_image_quality,
        )

    def _round_feasible_mask(
        self,
        problem,
        assignments,
        available_original_indices,
        round_index,
        max_distance_m,
    ):
        feasible = problem.feasible_mask.copy()
        for column, original_index in enumerate(problem.original_task_indices):
            if original_index not in available_original_indices:
                feasible[:, column] = False
        if round_index == 0:
            return feasible
        for row, uav_id in enumerate(problem.uav_ids):
            previous = assignments[uav_id]
            if not previous:
                continue
            if len(previous) >= 2:
                feasible[row, :] = False
                continue
            first_task = self._snapshot_tasks[previous[0][0]]
            for column, candidate in enumerate(problem.tasks):
                feasible[row, column] &= fov_com_pair_is_feasible(
                    first_task,
                    candidate,
                    max_distance_m=max_distance_m,
                )
        return feasible

    def assign_uav_tasks_k_times(
        self,
        uav_list,
        task_list,
        K=2,
        *,
        max_distance_m=FOV_COM_PAIR_MAX_DISTANCE_M,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
    ):
        rounds = min(max(int(K), 0), 2)
        self.assignments = {int(uid): [] for uid in uav_list}
        problem = self.build_problem(
            uav_list,
            task_list,
            coverage_threshold=coverage_threshold,
        )
        available = set(problem.original_task_indices)
        self.last_round_problems = []
        for round_index in range(rounds):
            if not available or not problem.tasks:
                break
            feasible = self._round_feasible_mask(
                problem,
                self.assignments,
                available,
                round_index,
                max_distance_m,
            )
            self.last_round_problems.append((problem.utility_matrix.copy(), feasible.copy()))
            for row, column in solve_assignment_with_dummies(
                problem.utility_matrix,
                feasible,
            ):
                original_index = problem.original_task_indices[column]
                if original_index not in available or not feasible[row, column]:
                    raise AssertionError("assignment result failed post-solve validation")
                task = problem.tasks[column]
                self.assignments[problem.uav_ids[row]].append(
                    (
                        original_index,
                        task.task_type,
                        float(problem.utility_matrix[row, column]),
                    )
                )
                available.remove(original_index)
        return self.assignments

    def random_assign_tasks(
        self,
        uav_list,
        task_list,
        *,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
    ):
        candidates = self._candidate_tasks(task_list, coverage_threshold)
        task_indices = [index for index, _ in candidates]
        random.shuffle(task_indices)
        self.assignments = {int(uid): [] for uid in uav_list}
        for uav_id, task_index in zip(uav_list, task_indices):
            task = task_list[task_index]
            self.assignments[int(uav_id)].append(
                (int(task_index), task.task_type, 0.0)
            )
        self.last_round_problems = []
        return self.assignments

    def build_uav_tasks_from_assignment(self):
        """Materialize explicit assignments plus phase fallback behavior."""

        snapshot = getattr(self, "_snapshot_tasks", self.env.task_list)
        coverage_threshold = float(
            getattr(
                self.env,
                "search_coverage_threshold",
                SEARCH_COVERAGE_THRESHOLD,
            )
        )
        search_active = (
            not bool(getattr(self.env, "_search_phase_over", False))
            and float(np.asarray(self.env.visited_bitmap, dtype=bool).mean())
            < coverage_threshold
        )
        self.env.multi_tasks = {}
        for uav_id in range(self.env.num_UAV):
            assigned = self.assignments.get(uav_id, [])
            uav = self.env.uav_dict[uav_id]
            entries = []
            for task_index, task_type, _ in assigned:
                task = snapshot[task_index]
                if task_type == "FOV":
                    target = self.env.gts[int(task.target_obj_id)]
                    position = target.get_position()
                    target_object_id = int(task.target_obj_id)
                elif task_type == "COM":
                    target = self.env.SR_teams[int(task.target_obj_id)]
                    position = target.get_position()
                    target_object_id = int(task.target_obj_id)
                else:
                    raise AssertionError(f"non-candidate task was assigned: {task_type}")
                entries.append(
                    {
                        "task_type": task_type,
                        "target_id": int(task_index),
                        "target_obj_id": target_object_id,
                        "target_pos": tuple(position),
                    }
                )
            if search_active and uav_id in tuple(
                getattr(self.env, "reserved_search_uav_ids", RESERVED_SEARCH_UAV_IDS)
            ):
                if entries:
                    raise AssertionError("reserved Search UAV entered service assignment")
                entries.append(
                    {
                        "task_type": "Search",
                        "target_id": None,
                        "target_obj_id": int(uav_id),
                        "target_pos": tuple(uav.get_position()),
                        "reserved_search": True,
                    }
                )
            elif not entries:
                fallback_type = "Search" if search_active else "Hovering"
                entries.append(
                    {
                        "task_type": fallback_type,
                        "target_id": None,
                        "target_obj_id": int(uav_id),
                        "target_pos": tuple(uav.get_position()),
                        "phase_fallback": True,
                    }
                )
            self.env.multi_tasks[uav_id] = entries
            if not search_active and any(
                entry["task_type"] == "Search" for entry in entries
            ):
                raise AssertionError("Search assignment created after coverage release")
            primary = sorted(
                entries,
                key=lambda item: (
                    {"FOV": 0, "COM": 1, "Search": 2, "Hovering": 3}[
                        item["task_type"]
                    ],
                    -1 if item.get("target_obj_id") is None else item["target_obj_id"],
                ),
            )[0]
            uav.active_task_index = 0
            uav.task_type = primary["task_type"]
            uav.assigned_target_id = primary["target_id"]
            uav.target_position = primary["target_pos"]


class Task:
    def __init__(self, task_id, task_type, target_obj, target_obj_id):
        self.task_id = task_id
        self.task_type = task_type
        self.target_obj = target_obj
        self.target_obj_id = target_obj_id
        self.target_type = type(target_obj).__name__
        self.is_assigned = False
