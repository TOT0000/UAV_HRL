"""Shared task-assignment strategies for K-KM, KM, and random baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from centralized_movement import (
    fov_quality_transform,
    fov_sensing_progress,
    fov_task_metrics,
)
from experiment_config import (
    ASSIGNMENT_DUMMY_UTILITY,
    FOV_COM_PAIR_MAX_DISTANCE_M,
    PERMANENT_GS_GATEWAY_UAV_ID,
    RESERVED_SEARCH_UAV_IDS,
    SEARCH_COVERAGE_THRESHOLD,
)
from relay_contract import relay_metrics_by_candidate


SERVICE_TASK_TYPES = ("Relay", "FOV", "COM")
FOV_COM_TASK_TYPES = ("FOV", "COM")


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
    raw_relay_utility: np.ndarray | None = None
    raw_fov_coverage: np.ndarray | None = None
    raw_fov_image_quality: np.ndarray | None = None
    relay_metrics_by_uav: dict[int, object] | None = None


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


def fov_com_pair_is_feasible(
    first_task,
    second_task,
    max_distance_m=FOV_COM_PAIR_MAX_DISTANCE_M,
):
    """Allow only symmetric FOV+COM pairs; distance is intentionally ignored."""

    del max_distance_m
    pair = {first_task.task_type, second_task.task_type}
    return pair == {"FOV", "COM"}


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
        self.last_relay_metrics = {}
        self.relay_handling_mode = None
        self.requested_relay_count = 0
        self.selected_relay_uav_ids = []
        self.last_relay_hungarian_plan = []

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
        del max_distance_m
        coverage_threshold = float(
            SEARCH_COVERAGE_THRESHOLD
            if coverage_threshold is None
            else coverage_threshold
        )
        if strategy == "random_one_to_one":
            result = self.random_assign_tasks(
                uav_id_list,
                task_list,
                coverage_threshold=coverage_threshold,
            )
            self.relay_handling_mode = "single_joint_relay_fov_com_named_rng"
            return result
        if strategy == "k_km":
            self.relay_handling_mode = "relay_first_quota_then_two_round_fov_com"
            return self.assign_relay_first_k_km(
                uav_id_list,
                task_list,
                K=min(int(K), 2),
                coverage_threshold=coverage_threshold,
            )
        self.relay_handling_mode = "single_joint_relay_fov_com_hungarian"
        return self.assign_uav_tasks_k_times(
            uav_id_list,
            task_list,
            K=1,
            coverage_threshold=coverage_threshold,
            candidate_task_types=SERVICE_TASK_TYPES,
        )

    def _candidate_tasks(
        self, task_list, coverage_threshold, candidate_task_types=SERVICE_TASK_TYPES
    ):
        del coverage_threshold
        candidates = []
        for index, task in enumerate(task_list):
            if task.task_type == "Hovering":
                continue
            if task.task_type not in candidate_task_types:
                continue
            candidates.append((index, task))
        return candidates

    def build_problem(
        self,
        uav_ids,
        task_list,
        *,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
        candidate_task_types=SERVICE_TASK_TYPES,
    ):
        candidates = self._candidate_tasks(
            task_list, coverage_threshold, candidate_task_types
        )
        uav_ids = tuple(int(uid) for uid in uav_ids)
        original_indices = tuple(index for index, _ in candidates)
        tasks = tuple(task for _, task in candidates)
        shape = (len(uav_ids), len(tasks))
        raw_fov = np.zeros(shape, dtype=float)
        raw_com = np.zeros(shape, dtype=float)
        raw_relay = np.zeros(shape, dtype=float)
        raw_fov_coverage = np.zeros(shape, dtype=float)
        raw_fov_image_quality = np.zeros(shape, dtype=float)
        fov_feasible = np.zeros(shape, dtype=bool)
        com_feasible = np.zeros(shape, dtype=bool)
        relay_feasible = np.zeros(shape, dtype=bool)
        relay_candidate_metrics = relay_metrics_by_candidate(
            self.env,
            uav_ids,
            backlog_bits=getattr(self.env, "assignment_backlog_snapshot", {}),
        ) if any(task.task_type == "Relay" for task in tasks) else {}

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
                    raw_fov[row, column] = fov_sensing_progress(
                        coverage, image_quality
                    )
                    if geometry_valid:
                        fov_feasible[row, column] = True
                elif task.task_type == "COM":
                    sr = self.env.SR_teams[int(task.target_obj_id)]
                    if sr.assigned_gt_id is None:
                        continue
                    raw = self.env.get_sr_uav_normalized_utility(
                        uav_id, int(task.target_obj_id)
                    )
                    if math.isfinite(raw):
                        raw_com[row, column] = raw
                        com_feasible[row, column] = True
                elif task.task_type == "Relay":
                    raw_relay[row, column] = relay_candidate_metrics[uav_id].utility
                    relay_feasible[row, column] = True

        normalized_fov = normalize_feasible_values(raw_fov, fov_feasible)
        utility = np.zeros(shape, dtype=float)
        utility[fov_feasible] = normalized_fov[fov_feasible]
        utility[com_feasible] = raw_com[com_feasible]
        utility[relay_feasible] = raw_relay[relay_feasible]
        feasible = fov_feasible | com_feasible | relay_feasible
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
            raw_relay_utility=raw_relay,
            raw_fov_coverage=raw_fov_coverage,
            raw_fov_image_quality=raw_fov_image_quality,
            relay_metrics_by_uav=relay_candidate_metrics,
        )

    def _round_feasible_mask(
        self,
        problem,
        assignments,
        available_original_indices,
        round_index,
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
        candidate_task_types=SERVICE_TASK_TYPES,
        initial_assignments=None,
    ):
        del max_distance_m
        rounds = min(max(int(K), 0), 2)
        if initial_assignments is None:
            self.assignments = {int(uid): [] for uid in uav_list}
        else:
            self.assignments = {
                int(uid): list(assignments)
                for uid, assignments in initial_assignments.items()
            }
            for uid in uav_list:
                self.assignments.setdefault(int(uid), [])
        problem = self.build_problem(
            uav_list,
            task_list,
            coverage_threshold=coverage_threshold,
            candidate_task_types=candidate_task_types,
        )
        if problem.relay_metrics_by_uav:
            self.last_relay_metrics = dict(problem.relay_metrics_by_uav)
            self.requested_relay_count = sum(
                task.task_type == "Relay" for task in problem.tasks
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
        self.selected_relay_uav_ids = sorted(
            uid
            for uid, assignments in self.assignments.items()
            if any(task_type == "Relay" for _index, task_type, _utility in assignments)
        )
        return self.assignments

    def assign_relay_first_k_km(
        self,
        uav_list,
        task_list,
        K=2,
        *,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
    ):
        """Fill identical Relay slots first, then retain formal FOV/COM rounds."""

        uav_ids = tuple(sorted({int(uid) for uid in uav_list}))
        relay_tasks = sorted(
            (
                (index, task)
                for index, task in enumerate(task_list)
                if task.task_type == "Relay"
            ),
            key=lambda item: (str(item[1].task_id), item[0]),
        )
        self.requested_relay_count = len(relay_tasks)
        self.last_relay_metrics = (
            relay_metrics_by_candidate(
                self.env,
                uav_ids,
                backlog_bits=getattr(self.env, "assignment_backlog_snapshot", {}),
            )
            if relay_tasks
            else {}
        )
        quota = min(len(relay_tasks), len(uav_ids))
        # No dummy columns are admitted in this Relay-only Hungarian stage, so
        # the real quota is filled even when every utility is zero. Independent
        # slots reduce the optimum to the top-quota set; UAV id then provides
        # the canonical exact-tie solution without perturbing real utilities.
        if quota:
            relay_utility = np.asarray(
                [
                    [self.last_relay_metrics[uid].utility] * quota
                    for uid in uav_ids
                ],
                dtype=float,
            )
            solver_rows, solver_columns = linear_sum_assignment(-relay_utility)
            solver_total = float(relay_utility[solver_rows, solver_columns].sum())
            selected = sorted(
                sorted(
                    uav_ids,
                    key=lambda uid: (-self.last_relay_metrics[uid].utility, uid),
                )[:quota]
            )
            canonical_total = sum(
                self.last_relay_metrics[uid].utility for uid in selected
            )
            if not math.isclose(
                canonical_total, solver_total, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise AssertionError("canonical Relay tie-break changed KM optimum")
            self.last_relay_hungarian_plan = [
                (uid, relay_tasks[column][0])
                for column, uid in enumerate(selected)
            ]
        else:
            selected = []
            self.last_relay_hungarian_plan = []
        self.selected_relay_uav_ids = selected
        assignments = {uid: [] for uid in uav_ids}
        for uid, (task_index, _task) in zip(selected, relay_tasks):
            assignments[uid].append(
                (
                    int(task_index),
                    "Relay",
                    float(self.last_relay_metrics[uid].utility),
                )
            )
        remaining = [uid for uid in uav_ids if uid not in set(selected)]
        return self.assign_uav_tasks_k_times(
            remaining,
            task_list,
            K=K,
            coverage_threshold=coverage_threshold,
            candidate_task_types=FOV_COM_TASK_TYPES,
            initial_assignments=assignments,
        )

    def random_assign_tasks(
        self,
        uav_list,
        task_list,
        *,
        coverage_threshold=SEARCH_COVERAGE_THRESHOLD,
    ):
        candidates = self._candidate_tasks(
            task_list, coverage_threshold, SERVICE_TASK_TYPES
        )
        uav_ids = tuple(int(uid) for uid in uav_list)
        self.requested_relay_count = sum(
            task.task_type == "Relay" for _index, task in candidates
        )
        self.last_relay_metrics = (
            relay_metrics_by_candidate(
                self.env,
                uav_ids,
                backlog_bits=getattr(self.env, "assignment_backlog_snapshot", {}),
            )
            if self.requested_relay_count
            else {}
        )
        task_indices = [index for index, _ in candidates]
        rng = getattr(self.env, "assignment_rng", None)
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(task_indices)
        self.assignments = {int(uid): [] for uid in uav_list}
        for uav_id, task_index in zip(uav_list, task_indices):
            task = task_list[task_index]
            self.assignments[int(uav_id)].append(
                (int(task_index), task.task_type, 0.0)
            )
        self.last_round_problems = []
        self.selected_relay_uav_ids = sorted(
            uid
            for uid, assignments in self.assignments.items()
            if any(task_type == "Relay" for _index, task_type, _utility in assignments)
        )
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
            if uav_id == PERMANENT_GS_GATEWAY_UAV_ID and assigned:
                raise AssertionError(
                    "permanent GS gateway entered Relay/FOV/COM service assignment"
                )
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
                elif task_type == "Relay":
                    position = None
                    target_object_id = None
                else:
                    raise AssertionError(f"non-candidate task was assigned: {task_type}")
                entries.append(
                    {
                        "task_type": task_type,
                        "target_id": (
                            task.task_id if task_type == "Relay" else int(task_index)
                        ),
                        "target_obj_id": target_object_id,
                        **(
                            {"target_pos": tuple(position)}
                            if position is not None
                            else {}
                        ),
                        **(
                            {
                                "relay_receive_score_at_assignment": float(
                                    self.last_relay_metrics[uav_id].receive_score
                                ),
                                "relay_forward_score_at_assignment": float(
                                    self.last_relay_metrics[uav_id].forward_score
                                ),
                                "relay_utility_at_assignment": float(
                                    self.last_relay_metrics[uav_id].utility
                                ),
                            }
                            if task_type == "Relay"
                            else {}
                        ),
                    }
                )
            if uav_id == PERMANENT_GS_GATEWAY_UAV_ID:
                entries.append(
                    {
                        "task_type": "Search" if search_active else "Hovering",
                        "target_id": None,
                        "target_obj_id": int(uav_id),
                        "target_pos": tuple(uav.get_position()),
                        "reserved_search": bool(search_active),
                        "permanent_gs_gateway": True,
                    }
                )
            elif search_active and uav_id in tuple(
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
            if any(entry["task_type"] == "Relay" for entry in entries) and len(entries) != 1:
                raise AssertionError("Relay assignment must be exclusive")
            self.env.multi_tasks[uav_id] = entries
            if not search_active and any(
                entry["task_type"] == "Search" for entry in entries
            ):
                raise AssertionError("Search assignment created after coverage release")
            primary = sorted(
                entries,
                key=lambda item: (
                    {"Relay": 0, "FOV": 1, "COM": 2, "Search": 3, "Hovering": 4}[
                        item["task_type"]
                    ],
                    -1 if item.get("target_obj_id") is None else item["target_obj_id"],
                ),
            )[0]
            uav.active_task_index = 0
            uav.task_type = primary["task_type"]
            uav.assigned_target_id = primary["target_id"]
            uav.target_position = primary.get("target_pos")


class Task:
    def __init__(self, task_id, task_type, target_obj, target_obj_id):
        self.task_id = task_id
        self.task_type = task_type
        self.target_obj = target_obj
        self.target_obj_id = target_obj_id
        self.target_type = None if target_obj is None else type(target_obj).__name__
        self.is_assigned = False
