"""Deterministic shared Search-region allocation and frontier navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from experiment_config import (
    GLOBAL_SEARCH_COMPLETION_THRESHOLD,
    GROUND_STATION_POSITION_M,
    GS_GATEWAY_HARD_RADIUS_M,
    PERMANENT_GS_GATEWAY_UAV_ID,
    SEARCH_FRONTIER_EPSILON,
    SEARCH_REGION_COMPLETION_THRESHOLD,
    SEARCH_REGION_TARGET_SIZE_M,
    SEARCH_TARGET_ALTITUDE_M,
)


@dataclass(frozen=True)
class SearchRegion:
    region_id: int
    row: int
    column: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int

    @property
    def center(self):
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def total_cells(self):
        return (self.x_stop - self.x_start) * (self.y_stop - self.y_start)

    @property
    def corners(self):
        return (
            (self.xmin, self.ymin),
            (self.xmin, self.ymax),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
        )


def build_search_regions(width_m, height_m, bit_resolution_m, target_size_m=SEARCH_REGION_TARGET_SIZE_M):
    """Partition every bitmap cell exactly once into equal physical regions."""

    width = float(width_m)
    height = float(height_m)
    resolution = float(bit_resolution_m)
    target = float(target_size_m)
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height, resolution, target)):
        raise ValueError("Search region dimensions must be positive and finite")
    cells_x_float, cells_y_float = width / resolution, height / resolution
    cells_x, cells_y = int(round(cells_x_float)), int(round(cells_y_float))
    if not math.isclose(cells_x_float, cells_x, abs_tol=1e-12) or not math.isclose(cells_y_float, cells_y, abs_tol=1e-12):
        raise ValueError("map dimensions must be exact multiples of bit resolution")
    nx, ny = int(math.ceil(width / target)), int(math.ceil(height / target))
    regions = []
    for row in range(ny):
        for column in range(nx):
            x_start = (column * cells_x) // nx
            x_stop = ((column + 1) * cells_x) // nx
            y_start = (row * cells_y) // ny
            y_stop = ((row + 1) * cells_y) // ny
            region = SearchRegion(
                region_id=row * nx + column,
                row=row,
                column=column,
                xmin=column * width / nx,
                xmax=(column + 1) * width / nx,
                ymin=row * height / ny,
                ymax=(row + 1) * height / ny,
                x_start=x_start,
                x_stop=x_stop,
                y_start=y_start,
                y_stop=y_stop,
            )
            if region.total_cells <= 0:
                raise ValueError("Search region contains no bitmap cells")
            regions.append(region)
    ownership = np.zeros((cells_x, cells_y), dtype=np.int16)
    for region in regions:
        ownership[region.x_start:region.x_stop, region.y_start:region.y_stop] += 1
    if not np.all(ownership == 1):
        raise AssertionError("Search region cells have gaps or overlaps")
    return tuple(regions)


def _minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Search cost criteria must be finite and non-empty")
    span = float(values.max() - values.min())
    if span <= SEARCH_FRONTIER_EPSILON:
        return np.zeros_like(values)
    return (values - values.min()) / span


def entropy_weights(*criteria):
    """Return finite entropy-dispersion weights for candidate criteria."""

    if not criteria or any(len(values) != len(criteria[0]) for values in criteria):
        raise ValueError("entropy criteria must have equal non-zero lengths")
    count = len(criteria[0])
    if count == 0:
        raise ValueError("entropy criteria must not be empty")
    gains = []
    for values in criteria:
        normalized = _minmax(values)
        total = float(normalized.sum())
        if total <= SEARCH_FRONTIER_EPSILON or count == 1:
            gains.append(0.0)
            continue
        probabilities = normalized / total
        entropy = -float(np.sum(probabilities * np.log(probabilities + SEARCH_FRONTIER_EPSILON))) / math.log(count)
        gains.append(max(0.0, 1.0 - entropy))
    gain_sum = float(sum(gains))
    if gain_sum <= SEARCH_FRONTIER_EPSILON:
        return tuple(1.0 / len(gains) for _ in gains)
    weights = tuple(float(gain / gain_sum) for gain in gains)
    if not np.isfinite(weights).all():
        raise FloatingPointError("Search entropy weights are not finite")
    return weights


class SearchPathManager:
    """Own Search UAV movement while leaving service UAVs to policy control."""

    def __init__(self, env):
        self.env = env
        self.regions = build_search_regions(
            env.env_width, env.env_height, env.bit_resolution
        )
        self.owner_by_region = {}
        self.active_region_by_uav = {}
        self.visit_counts = {region.region_id: 0 for region in self.regions}
        self.last_active_set = frozenset()
        self.last_commands = {}
        self._had_unfinished_assignment = {}
        self.last_reallocation_reason = None
        self.reallocation_count = 0
        self._effective_visit_keys = set()

    def region_coverage_ratio(self, region):
        patch = self.env.visited_bitmap[
            region.x_start:region.x_stop, region.y_start:region.y_stop
        ]
        if patch.size != region.total_cells or patch.size == 0:
            raise AssertionError("Search region bitmap mapping is invalid")
        ratio = float(patch.mean())
        if not math.isfinite(ratio):
            raise FloatingPointError("Search region coverage is not finite")
        return ratio

    def region_unknown_area(self, region):
        patch = self.env.visited_bitmap[
            region.x_start:region.x_stop, region.y_start:region.y_stop
        ]
        return float(np.count_nonzero(~patch) * self.env.bit_resolution**2)

    def region_completed(self, region):
        return self.region_coverage_ratio(region) >= SEARCH_REGION_COMPLETION_THRESHOLD

    def unfinished_regions(self):
        return tuple(region for region in self.regions if not self.region_completed(region))

    def globally_completed(self):
        ratio = float(np.asarray(self.env.visited_bitmap, dtype=bool).mean())
        if not math.isfinite(ratio):
            raise FloatingPointError("global Search coverage is not finite")
        return ratio >= GLOBAL_SEARCH_COMPLETION_THRESHOLD

    def active_search_uavs(self):
        active = []
        for uav_id in range(self.env.num_UAV):
            roles = tuple(task.get("task_type") for task in self.env.multi_tasks.get(uav_id, ()))
            role_set = set(roles)
            if "Search" in role_set and role_set & {"FOV", "COM"}:
                raise RuntimeError(f"UAV {uav_id} has illegal Search/service role overlap")
            if role_set == {"Search"}:
                active.append(uav_id)
        return tuple(active)

    def _gateway_region_feasible(self, region):
        gs = np.asarray(getattr(self.env, "GS_pos", GROUND_STATION_POSITION_M), dtype=float)
        for x, y in region.corners:
            distance = float(np.linalg.norm(np.asarray((x, y, SEARCH_TARGET_ALTITUDE_M)) - gs))
            if distance > GS_GATEWAY_HARD_RADIUS_M + SEARCH_FRONTIER_EPSILON:
                return False
        return True

    def region_feasible(self, uav_id, region):
        return uav_id != PERMANENT_GS_GATEWAY_UAV_ID or self._gateway_region_feasible(region)

    def _candidate_pairs(self, remaining, active, workloads):
        pairs = []
        for region in remaining:
            unknown_area = self.region_unknown_area(region)
            center = np.asarray(region.center, dtype=float)
            for uav_id in active:
                if not self.region_feasible(uav_id, region):
                    continue
                position = np.asarray(self.env.uav_dict[uav_id].get_position()[:2], dtype=float)
                pairs.append((
                    uav_id,
                    region,
                    float(np.linalg.norm(position - center)),
                    float(workloads[uav_id] + unknown_area),
                ))
        return pairs

    def _reallocate(self, active, reason):
        unfinished = {region.region_id: region for region in self.unfinished_regions()}
        locked = {}
        for uav_id, region_id in sorted(self.active_region_by_uav.items()):
            if uav_id in active and region_id in unfinished and self.owner_by_region.get(region_id) == uav_id:
                locked[region_id] = uav_id
        self.owner_by_region = dict(locked)
        workloads = {uav_id: 0.0 for uav_id in active}
        for region_id, uav_id in locked.items():
            workloads[uav_id] += self.region_unknown_area(unfinished[region_id])
        remaining = [region for region in unfinished.values() if region.region_id not in locked]
        while remaining:
            pairs = self._candidate_pairs(remaining, active, workloads)
            if not pairs:
                break
            distances = [pair[2] for pair in pairs]
            loads = [pair[3] for pair in pairs]
            normalized_distance = _minmax(distances)
            normalized_load = _minmax(loads)
            weight_distance, weight_load = entropy_weights(distances, loads)
            ranked = []
            for index, (uav_id, region, _distance, _load) in enumerate(pairs):
                cost = weight_distance * normalized_distance[index] + weight_load * normalized_load[index]
                if not math.isfinite(float(cost)):
                    raise FloatingPointError("Search allocation cost is not finite")
                ranked.append((float(cost), region.region_id, uav_id, region))
            _cost, region_id, uav_id, region = min(ranked, key=lambda item: item[:3])
            self.owner_by_region[region_id] = uav_id
            workloads[uav_id] += self.region_unknown_area(region)
            remaining = [candidate for candidate in remaining if candidate.region_id != region_id]
        for region in unfinished.values():
            feasible = [uav_id for uav_id in active if self.region_feasible(uav_id, region)]
            if feasible and region.region_id not in self.owner_by_region:
                raise RuntimeError(f"unfinished Search region {region.region_id} has no owner")
        self.active_region_by_uav = {
            uav_id: region_id
            for uav_id, region_id in self.active_region_by_uav.items()
            if self.owner_by_region.get(region_id) == uav_id
        }
        self.last_reallocation_reason = str(reason)
        self.reallocation_count += 1

    def _assigned_unfinished(self, uav_id):
        return tuple(
            region
            for region in self.regions
            if self.owner_by_region.get(region.region_id) == uav_id
            and not self.region_completed(region)
        )

    def _select_active_region(self, uav_id):
        candidates = self._assigned_unfinished(uav_id)
        if not candidates:
            self.active_region_by_uav.pop(uav_id, None)
            return None
        position = np.asarray(self.env.uav_dict[uav_id].get_position()[:2], dtype=float)
        scored = []
        for region in candidates:
            unknown_fraction = 1.0 - self.region_coverage_ratio(region)
            distance = float(np.linalg.norm(position - np.asarray(region.center)))
            score = unknown_fraction / (distance + SEARCH_FRONTIER_EPSILON)
            score /= 1.0 + self.visit_counts[region.region_id]
            if not math.isfinite(score):
                raise FloatingPointError("Search region priority is not finite")
            scored.append((-score, region.region_id, region))
        region = min(scored, key=lambda item: item[:2])[2]
        self.active_region_by_uav[uav_id] = region.region_id
        return region

    def _cell_position(self, x_index, y_index):
        resolution = float(self.env.bit_resolution)
        return ((x_index + 0.5) * resolution, (y_index + 0.5) * resolution)

    def _frontier_cells(self, region):
        bitmap = np.asarray(self.env.visited_bitmap, dtype=bool)
        frontier = []
        for x_index in range(region.x_start, region.x_stop):
            for y_index in range(region.y_start, region.y_stop):
                if bitmap[x_index, y_index]:
                    continue
                x0, x1 = max(0, x_index - 1), min(bitmap.shape[0], x_index + 2)
                y0, y1 = max(0, y_index - 1), min(bitmap.shape[1], y_index + 2)
                if bitmap[x0:x1, y0:y1].any():
                    frontier.append((x_index, y_index))
        return tuple(frontier)

    def _frontier_candidates(self, region):
        frontier = set(self._frontier_cells(region))
        components = []
        while frontier:
            seed = min(frontier, key=lambda cell: cell[0] * self.env.map_height + cell[1])
            frontier.remove(seed)
            stack, component = [seed], [seed]
            while stack:
                x_index, y_index = stack.pop()
                neighbors = sorted(
                    (
                        (x_index + dx, y_index + dy)
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        if (dx or dy) and (x_index + dx, y_index + dy) in frontier
                    ),
                    key=lambda cell: cell[0] * self.env.map_height + cell[1],
                )
                for neighbor in neighbors:
                    frontier.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
            components.append(tuple(component))
        candidates = []
        for component in components:
            centroid = np.asarray(component, dtype=float).mean(axis=0)
            cell = min(
                component,
                key=lambda item: (
                    float(np.linalg.norm(np.asarray(item) - centroid)),
                    item[0] * self.env.map_height + item[1],
                ),
            )
            candidates.append((cell, self._cell_position(*cell)))
        return tuple(candidates)

    def _initial_candidate(self, uav_id, region):
        position = np.asarray(self.env.uav_dict[uav_id].get_position()[:2], dtype=float)
        inside = region.xmin <= position[0] <= region.xmax and region.ymin <= position[1] <= region.ymax
        target = np.asarray(region.center if inside else (
            np.clip(position[0], region.xmin, region.xmax),
            np.clip(position[1], region.ymin, region.ymax),
        ))
        resolution = float(self.env.bit_resolution)
        x_index = int(np.clip(math.floor(target[0] / resolution), region.x_start, region.x_stop - 1))
        y_index = int(np.clip(math.floor(target[1] / resolution), region.y_start, region.y_stop - 1))
        return ((x_index, y_index), self._cell_position(x_index, y_index))

    def _gateway_waypoint_feasible(self, uav_id, waypoint):
        if uav_id != PERMANENT_GS_GATEWAY_UAV_ID:
            return True
        point = np.asarray((*waypoint, SEARCH_TARGET_ALTITUDE_M), dtype=float)
        gs = np.asarray(getattr(self.env, "GS_pos", GROUND_STATION_POSITION_M), dtype=float)
        return float(np.linalg.norm(point - gs)) <= GS_GATEWAY_HARD_RADIUS_M + SEARCH_FRONTIER_EPSILON

    def _select_waypoint(self, uav_id, region):
        candidates = self._frontier_candidates(region)
        if not candidates:
            candidates = (self._initial_candidate(uav_id, region),)
        candidates = tuple(candidate for candidate in candidates if self._gateway_waypoint_feasible(uav_id, candidate[1]))
        if not candidates:
            raise RuntimeError(f"Search UAV {uav_id} has no legal waypoint in region {region.region_id}")
        for (x_index, y_index), point in candidates:
            if not (region.x_start <= x_index < region.x_stop and region.y_start <= y_index < region.y_stop):
                raise RuntimeError("frontier candidate lies outside its Search region")
            if not (0.0 <= point[0] <= self.env.env_width and 0.0 <= point[1] <= self.env.env_height):
                raise RuntimeError("frontier candidate lies outside the map")
        if len(candidates) == 1:
            return candidates[0][1]
        position = np.asarray(self.env.uav_dict[uav_id].get_position()[:2], dtype=float)
        deltas = np.asarray([np.asarray(point) - position for _cell, point in candidates])
        distances = np.linalg.norm(deltas, axis=1)
        normalized_distances = _minmax(distances)
        velocity = np.asarray(self.last_commands.get(uav_id, (0.0, 0.0, 0.0))[:2], dtype=float)
        if float(np.linalg.norm(velocity)) <= SEARCH_FRONTIER_EPSILON:
            ranked = [(float(normalized_distances[index]), candidates[index][0], candidates[index][1]) for index in range(len(candidates))]
        else:
            angles = []
            for delta, distance in zip(deltas, distances):
                if distance <= SEARCH_FRONTIER_EPSILON:
                    angles.append(0.0)
                else:
                    cosine = float(np.dot(delta, velocity) / (distance * np.linalg.norm(velocity)))
                    angles.append(math.acos(float(np.clip(cosine, -1.0, 1.0))) / math.pi)
            normalized_angles = _minmax(angles)
            sigma_distance = float(np.std(distances))
            sigma_angle = float(np.std(angles))
            q_distance = 1.0 / (sigma_distance + SEARCH_FRONTIER_EPSILON)
            q_angle = 1.0 / (sigma_angle + SEARCH_FRONTIER_EPSILON)
            weight_distance = q_distance / (q_distance + q_angle)
            weight_angle = q_angle / (q_distance + q_angle)
            ranked = [
                (
                    float(weight_distance * normalized_distances[index] + weight_angle * normalized_angles[index]),
                    candidates[index][0],
                    candidates[index][1],
                )
                for index in range(len(candidates))
            ]
        if not np.isfinite([entry[0] for entry in ranked]).all():
            raise FloatingPointError("Search frontier cost is not finite")
        return min(ranked, key=lambda item: (item[0], item[1][0] * self.env.map_height + item[1][1]))[2]

    def plan_interval(self, interval_index):
        active = self.active_search_uavs()
        active_set = frozenset(active)
        unfinished = self.unfinished_regions()
        completed_ids = {region.region_id for region in self.regions if self.region_completed(region)}
        if completed_ids & set(self.owner_by_region):
            for region_id in completed_ids:
                self.owner_by_region.pop(region_id, None)
        for uav_id, region_id in tuple(self.active_region_by_uav.items()):
            if region_id in completed_ids:
                self.active_region_by_uav.pop(uav_id, None)
        active_set_changed = active_set != self.last_active_set
        exhausted = bool(unfinished) and any(
            self._had_unfinished_assignment.get(uav_id, False)
            and not self._assigned_unfinished(uav_id)
            for uav_id in active
        )
        if active_set_changed or exhausted:
            self._reallocate(active, "active_set_changed" if active_set_changed else "assigned_regions_completed")
        self.last_active_set = active_set
        commands = {}
        for uav_id in active:
            region = self._select_active_region(uav_id)
            if region is None:
                commands[uav_id] = np.zeros(3, dtype=np.float64)
                continue
            waypoint = np.asarray(self._select_waypoint(uav_id, region), dtype=float)
            position = np.asarray(self.env.uav_dict[uav_id].get_position(), dtype=float)
            horizontal_delta = waypoint - position[:2]
            distance = float(np.linalg.norm(horizontal_delta))
            speed = min(10.0, distance / 1.0)
            horizontal = np.zeros(2) if distance <= SEARCH_FRONTIER_EPSILON else horizontal_delta / distance * speed
            vertical = float(np.clip(SEARCH_TARGET_ALTITUDE_M - position[2], -2.0, 2.0))
            command = np.asarray((horizontal[0], horizontal[1], vertical), dtype=np.float64)
            if not np.isfinite(command).all() or np.linalg.norm(command[:2]) > 10.0 + 1e-12 or abs(command[2]) > 2.0 + 1e-12:
                raise RuntimeError("Search movement command violates speed limits")
            commands[uav_id] = command
        self.last_commands = {uav_id: tuple(command) for uav_id, command in commands.items()}
        self._had_unfinished_assignment = {
            uav_id: bool(self._assigned_unfinished(uav_id)) for uav_id in active
        }
        return commands

    def apply_control(self, policy_commands, interval_index):
        commands = np.asarray(policy_commands, dtype=np.float64).copy()
        if commands.shape != (self.env.num_UAV, 3) or not np.isfinite(commands).all():
            raise ValueError("movement policy commands are invalid")
        search_commands = self.plan_interval(interval_index)
        active = set(self.active_search_uavs())
        if active != set(search_commands):
            raise RuntimeError("Search UAV has no unique Search Path Manager control source")
        for uav_id, command in search_commands.items():
            commands[uav_id] = command
        return commands

    def replay_action(self, executed_action, movement_mask, hover_action):
        action = np.asarray(executed_action, dtype=np.float32).reshape(self.env.num_UAV, 3).copy()
        mask = np.asarray(movement_mask, dtype=bool)
        if mask.shape != (self.env.num_UAV,):
            raise ValueError("movement replay mask has invalid shape")
        action[~mask] = np.asarray(hover_action, dtype=np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError("movement replay action is not finite")
        return action.reshape(-1)

    def record_effective_visits(self, transitions, visited_before, interval_index):
        snapshot = np.asarray(visited_before, dtype=bool)
        if snapshot.shape != self.env.visited_bitmap.shape:
            raise ValueError("effective-visit snapshot shape mismatch")
        for transition in transitions:
            if not transition.coverage_contributor or transition.current_footprint is None:
                continue
            bx_min, bx_max, by_min, by_max = transition.current_footprint
            for region in self.regions:
                x0, x1 = max(bx_min, region.x_start), min(bx_max + 1, region.x_stop)
                y0, y1 = max(by_min, region.y_start), min(by_max + 1, region.y_stop)
                if x0 >= x1 or y0 >= y1 or not (~snapshot[x0:x1, y0:y1]).any():
                    continue
                key = (int(interval_index), int(transition.uav_id), region.region_id)
                if key not in self._effective_visit_keys:
                    self._effective_visit_keys.add(key)
                    self.visit_counts[region.region_id] += 1
