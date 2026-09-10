"""Snapshot Relay heuristic. No routing decisions, channel samples or RNG here.

Physical IDs sort numerically before virtual IDs (which sort lexicographically).
Endpoint ties use (distance, source-side UAV ID, GS-side UAV ID). BFS uses that
same node ordering. This is deletion-minimal greedy planning, not an optimum.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import combinations, product
import math

import numpy as np
from scipy.optimize import minimize

from Channel_model import (
    reference_u2u_max_capacity_mbps, u2u_capacity_mbps,
    a2g_capacity_mbps, U2U_U2G_TX_POWER_DBM,
)
from communication_contract import MAX_3D_COMMUNICATION_DISTANCE_M as RANGE_M
from experiment_config import RELAY_TASK_CONTRACT_VERSION, TOTAL_COMMUNICATION_BANDWIDTH_HZ

COUNT_RULE = "snapshot_component_bridges_greedy_deletion_minimal"
TRIGGER = "new_roi_at_existing_assignment_boundary_only"
MINIMAX_TOL_M = 1e-6
MINIMAX_MAX_ITER = 200
POSITION_MAX_ITER = 100


def node_key(node):
    return (0, int(node)) if isinstance(node, (int, np.integer)) else (1, str(node))


def initial_relay_count(distance):
    return max(0, math.ceil(float(distance) / RANGE_M) - 1)


def valid_in_air_backlog(packet_engine, current_time):
    """Read only UAV-owned FOV/COM remaining bits using production expiry epsilon."""
    from Packet_scheduler_v1 import PACKET_EPS
    result = {}
    for uid in range(packet_engine.num_UAV):
        result[uid] = math.fsum(
            max(float(p.get("rem_bits", 0.0)), 0.0)
            for p in packet_engine.uav_queues[uid]
            if p and not p.get("done", False)
            and p.get("task_type") in {"FOV", "COM"}
            and (p.get("deadline_abs") is None
                 or float(current_time) < float(p["deadline_abs"]) - PACKET_EPS)
            and math.isfinite(float(p.get("rem_bits", 0.0)))
        )
    return result


def movement_bounds(env):
    # All production UAVs share this box. Intersecting bounds also makes the
    # virtual target legal for every eligible vehicle in heterogeneous fixtures.
    return (np.array([0., 0., max(u.min_AGL for u in env.uav_dict.values())]),
            np.array([env.env_width, env.env_height,
                      min(u.max_AGL for u in env.uav_dict.values())], dtype=float))


def bounded_minimax_center(points, bounds, *, max_iter=MINIMAX_MAX_ITER):
    """Convex epigraph solve; exact support-sphere enumeration is the fallback.

    A 3-D minimax center has at most four supporting points plus box faces.
    The fallback enumerates these supports on the box's 27 faces/interior;
    this is unrelated to enumerating Relay subsets. It also handles GS neighbors
    below the movement altitude floor.
"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    low, high = bounds
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("minimax requires finite neighbor positions")
    if max_iter > 0 and len(points) <= 2 and np.all(points >= low) and np.all(points <= high):
        center = points.mean(axis=0)  # Exact minimax for one/two legal points.
        return center, {"converged": True, "fallback": False,
                        "fallback_method": None, "method": "analytic_one_two_points",
                        "iterations": 0, "iteration_cap": max_iter,
                        "tolerance_m": MINIMAX_TOL_M,
                        "radius_m": float(np.linalg.norm(points-center,axis=1).max())}
    start = np.clip(points.mean(axis=0), low, high)
    radius = float(np.linalg.norm(points - start, axis=1).max())
    result = minimize(
        lambda x: x[3], np.r_[start, radius], method="SLSQP",
        bounds=list(zip(low, high)) + [(0., None)],
        constraints=[{"type": "ineq", "fun": lambda x:
                      x[3] - np.linalg.norm(points - x[:3], axis=1)}],
        options={"ftol": 1e-10, "maxiter": max_iter},
    )
    center = np.clip(result.x[:3], low, high)
    converged = bool(result.success and np.isfinite(result.x).all()
                     and np.max(np.linalg.norm(points - center, axis=1))
                     <= result.x[3] + MINIMAX_TOL_M)
    # The numerical iterate must not be worse than its deterministic feasible
    # start (particularly on a chain whose two hops are exactly 400 m).
    if converged and np.linalg.norm(points-center,axis=1).max() > radius:
        center = start
    if not converged:
        candidates = []
        for face in product((-1, 0, 1), repeat=3):
            free = np.array([axis for axis in range(3) if face[axis] == 0], dtype=int)
            fixed = np.array([axis for axis in range(3) if face[axis] != 0], dtype=int)
            base = np.array([low[a] if face[a] == -1 else high[a] for a in range(3)])
            for size in range(1, min(len(free) + 1, len(points)) + 1):
                for indices in combinations(range(len(points)), size):
                    subset = points[list(indices)]
                    delta = subset[1:, free] - subset[0, free]
                    perpendicular = ((subset[:, fixed] - base[fixed])**2).sum(axis=1)
                    rhs = ((delta * delta).sum(axis=1) + perpendicular[1:] - perpendicular[0]) / 2
                    offset = np.linalg.lstsq(delta, rhs, rcond=None)[0] if len(delta) else np.zeros(len(free))
                    candidate = base.copy()
                    candidate[free] = subset[0, free] + offset
                    candidate = np.clip(candidate, low, high)
                    enclosing_radius = float(np.linalg.norm(points - candidate, axis=1).max())
                    candidates.append((enclosing_radius, tuple(candidate)))
        center = np.array(min(candidates)[1])
    return center, {"converged": converged, "fallback": not converged,
                    "fallback_method": "deterministic_support_spheres" if not converged else None,
                    "iterations": int(result.nit), "iteration_cap": max_iter,
                    "tolerance_m": MINIMAX_TOL_M,
                    "radius_m": float(np.linalg.norm(points - center, axis=1).max())}


def physical_positions(env):
    # Safe-DDQN exposes every UAV as a next hop; role is deliberately irrelevant.
    return {uid: np.asarray(env.uav_dict[uid].get_position(), dtype=float)
            for uid in range(env.num_UAV)}


def connectivity_graph(env, virtual=None):
    positions = physical_positions(env)
    positions.update(virtual or {})
    graph = {node: set() for node in positions}
    graph[env.GS_ID] = set()
    for first, second in combinations(sorted(positions, key=node_key), 2):
        if math.dist(positions[first], positions[second]) <= RANGE_M:
            graph[first].add(second)
            graph[second].add(first)
    # Apply the same production position eligibility contract to virtual nodes.
    for node, position in positions.items():
        eligible = (env.is_u2g_in_range(node) if isinstance(node, int)
                    else env.is_u2g_position_in_range(position))
        if eligible:
            graph[node].add(env.GS_ID)
            graph[env.GS_ID].add(node)
    return graph


def graph_from_node_positions(env, positions):
    """Build the canonical deterministic graph for an explicit node snapshot."""
    positions = {node: np.asarray(position, dtype=float)
                 for node, position in positions.items()}
    graph = {node: set() for node in positions}
    graph[env.GS_ID] = set()
    for first, second in combinations(sorted(positions, key=node_key), 2):
        if math.dist(positions[first], positions[second]) <= RANGE_M:
            graph[first].add(second)
            graph[second].add(first)
    for node, position in positions.items():
        if env.is_u2g_position_in_range(position):
            graph[node].add(env.GS_ID)
            graph[env.GS_ID].add(node)
    return graph


def reverse_bfs(graph, root):
    parents = {root: None}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph[node], key=node_key):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    return parents


def witness_paths(graph, gs, sources):
    parents = reverse_bfs(graph, gs)
    paths = {}
    for source in sorted(sources):
        path = []
        node = source
        while node is not None and node in parents:
            path.append(node)
            node = parents[node]
        paths[source] = path
    return paths


def empty_plan():
    return {"relay_task_contract_version": RELAY_TASK_CONTRACT_VERSION,
            "relay_count_rule": COUNT_RULE, "reassignment_trigger": TRIGGER,
            "optimality": "deletion-minimal greedy heuristic, not global minimum",
            "snapshot_policy_limitation": "No topology/backlog/fading/movement replanning; future connectivity is not guaranteed",
            "planning_source_ids": [], "actual_backlog_source_ids": [],
            "prospective_source_ids": [], "source_backlog_bits": {},
            "source_components": {}, "gs_component": [], "source_bridges": {},
            "initial_candidates": [], "redundancy_tests": [], "witness_paths": {},
            "merge_validation": [], "required_before_budget": 0,
            "available_relay_uavs": 0, "assigned_relay_count": 0, "shortage": 0,
            "budget_pruning": [], "unsupported_source_ids": [], "unsupported_backlog": 0.,
            "fully_supported_source_ids_after_budget": [],
            "partially_supported_source_ids_after_budget": [],
            "unsupported_source_ids_after_budget": [],
            "supported_backlog_bits_after_budget": 0.,
            "unsupported_backlog_bits_after_budget": 0.,
            "slots": [], "slot_to_uav": {}, "position_status": {},
            "current_source_reachability": {},
            "predicted_post_assignment_reachability": {},
            "predicted_fully_supported_source_ids": [],
            "predicted_partially_supported_source_ids": [],
            "predicted_unsupported_source_ids": [],
            "predicted_supported_backlog_bits": 0.,
            "predicted_unsupported_backlog_bits": 0.,
            "self_neighbor_conflicts": [], "relocated_anchor_conflicts": []}


def attach_witness_metadata(slots, paths):
    for slot in slots:
        sid = slot["slot_id"]
        neighbors, supported = set(), []
        slot_paths = {}
        for source, path in paths.items():
            if sid in path:
                idx = path.index(sid)
                neighbors.update(path[idx-1:idx] + path[idx+1:idx+2])
                supported.append(source)
                slot_paths[str(source)] = list(path)
        slot.update(neighbor_ids=sorted(neighbors, key=node_key),
                    supported_source_ids=sorted(supported), witness_paths=slot_paths,
                    shared=len(set(supported)) >= 2)


def original_chain_witnesses(env, slots, sources):
    """Conservative witness fallback after restoring all original bridges.

    Keep each source on its own original bridge instead of another source's
    shortcut. This removes a numerically/collectively invalid shared merge.
    """
    graph = connectivity_graph(env)
    gs_paths = witness_paths(graph, env.GS_ID, range(env.num_UAV))
    paths = {}
    for source in sources:
        chain = sorted((s for s in slots if s["origin_source_id"] == source),
                       key=lambda s: s["chain_index"])
        if not chain:
            paths[source] = gs_paths[source]
            continue
        left, far = chain[0]["L_s"], chain[0]["F_s"]
        source_path = witness_paths(graph, left, [source])[source]
        paths[source] = source_path + [s["slot_id"] for s in chain] + gs_paths[far]
    return paths


def virtual_positions(env, slots):
    """Pure simultaneous target refresh from immutable identities.

    Restart from bridge interpolation every time, avoiding observation-order
    dependence. Jacobi updates allow shared neighbors to be virtual slots.
    """
    physical = physical_positions(env)
    physical[env.GS_ID] = np.asarray(env.GS_pos, dtype=float)
    bounds = movement_bounds(env)
    positions, status = {}, {}
    for slot in slots:
        fraction = slot["chain_index"] / (slot["chain_count"] + 1)
        raw = physical[slot["L_s"]] + fraction * (physical[slot["F_s"]] - physical[slot["L_s"]])
        positions[slot["slot_id"]] = np.clip(raw, *bounds)
        status[slot["slot_id"]] = {"clipped": bool(np.any(raw != positions[slot["slot_id"]]))}
    converged = False
    for iteration in range(POSITION_MAX_ITER):
        previous = positions
        nodes = {**physical, **previous}
        positions = {}
        for slot in slots:
            sid = slot["slot_id"]
            neighbor_ids = slot.get("active_neighbor_ids_after_budget",
                                    slot["neighbor_ids"])
            neighbors = [nodes[n] for n in neighbor_ids if n in nodes]
            if slot["shared"] and len(neighbors) == len(neighbor_ids) and neighbors:
                positions[sid], solver = bounded_minimax_center(neighbors, bounds)
                status[sid]["minimax"] = solver
            else:
                positions[sid] = previous[sid]
                status[sid]["minimax"] = None
        delta = max((math.dist(positions[k], previous[k]) for k in positions), default=0.)
        if delta <= MINIMAX_TOL_M:
            converged = True
            break
    nodes = {**physical, **positions}
    for slot in slots:
        sid = slot["slot_id"]
        neighbor_ids = slot.get("active_neighbor_ids_after_budget",
                                slot["neighbor_ids"])
        missing = [n for n in neighbor_ids if n not in nodes]
        radius = max((math.dist(positions[sid], nodes[n])
                      for n in neighbor_ids if n in nodes), default=0.)
        budget_missing = list(slot.get("missing_neighbor_ids_after_budget", ()))
        status[sid].update(position=positions[sid].tolist(), radius_m=radius,
                           active_neighbors_feasible=not missing and radius <= RANGE_M,
                           feasible=not missing and not budget_missing and radius <= RANGE_M,
                           missing_neighbor_ids=missing, coupled_converged=converged,
                           coupled_fallback=None if converged else "last_bounded_finite_iterate",
                           coupled_iterations=iteration + 1,
                           coupled_iteration_cap=POSITION_MAX_ITER)
    return positions, status


def rebuild_after_budget_metadata(env, slots, positions, sources, weights):
    """Rebuild final witnesses while retaining incomplete bridge provenance."""
    retained = set(positions)
    final_graph = connectivity_graph(env, positions)
    final_paths = witness_paths(final_graph, env.GS_ID, sources)
    fully = sorted(source for source, path in final_paths.items() if path)
    fully_set = set(fully)
    partial = sorted(source for source in sources if source not in fully_set and any(
        source in slot["supported_source_ids_before_budget"] for slot in slots))
    unsupported = sorted(set(sources) - fully_set - set(partial))
    for slot in slots:
        before_neighbors = list(slot["neighbor_ids_before_budget"])
        active = [neighbor for neighbor in before_neighbors
                  if not isinstance(neighbor, str) or neighbor in retained]
        missing = [neighbor for neighbor in before_neighbors
                   if isinstance(neighbor, str) and neighbor not in retained]
        full_for_slot = [source for source in fully
                         if slot["slot_id"] in final_paths[source]]
        partial_for_slot = [source for source in partial
                            if source in slot["supported_source_ids_before_budget"]]
        if full_for_slot:
            support_status = "full"
        elif partial_for_slot:
            support_status = "partial"
        else:
            support_status = "infeasible"
        slot.update(
            active_neighbor_ids_after_budget=sorted(active, key=node_key),
            missing_neighbor_ids_after_budget=sorted(missing, key=node_key),
            fully_supported_source_ids_after_budget=full_for_slot,
            partially_supported_source_ids_after_budget=partial_for_slot,
            support_status=support_status,
            budget_witness_paths={str(source): list(final_paths[source])
                                  for source in full_for_slot},
        )
    positions, position_status = virtual_positions(env, slots)
    # Rebuild once more at the final updated targets, which is the actual
    # budget-pruned augmented topology used by planning diagnostics.
    final_paths = witness_paths(connectivity_graph(env, positions), env.GS_ID, sources)
    fully = sorted(source for source, path in final_paths.items() if path)
    fully_set = set(fully)
    partial = sorted(source for source in sources if source not in fully_set and any(
        source in slot["supported_source_ids_before_budget"] for slot in slots))
    unsupported = sorted(set(sources) - fully_set - set(partial))
    for slot in slots:
        full_for_slot = [source for source in fully
                         if slot["slot_id"] in final_paths[source]]
        partial_for_slot = [source for source in partial
                            if source in slot["supported_source_ids_before_budget"]]
        slot["fully_supported_source_ids_after_budget"] = full_for_slot
        slot["partially_supported_source_ids_after_budget"] = partial_for_slot
        slot["support_status"] = ("full" if full_for_slot else
                                  "partial" if partial_for_slot else "infeasible")
        slot["budget_witness_paths"] = {
            str(source): list(final_paths[source]) for source in full_for_slot}
        slot["virtual_position"] = positions[slot["slot_id"]].tolist()
    return {
        "positions": positions, "position_status": position_status,
        "witness_paths": final_paths, "fully": fully, "partial": partial,
        "unsupported": unsupported,
        "supported_backlog": math.fsum(weights[source] for source in fully),
        "unsupported_backlog": math.fsum(weights[source] for source in partial + unsupported),
    }


def removal_loss(env, positions, candidate, sources, weights):
    before = reverse_bfs(connectivity_graph(env, positions), env.GS_ID)
    after = reverse_bfs(connectivity_graph(env, {k: v for k, v in positions.items() if k != candidate}), env.GS_ID)
    lost = sorted(s for s in sources if s in before and s not in after)
    return {"slot_id": candidate, "lost_source_ids": lost,
            "backlog_loss": math.fsum(weights.get(s, 0.) for s in lost)}


def plan_relays(env, prospective, available_uavs):
    plan = empty_plan()
    physical = physical_positions(env)
    weights = {uid: float(getattr(env, "assignment_backlog_snapshot", {}).get(uid, 0.))
               for uid in physical}
    weights = {uid: q if math.isfinite(q) and q > 0 else 0. for uid, q in weights.items()}
    actual = sorted(uid for uid, q in weights.items() if q > 0.)
    sources = sorted(set(actual) | set(prospective))
    graph = connectivity_graph(env)
    connected = reverse_bfs(graph, env.GS_ID)
    gs_component = sorted(uid for uid in physical if uid in connected)
    plan.update(planning_source_ids=sources, actual_backlog_source_ids=actual,
                prospective_source_ids=sorted(set(prospective)),
                source_backlog_bits={str(k): weights[k] for k in sources},
                gs_component=gs_component, available_relay_uavs=len(available_uavs))
    slots = []
    for source in sources:
        component = sorted(uid for uid in reverse_bfs(graph, source) if uid in physical)
        plan["source_components"][str(source)] = component
        if source in connected or not gs_component:
            plan["source_bridges"][str(source)] = {
                "connected": source in connected, "no_gs_component": not gs_component,
                "L_s": None, "F_s": None, "endpoint_distance_m": None, "initial_N_s": 0}
            continue
        distance, left, far = min((math.dist(physical[i], physical[j]), i, j)
                                  for i in component for j in gs_component)
        count = initial_relay_count(distance)
        plan["source_bridges"][str(source)] = {"L_s": left, "F_s": far,
                                                "endpoint_distance_m": distance, "initial_N_s": count}
        for q in range(1, count + 1):
            raw = physical[left] + q / (count + 1) * (physical[far] - physical[left])
            position = np.clip(raw, *movement_bounds(env))
            slots.append({"slot_id": f"relay-{source:04d}-{q:04d}",
                          "virtual_position": position.tolist(), "shared": False,
                          "L_s": left, "F_s": far, "chain_index": q, "chain_count": count,
                          "origin_source_id": source, "neighbor_ids": [],
                          "neighbor_ids_before_budget": [],
                          "active_neighbor_ids_after_budget": [],
                          "missing_neighbor_ids_after_budget": [],
                          "supported_source_ids": [], "witness_paths": {},
                          "supported_source_ids_before_budget": [],
                          "fully_supported_source_ids_after_budget": [],
                          "partially_supported_source_ids_after_budget": [],
                          "support_status": "infeasible",
                          "planning_priority": None, "removal_backlog_loss": 0.,
                          "assigned_uav_id": None, "assignment_distance_m": None,
                          "clipped": bool(np.any(raw != position))})
    initial = deepcopy(slots)
    plan["initial_candidates"] = deepcopy(initial)
    positions = {s["slot_id"]: s["virtual_position"] for s in slots}
    removed = []
    while True:
        changed = False
        for sid in sorted(positions):
            trial = {k: v for k, v in positions.items() if k != sid}
            reachable = reverse_bfs(connectivity_graph(env, trial), env.GS_ID)
            unsupported = sorted(s for s in sources if s not in reachable)
            test = removal_loss(env, positions, sid, sources, weights)
            test.update(removed=not unsupported, unsupported_source_ids=unsupported)
            plan["redundancy_tests"].append(test)
            if not unsupported:
                positions = trial
                removed.append(sid)
                changed = True
        if not changed:
            break
    # Restore removed candidates in reverse deletion order on any shared-center
    # failure, rebuilding witnesses each time. Never keep an invalid merge.
    while True:
        slots = [deepcopy(s) for s in initial if s["slot_id"] in positions]
        paths = witness_paths(connectivity_graph(env, positions), env.GS_ID, sources)
        attach_witness_metadata(slots, paths)
        for slot in slots:
            slot["neighbor_ids_before_budget"] = list(slot["neighbor_ids"])
            slot["supported_source_ids_before_budget"] = list(
                slot["supported_source_ids"])
        updated, status = virtual_positions(env, slots)
        reachable = reverse_bfs(connectivity_graph(env, updated), env.GS_ID)
        invalid = [sid for sid, st in status.items()
                   if not st["feasible"] or not st["coupled_converged"]]
        lost = [s for s in sources if paths[s] and s not in reachable]
        valid = not invalid and not lost
        validation = {"valid": valid, "invalid_slot_ids": invalid,
                      "lost_source_ids": lost, "position_status": status,
                      "restored_candidate": None}
        plan["merge_validation"].append(validation)
        if valid:
            positions = updated
            break
        if not removed:
            # Fully restored original chains have a constructive geometric
            # witness. Do not preserve an invalid shortcut merely because the
            # coupled minimax iteration reached its cap.
            paths = original_chain_witnesses(env, slots, sources)
            attach_witness_metadata(slots, paths)
            for slot in slots:
                slot["neighbor_ids_before_budget"] = list(slot["neighbor_ids"])
                slot["supported_source_ids_before_budget"] = list(
                    slot["supported_source_ids"])
            updated, status = virtual_positions(env, slots)
            reachable = reverse_bfs(connectivity_graph(env, updated), env.GS_ID)
            valid = all(st["feasible"] for st in status.values()) and all(
                not path or source in reachable for source, path in paths.items())
            plan["merge_validation"].append({"valid": valid,
                "fallback": "restored_original_chain_witnesses", "position_status": status,
                "restored_candidate": None})
            if not valid:
                raise RuntimeError("legal original Relay bridge geometry could not be validated")
            positions = updated
            break
        restored = removed.pop()
        positions[restored] = next(s["virtual_position"] for s in initial if s["slot_id"] == restored)
        validation["restored_candidate"] = restored
    plan["required_before_budget"] = len(slots)
    # Keep the validated pre-budget identities. Partial bridges remain assigned
    # after pruning, with missing virtual neighbors diagnosed separately.
    while len(positions) > len(available_uavs):
        tests = [removal_loss(env, positions, sid, sources, weights) for sid in sorted(positions)]
        chosen = min(tests, key=lambda t: (t["backlog_loss"], len(t["lost_source_ids"]), t["slot_id"]))
        plan["budget_pruning"].append({**chosen, "candidate_tests": tests})
        del positions[chosen["slot_id"]]
    slots = [s for s in slots if s["slot_id"] in positions]
    rebuilt = rebuild_after_budget_metadata(env, slots, positions, sources, weights)
    positions = rebuilt["positions"]
    for slot in slots:
        loss = removal_loss(env, positions, slot["slot_id"], sources, weights)
        slot.update(virtual_position=list(map(float, positions[slot["slot_id"]])),
                    removal_backlog_loss=loss["backlog_loss"],
                    removal_source_ids=loss["lost_source_ids"],
                    shortage=plan["required_before_budget"] - len(slots),
                    budget_pruned_slot_ids=[t["slot_id"] for t in plan["budget_pruning"]])
    slots.sort(key=lambda s: (-s["removal_backlog_loss"], -len(s["removal_source_ids"]), s["slot_id"]))
    for priority, slot in enumerate(slots):
        slot["planning_priority"] = priority
    final_paths = rebuilt["witness_paths"]
    plan.update(slots=slots, witness_paths={str(k): v for k, v in paths.items()},
                budget_witness_paths={str(k): v for k, v in final_paths.items()},
                assigned_relay_count=len(slots), shortage=plan["required_before_budget"]-len(slots),
                fully_supported_source_ids_after_budget=rebuilt["fully"],
                partially_supported_source_ids_after_budget=rebuilt["partial"],
                unsupported_source_ids_after_budget=rebuilt["unsupported"],
                supported_backlog_bits_after_budget=rebuilt["supported_backlog"],
                unsupported_backlog_bits_after_budget=rebuilt["unsupported_backlog"],
                unsupported_source_ids=rebuilt["partial"] + rebuilt["unsupported"],
                unsupported_backlog=rebuilt["unsupported_backlog"],
                position_status=rebuilt["position_status"])
    return plan


def pair_relay_slots(env, plan, available_uavs, *, random=False):
    free = sorted(set(available_uavs))
    if random:
        env.assignment_rng.shuffle(free)
    for slot in plan["slots"]:
        uid = free[0] if random else min(free, key=lambda u: (
            math.dist(env.uav_dict[u].get_position(), slot["virtual_position"]), u))
        free.remove(uid)
        slot["assigned_uav_id"] = uid
        slot["assignment_distance_m"] = math.dist(env.uav_dict[uid].get_position(), slot["virtual_position"])
        plan["slot_to_uav"][slot["slot_id"]] = uid
    plan.update(predicted_post_assignment_diagnostics(env, plan))


def predicted_post_assignment_diagnostics(env, plan):
    """Predict topology after assigned Relay UAVs reach their targets.

    This function is observational. It replaces each assigned Relay UAV's
    physical coordinate with its virtual target exactly once and never adds an
    anonymous virtual copy. Results cannot change mapping, counts, or tasks.
    """
    sources = list(plan.get("planning_source_ids", ()))
    if not plan["slots"] and not sources:
        return {
            "predicted_post_assignment_reachability": {},
            "predicted_fully_supported_source_ids": [],
            "predicted_partially_supported_source_ids": [],
            "predicted_unsupported_source_ids": [],
            "predicted_supported_backlog_bits": 0.,
            "predicted_unsupported_backlog_bits": 0.,
            "self_neighbor_conflicts": [],
            "relocated_anchor_conflicts": [],
        }
    relay_by_uav = {int(slot["assigned_uav_id"]): slot for slot in plan["slots"]
                    if slot.get("assigned_uav_id") is not None}
    num_uav = getattr(env, "num_UAV", None)
    if num_uav is None:
        num_uav = len(env.uav_dict)
    num_uav = int(num_uav)
    predicted_positions = {
        uid: np.asarray(relay_by_uav[uid]["virtual_position"], dtype=float)
        if uid in relay_by_uav
        else np.asarray(env.uav_dict[uid].get_position(), dtype=float)
        for uid in range(num_uav)
    }
    graph = graph_from_node_positions(env, predicted_positions)
    reachable = reverse_bfs(graph, env.GS_ID)
    fully = sorted(source for source in sources if source in reachable)
    full_set = set(fully)
    retained_support = {
        source for slot in plan["slots"]
        for source in slot.get("supported_source_ids_before_budget", ())}
    partial = sorted(source for source in sources
                     if source not in full_set and source in retained_support)
    unsupported = sorted(set(sources) - full_set - set(partial))
    self_conflicts = []
    relocated_conflicts = []
    for slot in plan["slots"]:
        uid = int(slot["assigned_uav_id"])
        for neighbor in slot.get("active_neighbor_ids_after_budget", ()):
            if neighbor == uid:
                self_conflicts.append({"slot_id": slot["slot_id"],
                                       "assigned_uav_id": uid,
                                       "physical_neighbor_id": int(neighbor)})
            elif isinstance(neighbor, int) and neighbor in relay_by_uav:
                relocated_conflicts.append({
                    "slot_id": slot["slot_id"], "assigned_uav_id": uid,
                    "physical_neighbor_id": int(neighbor),
                    "neighbor_relay_slot_id": relay_by_uav[neighbor]["slot_id"],
                    "neighbor_original_position": list(map(
                        float, env.uav_dict[neighbor].get_position())),
                    "neighbor_predicted_position": list(map(
                        float, predicted_positions[neighbor])),
                })
    source_backlog = plan.get("source_backlog_bits", {})
    weights = {int(source): float(source_backlog.get(
        str(source), source_backlog.get(source, 0.)))
               for source in sources}
    return {
        "predicted_post_assignment_reachability": {
            str(source): source in reachable for source in sources},
        "predicted_fully_supported_source_ids": fully,
        "predicted_partially_supported_source_ids": partial,
        "predicted_unsupported_source_ids": unsupported,
        "predicted_supported_backlog_bits": math.fsum(weights[s] for s in fully),
        "predicted_unsupported_backlog_bits": math.fsum(
            weights[s] for s in partial + unsupported),
        "self_neighbor_conflicts": self_conflicts,
        "relocated_anchor_conflicts": relocated_conflicts,
    }


def relay_potential(env, uid, slot, positions=None):
    plan = getattr(env, "relay_plan", empty_plan())
    if positions is None:
        positions, _ = virtual_positions(env, plan["slots"])
    target = positions.get(slot["slot_id"], slot["virtual_position"])
    point = env.uav_dict[uid].get_position()
    p_pos = math.exp(-math.dist(point, target) / RANGE_M)
    capacities = []
    reference = float(reference_u2u_max_capacity_mbps(TOTAL_COMMUNICATION_BANDWIDTH_HZ))
    for neighbor in slot.get("active_neighbor_ids_after_budget",
                             slot["neighbor_ids"]):
        if neighbor == env.GS_ID:
            capacity = (a2g_capacity_mbps(point, env.GS_pos, TOTAL_COMMUNICATION_BANDWIDTH_HZ,
                                          U2U_U2G_TX_POWER_DBM)
                        if env.is_u2g_position_in_range(point) else 0.)
            capacities.append(float(np.clip(capacity / reference, 0., 1.)))
            continue
        neighbor_uid = neighbor if isinstance(neighbor, int) else plan["slot_to_uav"].get(neighbor)
        if neighbor_uid is None or neighbor_uid == uid:
            capacities.append(0.)
            continue
        neighbor_position = env.uav_dict[neighbor_uid].get_position()
        capacity = (u2u_capacity_mbps(point, neighbor_position, TOTAL_COMMUNICATION_BANDWIDTH_HZ)
                    if math.dist(point, neighbor_position) <= RANGE_M else 0.)
        capacities.append(float(np.clip(capacity / reference, 0., 1.)))
    p_link = min(capacities, default=0.)
    return {"P_pos": p_pos, "P_link": p_link, "Phi_relay": .3*p_pos + .7*p_link,
            "C_ref_mbps": reference}


def relay_snapshot(env):
    """Observational current geometry/potential diagnostics; never mutate plan."""
    snapshot = deepcopy(getattr(env, "relay_plan", empty_plan()))
    positions, status = virtual_positions(env, snapshot["slots"])
    reachable = reverse_bfs(connectivity_graph(env), env.GS_ID)
    snapshot["current_source_reachability"] = {
        str(s): s in reachable for s in snapshot["planning_source_ids"]}
    snapshot["position_status"] = status
    for slot in snapshot["slots"]:
        slot["virtual_position"] = positions[slot["slot_id"]].tolist()
        if slot["assigned_uav_id"] is not None:
            slot.update(relay_potential(env, slot["assigned_uav_id"], slot, positions))
    return snapshot


def relay_position_snapshot(env):
    """Compact movement observation; immutable planning details live in events."""
    plan = getattr(env, "relay_plan", empty_plan())
    positions, status = virtual_positions(env, plan["slots"])
    reachable = reverse_bfs(connectivity_graph(env), env.GS_ID)
    return {
        "assignment_invocation": int(getattr(env, "assignment_invocations", 0)),
        "position_status": status,
        "current_source_reachability": {
            str(s): s in reachable for s in plan["planning_source_ids"]},
        "slots": [{"slot_id": slot["slot_id"],
                   "assigned_uav_id": slot["assigned_uav_id"],
                   "virtual_position": positions[slot["slot_id"]].tolist(),
                   **relay_potential(env, slot["assigned_uav_id"], slot, positions)}
                  for slot in plan["slots"]],
    }


def refresh_relay_targets(env):
    """Boundary target update only; counts, anchors, slot IDs and owners persist."""
    plan = getattr(env, "relay_plan", None)
    if plan is None:
        return
    positions, status = virtual_positions(env, plan["slots"])
    plan["position_status"] = status
    for slot in plan["slots"]:
        sid, uid = slot["slot_id"], slot["assigned_uav_id"]
        slot["virtual_position"] = positions[sid].tolist()
        for task in env.multi_tasks.get(uid, []):
            if task.get("target_id") == sid:
                task["target_pos"] = tuple(positions[sid])
        if uid is not None:
            env.uav_dict[uid].target_position = tuple(positions[sid])
