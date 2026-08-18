import math

import numpy as np
import torch
from scipy.integrate import quad

from Fov_model_phase import FovModel


NUM_UAV = 16
TASK_TYPES = ("Search", "FOV", "COM", "Hovering")
LOCAL_MOVEMENT_DIM = 17
MOVEMENT_STATE_DIM = 532
JOINT_ACTION_DIM = NUM_UAV * 3
BACKLOG_NORM_REF_BITS = 5e7
GT_COUNT_MAX = 10
COVERAGE_GRID_SIZE = 16
VS_COVERAGE_EPS = 1e-6
HOVER_ACTION = (-1.0, 0.0, 0.0)
MOVEMENT_FEATURE_SCHEMA_VERSION = 1

LOCAL_MOVEMENT_FEATURES = (
    ("task_search", "binary", 0.0, 1.0, "Search task is active"),
    ("task_fov", "binary", 0.0, 1.0, "FOV task is active"),
    ("task_com", "binary", 0.0, 1.0, "COM task is active"),
    ("task_hovering", "binary", 0.0, 1.0, "Hovering task is active"),
    ("position_x", "continuous", 0.0, 1.0, "x / environment width"),
    ("position_y", "continuous", 0.0, 1.0, "y / environment height"),
    (
        "position_z",
        "continuous",
        0.0,
        1.0,
        "(z - UAV min AGL) / (UAV max AGL - UAV min AGL)",
    ),
    ("energy", "continuous", 0.0, 1.0, "remaining energy / E_max"),
    (
        "backlog",
        "continuous",
        0.0,
        1.0,
        "log1p(non-negative backlog bits) / log1p(5e7 bits)",
    ),
    (
        "fov_error",
        "continuous",
        -1.0,
        1.0,
        "clip((FOV image score - 1) / 3, -1, 1); zero without FOV task",
    ),
    ("fov_target_x", "continuous", 0.0, 1.0, "FOV target x / width"),
    ("fov_target_y", "continuous", 0.0, 1.0, "FOV target y / height"),
    ("fov_target_z", "continuous", 0.0, 1.0, "FOV target z / UAV max AGL"),
    ("com_target_x", "continuous", 0.0, 1.0, "COM target x / width"),
    ("com_target_y", "continuous", 0.0, 1.0, "COM target y / height"),
    ("com_target_z", "continuous", 0.0, 1.0, "COM target z / UAV max AGL"),
    (
        "com_capacity",
        "continuous",
        0.0,
        1.0,
        "non-negative COM capacity Mbps / calibrated c_ref_com",
    ),
)


def movement_state_feature_schema():
    """Return the authoritative fixed ordering used by get_global_movement_state."""

    features = []
    for uav_id in range(NUM_UAV):
        base = uav_id * LOCAL_MOVEMENT_DIM
        for local_offset, (name, kind, minimum, maximum, normalization) in enumerate(
            LOCAL_MOVEMENT_FEATURES
        ):
            features.append(
                {
                    "index": base + local_offset,
                    "name": f"uav_{uav_id}.{name}",
                    "kind": kind,
                    "minimum": minimum,
                    "maximum": maximum,
                    "normalization": normalization,
                }
            )
    coverage_base = NUM_UAV * LOCAL_MOVEMENT_DIM
    for row in range(COVERAGE_GRID_SIZE):
        for column in range(COVERAGE_GRID_SIZE):
            features.append(
                {
                    "index": coverage_base + row * COVERAGE_GRID_SIZE + column,
                    "name": f"coverage_macro[{row},{column}]",
                    "kind": "continuous",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "normalization": "mean of boolean visited cells in macro grid cell",
                }
            )
    for offset, (name, normalization) in enumerate(
        (
            ("global_coverage", "mean of boolean visited bitmap"),
            ("found_gt_ratio", "found GT count / current GT count"),
            ("num_gt", "clip(current GT count / 10, 0, 1)"),
            ("remaining_time", "remaining movement intervals / episode duration"),
        )
    ):
        features.append(
            {
                "index": coverage_base + COVERAGE_GRID_SIZE**2 + offset,
                "name": name,
                "kind": "continuous",
                "minimum": 0.0,
                "maximum": 1.0,
                "normalization": normalization,
            }
        )
    if len(features) != MOVEMENT_STATE_DIM:
        raise AssertionError("movement feature schema does not cover 532 dimensions")
    continuous = [item["index"] for item in features if item["kind"] == "continuous"]
    discrete = [item["index"] for item in features if item["kind"] == "binary"]
    return {
        "schema_version": MOVEMENT_FEATURE_SCHEMA_VERSION,
        "dimension": MOVEMENT_STATE_DIM,
        "ordering": "16 UAV blocks x 17, 16x16 coverage macro map row-major, 4 globals",
        "features": features,
        "continuous_indices": continuous,
        "discrete_indices": discrete,
    }


def projected_joint_action_schema():
    """Return the authoritative 16-by-3 projected raw actor-action ordering."""

    components = (
        ("speed_scalar", "decoded to horizontal speed in [0, 10] m/s"),
        ("heading_scalar", "decoded to heading in [-pi, pi] radians"),
        ("vertical_scalar", "decoded to vertical speed in [-2, 2] m/s"),
    )
    features = []
    for uav_id in range(NUM_UAV):
        for local_offset, (name, semantics) in enumerate(components):
            features.append(
                {
                    "index": uav_id * 3 + local_offset,
                    "name": f"uav_{uav_id}.{name}",
                    "kind": "continuous",
                    "minimum": -1.0,
                    "maximum": 1.0,
                    "semantics": semantics,
                }
            )
    return {
        "dimension": JOINT_ACTION_DIM,
        "ordering": "UAV id ascending; speed, heading, vertical scalar",
        "range": [-1.0, 1.0],
        "projection": (
            "task-inactive UAV blocks are replaced by hover raw action [-1, 0, 0]"
        ),
        "features": features,
        "continuous_indices": list(range(JOINT_ACTION_DIM)),
        "discrete_indices": [],
    }


def aggregate_coverage_map(visited_bitmap):
    bitmap = np.asarray(visited_bitmap, dtype=bool)
    if bitmap.ndim != 2:
        raise ValueError(f"visited_bitmap must be 2-D, got shape {bitmap.shape}")

    row_groups = np.array_split(np.arange(bitmap.shape[0]), COVERAGE_GRID_SIZE)
    col_groups = np.array_split(np.arange(bitmap.shape[1]), COVERAGE_GRID_SIZE)
    if any(group.size == 0 for group in (*row_groups, *col_groups)):
        raise ValueError(
            f"visited_bitmap {bitmap.shape} is too small for "
            f"{COVERAGE_GRID_SIZE}x{COVERAGE_GRID_SIZE} aggregation"
        )

    macro = np.empty((COVERAGE_GRID_SIZE, COVERAGE_GRID_SIZE), dtype=np.float32)
    covered_cells = 0
    for row_idx, rows in enumerate(row_groups):
        for col_idx, cols in enumerate(col_groups):
            patch = bitmap[np.ix_(rows, cols)]
            covered_cells += patch.size
            macro[row_idx, col_idx] = float(patch.mean())

    if covered_cells != bitmap.size:
        raise AssertionError(
            f"coverage aggregation lost cells: {covered_cells} != {bitmap.size}"
        )
    return macro.reshape(-1, order="C")


def _tasks_by_type(env, uav_id):
    grouped = {task_type: [] for task_type in TASK_TYPES}
    for task in env.multi_tasks.get(uav_id, []):
        task_type = task.get("task_type")
        if task_type in grouped:
            grouped[task_type].append(task)
    return grouped


def _assert_unique_target_tasks(uav_id, grouped_tasks):
    for task_type in ("FOV", "COM"):
        tasks = grouped_tasks[task_type]
        if len(tasks) > 1:
            target_ids = [
                task.get("target_obj_id", task.get("target_id")) for task in tasks
            ]
            raise ValueError(
                f"UAV {uav_id} has duplicate {task_type} targets: {target_ids}"
            )


def _target_object_id(task, task_type):
    if "target_obj_id" not in task:
        raise ValueError(
            f"{task_type} task is missing target_obj_id metadata: "
            f"target_id={task.get('target_id')}"
        )
    return int(task["target_obj_id"])


def _com_capacity_mbps(env, uav_id, task):
    sr_id = _target_object_id(task, "COM")
    capacity = float(env.get_sr_uav_capacity_mbps(uav_id, sr_id))
    if not math.isfinite(capacity):
        return 0.0
    return max(capacity, 0.0)


def get_global_movement_state(
    env, packet_engine, backlog_bits, c_ref_com, remaining_time
):
    if env.num_UAV != NUM_UAV:
        raise ValueError(f"centralized movement requires {NUM_UAV} UAVs, got {env.num_UAV}")
    if float(c_ref_com) <= 0 or not math.isfinite(float(c_ref_com)):
        raise ValueError(f"c_ref_com must be positive and finite, got {c_ref_com}")

    if backlog_bits is None:
        backlog_bits = packet_engine.backlog_bits

    local_features = []
    for uav_id in range(NUM_UAV):
        uav = env.uav_dict[uav_id]
        grouped = _tasks_by_type(env, uav_id)
        _assert_unique_target_tasks(uav_id, grouped)

        task_flags = [1.0 if grouped[name] else 0.0 for name in TASK_TYPES]

        z_min = float(getattr(uav, "min_AGL", 50.0))
        z_max = float(getattr(uav, "max_AGL", 200.0))
        z_span = max(z_max - z_min, 1e-9)
        position = [
            float(np.clip(uav.x_u / float(env.env_width), 0.0, 1.0)),
            float(np.clip(uav.y_u / float(env.env_height), 0.0, 1.0)),
            float(np.clip((uav.z_u - z_min) / z_span, 0.0, 1.0)),
        ]

        energy_norm = float(np.clip(uav.energy / float(env.E_max), 0.0, 1.0))
        queue_bits = max(float(backlog_bits.get(uav_id, 0.0)), 0.0)
        backlog_norm = float(
            np.clip(
                np.log1p(queue_bits) / np.log1p(BACKLOG_NORM_REF_BITS),
                0.0,
                1.0,
            )
        )

        fov_error = 0.0
        fov_target = [0.0, 0.0, 0.0]
        if grouped["FOV"]:
            task = grouped["FOV"][0]
            tx, ty, tz = map(float, task["target_pos"])
            fov_model = FovModel(
                f=0.004, wl=0.008, i_l=0.012, z_u=float(uav.z_u), gamma_g=80
            )
            image_score, _ = fov_model.calculate_fov_single(
                float(uav.x_u), float(uav.y_u), float(uav.z_u), tx, ty, tz
            )
            if math.isfinite(float(image_score)):
                fov_error = float(np.clip((float(image_score) - 1.0) / 3.0, -1.0, 1.0))
            fov_target = [
                float(np.clip(tx / float(env.env_width), 0.0, 1.0)),
                float(np.clip(ty / float(env.env_height), 0.0, 1.0)),
                float(np.clip(tz / max(z_max, 1e-9), 0.0, 1.0)),
            ]

        com_target = [0.0, 0.0, 0.0]
        com_capacity_norm = 0.0
        if grouped["COM"]:
            task = grouped["COM"][0]
            sr_id = _target_object_id(task, "COM")
            tx, ty, tz = map(float, env.SR_teams[sr_id].get_position())
            com_target = [
                float(np.clip(tx / float(env.env_width), 0.0, 1.0)),
                float(np.clip(ty / float(env.env_height), 0.0, 1.0)),
                float(np.clip(tz / max(z_max, 1e-9), 0.0, 1.0)),
            ]
            com_capacity_norm = float(
                np.clip(_com_capacity_mbps(env, uav_id, task) / float(c_ref_com), 0.0, 1.0)
            )

        uav_features = np.asarray(
            task_flags
            + position
            + [energy_norm, backlog_norm, fov_error]
            + fov_target
            + com_target
            + [com_capacity_norm],
            dtype=np.float32,
        )
        if uav_features.shape != (LOCAL_MOVEMENT_DIM,):
            raise AssertionError(
                f"UAV {uav_id} movement features have shape {uav_features.shape}"
            )
        local_features.append(uav_features)

    compressed_map = aggregate_coverage_map(env.visited_bitmap)
    num_gt = int(getattr(env, "num_GT", len(getattr(env, "gts", []))) or 0)
    found_count = int(env.count_found_targets()) if num_gt > 0 else 0
    remaining_time = float(remaining_time)
    if not math.isfinite(remaining_time) or not 0.0 <= remaining_time <= 1.0:
        raise ValueError(
            f"remaining_time must be finite and within [0, 1], got {remaining_time}"
        )
    global_scalars = np.asarray(
        [
            float(np.asarray(env.visited_bitmap, dtype=bool).mean()),
            float(found_count / num_gt) if num_gt > 0 else 0.0,
            float(np.clip(num_gt / GT_COUNT_MAX, 0.0, 1.0)),
            remaining_time,
        ],
        dtype=np.float32,
    )

    global_state = np.concatenate(local_features + [compressed_map, global_scalars]).astype(
        np.float32, copy=False
    )
    if global_state.shape != (MOVEMENT_STATE_DIM,):
        raise AssertionError(
            f"global movement state has shape {global_state.shape}, "
            f"expected ({MOVEMENT_STATE_DIM},)"
        )
    if not np.isfinite(global_state).all():
        raise ValueError("global movement state contains NaN or Inf")
    return global_state


def movement_mask_from_state(state):
    if torch.is_tensor(state):
        if state.shape[-1] != MOVEMENT_STATE_DIM:
            raise ValueError(f"movement state must end in {MOVEMENT_STATE_DIM} features")
        local = state[..., : NUM_UAV * LOCAL_MOVEMENT_DIM].reshape(
            *state.shape[:-1], NUM_UAV, LOCAL_MOVEMENT_DIM
        )
        return local[..., :3].amax(dim=-1) > 0.5

    state_array = np.asarray(state)
    if state_array.shape[-1] != MOVEMENT_STATE_DIM:
        raise ValueError(f"movement state must end in {MOVEMENT_STATE_DIM} features")
    local = state_array[..., : NUM_UAV * LOCAL_MOVEMENT_DIM].reshape(
        *state_array.shape[:-1], NUM_UAV, LOCAL_MOVEMENT_DIM
    )
    return np.max(local[..., :3], axis=-1) > 0.5


def project_joint_action(raw_action, movement_state):
    movable = movement_mask_from_state(movement_state)
    if torch.is_tensor(raw_action):
        if raw_action.shape[-1] != JOINT_ACTION_DIM:
            raise ValueError(f"joint action must end in {JOINT_ACTION_DIM} values")
        action_blocks = raw_action.reshape(*raw_action.shape[:-1], NUM_UAV, 3)
        mask = movable.to(dtype=raw_action.dtype).unsqueeze(-1)
        hover = raw_action.new_tensor(HOVER_ACTION)
        projected = action_blocks * mask + hover * (1.0 - mask)
        return projected.reshape_as(raw_action)

    action_array = np.asarray(raw_action, dtype=np.float32)
    if action_array.shape[-1] != JOINT_ACTION_DIM:
        raise ValueError(f"joint action must end in {JOINT_ACTION_DIM} values")
    blocks = action_array.reshape(*action_array.shape[:-1], NUM_UAV, 3).copy()
    blocks[~movable] = np.asarray(HOVER_ACTION, dtype=np.float32)
    return blocks.reshape(action_array.shape)


def _circle_rectangle_intersection_area(cx, cy, radius, xmin, xmax, ymin, ymax):
    if radius <= 0 or xmin >= xmax or ymin >= ymax:
        return 0.0
    if xmax <= cx - radius or xmin >= cx + radius:
        return 0.0
    if ymax <= cy - radius or ymin >= cy + radius:
        return 0.0
    if xmin <= cx - radius and xmax >= cx + radius and ymin <= cy - radius and ymax >= cy + radius:
        return math.pi * radius * radius

    x0 = max(xmin, cx - radius)
    x1 = min(xmax, cx + radius)

    def vertical_overlap(x_value):
        half_height = math.sqrt(max(radius * radius - (x_value - cx) ** 2, 0.0))
        lower = max(ymin, cy - half_height)
        upper = min(ymax, cy + half_height)
        return max(upper - lower, 0.0)

    area, _ = quad(vertical_overlap, x0, x1, epsabs=1e-7, epsrel=1e-7, limit=100)
    return float(np.clip(area, 0.0, math.pi * radius * radius))


def fov_task_metrics(env, uav_id, task):
    uav = env.uav_dict[uav_id]
    tx, ty, tz = map(float, task["target_pos"])
    target_obj_id = _target_object_id(task, "FOV")
    target = env.gts[target_obj_id]
    radius = float(getattr(target, "radius", 80.0))
    model = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=float(uav.z_u), gamma_g=radius)
    fov_width, fov_height = model.get_ground_fov_size(float(uav.z_u))
    image_score, _ = model.calculate_fov_single(
        float(uav.x_u), float(uav.y_u), float(uav.z_u), tx, ty, tz
    )

    z_min = float(getattr(uav, "min_AGL", 50.0))
    z_max = float(getattr(uav, "max_AGL", 200.0))
    geometry_valid = bool(
        z_min <= float(uav.z_u) <= z_max
        and float(uav.z_u) > tz
        and math.isfinite(float(fov_width))
        and math.isfinite(float(fov_height))
        and fov_width > 0
        and fov_height > 0
        and math.isfinite(float(image_score))
    )
    if not geometry_valid:
        return 0.0, 0.0, False

    xmin = float(uav.x_u) - fov_width / 2.0
    xmax = float(uav.x_u) + fov_width / 2.0
    ymin = float(uav.y_u) - fov_height / 2.0
    ymax = float(uav.y_u) + fov_height / 2.0
    overlap_area = _circle_rectangle_intersection_area(
        tx, ty, radius, xmin, xmax, ymin, ymax
    )
    roi_area = math.pi * radius * radius
    coverage_ratio = float(np.clip(overlap_area / max(roi_area, 1e-12), 0.0, 1.0))
    return coverage_ratio, float(image_score), True


def vs_data_valid(env, uav_id, task):
    coverage_ratio, image_score, geometry_valid = fov_task_metrics(
        env, uav_id, task
    )
    return bool(
        geometry_valid
        and math.isfinite(coverage_ratio)
        and math.isfinite(image_score)
        and coverage_ratio >= 1.0 - VS_COVERAGE_EPS
        and 0.0 < image_score <= 1.0 + VS_COVERAGE_EPS
    )


def calculate_movement_potentials(env, c_ref_com):
    phi_search = float(np.asarray(env.visited_bitmap, dtype=bool).mean())
    vs_progress = []
    com_progress = []
    for uav_id in range(env.num_UAV):
        grouped = _tasks_by_type(env, uav_id)
        _assert_unique_target_tasks(uav_id, grouped)
        for task in grouped["FOV"]:
            coverage, image_score, geometry_valid = fov_task_metrics(env, uav_id, task)
            progress = coverage * float(np.clip(image_score, 0.0, 1.0)) if geometry_valid else 0.0
            vs_progress.append(progress)
        for task in grouped["COM"]:
            com_progress.append(
                float(np.clip(_com_capacity_mbps(env, uav_id, task) / float(c_ref_com), 0.0, 1.0))
            )
    phi_vs = float(np.mean(vs_progress)) if vs_progress else 0.0
    phi_com = float(np.mean(com_progress)) if com_progress else 0.0
    return phi_search, phi_vs, phi_com


def build_joint_movement_proposals(env, model, projected_joint_action):
    action = np.asarray(projected_joint_action, dtype=np.float32)
    if action.shape != (JOINT_ACTION_DIM,):
        raise ValueError(f"joint action must have shape ({JOINT_ACTION_DIM},), got {action.shape}")
    proposals = []
    for uav_id in range(NUM_UAV):
        decoded = model.decode_action(action[uav_id * 3 : (uav_id + 1) * 3])
        proposal = env.uav_dict[uav_id].propose_movement(
            *decoded,
            step_time=1.0,
            mobility_params=env.mobility_params,
            env_width=env.env_width,
            env_height=env.env_height,
        )
        proposals.append(proposal)
    return proposals


def apply_joint_movement_proposals(env, proposals):
    if len(proposals) != NUM_UAV:
        raise ValueError(f"expected {NUM_UAV} movement proposals, got {len(proposals)}")
    energies = []
    for uav_id, proposal in enumerate(proposals):
        energy = env.uav_dict[uav_id].apply_movement_proposal(
            proposal, energy_model=env.energy_model, step_time=1.0
        )
        energies.append(float(energy))
    return np.asarray(energies, dtype=np.float64)
