import math

import numpy as np
import torch
from scipy.integrate import quad

from experiment_config import NUM_UAV, ROI_COUNT_MAX
from Fov_model_phase import FovModel


TASK_TYPES = ("Search", "FOV", "COM", "Hovering")
LOCAL_MOVEMENT_DIM = 17
COVERAGE_GRID_SIZE = 16
GLOBAL_MOVEMENT_DIM = 3
MOVEMENT_STATE_DIM = (
    NUM_UAV * LOCAL_MOVEMENT_DIM + COVERAGE_GRID_SIZE**2 + GLOBAL_MOVEMENT_DIM
)
JOINT_ACTION_DIM = NUM_UAV * 3
BACKLOG_NORM_REF_BITS = 5e7
GT_COUNT_MAX = ROI_COUNT_MAX
VS_COVERAGE_EPS = 1e-6
HOVER_ACTION = (-1.0, 0.0, 0.0)
MOVEMENT_FEATURE_SCHEMA_VERSION = 3

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
        "reference S2U capacity / fixed best-feasible S2U capacity",
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
            ("found_gt_ratio", f"discovered GT count / {GT_COUNT_MAX}"),
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
        raise AssertionError(
            f"movement feature schema does not cover {MOVEMENT_STATE_DIM} dimensions"
        )
    continuous = [item["index"] for item in features if item["kind"] == "continuous"]
    discrete = [item["index"] for item in features if item["kind"] == "binary"]
    return {
        "schema_version": MOVEMENT_FEATURE_SCHEMA_VERSION,
        "dimension": MOVEMENT_STATE_DIM,
        "ordering": (
            f"{NUM_UAV} UAV blocks x {LOCAL_MOVEMENT_DIM}, "
            f"{COVERAGE_GRID_SIZE}x{COVERAGE_GRID_SIZE} coverage macro map "
            f"row-major, {GLOBAL_MOVEMENT_DIM} globals"
        ),
        "features": features,
        "continuous_indices": continuous,
        "discrete_indices": discrete,
    }


def projected_joint_action_schema():
    """Return the authoritative NUM_UAV-by-3 actor-action ordering."""

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
            "speed/vertical clamp to [-1,1], heading wraps periodically to [-1,1), "
            "then task-inactive UAV blocks become hover [-1,0,0]"
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


def normalized_com_link_quality(env, uav_id, task):
    sr_id = _target_object_id(task, "COM")
    utility = float(env.get_sr_uav_normalized_utility(uav_id, sr_id))
    if not math.isfinite(utility) or not 0.0 <= utility <= 1.0:
        raise RuntimeError("canonical normalized COM link quality is invalid")
    return utility


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
            com_capacity_norm = normalized_com_link_quality(env, uav_id, task)

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
    found_count = int(env.count_found_targets())
    remaining_time = float(remaining_time)
    if not math.isfinite(remaining_time) or not 0.0 <= remaining_time <= 1.0:
        raise ValueError(
            f"remaining_time must be finite and within [0, 1], got {remaining_time}"
        )
    global_scalars = np.asarray(
        [
            float(np.asarray(env.visited_bitmap, dtype=bool).mean()),
            float(np.clip(found_count / GT_COUNT_MAX, 0.0, 1.0)),
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


def validate_movement_mask(movement_mask):
    """Validate an explicit per-UAV projection mask without shape inference."""

    if torch.is_tensor(movement_mask):
        if movement_mask.ndim not in (1, 2) or movement_mask.shape[-1] != NUM_UAV:
            raise ValueError(
                f"movement mask must have shape ({NUM_UAV},) or (batch, {NUM_UAV})"
            )
        if movement_mask.dtype == torch.bool:
            return movement_mask
        if not (
            torch.is_floating_point(movement_mask)
            or movement_mask.dtype
            in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("movement mask must be Boolean or numeric 0/1 values")
        if not torch.isfinite(movement_mask).all():
            raise ValueError("movement mask contains NaN or Inf")
        if not torch.logical_or(movement_mask == 0, movement_mask == 1).all():
            raise ValueError("movement mask numeric values must be exactly 0 or 1")
        return movement_mask.bool()

    mask = np.asarray(movement_mask)
    if mask.ndim not in (1, 2) or mask.shape[-1] != NUM_UAV:
        raise ValueError(
            f"movement mask must have shape ({NUM_UAV},) or (batch, {NUM_UAV})"
        )
    if mask.dtype == np.bool_:
        return mask
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError("movement mask must be Boolean or numeric 0/1 values")
    if not np.isfinite(mask).all():
        raise ValueError("movement mask contains NaN or Inf")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("movement mask numeric values must be exactly 0 or 1")
    return mask.astype(bool, copy=False)


def project_action_domain(raw_action):
    """Project speed/heading/vertical blocks without applying task masks."""

    if torch.is_tensor(raw_action):
        if raw_action.shape[-1] % 3 != 0 or not torch.isfinite(raw_action).all():
            raise ValueError("movement action must contain finite 3-value blocks")
        blocks = raw_action.reshape(*raw_action.shape[:-1], -1, 3)
        heading = blocks[..., 1]
        wrapped_heading = torch.where(
            torch.logical_and(heading >= -1.0, heading < 1.0),
            heading,
            torch.remainder(heading + 1.0, 2.0) - 1.0,
        )
        return torch.stack(
            (
                blocks[..., 0].clamp(-1.0, 1.0),
                wrapped_heading,
                blocks[..., 2].clamp(-1.0, 1.0),
            ),
            dim=-1,
        ).reshape_as(raw_action)

    action = np.asarray(raw_action, dtype=np.float32)
    if (
        action.shape[-1] % 3 != 0
        or not np.isfinite(action).all()
    ):
        raise ValueError("movement action must contain finite 3-value blocks")
    blocks = action.reshape(*action.shape[:-1], -1, 3).copy()
    blocks[..., 0] = np.clip(blocks[..., 0], -1.0, 1.0)
    heading = blocks[..., 1]
    blocks[..., 1] = np.where(
        np.logical_and(heading >= -1.0, heading < 1.0),
        heading,
        np.remainder(heading + 1.0, 2.0) - 1.0,
    )
    blocks[..., 2] = np.clip(blocks[..., 2], -1.0, 1.0)
    return blocks.reshape(action.shape)


def project_joint_action(raw_action, movement_state=None, *, movement_mask=None):
    """Apply fieldwise domain projection followed by the task movement mask.

    Existing callers may provide an unmasked movement state. Training callers
    should provide the explicit per-UAV true ``movement_mask`` stored with
    the transition. Exactly one control source is required.
    """

    if (movement_state is None) == (movement_mask is None):
        raise ValueError(
            "provide exactly one of movement_state or explicit movement_mask"
        )
    movable = (
        movement_mask_from_state(movement_state)
        if movement_mask is None
        else validate_movement_mask(movement_mask)
    )
    if torch.is_tensor(raw_action):
        if raw_action.shape[-1] != JOINT_ACTION_DIM:
            raise ValueError(f"joint action must end in {JOINT_ACTION_DIM} values")
        if not torch.is_tensor(movable):
            movable = torch.as_tensor(movable, device=raw_action.device)
        else:
            movable = movable.to(device=raw_action.device)
        if raw_action.shape[:-1] != movable.shape[:-1]:
            raise ValueError(
                "joint action batch dimensions must match movement mask batch dimensions"
            )
        projected_blocks = project_action_domain(raw_action).reshape(
            *raw_action.shape[:-1], NUM_UAV, 3
        )
        mask = movable.to(dtype=raw_action.dtype).unsqueeze(-1)
        hover = raw_action.new_tensor(HOVER_ACTION)
        projected = projected_blocks * mask + hover * (1.0 - mask)
        return projected.reshape_as(raw_action)

    action_array = np.asarray(raw_action, dtype=np.float32)
    if action_array.shape[-1] != JOINT_ACTION_DIM:
        raise ValueError(f"joint action must end in {JOINT_ACTION_DIM} values")
    if torch.is_tensor(movable):
        movable = movable.detach().cpu().numpy()
    movable = np.asarray(movable, dtype=bool)
    if action_array.shape[:-1] != movable.shape[:-1]:
        raise ValueError(
            "joint action batch dimensions must match movement mask batch dimensions"
        )
    blocks = project_action_domain(action_array).reshape(
        *action_array.shape[:-1], NUM_UAV, 3
    ).copy()
    blocks[~movable] = np.asarray(HOVER_ACTION, dtype=np.float32)
    return blocks.reshape(action_array.shape)


def project_local_action(raw_action):
    """Canonical projection for one decoded speed/heading/vertical block."""

    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (3,):
        raise ValueError("local movement action must be three finite values")
    return project_action_domain(action)


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
                normalized_com_link_quality(env, uav_id, task)
            )
    phi_vs = float(np.mean(vs_progress)) if vs_progress else 0.0
    phi_com = float(np.mean(com_progress)) if com_progress else 0.0
    return phi_search, phi_vs, phi_com


def decode_joint_velocity_commands(model, projected_joint_action):
    action = np.asarray(projected_joint_action, dtype=np.float32)
    if action.shape != (JOINT_ACTION_DIM,):
        raise ValueError(f"joint action must have shape ({JOINT_ACTION_DIM},), got {action.shape}")
    commands = np.asarray(
        [
            model.decode_action(action[uav_id * 3 : (uav_id + 1) * 3])
            for uav_id in range(NUM_UAV)
        ],
        dtype=np.float64,
    )
    if commands.shape != (NUM_UAV, 3) or not np.isfinite(commands).all():
        raise RuntimeError("decoded joint velocity command is invalid")
    return commands


def build_velocity_substep_proposals(env, velocity_commands, step_time):
    commands = np.asarray(velocity_commands, dtype=np.float64)
    if commands.shape != (NUM_UAV, 3) or not np.isfinite(commands).all():
        raise ValueError(f"velocity commands must have shape ({NUM_UAV}, 3)")
    step_time = float(step_time)
    if not np.isfinite(step_time) or step_time <= 0.0:
        raise ValueError("movement substep must be positive and finite")
    proposals = []
    for uav_id in range(NUM_UAV):
        proposal = env.uav_dict[uav_id].propose_movement(
            *commands[uav_id],
            step_time=step_time,
            mobility_params=env.mobility_params,
            env_width=env.env_width,
            env_height=env.env_height,
        )
        proposals.append(proposal)
    return proposals


def build_joint_movement_proposals(
    env, model, projected_joint_action, step_time=1.0
):
    """Compatibility facade for one proposal batch from a joint action."""

    return build_velocity_substep_proposals(
        env,
        decode_joint_velocity_commands(model, projected_joint_action),
        step_time,
    )


def apply_joint_movement_proposals(env, proposals, step_time=1.0):
    if len(proposals) != NUM_UAV:
        raise ValueError(f"expected {NUM_UAV} movement proposals, got {len(proposals)}")
    energies = []
    for uav_id, proposal in enumerate(proposals):
        energy = env.uav_dict[uav_id].apply_movement_proposal(
            proposal, energy_model=env.energy_model, step_time=step_time
        )
        energies.append(float(energy))
    return np.asarray(energies, dtype=np.float64)
