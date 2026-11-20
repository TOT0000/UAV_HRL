"""Authoritative feature ordering for centralized UAV movement."""

from dataclasses import dataclass


MOVEMENT_FEATURE_SCHEMA_VERSION = 8

TASK_TYPES = ("Search", "FOV", "COM", "Hovering")
# Only service roles are owned by the learned/random movement policy. Search is
# externally controlled and Hovering is canonical zero velocity.
ACTIVE_MOVEMENT_TASK_TYPES = ("FOV", "COM")

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

LOCAL_MOVEMENT_DIM = len(LOCAL_MOVEMENT_FEATURES)

SR_MOVEMENT_FEATURES = (
    ("valid", "binary", 0.0, 1.0, "one when this stable SR-id slot is populated"),
    ("position_x", "continuous", 0.0, 1.0, "SR x / environment width"),
    ("position_y", "continuous", 0.0, 1.0, "SR y / environment height"),
    ("position_z", "continuous", 0.0, 1.0, "SR z / UAV max AGL"),
    ("target_x", "continuous", 0.0, 1.0, "known assigned RoI x / environment width"),
    ("target_y", "continuous", 0.0, 1.0, "known assigned RoI y / environment height"),
    ("target_z", "continuous", 0.0, 1.0, "known assigned RoI z / UAV max AGL"),
    ("mode_direct", "binary", 0.0, 1.0, "S2G range is currently satisfied"),
    ("mode_support", "binary", 0.0, 1.0, "outside S2G with assigned in-range COM UAV"),
    ("mode_waiting", "binary", 0.0, 1.0, "outside S2G without usable S2U"),
    ("fifo_backlog", "continuous", 0.0, 1.0, "log1p(SR FIFO bits) / log1p(5e7)"),
)
SR_MOVEMENT_DIM = len(SR_MOVEMENT_FEATURES)


# These fixed extents are part of the movement observation contract.  Keep the
# block layout here so configuration metadata, the TD3 encoder, and LLM named
# projections cannot independently drift when a block is added.
MAX_SR_SLOTS = 8
COVERAGE_GRID_SIZE = 16
GLOBAL_MOVEMENT_DIM = 3


@dataclass(frozen=True)
class MovementStateLayout:
    uav_start: int
    uav_stop: int
    sr_start: int
    sr_stop: int
    coverage_start: int
    coverage_stop: int
    global_start: int
    global_stop: int

    @property
    def dimension(self) -> int:
        return self.global_stop


def movement_state_layout(num_uav: int) -> MovementStateLayout:
    """Return authoritative contiguous block boundaries for one state vector."""

    resolved_num_uav = int(num_uav)
    if resolved_num_uav <= 0:
        raise ValueError("num_uav must be positive")
    uav_start = 0
    uav_stop = resolved_num_uav * LOCAL_MOVEMENT_DIM
    sr_start = uav_stop
    sr_stop = sr_start + MAX_SR_SLOTS * SR_MOVEMENT_DIM
    coverage_start = sr_stop
    coverage_stop = coverage_start + COVERAGE_GRID_SIZE**2
    global_start = coverage_stop
    global_stop = global_start + GLOBAL_MOVEMENT_DIM
    return MovementStateLayout(
        uav_start=uav_start,
        uav_stop=uav_stop,
        sr_start=sr_start,
        sr_stop=sr_stop,
        coverage_start=coverage_start,
        coverage_stop=coverage_stop,
        global_start=global_start,
        global_stop=global_stop,
    )


def movement_state_dimension(num_uav: int) -> int:
    return movement_state_layout(num_uav).dimension
