"""Authoritative local feature ordering for centralized UAV movement."""


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
