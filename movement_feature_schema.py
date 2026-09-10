"""Authoritative local feature ordering for centralized UAV movement."""


MOVEMENT_FEATURE_SCHEMA_VERSION = 6

TASK_TYPES = ("Search", "FOV", "COM", "Relay", "Hovering")
ACTIVE_MOVEMENT_TASK_TYPES = ("Search", "FOV", "COM", "Relay")

LOCAL_MOVEMENT_FEATURES = (
    ("task_search", "binary", 0.0, 1.0, "Search task is active"),
    ("task_fov", "binary", 0.0, 1.0, "FOV task is active"),
    ("task_com", "binary", 0.0, 1.0, "COM task is active"),
    ("task_relay", "binary", 0.0, 1.0, "Relay task is active"),
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
    ("relay_target_dx", "continuous", -1.0, 1.0, "virtual target relative x / environment width; zero without Relay"),
    ("relay_target_dy", "continuous", -1.0, 1.0, "virtual target relative y / environment height; zero without Relay"),
    ("relay_target_dz", "continuous", -1.0, 1.0, "virtual target relative z / UAV altitude span; zero without Relay"),
)

LOCAL_MOVEMENT_DIM = len(LOCAL_MOVEMENT_FEATURES)
