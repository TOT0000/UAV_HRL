"""Single authoritative 3-D communication-range contract."""

from __future__ import annotations

import math


MAX_3D_COMMUNICATION_DISTANCE_M = 400.0
COMMUNICATION_RANGE_BOUNDARY_RULE = "euclidean_3d_distance_le_400m"
COMMUNICATION_LINK_TYPES = ("S2U", "U2G", "U2U")


def _finite_point_3d(position, name):
    try:
        point = tuple(float(value) for value in position)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-D position") from exc
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must be a finite 3-D position")
    return point


def normalized_gs_progress(
    sender_position,
    receiver_position,
    ground_station_position,
    *,
    is_wait=False,
):
    """Return canonical soft progress toward the GS for one routing action.

    ``receiver_position=None`` denotes the ground station itself. Wait is
    handled explicitly so its value is exactly zero rather than a floating
    point subtraction artifact.
    """

    sender = _finite_point_3d(sender_position, "sender_position")
    ground_station = _finite_point_3d(
        ground_station_position, "ground_station_position"
    )
    if bool(is_wait):
        return 0.0
    receiver_distance = (
        0.0
        if receiver_position is None
        else math.dist(
            _finite_point_3d(receiver_position, "receiver_position"),
            ground_station,
        )
    )
    sender_distance = math.dist(sender, ground_station)
    progress = (
        sender_distance - receiver_distance
    ) / MAX_3D_COMMUNICATION_DISTANCE_M
    return float(min(max(progress, -1.0), 1.0))


def validate_communication_range_aliases(*ranges):
    """Fail fast unless every compatibility alias equals the canonical range."""

    canonical = float(MAX_3D_COMMUNICATION_DISTANCE_M)
    if not math.isfinite(canonical) or canonical <= 0.0:
        raise RuntimeError(
            "canonical communication distance must be positive and finite"
        )
    values = tuple(float(value) for value in ranges)
    if any(
        not math.isfinite(value)
        or not math.isclose(value, canonical, rel_tol=0.0, abs_tol=0.0)
        for value in values
    ):
        raise RuntimeError(
            "communication-range aliases diverged from the canonical 400 m "
            f"contract: canonical={canonical}, aliases={values}"
        )
    return canonical
