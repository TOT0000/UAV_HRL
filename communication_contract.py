"""Single authoritative 3-D communication-range contract."""

from __future__ import annotations

import math


MAX_3D_COMMUNICATION_DISTANCE_M = 400.0
COMMUNICATION_RANGE_BOUNDARY_RULE = "euclidean_3d_distance_le_400m"
COMMUNICATION_LINK_TYPES = ("S2U", "U2G", "U2U")


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
