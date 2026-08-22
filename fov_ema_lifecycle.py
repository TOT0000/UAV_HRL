"""Shared fail-closed validation for serialized FOV EMA lifecycle state."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

from experiment_config import FOV_EMA_LIFECYCLE_VERSION


FOV_EMA_FIELDS = ("overlap", "unvisited", "frontier")


def _fail(message):
    raise RuntimeError(f"checkpoint FOV EMA lifecycle {message}")


def _uav_id(value, *, num_uav, field):
    if isinstance(value, bool):
        _fail(f"has invalid UAV ID in {field}: {value!r}")
    if isinstance(value, Integral):
        uav_id = int(value)
    elif isinstance(value, str):
        try:
            uav_id = int(value)
        except ValueError:
            _fail(f"has invalid UAV ID in {field}: {value!r}")
    else:
        _fail(f"has invalid UAV ID in {field}: {value!r}")
    if not 0 <= uav_id < int(num_uav):
        _fail(f"has out-of-range UAV ID in {field}: {uav_id}")
    return uav_id


def _id_mapping(value, *, num_uav, field):
    if not isinstance(value, Mapping):
        _fail(f"requires {field} to be a mapping")
    normalized = {}
    for raw_uav_id, item in value.items():
        uav_id = _uav_id(raw_uav_id, num_uav=num_uav, field=field)
        if uav_id in normalized:
            _fail(f"has duplicate or inconsistent UAV IDs in {field}: {uav_id}")
        normalized[uav_id] = item
    return normalized


def _initialized_ids(value, *, num_uav):
    if not isinstance(value, (list, tuple)):
        _fail("requires initialized_uav_ids to be a list")
    normalized = set()
    for raw_uav_id in value:
        uav_id = _uav_id(
            raw_uav_id,
            num_uav=num_uav,
            field="initialized_uav_ids",
        )
        if uav_id in normalized:
            _fail(f"has duplicate initialized UAV ID: {uav_id}")
        normalized.add(uav_id)
    return normalized


def _ema_values(value, *, num_uav):
    records = _id_mapping(value, num_uav=num_uav, field="values")
    normalized = {}
    for uav_id, record in records.items():
        if not isinstance(record, Mapping):
            _fail(f"has non-mapping EMA values for UAV {uav_id}")
        missing = [field for field in FOV_EMA_FIELDS if field not in record]
        if missing:
            _fail(f"is missing EMA values for UAV {uav_id}: {missing}")
        values = {}
        for field in FOV_EMA_FIELDS:
            raw_value = record[field]
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                _fail(f"has non-numeric {field} EMA for UAV {uav_id}")
            numeric = float(raw_value)
            if not math.isfinite(numeric):
                _fail(f"has non-finite {field} EMA for UAV {uav_id}")
            values[field] = numeric
        normalized[uav_id] = values
    return normalized


def _footprint(value, *, uav_id):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        _fail(f"has invalid previous-footprint length for UAV {uav_id}")
    indices = []
    for raw_index in value:
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            _fail(
                f"has non-integer or non-finite previous-footprint index "
                f"for UAV {uav_id}"
            )
        indices.append(int(raw_index))
    bx_min, bx_max, by_min, by_max = indices
    if bx_min > bx_max or by_min > by_max:
        _fail(f"has invalid previous-footprint bounds for UAV {uav_id}")
    return tuple(indices)


def _previous_footprints(value, *, num_uav):
    records = _id_mapping(
        value,
        num_uav=num_uav,
        field="previous_footprints",
    )
    return {
        uav_id: _footprint(footprint, uav_id=uav_id)
        for uav_id, footprint in records.items()
    }


def _marker(value, *, field):
    if value is not None and (not isinstance(value, str) or not value):
        _fail(f"has invalid {field}")
    return value


def validate_fov_ema_state(state, *, num_uav):
    """Validate and normalize a complete v3 FOV lifecycle checkpoint state."""

    if not isinstance(state, Mapping):
        _fail("state is missing")
    if state.get("lifecycle_version") != FOV_EMA_LIFECYCLE_VERSION:
        _fail("version is incompatible")
    values = _ema_values(state.get("values"), num_uav=num_uav)
    initialized = _initialized_ids(
        state.get("initialized_uav_ids"), num_uav=num_uav
    )
    previous = _previous_footprints(
        state.get("previous_footprints"), num_uav=num_uav
    )
    raw_count = state.get("update_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
        _fail("has invalid update_count")
    update_count = int(raw_count)
    if update_count < 0:
        _fail("has negative update_count")
    transition_marker = _marker(
        state.get("transition_marker"), field="transition_marker"
    )
    footprint_marker = _marker(
        state.get("footprint_transition_marker"),
        field="footprint_transition_marker",
    )

    value_ids = set(values)
    previous_ids = set(previous)
    if value_ids != initialized or previous_ids != initialized:
        _fail(
            "has inconsistent initialized, values, or previous-footprint UAV IDs"
        )

    if not initialized:
        if (
            values
            or previous
            or update_count != 0
            or transition_marker is not None
            or footprint_marker is not None
        ):
            _fail("has a non-empty payload for an uninitialized state")
    else:
        expected_ids = set(range(int(num_uav)))
        if initialized != expected_ids:
            _fail("is only partially initialized")
        if update_count <= 0:
            _fail("has initialized values with no EMA updates")
        if transition_marker is None or footprint_marker is None:
            _fail("has initialized values without transition markers")

    return {
        "values": values,
        "initialized_uav_ids": initialized,
        "previous_footprints": previous,
        "transition_marker": transition_marker,
        "footprint_transition_marker": footprint_marker,
        "update_count": update_count,
    }
