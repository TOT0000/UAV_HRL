"""Streaming Search-coverage diagnostics for environment-size evaluation.

``search_uav_ids`` is a legacy field name.  It contains every nadir Search
coverage contributor (Search, COM-only, and Hover), not only UAVs whose task
type is Search; FOV and FOV+COM UAVs are excluded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


SEARCH_DIAGNOSTICS_SCHEMA_VERSION = "uav-hrl-search-diagnostics-v2"
SEARCH_DIAGNOSTICS_INDEXING = {
    "episode_index": "zero_based",
    "interval_index": "zero_based",
    "time_seconds": "post_search_footprint_commit",
}

_REQUIRED_FIELDS = (
    "schema_version",
    "method_id",
    "scenario_id",
    "episode_index",
    "interval_index",
    "time_seconds",
    "coverage_ratio_before",
    "coverage_ratio_after",
    "discovered_roi_count_before",
    "discovered_roi_count_after",
    "newly_discovered_roi_ids",
    "search_uav_ids",
    "search_uav_count",
    "search_uavs",
    "gross_footprint_cell_count",
    "union_footprint_cell_count",
    "new_union_cell_count",
    "simultaneous_overlap_cell_count",
    "simultaneous_overlap_ratio",
    "historical_revisit_cell_count",
    "historical_revisit_ratio",
)


def _position(mapping, uav_id):
    value = mapping[uav_id]
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Search diagnostic position is invalid for UAV {uav_id}")
    return array


def build_search_diagnostics_record(
    *,
    method_id,
    scenario_id,
    episode_index,
    interval_index,
    time_seconds,
    visited_before,
    visited_after,
    discovered_roi_ids_before,
    discovered_roi_ids_after,
    search_uav_ids,
    footprint_transitions,
    footprint_transition_batches=(),
    interval_initial_positions,
    interval_final_positions,
):
    """Build one row from frozen bitmap state and per-subslot footprints.

    ``search_uav_ids`` retains its legacy name for reader compatibility but
    means Search coverage contributor IDs.  Each transition batch is one
    simultaneous 0.25-second subslot sample.
    """

    before = np.asarray(visited_before, dtype=bool)
    after = np.asarray(visited_after, dtype=bool)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("Search diagnostic bitmaps must be matching 2-D arrays")

    search_ids = sorted({int(value) for value in search_uav_ids})
    transition_by_uav = {
        int(transition.uav_id): transition
        for transition in footprint_transitions
        if bool(transition.coverage_contributor)
    }
    if set(transition_by_uav) != set(search_ids):
        raise ValueError("Search diagnostic contributors disagree with transitions")

    batches = tuple(footprint_transition_batches) + (tuple(footprint_transitions),)
    for batch in batches:
        contributors = {
            int(transition.uav_id)
            for transition in batch
            if bool(transition.coverage_contributor)
        }
        if contributors != set(search_ids):
            raise ValueError("Search diagnostic contributor set changed within interval")

    union_mask = np.zeros(before.shape, dtype=bool) if search_ids else None
    uav_masks = {
        uav_id: np.zeros(before.shape, dtype=bool) for uav_id in search_ids
    }
    footprint_samples_by_uav = {uav_id: 0 for uav_id in search_ids}
    gross_count = 0
    simultaneous_overlap_count = 0
    for batch in batches:
        batch_union_mask = np.zeros(before.shape, dtype=bool)
        batch_gross_count = 0
        transition_by_uav = {
            int(transition.uav_id): transition
            for transition in batch
            if bool(transition.coverage_contributor)
        }
        for uav_id in search_ids:
            footprint = transition_by_uav[uav_id].current_footprint
            if footprint is None:
                continue
            bx_min, bx_max, by_min, by_max = map(int, footprint)
            footprint_count = (
                (bx_max - bx_min + 1) * (by_max - by_min + 1)
            )
            footprint_samples_by_uav[uav_id] += footprint_count
            batch_gross_count += footprint_count
            batch_union_mask[
                bx_min : bx_max + 1, by_min : by_max + 1
            ] = True
            uav_masks[uav_id][
                bx_min : bx_max + 1, by_min : by_max + 1
            ] = True
        batch_union_count = int(np.count_nonzero(batch_union_mask))
        gross_count += batch_gross_count
        simultaneous_overlap_count += batch_gross_count - batch_union_count
        if union_mask is not None:
            union_mask |= batch_union_mask

    per_uav = []
    for uav_id in search_ids:
        uav_mask = uav_masks[uav_id]
        footprint_count = footprint_samples_by_uav[uav_id]
        new_count = int(np.count_nonzero(uav_mask & ~before))
        initial = _position(interval_initial_positions, uav_id)
        final = _position(interval_final_positions, uav_id)
        per_uav.append(
            {
                "uav_id": uav_id,
                "position_xyz_m": [float(value) for value in final],
                "displacement_m": float(np.linalg.norm(final - initial)),
                "footprint_cell_count": footprint_count,
                "new_cell_count": new_count,
                "new_cell_ratio": (
                    float(new_count) / float(footprint_count)
                    if footprint_count
                    else 0.0
                ),
            }
        )

    union_count = int(np.count_nonzero(union_mask)) if union_mask is not None else 0
    new_union_count = (
        int(np.count_nonzero(union_mask & ~before))
        if union_mask is not None
        else 0
    )
    historical_revisit_count = union_count - new_union_count
    expected_after = before if union_mask is None else before | union_mask
    if not np.array_equal(after, expected_after):
        raise ValueError("Search diagnostic bitmap does not match footprint commit")
    discovered_before = {int(value) for value in discovered_roi_ids_before}
    discovered_after = {int(value) for value in discovered_roi_ids_after}
    if not discovered_before.issubset(discovered_after):
        raise ValueError("Search discovery state regressed during an interval")

    record = {
        "schema_version": SEARCH_DIAGNOSTICS_SCHEMA_VERSION,
        "method_id": str(method_id),
        "scenario_id": str(scenario_id),
        "episode_index": int(episode_index),
        "interval_index": int(interval_index),
        "time_seconds": float(time_seconds),
        "coverage_ratio_before": float(before.mean()),
        "coverage_ratio_after": float(after.mean()),
        "discovered_roi_count_before": len(discovered_before),
        "discovered_roi_count_after": len(discovered_after),
        "newly_discovered_roi_ids": sorted(discovered_after - discovered_before),
        "search_uav_ids": search_ids,
        "search_uav_count": len(search_ids),
        "search_uavs": per_uav,
        "gross_footprint_cell_count": gross_count,
        "union_footprint_cell_count": union_count,
        "new_union_cell_count": new_union_count,
        "simultaneous_overlap_cell_count": simultaneous_overlap_count,
        "simultaneous_overlap_ratio": (
            float(simultaneous_overlap_count) / float(gross_count)
            if gross_count
            else 0.0
        ),
        "historical_revisit_cell_count": historical_revisit_count,
        "historical_revisit_ratio": (
            float(historical_revisit_count) / float(union_count)
            if union_count
            else 0.0
        ),
    }
    return validate_search_diagnostics_record(record)


def validate_search_diagnostics_record(record):
    if not isinstance(record, dict) or tuple(record) != _REQUIRED_FIELDS:
        raise ValueError("Search diagnostic row has an invalid field contract")
    if record["schema_version"] != SEARCH_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("Search diagnostic schema version is invalid")
    if record["episode_index"] < 0 or record["interval_index"] < 0:
        raise ValueError("Search diagnostic indices must be non-negative")
    if record["search_uav_count"] != len(record["search_uav_ids"]):
        raise ValueError("Search UAV count disagrees with Search UAV IDs")
    if record["search_uav_ids"] != sorted(record["search_uav_ids"]):
        raise ValueError("Search UAV IDs must be sorted")
    if record["newly_discovered_roi_ids"] != sorted(
        record["newly_discovered_roi_ids"]
    ):
        raise ValueError("newly discovered RoI IDs must be sorted")
    gross = record["gross_footprint_cell_count"]
    union = record["union_footprint_cell_count"]
    new_union = record["new_union_cell_count"]
    if not gross >= union >= new_union >= 0:
        raise ValueError("Search footprint cell counts are inconsistent")
    simultaneous_overlap = record["simultaneous_overlap_cell_count"]
    if not gross >= simultaneous_overlap >= 0:
        raise ValueError("simultaneous Search overlap count is inconsistent")
    if record["historical_revisit_cell_count"] != union - new_union:
        raise ValueError("historical Search revisit count is inconsistent")
    expected_overlap_ratio = (
        float(simultaneous_overlap) / float(gross) if gross else 0.0
    )
    expected_revisit_ratio = (
        float(union - new_union) / float(union) if union else 0.0
    )
    if not math.isclose(
        float(record["simultaneous_overlap_ratio"]),
        expected_overlap_ratio,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("simultaneous Search overlap ratio is inconsistent")
    if not math.isclose(
        float(record["historical_revisit_ratio"]),
        expected_revisit_ratio,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("historical Search revisit ratio is inconsistent")
    if record["discovered_roi_count_after"] - record[
        "discovered_roi_count_before"
    ] != len(record["newly_discovered_roi_ids"]):
        raise ValueError("newly discovered RoI count is inconsistent")
    for field in (
        "time_seconds",
        "coverage_ratio_before",
        "coverage_ratio_after",
        "simultaneous_overlap_ratio",
        "historical_revisit_ratio",
    ):
        if not math.isfinite(float(record[field])):
            raise ValueError(f"Search diagnostic value is non-finite: {field}")
    for field in (
        "coverage_ratio_before",
        "coverage_ratio_after",
        "simultaneous_overlap_ratio",
        "historical_revisit_ratio",
    ):
        if not 0.0 <= float(record[field]) <= 1.0:
            raise ValueError(f"Search diagnostic ratio is outside [0,1]: {field}")
    if len(record["search_uavs"]) != record["search_uav_count"]:
        raise ValueError("Search UAV details disagree with Search UAV count")
    if [item.get("uav_id") for item in record["search_uavs"]] != record[
        "search_uav_ids"
    ]:
        raise ValueError("Search UAV detail IDs disagree with Search UAV IDs")
    for item in record["search_uavs"]:
        if tuple(item) != (
            "uav_id",
            "position_xyz_m",
            "displacement_m",
            "footprint_cell_count",
            "new_cell_count",
            "new_cell_ratio",
        ):
            raise ValueError("per-UAV Search diagnostic fields are invalid")
        if not item["footprint_cell_count"] >= item["new_cell_count"] >= 0:
            raise ValueError("per-UAV Search cell counts are inconsistent")
        if not 0.0 <= float(item["new_cell_ratio"]) <= 1.0:
            raise ValueError("per-UAV new-cell ratio is outside [0,1]")
        expected_ratio = (
            float(item["new_cell_count"]) / float(item["footprint_cell_count"])
            if item["footprint_cell_count"]
            else 0.0
        )
        if not math.isclose(
            float(item["new_cell_ratio"]),
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("per-UAV new-cell ratio is inconsistent")
        values = (*item["position_xyz_m"], item["displacement_m"])
        if len(item["position_xyz_m"]) != 3 or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise ValueError("per-UAV Search geometry is invalid")
    return record


class SearchDiagnosticsJsonlWriter:
    """Write one validated post-commit Search row at a time."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._handle = None
        self.row_count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def write(self, record):
        if self._handle is None:
            raise RuntimeError("Search diagnostics writer is not open")
        validated = validate_search_diagnostics_record(record)
        line = json.dumps(
            validated,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        self.row_count += 1

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False
