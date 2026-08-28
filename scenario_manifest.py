"""Versioned, deterministic exogenous scenarios for comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable
import uuid

import numpy as np

from experiment_config import (
    A2A_COMMUNICATION_RANGE_M,
    A2G_COMMUNICATION_RANGE_M,
    CANONICAL_UAV_INITIAL_XY_M,
    COMMUNICATION_RANGE_CONTRACT_VERSION,
    GROUND_STATION_POSITION_M,
    GS_GATEWAY_CONTRACT_VERSION,
    GS_GATEWAY_HARD_RADIUS_M,
    GS_GATEWAY_PROJECTION_MODE,
    GS_GATEWAY_SOFT_RADIUS_M,
    INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
    NUM_UAV,
    PERMANENT_GS_GATEWAY_UAV_ID,
    RESERVED_SEARCH_UAV_IDS,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    UAV_INITIAL_LAYOUT_VERSION,
)


SCENARIO_SCHEMA_VERSION = "uav-hrl-scenario-v5"
OBSOLETE_SCHEMA_VERSIONS = frozenset(
    {
        "uav-hrl-scenario-v1",
        "uav-hrl-scenario-v2",
        "uav-hrl-scenario-v3",
        "uav-hrl-scenario-v4",
    }
)
# Singular compatibility name used by callers that construct an obsolete
# manifest explicitly for fail-fast tests.
OBSOLETE_SCHEMA_VERSION = "uav-hrl-scenario-v4"
UAV_INITIAL_LAYOUT = UAV_INITIAL_LAYOUT_VERSION
SUPPORTED_SPLITS = frozenset({"train", "validation", "test"})
POLICY_DEPENDENT_KEYS = frozenset(
    {
        "task_assignment",
        "uav_actions",
        "visited_map",
        "found_targets",
        "fov_validity",
        "source_uavs",
        "routing_decisions",
        "packets",
        "delivered_bits",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _topology_error(
    scenario_id,
    reason,
    *,
    nearest_u2g_distance_m,
    u2g_range_m,
    gs_component_uav_ids,
):
    nearest = (
        f"{float(nearest_u2g_distance_m):.12g}"
        if nearest_u2g_distance_m is not None
        and math.isfinite(float(nearest_u2g_distance_m))
        else "non-finite"
    )
    return ValueError(
        "initial communication topology is invalid: "
        f"scenario_id={scenario_id}; reason={reason}; "
        f"nearest_u2g_3d_distance_m={nearest}; "
        f"u2g_range_m={float(u2g_range_m):.12g}; "
        f"gs_component_uav_ids={list(gs_component_uav_ids)}"
    )


def validate_initial_communication_topology(
    uavs,
    *,
    scenario_id,
    gs_position=GROUND_STATION_POSITION_M,
    u2g_range_m=A2G_COMMUNICATION_RANGE_M,
    u2u_range_m=A2A_COMMUNICATION_RANGE_M,
):
    """Validate the finite inclusive 3-D range graph at episode start."""

    try:
        u2g_range_m = float(u2g_range_m)
        u2u_range_m = float(u2u_range_m)
        gs = tuple(float(value) for value in gs_position)
    except (TypeError, ValueError) as exc:
        raise _topology_error(
            scenario_id,
            "range or GS coordinate is not numeric",
            nearest_u2g_distance_m=None,
            u2g_range_m=A2G_COMMUNICATION_RANGE_M,
            gs_component_uav_ids=(),
        ) from exc
    if (
        len(gs) != 3
        or not all(math.isfinite(value) for value in gs)
        or not math.isfinite(u2g_range_m)
        or not math.isfinite(u2u_range_m)
        or u2g_range_m < 0.0
        or u2u_range_m < 0.0
    ):
        raise _topology_error(
            scenario_id,
            "range or GS coordinate is non-finite or invalid",
            nearest_u2g_distance_m=None,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=(),
        )

    positions = {}
    try:
        for item in uavs:
            uav_id = int(item["uav_id"])
            if uav_id in positions:
                raise ValueError(f"duplicate UAV ID {uav_id}")
            position = tuple(float(value) for value in item["position"])
            if len(position) != 3 or not all(
                math.isfinite(value) for value in position
            ):
                raise ValueError(f"UAV {uav_id} has non-finite 3-D coordinates")
            positions[uav_id] = position
    except (KeyError, TypeError, ValueError) as exc:
        raise _topology_error(
            scenario_id,
            str(exc),
            nearest_u2g_distance_m=None,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=(),
        ) from exc
    if not positions:
        raise _topology_error(
            scenario_id,
            "scenario has no UAV positions",
            nearest_u2g_distance_m=None,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=(),
        )

    u2g_distances = {
        uav_id: math.dist(position, gs)
        for uav_id, position in positions.items()
    }
    if not all(math.isfinite(distance) for distance in u2g_distances.values()):
        raise _topology_error(
            scenario_id,
            "U2G distance is non-finite",
            nearest_u2g_distance_m=None,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=(),
        )
    nearest_u2g_distance_m = min(u2g_distances.values())
    u2g_uav_ids = sorted(
        uav_id
        for uav_id, distance in u2g_distances.items()
        if distance <= u2g_range_m
    )
    adjacency = {uav_id: set() for uav_id in positions}
    u2u_edges = []
    ordered_ids = sorted(positions)
    for index, sender in enumerate(ordered_ids):
        for receiver in ordered_ids[index + 1 :]:
            distance = math.dist(positions[sender], positions[receiver])
            if not math.isfinite(distance):
                raise _topology_error(
                    scenario_id,
                    f"U2U distance is non-finite for UAVs {sender} and {receiver}",
                    nearest_u2g_distance_m=nearest_u2g_distance_m,
                    u2g_range_m=u2g_range_m,
                    gs_component_uav_ids=u2g_uav_ids,
                )
            if distance <= u2u_range_m:
                adjacency[sender].add(receiver)
                adjacency[receiver].add(sender)
                u2u_edges.append((sender, receiver))

    component = set(u2g_uav_ids)
    frontier = list(u2g_uav_ids)
    while frontier:
        sender = frontier.pop()
        for receiver in adjacency[sender]:
            if receiver not in component:
                component.add(receiver)
                frontier.append(receiver)
    gs_component_uav_ids = sorted(component)
    if not u2g_uav_ids:
        raise _topology_error(
            scenario_id,
            "no UAV has an inclusive U2G range edge",
            nearest_u2g_distance_m=nearest_u2g_distance_m,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=gs_component_uav_ids,
        )
    if len(gs_component_uav_ids) < 2:
        raise _topology_error(
            scenario_id,
            "GS component must contain at least two UAVs",
            nearest_u2g_distance_m=nearest_u2g_distance_m,
            u2g_range_m=u2g_range_m,
            gs_component_uav_ids=gs_component_uav_ids,
        )
    return {
        "scenario_id": str(scenario_id),
        "topology_contract_version": (
            INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION
        ),
        "u2g_range_m": u2g_range_m,
        "u2u_range_m": u2u_range_m,
        "nearest_u2g_3d_distance_m": nearest_u2g_distance_m,
        "u2g_uav_ids": u2g_uav_ids,
        "u2u_edges": [list(edge) for edge in u2u_edges],
        "gs_component_uav_ids": gs_component_uav_ids,
    }


def validate_permanent_gateway_initial_position(
    uavs,
    *,
    scenario_id,
    gs_position=GROUND_STATION_POSITION_M,
):
    """Fail before reset side effects if UAV 0 would require an initial teleport."""

    by_id = {int(item["uav_id"]): item for item in uavs}
    if PERMANENT_GS_GATEWAY_UAV_ID not in by_id:
        raise ValueError(
            "permanent GS gateway is absent from initial scenario: "
            f"scenario_id={scenario_id}; uav_id={PERMANENT_GS_GATEWAY_UAV_ID}"
        )
    try:
        position = tuple(
            float(value)
            for value in by_id[PERMANENT_GS_GATEWAY_UAV_ID]["position"]
        )
        station = tuple(float(value) for value in gs_position)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "permanent GS gateway initial position is invalid: "
            f"scenario_id={scenario_id}"
        ) from exc
    if len(position) != 3 or len(station) != 3 or not all(
        math.isfinite(value) for value in (*position, *station)
    ):
        raise ValueError(
            "permanent GS gateway initial position must be finite 3-D: "
            f"scenario_id={scenario_id}"
        )
    distance = math.dist(position, station)
    if distance > GS_GATEWAY_HARD_RADIUS_M:
        raise ValueError(
            "permanent GS gateway initial position exceeds hard radius: "
            f"scenario_id={scenario_id}; uav_id={PERMANENT_GS_GATEWAY_UAV_ID}; "
            f"distance_3d_m={distance:.12g}; "
            f"hard_radius_m={GS_GATEWAY_HARD_RADIUS_M:.12g}"
        )
    return {
        "uav_id": PERMANENT_GS_GATEWAY_UAV_ID,
        "position": list(position),
        "distance_3d_m": distance,
        "hard_radius_m": GS_GATEWAY_HARD_RADIUS_M,
        "contract_version": GS_GATEWAY_CONTRACT_VERSION,
    }


def current_environment_config() -> dict[str, Any]:
    return {
        "num_uav": NUM_UAV,
        "roi_count_min": ROI_COUNT_MIN,
        "roi_count_max": ROI_COUNT_MAX,
        "episode_seconds": 60,
        "routing_slot_seconds": 0.25,
        "environment_width_m": 1000,
        "environment_height_m": 1000,
        "bit_resolution_m": 2,
        "uav_energy_max_j": 10000.0,
        "active_link_bandwidth_hz": 10e6,
        "a2g_communication_range_m": A2G_COMMUNICATION_RANGE_M,
        "a2a_communication_range_m": A2A_COMMUNICATION_RANGE_M,
        "communication_range_contract_version": (
            COMMUNICATION_RANGE_CONTRACT_VERSION
        ),
        "ground_station_position_m": list(GROUND_STATION_POSITION_M),
        "uav_initial_layout": UAV_INITIAL_LAYOUT,
        "canonical_uav_initial_xy_m": [
            list(position) for position in CANONICAL_UAV_INITIAL_XY_M
        ],
        "initial_communication_topology_contract_version": (
            INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION
        ),
        "reserved_search_uav_ids": list(RESERVED_SEARCH_UAV_IDS),
        "permanent_gs_gateway_uav_id": PERMANENT_GS_GATEWAY_UAV_ID,
        "gs_gateway_soft_radius_m": GS_GATEWAY_SOFT_RADIUS_M,
        "gs_gateway_hard_radius_m": GS_GATEWAY_HARD_RADIUS_M,
        "gs_gateway_projection_mode": GS_GATEWAY_PROJECTION_MODE,
        "gs_gateway_contract_version": GS_GATEWAY_CONTRACT_VERSION,
        "gt_radius_m": 80.0,
        "com_deadline_seconds": 1.0,
        "fov_deadline_seconds": 1.5,
        "injection_cutoff_seconds": 58.5,
        "sr_initial_layout": "four-boundary-midpoints-cyclic",
        "sr_motion_model": "policy-triggered-straight-line-v1",
    }


def environment_config_fingerprint(config: dict[str, Any] | None = None) -> str:
    return sha256_json(config or current_environment_config())


def build_generation_profile(num_gt: int | None = None) -> dict[str, Any]:
    if num_gt is None:
        return {
            "num_gt_mode": "mixed",
            "fixed_num_gt": None,
            "mixed_num_gt_min": ROI_COUNT_MIN,
            "mixed_num_gt_max": ROI_COUNT_MAX,
        }
    if isinstance(num_gt, bool) or int(num_gt) != num_gt:
        raise ValueError(
            f"fixed num_GT must be an integer from {ROI_COUNT_MIN} "
            f"through {ROI_COUNT_MAX}"
        )
    fixed_num_gt = int(num_gt)
    if not ROI_COUNT_MIN <= fixed_num_gt <= ROI_COUNT_MAX:
        raise ValueError(
            "fixed num_GT must be in the inclusive range "
            f"[{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
        )
    return {
        "num_gt_mode": "fixed",
        "fixed_num_gt": fixed_num_gt,
        "mixed_num_gt_min": ROI_COUNT_MIN,
        "mixed_num_gt_max": ROI_COUNT_MAX,
    }


def _profile_id(profile: dict[str, Any]) -> str:
    if profile["num_gt_mode"] == "fixed":
        return f"fixed-{int(profile['fixed_num_gt'])}"
    return (
        f"mixed-{int(profile['mixed_num_gt_min'])}-"
        f"{int(profile['mixed_num_gt_max'])}"
    )


def _split_seed(
    split: str,
    manifest_seed: int,
    episode_index: int,
    generation_profile: dict[str, Any],
) -> int:
    material = (
        f"{SCENARIO_SCHEMA_VERSION}:{split}:{int(manifest_seed)}:"
        f"{_profile_id(generation_profile)}:{int(episode_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _uav_initial_data(py_rng: random.Random) -> list[dict[str, Any]]:
    return [
        {
            "uav_id": uav_id,
            "position": [x, y, py_rng.uniform(80.0, 120.0)],
            "energy_j": 10000.0,
        }
        for uav_id, (x, y) in enumerate(CANONICAL_UAV_INITIAL_XY_M)
    ]


def _gt_initial_data(
    py_rng: random.Random, num_gt: int
) -> list[dict[str, Any]]:
    radius = 80.0
    width = height = 1000.0
    minimum_distance = int(2 * radius)
    tries = 0
    points: list[tuple[float, float]] = []
    while len(points) < int(num_gt):
        if tries > 200:
            minimum_distance = max(int(minimum_distance * 0.9), int(radius))
            tries = 0
        x = py_rng.uniform(radius, width - radius)
        y = py_rng.uniform(radius, height - radius)
        if x * x + y * y < 200.0**2:
            tries += 1
            continue
        if any(
            np.hypot(x - previous_x, y - previous_y) < minimum_distance
            for previous_x, previous_y in points
        ):
            tries += 1
            continue
        points.append((x, y))
        tries = 0
    return [
        {
            "gt_id": gt_id,
            "position": [x, y, 0.0],
            "radius_m": radius,
        }
        for gt_id, (x, y) in enumerate(points)
    ]


def _sr_initial_data(num_gt: int) -> list[dict[str, Any]]:
    boundary_points = (
        (0.0, 500.0),
        (1000.0, 500.0),
        (500.0, 0.0),
        (500.0, 1000.0),
    )
    return [
        {
            "sr_id": sr_id,
            "position": [*boundary_points[sr_id % 4], 0.0],
            "movement_primitive": {
                "model": "policy-triggered-straight-line-v1",
                "speed_mps": 1.0,
            },
        }
        for sr_id in range(int(num_gt))
    ]


def generate_scenario_entry(
    split: str,
    manifest_seed: int,
    episode_index: int,
    num_gt: int | None = None,
) -> dict[str, Any]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported scenario split: {split}")
    generation_profile = build_generation_profile(num_gt)
    profile_id = _profile_id(generation_profile)
    scenario_seed = _split_seed(
        split, manifest_seed, episode_index, generation_profile
    )
    np_rng = np.random.default_rng(scenario_seed)
    py_rng = random.Random(scenario_seed)
    episode_num_gt = (
        int(np_rng.integers(ROI_COUNT_MIN, ROI_COUNT_MAX + 1))
        if generation_profile["num_gt_mode"] == "mixed"
        else int(generation_profile["fixed_num_gt"])
    )
    entry = {
        "scenario_id": (
            f"{split}:{SCENARIO_SCHEMA_VERSION}:{profile_id}:"
            f"{int(manifest_seed)}:"
            f"{int(episode_index):06d}"
        ),
        "scenario_seed": scenario_seed,
        "generation_profile_id": profile_id,
        "num_GT": episode_num_gt,
        "ground_targets": _gt_initial_data(py_rng, episode_num_gt),
        "uavs": _uav_initial_data(py_rng),
        "sr_teams": _sr_initial_data(episode_num_gt),
        "traffic_primitives": {
            "load_factor": 1.0,
            "base_fov_packets_per_second": 5.0,
            "base_com_packets_per_second": 50.0,
            "generation_model": "assigned-fov-rate-accumulator-v2",
        },
        "exogenous_primitives": {
            "channel_randomness": "none",
            "gt_placement_model": "nonoverlap-away-from-gs-v1",
            "uav_xy_layout": UAV_INITIAL_LAYOUT,
            "ground_station_position_m": list(GROUND_STATION_POSITION_M),
            "initial_communication_topology_contract_version": (
                INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION
            ),
            "num_uav": NUM_UAV,
            "reserved_search_uav_ids": list(RESERVED_SEARCH_UAV_IDS),
            "permanent_gs_gateway_uav_id": PERMANENT_GS_GATEWAY_UAV_ID,
            "gs_gateway_contract_version": GS_GATEWAY_CONTRACT_VERSION,
        },
    }
    validate_scenario_entry(entry)
    return entry


def validate_scenario_entry(entry: dict[str, Any]) -> dict[str, Any]:
    required = {
        "scenario_id",
        "scenario_seed",
        "generation_profile_id",
        "num_GT",
        "ground_targets",
        "uavs",
        "sr_teams",
        "traffic_primitives",
        "exogenous_primitives",
    }
    missing = required.difference(entry)
    if missing:
        raise ValueError(f"scenario entry is missing fields: {sorted(missing)}")
    forbidden = POLICY_DEPENDENT_KEYS.intersection(entry)
    if forbidden:
        raise ValueError(
            f"scenario entry contains policy-dependent fields: {sorted(forbidden)}"
        )
    if int(entry["num_GT"]) != len(entry["ground_targets"]):
        raise ValueError("scenario num_GT does not match ground target data")
    if not ROI_COUNT_MIN <= int(entry["num_GT"]) <= ROI_COUNT_MAX:
        raise ValueError(
            f"scenario num_GT must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
        )
    if len(entry["uavs"]) != NUM_UAV:
        raise ValueError(f"scenario must contain exactly {NUM_UAV} UAVs")
    validate_permanent_gateway_initial_position(
        entry["uavs"], scenario_id=entry["scenario_id"]
    )
    topology = validate_initial_communication_topology(
        entry["uavs"], scenario_id=entry["scenario_id"]
    )
    metadata = dict(entry["exogenous_primitives"])
    if metadata.get("uav_xy_layout") != UAV_INITIAL_LAYOUT:
        raise ValueError(
            "scenario UAV initial layout is incompatible: "
            f"scenario_id={entry['scenario_id']}; "
            f"declared={metadata.get('uav_xy_layout')}; "
            f"expected={UAV_INITIAL_LAYOUT}; "
            f"nearest_u2g_3d_distance_m="
            f"{topology['nearest_u2g_3d_distance_m']:.12g}; "
            f"u2g_range_m={topology['u2g_range_m']:.12g}; "
            f"gs_component_uav_ids={topology['gs_component_uav_ids']}"
        )
    if metadata.get("ground_station_position_m") != list(
        GROUND_STATION_POSITION_M
    ):
        raise ValueError("scenario ground-station position metadata is incompatible")
    if metadata.get(
        "initial_communication_topology_contract_version"
    ) != INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION:
        raise ValueError("scenario initial topology contract metadata is incompatible")
    if int(metadata.get("num_uav", -1)) != NUM_UAV:
        raise ValueError("scenario UAV count metadata is incompatible")
    if tuple(metadata.get("reserved_search_uav_ids", ())) != RESERVED_SEARCH_UAV_IDS:
        raise ValueError("scenario reserved Search UAV IDs are incompatible")
    if int(metadata.get("permanent_gs_gateway_uav_id", -1)) != (
        PERMANENT_GS_GATEWAY_UAV_ID
    ):
        raise ValueError("scenario permanent GS gateway UAV ID is incompatible")
    if metadata.get("gs_gateway_contract_version") != GS_GATEWAY_CONTRACT_VERSION:
        raise ValueError("scenario permanent GS gateway contract is incompatible")
    if entry["traffic_primitives"].get("generation_model") != (
        "assigned-fov-rate-accumulator-v2"
    ):
        raise ValueError("scenario packet generation model is incompatible")
    uav_ids = [int(item["uav_id"]) for item in entry["uavs"]]
    if uav_ids != list(range(NUM_UAV)):
        raise ValueError("scenario UAV IDs and order are incompatible")
    for item, expected_xy in zip(entry["uavs"], CANONICAL_UAV_INITIAL_XY_M):
        x, y, z = map(float, item["position"])
        energy = float(item["energy_j"])
        if (x, y) != tuple(expected_xy):
            raise ValueError(
                "scenario UAV coordinates disagree with declared initial layout: "
                f"scenario_id={entry['scenario_id']}; uav_id={item['uav_id']}"
            )
        if not 80.0 <= z <= 120.0:
            raise ValueError("scenario UAV initial altitude must be in [80, 120] m")
        if not math.isfinite(energy) or energy != 10000.0:
            raise ValueError("scenario UAV initial energy is incompatible")
    if len(entry["sr_teams"]) != int(entry["num_GT"]):
        raise ValueError("scenario SR team count must equal num_GT")
    return topology


def validate_manifest_initial_topologies(manifest, episode_count=None):
    """Revalidate formal manifest entries before run or checkpoint side effects."""

    if not isinstance(manifest, ScenarioManifest):
        raise TypeError("initial topology preflight requires a ScenarioManifest")
    count = manifest.episode_count if episode_count is None else int(episode_count)
    if count <= 0 or count > manifest.episode_count:
        raise ValueError("initial topology preflight episode count is invalid")
    return tuple(
        validate_scenario_entry(entry) for entry in manifest.episodes[:count]
    )


@dataclass(frozen=True)
class ScenarioManifest:
    schema_version: str
    split: str
    manifest_seed: int
    episode_count: int
    episodes: tuple[dict[str, Any], ...]
    generation_profile: dict[str, Any]
    generator_config: dict[str, Any]
    config_fingerprint: str
    content_hash: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "manifest_seed": int(self.manifest_seed),
            "episode_count": int(self.episode_count),
            "episodes": list(self.episodes),
            "generation_profile": self.generation_profile,
            "generator_config": self.generator_config,
            "config_fingerprint": self.config_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_hash": self.content_hash}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False
        ) + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def save_atomic(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Persist one complete manifest without exposing a partial JSON file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"scenario manifest already exists: {path}")
        temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(self.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists() and not overwrite:
                raise FileExistsError(f"scenario manifest already exists: {path}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioManifest":
        if data.get("schema_version") in {
            "uav-hrl-scenario-v1", "uav-hrl-scenario-v2"
        }:
            raise ValueError(
                "legacy 16-UAV scenario schema is incompatible; regenerate the "
                "manifest with the 10-UAV v5 generator"
            )
        if data.get("schema_version") == "uav-hrl-scenario-v3":
            raise ValueError(
                "legacy disconnected-GS scenario geometry is incompatible; "
                "regenerate the manifest with the v5 gateway layout"
            )
        if data.get("schema_version") == "uav-hrl-scenario-v4":
            raise ValueError(
                "legacy pre-permanent-gateway packet-generation scenario is "
                "incompatible; regenerate the manifest with the v5 generator"
            )
        if data.get("schema_version") != SCENARIO_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scenario schema: {data.get('schema_version')}"
            )
        split = str(data.get("split"))
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported scenario split: {split}")
        episodes = tuple(data.get("episodes", ()))
        if int(data.get("episode_count", -1)) != len(episodes):
            raise ValueError("manifest episode_count does not match entries")
        generation_profile = dict(data.get("generation_profile") or {})
        expected_profile = build_generation_profile(
            generation_profile.get("fixed_num_gt")
            if generation_profile.get("num_gt_mode") == "fixed"
            else None
        )
        if generation_profile != expected_profile:
            raise ValueError("manifest generation profile is invalid")
        profile_id = _profile_id(generation_profile)
        for entry in episodes:
            validate_scenario_entry(entry)
            if entry["generation_profile_id"] != profile_id:
                raise ValueError("scenario generation profile identity mismatch")
            num_gt = int(entry["num_GT"])
            if generation_profile["num_gt_mode"] == "fixed":
                if num_gt != int(generation_profile["fixed_num_gt"]):
                    raise ValueError("fixed num_GT manifest contains a mixed entry")
            elif not ROI_COUNT_MIN <= num_gt <= ROI_COUNT_MAX:
                raise ValueError(
                    "mixed num_GT entry is outside "
                    f"[{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
                )
        unsigned = {key: value for key, value in data.items() if key != "content_hash"}
        expected_hash = sha256_json(unsigned)
        if data.get("content_hash") != expected_hash:
            raise ValueError("manifest content hash mismatch")
        expected_config = environment_config_fingerprint()
        if data.get("config_fingerprint") != expected_config:
            raise ValueError("manifest environment configuration is incompatible")
        return cls(
            schema_version=SCENARIO_SCHEMA_VERSION,
            split=split,
            manifest_seed=int(data["manifest_seed"]),
            episode_count=len(episodes),
            episodes=episodes,
            generation_profile=generation_profile,
            generator_config=dict(data["generator_config"]),
            config_fingerprint=str(data["config_fingerprint"]),
            content_hash=str(data["content_hash"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def generate_manifest(
    split: str,
    manifest_seed: int,
    episode_count: int,
    num_gt: int | None = None,
) -> ScenarioManifest:
    if int(episode_count) <= 0:
        raise ValueError("episode_count must be positive")
    generation_profile = build_generation_profile(num_gt)
    unsigned = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "split": split,
        "manifest_seed": int(manifest_seed),
        "episode_count": int(episode_count),
        "episodes": [
            generate_scenario_entry(split, manifest_seed, index)
            if num_gt is None
            else generate_scenario_entry(
                split, manifest_seed, index, num_gt=num_gt
            )
            for index in range(int(episode_count))
        ],
        "generation_profile": generation_profile,
        "generator_config": {
            "generator": "local-python-and-numpy-rng-v1",
            "numpy_bit_generator": "PCG64",
            "policy_dependent_outcomes_excluded": True,
            "uav_initial_layout": UAV_INITIAL_LAYOUT,
            "ground_station_position_m": list(GROUND_STATION_POSITION_M),
            "initial_communication_topology_contract_version": (
                INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION
            ),
            "a2g_communication_range_m": A2G_COMMUNICATION_RANGE_M,
            "a2a_communication_range_m": A2A_COMMUNICATION_RANGE_M,
            "num_uav": NUM_UAV,
            "reserved_search_uav_ids": list(RESERVED_SEARCH_UAV_IDS),
        },
        "config_fingerprint": environment_config_fingerprint(),
    }
    return ScenarioManifest.from_dict(
        {**unsigned, "content_hash": sha256_json(unsigned)}
    )


def manifest_prefix(manifest: ScenarioManifest, episode_count: int) -> ScenarioManifest:
    """Return the canonical manifest represented by the requested leading episodes."""

    if not isinstance(manifest, ScenarioManifest):
        raise TypeError("manifest prefix validation requires a ScenarioManifest")
    episode_count = int(episode_count)
    if episode_count <= 0 or episode_count > manifest.episode_count:
        raise ValueError(
            "manifest prefix length must be positive and no greater than the manifest"
        )
    if episode_count == manifest.episode_count:
        return manifest
    unsigned = manifest.unsigned_dict()
    unsigned["episode_count"] = episode_count
    unsigned["episodes"] = list(manifest.episodes[:episode_count])
    return ScenarioManifest.from_dict(
        {**unsigned, "content_hash": sha256_json(unsigned)}
    )


def validate_manifest_prefix_extension(
    previous: ScenarioManifest,
    extended: ScenarioManifest,
) -> dict[str, Any]:
    """Validate a deterministic training-manifest extension and report provenance."""

    if previous.split != "train" or extended.split != "train":
        raise ValueError("horizon extension requires training scenario manifests")
    if extended.episode_count <= previous.episode_count:
        raise ValueError("extended training manifest must contain additional episodes")
    administrative_fields = {"episode_count", "episodes", "content_hash"}
    previous_contract = {
        key: value
        for key, value in previous.to_dict().items()
        if key not in administrative_fields
    }
    extended_contract = {
        key: value
        for key, value in extended.to_dict().items()
        if key not in administrative_fields
    }
    if previous_contract != extended_contract:
        raise ValueError("extended training manifest contract is incompatible")
    if tuple(extended.episodes[: previous.episode_count]) != previous.episodes:
        raise ValueError(
            "extended training manifest does not preserve the exact scenario prefix"
        )
    canonical_prefix = manifest_prefix(extended, previous.episode_count)
    if canonical_prefix.content_hash != previous.content_hash:
        raise ValueError(
            "extended training manifest canonical prefix hash is incompatible"
        )
    return {
        "previous_manifest_hash": previous.content_hash,
        "extended_manifest_hash": extended.content_hash,
        "preserved_prefix_length": previous.episode_count,
        "manifest_prefix_compatible": True,
    }


def extend_training_manifest(
    previous: ScenarioManifest,
    target_episode_count: int,
) -> tuple[ScenarioManifest, dict[str, Any]]:
    """Deterministically extend a training manifest after validating its prefix."""

    target_episode_count = int(target_episode_count)
    if target_episode_count <= previous.episode_count:
        raise ValueError("target manifest episode count must exceed the current count")
    fixed_num_gt = (
        previous.generation_profile.get("fixed_num_gt")
        if previous.generation_profile.get("num_gt_mode") == "fixed"
        else None
    )
    extended = generate_manifest(
        "train",
        manifest_seed=previous.manifest_seed,
        episode_count=target_episode_count,
        num_gt=fixed_num_gt,
    )
    provenance = validate_manifest_prefix_extension(previous, extended)
    return extended, provenance


def _manifest_path_within_run(run_directory, manifest_path) -> tuple[Path, str]:
    """Resolve one provenance path without allowing it to escape the run."""

    run_directory = Path(run_directory).resolve()
    configured = Path(str(manifest_path))
    resolved = (
        configured.resolve()
        if configured.is_absolute()
        else (run_directory / configured).resolve()
    )
    if not resolved.is_relative_to(run_directory):
        raise RuntimeError("training manifest segment path escapes the run directory")
    return resolved, resolved.relative_to(run_directory).as_posix()


def initial_training_manifest_segments(
    run_directory,
    manifest_path,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    """Build the canonical first segment for a new training run."""

    if not isinstance(manifest, ScenarioManifest) or manifest.split != "train":
        raise ValueError("training manifest segments require a training manifest")
    _, relative_path = _manifest_path_within_run(run_directory, manifest_path)
    return [
        {
            "episode_start": 1,
            "episode_end": int(manifest.episode_count),
            "manifest_hash": manifest.content_hash,
            "manifest_path": relative_path,
            "parent_manifest_hash": None,
        }
    ]


def append_training_manifest_segment(
    run_directory,
    segments,
    previous_manifest: ScenarioManifest,
    extended_manifest: ScenarioManifest,
    extended_manifest_path,
) -> list[dict[str, Any]]:
    """Append, never merge, one monotonic manifest-extension segment."""

    canonical = validate_training_manifest_segments(
        run_directory,
        segments,
        current_total_episodes=previous_manifest.episode_count,
    )
    if canonical[-1]["manifest_hash"] != previous_manifest.content_hash:
        raise RuntimeError(
            "training manifest segment tail disagrees with the active manifest"
        )
    validate_manifest_prefix_extension(previous_manifest, extended_manifest)
    _, relative_path = _manifest_path_within_run(
        run_directory, extended_manifest_path
    )
    return [
        *canonical,
        {
            "episode_start": int(previous_manifest.episode_count) + 1,
            "episode_end": int(extended_manifest.episode_count),
            "manifest_hash": extended_manifest.content_hash,
            "manifest_path": relative_path,
            "parent_manifest_hash": previous_manifest.content_hash,
        },
    ]


def validate_training_manifest_segments(
    run_directory,
    segments,
    *,
    current_total_episodes,
) -> list[dict[str, Any]]:
    """Validate complete, gap-free episode-to-manifest provenance."""

    try:
        current_total_episodes = int(current_total_episodes)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("training manifest segment horizon is invalid") from exc
    if current_total_episodes <= 0:
        raise RuntimeError("training manifest segment horizon must be positive")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise RuntimeError("training_manifest_segments is missing")

    canonical = []
    previous_manifest = None
    expected_start = 1
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, dict):
            raise RuntimeError("training manifest segment must be an object")
        required = {
            "episode_start",
            "episode_end",
            "manifest_hash",
            "manifest_path",
            "parent_manifest_hash",
        }
        missing = sorted(required.difference(raw_segment))
        if missing:
            raise RuntimeError(
                f"training manifest segment is incomplete: {missing}"
            )
        try:
            episode_start = int(raw_segment["episode_start"])
            episode_end = int(raw_segment["episode_end"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("training manifest segment episode range is invalid") from exc
        if episode_start != expected_start or episode_end < episode_start:
            raise RuntimeError(
                "training manifest segments contain a gap, overlap, or invalid range: "
                f"index={index}, expected_start={expected_start}, "
                f"actual={episode_start}..{episode_end}"
            )
        path, relative_path = _manifest_path_within_run(
            run_directory, raw_segment["manifest_path"]
        )
        manifest = ScenarioManifest.load(path)
        manifest_hash = str(raw_segment["manifest_hash"])
        if manifest.content_hash != manifest_hash:
            raise RuntimeError(
                "training manifest segment hash disagrees with manifest content: "
                f"segment={manifest_hash}, manifest={manifest.content_hash}"
            )
        if manifest.split != "train":
            raise RuntimeError("training manifest segment references a non-training manifest")
        if manifest.episode_count != episode_end:
            raise RuntimeError(
                "training manifest segment end disagrees with manifest length: "
                f"segment={episode_end}, manifest={manifest.episode_count}"
            )
        parent_hash = raw_segment["parent_manifest_hash"]
        if index == 0:
            if parent_hash is not None:
                raise RuntimeError(
                    "initial training manifest segment must not have a parent"
                )
        else:
            if parent_hash != previous_manifest.content_hash:
                raise RuntimeError(
                    "training manifest segment parent hash is incompatible"
                )
            try:
                validate_manifest_prefix_extension(previous_manifest, manifest)
            except ValueError as exc:
                raise RuntimeError(
                    "training manifest segment prefix is incompatible"
                ) from exc
        canonical.append(
            {
                "episode_start": episode_start,
                "episode_end": episode_end,
                "manifest_hash": manifest_hash,
                "manifest_path": relative_path,
                "parent_manifest_hash": parent_hash,
            }
        )
        previous_manifest = manifest
        expected_start = episode_end + 1

    if expected_start != current_total_episodes + 1:
        raise RuntimeError(
            "training manifest segments do not completely cover the current horizon: "
            f"covered_through={expected_start - 1}, "
            f"current={current_total_episodes}"
        )
    return canonical


def resolve_training_manifest_segment(
    run_directory,
    segments,
    episode,
    *,
    current_total_episodes=None,
) -> dict[str, Any]:
    """Resolve exactly one canonical active-manifest segment for an episode."""

    if current_total_episodes is None:
        if not isinstance(segments, (list, tuple)) or not segments:
            raise RuntimeError("training_manifest_segments is missing")
        current_total_episodes = segments[-1].get("episode_end")
    canonical = validate_training_manifest_segments(
        run_directory,
        segments,
        current_total_episodes=current_total_episodes,
    )
    try:
        episode = int(episode)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("training manifest segment episode is invalid") from exc
    matches = [
        segment
        for segment in canonical
        if segment["episode_start"] <= episode <= segment["episode_end"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "episode does not resolve to exactly one training manifest segment: "
            f"episode={episode}, matches={len(matches)}"
        )
    return dict(matches[0])


def resolve_training_manifest(run_directory, resolved_config) -> tuple[Path, ScenarioManifest]:
    """Load the active immutable training manifest declared by a run."""

    run_directory = Path(run_directory).resolve()
    configured = resolved_config.get(
        "training_manifest_path", "scenario_manifest.json"
    )
    configured_path = Path(str(configured))
    path = (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (run_directory / configured_path).resolve()
    )
    if not path.is_relative_to(run_directory):
        raise RuntimeError("training manifest path escapes the run directory")
    manifest = ScenarioManifest.load(path)
    if manifest.split != "train":
        raise RuntimeError("run scenario manifest is not a training manifest")
    expected_hash = resolved_config.get("training_manifest_hash")
    if expected_hash is not None and expected_hash != manifest.content_hash:
        raise RuntimeError("run training manifest hash disagrees with resolved config")
    return path, manifest


def resolve_training_manifest_segments_from_metadata(
    run_directory,
    resolved_config,
) -> tuple[Path, ScenarioManifest, list[dict[str, Any]]]:
    """Resolve canonical segments, including legacy extension metadata."""

    run_directory = Path(run_directory).resolve()
    manifest_path, manifest = resolve_training_manifest(
        run_directory, resolved_config
    )
    try:
        current_total = int(resolved_config["training_config"]["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("run training horizon is invalid") from exc
    stored = resolved_config.get("training_manifest_segments")
    if stored is not None:
        segments = stored
    else:
        history = list(resolved_config.get("horizon_extension_history") or ())
        if not history:
            segments = initial_training_manifest_segments(
                run_directory, manifest_path, manifest
            )
        else:
            first = history[0]
            segments = [
                {
                    "episode_start": 1,
                    "episode_end": int(first["previous_total_episodes"]),
                    "manifest_hash": str(first["previous_manifest_hash"]),
                    "manifest_path": str(first["previous_manifest_path"]),
                    "parent_manifest_hash": None,
                }
            ]
            for record in history:
                previous_total = int(record["previous_total_episodes"])
                if previous_total != int(segments[-1]["episode_end"]):
                    raise RuntimeError(
                        "legacy horizon extension provenance is not consecutive"
                    )
                segments.append(
                    {
                        "episode_start": previous_total + 1,
                        "episode_end": int(record["target_total_episodes"]),
                        "manifest_hash": str(record["extended_manifest_hash"]),
                        "manifest_path": str(record["extended_manifest_path"]),
                        "parent_manifest_hash": str(
                            record["previous_manifest_hash"]
                        ),
                    }
                )
    canonical = validate_training_manifest_segments(
        run_directory, segments, current_total_episodes=current_total
    )
    if canonical[-1]["manifest_hash"] != manifest.content_hash:
        raise RuntimeError(
            "training manifest segment tail disagrees with the active manifest"
        )
    return manifest_path, manifest, canonical


def validate_disjoint_manifests(manifests: Iterable[ScenarioManifest]) -> None:
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for manifest in manifests:
        ids = {str(entry["scenario_id"]) for entry in manifest.episodes}
        seeds = {int(entry["scenario_seed"]) for entry in manifest.episodes}
        if seen_ids.intersection(ids) or seen_seeds.intersection(seeds):
            raise ValueError("scenario identities or seeds overlap across splits")
        seen_ids.update(ids)
        seen_seeds.update(seeds)
