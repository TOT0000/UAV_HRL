"""Versioned, deterministic exogenous scenarios for comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from experiment_config import NUM_UAV, ROI_COUNT_MAX, ROI_COUNT_MIN


SCENARIO_SCHEMA_VERSION = "uav-hrl-scenario-v2"
OBSOLETE_SCHEMA_VERSION = "uav-hrl-scenario-v1"
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
    xy_positions = [
        (x, y)
        for y in (100.0, 300.0, 500.0, 700.0)
        for x in (100.0, 300.0, 500.0, 700.0)
    ]
    return [
        {
            "uav_id": uav_id,
            "position": [x, y, py_rng.uniform(80.0, 120.0)],
            "energy_j": 10000.0,
        }
        for uav_id, (x, y) in enumerate(xy_positions)
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
            "generation_model": "task-and-fov-gated-rate-accumulator-v1",
        },
        "exogenous_primitives": {
            "channel_randomness": "none",
            "gt_placement_model": "nonoverlap-away-from-gs-v1",
            "uav_xy_layout": "fixed-4x4-grid-v1",
        },
    }
    validate_scenario_entry(entry)
    return entry


def validate_scenario_entry(entry: dict[str, Any]) -> None:
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
    if len(entry["sr_teams"]) != int(entry["num_GT"]):
        raise ValueError("scenario SR team count must equal num_GT")


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioManifest":
        if data.get("schema_version") == OBSOLETE_SCHEMA_VERSION:
            raise ValueError(
                "scenario schema v1 is obsolete; regenerate the manifest with v2"
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
        },
        "config_fingerprint": environment_config_fingerprint(),
    }
    return ScenarioManifest.from_dict(
        {**unsigned, "content_hash": sha256_json(unsigned)}
    )


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
