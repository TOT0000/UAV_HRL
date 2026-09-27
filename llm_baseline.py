"""Reproducible offline sampling and empirical baseline Lipschitz estimates."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable
import uuid

import numpy as np

from centralized_movement import MOVEMENT_STATE_DIM
from replay_auxiliary import (
    REPLAY_AUXILIARY_SCHEMA_VERSION,
    SNAPSHOT_FIELD_SPECS,
    stable_scenario_id_hash,
)
from scenario_manifest import ScenarioManifest


BASELINE_SCHEMA_VERSION = "uav-hrl-offline-baseline-v1"
FIXED_SAMPLE_SCHEMA_VERSION = "uav-hrl-fixed-offline-sample-v1"
SAMPLING_METADATA_SCHEMA_VERSION = "uav-hrl-llm-sampling-v1"
DEFAULT_SAMPLES_PER_SOURCE = 1000
DEFAULT_SEED = 20260927
DEFAULT_BATCH_SIZE = 128
DEFAULT_DISTANCE_EPSILON = 1e-8
DEFAULT_REWARD_EPSILON = 1e-12
DEFAULT_LAMBDAS = (
    0.0,
    0.00007068881826573124,
    0.00013062248244154897,
    0.00013498938075010304,
    0.0001477291950503594,
)
TIME_SEGMENTS = ("early", "middle", "late")
CORE_FIELDS = (
    "state",
    "delivered_mbits",
    "total_mobility_energy",
    "c9_penalty",
    "c10_penalty",
    "com_range_penalty",
    "current_movement_mask",
    "movement_mask_valid",
)
TRACE_FIELDS = (
    "auxiliary_valid",
    "episode_id",
    "td3_step",
    "global_transition_id",
    "scenario_index",
    "scenario_id_hash",
    "dinkelbach_lambda",
)
CURRENT_FIELDS = tuple(f"current_{name}" for name in SNAPSHOT_FIELD_SPECS)
FIXED_ARRAY_FIELDS = CORE_FIELDS + TRACE_FIELDS + CURRENT_FIELDS


class BaselineDataError(ValueError):
    """The offline source or fixed artifact violates its declared contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(arrays: dict[str, np.ndarray], field_order: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for field in field_order:
        array = np.ascontiguousarray(arrays[field])
        digest.update(field.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def allocate_output_directory(
    *, output_root: str | Path, output_dir: str | Path | None = None
) -> Path:
    if output_dir is not None:
        candidate = Path(output_dir).resolve()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.mkdir()
        return candidate
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        candidate = root / f"baseline-{stamp}-{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique directory below {root}")


def _as_column(array: np.ndarray, field: str, rows: int) -> np.ndarray:
    if array.shape != (rows, 1):
        raise BaselineDataError(
            f"{field} must have shape ({rows}, 1), got {array.shape}"
        )
    return array[:, 0]


def _validate_source_files(directory: Path) -> tuple[Path, Path, Path]:
    replay = directory / "joint_replay.npz"
    metadata = directory / "metadata.json"
    manifest = directory / "scenario_manifest.json"
    missing = [str(path.name) for path in (replay, metadata, manifest) if not path.is_file()]
    if missing:
        raise BaselineDataError(
            f"sampling source {directory} is missing: {', '.join(missing)}"
        )
    return replay, metadata, manifest


def _compatibility_contract(metadata: dict, manifest: dict) -> dict:
    checkpoint_contract = json.loads(
        json.dumps(metadata.get("source_checkpoint_contract"))
    )
    routing = (checkpoint_contract or {}).get("routing_agent_configuration")
    if isinstance(routing, dict):
        # These describe checkpoint progress, not the observation/action/reward or
        # environment contract needed to combine offline samples.
        for key in (
            "routing_optimizer_update_count",
            "routing_target_update_count",
            "reward_optimizer_update_count",
            "reward_target_update_count",
            "cost_optimizer_update_count",
            "cost_target_update_count",
            "lambda_cost",
            "cost_multiplier_update_count",
            "episode_violation_accumulator",
            "episode_eligible_packet_accumulator",
            "last_episode_violation_count",
            "last_episode_eligible_packet_count",
            "last_episode_violation_probability",
            "last_lambda_cost_used",
            "last_lambda_cost_after",
            "last_lambda_update_status",
            "cost_multiplier_skipped_episode_count",
        ):
            routing.pop(key, None)
    return {
        "sampling_schema_version": metadata.get("schema_version"),
        "method_id": metadata.get("method_id"),
        "episode_seconds": metadata.get("episode_seconds"),
        "joint_replay_field_order": metadata.get("joint_replay_field_order"),
        "replay_auxiliary": metadata.get("replay_auxiliary"),
        "source_checkpoint_contract": checkpoint_contract,
        "source_training_environment_contract": metadata.get(
            "source_training_environment_contract"
        ),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_generator_config": manifest.get("generator_config"),
    }


def _time_segment(times: np.ndarray, horizon: float) -> np.ndarray:
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise BaselineDataError("episode_seconds must be finite and positive")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0) or np.any(times >= horizon):
        raise BaselineDataError(
            "current snapshot times must be finite and within [0, episode_seconds)"
        )
    result = np.full(times.shape, 2, dtype=np.int8)
    result[times < (2.0 * horizon / 3.0)] = 1
    result[times < (horizon / 3.0)] = 0
    return result


def _validate_id_padding(arrays: dict[str, np.ndarray]) -> None:
    checks = (
        ("current_sr_id", "current_sr_observable"),
        ("current_sr_roi_id", "current_sr_roi_mapping_valid"),
        ("current_roi_id", "current_roi_observable"),
        ("current_task_target_id", "current_task_target_valid"),
        ("current_vs_uav_id", "current_vs_pair_valid"),
        ("current_vs_roi_id", "current_vs_pair_valid"),
    )
    for value_field, valid_field in checks:
        invalid = ~np.asarray(arrays[valid_field], dtype=bool)
        if np.any(np.asarray(arrays[value_field])[invalid] != -1):
            raise BaselineDataError(
                f"{value_field} does not use -1 for entries invalidated by {valid_field}"
            )


def load_sampling_source(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    replay_path, metadata_path, manifest_path = _validate_source_files(directory)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The loader independently verifies schema/config/content fingerprints and
    # scenario identities instead of trusting the JSON's stored hash fields.
    ScenarioManifest.load(manifest_path)
    if metadata.get("schema_version") != SAMPLING_METADATA_SCHEMA_VERSION:
        raise BaselineDataError(
            "unsupported sampling metadata schema: "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("status") != "complete" or metadata.get("complete") is not True:
        raise BaselineDataError(f"sampling source is not complete: {directory}")
    if metadata.get("collector_wrapped") is not False:
        raise BaselineDataError("wrapped sampling replay is not supported")
    auxiliary = metadata.get("replay_auxiliary") or {}
    if auxiliary.get("schema_version") != REPLAY_AUXILIARY_SCHEMA_VERSION:
        raise BaselineDataError(
            "incompatible replay auxiliary schema: "
            f"{auxiliary.get('schema_version')!r}"
        )
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise BaselineDataError("scenario manifest has no episodes")
    if manifest.get("content_hash") != metadata.get("scenario_manifest_hash"):
        raise BaselineDataError("scenario manifest hash disagrees with metadata")
    if [str(item.get("scenario_id")) for item in episodes] != metadata.get(
        "scenario_ids_by_index"
    ):
        raise BaselineDataError("scenario id ordering disagrees with sampling metadata")

    expected_fields = tuple(metadata.get("joint_replay_field_order") or ())
    required = set(FIXED_ARRAY_FIELDS)
    if not expected_fields or not required.issubset(expected_fields):
        missing = sorted(required.difference(expected_fields))
        raise BaselineDataError(
            f"sampling metadata is missing required replay fields: {missing}"
        )
    with np.load(replay_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_fields):
            missing = sorted(set(expected_fields).difference(archive.files))
            extra = sorted(set(archive.files).difference(expected_fields))
            raise BaselineDataError(
                f"replay field set disagrees with metadata; missing={missing}, extra={extra}"
            )
        arrays = {field: np.asarray(archive[field]) for field in FIXED_ARRAY_FIELDS}

    rows = int(arrays["state"].shape[0])
    if rows <= 0 or metadata.get("transition_count") != rows:
        raise BaselineDataError("transition_count disagrees with replay rows")
    if arrays["state"].shape != (rows, MOVEMENT_STATE_DIM):
        raise BaselineDataError(
            f"state must have shape ({rows}, {MOVEMENT_STATE_DIM})"
        )
    if arrays["state"].dtype != np.dtype(np.float32):
        raise BaselineDataError("state must use float32 storage")
    for field in ("delivered_mbits", "total_mobility_energy", "c9_penalty", "c10_penalty", "com_range_penalty"):
        _as_column(arrays[field], field, rows)
        if arrays[field].dtype != np.dtype(np.float32):
            raise BaselineDataError(f"{field} must use float32 storage")
    for field in TRACE_FIELDS:
        _as_column(arrays[field], field, rows)
    if arrays["movement_mask_valid"].shape != (rows, 1):
        raise BaselineDataError("movement_mask_valid has an invalid shape")
    if arrays["movement_mask_valid"].dtype != np.dtype(bool):
        raise BaselineDataError("movement_mask_valid must use bool storage")
    if arrays["current_movement_mask"].shape[0] != rows:
        raise BaselineDataError("current_movement_mask row count is invalid")
    if arrays["current_movement_mask"].dtype != np.dtype(bool):
        raise BaselineDataError("current_movement_mask must use bool storage")
    for name, spec in SNAPSHOT_FIELD_SPECS.items():
        field = f"current_{name}"
        expected_shape = (rows, *tuple(spec["shape"]))
        if arrays[field].shape != expected_shape:
            raise BaselineDataError(
                f"{field} has shape {arrays[field].shape}, expected {expected_shape}"
            )
        if arrays[field].dtype != np.dtype(spec["dtype"]):
            raise BaselineDataError(
                f"{field} has dtype {arrays[field].dtype}, expected {np.dtype(spec['dtype'])}"
            )
    for field, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating) and np.any(~np.isfinite(array)):
            raise BaselineDataError(f"{field} contains non-finite values")
    _validate_id_padding(arrays)

    global_ids = _as_column(arrays["global_transition_id"], "global_transition_id", rows)
    if not np.array_equal(global_ids, np.arange(rows, dtype=global_ids.dtype)):
        raise BaselineDataError(
            "sampling replay must retain unwrapped monotonic global transition order"
        )
    episode_ids = _as_column(arrays["episode_id"], "episode_id", rows).astype(np.int64)
    steps = _as_column(arrays["td3_step"], "td3_step", rows).astype(np.int64)
    scenario_indices = _as_column(arrays["scenario_index"], "scenario_index", rows).astype(np.int64)
    scenario_hashes = _as_column(arrays["scenario_id_hash"], "scenario_id_hash", rows)
    if np.any(episode_ids < 0) or np.any(steps < 0):
        raise BaselineDataError("episode_id and td3_step must be non-negative")
    if np.any(scenario_indices < 0) or np.any(scenario_indices >= len(episodes)):
        raise BaselineDataError("scenario_index is outside the manifest")
    expected_hashes = np.asarray(
        [stable_scenario_id_hash(episodes[index]["scenario_id"]) for index in scenario_indices],
        dtype=np.int64,
    )
    if not np.array_equal(scenario_hashes, expected_hashes):
        raise BaselineDataError("scenario_id_hash does not match the manifest")
    if len(set(zip(episode_ids.tolist(), steps.tolist()))) != rows:
        raise BaselineDataError("episode_id/td3_step does not uniquely identify replay rows")

    horizon = float(metadata.get("episode_seconds", 0.0))
    times = _as_column(
        arrays["current_snapshot_time_s"], "current_snapshot_time_s", rows
    ).astype(np.float64)
    segments = _time_segment(times, horizon)
    roi_counts = np.asarray(
        [int(episodes[index]["num_GT"]) for index in scenario_indices], dtype=np.int16
    )
    scenario_ids = [str(episodes[index]["scenario_id"]) for index in scenario_indices]
    usable = (
        _as_column(arrays["auxiliary_valid"], "auxiliary_valid", rows).astype(bool)
        & _as_column(
            arrays["current_snapshot_valid"], "current_snapshot_valid", rows
        ).astype(bool)
        & _as_column(
            arrays["movement_mask_valid"], "movement_mask_valid", rows
        ).astype(bool)
    )
    exclusion = {
        "auxiliary_invalid": int(np.count_nonzero(~arrays["auxiliary_valid"][:, 0])),
        "current_snapshot_invalid": int(
            np.count_nonzero(~arrays["current_snapshot_valid"][:, 0])
        ),
        "movement_mask_invalid": int(
            np.count_nonzero(~arrays["movement_mask_valid"][:, 0])
        ),
        "excluded_union": int(np.count_nonzero(~usable)),
    }

    hashes = {
        "joint_replay_sha256": _file_hash(replay_path),
        "metadata_sha256": _file_hash(metadata_path),
        "scenario_manifest_sha256": _file_hash(manifest_path),
    }
    identity_payload = {
        # Deliberately exclude metadata-file bytes: they contain absolute output
        # paths and timestamps. Identical replay/manifest/checkpoint content must
        # retain the same seeded selection after a directory copy.
        "joint_replay_sha256": hashes["joint_replay_sha256"],
        "scenario_manifest_content_hash": manifest.get("content_hash"),
        "checkpoint_completed_episodes": metadata.get("checkpoint_completed_episodes"),
        "checkpoint_artifact_provenance": metadata.get(
            "source_checkpoint_artifact_provenance"
        ),
        "compatibility_sha256": _json_hash(
            _compatibility_contract(metadata, manifest)
        ),
    }
    source_id = _json_hash(identity_payload)
    return {
        "directory": directory,
        "metadata": metadata,
        "manifest": manifest,
        "arrays": arrays,
        "rows": rows,
        "usable": usable,
        "exclusion_counts": exclusion,
        "roi_counts": roi_counts,
        "time_segments": segments,
        "episode_ids": episode_ids,
        "steps": steps,
        "scenario_indices": scenario_indices,
        "scenario_ids": scenario_ids,
        "source_id": source_id,
        "hashes": hashes,
        "compatibility_contract": _compatibility_contract(metadata, manifest),
        "compatibility_hash": _json_hash(_compatibility_contract(metadata, manifest)),
    }


def _priority(keys: Iterable[Any], seed_material: str) -> list[Any]:
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(
            f"{seed_material}:{key}".encode("utf-8")
        ).digest(),
    )


def balanced_allocation(
    capacities: dict[Any, int], total: int, *, seed_material: str
) -> tuple[dict[Any, int], list[dict[str, Any]]]:
    """Water-fill a quota, with stable seeded ties and auditable shortages."""
    capacities = {key: max(0, int(value)) for key, value in capacities.items()}
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    if sum(capacities.values()) < total:
        raise BaselineDataError(
            f"strata provide {sum(capacities.values())} rows, fewer than requested {total}"
        )
    keys = _priority(capacities, seed_material)
    nominal = {key: total // len(keys) for key in keys} if keys else {}
    for key in keys[: total % len(keys) if keys else 0]:
        nominal[key] += 1
    allocation = {key: 0 for key in keys}
    for _ in range(total):
        available = [key for key in keys if allocation[key] < capacities[key]]
        chosen = min(available, key=lambda key: (allocation[key], keys.index(key)))
        allocation[chosen] += 1
    adjustments = []
    for key in keys:
        if allocation[key] != nominal[key]:
            adjustments.append(
                {
                    "stratum": str(key),
                    "capacity": capacities[key],
                    "nominal_quota": nominal[key],
                    "actual_quota": allocation[key],
                    "reason": "local_capacity_shortage_redistribution",
                }
            )
    return allocation, adjustments


def _sample_source(source: dict[str, Any], count: int, seed: int) -> dict[str, Any]:
    usable_rows = np.flatnonzero(source["usable"])
    if usable_rows.size < count:
        raise BaselineDataError(
            f"source {source['directory']} has {usable_rows.size} usable rows, "
            f"fewer than requested {count}"
        )
    base_seed = f"{int(seed)}:{source['source_id']}"
    roi_values = sorted(set(source["roi_counts"][usable_rows].tolist()))
    roi_capacities = {
        int(roi): int(np.count_nonzero(source["usable"] & (source["roi_counts"] == roi)))
        for roi in roi_values
    }
    roi_quota, adjustments = balanced_allocation(
        roi_capacities, count, seed_material=f"{base_seed}:roi"
    )
    selected: list[int] = []
    detail: dict[str, Any] = {}
    for roi in roi_values:
        roi_rows = usable_rows[source["roi_counts"][usable_rows] == roi]
        segment_capacities = {
            segment: int(
                np.count_nonzero(source["time_segments"][roi_rows] == segment)
            )
            for segment in range(3)
        }
        segment_quota, segment_adjustments = balanced_allocation(
            segment_capacities,
            roi_quota[roi],
            seed_material=f"{base_seed}:roi={roi}:segment",
        )
        roi_detail = {
            "capacity": int(roi_rows.size),
            "quota": int(roi_quota[roi]),
            "time_segments": {},
            "adjustments": segment_adjustments,
        }
        for segment in range(3):
            segment_rows = roi_rows[source["time_segments"][roi_rows] == segment]
            episode_values = sorted(set(source["episode_ids"][segment_rows].tolist()))
            episode_capacities = {
                int(episode): int(
                    np.count_nonzero(source["episode_ids"][segment_rows] == episode)
                )
                for episode in episode_values
            }
            episode_quota, episode_adjustments = balanced_allocation(
                episode_capacities,
                segment_quota[segment],
                seed_material=f"{base_seed}:roi={roi}:segment={segment}:episode",
            )
            for episode in episode_values:
                candidates = segment_rows[
                    source["episode_ids"][segment_rows] == episode
                ]
                rng_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{base_seed}:{roi}:{segment}:{episode}".encode("utf-8")
                    ).digest()[:8],
                    "little",
                )
                rng = np.random.default_rng(rng_seed)
                take = episode_quota[episode]
                if take:
                    selected.extend(
                        int(value)
                        for value in rng.choice(candidates, size=take, replace=False)
                    )
            roi_detail["time_segments"][TIME_SEGMENTS[segment]] = {
                "capacity": segment_capacities[segment],
                "quota": segment_quota[segment],
                "episode_quotas": {
                    str(key): value for key, value in sorted(episode_quota.items())
                },
                "adjustments": episode_adjustments,
            }
        detail[str(roi)] = roi_detail
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("stratified sampler produced an invalid selection")
    selected.sort(
        key=lambda row: (
            int(source["roi_counts"][row]),
            int(source["time_segments"][row]),
            int(source["episode_ids"][row]),
            int(source["steps"][row]),
            int(row),
        )
    )
    return {
        "rows": np.asarray(selected, dtype=np.int64),
        "roi_quota": roi_quota,
        "quota_adjustments": adjustments,
        "strata": detail,
    }


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def build_fixed_samples(
    source_directories: Iterable[str | Path],
    *,
    samples_per_source: int = DEFAULT_SAMPLES_PER_SOURCE,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    samples_per_source = int(samples_per_source)
    if samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive")
    resolved = [Path(path).resolve() for path in source_directories]
    if not resolved:
        raise ValueError("at least one sampling source is required")
    if len(set(resolved)) != len(resolved):
        raise BaselineDataError("the same sampling source directory was supplied twice")
    sources = sorted(
        (load_sampling_source(path) for path in resolved), key=lambda item: item["source_id"]
    )
    source_ids = [source["source_id"] for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise BaselineDataError("duplicate sampling source content was supplied")
    compatibility = sources[0]["compatibility_contract"]
    for source in sources[1:]:
        if source["compatibility_contract"] != compatibility:
            raise BaselineDataError(
                "sampling sources have incompatible schemas, state structures, or environment contracts"
            )

    array_parts: dict[str, list[np.ndarray]] = {field: [] for field in FIXED_ARRAY_FIELDS}
    array_parts.update(
        {
            "source_index": [],
            "source_row": [],
            "total_roi_count": [],
            "time_segment": [],
        }
    )
    source_records = []
    selection = []
    offset = 0
    for source_index, source in enumerate(sources):
        sampled = _sample_source(source, samples_per_source, seed)
        rows = sampled["rows"]
        for field in FIXED_ARRAY_FIELDS:
            array_parts[field].append(np.asarray(source["arrays"][field][rows]))
        array_parts["source_index"].append(
            np.full(rows.size, source_index, dtype=np.int16)
        )
        array_parts["source_row"].append(rows.astype(np.int64))
        array_parts["total_roi_count"].append(source["roi_counts"][rows].astype(np.int16))
        array_parts["time_segment"].append(source["time_segments"][rows].astype(np.int8))
        for local, row in enumerate(rows.tolist()):
            selection.append(
                {
                    "fixed_index": offset + local,
                    "source_index": source_index,
                    "source_id": source["source_id"],
                    "source_row": row,
                    "global_transition_id": int(source["arrays"]["global_transition_id"][row, 0]),
                    "checkpoint_completed_episodes": int(
                        source["metadata"].get("checkpoint_completed_episodes")
                    ),
                    "episode_id": int(source["episode_ids"][row]),
                    "td3_step": int(source["steps"][row]),
                    "scenario_index": int(source["scenario_indices"][row]),
                    "scenario_id": source["scenario_ids"][row],
                    "total_roi_count": int(source["roi_counts"][row]),
                    "time_segment": TIME_SEGMENTS[int(source["time_segments"][row])],
                }
            )
        offset += rows.size
        source_records.append(
            {
                "source_index": source_index,
                "source_id": source["source_id"],
                "directory": str(source["directory"]),
                "checkpoint_completed_episodes": source["metadata"].get(
                    "checkpoint_completed_episodes"
                ),
                "hashes": source["hashes"],
                "transition_count": source["rows"],
                "usable_transition_count": int(np.count_nonzero(source["usable"])),
                "exclusion_counts": source["exclusion_counts"],
                "selected_count": int(rows.size),
                "selected_roi_count_distribution": _distribution(
                    source["roi_counts"][rows].tolist()
                ),
                "selected_episode_distribution": _distribution(
                    source["episode_ids"][rows].tolist()
                ),
                "selected_time_segment_distribution": _distribution(
                    [TIME_SEGMENTS[value] for value in source["time_segments"][rows]]
                ),
                "quota_adjustments": sampled["quota_adjustments"],
                "strata": sampled["strata"],
            }
        )
    arrays = {
        field: np.concatenate(parts, axis=0) for field, parts in array_parts.items()
    }
    field_order = tuple(array_parts)
    sample_hash = _array_hash(arrays, field_order)
    metadata = {
        "schema_version": FIXED_SAMPLE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_contract": {
            "seed": int(seed),
            "samples_per_source": samples_per_source,
            "source_sort": "ascending source_id sha256",
            "source_balance": "exact equal count per source",
            "roi_balance": "seeded stable water-fill over actual scenario total RoI count",
            "within_roi_dispersion": "time-segment water-fill, then episode water-fill, then no-replacement seeded row choice",
            "time_segments": {
                "early": "[0,T/3)",
                "middle": "[T/3,2T/3)",
                "late": "[2T/3,T)",
            },
            "selection_ignores": "reward values and Lipschitz results",
        },
        "sample_count": int(arrays["state"].shape[0]),
        "state_dim": int(arrays["state"].shape[1]),
        "field_order": list(field_order),
        "sample_content_sha256": sample_hash,
        "compatibility_contract": compatibility,
        "compatibility_sha256": _json_hash(compatibility),
        "sources": source_records,
        "selection": selection,
        "overall_distributions": {
            "sources": _distribution(arrays["source_index"].tolist()),
            "roi_counts": _distribution(arrays["total_roi_count"].tolist()),
            "time_segments": _distribution(
                [TIME_SEGMENTS[value] for value in arrays["time_segment"]]
            ),
        },
    }
    return arrays, metadata


def reconstruct_baseline_rewards(
    arrays: dict[str, np.ndarray], lambdas: Iterable[float]
) -> dict[str, np.ndarray]:
    delivered = np.asarray(arrays["delivered_mbits"], dtype=np.float64).reshape(-1)
    energy = np.asarray(arrays["total_mobility_energy"], dtype=np.float64).reshape(-1)
    penalties = sum(
        np.asarray(arrays[field], dtype=np.float64).reshape(-1)
        for field in ("c9_penalty", "c10_penalty", "com_range_penalty")
    )
    result = {}
    for value in lambdas:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("lambda values must be finite")
        result[format(value, ".17g")] = delivered - value * energy - penalties
    return result


def _numeric_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
        }
    quantiles = np.quantile(values, [0.05, 0.5, 0.95, 0.99])
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
    }


def _pair_rank(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _sample_trace(metadata: dict[str, Any] | None, index: int) -> Any:
    if metadata is None:
        return {"fixed_index": int(index)}
    selection = metadata.get("selection") or []
    if index >= len(selection):
        return {"fixed_index": int(index)}
    return selection[index]


def _pair_detail(
    i: int,
    j: int,
    *,
    distance: float,
    reward_difference: float,
    rewards: np.ndarray,
    arrays: dict[str, np.ndarray],
    sample_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    def components(index: int) -> dict[str, float]:
        return {
            "timely_delivered_mbit": float(arrays["delivered_mbits"][index, 0]),
            "movement_energy_j": float(arrays["total_mobility_energy"][index, 0]),
            "c9_penalty": float(arrays["c9_penalty"][index, 0]),
            "c10_penalty": float(arrays["c10_penalty"][index, 0]),
            "com_range_penalty": float(arrays["com_range_penalty"][index, 0]),
            "reconstructed_reward": float(rewards[index]),
        }

    return {
        "i": int(i),
        "j": int(j),
        "distance": float(distance),
        "reward_difference": float(reward_difference),
        "ratio": None if distance == 0.0 else float(reward_difference / distance),
        "sample_i": _sample_trace(sample_metadata, i),
        "sample_j": _sample_trace(sample_metadata, j),
        "components_i": components(i),
        "components_j": components(j),
    }


def estimate_empirical_lipschitz(
    states: np.ndarray,
    rewards_by_lambda: dict[str, np.ndarray],
    *,
    arrays: dict[str, np.ndarray],
    sample_metadata: dict[str, Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
    reward_epsilon: float = DEFAULT_REWARD_EPSILON,
) -> dict[str, Any]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("states must be a two-dimensional array")
    n = states.shape[0]
    if n < 2:
        raise ValueError("at least two samples are required")
    if np.any(~np.isfinite(states)):
        raise ValueError("states contain non-finite values")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    distance_epsilon = float(distance_epsilon)
    reward_epsilon = float(reward_epsilon)
    if not math.isfinite(distance_epsilon) or distance_epsilon < 0.0:
        raise ValueError("distance_epsilon must be finite and non-negative")
    if not math.isfinite(reward_epsilon) or reward_epsilon < 0.0:
        raise ValueError("reward_epsilon must be finite and non-negative")
    rewards = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in rewards_by_lambda.items()
    }
    if any(value.size != n or np.any(~np.isfinite(value)) for value in rewards.values()):
        raise ValueError("every reward vector must be finite and aligned with states")

    pair_count = n * (n - 1) // 2
    distances = np.empty(pair_count, dtype=np.float64)
    trackers = {
        key: {
            "main_ratio": -math.inf,
            "main_rank": None,
            "main_pair": None,
            "zero_max_delta": -math.inf,
            "zero_rank": None,
            "zero_pair": None,
            "zero_conflicts": 0,
            "near_max_ratio": -math.inf,
            "near_rank": None,
            "near_pair": None,
            "near_conflicts": 0,
        }
        for key in rewards
    }
    for left_start in range(0, n, batch_size):
        left_stop = min(n, left_start + batch_size)
        for right_start in range(left_start, n, batch_size):
            right_stop = min(n, right_start + batch_size)
            if left_start == right_start:
                local_i, local_j = np.triu_indices(left_stop - left_start, k=1)
                pair_i = local_i + left_start
                pair_j = local_j + right_start
            else:
                pair_i = np.repeat(
                    np.arange(left_start, left_stop, dtype=np.int64),
                    right_stop - right_start,
                )
                pair_j = np.tile(
                    np.arange(right_start, right_stop, dtype=np.int64),
                    left_stop - left_start,
                )
            if pair_i.size == 0:
                continue
            difference = states[pair_i] - states[pair_j]
            distance = np.sqrt(np.einsum("ij,ij->i", difference, difference))
            ranks = _pair_rank(n, pair_i, pair_j)
            distances[ranks] = distance
            main = distance > distance_epsilon
            zero = distance == 0.0
            near = (distance > 0.0) & (distance <= distance_epsilon)
            for key, reward in rewards.items():
                delta = np.abs(reward[pair_i] - reward[pair_j])
                tracker = trackers[key]
                if np.any(main):
                    ratios = delta[main] / distance[main]
                    local = int(np.argmax(ratios))
                    value = float(ratios[local])
                    main_indices = np.flatnonzero(main)
                    position = int(main_indices[local])
                    candidate_rank = int(ranks[position])
                    if value > tracker["main_ratio"] or (
                        value == tracker["main_ratio"]
                        and (
                            tracker["main_rank"] is None
                            or candidate_rank < tracker["main_rank"]
                        )
                    ):
                        tracker["main_ratio"] = value
                        tracker["main_rank"] = candidate_rank
                        tracker["main_pair"] = (
                            int(pair_i[position]),
                            int(pair_j[position]),
                            float(distance[position]),
                            float(delta[position]),
                        )
                if np.any(zero):
                    tracker["zero_conflicts"] += int(
                        np.count_nonzero(delta[zero] > reward_epsilon)
                    )
                    zero_indices = np.flatnonzero(zero)
                    local = int(np.argmax(delta[zero]))
                    position = int(zero_indices[local])
                    value = float(delta[position])
                    candidate_rank = int(ranks[position])
                    if value > tracker["zero_max_delta"] or (
                        value == tracker["zero_max_delta"]
                        and (
                            tracker["zero_rank"] is None
                            or candidate_rank < tracker["zero_rank"]
                        )
                    ):
                        tracker["zero_max_delta"] = value
                        tracker["zero_rank"] = candidate_rank
                        tracker["zero_pair"] = (
                            int(pair_i[position]),
                            int(pair_j[position]),
                            0.0,
                            value,
                        )
                if np.any(near):
                    tracker["near_conflicts"] += int(
                        np.count_nonzero(delta[near] > reward_epsilon)
                    )
                    ratios = delta[near] / distance[near]
                    near_indices = np.flatnonzero(near)
                    local = int(np.argmax(ratios))
                    position = int(near_indices[local])
                    value = float(ratios[local])
                    candidate_rank = int(ranks[position])
                    if value > tracker["near_max_ratio"] or (
                        value == tracker["near_max_ratio"]
                        and (
                            tracker["near_rank"] is None
                            or candidate_rank < tracker["near_rank"]
                        )
                    ):
                        tracker["near_max_ratio"] = value
                        tracker["near_rank"] = candidate_rank
                        tracker["near_pair"] = (
                            int(pair_i[position]),
                            int(pair_j[position]),
                            float(distance[position]),
                            float(delta[position]),
                        )

    zero_mask = distances == 0.0
    near_mask = (distances > 0.0) & (distances <= distance_epsilon)
    main_mask = distances > distance_epsilon
    packed = np.packbits(main_mask, bitorder="little")
    pair_set_hash = hashlib.sha256(
        _canonical_json(
            {
                "sample_count": n,
                "pair_order": "lexicographic i<j using condensed rank",
                "distance_epsilon": distance_epsilon,
                "pair_count": pair_count,
            }
        ).encode("utf-8")
        + b"\0"
        + packed.tobytes()
    ).hexdigest()
    estimates = {}
    for key, reward in rewards.items():
        tracker = trackers[key]

        def detail(pair):
            if pair is None:
                return None
            return _pair_detail(
                pair[0],
                pair[1],
                distance=pair[2],
                reward_difference=pair[3],
                rewards=reward,
                arrays=arrays,
                sample_metadata=sample_metadata,
            )

        estimates[key] = {
            "status": "ok" if tracker["main_pair"] is not None else "unavailable_no_primary_pairs",
            "l_hat": (
                float(tracker["main_ratio"])
                if tracker["main_pair"] is not None
                else None
            ),
            "maximum_pair": detail(tracker["main_pair"]),
            "reward_distribution": _numeric_distribution(reward),
            "zero_distance": {
                "pair_count": int(np.count_nonzero(zero_mask)),
                "reward_conflict_count": int(tracker["zero_conflicts"]),
                "reward_conflict_tolerance": reward_epsilon,
                "maximum_reward_difference_pair": detail(tracker["zero_pair"]),
                "interpretation": "excluded from the primary estimate; nonzero reward differences are state-to-reward conflicts",
            },
            "near_zero_distance": {
                "pair_count": int(np.count_nonzero(near_mask)),
                "reward_difference_above_tolerance_count": int(
                    tracker["near_conflicts"]
                ),
                "maximum_calculable_ratio_pair": detail(tracker["near_pair"]),
                "interpretation": "actual finite ratios are reported but excluded from the primary estimate",
            },
        }
    return {
        "sample_count": n,
        "state_dim": int(states.shape[1]),
        "all_pair_count": pair_count,
        "pair_order": "all unique pairs i<j in fixed sample order",
        "distance_algorithm": (
            "direct float64 coordinate differences and sum of squares in bounded "
            "pair blocks; no N-by-N-by-state allocation and no Gram-matrix subtraction"
        ),
        "distance_dtype": "float64",
        "reward_dtype": "float64",
        "distance_epsilon": distance_epsilon,
        "reward_difference_tolerance": reward_epsilon,
        "primary_pair_count": int(np.count_nonzero(main_mask)),
        "zero_distance_pair_count": int(np.count_nonzero(zero_mask)),
        "near_zero_distance_pair_count": int(np.count_nonzero(near_mask)),
        "primary_pair_set_sha256": pair_set_hash,
        "distance_distribution_all_pairs": _numeric_distribution(distances),
        "distance_distribution_primary_pairs": _numeric_distribution(distances[main_mask]),
        "estimates_by_lambda_mbit_per_joule": estimates,
    }


def save_fixed_samples(
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    pair_contract: dict[str, Any],
    lambda_values: Iterable[float],
) -> tuple[Path, Path]:
    bundle = output / "fixed_samples.npz"
    np.savez_compressed(bundle, **arrays)
    fixed_metadata = {
        **metadata,
        "bundle_file": bundle.name,
        "bundle_sha256": _file_hash(bundle),
        "pair_contract": {
            "sample_order": "fixed_index ascending",
            "all_pairs": "all unique i<j",
            "primary_pair_rule": "original-state Euclidean distance > distance_epsilon",
            "distance_epsilon": pair_contract["distance_epsilon"],
            "reward_difference_tolerance": pair_contract[
                "reward_difference_tolerance"
            ],
            "all_pair_count": pair_contract["all_pair_count"],
            "primary_pair_count": pair_contract["primary_pair_count"],
            "zero_distance_pair_count": pair_contract["zero_distance_pair_count"],
            "near_zero_distance_pair_count": pair_contract[
                "near_zero_distance_pair_count"
            ],
            "primary_pair_set_sha256": pair_contract[
                "primary_pair_set_sha256"
            ],
            "baseline_lambda_values_mbit_per_joule": [
                float(value) for value in lambda_values
            ],
        },
    }
    metadata_path = output / "fixed_samples.json"
    _write_json(metadata_path, fixed_metadata)
    return bundle, metadata_path


def load_fixed_samples(directory: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(directory).resolve()
    metadata_path = directory / "fixed_samples.json"
    if not metadata_path.is_file():
        raise BaselineDataError(f"fixed sample metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != FIXED_SAMPLE_SCHEMA_VERSION:
        raise BaselineDataError("unsupported fixed sample schema")
    bundle = directory / str(metadata.get("bundle_file", "fixed_samples.npz"))
    if not bundle.is_file() or _file_hash(bundle) != metadata.get("bundle_sha256"):
        raise BaselineDataError("fixed sample bundle is missing or its SHA-256 changed")
    field_order = tuple(metadata.get("field_order") or ())
    with np.load(bundle, allow_pickle=False) as archive:
        if set(archive.files) != set(field_order):
            raise BaselineDataError("fixed sample bundle field set changed")
        arrays = {field: np.asarray(archive[field]) for field in field_order}
    if _array_hash(arrays, field_order) != metadata.get("sample_content_sha256"):
        raise BaselineDataError("fixed sample content hash changed")
    count = int(metadata.get("sample_count", -1))
    if arrays["state"].shape != (count, int(metadata.get("state_dim", -1))):
        raise BaselineDataError("fixed sample state shape disagrees with metadata")
    if len(metadata.get("selection") or []) != count:
        raise BaselineDataError("fixed sample trace list has an invalid length")
    return arrays, metadata


def verify_fixed_sources(metadata: dict[str, Any], directories: Iterable[str | Path]) -> None:
    provided = sorted(
        (load_sampling_source(path) for path in directories),
        key=lambda item: item["source_id"],
    )
    expected = metadata.get("sources") or []
    if len(provided) != len(expected):
        raise BaselineDataError("fixed artifact source count does not match supplied sources")
    for actual, saved in zip(provided, expected):
        if actual["source_id"] != saved.get("source_id") or actual["hashes"] != saved.get("hashes"):
            raise BaselineDataError(
                "a supplied sampling source changed since the fixed artifact was created"
            )


def run_baseline(
    *,
    source_directories: Iterable[str | Path] = (),
    fixed_sample_directory: str | Path | None = None,
    samples_per_source: int = DEFAULT_SAMPLES_PER_SOURCE,
    seed: int = DEFAULT_SEED,
    lambdas: Iterable[float] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    distance_epsilon: float | None = None,
    reward_epsilon: float | None = None,
    output_root: str | Path = "results/llm_baselines",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    source_directories = tuple(source_directories)
    if fixed_sample_directory is None:
        if distance_epsilon is None:
            distance_epsilon = DEFAULT_DISTANCE_EPSILON
        if reward_epsilon is None:
            reward_epsilon = DEFAULT_REWARD_EPSILON
        arrays, fixed_metadata = build_fixed_samples(
            source_directories,
            samples_per_source=samples_per_source,
            seed=seed,
        )
        reloaded = False
    else:
        arrays, fixed_metadata = load_fixed_samples(fixed_sample_directory)
        if source_directories:
            verify_fixed_sources(fixed_metadata, source_directories)
        reloaded = True
        saved_pair = fixed_metadata.get("pair_contract") or {}
        if distance_epsilon is None:
            distance_epsilon = saved_pair.get("distance_epsilon")
        if reward_epsilon is None:
            reward_epsilon = saved_pair.get("reward_difference_tolerance")
        if float(distance_epsilon) != float(saved_pair.get("distance_epsilon")):
            raise BaselineDataError(
                "reload must use the fixed artifact's distance_epsilon"
            )
        if float(reward_epsilon) != float(
            saved_pair.get("reward_difference_tolerance")
        ):
            raise BaselineDataError(
                "reload must use the fixed artifact's reward difference tolerance"
            )
    if lambdas is None:
        if reloaded:
            lambdas = (fixed_metadata.get("pair_contract") or {}).get(
                "baseline_lambda_values_mbit_per_joule"
            )
        if lambdas is None:
            lambdas = DEFAULT_LAMBDAS
    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values:
        raise ValueError("at least one lambda is required")
    if reloaded:
        saved_lambdas = tuple(
            float(value)
            for value in (fixed_metadata.get("pair_contract") or {}).get(
                "baseline_lambda_values_mbit_per_joule", ()
            )
        )
        if saved_lambdas and lambda_values != saved_lambdas:
            raise BaselineDataError(
                "reload must use the fixed artifact's baseline lambda list"
            )
    rewards = reconstruct_baseline_rewards(arrays, lambda_values)
    estimates = estimate_empirical_lipschitz(
        arrays["state"],
        rewards,
        arrays=arrays,
        sample_metadata=fixed_metadata,
        batch_size=batch_size,
        distance_epsilon=distance_epsilon,
        reward_epsilon=reward_epsilon,
    )
    if reloaded:
        saved_pair = fixed_metadata.get("pair_contract") or {}
        for key in ("all_pair_count", "primary_pair_count", "primary_pair_set_sha256"):
            if estimates[key] != saved_pair.get(key):
                raise BaselineDataError(
                    f"recomputed pair contract {key} disagrees with the fixed artifact"
                )
    output = allocate_output_directory(output_root=output_root, output_dir=output_dir)
    if not reloaded:
        bundle, fixed_path = save_fixed_samples(
            output, arrays, fixed_metadata, estimates, lambda_values
        )
        fixed_reference = {
            "directory": str(output),
            "metadata": fixed_path.name,
            "bundle": bundle.name,
            "sample_content_sha256": fixed_metadata["sample_content_sha256"],
        }
    else:
        fixed_reference = {
            "directory": str(Path(fixed_sample_directory).resolve()),
            "metadata": "fixed_samples.json",
            "bundle": fixed_metadata.get("bundle_file"),
            "sample_content_sha256": fixed_metadata["sample_content_sha256"],
        }
    report = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "empirical_scope": (
            "finite fixed sample and fixed original-state pair set; this is not a "
            "global Lipschitz constant"
        ),
        "reloaded_fixed_samples": reloaded,
        "fixed_samples": fixed_reference,
        "parameters": {
            "samples_per_source": fixed_metadata["sampling_contract"][
                "samples_per_source"
            ],
            "sampling_seed": fixed_metadata["sampling_contract"]["seed"],
            "lambda_values_mbit_per_joule": list(lambda_values),
            "batch_size": int(batch_size),
            "distance_epsilon": float(distance_epsilon),
            "reward_difference_tolerance": float(reward_epsilon),
        },
        "reward_contract": {
            "formula": "B_timely_Mbit - lambda_Mbit_per_J * E_movement_J - P_C9 - P_C10 - P_COM",
            "delivered_unit": "Mbit",
            "energy_unit": "J",
            "lambda_unit": "Mbit/J",
            "penalty_weights": {"C9": 1.0, "C10": 1.0, "COM": 1.0},
            "stored_checkpoint_lambda_used": False,
            "clipping_or_normalization": "none",
        },
        "source_summary": fixed_metadata["sources"],
        "sample_distributions": fixed_metadata["overall_distributions"],
        "lipschitz": estimates,
    }
    _write_json(output / "baseline_report.json", report)
    return {"output_directory": str(output), "report": report}
