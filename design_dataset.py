"""Deterministic offline collection of centralized joint design transitions."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np

from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    movement_state_feature_schema,
    projected_joint_action_schema,
)
from com_capacity_calibration import load_com_capacity_reference
from evaluation_metrics import (
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    METRIC_COLUMNS,
    OPTIONAL_METRIC_COLUMNS,
    PACKET_METRIC_COLUMNS,
)
from experiment_config import (
    MOVEMENT_REPLAY_CONTRACT_VERSION,
    TASK_POTENTIAL_CONTRACT_VERSION,
    FORMAL_EXPERIMENT_DEFAULTS,
    MethodSpec,
    effective_training_config,
)
from experiment_paths import (
    design_run_directory,
    design_run_identity,
    prepare_run_directory,
    validate_run_directory_preflight,
    write_run_status,
)
from scenario_manifest import (
    ScenarioManifest,
    validate_manifest_initial_topologies,
)
from training_checkpoint import (
    checkpoint_metadata_fingerprint,
    inspect_model_checkpoint,
)


DESIGN_DATASET_SCHEMA_VERSION = 4
DESIGN_TRANSITIONS_FILENAME = "design_transitions.npz"
DESIGN_METADATA_FILENAME = "design_dataset_metadata.json"
DESIGN_EPISODES_CSV = "per_episode.csv"
DESIGN_EPISODES_JSONL = "per_episode.jsonl"

ARRAY_NAMES = (
    "state",
    "projected_joint_action",
    "next_state",
    "done",
    "not_done",
    "delivered_mbits",
    "total_mobility_energy_j",
    "phi_search_t",
    "phi_search_t1",
    "phi_vs_t",
    "phi_vs_t1",
    "phi_com_t",
    "phi_com_t1",
    "phi_relay_t",
    "phi_relay_t1",
    "reward_at_checkpoint_lambda",
    "checkpoint_lambda",
    "episode_index",
    "movement_step",
    "global_transition_index",
    "scenario_index",
    "scenario_id",
)

FLOAT_COMPONENTS = (
    "delivered_mbits",
    "total_mobility_energy_j",
    "phi_search_t",
    "phi_search_t1",
    "phi_vs_t",
    "phi_vs_t1",
    "phi_com_t",
    "phi_com_t1",
    "phi_relay_t",
    "phi_relay_t1",
    "reward_at_checkpoint_lambda",
    "checkpoint_lambda",
)

INTEGER_EPISODE_COLUMNS = {
    "training_seed",
    "checkpoint_completed_episodes",
    "num_GT",
    "fov_timely_delivered_packets",
    "com_timely_delivered_packets",
    "fov_deadline_violations",
    "com_deadline_violations",
    "total_deadline_violations",
    "eligible_packet_count",
    "sr_admission_drop_count",
    "routing_wait_count",
    "partial_transmission_count",
    "slot_budget_violation_count",
}
PACKET_INTEGER_COLUMNS = {
    column for column in PACKET_METRIC_COLUMNS if column.endswith("_packets")
}


class DesignTransitionCollector:
    def __init__(self):
        self._records = []

    def __call__(self, transition):
        record = dict(transition)
        for field in ("state", "projected_joint_action", "next_state"):
            record[field] = np.asarray(record[field], dtype=np.float32).copy()
        self._records.append(record)

    def arrays(self):
        if not self._records:
            raise RuntimeError("design collector produced no transitions")
        arrays = {
            "state": np.stack([row["state"] for row in self._records]).astype(
                np.float32, copy=False
            ),
            "projected_joint_action": np.stack(
                [row["projected_joint_action"] for row in self._records]
            ).astype(np.float32, copy=False),
            "next_state": np.stack(
                [row["next_state"] for row in self._records]
            ).astype(np.float32, copy=False),
            "done": np.asarray(
                [row["done"] for row in self._records], dtype=np.bool_
            ),
            "not_done": np.asarray(
                [row["not_done"] for row in self._records], dtype=np.float32
            ),
        }
        arrays.update(
            {
                field: np.asarray(
                    [row[field] for row in self._records], dtype=np.float64
                )
                for field in FLOAT_COMPONENTS
            }
        )
        for field in (
            "episode_index",
            "movement_step",
            "global_transition_index",
            "scenario_index",
        ):
            arrays[field] = np.asarray(
                [row[field] for row in self._records], dtype=np.int64
            )
        arrays["scenario_id"] = np.asarray(
            [str(row["scenario_id"]) for row in self._records], dtype=np.str_
        )
        return arrays


def reconstruct_reward(arrays, *, beta_search, beta_vs, beta_com, beta_relay):
    return (
        arrays["delivered_mbits"]
        - arrays["checkpoint_lambda"]
        * arrays["total_mobility_energy_j"]
        + float(beta_search)
        * (arrays["phi_search_t1"] - arrays["phi_search_t"])
        + float(beta_vs) * (arrays["phi_vs_t1"] - arrays["phi_vs_t"])
        + float(beta_com) * (arrays["phi_com_t1"] - arrays["phi_com_t"])
        + float(beta_relay)
        * (arrays["phi_relay_t1"] - arrays["phi_relay_t"])
    )


def validate_design_arrays(
    arrays,
    *,
    episode_count,
    episode_seconds,
    beta_search,
    beta_vs,
    beta_com,
    beta_relay,
):
    missing = set(ARRAY_NAMES).difference(arrays)
    extra = set(arrays).difference(ARRAY_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"design dataset arrays are incompatible: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    expected_count = int(episode_count) * int(episode_seconds)
    expected_shapes = {
        "state": (expected_count, MOVEMENT_STATE_DIM),
        "projected_joint_action": (expected_count, JOINT_ACTION_DIM),
        "next_state": (expected_count, MOVEMENT_STATE_DIM),
    }
    for field, expected in expected_shapes.items():
        if tuple(arrays[field].shape) != expected:
            raise RuntimeError(
                f"design dataset {field} shape is incompatible: "
                f"{arrays[field].shape} != {expected}"
            )
    for field in ARRAY_NAMES[3:]:
        if tuple(arrays[field].shape) != (expected_count,):
            raise RuntimeError(
                f"design dataset {field} shape is incompatible: "
                f"{arrays[field].shape}"
            )
    for field, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise RuntimeError(f"design dataset {field} contains NaN or Inf")
    state_schema = movement_state_feature_schema()
    state_minimum = np.asarray(
        [feature["minimum"] for feature in state_schema["features"]],
        dtype=np.float32,
    )
    state_maximum = np.asarray(
        [feature["maximum"] for feature in state_schema["features"]],
        dtype=np.float32,
    )
    for field in ("state", "next_state"):
        if np.any(arrays[field] < state_minimum) or np.any(
            arrays[field] > state_maximum
        ):
            raise RuntimeError(f"design dataset {field} violates feature bounds")
    if np.any(arrays["projected_joint_action"] < -1.0) or np.any(
        arrays["projected_joint_action"] > 1.0
    ):
        raise RuntimeError("projected joint action violates [-1, 1] bounds")
    if not np.all(
        arrays["checkpoint_lambda"] == arrays["checkpoint_lambda"][0]
    ):
        raise RuntimeError("checkpoint lambda changed during design collection")
    expected_global = np.arange(expected_count, dtype=np.int64)
    expected_episode = np.repeat(
        np.arange(episode_count, dtype=np.int64), episode_seconds
    )
    expected_step = np.tile(
        np.arange(episode_seconds, dtype=np.int64), episode_count
    )
    expected_done = expected_step == int(episode_seconds) - 1
    checks = {
        "global_transition_index": np.array_equal(
            arrays["global_transition_index"], expected_global
        ),
        "episode_index": np.array_equal(arrays["episode_index"], expected_episode),
        "scenario_index": np.array_equal(
            arrays["scenario_index"], expected_episode
        ),
        "movement_step": np.array_equal(arrays["movement_step"], expected_step),
        "done": np.array_equal(arrays["done"], expected_done),
        "not_done": np.array_equal(
            arrays["not_done"], 1.0 - expected_done.astype(np.float32)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"design transition ordering is invalid: {failed}")
    scenario_ids = arrays["scenario_id"].reshape(episode_count, episode_seconds)
    if any(len(set(row.tolist())) != 1 for row in scenario_ids):
        raise RuntimeError("scenario_id changes within an episode")
    reconstructed = reconstruct_reward(
        arrays,
        beta_search=beta_search,
        beta_vs=beta_vs,
        beta_com=beta_com,
        beta_relay=beta_relay,
    )
    if not np.allclose(
        reconstructed,
        arrays["reward_at_checkpoint_lambda"],
        rtol=0.0,
        atol=1e-12,
    ):
        mismatch = int(
            np.flatnonzero(
                ~np.isclose(
                    reconstructed,
                    arrays["reward_at_checkpoint_lambda"],
                    rtol=0.0,
                    atol=1e-12,
                )
            )[0]
        )
        raise RuntimeError(
            "checkpoint-lambda reward reconstruction mismatch at transition "
            f"{mismatch}"
        )
    return arrays


def _array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _normalize_episode_row(row):
    normalized = {}
    for column in EPISODE_COLUMNS:
        value = row[column]
        if column in INTEGER_EPISODE_COLUMNS:
            normalized[column] = int(value)
        elif column in METRIC_COLUMNS:
            if column in OPTIONAL_METRIC_COLUMNS and value in (None, ""):
                normalized[column] = None
                continue
            number = float(value)
            if not np.isfinite(number):
                raise RuntimeError(f"evaluation metric {column} is non-finite")
            normalized[column] = number
        elif column in PACKET_METRIC_COLUMNS:
            if value is None or value == "":
                normalized[column] = None
                continue
            number = int(value) if column in PACKET_INTEGER_COLUMNS else float(value)
            if not np.isfinite(number):
                raise RuntimeError(f"packet metric {column} is non-finite")
            normalized[column] = number
        else:
            normalized[column] = str(value)
    return normalized


def metrics_fingerprint(rows):
    normalized = [_normalize_episode_row(row) for row in rows]
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def read_reference_episode_csv(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference per-episode CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(EPISODE_COLUMNS):
            raise RuntimeError("reference per-episode CSV schema is incompatible")
        rows = [_normalize_episode_row(row) for row in reader]
    return rows


def validate_reference_identity(
    rows,
    *,
    method_id,
    training_seed,
    split,
    evaluation_manifest_hash,
    training_manifest_hash,
    checkpoint_completed_episodes,
    checkpoint_fingerprint,
    expected_scenario_ids,
):
    by_scenario = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        if scenario_id in by_scenario:
            raise RuntimeError(
                f"reference contains duplicate scenario_id: {scenario_id}"
            )
        by_scenario[scenario_id] = row
    missing_scenarios = [
        scenario_id
        for scenario_id in expected_scenario_ids
        if scenario_id not in by_scenario
    ]
    if missing_scenarios:
        raise RuntimeError(
            "reference is missing requested scenarios: "
            f"{missing_scenarios[:3]}"
        )
    selected = [by_scenario[scenario_id] for scenario_id in expected_scenario_ids]
    expected = {
        "method_id": str(method_id),
        "training_seed": int(training_seed),
        "evaluation_split": str(split),
        "evaluation_manifest_hash": str(evaluation_manifest_hash),
        "training_manifest_hash": str(training_manifest_hash),
        "checkpoint_completed_episodes": int(checkpoint_completed_episodes),
        "checkpoint_metadata_fingerprint": str(checkpoint_fingerprint),
    }
    for index, row in enumerate(selected):
        for field, value in expected.items():
            if row[field] != value:
                raise RuntimeError(
                    "reference identity mismatch at row "
                    f"{index}, field {field}: reference={row[field]!r}, "
                    f"expected={value!r}"
                )
    return selected


def validate_reference_metrics(reference_rows, collected_rows):
    normalized = [_normalize_episode_row(row) for row in collected_rows]
    if len(reference_rows) != len(normalized):
        raise RuntimeError("reference and collected episode counts differ")
    for index, (reference, collected) in enumerate(zip(reference_rows, normalized)):
        for field in EPISODE_COLUMNS:
            if reference[field] != collected[field]:
                raise RuntimeError(
                    "reference metric mismatch at row "
                    f"{index}, scenario={collected['scenario_id']!r}, "
                    f"field={field}: reference={reference[field]!r}, "
                    f"collected={collected[field]!r}"
                )
    return normalized


def _git_commit_sha():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_metadata(preflight, arrays, result, run_dir, reference_rows):
    checkpoint_metadata = preflight["checkpoint"]["metadata"]
    experiment = checkpoint_metadata["experiment"]
    formal_config = experiment["formal_config"]
    array_descriptors = {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "sha256": _array_sha256(arrays[name]),
        }
        for name in ARRAY_NAMES
    }
    state_schema = movement_state_feature_schema()
    action_schema = projected_joint_action_schema()
    metadata = {
        "schema_version": DESIGN_DATASET_SCHEMA_VERSION,
        "method_id": preflight["method"].method_id,
        "training_seed": int(preflight["training_seed"]),
        "checkpoint_path": str(preflight["checkpoint"]["checkpoint_dir"]),
        "checkpoint_completed_episodes": int(
            preflight["checkpoint"]["completed_episode"]
        ),
        "checkpoint_metadata_fingerprint": preflight["checkpoint_fingerprint"],
        "training_manifest_hash": experiment["manifest_hash"],
        "validation_manifest_hash": preflight["manifest"].content_hash,
        "split": preflight["manifest"].split,
        "scenario_count": int(preflight["episode_count"]),
        "transition_count": int(arrays["state"].shape[0]),
        "centralized_actor_calls": int(result["environment_actor_calls"]),
        "state_dimension": MOVEMENT_STATE_DIM,
        "action_dimension": JOINT_ACTION_DIM,
        "episode_duration_seconds": int(preflight["config"].episode_seconds),
        "routing_slots_per_movement_interval": 4,
        "td3_gamma": 1.0,
        "ddqn_gamma": 0.99,
        "checkpoint_lambda": float(arrays["checkpoint_lambda"][0]),
        "exploration": {
            "td3_noise": 0.0,
            "ddqn_epsilon": 0.0,
            "ddqn_logits_noise": 0.0,
        },
        "action_perturbation": "disabled",
        "potential_weights": {
            "beta_search": float(formal_config["beta_search"]),
            "beta_vs": float(formal_config["beta_vs"]),
            "beta_com": float(formal_config["beta_com"]),
            "beta_relay": float(formal_config["beta_relay"]),
        },
        "task_potential_contract_version": TASK_POTENTIAL_CONTRACT_VERSION,
        "movement_replay_contract_version": MOVEMENT_REPLAY_CONTRACT_VERSION,
        "potential_boundary_semantics": (
            "phi_current uses current decision-state backlog; phi_next uses next "
            "decision-state backlog; terminal phi_next is zero"
        ),
        "reward_components": {
            "definition": (
                "delivered_mbits - lambda * total_mobility_energy_j + "
                "sum(beta_d * (phi_d_t1 - phi_d_t)); terminal phi_d_t1 is zero"
            ),
            "delivered_mbits_unit": "Mbit delivered to ground station within deadline",
            "total_mobility_energy_j_unit": "joule across all UAVs in one second",
            "potentials_unit": "dimensionless",
            "checkpoint_lambda_unit": "Mbit per joule",
            "reward_unit": "Mbit-equivalent shaped reward",
            "terminal_next_potential": 0.0,
        },
        "state_feature_schema": state_schema,
        "continuous_state_indices": state_schema["continuous_indices"],
        "discrete_state_indices": state_schema["discrete_indices"],
        "next_state_schema": "identical to state_feature_schema",
        "projected_joint_action_schema": action_schema,
        "terminal_semantics": (
            "done is true only at movement_step 59; not_done=0 and all "
            "effective next potentials are zero; episodes never link"
        ),
        "transition_definition": (
            "movement-boundary state -> one deterministic actor call -> projected "
            "derived joint raw action -> synchronous proposals and movement -> four routing "
            "slots -> next movement-boundary state"
        ),
        "git_commit_sha": _git_commit_sha(),
        "arrays": array_descriptors,
        "ordinary_evaluation_metrics_fingerprint": metrics_fingerprint(
            result["episode_metrics"]
        ),
        "reference_per_episode": (
            str(preflight["reference_path"])
            if preflight["reference_path"] is not None
            else None
        ),
        "reference_metrics_fingerprint": (
            metrics_fingerprint(reference_rows) if reference_rows is not None else None
        ),
        "evaluation_invariants": result["evaluation_invariants"],
        "learning_state_fingerprints": result["evaluation_state_fingerprints"],
        "run_directory": str(run_dir),
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    stable = dict(metadata)
    stable.pop("collection_timestamp")
    stable.pop("run_directory")
    stable.pop("reference_per_episode")
    metadata["deterministic_content_fingerprint"] = hashlib.sha256(
        _canonical_json_bytes(stable)
    ).hexdigest()
    return metadata


def _atomic_write_dataset(run_dir, arrays, metadata, episode_rows):
    run_dir = Path(run_dir).resolve()
    targets = {
        "npz": run_dir / DESIGN_TRANSITIONS_FILENAME,
        "metadata": run_dir / DESIGN_METADATA_FILENAME,
        "csv": run_dir / DESIGN_EPISODES_CSV,
        "jsonl": run_dir / DESIGN_EPISODES_JSONL,
    }
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("design dataset artifacts already exist")
    token = uuid.uuid4().hex
    temporary = {
        name: path.parent / f".{path.name}.tmp-{token}"
        for name, path in targets.items()
    }
    try:
        with temporary["npz"].open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary["npz"], allow_pickle=False) as loaded:
            if set(loaded.files) != set(ARRAY_NAMES):
                raise RuntimeError("temporary NPZ array schema is invalid")
            for name in ARRAY_NAMES:
                if not np.array_equal(loaded[name], arrays[name]):
                    raise RuntimeError(f"temporary NPZ verification failed: {name}")
        with temporary["csv"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            writer.writerows(
                [{column: row[column] for column in EPISODE_COLUMNS} for row in episode_rows]
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary["jsonl"].write_bytes(
            b"".join(
                _canonical_json_bytes(
                    {column: row[column] for column in EPISODE_COLUMNS}
                )
                for row in episode_rows
            )
        )
        temporary["metadata"].write_bytes(_canonical_json_bytes(metadata))
        for name in ("npz", "csv", "jsonl", "metadata"):
            temporary[name].replace(targets[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()
    return targets


def design_dataset_preflight(args):
    """Validate every collector input without weights, Simulator, or output writes."""

    if getattr(args, "resume", None) is not None:
        raise ValueError("design collection accepts --checkpoint, not --resume")
    method = args.method
    MethodSpec(
        **{
            key: value
            for key, value in method.to_dict().items()
            if key != "method_id"
        }
    )
    if method.method_id != "td3_dinkelbach":
        raise ValueError(
            "collect-design-dataset currently supports only td3_dinkelbach"
        )

    from HRL_task_aware import ROUTING_STATE_DIM, TrainingConfig, formal_training_config

    manifest = ScenarioManifest.load(args.manifest)
    if manifest.split != args.split:
        raise ValueError(
            f"manifest split mismatch: manifest={manifest.split}, requested={args.split}"
        )
    episode_count = int(args.episodes)
    if episode_count <= 0 or manifest.episode_count < episode_count:
        raise ValueError("manifest has fewer entries than requested design episodes")
    validate_manifest_initial_topologies(
        manifest, episode_count=episode_count
    )
    expected_training_config = effective_training_config(
        formal_training_config(
            FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"],
            random_seed=args.training_seed,
        ),
        method,
    )
    _, calibration = load_com_capacity_reference()
    checkpoint = inspect_model_checkpoint(
        args.checkpoint,
        movement_state_dim=MOVEMENT_STATE_DIM,
        joint_action_dim=JOINT_ACTION_DIM,
        routing_state_dim=ROUTING_STATE_DIM,
        td3_gamma=1.0,
        ddqn_gamma=0.99,
        calibration=calibration,
        expected_experiment_metadata={
            "method_spec_fingerprint": method.compatible_fingerprints,
            "training_seed": int(args.training_seed),
        },
        expected_completed_episodes=FORMAL_EXPERIMENT_DEFAULTS[
            "training_episodes_per_seed"
        ],
        expected_formal_config=expected_training_config,
        require_episode_directory=True,
        movement_agent_kind=method.agent,
    )
    checkpoint_fingerprint = checkpoint_metadata_fingerprint(
        checkpoint["metadata"]
    )
    identity = design_run_identity(
        method, manifest, args.training_seed, checkpoint_fingerprint
    )
    run_dir = design_run_directory(
        args.output_dir, method, manifest, args.training_seed
    )
    validate_run_directory_preflight(run_dir, identity)
    reference_path = (
        Path(args.reference_per_episode).resolve()
        if args.reference_per_episode is not None
        else None
    )
    reference_rows = (
        read_reference_episode_csv(reference_path)
        if reference_path is not None
        else None
    )
    if reference_rows is not None:
        experiment = checkpoint["metadata"]["experiment"]
        reference_rows = validate_reference_identity(
            reference_rows,
            method_id=method.method_id,
            training_seed=args.training_seed,
            split=manifest.split,
            evaluation_manifest_hash=manifest.content_hash,
            training_manifest_hash=experiment["manifest_hash"],
            checkpoint_completed_episodes=checkpoint["completed_episode"],
            checkpoint_fingerprint=checkpoint_fingerprint,
            expected_scenario_ids=[
                str(entry["scenario_id"])
                for entry in manifest.episodes[:episode_count]
            ],
        )
    config = TrainingConfig(
        total_episodes=episode_count,
        mode="custom",
        episode_seconds=FORMAL_EXPERIMENT_DEFAULTS["episode_seconds"],
        routing_slot_seconds=FORMAL_EXPERIMENT_DEFAULTS["routing_slot_seconds"],
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=args.training_seed,
    )
    return {
        "method": method,
        "manifest": manifest,
        "training_seed": int(args.training_seed),
        "episode_count": episode_count,
        "config": config,
        "expected_training_config": expected_training_config,
        "checkpoint": checkpoint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "identity": identity,
        "run_directory": Path(run_dir).resolve(),
        "reference_path": reference_path,
        "reference_rows": reference_rows,
    }


def run_design_dataset_command(args):
    from HRL_task_aware import train

    preflight = design_dataset_preflight(args)
    run_dir = prepare_run_directory(
        preflight["run_directory"], preflight["identity"]
    )
    try:
        write_run_status(run_dir, "RUNNING")
        collector = DesignTransitionCollector()
        result = train(
            preflight["config"],
            scenario_manifest=preflight["manifest"],
            method_spec=preflight["method"],
            evaluation=True,
            checkpoint_dir=args.checkpoint,
            expected_checkpoint_episodes=FORMAL_EXPERIMENT_DEFAULTS[
                "training_episodes_per_seed"
            ],
            expected_checkpoint_formal_config=preflight[
                "expected_training_config"
            ],
            transition_observer=collector,
        )
        formal_config = preflight["checkpoint"]["metadata"]["experiment"][
            "formal_config"
        ]
        arrays = collector.arrays()
        validate_design_arrays(
            arrays,
            episode_count=preflight["episode_count"],
            episode_seconds=preflight["config"].episode_seconds,
            beta_search=formal_config["beta_search"],
            beta_vs=formal_config["beta_vs"],
            beta_com=formal_config["beta_com"],
            beta_relay=formal_config["beta_relay"],
        )
        if preflight["reference_rows"] is not None:
            validate_reference_metrics(
                preflight["reference_rows"], result["episode_metrics"]
            )
        metadata = _build_metadata(
            preflight,
            arrays,
            result,
            run_dir,
            preflight["reference_rows"],
        )
        paths = _atomic_write_dataset(
            run_dir, arrays, metadata, result["episode_metrics"]
        )
        write_run_status(run_dir, "COMPLETED")
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()},
                ensure_ascii=False,
            )
        )
    except BaseException as exc:
        try:
            write_run_status(run_dir, "FAILED", exception=exc)
        except BaseException:
            pass
        raise
    return 0
