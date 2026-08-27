import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
import re
import shutil
import uuid
import zipfile

import numpy as np
import torch

from Channel_model import (
    CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
    CHANNEL_MODEL_VERSION,
    validate_channel_lifecycle_state,
)

from centralized_movement import (
    MOVEMENT_STATE_DIM,
    MOVEMENT_FEATURE_SCHEMA_VERSION,
    movement_mask_from_state,
)
from experiment_config import (
    COM_SESSION_LIFECYCLE_VERSION,
    MOVEMENT_ACTION_PROJECTION_CONTRACT_VERSION,
    MOVEMENT_REPLAY_CONTRACT_VERSION,
    MOVEMENT_WARMUP_CONTRACT_VERSION,
    NUM_UAV,
    SAFE_DDQN_ETA_C,
    SAFE_DDQN_INITIAL_LAMBDA_COST,
    SAFE_DDQN_QOS_TARGET_PROBABILITY,
)
from Packet_scheduler_v1 import (
    PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION,
)
from routing_transition_ledger import (
    validate_routing_transition_ledger_state,
)
from rng_contract import (
    CHANNEL_RNG_STREAMS,
    NamedRNGStreams,
    RNG_CONTRACT_VERSION,
    RNG_STREAM_IDS,
)
from fov_ema_lifecycle import validate_fov_ema_state
from scenario_manifest import (
    ScenarioManifest,
    manifest_prefix,
    resolve_training_manifest_segment,
    validate_training_manifest_segments,
)

from dinkelbach_blocks import (
    DINKELBACH_CONFIG_FIELDS,
    DinkelbachBlockState,
    dinkelbach_config_metadata,
)

CHECKPOINT_SCHEMA_VERSION = 15
ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION = 6
PRE_ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION = 5
PRE_ADAPTIVE_SAFE_DDQN_CHECKPOINT_SCHEMA_VERSION = 4
PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION = 3
LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION = 2
MODEL_CHECKPOINT_TYPE = "model-only"
FULL_CHECKPOINT_TYPE = "full-resume"
FULL_RESUME_LOGGING_SCHEMA_VERSION = 2

FULL_RESUME_LOGGING_STATE_FIELDS = (
    "full_resume_logging_schema_version",
    "reward_log",
    "delivered_log",
    "energy_log",
    "lambda_used_log",
    "lambda_after_episode_log",
)

LEGACY_JOINT_REPLAY_FIELDS = (
    "state",
    "action",
    "next_state",
    "not_done",
    "delivered_mbits",
    "total_mobility_energy",
    "phi_search_t",
    "phi_search_t1",
    "phi_vs_t",
    "phi_vs_t1",
    "phi_com_t",
    "phi_com_t1",
)
PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS = (
    *LEGACY_JOINT_REPLAY_FIELDS,
    "ratio_objective_reward",
)
JOINT_REPLAY_FIELDS = (
    *PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS,
    "current_movement_mask",
    "next_movement_mask",
    "movement_mask_valid",
)
ROUTING_REPLAY_FIELDS = (
    "state",
    "action",
    "next_state",
    "reward",
    "cost",
    "not_done",
    "tag_gt",
    "transition_id",
)

FORMAL_CORE_CONFIG_FIELDS = (
    "mode",
    "total_episodes",
    "episode_seconds",
    "routing_slot_seconds",
    "warmup_joint_transitions",
    "batch_size",
    "policy_delay",
    "beta_search",
    "beta_vs",
    "beta_com",
    "search_coverage_threshold",
    "replay_max_size",
    "routing_warmup_transitions",
    "routing_update_interval_slots",
    "routing_gradient_steps_per_update",
    "movement_exploration_decay_episodes",
    "routing_epsilon_decay_episodes",
    "dinkelbach_initial_lambda",
    "dinkelbach_update_interval_episodes",
    "dinkelbach_update_rule",
    "dinkelbach_numerator_unit",
    "dinkelbach_denominator_unit",
    "assignment_strategy",
    "assignment_rounds",
    "movement_policy",
    "movement_objective",
    "routing_policy",
    "task_observation_mode",
    "fov_com_pair_max_distance_m",
    "reserved_search_uav_ids",
    "service_assignment_only",
    "utility_normalization_mode",
    "task_compatibility_policy",
    "hover_assignment_candidate",
    "assignment_dummy_utility",
    "fov_assignment_utility_version",
    "fov_quality_transform",
    "fov_coverage_source",
    "safe_ddqn_qos_target_probability",
    "com_utility_contract_version",
    "reference_com_bandwidth_hz",
    "reference_s2u_max_capacity_mbps",
    "safe_ddqn_initial_lambda_cost",
    "safe_ddqn_eta_c",
    "safe_ddqn_lambda_update_scope",
    "safe_ddqn_evaluation_lambda_mode",
    "routing_mask_scope",
    "fov_ema_lifecycle_version",
    "sr_route_lifecycle_version",
    "packet_qos_contract_version",
    "com_session_lifecycle_version",
    "communication_range_contract_version",
    "a2g_communication_range_m",
    "a2a_communication_range_m",
    "packet_routing_causality_contract_version",
    "routing_cost_attribution_contract_version",
    "packet_service_contract_version",
    "qos_aggregate_contract_version",
    "evaluation_aggregation_schema_version",
    "routing_reward_contract_version",
    "routing_reward_alpha_capacity",
    "routing_reward_alpha_delay",
    "reference_u2u_max_capacity_mbps",
    "reference_u2g_max_capacity_mbps",
    "propulsion_model_id",
    "propulsion_parameters",
    "movement_channel_timing_version",
    "movement_replay_contract_version",
    "movement_substeps_per_interval",
    "movement_substep_seconds",
    "channel_model_version",
    "channel_environment_contract_version",
    "channel_fairness_contract_version",
    "channel_normalization_version",
    "channel_configuration",
    "large_scale_state_seconds",
    "fading_block_seconds",
    "fading_blocks_per_routing_slot",
    "rician_k_linear",
    "rician_k_db",
    "resolved_fov_deadline_seconds",
    "resolved_com_deadline_seconds",
    "packet_injection_cutoff_seconds",
)
ROUTING_EXPLORATION_CONFIG_FIELDS = frozenset(
    {
        "routing_warmup_transitions",
        "routing_update_interval_slots",
        "routing_gradient_steps_per_update",
        "movement_exploration_decay_episodes",
        "routing_epsilon_decay_episodes",
    }
)

FULL_RESUME_CONFIG_FIELDS = (
    *FORMAL_CORE_CONFIG_FIELDS,
    "model_checkpoint_every",
    "full_resume_every",
    "full_resume_keep_last",
    "formal_evaluation_episode",
    "random_seed",
)
HORIZON_EXTENSION_ADMINISTRATIVE_FIELDS = ("total_episodes",)
CHECKPOINT_HORIZON_COMPATIBILITY_FIELDS = (
    "checkpoint_episode",
    "checkpoint_planned_total_episodes",
    "current_training_run_total_episodes",
    "horizon_extension_compatible",
    "allowed_horizon_differences",
    "checkpoint_training_manifest_hash",
    "current_training_manifest_hash",
    "manifest_prefix_compatible",
)


def calibration_fingerprint(calibration):
    canonical = json.dumps(
        calibration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def checkpoint_metadata_fingerprint(metadata):
    return calibration_fingerprint(metadata)


CHECKPOINT_ARTIFACT_FINGERPRINT_SCHEMA = "uav-hrl-checkpoint-artifact-v1"
CHECKPOINT_HASH_CHUNK_BYTES = 1024 * 1024
CHECKPOINT_PROVENANCE_FIELDS = (
    "checkpoint_metadata_fingerprint",
    "checkpoint_models_sha256",
    "checkpoint_artifact_fingerprint",
)
EVALUATION_PROVENANCE_FIELDS = (
    "checkpoint_training_provenance",
    "evaluation_runtime_provenance",
    "checkpoint_training_episode_count",
    "evaluation_episode_count",
    "checkpoint_training_git_sha",
    "evaluation_git_sha",
)

ROUTING_LIFECYCLE_PROVENANCE_FIELDS = (
    "routing_global_slot_count",
    "routing_update_phase",
    "routing_slots_since_last_update",
    "routing_optimizer_update_count",
    "routing_target_update_count",
    "routing_epsilon_decay_start_slot",
    "routing_last_optimizer_update_slot",
    "routing_warmup_transitions",
    "routing_update_interval_slots",
    "routing_gradient_steps_per_update",
)


def _checkpoint_models_path(checkpoint_or_models_path):
    path = Path(checkpoint_or_models_path).resolve()
    return path if path.name == "models.pt" else path / "models.pt"


def checkpoint_models_sha256(checkpoint_or_models_path, *, chunk_bytes=None):
    """Hash a structurally valid torch-save ZIP without loading its weights."""

    models_path = _checkpoint_models_path(checkpoint_or_models_path)
    chunk_bytes = int(chunk_bytes or CHECKPOINT_HASH_CHUNK_BYTES)
    if chunk_bytes <= 0:
        raise ValueError("checkpoint hash chunk size must be positive")
    try:
        size = models_path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(
            f"checkpoint model payload is missing or unreadable: {models_path}"
        ) from exc
    if size <= 0:
        raise RuntimeError(f"checkpoint model payload is empty: {models_path}")
    try:
        with zipfile.ZipFile(models_path, "r") as archive:
            names = tuple(
                info.filename for info in archive.infolist() if not info.is_dir()
            )
            if not names:
                raise RuntimeError(
                    f"checkpoint model payload contains no records: {models_path}"
                )
            required_suffixes = ("data.pkl", "version")
            missing = [
                suffix
                for suffix in required_suffixes
                if not any(
                    name == suffix or name.endswith(f"/{suffix}") for name in names
                )
            ]
            if missing:
                raise RuntimeError(
                    "checkpoint model payload is not a canonical torch-save archive: "
                    f"path={models_path}, missing={missing}"
                )
            corrupt_record = archive.testzip()
            if corrupt_record is not None:
                raise RuntimeError(
                    "checkpoint model payload is truncated or corrupt: "
                    f"path={models_path}, record={corrupt_record}"
                )
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise RuntimeError(
            f"checkpoint model payload is truncated, corrupt, or unreadable: {models_path}"
        ) from exc
    digest = hashlib.sha256()
    try:
        with models_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(
            f"checkpoint model payload became unreadable while hashing: {models_path}"
        ) from exc
    return digest.hexdigest()


def checkpoint_artifact_fingerprint(metadata_fingerprint, models_sha256):
    """Bind canonical metadata and model bytes using a versioned JSON identity."""

    values = {
        "schema": CHECKPOINT_ARTIFACT_FINGERPRINT_SCHEMA,
        "checkpoint_metadata_fingerprint": str(metadata_fingerprint),
        "checkpoint_models_sha256": str(models_sha256),
    }
    for field in (
        "checkpoint_metadata_fingerprint",
        "checkpoint_models_sha256",
    ):
        value = values[field]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return calibration_fingerprint(values)


def checkpoint_artifact_provenance(checkpoint_dir, *, metadata=None):
    """Recompute paper provenance from the actual checkpoint artifacts."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    if metadata is None:
        metadata_path = checkpoint_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"checkpoint metadata is missing, unreadable, or invalid: {metadata_path}"
            ) from exc
    metadata_fingerprint = checkpoint_metadata_fingerprint(metadata)
    models_sha256 = checkpoint_models_sha256(checkpoint_dir)
    return {
        "checkpoint_metadata_fingerprint": metadata_fingerprint,
        "checkpoint_models_sha256": models_sha256,
        "checkpoint_artifact_fingerprint": checkpoint_artifact_fingerprint(
            metadata_fingerprint, models_sha256
        ),
    }


def checkpoint_episode_schedule(total_episodes, every):
    """Return periodic episodes plus a non-duplicate normal-completion save."""

    total_episodes = int(total_episodes)
    every = int(every)
    if total_episodes <= 0 or every <= 0:
        return []
    episodes = list(range(every, total_episodes + 1, every))
    if not episodes or episodes[-1] != total_episodes:
        episodes.append(total_episodes)
    return episodes


def _atomic_checkpoint_write(checkpoint_dir, writer):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    temporary = checkpoint_dir.parent / (
        f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        writer(temporary)
        temporary.replace(checkpoint_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return checkpoint_dir


def prune_full_resume_checkpoints(full_root, keep_last):
    full_root = Path(full_root).resolve()
    keep_last = int(keep_last)
    if keep_last <= 0:
        raise ValueError("full-resume retention must keep at least one checkpoint")
    if not full_root.is_dir():
        return []
    checkpoints = []
    for candidate in full_root.iterdir():
        match = re.fullmatch(r"ep_(\d+)", candidate.name)
        if match and candidate.is_dir() and candidate.resolve().parent == full_root:
            checkpoints.append((int(match.group(1)), candidate.resolve()))
    checkpoints.sort()
    removed = []
    for _, checkpoint in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def _agent_kind(agent):
    return str(getattr(agent, "agent_kind", "td3"))


def _routing_agent_kind(agent):
    return str(getattr(agent, "routing_agent_kind", "safe_ddqn"))


def _resolved_training_config_for_provenance(metadata):
    experiment = metadata.get("experiment") or {}
    resolved = deepcopy(experiment.get("formal_config") or {})
    resolved.update(
        {
            "method_key": experiment.get("method_id"),
            "method_id": experiment.get("method_id"),
            "method_spec": deepcopy(experiment.get("method_spec")),
            "method_spec_fingerprint": experiment.get(
                "method_spec_fingerprint"
            ),
            "training_episode_count": int(metadata["episode"]) + 1,
            "training_seed": experiment.get("training_seed"),
            "checkpoint_schema_version": int(
                metadata["checkpoint_schema_version"]
            ),
        }
    )
    return resolved


def _validate_routing_lifecycle_provenance(metadata, lifecycle):
    routing_kind = metadata.get("routing_agent_kind", "safe_ddqn")
    if routing_kind == "random":
        if lifecycle is not None:
            raise RuntimeError(
                "random-routing checkpoint must record routing_lifecycle=null"
            )
        return None
    if not isinstance(lifecycle, dict):
        raise RuntimeError(
            "learned-routing model checkpoint lacks training lifecycle provenance"
        )
    missing = sorted(
        set(ROUTING_LIFECYCLE_PROVENANCE_FIELDS).difference(lifecycle)
    )
    if missing:
        raise RuntimeError(
            "checkpoint training routing lifecycle provenance is incomplete: "
            f"{missing}"
        )
    try:
        global_slots = int(lifecycle["routing_global_slot_count"])
        interval = int(lifecycle["routing_update_interval_slots"])
        gradient_steps = int(lifecycle["routing_gradient_steps_per_update"])
        warmup = int(lifecycle["routing_warmup_transitions"])
        phase = int(lifecycle["routing_update_phase"])
        slots_since = int(lifecycle["routing_slots_since_last_update"])
        optimizer_count = int(lifecycle["routing_optimizer_update_count"])
        target_count = int(lifecycle["routing_target_update_count"])
        marker = lifecycle["routing_epsilon_decay_start_slot"]
        last_update = lifecycle["routing_last_optimizer_update_slot"]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint training routing lifecycle provenance is invalid"
        ) from exc
    if (
        global_slots < 0
        or interval <= 0
        or gradient_steps <= 0
        or warmup <= 0
        or phase != global_slots % interval
        or optimizer_count < 0
        or optimizer_count != target_count
    ):
        raise RuntimeError(
            "checkpoint training routing lifecycle counters are inconsistent"
        )
    if marker is not None and not 0 < int(marker) <= global_slots:
        raise RuntimeError(
            "checkpoint training routing epsilon marker is inconsistent"
        )
    if (optimizer_count == 0) != (last_update is None):
        raise RuntimeError(
            "checkpoint training routing last-update marker is inconsistent"
        )
    if last_update is not None:
        last_update = int(last_update)
        if not 0 < last_update <= global_slots or last_update % interval:
            raise RuntimeError(
                "checkpoint training routing last-update marker is inconsistent"
            )
    if slots_since != global_slots - int(last_update or 0):
        raise RuntimeError(
            "checkpoint training routing slots-since-update is inconsistent"
        )
    routing_config = metadata.get("routing_agent_configuration") or {}
    counter_mismatches = {
        "routing_optimizer_update_count": (
            routing_config.get("routing_optimizer_update_count"),
            optimizer_count,
        ),
        "routing_target_update_count": (
            routing_config.get("routing_target_update_count"),
            target_count,
        ),
    }
    counter_mismatches = {
        key: value
        for key, value in counter_mismatches.items()
        if value[0] is not None and int(value[0]) != value[1]
    }
    if counter_mismatches:
        raise RuntimeError(
            "checkpoint training routing lifecycle disagrees with agent counters: "
            f"{counter_mismatches}"
        )
    return deepcopy(lifecycle)


def _validate_complete_training_provenance(metadata, provenance):
    if not isinstance(provenance, dict):
        raise RuntimeError("checkpoint training_provenance is missing")
    required = {
        "training_episode_count",
        "training_git_sha",
        "resolved_training_config",
        "routing_lifecycle",
        "provenance_complete",
    }
    missing = sorted(required.difference(provenance))
    if missing:
        raise RuntimeError(f"checkpoint training_provenance is incomplete: {missing}")
    if provenance.get("provenance_complete") is not True:
        raise RuntimeError("checkpoint training provenance is not complete")
    completed_episodes = int(metadata["episode"]) + 1
    if int(provenance["training_episode_count"]) != completed_episodes:
        raise RuntimeError(
            "checkpoint training provenance episode count is inconsistent"
        )
    training_git_sha = provenance.get("training_git_sha")
    if not isinstance(training_git_sha, str) or not training_git_sha.strip():
        raise RuntimeError("checkpoint training provenance Git SHA is missing")
    experiment = metadata.get("experiment") or {}
    if experiment.get("git_sha") != training_git_sha:
        raise RuntimeError(
            "checkpoint training provenance Git SHA disagrees with experiment metadata"
        )
    latest_training_git_sha = experiment.get("latest_training_git_sha")
    if (
        latest_training_git_sha is not None
        and latest_training_git_sha != training_git_sha
    ):
        raise RuntimeError(
            "checkpoint latest training Git SHA disagrees with training provenance"
        )
    initial_training_git_sha = experiment.get("initial_training_git_sha")
    if initial_training_git_sha is not None and (
        not isinstance(initial_training_git_sha, str)
        or not initial_training_git_sha.strip()
    ):
        raise RuntimeError("checkpoint initial training Git SHA is invalid")
    history_identity_hash = experiment.get(
        "training_history_identity_manifest_hash"
    )
    legacy_history_hash = experiment.get("training_history_manifest_hash")
    if (
        history_identity_hash is not None
        and legacy_history_hash != history_identity_hash
    ):
        raise RuntimeError(
            "checkpoint legacy training history manifest hash is not its identity alias"
        )
    for field in (
        "training_manifest_segments",
        "training_history_identity_manifest_hash",
        "training_history_manifest_hash",
        "training_history_manifest_hash_semantics",
        "training_manifest_segments_semantics",
        "initial_training_git_sha",
        "latest_training_git_sha",
        "horizon_extension_provenance",
        "horizon_extension_history",
    ):
        if field in provenance and provenance[field] != experiment.get(field):
            raise RuntimeError(
                f"checkpoint training provenance disagrees with experiment metadata: {field}"
            )
    resolved = provenance.get("resolved_training_config")
    required_config = {
        "method_id",
        "method_spec",
        "assignment_strategy",
        "movement_policy",
        "task_observation_mode",
        "routing_policy",
        "total_episodes",
        "episode_seconds",
        "routing_slot_seconds",
        "random_seed",
        "movement_agent_configuration",
        "routing_agent_configuration",
        "exploration_schedule_configuration",
        "checkpoint_schema_version",
    }
    if not isinstance(resolved, dict):
        raise RuntimeError(
            "checkpoint resolved training configuration is missing"
        )
    missing_config = sorted(required_config.difference(resolved))
    if missing_config:
        raise RuntimeError(
            "checkpoint resolved training configuration is incomplete: "
            f"{missing_config}"
        )
    expected_resolved = _resolved_training_config_for_provenance(metadata)
    if resolved != expected_resolved:
        raise RuntimeError(
            "checkpoint resolved training configuration disagrees with metadata"
        )
    if resolved["routing_policy"] != metadata.get("routing_agent_kind"):
        raise RuntimeError(
            "checkpoint method routing policy disagrees with routing agent kind"
        )
    lifecycle = _validate_routing_lifecycle_provenance(
        metadata, provenance.get("routing_lifecycle")
    )
    return {
        **deepcopy(provenance),
        "resolved_training_config": deepcopy(resolved),
        "routing_lifecycle": lifecycle,
    }


def _build_training_provenance(metadata, routing_lifecycle_state):
    routing_kind = metadata.get("routing_agent_kind", "safe_ddqn")
    provenance = {
        "training_episode_count": int(metadata["episode"]) + 1,
        "training_git_sha": (metadata.get("experiment") or {}).get("git_sha"),
        "resolved_training_config": _resolved_training_config_for_provenance(
            metadata
        ),
        "routing_lifecycle": (
            None
            if routing_kind == "random"
            else deepcopy(routing_lifecycle_state)
        ),
        "safe_ddqn_constraint_state": (
            deepcopy(metadata.get("routing_agent_configuration"))
            if routing_kind == "safe_ddqn"
            else None
        ),
        "provenance_complete": True,
    }
    experiment = metadata.get("experiment") or {}
    for field in (
        "training_manifest_segments",
        "training_history_identity_manifest_hash",
        "training_history_manifest_hash",
        "training_history_manifest_hash_semantics",
        "training_manifest_segments_semantics",
        "initial_training_git_sha",
        "latest_training_git_sha",
        "horizon_extension_provenance",
        "horizon_extension_history",
    ):
        if field in experiment:
            provenance[field] = deepcopy(experiment[field])
    return _validate_complete_training_provenance(metadata, provenance)


def checkpoint_training_provenance(metadata, *, allow_incomplete=False):
    """Return immutable training provenance without consulting evaluator state."""

    schema = int(metadata.get("checkpoint_schema_version", -1))
    if schema >= CHECKPOINT_SCHEMA_VERSION:
        return _validate_complete_training_provenance(
            metadata, metadata.get("training_provenance")
        )
    routing_kind = metadata.get("routing_agent_kind", "safe_ddqn")
    if routing_kind != "random" and not allow_incomplete:
        raise RuntimeError(
            "schema-6 learned-routing model checkpoint lacks unambiguous "
            "training lifecycle provenance"
        )
    experiment = metadata.get("experiment") or {}
    inferred = {
        "training_episode_count": int(metadata["episode"]) + 1,
        "training_git_sha": experiment.get("git_sha"),
        "resolved_training_config": _resolved_training_config_for_provenance(
            metadata
        ),
        "routing_lifecycle": None,
        "safe_ddqn_constraint_state": (
            deepcopy(metadata.get("routing_agent_configuration"))
            if routing_kind == "safe_ddqn"
            else None
        ),
        "provenance_complete": routing_kind == "random",
    }
    return inferred


def _routing_agent_configuration(agent, resolved_configuration=None):
    kind = _routing_agent_kind(agent)
    resolved = dict(resolved_configuration or {})
    common = {
        **resolved,
        "routing_agent_kind": kind,
        "gamma": float(agent.gamma),
        "tau": float(agent.tau) if agent.tau is not None else None,
        "learning_rate": getattr(agent, "learning_rate", None),
        "target_update_scope": (
            "after_each_optimizer_event"
            if kind in {"safe_ddqn", "dqn"}
            else None
        ),
        "routing_optimizer_update_count": int(agent.num_training),
        "routing_target_update_count": int(
            getattr(agent, "target_update_count", 0)
        ),
        "reward_optimizer_update_count": int(
            getattr(agent, "reward_optimizer_update_count", 0)
        ),
        "reward_target_update_count": int(
            getattr(agent, "reward_target_update_count", 0)
        ),
    }
    if kind == "safe_ddqn":
        return {
            **common,
            "cost_optimizer_update_count": int(
                getattr(agent, "cost_optimizer_update_count", 0)
            ),
            "cost_target_update_count": int(
                getattr(agent, "cost_target_update_count", 0)
            ),
            **agent.constraint_state(),
        }
    return common


def _validate_safe_ddqn_constraint_metadata(metadata):
    if metadata.get("routing_agent_kind", "safe_ddqn") != "safe_ddqn":
        return
    routing = metadata.get("routing_agent_configuration") or {}
    expected = {
        "initial_lambda_cost": SAFE_DDQN_INITIAL_LAMBDA_COST,
        "eta_c": SAFE_DDQN_ETA_C,
        "qos_target_probability": SAFE_DDQN_QOS_TARGET_PROBABILITY,
        "lambda_update_scope": "episode_end",
        "cost_denominator": "eligible_packets",
        "mid_episode_checkpoint_supported": False,
    }
    mismatches = {
        field: (routing.get(field), value)
        for field, value in expected.items()
        if not _metadata_value_matches(routing.get(field), value)
    }
    if mismatches or "lambda_cost" not in routing:
        raise RuntimeError(
            "checkpoint safe-DDQN constraint metadata is incomplete or "
            f"incompatible: mismatches={mismatches}"
        )


def _movement_agent_configuration(agent, resolved_configuration=None):
    kind = _agent_kind(agent)
    actual = {
        "movement_agent_kind": kind,
        "movement_agent_gamma": float(agent.gamma),
        "tau": float(agent.tau) if hasattr(agent, "tau") else None,
        "policy_delay": getattr(agent, "policy_delay", None),
        "target_policy_noise": (
            float(agent.policy_noise)
            if kind == "td3"
            else getattr(agent, "target_policy_noise", None)
        ),
        "target_noise_clip": (
            float(agent.noise_clip)
            if kind == "td3"
            else getattr(agent, "target_noise_clip", None)
        ),
        "twin_critics": kind == "td3",
    }
    if resolved_configuration is None:
        return actual
    resolved = dict(resolved_configuration)
    mismatches = {
        field: (resolved.get(field), value)
        for field, value in actual.items()
        if not _metadata_value_matches(resolved.get(field), value)
    }
    if mismatches:
        raise RuntimeError(
            "resolved movement-agent metadata does not match the live agent: "
            f"{mismatches}"
        )
    return resolved


def _validate_effective_formal_movement_config(formal_config, kind, metadata=None):
    if not isinstance(formal_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    expected_delay = {"td3": 2, "ddpg": 1, "random": None}.get(kind)
    if kind not in {"td3", "ddpg", "random"}:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")
    if not _metadata_value_matches(formal_config.get("policy_delay"), expected_delay):
        raise RuntimeError(
            "checkpoint formal training config is incompatible with effective "
            f"{kind} policy delay: checkpoint={formal_config.get('policy_delay')}, "
            f"expected={expected_delay}"
        )
    nested = formal_config.get("movement_agent_configuration")
    if nested is not None and metadata is not None:
        actual = metadata.get("movement_agent_configuration")
        if nested != actual:
            raise RuntimeError(
                "checkpoint formal movement-agent configuration is incompatible "
                "with checkpoint metadata"
            )


def _movement_network_states(agent):
    kind = _agent_kind(agent)
    if kind == "td3":
        return {
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic_1": agent.critic_1.state_dict(),
            "critic_1_target": agent.critic_1_target.state_dict(),
            "critic_2": agent.critic_2.state_dict(),
            "critic_2_target": agent.critic_2_target.state_dict(),
        }
    if kind == "ddpg":
        return {
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
        }
    if kind == "random":
        return {}
    raise ValueError(f"unsupported movement agent kind: {kind}")


def _network_states(td3, ddqn):
    kind = _agent_kind(td3)
    routing_kind = _routing_agent_kind(ddqn)
    movement_key = "td3" if kind == "td3" else "movement_agent"
    states = {
        movement_key: {
            **({"kind": kind} if movement_key == "movement_agent" else {}),
            **_movement_network_states(td3),
        },
    }
    if routing_kind == "safe_ddqn":
        states["ddqn"] = {
            "q_network": ddqn.q_network.state_dict(),
            "target_q_network": ddqn.target_q_network.state_dict(),
            "cost_network": ddqn.cost_network.state_dict(),
            "target_cost_network": ddqn.target_cost_network.state_dict(),
            "constraint_state": ddqn.constraint_state(),
        }
    elif routing_kind == "dqn":
        states["routing_agent"] = {
            "kind": "dqn",
            "q_network": ddqn.q_network.state_dict(),
            "target_q_network": ddqn.target_q_network.state_dict(),
        }
    elif routing_kind == "random":
        states["routing_agent"] = {"kind": "random"}
    else:
        raise ValueError(f"unsupported routing agent kind: {routing_kind}")
    return states


def _load_network_states(payload, td3, ddqn):
    kind = _agent_kind(td3)
    if kind == "td3":
        td3_state = payload["td3"]
        td3.actor.load_state_dict(td3_state["actor"])
        td3.actor_target.load_state_dict(td3_state["actor_target"])
        td3.critic_1.load_state_dict(td3_state["critic_1"])
        td3.critic_1_target.load_state_dict(td3_state["critic_1_target"])
        td3.critic_2.load_state_dict(td3_state["critic_2"])
        td3.critic_2_target.load_state_dict(td3_state["critic_2_target"])
    elif kind == "ddpg":
        state = payload["movement_agent"]
        if state.get("kind") != "ddpg":
            raise RuntimeError("checkpoint movement agent kind is incompatible")
        td3.actor.load_state_dict(state["actor"])
        td3.actor_target.load_state_dict(state["actor_target"])
        td3.critic.load_state_dict(state["critic"])
        td3.critic_target.load_state_dict(state["critic_target"])
    elif kind == "random":
        state = payload.get("movement_agent") or {}
        if state.get("kind") != "random":
            raise RuntimeError("checkpoint movement agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")

    routing_kind = _routing_agent_kind(ddqn)
    if routing_kind == "safe_ddqn":
        ddqn_state = payload["ddqn"]
        ddqn.q_network.load_state_dict(ddqn_state["q_network"])
        ddqn.target_q_network.load_state_dict(ddqn_state["target_q_network"])
        ddqn.cost_network.load_state_dict(ddqn_state["cost_network"])
        ddqn.target_cost_network.load_state_dict(ddqn_state["target_cost_network"])
        if "constraint_state" not in ddqn_state:
            raise RuntimeError(
                "legacy safe-DDQN checkpoint lacks adaptive constraint state"
            )
        ddqn.load_constraint_state(ddqn_state["constraint_state"])
    elif routing_kind == "dqn":
        state = payload.get("routing_agent") or {}
        if state.get("kind") != "dqn" or "cost_network" in state:
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        ddqn.q_network.load_state_dict(state["q_network"])
        ddqn.target_q_network.load_state_dict(state["target_q_network"])
    elif routing_kind == "random":
        state = payload.get("routing_agent") or {}
        if state != {"kind": "random"}:
            raise RuntimeError("checkpoint routing agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported routing agent kind: {routing_kind}")


def _base_metadata(
    checkpoint_type,
    episode,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3,
    ddqn,
    calibration,
    experiment_metadata=None,
):
    kind = _agent_kind(td3)
    experiment = dict(experiment_metadata or {})
    formal_config = experiment.get("formal_config") or {}
    resolved_movement = formal_config.get("movement_agent_configuration")
    resolved_routing = formal_config.get("routing_agent_configuration")
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": checkpoint_type,
        "episode": int(episode),
        "movement_state_dim": int(movement_state_dim),
        "joint_action_dim": int(joint_action_dim),
        "routing_state_dim": int(routing_state_dim),
        "num_uav": NUM_UAV,
        "movement_feature_schema_version": MOVEMENT_FEATURE_SCHEMA_VERSION,
        "state_contract": "10-uav-no-hidden-num-gt-v1",
        "packet_lifecycle_contract": "sr-fifo-s2u-next-slot-routing-v1",
        "channel_contract": CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
        "channel_model_version": CHANNEL_MODEL_VERSION,
        "channel_configuration": deepcopy(
            formal_config.get("channel_configuration")
        ),
        "movement_agent_kind": kind,
        "movement_agent_gamma": float(td3.gamma),
        "movement_agent_configuration": _movement_agent_configuration(
            td3, resolved_movement
        ),
        "routing_ddqn_gamma": float(ddqn.gamma),
        "routing_agent_kind": _routing_agent_kind(ddqn),
        "routing_agent_configuration": _routing_agent_configuration(
            ddqn, resolved_routing
        ),
        "com_calibration_fingerprint": calibration_fingerprint(calibration),
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "master_seed": experiment.get("training_seed"),
        "rng_subsystem_mapping": dict(RNG_STREAM_IDS),
        "training_evaluation_rng_separation": (
            (experiment.get("rng_contract") or {}).get(
                "training_evaluation_separation"
            )
        ),
        "movement_action_projection_contract_version": (
            MOVEMENT_ACTION_PROJECTION_CONTRACT_VERSION
        ),
        "movement_heading_contract": "periodic-wrap-to-[-1,1)",
        "movement_replay_contract_version": MOVEMENT_REPLAY_CONTRACT_VERSION,
        "movement_warmup_contract_version": MOVEMENT_WARMUP_CONTRACT_VERSION,
        "capabilities": {
            "movement_learning": kind in {"td3", "ddpg"},
            "movement_replay": kind in {"td3", "ddpg"},
            "target_policy_smoothing": kind == "td3",
            "routing_learning": _routing_agent_kind(ddqn) in {"safe_ddqn", "dqn"},
            "routing_replay": _routing_agent_kind(ddqn) in {"safe_ddqn", "dqn"},
        },
    }
    reward_mode = (
        ((experiment.get("method_spec") or {}).get("reward_mode"))
        or experiment.get("reward_mode")
    )
    if reward_mode == "ratio":
        metadata["movement_objective"] = {
            "objective_unit": "bit_per_j",
            "numerator": "episode timely delivered bits",
            "denominator": "episode mobility energy joules",
            "semantics": "terminal-only ratio of sums",
        }
    if kind == "td3":
        metadata["centralized_td3_gamma"] = float(td3.gamma)
    if experiment_metadata is not None:
        metadata["experiment"] = experiment
    return metadata


def save_model_checkpoint(
    checkpoint_dir,
    *,
    episode,
    td3,
    ddqn,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    experiment_metadata=None,
    routing_lifecycle_state=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    metadata = _base_metadata(
        MODEL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
        experiment_metadata,
    )
    experiment = metadata.get("experiment") or {}
    if "formal_config" in experiment:
        _validate_effective_formal_movement_config(
            experiment["formal_config"], _agent_kind(td3), metadata
        )
    metadata["training_provenance"] = _build_training_provenance(
        metadata, routing_lifecycle_state
    )
    def write(temporary):
        torch.save(_network_states(td3, ddqn), temporary / "models.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    return _atomic_checkpoint_write(checkpoint_dir, write)


def _metadata_value_matches(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return bool(np.isclose(float(actual), float(expected)))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _checkpoint_directory_episode(checkpoint_dir, metadata):
    match = re.fullmatch(r"ep_(\d+)", Path(checkpoint_dir).name)
    if match is None:
        raise RuntimeError(
            f"checkpoint directory is not an episode checkpoint: {checkpoint_dir}"
        )
    try:
        metadata_episode = int(metadata["episode"]) + 1
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint episode metadata is invalid") from exc
    directory_episode = int(match.group(1))
    if directory_episode != metadata_episode:
        raise RuntimeError(
            "checkpoint directory episode is incompatible with metadata: "
            f"directory={directory_episode}, metadata={metadata_episode}"
        )
    return directory_episode


def _validate_formal_config(actual_config, expected_config, fields):
    if not isinstance(actual_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    mismatches = {
        key: (actual_config.get(key), expected_config.get(key))
        for key in fields
        if not _metadata_value_matches(
            actual_config.get(key), expected_config.get(key)
        )
    }
    if mismatches:
        raise RuntimeError(
            f"checkpoint formal training config is incompatible: {mismatches}"
        )


def _positive_horizon(config, label):
    if not isinstance(config, dict):
        raise RuntimeError(f"{label} formal training configuration is missing")
    raw_total = config.get("total_episodes")
    try:
        total = int(raw_total)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} total_episodes is invalid") from exc
    if isinstance(raw_total, bool) or total != raw_total or total <= 0:
        raise RuntimeError(f"{label} total_episodes must be a positive integer")
    return total


def validate_checkpoint_run_compatibility(
    checkpoint_formal_config,
    current_formal_config,
    *,
    checkpoint_episode,
    config_fields=FORMAL_CORE_CONFIG_FIELDS,
    checkpoint_training_manifest_hash=None,
    current_training_manifest=None,
):
    """Apply the canonical monotonic horizon-extension compatibility policy."""

    checkpoint_total = _positive_horizon(
        checkpoint_formal_config, "checkpoint planned"
    )
    current_total = _positive_horizon(current_formal_config, "current run")
    try:
        checkpoint_episode = int(checkpoint_episode)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint episode is invalid") from exc
    if checkpoint_episode <= 0:
        raise RuntimeError("checkpoint episode must be positive")
    if checkpoint_episode > checkpoint_total:
        raise RuntimeError(
            "checkpoint episode exceeds its planned training horizon: "
            f"checkpoint={checkpoint_episode}, planned={checkpoint_total}"
        )
    if checkpoint_episode > current_total:
        raise RuntimeError(
            "checkpoint episode exceeds the current run horizon: "
            f"checkpoint={checkpoint_episode}, current={current_total}"
        )
    if checkpoint_total > current_total:
        raise RuntimeError(
            "checkpoint planned horizon exceeds the current run horizon: "
            f"checkpoint={checkpoint_total}, current={current_total}"
        )

    strict_fields = tuple(
        field
        for field in config_fields
        if field not in HORIZON_EXTENSION_ADMINISTRATIVE_FIELDS
    )
    _validate_formal_config(
        checkpoint_formal_config,
        current_formal_config,
        strict_fields,
    )
    allowed_differences = [
        field
        for field in HORIZON_EXTENSION_ADMINISTRATIVE_FIELDS
        if not _metadata_value_matches(
            checkpoint_formal_config.get(field), current_formal_config.get(field)
        )
    ]
    horizon_extension = bool(allowed_differences)
    if horizon_extension and checkpoint_total >= current_total:
        raise RuntimeError("checkpoint horizon difference is not a monotonic extension")

    current_manifest_hash = None
    manifest_prefix_compatible = None
    if current_training_manifest is not None:
        if not isinstance(current_training_manifest, ScenarioManifest):
            raise TypeError("current training manifest must be a ScenarioManifest")
        if current_training_manifest.split != "train":
            raise RuntimeError("current run manifest is not a training manifest")
        if current_training_manifest.episode_count != current_total:
            raise RuntimeError(
                "current training manifest length disagrees with the run horizon: "
                f"manifest={current_training_manifest.episode_count}, "
                f"current={current_total}"
            )
        current_manifest_hash = current_training_manifest.content_hash
        if not checkpoint_training_manifest_hash:
            raise RuntimeError("checkpoint training manifest hash is missing")
        expected_manifest = (
            manifest_prefix(current_training_manifest, checkpoint_total)
            if horizon_extension
            else current_training_manifest
        )
        if checkpoint_training_manifest_hash != expected_manifest.content_hash:
            relationship = "prefix" if horizon_extension else "current manifest"
            raise RuntimeError(
                "checkpoint training manifest is incompatible with the "
                f"{relationship}: checkpoint={checkpoint_training_manifest_hash}, "
                f"expected={expected_manifest.content_hash}"
            )
        manifest_prefix_compatible = True
    elif horizon_extension:
        raise RuntimeError(
            "checkpoint horizon extension requires canonical training manifest "
            "prefix validation"
        )

    return {
        "checkpoint_episode": checkpoint_episode,
        "checkpoint_planned_total_episodes": checkpoint_total,
        "current_training_run_total_episodes": current_total,
        "horizon_extension_compatible": horizon_extension,
        "allowed_horizon_differences": allowed_differences,
        "checkpoint_training_manifest_hash": checkpoint_training_manifest_hash,
        "current_training_manifest_hash": current_manifest_hash,
        "manifest_prefix_compatible": manifest_prefix_compatible,
    }


def _validate_dinkelbach_checkpoint_metadata(metadata, expected_formal_config):
    experiment = metadata.get("experiment") or {}
    actual_formal_config = experiment.get("formal_config")
    if not isinstance(actual_formal_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    if not isinstance(experiment.get("dinkelbach_state"), dict):
        raise RuntimeError("checkpoint is missing Dinkelbach block state")
    config = expected_formal_config or actual_formal_config
    try:
        expected_config = dinkelbach_config_metadata(config)
        actual_config = dinkelbach_config_metadata(actual_formal_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint Dinkelbach configuration is incomplete"
        ) from exc
    if actual_config != expected_config:
        raise RuntimeError(
            "checkpoint Dinkelbach configuration is incompatible: "
            f"checkpoint={actual_config}, expected={expected_config}"
        )
    provenance_mismatches = {
        field: (experiment.get(field), actual_config[field])
        for field in DINKELBACH_CONFIG_FIELDS
        if experiment.get(field) != actual_config[field]
    }
    if provenance_mismatches:
        raise RuntimeError(
            "checkpoint Dinkelbach provenance is incompatible: "
            f"{provenance_mismatches}"
        )
    try:
        completed_episodes = int(metadata["episode"]) + 1
        state = DinkelbachBlockState.from_training_state(
            experiment.get("dinkelbach_state", {}),
            actual_formal_config,
            expected_completed_episodes=completed_episodes,
        )
        stored_lambda = float(experiment["lambda_ee"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint Dinkelbach metadata is incomplete"
        ) from exc
    if not np.isfinite(stored_lambda) or not np.isclose(
        stored_lambda, state.current_lambda
    ):
        raise RuntimeError(
            "checkpoint Dinkelbach lambda metadata is incompatible with block state"
        )
    return state


def _checkpoint_uses_dinkelbach(metadata):
    experiment = metadata.get("experiment") or {}
    method_spec = experiment.get("method_spec") or {}
    return method_spec.get("reward_mode", experiment.get("reward_mode", "dinkelbach")) == "dinkelbach"


def _validate_checkpoint_schema(metadata):
    schema = metadata.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "checkpoint_schema_version is incompatible with the boundary-aligned "
            "stochastic channel, COM QoS/routing-credit, all-participant FOV, "
            "named-RNG, projected-action and replay "
            "contract and must be retrained: "
            f"checkpoint={schema}, expected={CHECKPOINT_SCHEMA_VERSION}"
        )
    if schema in {
        CHECKPOINT_SCHEMA_VERSION,
        ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION,
        PRE_ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION,
        PRE_ADAPTIVE_SAFE_DDQN_CHECKPOINT_SCHEMA_VERSION,
        PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION,
    }:
        schema = int(schema)
        if (
            schema < PRE_ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
            and metadata.get("routing_agent_kind", "safe_ddqn") == "safe_ddqn"
        ):
            raise RuntimeError(
                "legacy safe-DDQN checkpoint lacks adaptive lambda_cost, eta_c, "
                "and canonical eligible-packet QoS target state"
            )
        required_contracts = {
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "channel_model_version": CHANNEL_MODEL_VERSION,
            "movement_action_projection_contract_version": (
                MOVEMENT_ACTION_PROJECTION_CONTRACT_VERSION
            ),
            "movement_replay_contract_version": MOVEMENT_REPLAY_CONTRACT_VERSION,
            "movement_warmup_contract_version": MOVEMENT_WARMUP_CONTRACT_VERSION,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in required_contracts.items()
            if key in metadata and metadata.get(key) != value
        }
        capabilities = metadata.get("capabilities")
        if mismatches or (
            capabilities is not None and not isinstance(capabilities, dict)
        ):
            raise RuntimeError(
                "checkpoint executable RNG/action/replay contract is incompatible: "
                f"{mismatches}"
            )
        return schema
    try:
        legacy_schema = (
            schema is None
            or int(schema) < PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        )
    except (TypeError, ValueError):
        legacy_schema = False
    if (
        not _checkpoint_uses_dinkelbach(metadata)
        and legacy_schema
    ):
        raise RuntimeError(
            "legacy ratio checkpoint uses per-transition B/E and cannot be "
            "resumed under the terminal ratio-of-sums reward contract"
        )
    if schema == LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION:
        return int(schema)
    raise RuntimeError(
        "checkpoint checkpoint_schema_version is incompatible: "
        f"checkpoint={schema}, expected={CHECKPOINT_SCHEMA_VERSION}"
    )


def _formal_config_for_validation(metadata, actual_config, expected_config):
    """Normalize documented legacy fields without weakening current checks."""

    if (
        metadata.get("checkpoint_schema_version")
        == LEGACY_DINKELBACH_CHECKPOINT_SCHEMA_VERSION
        and metadata.get("movement_agent_kind", "td3") == "ddpg"
        and isinstance(actual_config, dict)
        and isinstance(expected_config, dict)
    ):
        actual_config = dict(actual_config)
        actual_config["policy_delay"] = expected_config.get("policy_delay")
    experiment = metadata.get("experiment") or {}
    legacy_method_spec = experiment.get("method_spec") or {}
    if (
        isinstance(actual_config, dict)
        and isinstance(expected_config, dict)
        and "task_observation" not in legacy_method_spec
    ):
        actual_config = dict(actual_config)
        for field in FORMAL_CORE_CONFIG_FIELDS:
            if (
                field not in ROUTING_EXPLORATION_CONFIG_FIELDS
                and field not in actual_config
                and field in expected_config
            ):
                actual_config[field] = expected_config[field]
    return actual_config


def checkpoint_run_compatibility_from_metadata(
    metadata,
    current_formal_config,
    *,
    checkpoint_episode=None,
    config_fields=FORMAL_CORE_CONFIG_FIELDS,
    current_training_manifest=None,
):
    """Validate one checkpoint against its current run and return provenance."""

    experiment = metadata.get("experiment") or {}
    checkpoint_formal_config = _formal_config_for_validation(
        metadata,
        experiment.get("formal_config"),
        current_formal_config,
    )
    resolved_episode = (
        int(metadata["episode"]) + 1
        if checkpoint_episode is None
        else int(checkpoint_episode)
    )
    return validate_checkpoint_run_compatibility(
        checkpoint_formal_config,
        current_formal_config,
        checkpoint_episode=resolved_episode,
        config_fields=config_fields,
        checkpoint_training_manifest_hash=experiment.get("manifest_hash"),
        current_training_manifest=current_training_manifest,
    )


def _movement_gamma_key(metadata, schema):
    kind = metadata.get("movement_agent_kind", "td3")
    if kind == "td3" and "centralized_td3_gamma" in metadata:
        return "centralized_td3_gamma"
    return (
        "movement_agent_gamma"
        if schema >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        else "centralized_td3_gamma"
    )


def _validate_td3_gamma_alias(metadata):
    if metadata.get("movement_agent_kind", "td3") != "td3":
        return
    if "movement_agent_gamma" not in metadata or "centralized_td3_gamma" not in metadata:
        return
    if not _metadata_value_matches(
        metadata["movement_agent_gamma"], metadata["centralized_td3_gamma"]
    ):
        raise RuntimeError(
            "checkpoint centralized_td3_gamma is incompatible with "
            "movement_agent_gamma"
        )


def validate_model_checkpoint_metadata(
    metadata,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    td3_gamma=None,
    ddqn_gamma=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
    current_training_manifest=None,
    movement_agent_kind=None,
    allow_incomplete_provenance=False,
):
    """Validate evaluation provenance before any model payload is loaded."""

    schema = _validate_checkpoint_schema(metadata)
    checks = {"checkpoint_type": MODEL_CHECKPOINT_TYPE}
    optional_checks = {
        "movement_state_dim": movement_state_dim,
        "joint_action_dim": joint_action_dim,
        "routing_state_dim": routing_state_dim,
        "routing_ddqn_gamma": ddqn_gamma,
    }
    checks.update(
        {key: value for key, value in optional_checks.items() if value is not None}
    )
    if movement_agent_kind is not None:
        actual_kind = metadata.get("movement_agent_kind", "td3")
        if actual_kind != movement_agent_kind:
            raise RuntimeError(
                "checkpoint movement_agent_kind is incompatible: "
                f"checkpoint={actual_kind}, expected={movement_agent_kind}"
            )
    for key, expected in checks.items():
        actual = metadata.get(key)
        if not _metadata_value_matches(actual, expected):
            raise RuntimeError(
                f"checkpoint {key} is incompatible: "
                f"checkpoint={actual}, expected={expected}"
            )
    _validate_td3_gamma_alias(metadata)
    if td3_gamma is not None:
        gamma_key = _movement_gamma_key(metadata, schema)
        actual_gamma = metadata.get(gamma_key)
        if not _metadata_value_matches(actual_gamma, td3_gamma):
            raise RuntimeError(
                f"checkpoint {gamma_key} is incompatible: "
                f"checkpoint={actual_gamma}, expected={td3_gamma}"
            )

    if calibration is not None:
        expected_calibration = calibration_fingerprint(calibration)
        if metadata.get("com_calibration_fingerprint") != expected_calibration:
            raise RuntimeError("checkpoint COM calibration fingerprint is incompatible")

    if expected_experiment_metadata is not None:
        validate_checkpoint_experiment_metadata(
            metadata,
            expected_experiment_metadata,
            current_training_manifest=current_training_manifest,
        )

    if expected_completed_episodes is not None:
        try:
            completed_episodes = int(metadata["episode"]) + 1
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint episode metadata is invalid") from exc
        if completed_episodes != int(expected_completed_episodes):
            raise RuntimeError(
                "checkpoint completed training episodes is incompatible: "
                f"checkpoint={completed_episodes}, "
                f"expected={int(expected_completed_episodes)}"
            )

    if expected_formal_config is not None:
        checkpoint_run_compatibility_from_metadata(
            metadata,
            expected_formal_config,
            checkpoint_episode=(
                int(metadata["episode"]) + 1
                if expected_completed_episodes is None
                else int(expected_completed_episodes)
            ),
            config_fields=FORMAL_CORE_CONFIG_FIELDS,
            current_training_manifest=current_training_manifest,
        )
        if _checkpoint_uses_dinkelbach(metadata):
            _validate_dinkelbach_checkpoint_metadata(
                metadata, expected_formal_config
            )
    elif current_training_manifest is not None:
        raise ValueError(
            "current training manifest validation requires the current formal config"
        )
    experiment = metadata.get("experiment") or {}
    if (
        schema >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
        and "formal_config" in experiment
    ):
        _validate_effective_formal_movement_config(
            experiment["formal_config"], metadata.get("movement_agent_kind", "td3"), metadata
        )
    _validate_safe_ddqn_constraint_metadata(metadata)
    checkpoint_training_provenance(
        metadata, allow_incomplete=allow_incomplete_provenance
    )
    return metadata


def inspect_model_checkpoint(
    checkpoint_dir,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    td3_gamma=None,
    ddqn_gamma=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
    current_training_manifest=None,
    current_training_manifest_segments=None,
    training_run_directory=None,
    require_episode_directory=False,
    movement_agent_kind=None,
    allow_incomplete_provenance=False,
):
    """Validate model metadata and required files without loading weights."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    models_path = checkpoint_dir / "models.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_model_checkpoint_metadata(
        metadata,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3_gamma,
        ddqn_gamma=ddqn_gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_completed_episodes=expected_completed_episodes,
        expected_formal_config=expected_formal_config,
        current_training_manifest=current_training_manifest,
        movement_agent_kind=movement_agent_kind,
        allow_incomplete_provenance=allow_incomplete_provenance,
    )
    if not models_path.is_file():
        raise RuntimeError(f"checkpoint model payload is missing: {models_path}")
    completed_episode = (
        _checkpoint_directory_episode(checkpoint_dir, metadata)
        if require_episode_directory
        else int(metadata["episode"]) + 1
    )
    horizon_compatibility = (
        checkpoint_run_compatibility_from_metadata(
            metadata,
            expected_formal_config,
            checkpoint_episode=completed_episode,
            config_fields=FORMAL_CORE_CONFIG_FIELDS,
            current_training_manifest=current_training_manifest,
        )
        if expected_formal_config is not None
        else None
    )
    _validate_checkpoint_manifest_segments(
        checkpoint_dir,
        metadata,
        (metadata.get("experiment") or {}).get("formal_config"),
        completed_episode,
        current_training_manifest_segments=current_training_manifest_segments,
        current_formal_config=expected_formal_config,
        training_run_directory=training_run_directory,
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "completed_episode": completed_episode,
        "metadata": metadata,
        "horizon_compatibility": horizon_compatibility,
    }


def load_model_checkpoint(
    checkpoint_dir,
    td3,
    ddqn,
    *,
    movement_state_dim=None,
    joint_action_dim=None,
    routing_state_dim=None,
    calibration=None,
    expected_experiment_metadata=None,
    expected_completed_episodes=None,
    expected_formal_config=None,
    current_training_manifest=None,
    current_training_manifest_segments=None,
    training_run_directory=None,
    allow_incomplete_provenance=False,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    inspected = inspect_model_checkpoint(
        checkpoint_dir,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3.gamma,
        ddqn_gamma=ddqn.gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_completed_episodes=expected_completed_episodes,
        expected_formal_config=expected_formal_config,
        current_training_manifest=current_training_manifest,
        current_training_manifest_segments=current_training_manifest_segments,
        training_run_directory=training_run_directory,
        movement_agent_kind=_agent_kind(td3),
        allow_incomplete_provenance=allow_incomplete_provenance,
    )
    metadata = inspected["metadata"]
    payload = torch.load(
        checkpoint_dir / "models.pt", map_location="cpu", weights_only=False
    )
    _load_network_states(payload, td3, ddqn)
    return metadata


def _save_replay(path, replay, fields):
    size = int(replay.size)
    arrays = {field: np.asarray(getattr(replay, field)[:size]) for field in fields}
    np.savez_compressed(path, **arrays)
    return {
        "ptr": int(replay.ptr),
        "size": size,
        "max_size": int(replay.max_size),
        "n_step": int(replay.n_step),
        "gamma": float(getattr(replay, "gamma", 1.0)),
        "total_added": int(getattr(replay, "total_added", size)),
        "n_step_buffer": list(getattr(replay, "n_step_buffer", [])),
    }


def _validate_replay_payload(path, replay, fields, metadata):
    if not isinstance(metadata, dict):
        raise RuntimeError("checkpoint replay metadata is invalid")
    expected = {
        "max_size": int(replay.max_size),
        "n_step": int(replay.n_step),
    }
    for key, expected_value in expected.items():
        actual = int(metadata[key])
        if actual != expected_value:
            raise RuntimeError(
                f"replay {key} mismatch: checkpoint={actual}, current={expected_value}"
            )
    if hasattr(replay, "gamma"):
        checkpoint_gamma = float(metadata["gamma"])
        if not np.isclose(checkpoint_gamma, replay.gamma):
            raise RuntimeError(
                "replay gamma mismatch: "
                f"checkpoint={checkpoint_gamma}, current={replay.gamma}"
            )
    size = int(metadata["size"])
    ptr = int(metadata["ptr"])
    total_added = int(metadata.get("total_added", size))
    if not 0 <= size <= replay.max_size or not 0 <= ptr < replay.max_size:
        raise RuntimeError(f"invalid replay size/ptr in checkpoint: {size}/{ptr}")
    if total_added < size or ptr != total_added % replay.max_size:
        raise RuntimeError(
            "replay wrap diagnostics are inconsistent: "
            f"size={size}, ptr={ptr}, total_added={total_added}"
        )
    with np.load(path, allow_pickle=False) as arrays:
        for field in fields:
            if field not in arrays:
                raise RuntimeError(f"replay field is missing from checkpoint: {field}")
            saved = arrays[field]
            target = getattr(replay, field)
            if saved.shape != target[:size].shape:
                raise RuntimeError(
                    f"replay field {field} shape mismatch: "
                    f"checkpoint={saved.shape}, current={target[:size].shape}"
                )
    if not isinstance(metadata.get("n_step_buffer"), (list, tuple)):
        raise RuntimeError("checkpoint replay n-step buffer is invalid")


def _load_replay(path, replay, fields, metadata):
    _validate_replay_payload(path, replay, fields, metadata)
    size = int(metadata["size"])
    ptr = int(metadata["ptr"])
    with np.load(path, allow_pickle=False) as arrays:
        for field in fields:
            target = getattr(replay, field)
            target[:size] = arrays[field]
    replay.size = size
    replay.ptr = ptr
    replay.total_added = int(metadata.get("total_added", size))
    replay.n_step_buffer = list(metadata.get("n_step_buffer", []))


def _checkpoint_task_observation_mode(metadata):
    experiment = metadata.get("experiment") or {}
    method_spec = experiment.get("method_spec") or {}
    formal_config = experiment.get("formal_config") or {}
    return str(
        method_spec.get(
            "task_observation",
            formal_config.get("task_observation_mode", "full"),
        )
    )


def _validate_joint_replay_projection_masks(checkpoint_dir, metadata):
    """Validate mask availability before exact-resume state can be mutated."""

    if not bool((metadata.get("capabilities") or {}).get("movement_replay")):
        return
    schema = int(metadata["checkpoint_schema_version"])
    observation_mode = _checkpoint_task_observation_mode(metadata)
    if schema < ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION:
        if observation_mode == "masked":
            raise RuntimeError(
                "legacy masked-observation full-resume checkpoint lacks true "
                "movement projection masks and cannot be resumed safely"
            )
        return

    replay_path = Path(checkpoint_dir) / "joint_replay.npz"
    required = {
        "current_movement_mask",
        "next_movement_mask",
        "movement_mask_valid",
    }
    with np.load(replay_path, allow_pickle=False) as arrays:
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise RuntimeError(
                "full-resume checkpoint is missing movement projection mask "
                f"fields: {missing}"
            )
        if int(metadata.get("movement_state_dim", -1)) == MOVEMENT_STATE_DIM:
            validity = np.asarray(arrays["movement_mask_valid"], dtype=bool)
            if not validity.all():
                raise RuntimeError(
                    "full-resume checkpoint contains joint transitions without "
                    "authoritative movement projection masks"
                )


def _reconstruct_legacy_full_observation_masks(replay, metadata):
    """Migrate pre-mask full-observation replay without guessing assignments."""

    if _checkpoint_task_observation_mode(metadata) != "full":
        raise RuntimeError(
            "legacy masked-observation full-resume checkpoint lacks true "
            "movement projection masks and cannot be resumed safely"
        )
    if replay.state.shape[1] != MOVEMENT_STATE_DIM:
        raise RuntimeError(
            "legacy full-observation replay cannot reconstruct movement masks "
            f"from state dimension {replay.state.shape[1]}"
        )
    size = int(replay.size)
    replay.current_movement_mask[:size] = movement_mask_from_state(
        replay.state[:size]
    )
    replay.next_movement_mask[:size] = movement_mask_from_state(
        replay.next_state[:size]
    )
    replay.movement_mask_valid[:size] = True


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _movement_training_payload(agent):
    kind = _agent_kind(agent)
    counters = {
        "critic_updates": int(agent.num_critic_update_iteration),
        "actor_updates": int(agent.num_actor_update_iteration),
        "training_updates": int(agent.num_training),
    }
    if kind == "td3":
        return {
            "td3_optimizers": {
                "actor": agent.actor_optimizer.state_dict(),
                "critic_1": agent.critic_1_optimizer.state_dict(),
                "critic_2": agent.critic_2_optimizer.state_dict(),
            },
            "td3_counters": counters,
            "td3_hyperparameters": {
                "gamma": float(agent.gamma),
                "tau": float(agent.tau),
                "policy_delay": int(agent.policy_delay),
                "policy_noise": float(agent.policy_noise),
                "noise_clip": float(agent.noise_clip),
                "max_action": float(agent.max_action),
            },
        }
    if kind == "ddpg":
        return {
            "movement_agent_optimizers": {
                "actor": agent.actor_optimizer.state_dict(),
                "critic": agent.critic_optimizer.state_dict(),
            },
            "movement_agent_counters": counters,
            "movement_agent_hyperparameters": {
                "gamma": float(agent.gamma),
                "tau": float(agent.tau),
                "policy_delay": int(agent.policy_delay),
                "target_policy_noise": agent.target_policy_noise,
                "target_noise_clip": agent.target_noise_clip,
                "twin_critics": False,
                "max_action": float(agent.max_action),
            },
        }
    if kind == "random":
        return {
            "movement_agent_optimizers": {},
            "movement_agent_counters": counters,
            "movement_agent_hyperparameters": {"gamma": float(agent.gamma)},
        }
    raise ValueError(f"unsupported movement agent kind: {kind}")


def _restore_movement_training_payload(agent, payload):
    kind = _agent_kind(agent)
    if kind == "td3":
        agent.actor_optimizer.load_state_dict(payload["td3_optimizers"]["actor"])
        agent.critic_1_optimizer.load_state_dict(
            payload["td3_optimizers"]["critic_1"]
        )
        agent.critic_2_optimizer.load_state_dict(
            payload["td3_optimizers"]["critic_2"]
        )
        counters = payload["td3_counters"]
        hyperparameters = payload["td3_hyperparameters"]
    elif kind == "ddpg":
        optimizers = payload["movement_agent_optimizers"]
        agent.actor_optimizer.load_state_dict(optimizers["actor"])
        agent.critic_optimizer.load_state_dict(optimizers["critic"])
        counters = payload["movement_agent_counters"]
        hyperparameters = payload["movement_agent_hyperparameters"]
    elif kind == "random":
        counters = payload["movement_agent_counters"]
        hyperparameters = payload["movement_agent_hyperparameters"]
    else:
        raise RuntimeError(f"unsupported movement agent kind: {kind}")
    agent.num_critic_update_iteration = int(counters["critic_updates"])
    agent.num_actor_update_iteration = int(counters["actor_updates"])
    agent.num_training = int(counters["training_updates"])
    for name, value in hyperparameters.items():
        setattr(agent, name, value)


def _routing_training_payload(agent):
    kind = _routing_agent_kind(agent)
    if kind in {"safe_ddqn", "dqn"}:
        expected = int(agent.num_training)
        counters = {
            "target": int(agent.target_update_count),
            "reward_optimizer": int(agent.reward_optimizer_update_count),
            "reward_target": int(agent.reward_target_update_count),
        }
        if kind == "safe_ddqn":
            counters.update(
                cost_optimizer=int(agent.cost_optimizer_update_count),
                cost_target=int(agent.cost_target_update_count),
            )
        if any(value != expected for value in counters.values()):
            raise RuntimeError(
                "routing optimizer/target counters are inconsistent: "
                f"training={expected}, counters={counters}"
            )
    state = {
        "kind": kind,
        "gamma": float(agent.gamma),
        "tau": float(agent.tau) if agent.tau is not None else None,
        "training_updates": int(agent.num_training),
        "target_update_count": int(getattr(agent, "target_update_count", 0)),
        "loss_log": list(agent.loss_log),
        "reward_optimizer_update_count": int(
            getattr(agent, "reward_optimizer_update_count", 0)
        ),
        "reward_target_update_count": int(
            getattr(agent, "reward_target_update_count", 0)
        ),
    }
    if kind == "safe_ddqn":
        return {
            "ddqn_optimizers": {
                "reward": agent.optimizer.state_dict(),
                "cost": agent.cost_optimizer.state_dict(),
            },
            "ddqn_state": {
                **state,
                "cost_loss_log": list(agent.cost_loss_log),
                "constraint_state": agent.constraint_state(),
                "cost_optimizer_update_count": int(
                    agent.cost_optimizer_update_count
                ),
                "cost_target_update_count": int(agent.cost_target_update_count),
            },
        }
    if kind == "dqn":
        return {
            "routing_agent_optimizer": agent.optimizer.state_dict(),
            "routing_agent_state": state,
        }
    if kind == "random":
        return {"routing_agent_state": state}
    raise ValueError(f"unsupported routing agent kind: {kind}")


def _restore_routing_training_payload(agent, payload):
    kind = _routing_agent_kind(agent)
    if kind == "safe_ddqn":
        agent.optimizer.load_state_dict(payload["ddqn_optimizers"]["reward"])
        agent.cost_optimizer.load_state_dict(payload["ddqn_optimizers"]["cost"])
        state = payload["ddqn_state"]
        if state.get("kind", "safe_ddqn") != "safe_ddqn":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        if "constraint_state" not in state:
            raise RuntimeError(
                "legacy safe-DDQN resume checkpoint lacks constraint state"
            )
        agent.load_constraint_state(state["constraint_state"])
        agent.cost_loss_log = list(state["cost_loss_log"])
        agent.cost_optimizer_update_count = int(
            state["cost_optimizer_update_count"]
        )
        agent.cost_target_update_count = int(state["cost_target_update_count"])
    elif kind == "dqn":
        state = payload.get("routing_agent_state") or {}
        if state.get("kind") != "dqn":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
        agent.optimizer.load_state_dict(payload["routing_agent_optimizer"])
    elif kind == "random":
        state = payload.get("routing_agent_state") or {}
        if state.get("kind") != "random":
            raise RuntimeError("checkpoint routing agent kind is incompatible")
    else:
        raise RuntimeError(f"unsupported routing agent kind: {kind}")
    agent.gamma = float(state["gamma"])
    if state.get("tau") is not None:
        agent.tau = float(state["tau"])
    agent.num_training = int(state["training_updates"])
    agent.target_update_count = int(state.get("target_update_count", 0))
    agent.loss_log = list(state["loss_log"])
    agent.reward_optimizer_update_count = int(
        state["reward_optimizer_update_count"]
    )
    agent.reward_target_update_count = int(state["reward_target_update_count"])
    if kind in {"safe_ddqn", "dqn"}:
        expected = int(agent.num_training)
        counters = {
            "target": int(agent.target_update_count),
            "reward_optimizer": int(agent.reward_optimizer_update_count),
            "reward_target": int(agent.reward_target_update_count),
        }
        if kind == "safe_ddqn":
            counters.update(
                cost_optimizer=int(agent.cost_optimizer_update_count),
                cost_target=int(agent.cost_target_update_count),
            )
        if any(value != expected for value in counters.values()):
            raise RuntimeError(
                "checkpoint routing optimizer/target counters are inconsistent"
            )


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_rng_state_payload(state):
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint RNG state is invalid")
    missing = sorted({"python", "numpy", "torch_cpu", "torch_cuda"}.difference(state))
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete: {missing}")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("checkpoint RNG state is incompatible") from exc
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not (
            isinstance(cuda_state, (list, tuple))
            and all(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.ndim == 1
                for value in cuda_state
            )
        ):
            raise RuntimeError("checkpoint CUDA RNG state is invalid")
        if torch.cuda.is_available():
            if len(cuda_state) != torch.cuda.device_count():
                raise RuntimeError("checkpoint CUDA RNG device count is incompatible")
            try:
                for index, value in enumerate(cuda_state):
                    torch.Generator(device=f"cuda:{index}").set_state(value)
            except RuntimeError as exc:
                raise RuntimeError("checkpoint CUDA RNG state is incompatible") from exc


def _validate_named_rng_state_payload(state, expected_master_seed):
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint named subsystem RNG state is missing")
    try:
        master_seed = int(state["master_seed"])
        if expected_master_seed is not None and master_seed != int(
            expected_master_seed
        ):
            raise RuntimeError("checkpoint named RNG master seed is incompatible")
        NamedRNGStreams(master_seed).load_state_dict(state)
        missing_channel_streams = set(CHANNEL_RNG_STREAMS).difference(
            (state.get("numpy") or {}).keys()
        )
        if missing_channel_streams:
            raise RuntimeError(
                "checkpoint named channel RNG states are missing: "
                f"{sorted(missing_channel_streams)}"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("checkpoint named"):
            raise
        raise RuntimeError("checkpoint named subsystem RNG state is incompatible") from exc


def save_full_resume_checkpoint(
    checkpoint_dir,
    *,
    episode,
    td3,
    ddqn,
    joint_replay,
    routing_replay,
    training_state,
    formal_config,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    experiment_metadata=None,
    keep_last=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    if not isinstance(training_state, dict):
        raise TypeError("training_state must be an object")
    if not isinstance(training_state.get("named_rng_state"), dict):
        rng_streams = NamedRNGStreams(int(formal_config.get("random_seed") or 0))
        for stream_name in CHANNEL_RNG_STREAMS:
            rng_streams.numpy(stream_name)
        training_state["named_rng_state"] = rng_streams.state_dict()
    _validate_full_resume_logging_state(
        training_state,
        int(episode) + 1,
        _routing_agent_kind(ddqn),
        CHECKPOINT_SCHEMA_VERSION,
    )
    resolved_experiment_metadata = dict(experiment_metadata or {})
    resolved_experiment_metadata.setdefault("formal_config", dict(formal_config))
    if experiment_metadata is None:
        try:
            dinkelbach_state = DinkelbachBlockState.from_training_state(
                training_state,
                formal_config,
                expected_completed_episodes=int(episode) + 1,
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            pass
        else:
            resolved_experiment_metadata.update(
                dinkelbach_config_metadata(formal_config),
                lambda_ee=dinkelbach_state.current_lambda,
                dinkelbach_state=dinkelbach_state.training_state(),
            )
    metadata = _base_metadata(
        FULL_CHECKPOINT_TYPE,
        episode,
        movement_state_dim,
        joint_action_dim,
        routing_state_dim,
        td3,
        ddqn,
        calibration,
        resolved_experiment_metadata,
    )
    _validate_effective_formal_movement_config(
        formal_config, _agent_kind(td3), metadata
    )
    experiment = metadata.get("experiment") or {}
    if "formal_config" in experiment:
        _validate_effective_formal_movement_config(
            experiment["formal_config"], _agent_kind(td3), metadata
        )
    if all(
        experiment.get(field) is not None
        for field in ("method_id", "git_sha", "formal_config")
    ):
        metadata["training_provenance"] = _build_training_provenance(
            metadata, training_state.get("routing_lifecycle_state")
        )
    def write(temporary):
        replay_metadata = {
            "joint": None,
            "routing": None,
        }
        movement_kind = _agent_kind(td3)
        if movement_kind == "random":
            if joint_replay is not None:
                raise ValueError("random movement must not allocate a joint replay")
        else:
            if joint_replay is None:
                raise ValueError(f"{movement_kind} movement requires a joint replay")
            replay_metadata["joint"] = _save_replay(
                temporary / "joint_replay.npz", joint_replay, JOINT_REPLAY_FIELDS
            )
        routing_kind = _routing_agent_kind(ddqn)
        if routing_kind == "random":
            if routing_replay is not None:
                raise ValueError("random routing must not allocate a routing replay")
        else:
            if routing_replay is None:
                raise ValueError(f"{routing_kind} routing requires a replay")
            replay_metadata["routing"] = _save_replay(
                temporary / "routing_replay.npz",
                routing_replay,
                ROUTING_REPLAY_FIELDS,
            )
        payload = {
            "networks": _network_states(td3, ddqn),
            **_movement_training_payload(td3),
            **_routing_training_payload(ddqn),
            "training_state": dict(training_state),
            "formal_config": dict(formal_config),
            "replay_metadata": replay_metadata,
            "rng_state": _rng_state(),
        }
        torch.save(payload, temporary / "training_state.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    saved = _atomic_checkpoint_write(checkpoint_dir, write)
    if keep_last is not None:
        prune_full_resume_checkpoints(saved.parent, keep_last)
    return saved


def _validate_full_resume_logging_state(
    training_state, completed_episode, routing_agent_kind, checkpoint_schema_version
):
    if not isinstance(training_state, dict):
        raise RuntimeError("checkpoint training state is invalid")
    schema_version = training_state.get("full_resume_logging_schema_version")
    if schema_version != FULL_RESUME_LOGGING_SCHEMA_VERSION:
        raise RuntimeError(
            "checkpoint full-resume logging schema is incompatible: "
            f"checkpoint={schema_version}, "
            f"expected={FULL_RESUME_LOGGING_SCHEMA_VERSION}"
        )
    expected_length = int(completed_episode)
    dinkelbach_active = training_state.get("dinkelbach_active", True) is not False
    lambda_fields = {"lambda_used_log", "lambda_after_episode_log"}
    for field in FULL_RESUME_LOGGING_STATE_FIELDS[1:]:
        values = training_state.get(field)
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(
                f"checkpoint full-resume logging state is incomplete: {field}"
            )
        if len(values) != expected_length:
            raise RuntimeError(
                "checkpoint full-resume logging length is incompatible: "
                f"{field}={len(values)}, completed_episodes={expected_length}"
            )
        if not dinkelbach_active and field in lambda_fields:
            finite = all(value is None for value in values)
        else:
            try:
                finite = all(np.isfinite(float(value)) for value in values)
            except (TypeError, ValueError):
                finite = False
        if not finite:
            raise RuntimeError(
                f"checkpoint full-resume logging state is non-finite: {field}"
            )
    if routing_agent_kind == "safe_ddqn":
        for field in (
            "lambda_cost_used_log",
            "lambda_cost_after_episode_log",
        ):
            values = training_state.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != expected_length:
                raise RuntimeError(
                    "safe-DDQN checkpoint multiplier logging state is "
                    f"incompatible: {field}"
                )
            try:
                finite = all(np.isfinite(float(value)) for value in values)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise RuntimeError(
                    f"safe-DDQN checkpoint multiplier log is non-finite: {field}"
                )
    if (
        int(checkpoint_schema_version)
        >= ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
    ):
        fov_ema_state = training_state.get("fov_ema_state")
        validated_fov_state = validate_fov_ema_state(
            fov_ema_state, num_uav=NUM_UAV
        )
        if expected_length > 0 and not validated_fov_state[
            "initialized_uav_ids"
        ]:
            raise RuntimeError(
                "checkpoint FOV EMA lifecycle is uninitialized after completed episodes"
            )
    if (
        int(checkpoint_schema_version)
        >= ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
        and not isinstance(training_state.get("sr_route_state"), dict)
    ):
        raise RuntimeError("checkpoint lacks SR route lifecycle state")
    if int(checkpoint_schema_version) >= CHECKPOINT_SCHEMA_VERSION:
        validate_channel_lifecycle_state(
            training_state.get("channel_lifecycle_state"), num_uav=NUM_UAV
        )
        packet_state = training_state.get("packet_engine_state")
        if not isinstance(packet_state, dict):
            raise RuntimeError("checkpoint packet-engine state is missing")
        if (
            packet_state.get("schema_version")
            != PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION
            or packet_state.get("checkpoint_scope")
            != "episode_boundary_terminal_snapshot"
            or packet_state.get("mid_episode_checkpoint_supported") is not False
        ):
            raise RuntimeError("checkpoint packet-engine lifecycle is incompatible")
        if packet_state.get("active_packets"):
            raise RuntimeError("episode-boundary checkpoint contains active packets")
        uav_queues = packet_state.get("uav_queue_packet_ids")
        if (
            not isinstance(uav_queues, dict)
            or set(uav_queues) != {str(uid) for uid in range(NUM_UAV)}
            or any(value for value in uav_queues.values())
            or any(value for value in dict(packet_state.get("sr_queue_packet_ids") or {}).values())
        ):
            raise RuntimeError("episode-boundary checkpoint packet queues are not empty")
        packet_refs = packet_state.get("routing_transition_reference_counts")
        if not isinstance(packet_refs, dict) or packet_refs:
            raise RuntimeError("episode-boundary checkpoint has packet routing references")
        if packet_state.get("pending_terminal_violation_events"):
            raise RuntimeError("checkpoint has unprocessed terminal violation events")
        packet_qos_fields = (
            "system_qos_eligible_packet_count",
            "system_qos_violation_count",
            "routing_credit_eligible_packet_count",
            "routing_credit_violation_count",
            "unattributed_transition_violation_count",
            "unattributed_pre_routing_violation_count",
        )
        if any(
            type(packet_state.get(field)) is not int
            or int(packet_state[field]) < 0
            for field in packet_qos_fields
        ):
            raise RuntimeError("checkpoint packet QoS/credit counters are invalid")
        system_eligible = packet_state["system_qos_eligible_packet_count"]
        system_violations = packet_state["system_qos_violation_count"]
        routing_eligible = packet_state[
            "routing_credit_eligible_packet_count"
        ]
        routing_violations = packet_state["routing_credit_violation_count"]
        unattributed = packet_state[
            "unattributed_transition_violation_count"
        ]
        if (
            routing_eligible > system_eligible
            or system_violations > system_eligible
            or routing_violations > routing_eligible
            or system_violations != routing_violations + unattributed
            or packet_state["unattributed_pre_routing_violation_count"]
            > unattributed
        ):
            raise RuntimeError(
                "checkpoint packet system-QoS/routing-credit conservation failed"
            )
        replay_cost = packet_state.get(
            "replay_attributed_violation_cost_count"
        )
        if (
            not isinstance(replay_cost, (int, float))
            or not np.isfinite(float(replay_cost))
            or not np.isclose(float(replay_cost), routing_violations)
        ):
            raise RuntimeError(
                "checkpoint replay-attributed routing cost is inconsistent"
            )
        com_state = packet_state.get("com_session_state")
        if (
            not isinstance(com_state, dict)
            or com_state.get("lifecycle_version")
            != COM_SESSION_LIFECYCLE_VERSION
        ):
            raise RuntimeError("checkpoint COM-session lifecycle is incompatible")
        ledger_state = validate_routing_transition_ledger_state(
            training_state.get("routing_transition_state"),
            reference_counts=packet_refs,
        )
        if ledger_state["entries"]:
            raise RuntimeError("episode-boundary checkpoint routing ledger is not drained")
    if (
        int(checkpoint_schema_version)
        >= ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
    ):
        exploration_fields = (
            "exploration_schedule_version",
            "movement_exploration_decay_episodes",
            "routing_epsilon_decay_episodes",
            "resolved_movement_decay_steps",
            "resolved_routing_decay_slots",
            "movement_post_warmup_transition_count",
            "movement_noise_start",
            "movement_noise_end",
            "routing_epsilon_start",
            "routing_epsilon_end",
            "routing_epsilon_decay_start_slot",
        )
        missing = [field for field in exploration_fields if field not in training_state]
        if missing:
            raise RuntimeError(
                f"checkpoint exploration lifecycle state is incomplete: {missing}"
            )
        lifecycle = training_state.get("routing_lifecycle_state")
        if routing_agent_kind == "random":
            if lifecycle is not None:
                raise RuntimeError(
                    "random-routing checkpoint must not contain learner lifecycle state"
                )
        else:
            if not isinstance(lifecycle, dict):
                raise RuntimeError("checkpoint routing lifecycle state is missing")
            required_lifecycle = {
                "routing_optimizer_update_scope",
                "routing_update_interval_slots",
                "routing_gradient_steps_per_update",
                "routing_warmup_transitions",
                "routing_global_slot_count",
                "routing_update_phase",
                "routing_optimizer_update_count",
                "routing_target_update_count",
                "routing_slots_since_last_update",
                "routing_warmup_complete",
                "routing_epsilon_decay_start_slot",
                "routing_last_optimizer_update_slot",
            }
            lifecycle_missing = sorted(required_lifecycle.difference(lifecycle))
            if lifecycle_missing:
                raise RuntimeError(
                    "checkpoint routing lifecycle state is incomplete: "
                    f"{lifecycle_missing}"
                )
            interval = int(lifecycle["routing_update_interval_slots"])
            global_slots = int(lifecycle["routing_global_slot_count"])
            if (
                interval <= 0
                or int(lifecycle["routing_update_phase"])
                != global_slots % interval
                or int(lifecycle["routing_optimizer_update_count"])
                != int(lifecycle["routing_target_update_count"])
                or training_state["routing_epsilon_decay_start_slot"]
                != lifecycle["routing_epsilon_decay_start_slot"]
            ):
                raise RuntimeError("checkpoint routing lifecycle counters are inconsistent")
    if training_state.get("named_rng_state") is not None:
        _validate_named_rng_state_payload(
            training_state["named_rng_state"], None
        )


def _validate_full_metadata(
    metadata,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
    expected_experiment_metadata=None,
    current_training_manifest=None,
    movement_agent_kind=None,
):
    if metadata.get("checkpoint_type") != FULL_CHECKPOINT_TYPE:
        raise RuntimeError(
            "model-only checkpoint can only be used for evaluation, not exact resume"
        )
    schema = _validate_checkpoint_schema(metadata)
    if schema < ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "legacy full-resume checkpoint lacks unambiguous routing cadence and "
            "exploration schedule state"
        )
    checks = {
        "movement_state_dim": int(movement_state_dim),
        "joint_action_dim": int(joint_action_dim),
        "routing_state_dim": int(routing_state_dim),
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint {key} is incompatible: checkpoint={actual}, current={expected}"
            )
    if movement_agent_kind is not None:
        actual_kind = metadata.get("movement_agent_kind", "td3")
        if actual_kind != movement_agent_kind:
            raise RuntimeError(
                "checkpoint movement_agent_kind is incompatible: "
                f"checkpoint={actual_kind}, current={movement_agent_kind}"
            )
    _validate_td3_gamma_alias(metadata)
    movement_gamma_key = _movement_gamma_key(metadata, schema)
    gamma_checks = {
        movement_gamma_key: float(td3_gamma),
        "routing_ddqn_gamma": float(ddqn_gamma),
    }
    for key, expected in gamma_checks.items():
        actual = float(metadata.get(key, float("nan")))
        if not np.isclose(actual, expected):
            raise RuntimeError(
                f"checkpoint {key} is incompatible: checkpoint={actual}, current={expected}"
            )
    expected_fingerprint = calibration_fingerprint(calibration)
    if metadata.get("com_calibration_fingerprint") != expected_fingerprint:
        raise RuntimeError("checkpoint COM calibration fingerprint is incompatible")
    if expected_experiment_metadata is not None:
        validate_checkpoint_experiment_metadata(
            metadata,
            expected_experiment_metadata,
            current_training_manifest=current_training_manifest,
        )
    _validate_safe_ddqn_constraint_metadata(metadata)


def validate_checkpoint_experiment_metadata(
    metadata,
    expected,
    *,
    current_training_manifest=None,
):
    """Validate only requested experiment identity fields for compatibility."""

    actual = metadata.get("experiment")
    if actual is None:
        raise RuntimeError("checkpoint has no experiment identity metadata")
    mismatches = {}
    for key, value in expected.items():
        if key == "manifest_hash" and current_training_manifest is not None:
            continue
        actual_value = actual.get(key)
        if isinstance(value, list):
            matches = actual_value == value
        elif isinstance(value, (tuple, set, frozenset)):
            matches = actual_value in value
        else:
            matches = actual_value == value
        if not matches:
            mismatches[key] = (actual_value, value)
    if mismatches:
        raise RuntimeError(
            f"checkpoint experiment metadata is incompatible: {mismatches}"
        )


def _infer_training_run_directory(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if (
        checkpoint_dir.parent.name in {"full", "models"}
        and checkpoint_dir.parent.parent.name == "checkpoints"
    ):
        return checkpoint_dir.parent.parent.parent
    return None


def _validate_checkpoint_manifest_segments(
    checkpoint_dir,
    metadata,
    checkpoint_formal_config,
    completed_episode,
    *,
    current_training_manifest_segments=None,
    current_formal_config=None,
    training_run_directory=None,
):
    experiment = metadata.get("experiment") or {}
    checkpoint_segments = experiment.get("training_manifest_segments")
    run_directory = (
        Path(training_run_directory).resolve()
        if training_run_directory is not None
        else _infer_training_run_directory(checkpoint_dir)
    )
    if checkpoint_segments is not None:
        if run_directory is None:
            raise RuntimeError(
                "checkpoint manifest segment paths require the training run directory"
            )
        checkpoint_total = _positive_horizon(
            checkpoint_formal_config, "checkpoint planned"
        )
        canonical = validate_training_manifest_segments(
            run_directory,
            checkpoint_segments,
            current_total_episodes=checkpoint_total,
        )
        active_segment = resolve_training_manifest_segment(
            run_directory,
            canonical,
            completed_episode,
            current_total_episodes=checkpoint_total,
        )
        if active_segment["manifest_hash"] != experiment.get("manifest_hash"):
            raise RuntimeError(
                "checkpoint active manifest segment disagrees with experiment metadata"
            )
    if current_training_manifest_segments is not None:
        if run_directory is None:
            raise RuntimeError(
                "current manifest segment paths require the training run directory"
            )
        if current_formal_config is None:
            raise RuntimeError(
                "current manifest segments require the current formal config"
            )
        validate_training_manifest_segments(
            run_directory,
            current_training_manifest_segments,
            current_total_episodes=_positive_horizon(
                current_formal_config, "current run"
            ),
        )


def preflight_full_resume_checkpoint_metadata(
    checkpoint_dir,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
    expected_experiment_metadata=None,
    expected_formal_config=None,
    current_training_manifest=None,
    current_training_manifest_segments=None,
    training_run_directory=None,
    require_episode_directory=False,
    movement_agent_kind=None,
):
    """Metadata-only exact-resume preflight; never deserializes torch payloads."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        legacy_metadata_path = checkpoint_dir / "meta.json"
        if legacy_metadata_path.is_file():
            legacy = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
            legacy_dim = legacy.get("movement_state_dim", "unknown")
            raise RuntimeError(
                "legacy/model-only checkpoint "
                f"({legacy_dim}-D movement state) can only be used for "
                "evaluation and is incompatible with exact resume"
            )
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if current_training_manifest is not None and expected_formal_config is None:
        raise ValueError(
            "current training manifest validation requires the current formal config"
        )
    _validate_full_metadata(
        metadata,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3_gamma,
        ddqn_gamma=ddqn_gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        current_training_manifest=current_training_manifest,
        movement_agent_kind=movement_agent_kind,
    )
    completed_episode = (
        _checkpoint_directory_episode(checkpoint_dir, metadata)
        if require_episode_directory
        else int(metadata["episode"]) + 1
    )
    experiment = metadata.get("experiment") or {}
    checkpoint_formal_config = experiment.get("formal_config")
    if not isinstance(checkpoint_formal_config, dict):
        raise RuntimeError("checkpoint has no formal training configuration")
    if expected_formal_config is not None:
        horizon_compatibility = checkpoint_run_compatibility_from_metadata(
            metadata,
            expected_formal_config,
            checkpoint_episode=completed_episode,
            config_fields=FULL_RESUME_CONFIG_FIELDS,
            current_training_manifest=current_training_manifest,
        )
    else:
        checkpoint_total = _positive_horizon(
            checkpoint_formal_config, "checkpoint planned"
        )
        if completed_episode > checkpoint_total:
            raise RuntimeError(
                "checkpoint episode exceeds its planned training horizon: "
                f"checkpoint={completed_episode}, planned={checkpoint_total}"
            )
        horizon_compatibility = None
    if _checkpoint_uses_dinkelbach(metadata):
        _validate_dinkelbach_checkpoint_metadata(
            metadata, expected_formal_config or checkpoint_formal_config
        )
    if "training_provenance" in metadata:
        _validate_complete_training_provenance(
            metadata, metadata["training_provenance"]
        )
    if metadata["checkpoint_schema_version"] >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION:
        _validate_effective_formal_movement_config(
            checkpoint_formal_config,
            metadata.get("movement_agent_kind", "td3"),
            metadata,
        )
    _validate_checkpoint_manifest_segments(
        checkpoint_dir,
        metadata,
        checkpoint_formal_config,
        completed_episode,
        current_training_manifest_segments=current_training_manifest_segments,
        current_formal_config=expected_formal_config,
        training_run_directory=training_run_directory,
    )
    required_paths = [checkpoint_dir / "training_state.pt"]
    if bool((metadata.get("capabilities") or {}).get("movement_replay")):
        required_paths.append(checkpoint_dir / "joint_replay.npz")
    if metadata.get("routing_agent_kind", "safe_ddqn") != "random":
        required_paths.append(checkpoint_dir / "routing_replay.npz")
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"checkpoint has incomplete full-resume state; missing={missing}"
        )
    return {
        "checkpoint_dir": checkpoint_dir,
        "completed_episode": completed_episode,
        "metadata": metadata,
        "checkpoint_formal_config": checkpoint_formal_config,
        "horizon_compatibility": horizon_compatibility,
    }


def inspect_full_resume_checkpoint(
    checkpoint_dir,
    *,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    td3_gamma,
    ddqn_gamma,
    calibration,
    expected_experiment_metadata=None,
    expected_formal_config=None,
    current_training_manifest=None,
    current_training_manifest_segments=None,
    training_run_directory=None,
    require_episode_directory=False,
    movement_agent_kind=None,
):
    """Validate an exact-resume checkpoint without mutating training state."""

    preflight = preflight_full_resume_checkpoint_metadata(
        checkpoint_dir,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3_gamma,
        ddqn_gamma=ddqn_gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_formal_config=expected_formal_config,
        current_training_manifest=current_training_manifest,
        current_training_manifest_segments=current_training_manifest_segments,
        training_run_directory=training_run_directory,
        require_episode_directory=require_episode_directory,
        movement_agent_kind=movement_agent_kind,
    )
    checkpoint_dir = preflight["checkpoint_dir"]
    metadata = preflight["metadata"]
    completed_episode = preflight["completed_episode"]
    horizon_compatibility = preflight["horizon_compatibility"]

    # Phase two begins here. No torch payload is touched before preflight passes.
    _validate_joint_replay_projection_masks(checkpoint_dir, metadata)
    payload = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint full-resume payload is invalid")
    replay_metadata = payload.get("replay_metadata") or {}
    routing_kind = metadata.get("routing_agent_kind", "safe_ddqn")
    routing_replay_path = checkpoint_dir / "routing_replay.npz"
    if routing_kind == "random":
        if routing_replay_path.exists() or replay_metadata.get("routing") is not None:
            raise RuntimeError("random-routing checkpoint must not contain a routing replay")
    training_state = payload.get("training_state")
    formal_config = payload.get("formal_config")
    if not isinstance(training_state, dict):
        raise RuntimeError("checkpoint training state is invalid")
    if metadata.get("rng_contract_version") is not None:
        _validate_named_rng_state_payload(
            training_state.get("named_rng_state"), metadata.get("master_seed")
        )
    try:
        state_completed = int(training_state["completed_episode_index"]) + 1
        state_next = int(training_state["next_episode_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint training episode state is invalid") from exc
    if state_completed != completed_episode or state_next != completed_episode:
        raise RuntimeError(
            "checkpoint training episode state is incompatible with metadata: "
            f"completed={state_completed}, next={state_next}, "
            f"metadata={completed_episode}"
        )
    if not isinstance(formal_config, dict):
        raise RuntimeError("checkpoint formal training configuration is invalid")
    experiment = metadata.get("experiment") or {}
    experiment_formal_config = experiment["formal_config"]
    _validate_formal_config(
        formal_config,
        experiment_formal_config,
        FULL_RESUME_CONFIG_FIELDS,
    )
    if (
        metadata["checkpoint_schema_version"]
        >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
    ):
        kind = metadata.get("movement_agent_kind", "td3")
        experiment = metadata.get("experiment") or {}
        _validate_effective_formal_movement_config(formal_config, kind, metadata)
        if "formal_config" in experiment:
            _validate_effective_formal_movement_config(
                experiment["formal_config"], kind, metadata
            )
    _validate_full_resume_logging_state(
        training_state,
        completed_episode,
        routing_kind,
        metadata["checkpoint_schema_version"],
    )
    if routing_kind != "random":
        lifecycle = training_state["routing_lifecycle_state"]
        routing_agent_state = (
            payload.get("ddqn_state")
            if routing_kind == "safe_ddqn"
            else payload.get("routing_agent_state")
        ) or {}
        if (
            int(lifecycle["routing_optimizer_update_count"])
            != int(routing_agent_state.get("training_updates", -1))
            or int(lifecycle["routing_target_update_count"])
            != int(routing_agent_state.get("target_update_count", -1))
        ):
            raise RuntimeError(
                "checkpoint routing lifecycle counters disagree with agent state"
            )
        routing_replay_metadata = replay_metadata.get("routing")
        if not isinstance(routing_replay_metadata, dict):
            raise RuntimeError("checkpoint routing replay metadata is missing")
        replay_size = int(routing_replay_metadata.get("size", -1))
        warmup = int(lifecycle["routing_warmup_transitions"])
        warmup_complete = bool(lifecycle["routing_warmup_complete"])
        marker = lifecycle["routing_epsilon_decay_start_slot"]
        if (
            replay_size < 0
            or warmup_complete != (replay_size >= warmup)
            or warmup_complete != (marker is not None)
            or (
                marker is not None
                and not 0 < int(marker) <= int(lifecycle["routing_global_slot_count"])
            )
        ):
            raise RuntimeError(
                "checkpoint routing replay warm-up and epsilon marker are inconsistent"
            )
    if (
        int(metadata["checkpoint_schema_version"])
        >= CHECKPOINT_SCHEMA_VERSION
        and "training_provenance" in metadata
    ):
        provenance = _validate_complete_training_provenance(
            metadata, metadata["training_provenance"]
        )
        if provenance["routing_lifecycle"] != training_state.get(
            "routing_lifecycle_state"
        ):
            raise RuntimeError(
                "checkpoint metadata provenance disagrees with full-resume lifecycle"
            )
    if _checkpoint_uses_dinkelbach(metadata):
        dinkelbach_state = DinkelbachBlockState.from_training_state(
            training_state,
            formal_config,
            expected_completed_episodes=completed_episode,
        )
        if completed_episode > 0 and not np.isclose(
            float(training_state["lambda_after_episode_log"][-1]),
            dinkelbach_state.current_lambda,
        ):
            raise RuntimeError(
                "checkpoint lambda-after-episode log is incompatible with "
                "Dinkelbach block state"
            )
    return {
        "checkpoint_dir": checkpoint_dir,
        "completed_episode": completed_episode,
        "metadata": metadata,
        "training_state": training_state,
        "formal_config": formal_config,
        "payload": payload,
        "horizon_compatibility": horizon_compatibility,
    }


def load_full_resume_checkpoint(
    checkpoint_dir,
    *,
    td3,
    ddqn,
    joint_replay,
    routing_replay,
    movement_state_dim,
    joint_action_dim,
    routing_state_dim,
    calibration,
    expected_experiment_metadata=None,
    expected_formal_config=None,
    current_training_manifest=None,
    current_training_manifest_segments=None,
    training_run_directory=None,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    inspected = inspect_full_resume_checkpoint(
        checkpoint_dir,
        movement_state_dim=movement_state_dim,
        joint_action_dim=joint_action_dim,
        routing_state_dim=routing_state_dim,
        td3_gamma=td3.gamma,
        ddqn_gamma=ddqn.gamma,
        calibration=calibration,
        expected_experiment_metadata=expected_experiment_metadata,
        expected_formal_config=expected_formal_config,
        current_training_manifest=current_training_manifest,
        current_training_manifest_segments=current_training_manifest_segments,
        training_run_directory=training_run_directory,
        movement_agent_kind=_agent_kind(td3),
    )
    metadata = inspected["metadata"]
    payload = inspected["payload"]
    replay_metadata = payload["replay_metadata"]
    movement_replay_enabled = bool(
        (metadata.get("capabilities") or {}).get("movement_replay")
    )
    if movement_replay_enabled:
        if not isinstance(replay_metadata.get("joint"), dict):
            raise RuntimeError("checkpoint joint replay metadata is missing")
    elif replay_metadata.get("joint") is not None:
        raise RuntimeError("random-movement checkpoint contains a joint replay")
    if not isinstance(payload.get("networks"), dict):
        raise RuntimeError("checkpoint network payload is invalid")
    _validate_rng_state_payload(payload.get("rng_state"))
    joint_fields = (
        JOINT_REPLAY_FIELDS
        if metadata["checkpoint_schema_version"]
        >= ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
        else (
            PRE_MOVEMENT_MASK_JOINT_REPLAY_FIELDS
            if metadata["checkpoint_schema_version"]
            >= PRE_MOVEMENT_MASK_CHECKPOINT_SCHEMA_VERSION
            else LEGACY_JOINT_REPLAY_FIELDS
        )
    )
    if movement_replay_enabled:
        if joint_replay is None:
            raise RuntimeError("learned movement requires a joint replay")
        _validate_replay_payload(
            checkpoint_dir / "joint_replay.npz",
            joint_replay,
            joint_fields,
            replay_metadata["joint"],
        )
    elif joint_replay is not None:
        raise RuntimeError("random movement must not allocate a joint replay")
    routing_kind = metadata.get("routing_agent_kind", "safe_ddqn")
    if routing_kind == "random":
        if routing_replay is not None or replay_metadata.get("routing") is not None:
            raise RuntimeError("random-routing checkpoint contains a routing replay")
    else:
        if routing_replay is None:
            raise RuntimeError(f"{routing_kind} routing requires a replay")
        _validate_replay_payload(
            checkpoint_dir / "routing_replay.npz",
            routing_replay,
            ROUTING_REPLAY_FIELDS,
            replay_metadata["routing"],
        )

    # Validate networks, optimizers and constraint state on isolated agents. Live
    # agents remain untouched until every payload and replay contract has passed.
    td3_probe = deepcopy(td3)
    ddqn_probe = deepcopy(ddqn)
    _load_network_states(payload["networks"], td3_probe, ddqn_probe)
    _restore_movement_training_payload(td3_probe, payload)
    _restore_routing_training_payload(ddqn_probe, payload)

    _load_network_states(payload["networks"], td3, ddqn)
    _restore_movement_training_payload(td3, payload)
    _restore_routing_training_payload(ddqn, payload)
    if movement_replay_enabled:
        _load_replay(
            checkpoint_dir / "joint_replay.npz",
            joint_replay,
            joint_fields,
            replay_metadata["joint"],
        )
        if (
            metadata["checkpoint_schema_version"]
            < ROUTING_LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
        ):
            _reconstruct_legacy_full_observation_masks(joint_replay, metadata)
    if routing_kind != "random":
        _load_replay(
            checkpoint_dir / "routing_replay.npz",
            routing_replay,
            ROUTING_REPLAY_FIELDS,
            replay_metadata["routing"],
        )
    _restore_rng_state(payload["rng_state"])
    return {
        "metadata": metadata,
        "training_state": payload["training_state"],
        "formal_config": payload["formal_config"],
        "horizon_compatibility": inspected["horizon_compatibility"],
    }
