"""Read-only planning and non-destructive reconciliation for exact resume."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid


_EPISODE_DIRECTORY = re.compile(r"ep_(\d+)")


@dataclass(frozen=True)
class StaleModelArtifact:
    source: Path
    episode: int
    metadata: dict


@dataclass(frozen=True)
class ResumeReconciliationPlan:
    run_directory: Path
    resume_checkpoint: Path
    resume_episode: int
    resume_checkpoint_metadata: dict
    resume_training_state: dict
    stale_models: tuple[StaleModelArtifact, ...]
    transaction_id: str
    timestamp: str


def _direct_episode_directories(root):
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    candidates = []
    for candidate in root.iterdir():
        match = _EPISODE_DIRECTORY.fullmatch(candidate.name)
        if match and candidate.is_dir() and candidate.resolve().parent == root:
            candidates.append((int(match.group(1)), candidate.resolve()))
    return sorted(candidates)


def plan_resume_reconciliation(
    run_directory,
    resume_checkpoint,
    *,
    inspect_full,
    inspect_model,
    transaction_id=None,
    timestamp=None,
):
    """Build a reconciliation plan without changing the filesystem."""

    run_directory = Path(run_directory).resolve()
    resume_checkpoint = Path(resume_checkpoint).resolve()
    full_root = (run_directory / "checkpoints" / "full").resolve()
    model_root = (run_directory / "checkpoints" / "models").resolve()
    if resume_checkpoint.parent != full_root:
        raise RuntimeError(
            "resume checkpoint is not in the canonical full-resume root: "
            f"{resume_checkpoint}"
        )
    selected = inspect_full(resume_checkpoint)
    resume_episode = int(selected["completed_episode"])

    newer_valid_full = []
    for episode, candidate in _direct_episode_directories(full_root):
        if episode <= resume_episode:
            continue
        try:
            inspected = inspect_full(candidate)
        except Exception:
            continue
        newer_valid_full.append((int(inspected["completed_episode"]), candidate))
    if newer_valid_full:
        episode, _ = max(newer_valid_full)
        raise RuntimeError(
            "A newer valid full-resume checkpoint exists: "
            f"ep_{episode:04d}. Resume from the latest valid checkpoint instead."
        )

    stale_models = []
    for episode, candidate in _direct_episode_directories(model_root):
        if episode <= resume_episode:
            continue
        try:
            inspected = inspect_model(candidate)
        except Exception:
            continue
        stale_models.append(
            StaleModelArtifact(
                source=candidate,
                episode=int(inspected["completed_episode"]),
                metadata=dict(inspected["metadata"]),
            )
        )

    return ResumeReconciliationPlan(
        run_directory=run_directory,
        resume_checkpoint=resume_checkpoint,
        resume_episode=resume_episode,
        resume_checkpoint_metadata=dict(selected["metadata"]),
        resume_training_state=dict(selected["training_state"]),
        stale_models=tuple(stale_models),
        transaction_id=transaction_id or uuid.uuid4().hex,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex[:16]}"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_resume_reconciliation(plan):
    """Move only planned canonical model artifacts into a recovery transaction."""

    if not plan.stale_models:
        return None
    recovery_directory = (
        plan.run_directory
        / "recovery"
        / f"resume-from-ep_{plan.resume_episode:04d}-{plan.transaction_id}"
    ).resolve()
    expected_recovery_parent = (plan.run_directory / "recovery").resolve()
    if recovery_directory.parent != expected_recovery_parent:
        raise RuntimeError("invalid recovery transaction path")
    models_directory = recovery_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for artifact in plan.stale_models:
        source = artifact.source.resolve()
        expected_source_parent = (
            plan.run_directory / "checkpoints" / "models"
        ).resolve()
        if source.parent != expected_source_parent:
            raise RuntimeError(f"refusing to move non-canonical artifact: {source}")
        quarantine = (models_directory / source.name).resolve()
        artifacts.append(
            {
                "episode": artifact.episode,
                "original_path": str(source),
                "quarantine_path": str(quarantine),
                "checkpoint_metadata": artifact.metadata,
                "moved": quarantine.exists() and not source.exists(),
            }
        )

    manifest_path = recovery_directory / "recovery_manifest.json"
    manifest = {
        "schema_version": 1,
        "transaction_id": plan.transaction_id,
        "reconciliation_timestamp": plan.timestamp,
        "status": "PREPARING",
        "resume_checkpoint": str(plan.resume_checkpoint),
        "resume_episode": plan.resume_episode,
        "artifacts": artifacts,
    }
    _atomic_write_json(manifest_path, manifest)

    for record in artifacts:
        source = Path(record["original_path"])
        quarantine = Path(record["quarantine_path"])
        if source.exists():
            if quarantine.exists():
                raise FileExistsError(
                    f"recovery quarantine already exists: {quarantine}"
                )
            source.replace(quarantine)
        elif not quarantine.exists():
            raise FileNotFoundError(
                f"planned checkpoint disappeared before reconciliation: {source}"
            )
        record["moved"] = True
        _atomic_write_json(manifest_path, manifest)

    manifest["status"] = "COMPLETED"
    _atomic_write_json(manifest_path, manifest)
    return {
        "recovery_directory": str(recovery_directory),
        "manifest_path": str(manifest_path),
        "transaction_id": plan.transaction_id,
        "resume_episode": plan.resume_episode,
        "moved_model_checkpoints": [
            record["original_path"] for record in artifacts
        ],
    }
