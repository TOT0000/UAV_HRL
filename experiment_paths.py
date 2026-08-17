"""Canonical, collision-safe filesystem layout for comparison runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
import uuid


RUN_IDENTITY_FILENAME = "run_identity.json"
RUN_STATUS_FILENAME = "run_status.json"
RUN_STATUS_SCHEMA_VERSION = 1
RUN_STATES = frozenset(
    {"PREPARING", "RUNNING", "COMPLETED", "FAILED", "RESUMING"}
)
SHORT_HASH_LENGTH = 8


def filesystem_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    if not slug:
        raise ValueError("method_id cannot be converted to a filesystem-safe slug")
    return slug


def manifest_short_hash(content_hash: str) -> str:
    normalized = str(content_hash).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("manifest content hash must be a canonical SHA-256 hex digest")
    return normalized[:SHORT_HASH_LENGTH]


def training_run_identity(method, manifest, training_seed) -> dict:
    return {
        "run_kind": "train",
        "method_id": method.method_id,
        "method_slug": filesystem_slug(method.method_id),
        "training_seed": int(training_seed),
        "training_manifest_hash": manifest.content_hash,
        "manifest_split": manifest.split,
    }


def evaluation_run_identity(method, manifest, training_seed) -> dict:
    return {
        "run_kind": "evaluate",
        "method_id": method.method_id,
        "method_slug": filesystem_slug(method.method_id),
        "training_seed": int(training_seed),
        "evaluation_split": manifest.split,
        "evaluation_manifest_hash": manifest.content_hash,
    }


def training_run_directory(output_root, method, manifest, training_seed) -> Path:
    return (
        Path(output_root)
        / filesystem_slug(method.method_id)
        / "train"
        / manifest_short_hash(manifest.content_hash)
        / f"seed-{int(training_seed)}"
    )


def evaluation_run_directory(output_root, method, manifest, training_seed) -> Path:
    return (
        Path(output_root)
        / filesystem_slug(method.method_id)
        / "evaluate"
        / manifest.split
        / manifest_short_hash(manifest.content_hash)
        / f"seed-{int(training_seed)}"
    )


def resume_run_directory(checkpoint_dir) -> Path:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if (
        checkpoint_dir.parent.name != "full"
        or checkpoint_dir.parent.parent.name != "checkpoints"
    ):
        raise ValueError(
            "resume checkpoint must be inside "
            "<run-directory>/checkpoints/full/<checkpoint>"
        )
    return checkpoint_dir.parent.parent.parent


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex[:16]}"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_run_status(run_dir):
    path = Path(run_dir).resolve() / RUN_STATUS_FILENAME
    if not path.is_file():
        return None
    status = json.loads(path.read_text(encoding="utf-8"))
    transitions = status.get("transitions")
    if (
        status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION
        or status.get("state") not in RUN_STATES
        or not isinstance(transitions, list)
        or not transitions
        or any(item.get("state") not in RUN_STATES for item in transitions)
        or transitions[-1].get("state") != status.get("state")
    ):
        raise RuntimeError(f"run lifecycle marker is invalid: {path}")
    return status


def write_run_status(run_dir, state, *, exception=None):
    state = str(state)
    if state not in RUN_STATES:
        raise ValueError(f"unsupported run lifecycle state: {state}")
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    previous = read_run_status(run_dir)
    transitions = list(previous["transitions"]) if previous is not None else []
    transition = {
        "state": state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if exception is not None:
        transition["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception)[:1000],
        }
    transitions.append(transition)
    payload = {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "state": state,
        "updated_at": transition["timestamp"],
        "transitions": transitions,
    }
    if "exception" in transition:
        payload["exception"] = transition["exception"]
    _atomic_write_json(run_dir / RUN_STATUS_FILENAME, payload)
    return payload


def validate_run_directory_preflight(
    run_dir, identity, *, resume_checkpoint=None
) -> Path:
    """Validate run ownership/collision rules without creating or writing it."""

    run_dir = Path(run_dir).resolve()
    identity_path = run_dir / RUN_IDENTITY_FILENAME
    if resume_checkpoint is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"run directory already contains output; use explicit resume: {run_dir}"
            )
        return run_dir

    checkpoint_run_dir = resume_run_directory(resume_checkpoint)
    if checkpoint_run_dir != run_dir:
        raise RuntimeError(
            "resume checkpoint belongs to a different canonical run directory: "
            f"checkpoint={checkpoint_run_dir}, expected={run_dir}"
        )
    if not identity_path.is_file():
        raise RuntimeError(f"resume run identity is missing: {identity_path}")
    actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if actual_identity != identity:
        raise RuntimeError(
            "resume run identity is incompatible: "
            f"stored={actual_identity}, expected={identity}"
        )
    status = read_run_status(run_dir)
    if status is not None and status["state"] == "COMPLETED":
        raise RuntimeError("a completed training run cannot be resumed")
    return run_dir


def prepare_run_directory(run_dir, identity, *, resume_checkpoint=None) -> Path:
    run_dir = validate_run_directory_preflight(
        run_dir, identity, resume_checkpoint=resume_checkpoint
    )
    identity_path = run_dir / RUN_IDENTITY_FILENAME
    if resume_checkpoint is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(identity_path, identity)
        write_run_status(run_dir, "PREPARING")
        return run_dir
    write_run_status(run_dir, "RESUMING")
    return run_dir
