"""Canonical, collision-safe filesystem layout for comparison runs."""

from __future__ import annotations

import json
from pathlib import Path
import re


RUN_IDENTITY_FILENAME = "run_identity.json"
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


def prepare_run_directory(run_dir, identity, *, resume_checkpoint=None) -> Path:
    run_dir = Path(run_dir).resolve()
    identity_path = run_dir / RUN_IDENTITY_FILENAME
    if resume_checkpoint is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"run directory already contains output; use explicit resume: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
    return run_dir
