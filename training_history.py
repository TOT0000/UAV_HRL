"""Atomic, exact-resume-safe persistence for per-episode training metrics."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import uuid

from evaluation_metrics import safe_energy_efficiency


TRAINING_HISTORY_COLUMNS = (
    "method_id",
    "training_seed",
    "training_manifest_hash",
    "episode",
    "reward",
    "timely_goodput_mbits",
    "mobility_energy_j",
    "energy_efficiency_mbit_per_j",
    "dinkelbach_lambda",
)

FLOAT_COLUMNS = (
    "reward",
    "timely_goodput_mbits",
    "mobility_energy_j",
    "energy_efficiency_mbit_per_j",
    "dinkelbach_lambda",
)


def training_history_identity(method_id, training_seed, training_manifest_hash):
    return {
        "method_id": str(method_id),
        "training_seed": int(training_seed),
        "training_manifest_hash": str(training_manifest_hash),
    }


def build_training_history_row(
    identity,
    *,
    episode,
    reward,
    timely_goodput_mbits,
    mobility_energy_j,
    dinkelbach_lambda,
):
    timely_goodput_mbits = float(timely_goodput_mbits)
    mobility_energy_j = float(mobility_energy_j)
    row = {
        **identity,
        "episode": int(episode),
        "reward": float(reward),
        "timely_goodput_mbits": timely_goodput_mbits,
        "mobility_energy_j": mobility_energy_j,
        "energy_efficiency_mbit_per_j": safe_energy_efficiency(
            timely_goodput_mbits, mobility_energy_j
        ),
        "dinkelbach_lambda": float(dinkelbach_lambda),
    }
    return normalize_training_history_row(row)


def normalize_training_history_row(row):
    missing = set(TRAINING_HISTORY_COLUMNS).difference(row)
    if missing:
        raise ValueError(f"training history row is missing: {sorted(missing)}")
    normalized = {
        "method_id": str(row["method_id"]),
        "training_seed": int(row["training_seed"]),
        "training_manifest_hash": str(row["training_manifest_hash"]),
        "episode": int(row["episode"]),
    }
    normalized.update({column: float(row[column]) for column in FLOAT_COLUMNS})
    return normalized


def validate_training_history(rows, identity):
    normalized = [normalize_training_history_row(row) for row in rows]
    expected_episodes = list(range(1, len(normalized) + 1))
    actual_episodes = [row["episode"] for row in normalized]
    if actual_episodes != expected_episodes:
        raise ValueError(
            "training history episodes must be unique and consecutive from 1: "
            f"{actual_episodes}"
        )
    for row in normalized:
        for key, expected in identity.items():
            if row[key] != expected:
                raise RuntimeError(
                    f"training history identity mismatch for {key}: "
                    f"row={row[key]}, expected={expected}"
                )
        for column in FLOAT_COLUMNS:
            if not math.isfinite(row[column]):
                raise ValueError(f"training history {column} must be finite")
    return normalized


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_training_history(run_directory, rows, identity):
    run_directory = Path(run_directory)
    normalized = validate_training_history(rows, identity)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=TRAINING_HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(normalized)
    jsonl = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in normalized
    )
    _atomic_write_text(run_directory / "training_history.csv", csv_buffer.getvalue())
    _atomic_write_text(run_directory / "training_history.jsonl", jsonl)
    return normalized


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _same_rows(left, right):
    return left == right


def prepare_training_history(
    run_directory,
    identity,
    *,
    checkpoint_rows=None,
):
    """Initialize fresh history or reconcile disk state to an exact checkpoint."""

    run_directory = Path(run_directory)
    csv_path = run_directory / "training_history.csv"
    jsonl_path = run_directory / "training_history.jsonl"
    existing_sets = []
    if csv_path.is_file():
        existing_sets.append(validate_training_history(_read_csv(csv_path), identity))
    if jsonl_path.is_file():
        existing_sets.append(
            validate_training_history(_read_jsonl(jsonl_path), identity)
        )

    if checkpoint_rows is None:
        if existing_sets:
            raise FileExistsError(
                "fresh training run already has training history; use exact resume"
            )
        return []

    canonical = validate_training_history(checkpoint_rows, identity)
    for existing in existing_sets:
        prefix_length = min(len(existing), len(canonical))
        if not _same_rows(existing[:prefix_length], canonical[:prefix_length]):
            raise RuntimeError(
                "training history does not match the exact-resume checkpoint prefix"
            )
    write_training_history(run_directory, canonical, identity)
    return canonical
