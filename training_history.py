"""Atomic, exact-resume-safe persistence for per-episode training metrics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import uuid

from evaluation_metrics import safe_energy_efficiency


TRAINING_HISTORY_COLUMNS = (
    "method_id",
    "training_seed",
    "training_manifest_hash",
    "episode",
    "reward",
    "timely_goodput_mbits",
    "total_timely_useful_mbits",
    "mobility_energy_j",
    "energy_efficiency_mbit_per_j",
    "dinkelbach_lambda_used",
    "dinkelbach_lambda_after_episode",
    "dinkelbach_lambda_updated",
    "dinkelbach_update_status",
    "dinkelbach_block_index",
    "dinkelbach_block_episode",
    "dinkelbach_block_timely_mbits_so_far",
    "dinkelbach_block_energy_joules_so_far",
    "eligible_packet_count",
    "delay_violation_count",
    "delay_violation_probability",
    "lambda_cost_used",
    "lambda_cost_after_episode",
)

FLOAT_COLUMNS = (
    "reward",
    "timely_goodput_mbits",
    "total_timely_useful_mbits",
    "mobility_energy_j",
    "energy_efficiency_mbit_per_j",
    "dinkelbach_lambda_used",
    "dinkelbach_lambda_after_episode",
    "dinkelbach_block_timely_mbits_so_far",
    "dinkelbach_block_energy_joules_so_far",
    "delay_violation_probability",
    "lambda_cost_used",
    "lambda_cost_after_episode",
)

INTEGER_COLUMNS = (
    "dinkelbach_block_index",
    "dinkelbach_block_episode",
    "eligible_packet_count",
    "delay_violation_count",
)

BOOLEAN_COLUMNS = ("dinkelbach_lambda_updated",)

STRING_COLUMNS = ("dinkelbach_update_status",)

TRAINING_HISTORY_SCHEMA_VERSION = 6
TRAINING_HISTORY_CSV = "training_history.csv"
TRAINING_HISTORY_JSONL = "training_history.jsonl"
TRAINING_HISTORY_COMMIT = "training_history_commit.json"


class TrainingHistoryError(RuntimeError):
    pass


class TrainingHistoryIdentityError(TrainingHistoryError):
    pass


class TrainingHistoryConsistencyError(TrainingHistoryError):
    pass


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
    total_timely_useful_mbits=None,
    mobility_energy_j,
    dinkelbach_lambda_used,
    dinkelbach_lambda_after_episode,
    dinkelbach_lambda_updated,
    dinkelbach_update_status,
    dinkelbach_block_index,
    dinkelbach_block_episode,
    dinkelbach_block_timely_mbits_so_far,
    dinkelbach_block_energy_joules_so_far,
    dinkelbach_block_completed=None,
    eligible_packet_count=0,
    delay_violation_count=0,
    delay_violation_probability=None,
    lambda_cost_used=None,
    lambda_cost_after_episode=None,
):
    timely_goodput_mbits = float(timely_goodput_mbits)
    total_timely_useful_mbits = (
        timely_goodput_mbits
        if total_timely_useful_mbits is None
        else float(total_timely_useful_mbits)
    )
    mobility_energy_j = float(mobility_energy_j)
    row = {
        **identity,
        "episode": int(episode),
        "reward": float(reward),
        "timely_goodput_mbits": timely_goodput_mbits,
        "total_timely_useful_mbits": total_timely_useful_mbits,
        "mobility_energy_j": mobility_energy_j,
        "energy_efficiency_mbit_per_j": safe_energy_efficiency(
            timely_goodput_mbits, mobility_energy_j
        ),
        "dinkelbach_lambda_used": dinkelbach_lambda_used,
        "dinkelbach_lambda_after_episode": dinkelbach_lambda_after_episode,
        "dinkelbach_lambda_updated": _normalize_bool(
            dinkelbach_lambda_updated
        ),
        "dinkelbach_update_status": str(dinkelbach_update_status),
        "dinkelbach_block_index": dinkelbach_block_index,
        "dinkelbach_block_episode": dinkelbach_block_episode,
        "dinkelbach_block_timely_mbits_so_far": dinkelbach_block_timely_mbits_so_far,
        "dinkelbach_block_energy_joules_so_far": dinkelbach_block_energy_joules_so_far,
        "eligible_packet_count": int(eligible_packet_count),
        "delay_violation_count": int(delay_violation_count),
        "delay_violation_probability": delay_violation_probability,
        "lambda_cost_used": lambda_cost_used,
        "lambda_cost_after_episode": lambda_cost_after_episode,
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
    nullable = {
        "dinkelbach_lambda_used",
        "dinkelbach_lambda_after_episode",
        "dinkelbach_block_index",
        "dinkelbach_block_episode",
        "dinkelbach_block_timely_mbits_so_far",
        "dinkelbach_block_energy_joules_so_far",
        "delay_violation_probability",
        "lambda_cost_used",
        "lambda_cost_after_episode",
    }
    normalized.update(
        {
            column: (
                None
                if column in nullable and row[column] in (None, "")
                else float(row[column])
            )
            for column in FLOAT_COLUMNS
        }
    )
    normalized.update(
        {
            column: (
                None
                if column in nullable and row[column] in (None, "")
                else int(row[column])
            )
            for column in INTEGER_COLUMNS
        }
    )
    normalized.update(
        {column: _normalize_bool(row[column]) for column in BOOLEAN_COLUMNS}
    )
    normalized.update({column: str(row[column]) for column in STRING_COLUMNS})
    return normalized


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"training history boolean is invalid: {value!r}")


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
                raise TrainingHistoryIdentityError(
                    f"training history identity mismatch for {key}: "
                    f"row={row[key]}, expected={expected}"
                )
        for column in FLOAT_COLUMNS:
            if row[column] is not None and not math.isfinite(row[column]):
                raise ValueError(f"training history {column} must be finite")
        if row["dinkelbach_block_index"] is not None and row["dinkelbach_block_index"] <= 0:
            raise ValueError("training history Dinkelbach block index must be positive")
        if row["dinkelbach_block_episode"] is not None and row["dinkelbach_block_episode"] <= 0:
            raise ValueError("training history Dinkelbach block episode must be positive")
        if not row["dinkelbach_update_status"]:
            raise ValueError("training history Dinkelbach status must be non-empty")
        if not math.isclose(
            row["timely_goodput_mbits"],
            row["total_timely_useful_mbits"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "training history timely_goodput_mbits must alias total timely useful Mbit"
            )
    return normalized


def _history_bytes(normalized):
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=TRAINING_HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(normalized)
    jsonl = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in normalized
    )
    return csv_buffer.getvalue().encode("utf-8"), jsonl.encode("utf-8")


def _write_fsynced(path, payload):
    with Path(path).open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _commit_metadata(identity, normalized, csv_bytes, jsonl_bytes, transaction_id):
    return {
        "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
        **identity,
        "row_count": len(normalized),
        "last_episode": normalized[-1]["episode"] if normalized else 0,
        "csv_sha256": _sha256(csv_bytes),
        "jsonl_sha256": _sha256(jsonl_bytes),
        "transaction_id": str(transaction_id),
    }


def write_training_history(
    run_directory,
    rows,
    identity,
    *,
    transaction_id=None,
    _fault_inject=None,
):
    """Commit JSONL, its CSV projection, then commit metadata last."""

    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    normalized = validate_training_history(rows, identity)
    csv_bytes, jsonl_bytes = _history_bytes(normalized)
    transaction_id = str(transaction_id or uuid.uuid4().hex)
    if (
        not transaction_id
        or transaction_id != Path(transaction_id).name
        or "/" in transaction_id
        or "\\" in transaction_id
    ):
        raise ValueError("training history transaction id is not filesystem-safe")
    transaction_slug = hashlib.sha256(transaction_id.encode("utf-8")).hexdigest()[:16]
    transaction_directory = run_directory / f".th-{transaction_slug}"
    transaction_directory.mkdir()
    jsonl_temporary = transaction_directory / "j.tmp"
    csv_temporary = transaction_directory / "c.tmp"
    commit_temporary = transaction_directory / "m.tmp"
    commit = _commit_metadata(
        identity, normalized, csv_bytes, jsonl_bytes, transaction_id
    )
    commit_bytes = (
        json.dumps(commit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        _write_fsynced(jsonl_temporary, jsonl_bytes)
        _write_fsynced(csv_temporary, csv_bytes)
        _write_fsynced(commit_temporary, commit_bytes)
        jsonl_temporary.replace(run_directory / TRAINING_HISTORY_JSONL)
        if _fault_inject is not None:
            _fault_inject("after_jsonl_replace")
        csv_temporary.replace(run_directory / TRAINING_HISTORY_CSV)
        if _fault_inject is not None:
            _fault_inject("after_csv_replace")
        commit_temporary.replace(run_directory / TRAINING_HISTORY_COMMIT)
    finally:
        if transaction_directory.exists():
            try:
                shutil.rmtree(transaction_directory)
            except OSError:
                # Abandoned transaction directories are intentionally ignored.
                pass
    return normalized


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _same_rows(left, right):
    return left == right


def _history_paths(run_directory):
    run_directory = Path(run_directory)
    return {
        "csv": run_directory / TRAINING_HISTORY_CSV,
        "jsonl": run_directory / TRAINING_HISTORY_JSONL,
        "commit": run_directory / TRAINING_HISTORY_COMMIT,
    }


def _validate_commit_identity(commit, identity):
    if commit.get("schema_version") not in {2, TRAINING_HISTORY_SCHEMA_VERSION}:
        raise TrainingHistoryConsistencyError(
            "training history commit schema is incompatible"
        )
    mismatches = {
        key: (commit.get(key), expected)
        for key, expected in identity.items()
        if commit.get(key) != expected
    }
    if mismatches:
        raise TrainingHistoryIdentityError(
            f"training history commit identity mismatch: {mismatches}"
        )


def read_committed_training_history(run_directory, identity):
    """Read history only when JSONL, CSV, and commit metadata agree."""

    paths = _history_paths(run_directory)
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        return []
    if not all(present.values()):
        raise TrainingHistoryConsistencyError(
            f"training history transaction is incomplete: {present}"
        )
    try:
        commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise TrainingHistoryConsistencyError(
            "training history commit metadata is invalid"
        ) from exc
    _validate_commit_identity(commit, identity)

    csv_bytes = paths["csv"].read_bytes()
    jsonl_bytes = paths["jsonl"].read_bytes()
    if _sha256(csv_bytes) != commit.get("csv_sha256"):
        raise TrainingHistoryConsistencyError(
            "training history CSV hash does not match its commit"
        )
    if _sha256(jsonl_bytes) != commit.get("jsonl_sha256"):
        raise TrainingHistoryConsistencyError(
            "training history JSONL hash does not match its commit"
        )
    try:
        canonical = validate_training_history(_read_jsonl(paths["jsonl"]), identity)
        csv_rows = validate_training_history(_read_csv(paths["csv"]), identity)
    except TrainingHistoryIdentityError:
        raise
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise TrainingHistoryConsistencyError(
            "training history committed rows are invalid"
        ) from exc
    if not _same_rows(csv_rows, canonical):
        raise TrainingHistoryConsistencyError(
            "training history CSV is not the JSONL canonical projection"
        )
    expected_row_count = len(canonical)
    expected_last_episode = canonical[-1]["episode"] if canonical else 0
    if commit.get("row_count") != expected_row_count:
        raise TrainingHistoryConsistencyError(
            "training history row count does not match its commit"
        )
    if commit.get("last_episode") != expected_last_episode:
        raise TrainingHistoryConsistencyError(
            "training history last episode does not match its commit"
        )
    if not isinstance(commit.get("transaction_id"), str) or not commit[
        "transaction_id"
    ]:
        raise TrainingHistoryConsistencyError(
            "training history transaction id is invalid"
        )
    return canonical


def prepare_training_history(
    run_directory,
    identity,
    *,
    checkpoint_rows=None,
):
    """Initialize fresh history or reconcile disk state to an exact checkpoint."""

    run_directory = Path(run_directory)
    paths = _history_paths(run_directory)
    any_history_artifact = any(path.is_file() for path in paths.values())

    if checkpoint_rows is None:
        if any_history_artifact:
            read_committed_training_history(run_directory, identity)
            raise FileExistsError(
                "fresh training run already has training history; use exact resume"
            )
        return []

    canonical = validate_training_history(checkpoint_rows, identity)
    _validate_resume_history_on_disk(run_directory, identity, canonical)
    write_training_history(run_directory, canonical, identity)
    return canonical


def _validate_resume_history_on_disk(run_directory, identity, canonical):
    """Validate committed history/prefix without repairing or writing it."""

    paths = _history_paths(run_directory)
    any_history_artifact = any(path.is_file() for path in paths.values())
    existing_sets = []
    if any_history_artifact:
        try:
            existing_sets.append(
                read_committed_training_history(run_directory, identity)
            )
        except TrainingHistoryIdentityError:
            raise
        except TrainingHistoryConsistencyError:
            # An exact-resume checkpoint is the canonical repair source.
            existing_sets = []
    for existing in existing_sets:
        prefix_length = min(len(existing), len(canonical))
        if not _same_rows(existing[:prefix_length], canonical[:prefix_length]):
            raise RuntimeError(
                "training history does not match the exact-resume checkpoint prefix"
            )
    return canonical


def preflight_resume_training_history(
    run_directory, identity, *, checkpoint_rows
):
    """Check whether checkpoint rows can safely repair/continue disk history."""

    canonical = validate_training_history(checkpoint_rows, identity)
    _validate_resume_history_on_disk(run_directory, identity, canonical)
    return canonical
