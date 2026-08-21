"""Fail-closed paper-figure builder backed only by audited legacy contracts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from experiment_config import FORMAL_CHECKPOINT_EPISODE, MethodSpec
from paper_figure_registry import (
    FIGURE_REGISTRY,
    LEGACY_SOURCE_COMMIT,
    LegacyFigureSourceUnavailable,
    require_legacy_figure_contract,
)


class PaperFigureSpecError(ValueError):
    pass


class AmbiguousPaperRunError(PaperFigureSpecError):
    pass


class IncompatiblePaperRunError(PaperFigureSpecError):
    pass


def causal_trailing_average(values, window=50):
    window = int(window)
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    result = []
    running = 0.0
    finite_values = []
    for index, value in enumerate(values):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("energy-efficiency series must be finite")
        finite_values.append(value)
        running += value
        if index >= window:
            running -= finite_values[index - window]
        result.append(running / min(index + 1, window))
    return result


def normalize_episode_ee(method_id, history_rows, window=50):
    ordered = sorted(history_rows, key=lambda row: int(row["episode"]))
    episodes = [int(row["episode"]) for row in ordered]
    if episodes != list(range(1, len(episodes) + 1)):
        raise PaperFigureSpecError(
            f"{method_id} training history must contain contiguous episodes from 1"
        )
    raw = []
    for row in ordered:
        timely_mbits = float(row["timely_goodput_mbits"])
        mobility_joules = float(row["mobility_energy_j"])
        # The production history is Mbit/J; the requested legacy-compatible
        # y-axis is bit/J, so conversion occurs once at this adapter boundary.
        value = (
            timely_mbits * 1e6 / mobility_joules
            if math.isfinite(timely_mbits)
            and math.isfinite(mobility_joules)
            and mobility_joules > 0.0
            else 0.0
        )
        raw.append(value if math.isfinite(value) else 0.0)
    averaged = causal_trailing_average(raw, window=window)
    return [
        {
            "method_id": str(method_id),
            "episode": episode,
            "timely_goodput_mbits": float(source["timely_goodput_mbits"]),
            "mobility_energy_j": float(source["mobility_energy_j"]),
            "raw_energy_efficiency_bit_per_j": raw_value,
            "trailing_50_energy_efficiency_bit_per_j": average,
        }
        for episode, source, raw_value, average in zip(
            episodes, ordered, raw, averaged
        )
    ]


def _read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PaperFigureSpecError(f"required paper artifact is missing: {path}") from exc
    return value


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise PaperFigureSpecError(f"training history is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise PaperFigureSpecError(f"training history is empty: {path}")
    return rows


def _metadata_fingerprint(metadata):
    canonical = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_run_entry(method_id, entry, spec_dir):
    if not isinstance(entry, dict):
        raise PaperFigureSpecError(f"run entry for {method_id} must be an object")
    if "run_dir" in entry:
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = spec_dir / run_dir
    else:
        candidates = [Path(value) for value in entry.get("candidates", [])]
        if len(candidates) != 1:
            raise AmbiguousPaperRunError(
                f"{method_id} must specify exactly one run_dir; candidates={candidates}"
            )
        run_dir = candidates[0]
        if not run_dir.is_absolute():
            run_dir = spec_dir / run_dir
    run_dir = run_dir.resolve()
    resolved = _read_json(run_dir / "resolved_config.json")
    if resolved.get("status") not in (None, "COMPLETED"):
        raise IncompatiblePaperRunError(
            f"paper figure requires a completed run for {method_id}"
        )
    if str(resolved.get("method")) != str(method_id):
        raise IncompatiblePaperRunError(
            f"run method mismatch: expected={method_id}, found={resolved.get('method')}"
        )
    method = MethodSpec.parse(method_id)
    if resolved.get("method_spec") != method.to_dict():
        raise IncompatiblePaperRunError(
            f"run method specification is stale for {method_id}"
        )
    checkpoint_episode = int(resolved.get("formal_checkpoint_episode", -1))
    if checkpoint_episode != FORMAL_CHECKPOINT_EPISODE:
        raise IncompatiblePaperRunError(
            f"{method_id} must resolve to formal ep_{FORMAL_CHECKPOINT_EPISODE}"
        )
    checkpoint_dir = (
        run_dir / "checkpoints" / "models" / f"ep_{checkpoint_episode:04d}"
    )
    checkpoint_metadata = _read_json(checkpoint_dir / "metadata.json")
    if int(checkpoint_metadata.get("episode", -2)) + 1 != checkpoint_episode:
        raise IncompatiblePaperRunError(
            f"formal checkpoint episode mismatch for {method_id}"
        )
    if not (checkpoint_dir / "models.pt").is_file():
        raise PaperFigureSpecError(
            f"formal checkpoint model payload is missing: {checkpoint_dir / 'models.pt'}"
        )
    history_path = Path(entry.get("training_history", run_dir / "training_history.jsonl"))
    if not history_path.is_absolute():
        history_path = spec_dir / history_path
    rows = _read_jsonl(history_path.resolve())
    if {str(row.get("method_id")) for row in rows} != {method_id}:
        raise IncompatiblePaperRunError(
            f"training history method mismatch for {method_id}"
        )
    return {
        "run_dir": run_dir,
        "resolved": resolved,
        "history_path": history_path.resolve(),
        "history_rows": rows,
        "checkpoint_path": checkpoint_dir.resolve(),
        "checkpoint_metadata_fingerprint": _metadata_fingerprint(
            checkpoint_metadata
        ),
    }


def _git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _unique_output_directory(output_root, git_sha):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{stamp}_{git_sha[:12]}"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt:02d}"
        candidate = output_root / f"{base}{suffix}"
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate output below {output_root}")


def _validate_fig2_compatibility(runs):
    identities = {
        (
            int(run["resolved"]["seed"]),
            str(run["resolved"]["training_manifest_hash"]),
            int(run["resolved"]["formal_checkpoint_episode"]),
        )
        for run in runs.values()
    }
    if len(identities) != 1:
        raise IncompatiblePaperRunError(
            "Fig.2 methods must share training seed, manifest hash, and checkpoint episode"
        )
    lengths = {len(run["history_rows"]) for run in runs.values()}
    if len(lengths) != 1:
        raise IncompatiblePaperRunError(
            "Fig.2 methods must have equal training-history lengths"
        )


def _write_csv(path, rows):
    path = Path(path)
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_fig2(spec, spec_path, output_dir, git_sha):
    contract = require_legacy_figure_contract("fig2")
    required_methods = tuple(contract["methods"])
    entries = spec.get("methods")
    if not isinstance(entries, dict):
        raise PaperFigureSpecError("paper spec must contain a methods object")
    if set(entries) != set(required_methods):
        raise PaperFigureSpecError(
            "Fig.2 spec must explicitly map exactly these methods: "
            f"{list(required_methods)}"
        )
    runs = {
        method_id: _resolve_run_entry(method_id, entries[method_id], spec_path.parent)
        for method_id in required_methods
    }
    _validate_fig2_compatibility(runs)
    normalized = {
        method_id: normalize_episode_ee(method_id, run["history_rows"])
        for method_id, run in runs.items()
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=tuple(contract["figure_size_inches"]))
    for index, method_id in enumerate(required_methods):
        rows = normalized[method_id]
        color = f"C{index}"
        axis.plot(
            [row["episode"] for row in rows],
            [row["raw_energy_efficiency_bit_per_j"] for row in rows],
            color=color,
            linewidth=contract["style"]["raw_line"]["linewidth"],
            alpha=contract["style"]["raw_line"]["alpha"],
            label="_nolegend_",
        )
        axis.plot(
            [row["episode"] for row in rows],
            [row["trailing_50_energy_efficiency_bit_per_j"] for row in rows],
            color=color,
            linewidth=contract["style"]["trailing_average_line"]["linewidth"],
            linestyle=contract["style"]["trailing_average_line"]["linestyle"],
            label=MethodSpec.parse(method_id).label,
        )
    axis.set_xlabel(contract["x_axis"]["label"])
    axis.set_ylabel(contract["y_axis"]["label"])
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    stem = contract["legacy_output_stem"]
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300)
    figure.savefig(pdf)
    plt.close(figure)

    flat_rows = [row for method_id in required_methods for row in normalized[method_id]]
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    _write_csv(csv_path, flat_rows)
    json_path.write_text(
        json.dumps(flat_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    resolved_spec = {
        "figure": "fig2",
        "contract": contract,
        "method_to_run_mapping": {
            method_id: str(run["run_dir"]) for method_id, run in runs.items()
        },
        "method_display_names": {
            method_id: MethodSpec.parse(method_id).label
            for method_id in required_methods
        },
        "source_histories": {
            method_id: str(run["history_path"]) for method_id, run in runs.items()
        },
        "checkpoint_paths": {
            method_id: str(run["checkpoint_path"])
            for method_id, run in runs.items()
        },
        "checkpoint_metadata_fingerprints": {
            method_id: run["checkpoint_metadata_fingerprint"]
            for method_id, run in runs.items()
        },
        "git_sha": git_sha,
        "legacy_source_commit": LEGACY_SOURCE_COMMIT,
        "metric_definition": (
            "episode timely delivered Mbit * 1e6 / max(episode mobility J, epsilon); "
            "no reward or cross-episode accumulation"
        ),
        "moving_average_definition": (
            "causal trailing mean over episodes max(1,e-49)..e"
        ),
    }
    (output_dir / "resolved_figure_spec.json").write_text(
        json.dumps(resolved_spec, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def build_paper_figures(spec_path, figure="all", output_root="results/paper_figures"):
    spec_path = Path(spec_path).resolve()
    spec = _read_json(spec_path)
    git_sha = _git_sha()
    output_dir = _unique_output_directory(output_root, git_sha)
    requested = str(figure).lower()
    if requested != "all":
        require_legacy_figure_contract(requested)
        figure_ids = [requested]
    else:
        figure_ids = [
            key for key, contract in FIGURE_REGISTRY.items() if contract["available"]
        ]
        unavailable = {
            key: contract["missing"]
            for key, contract in FIGURE_REGISTRY.items()
            if not contract["available"]
        }
        (output_dir / "unavailable_figures.json").write_text(
            json.dumps(unavailable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    outputs = {}
    for figure_id in figure_ids:
        if figure_id == "fig2":
            outputs[figure_id] = _build_fig2(
                spec, spec_path, output_dir, git_sha
            )
        else:
            raise LegacyFigureSourceUnavailable(
                f"no audited builder is available for {figure_id}"
            )
    metadata = {
        "output_directory": str(output_dir),
        "source_spec": str(spec_path),
        "git_sha": git_sha,
        "figures": outputs,
        "legacy_source_commit": LEGACY_SOURCE_COMMIT,
    }
    (output_dir / "paper_figure_build.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
