"""Build semantic paper figures exclusively from unified-runner artifacts."""

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
    METHOD_DISPLAY_NAMES,
    PAPER_METHOD_MAPPINGS,
    PLOT_STYLES,
    resolve_figure_id,
)


PAPER_EE_EPSILON_J = 1e-12


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


def paper_energy_efficiency(timely_goodput_mbits, mobility_energy_j):
    """Strict paper adapter: Mbit/J with an explicit positive epsilon."""

    numerator = float(timely_goodput_mbits)
    denominator = float(mobility_energy_j)
    if not math.isfinite(numerator) or numerator < 0.0:
        raise ValueError("timely goodput must be finite and non-negative")
    if not math.isfinite(denominator) or denominator < 0.0:
        raise ValueError("mobility energy must be finite and non-negative")
    value = numerator / max(denominator, PAPER_EE_EPSILON_J)
    if not math.isfinite(value):
        raise ValueError("paper energy efficiency is non-finite")
    return value


def normalize_episode_ee(method_id, history_rows, window=50):
    ordered = sorted(history_rows, key=lambda row: int(row["episode"]))
    episodes = [int(row["episode"]) for row in ordered]
    if not episodes or episodes != list(range(1, len(episodes) + 1)):
        raise PaperFigureSpecError(
            f"{method_id} training history must contain contiguous episodes from 1"
        )
    raw_mbit = []
    for row in ordered:
        try:
            timely_mbits = row["timely_goodput_mbits"]
            mobility_joules = row["mobility_energy_j"]
        except KeyError as exc:
            raise PaperFigureSpecError(
                f"{method_id} training history lacks production EE inputs"
            ) from exc
        raw_mbit.append(paper_energy_efficiency(timely_mbits, mobility_joules))
    raw_bits = [value * 1e6 for value in raw_mbit]
    averaged = causal_trailing_average(raw_bits, window=window)
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
            episodes, ordered, raw_bits, averaged
        )
    ]


def _read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PaperFigureSpecError(f"required paper artifact is missing: {path}") from exc


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


def _entry_directory(method_id, entry, spec_dir, key):
    if isinstance(entry, str):
        path = Path(entry)
    elif isinstance(entry, dict):
        if key in entry:
            path = Path(entry[key])
        elif "run_dir" in entry:
            path = Path(entry["run_dir"])
        else:
            candidates = [Path(value) for value in entry.get("candidates", [])]
            if len(candidates) != 1:
                raise AmbiguousPaperRunError(
                    f"{method_id} must specify exactly one {key}; candidates={candidates}"
                )
            path = candidates[0]
    else:
        raise PaperFigureSpecError(f"run entry for {method_id} must be a path or object")
    if not path.is_absolute():
        path = spec_dir / path
    return path.resolve()


def _resolve_training_run(method_id, entry, spec_dir):
    run_dir = _entry_directory(method_id, entry, spec_dir, "run_dir")
    resolved = _read_json(run_dir / "resolved_config.json")
    if resolved.get("status") not in (None, "COMPLETED"):
        raise IncompatiblePaperRunError(f"paper figure requires a completed run for {method_id}")
    if str(resolved.get("method")) != str(method_id):
        raise IncompatiblePaperRunError(
            f"run method mismatch: expected={method_id}, found={resolved.get('method')}"
        )
    method = MethodSpec.parse(method_id)
    if resolved.get("method_spec") != method.to_dict():
        raise IncompatiblePaperRunError(f"run method specification is stale for {method_id}")
    checkpoint_episode = int(resolved.get("formal_checkpoint_episode", -1))
    if checkpoint_episode != FORMAL_CHECKPOINT_EPISODE:
        raise IncompatiblePaperRunError(f"{method_id} must resolve to formal ep_2500")
    checkpoint_dir = run_dir / "checkpoints" / "models" / "ep_2500"
    checkpoint_metadata = _read_json(checkpoint_dir / "metadata.json")
    if int(checkpoint_metadata.get("episode", -2)) + 1 != FORMAL_CHECKPOINT_EPISODE:
        raise IncompatiblePaperRunError(f"formal checkpoint episode mismatch for {method_id}")
    if not (checkpoint_dir / "models.pt").is_file():
        raise PaperFigureSpecError(f"formal checkpoint model payload is missing: {checkpoint_dir}")
    history_path = run_dir / "training_history.jsonl"
    if isinstance(entry, dict) and entry.get("training_history"):
        history_path = Path(entry["training_history"])
        if not history_path.is_absolute():
            history_path = spec_dir / history_path
    rows = _read_jsonl(history_path.resolve())
    if {str(row.get("method_id")) for row in rows} != {method_id}:
        raise IncompatiblePaperRunError(f"training history method mismatch for {method_id}")
    return {
        "run_dir": run_dir,
        "resolved": resolved,
        "history_path": history_path.resolve(),
        "history_rows": rows,
        "checkpoint_path": checkpoint_dir.resolve(),
        "checkpoint_metadata_fingerprint": _metadata_fingerprint(checkpoint_metadata),
    }


def _resolve_evaluation_run(method_id, entry, spec_dir, expected_suite):
    evaluation_dir = _entry_directory(method_id, entry, spec_dir, "evaluation_dir")
    metadata = _read_json(evaluation_dir / "paper_evaluation_metadata.json")
    if metadata.get("semantic_suite") != expected_suite:
        raise IncompatiblePaperRunError(
            f"evaluation suite mismatch for {method_id}: {metadata.get('semantic_suite')}"
        )
    if metadata.get("method_id") != method_id:
        raise IncompatiblePaperRunError(
            f"evaluation method mismatch: expected={method_id}, found={metadata.get('method_id')}"
        )
    method = MethodSpec.parse(method_id)
    if metadata.get("method_spec") != method.to_dict():
        raise IncompatiblePaperRunError(f"evaluation method specification is stale for {method_id}")
    checkpoint_required = bool(method.learns_movement or method.learns_routing)
    if bool(metadata.get("checkpoint_required")) != checkpoint_required:
        raise IncompatiblePaperRunError(f"checkpoint requirement mismatch for {method_id}")
    if checkpoint_required:
        if int(metadata.get("checkpoint_episode", -1)) != FORMAL_CHECKPOINT_EPISODE:
            raise IncompatiblePaperRunError(f"{method_id} evaluation is not from ep_2500")
        if not metadata.get("checkpoint_path"):
            raise IncompatiblePaperRunError(f"{method_id} lacks checkpoint provenance")
    elif metadata.get("checkpoint_path") is not None:
        raise IncompatiblePaperRunError("pure-random evaluation must have no checkpoint path")
    aggregate_path = evaluation_dir / "aggregated_plot_data.json"
    aggregate_rows = _read_json(aggregate_path)
    if not isinstance(aggregate_rows, list) or not aggregate_rows:
        raise PaperFigureSpecError(f"evaluation aggregate is empty: {aggregate_path}")
    return {
        "evaluation_dir": evaluation_dir,
        "metadata": metadata,
        "aggregate_path": aggregate_path,
        "aggregate_rows": aggregate_rows,
    }


def _git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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


def _write_csv(path, rows):
    if not rows:
        raise PaperFigureSpecError(f"cannot write empty source CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready_contract(figure_id):
    return json.loads(json.dumps(dict(FIGURE_REGISTRY[figure_id]), ensure_ascii=False))


def _emit(figure_id, figure, output_dir, rows, resolved, json_value=None):
    contract = FIGURE_REGISTRY[figure_id]
    stem = contract["output_stem"]
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    spec_path = output_dir / f"{stem}_resolved_spec.json"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(json_value if json_value is not None else rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    spec_path.write_text(
        json.dumps(
            {
                "semantic_figure_id": figure_id,
                "contract": _json_ready_contract(figure_id),
                "resolved_style": (
                    PLOT_STYLES.get(figure_id)
                    if figure_id != "energy_efficiency_design_comparisons"
                    else {
                        component: PLOT_STYLES[component]
                        for component in FIGURE_REGISTRY[figure_id]["components"]
                    }
                ),
                **resolved,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "json": str(json_path),
        "resolved_spec": str(spec_path),
    }


def _training_entries(spec):
    entries = spec.get("training_runs", spec.get("methods"))
    if not isinstance(entries, dict):
        raise PaperFigureSpecError("paper spec must contain a training_runs object")
    return entries


def _evaluation_entries(spec, suite):
    groups = spec.get("evaluation_runs")
    if not isinstance(groups, dict) or not isinstance(groups.get(suite), dict):
        raise PaperFigureSpecError(f"paper spec must contain evaluation_runs.{suite}")
    return groups[suite]


def _validate_training_compatibility(runs):
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
            "training convergence methods must share seed, manifest hash, and checkpoint episode"
        )
    if len({len(run["history_rows"]) for run in runs.values()}) != 1:
        raise IncompatiblePaperRunError("training convergence histories must have equal lengths")


def _build_training(spec, spec_path, output_dir, git_sha):
    figure_id = "training_ee_vs_episode"
    methods = tuple(FIGURE_REGISTRY[figure_id]["methods"])
    entries = _training_entries(spec)
    if not set(methods).issubset(entries):
        raise PaperFigureSpecError(f"training_runs must map these methods: {list(methods)}")
    runs = {
        method: _resolve_training_run(method, entries[method], spec_path.parent)
        for method in methods
    }
    _validate_training_compatibility(runs)
    normalized = {
        method: normalize_episode_ee(method, run["history_rows"])
        for method, run in runs.items()
    }
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    flat_rows = []
    for method in methods:
        style = PLOT_STYLES[figure_id][method]
        rows = normalized[method]
        display = style["label"]
        axis.plot(
            [row["episode"] for row in rows],
            [row["raw_energy_efficiency_bit_per_j"] for row in rows],
            color=style["raw_color"], linewidth=1, alpha=style["raw_alpha"], label="_nolegend_",
        )
        axis.plot(
            [row["episode"] for row in rows],
            [row["trailing_50_energy_efficiency_bit_per_j"] for row in rows],
            color=style["color"], linewidth=3, label=display,
        )
        flat_rows.extend({**row, "display_name": display} for row in rows)
    axis.set_xlabel("Episodes")
    axis.set_ylabel("Energy efficiency (bit/J)")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc="upper left")
    figure.tight_layout()
    return _emit(
        figure_id,
        figure,
        output_dir,
        flat_rows,
        {
            "source_spec": str(spec_path),
            "method_to_training_run": {method: str(run["run_dir"]) for method, run in runs.items()},
            "checkpoint_paths": {method: str(run["checkpoint_path"]) for method, run in runs.items()},
            "checkpoint_metadata_fingerprints": {method: run["checkpoint_metadata_fingerprint"] for method, run in runs.items()},
            "metric_definition": "episode timely delivered Mbit * 1e6 / max(episode mobility J, 1e-12 J)",
            "trailing_average_definition": "causal mean over max(1,e-49)..e",
            "git_sha": git_sha,
        },
    )


def _resolve_suite_runs(spec, spec_path, suite, methods):
    entries = _evaluation_entries(spec, suite)
    missing = set(methods).difference(entries)
    if missing:
        raise PaperFigureSpecError(f"evaluation_runs.{suite} lacks methods: {sorted(missing)}")
    runs = {
        method: _resolve_evaluation_run(method, entries[method], spec_path.parent, suite)
        for method in methods
    }
    reference = None
    for method, run in runs.items():
        points = tuple(
            (point.get("point_id"), point.get("manifest_hash"), point.get("x_value"), point.get("x_unit"))
            for point in run["metadata"].get("points", [])
        )
        if reference is None:
            reference = points
        elif points != reference:
            raise IncompatiblePaperRunError(
                f"{suite} methods do not share identical sweep points and manifests; mismatch={method}"
            )
    expected = _expected_sweep_signature(suite)
    actual = tuple((point_id, x_value, x_unit) for point_id, _, x_value, x_unit in reference)
    if actual != expected:
        raise IncompatiblePaperRunError(
            f"{suite} sweep contract mismatch: expected={expected}, found={actual}"
        )
    return runs


def _expected_sweep_signature(suite):
    if suite == "uav_trajectory_snapshots":
        return (("trajectory", None, None),)
    if suite == "fixed_roi":
        return tuple((f"roi_{value}", value, "RoIs") for value in range(2, 9))
    if suite == "task_type_delay_vs_arrival_rate":
        return tuple(
            [
                (f"com_rate_{value:g}", value, "packets/s")
                for value in (50.0, 100.0, 150.0, 200.0)
            ]
            + [
                (f"fov_rate_{value:g}", value, "packets/s")
                for value in (10.0, 20.0, 30.0, 40.0)
            ]
        )
    if suite == "task_type_delay_violation_vs_target_delay":
        return tuple(
            (f"{task.lower()}_deadline_{value:g}s", value, "seconds")
            for task in ("COM", "FOV")
            for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        )
    raise PaperFigureSpecError(f"no sweep contract for semantic suite: {suite}")


def _metric_rows(runs, metric, task_type=None):
    result = []
    for method, run in runs.items():
        selected = [
            row
            for row in run["aggregate_rows"]
            if row.get("metric") == metric
            and (task_type is None or row.get("task_type") == task_type)
        ]
        if not selected:
            raise PaperFigureSpecError(f"{method} has no {metric}/{task_type} rows")
        for row in selected:
            if row.get("missing") and metric == "energy_efficiency_mbit_per_j":
                raise PaperFigureSpecError(f"{method} has missing EE at {row.get('point_id')}")
            result.append(dict(row))
    return result


def _plot_ee_panel(axis, figure_id, rows):
    method_contract = PAPER_METHOD_MAPPINGS[figure_id]
    if isinstance(method_contract, dict):
        ordered = tuple(method_contract.values())
        labels = {method: label for label, method in method_contract.items()}
    else:
        ordered = tuple(method_contract)
        labels = {
            method: PLOT_STYLES[figure_id][method]["label"] for method in ordered
        }
    plotted = []
    for method in ordered:
        values = sorted(
            (row for row in rows if row["method_id"] == method),
            key=lambda row: float(row["x_value"]),
        )
        style = PLOT_STYLES[figure_id][method]
        axis.plot(
            [row["x_value"] for row in values],
            [row["value"] for row in values],
            color=style["color"], marker=style["marker"], linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 2.5), markersize=style.get("markersize", 8),
            label=labels[method],
        )
        plotted.extend({**row, "display_name": labels[method], "semantic_figure_id": figure_id} for row in values)
    axis.set_xlabel("Number of RoIs")
    axis.set_ylabel("Energy efficiency (Mbit/J)")
    axis.set_xticks(range(2, 9))
    axis.grid(True, alpha=0.6)
    axis.legend(fontsize="small")
    return plotted


def _fixed_ee_data(spec, spec_path, figure_id):
    mapping = PAPER_METHOD_MAPPINGS[figure_id]
    methods = tuple(mapping.values()) if isinstance(mapping, dict) else tuple(mapping)
    runs = _resolve_suite_runs(spec, spec_path, "fixed_roi", methods)
    rows = _metric_rows(runs, "energy_efficiency_mbit_per_j")
    return runs, rows


def _run_mapping(runs):
    return {method: str(run["evaluation_dir"]) for method, run in runs.items()}


def _build_standalone_ee(spec, spec_path, output_dir, git_sha, figure_id):
    runs, rows = _fixed_ee_data(spec, spec_path, figure_id)
    import matplotlib.pyplot as plt

    size = (7, 4.2) if figure_id == "task_assignment_ee_vs_number_of_rois" else (9, 5.4)
    figure, axis = plt.subplots(figsize=size)
    plotted = _plot_ee_panel(axis, figure_id, rows)
    figure.tight_layout()
    return _emit(
        figure_id,
        figure,
        output_dir,
        plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "aggregation": "ratio of pooled timely delivered Mbit to pooled mobility J",
            "git_sha": git_sha,
        },
    )


def _build_ee_composite(spec, spec_path, output_dir, git_sha):
    figure_id = "energy_efficiency_design_comparisons"
    components = FIGURE_REGISTRY[figure_id]["components"]
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    all_rows = []
    all_runs = {}
    titles = ("(a) Task-assignment strategies", "(b) Trajectory-design methods", "(c) Hierarchical architectures")
    for axis, component, title in zip(axes, components, titles):
        runs, rows = _fixed_ee_data(spec, spec_path, component)
        all_runs.update(runs)
        all_rows.extend(_plot_ee_panel(axis, component, rows))
        axis.set_title(title, y=-0.27)
    figure.subplots_adjust(bottom=0.25, wspace=0.3)
    return _emit(
        figure_id,
        figure,
        output_dir,
        all_rows,
        {
            "source_spec": str(spec_path),
            "components": list(components),
            "method_to_evaluation_run": _run_mapping(all_runs),
            "aggregation": "ratio of pooled timely delivered Mbit to pooled mobility J",
            "git_sha": git_sha,
        },
    )


def _trajectory_rows(artifact):
    rows = []
    for snapshot in artifact["snapshots"]:
        for uav in snapshot["uavs"]:
            rows.append(
                {
                    "scenario_id": artifact["scenario_id"],
                    "target_uav_id": artifact["target_uav_id"],
                    "requested_time_seconds": snapshot["requested_time_seconds"],
                    "actual_time_seconds": snapshot["actual_time_seconds"],
                    "target_uav_phase": snapshot["target_uav_phase"],
                    **uav,
                }
            )
    return rows


def _draw_ground_circle(axis, x, y, radius, *, color, linewidth=1.0):
    import numpy as np

    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    axis.plot(x + radius * np.cos(theta), y + radius * np.sin(theta), 0.0, color=color, linewidth=linewidth)


def _phase_subtitle(phase):
    return {
        "Search": "UAV in search mode",
        "FOV": "UAV in VS mode",
        "COM": "UAV in COM mode",
        "FOV+COM": "UAV in VS and COM mode",
        "Hover": "UAV in hovering mode",
    }.get(str(phase), f"UAV in {phase} mode")


def _build_trajectory(spec, spec_path, output_dir, git_sha):
    figure_id = "uav_trajectory_snapshots"
    method = "td3_dinkelbach"
    runs = _resolve_suite_runs(spec, spec_path, figure_id, (method,))
    run = runs[method]
    target_uav_id = spec.get("target_uav_id")
    if target_uav_id is None:
        raise PaperFigureSpecError("paper spec requires target_uav_id")
    metadata_target = run["metadata"].get("target_uav_id")
    if int(metadata_target) != int(target_uav_id):
        raise IncompatiblePaperRunError(
            f"trajectory target mismatch: spec={target_uav_id}, evaluation={metadata_target}"
        )
    points = run["metadata"].get("points", [])
    if len(points) != 1:
        raise IncompatiblePaperRunError("trajectory evaluation must contain exactly one point")
    artifacts = _read_json(Path(points[0]["output_directory"]) / "trajectory_artifacts.json")
    if len(artifacts) != 1:
        raise IncompatiblePaperRunError("trajectory renderer requires exactly one scenario artifact")
    artifact = artifacts[0]
    if int(artifact.get("target_uav_id", -1)) != int(target_uav_id):
        raise IncompatiblePaperRunError("trajectory artifact target_uav_id mismatch")
    snapshots = artifact.get("snapshots", [])
    requested = tuple(float(item["requested_time_seconds"]) for item in snapshots)
    if requested != (5.0, 10.0, 15.0, 25.0):
        raise IncompatiblePaperRunError(f"unexpected trajectory snapshot times: {requested}")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(24, 6.4))
    target_path = artifact["uav_paths"][str(target_uav_id)]
    sr_paths = artifact["sr_paths"]
    for index, snapshot in enumerate(snapshots, 1):
        axis = figure.add_subplot(1, 4, index, projection="3d")
        actual = float(snapshot["actual_time_seconds"])
        uavs = {int(item["uav_id"]): item for item in snapshot["uavs"]}
        target = uavs[int(target_uav_id)]
        path = [item for item in target_path if float(item["actual_time_seconds"]) <= actual]
        axis.plot([p["x"] for p in path], [p["y"] for p in path], [p["z"] for p in path], color="green", linewidth=1.5)
        axis.scatter(path[0]["x"], path[0]["y"], path[0]["z"], color="green", s=36)
        axis.scatter(target["x"], target["y"], target["z"], facecolors="none", edgecolors="green", s=42)
        others = [item for uid, item in uavs.items() if uid != int(target_uav_id)]
        axis.scatter([u["x"] for u in others], [u["y"] for u in others], [u["z"] for u in others], color="lightgray", alpha=0.45, s=24)
        for sr in snapshot["sr_teams"]:
            sr_path = [item for item in sr_paths[str(sr["sr_id"])] if float(item["actual_time_seconds"]) <= actual]
            axis.plot([p["x"] for p in sr_path], [p["y"] for p in sr_path], [p["z"] for p in sr_path], color="#243BFF", linestyle="--", linewidth=1.0)
            axis.scatter(sr_path[0]["x"], sr_path[0]["y"], sr_path[0]["z"], color="#243BFF", marker="s", s=26)
            axis.scatter(sr["x"], sr["y"], sr["z"], facecolors="none", edgecolors="#243BFF", marker="s", s=30)
        gs = snapshot["ground_station"]
        axis.scatter(gs["x"], gs["y"], gs["z"], color="red", marker="^", s=50)
        for gt in snapshot["ground_targets"]:
            _draw_ground_circle(axis, gt["x"], gt["y"], gt["radius_m"], color="magenta" if gt["detected"] else "gray")
        for coverage in snapshot["sensing_coverage"]:
            bounds = coverage["clipped_bounds"]
            verts = [[
                (bounds["x_min"], bounds["y_min"], 0.0),
                (bounds["x_max"], bounds["y_min"], 0.0),
                (bounds["x_max"], bounds["y_max"], 0.0),
                (bounds["x_min"], bounds["y_max"], 0.0),
            ]]
            axis.add_collection3d(Poly3DCollection(verts, facecolor="#E8B95A", alpha=0.25, edgecolor="#C6902E"))
        for link in snapshot["active_links"]:
            sender = uavs[link["sender_id"]]
            receiver = gs if link["receiver_id"] == gs["gs_id"] else uavs[link["receiver_id"]]
            color = "red" if link["link_type"] == "U2U" else "purple"
            axis.plot([sender["x"], receiver["x"]], [sender["y"], receiver["y"]], [sender["z"], receiver["z"]], color=color, linestyle="--", linewidth=1.3)
        axis.set_xlim(0, 1000)
        axis.set_ylim(0, 1000)
        axis.set_zlim(0, 180)
        axis.set_xlabel("X(m)")
        axis.set_ylabel("Y(m)")
        axis.set_zlabel("Z(m)")
        axis.view_init(elev=20, azim=60)
        axis.set_title(f"({chr(96 + index)}) $t={snapshot['requested_time_seconds']:g}$: {_phase_subtitle(snapshot['target_uav_phase'])}", y=-0.19)
        handles = [
            Line2D([0], [0], color="green", label="Target UAV"),
            Line2D([0], [0], marker="o", color="green", linestyle="", label="UAV initial point"),
            Line2D([0], [0], marker="o", color="lightgray", linestyle="", label="Other UAVs"),
            Line2D([0], [0], color="#243BFF", linestyle="--", label="SR Path"),
            Line2D([0], [0], marker="s", color="#243BFF", linestyle="", label="SR team initial point"),
            Line2D([0], [0], marker="^", color="red", linestyle="", label="GS"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="magenta", color="none", label="Detected RoIs"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="gray", color="none", label="Other RoIs"),
            Line2D([0], [0], color="red", linestyle="--", label="U2U Link"),
            Line2D([0], [0], color="purple", linestyle="--", label="U2G Link"),
            Patch(facecolor="#E8B95A", alpha=0.25, label="Sensing coverage"),
        ]
        axis.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")
    figure.subplots_adjust(bottom=0.18, wspace=0.12)
    return _emit(
        figure_id,
        figure,
        output_dir,
        _trajectory_rows(artifact),
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "target_uav_id": int(target_uav_id),
            "scenario_id": artifact["scenario_id"],
            "scenario_manifest_hash": artifact["scenario_manifest_hash"],
            "checkpoint_path": artifact.get("checkpoint_path"),
            "checkpoint_fingerprint": artifact.get("checkpoint_fingerprint"),
            "subtitle_definition": "derived from each snapshot target_uav_phase",
            "git_sha": git_sha,
        },
        json_value=artifact,
    )


def _build_arrival(spec, spec_path, output_dir, git_sha):
    figure_id = "task_type_delay_vs_arrival_rate"
    mapping = PAPER_METHOD_MAPPINGS[figure_id]
    methods = tuple(mapping.values())
    runs = _resolve_suite_runs(spec, spec_path, figure_id, methods)
    rows = _metric_rows(runs, "average_e2e_delay_seconds")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    plotted = []
    for axis, task_type, title in zip(axes, ("COM", "FOV"), ("(a) COM task", "(b) VS task")):
        task_rows = [row for row in rows if row["task_type"] == task_type and row["swept_task"] == task_type]
        x_values = sorted({float(row["x_value"]) for row in task_rows})
        positions = np.arange(len(x_values), dtype=float)
        width = 0.22
        for method_index, (label, method) in enumerate(mapping.items()):
            selected = {float(row["x_value"]): row for row in task_rows if row["method_id"] == method}
            values = [
                (float(selected[x]["value"]) * 1000.0 if selected[x]["value"] is not None else math.nan)
                for x in x_values
            ]
            style = PLOT_STYLES[figure_id][method]
            axis.bar(positions + (method_index - 1) * width, values, width, color=style["color"], label=label)
            plotted.extend({**selected[x], "display_name": label, "plot_value_milliseconds": (None if selected[x]["value"] is None else selected[x]["value"] * 1000.0)} for x in x_values)
        axis.set_xticks(positions, [f"{value:g}" for value in x_values])
        axis.set_xlabel("Arrival rate (packet/s)")
        axis.set_ylabel("Average E2E delay (ms)")
        axis.set_title(title, y=-0.28)
        axis.grid(True, axis="y", linestyle="--", alpha=0.5)
        axis.legend(fontsize="small")
    figure.subplots_adjust(bottom=0.25, wspace=0.32)
    return _emit(
        figure_id, figure, output_dir, plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "aggregation": "pooled delivered-delay numerator / pooled delivered-packet denominator",
            "display_conversion": "seconds * 1000 = milliseconds",
            "missing_delay": "not plotted; retained as null with missing=true",
            "git_sha": git_sha,
        },
    )


def _build_violation(spec, spec_path, output_dir, git_sha):
    figure_id = "task_type_delay_violation_vs_target_delay"
    methods = tuple(PAPER_METHOD_MAPPINGS[figure_id])
    runs = _resolve_suite_runs(spec, spec_path, figure_id, methods)
    rows = _metric_rows(runs, "violation_probability")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = []
    for method in methods:
        style = PLOT_STYLES[figure_id][method]
        for task_type in ("FOV", "COM"):
            selected = sorted(
                (row for row in rows if row["method_id"] == method and row["task_type"] == task_type and row["swept_task"] == task_type),
                key=lambda row: float(row["x_value"]),
            )
            y_values = [
                float(row["value"]) if row["value"] is not None and float(row["value"]) > 0.0 else np.nan
                for row in selected
            ]
            task_label = "VS" if task_type == "FOV" else "COM"
            axis.plot(
                [row["x_value"] for row in selected], y_values,
                color=style["color"], marker=style["marker"], linewidth=2.5,
                linestyle="-" if task_type == "FOV" else "--",
                markerfacecolor=style["color"] if task_type == "FOV" else "none",
                label=f"{style['label']} ({task_label})",
            )
            plotted.extend({**row, "display_name": style["label"], "plot_value": (row["value"] if row["value"] and row["value"] > 0 else None), "log_zero_omitted": row["value"] == 0} for row in selected)
    axis.set_xlabel("Delay threshold (s)")
    axis.set_ylabel("Violation probability")
    axis.set_yscale("log")
    axis.set_xticks((0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    axis.legend(ncol=2, loc="lower left", fontsize="small")
    figure.tight_layout()
    return _emit(
        figure_id, figure, output_dir, plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "aggregation": "pooled violation count / pooled generated-packet count",
            "zero_probability_log_handling": "omitted as non-representable on log scale; no epsilon fabricated",
            "git_sha": git_sha,
        },
    )


def _build_roi_delay(spec, spec_path, output_dir, git_sha):
    figure_id = "task_type_delay_vs_number_of_rois"
    methods = tuple(PAPER_METHOD_MAPPINGS[figure_id])
    runs = _resolve_suite_runs(spec, spec_path, "fixed_roi", methods)
    rows = _metric_rows(runs, "average_e2e_delay_seconds")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = []
    for method in methods:
        style = PLOT_STYLES[figure_id][method]
        for task_type in ("FOV", "COM"):
            selected = sorted(
                (row for row in rows if row["method_id"] == method and row["task_type"] == task_type),
                key=lambda row: float(row["x_value"]),
            )
            y_values = [float(row["value"]) * 1000.0 if row["value"] is not None else np.nan for row in selected]
            task_label = "VS" if task_type == "FOV" else "COM"
            axis.plot(
                [row["x_value"] for row in selected], y_values,
                color=style["color"], marker=style["marker"], linewidth=2.5,
                linestyle="-" if task_type == "FOV" else "--",
                markerfacecolor=style["color"] if task_type == "FOV" else "none",
                label=f"{style['label']} ({task_label})",
            )
            plotted.extend({**row, "display_name": style["label"], "plot_value_milliseconds": (None if row["value"] is None else row["value"] * 1000.0)} for row in selected)
    axis.set_xlabel("Number of RoIs")
    axis.set_ylabel("Average E2E delays (ms)")
    axis.set_xticks(range(2, 9))
    axis.grid(True, alpha=0.6)
    axis.legend(loc="upper left", fontsize="small")
    figure.tight_layout()
    return _emit(
        figure_id, figure, output_dir, plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "aggregation": "pooled delivered-delay numerator / pooled delivered-packet denominator",
            "display_conversion": "seconds * 1000 = milliseconds",
            "missing_delay": "not plotted; retained as null with missing=true",
            "git_sha": git_sha,
        },
    )


def build_paper_figures(spec_path, figure="all", output_root="results/paper_figures"):
    spec_path = Path(spec_path).resolve()
    spec = _read_json(spec_path)
    if not isinstance(spec, dict):
        raise PaperFigureSpecError("paper spec root must be an object")
    git_sha = _git_sha()
    build_started = datetime.now(timezone.utc).isoformat()
    output_dir = _unique_output_directory(output_root, git_sha)
    if str(figure).lower() == "all":
        figure_ids = tuple(FIGURE_REGISTRY)
    else:
        figure_ids = (resolve_figure_id(figure),)
    outputs = {}
    for figure_id in figure_ids:
        if figure_id == "training_ee_vs_episode":
            outputs[figure_id] = _build_training(spec, spec_path, output_dir, git_sha)
        elif figure_id == "uav_trajectory_snapshots":
            outputs[figure_id] = _build_trajectory(spec, spec_path, output_dir, git_sha)
        elif figure_id == "energy_efficiency_design_comparisons":
            outputs[figure_id] = _build_ee_composite(spec, spec_path, output_dir, git_sha)
        elif figure_id in {
            "task_assignment_ee_vs_number_of_rois",
            "trajectory_design_ee_vs_number_of_rois",
            "hierarchical_architecture_ee_vs_number_of_rois",
        }:
            outputs[figure_id] = _build_standalone_ee(spec, spec_path, output_dir, git_sha, figure_id)
        elif figure_id == "task_type_delay_vs_arrival_rate":
            outputs[figure_id] = _build_arrival(spec, spec_path, output_dir, git_sha)
        elif figure_id == "task_type_delay_violation_vs_target_delay":
            outputs[figure_id] = _build_violation(spec, spec_path, output_dir, git_sha)
        elif figure_id == "task_type_delay_vs_number_of_rois":
            outputs[figure_id] = _build_roi_delay(spec, spec_path, output_dir, git_sha)
        else:
            raise AssertionError(f"unhandled semantic figure: {figure_id}")
    metadata = {
        "output_directory": str(output_dir),
        "source_spec": str(spec_path),
        "git_sha": git_sha,
        "build_started_utc": build_started,
        "semantic_figures": outputs,
        "new_training_started": False,
        "canonical_figure_ids": list(FIGURE_REGISTRY),
    }
    (output_dir / "paper_figure_build.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata
