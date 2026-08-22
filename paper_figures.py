"""Build semantic paper figures exclusively from unified-runner artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from experiment_config import (
    FORMAL_CHECKPOINT_EPISODE,
    MethodSpec,
    NUM_UAV,
    comparison_method_configuration,
)
from Packet_scheduler_v1 import (
    EPISODE_INJECTION_CUTOFF_SECONDS,
    TASK_DEADLINE_SECONDS,
)
from paper_figure_registry import (
    FIGURE_REGISTRY,
    METHOD_DISPLAY_NAMES,
    PAPER_METHOD_MAPPINGS,
    PLOT_STYLES,
    resolve_figure_ids,
)
from paper_metrics import (
    aggregate_paper_point_metrics,
    causal_trailing_average as _causal_trailing_average,
    compare_aggregate_collections,
    normalize_episode_ee as _normalize_episode_ee,
    paper_energy_efficiency as _paper_energy_efficiency,
    validate_aggregate_collection,
    validate_canonical_aggregate_rows,
)
from paper_trajectory import (
    build_standalone_trajectory_source,
    standalone_trajectory_csv_rows,
    validate_standalone_trajectory_source,
)
from scenario_manifest import ScenarioManifest, current_environment_config
from training_checkpoint import (
    CHECKPOINT_PROVENANCE_FIELDS,
    checkpoint_artifact_provenance,
    inspect_model_checkpoint,
)


class PaperFigureSpecError(ValueError):
    pass


class AmbiguousPaperRunError(PaperFigureSpecError):
    pass


class IncompatiblePaperRunError(PaperFigureSpecError):
    pass


def causal_trailing_average(values, window=50):
    return _causal_trailing_average(values, window=window)


def paper_energy_efficiency(timely_goodput_mbits, mobility_energy_j):
    return _paper_energy_efficiency(timely_goodput_mbits, mobility_energy_j)


def normalize_episode_ee(method_id, history_rows, window=50):
    try:
        return _normalize_episode_ee(method_id, history_rows, window=window)
    except ValueError as exc:
        raise PaperFigureSpecError(str(exc)) from exc


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


def _checkpoint_identity(method, training_seed):
    return {
        "method_id": method.method_id,
        "method_spec": method.to_dict(),
        "method_spec_fingerprint": method.compatible_fingerprints,
        "training_seed": int(training_seed),
        "movement_agent": method.agent,
        "reward_mode": method.reward_mode,
        "task_potential_enabled": bool(method.task_potential_enabled),
        **comparison_method_configuration(method),
    }


def _inspect_checkpoint(method_id, checkpoint_dir, training_seed, formal_config):
    method = MethodSpec.parse(method_id)
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if checkpoint_dir.name != f"ep_{FORMAL_CHECKPOINT_EPISODE:04d}":
        raise IncompatiblePaperRunError(
            f"{method_id} checkpoint is not the formal ep_2500 directory: {checkpoint_dir}"
        )
    try:
        inspected = inspect_model_checkpoint(
            checkpoint_dir,
            expected_experiment_metadata=_checkpoint_identity(method, training_seed),
            expected_completed_episodes=FORMAL_CHECKPOINT_EPISODE,
            expected_formal_config=formal_config,
            require_episode_directory=True,
            movement_agent_kind=method.agent,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise IncompatiblePaperRunError(
            f"checkpoint provenance is invalid for {method_id}: {exc}"
        ) from exc
    provenance = checkpoint_artifact_provenance(
        checkpoint_dir, metadata=inspected["metadata"]
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_metadata": inspected["metadata"],
        **provenance,
    }


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
    inspected = _inspect_checkpoint(
        method_id,
        checkpoint_dir,
        int(resolved["seed"]),
        resolved.get("training_config"),
    )
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
        **{field: inspected[field] for field in CHECKPOINT_PROVENANCE_FIELDS},
    }


def _expected_resolved_overrides(suite, point):
    rates = {"FOV": None, "COM": None}
    deadlines = {
        "FOV": float(TASK_DEADLINE_SECONDS["FOV"]),
        "COM": float(TASK_DEADLINE_SECONDS["COM"]),
    }
    flat = {}
    injection_cutoff = float(
        point.get("overrides", {}).get(
            "packet_injection_cutoff_seconds",
            EPISODE_INJECTION_CUTOFF_SECONDS,
        )
    )
    if suite == "task_type_delay_vs_arrival_rate":
        value = float(point["x_value"])
        if point["swept_task"] == "COM":
            rates = {"FOV": 5.0, "COM": value}
        elif point["swept_task"] == "FOV":
            rates = {"FOV": value, "COM": 50.0}
        else:
            raise IncompatiblePaperRunError("arrival point has an invalid swept_task")
        flat = {
            "fov_rate_packets_per_second": rates["FOV"],
            "com_rate_packets_per_second": rates["COM"],
        }
    elif suite == "task_type_delay_violation_vs_target_delay":
        value = float(point["x_value"])
        if point["swept_task"] not in deadlines:
            raise IncompatiblePaperRunError("deadline point has an invalid swept_task")
        deadlines[point["swept_task"]] = value
        flat = {
            "fov_deadline_seconds": deadlines["FOV"],
            "com_deadline_seconds": deadlines["COM"],
            "packet_injection_cutoff_seconds": injection_cutoff,
        }
    expected = {
        "traffic_rates_packets_per_second": rates,
        "task_deadlines_seconds": deadlines,
        "packet_injection_cutoff_seconds": injection_cutoff,
        "units": {
            "traffic_rate": "packets/s",
            "deadline": "seconds",
            "packet_injection_cutoff": "seconds",
        },
    }
    return flat, expected


def _validate_manifest_point(point, metadata, suite):
    path_value = point.get("scenario_manifest_path")
    if not path_value:
        raise IncompatiblePaperRunError(
            f"{suite}/{point.get('point_id')} lacks scenario_manifest_path"
        )
    manifest_path = Path(path_value).resolve()
    output_directory = Path(point.get("output_directory", "")).resolve()
    if output_directory != manifest_path.parent:
        raise IncompatiblePaperRunError(
            f"manifest/output directory mismatch at {point.get('point_id')}"
        )
    try:
        manifest = ScenarioManifest.load(manifest_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise IncompatiblePaperRunError(
            f"scenario manifest is invalid at {manifest_path}: {exc}"
        ) from exc
    hashes = {
        point.get("scenario_manifest_hash"),
        point.get("manifest_hash"),
        manifest.content_hash,
    }
    if None in hashes or len(hashes) != 1:
        raise IncompatiblePaperRunError(
            f"scenario manifest hash mismatch at {point.get('point_id')}"
        )
    scenario_ids = [entry["scenario_id"] for entry in manifest.episodes]
    if point.get("scenario_ids") != scenario_ids:
        raise IncompatiblePaperRunError(
            f"scenario IDs mismatch at {point.get('point_id')}"
        )
    if int(point.get("evaluation_episode_count", -1)) != manifest.episode_count:
        raise IncompatiblePaperRunError(
            f"evaluation episode count mismatch at {point.get('point_id')}"
        )
    if int(metadata.get("evaluation_episodes_per_point", -1)) != manifest.episode_count:
        raise IncompatiblePaperRunError("top-level evaluation episode count is incompatible")
    horizon = int(point.get("evaluation_horizon_seconds", -1))
    if horizon != int(metadata.get("evaluation_horizon_seconds", -2)):
        raise IncompatiblePaperRunError(
            f"evaluation horizon mismatch at {point.get('point_id')}"
        )
    if horizon != int(current_environment_config()["episode_seconds"]):
        raise IncompatiblePaperRunError(
            f"manifest horizon mismatch at {point.get('point_id')}"
        )
    if int(point.get("evaluation_seed", -1)) != int(metadata.get("evaluation_seed", -2)):
        raise IncompatiblePaperRunError(
            f"evaluation seed mismatch at {point.get('point_id')}"
        )
    if int(point.get("manifest_seed", -1)) != manifest.manifest_seed:
        raise IncompatiblePaperRunError(
            f"manifest seed mismatch at {point.get('point_id')}"
        )
    if int(point["manifest_seed"]) != int(metadata.get("manifest_seed", -2)):
        raise IncompatiblePaperRunError(
            f"top-level manifest seed mismatch at {point.get('point_id')}"
        )
    if int(point.get("num_uav", -1)) != NUM_UAV or any(
        len(entry.get("uavs", ())) != NUM_UAV for entry in manifest.episodes
    ):
        raise IncompatiblePaperRunError(
            f"paper evaluation requires {NUM_UAV} UAVs at {point.get('point_id')}"
        )
    if suite == "fixed_roi":
        expected_num_gt = int(point["x_value"])
        if int(point.get("fixed_num_gt", -1)) != expected_num_gt or any(
            int(entry.get("num_GT", -1)) != expected_num_gt
            for entry in manifest.episodes
        ):
            raise IncompatiblePaperRunError(
                f"fixed-RoI manifest mismatch at {point.get('point_id')}"
            )
    expected_flat, expected_resolved = _expected_resolved_overrides(suite, point)
    if point.get("overrides") != expected_flat:
        raise IncompatiblePaperRunError(
            f"sweep overrides mismatch at {point.get('point_id')}"
        )
    if point.get("resolved_overrides") != expected_resolved:
        raise IncompatiblePaperRunError(
            f"resolved sweep overrides mismatch at {point.get('point_id')}"
        )
    return {
        "point_id": point.get("point_id"),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest.content_hash,
        "scenario_ids": scenario_ids,
        "evaluation_episode_count": manifest.episode_count,
        "evaluation_horizon_seconds": horizon,
        "evaluation_seed": int(point["evaluation_seed"]),
        "manifest_seed": manifest.manifest_seed,
        "num_uav": NUM_UAV,
        "resolved_overrides": expected_resolved,
    }


def _validate_checkpoint_record(record, expected, *, method_id, point_id, layer):
    expected_path = expected.get("checkpoint_path")
    actual_path = record.get("checkpoint_path")
    if expected_path is None:
        if actual_path is not None:
            raise IncompatiblePaperRunError(
                f"method={method_id}, point={point_id}, layer={layer}: "
                f"checkpoint_path mismatch: expected=None, actual={actual_path!r}"
            )
    elif not actual_path or Path(actual_path).resolve() != Path(expected_path).resolve():
        raise IncompatiblePaperRunError(
            f"method={method_id}, point={point_id}, layer={layer}: "
            f"checkpoint_path mismatch: expected={expected_path!r}, actual={actual_path!r}"
        )
    for field in CHECKPOINT_PROVENANCE_FIELDS:
        expected_value = expected.get(field)
        actual_value = record.get(field)
        if actual_value != expected_value:
            raise IncompatiblePaperRunError(
                f"method={method_id}, point={point_id}, layer={layer}: "
                f"{field} mismatch: expected={expected_value!r}, actual={actual_value!r}"
            )


def _read_point_episode_rows(point, *, method_id):
    outputs = point.get("outputs") or {}
    path_value = outputs.get("per_episode_jsonl")
    if not path_value:
        raise IncompatiblePaperRunError(
            f"method={method_id}, point={point.get('point_id')}: "
            "point metadata lacks outputs.per_episode_jsonl"
        )
    path = Path(path_value).resolve()
    expected = Path(point["output_directory"]).resolve() / "per_episode.jsonl"
    if path != expected:
        raise IncompatiblePaperRunError(
            f"method={method_id}, point={point.get('point_id')}: per-episode path "
            f"mismatch: expected={expected}, actual={path}"
        )
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise IncompatiblePaperRunError(
            f"method={method_id}, point={point.get('point_id')}: "
            f"per-episode artifact is missing or invalid: {path}"
        ) from exc
    if not rows:
        raise IncompatiblePaperRunError(
            f"method={method_id}, point={point.get('point_id')}: "
            "per-episode artifact is empty"
        )
    return rows


def _validate_aggregate_layers(evaluation_dir, metadata, top_rows):
    suite = metadata["semantic_suite"]
    method_id = metadata["method_id"]
    if suite == "uav_trajectory_snapshots":
        return top_rows
    points = metadata["points"]
    point_ids = [point["point_id"] for point in points]
    combined = []
    try:
        validate_aggregate_collection(top_rows, method_id, point_ids)
        for point in points:
            point_id = point["point_id"]
            aggregate_path = Path(point.get("aggregated_plot_data", "")).resolve()
            expected_path = (
                Path(point["output_directory"]).resolve()
                / "aggregated_plot_data.json"
            )
            if aggregate_path != expected_path:
                raise ValueError(
                    f"method={method_id}, point={point_id}: point aggregate path "
                    f"mismatch: expected={expected_path}, actual={aggregate_path}"
                )
            point_rows = _read_json(aggregate_path)
            validate_canonical_aggregate_rows(point_rows, method_id, point_id)
            episode_rows = _read_point_episode_rows(point, method_id=method_id)
            recomputed = aggregate_paper_point_metrics(
                method_id, suite, point, episode_rows
            )
            compare_aggregate_collections(
                recomputed,
                point_rows,
                context=(
                    f"point-level/per-episode mismatch: method={method_id}, "
                    f"point={point_id}"
                ),
            )
            combined.extend(point_rows)
        compare_aggregate_collections(
            combined,
            top_rows,
            context=(
                f"top-level/point-level mismatch: method={method_id}, "
                f"evaluation_dir={Path(evaluation_dir).resolve()}"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, IncompatiblePaperRunError):
            raise
        raise IncompatiblePaperRunError(str(exc)) from exc
    return top_rows


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
        training_run_value = metadata.get("training_run")
        if not training_run_value:
            raise IncompatiblePaperRunError(f"{method_id} lacks training-run provenance")
        training_run = Path(training_run_value).resolve()
        training_resolved = _read_json(training_run / "resolved_config.json")
        if training_resolved.get("method") != method_id:
            raise IncompatiblePaperRunError("evaluation training-run method mismatch")
        if training_resolved.get("method_spec") != method.to_dict():
            raise IncompatiblePaperRunError("evaluation training-run method spec mismatch")
        expected_checkpoint = (
            training_run / "checkpoints" / "models" / f"ep_{FORMAL_CHECKPOINT_EPISODE:04d}"
        ).resolve()
        checkpoint_path = Path(metadata["checkpoint_path"]).resolve()
        if checkpoint_path != expected_checkpoint:
            raise IncompatiblePaperRunError(
                f"checkpoint path does not belong to the resolved training run: {checkpoint_path}"
            )
        formal_config = metadata.get("formal_training_config")
        if formal_config != training_resolved.get("training_config"):
            raise IncompatiblePaperRunError("formal training config provenance mismatch")
        training_seed = int(metadata.get("training_seed", -1))
        if training_seed != int(training_resolved.get("seed", -2)):
            raise IncompatiblePaperRunError("checkpoint training seed mismatch")
        checkpoint = _inspect_checkpoint(
            method_id, checkpoint_path, training_seed, formal_config
        )
        checkpoint_provenance = {
            "checkpoint_path": str(checkpoint_path),
            **{
                field: checkpoint[field] for field in CHECKPOINT_PROVENANCE_FIELDS
            },
        }
    else:
        checkpoint_path = None
        checkpoint_provenance = {
            "checkpoint_path": None,
            **{field: None for field in CHECKPOINT_PROVENANCE_FIELDS},
        }
        if metadata.get("training_run") is not None or any(
            metadata.get(field) is not None
            for field in ("checkpoint_path", *CHECKPOINT_PROVENANCE_FIELDS)
        ):
            raise IncompatiblePaperRunError(
                f"method={method_id}: pure-random evaluation must have no neural "
                "checkpoint provenance or training run"
            )
    _validate_checkpoint_record(
        metadata,
        checkpoint_provenance,
        method_id=method_id,
        point_id="<top-level>",
        layer="paper_evaluation_metadata.json",
    )
    points = metadata.get("points")
    if not isinstance(points, list) or not points:
        raise PaperFigureSpecError(f"evaluation has no point metadata: {evaluation_dir}")
    for point in points:
        if bool(point.get("checkpoint_required")) != checkpoint_required:
            raise IncompatiblePaperRunError(
                f"checkpoint requirement mismatch at {point.get('point_id')}"
            )
        point_id = point.get("point_id")
        _validate_checkpoint_record(
            point,
            checkpoint_provenance,
            method_id=method_id,
            point_id=point_id,
            layer="point metadata",
        )
        outputs = point.get("outputs") or {}
        run_metadata_path = Path(outputs.get("run_metadata", "")).resolve()
        expected_run_metadata_path = (
            Path(point["output_directory"]).resolve() / "run_metadata.json"
        )
        if run_metadata_path != expected_run_metadata_path:
            raise IncompatiblePaperRunError(
                f"method={method_id}, point={point_id}: run metadata path mismatch: "
                f"expected={expected_run_metadata_path}, actual={run_metadata_path}"
            )
        point_run_metadata = _read_json(run_metadata_path)
        _validate_checkpoint_record(
            point_run_metadata,
            checkpoint_provenance,
            method_id=method_id,
            point_id=point_id,
            layer="run_metadata.json",
        )
    point_provenance = [
        _validate_manifest_point(point, metadata, expected_suite) for point in points
    ]
    aggregate_path = evaluation_dir / "aggregated_plot_data.json"
    aggregate_rows = _read_json(aggregate_path)
    if not isinstance(aggregate_rows, list) or not aggregate_rows:
        raise PaperFigureSpecError(f"evaluation aggregate is empty: {aggregate_path}")
    _validate_aggregate_layers(evaluation_dir, metadata, aggregate_rows)
    return {
        "evaluation_dir": evaluation_dir,
        "metadata": metadata,
        "aggregate_path": aggregate_path,
        "aggregate_rows": aggregate_rows,
        **{
            field: checkpoint_provenance[field]
            for field in CHECKPOINT_PROVENANCE_FIELDS
        },
        "checkpoint_provenance": checkpoint_provenance,
        "point_provenance": point_provenance,
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
    axes = list(figure.axes)
    render_contract = {
        "axes_count": len(axes),
        "axes_titles": [axis.get_title() for axis in axes],
        "axes_title_positions": [list(axis.title.get_position()) for axis in axes],
    }
    if render_contract["axes_count"] != 1:
        raise PaperFigureSpecError(
            f"formal figure {figure_id} must contain exactly one axes"
        )
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
                "resolved_style": PLOT_STYLES.get(
                    figure_id,
                    PLOT_STYLES.get("uav_trajectory_snapshots"),
                ),
                "render_contract": render_contract,
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
        "method_to_checkpoint_provenance": resolved.get(
            "method_to_checkpoint_provenance", {}
        ),
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
    for method, run in runs.items():
        episodes = sorted(int(row["episode"]) for row in run["history_rows"])
        expected = list(range(1, FORMAL_CHECKPOINT_EPISODE + 1))
        if episodes != expected:
            raise IncompatiblePaperRunError(
                f"{method} training history must contain exactly episodes 1..2500"
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
            "method_to_checkpoint_provenance": {
                method: {
                    "checkpoint_path": str(run["checkpoint_path"]),
                    **{
                        field: run[field]
                        for field in CHECKPOINT_PROVENANCE_FIELDS
                    },
                }
                for method, run in runs.items()
            },
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
            (
                point["point_id"],
                point["manifest_hash"],
                tuple(point["scenario_ids"]),
                point["evaluation_episode_count"],
                point["evaluation_horizon_seconds"],
                point["evaluation_seed"],
                point["num_uav"],
            )
            for point in run["point_provenance"]
        )
        if reference is None:
            reference = points
        elif points != reference:
            raise IncompatiblePaperRunError(
                f"{suite} methods do not share identical sweep points and manifests; mismatch={method}"
            )
    expected = _expected_sweep_signature(suite)
    actual = tuple(
        (point.get("point_id"), point.get("x_value"), point.get("x_unit"))
        for point in next(iter(runs.values()))["metadata"]["points"]
    )
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


def _metric_rows(
    runs,
    metric,
    *,
    point_ids,
    task_types=(None,),
    swept_only=False,
):
    expected_points = tuple(point_ids)
    expected_tasks = tuple(task_types)
    if swept_only:
        point_tasks = {
            point_id: (
                "COM" if point_id.startswith("com_") else "FOV"
                if point_id.startswith("fov_")
                else None
            )
            for point_id in expected_points
        }
        if any(task not in expected_tasks for task in point_tasks.values()):
            raise PaperFigureSpecError(
                f"cannot infer swept task from point IDs for {metric}"
            )
        expected = {
            (method, point_id, point_tasks[point_id])
            for method in runs
            for point_id in expected_points
        }
    else:
        expected = {
            (method, point_id, task_type)
            for method in runs
            for point_id in expected_points
            for task_type in expected_tasks
        }
    found = {}
    for method, run in runs.items():
        for source in run["aggregate_rows"]:
            if source.get("metric") != metric:
                continue
            task_type = source.get("task_type")
            if swept_only and task_type != source.get("swept_task"):
                continue
            if task_type not in expected_tasks:
                if swept_only and task_type in {"FOV", "COM"}:
                    continue
                raise IncompatiblePaperRunError(
                    f"unexpected task for {metric}: {task_type}"
                )
            point_id = source.get("point_id")
            if point_id not in expected_points:
                raise IncompatiblePaperRunError(
                    f"unexpected point for {metric}: {point_id}"
                )
            if source.get("method_id") != method:
                raise IncompatiblePaperRunError(
                    f"aggregate method mismatch: expected={method}, found={source.get('method_id')}"
                )
            key = (method, point_id, task_type)
            if key in found:
                raise IncompatiblePaperRunError(f"duplicate aggregate row: {key}")
            if source.get("missing") and metric == "energy_efficiency_mbit_per_j":
                raise PaperFigureSpecError(f"{method} has missing EE at {point_id}")
            found[key] = dict(source)
    missing = expected.difference(found)
    extra = set(found).difference(expected)
    if missing or extra:
        raise IncompatiblePaperRunError(
            f"{metric} aggregate Cartesian mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [found[key] for key in sorted(found, key=lambda item: tuple(str(v) for v in item))]


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
    rows = _metric_rows(
        runs,
        "energy_efficiency_mbit_per_j",
        point_ids=tuple(f"roi_{value}" for value in range(2, 9)),
    )
    return runs, rows


def _run_mapping(runs):
    return {method: str(run["evaluation_dir"]) for method, run in runs.items()}


def _validated_provenance(runs):
    return {
        "method_to_checkpoint_provenance": {
            method: dict(run["checkpoint_provenance"])
            for method, run in runs.items()
        },
        "validated_points": {
            method: run["point_provenance"] for method, run in runs.items()
        },
    }


def _build_standalone_ee(spec, spec_path, output_dir, git_sha, figure_id):
    runs, rows = _fixed_ee_data(spec, spec_path, figure_id)
    import matplotlib.pyplot as plt

    size = (7, 4.2) if figure_id == "task_assignment_ee_vs_number_of_rois" else (9, 5.4)
    figure, axis = plt.subplots(figsize=size)
    plotted = _plot_ee_panel(axis, figure_id, rows)
    title = {
        "task_assignment_ee_vs_number_of_rois": "Task-assignment strategies",
        "trajectory_design_ee_vs_number_of_rois": "Trajectory-design methods",
        "hierarchical_architecture_ee_vs_number_of_rois": "Hierarchical learning architectures",
    }[figure_id]
    axis.set_title(title)
    figure.tight_layout()
    return _emit(
        figure_id,
        figure,
        output_dir,
        plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            **_validated_provenance(runs),
            "aggregation": "ratio of pooled timely delivered Mbit to pooled mobility J",
            "title": title,
            "git_sha": git_sha,
        },
    )


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


def render_standalone_trajectory_source(source):
    """Render one axes using only a standalone trajectory source object."""

    validate_standalone_trajectory_source(source)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(7.2, 6.4))
    axis = figure.add_subplot(projection="3d")
    actual = float(source["actual_time_seconds"])
    target_uav_id = int(source["target_uav_id"])
    uavs = {int(item["uav_id"]): item for item in source["uavs"]}
    target = uavs[target_uav_id]
    target_path = source["uav_paths"][str(target_uav_id)]
    axis.plot(
        [point["x"] for point in target_path],
        [point["y"] for point in target_path],
        [point["z"] for point in target_path],
        color="green",
        linewidth=1.5,
    )
    axis.scatter(
        target_path[0]["x"],
        target_path[0]["y"],
        target_path[0]["z"],
        color="green",
        s=36,
    )
    axis.scatter(
        target["x"],
        target["y"],
        target["z"],
        facecolors="none",
        edgecolors="green",
        s=42,
    )
    others = [item for uid, item in uavs.items() if uid != target_uav_id]
    axis.scatter(
        [uav["x"] for uav in others],
        [uav["y"] for uav in others],
        [uav["z"] for uav in others],
        color="lightgray",
        alpha=0.45,
        s=24,
    )
    for sr in source["sr_teams"]:
        sr_path = source["sr_paths"][str(sr["sr_id"])]
        axis.plot(
            [point["x"] for point in sr_path],
            [point["y"] for point in sr_path],
            [point["z"] for point in sr_path],
            color="#243BFF",
            linestyle="--",
            linewidth=1.0,
        )
        axis.scatter(
            sr_path[0]["x"],
            sr_path[0]["y"],
            sr_path[0]["z"],
            color="#243BFF",
            marker="s",
            s=26,
        )
        axis.scatter(
            sr["x"],
            sr["y"],
            sr["z"],
            facecolors="none",
            edgecolors="#243BFF",
            marker="s",
            s=30,
        )
    ground_station = source["ground_station"]
    axis.scatter(
        ground_station["x"],
        ground_station["y"],
        ground_station["z"],
        color="red",
        marker="^",
        s=50,
    )
    for target_row in source["ground_targets"]:
        _draw_ground_circle(
            axis,
            target_row["x"],
            target_row["y"],
            target_row["radius_m"],
            color="magenta" if target_row["detected"] else "gray",
        )
    for coverage in source["sensing_coverage"]:
        bounds = coverage["clipped_bounds"]
        vertices = [[
            (bounds["x_min"], bounds["y_min"], 0.0),
            (bounds["x_max"], bounds["y_min"], 0.0),
            (bounds["x_max"], bounds["y_max"], 0.0),
            (bounds["x_min"], bounds["y_max"], 0.0),
        ]]
        axis.add_collection3d(
            Poly3DCollection(
                vertices,
                facecolor="#E8B95A",
                alpha=0.25,
                edgecolor="#C6902E",
            )
        )
    for link in source["active_links"]:
        sender = uavs[int(link["sender_id"])]
        receiver = (
            ground_station
            if int(link["receiver_id"]) == int(ground_station["gs_id"])
            else uavs[int(link["receiver_id"])]
        )
        color = "red" if link["link_type"] == "U2U" else "purple"
        axis.plot(
            [sender["x"], receiver["x"]],
            [sender["y"], receiver["y"]],
            [sender["z"], receiver["z"]],
            color=color,
            linestyle="--",
            linewidth=1.3,
        )
    contract = source["render_contract"]
    limits = contract["axis_limits"]
    labels = contract["axis_labels"]
    axis.set_xlim(*limits["x"])
    axis.set_ylim(*limits["y"])
    axis.set_zlim(*limits["z"])
    axis.set_xlabel(labels["x"])
    axis.set_ylabel(labels["y"])
    axis.set_zlabel(labels["z"])
    camera = contract["camera"]
    axis.view_init(
        elev=float(camera["elevation_degrees"]),
        azim=float(camera["azimuth_degrees"]),
    )
    title = f"t = {actual:g} s: {_phase_subtitle(source['actual_phase'])}"
    axis.set_title(title)
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
    figure.tight_layout()
    return figure


def _build_trajectory(spec, spec_path, output_dir, git_sha, figure_id):
    suite = "uav_trajectory_snapshots"
    method = "td3_dinkelbach"
    runs = _resolve_suite_runs(spec, spec_path, suite, (method,))
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
    point_provenance = run["point_provenance"][0]
    if artifact.get("scenario_manifest_hash") != point_provenance["manifest_hash"]:
        raise IncompatiblePaperRunError("trajectory artifact manifest hash mismatch")
    if artifact.get("scenario_id") not in point_provenance["scenario_ids"]:
        raise IncompatiblePaperRunError("trajectory artifact scenario ID mismatch")
    if Path(artifact.get("checkpoint_path", "")).resolve() != Path(
        run["metadata"]["checkpoint_path"]
    ).resolve():
        raise IncompatiblePaperRunError("trajectory artifact checkpoint path mismatch")
    _validate_checkpoint_record(
        artifact,
        run["checkpoint_provenance"],
        method_id=method,
        point_id="trajectory",
        layer="trajectory artifact",
    )
    snapshots = artifact.get("snapshots", [])
    requested = tuple(float(item["requested_time_seconds"]) for item in snapshots)
    if requested != (5.0, 10.0, 15.0, 25.0):
        raise IncompatiblePaperRunError(f"unexpected trajectory snapshot times: {requested}")
    requested_time = float(FIGURE_REGISTRY[figure_id]["requested_time_seconds"])
    selected = [
        snapshot
        for snapshot in snapshots
        if float(snapshot["requested_time_seconds"]) == requested_time
    ]
    if len(selected) != 1:
        raise IncompatiblePaperRunError(
            f"trajectory artifact must contain one snapshot requested at {requested_time:g} s"
        )
    snapshot = selected[0]
    source = build_standalone_trajectory_source(
        figure_id=figure_id,
        method_id=method,
        artifact=artifact,
        snapshot=snapshot,
        requested_time_seconds=requested_time,
        git_sha=git_sha,
        checkpoint_provenance=run["checkpoint_provenance"],
        camera=FIGURE_REGISTRY[figure_id]["camera"],
        style=PLOT_STYLES[suite],
    )
    figure = render_standalone_trajectory_source(source)
    actual = source["actual_time_seconds"]
    title = figure.axes[0].get_title()
    rows = standalone_trajectory_csv_rows(source)
    return _emit(
        figure_id,
        figure,
        output_dir,
        rows,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            "target_uav_id": int(target_uav_id),
            "scenario_id": artifact["scenario_id"],
            "scenario_manifest_hash": artifact["scenario_manifest_hash"],
            "checkpoint_path": source["checkpoint_path"],
            **{field: source[field] for field in CHECKPOINT_PROVENANCE_FIELDS},
            **_validated_provenance(runs),
            "requested_time_seconds": requested_time,
            "actual_time_seconds": actual,
            "target_uav_phase": source["actual_phase"],
            "title": title,
            "title_definition": "actual artifact time and target_uav_phase",
            "git_sha": git_sha,
        },
        json_value=source,
    )


def _build_arrival(spec, spec_path, output_dir, git_sha, figure_id):
    suite = "task_type_delay_vs_arrival_rate"
    mapping = PAPER_METHOD_MAPPINGS[figure_id]
    methods = tuple(mapping.values())
    runs = _resolve_suite_runs(spec, spec_path, suite, methods)
    task_type = FIGURE_REGISTRY[figure_id]["task_type"]
    x_values = (
        (50.0, 100.0, 150.0, 200.0)
        if task_type == "COM"
        else (10.0, 20.0, 30.0, 40.0)
    )
    point_ids = tuple(f"{task_type.lower()}_rate_{value:g}" for value in x_values)
    rows = _metric_rows(
        runs,
        "average_e2e_delay_seconds",
        point_ids=point_ids,
        task_types=(task_type,),
        swept_only=True,
    )
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(6, 4.3))
    plotted = []
    positions = np.arange(len(x_values), dtype=float)
    width = 0.22
    for method_index, (label, method) in enumerate(mapping.items()):
        selected = {
            float(row["x_value"]): row
            for row in rows
            if row["method_id"] == method
        }
        values = [
            (float(selected[x]["value"]) * 1000.0 if selected[x]["value"] is not None else math.nan)
            for x in x_values
        ]
        style = PLOT_STYLES[figure_id][method]
        axis.bar(positions + (method_index - 1) * width, values, width, color=style["color"], label=label)
        plotted.extend({**selected[x], "display_name": label, "plot_value_milliseconds": (None if selected[x]["value"] is None else selected[x]["value"] * 1000.0)} for x in x_values)
    title = "COM task" if task_type == "COM" else "VS task"
    axis.set_xticks(positions, [f"{value:g}" for value in x_values])
    axis.set_xlabel("Arrival rate (packet/s)")
    axis.set_ylabel("Average E2E delay (ms)")
    axis.set_title(title)
    axis.grid(True, axis="y", linestyle="--", alpha=0.5)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return _emit(
        figure_id, figure, output_dir, plotted,
        {
            "source_spec": str(spec_path),
            "method_to_evaluation_run": _run_mapping(runs),
            **_validated_provenance(runs),
            "aggregation": "pooled delivered-delay numerator / pooled delivered-packet denominator",
            "display_conversion": "seconds * 1000 = milliseconds",
            "missing_delay": "not plotted; retained as null with missing=true",
            "task_type": task_type,
            "title": title,
            "git_sha": git_sha,
        },
    )


def _build_violation(spec, spec_path, output_dir, git_sha):
    figure_id = "task_type_delay_violation_vs_target_delay"
    methods = tuple(PAPER_METHOD_MAPPINGS[figure_id])
    runs = _resolve_suite_runs(spec, spec_path, figure_id, methods)
    point_ids = tuple(
        f"{task.lower()}_deadline_{value:g}s"
        for task in ("COM", "FOV")
        for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    )
    rows = _metric_rows(
        runs,
        "violation_probability",
        point_ids=point_ids,
        task_types=("FOV", "COM"),
        swept_only=True,
    )
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
            **_validated_provenance(runs),
            "aggregation": "pooled violation count / pooled generated-packet count",
            "zero_probability_log_handling": "omitted as non-representable on log scale; no epsilon fabricated",
            "git_sha": git_sha,
        },
    )


def _build_roi_delay(spec, spec_path, output_dir, git_sha):
    figure_id = "task_type_delay_vs_number_of_rois"
    methods = tuple(PAPER_METHOD_MAPPINGS[figure_id])
    runs = _resolve_suite_runs(spec, spec_path, "fixed_roi", methods)
    rows = _metric_rows(
        runs,
        "average_e2e_delay_seconds",
        point_ids=tuple(f"roi_{value}" for value in range(2, 9)),
        task_types=("FOV", "COM"),
    )
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
            **_validated_provenance(runs),
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
        figure_ids = resolve_figure_ids(figure)
    outputs = {}
    for figure_id in figure_ids:
        if figure_id == "training_ee_vs_episode":
            outputs[figure_id] = _build_training(spec, spec_path, output_dir, git_sha)
        elif figure_id.startswith("uav_trajectory_t_"):
            outputs[figure_id] = _build_trajectory(
                spec, spec_path, output_dir, git_sha, figure_id
            )
        elif figure_id in {
            "task_assignment_ee_vs_number_of_rois",
            "trajectory_design_ee_vs_number_of_rois",
            "hierarchical_architecture_ee_vs_number_of_rois",
        }:
            outputs[figure_id] = _build_standalone_ee(spec, spec_path, output_dir, git_sha, figure_id)
        elif figure_id in {
            "com_task_delay_vs_arrival_rate",
            "vs_task_delay_vs_arrival_rate",
        }:
            outputs[figure_id] = _build_arrival(
                spec, spec_path, output_dir, git_sha, figure_id
            )
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
