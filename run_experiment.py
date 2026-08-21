"""Run exactly one registered trajectory method in an isolated directory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid

from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    METHOD_REGISTRY,
    MethodSpec,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
)
from HRL_task_aware import formal_training_config, train
from scenario_manifest import generate_manifest


def _git_short_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def create_unique_run_directory(output_root, method_key, seed, git_sha=None):
    """Atomically create a new leaf; a collision is never silently reused."""

    parent = Path(output_root) / str(method_key)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sha = str(git_sha or _git_short_sha())[:12]
    base = f"{stamp}_seed{int(seed)}_{sha}"
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt:02d}-{uuid.uuid4().hex[:8]}"
        candidate = parent / f"{base}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate.resolve()
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique run directory below {parent}")


def _method(value):
    try:
        return MethodSpec.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _roi_count(value):
    value = int(value)
    if not ROI_COUNT_MIN <= value <= ROI_COUNT_MAX:
        raise argparse.ArgumentTypeError(
            f"RoI count must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
        )
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one controlled trajectory experiment"
    )
    parser.add_argument("method", type=_method, metavar="METHOD")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episode-seconds", type=int)
    parser.add_argument("--roi-count", type=_roi_count)
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="default to one episode/one movement transition and no checkpoints",
    )
    return parser


def _resolved_values(args):
    defaults = FORMAL_EXPERIMENT_DEFAULTS
    episodes = args.episodes
    if episodes is None:
        episodes = 1 if args.smoke else defaults["training_episodes_per_seed"]
    episode_seconds = args.episode_seconds
    if episode_seconds is None:
        episode_seconds = 1 if args.smoke else defaults["episode_seconds"]
    seed = defaults["training_seed"] if args.seed is None else args.seed
    checkpoint_interval = (
        defaults["checkpoint_interval_episodes"]
        if args.checkpoint_interval is None
        else args.checkpoint_interval
    )
    output_root = args.output_root or defaults["output_root"]
    if int(episodes) <= 0 or int(episode_seconds) <= 0:
        raise ValueError("episodes and episode-seconds must be positive")
    if int(checkpoint_interval) <= 0:
        raise ValueError("checkpoint interval must be positive")
    return {
        "episodes": int(episodes),
        "episode_seconds": int(episode_seconds),
        "seed": int(seed),
        "checkpoint_interval": int(checkpoint_interval),
        "output_root": str(output_root),
    }


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_training_history(run_dir, rows):
    """Persist every completed episode; JSONL is the canonical append log."""

    jsonl_path = Path(run_dir) / "training_history.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = Path(run_dir) / "training_history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]) if rows else ())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    values = _resolved_values(args)
    method = args.method
    git_sha = _git_short_sha()
    run_dir = create_unique_run_directory(
        values["output_root"], method.method_key, values["seed"], git_sha
    )
    started_at = datetime.now(timezone.utc).isoformat()
    checkpoints_enabled = not args.smoke
    config = formal_training_config(
        values["episodes"],
        mode="smoke" if args.smoke else "train",
        episode_seconds=values["episode_seconds"],
        random_seed=values["seed"],
        warmup_joint_transitions=(
            0
            if args.smoke
            else FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
                "warmup_joint_transitions"
            ]
        ),
        batch_size=(
            1
            if args.smoke
            else FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]["batch_size"]
        ),
        replay_max_size=FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"][
            "replay_size"
        ],
        model_checkpoint_every=values["checkpoint_interval"],
        full_resume_every=values["checkpoint_interval"],
        checkpoint_root=str(run_dir / "checkpoints"),
        enable_model_checkpoints=checkpoints_enabled,
        enable_full_resume=checkpoints_enabled,
        enable_plots=False,
        enable_csv=False,
        run_directory=None,
    )
    manifest = generate_manifest(
        "train",
        manifest_seed=values["seed"],
        episode_count=values["episodes"],
        num_gt=args.roi_count,
    )
    manifest.save(run_dir / "scenario_manifest.json")
    resolved = {
        "status": "RUNNING",
        "method": method.method_key,
        "method_spec": method.to_dict(),
        "agent": method.agent,
        "reward_mode": method.reward_mode,
        "task_potential_enabled": method.task_potential_enabled,
        "seed": values["seed"],
        "episodes": values["episodes"],
        "num_uav": FORMAL_EXPERIMENT_DEFAULTS["num_uav"],
        "roi_count": (
            args.roi_count
            if args.roi_count is not None
            else [ROI_COUNT_MIN, ROI_COUNT_MAX]
        ),
        "checkpoint_interval": values["checkpoint_interval"],
        "formal_checkpoint_episode": FORMAL_EXPERIMENT_DEFAULTS[
            "formal_checkpoint_episode"
        ],
        "movement_hyperparameters": FORMAL_EXPERIMENT_DEFAULTS[
            "movement_hyperparameters"
        ],
        "training_config": asdict(config),
        "started_at": started_at,
        "git_sha": git_sha,
        "run_directory": str(run_dir),
    }
    _write_json(run_dir / "resolved_config.json", resolved)
    try:
        history = []

        def persist_episode(row):
            history.append(dict(row))
            _write_training_history(run_dir, history)

        result = train(
            config,
            scenario_manifest=manifest,
            method_spec=method,
            episode_observer=persist_episode,
        )
        resolved.update(
            status="COMPLETED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            history_rows=len(history),
            dinkelbach_update_count=result["dinkelbach_update_count"],
        )
        _write_json(run_dir / "run_metadata.json", {**result["run_metadata"], **resolved})
        _write_json(run_dir / "resolved_config.json", resolved)
    except BaseException as exc:
        resolved.update(
            status="FAILED",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_json(run_dir / "resolved_config.json", resolved)
        raise
    print(json.dumps({"run_directory": str(run_dir), "status": "COMPLETED"}))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
