"""Unified CLI for manifest-driven comparison experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from experiment_config import FORMAL_EXPERIMENT_DEFAULTS, MethodSpec
from HRL_task_aware import formal_training_config, smoke_training_config, train
from scenario_manifest import ScenarioManifest, generate_manifest


DEFAULT_OUTPUT_DIR = Path("runs") / "comparison"


def _method(value):
    try:
        return MethodSpec.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_manifest(path, expected_split=None):
    if path is None:
        raise ValueError("this command requires --manifest")
    manifest = ScenarioManifest.load(path)
    if expected_split is not None and manifest.split != expected_split:
        raise ValueError(
            f"manifest split mismatch: manifest={manifest.split}, "
            f"requested={expected_split}"
        )
    return manifest


def _write_run_metadata(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    path.write_text(
        json.dumps(result["run_metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def command_generate_manifest(args):
    manifest = generate_manifest(
        args.split, args.manifest_seed, args.episodes
    )
    path = Path(args.manifest) if args.manifest else (
        Path(args.output_dir) / "manifests" / f"{args.split}.json"
    )
    manifest.save(path)
    print(
        json.dumps(
            {"manifest": str(path), "content_hash": manifest.content_hash},
            ensure_ascii=False,
        )
    )
    return 0


def command_smoke(args):
    method = args.method
    manifest = (
        _load_manifest(args.manifest, expected_split=args.split)
        if args.manifest
        else generate_manifest(
            args.split, manifest_seed=20260817, episode_count=args.episodes
        )
    )
    config = replace(
        smoke_training_config(),
        total_episodes=args.episodes,
        random_seed=args.training_seed,
    )
    result = train(config, scenario_manifest=manifest, method_spec=method)
    _write_run_metadata(args.output_dir, result)
    print(json.dumps(result["run_metadata"], ensure_ascii=False))
    return 0


def command_train(args):
    method = args.method
    manifest = _load_manifest(args.manifest, expected_split=args.split)
    config = formal_training_config(
        args.episodes,
        random_seed=args.training_seed,
        resume_dir=args.resume,
        checkpoint_root=str(
            Path(args.checkpoint)
            if args.checkpoint
            else Path(args.output_dir) / "checkpoints"
        ),
        enable_plots=False,
        enable_csv=False,
    )
    result = train(
        config, scenario_manifest=manifest, method_spec=method
    )
    _write_run_metadata(args.output_dir, result)
    return 0


def command_evaluate(args):
    from evaluation_metrics import run_evaluation_command

    return run_evaluation_command(args)


def command_aggregate(args):
    from evaluation_metrics import run_aggregate_command

    return run_aggregate_command(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manifest-driven corrected UAV-HRL comparison runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-manifest")
    generate.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    generate.add_argument("--manifest")
    generate.add_argument("--manifest-seed", type=int, required=True)
    generate.add_argument("--episodes", type=int, required=True)
    generate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    generate.set_defaults(handler=command_generate_manifest)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--method", type=_method, default=MethodSpec())
    smoke.add_argument("--split", default="test", choices=("train", "validation", "test"))
    smoke.add_argument("--manifest")
    smoke.add_argument("--training-seed", type=int, default=20260817)
    smoke.add_argument("--episodes", type=int, default=1)
    smoke.add_argument("--checkpoint")
    smoke.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "smoke"))
    smoke.add_argument("--resume")
    smoke.set_defaults(handler=command_smoke)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--method", type=_method, default=MethodSpec())
    train_parser.add_argument("--split", default="train", choices=("train",))
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--training-seed", type=int, required=True)
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS["training_episodes_per_seed"],
    )
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "train"))
    train_parser.add_argument("--resume")
    train_parser.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--method", type=_method, default=MethodSpec())
    evaluate.add_argument("--split", default="test", choices=("validation", "test"))
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--training-seed", type=int, required=True)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "evaluate"))
    evaluate.add_argument("--resume")
    evaluate.set_defaults(handler=command_evaluate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--method", type=_method, default=MethodSpec())
    aggregate.add_argument("--split", default="test", choices=("validation", "test"))
    aggregate.add_argument("--manifest")
    aggregate.add_argument("--training-seed", type=int)
    aggregate.add_argument("--episodes", type=int)
    aggregate.add_argument("--checkpoint")
    aggregate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "aggregate"))
    aggregate.add_argument("--resume")
    aggregate.add_argument("--input-dir", required=True)
    aggregate.add_argument(
        "--expected-seed-count",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS["training_seed_count"],
    )
    aggregate.add_argument(
        "--expected-episodes-per-seed",
        type=int,
        default=FORMAL_EXPERIMENT_DEFAULTS[
            "evaluation_episodes_per_trained_seed"
        ],
    )
    aggregate.set_defaults(handler=command_aggregate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
