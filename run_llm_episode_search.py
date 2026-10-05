"""CLI for complete-episode, four-candidate LLM feature search."""

from __future__ import annotations

import argparse
import sys

from llm_design import (
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    redact_provider_secrets,
)
from llm_episode_search import (
    DEFAULT_OUTPUT_ROOT,
    run_episode_search,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate four candidates, screen on complete episodes, then train/evaluate one"
    )
    parser.add_argument("--episode-source", action="append", dest="episode_sources")
    parser.add_argument("--provider", choices=("lmstudio", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_API_TIMEOUT_SECONDS)
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_API_CONNECT_TIMEOUT_SECONDS)
    parser.add_argument("--total-timeout", type=float, default=DEFAULT_API_TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--progress-interval", type=float, default=DEFAULT_API_PROGRESS_INTERVAL_SECONDS)
    parser.add_argument("--worker-timeout", type=float, default=DEFAULT_WORKER_TIMEOUT_SECONDS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    lambda_group = parser.add_mutually_exclusive_group()
    lambda_group.add_argument("--evaluation-lambda", type=float)
    lambda_group.add_argument("--lambda-training-run")
    parser.add_argument(
        "--max-search-rounds",
        type=int,
        default=None,
        help="new-run default is 5; on resume, omit to retain the saved limit",
    )
    parser.add_argument(
        "--max-repairs-per-round",
        type=int,
        default=None,
        help="new-run default is 5; on resume, an explicit larger value adds budget",
    )
    parser.add_argument("--ee-tolerance", type=float, default=1e-12)
    parser.add_argument("--reward-tolerance", type=float, default=1e-12)
    parser.add_argument("--baseline-run")
    parser.add_argument("--baseline-evaluation")
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument(
        "--revalidate-only",
        action="store_true",
        help=(
            "revalidate saved generation-stage candidates under the current numeric "
            "operation rules and stop before model calls, pre-evaluation, or training"
        ),
    )
    parser.add_argument(
        "--recover-training-initialization",
        action="store_true",
        help=(
            "perform the bounded ef10fa3 train-stage compatibility migration, "
            "preserve saved selection, and continue through the short training root"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.revalidate_only and args.resume is None:
        print("--revalidate-only requires --resume", file=sys.stderr)
        return 2
    if args.recover_training_initialization and args.resume is None:
        print("--recover-training-initialization requires --resume", file=sys.stderr)
        return 2
    if args.revalidate_only and args.recover_training_initialization:
        print(
            "--revalidate-only and --recover-training-initialization are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if args.resume is None:
        missing = [
            option for option, value in (
                ("--episode-source", args.episode_sources),
                ("--provider", args.provider),
                ("--model", args.model),
            ) if not value
        ]
        if args.evaluation_lambda is None and args.lambda_training_run is None:
            missing.append("--evaluation-lambda or --lambda-training-run")
        if not args.dry_run:
            missing.extend(
                option for option, value in (
                    ("--baseline-run", args.baseline_run),
                    ("--baseline-evaluation", args.baseline_evaluation),
                    ("--evaluation-manifest", args.evaluation_manifest),
                ) if not value
            )
        if missing:
            print("Missing required arguments: " + ", ".join(missing), file=sys.stderr)
            return 2
    try:
        result = run_episode_search(**vars(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(redact_provider_secrets(f"LLM episode search failed: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 2
    print(f"LLM episode search status: {result['status']}")
    print(f"Output: {result['output_directory']}")
    print(f"Round: {result['search_round']}; model calls: {result['model_calls']}")
    if result.get("best_trained_candidate"):
        print(f"Best candidate: {result['best_trained_candidate']['candidate_id']}")
    if result.get("stop_reason"):
        print(f"Stop reason: {result['stop_reason']}")
    if args.revalidate_only:
        return 0
    return 0 if result["status"] in {"complete", "dry_run_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
