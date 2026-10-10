"""CLI for multi-candidate actual-training LLM reward search."""

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
from llm_training_search import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNTIME_ROOT,
    run_training_search,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four candidates one at a time, then independently train and "
            "evaluate every valid nonduplicate candidate"
        )
    )
    parser.add_argument(
        "--validation-source",
        action="append",
        dest="validation_sources",
        help="complete-episode sample source used only for executable validation",
    )
    parser.add_argument("--baseline-run")
    parser.add_argument("--baseline-evaluation")
    parser.add_argument("--provider", choices=("lmstudio", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="model sampling seed; candidate training seed is fixed at 20260817",
    )
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_API_TIMEOUT_SECONDS)
    parser.add_argument(
        "--connect-timeout", type=float, default=DEFAULT_API_CONNECT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--total-timeout", type=float, default=DEFAULT_API_TOTAL_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    )
    parser.add_argument("--worker-timeout", type=float, default=DEFAULT_WORKER_TIMEOUT_SECONDS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="new-run default is 5; on resume only the same or a larger value is allowed",
    )
    parser.add_argument(
        "--max-repairs-per-candidate",
        type=int,
        default=None,
        help="new-run default is 5 corrections after the first response",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--runtime-root",
        default=str(DEFAULT_RUNTIME_ROOT),
        help="short persistent root for deep training checkpoint and evaluation paths",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is None:
        missing = [
            option
            for option, value in (
                ("--validation-source", args.validation_sources),
                ("--provider", args.provider),
                ("--model", args.model),
            )
            if not value
        ]
        if not args.dry_run:
            missing.extend(
                option
                for option, value in (
                    ("--baseline-run", args.baseline_run),
                    ("--baseline-evaluation", args.baseline_evaluation),
                )
                if not value
            )
        if missing:
            print("Missing required arguments: " + ", ".join(missing), file=sys.stderr)
            return 2
    try:
        result = run_training_search(**vars(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            redact_provider_secrets(
                f"LLM training search failed: {type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 2
    print(f"LLM training search status: {result['status']}")
    print(f"Output: {result['output_directory']}")
    print(f"Round: {result['search_round']}; model calls: {result['model_calls']}")
    best = result.get("best_candidate")
    if best:
        print(
            "Historical best: "
            f"{best['candidate_id']} "
            f"mean_EE={best['mean_episode_energy_efficiency_mbit_per_j']:.9g}"
        )
    if result.get("stop_reason"):
        print(f"Stop reason: {result['stop_reason']}")
        print(
            "Resume with: python run_llm_training_search.py --resume "
            f"\"{result['output_directory']}\""
        )
    return 0 if result["status"] in {"complete", "dry_run_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
