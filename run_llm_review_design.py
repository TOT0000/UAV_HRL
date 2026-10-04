"""CLI for proposer/reviewer offline LLM feature design."""

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
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    redact_provider_secrets,
)
from llm_review_design import (
    DEFAULT_MAX_CODE_REPAIRS,
    DEFAULT_MAX_REVIEW_ROUNDS,
    DEFAULT_REVIEW_OUTPUT_ROOT,
    run_review_design,
)


def _role_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    prefix = f"--{role}-"
    parser.add_argument(prefix + "provider", choices=("lmstudio", "openai"))
    parser.add_argument(prefix + "model")
    parser.add_argument(prefix + "base-url")
    parser.add_argument(
        prefix + "context-length", type=int, default=DEFAULT_CONTEXT_LENGTH
    )
    parser.add_argument(
        prefix + "max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument(
        prefix + "temperature", type=float, default=DEFAULT_TEMPERATURE
    )
    parser.add_argument(prefix + "seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        prefix + "reasoning-effort",
        choices=("low", "medium", "high"),
        help="optional provider reasoning setting when the selected model supports it",
    )
    parser.add_argument(
        prefix + "timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT_SECONDS,
        help="stream idle timeout for each request by this role",
    )
    parser.add_argument(
        prefix + "connect-timeout",
        type=float,
        default=DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        help="TCP/TLS connection timeout for each request by this role",
    )
    parser.add_argument(
        prefix + "total-timeout",
        type=float,
        default=DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        help="total timeout for each request by this role",
    )
    parser.add_argument(
        prefix + "progress-interval",
        type=float,
        default=DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate features with one model, validate them on all fixed samples, "
            "and obtain an independent review from another model."
        )
    )
    parser.add_argument("--fixed-sample")
    _role_arguments(parser, "proposer")
    _role_arguments(parser, "reviewer")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument(
        "--worker-timeout", type=float, default=DEFAULT_WORKER_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=None,
        help=f"completed valid reviews allowed (new-run default: {DEFAULT_MAX_REVIEW_ROUNDS})",
    )
    parser.add_argument(
        "--max-code-repairs",
        type=int,
        default=None,
        help=(
            "repairs after the initial proposer output in each proposal cycle "
            f"(new-run default: {DEFAULT_MAX_CODE_REPAIRS})"
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_REVIEW_OUTPUT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        help=(
            "resume a compatible review-design run; saved models and fixed samples "
            "are reused, and only explicitly higher limits are accepted"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is None:
        missing = [
            name
            for name in (
                "fixed_sample",
                "proposer_provider",
                "proposer_model",
                "reviewer_provider",
                "reviewer_model",
            )
            if getattr(args, name) in (None, "")
        ]
        if missing:
            print(
                "New review-design run is missing required arguments: "
                + ", ".join("--" + item.replace("_", "-") for item in missing),
                file=sys.stderr,
            )
            return 2
    values = vars(args).copy()
    try:
        result = run_review_design(**values)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            redact_provider_secrets(
                f"LLM review design failed: {type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 2
    print(f"LLM review design status: {result['status']}")
    print(f"Output: {result['output_directory']}")
    print(
        "Progress: "
        f"proposer_calls={result['proposer_calls']}, "
        f"reviewer_calls={result['reviewer_calls']}, "
        f"completed_reviews={result['review_rounds_completed']}, "
        f"repairs_in_current_cycle={result['code_repairs_used_in_cycle']}"
    )
    if result.get("approved_artifact"):
        print(f"Approved artifact: {result['approved_artifact']}")
    elif result.get("stop_reason"):
        print(f"Stop reason: {result['stop_reason']}")
    return 0 if result["status"] in {"approved", "dry_run_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
