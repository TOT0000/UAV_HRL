"""CLI for offline LM-provider state/reward design against fixed samples."""

from __future__ import annotations

import argparse
import sys

from llm_design import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    redact_provider_secrets,
    run_design,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate a current-only LLM state/reward design using "
            "one fixed offline baseline artifact."
        )
    )
    parser.add_argument("--fixed-sample", required=True)
    parser.add_argument(
        "--provider", choices=("lmstudio", "openai"), default="lmstudio"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "provider endpoint; defaults to http://127.0.0.1:1234/v1 for "
            "lmstudio and https://api.openai.com/v1 for openai"
        ),
    )
    parser.add_argument("--model", required=True, help="exact provider API identifier")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT_SECONDS,
        help=(
            "maximum idle seconds while reading one provider stream "
            "(default: 600); retained as the backward-compatible API timeout option"
        ),
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        help=(
            "maximum seconds for TCP/TLS connection establishment only (default: 30); "
            "response headers and SSE reads use --timeout and --total-timeout"
        ),
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        help="maximum total seconds for one generation request (default: 1800)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
        help="seconds between concise streaming progress messages (default: 10)",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="isolated candidate worker timeout in seconds (default: 120)",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_design(
            fixed_sample=args.fixed_sample,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            context_length=args.context_length,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            seed=args.seed,
            max_attempts=args.max_attempts,
            beta=args.beta,
            batch_size=args.batch_size,
            timeout=args.timeout,
            connect_timeout=args.connect_timeout,
            total_timeout=args.total_timeout,
            progress_interval=args.progress_interval,
            worker_timeout=args.worker_timeout,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            output_root=args.output_root,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            redact_provider_secrets(
                f"LLM design failed: {type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 2
    print(f"LLM design status: {result['status']}")
    print(f"Output: {result['output_directory']}")
    metadata = result.get("metadata", {})
    generation = metadata.get("generation", {})
    context = metadata.get("context", {})
    budget = metadata.get("first_prompt_token_budget", {})
    model_metadata = metadata.get("model", {})
    print(f"Model: {model_metadata.get('requested_api_identifier', args.model)}")
    print(f"Provider: {model_metadata.get('provider', args.provider)}")
    print(
        "Generation: "
        f"temperature={generation.get('temperature', args.temperature)}, "
        f"max_output_tokens={generation.get('max_output_tokens', args.max_output_tokens)}, "
        f"seed={generation.get('seed_requested', args.seed)}, "
        f"max_attempts={generation.get('max_attempts', args.max_attempts)}"
    )
    print(
        "Context budget: "
        f"effective={context.get('effective_budget', args.context_length)}, "
        f"prompt_estimate={budget.get('estimated_prompt_tokens_lower')}.."
        f"{budget.get('estimated_prompt_tokens_upper')}, "
        f"reserved_output={budget.get('reserved_output_tokens', args.max_output_tokens)}, "
        f"estimated_total_upper={budget.get('estimated_total_upper')}, "
        f"fits={budget.get('fits_client_budget')}"
    )
    print(
        "Timeouts: "
        f"connect={generation.get('api_connect_timeout_seconds', args.connect_timeout)}, "
        f"stream_idle={generation.get('api_stream_idle_timeout_seconds', args.timeout)}, "
        f"request_total={generation.get('api_total_timeout_seconds_per_request', args.total_timeout)}, "
        f"worker={generation.get('worker_timeout_seconds', args.worker_timeout)}; "
        "limits apply independently per request"
    )
    if result.get("approved_artifact"):
        print(f"Approved artifact: {result['approved_artifact']}")
    return 0 if result["status"] in {"approved", "dry_run_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
