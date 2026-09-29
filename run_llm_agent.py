"""CLI for the optional smolagents offline feature-design agent."""

from __future__ import annotations

import argparse
import sys

from llm_design import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    redact_provider_secrets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the optional smolagents ToolCallingAgent against an existing "
            "fixed offline LLM baseline. This never starts training."
        )
    )
    parser.add_argument("--fixed-sample")
    parser.add_argument("--provider", choices=("lmstudio", "openai"))
    parser.add_argument("--base-url")
    parser.add_argument("--model", help="exact provider API identifier")
    parser.add_argument("--max-model-calls", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        help="optional GPT-OSS reasoning setting; rejected for other models",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT_SECONDS,
        help="stream idle timeout per model request (default: 600 seconds)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_API_CONNECT_TIMEOUT_SECONDS,
        help="TCP/TLS connection timeout per request (default: 30 seconds)",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=DEFAULT_API_TOTAL_TIMEOUT_SECONDS,
        help="total timeout per model request (default: 1800 seconds)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_API_PROGRESS_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--output-root", default="results/llm_agents")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        help=(
            "resume an existing agent run directory; fixed sample, provider, model, "
            "evaluation settings, and call budget are restored and compatibility-checked"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is None and not (args.fixed_sample and args.provider and args.model):
        print(
            "New runs require --fixed-sample, --provider, and --model; use only "
            "--resume RUN_DIR to continue a saved run.",
            file=sys.stderr,
        )
        return 2
    try:
        from llm_agent import run_agent

        result = run_agent(
            fixed_sample=args.fixed_sample,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            max_model_calls=args.max_model_calls,
            context_length=args.context_length,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
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
            resume=args.resume,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(
            redact_provider_secrets(
                f"LLM feature-design agent failed: {type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 2
    print(f"Agent status: {result['status']}")
    print(f"Output: {result['output_directory']}")
    print(f"Model calls: {result.get('model_calls_used', 0)}")
    print(f"Tool operations: {result.get('tool_operations', 0)}")
    if result.get("approved_artifact"):
        print(f"Approved artifact: {result['approved_artifact']}")
    if result.get("stop_reason"):
        print(f"Stop reason: {result['stop_reason']}")
    return 0 if result["status"] in {"approved", "dry_run_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
