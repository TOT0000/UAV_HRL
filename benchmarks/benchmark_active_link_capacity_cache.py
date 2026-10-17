"""Short deterministic benchmark for event-time link-capacity allocation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


PROCESS_STARTED = time.perf_counter()
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-seconds", type=int, default=5)
    parser.add_argument("--roi-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    import HRL_task_aware
    import Simulator as simulator_module
    from Packet_scheduler_v1 import PacketEngine
    from run_experiment import main as run_experiment
    from Simulator import Simulator

    import_seconds = time.perf_counter() - PROCESS_STARTED
    if args.disable_cache:
        if not hasattr(Simulator, "_ACTIVE_LINK_CAPACITY_CACHE_ENABLED"):
            raise RuntimeError("this revision does not expose the cache test hook")
        Simulator._ACTIVE_LINK_CAPACITY_CACHE_ENABLED = False

    counters = {
        "allocation_calls": 0,
        "profile_requests": 0,
        "profile_computations": 0,
    }
    timings = {
        "allocation_seconds": 0.0,
        "profile_computation_seconds": 0.0,
        "transmission_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }

    if args.profile:
        original_profile = simulator_module.block_capacity_profile_mbps
        original_allocate = Simulator.allocate_active_link_capacities
        original_serve = PacketEngine.serve_event_driven_slot
        original_model_checkpoint = HRL_task_aware.save_model_checkpoint
        original_full_checkpoint = HRL_task_aware.save_full_resume_checkpoint

        def counted_profile(*profile_args, **profile_kwargs):
            started = time.perf_counter()
            try:
                return original_profile(*profile_args, **profile_kwargs)
            finally:
                counters["profile_computations"] += 1
                timings["profile_computation_seconds"] += (
                    time.perf_counter() - started
                )

        def counted_allocate(environment, *allocate_args, **allocate_kwargs):
            started = time.perf_counter()
            try:
                result = original_allocate(
                    environment, *allocate_args, **allocate_kwargs
                )
                counters["allocation_calls"] += 1
                counters["profile_requests"] += (
                    len(environment.active_link_capacity_profiles_mbps)
                    + len(environment.active_s2u_capacity_profiles_mbps)
                )
                return result
            finally:
                timings["allocation_seconds"] += time.perf_counter() - started

        def counted_serve(engine, *serve_args, **serve_kwargs):
            started = time.perf_counter()
            try:
                return original_serve(engine, *serve_args, **serve_kwargs)
            finally:
                timings["transmission_seconds"] += time.perf_counter() - started

        def timed_model_checkpoint(*checkpoint_args, **checkpoint_kwargs):
            started = time.perf_counter()
            try:
                return original_model_checkpoint(
                    *checkpoint_args, **checkpoint_kwargs
                )
            finally:
                timings["checkpoint_seconds"] += time.perf_counter() - started

        def timed_full_checkpoint(*checkpoint_args, **checkpoint_kwargs):
            started = time.perf_counter()
            try:
                return original_full_checkpoint(
                    *checkpoint_args, **checkpoint_kwargs
                )
            finally:
                timings["checkpoint_seconds"] += time.perf_counter() - started

        simulator_module.block_capacity_profile_mbps = counted_profile
        Simulator.allocate_active_link_capacities = counted_allocate
        PacketEngine.serve_event_driven_slot = counted_serve
        HRL_task_aware.save_model_checkpoint = timed_model_checkpoint
        HRL_task_aware.save_full_resume_checkpoint = timed_full_checkpoint

    with tempfile.TemporaryDirectory() as temp_dir:
        output_root = Path(temp_dir) / "results"
        command = [
            "td3_dinkelbach",
            "--episodes",
            "1",
            "--episode-seconds",
            str(args.episode_seconds),
            "--checkpoint-interval",
            "1",
            "--roi-count",
            str(args.roi_count),
            "--seed",
            str(args.seed),
            "--output-root",
            str(output_root),
            "--routing-training-diagnostics",
        ]
        started = time.perf_counter()
        exit_code = run_experiment(command)
        overall_seconds = time.perf_counter() - started
        if exit_code != 0:
            raise RuntimeError(
                f"benchmark training failed with exit code {exit_code}"
            )

    result = {
        "episode_seconds": int(args.episode_seconds),
        "roi_count": int(args.roi_count),
        "seed": int(args.seed),
        "cache_enabled": not bool(args.disable_cache),
        "profiled": bool(args.profile),
        "import_seconds": float(import_seconds),
        "overall_seconds": float(overall_seconds),
    }
    if args.profile:
        cache_hits = (
            counters["profile_requests"] - counters["profile_computations"]
        )
        result.update(
            counters,
            cache_hits=int(cache_hits),
            cache_misses=int(counters["profile_computations"]),
            **timings,
            training_excluding_checkpoint_seconds=float(
                overall_seconds - timings["checkpoint_seconds"]
            ),
        )
    print("CAPACITY_CACHE_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
