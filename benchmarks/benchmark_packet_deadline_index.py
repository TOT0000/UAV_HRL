"""Deterministic short benchmark for event-time deadline handling."""

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
    parser.add_argument("--disable-index", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    import HRL_task_aware
    from Packet_scheduler_v1 import PacketEngine
    from run_experiment import main as run_experiment

    import_seconds = time.perf_counter() - PROCESS_STARTED
    PacketEngine._DEADLINE_INDEX_ENABLED = not bool(args.disable_index)

    engines = []
    timings = {
        "expiration_check_seconds": 0.0,
        "nearest_deadline_query_seconds": 0.0,
        "full_scan_seconds": 0.0,
        "transmission_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }

    if args.profile:
        original_init = PacketEngine.__init__
        original_expire = PacketEngine.expire_packets
        original_next = PacketEngine.next_packet_deadline
        original_full_scan = PacketEngine._expire_packets_full_scan
        original_serve = PacketEngine.serve_event_driven_slot
        original_model_checkpoint = HRL_task_aware.save_model_checkpoint
        original_full_checkpoint = HRL_task_aware.save_full_resume_checkpoint

        def tracked_init(engine, *init_args, **init_kwargs):
            original_init(engine, *init_args, **init_kwargs)
            engines.append(engine)

        def timed_expire(engine, *expire_args, **expire_kwargs):
            started = time.perf_counter()
            try:
                return original_expire(engine, *expire_args, **expire_kwargs)
            finally:
                timings["expiration_check_seconds"] += (
                    time.perf_counter() - started
                )

        def timed_next(engine, *next_args, **next_kwargs):
            started = time.perf_counter()
            try:
                return original_next(engine, *next_args, **next_kwargs)
            finally:
                timings["nearest_deadline_query_seconds"] += (
                    time.perf_counter() - started
                )

        def timed_full_scan(engine, *scan_args, **scan_kwargs):
            started = time.perf_counter()
            try:
                return original_full_scan(engine, *scan_args, **scan_kwargs)
            finally:
                timings["full_scan_seconds"] += time.perf_counter() - started

        def timed_serve(engine, *serve_args, **serve_kwargs):
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

        PacketEngine.__init__ = tracked_init
        PacketEngine.expire_packets = timed_expire
        PacketEngine.next_packet_deadline = timed_next
        PacketEngine._expire_packets_full_scan = timed_full_scan
        PacketEngine.serve_event_driven_slot = timed_serve
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
        "deadline_index_enabled": not bool(args.disable_index),
        "profiled": bool(args.profile),
        "import_seconds": float(import_seconds),
        "overall_seconds": float(overall_seconds),
    }
    if args.profile:
        counter_names = (
            "expiration_checks",
            "nearest_deadline_queries",
            "lightweight_heap_queries",
            "full_scans",
            "uav_queue_entries_visited",
            "sr_queue_entries_visited",
            "detached_entries_visited",
            "linear_active_packet_visits",
            "stale_heap_entries_removed",
            "heap_compactions",
            "index_rebuilds",
        )
        counters = {name: 0 for name in counter_names}
        for engine in engines:
            stats = engine.deadline_index_statistics()
            for name in counter_names:
                counters[name] += int(stats[name])
        counters["queue_entries_visited"] = (
            counters["uav_queue_entries_visited"]
            + counters["sr_queue_entries_visited"]
            + counters["detached_entries_visited"]
        )
        result.update(
            counters,
            **timings,
            training_excluding_checkpoint_seconds=float(
                overall_seconds - timings["checkpoint_seconds"]
            ),
        )
    print("DEADLINE_INDEX_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
