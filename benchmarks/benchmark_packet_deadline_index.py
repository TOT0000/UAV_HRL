"""Deterministic short benchmark for event-time deadline handling."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--disable-event-service-cache", action="store_true")
    parser.add_argument("--disable-geometry-cache", action="store_true")
    parser.add_argument("--routing-diagnostics", choices=("on", "off"), default="on")
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = _parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.cpu_threads)

    import HRL_task_aware
    import torch
    from DDQN import DDQN
    from Packet_scheduler_v1 import PacketEngine
    from Simulator import Simulator
    from run_experiment import main as run_experiment
    from td3 import TD3

    import_seconds = time.perf_counter() - PROCESS_STARTED
    PacketEngine._DEADLINE_INDEX_ENABLED = not bool(args.disable_index)
    PacketEngine._EVENT_SERVICE_CACHE_ENABLED = not bool(
        args.disable_event_service_cache
    )
    Simulator._ROUTING_GEOMETRY_CACHE_ENABLED = not bool(
        args.disable_geometry_cache
    )
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(int(args.cpu_threads))

    engines = []
    simulators = {}
    run_started = [None]
    timings = {
        "expiration_check_seconds": 0.0,
        "nearest_deadline_query_seconds": 0.0,
        "full_scan_seconds": 0.0,
        "transmission_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "allocation_seconds": 0.0,
        "routing_update_seconds": 0.0,
        "movement_update_seconds": 0.0,
        "initialization_until_first_transmission_seconds": None,
    }
    profile_counts = {
        "allocation_calls": 0,
        "routing_update_calls": 0,
        "movement_update_calls": 0,
    }

    if args.profile:
        original_init = PacketEngine.__init__
        original_expire = PacketEngine.expire_packets
        original_next = PacketEngine.next_packet_deadline
        original_full_scan = PacketEngine._expire_packets_full_scan
        original_serve = PacketEngine.serve_event_driven_slot
        original_model_checkpoint = HRL_task_aware.save_model_checkpoint
        original_full_checkpoint = HRL_task_aware.save_full_resume_checkpoint
        original_allocate = Simulator.allocate_active_link_capacities
        original_routing_update = DDQN.train
        original_movement_update = TD3.update_joint

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
            if timings["initialization_until_first_transmission_seconds"] is None:
                timings["initialization_until_first_transmission_seconds"] = (
                    started - run_started[0]
                )
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

        def timed_allocate(simulator, *allocate_args, **allocate_kwargs):
            simulators[id(simulator)] = simulator
            profile_counts["allocation_calls"] += 1
            started = time.perf_counter()
            try:
                return original_allocate(simulator, *allocate_args, **allocate_kwargs)
            finally:
                timings["allocation_seconds"] += time.perf_counter() - started

        def timed_routing_update(agent, *update_args, **update_kwargs):
            profile_counts["routing_update_calls"] += 1
            started = time.perf_counter()
            try:
                return original_routing_update(agent, *update_args, **update_kwargs)
            finally:
                timings["routing_update_seconds"] += time.perf_counter() - started

        def timed_movement_update(agent, *update_args, **update_kwargs):
            profile_counts["movement_update_calls"] += 1
            started = time.perf_counter()
            try:
                return original_movement_update(agent, *update_args, **update_kwargs)
            finally:
                timings["movement_update_seconds"] += time.perf_counter() - started

        PacketEngine.__init__ = tracked_init
        PacketEngine.expire_packets = timed_expire
        PacketEngine.next_packet_deadline = timed_next
        PacketEngine._expire_packets_full_scan = timed_full_scan
        PacketEngine.serve_event_driven_slot = timed_serve
        HRL_task_aware.save_model_checkpoint = timed_model_checkpoint
        HRL_task_aware.save_full_resume_checkpoint = timed_full_checkpoint
        Simulator.allocate_active_link_capacities = timed_allocate
        DDQN.train = timed_routing_update
        TD3.update_joint = timed_movement_update

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
        ]
        if args.routing_diagnostics == "on":
            command.append("--routing-training-diagnostics")
        started = time.perf_counter()
        run_started[0] = started
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
        "event_service_cache_enabled": not bool(
            args.disable_event_service_cache
        ),
        "geometry_cache_enabled": not bool(args.disable_geometry_cache),
        "routing_diagnostics": args.routing_diagnostics,
        "cpu_threads": int(args.cpu_threads),
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
        event_counters = {
            "active_link_queries": 0,
            "active_link_full_scans": 0,
            "active_link_cache_hits": 0,
            "active_link_incremental_updates": 0,
            "active_link_nodes_rechecked": 0,
            "allocation_rebuilds": 0,
            "resource_bandwidth_serializations": 0,
        }
        for engine in engines:
            stats = engine.event_service_cache_statistics()
            for name in event_counters:
                event_counters[name] += int(stats[name])
        geometry_counters = {
            "geometry_cache_builds": 0,
            "geometry_cache_hits": 0,
        }
        for simulator in simulators.values():
            stats = simulator.routing_geometry_cache_statistics()
            geometry_counters["geometry_cache_builds"] += int(stats["builds"])
            geometry_counters["geometry_cache_hits"] += int(stats["hits"])
        counters["queue_entries_visited"] = (
            counters["uav_queue_entries_visited"]
            + counters["sr_queue_entries_visited"]
            + counters["detached_entries_visited"]
        )
        result.update(
            counters,
            **event_counters,
            **geometry_counters,
            **profile_counts,
            **timings,
            training_excluding_checkpoint_seconds=float(
                overall_seconds - timings["checkpoint_seconds"]
            ),
            other_initialization_and_loop_seconds=float(
                overall_seconds
                - timings["transmission_seconds"]
                - timings["routing_update_seconds"]
                - timings["movement_update_seconds"]
                - timings["checkpoint_seconds"]
            ),
        )
    print("DEADLINE_INDEX_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
