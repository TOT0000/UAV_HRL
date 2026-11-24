import json
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from Packet_scheduler_v1 import MAX_PACKET_HOPS, PacketEngine
from run_experiment import main as run_experiment


def _legacy_drop_expired_packets(engine, current_time):
    """Pre-routing-index implementation retained only as an exact oracle."""

    violations = []
    queued_indices = set()
    for uav_id in range(engine.num_UAV):
        kept = deque()
        dropped_bits = 0.0
        for packet in engine.uav_queues[uav_id]:
            if packet is None:
                continue
            pool_idx = int(packet.get("_pool_idx", -1))
            if pool_idx >= 0:
                queued_indices.add(pool_idx)
            if packet.get("done", False):
                packet["_queued_uav"] = None
                dropped_bits += max(float(packet.get("rem_bits", 0.0)), 0.0)
                continue
            if int(packet.get("hops", 0)) >= MAX_PACKET_HOPS:
                dropped_bits += max(float(packet.get("rem_bits", 0.0)), 0.0)
                event = engine._mark_deadline_violation(
                    packet,
                    current_time,
                    sender=uav_id,
                    reason="max_hops",
                    remove_from_queue=False,
                )
                packet["_queued_uav"] = None
                if event is not None:
                    violations.append(event)
            else:
                kept.append(packet)
        engine.uav_queues[uav_id] = kept
        engine._decrease_backlog(uav_id, dropped_bits)

    detached_indices = set(engine._active_idx).difference(queued_indices)
    for pool_idx in sorted(detached_indices):
        if not (0 <= pool_idx < len(engine.packet_pool)):
            engine._active_idx.discard(pool_idx)
            continue
        packet = engine.packet_pool[pool_idx]
        if packet is not None and int(packet.get("hops", 0)) >= MAX_PACKET_HOPS:
            if not bool(packet.get("routing_eligible", False)):
                raise AssertionError("detached non-eligible packet reached max hops")
            event = engine._mark_deadline_violation(
                packet,
                current_time,
                reason="max_hops",
                remove_from_queue=False,
            )
            if event is not None:
                violations.append(event)
    return violations


class DeadlineIndexTrainingEquivalenceTest(unittest.TestCase):
    def _normalize_run_paths(self, value, run_directory):
        run_directory = str(Path(run_directory).resolve())
        if isinstance(value, str):
            return value.replace(run_directory, "<RUN_DIRECTORY>")
        if isinstance(value, dict):
            return {
                key: self._normalize_run_paths(item, run_directory)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._normalize_run_paths(item, run_directory)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._normalize_run_paths(item, run_directory)
                for item in value
            )
        return value

    def assertNestedEqual(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assertNestedEqual(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for first_item, second_item in zip(first, second):
                self.assertNestedEqual(first_item, second_item)
        else:
            self.assertEqual(first, second)

    def _run(
        self,
        root,
        *,
        index_enabled,
        episode_seconds,
        resume=False,
        drop_expired_impl=None,
    ):
        command = [
            "td3_dinkelbach",
            "--episodes",
            "1",
            "--episode-seconds",
            str(episode_seconds),
            "--checkpoint-interval",
            "1",
            "--roi-count",
            "8",
            "--seed",
            "20260817",
            "--output-root",
            str(root),
            "--routing-training-diagnostics",
        ]
        drop_context = (
            mock.patch.object(
                PacketEngine,
                "drop_expired_packets",
                drop_expired_impl,
            )
            if drop_expired_impl is not None
            else nullcontext()
        )
        with mock.patch.object(
            PacketEngine, "_DEADLINE_INDEX_ENABLED", bool(index_enabled)
        ), drop_context:
            self.assertEqual(run_experiment(command), 0)
            run_dir = next((Path(root) / "td3_dinkelbach").iterdir())
            if resume:
                self.assertEqual(
                    run_experiment(
                        ["resume", str(run_dir), "--target-episodes", "2"]
                    ),
                    0,
                )
        return run_dir

    def _assert_checkpoint_equal(self, reference, indexed, episode):
        reference_checkpoint = (
            reference / "checkpoints" / "full" / f"ep_{episode:04d}"
        )
        indexed_checkpoint = (
            indexed / "checkpoints" / "full" / f"ep_{episode:04d}"
        )
        for replay_name in ("joint_replay.npz", "routing_replay.npz"):
            with np.load(reference_checkpoint / replay_name) as first:
                with np.load(indexed_checkpoint / replay_name) as second:
                    self.assertEqual(set(first.files), set(second.files))
                    for field in first.files:
                        np.testing.assert_array_equal(first[field], second[field])

        reference_state = torch.load(
            reference_checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        indexed_state = torch.load(
            indexed_checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertNestedEqual(
            self._normalize_run_paths(reference_state, reference),
            self._normalize_run_paths(indexed_state, indexed),
        )
        self.assertFalse(
            any(
                "deadline" in key
                for key in indexed_state["training_state"][
                    "packet_engine_state"
                ]
            )
        )

        reference_models = torch.load(
            reference
            / "checkpoints"
            / "models"
            / f"ep_{episode:04d}"
            / "models.pt",
            map_location="cpu",
            weights_only=False,
        )
        indexed_models = torch.load(
            indexed
            / "checkpoints"
            / "models"
            / f"ep_{episode:04d}"
            / "models.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertNestedEqual(
            self._normalize_run_paths(reference_models, reference),
            self._normalize_run_paths(indexed_models, indexed),
        )

    def _assert_jsonl_equal(self, reference, indexed):
        for filename in (
            "training_history.jsonl",
            "routing_training_diagnostics.jsonl",
        ):
            self.assertEqual(
                (reference / filename).read_text(encoding="utf-8"),
                (indexed / filename).read_text(encoding="utf-8"),
            )

    def test_exact_training_state_after_formal_routing_warmup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = self._run(
                root / "reference",
                index_enabled=False,
                episode_seconds=16,
            )
            indexed = self._run(
                root / "indexed",
                index_enabled=True,
                episode_seconds=16,
            )
            self._assert_jsonl_equal(reference, indexed)
            self._assert_checkpoint_equal(reference, indexed, 1)

            row = json.loads(
                (indexed / "routing_training_diagnostics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(row["decision_count"], 1_024)
            self.assertEqual(row["routing_warmup_transition_threshold"], 1_000)
            self.assertTrue(row["routing_optimizer_warmup_complete"])
            self.assertGreater(row["routing_reward_optimizer_updates"], 0)
            self.assertGreater(row["routing_cost_optimizer_updates"], 0)

    def test_checkpoint_resume_remains_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = self._run(
                root / "reference",
                index_enabled=False,
                episode_seconds=2,
                resume=True,
            )
            indexed = self._run(
                root / "indexed",
                index_enabled=True,
                episode_seconds=2,
                resume=True,
            )
            self._assert_jsonl_equal(reference, indexed)
            self._assert_checkpoint_equal(reference, indexed, 2)

            rows = [
                json.loads(line)
                for line in (indexed / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["episode"] for row in rows], [1, 2])

    def test_routing_active_index_preserves_training_and_replay_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = self._run(
                root / "legacy",
                index_enabled=True,
                episode_seconds=2,
                drop_expired_impl=_legacy_drop_expired_packets,
            )
            optimized = self._run(
                root / "optimized",
                index_enabled=True,
                episode_seconds=2,
            )

            self._assert_jsonl_equal(legacy, optimized)
            self._assert_checkpoint_equal(legacy, optimized, 1)
            row = json.loads(
                (optimized / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertGreater(row["system_eligible_packets"], 0)
            self.assertGreater(row["routing_completed_uav_hop_count"], 0)


if __name__ == "__main__":
    unittest.main()
