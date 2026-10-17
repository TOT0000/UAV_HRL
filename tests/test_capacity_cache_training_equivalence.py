import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from packet_outcome_artifacts import PACKET_OUTCOME_MODE_BOUNDED
from Packet_scheduler_v1 import PacketEngine
from run_experiment import main as run_experiment
from scenario_manifest import generate_manifest
from Simulator import Simulator


class CapacityCacheTrainingEquivalenceTest(unittest.TestCase):
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

    def _run(self, *, cache_enabled):
        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=16,
            warmup_joint_transitions=10_000,
            routing_warmup_transitions=1_000,
            batch_size=256,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
            collect_packet_outcomes=True,
            packet_outcome_artifact_mode=PACKET_OUTCOME_MODE_BOUNDED,
            packet_outcome_collection_limit=1,
            routing_training_diagnostics=True,
        )
        manifest = generate_manifest("train", 20260817, 1, num_gt=8)
        with mock.patch.object(
            Simulator,
            "_ACTIVE_LINK_CAPACITY_CACHE_ENABLED",
            bool(cache_enabled),
        ), mock.patch.object(
            Simulator,
            "_ROUTING_GEOMETRY_CACHE_ENABLED",
            bool(cache_enabled),
        ), mock.patch.object(
            PacketEngine,
            "_EVENT_SERVICE_CACHE_ENABLED",
            bool(cache_enabled),
        ):
            return train(
                config,
                scenario_manifest=manifest,
                method_spec=MethodSpec.parse("td3_dinkelbach"),
            )

    def test_training_is_exact_after_formal_routing_warmup(self):
        uncached = self._run(cache_enabled=False)
        cached = self._run(cache_enabled=True)

        self.assertEqual(
            uncached["packet_outcome_artifacts"],
            cached["packet_outcome_artifacts"],
        )
        self.assertEqual(uncached["episode_metrics"], cached["episode_metrics"])
        self.assertEqual(
            uncached["training_history_rows"],
            cached["training_history_rows"],
        )
        for field in (
            "reward_log",
            "delivered_log",
            "energy_log",
            "lambda_used_log",
            "lambda_after_episode_log",
            "lambda_cost_used_log",
            "lambda_cost_after_episode_log",
            "routing_epsilon_log",
            "routing_replay_size",
            "joint_replay_size",
            "ddqn_training_updates",
            "routing_reward_optimizer_update_count",
            "routing_cost_optimizer_update_count",
            "routing_reward_target_update_count",
            "routing_cost_target_update_count",
        ):
            with self.subTest(field=field):
                self.assertEqual(uncached[field], cached[field])

        self.assertEqual(cached["routing_replay_size"], 1_024)
        self.assertGreater(cached["routing_reward_optimizer_update_count"], 0)
        self.assertGreater(cached["routing_cost_optimizer_update_count"], 0)
        self.assertGreater(
            cached["episode_metrics"][0]["packet_path_cost_transition_count"],
            0,
        )

    def _run_checkpoint_resume(self, root, *, cache_enabled):
        command = [
            "td3_dinkelbach",
            "--episodes",
            "1",
            "--episode-seconds",
            "2",
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
        with mock.patch.object(
            Simulator,
            "_ACTIVE_LINK_CAPACITY_CACHE_ENABLED",
            bool(cache_enabled),
        ), mock.patch.object(
            Simulator,
            "_ROUTING_GEOMETRY_CACHE_ENABLED",
            bool(cache_enabled),
        ), mock.patch.object(
            PacketEngine,
            "_EVENT_SERVICE_CACHE_ENABLED",
            bool(cache_enabled),
        ):
            self.assertEqual(run_experiment(command), 0)
            run_dir = next((Path(root) / "td3_dinkelbach").iterdir())
            self.assertEqual(
                run_experiment(
                    ["resume", str(run_dir), "--target-episodes", "2"]
                ),
                0,
            )
        return run_dir

    def test_checkpoint_resume_is_exact_against_uncached_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uncached = self._run_checkpoint_resume(
                root / "uncached", cache_enabled=False
            )
            cached = self._run_checkpoint_resume(
                root / "cached", cache_enabled=True
            )

            for filename in (
                "training_history.jsonl",
                "routing_training_diagnostics.jsonl",
            ):
                self.assertEqual(
                    (uncached / filename).read_text(encoding="utf-8"),
                    (cached / filename).read_text(encoding="utf-8"),
                )

            uncached_checkpoint = uncached / "checkpoints" / "full" / "ep_0002"
            cached_checkpoint = cached / "checkpoints" / "full" / "ep_0002"
            for replay_name in ("joint_replay.npz", "routing_replay.npz"):
                with np.load(uncached_checkpoint / replay_name) as first:
                    with np.load(cached_checkpoint / replay_name) as second:
                        self.assertEqual(set(first.files), set(second.files))
                        for field in first.files:
                            np.testing.assert_array_equal(
                                first[field], second[field]
                            )

            uncached_state = torch.load(
                uncached_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            cached_state = torch.load(
                cached_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertNestedEqual(
                self._normalize_run_paths(uncached_state, uncached),
                self._normalize_run_paths(cached_state, cached),
            )

            uncached_model_path = (
                uncached / "checkpoints" / "models" / "ep_0002" / "models.pt"
            )
            cached_model_path = (
                cached / "checkpoints" / "models" / "ep_0002" / "models.pt"
            )
            uncached_models = torch.load(
                uncached_model_path,
                map_location="cpu",
                weights_only=False,
            )
            cached_models = torch.load(
                cached_model_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertNestedEqual(
                self._normalize_run_paths(uncached_models, uncached),
                self._normalize_run_paths(cached_models, cached),
            )

            uncached_rows = [
                json.loads(line)
                for line in (uncached / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["episode"] for row in uncached_rows], [1, 2])


if __name__ == "__main__":
    unittest.main()
