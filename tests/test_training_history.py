import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from scenario_manifest import generate_manifest
from training_history import (
    build_training_history_row,
    prepare_training_history,
    training_history_identity,
    write_training_history,
)


class TrainingHistoryPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.identity = training_history_identity("method", 7, "manifest")

    def _row(self, episode, reward=None, energy=2.0):
        return build_training_history_row(
            self.identity,
            episode=episode,
            reward=float(episode if reward is None else reward),
            timely_goodput_mbits=float(episode),
            mobility_energy_j=energy,
            dinkelbach_lambda=0.1 * episode,
        )

    def test_csv_and_jsonl_contain_the_same_finite_episode_rows(self):
        rows = [self._row(1), self._row(2, energy=0.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_training_history(temp_dir, rows, self.identity)
            with (Path(temp_dir) / "training_history.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            json_rows = [
                json.loads(line)
                for line in (
                    Path(temp_dir) / "training_history.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(csv_rows), len(json_rows), 2)
        self.assertEqual([int(row["episode"]) for row in csv_rows], [1, 2])
        self.assertEqual(json_rows, written)
        self.assertEqual(json_rows[1]["energy_efficiency_mbit_per_j"], 0.0)
        for row in json_rows:
            self.assertTrue(
                all(
                    math.isfinite(float(row[key]))
                    for key in (
                        "reward",
                        "timely_goodput_mbits",
                        "mobility_energy_j",
                        "energy_efficiency_mbit_per_j",
                        "dinkelbach_lambda",
                    )
                )
            )

    def test_resume_truncates_only_validated_rows_after_checkpoint(self):
        rows = [self._row(1), self._row(2), self._row(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, rows, self.identity)

            reconciled = prepare_training_history(
                temp_dir,
                self.identity,
                checkpoint_rows=rows[:2],
            )
            reconciled.append(self._row(3, reward=30.0))
            final_rows = write_training_history(
                temp_dir, reconciled, self.identity
            )

        self.assertEqual([row["episode"] for row in final_rows], [1, 2, 3])
        self.assertEqual(final_rows[-1]["reward"], 30.0)

    def test_resume_refuses_mismatched_history_prefix_or_identity(self):
        checkpoint_rows = [self._row(1), self._row(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            mismatched = [self._row(1, reward=999.0), self._row(2)]
            write_training_history(temp_dir, mismatched, self.identity)
            with self.assertRaisesRegex(RuntimeError, "checkpoint prefix"):
                prepare_training_history(
                    temp_dir,
                    self.identity,
                    checkpoint_rows=checkpoint_rows,
                )

        other_identity = training_history_identity("method", 8, "manifest")
        with tempfile.TemporaryDirectory() as temp_dir:
            other_rows = [
                build_training_history_row(
                    other_identity,
                    episode=1,
                    reward=1.0,
                    timely_goodput_mbits=1.0,
                    mobility_energy_j=1.0,
                    dinkelbach_lambda=0.1,
                )
            ]
            write_training_history(temp_dir, other_rows, other_identity)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                prepare_training_history(
                    temp_dir,
                    self.identity,
                    checkpoint_rows=checkpoint_rows[:1],
                )


class TrainingHistoryResumeIntegrationTest(unittest.TestCase):
    def test_short_exact_resume_reconciles_history_without_duplicate_rows(self):
        method = MethodSpec()
        manifest = generate_manifest("train", 6601, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            checkpoint_root = run_dir / "checkpoints"
            base = dict(
                total_episodes=2,
                mode="custom",
                episode_seconds=1,
                routing_slot_seconds=0.25,
                warmup_joint_transitions=0,
                batch_size=1,
                model_checkpoint_every=1,
                full_resume_every=1,
                full_resume_keep_last=2,
                checkpoint_root=str(checkpoint_root),
                enable_model_checkpoints=False,
                enable_full_resume=True,
                enable_plots=False,
                enable_csv=False,
                random_seed=12345,
                run_directory=str(run_dir),
            )
            first = train(
                TrainingConfig(**base),
                scenario_manifest=manifest,
                method_spec=method,
            )
            expected_history = first["training_history_rows"]
            checkpoint_one = checkpoint_root / "full" / "ep_0001"
            checkpoint_two = checkpoint_root / "full" / "ep_0002"
            shutil.rmtree(checkpoint_two)

            resumed = train(
                TrainingConfig(**(base | {"resume_dir": str(checkpoint_one)})),
                scenario_manifest=manifest,
                method_spec=method,
            )

            persisted = [
                json.loads(line)
                for line in (run_dir / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(resumed["episodes_run"], 1)
        self.assertEqual(resumed["training_history_rows"], expected_history)
        self.assertEqual(persisted, expected_history)
        self.assertEqual([row["episode"] for row in persisted], [1, 2])
        self.assertEqual(
            resumed["run_metadata"]["training_history"]["row_count"], 2
        )
        self.assertEqual(
            resumed["run_metadata"]["training_history"]["last_episode"], 2
        )


if __name__ == "__main__":
    unittest.main()
