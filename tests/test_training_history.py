import csv
import hashlib
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
    TRAINING_HISTORY_COMMIT,
    TrainingHistoryConsistencyError,
    TrainingHistoryIdentityError,
    build_training_history_row,
    prepare_training_history,
    read_committed_training_history,
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
            dinkelbach_lambda_used=0.0,
            dinkelbach_lambda_after_episode=0.0,
            dinkelbach_lambda_updated=False,
            dinkelbach_update_status="accumulating",
            dinkelbach_block_index=1,
            dinkelbach_block_episode=episode,
            dinkelbach_block_timely_mbits_so_far=(
                episode * (episode + 1) / 2
            ),
            dinkelbach_block_energy_joules_so_far=episode * energy,
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
            commit = json.loads(
                (Path(temp_dir) / TRAINING_HISTORY_COMMIT).read_text(
                    encoding="utf-8"
                )
            )
            csv_sha256 = hashlib.sha256(
                (Path(temp_dir) / "training_history.csv").read_bytes()
            ).hexdigest()
            jsonl_sha256 = hashlib.sha256(
                (Path(temp_dir) / "training_history.jsonl").read_bytes()
            ).hexdigest()

        self.assertEqual(len(csv_rows), len(json_rows), 2)
        self.assertEqual([int(row["episode"]) for row in csv_rows], [1, 2])
        self.assertEqual(json_rows, written)
        self.assertEqual(commit["row_count"], 2)
        self.assertEqual(commit["last_episode"], 2)
        self.assertEqual(commit["csv_sha256"], csv_sha256)
        self.assertEqual(commit["jsonl_sha256"], jsonl_sha256)
        self.assertEqual(commit["method_id"], self.identity["method_id"])
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
                        "dinkelbach_lambda_used",
                        "dinkelbach_lambda_after_episode",
                        "dinkelbach_block_timely_mbits_so_far",
                        "dinkelbach_block_energy_joules_so_far",
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
                    dinkelbach_lambda_used=0.0,
                    dinkelbach_lambda_after_episode=0.0,
                    dinkelbach_lambda_updated=False,
                    dinkelbach_update_status="accumulating",
                    dinkelbach_block_index=1,
                    dinkelbach_block_episode=1,
                    dinkelbach_block_timely_mbits_so_far=1.0,
                    dinkelbach_block_energy_joules_so_far=1.0,
                )
            ]
            write_training_history(temp_dir, other_rows, other_identity)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                prepare_training_history(
                    temp_dir,
                    self.identity,
                    checkpoint_rows=checkpoint_rows[:1],
                )

    def test_crash_after_jsonl_replace_is_detected_and_exact_resume_repairs(self):
        checkpoint_rows = [self._row(1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, checkpoint_rows, self.identity)

            def fail(stage):
                if stage == "after_jsonl_replace":
                    raise RuntimeError("injected crash")

            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                write_training_history(
                    temp_dir,
                    [*checkpoint_rows, self._row(2)],
                    self.identity,
                    _fault_inject=fail,
                )
            with self.assertRaisesRegex(
                TrainingHistoryConsistencyError, "JSONL hash"
            ):
                read_committed_training_history(temp_dir, self.identity)

            first_repair = prepare_training_history(
                temp_dir, self.identity, checkpoint_rows=checkpoint_rows
            )
            second_repair = prepare_training_history(
                temp_dir, self.identity, checkpoint_rows=checkpoint_rows
            )

            self.assertEqual(first_repair, checkpoint_rows)
            self.assertEqual(second_repair, checkpoint_rows)
            self.assertEqual(
                read_committed_training_history(temp_dir, self.identity),
                checkpoint_rows,
            )

    def test_crash_after_csv_replace_before_commit_is_detected_and_repaired(self):
        checkpoint_rows = [self._row(1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, checkpoint_rows, self.identity)

            def fail(stage):
                if stage == "after_csv_replace":
                    raise RuntimeError("injected crash")

            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                write_training_history(
                    temp_dir,
                    [*checkpoint_rows, self._row(2)],
                    self.identity,
                    _fault_inject=fail,
                )
            with self.assertRaisesRegex(
                TrainingHistoryConsistencyError, "hash"
            ):
                read_committed_training_history(temp_dir, self.identity)

            repaired = prepare_training_history(
                temp_dir, self.identity, checkpoint_rows=checkpoint_rows
            )
            self.assertEqual(repaired, checkpoint_rows)
            self.assertEqual(
                read_committed_training_history(temp_dir, self.identity),
                checkpoint_rows,
            )

    def test_commit_identity_and_hash_mismatches_are_rejected(self):
        rows = [self._row(1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, rows, self.identity)
            commit_path = Path(temp_dir) / TRAINING_HISTORY_COMMIT
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            commit["training_seed"] = 999
            commit_path.write_text(json.dumps(commit), encoding="utf-8")

            with self.assertRaisesRegex(
                TrainingHistoryIdentityError, "identity mismatch"
            ):
                read_committed_training_history(temp_dir, self.identity)
            with self.assertRaisesRegex(
                TrainingHistoryIdentityError, "identity mismatch"
            ):
                prepare_training_history(
                    temp_dir, self.identity, checkpoint_rows=rows
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, rows, self.identity)
            commit_path = Path(temp_dir) / TRAINING_HISTORY_COMMIT
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            commit["jsonl_sha256"] = "0" * 64
            commit_path.write_text(json.dumps(commit), encoding="utf-8")

            with self.assertRaisesRegex(
                TrainingHistoryConsistencyError, "JSONL hash"
            ):
                read_committed_training_history(temp_dir, self.identity)

    def test_fresh_run_rejects_partial_commit_but_ignores_temp_transaction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary = Path(temp_dir) / ".training-history-txn-abandoned"
            temporary.mkdir()
            (temporary / "training_history.jsonl").write_text(
                "not committed\n", encoding="utf-8"
            )
            self.assertEqual(
                prepare_training_history(temp_dir, self.identity), []
            )

            (Path(temp_dir) / "training_history.jsonl").write_text(
                json.dumps(self._row(1)) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                TrainingHistoryConsistencyError, "incomplete"
            ):
                prepare_training_history(temp_dir, self.identity)

    def test_transaction_work_files_fit_a_long_canonical_run_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / ("method-" + "m" * 72) / ("seed-" + "s" * 62)
            written = write_training_history(
                run_dir, [self._row(1)], self.identity
            )

            self.assertEqual(written, [self._row(1)])
            self.assertEqual(
                read_committed_training_history(run_dir, self.identity),
                written,
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
        self.assertEqual(resumed["dinkelbach_update_count"], 0)
        self.assertEqual(resumed["dinkelbach_state"], first["dinkelbach_state"])
        self.assertEqual(
            [row["dinkelbach_block_episode"] for row in persisted], [1, 2]
        )
        self.assertEqual(
            [row["dinkelbach_lambda_updated"] for row in persisted],
            [False, False],
        )
        self.assertEqual(
            resumed["run_metadata"]["training_history"]["row_count"], 2
        )
        self.assertEqual(
            resumed["run_metadata"]["training_history"]["last_episode"], 2
        )


if __name__ == "__main__":
    unittest.main()
