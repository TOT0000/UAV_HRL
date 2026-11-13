import csv
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from Packet_scheduler_v1 import PacketEngine
from scenario_manifest import generate_manifest
from training_history import (
    TRAINING_HISTORY_COMMIT,
    TRAINING_HISTORY_COLUMNS,
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

    def _row(
        self,
        episode,
        reward=None,
        energy=2.0,
        *,
        routing_delivery=0.0,
        routing_positive_progress=0.0,
        routing_negative_progress=0.0,
        completed_hops=0,
        cross_slot_hops=0,
    ):
        routing_progress = routing_positive_progress + routing_negative_progress
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
            dinkelbach_block_complete_mbits_so_far=(
                episode * (episode + 1) / 2
            ),
            dinkelbach_block_energy_joules_so_far=episode * energy,
            routing_reward_total=routing_delivery + routing_progress,
            routing_delivery_reward_total=routing_delivery,
            routing_progress_reward_total=routing_progress,
            routing_positive_progress_reward_total=routing_positive_progress,
            routing_negative_progress_reward_total=routing_negative_progress,
            routing_completed_uav_hop_count=completed_hops,
            routing_cross_slot_completed_uav_hop_count=cross_slot_hops,
            routing_cross_slot_completed_uav_hop_fraction=(
                cross_slot_hops / float(completed_hops)
                if completed_hops
                else None
            ),
        )

    def test_nonzero_routing_reward_components_are_persisted_authoritatively(self):
        row = self._row(
            1,
            routing_delivery=2.5,
            routing_positive_progress=0.75,
            routing_negative_progress=-0.25,
            completed_hops=4,
            cross_slot_hops=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, [row], self.identity)
            json_row = json.loads(
                (Path(temp_dir) / "training_history.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            with (Path(temp_dir) / "training_history.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_row = next(csv.DictReader(handle))

        self.assertEqual(json_row["reward"], 1.0)
        self.assertEqual(json_row["routing_delivery_reward_total"], 2.5)
        self.assertEqual(json_row["routing_positive_progress_reward_total"], 0.75)
        self.assertEqual(json_row["routing_negative_progress_reward_total"], -0.25)
        self.assertEqual(json_row["routing_progress_reward_total"], 0.5)
        self.assertEqual(json_row["routing_reward_total"], 3.0)
        self.assertEqual(json_row["routing_completed_uav_hop_count"], 4)
        self.assertEqual(
            json_row["routing_cross_slot_completed_uav_hop_count"], 1
        )
        self.assertEqual(
            json_row["routing_cross_slot_completed_uav_hop_fraction"], 0.25
        )
        self.assertEqual(float(csv_row["routing_reward_total"]), 3.0)

    def test_packet_engine_cancellation_round_trips_without_rewriting_values(self):
        env = SimpleNamespace(
            num_UAV=2,
            GS_ID=2,
            GS_pos=(0.0, 0.0, 0.0),
            uav_dict={
                0: SimpleNamespace(
                    get_position=lambda: (math.sqrt(80_000.0), 0.0, 100.0)
                ),
                1: SimpleNamespace(
                    get_position=lambda: (0.0, 0.0, 100.0)
                ),
            },
        )
        engine = PacketEngine(2)
        rng = np.random.default_rng(0)
        coverages = rng.random(500)
        packets = [
            engine.create_packet(
                0,
                "FOV",
                31_600.0,
                0.0,
                capture_coverage_ratio=float(coverage),
            )
            for coverage in coverages
        ]
        packets.extend(
            engine.create_packet(0, "COM", 256.0, 0.0)
            for _index in range(4_500)
        )
        for packet in packets:
            packet["hop_service_start_time"] = 0.0
        result = engine._event_result()
        for packet in packets:
            engine._record_completed_uav_hop_reward(
                env,
                packet,
                0,
                1,
                0.0,
                result,
                slot_start=0.0,
            )
        for packet_index in rng.permutation(5_000):
            engine._record_completed_uav_hop_reward(
                env,
                packets[int(packet_index)],
                1,
                0,
                0.0,
                result,
                slot_start=0.0,
            )

        authoritative = {
            "routing_reward_total": engine.routing_progress_reward_total,
            "routing_delivery_reward_total": 0.0,
            "routing_progress_reward_total": (
                engine.routing_progress_reward_total
            ),
            "routing_positive_progress_reward_total": (
                engine.routing_positive_progress_reward_total
            ),
            "routing_negative_progress_reward_total": (
                engine.routing_negative_progress_reward_total
            ),
            "routing_completed_uav_hop_count": (
                engine.completed_uav_hop_count
            ),
            "routing_cross_slot_completed_uav_hop_count": 0,
            "routing_cross_slot_completed_uav_hop_fraction": 0.0,
        }
        component_sum = (
            authoritative["routing_positive_progress_reward_total"]
            + authoritative["routing_negative_progress_reward_total"]
        )
        self.assertEqual(engine.completed_uav_hop_count, 10_000)
        self.assertAlmostEqual(
            authoritative["routing_positive_progress_reward_total"],
            476.9003806143868,
            places=10,
        )
        self.assertAlmostEqual(
            authoritative["routing_negative_progress_reward_total"],
            -476.90038061431693,
            places=10,
        )
        self.assertAlmostEqual(
            authoritative["routing_progress_reward_total"],
            6.712478316239778e-11,
            delta=1e-18,
        )
        self.assertAlmostEqual(
            component_sum,
            6.986056177993305e-11,
            delta=1e-18,
        )
        self.assertGreater(
            abs(authoritative["routing_progress_reward_total"] - component_sum),
            1e-12,
        )

        row = self._row(1)
        row.update(authoritative)
        with tempfile.TemporaryDirectory() as temp_dir:
            write_training_history(temp_dir, [row], self.identity)
            saved_json = json.loads(
                (Path(temp_dir) / "training_history.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            read_back = read_committed_training_history(
                temp_dir, self.identity
            )[0]

        for field, value in authoritative.items():
            self.assertEqual(saved_json[field], value)
            self.assertEqual(read_back[field], value)

    def test_total_reward_cancellation_uses_gross_progress_scale(self):
        delivery = 476.9003806143868
        progress = -476.90038061431693
        authoritative_total = 6.712478316239778e-11
        row = self._row(1)
        row.update(
            routing_reward_total=authoritative_total,
            routing_delivery_reward_total=delivery,
            routing_progress_reward_total=progress,
            routing_positive_progress_reward_total=0.0,
            routing_negative_progress_reward_total=progress,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_training_history(
                temp_dir, [row], self.identity
            )[0]

        self.assertEqual(written["routing_reward_total"], authoritative_total)
        self.assertEqual(written["routing_progress_reward_total"], progress)
        self.assertGreater(
            abs(authoritative_total - (delivery + progress)), 1e-12
        )

    def test_routing_reward_consistency_accepts_normal_values_at_multiple_scales(self):
        cases = (
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.75, -0.25, 0.5),
            (2e-9, 3e-9, -1e-9, 2e-9),
            (2e6 + 1e-7, 3e6, -1e6, 2e6 + 1e-7),
        )
        for progress, positive, negative, total in cases:
            with self.subTest(progress=progress, total=total):
                row = self._row(1)
                row.update(
                    routing_reward_total=total,
                    routing_delivery_reward_total=0.0,
                    routing_progress_reward_total=progress,
                    routing_positive_progress_reward_total=positive,
                    routing_negative_progress_reward_total=negative,
                )
                with tempfile.TemporaryDirectory() as temp_dir:
                    written = write_training_history(
                        temp_dir, [row], self.identity
                    )[0]
                self.assertEqual(written["routing_progress_reward_total"], progress)
                self.assertEqual(written["routing_reward_total"], total)

    def test_routing_reward_consistency_still_rejects_material_errors(self):
        bad_progress = self._row(1)
        bad_progress.update(
            routing_reward_total=1.01,
            routing_delivery_reward_total=0.0,
            routing_progress_reward_total=1.01,
            routing_positive_progress_reward_total=1.0,
            routing_negative_progress_reward_total=0.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "routing progress reward is inconsistent"
            ):
                write_training_history(
                    temp_dir, [bad_progress], self.identity
                )

        bad_total = self._row(1)
        bad_total.update(
            routing_reward_total=1.01,
            routing_delivery_reward_total=0.5,
            routing_progress_reward_total=0.5,
            routing_positive_progress_reward_total=0.5,
            routing_negative_progress_reward_total=0.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "routing total reward is inconsistent"
            ):
                write_training_history(temp_dir, [bad_total], self.identity)

    def test_routing_reward_validation_keeps_nonfinite_and_count_guards(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(nonfinite=value):
                nonfinite = self._row(1)
                nonfinite["routing_progress_reward_total"] = value
                with tempfile.TemporaryDirectory() as temp_dir:
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        write_training_history(
                            temp_dir, [nonfinite], self.identity
                        )

        incomplete = self._row(1)
        incomplete["routing_positive_progress_reward_total"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "routing reward diagnostics are incomplete"
            ):
                write_training_history(temp_dir, [incomplete], self.identity)

        illegal_count = self._row(1)
        illegal_count["routing_cross_slot_completed_uav_hop_count"] = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "completed-hop counts are inconsistent"
            ):
                write_training_history(
                    temp_dir, [illegal_count], self.identity
                )

    def test_legacy_rows_preserve_unknown_routing_diagnostics(self):
        legacy = self._row(1)
        for column in (
            "routing_reward_total",
            "routing_delivery_reward_total",
            "routing_progress_reward_total",
            "routing_positive_progress_reward_total",
            "routing_negative_progress_reward_total",
            "routing_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_fraction",
        ):
            legacy.pop(column)

        with tempfile.TemporaryDirectory() as temp_dir:
            normalized = prepare_training_history(
                temp_dir, self.identity, checkpoint_rows=[legacy]
            )

        self.assertTrue(
            all(
                normalized[0][column] is None
                for column in (
                    "routing_reward_total",
                    "routing_delivery_reward_total",
                    "routing_progress_reward_total",
                    "routing_positive_progress_reward_total",
                    "routing_negative_progress_reward_total",
                    "routing_completed_uav_hop_count",
                    "routing_cross_slot_completed_uav_hop_count",
                    "routing_cross_slot_completed_uav_hop_fraction",
                )
            )
        )

    def test_resume_keeps_legacy_unknown_row_and_appends_new_routing_values(self):
        legacy = self._row(1)
        for column in (
            "routing_reward_total",
            "routing_delivery_reward_total",
            "routing_progress_reward_total",
            "routing_positive_progress_reward_total",
            "routing_negative_progress_reward_total",
            "routing_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_fraction",
        ):
            legacy.pop(column)
        with tempfile.TemporaryDirectory() as temp_dir:
            resumed = prepare_training_history(
                temp_dir, self.identity, checkpoint_rows=[legacy]
            )
            resumed.append(
                self._row(
                    2,
                    routing_delivery=1.0,
                    routing_positive_progress=0.5,
                    completed_hops=2,
                    cross_slot_hops=1,
                )
            )
            persisted = write_training_history(
                temp_dir, resumed, self.identity
            )

        self.assertEqual([row["episode"] for row in persisted], [1, 2])
        self.assertIsNone(persisted[0]["routing_reward_total"])
        self.assertIsNone(
            persisted[0]["routing_completed_uav_hop_count"]
        )
        self.assertEqual(persisted[1]["routing_reward_total"], 1.5)
        self.assertEqual(
            persisted[1]["routing_cross_slot_completed_uav_hop_fraction"],
            0.5,
        )

    def test_csv_and_jsonl_contain_the_same_finite_episode_rows(self):
        rows = [self._row(1), self._row(2, energy=0.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_training_history(temp_dir, rows, self.identity)
            with (Path(temp_dir) / "training_history.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)
                csv_columns = tuple(reader.fieldnames or ())
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
        self.assertEqual(csv_columns, TRAINING_HISTORY_COLUMNS)
        self.assertEqual(set(json_rows[0]), set(TRAINING_HISTORY_COLUMNS))
        self.assertEqual([int(row["episode"]) for row in csv_rows], [1, 2])
        self.assertEqual(json_rows, written)
        self.assertEqual(commit["row_count"], 2)
        self.assertEqual(commit["last_episode"], 2)
        self.assertEqual(commit["csv_sha256"], csv_sha256)
        self.assertEqual(commit["jsonl_sha256"], jsonl_sha256)
        self.assertEqual(commit["method_id"], self.identity["method_id"])
        self.assertEqual(json_rows[1]["energy_efficiency_mbit_per_j"], 0.0)
        self.assertEqual(json_rows[0]["dinkelbach_lambda_used"], 0.0)
        self.assertEqual(json_rows[0]["dinkelbach_lambda_after_episode"], 0.0)
        self.assertFalse(json_rows[0]["dinkelbach_lambda_updated"])
        self.assertEqual(json_rows[0]["dinkelbach_update_status"], "accumulating")
        self.assertEqual(json_rows[0]["dinkelbach_block_index"], 1)
        self.assertEqual(json_rows[0]["dinkelbach_block_episode"], 1)
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
                        "dinkelbach_block_complete_mbits_so_far",
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
                    dinkelbach_block_complete_mbits_so_far=1.0,
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
    def test_shared_training_entry_persists_routing_fields_for_all_routing_modes(self):
        routing_fields = (
            "routing_reward_total",
            "routing_delivery_reward_total",
            "routing_progress_reward_total",
            "routing_positive_progress_reward_total",
            "routing_negative_progress_reward_total",
            "routing_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_fraction",
        )
        manifest = generate_manifest("train", 7711, 1)
        methods = (
            "td3_dinkelbach",
            "td3_dinkelbach_dqn",
            "td3_dinkelbach_ddqn",
            "td3_dinkelbach_random_routing",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for method_id in methods:
                with self.subTest(method_id=method_id):
                    run_dir = root / method_id
                    result = train(
                        TrainingConfig(
                            total_episodes=1,
                            mode="custom",
                            episode_seconds=1,
                            routing_slot_seconds=0.25,
                            warmup_joint_transitions=0,
                            routing_warmup_transitions=1,
                            batch_size=1,
                            enable_model_checkpoints=False,
                            enable_full_resume=False,
                            enable_plots=False,
                            enable_csv=False,
                            random_seed=7711,
                            run_directory=str(run_dir),
                        ),
                        scenario_manifest=manifest,
                        method_spec=MethodSpec.parse(method_id),
                    )
                    json_row = json.loads(
                        (run_dir / "training_history.jsonl").read_text(
                            encoding="utf-8"
                        )
                    )
                    with (run_dir / "training_history.csv").open(
                        "r", encoding="utf-8", newline=""
                    ) as handle:
                        csv_row = next(csv.DictReader(handle))
                    for field in routing_fields:
                        self.assertIn(field, json_row)
                        self.assertIn(field, csv_row)
                        self.assertEqual(
                            json_row[field], result["episode_metrics"][0][field]
                        )

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
            checkpoint_payload = torch.load(
                checkpoint_one / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )

            def nested_mapping_keys(value):
                if isinstance(value, dict):
                    return set(value).union(
                        *(
                            nested_mapping_keys(child)
                            for child in value.values()
                        )
                    )
                if isinstance(value, (list, tuple)):
                    return set().union(
                        *(nested_mapping_keys(child) for child in value)
                    )
                return set()

            checkpoint_keys = nested_mapping_keys(
                checkpoint_payload["training_state"]
            )
            self.assertNotIn("packet_outcomes", checkpoint_keys)
            self.assertNotIn("packet_outcome_artifacts", checkpoint_keys)

            with (
                mock.patch(
                    "training_checkpoint._load_network_states"
                ) as load_networks,
                mock.patch("training_checkpoint._load_replay") as load_replay,
                mock.patch(
                    "training_checkpoint._restore_rng_state"
                ) as restore_rng,
            ):
                with self.assertRaisesRegex(RuntimeError, "batch_size"):
                    train(
                        TrainingConfig(
                            **(
                                base
                                | {
                                    "batch_size": 2,
                                    "resume_dir": str(checkpoint_one),
                                }
                            )
                        ),
                        scenario_manifest=manifest,
                        method_spec=method,
                    )
                load_networks.assert_not_called()
                load_replay.assert_not_called()
                restore_rng.assert_not_called()

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
        self.assertIsNone(first["packet_outcome_artifacts"])
        self.assertIsNone(resumed["packet_outcome_artifacts"])
        self.assertFalse(
            resumed["run_metadata"]["formal_config"][
                "collect_packet_outcomes"
            ]
        )
        self.assertEqual(
            resumed["run_metadata"]["formal_config"][
                "packet_outcome_artifact_mode"
            ],
            "disabled",
        )
        self.assertEqual(resumed["training_history_rows"], expected_history)
        self.assertEqual(persisted, expected_history)
        self.assertEqual([row["episode"] for row in persisted], [1, 2])
        routing_fields = (
            "routing_reward_total",
            "routing_delivery_reward_total",
            "routing_progress_reward_total",
            "routing_positive_progress_reward_total",
            "routing_negative_progress_reward_total",
            "routing_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_count",
            "routing_cross_slot_completed_uav_hop_fraction",
        )
        for persisted_row, metric_row in zip(
            persisted, first["episode_metrics"]
        ):
            for field in routing_fields:
                self.assertIn(field, persisted_row)
                self.assertEqual(persisted_row[field], metric_row[field])
        self.assertEqual(resumed["dinkelbach_update_count"], 0)
        self.assertEqual(resumed["dinkelbach_state"], first["dinkelbach_state"])
        self.assertEqual(resumed["lambda_used_log"], first["lambda_used_log"])
        self.assertEqual(
            resumed["lambda_after_episode_log"],
            first["lambda_after_episode_log"],
        )
        self.assertEqual(len(resumed["lambda_used_log"]), 2)
        self.assertEqual(
            resumed["lambda_used_log"],
            [row["dinkelbach_lambda_used"] for row in persisted],
        )
        self.assertEqual(
            resumed["lambda_after_episode_log"],
            [row["dinkelbach_lambda_after_episode"] for row in persisted],
        )
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
