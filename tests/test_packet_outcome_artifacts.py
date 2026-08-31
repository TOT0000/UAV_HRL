import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import HRL_task_aware
from experiment_config import METHOD_REGISTRY, MethodSpec, effective_training_config
from HRL_task_aware import TrainingConfig, formal_training_config, train
from packet_outcome_artifacts import (
    MAX_BOUNDED_PACKET_OUTCOME_EPISODES,
    PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
    PACKET_OUTCOME_MODE_BOUNDED,
    PACKET_OUTCOME_MODE_DISABLED,
    PACKET_OUTCOME_MODE_STREAMING,
    PACKET_OUTCOME_REQUIRED_FIELDS,
    PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
    PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS,
    PacketOutcomeJsonlWriter,
    packet_outcome_episode_record,
)
from scenario_manifest import generate_manifest


PURE_RANDOM_METHOD = MethodSpec.parse("kkm_random_action_random_routing")


def _short_config(episodes, *, artifact_mode=PACKET_OUTCOME_MODE_DISABLED):
    bounded = artifact_mode == PACKET_OUTCOME_MODE_BOUNDED
    return TrainingConfig(
        total_episodes=int(episodes),
        mode="custom",
        episode_seconds=5,
        routing_slot_seconds=0.25,
        warmup_joint_transitions=0,
        batch_size=1,
        enable_model_checkpoints=False,
        enable_full_resume=False,
        enable_plots=False,
        enable_csv=False,
        random_seed=20260825,
        collect_packet_outcomes=bounded,
        packet_outcome_artifact_mode=artifact_mode,
        packet_outcome_collection_limit=int(episodes) if bounded else 0,
    )


class PacketOutcomeTrainingContractTest(unittest.TestCase):
    def test_disabled_training_does_not_retain_cross_episode_raw_outcomes(self):
        manifest = generate_manifest("train", 8501, 3, num_gt=2)

        result = train(
            _short_config(3),
            scenario_manifest=manifest,
            method_spec=PURE_RANDOM_METHOD,
        )

        self.assertIsNone(result["packet_outcome_artifacts"])
        self.assertEqual(result["packet_outcome_streamed_episode_count"], 0)
        self.assertEqual(len(result["episode_metrics"]), 3)
        self.assertEqual(
            result["run_metadata"]["packet_outcome_artifact_mode"],
            PACKET_OUTCOME_MODE_DISABLED,
        )
        self.assertFalse(result["run_metadata"]["collect_packet_outcomes"])
        self.assertIsNone(
            result["run_metadata"]["packet_outcome_artifact_schema_version"]
        )
        self.assertIsNone(
            result["run_metadata"][
                "packet_routing_diagnostic_contract_version"
            ]
        )
        for field in PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS:
            self.assertIsNone(result["run_metadata"][field])

    def test_bounded_collection_is_explicit_small_and_metric_equivalent(self):
        manifest = generate_manifest("train", 8502, 2, num_gt=2)

        def run_and_capture_actions(config):
            actions = []
            select_actions = HRL_task_aware._select_routing_actions

            def record_actions(*args, **kwargs):
                selected = select_actions(*args, **kwargs)
                actions.append(tuple(sorted(selected.items())))
                return selected

            with mock.patch.object(
                HRL_task_aware,
                "_select_routing_actions",
                side_effect=record_actions,
            ):
                result = train(
                    config,
                    scenario_manifest=manifest,
                    method_spec=PURE_RANDOM_METHOD,
                )
            return result, actions

        disabled, disabled_actions = run_and_capture_actions(_short_config(2))
        bounded, bounded_actions = run_and_capture_actions(
            _short_config(2, artifact_mode=PACKET_OUTCOME_MODE_BOUNDED)
        )

        self.assertEqual(disabled["episode_metrics"], bounded["episode_metrics"])
        self.assertEqual(disabled_actions, bounded_actions)
        for field in (
            "reward_log",
            "delivered_log",
            "energy_log",
            "lambda_used_log",
            "lambda_after_episode_log",
        ):
            self.assertEqual(disabled[field], bounded[field])
        self.assertEqual(len(bounded["packet_outcome_artifacts"]), 2)
        self.assertEqual(
            bounded["run_metadata"][
                "packet_routing_diagnostic_contract_version"
            ],
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
        )
        for field, definition in PACKET_ROUTING_DIAGNOSTIC_DEFINITIONS.items():
            self.assertEqual(bounded["run_metadata"][field], definition)
        for record, scenario_id, metrics in zip(
            bounded["packet_outcome_artifacts"],
            bounded["scenario_ids"],
            bounded["episode_metrics"],
        ):
            self.assertEqual(record["scenario_id"], scenario_id)
            self.assertEqual(
                record["artifact_schema_version"],
                PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
            )
            expected_outcomes = sum(
                metrics[f"{task}_source_generated_packets"]
                for task in ("fov", "com")
            )
            self.assertGreaterEqual(expected_outcomes, 0)
            self.assertEqual(len(record["packet_outcomes"]), expected_outcomes)
            self.assertEqual(
                record["summary"]["FOV"]["eligible_packets"],
                metrics["fov_eligible_packets"],
            )
            self.assertEqual(
                record["summary"]["COM"]["violation_packets"],
                metrics["com_violation_packets"],
            )

    def test_formal_registry_and_resume_cannot_enable_collection(self):
        config = formal_training_config(1500)
        self.assertFalse(config.collect_packet_outcomes)
        self.assertEqual(
            config.packet_outcome_artifact_mode, PACKET_OUTCOME_MODE_DISABLED
        )
        for method_id in METHOD_REGISTRY:
            resolved = effective_training_config(
                config, MethodSpec.parse(method_id)
            )
            with self.subTest(method=method_id):
                self.assertFalse(resolved["collect_packet_outcomes"])
                self.assertEqual(
                    resolved["packet_outcome_artifact_mode"],
                    PACKET_OUTCOME_MODE_DISABLED,
                )
                self.assertEqual(resolved["packet_outcome_collection_limit"], 0)

        with self.assertRaisesRegex(ValueError, "formal training"):
            formal_training_config(
                1500,
                collect_packet_outcomes=True,
                packet_outcome_artifact_mode=PACKET_OUTCOME_MODE_BOUNDED,
                packet_outcome_collection_limit=1500,
            )
        with self.assertRaisesRegex(ValueError, "full-resume training"):
            TrainingConfig(
                total_episodes=1,
                mode="custom",
                resume_dir="checkpoint",
                collect_packet_outcomes=True,
                packet_outcome_artifact_mode=PACKET_OUTCOME_MODE_BOUNDED,
                packet_outcome_collection_limit=1,
            )
        with self.assertRaisesRegex(ValueError, "no greater than"):
            TrainingConfig(
                total_episodes=MAX_BOUNDED_PACKET_OUTCOME_EPISODES + 1,
                mode="custom",
                collect_packet_outcomes=True,
                packet_outcome_artifact_mode=PACKET_OUTCOME_MODE_BOUNDED,
                packet_outcome_collection_limit=(
                    MAX_BOUNDED_PACKET_OUTCOME_EPISODES + 1
                ),
            )


class PacketOutcomeStreamingTest(unittest.TestCase):
    def test_streams_one_flushed_record_per_episode_without_result_list(self):
        manifest = generate_manifest("train", 8503, 2, num_gt=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "packet_outcomes.jsonl"
            flushed_line_counts = []
            with PacketOutcomeJsonlWriter(path) as writer:
                def sink(record):
                    writer.write_episode(record)
                    flushed_line_counts.append(
                        len(path.read_text(encoding="utf-8").splitlines())
                    )

                result = train(
                    _short_config(
                        2, artifact_mode=PACKET_OUTCOME_MODE_STREAMING
                    ),
                    scenario_manifest=manifest,
                    method_spec=PURE_RANDOM_METHOD,
                    packet_outcome_sink=sink,
                )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(writer.closed)
        self.assertEqual(flushed_line_counts, [1, 2])
        self.assertEqual(writer.episode_count, 2)
        self.assertIsNone(result["packet_outcome_artifacts"])
        self.assertEqual(result["packet_outcome_streamed_episode_count"], 2)
        self.assertEqual(
            result["run_metadata"]["packet_outcome_artifact_mode"],
            PACKET_OUTCOME_MODE_STREAMING,
        )
        self.assertEqual(
            [record["scenario_id"] for record in records],
            result["scenario_ids"],
        )
        for record, metrics in zip(records, result["episode_metrics"]):
            expected_outcomes = sum(
                metrics[f"{task}_source_generated_packets"]
                for task in ("fov", "com")
            )
            self.assertGreaterEqual(expected_outcomes, 0)
            self.assertEqual(len(record["packet_outcomes"]), expected_outcomes)
            if not record["packet_outcomes"]:
                continue
            self.assertTrue(
                {
                    "packet_id",
                    "source_uav_id",
                    "source_sr_id",
                    "source_kind",
                    "task_type",
                    "outcome",
                    "generation_time_seconds",
                    "finish_time_seconds",
                    "deadline_seconds",
                    "e2e_delay_seconds",
                    "size_bits",
                    "delivered_to_gs",
                    "routing_eligible",
                    "sr_waiting_seconds",
                    "remaining_bits_at_drop",
                }.issubset(record["packet_outcomes"][0])
            )
            self.assertTrue(
                PACKET_OUTCOME_REQUIRED_FIELDS.issubset(
                    record["packet_outcomes"][0]
                )
            )
            self.assertEqual(
                record["summary"]["FOV"]["generated_packets"],
                metrics["fov_generated_packets"],
            )

    def test_writer_closes_and_propagates_normal_and_exceptional_failures(self):
        record = packet_outcome_episode_record(
            "scenario-a",
            {"FOV": {"generated_packets": 0}},
            [],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            normal = PacketOutcomeJsonlWriter(Path(temp_dir) / "normal.jsonl")
            with normal:
                normal.write_episode(record)
            self.assertTrue(normal.closed)

            exceptional = PacketOutcomeJsonlWriter(
                Path(temp_dir) / "exceptional.jsonl"
            )
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                with exceptional:
                    exceptional.write_episode(record)
                    raise RuntimeError("forced failure")
            self.assertTrue(exceptional.closed)

            serialization_failure = PacketOutcomeJsonlWriter(
                Path(temp_dir) / "serialization-failure.jsonl"
            )
            invalid_record = packet_outcome_episode_record(
                "scenario-invalid",
                {"FOV": {"generated_packets": float("nan")}},
                [],
            )
            with self.assertRaisesRegex(ValueError, "Out of range float"):
                with serialization_failure:
                    serialization_failure.write_episode(invalid_record)
            self.assertTrue(serialization_failure.closed)

            failing_config = _short_config(
                1, artifact_mode=PACKET_OUTCOME_MODE_STREAMING
            )
            manifest = generate_manifest("train", 8504, 1, num_gt=2)

            def failed_write(_record):
                raise OSError("disk write failed")

            with self.assertRaisesRegex(OSError, "disk write failed"):
                train(
                    failing_config,
                    scenario_manifest=manifest,
                    method_spec=PURE_RANDOM_METHOD,
                    packet_outcome_sink=failed_write,
                )


if __name__ == "__main__":
    unittest.main()
