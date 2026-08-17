from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from HRL_task_aware import formal_training_config
from com_capacity_calibration import load_com_capacity_reference
from comparison_experiment import main as comparison_main
from experiment_config import MethodSpec
from experiment_paths import (
    evaluation_run_directory,
    read_run_status,
    training_run_directory,
)
from scenario_manifest import generate_manifest
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    calibration_fingerprint,
)


class ExperimentPreflightTest(unittest.TestCase):
    def setUp(self):
        self.method = MethodSpec()

    def _train_args(self, manifest_path, output_root, *, episodes=1, resume=None):
        args = [
            "train",
            "--manifest",
            str(manifest_path),
            "--training-seed",
            "17",
            "--episodes",
            str(episodes),
            "--output-dir",
            str(output_root),
        ]
        if resume is not None:
            args.extend(("--resume", str(resume)))
        return args

    def _model_checkpoint(self, root, training_seed=17, mutate=None):
        _, calibration = load_com_capacity_reference()
        formal_config = asdict(
            formal_training_config(1500, random_seed=training_seed)
        )
        metadata = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 1499,
            "movement_state_dim": 532,
            "joint_action_dim": 48,
            "routing_state_dim": 126,
            "centralized_td3_gamma": 1.0,
            "routing_ddqn_gamma": 0.99,
            "com_calibration_fingerprint": calibration_fingerprint(calibration),
            "experiment": {
                "method_spec_fingerprint": self.method.fingerprint,
                "training_seed": training_seed,
                "manifest_hash": "training-manifest",
                "formal_config": formal_config,
            },
        }
        if mutate is not None:
            mutate(metadata)
        checkpoint = Path(root) / "checkpoints" / "models" / "ep_1500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (checkpoint / "models.pt").write_bytes(b"weights-not-loaded-by-preflight")
        return checkpoint

    def _evaluate_args(self, manifest_path, checkpoint, output_root, seed=17):
        return [
            "evaluate",
            "--split",
            "validation",
            "--manifest",
            str(manifest_path),
            "--training-seed",
            str(seed),
            "--episodes",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_root),
        ]

    def test_invalid_or_short_training_manifest_creates_no_run_or_simulator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest_path = root / "train.json"
            generate_manifest("train", 7001, 1).save(manifest_path)

            with mock.patch("HRL_task_aware.Simulator") as simulator:
                with self.assertRaisesRegex(ValueError, "fewer entries"):
                    comparison_main(
                        self._train_args(manifest_path, output, episodes=2)
                    )
                simulator.assert_not_called()
            self.assertFalse(output.exists())

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["content_hash"] = "0" * 64
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash"):
                comparison_main(self._train_args(manifest_path, output))
            self.assertFalse(output.exists())

    def test_training_lifecycle_marks_failure_and_blocks_fresh_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest = generate_manifest("train", 7002, 1)
            manifest_path = root / "train.json"
            manifest.save(manifest_path)
            run_dir = training_run_directory(output, self.method, manifest, 17)

            with mock.patch(
                "comparison_experiment.train",
                side_effect=RuntimeError("simulator exploded"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulator exploded"):
                    comparison_main(self._train_args(manifest_path, output))

            status = read_run_status(run_dir)
            self.assertEqual(status["state"], "FAILED")
            self.assertEqual(status["exception"]["type"], "RuntimeError")
            self.assertEqual(
                [item["state"] for item in status["transitions"]],
                ["PREPARING", "RUNNING", "FAILED"],
            )

            with mock.patch("comparison_experiment.train") as train_mock:
                with self.assertRaisesRegex(FileExistsError, "explicit resume"):
                    comparison_main(self._train_args(manifest_path, output))
                train_mock.assert_not_called()

    def test_valid_fresh_training_writes_identity_and_completed_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest = generate_manifest("train", 7010, 1)
            manifest_path = root / "train.json"
            manifest.save(manifest_path)
            run_dir = training_run_directory(output, self.method, manifest, 17)

            with mock.patch(
                "comparison_experiment.train", return_value={"run_metadata": {}}
            ):
                comparison_main(self._train_args(manifest_path, output))

            identity = json.loads(
                (run_dir / "run_identity.json").read_text(encoding="utf-8")
            )
            status = read_run_status(run_dir)
            self.assertEqual(identity["method_id"], self.method.method_id)
            self.assertEqual(identity["training_manifest_hash"], manifest.content_hash)
            self.assertEqual(status["state"], "COMPLETED")
            self.assertEqual(
                [item["state"] for item in status["transitions"]],
                ["PREPARING", "RUNNING", "COMPLETED"],
            )

    def test_failed_training_exact_resume_records_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest = generate_manifest("train", 7003, 1)
            manifest_path = root / "train.json"
            manifest.save(manifest_path)
            run_dir = training_run_directory(output, self.method, manifest, 17)

            with mock.patch(
                "comparison_experiment.train", side_effect=RuntimeError("stop")
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    comparison_main(self._train_args(manifest_path, output))

            resume = run_dir / "checkpoints" / "full" / "ep_0001"
            resume.mkdir(parents=True)
            resumed_result = {"run_metadata": {}}
            training_state = {
                key: []
                for key in (
                    "reward_log",
                    "delivered_log",
                    "energy_log",
                    "lambda_log",
                    "td3_noise_log",
                    "routing_epsilon_log",
                    "training_history_rows",
                )
            }
            training_state.update(
                {
                    "lambda_EE_global": 0.1,
                    "total_joint_transitions": 0,
                    "global_routing_slot": 0,
                    "td3_post_warmup_transition": 0,
                    "ddqn_schedule_slot": 0,
                }
            )
            plan = mock.Mock(
                resume_training_state=training_state,
                resume_episode=1,
            )
            with (
                mock.patch(
                    "comparison_experiment.plan_resume_reconciliation",
                    return_value=plan,
                ),
                mock.patch(
                    "comparison_experiment.preflight_resume_training_history",
                    return_value=[{}],
                ),
                mock.patch(
                    "comparison_experiment.execute_resume_reconciliation",
                    return_value=None,
                ),
                mock.patch(
                    "comparison_experiment.train", return_value=resumed_result
                ),
            ):
                comparison_main(
                    self._train_args(manifest_path, output, resume=resume)
                )

            status = read_run_status(run_dir)
            self.assertEqual(status["state"], "COMPLETED")
            self.assertEqual(
                [item["state"] for item in status["transitions"]],
                [
                    "PREPARING",
                    "RUNNING",
                    "FAILED",
                    "RESUMING",
                    "RUNNING",
                    "COMPLETED",
                ],
            )
            identity = json.loads(
                (run_dir / "run_identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(identity["training_seed"], 17)

    def test_invalid_evaluation_checkpoint_preflight_loads_no_weights_or_simulator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest_path = root / "validation.json"
            generate_manifest("validation", 7004, 1).save(manifest_path)
            checkpoint = self._model_checkpoint(
                root,
                mutate=lambda metadata: metadata.update(
                    {"movement_state_dim": 531}
                ),
            )

            with (
                mock.patch("training_checkpoint.torch.load") as torch_load,
                mock.patch("HRL_task_aware.Simulator") as simulator,
            ):
                with self.assertRaisesRegex(RuntimeError, "movement_state_dim"):
                    comparison_main(
                        self._evaluate_args(
                            manifest_path, checkpoint, output
                        )
                    )
                torch_load.assert_not_called()
                simulator.assert_not_called()
            self.assertFalse(output.exists())

    def test_short_evaluation_manifest_creates_no_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest_path = root / "validation.json"
            generate_manifest("validation", 7011, 1).save(manifest_path)

            args = self._evaluate_args(
                manifest_path, root / "missing-checkpoint", output
            )
            args[args.index("--episodes") + 1] = "2"
            with mock.patch("HRL_task_aware.Simulator") as simulator:
                with self.assertRaisesRegex(ValueError, "fewer entries"):
                    comparison_main(args)
                simulator.assert_not_called()
            self.assertFalse(output.exists())

    def test_evaluation_seed_and_formal_config_mismatches_create_no_run(self):
        cases = (
            (
                "method",
                lambda metadata: metadata["experiment"].update(
                    {"method_spec_fingerprint": "wrong"}
                ),
                "method_spec_fingerprint",
            ),
            (
                "seed",
                lambda metadata: metadata["experiment"].update(
                    {"training_seed": 18}
                ),
                "training_seed",
            ),
            (
                "config",
                lambda metadata: metadata["experiment"]["formal_config"].update(
                    {"batch_size": 32}
                ),
                "batch_size",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                output = root / "output"
                manifest_path = root / "validation.json"
                generate_manifest("validation", 7005, 1).save(manifest_path)
                checkpoint = self._model_checkpoint(root, mutate=mutate)

                with self.assertRaisesRegex(RuntimeError, message):
                    comparison_main(
                        self._evaluate_args(
                            manifest_path, checkpoint, output
                        )
                    )
                self.assertFalse(output.exists())

    def test_valid_evaluation_completes_and_collision_rerun_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            manifest = generate_manifest("validation", 7006, 1)
            manifest_path = root / "validation.json"
            manifest.save(manifest_path)
            checkpoint = self._model_checkpoint(root)
            run_dir = evaluation_run_directory(
                output, self.method, manifest, 17
            )
            result = {
                "run_metadata": {},
                "evaluation_invariants": {"weights_unchanged": True},
                "episode_metrics": [],
            }

            with mock.patch("HRL_task_aware.train", return_value=result):
                comparison_main(
                    self._evaluate_args(manifest_path, checkpoint, output)
                )

            status = read_run_status(run_dir)
            self.assertEqual(status["state"], "COMPLETED")
            self.assertEqual(
                [item["state"] for item in status["transitions"]],
                ["PREPARING", "RUNNING", "COMPLETED"],
            )
            self.assertTrue((run_dir / "run_identity.json").is_file())

            with mock.patch("HRL_task_aware.train") as train_mock:
                with self.assertRaisesRegex(FileExistsError, "explicit resume"):
                    comparison_main(
                        self._evaluate_args(manifest_path, checkpoint, output)
                    )
                train_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
