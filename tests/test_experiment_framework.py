import unittest

from comparison_experiment import build_parser
from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from scenario_manifest import generate_manifest
from training_checkpoint import validate_checkpoint_experiment_metadata


class ExperimentFrameworkTest(unittest.TestCase):
    def test_unsupported_method_spec_fails_fast(self):
        for kwargs in (
            {"movement": "ddpg"},
            {"routing": "dqn"},
            {"assignment": "km"},
            {"assignment": "random"},
            {"lambda_mode": "fixed_lambda"},
            {"llm_enabled": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    MethodSpec(**kwargs)

    def test_cli_does_not_fallback_unknown_method(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["smoke", "--method", "ddpg"])

    def test_aggregate_cli_uses_formal_completeness_defaults(self):
        args = build_parser().parse_args(
            ["aggregate", "--input-dir", "evaluation-results"]
        )

        self.assertEqual(args.expected_seed_count, 1)
        self.assertEqual(args.expected_episodes_per_seed, 100)

    def test_manifest_cli_accepts_only_supported_fixed_num_gt(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "generate-manifest",
                "--split",
                "test",
                "--manifest-seed",
                "3",
                "--episodes",
                "5",
                "--num-gt",
                "4",
            ]
        )
        self.assertEqual(args.num_gt, 4)

        for value in ("1", "10"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "generate-manifest",
                        "--split",
                        "test",
                        "--manifest-seed",
                        "3",
                        "--episodes",
                        "5",
                        "--num-gt",
                        value,
                    ]
                )

    def test_checkpoint_identity_mismatch_fails_fast(self):
        metadata = {
            "experiment": {
                "method_spec_fingerprint": "method-a",
                "manifest_hash": "manifest-a",
                "training_seed": 7,
            }
        }

        with self.assertRaisesRegex(RuntimeError, "manifest_hash"):
            validate_checkpoint_experiment_metadata(
                metadata,
                {
                    "method_spec_fingerprint": "method-a",
                    "manifest_hash": "manifest-b",
                    "training_seed": 7,
                },
            )

    def test_manifest_driven_training_uses_entry_and_metadata_hash(self):
        manifest = generate_manifest("train", 909, 1)
        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=1,
            routing_slot_seconds=0.25,
            warmup_joint_transitions=0,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=1234,
        )

        result = train(
            config,
            scenario_manifest=manifest,
            method_spec=MethodSpec(),
        )

        self.assertEqual(
            result["scenario_ids"], [manifest.episodes[0]["scenario_id"]]
        )
        self.assertEqual(
            result["run_metadata"]["manifest_hash"], manifest.content_hash
        )
        self.assertEqual(result["movement_state_dim"], 519)
        self.assertEqual(result["joint_action_dim"], 30)
        self.assertEqual(result["routing_state_dim"], 101)


if __name__ == "__main__":
    unittest.main()
