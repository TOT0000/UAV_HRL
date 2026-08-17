import tempfile
import unittest
from pathlib import Path

from experiment_config import MethodSpec
from experiment_paths import (
    evaluation_run_directory,
    evaluation_run_identity,
    filesystem_slug,
    prepare_run_directory,
    training_run_directory,
    training_run_identity,
)
from scenario_manifest import generate_manifest


class ExperimentPathTest(unittest.TestCase):
    def setUp(self):
        self.method = MethodSpec()
        self.train_manifest = generate_manifest("train", 1001, 1)
        self.other_train_manifest = generate_manifest("train", 1002, 1)
        self.validation_manifest = generate_manifest("validation", 1003, 1)
        self.test_manifest = generate_manifest("test", 1004, 1)

    def test_method_slug_is_filesystem_safe(self):
        slug = filesystem_slug(" Corrected / No LLM: TD3 ")

        self.assertEqual(slug, "corrected-no-llm-td3")

    def test_training_paths_isolate_seed_manifest_and_run_kind(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = training_run_directory(
                temp_dir, self.method, self.train_manifest, 11
            )
            other_seed = training_run_directory(
                temp_dir, self.method, self.train_manifest, 22
            )
            other_manifest = training_run_directory(
                temp_dir, self.method, self.other_train_manifest, 11
            )
            evaluation = evaluation_run_directory(
                temp_dir, self.method, self.test_manifest, 11
            )

        self.assertNotEqual(first, other_seed)
        self.assertNotEqual(first, other_manifest)
        self.assertNotEqual(first, evaluation)
        self.assertEqual(first.name, "seed-11")
        self.assertEqual(first.parent.name, self.train_manifest.content_hash[:8])
        self.assertEqual(first.parent.parent.name, "train")

    def test_evaluation_paths_isolate_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validation = evaluation_run_directory(
                temp_dir, self.method, self.validation_manifest, 11
            )
            test = evaluation_run_directory(
                temp_dir, self.method, self.test_manifest, 11
            )

        self.assertNotEqual(validation, test)
        self.assertIn("validation", validation.parts)
        self.assertIn("test", test.parts)

    def test_existing_nonempty_run_requires_explicit_resume(self):
        identity = training_run_identity(
            self.method, self.train_manifest, 11
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = training_run_directory(
                temp_dir, self.method, self.train_manifest, 11
            )
            prepare_run_directory(run_dir, identity)

            with self.assertRaisesRegex(FileExistsError, "explicit resume"):
                prepare_run_directory(run_dir, identity)

    def test_matching_resume_reuses_only_its_canonical_run_directory(self):
        identity = training_run_identity(
            self.method, self.train_manifest, 11
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = training_run_directory(
                temp_dir, self.method, self.train_manifest, 11
            )
            prepared = prepare_run_directory(run_dir, identity)
            checkpoint = prepared / "checkpoints" / "full" / "ep_0050"
            checkpoint.mkdir(parents=True)

            resumed = prepare_run_directory(
                run_dir, identity, resume_checkpoint=checkpoint
            )
            self.assertEqual(resumed, prepared)

            wrong_run = training_run_directory(
                temp_dir, self.method, self.train_manifest, 22
            )
            with self.assertRaisesRegex(RuntimeError, "different canonical"):
                prepare_run_directory(
                    wrong_run, identity, resume_checkpoint=checkpoint
                )

    def test_identity_marker_matches_resolved_run_identity(self):
        identity = evaluation_run_identity(
            self.method, self.test_manifest, 77
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = evaluation_run_directory(
                temp_dir, self.method, self.test_manifest, 77
            )
            prepared = prepare_run_directory(run_dir, identity)

            self.assertTrue((prepared / "run_identity.json").is_file())
            self.assertEqual(identity["evaluation_split"], "test")
            self.assertEqual(
                identity["evaluation_manifest_hash"],
                self.test_manifest.content_hash,
            )


if __name__ == "__main__":
    unittest.main()
