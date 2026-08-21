import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import zipfile

from training_checkpoint import (
    checkpoint_artifact_fingerprint,
    checkpoint_artifact_provenance,
    checkpoint_metadata_fingerprint,
    checkpoint_models_sha256,
)


class PaperCheckpointArtifactProvenanceTest(unittest.TestCase):
    @staticmethod
    def _write_models(path, marker=b"payload"):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", marker)
            archive.writestr("archive/version", b"3\n")

    def _checkpoint(self, root, name="ep_2500", marker=b"payload"):
        checkpoint = root / name
        checkpoint.mkdir()
        metadata = {"episode": 2499, "experiment": {"method_id": "synthetic"}}
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        self._write_models(checkpoint / "models.pt", marker)
        return checkpoint, metadata

    def test_combined_fingerprint_binds_metadata_and_streamed_model_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint, metadata = self._checkpoint(Path(temp_dir))
            with mock.patch(
                "training_checkpoint.torch.load",
                side_effect=AssertionError("provenance loaded weights"),
            ):
                provenance = checkpoint_artifact_provenance(
                    checkpoint, metadata=metadata
                )
            self.assertEqual(
                provenance["checkpoint_metadata_fingerprint"],
                checkpoint_metadata_fingerprint(metadata),
            )
            self.assertEqual(
                provenance["checkpoint_models_sha256"],
                checkpoint_models_sha256(checkpoint),
            )
            self.assertEqual(
                provenance["checkpoint_artifact_fingerprint"],
                checkpoint_artifact_fingerprint(
                    provenance["checkpoint_metadata_fingerprint"],
                    provenance["checkpoint_models_sha256"],
                ),
            )

    def test_payload_mutation_and_checkpoint_payload_swap_change_or_fail_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first, metadata = self._checkpoint(root, "ep_2500", b"first")
            second, _ = self._checkpoint(root, "ep_2499", b"second")
            original = checkpoint_artifact_provenance(first, metadata=metadata)

            self._write_models(first / "models.pt", b"mutated")
            mutated = checkpoint_artifact_provenance(first, metadata=metadata)
            self.assertNotEqual(
                original["checkpoint_models_sha256"],
                mutated["checkpoint_models_sha256"],
            )
            self.assertNotEqual(
                original["checkpoint_artifact_fingerprint"],
                mutated["checkpoint_artifact_fingerprint"],
            )

            (first / "models.pt").write_bytes((second / "models.pt").read_bytes())
            swapped = checkpoint_artifact_provenance(first, metadata=metadata)
            self.assertEqual(
                swapped["checkpoint_models_sha256"],
                checkpoint_models_sha256(second),
            )
            self.assertNotEqual(
                original["checkpoint_artifact_fingerprint"],
                swapped["checkpoint_artifact_fingerprint"],
            )

    def test_missing_empty_truncated_and_arbitrary_payloads_fail_loudly(self):
        cases = {
            "missing": None,
            "empty": b"",
            "truncated": b"PK\x03\x04truncated",
            "arbitrary": b"weights-not-a-torch-archive",
        }
        for name, payload in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = Path(temp_dir) / "ep_2500"
                checkpoint.mkdir()
                if payload is not None:
                    (checkpoint / "models.pt").write_bytes(payload)
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    checkpoint_models_sha256(checkpoint)

    def test_metadata_mutation_changes_only_metadata_and_combined_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint, metadata = self._checkpoint(Path(temp_dir))
            original = checkpoint_artifact_provenance(checkpoint, metadata=metadata)
            changed_metadata = {
                **metadata,
                "experiment": {"method_id": "different"},
            }
            changed = checkpoint_artifact_provenance(
                checkpoint, metadata=changed_metadata
            )
            self.assertEqual(
                original["checkpoint_models_sha256"],
                changed["checkpoint_models_sha256"],
            )
            self.assertNotEqual(
                original["checkpoint_metadata_fingerprint"],
                changed["checkpoint_metadata_fingerprint"],
            )
            self.assertNotEqual(
                original["checkpoint_artifact_fingerprint"],
                changed["checkpoint_artifact_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
