import tempfile
import unittest
from pathlib import Path

from training_checkpoint import (
    _atomic_checkpoint_write,
    checkpoint_episode_schedule,
    prune_full_resume_checkpoints,
)


class CheckpointScheduleTest(unittest.TestCase):
    def test_formal_model_and_full_schedules_are_every_50_with_final_1500(self):
        expected = list(range(50, 1501, 50))

        model_schedule = checkpoint_episode_schedule(1500, 50)
        full_schedule = checkpoint_episode_schedule(1500, 50)

        self.assertEqual(model_schedule, expected)
        self.assertEqual(full_schedule, expected)
        self.assertEqual(len(model_schedule), 30)
        self.assertEqual(model_schedule.count(1500), 1)

    def test_custom_75_episode_run_adds_one_nonduplicate_final_checkpoint(self):
        self.assertEqual(checkpoint_episode_schedule(75, 50), [50, 75])
        self.assertEqual(checkpoint_episode_schedule(50, 50), [50])


class FullResumeRetentionTest(unittest.TestCase):
    def _checkpoint_dir(self, root, episode):
        path = Path(root) / "full" / f"ep_{episode:04d}"
        path.mkdir(parents=True)
        (path / "sentinel").write_text(str(episode), encoding="utf-8")
        return path

    def test_retention_keeps_only_latest_two_full_resume_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for episode in (50, 100, 150, 200):
                self._checkpoint_dir(root, episode)
            model = root / "models" / "ep_0050"
            model.mkdir(parents=True)
            unrelated = root / "full" / "notes"
            unrelated.mkdir()

            removed = prune_full_resume_checkpoints(root / "full", keep_last=2)

            self.assertEqual(
                {path.name for path in removed}, {"ep_0050", "ep_0100"}
            )
            self.assertEqual(
                {
                    path.name
                    for path in (root / "full").iterdir()
                    if path.name.startswith("ep_")
                },
                {"ep_0150", "ep_0200"},
            )
            self.assertTrue(model.is_dir())
            self.assertTrue(unrelated.is_dir())

    def test_failed_atomic_write_preserves_existing_retention_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = {
                self._checkpoint_dir(root, 50),
                self._checkpoint_dir(root, 100),
            }
            target = root / "full" / "ep_0150"

            def fail(temporary):
                (temporary / "partial").write_text("partial", encoding="utf-8")
                raise OSError("simulated write failure")

            with self.assertRaisesRegex(OSError, "simulated"):
                _atomic_checkpoint_write(target, fail)

            self.assertFalse(target.exists())
            self.assertTrue(all(path.is_dir() for path in existing))
            self.assertEqual(
                list((root / "full").glob(".*.tmp-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
