import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_experiment import main


class SimpleRunnerLifecycleIntegrationTest(unittest.TestCase):
    def test_td3_ratio_full_checkpoint_round_trip_keeps_terminal_objective(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            main(
                [
                    "td3_ratio",
                    "--episodes",
                    "1",
                    "--episode-seconds",
                    "1",
                    "--checkpoint-interval",
                    "1",
                    "--roi-count",
                    "2",
                    "--output-root",
                    str(output),
                ]
            )
            run_dir = next((output / "td3_ratio").iterdir())
            history = json.loads(
                (run_dir / "training_history.jsonl").read_text(encoding="utf-8")
            )
            expected_ratio = history["energy_efficiency_mbit_per_j"]
            replay_path = run_dir / "checkpoints" / "full" / "ep_0001" / "joint_replay.npz"
            with np.load(replay_path, allow_pickle=False) as replay:
                self.assertIn("ratio_objective_reward", replay.files)
                self.assertAlmostEqual(
                    float(replay["ratio_objective_reward"][0, 0]),
                    expected_ratio,
                    places=9,
                )
            self.assertEqual(main(["resume", str(run_dir)]), 0)

    def test_new_resume_and_collision_free_smoke_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            self.assertEqual(
                main(
                    [
                        "ddpg_ratio",
                        "--episodes",
                        "1",
                        "--episode-seconds",
                        "1",
                        "--checkpoint-interval",
                        "1",
                        "--roi-count",
                        "2",
                        "--output-root",
                        str(output),
                    ]
                ),
                0,
            )
            runs = list((output / "ddpg_ratio").iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            history_before = (run_dir / "training_history.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(len(history_before.splitlines()), 1)

            self.assertEqual(main(["resume", str(run_dir)]), 0)
            history_after = (run_dir / "training_history.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(history_after, history_before)
            metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["method"], "ddpg_ratio")
            self.assertEqual(metadata["agent"], "ddpg")
            self.assertEqual(metadata["reward_mode"], "ratio")
            self.assertEqual(metadata["history_rows"], 1)

            command = [
                "evaluate",
                str(run_dir),
                "--checkpoint-episode",
                "1",
                "--smoke",
            ]
            self.assertEqual(main(command), 0)
            self.assertEqual(main(command), 0)
            evaluations = list((run_dir / "evaluation" / "ep_1").iterdir())
            self.assertEqual(len(evaluations), 2)
            self.assertNotEqual(evaluations[0], evaluations[1])
            for evaluation in evaluations:
                self.assertTrue((evaluation / "per_episode.csv").is_file())
                self.assertTrue((evaluation / "per_episode.jsonl").is_file())
                self.assertTrue((evaluation / "run_metadata.json").is_file())
                self.assertEqual(len(list((evaluation / "plots").glob("*.png"))), 3)
                evaluation_metadata = json.loads(
                    (evaluation / "run_metadata.json").read_text(encoding="utf-8")
                )
                self.assertFalse(evaluation_metadata["formal_evaluation"])
                self.assertTrue(evaluation_metadata["smoke_evaluation"])


if __name__ == "__main__":
    unittest.main()
