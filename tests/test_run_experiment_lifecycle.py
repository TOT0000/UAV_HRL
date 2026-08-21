import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from run_experiment import main
from Simulator import Simulator


class SimpleRunnerLifecycleIntegrationTest(unittest.TestCase):
    def test_td3_ratio_partial_resume_runs_remaining_episodes_exactly_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            command = [
                "td3_ratio",
                "--episodes",
                "3",
                "--episode-seconds",
                "2",
                "--checkpoint-interval",
                "1",
                "--roi-count",
                "2",
                "--output-root",
                str(output),
            ]
            original_apply = Simulator.apply_scenario_entry
            attempted_scenarios = []

            def interrupt_before_second_episode(simulator, scenario_entry):
                attempted_scenarios.append(str(scenario_entry["scenario_id"]))
                if len(attempted_scenarios) == 2:
                    raise RuntimeError("controlled interruption after episode 1")
                return original_apply(simulator, scenario_entry)

            with mock.patch.object(
                Simulator,
                "apply_scenario_entry",
                new=interrupt_before_second_episode,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "controlled interruption after episode 1"
                ):
                    main(command)

            run_dir = next((output / "td3_ratio").iterdir())
            manifest = json.loads(
                (run_dir / "scenario_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                attempted_scenarios,
                [
                    manifest["episodes"][0]["scenario_id"],
                    manifest["episodes"][1]["scenario_id"],
                ],
            )
            ep1_checkpoint = run_dir / "checkpoints" / "full" / "ep_0001"
            self.assertTrue((ep1_checkpoint / "training_state.pt").is_file())
            ep1_payload = torch.load(
                ep1_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            ep1_state = ep1_payload["training_state"]
            self.assertEqual(ep1_state["next_episode_index"], 1)
            self.assertEqual(ep1_state["total_joint_transitions"], 2)
            self.assertEqual(ep1_state["global_routing_slot"], 8)
            self.assertEqual(len(ep1_state["routing_epsilon_log"]), 8)
            self.assertEqual(ep1_state["td3_noise_log"], [])
            self.assertEqual(ep1_payload["replay_metadata"]["joint"]["size"], 2)
            with np.load(
                ep1_checkpoint / "joint_replay.npz", allow_pickle=False
            ) as ep1_replay:
                ep1_joint_replay = {
                    name: ep1_replay[name].copy() for name in ep1_replay.files
                }

            history_path = run_dir / "training_history.jsonl"
            history_before = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual([row["episode"] for row in history_before], [1])
            self.assertEqual(
                json.loads(
                    (run_dir / "resolved_config.json").read_text(encoding="utf-8")
                )["status"],
                "FAILED",
            )

            resumed_scenarios = []

            def record_resumed_scenario(simulator, scenario_entry):
                resumed_scenarios.append(str(scenario_entry["scenario_id"]))
                return original_apply(simulator, scenario_entry)

            with mock.patch.object(
                Simulator,
                "apply_scenario_entry",
                new=record_resumed_scenario,
            ):
                self.assertEqual(main(["resume", str(run_dir)]), 0)

            self.assertEqual(
                resumed_scenarios,
                [
                    manifest["episodes"][1]["scenario_id"],
                    manifest["episodes"][2]["scenario_id"],
                ],
            )
            history = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual([row["episode"] for row in history], [1, 2, 3])
            self.assertEqual(history[0], history_before[0])
            self.assertTrue(
                all(row["dinkelbach_lambda_used"] is None for row in history)
            )
            self.assertTrue(
                all(
                    row["dinkelbach_lambda_after_episode"] is None
                    for row in history
                )
            )
            self.assertTrue(
                all(
                    row["dinkelbach_update_status"]
                    == "disabled_for_reward_mode"
                    for row in history
                )
            )

            final_checkpoint = run_dir / "checkpoints" / "full" / "ep_0003"
            final_payload = torch.load(
                final_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            final_state = final_payload["training_state"]
            self.assertEqual(final_state["next_episode_index"], 3)
            self.assertEqual(final_state["total_joint_transitions"], 6)
            self.assertEqual(final_state["global_routing_slot"], 24)
            self.assertEqual(final_state["ddqn_schedule_slot"], 24)
            self.assertEqual(final_state["td3_post_warmup_transition"], 0)
            self.assertEqual(len(final_state["routing_epsilon_log"]), 24)
            self.assertEqual(final_state["td3_noise_log"], [])
            self.assertEqual(final_state["lambda_used_log"], [None, None, None])
            self.assertEqual(
                final_state["lambda_after_episode_log"], [None, None, None]
            )
            self.assertFalse(final_state["dinkelbach_active"])
            self.assertEqual(final_payload["replay_metadata"]["joint"]["size"], 6)
            self.assertEqual(final_payload["formal_config"]["total_episodes"], 3)

            with np.load(
                final_checkpoint / "joint_replay.npz", allow_pickle=False
            ) as replay:
                for name, values in ep1_joint_replay.items():
                    np.testing.assert_array_equal(replay[name][:2], values)
                not_done = replay["not_done"][:, 0]
                objectives = replay["ratio_objective_reward"][:, 0]
                delivered = replay["delivered_mbits"][:, 0]
                energy = replay["total_mobility_energy"][:, 0]
                shaping = (
                    not_done * replay["phi_search_t1"][:, 0]
                    - replay["phi_search_t"][:, 0]
                    + not_done * replay["phi_vs_t1"][:, 0]
                    - replay["phi_vs_t"][:, 0]
                    + not_done * replay["phi_com_t1"][:, 0]
                    - replay["phi_com_t"][:, 0]
                )
            np.testing.assert_array_equal(not_done, [1.0, 0.0] * 3)
            np.testing.assert_array_equal(objectives[::2], np.zeros(3))
            for episode_index, row in enumerate(history):
                start = episode_index * 2
                stop = start + 2
                expected_terminal_ratio = float(
                    delivered[start:stop].sum() / energy[start:stop].sum()
                )
                self.assertAlmostEqual(
                    float(objectives[stop - 1]),
                    expected_terminal_ratio,
                    places=6,
                )
                self.assertAlmostEqual(
                    float((objectives[start:stop] + shaping[start:stop]).sum()),
                    row["reward"],
                    places=5,
                )

            metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(Path(metadata["resume_checkpoint"]).name, "ep_0001")
            self.assertEqual(metadata["resume_episode"], 1)
            self.assertEqual(metadata["history_rows"], 3)
            self.assertEqual(metadata["dinkelbach_update_count"], 0)
            self.assertEqual(
                metadata["dinkelbach_state"],
                {"active": False, "update_count": 0},
            )
            self.assertIsNone(metadata["resume_reconciliation"])
            self.assertEqual(metadata["formal_config"]["total_episodes"], 3)
            self.assertEqual(metadata["training_config"]["total_episodes"], 3)

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
