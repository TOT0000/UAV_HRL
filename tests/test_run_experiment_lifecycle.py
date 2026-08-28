import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from observation_strategy import MOVEMENT_TASK_ASSIGNMENT_INDICES
from evaluation_selection import resolve_training_run_checkpoint
from run_experiment import main
from scenario_manifest import ScenarioManifest, manifest_prefix
from Simulator import Simulator


class SimpleRunnerLifecycleIntegrationTest(unittest.TestCase):
    def test_explicit_horizon_extension_preserves_old_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            with mock.patch("run_experiment._git_short_sha", return_value="sha-A"):
                self.assertEqual(
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
                    ),
                    0,
                )
            run_dir = next((output / "td3_ratio").iterdir())
            original_manifest_path = run_dir / "scenario_manifest.json"
            original_manifest_bytes = original_manifest_path.read_bytes()
            original_manifest = ScenarioManifest.load(original_manifest_path)
            old_checkpoint = run_dir / "checkpoints" / "models" / "ep_0001"
            old_models_bytes = (old_checkpoint / "models.pt").read_bytes()
            old_metadata_bytes = (old_checkpoint / "metadata.json").read_bytes()
            history_before = [
                json.loads(line)
                for line in (run_dir / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            evaluation_marker = run_dir / "evaluation" / "existing.txt"
            evaluation_marker.parent.mkdir()
            evaluation_marker.write_text("preserve", encoding="utf-8")

            with mock.patch("run_experiment._git_short_sha", return_value="sha-B"):
                self.assertEqual(
                    main(["resume", str(run_dir), "--target-episodes", "3"]), 0
                )

            resolved = json.loads(
                (run_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            active_manifest = ScenarioManifest.load(
                run_dir / resolved["training_manifest_path"]
            )
            self.assertEqual(resolved["episodes"], 3)
            self.assertEqual(resolved["training_config"]["total_episodes"], 3)
            self.assertEqual(active_manifest.episode_count, 3)
            self.assertEqual(
                manifest_prefix(active_manifest, 1).content_hash,
                original_manifest.content_hash,
            )
            provenance = resolved["horizon_extension_provenance"]
            self.assertEqual(provenance["previous_total_episodes"], 1)
            self.assertEqual(provenance["target_total_episodes"], 3)
            self.assertEqual(provenance["preserved_prefix_length"], 1)
            self.assertEqual(provenance["extension_git_sha"], "sha-B")
            self.assertEqual(resolved["initial_training_git_sha"], "sha-A")
            self.assertEqual(resolved["latest_training_git_sha"], "sha-B")
            self.assertEqual(resolved["git_sha"], "sha-B")
            self.assertEqual(
                resolved["training_history_identity_manifest_hash"],
                original_manifest.content_hash,
            )
            self.assertEqual(
                resolved["training_history_manifest_hash"],
                resolved["training_history_identity_manifest_hash"],
            )
            self.assertIn(
                "not_active_manifest",
                resolved["training_history_manifest_hash_semantics"],
            )
            segments = resolved["training_manifest_segments"]
            self.assertEqual(
                [(item["episode_start"], item["episode_end"]) for item in segments],
                [(1, 1), (2, 3)],
            )
            self.assertEqual(segments[0]["manifest_hash"], original_manifest.content_hash)
            self.assertEqual(segments[1]["manifest_hash"], active_manifest.content_hash)
            self.assertEqual(
                segments[1]["parent_manifest_hash"], original_manifest.content_hash
            )
            self.assertEqual(original_manifest_path.read_bytes(), original_manifest_bytes)
            self.assertEqual((old_checkpoint / "models.pt").read_bytes(), old_models_bytes)
            self.assertEqual(
                (old_checkpoint / "metadata.json").read_bytes(), old_metadata_bytes
            )
            self.assertEqual(evaluation_marker.read_text(encoding="utf-8"), "preserve")
            self.assertTrue(
                (run_dir / "checkpoints" / "models" / "ep_0002").is_dir()
            )
            self.assertTrue(
                (run_dir / "checkpoints" / "models" / "ep_0003").is_dir()
            )
            self.assertTrue(
                (run_dir / "checkpoints" / "full" / "ep_0003").is_dir()
            )
            history_after = [
                json.loads(line)
                for line in (run_dir / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual([row["episode"] for row in history_after], [1, 2, 3])
            self.assertEqual(history_after[0], history_before[0])
            self.assertEqual(
                {row["training_manifest_hash"] for row in history_after},
                {original_manifest.content_hash},
            )
            checkpoint_three_metadata = json.loads(
                (
                    run_dir
                    / "checkpoints"
                    / "full"
                    / "ep_0003"
                    / "metadata.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_three_metadata["experiment"]["latest_training_git_sha"],
                "sha-B",
            )
            self.assertEqual(
                checkpoint_three_metadata["training_provenance"]["training_git_sha"],
                "sha-B",
            )

            old_context = resolve_training_run_checkpoint(run_dir, 1)
            self.assertEqual(old_context["checkpoint_planned_total_episodes"], 1)
            self.assertEqual(old_context["current_training_run_total_episodes"], 3)
            self.assertTrue(old_context["horizon_extension_compatible"])
            self.assertEqual(
                old_context["allowed_horizon_differences"], ["total_episodes"]
            )
            self.assertTrue(old_context["manifest_prefix_compatible"])
            new_context = resolve_training_run_checkpoint(run_dir, 3)
            self.assertEqual(new_context["checkpoint_planned_total_episodes"], 3)
            self.assertFalse(new_context["horizon_extension_compatible"])
            self.assertEqual(new_context["allowed_horizon_differences"], [])

            self.assertEqual(
                main(
                    [
                        "evaluate",
                        str(run_dir),
                        "--checkpoint-episode",
                        "1",
                        "--smoke",
                    ]
                ),
                0,
            )
            evaluation_dir = next(
                (run_dir / "evaluation" / "ep_1").iterdir()
            )
            evaluation_metadata = json.loads(
                (evaluation_dir / "run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                evaluation_metadata["checkpoint_planned_total_episodes"], 1
            )
            self.assertEqual(
                evaluation_metadata["current_training_run_total_episodes"], 3
            )
            self.assertTrue(
                evaluation_metadata["horizon_extension_compatible"]
            )
            self.assertTrue(evaluation_metadata["manifest_prefix_compatible"])

            with mock.patch("run_experiment._git_short_sha", return_value="sha-C"):
                self.assertEqual(
                    main(["resume", str(run_dir), "--target-episodes", "5"]), 0
                )
            repeated = json.loads(
                (run_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(repeated["initial_training_git_sha"], "sha-A")
            self.assertEqual(repeated["latest_training_git_sha"], "sha-C")
            self.assertEqual(repeated["git_sha"], "sha-C")
            repeated_run_metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                repeated_run_metadata["initial_training_git_sha"], "sha-A"
            )
            self.assertEqual(
                repeated_run_metadata["latest_training_git_sha"], "sha-C"
            )
            self.assertEqual(repeated_run_metadata["git_sha"], "sha-C")
            self.assertEqual(
                [record["extension_git_sha"] for record in repeated["horizon_extension_history"]],
                ["sha-B", "sha-C"],
            )
            repeated_segments = repeated["training_manifest_segments"]
            self.assertEqual(
                [
                    (segment["episode_start"], segment["episode_end"])
                    for segment in repeated_segments
                ],
                [(1, 1), (2, 3), (4, 5)],
            )
            self.assertEqual(
                repeated_segments[2]["parent_manifest_hash"],
                repeated_segments[1]["manifest_hash"],
            )
            checkpoint_five_metadata = json.loads(
                (
                    run_dir
                    / "checkpoints"
                    / "full"
                    / "ep_0005"
                    / "metadata.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_five_metadata["experiment"]["initial_training_git_sha"],
                "sha-A",
            )
            self.assertEqual(
                checkpoint_five_metadata["experiment"]["latest_training_git_sha"],
                "sha-C",
            )
            self.assertEqual(
                [
                    record["extension_git_sha"]
                    for record in checkpoint_five_metadata["experiment"][
                        "horizon_extension_history"
                    ]
                ],
                ["sha-B", "sha-C"],
            )

            with self.assertRaisesRegex(ValueError, "planned training horizon"):
                main(["resume", str(run_dir), "--target-episodes", "5"])

    def test_masked_td3_exact_resume_preserves_true_projection_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            command = [
                "td3_dinkelbach_wo_ta",
                "--episodes", "2",
                "--episode-seconds", "1",
                "--checkpoint-interval", "1",
                "--roi-count", "2",
                "--output-root", str(output),
            ]
            original_apply = Simulator.apply_scenario_entry
            calls = []

            def interrupt(simulator, scenario_entry):
                calls.append(str(scenario_entry["scenario_id"]))
                if len(calls) == 2:
                    raise RuntimeError("masked TD3 interruption")
                return original_apply(simulator, scenario_entry)

            with mock.patch.object(Simulator, "apply_scenario_entry", new=interrupt):
                with self.assertRaisesRegex(RuntimeError, "masked TD3 interruption"):
                    main(command)

            run_dir = next((output / "td3_dinkelbach_wo_ta").iterdir())
            first = run_dir / "checkpoints" / "full" / "ep_0001"
            with np.load(first / "joint_replay.npz", allow_pickle=False) as replay:
                self.assertEqual(replay["current_movement_mask"].shape, (1, 10))
                self.assertTrue(replay["movement_mask_valid"].all())
                np.testing.assert_array_equal(
                    replay["state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)],
                    0.0,
                )

            self.assertEqual(main(["resume", str(run_dir)]), 0)
            final = run_dir / "checkpoints" / "full" / "ep_0002"
            with np.load(final / "joint_replay.npz", allow_pickle=False) as replay:
                self.assertEqual(replay["current_movement_mask"].shape, (2, 10))
                self.assertEqual(replay["next_movement_mask"].shape, (2, 10))
                self.assertTrue(replay["movement_mask_valid"].all())
                self.assertTrue(replay["current_movement_mask"].any())
                np.testing.assert_array_equal(
                    replay["state"][:, list(MOVEMENT_TASK_ASSIGNMENT_INDICES)],
                    0.0,
                )
            history = [
                json.loads(line)
                for line in (run_dir / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual([row["episode"] for row in history], [1, 2])

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
            ep1_constraint_state = ep1_payload["ddqn_state"][
                "constraint_state"
            ]
            self.assertEqual(ep1_state["next_episode_index"], 1)
            self.assertEqual(ep1_state["total_joint_transitions"], 2)
            self.assertEqual(ep1_state["global_routing_slot"], 8)
            self.assertEqual(len(ep1_state["routing_epsilon_log"]), 8)
            self.assertEqual(ep1_state["lambda_cost_used_log"], [0.0])
            self.assertEqual(len(ep1_state["lambda_cost_after_episode_log"]), 1)
            self.assertGreaterEqual(
                ep1_state["lambda_cost_after_episode_log"][0], 0.0
            )
            self.assertIn("fov_ema_state", ep1_state)
            self.assertEqual(ep1_state["td3_noise_log"], [])
            self.assertEqual(
                ep1_constraint_state["cost_multiplier_update_count"], 1
            )
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
            self.assertEqual(len(final_state["lambda_cost_used_log"]), 3)
            self.assertEqual(len(final_state["lambda_cost_after_episode_log"]), 3)
            self.assertEqual(
                final_state["lambda_cost_used_log"][1],
                ep1_state["lambda_cost_after_episode_log"][0],
            )
            # Assigned FOV sources are immediately QoS eligible under the
            # current packet contract, so each resumed episode contributes a
            # valid constraint update instead of being skipped for lack of
            # full-coverage FOV packets.
            self.assertEqual(
                final_payload["ddqn_state"]["constraint_state"][
                    "cost_multiplier_update_count"
                ],
                3,
            )
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
                    delivered[start:stop].sum() * 1e6 / energy[start:stop].sum()
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

    def test_controlled_dqn_partial_resume_preserves_epsilon_and_target_counters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            command = [
                "td3_dinkelbach_dqn",
                "--episodes", "3",
                "--episode-seconds", "1",
                "--checkpoint-interval", "1",
                "--roi-count", "2",
                "--output-root", str(output),
            ]
            original_apply = Simulator.apply_scenario_entry
            calls = []

            def interrupt(simulator, scenario_entry):
                calls.append(str(scenario_entry["scenario_id"]))
                if len(calls) == 2:
                    raise RuntimeError("controlled DQN interruption")
                return original_apply(simulator, scenario_entry)

            with mock.patch.object(
                Simulator, "apply_scenario_entry", new=interrupt
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "controlled DQN interruption"
                ):
                    main(command)

            run_dir = next((output / "td3_dinkelbach_dqn").iterdir())
            ep1 = run_dir / "checkpoints" / "full" / "ep_0001"
            self.assertTrue(ep1.is_dir())
            payload1 = torch.load(
                ep1 / "training_state.pt", map_location="cpu", weights_only=False
            )
            routing1 = payload1["routing_agent_state"]
            self.assertEqual(routing1["kind"], "dqn")
            self.assertEqual(routing1["target_update_count"], 0)
            self.assertEqual(payload1["training_state"]["global_routing_slot"], 4)
            self.assertEqual(
                len(payload1["training_state"]["routing_epsilon_log"]), 4
            )

            resumed = []

            def record(simulator, scenario_entry):
                resumed.append(str(scenario_entry["scenario_id"]))
                return original_apply(simulator, scenario_entry)

            with mock.patch.object(
                Simulator, "apply_scenario_entry", new=record
            ):
                self.assertEqual(main(["resume", str(run_dir)]), 0)

            manifest = json.loads(
                (run_dir / "scenario_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resumed,
                [
                    manifest["episodes"][1]["scenario_id"],
                    manifest["episodes"][2]["scenario_id"],
                ],
            )
            final = run_dir / "checkpoints" / "full" / "ep_0003"
            payload3 = torch.load(
                final / "training_state.pt", map_location="cpu", weights_only=False
            )
            routing3 = payload3["routing_agent_state"]
            state3 = payload3["training_state"]
            self.assertEqual(routing3["kind"], "dqn")
            self.assertEqual(routing3["target_update_count"], 0)
            self.assertEqual(routing3["training_updates"], 0)
            self.assertEqual(state3["global_routing_slot"], 12)
            self.assertEqual(state3["ddqn_schedule_slot"], 12)
            self.assertEqual(len(state3["routing_epsilon_log"]), 12)
            self.assertEqual(state3["routing_epsilon_log"], [1.0] * 12)
            self.assertEqual(
                state3["routing_lifecycle_state"][
                    "routing_optimizer_update_count"
                ],
                0,
            )
            np.testing.assert_array_equal(
                state3["routing_epsilon_log"][:4],
                payload1["training_state"]["routing_epsilon_log"],
            )
            history = [
                json.loads(line)
                for line in (run_dir / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual([row["episode"] for row in history], [1, 2, 3])
            metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["routing_policy"], "dqn")
            self.assertEqual(Path(metadata["resume_checkpoint"]).name, "ep_0001")
            self.assertEqual(metadata["history_rows"], 3)
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
