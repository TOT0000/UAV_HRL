import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiment_config import MethodSpec, effective_training_config
from HRL_task_aware import TrainingConfig, train
from scenario_manifest import generate_manifest


NEW_PAPER_METHODS = (
    "km_ddpg_dinkelbach",
    "ddpg_dinkelbach_wo_ta",
    "td3_dinkelbach_random_routing",
    "td3_dinkelbach_dqn_wo_ta",
)


class PaperMethodSmokeTest(unittest.TestCase):
    def _training_config(self, root):
        return TrainingConfig(
            total_episodes=1,
            mode="train",
            episode_seconds=1,
            routing_slot_seconds=0.25,
            warmup_joint_transitions=0,
            batch_size=1,
            model_checkpoint_every=1,
            full_resume_every=1,
            checkpoint_root=str(root),
            enable_model_checkpoints=True,
            enable_full_resume=True,
            enable_plots=False,
            enable_csv=False,
            random_seed=20260817,
        )

    def test_four_methods_train_and_model_checkpoint_round_trip(self):
        for method_id in NEW_PAPER_METHODS:
            with self.subTest(method=method_id), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "checkpoints"
                method = MethodSpec.parse(method_id)
                training_manifest = generate_manifest("train", 7123, 1, num_gt=2)
                config = self._training_config(root)
                training = train(
                    config,
                    scenario_manifest=training_manifest,
                    method_spec=method,
                )
                checkpoint = root / "models" / "ep_0001"
                full = root / "full" / "ep_0001"
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertEqual(metadata["movement_agent_kind"], method.agent)
                self.assertEqual(metadata["routing_agent_kind"], method.routing)
                self.assertEqual(
                    metadata["experiment"]["task_observation_mode"],
                    method.task_observation,
                )

                networks = torch.load(
                    checkpoint / "models.pt", map_location="cpu", weights_only=False
                )
                if method.agent == "ddpg":
                    self.assertIn("critic", networks["movement_agent"])
                    self.assertNotIn("critic_1", networks["movement_agent"])
                    self.assertNotIn("critic_2", networks["movement_agent"])
                if method.routing == "random":
                    self.assertEqual(networks["routing_agent"], {"kind": "random"})
                    self.assertFalse((full / "routing_replay.npz").exists())
                    self.assertEqual(training["routing_replay_size"], 0)
                elif method.routing == "dqn":
                    self.assertNotIn("cost_network", networks["routing_agent"])

                evaluation_manifest = generate_manifest(
                    "validation", 8123, 1, num_gt=2
                )
                evaluation = train(
                    TrainingConfig(
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
                        random_seed=20260817,
                    ),
                    scenario_manifest=evaluation_manifest,
                    method_spec=method,
                    evaluation=True,
                    checkpoint_dir=checkpoint,
                    expected_checkpoint_episodes=1,
                    expected_checkpoint_formal_config=effective_training_config(
                        config, method
                    ),
                    trajectory_snapshot_times=(1.0,)
                    if method.routing == "random"
                    else None,
                    trajectory_target_uav_id=0
                    if method.routing == "random"
                    else None,
                )
                self.assertTrue(all(evaluation["evaluation_invariants"].values()))
                self.assertEqual(evaluation["movement_agent_kind"], method.agent)
                self.assertEqual(evaluation["routing_agent_kind"], method.routing)
                for field in (
                    "checkpoint_metadata_fingerprint",
                    "checkpoint_models_sha256",
                    "checkpoint_artifact_fingerprint",
                ):
                    self.assertEqual(len(evaluation["run_metadata"][field]), 64)
                if method.routing == "random":
                    artifact = evaluation["trajectory_artifacts"][0]
                    self.assertEqual(artifact["requested_times_seconds"], [1.0])
                    self.assertEqual(
                        artifact["snapshots"][0]["requested_time_seconds"], 1.0
                    )
                    self.assertGreaterEqual(
                        artifact["snapshots"][0]["actual_time_seconds"], 1.0
                    )
                    self.assertEqual(len(artifact["snapshots"][0]["uavs"]), 16)
                    self.assertEqual(artifact["target_uav_id"], 0)
                    self.assertEqual(artifact["method_id"], method_id)
                    self.assertIn("uav_paths", artifact)
                    self.assertIn("sr_paths", artifact)
                    self.assertIn("ground_station", artifact["snapshots"][0])
                    self.assertIn("active_links", artifact["snapshots"][0])
                    self.assertIn("sensing_coverage", artifact["snapshots"][0])
                    self.assertIn("ground_targets", artifact)
                    self.assertIn("initial_sr_teams", artifact)
                    for field in (
                        "checkpoint_metadata_fingerprint",
                        "checkpoint_models_sha256",
                        "checkpoint_artifact_fingerprint",
                    ):
                        self.assertEqual(
                            artifact[field], evaluation["run_metadata"][field]
                        )


if __name__ == "__main__":
    unittest.main()
