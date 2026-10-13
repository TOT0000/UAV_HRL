import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiment_config import MethodSpec
from HRL_task_aware import TrainingConfig, train
from rnd_exploration import RNDConfig, RNDExplorer, RND_METHOD_ID, RunningMeanVariance
from run_rnd_sampling import _completed_output_is_valid
from scenario_manifest import generate_manifest
from utils_update_v2 import ReplayBufferJoint


class RNDCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = RNDConfig(
            eta=0.5,
            gamma_rnd=0.5,
            learning_rate=1e-3,
            predictor_batch_size=1,
            initialization_seed=17,
        )

    def test_target_is_fixed_and_predictor_learns_fixed_input(self):
        global_rng_before = torch.get_rng_state().clone()
        rnd = RNDExplorer(3, self.config, device="cpu")
        self.assertTrue(torch.equal(global_rng_before, torch.get_rng_state()))
        target_before = {
            key: value.detach().clone() for key, value in rnd.target.state_dict().items()
        }
        observation = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
        rnd.observation_stats.update(observation.reshape(1, -1))
        before = float(rnd.raw_error(observation.reshape(1, -1)).item())
        for _ in range(150):
            rnd.pending_observations.append(observation.copy())
            rnd.train_pending()
        after = float(rnd.raw_error(observation.reshape(1, -1)).item())
        self.assertLess(after, before)
        for key, value in rnd.target.state_dict().items():
            self.assertTrue(torch.equal(value, target_before[key]))
        self.assertTrue(all(parameter.grad is None for parameter in rnd.target.parameters()))

    def test_running_statistics_and_causal_q_reward_are_finite(self):
        stats = RunningMeanVariance((2,))
        stats.update(np.asarray([[1.0, 5.0], [3.0, 5.0]], dtype=np.float64))
        np.testing.assert_allclose(stats.mean, [2.0, 5.0])
        np.testing.assert_allclose(stats.variance, [1.0, 0.0])

        rnd = RNDExplorer(2, self.config, device="cpu")
        original = np.asarray([1.0, 1.0], dtype=np.float32)
        first = rnd.observe(original, queue_for_predictor=False)
        np.testing.assert_array_equal(original, [1.0, 1.0])
        self.assertAlmostEqual(
            first["q_value"], first["raw_error"], places=7
        )
        self.assertAlmostEqual(first["q_variance_previous"], 1.0)
        second = rnd.observe(original, queue_for_predictor=False)
        self.assertAlmostEqual(
            second["q_value"],
            0.5 * first["q_value"] + second["raw_error"],
            places=7,
        )
        for record in (first, second):
            self.assertTrue(
                np.isfinite(
                    [
                        record["raw_error"],
                        record["normalized_bonus"],
                        record["weighted_bonus"],
                        record["q_value"],
                        record["q_variance_previous"],
                    ]
                ).all()
            )

    def test_replay_rescoring_does_not_change_original_reward_or_statistics(self):
        replay = ReplayBufferJoint(2, 3, max_size=4, rng=np.random.default_rng(3))
        replay.add(
            [0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0], False,
            delivered_mbits=7.0, total_mobility_energy=2.0,
            c9_penalty=0.25, c10_penalty=0.5, com_range_penalty=0.75,
        )
        original = replay._reward_numpy(
            np.asarray([0]), current_lambda=0.5, gamma=1.0
        ).copy()
        rnd = RNDExplorer(2, self.config, device="cpu")
        rnd.observe([2.0, 3.0], queue_for_predictor=False)
        stats_before = copy.deepcopy(rnd.state_dict(full=True))
        bonus = rnd.intrinsic_reward(torch.tensor([[2.0, 3.0]]))
        self.assertTrue(torch.isfinite(bonus).all())
        np.testing.assert_array_equal(
            replay._reward_numpy(np.asarray([0]), current_lambda=0.5, gamma=1.0),
            original,
        )
        self.assertEqual(rnd.observation_stats.count, stats_before["observation_stats"]["count"])
        self.assertEqual(rnd.q_stats.count, stats_before["q_stats"]["count"])

    def test_full_checkpoint_state_restores_optimizer_statistics_and_pending_batch(self):
        rnd = RNDExplorer(3, self.config, device="cpu")
        rnd.observe([1.0, 2.0, 3.0], queue_for_predictor=True)
        rnd.train_pending(force=True)
        rnd.observe([2.0, 3.0, 4.0], queue_for_predictor=True)
        state = rnd.state_dict(full=True)
        restored = RNDExplorer(3, self.config, device="cpu")
        restored.load_state_dict(state, require_full=True)
        self.assertEqual(restored.fingerprint(), rnd.fingerprint())
        self.assertEqual(len(restored.pending_observations), 1)
        self.assertTrue(restored.optimizer.state_dict()["state"])
        with self.assertRaisesRegex(RuntimeError, "dimension"):
            RNDExplorer(4, self.config, device="cpu").load_state_dict(
                state, require_full=True
            )
        incomplete = dict(state)
        incomplete.pop("optimizer")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            restored.load_state_dict(incomplete, require_full=True)

    def test_non_rnd_method_parse_consumes_no_global_rng_and_has_no_rnd_configuration(self):
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        method = MethodSpec.parse("td3_dinkelbach")
        numpy_after = np.random.get_state()
        self.assertEqual(method.method_id, "td3_dinkelbach")
        self.assertNotEqual(method.method_id, RND_METHOD_ID)
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_non_rnd_training_never_constructs_rnd(self):
        config = TrainingConfig(
            total_episodes=1,
            mode="smoke",
            episode_seconds=1,
            warmup_joint_transitions=0,
            routing_warmup_transitions=1,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=41,
        )
        with mock.patch(
            "HRL_task_aware.RNDExplorer",
            side_effect=AssertionError("baseline attempted to construct RND"),
        ):
            result = train(
                config,
                scenario_manifest=generate_manifest("train", 41, 1, num_gt=2),
                method_spec=MethodSpec.parse("td3_dinkelbach"),
            )
        self.assertIsNone(result["rnd"])

    def test_completed_sampling_work_is_skipped_only_after_integrity_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            replay = output / "joint_replay.npz"
            diagnostics = output / "rnd_diagnostics.npz"
            np.savez_compressed(replay, state=np.zeros((2, 3), dtype=np.float32))
            np.savez_compressed(
                diagnostics, raw_error=np.zeros((2,), dtype=np.float32)
            )
            hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (replay, diagnostics)
            }
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "checkpoint_completed_episodes": 100,
                        "manifest_hash": "manifest",
                        "completed_episodes": 1,
                        "transition_count": 2,
                        "file_sha256": hashes,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _completed_output_is_valid(
                    output,
                    checkpoint_episode=100,
                    manifest_hash="manifest",
                    episodes=1,
                )
            )
            diagnostics.write_bytes(diagnostics.read_bytes() + b"corrupt")
            self.assertFalse(
                _completed_output_is_valid(
                    output,
                    checkpoint_episode=100,
                    manifest_hash="manifest",
                    episodes=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
