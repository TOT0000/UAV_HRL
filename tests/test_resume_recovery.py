from dataclasses import asdict
from functools import partial
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from HRL_task_aware import formal_training_config
from experiment_config import (
    FOV_EMA_LIFECYCLE_VERSION,
    MethodSpec,
    SR_ROUTE_LIFECYCLE_VERSION,
)
from resume_recovery import (
    execute_resume_reconciliation,
    plan_resume_reconciliation,
)
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    FULL_CHECKPOINT_TYPE,
    FULL_RESUME_LOGGING_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    _atomic_checkpoint_write,
    calibration_fingerprint,
    inspect_full_resume_checkpoint,
    inspect_model_checkpoint,
)


class ResumeRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.calibration = {"c_ref_com": 12.5, "seed": 7}
        self.method = MethodSpec()
        self.formal_config = asdict(
            formal_training_config(2500, random_seed=101)
        )
        self.experiment = {
            "method_spec_fingerprint": self.method.fingerprint,
            "manifest_hash": "a" * 64,
            "training_seed": 101,
            "formal_config": self.formal_config,
            **dinkelbach_config_metadata(self.formal_config),
        }

    def _dinkelbach_state(self, completed_episode):
        state = DinkelbachBlockState.from_config(self.formal_config)
        for _ in range(completed_episode):
            state.record_episode(1.0, 2.0)
        return state.training_state()

    def _training_state(self, completed_episode):
        state = DinkelbachBlockState.from_config(self.formal_config)
        lambda_used_log = []
        lambda_after_episode_log = []
        for _ in range(completed_episode):
            event = state.record_episode(1.0, 2.0)
            lambda_used_log.append(event["dinkelbach_lambda_used"])
            lambda_after_episode_log.append(
                event["dinkelbach_lambda_after_episode"]
            )
        return {
            "completed_episode_index": completed_episode - 1,
            "next_episode_index": completed_episode,
            "full_resume_logging_schema_version": (
                FULL_RESUME_LOGGING_SCHEMA_VERSION
            ),
            "reward_log": [0.0] * completed_episode,
            "delivered_log": [1.0] * completed_episode,
            "energy_log": [2.0] * completed_episode,
            "lambda_used_log": lambda_used_log,
            "lambda_after_episode_log": lambda_after_episode_log,
            "lambda_cost_used_log": [0.0] * completed_episode,
            "lambda_cost_after_episode_log": [0.0] * completed_episode,
            "fov_ema_state": {
                "lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
                "values": {},
                "initialized_uav_ids": [],
                "previous_footprints": {},
                "transition_marker": None,
                "update_count": 0,
            },
            "sr_route_state": {
                "lifecycle_version": SR_ROUTE_LIFECYCLE_VERSION,
                "teams": [],
                "trajectory": {},
                "checkpoint_scope": "episode_boundary_terminal_snapshot",
                "mid_episode_checkpoint_supported": False,
            },
            **state.training_state(),
        }

    def _metadata(self, checkpoint_type, completed_episode):
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": checkpoint_type,
            "episode": completed_episode - 1,
            "movement_state_dim": 532,
            "joint_action_dim": 48,
            "routing_state_dim": 126,
            "centralized_td3_gamma": 1.0,
            "routing_ddqn_gamma": 0.99,
            "routing_agent_kind": "safe_ddqn",
            "routing_agent_configuration": {
                "lambda_cost": 0.0,
                "initial_lambda_cost": 0.0,
                "eta_c": 0.01,
                "qos_cost_budget": 12.0,
                "lambda_update_scope": "episode_end",
                "cost_denominator": "network_routing_slot_steps",
                "mid_episode_checkpoint_supported": False,
            },
            "com_calibration_fingerprint": calibration_fingerprint(
                self.calibration
            ),
            "experiment": {
                **self.experiment,
                "lambda_ee": self._dinkelbach_state(completed_episode)[
                    "current_dinkelbach_lambda"
                ],
                "dinkelbach_state": self._dinkelbach_state(completed_episode),
            },
        }

    def _full(self, run_dir, completed_episode, *, complete=True):
        path = (
            Path(run_dir)
            / "checkpoints"
            / "full"
            / f"ep_{completed_episode:04d}"
        )
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(
            json.dumps(self._metadata(FULL_CHECKPOINT_TYPE, completed_episode)),
            encoding="utf-8",
        )
        if complete:
            torch.save(
                {
                    "training_state": self._training_state(completed_episode),
                    "formal_config": self.formal_config,
                },
                path / "training_state.pt",
            )
            np.savez_compressed(
                path / "joint_replay.npz",
                current_movement_mask=np.zeros((0, 16), dtype=bool),
                next_movement_mask=np.zeros((0, 16), dtype=bool),
                movement_mask_valid=np.zeros((0, 1), dtype=bool),
            )
            (path / "routing_replay.npz").write_bytes(b"routing")
        return path

    def _model(self, run_dir, completed_episode, *, valid=True):
        path = (
            Path(run_dir)
            / "checkpoints"
            / "models"
            / f"ep_{completed_episode:04d}"
        )
        path.mkdir(parents=True)
        if valid:
            (path / "metadata.json").write_text(
                json.dumps(
                    self._metadata(MODEL_CHECKPOINT_TYPE, completed_episode)
                ),
                encoding="utf-8",
            )
            (path / "models.pt").write_bytes(b"models")
        return path

    def _inspectors(self):
        common = {
            "movement_state_dim": 532,
            "joint_action_dim": 48,
            "routing_state_dim": 126,
            "td3_gamma": 1.0,
            "ddqn_gamma": 0.99,
            "calibration": self.calibration,
            "expected_experiment_metadata": {
                "method_spec_fingerprint": self.method.fingerprint,
                "manifest_hash": "a" * 64,
                "training_seed": 101,
            },
            "expected_formal_config": self.formal_config,
            "require_episode_directory": True,
        }
        return (
            partial(inspect_full_resume_checkpoint, **common),
            partial(inspect_model_checkpoint, **common),
        )

    def _plan(self, run_dir, selected):
        inspect_full, inspect_model = self._inspectors()
        return plan_resume_reconciliation(
            run_dir,
            selected,
            inspect_full=inspect_full,
            inspect_model=inspect_model,
            transaction_id="txn-001",
            timestamp="2026-08-17T00:00:00+00:00",
        )

    def test_stale_model_is_moved_and_checkpoint_names_become_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            selected = self._full(run_dir, 50)
            stale = self._model(run_dir, 100)

            plan = self._plan(run_dir, selected)
            recovery = execute_resume_reconciliation(plan)
            execute_resume_reconciliation(plan)

            quarantined = (
                Path(recovery["recovery_directory"]) / "models" / "ep_0100"
            )
            self.assertFalse(stale.exists())
            self.assertTrue(quarantined.is_dir())
            self.assertTrue(selected.is_dir())
            manifest = json.loads(
                Path(recovery["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertEqual(manifest["resume_checkpoint"], str(selected.resolve()))
            self.assertEqual(manifest["resume_episode"], 50)
            self.assertEqual(manifest["transaction_id"], "txn-001")
            self.assertEqual(
                manifest["artifacts"][0]["checkpoint_metadata"]["episode"], 99
            )
            self.assertEqual(
                manifest["artifacts"][0]["original_path"], str(stale.resolve())
            )
            self.assertEqual(
                manifest["artifacts"][0]["quarantine_path"],
                str(quarantined.resolve()),
            )

            repeated = self._plan(run_dir, selected)
            self.assertEqual(repeated.stale_models, ())
            self.assertIsNone(execute_resume_reconciliation(repeated))

            for kind in ("models", "full"):
                target = run_dir / "checkpoints" / kind / "ep_0100"
                _atomic_checkpoint_write(
                    target, lambda temporary: (temporary / "saved").touch()
                )
                self.assertTrue(target.is_dir())

    def test_newer_valid_full_checkpoint_rejects_rollback_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            selected = self._full(run_dir, 50)
            newer = self._full(run_dir, 100)
            stale = self._model(run_dir, 100)

            with self.assertRaisesRegex(
                RuntimeError,
                "A newer valid full-resume checkpoint exists: ep_0100.*latest",
            ):
                self._plan(run_dir, selected)

            self.assertTrue(newer.is_dir())
            self.assertTrue(stale.is_dir())
            self.assertFalse((run_dir / "recovery").exists())

    def test_incomplete_full_and_invalid_or_temporary_models_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            selected = self._full(run_dir, 50)
            incomplete_full = self._full(run_dir, 100, complete=False)
            invalid_model = self._model(run_dir, 100, valid=False)
            hidden = run_dir / "checkpoints" / "models" / ".ep_0150.tmp"
            hidden.mkdir(parents=True)

            plan = self._plan(run_dir, selected)

            self.assertEqual(plan.stale_models, ())
            self.assertTrue(incomplete_full.is_dir())
            self.assertTrue(invalid_model.is_dir())
            self.assertTrue(hidden.is_dir())

    def test_reconciliation_never_scans_or_mutates_another_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            other = Path(temp_dir) / "other"
            selected = self._full(first, 50)
            own_stale = self._model(first, 100)
            other_stale = self._model(other, 100)

            execute_resume_reconciliation(self._plan(first, selected))

            self.assertFalse(own_stale.exists())
            self.assertTrue(other_stale.is_dir())
            self.assertFalse((other / "recovery").exists())

    def test_selected_checkpoint_directory_episode_must_match_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            selected = self._full(run_dir, 50)
            metadata_path = selected / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["episode"] = 99
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "directory episode"):
                self._plan(run_dir, selected)


if __name__ == "__main__":
    unittest.main()
