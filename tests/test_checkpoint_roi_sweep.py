import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from checkpoint_roi_sweep import (
    SweepExecutionError,
    SweepPreflightError,
    build_checkpoint_roi_sweep_plan,
    execute_checkpoint_roi_sweep,
)
from evaluation_selection import (
    DEFAULT_FIXED_ROI_COUNTS,
    resolve_checkpoint_episodes,
    resolve_roi_counts,
    resolve_training_run_checkpoint,
)
from experiment_config import FORMAL_CHECKPOINT_EPISODE, MethodSpec
from paper_evaluation import (
    _load_training_run,
    evaluation_sweep_points,
    run_paper_evaluation,
)
from paper_metrics import aggregate_paper_point_metrics
from run_checkpoint_roi_sweep import build_parser as build_batch_parser
from run_paper_evaluation import build_parser as build_paper_parser
from scenario_manifest import generate_manifest


PROVENANCE = {
    "checkpoint_metadata_fingerprint": "a" * 64,
    "checkpoint_models_sha256": "b" * 64,
    "checkpoint_artifact_fingerprint": "c" * 64,
}


class CheckpointRoiSelectorTest(unittest.TestCase):
    def test_defaults_and_explicit_singular_or_batch_values(self):
        self.assertEqual(
            resolve_checkpoint_episodes(), (FORMAL_CHECKPOINT_EPISODE,)
        )
        self.assertEqual(resolve_roi_counts(), DEFAULT_FIXED_ROI_COUNTS)
        self.assertEqual(resolve_checkpoint_episodes(50), (50,))
        self.assertEqual(
            resolve_checkpoint_episodes(checkpoint_episodes=(50, 100, 50)),
            (50, 100),
        )
        self.assertEqual(resolve_roi_counts(roi_count=4), (4,))
        self.assertEqual(
            resolve_roi_counts(roi_counts=(2, 5, 8, 5)), (2, 5, 8)
        )
        self.assertEqual(
            tuple(
                point["fixed_num_gt"]
                for point in evaluation_sweep_points("fixed_roi")
            ),
            tuple(range(2, 9)),
        )
        self.assertEqual(
            tuple(
                point["fixed_num_gt"]
                for point in evaluation_sweep_points("fixed_roi", (3, 7))
            ),
            (3, 7),
        )

    def test_invalid_or_conflicting_selectors_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_checkpoint_episodes(50, (100,))
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_checkpoint_episodes(0)
        with self.assertRaisesRegex(ValueError, "inclusive range"):
            resolve_roi_counts(roi_counts=(1, 4, 9))
        with self.assertRaisesRegex(ValueError, "only for fixed_roi or environment_size"):
            evaluation_sweep_points("uav_trajectory_snapshots", (2,))

    def test_cli_exposes_singular_and_batch_forms(self):
        default_paper = build_paper_parser().parse_args(
            [
                "td3_dinkelbach",
                "--run-dir",
                "run",
                "--suite",
                "fixed_roi",
            ]
        )
        self.assertIsNone(default_paper.checkpoint_episode)
        self.assertIsNone(default_paper.checkpoint_episodes)
        self.assertIsNone(default_paper.roi_count)
        self.assertIsNone(default_paper.roi_counts)
        paper = build_paper_parser().parse_args(
            [
                "td3_dinkelbach",
                "--run-dir",
                "run",
                "--suite",
                "fixed_roi",
                "--checkpoint-episode",
                "50",
                "--roi-count",
                "4",
            ]
        )
        self.assertEqual((paper.checkpoint_episode, paper.roi_count), (50, 4))
        batch = build_batch_parser().parse_args(
            [
                "--run-dir",
                "run-a",
                "--run-dir",
                "run-b",
                "--checkpoint-episodes",
                "50",
                "100",
                "--roi-counts",
                "2",
                "8",
            ]
        )
        self.assertEqual(batch.run_directories, ["run-a", "run-b"])
        self.assertEqual(batch.checkpoint_episodes, [50, 100])
        self.assertEqual(batch.roi_counts, [2, 8])

    def test_paper_training_run_loader_keeps_formal_checkpoint_default(self):
        expected = {"checkpoint_episode": FORMAL_CHECKPOINT_EPISODE}
        with mock.patch(
            "paper_evaluation.resolve_training_run_checkpoint",
            return_value=expected,
        ) as resolve:
            actual = _load_training_run("run", "td3_dinkelbach")
        self.assertIs(actual, expected)
        self.assertEqual(resolve.call_args.args[1], FORMAL_CHECKPOINT_EPISODE)


class TrainingRunCheckpointSelectorTest(unittest.TestCase):
    def _run(
        self,
        root,
        method_id="td3_dinkelbach",
        checkpoints=(50, 100),
        status="COMPLETED",
    ):
        method = MethodSpec.parse(method_id)
        run_dir = Path(root)
        run_dir.mkdir(parents=True)
        training_config = {
            "total_episodes": 1500,
            "mode": "train",
            "random_seed": 17,
        }
        manifest = generate_manifest("train", 17, 1500)
        manifest.save(run_dir / "scenario_manifest.json")
        resolved = {
            "status": status,
            "method": method.method_id,
            "method_spec": method.to_dict(),
            "seed": 17,
            "episodes": 1500,
            "training_config": training_config,
            "training_manifest_hash": manifest.content_hash,
            "training_manifest_path": "scenario_manifest.json",
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"method_id": method.method_id}), encoding="utf-8"
        )
        for episode in checkpoints:
            checkpoint = (
                run_dir / "checkpoints" / "models" / f"ep_{episode:04d}"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
            (checkpoint / "models.pt").write_bytes(b"fixture")
        return run_dir

    @staticmethod
    def _inspect(checkpoint, **kwargs):
        episode = int(Path(checkpoint).name.removeprefix("ep_"))
        current_manifest = kwargs["current_training_manifest"]
        current_total = int(kwargs["expected_formal_config"]["total_episodes"])
        return {
            "checkpoint_dir": Path(checkpoint),
            "completed_episode": episode,
            "metadata": {"episode": episode - 1},
            "horizon_compatibility": {
                "checkpoint_episode": episode,
                "checkpoint_planned_total_episodes": current_total,
                "current_training_run_total_episodes": current_total,
                "horizon_extension_compatible": False,
                "allowed_horizon_differences": [],
                "checkpoint_training_manifest_hash": current_manifest.content_hash,
                "current_training_manifest_hash": current_manifest.content_hash,
                "manifest_prefix_compatible": True,
            },
        }

    def test_intermediate_checkpoint_uses_run_total_and_full_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._run(Path(temp_dir) / "long-run")
            with (
                mock.patch(
                    "evaluation_selection.inspect_model_checkpoint",
                    side_effect=self._inspect,
                ) as inspect,
                mock.patch(
                    "evaluation_selection.checkpoint_artifact_provenance",
                    return_value=PROVENANCE,
                ) as fingerprint,
            ):
                context = resolve_training_run_checkpoint(
                    run_dir, 50, require_run_metadata=True
                )

        self.assertEqual(context["checkpoint_episode"], 50)
        self.assertEqual(context["training_run_status_at_evaluation"], "COMPLETED")
        self.assertTrue(context["training_run_completed_at_evaluation"])
        self.assertFalse(context["interim_checkpoint_evaluation"])
        self.assertEqual(context["training_total_episodes"], 1500)
        self.assertEqual(context["method"].method_id, "td3_dinkelbach")
        self.assertEqual(context["checkpoint_artifact_provenance"], PROVENANCE)
        self.assertEqual(inspect.call_args.kwargs["expected_completed_episodes"], 50)
        self.assertTrue(inspect.call_args.kwargs["require_episode_directory"])
        self.assertEqual(
            inspect.call_args.kwargs["expected_experiment_metadata"][
                "training_seed"
            ],
            17,
        )
        fingerprint.assert_called_once()

    def test_running_and_interrupted_runs_accept_complete_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for status in ("RUNNING", "INTERRUPTED"):
                with self.subTest(status=status):
                    run_dir = self._run(
                        root / status.lower(), status=status, checkpoints=(50,)
                    )
                    with (
                        mock.patch(
                            "evaluation_selection.inspect_model_checkpoint",
                            side_effect=self._inspect,
                        ) as inspect,
                        mock.patch(
                            "evaluation_selection.checkpoint_artifact_provenance",
                            return_value=PROVENANCE,
                        ) as fingerprint,
                    ):
                        context = resolve_training_run_checkpoint(run_dir, 50)
                    self.assertEqual(context["checkpoint_episode"], 50)
                    self.assertEqual(
                        context["training_run_status_at_evaluation"], status
                    )
                    self.assertFalse(
                        context["training_run_completed_at_evaluation"]
                    )
                    self.assertTrue(context["interim_checkpoint_evaluation"])
                    inspect.assert_called_once()
                    fingerprint.assert_called_once()

    def test_missing_or_invalid_training_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, status in enumerate((None, "", "FAILED", 123)):
                with self.subTest(status=status):
                    run_dir = self._run(root / f"run-{index}", checkpoints=(50,))
                    resolved_path = run_dir / "resolved_config.json"
                    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                    if status is None:
                        resolved.pop("status")
                    else:
                        resolved["status"] = status
                    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "status"):
                        resolve_training_run_checkpoint(run_dir, 50)

    def test_running_run_still_rejects_incomplete_and_incompatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = self._run(
                root / "missing", status="RUNNING", checkpoints=()
            )
            with self.assertRaisesRegex(FileNotFoundError, "ep_0050"):
                resolve_training_run_checkpoint(missing, 50)

            incomplete = self._run(
                root / "incomplete", status="RUNNING", checkpoints=(50,)
            )
            (incomplete / "checkpoints" / "models" / "ep_0050" / "metadata.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                resolve_training_run_checkpoint(incomplete, 50)

            incomplete_fields = self._run(
                root / "incomplete-fields",
                status="RUNNING",
                checkpoints=(50,),
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint_schema_version"):
                resolve_training_run_checkpoint(incomplete_fields, 50)

            incompatible = self._run(
                root / "incompatible", status="RUNNING", checkpoints=(50,)
            )
            with mock.patch(
                "evaluation_selection.inspect_model_checkpoint",
                side_effect=RuntimeError("movement state dimension is incompatible"),
            ):
                with self.assertRaisesRegex(RuntimeError, "dimension"):
                    resolve_training_run_checkpoint(incompatible, 50)

            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                resolve_training_run_checkpoint(
                    incompatible, 50, expected_method="ddpg_dinkelbach"
                )

    def test_missing_or_mismatched_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._run(
                Path(temp_dir) / "run", checkpoints=(50,), status="RUNNING"
            )
            with self.assertRaisesRegex(FileNotFoundError, "ep_0100"):
                resolve_training_run_checkpoint(run_dir, 100)

            with (
                mock.patch(
                    "evaluation_selection.inspect_model_checkpoint",
                    return_value={
                        "completed_episode": 49,
                        "metadata": {"episode": 48},
                    },
                ),
                mock.patch(
                    "evaluation_selection.checkpoint_artifact_provenance",
                    return_value=PROVENANCE,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "disagrees"):
                    resolve_training_run_checkpoint(run_dir, 50)

    def test_checkpoint_over_run_total_and_fingerprint_failure_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._run(
                Path(temp_dir) / "run", checkpoints=(50,), status="INTERRUPTED"
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                resolve_training_run_checkpoint(run_dir, 1501)
            with (
                mock.patch(
                    "evaluation_selection.inspect_model_checkpoint",
                    side_effect=self._inspect,
                ),
                mock.patch(
                    "evaluation_selection.checkpoint_artifact_provenance",
                    side_effect=RuntimeError("invalid fingerprint payload"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    resolve_training_run_checkpoint(run_dir, 50)

    def test_training_history_metadata_records_each_training_run_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "training"
            run_dir.mkdir()
            (run_dir / "training_history.jsonl").write_text(
                json.dumps(
                    {
                        "episode": 1,
                        "timely_goodput_mbits": 2.0,
                        "total_timely_useful_mbits": 2.0,
                        "mobility_energy_j": 4.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for status in ("COMPLETED", "RUNNING", "INTERRUPTED"):
                with self.subTest(status=status):
                    output = root / f"evaluation-{status.lower()}"
                    completed = status == "COMPLETED"
                    context = {
                        "run_dir": run_dir,
                        "checkpoint": run_dir / "checkpoints" / "models" / "ep_0050",
                        "checkpoint_episode": 50,
                        "training_run_id": run_dir.name,
                        "training_total_episodes": 1500,
                        "training_run_status_at_evaluation": status,
                        "training_run_completed_at_evaluation": completed,
                        "interim_checkpoint_evaluation": not completed,
                    }
                    with (
                        mock.patch(
                            "paper_evaluation._load_training_run",
                            return_value=context,
                        ),
                        mock.patch(
                            "paper_evaluation._git_sha",
                            return_value="a" * 40,
                        ),
                    ):
                        result = run_paper_evaluation(
                            "td3_dinkelbach",
                            run_directory=run_dir,
                            suite="training_ee_vs_episode",
                            checkpoint_episode=50,
                            output_directory=output,
                        )
                    metadata = json.loads(
                        (output / "paper_evaluation_metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for value in (result, metadata):
                        self.assertEqual(
                            value["training_run_status_at_evaluation"], status
                        )
                        self.assertIs(
                            value["training_run_completed_at_evaluation"],
                            completed,
                        )
                        self.assertIs(
                            value["interim_checkpoint_evaluation"],
                            not completed,
                        )
                        self.assertEqual(value["checkpoint_episode"], 50)


class CheckpointRoiBatchPlanTest(unittest.TestCase):
    _run = TrainingRunCheckpointSelectorTest._run
    _inspect = staticmethod(TrainingRunCheckpointSelectorTest._inspect)

    def _plan(self, root, *, rois=(2, 5), checkpoints=(50, 100)):
        run_a = self._run(root / "run-a", "td3_dinkelbach", checkpoints)
        run_b = self._run(root / "run-b", "ddpg_dinkelbach", checkpoints)
        with (
            mock.patch(
                "evaluation_selection.inspect_model_checkpoint",
                side_effect=self._inspect,
            ),
            mock.patch(
                "evaluation_selection.checkpoint_artifact_provenance",
                return_value=PROVENANCE,
            ),
        ):
            return build_checkpoint_roi_sweep_plan(
                (run_a, run_b),
                checkpoint_episodes=checkpoints,
                roi_counts=rois,
                evaluation_episodes=2,
                episode_seconds=5,
                manifest_seed=701,
                output_root=root / "outputs",
                batch_id="batch",
            )

    def test_dynamic_cartesian_plan_and_shared_scenario_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._plan(Path(temp_dir))

        self.assertEqual(len(plan["points"]), 2 * 2 * 2)
        self.assertFalse(plan["batch_output_directory"].exists())
        self.assertEqual(
            {point["method_id"] for point in plan["points"]},
            {"td3_dinkelbach", "ddpg_dinkelbach"},
        )
        for roi in (2, 5):
            roi_points = [
                point for point in plan["points"] if point["roi_count"] == roi
            ]
            self.assertEqual(
                {point["manifest_hash"] for point in roi_points},
                {plan["manifests"][roi].content_hash},
            )
            self.assertEqual(
                {tuple(point["scenario_ids"]) for point in roi_points},
                {
                    tuple(
                        entry["scenario_id"]
                        for entry in plan["manifests"][roi].episodes
                    )
                },
            )
        self.assertNotEqual(
            plan["manifests"][2].content_hash,
            plan["manifests"][5].content_hash,
        )
        self.assertEqual(
            len({point["result_directory"] for point in plan["points"]}),
            len(plan["points"]),
        )
        for point in plan["points"]:
            self.assertEqual(
                point["training_run_status_at_evaluation"], "COMPLETED"
            )
            self.assertTrue(point["training_run_completed_at_evaluation"])
            self.assertFalse(point["interim_checkpoint_evaluation"])
            self.assertEqual(point["checkpoint_planned_total_episodes"], 1500)
            self.assertEqual(point["current_training_run_total_episodes"], 1500)
            self.assertFalse(point["horizon_extension_compatible"])
            self.assertEqual(point["allowed_horizon_differences"], [])
            self.assertTrue(point["manifest_prefix_compatible"])

    def test_running_run_without_final_run_metadata_can_build_sweep_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self._run(
                root / "run", checkpoints=(50,), status="RUNNING"
            )
            (run_dir / "run_metadata.json").unlink()
            with (
                mock.patch(
                    "evaluation_selection.inspect_model_checkpoint",
                    side_effect=self._inspect,
                ),
                mock.patch(
                    "evaluation_selection.checkpoint_artifact_provenance",
                    return_value=PROVENANCE,
                ),
            ):
                plan = build_checkpoint_roi_sweep_plan(
                    (run_dir,),
                    checkpoint_episodes=(50,),
                    roi_counts=(2,),
                    evaluation_episodes=1,
                    episode_seconds=5,
                    manifest_seed=701,
                    output_root=root / "outputs",
                    batch_id="batch",
                )
            point = plan["points"][0]
            self.assertEqual(
                point["training_run_status_at_evaluation"], "RUNNING"
            )
            self.assertFalse(point["training_run_completed_at_evaluation"])
            self.assertTrue(point["interim_checkpoint_evaluation"])

    def _fake_evaluator(self, calls):
        def evaluate(method_id, **kwargs):
            calls.append((method_id, kwargs))
            output = Path(kwargs["output_directory"])
            output.mkdir(parents=True)
            roi = int(kwargs["roi_counts"][0])
            manifest = kwargs["fixed_roi_manifests"][roi]
            rows = [
                {
                    "num_GT": roi,
                    "timely_goodput_mbits": 1.0,
                    "total_mobility_energy_j": 5.0,
                    "fov_delivered_packets": 0,
                    "fov_delivered_e2e_delay_sum_seconds": 0.0,
                    "fov_eligible_packets": 0,
                    "fov_violation_packets": 0,
                    "com_delivered_packets": 0,
                    "com_delivered_e2e_delay_sum_seconds": 0.0,
                    "com_eligible_packets": 0,
                    "com_violation_packets": 0,
                    "coverage": 0.5,
                    "sr_admission_drop_count": 3,
                }
                for _ in range(2)
            ]
            per_episode = output / "per_episode.jsonl"
            per_episode.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            point = {
                "point_id": f"roi_{roi}",
                "fixed_num_gt": roi,
                "x_value": roi,
                "x_unit": "RoIs",
            }
            aggregates = aggregate_paper_point_metrics(
                method_id, "fixed_roi", point, rows
            )
            aggregate_path = output / "aggregated_plot_data.json"
            aggregate_path.write_text(json.dumps(aggregates), encoding="utf-8")
            return {
                "points": [
                    {
                        "manifest_hash": manifest.content_hash,
                        "scenario_ids": [
                            entry["scenario_id"] for entry in manifest.episodes
                        ],
                        "roi_count": roi,
                        "checkpoint_episode": kwargs["checkpoint_episode"],
                        "training_run_status_at_evaluation": "COMPLETED",
                        "training_run_completed_at_evaluation": True,
                        "interim_checkpoint_evaluation": False,
                        "output_directory": str(output),
                        "outputs": {
                            "per_episode_jsonl": str(per_episode)
                        },
                        "aggregated_plot_data": str(aggregate_path),
                    }
                ]
            }

        return evaluate

    def test_sequential_fake_execution_writes_canonical_batch_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._plan(Path(temp_dir))
            calls = []
            result = execute_checkpoint_roi_sweep(
                plan, evaluator=self._fake_evaluator(calls)
            )
            batch_dir = Path(result["output_directory"])
            metadata = json.loads(
                (batch_dir / "checkpoint_roi_sweep_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (batch_dir / "checkpoint_roi_sweep_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(calls), 8)
        self.assertEqual(metadata["status"], "COMPLETED")
        self.assertEqual(metadata["completed_point_count"], 8)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(row["status"] == "COMPLETED" for row in summary))
        self.assertTrue(
            all(row["energy_efficiency_bit_per_joule"] == 200_000.0 for row in summary)
        )
        self.assertTrue(all(row["timely_useful_bits"] == 2_000_000.0 for row in summary))
        self.assertTrue(
            all(row["timely_useful_goodput_bps"] == 200_000.0 for row in summary)
        )
        self.assertTrue(all(row["delay_violation_probability"] is None for row in summary))
        self.assertTrue(all(row["sr_admission_drop_count"] == 6 for row in summary))

    def test_first_failure_stops_later_points_and_preserves_completed_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self._run(root / "run-a", checkpoints=(50,))
            with (
                mock.patch(
                    "evaluation_selection.inspect_model_checkpoint",
                    side_effect=self._inspect,
                ),
                mock.patch(
                    "evaluation_selection.checkpoint_artifact_provenance",
                    return_value=PROVENANCE,
                ),
            ):
                plan = build_checkpoint_roi_sweep_plan(
                    (run_dir,),
                    checkpoint_episodes=(50,),
                    roi_counts=(2, 3, 4),
                    evaluation_episodes=2,
                    episode_seconds=5,
                    manifest_seed=702,
                    output_root=root / "outputs",
                    batch_id="batch",
                )
            calls = []
            successful = self._fake_evaluator(calls)

            def fail_second(method_id, **kwargs):
                if len(calls) == 1:
                    calls.append((method_id, kwargs))
                    raise RuntimeError("forced point failure")
                return successful(method_id, **kwargs)

            with self.assertRaisesRegex(SweepExecutionError, "roi=3"):
                execute_checkpoint_roi_sweep(plan, evaluator=fail_second)
            batch_dir = Path(plan["batch_output_directory"])
            metadata = json.loads(
                (batch_dir / "checkpoint_roi_sweep_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (batch_dir / "checkpoint_roi_sweep_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(metadata["status"], "FAILED")
        self.assertEqual(
            [point["status"] for point in metadata["points"]],
            ["COMPLETED", "FAILED", "PENDING"],
        )
        self.assertEqual([row["status"] for row in summary], ["COMPLETED", "FAILED"])

    def test_preflight_collects_errors_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_a = root / "missing-a"
            missing_b = root / "missing-b"
            with self.assertRaises(SweepPreflightError) as raised:
                build_checkpoint_roi_sweep_plan(
                    (missing_a, missing_b),
                    checkpoint_episodes=(50, 100),
                    roi_counts=(2,),
                    evaluation_episodes=1,
                    episode_seconds=1,
                    manifest_seed=703,
                    output_root=root / "outputs",
                    batch_id="batch",
                )
            self.assertGreaterEqual(len(raised.exception.errors), 4)
            self.assertFalse((root / "outputs" / "batch").exists())


if __name__ == "__main__":
    unittest.main()
