from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path

from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import (
    MethodSpec,
    comparison_method_configuration,
    effective_training_config,
)
from HRL_task_aware import formal_training_config
from Packet_scheduler_v1 import TASK_DEADLINE_SECONDS
from paper_evaluation import (
    PAPER_EVALUATION_SUITES,
    aggregate_paper_point_metrics,
    evaluation_sweep_points,
)
from paper_figure_registry import FIGURE_REGISTRY
from paper_figures import (
    AmbiguousPaperRunError,
    IncompatiblePaperRunError,
    PaperFigureSpecError,
    build_paper_figures,
    causal_trailing_average,
    normalize_episode_ee,
    paper_energy_efficiency,
)
from scenario_manifest import generate_manifest
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    checkpoint_metadata_fingerprint,
)


TRAINING_SEED = 20260817
EVALUATION_EPISODES = 3
EVALUATION_HORIZON_SECONDS = 60


class EpisodeEnergyEfficiencyTest(unittest.TestCase):
    def test_causal_average_never_reads_a_future_episode(self):
        values = list(range(1, 52))
        averaged = causal_trailing_average(values, window=50)
        self.assertEqual(averaged[0], 1.0)
        self.assertEqual(averaged[49], sum(range(1, 51)) / 50)
        self.assertEqual(averaged[50], sum(range(2, 52)) / 50)

    def test_adapter_uses_episode_bits_and_energy_not_reward(self):
        rows = [
            {"method_id": "td3_dinkelbach", "episode": 1, "reward": 999.0, "timely_goodput_mbits": 2.0, "mobility_energy_j": 4.0},
            {"method_id": "td3_dinkelbach", "episode": 2, "reward": -999.0, "timely_goodput_mbits": 9.0, "mobility_energy_j": 3.0},
        ]
        normalized = normalize_episode_ee("td3_dinkelbach", rows)
        self.assertEqual(
            [row["raw_energy_efficiency_bit_per_j"] for row in normalized],
            [500000.0, 3000000.0],
        )
        self.assertNotIn("reward", normalized[0])

    def test_safe_division_is_explicit_and_broken_rows_fail_loudly(self):
        self.assertEqual(paper_energy_efficiency(0.0, 0.0), 0.0)
        self.assertGreater(paper_energy_efficiency(1.0, 0.0), 1e11)
        with self.assertRaises(PaperFigureSpecError):
            normalize_episode_ee(
                "td3_dinkelbach",
                [{"episode": 1, "timely_goodput_mbits": 1.0}],
            )


class PooledMetricTest(unittest.TestCase):
    def test_delay_and_violation_pool_counts_not_episode_averages(self):
        base = {
            "timely_goodput_mbits": 2.0,
            "total_mobility_energy_j": 4.0,
            "fov_delivered_packets": 1,
            "fov_delivered_e2e_delay_sum_seconds": 1.0,
            "fov_generated_packets": 2,
            "fov_violation_packets": 1,
            "com_delivered_packets": 0,
            "com_delivered_e2e_delay_sum_seconds": 0.0,
            "com_generated_packets": 0,
            "com_violation_packets": 0,
        }
        second = {
            **base,
            "fov_delivered_packets": 9,
            "fov_delivered_e2e_delay_sum_seconds": 0.9,
            "fov_generated_packets": 18,
            "fov_violation_packets": 3,
        }
        rows = aggregate_paper_point_metrics(
            "td3_dinkelbach",
            "fixed_roi",
            {"point_id": "roi_2", "x_value": 2, "x_unit": "RoIs", "fixed_num_gt": 2},
            [base, second],
        )
        delay = next(row for row in rows if row["metric"] == "average_e2e_delay_seconds" and row["task_type"] == "FOV")
        violation = next(row for row in rows if row["metric"] == "violation_probability" and row["task_type"] == "FOV")
        missing = next(row for row in rows if row["metric"] == "average_e2e_delay_seconds" and row["task_type"] == "COM")
        self.assertAlmostEqual(delay["value"], 1.9 / 10)
        self.assertAlmostEqual(violation["value"], 4 / 20)
        self.assertIsNone(missing["value"])
        self.assertTrue(missing["missing"])


class SyntheticFigureBuildTest(unittest.TestCase):
    @staticmethod
    def _formal_config(method):
        base = asdict(formal_training_config(2500, random_seed=TRAINING_SEED))
        return effective_training_config(base, method)

    @classmethod
    def _checkpoint_metadata(cls, method, formal_config):
        experiment = {
            "method_id": method.method_id,
            "method_spec": method.to_dict(),
            "method_spec_fingerprint": method.fingerprint,
            "training_seed": TRAINING_SEED,
            "movement_agent": method.agent,
            "reward_mode": method.reward_mode,
            "task_potential_enabled": method.task_potential_enabled,
            "formal_config": formal_config,
            **comparison_method_configuration(method),
        }
        if method.uses_dinkelbach:
            state = DinkelbachBlockState.from_config(formal_config)
            for _ in range(2500):
                state.record_episode(1.0, 2.0)
            experiment.update(
                {
                    **dinkelbach_config_metadata(formal_config),
                    "lambda_ee": state.current_lambda,
                    "dinkelbach_state": state.training_state(),
                }
            )
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 2499,
            "movement_agent_kind": method.agent,
            "movement_agent_configuration": formal_config[
                "movement_agent_configuration"
            ],
            "experiment": experiment,
        }

    def _write_training_run(self, root, method_id, *, include_history=False):
        run_dir = root / "training" / method_id
        run_dir.mkdir(parents=True)
        method = MethodSpec.parse(method_id)
        formal_config = self._formal_config(method)
        resolved = {
            "method": method_id,
            "method_spec": method.to_dict(),
            "seed": TRAINING_SEED,
            "training_manifest_hash": "a" * 64,
            "formal_checkpoint_episode": 2500,
            "training_config": formal_config,
            "status": "COMPLETED",
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        checkpoint = run_dir / "checkpoints" / "models" / "ep_2500"
        checkpoint.mkdir(parents=True)
        metadata = self._checkpoint_metadata(method, formal_config)
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (checkpoint / "models.pt").write_bytes(b"synthetic-not-loaded")
        if include_history:
            rows = [
                {
                    "method_id": method_id,
                    "episode": episode,
                    "reward": 1e9,
                    "timely_goodput_mbits": episode * (1.0 + len(method_id) / 100),
                    "mobility_energy_j": 2.0,
                }
                for episode in range(1, 2501)
            ]
            (run_dir / "training_history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        return run_dir, metadata

    @staticmethod
    def _aggregate_rows(method_id, suite, points):
        method_offset = sum(ord(char) for char in method_id) % 7
        rows = []
        for point_index, point in enumerate(points):
            common = {
                "semantic_suite": suite,
                "method_id": method_id,
                "point_id": point["point_id"],
                "x_value": point.get("x_value"),
                "x_unit": point.get("x_unit"),
                "fixed_num_gt": point.get("fixed_num_gt"),
                "swept_task": point.get("swept_task"),
                "evaluation_episode_count": EVALUATION_EPISODES,
            }
            rows.append({**common, "metric": "energy_efficiency_mbit_per_j", "task_type": None, "display_task_type": None, "numerator": 10 + point_index, "numerator_unit": "Mbit", "denominator": 4, "denominator_unit": "J", "value": 1.0 + 0.15 * point_index + 0.05 * method_offset, "value_unit": "Mbit/J", "missing": False})
            for task_type in ("FOV", "COM"):
                delay = 0.004 + 0.002 * point_index + 0.0003 * method_offset + (0.003 if task_type == "COM" else 0.0)
                probability = max(0.0005, 0.4 / (point_index + 2 + method_offset / 4))
                rows.append({**common, "metric": "average_e2e_delay_seconds", "task_type": task_type, "display_task_type": "VS" if task_type == "FOV" else "COM", "numerator": delay * 20, "numerator_unit": "seconds", "denominator": 20, "denominator_unit": "delivered_packets", "value": delay, "value_unit": "seconds", "missing": False})
                rows.append({**common, "metric": "violation_probability", "task_type": task_type, "display_task_type": "VS" if task_type == "FOV" else "COM", "numerator": probability * 100, "numerator_unit": "violation_packets", "denominator": 100, "denominator_unit": "generated_packets", "value": probability, "value_unit": "probability", "missing": False})
        return rows

    @staticmethod
    def _resolved_overrides(suite, point):
        rates = {"FOV": None, "COM": None}
        deadlines = {
            "FOV": float(TASK_DEADLINE_SECONDS["FOV"]),
            "COM": float(TASK_DEADLINE_SECONDS["COM"]),
        }
        if suite == "task_type_delay_vs_arrival_rate":
            if point["swept_task"] == "COM":
                rates = {"FOV": 5.0, "COM": float(point["x_value"])}
            else:
                rates = {"FOV": float(point["x_value"]), "COM": 50.0}
        elif suite == "task_type_delay_violation_vs_target_delay":
            deadlines[point["swept_task"]] = float(point["x_value"])
        return {
            "traffic_rates_packets_per_second": rates,
            "task_deadlines_seconds": deadlines,
            "units": {"traffic_rate": "packets/s", "deadline": "seconds"},
        }

    @staticmethod
    def _trajectory_artifact(manifest, checkpoint, fingerprint):
        times = (5.0, 10.0, 15.0, 25.0)
        phases = ("Search", "FOV", "FOV+COM", "Hover")
        history_times = (0.0, *times)
        uav_paths = {
            str(uid): [
                {"actual_time_seconds": time, "x": 100 + (uid % 4) * 200 + time, "y": 100 + (uid // 4) * 200, "z": 90 + uid}
                for time in history_times
            ]
            for uid in range(16)
        }
        sr_paths = {
            str(uid): [
                {"actual_time_seconds": time, "x": (0 if uid == 0 else 1000) + (time if uid == 0 else -time), "y": 500, "z": 0}
                for time in history_times
            ]
            for uid in range(2)
        }
        snapshots = []
        for time, phase in zip(times, phases):
            snapshots.append({
                "requested_time_seconds": time,
                "actual_time_seconds": time,
                "target_uav_id": 0,
                "target_uav_phase": phase,
                "uavs": [{"uav_id": uid, "x": uav_paths[str(uid)][history_times.index(time)]["x"], "y": uav_paths[str(uid)][history_times.index(time)]["y"], "z": 90 + uid, "task_phase": phase if uid == 0 else "Hover"} for uid in range(16)],
                "sr_teams": [{"sr_id": uid, "x": sr_paths[str(uid)][history_times.index(time)]["x"], "y": 500, "z": 0, "active": True} for uid in range(2)],
                "ground_targets": [{"gt_id": 0, "x": 650.0, "y": 650.0, "z": 0.0, "radius_m": 80.0, "detected": time >= 10, "detected_by_uav_id": 0 if time >= 10 else None}],
                "ground_station": {"gs_id": 16, "x": 0.0, "y": 0.0, "z": 0.0},
                "active_links": [{"sender_id": 0, "receiver_id": 1, "link_type": "U2U", "capacity_bits_per_second": 1e6}],
                "sensing_coverage": [{"uav_id": 0, "geometry": "axis_aligned_ground_rectangle", "center_x": 100 + time, "center_y": 100, "ground_z": 0, "width_m": 200, "height_m": 200, "clipped_bounds": {"x_min": 0, "x_max": 200 + time, "y_min": 0, "y_max": 200}, "model": {"f_m": 0.004, "image_width_m": 0.008, "image_length_m": 0.012}}],
            })
        return {
            "scenario_id": manifest.episodes[0]["scenario_id"],
            "scenario_manifest_hash": manifest.content_hash,
            "requested_times_seconds": list(times),
            "target_uav_id": 0,
            "snapshots": snapshots,
            "trajectory_history": [],
            "uav_paths": uav_paths,
            "sr_paths": sr_paths,
            "ground_targets": [],
            "initial_sr_teams": [],
            "checkpoint_path": str(checkpoint),
            "checkpoint_required": True,
            "checkpoint_fingerprint": fingerprint,
        }

    def _write_evaluation_run(self, root, suite, method_id, training_runs):
        evaluation_dir = root / "evaluations" / suite / method_id
        evaluation_dir.mkdir(parents=True)
        method = MethodSpec.parse(method_id)
        checkpoint_required = method.learns_movement or method.learns_routing
        training_run = training_runs.get(method_id)
        checkpoint = (
            training_run / "checkpoints" / "models" / "ep_2500"
            if checkpoint_required
            else None
        )
        checkpoint_metadata = (
            json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            if checkpoint_required
            else None
        )
        fingerprint = (
            checkpoint_metadata_fingerprint(checkpoint_metadata)
            if checkpoint_required
            else None
        )
        points = list(evaluation_sweep_points(suite))
        metadata_points = []
        for index, point in enumerate(points):
            point_dir = evaluation_dir / point["point_id"]
            point_dir.mkdir()
            manifest = generate_manifest(
                "test",
                TRAINING_SEED,
                EVALUATION_EPISODES,
                num_gt=point.get("fixed_num_gt"),
            )
            manifest_path = manifest.save(point_dir / "scenario_manifest.json")
            if suite == "uav_trajectory_snapshots":
                artifact = self._trajectory_artifact(
                    manifest, checkpoint, fingerprint
                )
                (point_dir / "trajectory_artifacts.json").write_text(
                    json.dumps([artifact]), encoding="utf-8"
                )
            metadata_points.append({
                **point,
                "scenario_manifest_path": str(manifest_path.resolve()),
                "scenario_manifest_hash": manifest.content_hash,
                "manifest_hash": manifest.content_hash,
                "scenario_ids": [entry["scenario_id"] for entry in manifest.episodes],
                "evaluation_episode_count": EVALUATION_EPISODES,
                "evaluation_horizon_seconds": EVALUATION_HORIZON_SECONDS,
                "evaluation_seed": TRAINING_SEED,
                "manifest_seed": TRAINING_SEED,
                "num_uav": 16,
                "resolved_overrides": self._resolved_overrides(suite, point),
                "checkpoint_required": checkpoint_required,
                "checkpoint_path": str(checkpoint) if checkpoint_required else None,
                "checkpoint_metadata_fingerprint": fingerprint,
                "output_directory": str(point_dir.resolve()),
            })
        rows = self._aggregate_rows(method_id, suite, points)
        (evaluation_dir / "aggregated_plot_data.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )
        training_resolved = (
            json.loads((training_run / "resolved_config.json").read_text(encoding="utf-8"))
            if checkpoint_required
            else None
        )
        metadata = {
            "semantic_suite": suite,
            "method_id": method_id,
            "method_spec": method.to_dict(),
            "training_run": str(training_run) if checkpoint_required else None,
            "checkpoint_required": checkpoint_required,
            "checkpoint_episode": 2500 if checkpoint_required else None,
            "checkpoint_path": str(checkpoint) if checkpoint_required else None,
            "checkpoint_metadata_fingerprint": fingerprint,
            "formal_training_config": training_resolved["training_config"] if checkpoint_required else None,
            "training_seed": TRAINING_SEED if checkpoint_required else None,
            "evaluation_seed": TRAINING_SEED,
            "manifest_seed": TRAINING_SEED,
            "evaluation_episodes_per_point": EVALUATION_EPISODES,
            "evaluation_horizon_seconds": EVALUATION_HORIZON_SECONDS,
            "target_uav_id": 0 if suite == "uav_trajectory_snapshots" else None,
            "points": metadata_points,
        }
        (evaluation_dir / "paper_evaluation_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return evaluation_dir

    def _fixture(self, root):
        convergence_methods = tuple(
            FIGURE_REGISTRY["training_ee_vs_episode"]["methods"]
        )
        learned_methods = {
            method
            for definition in PAPER_EVALUATION_SUITES.values()
            for method in definition["methods"]
            if MethodSpec.parse(method).learns_movement
            or MethodSpec.parse(method).learns_routing
        }
        training_runs = {}
        for method in sorted(set(convergence_methods).union(learned_methods)):
            training_runs[method], _ = self._write_training_run(
                root, method, include_history=method in convergence_methods
            )
        evaluations = {}
        for suite, definition in PAPER_EVALUATION_SUITES.items():
            if definition["kind"] == "training_history":
                continue
            evaluations[suite] = {
                method: self._write_evaluation_run(
                    root, suite, method, training_runs
                )
                for method in definition["methods"]
            }
        spec = root / "paper_runs.json"
        spec.write_text(
            json.dumps({
                "target_uav_id": 0,
                "training_runs": {
                    method: {"run_dir": str(training_runs[method])}
                    for method in convergence_methods
                },
                "evaluation_runs": {
                    suite: {
                        method: {"evaluation_dir": str(path)}
                        for method, path in methods.items()
                    }
                    for suite, methods in evaluations.items()
                },
            }),
            encoding="utf-8",
        )
        return spec, training_runs, evaluations

    def test_build_all_writes_exactly_twelve_standalone_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, _ = self._fixture(root)
            result = build_paper_figures(spec, output_root=root / "output")
            self.assertEqual(tuple(result["semantic_figures"]), tuple(FIGURE_REGISTRY))
            self.assertEqual(len(result["semantic_figures"]), 12)
            output = Path(result["output_directory"])
            expected_titles = {
                "task_assignment_ee_vs_number_of_rois": "Task-assignment strategies",
                "trajectory_design_ee_vs_number_of_rois": "Trajectory-design methods",
                "hierarchical_architecture_ee_vs_number_of_rois": "Hierarchical learning architectures",
                "com_task_delay_vs_arrival_rate": "COM task",
                "vs_task_delay_vs_arrival_rate": "VS task",
            }
            for figure_id, contract in FIGURE_REGISTRY.items():
                stem = contract["output_stem"]
                for suffix in (".png", ".pdf", ".csv", ".json", "_resolved_spec.json"):
                    self.assertTrue((output / f"{stem}{suffix}").is_file(), f"{figure_id}{suffix}")
                resolved = json.loads((output / f"{stem}_resolved_spec.json").read_text(encoding="utf-8"))
                self.assertEqual(resolved["semantic_figure_id"], figure_id)
                self.assertEqual(resolved["render_contract"]["axes_count"], 1)
                title = resolved["render_contract"]["axes_titles"][0]
                if figure_id.startswith("uav_trajectory_t_"):
                    self.assertRegex(title, r"^t = (5|10|15|25) s: UAV in .+ mode$")
                else:
                    self.assertEqual(title, expected_titles.get(figure_id, ""))
                if title:
                    self.assertGreaterEqual(
                        resolved["render_contract"]["axes_title_positions"][0][1],
                        1.0,
                    )
            for forbidden in (
                "UAV_trajectory_snapshots",
                "Energy_efficiency_design_comparisons",
                "Task_type_delay_Vs_arrival_rate",
            ):
                self.assertFalse((output / f"{forbidden}.png").exists())
                self.assertFalse((output / f"{forbidden}.pdf").exists())
            build_metadata = json.loads((output / "paper_figure_build.json").read_text(encoding="utf-8"))
            self.assertEqual(len(build_metadata["semantic_figures"]), 12)

    def test_deprecated_family_alias_expands_without_composite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, _ = self._fixture(root)
            result = build_paper_figures(spec, figure="fig5", output_root=root / "output")
            self.assertEqual(
                tuple(result["semantic_figures"]),
                ("com_task_delay_vs_arrival_rate", "vs_task_delay_vs_arrival_rate"),
            )

    def test_checkpoint_paths_and_fingerprints_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            method = "td3_dinkelbach"
            evaluation_dir = evaluations["fixed_roi"][method]
            metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_metadata_fingerprint"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "fingerprint"):
                build_paper_figures(spec, figure="task_assignment_ee_vs_number_of_rois", output_root=root / "bad-fingerprint")

            spec, _, evaluations = self._fixture(root / "missing")
            evaluation_dir = evaluations["fixed_roi"][method]
            metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(root / "does-not-exist" / "ep_2500")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "checkpoint path"):
                build_paper_figures(spec, figure="task_assignment_ee_vs_number_of_rois", output_root=root / "missing-checkpoint")

    def test_random_baseline_rejects_any_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            method = "kkm_random_action_random_routing"
            evaluation_dir = evaluations["fixed_roi"][method]
            metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(root / "fake" / "ep_2500")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "pure-random"):
                build_paper_figures(spec, figure="hierarchical_architecture_ee_vs_number_of_rois", output_root=root / "random-checkpoint")

    def test_manifest_overrides_and_exact_cartesian_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            method = "td3_dinkelbach"
            evaluation_dir = evaluations["task_type_delay_vs_arrival_rate"][method]
            metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["points"][0]["resolved_overrides"]["traffic_rates_packets_per_second"]["FOV"] = 6.0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "resolved sweep overrides"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "bad-overrides")
            metadata["points"][0]["resolved_overrides"] = self._resolved_overrides(
                "task_type_delay_vs_arrival_rate", metadata["points"][0]
            )
            metadata["points"][0]["scenario_manifest_hash"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "manifest hash"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "bad-manifest")

            spec, _, evaluations = self._fixture(root / "missing-row")
            evaluation_dir = evaluations["task_type_delay_vs_arrival_rate"][method]
            aggregate_path = evaluation_dir / "aggregated_plot_data.json"
            original_rows = json.loads(aggregate_path.read_text(encoding="utf-8"))
            rows = [
                row for row in original_rows
                if not (
                    row["metric"] == "average_e2e_delay_seconds"
                    and row["task_type"] == "COM"
                    and row["point_id"] == "com_rate_50"
                )
            ]
            aggregate_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "Cartesian"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "missing-row-out")
            selected = next(
                row for row in original_rows
                if row["metric"] == "average_e2e_delay_seconds"
                and row["task_type"] == "COM"
                and row["point_id"] == "com_rate_50"
            )
            aggregate_path.write_text(
                json.dumps([*original_rows, selected]), encoding="utf-8"
            )
            with self.assertRaisesRegex(IncompatiblePaperRunError, "duplicate"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "duplicate-row-out")
            aggregate_path.write_text(
                json.dumps([*original_rows, {**selected, "point_id": "com_rate_999"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(IncompatiblePaperRunError, "unexpected aggregate point"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "extra-row-out")
            aggregate_path.write_text(
                json.dumps([*original_rows, {**selected, "metric": "invented_metric"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(IncompatiblePaperRunError, "unexpected aggregate metric"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "bad-metric-out")

    def test_ambiguous_mapping_and_incomplete_training_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, training, _ = self._fixture(root)
            value = json.loads(spec.read_text(encoding="utf-8"))
            method = FIGURE_REGISTRY["training_ee_vs_episode"]["methods"][0]
            value["training_runs"][method] = {
                "candidates": [str(training[method]), str(training[method])]
            }
            spec.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AmbiguousPaperRunError):
                build_paper_figures(spec, figure="training_ee_vs_episode", output_root=root / "ambiguous")

            spec, training, _ = self._fixture(root / "short")
            method = FIGURE_REGISTRY["training_ee_vs_episode"]["methods"][0]
            history = training[method] / "training_history.jsonl"
            history.write_text("\n".join(history.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "1..2500"):
                build_paper_figures(spec, figure="training_ee_vs_episode", output_root=root / "short-out")


if __name__ == "__main__":
    unittest.main()
