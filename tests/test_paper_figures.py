import json
import tempfile
import unittest
from pathlib import Path

from experiment_config import MethodSpec
from paper_evaluation import (
    ARRIVAL_RATE_SWEEPS,
    DEADLINE_SWEEP_SECONDS,
    FIXED_ROI_VALUES,
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
        self.assertEqual([row["raw_energy_efficiency_bit_per_j"] for row in normalized], [500000.0, 3000000.0])
        self.assertNotIn("reward", normalized[0])

    def test_safe_division_is_explicit_and_broken_rows_fail_loudly(self):
        self.assertEqual(paper_energy_efficiency(0.0, 0.0), 0.0)
        self.assertGreater(paper_energy_efficiency(1.0, 0.0), 1e11)
        with self.assertRaises(PaperFigureSpecError):
            normalize_episode_ee("td3_dinkelbach", [{"episode": 1, "timely_goodput_mbits": 1.0}])


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
        second = {**base, "fov_delivered_packets": 9, "fov_delivered_e2e_delay_sum_seconds": 0.9, "fov_generated_packets": 18, "fov_violation_packets": 3}
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
    def _write_training_run(self, root, method_id):
        run_dir = root / "training" / method_id
        run_dir.mkdir(parents=True)
        method = MethodSpec.parse(method_id)
        (run_dir / "resolved_config.json").write_text(
            json.dumps({
                "method": method_id,
                "method_spec": method.to_dict(),
                "seed": 20260817,
                "training_manifest_hash": "a" * 64,
                "formal_checkpoint_episode": 2500,
                "status": "COMPLETED",
            }), encoding="utf-8"
        )
        checkpoint = run_dir / "checkpoints" / "models" / "ep_2500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(json.dumps({"episode": 2499, "method": method_id}), encoding="utf-8")
        (checkpoint / "models.pt").write_bytes(b"synthetic")
        rows = [
            {"method_id": method_id, "episode": episode, "reward": 1e9, "timely_goodput_mbits": episode * (1.0 + len(method_id) / 100), "mobility_energy_j": 2.0}
            for episode in range(1, 61)
        ]
        (run_dir / "training_history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return run_dir

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
                "evaluation_episode_count": 3,
            }
            rows.append({**common, "metric": "energy_efficiency_mbit_per_j", "task_type": None, "display_task_type": None, "numerator": 10 + point_index, "numerator_unit": "Mbit", "denominator": 4, "denominator_unit": "J", "value": 1.0 + 0.15 * point_index + 0.05 * method_offset, "value_unit": "Mbit/J", "missing": False})
            for task_type in ("FOV", "COM"):
                delay = 0.004 + 0.002 * point_index + 0.0003 * method_offset + (0.003 if task_type == "COM" else 0.0)
                probability = max(0.0005, 0.4 / (point_index + 2 + method_offset / 4))
                rows.append({**common, "metric": "average_e2e_delay_seconds", "task_type": task_type, "display_task_type": "VS" if task_type == "FOV" else "COM", "numerator": delay * 20, "numerator_unit": "seconds", "denominator": 20, "denominator_unit": "delivered_packets", "value": delay, "value_unit": "seconds", "missing": False})
                rows.append({**common, "metric": "violation_probability", "task_type": task_type, "display_task_type": "VS" if task_type == "FOV" else "COM", "numerator": probability * 100, "numerator_unit": "violation_packets", "denominator": 100, "denominator_unit": "generated_packets", "value": probability, "value_unit": "probability", "missing": False})
        return rows

    @staticmethod
    def _trajectory_artifact():
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
            "scenario_id": "test:synthetic:0",
            "scenario_manifest_hash": "f" * 64,
            "requested_times_seconds": list(times),
            "target_uav_id": 0,
            "snapshots": snapshots,
            "trajectory_history": [],
            "uav_paths": uav_paths,
            "sr_paths": sr_paths,
            "ground_targets": [],
            "initial_sr_teams": [],
            "checkpoint_path": "C:/synthetic/ep_2500",
            "checkpoint_fingerprint": "b" * 64,
        }

    def _write_evaluation_run(self, root, suite, method_id):
        evaluation_dir = root / "evaluations" / suite / method_id
        evaluation_dir.mkdir(parents=True)
        method = MethodSpec.parse(method_id)
        points = list(evaluation_sweep_points(suite))
        metadata_points = []
        for index, point in enumerate(points):
            point_dir = evaluation_dir / point["point_id"]
            point_dir.mkdir()
            if suite == "uav_trajectory_snapshots":
                (point_dir / "trajectory_artifacts.json").write_text(json.dumps([self._trajectory_artifact()]), encoding="utf-8")
            metadata_points.append({**point, "manifest_hash": f"{index:064x}"[-64:], "scenario_ids": [f"scenario-{index}"], "output_directory": str(point_dir)})
        rows = self._aggregate_rows(method_id, suite, points)
        (evaluation_dir / "aggregated_plot_data.json").write_text(json.dumps(rows), encoding="utf-8")
        checkpoint_required = method.learns_movement or method.learns_routing
        metadata = {
            "semantic_suite": suite,
            "method_id": method_id,
            "method_spec": method.to_dict(),
            "checkpoint_required": checkpoint_required,
            "checkpoint_episode": 2500 if checkpoint_required else None,
            "checkpoint_path": "C:/synthetic/ep_2500" if checkpoint_required else None,
            "target_uav_id": 0 if suite == "uav_trajectory_snapshots" else None,
            "points": metadata_points,
        }
        (evaluation_dir / "paper_evaluation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return evaluation_dir

    def _fixture(self, root):
        training_methods = FIGURE_REGISTRY["training_ee_vs_episode"]["methods"]
        training = {method: self._write_training_run(root, method) for method in training_methods}
        evaluations = {}
        for suite, definition in PAPER_EVALUATION_SUITES.items():
            if definition["kind"] == "training_history":
                continue
            evaluations[suite] = {
                method: self._write_evaluation_run(root, suite, method)
                for method in definition["methods"]
            }
        spec = root / "paper_runs.json"
        spec.write_text(json.dumps({
            "target_uav_id": 0,
            "training_runs": {method: {"run_dir": str(path)} for method, path in training.items()},
            "evaluation_runs": {suite: {method: {"evaluation_dir": str(path)} for method, path in methods.items()} for suite, methods in evaluations.items()},
        }), encoding="utf-8")
        return spec, training, evaluations

    def test_build_all_writes_every_semantic_png_pdf_csv_json_and_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, _ = self._fixture(root)
            first = build_paper_figures(spec, output_root=root / "output")
            second = build_paper_figures(spec, figure="training_ee_vs_episode", output_root=root / "output")
            self.assertNotEqual(first["output_directory"], second["output_directory"])
            self.assertEqual(set(first["semantic_figures"]), set(FIGURE_REGISTRY))
            output = Path(first["output_directory"])
            for figure_id, contract in FIGURE_REGISTRY.items():
                stem = contract["output_stem"]
                for suffix in (".png", ".pdf", ".csv", ".json", "_resolved_spec.json"):
                    self.assertTrue((output / f"{stem}{suffix}").is_file(), f"{figure_id}{suffix}")
                resolved = json.loads(
                    (output / f"{stem}_resolved_spec.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(resolved["semantic_figure_id"], figure_id)
            self.assertFalse((output / "unavailable_figures.json").exists())

    def test_ambiguous_and_incompatible_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, training, evaluations = self._fixture(root)
            value = json.loads(spec.read_text())
            method = FIGURE_REGISTRY["training_ee_vs_episode"]["methods"][0]
            value["training_runs"][method] = {"candidates": [str(training[method]), str(training[method])]}
            spec.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AmbiguousPaperRunError):
                build_paper_figures(spec, figure="training_ee_vs_episode", output_root=root / "ambiguous")

            spec, _, evaluations = self._fixture(root / "second")
            bad_method = next(iter(evaluations["task_type_delay_vs_arrival_rate"]))
            metadata_path = evaluations["task_type_delay_vs_arrival_rate"][bad_method] / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["semantic_suite"] = "fixed_roi"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(IncompatiblePaperRunError):
                build_paper_figures(spec, figure="task_type_delay_vs_arrival_rate", output_root=root / "incompatible")

    def test_missing_evaluation_mapping_fails_without_starting_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = root / "paper_runs.json"
            spec.write_text(json.dumps({"evaluation_runs": {}}), encoding="utf-8")
            with self.assertRaises(PaperFigureSpecError):
                build_paper_figures(spec, figure="task_type_delay_vs_arrival_rate", output_root=root / "out")

    def test_incomplete_sweep_is_rejected_even_when_methods_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            for evaluation_dir in evaluations["task_type_delay_vs_arrival_rate"].values():
                metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["points"] = metadata["points"][:-1]
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(IncompatiblePaperRunError):
                build_paper_figures(
                    spec,
                    figure="task_type_delay_vs_arrival_rate",
                    output_root=root / "incomplete",
                )


if __name__ == "__main__":
    unittest.main()
