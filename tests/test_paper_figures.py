from copy import deepcopy
from dataclasses import asdict
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import zipfile

from dinkelbach_blocks import DinkelbachBlockState, dinkelbach_config_metadata
from experiment_config import (
    MethodSpec,
    comparison_method_configuration,
    effective_training_config,
)
from HRL_task_aware import formal_training_config
from Packet_scheduler_v1 import (
    EPISODE_INJECTION_CUTOFF_SECONDS,
    TASK_DEADLINE_SECONDS,
)
from paper_evaluation import (
    PAPER_EVALUATION_SUITES,
    aggregate_paper_point_metrics,
    evaluation_sweep_points,
)
from paper_figure_registry import FIGURE_REGISTRY, PAPER_METHOD_MAPPINGS
from paper_figures import (
    AmbiguousPaperRunError,
    IncompatiblePaperRunError,
    PaperFigureSpecError,
    _build_violation,
    build_paper_figures,
    causal_trailing_average,
    normalize_episode_ee,
    paper_energy_efficiency,
    render_standalone_trajectory_source,
)
from scenario_manifest import generate_manifest
from routing_lifecycle import RoutingLearnerLifecycle
from training_checkpoint import (
    CHECKPOINT_PROVENANCE_FIELDS,
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CHECKPOINT_TYPE,
    checkpoint_artifact_provenance,
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
            "fov_eligible_packets": 2,
            "fov_violation_packets": 1,
            "com_delivered_packets": 0,
            "com_delivered_e2e_delay_sum_seconds": 0.0,
            "com_generated_packets": 0,
            "com_eligible_packets": 0,
            "com_violation_packets": 0,
        }
        second = {
            **base,
            "fov_delivered_packets": 9,
            "fov_delivered_e2e_delay_sum_seconds": 0.9,
            "fov_generated_packets": 18,
            "fov_eligible_packets": 18,
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

    def test_violation_figure_reads_only_combined_all_rows(self):
        figure_id = "task_type_delay_violation_vs_target_delay"
        methods = tuple(PAPER_METHOD_MAPPINGS[figure_id])
        aggregate_rows = []
        for method in methods:
            for swept_task in ("COM", "FOV"):
                for threshold in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                    point_id = (
                        f"{swept_task.lower()}_deadline_{threshold:g}s"
                    )
                    aggregate_rows.extend(
                        (
                            {
                                "method_id": method,
                                "point_id": point_id,
                                "metric": "violation_probability",
                                "task_type": "ALL",
                                "swept_task": swept_task,
                                "x_value": threshold,
                                "value": 0.9,
                                "missing": False,
                            },
                            {
                                "method_id": method,
                                "point_id": point_id,
                                "metric": "violation_probability",
                                "task_type": swept_task,
                                "swept_task": swept_task,
                                "x_value": threshold,
                                "value": 0.1,
                                "missing": False,
                            },
                        )
                    )
        runs = {
            method: {
                "evaluation_dir": "synthetic",
                "aggregate_rows": [
                    row for row in aggregate_rows if row["method_id"] == method
                ],
                "checkpoint_provenance": {},
                "point_provenance": {},
            }
            for method in methods
        }
        with mock.patch(
            "paper_figures._resolve_suite_runs", return_value=runs
        ), mock.patch("paper_figures._emit", return_value={}) as emit:
            _build_violation({}, Path("synthetic.json"), Path("."), "sha")

        figure = emit.call_args.args[1]
        plotted = emit.call_args.args[3]
        try:
            self.assertEqual(
                figure.axes[0].get_ylabel(), "Delay Violation Probability"
            )
            self.assertTrue(plotted)
            self.assertTrue(all(row["task_type"] == "ALL" for row in plotted))
            self.assertTrue(all(row["value"] == 0.9 for row in plotted))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


class SyntheticFigureBuildTest(unittest.TestCase):
    @staticmethod
    def _formal_config(method):
        base = asdict(formal_training_config(1500, random_seed=TRAINING_SEED))
        return effective_training_config(base, method)

    @classmethod
    def _checkpoint_metadata(cls, method, formal_config):
        experiment = {
            "method_id": method.method_id,
            "method_spec": method.to_dict(),
            "method_spec_fingerprint": method.fingerprint,
            "training_seed": TRAINING_SEED,
            "git_sha": "synthetic-training-sha",
            "movement_agent": method.agent,
            "reward_mode": method.reward_mode,
            "task_potential_enabled": method.task_potential_enabled,
            "formal_config": formal_config,
            **comparison_method_configuration(method),
        }
        if method.uses_dinkelbach:
            state = DinkelbachBlockState.from_config(formal_config)
            for _ in range(1500):
                state.record_episode(1.0, 2.0)
            experiment.update(
                {
                    **dinkelbach_config_metadata(formal_config),
                    "lambda_ee": state.current_lambda,
                    "dinkelbach_state": state.training_state(),
                }
            )
        metadata = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": MODEL_CHECKPOINT_TYPE,
            "episode": 1499,
            "movement_agent_kind": method.agent,
            "movement_agent_configuration": formal_config[
                "movement_agent_configuration"
            ],
            "experiment": experiment,
        }
        metadata["routing_agent_kind"] = method.routing
        if method.routing == "safe_ddqn":
            metadata["routing_agent_configuration"] = {
                "lambda_cost": 0.0,
                "initial_lambda_cost": 0.0,
                "eta_c": 0.01,
                "qos_target_probability": 0.1,
                "lambda_update_scope": "episode_end",
                "cost_denominator": "eligible_packets",
                "mid_episode_checkpoint_supported": False,
                "routing_optimizer_update_count": 17,
                "routing_target_update_count": 17,
            }
        lifecycle = (
            RoutingLearnerLifecycle(
                global_slot_count=600000,
                optimizer_update_count=17,
                target_update_count=17,
                epsilon_decay_start_slot=63,
                last_optimizer_update_slot=600000,
            ).state_dict()
            if method.learns_routing
            else None
        )
        resolved = deepcopy(formal_config)
        resolved.update(
            {
                "method_key": method.method_id,
                "method_id": method.method_id,
                "method_spec": method.to_dict(),
                "method_spec_fingerprint": method.fingerprint,
                "training_episode_count": 1500,
                "training_seed": TRAINING_SEED,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            }
        )
        metadata["training_provenance"] = {
            "training_episode_count": 1500,
            "training_git_sha": "synthetic-training-sha",
            "resolved_training_config": resolved,
            "routing_lifecycle": lifecycle,
            "safe_ddqn_constraint_state": (
                deepcopy(metadata.get("routing_agent_configuration"))
                if method.routing == "safe_ddqn"
                else None
            ),
            "provenance_complete": True,
        }
        return metadata

    @staticmethod
    def _write_synthetic_models(path, marker):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", f"synthetic:{marker}".encode())
            archive.writestr("archive/version", b"3\n")

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
            "formal_checkpoint_episode": 1500,
            "training_config": formal_config,
            "status": "COMPLETED",
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        checkpoint = run_dir / "checkpoints" / "models" / "ep_1500"
        checkpoint.mkdir(parents=True)
        metadata = self._checkpoint_metadata(method, formal_config)
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        self._write_synthetic_models(checkpoint / "models.pt", method_id)
        if include_history:
            rows = [
                {
                    "method_id": method_id,
                    "episode": episode,
                    "reward": 1e9,
                    "timely_goodput_mbits": episode * (1.0 + len(method_id) / 100),
                    "mobility_energy_j": 2.0,
                }
                for episode in range(1, 1501)
            ]
            (run_dir / "training_history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        return run_dir, metadata

    @staticmethod
    def _episode_rows(method_id, point_index):
        method_offset = sum(ord(char) for char in method_id) % 7
        rows = []
        for episode in range(EVALUATION_EPISODES):
            fov_delivered = 3 + episode
            com_delivered = 2 + episode
            fov_generated = fov_delivered + 2
            com_generated = com_delivered + 3
            rows.append(
                {
                    "episode": episode + 1,
                    "timely_goodput_mbits": 2.0 + point_index + method_offset / 10,
                    "total_mobility_energy_j": 4.0 + episode,
                    "fov_delivered_packets": fov_delivered,
                    "fov_delivered_e2e_delay_sum_seconds": fov_delivered
                    * (0.01 + point_index / 1000),
                    "fov_generated_packets": fov_generated,
                    "fov_eligible_packets": fov_generated,
                    "fov_violation_packets": 1 + episode,
                    "com_delivered_packets": com_delivered,
                    "com_delivered_e2e_delay_sum_seconds": com_delivered
                    * (0.02 + point_index / 1000),
                    "com_generated_packets": com_generated,
                    "com_eligible_packets": com_generated,
                    "com_violation_packets": 2 + episode,
                }
            )
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
            "packet_injection_cutoff_seconds": float(
                point["overrides"].get(
                    "packet_injection_cutoff_seconds",
                    EPISODE_INJECTION_CUTOFF_SECONDS,
                )
            ),
            "units": {
                "traffic_rate": "packets/s",
                "deadline": "seconds",
                "packet_injection_cutoff": "seconds",
            },
        }

    @staticmethod
    def _trajectory_artifact(manifest, checkpoint, provenance):
        times = (5.0, 10.0, 15.0, 25.0)
        phases = ("Search", "FOV", "FOV+COM", "Hover")
        history_times = (0.0, *times)
        uav_paths = {
            str(uid): [
                {
                    "actual_time_seconds": time,
                    "x": 100 + (uid % 4) * 200 + time,
                    "y": 100 + (uid // 4) * 200,
                    "z": 90 + uid,
                    "task_phase": (
                        "Search"
                        if time == 0.0
                        else phases[times.index(time)]
                        if uid == 0
                        else "Hover"
                    ),
                    "assigned_tasks": [
                        {
                            "task_type": (
                                "Search"
                                if time == 0.0
                                else phases[times.index(time)]
                                if uid == 0
                                else "Hovering"
                            )
                        }
                    ],
                }
                for time in history_times
            ]
            for uid in range(10)
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
                "uavs": [{"uav_id": uid, "x": uav_paths[str(uid)][history_times.index(time)]["x"], "y": uav_paths[str(uid)][history_times.index(time)]["y"], "z": 90 + uid, "task_phase": phase if uid == 0 else "Hover", "assigned_tasks": [{"task_type": phase if uid == 0 else "Hovering"}]} for uid in range(10)],
                "sr_teams": [{"sr_id": uid, "x": sr_paths[str(uid)][history_times.index(time)]["x"], "y": 500, "z": 0, "active": True} for uid in range(2)],
                "ground_targets": [{"gt_id": 0, "x": 650.0, "y": 650.0, "z": 0.0, "radius_m": 80.0, "detected": time >= 10, "detected_by_uav_id": 0 if time >= 10 else None}],
                "ground_station": {"gs_id": 10, "x": 0.0, "y": 0.0, "z": 0.0},
                "active_links": [
                    {"sender_id": 0, "receiver_id": 1, "link_type": "U2U", "bandwidth_hz": 5e6, "capacity_bits_per_second": 1e6},
                    {"sender_id": 0, "receiver_id": 2, "link_type": "S2U", "bandwidth_hz": 5e6, "capacity_bits_per_second": 0.5e6},
                ],
                "sensing_coverage": [{"uav_id": 0, "geometry": "axis_aligned_ground_rectangle", "center_x": 100 + time, "center_y": 100, "ground_z": 0, "width_m": 200, "height_m": 200, "clipped_bounds": {"x_min": 0, "x_max": 200 + time, "y_min": 0, "y_max": 200}, "model": {"f_m": 0.004, "image_width_m": 0.008, "image_length_m": 0.012}}],
            })
        return {
            "scenario_id": manifest.episodes[0]["scenario_id"],
            "scenario_manifest_hash": manifest.content_hash,
            "requested_times_seconds": list(times),
            "target_uav_id": 0,
            "snapshots": snapshots,
            "trajectory_history": [
                {
                    "actual_time_seconds": time,
                    "uavs": [
                        {
                            "uav_id": uid,
                            **{
                                key: value
                                for key, value in uav_paths[str(uid)][
                                    history_times.index(time)
                                ].items()
                                if key != "actual_time_seconds"
                            },
                        }
                        for uid in range(10)
                    ],
                    "sr_teams": [
                        {
                            "sr_id": uid,
                            **{
                                key: value
                                for key, value in sr_paths[str(uid)][
                                    history_times.index(time)
                                ].items()
                                if key != "actual_time_seconds"
                            },
                            "active": True,
                        }
                        for uid in range(2)
                    ],
                }
                for time in history_times
            ],
            "uav_paths": uav_paths,
            "sr_paths": sr_paths,
            "ground_targets": [],
            "initial_sr_teams": [],
            "checkpoint_path": str(checkpoint),
            "checkpoint_required": True,
            "checkpoint_fingerprint": provenance[
                "checkpoint_metadata_fingerprint"
            ],
            **provenance,
        }

    def _write_evaluation_run(self, root, suite, method_id, training_runs):
        evaluation_dir = root / "evaluations" / suite / method_id
        evaluation_dir.mkdir(parents=True)
        method = MethodSpec.parse(method_id)
        checkpoint_required = method.learns_movement or method.learns_routing
        training_run = training_runs.get(method_id)
        checkpoint = (
            training_run / "checkpoints" / "models" / "ep_1500"
            if checkpoint_required
            else None
        )
        checkpoint_metadata = (
            json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            if checkpoint_required
            else None
        )
        provenance = (
            checkpoint_artifact_provenance(
                checkpoint, metadata=checkpoint_metadata
            )
            if checkpoint_required
            else {field: None for field in CHECKPOINT_PROVENANCE_FIELDS}
        )
        checkpoint_training = (
            deepcopy(checkpoint_metadata["training_provenance"])
            if checkpoint_required
            else None
        )
        runtime = {
            "evaluation_episode_count": EVALUATION_EPISODES,
            "evaluation_git_sha": "synthetic-evaluation-sha",
            "resolved_evaluation_config": {
                "method_id": method_id,
                "evaluation_episode_count": EVALUATION_EPISODES,
                "learning_state_frozen": True,
            },
            "routing_lifecycle": (
                RoutingLearnerLifecycle().state_dict()
                if method.learns_routing
                else None
            ),
            "lambda_cost_source": (
                "checkpoint_frozen"
                if method.routing == "safe_ddqn"
                else None
            ),
            "safe_ddqn_constraint_state": (
                deepcopy(
                    checkpoint_metadata["routing_agent_configuration"]
                )
                if method.routing == "safe_ddqn"
                else None
            ),
        }
        provenance.update(
            {
                "checkpoint_training_provenance": checkpoint_training,
                "evaluation_runtime_provenance": runtime,
                "checkpoint_training_episode_count": (
                    1500 if checkpoint_training is not None else None
                ),
                "evaluation_episode_count": EVALUATION_EPISODES,
                "checkpoint_training_git_sha": (
                    "synthetic-training-sha"
                    if checkpoint_training is not None
                    else None
                ),
                "evaluation_git_sha": "synthetic-evaluation-sha",
            }
        )
        points = list(evaluation_sweep_points(suite))
        metadata_points = []
        all_aggregates = []
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
                    manifest, checkpoint, provenance
                )
                (point_dir / "trajectory_artifacts.json").write_text(
                    json.dumps([artifact]), encoding="utf-8"
                )
            episode_rows = self._episode_rows(method_id, index)
            per_episode_path = point_dir / "per_episode.jsonl"
            per_episode_path.write_text(
                "".join(json.dumps(row) + "\n" for row in episode_rows),
                encoding="utf-8",
            )
            point_aggregates = aggregate_paper_point_metrics(
                method_id, suite, point, episode_rows
            )
            aggregate_path = point_dir / "aggregated_plot_data.json"
            aggregate_path.write_text(
                json.dumps(point_aggregates), encoding="utf-8"
            )
            all_aggregates.extend(point_aggregates)
            run_metadata_path = point_dir / "run_metadata.json"
            run_metadata_path.write_text(
                json.dumps(
                    {
                        "method_id": method_id,
                        "semantic_suite": suite,
                        "checkpoint_required": checkpoint_required,
                        "checkpoint_path": (
                            str(checkpoint) if checkpoint_required else None
                        ),
                        **provenance,
                    }
                ),
                encoding="utf-8",
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
                "num_uav": 10,
                "resolved_overrides": self._resolved_overrides(suite, point),
                "checkpoint_required": checkpoint_required,
                "checkpoint_path": str(checkpoint) if checkpoint_required else None,
                **provenance,
                "output_directory": str(point_dir.resolve()),
                "outputs": {
                    "per_episode_jsonl": str(per_episode_path.resolve()),
                    "run_metadata": str(run_metadata_path.resolve()),
                },
                "aggregated_plot_data": str(aggregate_path.resolve()),
            })
        (evaluation_dir / "aggregated_plot_data.json").write_text(
            json.dumps(all_aggregates), encoding="utf-8"
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
            "checkpoint_episode": 1500 if checkpoint_required else None,
            "checkpoint_path": str(checkpoint) if checkpoint_required else None,
            **provenance,
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
            with mock.patch(
                "training_checkpoint.torch.load",
                side_effect=AssertionError("figure builder loaded model weights"),
            ):
                result = build_paper_figures(spec, output_root=root / "output")
                second = build_paper_figures(
                    spec,
                    figure="training_ee_vs_episode",
                    output_root=root / "output",
                )
            self.assertNotEqual(result["output_directory"], second["output_directory"])
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
                for provenance in resolved[
                    "method_to_checkpoint_provenance"
                ].values():
                    if provenance["checkpoint_path"] is not None:
                        for field in CHECKPOINT_PROVENANCE_FIELDS:
                            self.assertEqual(len(provenance[field]), 64)
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

    def test_standalone_trajectory_source_redraws_without_evaluation_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            result = build_paper_figures(
                spec, figure="fig3", output_root=root / "output"
            )
            self.assertEqual(len(result["semantic_figures"]), 4)
            output = Path(result["output_directory"])
            artifact_path = (
                evaluations["uav_trajectory_snapshots"]["td3_dinkelbach"]
                / "trajectory"
                / "trajectory_artifacts.json"
            )
            artifact_path.rename(artifact_path.with_suffix(".unavailable"))

            source_5 = json.loads(
                (output / "UAV_trajectory_t_5s.json").read_text(encoding="utf-8")
            )
            source_10 = json.loads(
                (output / "UAV_trajectory_t_10s.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(source_5["uavs"]), 10)
            self.assertEqual(len(source_5["uav_paths"]), 10)
            self.assertTrue(source_5["sr_paths"])
            self.assertTrue(source_5["ground_targets"])
            self.assertTrue(source_5["ground_station"])
            self.assertTrue(source_5["active_links"])
            self.assertTrue(source_5["sensing_coverage"])
            self.assertTrue(
                all(
                    state["actual_time_seconds"] <= 5.0
                    for path in source_5["uav_paths"].values()
                    for state in path
                )
            )
            self.assertEqual(source_10["actual_phase"], "FOV")
            for field in CHECKPOINT_PROVENANCE_FIELDS:
                self.assertEqual(len(source_10[field]), 64)

            figure = render_standalone_trajectory_source(source_10)
            try:
                self.assertEqual(len(figure.axes), 1)
                self.assertEqual(
                    figure.axes[0].get_title(), "t = 10 s: UAV in VS mode"
                )
                self.assertNotIn("FOV", figure.axes[0].get_title())
                self.assertGreater(len(figure.axes[0].collections), 2)
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)

            with (output / "UAV_trajectory_t_5s.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                record_types = {row["record_type"] for row in csv.DictReader(handle)}
            self.assertTrue(
                {
                    "uav_path",
                    "uav_snapshot",
                    "sr_path",
                    "sr_snapshot",
                    "ground_target",
                    "ground_station",
                    "active_link",
                    "sensing_coverage",
                }.issubset(record_types)
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
            metadata["checkpoint_path"] = str(root / "does-not-exist" / "ep_1500")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatiblePaperRunError, "checkpoint.*path"):
                build_paper_figures(spec, figure="task_assignment_ee_vs_number_of_rois", output_root=root / "missing-checkpoint")

            spec, training, _ = self._fixture(root / "mutated-models")
            checkpoint = training[method] / "checkpoints" / "models" / "ep_1500"
            self._write_synthetic_models(checkpoint / "models.pt", "mutated-payload")
            with self.assertRaisesRegex(
                IncompatiblePaperRunError, "checkpoint_models_sha256"
            ):
                build_paper_figures(
                    spec,
                    figure="task_assignment_ee_vs_number_of_rois",
                    output_root=root / "mutated-models-out",
                )

            spec, training, _ = self._fixture(root / "missing-models")
            checkpoint = training[method] / "checkpoints" / "models" / "ep_1500"
            (checkpoint / "models.pt").rename(checkpoint / "models.unavailable")
            with self.assertRaisesRegex(
                IncompatiblePaperRunError, "model payload is missing"
            ):
                build_paper_figures(
                    spec,
                    figure="training_ee_vs_episode",
                    output_root=root / "missing-models-out",
                )

    def test_random_baseline_rejects_any_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _, evaluations = self._fixture(root)
            method = "kkm_random_action_random_routing"
            evaluation_dir = evaluations["fixed_roi"][method]
            metadata_path = evaluation_dir / "paper_evaluation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_path"] = str(root / "fake" / "ep_1500")
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
            with self.assertRaisesRegex(IncompatiblePaperRunError, "missing canonical"):
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
            with self.assertRaisesRegex(IncompatiblePaperRunError, "unexpected canonical"):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "bad-metric-out")

            aggregate_path.write_text(json.dumps(original_rows), encoding="utf-8")
            top_mutated = [dict(row) for row in original_rows]
            top_ee = next(
                row for row in top_mutated
                if row["point_id"] == "com_rate_50"
                and row["metric"] == "energy_efficiency_mbit_per_j"
            )
            top_ee["numerator"] += 1.0
            top_ee["value"] = top_ee["numerator"] / top_ee["denominator"]
            aggregate_path.write_text(json.dumps(top_mutated), encoding="utf-8")
            with self.assertRaisesRegex(
                IncompatiblePaperRunError, "top-level/point-level mismatch"
            ):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "top-point-mismatch")

            aggregate_path.write_text(json.dumps(original_rows), encoding="utf-8")
            point_dir = evaluation_dir / "com_rate_50"
            point_aggregate_path = point_dir / "aggregated_plot_data.json"
            point_rows = json.loads(point_aggregate_path.read_text(encoding="utf-8"))
            point_ee = next(
                row for row in point_rows
                if row["metric"] == "energy_efficiency_mbit_per_j"
            )
            point_ee["numerator"] += 1.0
            point_ee["value"] = point_ee["numerator"] / point_ee["denominator"]
            point_aggregate_path.write_text(json.dumps(point_rows), encoding="utf-8")
            with self.assertRaisesRegex(
                IncompatiblePaperRunError, "point-level/per-episode mismatch"
            ):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "point-episode-mismatch")

            point_aggregate_path.write_text(
                json.dumps(
                    [
                        row
                        for row in original_rows
                        if row["point_id"] == "com_rate_50"
                    ]
                ),
                encoding="utf-8",
            )
            per_episode_path = point_dir / "per_episode.jsonl"
            episode_rows = [
                json.loads(line)
                for line in per_episode_path.read_text(encoding="utf-8").splitlines()
            ]
            episode_rows[0]["timely_goodput_mbits"] += 1.0
            per_episode_path.write_text(
                "".join(json.dumps(row) + "\n" for row in episode_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                IncompatiblePaperRunError, "point-level/per-episode mismatch"
            ):
                build_paper_figures(spec, figure="com_task_delay_vs_arrival_rate", output_root=root / "episode-recompute-mismatch")

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
            with self.assertRaisesRegex(IncompatiblePaperRunError, "1..1500"):
                build_paper_figures(spec, figure="training_ee_vs_episode", output_root=root / "short-out")


if __name__ == "__main__":
    unittest.main()
