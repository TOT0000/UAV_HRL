import json
import math
from pathlib import Path
from statistics import stdev
import tempfile
import unittest
from unittest import mock

import numpy as np

from centralized_movement import MOVEMENT_STATE_DIM, aggregate_coverage_map
from evaluation_selection import (
    DEFAULT_ENVIRONMENT_SIZES_M,
    resolve_episode_horizons_s,
    resolve_environment_sizes_m,
)
from experiment_config import (
    CANONICAL_UAV_INITIAL_XY_M,
    ENVIRONMENT_SIZE_REFERENCE_M,
    METHOD_REGISTRY,
)
from HRL_task_aware import _normalize_evaluation_overrides, normalized_remaining_time
from observation_strategy import ROUTING_STATE_DIM
from paper_evaluation import (
    PAPER_EVALUATION_SUITES,
    evaluation_sweep_points,
    run_paper_evaluation,
)
from paper_metrics import (
    ENVIRONMENT_SIZE_AGGREGATE_CONTRACT_VERSION,
    LEGACY_ENVIRONMENT_SIZE_AGGREGATE_CONTRACT_VERSION,
    aggregate_paper_point_metrics,
    validate_canonical_aggregate_rows,
)
from run_paper_evaluation import main as run_evaluation_main
from scenario_manifest import (
    ENVIRONMENT_SIZE_SCENARIO_SCHEMA_VERSION,
    SCENARIO_SCHEMA_VERSION,
    ScenarioManifest,
    environment_size_uav_initial_xy_m,
    generate_manifest,
)
from Simulator import Simulator


class EnvironmentSizeSelectionTest(unittest.TestCase):
    def test_default_and_explicit_environment_sizes(self):
        self.assertEqual(
            resolve_environment_sizes_m(),
            (750, 1000, 1250, 1500, 1750, 2000),
        )
        self.assertEqual(
            resolve_environment_sizes_m(environment_sizes_m=(1500, 750, 1500)),
            (1500, 750),
        )
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_environment_sizes_m(750, (1000,))
        with self.assertRaisesRegex(ValueError, "one of"):
            resolve_environment_sizes_m(900)

    def test_suite_supports_every_registered_method_and_one_fixed_roi(self):
        self.assertEqual(
            set(PAPER_EVALUATION_SUITES["environment_size"]["methods"]),
            set(METHOD_REGISTRY),
        )
        points = evaluation_sweep_points("environment_size")
        self.assertEqual(
            tuple(point["x_value"] for point in points),
            DEFAULT_ENVIRONMENT_SIZES_M,
        )
        self.assertEqual({point["fixed_num_gt"] for point in points}, {8})
        self.assertEqual({point["x_unit"] for point in points}, {"m"})
        fixed_points = evaluation_sweep_points("fixed_roi")
        self.assertEqual(
            tuple(point["point_id"] for point in fixed_points),
            tuple(f"roi_{value}" for value in range(2, 9)),
        )
        self.assertTrue(all(point["overrides"] == {} for point in fixed_points))
        self.assertTrue(
            all("point_evaluation_purpose" not in point for point in fixed_points)
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation_sweep_points("environment_size", roi_counts=(4, 8))

    def test_horizon_selectors_are_strict_and_cartesian(self):
        self.assertEqual(resolve_episode_horizons_s(), (60,))
        self.assertEqual(
            resolve_episode_horizons_s(episode_horizons_s=(120, 60, 120)),
            (120, 60),
        )
        self.assertEqual(resolve_episode_horizons_s(episode_horizon_s=137), (137,))
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_episode_horizons_s(60, (120,))
        with self.assertRaisesRegex(ValueError, "at least one"):
            resolve_episode_horizons_s(episode_horizons_s=())
        for invalid in (-1, 0, 1, 2, 2.5, True, "120"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_episode_horizons_s(episode_horizon_s=invalid)
        points = evaluation_sweep_points(
            "environment_size",
            environment_sizes_m=(750, 1000),
            episode_horizons_s=(60, 120, 600),
        )
        self.assertEqual(
            [point["point_id"] for point in points],
            [
                "map_750m", "map_750m_t120s", "map_750m_t600s",
                "map_1000m", "map_1000m_t120s", "map_1000m_t600s",
            ],
        )
        self.assertEqual(
            [point["packet_injection_cutoff_s"] for point in points],
            [57.5, 117.5, 597.5, 57.5, 117.5, 597.5],
        )
        self.assertEqual(points[-1]["movement_transition_count"], 600)
        self.assertEqual(points[-1]["routing_slot_count"], 2400)
        self.assertEqual(len({point["point_id"] for point in points}), len(points))
        self.assertEqual(normalized_remaining_time(240, 60), 0.75)

    def test_environment_size_point_shift_semantics(self):
        points = {
            point["x_value"]: point
            for point in evaluation_sweep_points(
                "environment_size", environment_sizes_m=(750, 1000, 2000)
            )
        }
        baseline = points[1000]
        self.assertFalse(baseline["zero_shot_environment_shift"])
        self.assertIsNone(baseline["environment_shift_type"])
        self.assertEqual(
            baseline["point_evaluation_purpose"],
            "in_distribution_environment_size_baseline",
        )
        for size in (750, 2000):
            with self.subTest(size=size):
                self.assertTrue(points[size]["zero_shot_environment_shift"])
                self.assertEqual(points[size]["environment_shift_type"], "map_size")
                self.assertEqual(
                    points[size]["point_evaluation_purpose"],
                    "zero_shot_environment_shift_evaluation",
                )

    def test_cli_selector_scope_and_forwarding(self):
        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation", return_value={}
        ) as runner:
            self.assertEqual(
                run_evaluation_main(
                    [
                        "kkm_random_action_random_routing",
                        "--suite",
                        "environment_size",
                        "--environment-sizes-m",
                        "2000",
                        "750",
                        "2000",
                        "--roi-count",
                        "6",
                    ]
                ),
                0,
            )
        self.assertEqual(runner.call_args.kwargs["environment_sizes_m"], (2000, 750))
        self.assertEqual(runner.call_args.kwargs["roi_counts"], (6,))
        self.assertEqual(runner.call_args.kwargs["episode_horizons_s"], (60,))
        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation", return_value={}
        ) as horizon_runner:
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite", "environment_size",
                    "--environment-size-m", "1000",
                    "--episode-horizons-s", "60", "120",
                    "--episodes", "2",
                ]
            )
        self.assertEqual(
            horizon_runner.call_args.kwargs["episode_horizons_s"], (60, 120)
        )
        with self.assertRaisesRegex(ValueError, "not --roi-counts"):
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite",
                    "environment_size",
                    "--roi-counts",
                    "4",
                    "8",
                ]
            )
        with self.assertRaisesRegex(ValueError, "only for environment_size"):
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite",
                    "fixed_roi",
                    "--environment-size-m",
                    "750",
                ]
            )
        with self.assertRaisesRegex(ValueError, "only for environment_size"):
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite", "fixed_roi",
                    "--episode-horizon-s", "120",
                ]
            )


class EnvironmentSizeManifestAndSimulatorTest(unittest.TestCase):
    def test_fixed_roi_v8_identity_path_is_unchanged(self):
        manifest = generate_manifest("test", 20260817, 1, num_gt=8)
        self.assertEqual(manifest.schema_version, SCENARIO_SCHEMA_VERSION)
        self.assertEqual(manifest.environment_width_m, 1000)
        self.assertEqual(manifest.environment_height_m, 1000)
        self.assertNotIn("environment_size_m", manifest.to_dict())
        self.assertEqual(
            manifest.episodes[0]["scenario_id"],
            "test:uav-hrl-scenario-v8:fixed-8:20260817:000000",
        )
        self.assertEqual(
            manifest.content_hash,
            "afcb7256a161896d6ecc48e83948410fe9cadbb83610fabccd7a6790a634efed",
        )

    def test_size_manifests_are_deterministic_distinct_and_in_bounds(self):
        manifests = {
            size: generate_manifest(
                "test", 20260817, 2, num_gt=8, environment_size_m=size
            )
            for size in (750, 1000, 2000)
        }
        self.assertEqual(len({item.content_hash for item in manifests.values()}), 3)
        for size, manifest in manifests.items():
            with self.subTest(size=size):
                self.assertEqual(
                    manifest.to_dict(),
                    generate_manifest(
                        "test", 20260817, 2, num_gt=8, environment_size_m=size
                    ).to_dict(),
                )
                self.assertEqual(
                    manifest.schema_version,
                    ENVIRONMENT_SIZE_SCENARIO_SCHEMA_VERSION,
                )
                loaded = ScenarioManifest.from_dict(manifest.to_dict())
                self.assertEqual(loaded.environment_size_m, size)
                entry = manifest.episodes[0]
                expected_boundary = {
                    0: (0.0, size / 2.0),
                    1: (float(size), size / 2.0),
                    2: (size / 2.0, 0.0),
                    3: (size / 2.0, float(size)),
                }
                for sr in entry["sr_teams"]:
                    self.assertEqual(
                        tuple(sr["position"][:2]), expected_boundary[sr["sr_id"] % 4]
                    )
                for gt in entry["ground_targets"]:
                    x, y, _ = gt["position"]
                    radius = gt["radius_m"]
                    self.assertLessEqual(radius, x)
                    self.assertLessEqual(x, size - radius)
                    self.assertLessEqual(radius, y)
                    self.assertLessEqual(y, size - radius)

    def test_size_manifests_pair_latent_roi_layout_and_keep_distinct_identity(self):
        sizes = (750, 1000, 2000)
        manifests = {
            size: generate_manifest(
                "test", 20260817, 3, num_gt=8, environment_size_m=size
            )
            for size in sizes
        }
        self.assertEqual(ENVIRONMENT_SIZE_REFERENCE_M, 750)
        self.assertEqual(len({item.content_hash for item in manifests.values()}), 3)
        self.assertEqual(
            len({item.config_fingerprint for item in manifests.values()}), 3
        )
        for episode_index in range(3):
            reference = manifests[750].episodes[episode_index]["ground_targets"]
            reference_by_id = {int(gt["gt_id"]): gt for gt in reference}
            for size in sizes:
                entry = manifests[size].episodes[episode_index]
                targets = {int(gt["gt_id"]): gt for gt in entry["ground_targets"]}
                self.assertEqual(set(targets), set(reference_by_id))
                self.assertIn(f"map-{size}m", entry["scenario_id"])
                for gt_id, target in targets.items():
                    x, y, _ = map(float, target["position"])
                    ref_x, ref_y, _ = map(
                        float, reference_by_id[gt_id]["position"]
                    )
                    expected_u = (
                        (ref_x - 80.0) / (750.0 - 160.0),
                        (ref_y - 80.0) / (750.0 - 160.0),
                    )
                    actual_u = (
                        (x - 80.0) / (float(size) - 160.0),
                        (y - 80.0) / (float(size) - 160.0),
                    )
                    np.testing.assert_allclose(
                        actual_u, expected_u, rtol=0, atol=1e-14
                    )
                    self.assertGreaterEqual(math.hypot(x, y), 200.0)
                    self.assertGreaterEqual(
                        math.hypot(x, y), math.hypot(ref_x, ref_y)
                    )
                positions = [
                    tuple(map(float, target["position"][:2]))
                    for target in targets.values()
                ]
                for index, first in enumerate(positions):
                    for second in positions[index + 1 :]:
                        self.assertGreaterEqual(math.dist(first, second), 80.0)
        coordinates = {
            size: tuple(
                tuple(gt["position"])
                for gt in manifests[size].episodes[0]["ground_targets"]
            )
            for size in sizes
        }
        self.assertEqual(len(set(coordinates.values())), len(sizes))


class EnvironmentSizeAggregateTest(unittest.TestCase):
    point = {
        "point_id": "map_750m",
        "x_value": 750,
        "x_unit": "m",
        "fixed_num_gt": 8,
        "environment_width_m": 750,
        "environment_height_m": 750,
    }

    @staticmethod
    def _episode(seed, bits, horizon, energy, discovery, coverage):
        return {
            "training_seed": seed,
            "timely_goodput_mbits": bits / 1e6,
            "total_timely_useful_mbits": bits / 1e6,
            "total_timely_useful_bits": bits,
            "episode_horizon_seconds": horizon,
            "total_mobility_energy_j": energy,
            "found_GT_ratio": discovery,
            "coverage": coverage,
            "all_rois_discovered": discovery >= 0.9,
            "fov_delivered_packets": 2,
            "fov_delivered_e2e_delay_sum_seconds": 0.5,
            "fov_eligible_packets": 4,
            "fov_violation_packets": 1,
            "com_delivered_packets": 1,
            "com_delivered_e2e_delay_sum_seconds": 0.25,
            "com_eligible_packets": 3,
            "com_violation_packets": 1,
        }

    def _rows(self):
        episodes = [
            self._episode(11, 100_000, 10, 100, 0.2, 0.4),
            self._episode(11, 600_000, 30, 300, 0.4, 0.6),
            self._episode(22, 60_000, 20, 50, 0.9, 0.1),
        ]
        return aggregate_paper_point_metrics(
            "method", "environment_size", self.point, episodes
        )

    def test_new_metrics_use_per_seed_ratios_then_equal_seed_weighting(self):
        rows = self._rows()
        self.assertEqual(len(rows), 11)
        validate_canonical_aggregate_rows(
            rows, "method", "map_750m", suite="environment_size"
        )
        metrics = {row["metric"]: row for row in rows if row["task_type"] is None}
        expected = {
            "timely_throughput_kbit_per_s": ([17.5, 3.0], "kbit/s"),
            "mobility_energy_j_per_episode": ([200.0, 50.0], "J/episode"),
            "roi_discovery_ratio": ([0.3, 0.9], "ratio"),
            "terminal_coverage_ratio": ([0.5, 0.1], "ratio"),
            "all_rois_discovered_probability": ([0.0, 1.0], "probability"),
        }
        for metric, (seed_values, unit) in expected.items():
            with self.subTest(metric=metric):
                row = metrics[metric]
                np.testing.assert_allclose(row["per_seed_values"], seed_values)
                self.assertAlmostEqual(row["value"], sum(seed_values) / 2.0)
                self.assertAlmostEqual(row["sample_stddev"], stdev(seed_values))
                self.assertEqual(row["value_unit"], unit)
                self.assertEqual(row["valid_training_seed_count"], 2)
                self.assertEqual(
                    row["suite_aggregate_contract_version"],
                    ENVIRONMENT_SIZE_AGGREGATE_CONTRACT_VERSION,
                )
                self.assertGreater(row["ci95_half_width"], 0.0)
        incomplete = [
            row
            for row in rows
            if row["metric"] != "terminal_coverage_ratio"
        ]
        with self.assertRaisesRegex(ValueError, "missing canonical"):
            validate_canonical_aggregate_rows(
                incomplete,
                "method",
                "map_750m",
                suite="environment_size",
            )

    def test_invalid_environment_metric_inputs_fail_explicitly(self):
        base = self._episode(11, 100_000, 10, 100, 0.2, 0.4)
        cases = {
            "missing": {key: value for key, value in base.items() if key != "coverage"},
            "zero horizon": {**base, "episode_horizon_seconds": 0},
            "nonfinite energy": {**base, "total_mobility_energy_j": math.inf},
            "ratio out of range": {**base, "found_GT_ratio": 1.01},
        }
        for name, episode in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                aggregate_paper_point_metrics(
                    "method", "environment_size", self.point, [episode]
                )

    def test_fixed_roi_contract_stays_at_six_rows_and_accepts_legacy_rows(self):
        episode = self._episode(11, 100_000, 10, 100, 0.2, 0.4)
        fixed = aggregate_paper_point_metrics(
            "method",
            "fixed_roi",
            {
                "point_id": "roi_8",
                "x_value": 8,
                "x_unit": "RoIs",
                "fixed_num_gt": 8,
            },
            [episode],
        )
        self.assertEqual(len(fixed), 6)
        self.assertTrue(
            all(
                "suite_aggregate_contract_version" not in row
                and "environment_width_m" not in row
                and "environment_height_m" not in row
                and "display_name" not in row
                for row in fixed
            )
        )
        legacy_environment = [
            {**row, "semantic_suite": "environment_size"} for row in fixed
        ]
        validate_canonical_aggregate_rows(
            legacy_environment, "method", "roi_8", suite="environment_size"
        )

    def test_environment_v1_ten_row_artifact_remains_readable(self):
        rows = [
            row
            for row in self._rows()
            if row["metric"] != "all_rois_discovered_probability"
        ]
        for row in rows:
            row["suite_aggregate_contract_version"] = (
                LEGACY_ENVIRONMENT_SIZE_AGGREGATE_CONTRACT_VERSION
            )
        validate_canonical_aggregate_rows(
            rows,
            "method",
            "map_750m",
            suite="environment_size",
            suite_contract_version=LEGACY_ENVIRONMENT_SIZE_AGGREGATE_CONTRACT_VERSION,
        )


class EnvironmentSizeRuntimeContractTest(unittest.TestCase):
    def test_bitmap_and_observation_shape_follow_map_without_state_dim_change(self):
        for size in (750, 2000):
            with self.subTest(size=size):
                manifest = generate_manifest(
                    "test", 19, 1, num_gt=8, environment_size_m=size
                )
                env = Simulator(
                    num_UAV=16,
                    environment_width_m=size,
                    environment_height_m=size,
                )
                env.reset_environment(manifest.episodes[0])
                self.assertEqual(env.visited_bitmap.shape, (size // 2, size // 2))
                self.assertEqual(env.explorer_id_map.shape, env.visited_bitmap.shape)
                self.assertEqual(
                    tuple((uav.x_u, uav.y_u) for uav in env.UAVs),
                    environment_size_uav_initial_xy_m(size),
                )
                self.assertEqual(tuple(env.GS_pos), (0.0, 0.0, 0.0))
                self.assertLessEqual(
                    math.dist(env.uav_dict[0].get_position(), tuple(env.GS_pos)),
                    400.0,
                )
                compressed = aggregate_coverage_map(env.visited_bitmap)
                self.assertEqual(compressed.shape, (256,))
                self.assertTrue(np.isfinite(compressed).all())
                self.assertEqual(MOVEMENT_STATE_DIM, 531)
                self.assertEqual(ROUTING_STATE_DIM, 143)

    def test_runtime_override_is_evaluation_only_and_normalized_by_current_map(self):
        resolved = _normalize_evaluation_overrides(
            {"environment_width_m": 1750, "environment_height_m": 1750}
        )
        self.assertEqual(resolved["environment_width_m"], 1750)
        self.assertEqual(resolved["environment_height_m"], 1750)
        self.assertEqual(resolved["units"]["environment_dimensions"], "metres")
        with self.assertRaisesRegex(ValueError, "square"):
            _normalize_evaluation_overrides(
                {"environment_width_m": 750, "environment_height_m": 1000}
            )


class EnvironmentSizeSmokeTest(unittest.TestCase):
    def test_horizon_points_share_scenarios_but_have_distinct_config_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evaluation"
            result = run_paper_evaluation(
                "kkm_random_action_random_routing",
                suite="environment_size",
                manifest_seed=47,
                episodes=1,
                episode_horizons_s=(3, 4),
                roi_counts=(8,),
                environment_sizes_m=(1000,),
                output_directory=output,
            )
            self.assertEqual(result["environment_size_horizon_cartesian_point_count"], 2)
            self.assertEqual(result["environment_size_horizon_total_episode_count"], 2)
            self.assertIsNone(result["evaluation_horizon_seconds"])
            self.assertEqual(result["evaluation_episode_horizons_s"], [3, 4])
            first, second = result["points"]
            self.assertEqual(first["scenario_manifest_hash"], second["scenario_manifest_hash"])
            self.assertEqual(first["scenario_ids"], second["scenario_ids"])
            self.assertNotEqual(
                first["evaluation_config_fingerprint"],
                second["evaluation_config_fingerprint"],
            )

    def test_discovery_diagnostics_use_canonical_found_state_and_first_time(self):
        def discover_all(env, **_kwargs):
            for gt in env.gts:
                gt.is_found = True
            return ()

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "HRL_task_aware._mark_search_observations", side_effect=discover_all
        ):
            output = Path(temp_dir) / "evaluation"
            run_paper_evaluation(
                "kkm_random_action_random_routing",
                suite="environment_size",
                manifest_seed=53,
                episodes=1,
                episode_horizons_s=(3,),
                roi_counts=(8,),
                environment_sizes_m=(1000,),
                output_directory=output,
                flatten_single_point=True,
            )
            row = json.loads(
                (output / "per_episode.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(row["discovered_roi_count"], 8)
            self.assertEqual(row["roi_discovery_ratio"], 1.0)
            self.assertTrue(row["all_rois_discovered"])
            self.assertEqual(row["all_rois_discovered_time_s"], 1.0)
            self.assertEqual(row["movement_transition_count"], 3)
            self.assertEqual(row["routing_slot_count"], 12)
            self.assertEqual(set(row["first_discovery_time_by_roi_s"]), {str(i) for i in range(8)})
            self.assertEqual(set(row["first_discovery_time_by_roi_s"].values()), {1.0})

    def test_incomplete_discovery_keeps_null_completion_and_first_time_once(self):
        calls = {"count": 0}

        def discover_incrementally(env, **kwargs):
            calls["count"] += 1
            env.gts[0].is_found = True
            if kwargs["effective_visit_interval_index"] >= 1:
                env.gts[1].is_found = True
            return ()

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "HRL_task_aware._mark_search_observations",
            side_effect=discover_incrementally,
        ):
            output = Path(temp_dir) / "evaluation"
            run_paper_evaluation(
                "kkm_random_action_random_routing",
                suite="environment_size",
                manifest_seed=59,
                episodes=1,
                episode_horizons_s=(3,),
                roi_counts=(8,),
                environment_sizes_m=(1500,),
                output_directory=output,
                flatten_single_point=True,
            )
            row = json.loads(
                (output / "per_episode.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(row["discovered_roi_count"], 2)
            self.assertEqual(row["roi_discovery_ratio"], 0.25)
            self.assertFalse(row["all_rois_discovered"])
            self.assertIsNone(row["all_rois_discovered_time_s"])
            self.assertEqual(
                row["first_discovery_time_by_roi_s"], {"0": 1.0, "1": 2.0}
            )

    def test_mixed_sweep_uses_point_authority_and_suite_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evaluation"
            result = run_paper_evaluation(
                "kkm_random_action_random_routing",
                suite="environment_size",
                manifest_seed=43,
                episodes=1,
                episode_horizons_s=(3,),
                roi_counts=(8,),
                environment_sizes_m=(750, 1000, 2000),
                output_directory=output,
            )
            self.assertEqual(
                result["evaluation_purpose"],
                "environment_size_sweep_evaluation",
            )
            self.assertTrue(result["contains_zero_shot_environment_shift"])
            points = {point["x_value"]: point for point in result["points"]}
            self.assertFalse(points[1000]["zero_shot_environment_shift"])
            self.assertEqual(points[1000]["environment_shift_type"], "episode_horizon")
            self.assertTrue(points[1000]["zero_shot_horizon_shift"])
            self.assertTrue(points[750]["zero_shot_environment_shift"])
            self.assertTrue(points[2000]["zero_shot_environment_shift"])
            self.assertEqual(
                points[750]["environment_shift_types"],
                ["map_size", "episode_horizon"],
            )

    def test_one_episode_random_smoke_at_1500m_and_2000m(self):
        for size in (1500, 2000):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "evaluation"
                result = run_paper_evaluation(
                    "kkm_random_action_random_routing",
                    suite="environment_size",
                    manifest_seed=41,
                    episodes=1,
                    episode_horizons_s=(3,),
                    roi_counts=(8,),
                    environment_sizes_m=(size,),
                    output_directory=output,
                    flatten_single_point=True,
                )
                point = result["points"][0]
                self.assertEqual(point["evaluation_environment_width_m"], size)
                self.assertEqual(point["evaluation_environment_height_m"], size)
                self.assertEqual(point["training_environment_width_m"], 1000)
                self.assertEqual(point["new_training_started"], False)
                self.assertFalse(point["environment_size_observed_by_policy"])
                self.assertEqual(
                    point["initial_uav_deployment_strategy"],
                    "scaled_canonical_connectivity_preserving",
                )
                self.assertEqual(
                    point["initial_uav_xy_scale"], min(size / 1000.0, 1.9)
                )
                self.assertTrue(point["initial_topology_connected"])
                self.assertEqual(
                    point["initial_uav_deployment_source"], "evaluation_map_size"
                )
                self.assertEqual(
                    point["coordinate_normalization"],
                    "current_environment_width_height",
                )
                self.assertEqual(
                    point["zero_shot_environment_shift"], size != 1000
                )
                self.assertEqual(
                    point["environment_shift_type"],
                    "map_size" if size != 1000 else "episode_horizon",
                )
                self.assertEqual(
                    point["point_evaluation_purpose"],
                    "zero_shot_environment_shift_evaluation",
                )
                metadata = json.loads(
                    (output / "paper_evaluation_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metadata["evaluation_environment_sizes_m"], [size])
                self.assertEqual(
                    metadata["evaluation_purpose"],
                    "environment_size_sweep_evaluation",
                )
                self.assertEqual(
                    metadata["contains_zero_shot_environment_shift"],
                    size != 1000,
                )
                self.assertTrue(metadata["contains_zero_shot_horizon_shift"])
                self.assertEqual(metadata["points"][0]["roi_count"], 8)
                self.assertEqual(metadata["points"][0]["environment_size_m"], size)
                self.assertEqual(
                    metadata["environment_size_uav_deployments"][0][
                        "initial_uav_deployment_strategy"
                    ],
                    "scaled_canonical_connectivity_preserving",
                )
                self.assertEqual(metadata["points"][0]["training_episode_horizon_s"], 60)
                self.assertEqual(metadata["points"][0]["evaluation_episode_horizon_s"], 3)
                self.assertEqual(
                    metadata["points"][0]["remaining_time_normalization"],
                    "current_evaluation_episode_horizon",
                )
                self.assertFalse(
                    metadata["points"][0]["episode_horizon_seconds_observed_by_policy"]
                )
                persisted_episode = json.loads(
                    (output / "per_episode.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                self.assertEqual(persisted_episode["episode_horizon_seconds"], 3.0)
                self.assertEqual(persisted_episode["movement_transition_count"], 3)
                self.assertEqual(persisted_episode["routing_slot_count"], 12)
                self.assertEqual(persisted_episode["packet_injection_cutoff_seconds"], 0.5)
                aggregate_json = json.loads(
                    (output / "aggregated_plot_data.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(len(aggregate_json), 11)
                expected_metrics = {
                    "timely_throughput_kbit_per_s",
                    "mobility_energy_j_per_episode",
                    "roi_discovery_ratio",
                    "terminal_coverage_ratio",
                    "all_rois_discovered_probability",
                }
                self.assertTrue(
                    expected_metrics.issubset(
                        {row["metric"] for row in aggregate_json}
                    )
                )
                aggregate_csv = (output / "aggregated_plot_data.csv").read_text(
                    encoding="utf-8"
                )
                for metric in expected_metrics:
                    self.assertIn(metric, aggregate_csv)


if __name__ == "__main__":
    unittest.main()
