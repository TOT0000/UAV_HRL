import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from centralized_movement import MOVEMENT_STATE_DIM, aggregate_coverage_map
from evaluation_selection import (
    DEFAULT_ENVIRONMENT_SIZES_M,
    resolve_environment_sizes_m,
)
from experiment_config import CANONICAL_UAV_INITIAL_XY_M, METHOD_REGISTRY
from HRL_task_aware import _normalize_evaluation_overrides
from observation_strategy import ROUTING_STATE_DIM
from paper_evaluation import (
    PAPER_EVALUATION_SUITES,
    evaluation_sweep_points,
    run_paper_evaluation,
)
from run_paper_evaluation import main as run_evaluation_main
from scenario_manifest import (
    ENVIRONMENT_SIZE_SCENARIO_SCHEMA_VERSION,
    SCENARIO_SCHEMA_VERSION,
    ScenarioManifest,
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
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation_sweep_points("environment_size", roi_counts=(4, 8))

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
                    CANONICAL_UAV_INITIAL_XY_M,
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
    def test_one_episode_random_smoke_at_smallest_and_largest_maps(self):
        for size in (750, 1000, 2000):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "evaluation"
                result = run_paper_evaluation(
                    "kkm_random_action_random_routing",
                    suite="environment_size",
                    manifest_seed=41,
                    episodes=1,
                    episode_seconds=1,
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
                    point["coordinate_normalization"],
                    "current_environment_width_height",
                )
                self.assertEqual(
                    point["zero_shot_environment_shift"], size != 1000
                )
                self.assertEqual(point["environment_shift_type"], "map_size")
                metadata = json.loads(
                    (output / "paper_evaluation_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metadata["evaluation_environment_sizes_m"], [size])
                self.assertEqual(metadata["points"][0]["roi_count"], 8)


if __name__ == "__main__":
    unittest.main()
