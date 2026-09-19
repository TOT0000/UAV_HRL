import copy
import unittest

from experiment_config import (
    CANONICAL_ENVIRONMENT_SIZE_M,
    CANONICAL_UAV_INITIAL_XY_M,
    ENVIRONMENT_SIZE_UAV_DEPLOYMENT_STRATEGY,
)
from scenario_manifest import (
    ENVIRONMENT_SIZE_SCENARIO_SCHEMA_VERSION,
    ScenarioManifest,
    current_environment_config,
    environment_config_fingerprint,
    environment_size_uav_initial_xy_m,
    environment_size_uav_xy_scale,
    generate_manifest,
    manifest_prefix,
    validate_initial_communication_topology,
    validate_scenario_entry,
)
from Simulator import Simulator


SIZES = (750, 1000, 1250, 1500, 1750, 2000)
EXPECTED_GRID = {
    750: (75.0, 225.0, 375.0, 525.0),
    1000: (100.0, 300.0, 500.0, 700.0),
    1250: (125.0, 375.0, 625.0, 875.0),
    1500: (150.0, 450.0, 750.0, 1050.0),
    1750: (175.0, 525.0, 875.0, 1225.0),
    2000: (190.0, 570.0, 950.0, 1330.0),
}


class EnvironmentSizeDeploymentTest(unittest.TestCase):
    def test_scaled_grid_is_deterministic_in_bounds_and_preserves_altitudes(self):
        manifests = {
            size: generate_manifest(
                "test", 20260817, 1, num_gt=8, environment_size_m=size
            )
            for size in SIZES
        }
        reference_z = tuple(
            item["position"][2] for item in manifests[1000].episodes[0]["uavs"]
        )
        for size, manifest in manifests.items():
            with self.subTest(size=size):
                expected_scale = min(size / 1000.0, 1.9)
                self.assertEqual(environment_size_uav_xy_scale(size), expected_scale)
                expected_xy = environment_size_uav_initial_xy_m(size)
                self.assertEqual(
                    tuple(sorted({x for x, _ in expected_xy})), EXPECTED_GRID[size]
                )
                self.assertEqual(
                    tuple(sorted({y for _, y in expected_xy})), EXPECTED_GRID[size]
                )
                entry = manifest.episodes[0]
                actual_xy = tuple(
                    tuple(map(float, item["position"][:2])) for item in entry["uavs"]
                )
                actual_z = tuple(item["position"][2] for item in entry["uavs"])
                self.assertEqual(actual_xy, expected_xy)
                self.assertEqual(actual_z, reference_z)
                self.assertTrue(
                    all(0.0 <= x <= size and 0.0 <= y <= size for x, y in actual_xy)
                )
                self.assertEqual(
                    manifest.to_dict(),
                    generate_manifest(
                        "test", 20260817, 1, num_gt=8, environment_size_m=size
                    ).to_dict(),
                )
        self.assertEqual(
            environment_size_uav_initial_xy_m(1000),
            CANONICAL_UAV_INITIAL_XY_M,
        )

    def test_every_supported_map_is_fully_connected_in_three_dimensions(self):
        for size in SIZES:
            with self.subTest(size=size):
                entry = generate_manifest(
                    "test", 91, 1, num_gt=8, environment_size_m=size
                ).episodes[0]
                topology = validate_initial_communication_topology(
                    entry["uavs"],
                    scenario_id=entry["scenario_id"],
                    environment_size_m=size,
                )
                self.assertTrue(topology["initial_topology_connected"])
                self.assertEqual(topology["gs_component_uav_ids"], list(range(16)))
                self.assertEqual(topology["unreachable_uav_ids"], [])
                self.assertLessEqual(topology["gateway_gs_3d_distance_m"], 400.0)

    def test_disconnected_fixture_reports_map_gateway_and_pair_distances(self):
        uavs = [
            {"uav_id": 0, "position": [100.0, 100.0, 100.0], "energy_j": 10000.0},
            {"uav_id": 1, "position": [700.0, 700.0, 100.0], "energy_j": 10000.0},
            {"uav_id": 2, "position": [900.0, 900.0, 100.0], "energy_j": 10000.0},
        ]
        with self.assertRaises(ValueError) as raised:
            validate_initial_communication_topology(
                uavs,
                scenario_id="disconnected-map",
                environment_size_m=2000,
            )
        message = str(raised.exception)
        self.assertIn("environment_size_m=2000", message)
        self.assertIn("unreachable_uav_ids=[1, 2]", message)
        self.assertIn("gateway_gs_3d_distance_m=", message)
        self.assertIn("relevant_uav_pair_3d_distances_m=", message)

    def test_manifest_identity_fingerprint_and_metadata_bind_deployment(self):
        manifest = generate_manifest(
            "test", 73, 1, num_gt=8, environment_size_m=1500
        )
        self.assertEqual(
            manifest.schema_version, ENVIRONMENT_SIZE_SCENARIO_SCHEMA_VERSION
        )
        entry = manifest.episodes[0]
        metadata = entry["exogenous_primitives"]
        self.assertEqual(
            metadata["initial_uav_deployment_strategy"],
            ENVIRONMENT_SIZE_UAV_DEPLOYMENT_STRATEGY,
        )
        self.assertEqual(
            metadata["canonical_environment_size_m"],
            CANONICAL_ENVIRONMENT_SIZE_M,
        )
        self.assertEqual(metadata["initial_uav_xy_scale"], 1.5)
        self.assertEqual(
            metadata["initial_uav_positions_xyz_m"],
            [item["position"] for item in entry["uavs"]],
        )
        self.assertTrue(metadata["initial_topology_connected"])
        self.assertIn("deployment-", entry["scenario_id"])
        self.assertNotEqual(
            entry["scenario_id"],
            "test:uav-hrl-scenario-v9:fixed-8:map-1500m:73:000000",
        )
        positions = manifest.generator_config["initial_uav_positions_xyz_m"]
        expected_fingerprint = environment_config_fingerprint(
            current_environment_config(
                1500,
                1500,
                environment_size_m=1500,
                initial_uav_positions_xyz_m=positions,
            )
        )
        self.assertEqual(manifest.config_fingerprint, expected_fingerprint)
        changed_positions = copy.deepcopy(positions)
        changed_positions[0][0][0] += 1.0
        self.assertNotEqual(
            expected_fingerprint,
            environment_config_fingerprint(
                current_environment_config(
                    1500,
                    1500,
                    environment_size_m=1500,
                    initial_uav_positions_xyz_m=changed_positions,
                )
            ),
        )

        stale = copy.deepcopy(entry)
        stale["exogenous_primitives"]["initial_uav_deployment_strategy"] = (
            "fixed_absolute_coordinates"
        )
        with self.assertRaisesRegex(ValueError, "deployment strategy"):
            validate_scenario_entry(stale)

        legacy = copy.deepcopy(manifest.to_dict())
        legacy["schema_version"] = "uav-hrl-scenario-v9"
        with self.assertRaisesRegex(ValueError, "fixed-coordinate deployment"):
            ScenarioManifest.from_dict(legacy)

    def test_simulator_rejects_applied_position_drift(self):
        manifest = generate_manifest(
            "test", 81, 1, num_gt=8, environment_size_m=1500
        )
        entry = manifest.episodes[0]
        env = Simulator(16, environment_width_m=1500, environment_height_m=1500)
        env.apply_scenario_entry(entry)
        env.uav_dict[4].x_u += 1.0
        with self.assertRaisesRegex(RuntimeError, "applied UAV initial state mismatch"):
            env.validate_applied_scenario(entry)

    def test_fixed_roi_manifest_remains_byte_compatible(self):
        manifest = generate_manifest("test", 20260817, 1, num_gt=8)
        self.assertEqual(
            manifest.content_hash,
            "afcb7256a161896d6ecc48e83948410fe9cadbb83610fabccd7a6790a634efed",
        )
        self.assertEqual(
            tuple(tuple(item["position"][:2]) for item in manifest.episodes[0]["uavs"]),
            CANONICAL_UAV_INITIAL_XY_M,
        )
        self.assertNotIn(
            "initial_uav_deployment_strategy", manifest.generator_config
        )

    def test_environment_size_manifest_prefix_rebinds_position_fingerprint(self):
        manifest = generate_manifest(
            "test", 95, 2, num_gt=8, environment_size_m=1750
        )
        prefix = manifest_prefix(manifest, 1)
        self.assertEqual(prefix.episodes, manifest.episodes[:1])
        self.assertEqual(
            prefix.generator_config["initial_uav_positions_xyz_m"],
            [
                manifest.episodes[0]["exogenous_primitives"][
                    "initial_uav_positions_xyz_m"
                ]
            ],
        )
        self.assertNotEqual(prefix.config_fingerprint, manifest.config_fingerprint)


if __name__ == "__main__":
    unittest.main()
