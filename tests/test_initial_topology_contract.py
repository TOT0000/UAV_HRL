import copy
import math
import unittest
from unittest import mock

from experiment_config import (
    CANONICAL_UAV_INITIAL_XY_M,
    INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
    METHOD_REGISTRY,
    MethodSpec,
    UAV_INITIAL_LAYOUT_VERSION,
    effective_training_config,
)
from HRL_task_aware import TrainingConfig, formal_training_config, train
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from rng_contract import NamedRNGStreams
from scenario_manifest import (
    UAV_INITIAL_LAYOUT,
    environment_config_fingerprint,
    generate_manifest,
    validate_initial_communication_topology,
    validate_manifest_initial_topologies,
    validate_scenario_entry,
)
from training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    _validate_checkpoint_schema,
)


def _topology_uav(uav_id, position):
    return {
        "uav_id": int(uav_id),
        "position": list(position),
        "energy_j": 10000.0,
    }


class InitialTopologyGeometryTest(unittest.TestCase):
    def test_gateway_is_in_range_for_full_altitude_interval_and_boundary_is_inclusive(
        self,
    ):
        for altitude in (80.0, 100.0, 120.0):
            uavs = [
                _topology_uav(
                    uav_id,
                    (x, y, altitude),
                )
                for uav_id, (x, y) in enumerate(CANONICAL_UAV_INITIAL_XY_M)
            ]
            topology = validate_initial_communication_topology(
                uavs, scenario_id=f"gateway-z-{altitude}"
            )
            self.assertIn(0, topology["u2g_uav_ids"])
            self.assertLess(topology["nearest_u2g_3d_distance_m"], 400.0)

        boundary = validate_initial_communication_topology(
            [
                _topology_uav(0, (0.0, 0.0, 400.0)),
                _topology_uav(1, (400.0, 0.0, 400.0)),
            ],
            scenario_id="inclusive-boundary",
        )
        self.assertEqual(boundary["u2g_uav_ids"], [0])
        self.assertEqual(boundary["gs_component_uav_ids"], [0, 1])
        self.assertEqual(boundary["nearest_u2g_3d_distance_m"], 400.0)

        with self.assertRaisesRegex(ValueError, "no UAV.*U2G range edge"):
            validate_initial_communication_topology(
                [
                    _topology_uav(0, (0.0, 0.0, 400.0001)),
                    _topology_uav(1, (400.0, 0.0, 400.0001)),
                ],
                scenario_id="outside-boundary",
            )

    def test_generated_manifests_pass_for_all_splits_roi_counts_and_seeds(self):
        for split in ("train", "validation", "test"):
            for num_gt in range(2, 9):
                for seed in (17, 20260817, 99173):
                    with self.subTest(split=split, num_gt=num_gt, seed=seed):
                        manifest = generate_manifest(
                            split, seed, 2, num_gt=num_gt
                        )
                        topologies = validate_manifest_initial_topologies(manifest)
                        self.assertEqual(
                            manifest.config_fingerprint,
                            environment_config_fingerprint(),
                        )
                        self.assertEqual(
                            manifest.generator_config[
                                "initial_communication_topology_contract_version"
                            ],
                            INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
                        )
                        self.assertEqual(len(topologies), 2)
                        self.assertTrue(
                            all(item["u2g_uav_ids"] for item in topologies)
                        )
                        self.assertTrue(
                            all(
                                len(item["gs_component_uav_ids"]) >= 2
                                for item in topologies
                            )
                        )

    def test_invalid_topologies_and_layout_metadata_fail_fast(self):
        with self.assertRaisesRegex(
            ValueError,
            r"scenario_id=all-far;.*nearest_u2g_3d_distance_m=.*"
            r"u2g_range_m=400;.*gs_component_uav_ids=\[\]",
        ):
            validate_initial_communication_topology(
                [
                    _topology_uav(0, (500.0, 500.0, 100.0)),
                    _topology_uav(1, (800.0, 800.0, 100.0)),
                ],
                scenario_id="all-far",
            )

        with self.assertRaisesRegex(
            ValueError,
            r"scenario_id=isolated;.*u2g_range_m=400;.*"
            r"gs_component_uav_ids=\[0\]",
        ):
            validate_initial_communication_topology(
                [
                    _topology_uav(0, (0.0, 0.0, 100.0)),
                    _topology_uav(1, (1000.0, 1000.0, 100.0)),
                ],
                scenario_id="isolated",
            )

        with self.assertRaisesRegex(
            ValueError,
            r"scenario_id=non-finite;.*nearest_u2g_3d_distance_m=non-finite;"
            r".*u2g_range_m=400;.*gs_component_uav_ids=\[\]",
        ):
            validate_initial_communication_topology(
                [
                    _topology_uav(0, (math.nan, 0.0, 100.0)),
                    _topology_uav(1, (100.0, 0.0, 100.0)),
                ],
                scenario_id="non-finite",
            )

        entry = copy.deepcopy(generate_manifest("test", 41, 1).episodes[0])
        entry["exogenous_primitives"]["uav_xy_layout"] = "stale-layout"
        with self.assertRaisesRegex(
            ValueError,
            rf"declared=stale-layout; expected={UAV_INITIAL_LAYOUT}",
        ):
            validate_scenario_entry(entry)

        entry = copy.deepcopy(generate_manifest("test", 42, 1).episodes[0])
        entry["uavs"][0]["position"][0] = 60.0
        with self.assertRaisesRegex(ValueError, "coordinates disagree"):
            validate_scenario_entry(entry)

    def test_simulator_fallback_uses_the_canonical_layout(self):
        env = Simulator(10, rng_streams=NamedRNGStreams(501))
        env.num_GT = 2
        env.reset_environment()

        self.assertEqual(
            [(uav.x_u, uav.y_u) for uav in env.UAVs],
            list(CANONICAL_UAV_INITIAL_XY_M),
        )
        self.assertTrue(all(80.0 <= uav.z_u <= 120.0 for uav in env.UAVs))
        topology = validate_initial_communication_topology(
            [
                _topology_uav(uav.id, uav.get_position())
                for uav in env.UAVs
            ],
            scenario_id="fallback",
            gs_position=env.GS_pos,
        )
        self.assertIn(0, topology["u2g_uav_ids"])
        self.assertGreaterEqual(len(topology["gs_component_uav_ids"]), 2)

    def test_all_methods_receive_identical_manifest_geometry_and_topology(self):
        entry = generate_manifest("train", 20260817, 1, num_gt=4).episodes[0]
        expected_geometry = None
        expected_topology = None
        self.assertEqual(len(METHOD_REGISTRY), 16)
        for method_id in METHOD_REGISTRY:
            with self.subTest(method=method_id):
                env = Simulator(10, rng_streams=NamedRNGStreams(88))
                env.configure_method(MethodSpec.parse(method_id))
                env.apply_scenario_entry(entry)
                geometry = [uav.get_position() for uav in env.UAVs]
                topology = validate_initial_communication_topology(
                    entry["uavs"], scenario_id=entry["scenario_id"]
                )
                if expected_geometry is None:
                    expected_geometry = geometry
                    expected_topology = topology
                self.assertEqual(geometry, expected_geometry)
                self.assertEqual(topology, expected_topology)
                formal = effective_training_config(
                    formal_training_config(1, enable_model_checkpoints=False),
                    MethodSpec.parse(method_id),
                )
                self.assertEqual(
                    formal["uav_initial_layout_version"],
                    UAV_INITIAL_LAYOUT_VERSION,
                )
                self.assertEqual(
                    formal["initial_communication_topology_contract_version"],
                    INITIAL_COMMUNICATION_TOPOLOGY_CONTRACT_VERSION,
                )

    def test_schema_15_checkpoint_is_rejected_for_old_initial_geometry(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 22)
        with self.assertRaisesRegex(
            RuntimeError, "GS-reachable initial topology.*must be retrained"
        ):
            _validate_checkpoint_schema({"checkpoint_schema_version": 15})

    def test_invalid_manifest_is_rejected_before_simulator_or_weight_load(self):
        manifest = generate_manifest("validation", 75, 1, num_gt=2)
        for item in manifest.episodes[0]["uavs"]:
            item["position"] = [800.0 + item["uav_id"], 800.0, 100.0]
        config = TrainingConfig(
            total_episodes=1,
            mode="custom",
            episode_seconds=1,
            warmup_joint_transitions=0,
            batch_size=1,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            random_seed=75,
        )
        with (
            mock.patch("HRL_task_aware.Simulator") as simulator,
            mock.patch("HRL_task_aware.load_model_checkpoint") as load_weights,
            self.assertRaisesRegex(ValueError, "permanent GS gateway"),
        ):
            train(
                config,
                scenario_manifest=manifest,
                method_spec=MethodSpec.parse("td3_dinkelbach"),
                evaluation=True,
                checkpoint_dir="never-read",
            )
        simulator.assert_not_called()
        load_weights.assert_not_called()


class InitialTopologyDeliveryTest(unittest.TestCase):
    def test_non_gateway_packet_reaches_gs_through_production_service(self):
        env = Simulator(10, rng_streams=NamedRNGStreams(123))
        env.apply_scenario_entry(
            generate_manifest("test", 123, 1, num_gt=2).episodes[0]
        )
        self.assertTrue(env.is_routing_link_in_range(1, 0))
        self.assertTrue(env.is_u2g_in_range(0))

        engine = PacketEngine(10)
        packet = engine.create_packet(1, "COM", 256.0, 0.0)
        env.prepare_channel_routing_slot(0)
        capacities, bandwidths = env.allocate_active_link_capacities({1: 0})
        self.assertGreater(capacities[(1, 0)], 0.0)
        self.assertLessEqual(sum(bandwidths.values()), env.B_tot)
        self.assertEqual(env.active_link_capacity_profiles_mbps[(1, 0)].shape, (50,))
        engine.serve_active_links(
            env,
            {1: 0},
            capacities,
            current_time=0.0,
            block_capacity_profiles=env.active_link_capacity_profiles_mbps,
        )
        self.assertEqual(packet["current"], 0)

        env.prepare_channel_routing_slot(1)
        capacities, bandwidths = env.allocate_active_link_capacities(
            {0: env.GS_ID}
        )
        self.assertGreater(capacities[(0, env.GS_ID)], 0.0)
        self.assertLessEqual(sum(bandwidths.values()), env.B_tot)
        self.assertEqual(
            env.active_link_capacity_profiles_mbps[(0, env.GS_ID)].shape,
            (50,),
        )
        result = engine.serve_active_links(
            env,
            {0: env.GS_ID},
            capacities,
            current_time=0.25,
            block_capacity_profiles=env.active_link_capacity_profiles_mbps,
        )
        self.assertEqual(packet["current"], env.GS_ID)
        self.assertGreater(result["raw_final_hop_bits"], 0.0)
        self.assertGreater(result["timely_goodput_bits"], 0.0)
        self.assertGreater(engine.raw_final_hop_bits, 0.0)
        self.assertGreater(engine.timely_goodput_bits, 0.0)

    def test_seed_20260817_first_20_warmup_episodes_deliver_bits(self):
        seed = 20260817
        config = formal_training_config(
            20,
            random_seed=seed,
            enable_model_checkpoints=False,
            enable_full_resume=False,
            enable_plots=False,
            enable_csv=False,
            run_directory=None,
        )
        result = train(
            config,
            scenario_manifest=generate_manifest("train", seed, 20),
            method_spec=MethodSpec.parse("td3_dinkelbach"),
        )
        raw_final_hop_bits = sum(
            float(row["raw_final_hop_mbits"]) * 1e6
            for row in result["episode_metrics"]
        )
        timely_goodput_bits = sum(
            float(row["timely_goodput_mbits"]) * 1e6
            for row in result["episode_metrics"]
        )
        self.assertEqual(len(result["episode_metrics"]), 20)
        self.assertEqual(config.warmup_joint_transitions, 10_000)
        self.assertGreater(raw_final_hop_bits, 0.0)
        self.assertGreater(timely_goodput_bits, 0.0)


if __name__ == "__main__":
    unittest.main()
