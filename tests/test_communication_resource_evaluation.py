import math
import unittest
from unittest import mock

import numpy as np

from Channel_model import (
    S2U_TX_POWER_DBM,
    U2U_U2G_TX_POWER_DBM,
    transmit_power_mw,
)
from HRL_task_aware import _normalize_evaluation_overrides
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from experiment_config import TOTAL_COMMUNICATION_BANDWIDTH_HZ
from paper_evaluation import evaluation_sweep_points
from run_paper_evaluation import main as run_evaluation_main
from scenario_manifest import generate_manifest


RESOURCE_CASES = (
    ("bandwidth_sweep", "bandwidth_mhz", (2.5, 10.125)),
    ("sr_transmit_power_sweep", "sr_tx_power_dbm", (-3.25, 23.5)),
    ("uav_transmit_power_sweep", "uav_tx_power_dbm", (-1.75, 30.25)),
)


class CommunicationResourcePointTest(unittest.TestCase):
    def test_all_suites_accept_single_multiple_and_fractional_values(self):
        for suite, argument, values in RESOURCE_CASES:
            with self.subTest(suite=suite):
                points = evaluation_sweep_points(
                    suite,
                    **{argument: values},
                )
                self.assertEqual([point["x_value"] for point in points], list(values))
                self.assertEqual(len({point["point_id"] for point in points}), 2)
                self.assertEqual({point["fixed_num_gt"] for point in points}, {8})
                for point in points:
                    self.assertEqual(
                        point["total_communication_bandwidth_hz"],
                        point["total_communication_bandwidth_mhz"] * 1e6,
                    )
                    self.assertTrue(math.isfinite(point["sr_tx_power_dbm"]))
                    self.assertTrue(math.isfinite(point["uav_tx_power_dbm"]))

    def test_non_scan_values_are_single_fixed_values_and_defaults_are_exact(self):
        points = evaluation_sweep_points(
            "bandwidth_sweep",
            bandwidth_mhz=(4.5, 9.0),
            sr_tx_power_dbm=(-2.5,),
            uav_tx_power_dbm=(17.25,),
            roi_counts=(6,),
        )
        self.assertEqual({point["sr_tx_power_dbm"] for point in points}, {-2.5})
        self.assertEqual({point["uav_tx_power_dbm"] for point in points}, {17.25})
        self.assertEqual({point["fixed_num_gt"] for point in points}, {6})

        defaults = evaluation_sweep_points(
            "bandwidth_sweep", bandwidth_mhz=(10.0,)
        )[0]
        self.assertEqual(
            defaults["total_communication_bandwidth_hz"],
            TOTAL_COMMUNICATION_BANDWIDTH_HZ,
        )
        self.assertEqual(defaults["sr_tx_power_dbm"], S2U_TX_POWER_DBM)
        self.assertEqual(defaults["uav_tx_power_dbm"], U2U_U2G_TX_POWER_DBM)

        with self.assertRaisesRegex(ValueError, "exactly one fixed value"):
            evaluation_sweep_points(
                "bandwidth_sweep",
                bandwidth_mhz=(10.0,),
                sr_tx_power_dbm=(20.0, 23.0),
            )

    def test_invalid_values_fail_before_evaluation(self):
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            evaluation_sweep_points("bandwidth_sweep")
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bandwidth=invalid), self.assertRaises(ValueError):
                evaluation_sweep_points(
                    "bandwidth_sweep", bandwidth_mhz=(invalid,)
                )
        for invalid in (float("nan"), float("inf"), float("-inf"), -4000.0):
            with self.subTest(power=invalid), self.assertRaises(ValueError):
                evaluation_sweep_points(
                    "sr_transmit_power_sweep",
                    sr_tx_power_dbm=(invalid,),
                )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            evaluation_sweep_points(
                "uav_transmit_power_sweep",
                uav_tx_power_dbm=(1.25, 1.25),
            )
        self.assertGreater(transmit_power_mw(-25.5), 0.0)

    def test_point_ids_do_not_collide_when_display_prefix_matches(self):
        points = evaluation_sweep_points(
            "bandwidth_sweep",
            bandwidth_mhz=(1.0000000000001, 1.0000000000002),
        )
        self.assertEqual(len({point["point_id"] for point in points}), 2)

    def test_cli_forwards_resource_values_and_preserves_old_suite_scope(self):
        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation", return_value={}
        ) as runner:
            self.assertEqual(
                run_evaluation_main(
                    [
                        "kkm_random_action_random_routing",
                        "--suite",
                        "sr_transmit_power_sweep",
                        "--sr-tx-power-dbm",
                        "-2.5",
                        "23.25",
                        "--bandwidth-mhz",
                        "7.5",
                        "--uav-tx-power-dbm",
                        "28.5",
                        "--roi-count",
                        "5",
                    ]
                ),
                0,
            )
        self.assertEqual(runner.call_args.kwargs["sr_tx_power_dbm"], [-2.5, 23.25])
        self.assertEqual(runner.call_args.kwargs["bandwidth_mhz"], [7.5])
        self.assertEqual(runner.call_args.kwargs["uav_tx_power_dbm"], [28.5])
        self.assertEqual(runner.call_args.kwargs["roi_counts"], (5,))
        with self.assertRaisesRegex(ValueError, "require a communication-resource"):
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite",
                    "fixed_roi",
                    "--bandwidth-mhz",
                    "10",
                ]
            )

    def test_points_share_the_same_manifest_and_exogenous_conditions(self):
        points = evaluation_sweep_points(
            "uav_transmit_power_sweep",
            uav_tx_power_dbm=(10.0, 20.0, 30.0),
            roi_counts=(4,),
        )
        manifests = [
            generate_manifest("test", 12345, 2, num_gt=point["fixed_num_gt"])
            for point in points
        ]
        self.assertEqual(len({manifest.content_hash for manifest in manifests}), 1)
        self.assertEqual(
            len(
                {
                    tuple(
                        episode["scenario_id"] for episode in manifest.episodes
                    )
                    for manifest in manifests
                }
            ),
            1,
        )


class CommunicationResourceRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = generate_manifest("test", 3819, 1, num_gt=2).episodes[0]

    def _environment(self, *, bandwidth=10e6, sr_power=23.0, uav_power=30.0):
        env = Simulator(
            num_UAV=16,
            total_communication_bandwidth_hz=bandwidth,
            s2u_tx_power_dbm=sr_power,
            uav_tx_power_dbm=uav_power,
        )
        env.apply_scenario_entry(self.scenario)
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
            100.0,
            100.0,
            100.0,
        )
        env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (
            150.0,
            100.0,
            100.0,
        )
        env.SR_teams[0].x, env.SR_teams[0].y, env.SR_teams[0].z = (
            100.0,
            100.0,
            0.0,
        )
        env.multi_tasks = {uav_id: [] for uav_id in range(env.num_UAV)}
        env.multi_tasks[0] = [
            {
                "task_type": "COM",
                "target_obj_id": 0,
                "target_pos": (100.0, 100.0, 0.0),
            }
        ]
        env.SR_teams[0].assigned_gt_id = 0
        env.update_u2u_channels()
        env.update_u2g_channels()
        env.prepare_channel_routing_slot(0)
        return env

    def test_bandwidth_and_uav_power_change_actual_u2u_service(self):
        configurations = (
            self._environment(bandwidth=2e6, uav_power=5.0),
            self._environment(bandwidth=20e6, uav_power=35.0),
        )
        transmitted = []
        for env in configurations:
            engine = PacketEngine(num_uav=16)
            engine.create_packet(0, "COM", 1e9, generation_time=0.0)
            capacities, bandwidths = env.allocate_active_link_capacities({0: 1})
            result = engine.serve_active_links(
                env,
                {0: 1},
                capacities,
                current_time=0.0,
                block_capacity_profiles=env.active_link_capacity_profiles_mbps,
            )
            self.assertEqual(bandwidths[(0, 1)], env.B_tot)
            transmitted.append(result["transmitted_bits_by_link"][(0, 1)])
        self.assertGreater(transmitted[1], transmitted[0])

    def test_sr_power_changes_actual_s2u_service(self):
        transmitted = []
        for power in (-10.5, 30.25):
            env = self._environment(sr_power=power)
            engine = PacketEngine(num_uav=16)
            engine.create_sr_packet(0, 1e9, generation_time=0.0)
            env.allocate_active_link_capacities({}, s2u_links={0: 0})
            result = engine.serve_active_links(
                env,
                {},
                {},
                current_time=0.0,
                s2u_block_capacity_profiles=(
                    env.active_s2u_capacity_profiles_mbps
                ),
            )
            transmitted.append(
                result["transmitted_bits_by_link"][("S2U", 0, 0)]
            )
        self.assertGreater(transmitted[1], transmitted[0])

    def test_runtime_points_are_isolated_and_explicit_defaults_are_identical(self):
        default_env = self._environment()
        explicit_default_env = self._environment(
            bandwidth=TOTAL_COMMUNICATION_BANDWIDTH_HZ,
            sr_power=S2U_TX_POWER_DBM,
            uav_power=U2U_U2G_TX_POWER_DBM,
        )
        default_env.allocate_active_link_capacities({0: 1})
        explicit_default_env.allocate_active_link_capacities({0: 1})
        np.testing.assert_array_equal(
            default_env.active_link_capacity_profiles_mbps[(0, 1)],
            explicit_default_env.active_link_capacity_profiles_mbps[(0, 1)],
        )

        first = self._environment(bandwidth=3e6, sr_power=-2.0, uav_power=9.0)
        second = self._environment(bandwidth=17e6, sr_power=26.0, uav_power=33.0)
        self.assertEqual(first.B_tot, 3e6)
        self.assertEqual(first.s2u_tx_power_dbm, -2.0)
        self.assertEqual(first.uav_tx_power_dbm, 9.0)
        self.assertEqual(second.B_tot, 17e6)
        self.assertEqual(
            first.routing_capacity_reference_mbps(0, 1),
            second.routing_capacity_reference_mbps(0, 1),
        )
        self.assertNotEqual(
            first.u2u_nominal_capacity[0, 1],
            second.u2u_nominal_capacity[0, 1],
        )

    def test_normalized_overrides_keep_checkpoint_calibration_separate(self):
        resolved = _normalize_evaluation_overrides(
            {
                "total_communication_bandwidth_hz": 3.75e6,
                "s2u_tx_power_dbm": -4.5,
                "uav_tx_power_dbm": 18.25,
            }
        )
        self.assertEqual(resolved["total_communication_bandwidth_hz"], 3.75e6)
        self.assertEqual(resolved["s2u_tx_power_dbm"], -4.5)
        self.assertEqual(resolved["uav_tx_power_dbm"], 18.25)


if __name__ == "__main__":
    unittest.main()
