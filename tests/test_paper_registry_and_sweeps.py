import unittest

from experiment_config import METHOD_REGISTRY, MethodSpec, movement_agent_configuration
from paper_evaluation import (
    ARRIVAL_RATE_SWEEPS,
    DEADLINE_SWEEP_SECONDS,
    FIXED_ROI_VALUES,
    TRAJECTORY_SNAPSHOT_SECONDS,
    evaluation_sweep_points,
    validate_production_deadlines,
)
from paper_figure_registry import (
    FIGURE_REGISTRY,
    LegacyFigureSourceUnavailable,
    PAPER_METHOD_MAPPINGS,
    require_legacy_figure_contract,
)


class PaperMethodRegistryTest(unittest.TestCase):
    def test_registry_contains_the_exact_sixteen_methods(self):
        self.assertEqual(
            tuple(METHOD_REGISTRY),
            (
                "td3_dinkelbach",
                "ddpg_dinkelbach",
                "td3_ratio",
                "ddpg_ratio",
                "random_action",
                "td3_dinkelbach_no_task_potential",
                "ddpg_dinkelbach_no_task_potential",
                "td3_dinkelbach_wo_ta",
                "td3_dinkelbach_dqn",
                "kkm_random_action_random_routing",
                "km_td3_dinkelbach",
                "random_assignment_td3_dinkelbach",
                "km_ddpg_dinkelbach",
                "ddpg_dinkelbach_wo_ta",
                "td3_dinkelbach_random_routing",
                "td3_dinkelbach_dqn_wo_ta",
            ),
        )

    def test_new_methods_are_orthogonal_combinations(self):
        km_ddpg = MethodSpec.parse("km_ddpg_dinkelbach")
        self.assertEqual(km_ddpg.assignment, "km")
        self.assertEqual(km_ddpg.assignment_rounds, 1)
        self.assertEqual(km_ddpg.agent, "ddpg")
        self.assertEqual(km_ddpg.routing, "safe_ddqn")
        self.assertFalse(movement_agent_configuration(km_ddpg)["twin_critics"])

        masked_ddpg = MethodSpec.parse("ddpg_dinkelbach_wo_ta")
        self.assertEqual(masked_ddpg.task_observation, "masked")
        self.assertTrue(masked_ddpg.task_potential_enabled)
        self.assertEqual(masked_ddpg.agent, "ddpg")
        self.assertEqual(masked_ddpg.routing, "safe_ddqn")

        random_routing = MethodSpec.parse("td3_dinkelbach_random_routing")
        self.assertEqual(random_routing.agent, "td3")
        self.assertEqual(random_routing.routing, "random")
        self.assertTrue(random_routing.learns_movement)
        self.assertFalse(random_routing.learns_routing)

        masked_dqn = MethodSpec.parse("td3_dinkelbach_dqn_wo_ta")
        self.assertEqual(masked_dqn.agent, "td3")
        self.assertEqual(masked_dqn.routing, "dqn")
        self.assertEqual(masked_dqn.task_observation, "masked")
        self.assertTrue(masked_dqn.task_potential_enabled)


class LegacyContractTest(unittest.TestCase):
    def test_fig2_contract_is_traceable_and_structural(self):
        contract = require_legacy_figure_contract("fig2")
        self.assertEqual(contract["legacy_source_commit"][:7], "57f6621")
        self.assertEqual(contract["legacy_source_file"], "HRL_task_aware.py")
        self.assertEqual(contract["figure_size_inches"], [8, 6])
        self.assertEqual(contract["subplots"]["count"], 1)
        self.assertEqual(contract["x_axis"]["label"], "Episodes")
        self.assertEqual(contract["y_axis"]["unit"], "bit/J")
        self.assertEqual(contract["legacy_output_stem"], "Total_reward")
        self.assertEqual(len(contract["methods"]), 5)

    def test_missing_legacy_visuals_fail_closed(self):
        for figure_id in ("fig3", "fig4a", "fig4b", "fig4c", "fig5", "fig6", "fig7"):
            with self.subTest(figure=figure_id):
                self.assertFalse(FIGURE_REGISTRY[figure_id]["available"])
                with self.assertRaises(LegacyFigureSourceUnavailable):
                    require_legacy_figure_contract(figure_id)


class PaperSweepContractTest(unittest.TestCase):
    def test_arrival_sweep_changes_only_the_requested_rate(self):
        points = evaluation_sweep_points("fig5_arrival")
        self.assertEqual(len(points), 8)
        com = [point for point in points if point["swept_task"] == "COM"]
        fov = [point for point in points if point["swept_task"] == "FOV"]
        self.assertEqual(
            tuple(point["x_value"] for point in com),
            ARRIVAL_RATE_SWEEPS["COM"]["values"],
        )
        self.assertEqual(
            {point["overrides"]["fov_rate_packets_per_second"] for point in com},
            {5.0},
        )
        self.assertEqual(
            tuple(point["x_value"] for point in fov),
            ARRIVAL_RATE_SWEEPS["FOV"]["values"],
        )
        self.assertEqual(
            {point["overrides"]["com_rate_packets_per_second"] for point in fov},
            {50.0},
        )

    def test_deadline_sweep_is_seconds_scoped_and_preserves_defaults(self):
        defaults = validate_production_deadlines()
        points = evaluation_sweep_points("fig6_deadline")
        self.assertEqual(len(points), 12)
        self.assertEqual(DEADLINE_SWEEP_SECONDS, (0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        for point in points:
            overrides = point["overrides"]
            self.assertEqual(point["x_unit"], "seconds")
            other = "FOV" if point["swept_task"] == "COM" else "COM"
            self.assertEqual(
                overrides[f"{other.lower()}_deadline_seconds"], defaults[other]
            )
        self.assertEqual(validate_production_deadlines(), defaults)

    def test_fixed_roi_and_trajectory_contracts(self):
        self.assertEqual(FIXED_ROI_VALUES, (2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(
            tuple(point["fixed_num_gt"] for point in evaluation_sweep_points("fig7_fixed_roi")),
            FIXED_ROI_VALUES,
        )
        self.assertEqual(TRAJECTORY_SNAPSHOT_SECONDS, (5.0, 10.0, 15.0, 25.0))
        self.assertEqual(
            tuple(PAPER_METHOD_MAPPINGS["fig7"]),
            (
                "td3_dinkelbach_random_routing",
                "td3_dinkelbach_dqn_wo_ta",
                "td3_dinkelbach_wo_ta",
                "td3_dinkelbach",
            ),
        )


if __name__ == "__main__":
    unittest.main()
