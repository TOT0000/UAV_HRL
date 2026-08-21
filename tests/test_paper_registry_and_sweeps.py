import unittest

from experiment_config import METHOD_REGISTRY, MethodSpec, movement_agent_configuration
from paper_evaluation import (
    ARRIVAL_RATE_SWEEPS,
    DEADLINE_SWEEP_SECONDS,
    FIXED_ROI_VALUES,
    PAPER_EVALUATION_SUITES,
    TRAJECTORY_SNAPSHOT_SECONDS,
    evaluation_sweep_points,
)
from paper_figure_registry import (
    DEPRECATED_FIGURE_ALIASES,
    DRIVE_SOURCES,
    FIGURE_REGISTRY,
    PAPER_METHOD_MAPPINGS,
    resolve_figure_id,
    resolve_figure_ids,
)
from build_paper_figures import build_parser as build_figure_parser
from run_paper_evaluation import build_parser as build_evaluation_parser


EXPECTED_SEMANTIC_FIGURES = (
    "uav_trajectory_t_5s",
    "uav_trajectory_t_10s",
    "uav_trajectory_t_15s",
    "uav_trajectory_t_25s",
    "training_ee_vs_episode",
    "task_assignment_ee_vs_number_of_rois",
    "trajectory_design_ee_vs_number_of_rois",
    "hierarchical_architecture_ee_vs_number_of_rois",
    "com_task_delay_vs_arrival_rate",
    "vs_task_delay_vs_arrival_rate",
    "task_type_delay_violation_vs_target_delay",
    "task_type_delay_vs_number_of_rois",
)


class PaperMethodRegistryTest(unittest.TestCase):
    def test_registry_contains_the_exact_sixteen_methods(self):
        self.assertEqual(len(METHOD_REGISTRY), 16)
        self.assertIn("kkm_random_action_random_routing", METHOD_REGISTRY)

    def test_comparison_methods_are_orthogonal_combinations(self):
        km_ddpg = MethodSpec.parse("km_ddpg_dinkelbach")
        self.assertEqual((km_ddpg.assignment, km_ddpg.agent), ("km", "ddpg"))
        self.assertFalse(movement_agent_configuration(km_ddpg)["twin_critics"])
        pure_random = MethodSpec.parse("kkm_random_action_random_routing")
        self.assertFalse(pure_random.learns_movement)
        self.assertFalse(pure_random.learns_routing)


class SemanticContractTest(unittest.TestCase):
    def test_only_semantic_ids_are_canonical(self):
        self.assertEqual(tuple(FIGURE_REGISTRY), EXPECTED_SEMANTIC_FIGURES)
        self.assertFalse(any(key.startswith("fig") for key in FIGURE_REGISTRY))
        self.assertEqual(resolve_figure_id("fig6"), "task_type_delay_violation_vs_target_delay")
        self.assertEqual(
            resolve_figure_ids("fig4"),
            (
                "task_assignment_ee_vs_number_of_rois",
                "trajectory_design_ee_vs_number_of_rois",
                "hierarchical_architecture_ee_vs_number_of_rois",
            ),
        )
        self.assertIn("fig6", DEPRECATED_FIGURE_ALIASES)

    def test_cli_choices_expose_only_semantic_names(self):
        evaluation_help = build_evaluation_parser().format_help()
        figure_help = build_figure_parser().format_help()
        self.assertNotIn("fig3_trajectory", evaluation_help)
        self.assertNotIn("fig4a", figure_help)
        self.assertIn("uav_trajectory_snapshots", evaluation_help)
        self.assertNotIn("uav_trajectory_snapshots", figure_help)
        self.assertNotIn("energy_efficiency_design_comparisons", figure_help)
        self.assertNotIn("task_type_delay_vs_arrival_rate", figure_help)
        self.assertIn("uav_trajectory_t_5s", figure_help)
        self.assertIn("training_ee_vs_episode", figure_help)

    def test_drive_provenance_has_read_only_audit_fingerprints(self):
        for source in DRIVE_SOURCES.values():
            self.assertTrue(source["file_id"])
            self.assertEqual(len(source["sha256"]), 64)
            self.assertTrue(source["path"].endswith(".py"))
        for contract in FIGURE_REGISTRY.values():
            self.assertTrue(contract["sources"])
            self.assertTrue(contract["production_source"])
            self.assertTrue(contract["screenshot_reference"])


class PaperSweepContractTest(unittest.TestCase):
    def test_cli_suites_have_no_deprecated_fig_names(self):
        self.assertFalse(any(name.startswith("fig") for name in PAPER_EVALUATION_SUITES))

    def test_arrival_sweep_changes_only_the_requested_rate(self):
        points = evaluation_sweep_points("task_type_delay_vs_arrival_rate")
        self.assertEqual(len(points), 8)
        com = [point for point in points if point["swept_task"] == "COM"]
        fov = [point for point in points if point["swept_task"] == "FOV"]
        self.assertEqual(tuple(point["x_value"] for point in com), ARRIVAL_RATE_SWEEPS["COM"]["values"])
        self.assertEqual({point["overrides"]["fov_rate_packets_per_second"] for point in com}, {5.0})
        self.assertEqual(tuple(point["x_value"] for point in fov), ARRIVAL_RATE_SWEEPS["FOV"]["values"])
        self.assertEqual({point["overrides"]["com_rate_packets_per_second"] for point in fov}, {50.0})

    def test_deadline_and_fixed_roi_contracts(self):
        points = evaluation_sweep_points("task_type_delay_violation_vs_target_delay")
        self.assertEqual(len(points), 12)
        self.assertEqual(DEADLINE_SWEEP_SECONDS, (0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        self.assertEqual(FIXED_ROI_VALUES, tuple(range(2, 9)))
        self.assertEqual(
            tuple(point["fixed_num_gt"] for point in evaluation_sweep_points("fixed_roi")),
            FIXED_ROI_VALUES,
        )
        self.assertEqual(TRAJECTORY_SNAPSHOT_SECONDS, (5.0, 10.0, 15.0, 25.0))
        self.assertEqual(
            tuple(PAPER_METHOD_MAPPINGS["task_type_delay_vs_number_of_rois"]),
            (
                "td3_dinkelbach_random_routing",
                "td3_dinkelbach_dqn_wo_ta",
                "td3_dinkelbach_wo_ta",
                "td3_dinkelbach",
            ),
        )


if __name__ == "__main__":
    unittest.main()
