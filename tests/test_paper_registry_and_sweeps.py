import math
import unittest
from unittest import mock

from experiment_config import (
    METHOD_REGISTRY,
    MethodSpec,
    comparison_method_configuration,
    movement_agent_configuration,
)
from paper_evaluation import (
    ARRIVAL_RATE_SWEEPS,
    DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS,
    DEADLINE_SWEEP_SECONDS,
    FIXED_ROI_VALUES,
    PAPER_EVALUATION_SUITES,
    TRAJECTORY_SNAPSHOT_SECONDS,
    evaluation_sweep_points,
    run_paper_evaluation,
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
from run_paper_evaluation import (
    build_parser as build_evaluation_parser,
    main as run_evaluation_main,
)


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

    def test_shared_lifecycle_metadata_is_resolved_for_all_methods(self):
        safe_method_count = 0
        for method_id in METHOD_REGISTRY:
            spec = MethodSpec.parse(method_id)
            resolved = comparison_method_configuration(spec)
            self.assertEqual(resolved["routing_mask_scope"], "every_slot")
            self.assertEqual(resolved["packet_injection_cutoff_seconds"], 57.5)
            self.assertEqual(resolved["resolved_fov_deadline_seconds"], 2.5)
            self.assertEqual(resolved["resolved_com_deadline_seconds"], 2.0)
            if spec.routing == "safe_ddqn":
                safe_method_count += 1
                self.assertEqual(resolved["safe_ddqn_qos_target_probability"], 0.05)
                self.assertEqual(
                    resolved["safe_ddqn_dual_normalization_reference_packets"],
                    10_000,
                )
                self.assertEqual(resolved["safe_ddqn_initial_lambda_cost"], 0.0)
                self.assertEqual(resolved["safe_ddqn_eta_c"], 0.01)
            else:
                self.assertIsNone(resolved["safe_ddqn_qos_target_probability"])
                self.assertIsNone(resolved["safe_ddqn_eta_c"])
        self.assertEqual(safe_method_count, 12)

    def test_td3_and_ddpg_hyperparameters_are_explicit_and_unchanged(self):
        td3 = movement_agent_configuration(MethodSpec.parse("td3_dinkelbach"))
        ddpg = movement_agent_configuration(MethodSpec.parse("ddpg_dinkelbach"))
        for resolved in (td3, ddpg):
            self.assertEqual(resolved["actor_learning_rate"], 6e-5)
            self.assertEqual(resolved["critic_learning_rate"], 2e-4)
            self.assertEqual(resolved["movement_agent_gamma"], 1.0)
            self.assertEqual(resolved["tau"], 0.005)
            self.assertEqual(resolved["batch_size"], 64)
            self.assertEqual(resolved["replay_capacity"], 50_000)
            self.assertEqual(resolved["warmup_joint_transitions"], 10_000)
            self.assertEqual(resolved["exploration_noise_start"], 0.20)
            self.assertEqual(resolved["exploration_noise_end"], 0.05)
        self.assertEqual(td3["policy_delay"], 2)
        self.assertEqual(td3["target_policy_noise"], 0.10)
        self.assertEqual(td3["target_noise_clip"], 0.25)
        self.assertTrue(td3["twin_critics"])
        self.assertEqual(ddpg["policy_delay"], 1)
        self.assertIsNone(ddpg["target_policy_noise"])
        self.assertIsNone(ddpg["target_noise_clip"])
        self.assertFalse(ddpg["twin_critics"])


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
        self.assertEqual(DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS, 57.0)
        self.assertEqual(
            {
                point["overrides"]["packet_injection_cutoff_seconds"]
                for point in points
            },
            {57.0},
        )
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

    def test_custom_deadline_sweep_preserves_order_and_opposite_defaults(self):
        points = evaluation_sweep_points(
            "task_type_delay_violation_vs_target_delay",
            deadline_seconds=(1, 2, 4, 5),
            episode_seconds=60,
        )
        self.assertEqual(len(points), 8)
        com = [point for point in points if point["swept_task"] == "COM"]
        fov = [point for point in points if point["swept_task"] == "FOV"]
        self.assertEqual(tuple(point["x_value"] for point in com), (1.0, 2.0, 4.0, 5.0))
        self.assertEqual(tuple(point["x_value"] for point in fov), (1.0, 2.0, 4.0, 5.0))
        self.assertEqual(
            {point["overrides"]["fov_deadline_seconds"] for point in com},
            {2.5},
        )
        self.assertEqual(
            {point["overrides"]["com_deadline_seconds"] for point in fov},
            {2.0},
        )
        self.assertEqual(
            {
                point["overrides"]["packet_injection_cutoff_seconds"]
                for point in points
            },
            {55.0},
        )
        for point in points:
            self.assertEqual(point["x_unit"], "seconds")
            self.assertIn("fov_deadline_seconds", point["overrides"])
            self.assertIn("com_deadline_seconds", point["overrides"])

    def test_invalid_custom_deadline_values_fail_fast(self):
        invalid_cases = (
            ((), "at least one"),
            ((0,), "greater than zero"),
            ((-1, 1), "greater than zero"),
            ((math.nan,), "finite"),
            ((math.inf,), "finite"),
            ((1, 1, 2), "duplicates"),
        )
        for values, message in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    evaluation_sweep_points(
                        "task_type_delay_violation_vs_target_delay",
                        deadline_seconds=values,
                        episode_seconds=60,
                    )

    def test_custom_deadline_must_fit_episode_horizon(self):
        with self.assertRaisesRegex(ValueError, "less than episode_seconds"):
            evaluation_sweep_points(
                "task_type_delay_violation_vs_target_delay",
                deadline_seconds=(1, 5),
                episode_seconds=5,
            )

    def test_run_api_rejects_invalid_deadline_before_loading_checkpoint(self):
        with mock.patch("paper_evaluation._load_training_run") as load_run:
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                run_paper_evaluation(
                    "td3_dinkelbach",
                    run_directory="unused",
                    suite="task_type_delay_violation_vs_target_delay",
                    deadline_seconds=(0,),
                    episode_seconds=60,
                )
        load_run.assert_not_called()

    def test_deadline_selector_is_rejected_for_other_suites(self):
        with self.assertRaisesRegex(ValueError, "only for"):
            evaluation_sweep_points(
                "fixed_roi",
                deadline_seconds=(1,),
                episode_seconds=60,
            )
        with self.assertRaisesRegex(ValueError, "only for"):
            run_paper_evaluation(
                "td3_dinkelbach",
                suite="fixed_roi",
                deadline_seconds=(1,),
            )
        with self.assertRaisesRegex(ValueError, "only for"):
            run_evaluation_main(
                [
                    "td3_dinkelbach",
                    "--suite",
                    "fixed_roi",
                    "--deadline-seconds",
                    "1",
                ]
            )

    def test_cli_parses_custom_deadline_seconds_as_floats(self):
        args = build_evaluation_parser().parse_args(
            [
                "td3_dinkelbach",
                "--suite",
                "task_type_delay_violation_vs_target_delay",
                "--deadline-seconds",
                "0.5",
                "1",
                "4",
            ]
        )
        self.assertEqual(args.deadline_seconds, [0.5, 1.0, 4.0])

    def test_cli_passes_custom_deadline_seconds_to_evaluation(self):
        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation",
            return_value={},
        ) as run_evaluation:
            exit_code = run_evaluation_main(
                [
                    "td3_dinkelbach",
                    "--run-dir",
                    "run",
                    "--suite",
                    "task_type_delay_violation_vs_target_delay",
                    "--deadline-seconds",
                    "0.5",
                    "2",
                    "5",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run_evaluation.call_args.kwargs["deadline_seconds"],
            [0.5, 2.0, 5.0],
        )


if __name__ == "__main__":
    unittest.main()
