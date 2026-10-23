import math
import json
from pathlib import Path
import tempfile
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
    DEADLINE_SWEEP_DISPLAY_MILLISECONDS,
    DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS,
    DEADLINE_SWEEP_SECONDS,
    FIXED_ROI_VALUES,
    PAPER_EVALUATION_SUITES,
    TRAJECTORY_SNAPSHOT_SECONDS,
    evaluation_method_supported,
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
from run_experiment import main as run_experiment_main
from scenario_manifest import generate_manifest


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
    def test_registry_contains_the_exact_methods(self):
        self.assertEqual(len(METHOD_REGISTRY), 26)
        self.assertIn("td3_dinkelbach_llm", METHOD_REGISTRY)
        self.assertIn("td3_dinkelbach_llm_search", METHOD_REGISTRY)
        self.assertIn("td3_dinkelbach_llm_train_search", METHOD_REGISTRY)
        self.assertIn("td3_dinkelbach_llm_potential", METHOD_REGISTRY)
        self.assertIn("kkm_random_action_random_routing", METHOD_REGISTRY)
        self.assertIn("td3_dinkelbach_ddqn", METHOD_REGISTRY)

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
            self.assertEqual(resolved["packet_injection_cutoff_seconds"], 59.985)
            self.assertEqual(resolved["resolved_fov_deadline_seconds"], 0.015)
            self.assertEqual(resolved["resolved_com_deadline_seconds"], 0.010)
            if spec.routing == "safe_ddqn":
                safe_method_count += 1
                self.assertEqual(resolved["safe_ddqn_qos_target_probability"], 0.01)
                self.assertEqual(
                    resolved["safe_ddqn_lambda_update_mode"],
                    "direct_episode_system_dvp",
                )
                self.assertEqual(resolved["safe_ddqn_initial_lambda_cost"], 0.0)
                self.assertEqual(resolved["safe_ddqn_eta_c"], 0.01)
            else:
                self.assertIsNone(resolved["safe_ddqn_qos_target_probability"])
                self.assertIsNone(resolved["safe_ddqn_eta_c"])
        self.assertEqual(safe_method_count, 21)

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
        self.assertEqual(
            DEADLINE_SWEEP_SECONDS,
            (0.005, 0.010, 0.015, 0.020, 0.025, 0.030),
        )
        self.assertEqual(
            DEADLINE_SWEEP_DISPLAY_MILLISECONDS,
            (5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
        )
        self.assertEqual(DEADLINE_SWEEP_INJECTION_CUTOFF_SECONDS, 59.97)
        self.assertEqual(
            {
                point["overrides"]["packet_injection_cutoff_seconds"]
                for point in points
            },
            {59.97},
        )
        com = [point for point in points if point["swept_task"] == "COM"]
        fov = [point for point in points if point["swept_task"] == "FOV"]
        self.assertEqual(
            tuple(point["x_value"] for point in com),
            DEADLINE_SWEEP_DISPLAY_MILLISECONDS,
        )
        self.assertEqual(
            tuple(point["simulation_value_seconds"] for point in fov),
            DEADLINE_SWEEP_SECONDS,
        )
        self.assertEqual(
            {point["overrides"]["fov_deadline_seconds"] for point in com},
            {0.015},
        )
        self.assertEqual(
            {point["overrides"]["com_deadline_seconds"] for point in fov},
            {0.010},
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
            deadline_seconds=(0.007, 0.011, 0.019, 0.023),
            episode_seconds=45,
        )
        self.assertEqual(len(points), 8)
        com = [point for point in points if point["swept_task"] == "COM"]
        fov = [point for point in points if point["swept_task"] == "FOV"]
        self.assertEqual(
            tuple(point["x_value"] for point in com),
            (7.0, 11.0, 19.0, 23.0),
        )
        self.assertEqual(
            tuple(point["simulation_value_seconds"] for point in fov),
            (0.007, 0.011, 0.019, 0.023),
        )
        self.assertEqual(
            {point["overrides"]["fov_deadline_seconds"] for point in com},
            {0.015},
        )
        self.assertEqual(
            {point["overrides"]["com_deadline_seconds"] for point in fov},
            {0.010},
        )
        self.assertEqual(
            {
                point["overrides"]["packet_injection_cutoff_seconds"]
                for point in points
            },
            {44.977},
        )
        for point in points:
            self.assertEqual(point["x_unit"], "ms")
            self.assertEqual(point["simulation_unit"], "seconds")
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

    def test_llm_explorer_methods_are_explicitly_eligible_without_changing_defaults(self):
        explorer_methods = (
            "td3_dinkelbach_llm_explorer",
            "td3_dinkelbach_llm_explorer_v3",
            "td3_dinkelbach_llm_explorer_td3_only",
            "td3_dinkelbach_llm_explorer_state_rule",
        )
        suites = (
            "task_type_delay_vs_arrival_rate",
            "task_type_delay_violation_vs_target_delay",
            "uav_trajectory_snapshots",
        )
        for method_id in explorer_methods:
            for suite in suites:
                with self.subTest(method=method_id, suite=suite):
                    self.assertTrue(evaluation_method_supported(method_id, suite))
                    self.assertNotIn(
                        method_id,
                        PAPER_EVALUATION_SUITES[suite]["methods"],
                    )

    def test_llm_explorer_trajectory_path_completes_offline_with_real_identity(self):
        method_id = "td3_dinkelbach_llm_explorer_v3"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_root = root / "training"
            self.assertEqual(
                run_experiment_main(
                    [
                        method_id,
                        "--llm-explorer-mock",
                        "--episodes",
                        "1",
                        "--episode-seconds",
                        "1",
                        "--checkpoint-interval",
                        "1",
                        "--roi-count",
                        "2",
                        "--seed",
                        "20260817",
                        "--output-root",
                        str(training_root),
                    ]
                ),
                0,
            )
            run_dir = next((training_root / method_id).iterdir())
            manifest = generate_manifest("test", 20260818, 1, num_gt=2)
            manifest_path = root / "trajectory_manifest.json"
            manifest.save(manifest_path)
            output = root / "trajectory_evaluation"
            with mock.patch(
                "paper_evaluation.TRAJECTORY_SNAPSHOT_SECONDS", (1.0,)
            ), mock.patch(
                "llm_design.OpenAIClient",
                side_effect=AssertionError("evaluation must not create a live client"),
            ):
                result = run_paper_evaluation(
                    method_id,
                    run_directory=run_dir,
                    suite="uav_trajectory_snapshots",
                    manifest_path=manifest_path,
                    episodes=1,
                    episode_seconds=1,
                    checkpoint_episode=1,
                    target_uav_id=0,
                    output_directory=output,
                )
            self.assertEqual(result["method_id"], method_id)
            self.assertEqual(
                result["points"][0]["scenario_ids"],
                [entry["scenario_id"] for entry in manifest.episodes],
            )
            point_dir = output / "trajectory"
            artifacts = json.loads(
                (point_dir / "trajectory_artifacts.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["method_id"], method_id)
            self.assertEqual(artifacts[0]["target_uav_id"], 0)
            self.assertEqual(artifacts[0]["requested_times_seconds"], [1.0])
            self.assertEqual(
                artifacts[0]["scenario_manifest_hash"], manifest.content_hash
            )
            self.assertTrue(artifacts[0]["checkpoint_path"].endswith("ep_0001"))
            point_metadata = json.loads(
                (point_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(all(point_metadata["evaluation_invariants"].values()))

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
