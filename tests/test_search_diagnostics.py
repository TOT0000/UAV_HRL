import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from HRL_task_aware import _mark_search_observations
from Simulator import FovCoverageTransition, Simulator
from paper_evaluation import run_paper_evaluation
from run_paper_evaluation import main as run_evaluation_main
from scenario_manifest import generate_manifest
from search_diagnostics import (
    SEARCH_DIAGNOSTICS_SCHEMA_VERSION,
    SearchDiagnosticsJsonlWriter,
    build_search_diagnostics_record,
    validate_search_diagnostics_record,
)


class SearchDiagnosticCalculationTest(unittest.TestCase):
    @staticmethod
    def _transition(uav_id, footprint):
        return FovCoverageTransition(
            uav_id=uav_id,
            previous_footprint=None,
            current_footprint=footprint,
            map_changed=True,
            raw_overlap=0.0,
            raw_unvisited=1.0,
            raw_frontier=1.0,
            coverage_contributor=True,
        )

    def test_two_partially_overlapping_footprints_use_frozen_bitmap(self):
        before = np.zeros((4, 4), dtype=bool)
        before[0, 0] = True
        after = before.copy()
        after[0:2, 0:2] = True
        after[1:3, 1:3] = True
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=2,
            time_seconds=3.0,
            visited_before=before,
            visited_after=after,
            discovered_roi_ids_before={0},
            discovered_roi_ids_after={0, 2},
            search_uav_ids=(2, 1),
            footprint_transitions=(
                self._transition(2, (1, 2, 1, 2)),
                self._transition(1, (0, 1, 0, 1)),
            ),
            interval_initial_positions={
                1: (0.0, 0.0, 100.0),
                2: (10.0, 10.0, 100.0),
            },
            interval_final_positions={
                1: (3.0, 4.0, 100.0),
                2: (10.0, 10.0, 101.0),
            },
        )
        self.assertEqual(record["schema_version"], SEARCH_DIAGNOSTICS_SCHEMA_VERSION)
        self.assertEqual(record["newly_discovered_roi_ids"], [2])
        self.assertEqual(record["search_uav_ids"], [1, 2])
        self.assertEqual(record["gross_footprint_cell_count"], 8)
        self.assertEqual(record["union_footprint_cell_count"], 7)
        self.assertEqual(record["new_union_cell_count"], 6)
        self.assertEqual(record["simultaneous_overlap_cell_count"], 1)
        self.assertAlmostEqual(record["simultaneous_overlap_ratio"], 1.0 / 8.0)
        self.assertEqual(record["historical_revisit_cell_count"], 1)
        self.assertAlmostEqual(record["historical_revisit_ratio"], 1.0 / 7.0)
        per_uav = {item["uav_id"]: item for item in record["search_uavs"]}
        self.assertEqual(per_uav[1]["footprint_cell_count"], 4)
        self.assertEqual(per_uav[1]["new_cell_count"], 3)
        self.assertEqual(per_uav[1]["new_cell_ratio"], 0.75)
        self.assertEqual(per_uav[1]["displacement_m"], 5.0)
        self.assertEqual(per_uav[2]["footprint_cell_count"], 4)
        self.assertEqual(per_uav[2]["new_cell_count"], 4)
        self.assertEqual(per_uav[2]["new_cell_ratio"], 1.0)
        self.assertEqual(per_uav[2]["displacement_m"], 1.0)

    def test_same_subslot_multi_uav_overlap_is_simultaneous(self):
        before = np.zeros((2, 2), dtype=bool)
        after = before.copy()
        after[0, 0] = True
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=before,
            visited_after=after,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(1, 2),
            footprint_transitions=(
                self._transition(1, (0, 0, 0, 0)),
                self._transition(2, (0, 0, 0, 0)),
            ),
            interval_initial_positions={1: (0, 0, 0), 2: (0, 0, 0)},
            interval_final_positions={1: (0, 0, 0), 2: (0, 0, 0)},
        )
        self.assertEqual(record["gross_footprint_cell_count"], 2)
        self.assertEqual(record["union_footprint_cell_count"], 1)
        self.assertEqual(record["simultaneous_overlap_cell_count"], 1)
        self.assertEqual(record["simultaneous_overlap_ratio"], 0.5)

    def test_cross_subslot_overlap_is_not_simultaneous(self):
        before = np.zeros((3, 3), dtype=bool)
        after = before.copy()
        after[0, 0] = True
        after[1, 1] = True
        after[2, 2] = True
        first = (
            self._transition(1, (0, 0, 0, 0)),
            self._transition(2, (2, 2, 2, 2)),
        )
        second = (
            self._transition(1, (1, 1, 1, 1)),
            self._transition(2, (0, 0, 0, 0)),
        )
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=before,
            visited_after=after,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(1, 2),
            footprint_transition_batches=(first,),
            footprint_transitions=second,
            interval_initial_positions={1: (0, 0, 0), 2: (0, 0, 0)},
            interval_final_positions={1: (0, 0, 0), 2: (0, 0, 0)},
        )
        self.assertEqual(record["gross_footprint_cell_count"], 4)
        self.assertEqual(record["union_footprint_cell_count"], 3)
        self.assertEqual(record["new_union_cell_count"], 3)
        self.assertEqual(record["historical_revisit_cell_count"], 0)
        self.assertEqual(record["simultaneous_overlap_cell_count"], 0)
        self.assertEqual(record["simultaneous_overlap_ratio"], 0.0)
        self.assertTrue(
            all(item["footprint_cell_count"] == 2 for item in record["search_uavs"])
        )

    def test_same_uav_cross_subslot_revisit_is_not_multi_uav_overlap(self):
        before = np.zeros((1, 1), dtype=bool)
        after = np.ones((1, 1), dtype=bool)
        transition = (self._transition(1, (0, 0, 0, 0)),)
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=before,
            visited_after=after,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(1,),
            footprint_transition_batches=(transition,),
            footprint_transitions=transition,
            interval_initial_positions={1: (0, 0, 0)},
            interval_final_positions={1: (0, 0, 0)},
        )
        self.assertEqual(record["gross_footprint_cell_count"], 2)
        self.assertEqual(record["union_footprint_cell_count"], 1)
        self.assertEqual(record["simultaneous_overlap_cell_count"], 0)
        self.assertEqual(record["search_uavs"][0]["footprint_cell_count"], 2)
        self.assertEqual(record["search_uavs"][0]["new_cell_count"], 1)
        self.assertEqual(record["search_uavs"][0]["new_cell_ratio"], 0.5)

    def test_four_subslot_counts_preserve_interval_and_sample_semantics(self):
        before = np.zeros((3, 3), dtype=bool)
        before[0, 0] = True
        after = before.copy()
        for x_index, y_index in ((0, 1), (1, 1), (2, 2), (2, 0)):
            after[x_index, y_index] = True
        batches = (
            (
                self._transition(1, (0, 0, 0, 0)),
                self._transition(2, (0, 0, 1, 1)),
            ),
            (
                self._transition(1, (0, 0, 1, 1)),
                self._transition(2, (0, 0, 1, 1)),
            ),
            (
                self._transition(1, (1, 1, 1, 1)),
                self._transition(2, (2, 2, 2, 2)),
            ),
            (
                self._transition(1, (1, 1, 1, 1)),
                self._transition(2, (2, 2, 0, 0)),
            ),
        )
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=before,
            visited_after=after,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(1, 2),
            footprint_transition_batches=batches[:3],
            footprint_transitions=batches[3],
            interval_initial_positions={1: (0, 0, 0), 2: (0, 0, 0)},
            interval_final_positions={1: (0, 0, 0), 2: (0, 0, 0)},
        )
        self.assertEqual(record["gross_footprint_cell_count"], 8)
        self.assertEqual(record["union_footprint_cell_count"], 5)
        self.assertEqual(record["new_union_cell_count"], 4)
        self.assertEqual(record["historical_revisit_cell_count"], 1)
        self.assertEqual(record["simultaneous_overlap_cell_count"], 1)
        self.assertEqual(record["simultaneous_overlap_ratio"], 1.0 / 8.0)
        per_uav = {item["uav_id"]: item for item in record["search_uavs"]}
        self.assertEqual(per_uav[1]["footprint_cell_count"], 4)
        self.assertEqual(per_uav[1]["new_cell_count"], 2)
        self.assertEqual(per_uav[1]["new_cell_ratio"], 0.5)
        self.assertEqual(per_uav[2]["footprint_cell_count"], 4)
        self.assertEqual(per_uav[2]["new_cell_count"], 3)
        self.assertEqual(per_uav[2]["new_cell_ratio"], 0.75)

    def test_subslot_contributor_set_and_schema_version_are_validated(self):
        bitmap = np.zeros((2, 2), dtype=bool)
        valid_batch = (
            self._transition(1, (0, 0, 0, 0)),
            self._transition(2, (1, 1, 1, 1)),
        )
        invalid_batch = (self._transition(1, (0, 0, 0, 0)),)
        with self.assertRaisesRegex(ValueError, "contributor set changed"):
            build_search_diagnostics_record(
                method_id="method",
                scenario_id="scenario",
                episode_index=0,
                interval_index=0,
                time_seconds=1.0,
                visited_before=bitmap,
                visited_after=np.ones((2, 2), dtype=bool),
                discovered_roi_ids_before=(),
                discovered_roi_ids_after=(),
                search_uav_ids=(1, 2),
                footprint_transition_batches=(
                    valid_batch,
                    invalid_batch,
                    valid_batch,
                ),
                footprint_transitions=valid_batch,
                interval_initial_positions={1: (0, 0, 0), 2: (0, 0, 0)},
                interval_final_positions={1: (0, 0, 0), 2: (0, 0, 0)},
            )

        empty_record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=bitmap,
            visited_after=bitmap,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(),
            footprint_transitions=(),
            interval_initial_positions={},
            interval_final_positions={},
        )
        legacy_record = dict(empty_record)
        legacy_record["schema_version"] = "uav-hrl-search-diagnostics-v1"
        with self.assertRaisesRegex(ValueError, "schema version"):
            validate_search_diagnostics_record(legacy_record)

    def test_legacy_search_uav_ids_are_all_nadir_coverage_contributors(self):
        env = Simulator(num_UAV=16, evaluation=True)
        env.num_GT = 2
        env.reset_environment()
        env.multi_tasks[1] = [{"task_type": "Search"}]
        env.multi_tasks[2] = [{"task_type": "COM", "target_obj_id": 0}]
        env.multi_tasks[3] = [{"task_type": "Hovering"}]
        env.multi_tasks[4] = [{"task_type": "FOV", "target_obj_id": 0}]
        env.multi_tasks[5] = [
            {"task_type": "FOV", "target_obj_id": 0},
            {"task_type": "COM", "target_obj_id": 0},
        ]
        self.assertTrue(env.is_search_contributor(1))
        self.assertTrue(env.is_search_contributor(2))
        self.assertTrue(env.is_search_contributor(3))
        self.assertFalse(env.is_search_contributor(4))
        self.assertFalse(env.is_search_contributor(5))

    def test_no_search_uav_produces_finite_zero_diagnostics(self):
        bitmap = np.zeros((3, 3), dtype=bool)
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=bitmap,
            visited_after=bitmap,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(),
            footprint_transitions=(),
            interval_initial_positions={},
            interval_final_positions={},
        )
        self.assertEqual(record["search_uav_ids"], [])
        self.assertEqual(record["search_uavs"], [])
        self.assertEqual(record["search_uav_count"], 0)
        for field in (
            "gross_footprint_cell_count",
            "union_footprint_cell_count",
            "new_union_cell_count",
            "simultaneous_overlap_cell_count",
            "historical_revisit_cell_count",
            "simultaneous_overlap_ratio",
            "historical_revisit_ratio",
        ):
            self.assertEqual(record[field], 0)
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in (
                    record["coverage_ratio_before"],
                    record["coverage_ratio_after"],
                    record["simultaneous_overlap_ratio"],
                    record["historical_revisit_ratio"],
                )
            )
        )

    def test_writer_closes_after_exception(self):
        bitmap = np.zeros((1, 1), dtype=bool)
        record = build_search_diagnostics_record(
            method_id="method",
            scenario_id="scenario",
            episode_index=0,
            interval_index=0,
            time_seconds=1.0,
            visited_before=bitmap,
            visited_after=bitmap,
            discovered_roi_ids_before=(),
            discovered_roi_ids_after=(),
            search_uav_ids=(),
            footprint_transitions=(),
            interval_initial_positions={},
            interval_final_positions={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "search.jsonl"
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with SearchDiagnosticsJsonlWriter(path) as writer:
                    writer.write(record)
                    raise RuntimeError("stop")
            path.unlink()
            self.assertFalse(path.exists())

    def test_canonical_boundary_emits_empty_row_after_search_phase(self):
        manifest = generate_manifest(
            "test", 20260919, 1, num_gt=8, environment_size_m=750
        )
        env = Simulator(
            num_UAV=16,
            evaluation=True,
            environment_width_m=750,
            environment_height_m=750,
        )
        env.reset_environment(manifest.episodes[0])
        env._search_phase_over = True
        rows = []
        initial_positions = np.asarray(
            [env.uav_dict[uav_id].get_position() for uav_id in range(env.num_UAV)]
        )
        transitions = _mark_search_observations(
            env,
            search_diagnostics_sink=rows.append,
            search_diagnostics_context={
                "method_id": "method",
                "scenario_id": manifest.episodes[0]["scenario_id"],
                "episode_index": 0,
                "interval_index": 0,
                "time_seconds": 1.0,
                "discovered_roi_ids_before": set(),
                "interval_initial_positions": initial_positions,
            },
        )
        self.assertEqual(transitions, ())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["search_uav_count"], 0)
        self.assertEqual(rows[0]["search_uav_ids"], [])
        self.assertEqual(rows[0]["search_uavs"], [])
        self.assertEqual(rows[0]["gross_footprint_cell_count"], 0)


class SearchDiagnosticCliTest(unittest.TestCase):
    def test_cli_forwards_environment_size_flag_and_defaults_off(self):
        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation", return_value={}
        ) as runner:
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite", "environment_size",
                    "--environment-size-m", "1500",
                    "--episode-horizon-s", "3",
                    "--episodes", "1",
                    "--collect-search-diagnostics",
                ]
            )
        self.assertTrue(runner.call_args.kwargs["collect_search_diagnostics"])

        with mock.patch(
            "run_paper_evaluation.run_paper_evaluation", return_value={}
        ) as runner:
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite", "environment_size",
                    "--environment-size-m", "1500",
                    "--episodes", "1",
                ]
            )
        self.assertFalse(runner.call_args.kwargs["collect_search_diagnostics"])

    def test_cli_rejects_non_environment_size_suite(self):
        with self.assertRaisesRegex(ValueError, "only for environment_size"):
            run_evaluation_main(
                [
                    "kkm_random_action_random_routing",
                    "--suite", "fixed_roi",
                    "--collect-search-diagnostics",
                ]
            )


class SearchDiagnosticArtifactTest(unittest.TestCase):
    @staticmethod
    def _run(output, enabled):
        return run_paper_evaluation(
            "kkm_random_action_random_routing",
            suite="environment_size",
            manifest_seed=20260919,
            episodes=1,
            episode_horizons_s=(3,),
            roi_counts=(8,),
            environment_sizes_m=(1500,),
            output_directory=output,
            flatten_single_point=True,
            collect_search_diagnostics=enabled,
        )

    def test_streaming_artifact_and_canonical_results_are_observational_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            disabled_dir = root / "disabled"
            enabled_dir = root / "enabled"
            disabled = self._run(disabled_dir, False)
            enabled = self._run(enabled_dir, True)

            disabled_path = disabled_dir / "search_diagnostics.jsonl"
            enabled_path = enabled_dir / "search_diagnostics.jsonl"
            self.assertFalse(disabled_path.exists())
            self.assertTrue(enabled_path.is_file())
            rows = [
                json.loads(line)
                for line in enabled_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [row["time_seconds"] for row in rows], [1.0, 2.0, 3.0]
            )
            self.assertEqual([row["episode_index"] for row in rows], [0, 0, 0])
            self.assertEqual([row["interval_index"] for row in rows], [0, 1, 2])
            for row in rows:
                validate_search_diagnostics_record(row)
                self.assertEqual(row["search_uav_count"], len(row["search_uav_ids"]))
                self.assertGreaterEqual(
                    row["gross_footprint_cell_count"],
                    row["union_footprint_cell_count"],
                )
                self.assertGreaterEqual(
                    row["union_footprint_cell_count"], row["new_union_cell_count"]
                )
                for field in (
                    "simultaneous_overlap_ratio",
                    "historical_revisit_ratio",
                ):
                    self.assertGreaterEqual(row[field], 0.0)
                    self.assertLessEqual(row[field], 1.0)
                for item in row["search_uavs"]:
                    self.assertGreaterEqual(
                        item["footprint_cell_count"], item["new_cell_count"]
                    )
                    self.assertGreaterEqual(item["new_cell_count"], 0)
                    self.assertGreaterEqual(item["new_cell_ratio"], 0.0)
                    self.assertLessEqual(item["new_cell_ratio"], 1.0)

            disabled_episodes = (disabled_dir / "per_episode.jsonl").read_text(
                encoding="utf-8"
            )
            enabled_episodes = (enabled_dir / "per_episode.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(disabled_episodes, enabled_episodes)
            self.assertEqual(
                (disabled_dir / "aggregated_plot_data.json").read_text(
                    encoding="utf-8"
                ),
                (enabled_dir / "aggregated_plot_data.json").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertNotIn("search_diagnostics_enabled", disabled)
            self.assertTrue(enabled["search_diagnostics_enabled"])
            point = enabled["points"][0]
            self.assertEqual(point["search_diagnostics_row_count"], 3)
            self.assertEqual(
                point["search_diagnostics_schema_version"],
                SEARCH_DIAGNOSTICS_SCHEMA_VERSION,
            )
            self.assertEqual(
                Path(point["search_diagnostics_jsonl"]), enabled_path.resolve()
            )
            self.assertEqual(
                Path(point["outputs"]["search_diagnostics_jsonl"]),
                enabled_path.resolve(),
            )
            run_metadata = json.loads(
                (enabled_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(run_metadata["search_diagnostics_enabled"])
            self.assertEqual(
                run_metadata["search_diagnostics_schema_version"],
                SEARCH_DIAGNOSTICS_SCHEMA_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
