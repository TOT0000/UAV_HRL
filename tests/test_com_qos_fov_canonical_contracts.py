import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluation_metrics
from evaluation_aggregation import canonical_aggregation
from evaluation_metrics import (
    DESCRIPTIVE_EPISODE_METRIC_COLUMNS,
    EPISODE_COLUMNS,
    METRIC_COLUMNS,
    aggregate_descriptive_seed_metrics,
    build_generic_cross_seed_artifact,
    canonical_cross_seed_artifact_rows,
    run_aggregate_command,
    summarize_training_seeds,
)
from HRL_task_aware import _mark_search_observations
from observation_strategy import routing_state_feature_names
from Packet_scheduler_v1 import PacketEngine
from paper_metrics import aggregate_paper_point_metrics
from scenario_manifest import generate_manifest
from Simulator import Simulator


class ActivatedComQosContractTest(unittest.TestCase):
    @staticmethod
    def _environment(in_range):
        return SimpleNamespace(
            source_uavs=set(),
            multi_tasks={0: [{"task_type": "COM", "target_obj_id": 0}]},
            SR_teams=[SimpleNamespace(id=0, assigned_gt_id=0)],
            load_factor=1.0,
            is_s2u_in_range=lambda sr_id, uav_id: in_range["value"],
        )

    def test_four_activated_packets_expire_at_sr_as_system_only_violations(self):
        in_range = {"value": True}
        env = self._environment(in_range)
        engine = PacketEngine(
            num_uav=1, task_deadlines_seconds={"COM": 0.5}
        )
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=0.0,
            step_time=0.25,
            rate_overrides={"FOV": 0.0, "COM": 0.0},
        )
        self.assertTrue(engine.com_sessions[0]["session_active"])

        in_range["value"] = False
        engine.inject_packets(
            env,
            delay_bound_steps=20,
            current_time=0.25,
            step_time=1.0,
            rate_overrides={"FOV": 0.0, "COM": 4.0},
        )
        events = engine.expire_packets(0.75, inclusive=True)
        summary = engine.packet_metric_summary()["COM"]

        self.assertEqual(engine.generated_packet_counts["COM"], 4)
        self.assertEqual(engine.eligible_packet_counts["COM"], 4)
        self.assertEqual(len(events), 4)
        self.assertEqual(summary["expired_dropped_packets"], 4)
        self.assertEqual(summary["violation_packets"], 4)
        self.assertEqual(summary["violation_probability"], 1.0)
        self.assertEqual(engine.routing_constraint_counts(), (0, 0))
        self.assertEqual(engine.routing_immediate_cost_sum, 0.0)
        self.assertEqual(engine.pre_routing_violation_count, 4)

    def test_s2u_completion_changes_only_routing_eligibility(self):
        engine = PacketEngine(num_uav=1)
        packet = engine.create_sr_packet(0, 100.0, generation_time=0.0)
        env = SimpleNamespace(
            multi_tasks={0: [{"task_type": "COM", "target_obj_id": 0}]}
        )

        self.assertTrue(packet["qos_eligible"])
        self.assertFalse(packet["routing_eligible"])
        self.assertEqual(engine.eligible_packet_counts["COM"], 1)
        result = engine.serve_s2u_links(
            env, {(0, 0): 0.001}, current_time=0.0
        )

        self.assertEqual(result["violations"], [])
        self.assertTrue(packet["routing_eligible"])
        self.assertEqual(engine.eligible_packet_counts["COM"], 1)

    def test_s2u_deadline_and_repeated_settlement_count_once(self):
        engine = PacketEngine(
            num_uav=1, task_deadlines_seconds={"COM": 0.1}
        )
        packet = engine.create_sr_packet(0, 100.0, generation_time=0.0)
        env = SimpleNamespace(
            multi_tasks={0: [{"task_type": "COM", "target_obj_id": 0}]}
        )
        result = engine.serve_s2u_links(
            env, {(0, 0): 0.001}, current_time=0.0
        )
        engine.expire_packets(0.25, inclusive=True)
        summary = engine.finalize_episode(0.25)["COM"]

        self.assertTrue(packet["done"])
        self.assertEqual(len(result["violations"]), 1)
        self.assertEqual(summary["eligible_packets"], 1)
        self.assertEqual(summary["violation_packets"], 1)
        self.assertEqual(engine.total_violated, 1)
        self.assertEqual(engine.pre_routing_violation_count, 1)

    def test_never_activated_session_has_missing_probability(self):
        in_range = {"value": False}
        engine = PacketEngine(num_uav=1)
        engine.inject_packets(
            self._environment(in_range),
            delay_bound_steps=20,
            current_time=0.0,
            step_time=1.0,
            rate_overrides={"FOV": 0.0, "COM": 4.0},
        )
        summary = engine.finalize_episode(1.0)["COM"]

        self.assertEqual(summary["source_generated_packets"], 0)
        self.assertEqual(summary["eligible_packets"], 0)
        self.assertEqual(summary["violation_packets"], 0)
        self.assertIsNone(summary["violation_probability"])

    def test_terminal_sr_queue_conserves_formal_qos_outcomes(self):
        engine = PacketEngine(num_uav=1)
        packets = [engine.create_sr_packet(0, 256.0, 0.0) for _ in range(3)]
        summary = engine.finalize_episode(1.0)["COM"]

        self.assertTrue(all(packet["done"] for packet in packets))
        self.assertEqual(summary["source_generated_packets"], 3)
        self.assertEqual(summary["eligible_packets"], 3)
        self.assertEqual(summary["expired_dropped_packets"], 3)
        self.assertEqual(summary["violation_packets"], 3)
        self.assertEqual(engine.routing_constraint_counts(), (0, 0))
        self.assertEqual(engine.pre_routing_violation_count, 3)


class AllParticipantFovSnapshotContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = generate_manifest("test", 9501, 1).episodes[0]

    def _environment(self):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        env._search_phase_over = False
        env.visited_bitmap[:] = False
        env.multi_tasks = {uav_id: [] for uav_id in range(env.num_UAV)}
        env.multi_tasks[0] = [{"task_type": "Search"}]
        env.uav_dict[0].x_u, env.uav_dict[0].y_u = 100.0, 100.0
        env.uav_dict[1].x_u, env.uav_dict[1].y_u = 900.0, 900.0
        return env

    def test_non_search_observes_precommit_without_contributing_coverage(self):
        env = self._environment()
        transitions = _mark_search_observations(env)
        by_uav = {transition.uav_id: transition for transition in transitions}
        search = by_uav[0]
        non_search = by_uav[1]

        self.assertEqual(len(transitions), env.num_UAV)
        self.assertTrue(search.coverage_contributor)
        self.assertFalse(non_search.coverage_contributor)
        self.assertEqual(search.raw_unvisited, 1.0)
        self.assertEqual(non_search.raw_unvisited, 1.0)
        bx0, bx1, by0, by1 = non_search.current_footprint
        self.assertFalse(env.visited_bitmap[bx0 : bx1 + 1, by0 : by1 + 1].any())

        first = PacketEngine(num_uav=10)
        second = PacketEngine(num_uav=10)
        self.assertTrue(first.process_fov_transitions(env, "a", transitions))
        self.assertTrue(
            second.process_fov_transitions(env, "a", tuple(reversed(transitions)))
        )
        self.assertEqual(first.fov_ema_state(), second.fov_ema_state())
        self.assertEqual(first.fov_ema_update_count, 1)
        self.assertEqual(len(first.fov_ema), env.num_UAV)
        self.assertAlmostEqual(first.fov_ema[1]["unvisited"], 0.3)

        env.update_u2u_channels()
        env.update_u2g_channels()
        state = first.get_state_ta(
            env,
            1,
            backlog_bits=first.backlog_bits,
            action_mask=env.get_routing_action_mask(1),
        )
        index = routing_state_feature_names().index("coverage_unvisited_ema")
        self.assertAlmostEqual(float(state[index]), 0.3)

    def test_missing_participant_batch_is_rejected_after_commit(self):
        env = self._environment()
        transitions = _mark_search_observations(env)
        with self.assertRaisesRegex(RuntimeError, "every UAV"):
            PacketEngine(num_uav=10).process_fov_transitions(
                env, "incomplete", transitions[:-1]
            )


class CanonicalArtifactSourceContractTest(unittest.TestCase):
    @staticmethod
    def _row(seed, episode, violations, eligible):
        row = {
            "method_id": "method",
            "training_seed": seed,
            "evaluation_split": "test",
            "evaluation_manifest_hash": "evaluation",
            "training_manifest_hash": "training",
            "checkpoint_completed_episodes": 1500,
            "checkpoint_metadata_fingerprint": f"checkpoint-{seed}",
            "scenario_id": f"episode-{episode}",
            "timely_goodput_mbits": float(eligible),
            "total_timely_useful_mbits": float(eligible),
            "total_mobility_energy_j": float(max(eligible, 1)),
            "fov_delivered_packets": eligible,
            "fov_delivered_e2e_delay_sum_seconds": float(eligible),
            "fov_violation_packets": violations,
            "fov_eligible_packets": eligible,
            "com_delivered_packets": 0,
            "com_delivered_e2e_delay_sum_seconds": 0.0,
            "com_violation_packets": 0,
            "com_eligible_packets": 0,
        }
        for metric in METRIC_COLUMNS:
            row.setdefault(metric, 0.0)
        return row

    def test_generic_and_paper_serialize_the_same_canonical_statistics(self):
        rows = [
            self._row(1, 0, 1, 1),
            self._row(1, 1, 0, 99),
            self._row(2, 0, 1, 2),
            self._row(3, 0, 0, 0),
        ]
        per_seed, canonical = canonical_aggregation(rows)
        generic = canonical_cross_seed_artifact_rows(canonical, rows)
        paper = aggregate_paper_point_metrics(
            "method",
            "fixed_roi",
            {"point_id": "roi_2", "x_value": 2, "x_unit": "RoIs"},
            rows,
        )
        identity = ("violation_probability", "FOV")
        canonical_row = next(
            row
            for row in canonical
            if (row["metric"], row["task_type"]) == identity
        )
        generic_row = next(
            row
            for row in generic
            if (row["metric"], row["task_type"]) == identity
        )
        paper_row = next(
            row
            for row in paper
            if (row["metric"], row["task_type"]) == identity
        )

        for field in (
            "valid_training_seed_count",
            "missing_training_seed_count",
            "training_seed_count",
            "valid_training_seeds",
            "sample_stddev",
            "degrees_of_freedom",
            "t_critical_975",
            "ci95_half_width",
            "ci95_lower",
            "ci95_upper",
            "per_seed_numerators",
            "per_seed_denominators",
            "per_seed_values",
        ):
            self.assertEqual(generic_row[field], canonical_row[field])
            self.assertEqual(paper_row[field], canonical_row[field])
        self.assertEqual(generic_row["mean"], paper_row["value"])
        self.assertEqual(canonical_row["valid_training_seed_count"], 2)
        self.assertEqual(canonical_row["per_seed_values"], [0.01, 0.5, None])
        self.assertFalse(hasattr(evaluation_metrics, "aggregate_seed_means"))
        self.assertEqual(len(per_seed), 18)

    def test_generic_artifacts_restore_diagnostics_and_preserve_canonical_rows(self):
        rows = [
            self._row(1, 0, 1, 1),
            self._row(1, 1, 0, 99),
            self._row(2, 0, 1, 2),
            self._row(2, 1, 0, 0),
        ]
        for index, row in enumerate(rows, start=1):
            for metric in DESCRIPTIVE_EPISODE_METRIC_COLUMNS:
                row[metric] = float(index)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input" / "seed-results"
            output_dir = root / "aggregate"
            input_dir.mkdir(parents=True)
            with (input_dir / "per_episode.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            args = SimpleNamespace(
                input_dir=input_dir.parent,
                output_dir=output_dir,
                method=SimpleNamespace(method_id="method"),
                split="test",
                expected_seed_count=2,
                expected_episodes_per_seed=2,
                manifest=None,
            )
            self.assertEqual(run_aggregate_command(args), 0)
            with (output_dir / "cross_seed_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            generic_rows = json.loads(
                (output_dir / "cross_seed_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            canonical_rows = json.loads(
                (output_dir / "canonical_cross_seed_aggregation.json").read_text(
                    encoding="utf-8"
                )
            )
            aggregation_metadata = json.loads(
                (output_dir / "aggregation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(canonical_rows), 6)
        expected_rows = 7 + len(DESCRIPTIVE_EPISODE_METRIC_COLUMNS)
        self.assertEqual(len(generic_rows), expected_rows)
        self.assertEqual(len(csv_rows), expected_rows)
        self.assertEqual(
            aggregation_metadata["generic_cross_seed_artifact_schema_version"],
            "uav-hrl-generic-cross-seed-aggregate-v3",
        )
        kind_counts = {
            kind: sum(row["aggregation_kind"] == kind for row in generic_rows)
            for kind in (
                "canonical_ratio",
                "canonical_alias",
                "descriptive_seed_mean",
            )
        }
        self.assertEqual(
            kind_counts,
            {
                "canonical_ratio": 6,
                "canonical_alias": 1,
                "descriptive_seed_mean": len(
                    DESCRIPTIVE_EPISODE_METRIC_COLUMNS
                ),
            },
        )
        identities = {
            (row["metric"], row.get("task_type")) for row in generic_rows
        }
        for row in generic_rows:
            self.assertEqual(row["method_id"], "method")
            self.assertEqual(row["evaluation_split"], "test")
            self.assertEqual(row["evaluation_manifest_hash"], "evaluation")
            self.assertEqual(row["training_manifest_hash"], "training")
            self.assertEqual(row["checkpoint_completed_episodes"], 1500)
            self.assertEqual(row["training_seeds"], [1, 2])
            self.assertEqual(
                row["checkpoint_identities"],
                [
                    {
                        "training_seed": 1,
                        "checkpoint_metadata_fingerprint": "checkpoint-1",
                    },
                    {
                        "training_seed": 2,
                        "checkpoint_metadata_fingerprint": "checkpoint-2",
                    },
                ],
            )
        for metric in (
            "coverage",
            "found_GT_ratio",
            "raw_final_hop_mbits",
            "total_mobility_energy_j",
            "routing_wait_count",
            "system_qos_violation_count",
            "routing_stage_violated_packets",
            "pre_routing_violation_count",
        ):
            self.assertIn((metric, None), identities)

        generic = next(
            row
            for row in generic_rows
            if row["metric"] == "violation_probability"
            and row["task_type"] == "FOV"
        )
        canonical = next(
            row
            for row in canonical_rows
            if row["metric"] == "violation_probability"
            and row["task_type"] == "FOV"
        )
        for field in (
            "mean",
            "sample_stddev",
            "ci95_lower",
            "ci95_upper",
            "valid_training_seed_count",
            "pooled_numerator",
            "pooled_denominator",
        ):
            self.assertEqual(generic[field], canonical[field])
        self.assertEqual(
            generic["aggregation_schema_version"],
            canonical["aggregation_schema_version"],
        )
        for field, value in canonical.items():
            self.assertEqual(generic[field], value)

        alias = next(
            row
            for row in generic_rows
            if row["metric"] == "delay_violation_probability"
        )
        all_violation = next(
            row
            for row in generic_rows
            if (row["metric"], row.get("task_type"))
            == ("violation_probability", "ALL")
        )
        self.assertEqual(alias["alias_of_metric"], "violation_probability")
        self.assertEqual(alias["alias_of_task_type"], "ALL")
        for field in (
            "mean",
            "sample_stddev",
            "valid_training_seed_count",
            "missing_training_seed_count",
            "t_critical_975",
            "ci95_half_width",
            "ci95_lower",
            "ci95_upper",
            "per_seed_values",
            "per_seed_numerators",
            "per_seed_denominators",
        ):
            self.assertEqual(alias[field], all_violation[field])

        csv_by_identity = {
            (
                row["metric"],
                row["task_type"] or None,
                row["aggregation_kind"],
            ): row
            for row in csv_rows
        }
        for json_row in generic_rows:
            csv_row = csv_by_identity[
                (
                    json_row["metric"],
                    json_row.get("task_type"),
                    json_row["aggregation_kind"],
                )
            ]
            for field, value in json_row.items():
                if isinstance(value, (list, dict)):
                    self.assertEqual(json.loads(csv_row[field]), value)

        # The union schema carries fields that only apply to other row kinds.
        self.assertIn("alias_of_metric", csv_rows[0])
        self.assertIn("missing_training_seeds", csv_rows[0])
        self.assertIn("per_seed_episode_counts", csv_rows[0])

    def test_descriptive_metrics_weight_seed_means_and_preserve_missing(self):
        rows = [
            self._row(1, 0, 0, 1),
            self._row(1, 1, 0, 1),
            self._row(2, 0, 0, 1),
            self._row(3, 0, 0, 0),
        ]
        rows[0]["coverage"] = 0.0
        rows[1]["coverage"] = 2.0
        rows[2]["coverage"] = 10.0
        rows[3]["coverage"] = None
        summaries = summarize_training_seeds(rows)
        descriptive = aggregate_descriptive_seed_metrics(summaries)
        coverage = next(row for row in descriptive if row["metric"] == "coverage")

        self.assertEqual(coverage["per_seed_values"], [1.0, 10.0, None])
        self.assertEqual(coverage["per_seed_episode_counts"], [2, 1, 1])
        self.assertEqual(coverage["mean"], 5.5)
        self.assertNotEqual(coverage["mean"], (0.0 + 2.0 + 10.0) / 3.0)
        self.assertAlmostEqual(coverage["sample_stddev"], 9.0 / math.sqrt(2.0))
        self.assertEqual(coverage["degrees_of_freedom"], 1)
        self.assertEqual(coverage["valid_training_seed_count"], 2)
        self.assertEqual(coverage["missing_training_seed_count"], 1)
        self.assertEqual(coverage["valid_training_seeds"], [1, 2])
        self.assertEqual(coverage["missing_training_seeds"], [3])

        _per_seed, canonical = canonical_aggregation(rows)
        artifact = build_generic_cross_seed_artifact(
            canonical, descriptive, rows
        )
        fov = next(
            row
            for row in artifact
            if (row["metric"], row.get("task_type"))
            == ("violation_probability", "FOV")
        )
        self.assertEqual(fov["per_seed_values"], [0.0, 0.0, None])
        self.assertEqual(fov["valid_training_seed_count"], 2)
        self.assertEqual(fov["missing_training_seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
