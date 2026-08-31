import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Packet_scheduler_v1 import BlockServiceCursor, PacketEngine
from paper_evaluation import run_paper_evaluation
from packet_outcome_artifacts import (
    PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
    PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
    packet_outcome_episode_record,
    packet_routing_diagnostics_from_outcomes,
    write_packet_routing_diagnostic_artifacts,
)


def routing_env(num_uav=3, *, routing_link_in_range=True):
    return SimpleNamespace(
        GS_ID=num_uav,
        is_routing_link_in_range=mock.Mock(
            return_value=bool(routing_link_in_range)
        ),
    )


def action_mask(num_uav, *enabled):
    mask = np.zeros(num_uav + 1, dtype=bool)
    mask[list(enabled)] = True
    return mask


def diagnostic_engine(num_uav=3):
    return PacketEngine(
        num_uav=num_uav,
        step_time=0.25,
        enable_packet_diagnostic_artifacts=True,
    )


class PacketActualServiceTimeTest(unittest.TestCase):
    def test_cursor_actual_airtime_excludes_zero_capacity_block_gap(self):
        profile = np.zeros(50, dtype=float)
        profile[10] = 1.0
        cursor = BlockServiceCursor(
            profile,
            slot_start_time=0.0,
            track_actual_service_time=True,
        )

        consumed, completion_time = cursor.consume(1_000.0)

        self.assertAlmostEqual(consumed, 1_000.0)
        self.assertAlmostEqual(cursor.last_consumed_service_seconds, 0.001)
        self.assertAlmostEqual(completion_time, 0.051)


class PacketRoutingDecisionDiagnosticTest(unittest.TestCase):
    def test_normal_direct_delivery_has_no_loop_or_wait(self):
        env = routing_env()
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        physical = action_mask(3, 0, env.GS_ID)

        engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.0,
            start_of_slot_physical_masks_by_sender={0: physical},
            start_of_slot_effective_masks_by_sender={0: physical},
        )

        outcome = engine.packet_outcomes[0]
        self.assertEqual(outcome["terminal_node_type"], "GS")
        self.assertEqual(outcome["terminal_node_id"], env.GS_ID)
        self.assertIsNone(outcome["terminal_uav_id"])
        self.assertFalse(outcome["has_repeated_uav"])
        self.assertEqual(outcome["routing_decision_slot_count"], 1)
        self.assertEqual(outcome["routing_wait_slot_count"], 0)
        self.assertEqual(
            outcome["locked_receiver_out_of_range_wait_slot_count"], 0
        )
        self.assertEqual(outcome["completed_uav_hop_count"], 1)
        self.assertEqual(outcome["per_hop"][0]["link_type"], "U2G")

    def test_voluntary_wait_counts_only_the_frozen_hol_packet(self):
        env = routing_env()
        engine = diagnostic_engine()
        hol = engine.create_packet(0, "FOV", 100.0, 0.0)
        queued = engine.create_packet(0, "FOV", 100.0, 0.0)
        physical = action_mask(3, 0, 1, env.GS_ID)

        engine.serve_active_links(
            env,
            actions={0: 0},
            capacities={},
            current_time=0.0,
            start_of_slot_physical_masks_by_sender={0: physical},
            start_of_slot_effective_masks_by_sender={0: physical},
        )

        self.assertEqual(hol["routing_decision_slot_count"], 1)
        self.assertEqual(hol["routing_wait_slot_count"], 1)
        self.assertAlmostEqual(hol["routing_wait_seconds"], 0.25)
        self.assertEqual(
            hol["locked_receiver_out_of_range_wait_slot_count"], 0
        )
        self.assertEqual(queued["routing_decision_slot_count"], 0)
        self.assertEqual(queued["routing_wait_slot_count"], 0)
        self.assertEqual(engine.wait_actions, 1)

    def test_out_of_range_locked_receiver_is_a_forced_wait(self):
        env = routing_env(routing_link_in_range=False)
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        self.assertFalse(engine.record_hop_transmission(packet, 0, 1, 40.0))
        physical = action_mask(3, 0)
        effective = action_mask(3, 0)

        engine.serve_active_links(
            env,
            actions={0: 0},
            capacities={},
            current_time=0.25,
            start_of_slot_physical_masks_by_sender={0: physical},
            start_of_slot_effective_masks_by_sender={0: effective},
        )

        self.assertEqual(packet["routing_wait_slot_count"], 1)
        self.assertEqual(
            packet["locked_receiver_out_of_range_wait_slot_count"], 1
        )
        self.assertAlmostEqual(
            packet["locked_receiver_out_of_range_wait_seconds"], 0.25
        )
        env.is_routing_link_in_range.assert_called_once_with(0, 1)

    def test_in_range_but_capacity_unavailable_lock_is_not_out_of_range(self):
        env = routing_env(routing_link_in_range=True)
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        self.assertFalse(engine.record_hop_transmission(packet, 0, 1, 40.0))
        physical = action_mask(3, 0)
        effective = action_mask(3, 0)

        engine.serve_active_links(
            env,
            actions={0: 0},
            capacities={},
            current_time=0.25,
            start_of_slot_physical_masks_by_sender={0: physical},
            start_of_slot_effective_masks_by_sender={0: effective},
        )

        self.assertEqual(packet["routing_wait_slot_count"], 1)
        self.assertEqual(
            packet["locked_receiver_out_of_range_wait_slot_count"], 0
        )
        env.is_routing_link_in_range.assert_called_once_with(0, 1)


class PacketTerminalDiagnosticTest(unittest.TestCase):
    def test_repeated_uav_path_excludes_sr_and_gs(self):
        engine = diagnostic_engine()
        packet = engine.create_packet(1, "FOV", 100.0, 0.0)
        packet["path"] = [1, 2, 1]
        packet["current"] = 1
        packet["deadline_abs"] = 0.5

        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        self.assertTrue(outcome["has_repeated_uav"])
        self.assertEqual(outcome["repeated_uav_ids"], [1])
        self.assertEqual(outcome["repeated_uav_count"], 1)
        self.assertEqual(outcome["unique_uav_count"], 2)
        self.assertEqual(outcome["terminal_node_type"], "UAV")
        self.assertEqual(outcome["terminal_uav_id"], 1)

    def test_sr_uav_and_gs_terminal_locations_and_s2u_classification(self):
        pre_engine = diagnostic_engine()
        pre = pre_engine.create_sr_packet(7, 100.0, 0.0)
        pre["deadline_abs"] = 0.5
        pre_engine.expire_packets(0.5)
        pre_outcome = pre_engine.packet_outcomes[0]
        self.assertEqual(pre_outcome["terminal_node_type"], "SR")
        self.assertEqual(pre_outcome["terminal_node_id"], 7)
        self.assertIsNone(pre_outcome["terminal_uav_id"])
        self.assertFalse(pre_outcome["s2u_completed"])

        post_engine = diagnostic_engine()
        post = post_engine.create_sr_packet(8, 100.0, 0.0)
        self.assertTrue(post_engine._remove_from_sr_queue(post))
        post["s2u_completion_time"] = 0.1
        post["routing_eligible"] = True
        post["routing_eligible_time"] = 0.25
        post["path"].append(2)
        post["rem_bits"] = post["size_bits"]
        post["deadline_abs"] = 0.5
        post_engine.enqueue_packet(post, 2, 0.25)
        post_engine.expire_packets(0.5)
        post_outcome = post_engine.packet_outcomes[0]
        self.assertEqual(post_outcome["terminal_node_type"], "UAV")
        self.assertEqual(post_outcome["terminal_uav_id"], 2)
        self.assertTrue(post_outcome["s2u_completed"])
        self.assertEqual(post_outcome["s2u_completion_time_seconds"], 0.1)

        delivered_engine = diagnostic_engine()
        env = routing_env()
        delivered_engine.create_packet(0, "FOV", 100.0, 0.0)
        physical = action_mask(3, 0, env.GS_ID)
        delivered_engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.0,
            start_of_slot_physical_masks_by_sender={0: physical},
            start_of_slot_effective_masks_by_sender={0: physical},
        )
        delivered_outcome = delivered_engine.packet_outcomes[0]
        self.assertEqual(delivered_outcome["terminal_node_type"], "GS")
        self.assertIsNone(delivered_outcome["s2u_completed"])
        self.assertIsNone(delivered_outcome["s2u_completion_time_seconds"])
        self.assertIsNone(delivered_outcome["s2u_queue_delay_seconds"])
        self.assertIsNone(delivered_outcome["s2u_tx_delay_seconds"])

    def test_completed_s2u_at_deadline_uses_path_receiver_as_terminal_uav(self):
        engine = diagnostic_engine()
        packet = engine.create_sr_packet(9, 100.0, 0.0)
        packet["s2u_completion_time"] = 0.5
        packet["path"].append(2)
        packet["deadline_abs"] = 0.5

        engine._mark_deadline_violation(
            packet,
            current_time=0.5,
            remove_from_queue=False,
        )

        outcome = engine.packet_outcomes[0]
        self.assertTrue(outcome["s2u_completed"])
        self.assertEqual(outcome["terminal_node_type"], "UAV")
        self.assertEqual(outcome["terminal_uav_id"], 2)

    def test_partial_hop_expiry_remains_at_sender_and_retains_lock(self):
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        self.assertFalse(engine.record_hop_transmission(packet, 0, 1, 40.0))
        packet["deadline_abs"] = 0.5

        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        self.assertEqual(outcome["terminal_node_type"], "UAV")
        self.assertEqual(outcome["terminal_uav_id"], 0)
        self.assertEqual(outcome["locked_hop_receiver_at_terminal"], 1)

    def test_uav_queue_expiry_before_service_includes_terminal_wait(self):
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        packet["deadline_abs"] = 0.5

        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        self.assertAlmostEqual(
            outcome["cumulative_uav_queue_delay_seconds"], 0.5
        )
        self.assertAlmostEqual(outcome["cumulative_uav_tx_delay_seconds"], 0.0)

    def test_partial_uav_transmission_records_actual_airtime_before_expiry(self):
        env = routing_env()
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 1_000_000.0, 0.0)
        packet["deadline_abs"] = 0.5
        mask = action_mask(3, 0, 1)

        engine.serve_active_links(
            env,
            actions={0: 1},
            capacities={(0, 1): 1.0},
            current_time=0.0,
            start_of_slot_physical_masks_by_sender={0: mask},
            start_of_slot_effective_masks_by_sender={0: mask},
        )
        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        self.assertEqual(outcome["completed_uav_hop_count"], 0)
        self.assertAlmostEqual(outcome["cumulative_uav_tx_delay_seconds"], 0.25)
        self.assertAlmostEqual(
            outcome["cumulative_uav_queue_delay_seconds"], 0.0
        )

    def test_completed_hop_plus_current_queue_is_added_without_double_count(self):
        env = routing_env()
        engine = diagnostic_engine()
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        packet["deadline_abs"] = 0.75
        mask = action_mask(3, 0, 1)

        engine.serve_active_links(
            env,
            actions={0: 1},
            capacities={(0, 1): 0.0004},
            current_time=0.2,
            start_of_slot_physical_masks_by_sender={0: mask},
            start_of_slot_effective_masks_by_sender={0: mask},
        )
        engine.expire_packets(0.75)

        outcome = engine.packet_outcomes[0]
        self.assertEqual(outcome["completed_uav_hop_count"], 1)
        self.assertAlmostEqual(outcome["per_hop"][0]["queue_s"], 0.2)
        self.assertAlmostEqual(outcome["per_hop"][0]["actual_tx_s"], 0.25)
        self.assertAlmostEqual(
            outcome["cumulative_uav_queue_delay_seconds"], 0.5
        )
        self.assertAlmostEqual(outcome["cumulative_uav_tx_delay_seconds"], 0.25)

    def test_pre_s2u_queue_expiry_reports_elapsed_wait_and_zero_airtime(self):
        engine = diagnostic_engine()
        packet = engine.create_sr_packet(4, 100.0, 0.0)
        packet["deadline_abs"] = 0.5

        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        self.assertFalse(outcome["s2u_completed"])
        self.assertAlmostEqual(outcome["s2u_queue_delay_seconds"], 0.5)
        self.assertAlmostEqual(outcome["s2u_tx_delay_seconds"], 0.0)

    def test_partial_s2u_transmission_reports_queue_and_actual_airtime(self):
        env = routing_env()
        engine = diagnostic_engine()
        packet = engine.create_sr_packet(5, 1_000_000.0, 0.0)
        packet["s2u_receiver"] = 0
        packet["deadline_abs"] = 0.75

        engine.serve_s2u_links(
            env,
            capacities={(5, 0): 1.0},
            current_time=0.2,
        )
        engine.expire_packets(0.75)

        outcome = engine.packet_outcomes[0]
        self.assertFalse(outcome["s2u_completed"])
        self.assertAlmostEqual(outcome["s2u_queue_delay_seconds"], 0.2)
        self.assertAlmostEqual(outcome["s2u_tx_delay_seconds"], 0.25)

    def test_disabled_diagnostics_do_not_expand_terminal_outcome(self):
        engine = PacketEngine(num_uav=3, step_time=0.25)
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        packet["deadline_abs"] = 0.5

        self.assertNotIn("routing_decision_slot_count", packet)
        engine.expire_packets(0.5)

        self.assertNotIn("path_hop_count", engine.packet_outcomes[0])
        self.assertFalse(engine.enable_packet_diagnostic_artifacts)


class PacketRoutingAggregateTest(unittest.TestCase):
    @staticmethod
    def _violation_outcomes():
        pre_engine = diagnostic_engine()
        pre = pre_engine.create_sr_packet(0, 100.0, 0.0)
        pre["deadline_abs"] = 0.5
        pre_engine.expire_packets(0.5)

        post_engine = diagnostic_engine()
        post = post_engine.create_sr_packet(1, 100.0, 0.0)
        post_engine._remove_from_sr_queue(post)
        post["s2u_completion_time"] = 0.1
        post["routing_eligible"] = True
        post["routing_eligible_time"] = 0.25
        post["path"].append(2)
        post["rem_bits"] = post["size_bits"]
        post["deadline_abs"] = 0.5
        post_engine.enqueue_packet(post, 2, 0.25)
        post_engine.expire_packets(0.5)

        fov_engine = diagnostic_engine()
        fov = fov_engine.create_packet(1, "FOV", 100.0, 0.0)
        fov["path"] = [1, 2, 1]
        fov["current"] = 1
        fov["deadline_abs"] = 0.5
        fov["routing_decision_slot_count"] = 2
        fov["routing_wait_slot_count"] = 1
        fov["routing_wait_seconds"] = 0.25
        fov["locked_receiver_out_of_range_wait_slot_count"] = 1
        fov["locked_receiver_out_of_range_wait_seconds"] = 0.25
        fov_engine.expire_packets(0.5)
        return (
            pre_engine.packet_outcomes
            + post_engine.packet_outcomes
            + fov_engine.packet_outcomes
        )

    def test_schema_v4_and_grouped_diagnostic_summary(self):
        outcomes = self._violation_outcomes()
        record = packet_outcome_episode_record(
            "scenario-diagnostics",
            {"COM": {"eligible_packets": 2}},
            outcomes,
        )
        self.assertEqual(
            record["artifact_schema_version"],
            "uav-hrl-packet-outcomes-jsonl-v5",
        )
        self.assertEqual(
            record["artifact_schema_version"],
            PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
        )

        diagnostics = packet_routing_diagnostics_from_outcomes(outcomes)
        self.assertEqual(
            diagnostics["packet_routing_diagnostic_contract_version"],
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
        )
        com = diagnostics["groups"]["COM"]
        fov = diagnostics["groups"]["FOV"]
        all_packets = diagnostics["groups"]["ALL"]
        self.assertEqual(com["pre_s2u_violation_count"], 1)
        self.assertEqual(com["post_s2u_violation_count"], 1)
        self.assertEqual(com["expired_at_sr_count"], 1)
        self.assertEqual(com["expired_packet_count_by_terminal_uav"], {"2": 1})
        self.assertIsNone(fov["pre_s2u_violation_count"])
        self.assertIsNone(fov["post_s2u_violation_count"])
        self.assertEqual(fov["loop_violation_count"], 1)
        self.assertEqual(fov["violations_with_forced_locked_wait"], 1)
        self.assertEqual(all_packets["eligible_packets"], 3)
        self.assertEqual(all_packets["violated_packets"], 3)

    def test_loop_mean_hops_excludes_s2u_and_terminal_partial_hop(self):
        engine = diagnostic_engine()
        packet = engine.create_sr_packet(0, 100.0, 0.0)
        self.assertTrue(engine._remove_from_sr_queue(packet))
        packet.update(
            {
                "s2u_completion_time": 0.1,
                "routing_eligible": True,
                "routing_eligible_time": 0.25,
                "path": ["SR:0", 1, 2, 1],
                "current": 1,
                "hops": 2,
                "deadline_abs": 0.5,
                "rem_bits": packet["size_bits"],
                "s2u_actual_tx_seconds": 0.05,
                "cumulative_actual_uav_tx_seconds": 0.2,
                "per_hop": [
                    {
                        "from": "SR:0",
                        "to": 1,
                        "queue_s": 0.05,
                        "tx_s": 0.05,
                        "actual_tx_s": 0.05,
                        "link_type": "S2U",
                    },
                    {
                        "from": 1,
                        "to": 2,
                        "queue_s": 0.1,
                        "tx_s": 0.1,
                        "actual_tx_s": 0.1,
                    },
                    {
                        "from": 2,
                        "to": 1,
                        "queue_s": 0.1,
                        "tx_s": 0.1,
                        "actual_tx_s": 0.1,
                    },
                ],
            }
        )
        engine.enqueue_packet(packet, 1, 0.4)
        engine.expire_packets(0.5)

        outcome = engine.packet_outcomes[0]
        diagnostics = packet_routing_diagnostics_from_outcomes([outcome])
        self.assertEqual(outcome["path_hop_count"], 3)
        self.assertEqual(outcome["completed_uav_hop_count"], 2)
        self.assertEqual(
            diagnostics["groups"]["COM"]["mean_hops_loop_packets"], 2.0
        )

    def test_writes_json_csv_and_terminal_distribution(self):
        diagnostics = packet_routing_diagnostics_from_outcomes(
            self._violation_outcomes()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_packet_routing_diagnostic_artifacts(
                temp_dir, diagnostics
            )
            loaded = json.loads(
                Path(outputs["packet_routing_diagnostics_json"]).read_text(
                    encoding="utf-8"
                )
            )
            with Path(outputs["packet_routing_diagnostics_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            with Path(outputs["terminal_uav_distribution_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                terminal_rows = list(csv.DictReader(handle))

        self.assertEqual(loaded, diagnostics)
        self.assertEqual([row["task_type"] for row in rows], ["ALL", "COM", "FOV"])
        self.assertTrue(
            any(
                row["task_type"] == "COM" and row["terminal_uav_id"] == "2"
                for row in terminal_rows
            )
        )


class PacketRoutingNoBehaviorChangeTest(unittest.TestCase):
    @staticmethod
    def _run_two_slot_scenario(*, diagnostics_enabled):
        env = routing_env()
        engine = PacketEngine(
            num_uav=3,
            step_time=0.25,
            enable_packet_diagnostic_artifacts=diagnostics_enabled,
        )
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        mask = action_mask(3, 0, env.GS_ID)
        first = engine.serve_active_links(
            env,
            actions={0: 0},
            capacities={},
            current_time=0.0,
            start_of_slot_physical_masks_by_sender={0: mask},
            start_of_slot_effective_masks_by_sender={0: mask},
        )
        second = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): 0.0004},
            current_time=0.25,
            start_of_slot_physical_masks_by_sender={0: mask},
            start_of_slot_effective_masks_by_sender={0: mask},
        )
        core_packet = {
            key: packet.get(key)
            for key in (
                "done",
                "reason",
                "finish_time",
                "current",
                "path",
                "rem_bits",
                "e2e_delay_ms",
            )
        }
        core_engine = {
            "total_delivered": engine.total_delivered,
            "total_violated": engine.total_violated,
            "timely_goodput_bits": engine.timely_goodput_bits,
            "raw_final_hop_bits": engine.raw_final_hop_bits,
            "wait_actions": engine.wait_actions,
            "energy": engine.energy,
            "backlog": dict(engine.backlog_bits),
        }
        core_results = {
            "first_transmitted": dict(first["transmitted_bits_by_link"]),
            "second_transmitted": dict(second["transmitted_bits_by_link"]),
            "second_goodput": second["timely_goodput_bits"],
        }
        terminal = engine.packet_outcomes[0]
        core_terminal = {
            key: terminal[key]
            for key in (
                "packet_id",
                "source_uav_id",
                "source_sr_id",
                "source_kind",
                "task_type",
                "outcome",
                "generation_time_seconds",
                "finish_time_seconds",
                "deadline_seconds",
                "e2e_delay_seconds",
                "size_bits",
                "delivered_to_gs",
                "qos_eligible",
            )
        }
        return core_packet, core_engine, core_results, core_terminal

    def test_diagnostics_do_not_change_service_or_core_terminal_outcomes(self):
        without_diagnostics = self._run_two_slot_scenario(
            diagnostics_enabled=False
        )
        with_diagnostics = self._run_two_slot_scenario(
            diagnostics_enabled=True
        )
        self.assertEqual(with_diagnostics, without_diagnostics)


class PacketRoutingPaperEvaluationIntegrationTest(unittest.TestCase):
    def test_paper_evaluation_automatically_writes_diagnostic_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_directory = Path(temp_dir) / "evaluation"
            result = run_paper_evaluation(
                "kkm_random_action_random_routing",
                suite="fixed_roi",
                manifest_seed=20260901,
                episodes=1,
                episode_seconds=5,
                roi_counts=(2,),
                output_directory=output_directory,
                flatten_single_point=True,
            )
            for filename in (
                "packet_outcomes.jsonl",
                "packet_routing_diagnostics.json",
                "packet_routing_diagnostics.csv",
                "terminal_uav_distribution.csv",
            ):
                self.assertTrue((output_directory / filename).is_file())
            metadata = json.loads(
                (output_directory / "run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            diagnostics = json.loads(
                (output_directory / "packet_routing_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            metadata["packet_routing_diagnostic_contract_version"],
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
        )
        self.assertEqual(
            diagnostics["packet_routing_diagnostic_contract_version"],
            PACKET_ROUTING_DIAGNOSTIC_CONTRACT_VERSION,
        )
        self.assertEqual(
            set(diagnostics["groups"]), {"ALL", "COM", "FOV"}
        )
        self.assertIn(
            "packet_routing_diagnostics_json",
            result["points"][0]["outputs"],
        )


if __name__ == "__main__":
    unittest.main()
