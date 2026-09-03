import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from DDQN import DDQN
from experiment_config import MethodSpec
from HRL_task_aware import _run_routing_slot
from Packet_scheduler_v1 import PacketEngine
from paper_evaluation import _write_routing_q_score_outputs
from rng_contract import NamedRNGStreams
from routing_q_score_diagnostics import (
    ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION,
    VOLUNTARY_WAIT_EVENT_FIELDS,
    RoutingQScoreDiagnosticAccumulator,
    write_routing_q_score_diagnostic_artifacts,
)
from Simulator import Simulator
from utils_update_v2 import ReplayBufferDiscrete


class FixedNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values", torch.as_tensor(values, dtype=torch.float32).reshape(1, -1)
        )

    def forward(self, state):
        return self.values.expand(state.shape[0], -1)


def fixed_agent(q_r, q_c, lambda_cost):
    agent = DDQN(
        state_dim=2,
        action_dim=len(q_r),
        lambda_cost=float(lambda_cost),
        master_seed=20260901,
    )
    agent.q_network = FixedNetwork(q_r)
    agent.cost_network = FixedNetwork(q_c)
    return agent


def inspect_and_add(
    q_r,
    q_c,
    lambda_cost,
    mask,
    *,
    sender=0,
    task_type="FOV",
):
    agent = fixed_agent(q_r, q_c, lambda_cost)
    state = np.asarray([0.25, -0.5], dtype=np.float32)
    selected = agent.select_action(
        state,
        sender,
        mask=mask,
        epsilon=0.0,
        logits_noise_std=0.0,
    )
    inspection = agent.inspect_action_scores(state, mask)
    accumulator = RoutingQScoreDiagnosticAccumulator()
    event = accumulator.add_decision(
        inspection,
        selected_action=selected,
        sender_uav_id=sender,
        hol_task_type=task_type,
        scenario_id="scenario-q",
        episode_index=3,
        slot_index=7,
        time_seconds=1.75,
    )
    return selected, inspection, accumulator, event


class RoutingQScoreDecisionTest(unittest.TestCase):
    def test_reward_and_safe_both_select_forward(self):
        selected, inspection, accumulator, event = inspect_and_add(
            [0.0, 2.0, 100.0],
            [0.0, 0.1, -100.0],
            1.0,
            [True, True, False],
        )

        self.assertEqual(selected, 1)
        self.assertEqual(inspection["reward_argmax_action"], 1)
        self.assertEqual(inspection["safe_argmax_action"], 1)
        self.assertIsNone(event)
        all_group = accumulator.summary()["groups"]["ALL"]
        self.assertEqual(all_group["selected_forward_count"], 1)
        self.assertEqual(
            all_group["reward_argmax_forward_but_safe_argmax_wait_count"], 0
        )

    def test_reward_and_safe_both_select_wait(self):
        selected, _, accumulator, event = inspect_and_add(
            [2.0, 1.0, 100.0],
            [0.0, 0.0, -100.0],
            1.0,
            [True, True, False],
        )

        self.assertEqual(selected, 0)
        self.assertTrue(event["reward_also_prefers_wait"])
        self.assertFalse(event["cost_induced_forward_to_wait_flip"])
        self.assertEqual(
            accumulator.summary()["groups"]["ALL"][
                "voluntary_wait_reward_also_prefers_wait_count"
            ],
            1,
        )

    def test_negative_qc_causes_forward_to_wait_flip(self):
        selected, inspection, accumulator, event = inspect_and_add(
            [-0.8, 0.5],
            [-0.10, 0.02],
            15.0,
            [True, True],
        )

        self.assertEqual(selected, 0)
        self.assertEqual(inspection["reward_argmax_action"], 1)
        self.assertEqual(inspection["safe_argmax_action"], 0)
        self.assertAlmostEqual(event["q_safe_wait"], 0.7, places=6)
        self.assertAlmostEqual(event["q_safe_best_safe_forward"], 0.2, places=6)
        self.assertTrue(event["cost_induced_forward_to_wait_flip"])
        self.assertTrue(event["q_c_wait_is_negative"])
        group = accumulator.summary()["groups"]["ALL"]
        self.assertEqual(group["voluntary_wait_cost_induced_flip_count"], 1)
        self.assertEqual(
            group["cost_induced_flip_and_qc_wait_negative_count"], 1
        )

    def test_positive_qc_can_also_cause_flip(self):
        selected, _, accumulator, event = inspect_and_add(
            [0.0, 1.0],
            [0.1, 1.0],
            2.0,
            [True, True],
        )

        self.assertEqual(selected, 0)
        self.assertTrue(event["cost_induced_forward_to_wait_flip"])
        self.assertFalse(event["q_c_wait_is_negative"])
        group = accumulator.summary()["groups"]["ALL"]
        self.assertEqual(group["voluntary_wait_cost_induced_flip_count"], 1)
        self.assertEqual(
            group["cost_induced_flip_and_qc_wait_negative_count"], 0
        )

    def test_illegal_high_score_never_wins_any_argmax_or_best_forward(self):
        selected, inspection, _, event = inspect_and_add(
            [0.0, 2.0, 1_000.0],
            [0.0, 0.0, -1_000.0],
            10.0,
            [True, True, False],
        )

        self.assertEqual(selected, 1)
        self.assertEqual(inspection["reward_argmax_action"], 1)
        self.assertEqual(inspection["safe_argmax_action"], 1)
        self.assertIsNone(event)

        _, _, _, wait_event = inspect_and_add(
            [2.0, 1.0, 1_000.0],
            [0.0, 0.0, -1_000.0],
            10.0,
            [True, True, False],
        )
        self.assertEqual(wait_event["best_qr_forward_action"], 1)
        self.assertEqual(wait_event["best_safe_forward_action"], 1)

    def test_only_wait_mask_is_not_a_voluntary_wait_event(self):
        selected, _, accumulator, event = inspect_and_add(
            [2.0, 100.0],
            [0.0, -100.0],
            15.0,
            [True, False],
        )

        self.assertEqual(selected, 0)
        self.assertIsNone(event)
        self.assertEqual(accumulator.voluntary_wait_events, [])
        self.assertEqual(
            accumulator.summary()["groups"]["ALL"]["voluntary_wait_count"], 0
        )

    def test_diagnostic_safe_argmax_must_match_production_action(self):
        agent = fixed_agent([0.0, 2.0], [0.0, 0.0], 1.0)
        state = np.zeros(2, dtype=np.float32)
        inspection = agent.inspect_action_scores(state, [True, True])
        self.assertEqual(
            inspection["safe_argmax_action"],
            agent.select_action(
                state,
                0,
                mask=[True, True],
                epsilon=0.0,
                logits_noise_std=0.0,
            ),
        )
        with self.assertRaisesRegex(AssertionError, "production action"):
            RoutingQScoreDiagnosticAccumulator().add_decision(
                inspection,
                selected_action=0,
                sender_uav_id=0,
                hol_task_type="FOV",
                scenario_id="scenario-q",
                episode_index=0,
                slot_index=0,
                time_seconds=0.0,
            )

    def test_legal_nonfinite_q_value_fails_fast(self):
        agent = fixed_agent([0.0, np.nan], [0.0, 0.0], 1.0)
        with self.assertRaisesRegex(ValueError, "Q_r.*finite"):
            agent.inspect_action_scores(np.zeros(2), [True, True])

    def test_native_float32_safe_consistency_avoids_float64_false_positive(self):
        lambda_cost = 15.0595
        q_c = np.asarray([1000.123, 2000.456], dtype=np.float32)
        q_r = np.asarray(lambda_cost * q_c, dtype=np.float32)
        agent = fixed_agent(q_r, q_c, lambda_cost)
        state = np.zeros(2, dtype=np.float32)
        inspection = agent.inspect_action_scores(state, [True, True])

        self.assertEqual(inspection["q_r"].dtype, np.dtype(np.float32))
        self.assertEqual(inspection["q_c"].dtype, np.dtype(np.float32))
        self.assertEqual(inspection["q_safe"].dtype, np.dtype(np.float32))
        float64_recomputed = inspection["q_r"].astype(np.float64) - (
            lambda_cost * inspection["q_c"].astype(np.float64)
        )
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        inspection["q_safe"].astype(np.float64)
                        - float64_recomputed
                    )
                )
            ),
            1e-6,
        )
        self.assertFalse(
            np.allclose(
                inspection["q_safe"].astype(np.float64),
                float64_recomputed,
                rtol=1e-6,
                atol=1e-6,
            )
        )

        accumulator = RoutingQScoreDiagnosticAccumulator()
        event = accumulator.add_decision(
            inspection,
            selected_action=inspection["safe_argmax_action"],
            sender_uav_id=0,
            hol_task_type="FOV",
            scenario_id="float32-regression",
            episode_index=0,
            slot_index=0,
            time_seconds=0.0,
        )
        self.assertIsNotNone(event)

        corrupted = copy.deepcopy(inspection)
        corrupted["q_safe"][0] += np.float32(1.0)
        with self.assertRaisesRegex(
            AssertionError,
            "routing Q_safe does not equal Q_r - lambda_cost \\* Q_c",
        ):
            RoutingQScoreDiagnosticAccumulator().add_decision(
                corrupted,
                selected_action=inspection["safe_argmax_action"],
                sender_uav_id=0,
                hol_task_type="FOV",
                scenario_id="float32-corruption",
                episode_index=0,
                slot_index=0,
                time_seconds=0.0,
            )


class RoutingQScoreArtifactTest(unittest.TestCase):
    def test_group_statistics_and_three_artifacts(self):
        _, _, accumulator, event = inspect_and_add(
            [-0.8, 0.5],
            [-0.10, 0.02],
            15.0,
            [True, True],
            task_type="COM",
        )
        diagnostics = accumulator.summary()
        all_group = diagnostics["groups"]["ALL"]

        self.assertEqual(
            diagnostics["routing_q_score_diagnostic_contract_version"],
            ROUTING_Q_SCORE_DIAGNOSTIC_CONTRACT_VERSION,
        )
        self.assertEqual(set(diagnostics["groups"]), {"ALL", "COM", "FOV"})
        self.assertEqual(all_group["voluntary_wait_qc_wait_negative_fraction"], 1.0)
        self.assertAlmostEqual(all_group["q_c_wait_mean"], -0.1, places=6)
        self.assertAlmostEqual(all_group["q_c_wait_median"], -0.1, places=6)
        self.assertEqual(tuple(event), VOLUNTARY_WAIT_EVENT_FIELDS)

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_routing_q_score_diagnostic_artifacts(
                temp_dir, diagnostics, accumulator.voluntary_wait_events
            )
            loaded = json.loads(
                Path(outputs["routing_q_score_diagnostics_json"]).read_text(
                    encoding="utf-8"
                )
            )
            with Path(outputs["routing_q_score_diagnostics_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                aggregate_rows = list(csv.DictReader(handle))
            with Path(outputs["routing_q_score_voluntary_waits_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                event_rows = list(csv.DictReader(handle))

        self.assertEqual(loaded, diagnostics)
        self.assertEqual(
            [row["task_type"] for row in aggregate_rows], ["ALL", "COM", "FOV"]
        )
        self.assertEqual(len(event_rows), 1)
        self.assertEqual(tuple(event_rows[0]), VOLUNTARY_WAIT_EVENT_FIELDS)

    def test_paper_output_hook_writes_safe_ddqn_only(self):
        _, _, accumulator, _ = inspect_and_add(
            [2.0, 1.0], [0.0, 0.0], 1.0, [True, True]
        )
        result = {
            "routing_q_score_diagnostics": accumulator.summary(),
            "routing_q_score_voluntary_waits": accumulator.voluntary_wait_events,
        }
        safe_method = MethodSpec.parse("td3_dinkelbach")
        random_method = MethodSpec.parse("kkm_random_action_random_routing")

        with tempfile.TemporaryDirectory() as temp_dir:
            safe_dir = Path(temp_dir) / "safe"
            random_dir = Path(temp_dir) / "random"
            safe_dir.mkdir()
            random_dir.mkdir()
            safe_outputs = _write_routing_q_score_outputs(
                safe_dir, safe_method, result
            )
            random_outputs = _write_routing_q_score_outputs(
                random_dir, random_method, result
            )

            self.assertEqual(len(safe_outputs), 3)
            self.assertTrue(
                (safe_dir / "routing_q_score_diagnostics.json").is_file()
            )
            self.assertTrue(
                (safe_dir / "routing_q_score_diagnostics.csv").is_file()
            )
            self.assertTrue(
                (safe_dir / "routing_q_score_voluntary_waits.csv").is_file()
            )
            self.assertEqual(random_outputs, {})
            self.assertEqual(list(random_dir.iterdir()), [])


class RoutingQScoreNoBehaviorChangeTest(unittest.TestCase):
    @staticmethod
    def _run(*, diagnostics_enabled):
        agent = fixed_agent([-0.8, 0.5, -2.0], [-0.10, 0.02, 0.0], 15.0)
        states = (
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.1, -0.2], dtype=np.float32),
            np.asarray([-0.5, 0.25], dtype=np.float32),
        )
        mask = np.asarray([True, True, False])
        accumulator = (
            RoutingQScoreDiagnosticAccumulator() if diagnostics_enabled else None
        )
        actions = []
        replay = []
        for slot, state in enumerate(states):
            selected = agent.select_action(
                state,
                0,
                mask=mask,
                epsilon=0.0,
                logits_noise_std=0.0,
            )
            actions.append(selected)
            if accumulator is not None:
                accumulator.add_decision(
                    agent.inspect_action_scores(state, mask),
                    selected_action=selected,
                    sender_uav_id=0,
                    hol_task_type="FOV",
                    scenario_id="deterministic",
                    episode_index=0,
                    slot_index=slot,
                    time_seconds=slot * 0.25,
                )

        engine = PacketEngine(num_uav=2, step_time=0.25)
        packet = engine.create_packet(0, "FOV", 100.0, 0.0)
        packet["deadline_abs"] = 0.75
        env = SimpleNamespace(
            GS_ID=2,
            GS_pos=(0.0, 0.0, 0.0),
            uav_dict={
                0: SimpleNamespace(get_position=lambda: (100.0, 0.0, 0.0)),
                1: SimpleNamespace(get_position=lambda: (50.0, 0.0, 0.0)),
            },
        )
        engine.serve_active_links(
            env,
            actions={0: actions[0]},
            capacities={},
            current_time=0.0,
        )
        engine.expire_packets(0.75)
        terminal = engine.packet_outcomes[0]
        return {
            "actions": actions,
            "rng_state": copy.deepcopy(agent.exploration_rng.bit_generator.state),
            "replay": replay,
            "packet_outcome": {
                key: terminal[key]
                for key in (
                    "packet_id",
                    "task_type",
                    "outcome",
                    "finish_time_seconds",
                    "remaining_bits_at_drop",
                )
            },
        }

    def test_diagnostics_on_off_preserve_actions_rng_replay_and_packets(self):
        self.assertEqual(
            self._run(diagnostics_enabled=True),
            self._run(diagnostics_enabled=False),
        )

    @staticmethod
    def _run_formal_slot(*, diagnostics_enabled):
        env = Simulator(num_UAV=10, rng_streams=NamedRNGStreams(20260901))
        env.num_GT = 2
        env.reset_environment()
        env.source_uavs = set()
        env.current_time = 0.0
        env.get_routing_action_mask = lambda _uid: np.ones(11, dtype=bool)
        env.update_u2u_channels()
        env.update_u2g_channels()

        engine = PacketEngine(num_uav=10, step_time=0.25)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)
        packet["deadline_abs"] = 0.25
        replay = ReplayBufferDiscrete(101, 11, max_size=16, n_step=1)
        q_r = np.asarray([2.0, 1.0, *([0.0] * 9)])
        agent = fixed_agent(q_r, np.zeros(11), lambda_cost=15.0)
        accumulator = (
            RoutingQScoreDiagnosticAccumulator() if diagnostics_enabled else None
        )
        rng_before = copy.deepcopy(agent.exploration_rng.bit_generator.state)
        slot_result = _run_routing_slot(
            env,
            engine,
            agent,
            replay,
            None,
            current_time=0.0,
            done=True,
            delay_bound_steps=20,
            violation_stats={
                "COM": {
                    "timely_delivered_packets": 0,
                    "deadline_violated_packets": 0,
                },
                "FOV": {
                    "timely_delivered_packets": 0,
                    "deadline_violated_packets": 0,
                },
            },
            epsilon=0.0,
            traffic_rate_overrides={"FOV": 0.0, "COM": 0.0},
            routing_q_score_accumulator=accumulator,
            routing_q_score_context={
                "scenario_id": "formal-slot-smoke",
                "episode_index": 0,
                "slot_index": 0,
            }
            if diagnostics_enabled
            else None,
        )
        if accumulator is not None:
            event = accumulator.voluntary_wait_events[0]
            if event["hol_task_type"] != "COM":
                raise AssertionError("diagnostic did not use frozen HOL task type")
            if accumulator.summary()["groups"]["ALL"][
                "total_routing_q_decisions"
            ] != 1:
                raise AssertionError("formal slot diagnostic decision was not recorded")
        rng_after = copy.deepcopy(agent.exploration_rng.bit_generator.state)
        if rng_after != rng_before:
            raise AssertionError("routing Q-score diagnostics consumed agent RNG")
        return {
            "slot_result": slot_result,
            "routing_actions": replay.action[: replay.size].tolist(),
            "replay_state": replay.state[: replay.size].tolist(),
            "replay_next_state": replay.next_state[: replay.size].tolist(),
            "replay_reward": replay.reward[: replay.size].tolist(),
            "replay_cost": replay.cost[: replay.size].tolist(),
            "replay_not_done": replay.not_done[: replay.size].tolist(),
            "packet_outcomes": copy.deepcopy(engine.packet_outcomes),
            "rng_state": rng_after,
        }

    def test_formal_slot_wiring_is_read_only_and_records_frozen_hol(self):
        self.assertEqual(
            self._run_formal_slot(diagnostics_enabled=True),
            self._run_formal_slot(diagnostics_enabled=False),
        )


if __name__ == "__main__":
    unittest.main()
