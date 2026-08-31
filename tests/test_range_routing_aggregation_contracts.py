import unittest
from types import SimpleNamespace

import numpy as np

from evaluation_aggregation import canonical_aggregation
from HRL_task_aware import _mark_search_observations
from Packet_scheduler_v1 import PacketEngine
from communication_contract import validate_communication_range_aliases
from experiment_config import (
    A2A_COMMUNICATION_RANGE_M,
    A2G_COMMUNICATION_RANGE_M,
    COMMUNICATION_RANGE_M,
    S2U_COMMUNICATION_RANGE_M,
    U2G_COMMUNICATION_RANGE_M,
    U2U_COMMUNICATION_RANGE_M,
)
from paper_metrics import aggregate_paper_point_metrics
from routing_transition_ledger import RoutingTransitionLedger
from scenario_manifest import generate_manifest
from Simulator import Simulator
from utils_update_v2 import ReplayBufferDiscrete


class HardRangeAndComSessionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = generate_manifest("test", 9401, 1).episodes[0]

    def test_all_communication_aliases_share_one_fail_fast_canonical_value(self):
        aliases = (
            A2G_COMMUNICATION_RANGE_M,
            A2A_COMMUNICATION_RANGE_M,
            S2U_COMMUNICATION_RANGE_M,
            U2G_COMMUNICATION_RANGE_M,
            U2U_COMMUNICATION_RANGE_M,
        )
        self.assertEqual(aliases, (400.0,) * len(aliases))
        self.assertEqual(validate_communication_range_aliases(*aliases), 400.0)
        with self.assertRaisesRegex(RuntimeError, "aliases diverged"):
            validate_communication_range_aliases(400.0, 200.0)

    def _routing_boundary_result(self, link_type, distance):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        for uav_id, uav in env.uav_dict.items():
            uav.x_u = 800.0 + 10.0 * uav_id
            uav.y_u = 900.0
            uav.z_u = 100.0
        if link_type == "U2U":
            env.GS_pos = np.array([1000.0, 1000.0, 0.0])
            env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
                0.0,
                0.0,
                50.0,
            )
            env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (
                distance,
                0.0,
                50.0,
            )
            receiver = 1
        else:
            env.GS_pos = np.array([0.0, 0.0, 0.0])
            env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
                distance,
                0.0,
                0.0,
            )
            receiver = env.GS_ID
        env.update_u2u_channels()
        env.update_u2g_channels()
        engine = PacketEngine(num_uav=10)
        packet = engine.create_packet(0, "COM", 256.0, 0.0)
        physical_mask = env.get_routing_action_mask(0).astype(bool)
        state = engine.get_state_ta(env, 0, action_mask=physical_mask)
        capacity_start = env.num_UAV + 8 + 2 * (env.num_UAV + 1)
        env.prepare_channel_routing_slot(0)
        capacities, bandwidths = env.allocate_active_link_capacities({0: receiver})
        result = engine.serve_active_links(
            env,
            {0: receiver},
            capacities,
            current_time=0.0,
            block_capacity_profiles=env.active_link_capacity_profiles_mbps,
        )
        return {
            "env": env,
            "packet": packet,
            "receiver": receiver,
            "mask": physical_mask,
            "observation_capacity": state[capacity_start + receiver],
            "capacities": capacities,
            "bandwidths": bandwidths,
            "service_bits": result["transmitted_bits_by_link"].get(
                (0, receiver), 0.0
            ),
        }

    def _s2u_boundary_result(self, distance):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        env.SR_teams[0].x, env.SR_teams[0].y, env.SR_teams[0].z = (0.0, 0.0, 0.0)
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
            distance,
            0.0,
            0.0,
        )
        env.multi_tasks = {uav_id: [] for uav_id in range(env.num_UAV)}
        env.multi_tasks[0] = [
            {"task_type": "COM", "target_obj_id": 0, "target_pos": (0.0, 0.0, 0.0)}
        ]
        env.SR_teams[0].assigned_gt_id = 0
        engine = PacketEngine(num_uav=10)
        engine.create_sr_packet(0, 256.0, generation_time=0.0)
        env.prepare_channel_routing_slot(0)
        _, bandwidths = env.allocate_active_link_capacities({}, s2u_links={0: 0})
        result = engine.serve_active_links(
            env,
            {},
            {},
            current_time=0.0,
            s2u_block_capacity_profiles=env.active_s2u_capacity_profiles_mbps,
        )
        return {
            "env": env,
            "bandwidths": bandwidths,
            "service_bits": result["transmitted_bits_by_link"].get(
                ("S2U", 0, 0), 0.0
            ),
        }

    def test_s2u_u2g_u2u_boundaries_align_observation_mask_fdma_service_and_diagnostics(self):
        self.assertEqual(COMMUNICATION_RANGE_M, 400.0)
        for distance, legal in ((399.999, True), (400.0, True), (400.001, False)):
            for link_type in ("U2U", "U2G"):
                with self.subTest(link_type=link_type, distance=distance):
                    result = self._routing_boundary_result(link_type, distance)
                    receiver = result["receiver"]
                    self.assertEqual(bool(result["mask"][receiver]), legal)
                    self.assertEqual(result["observation_capacity"] > 0.0, legal)
                    self.assertEqual((0, receiver) in result["capacities"], legal)
                    self.assertEqual(result["service_bits"] > 0.0, legal)
                    self.assertEqual(
                        bool(result["env"].active_link_diagnostics), legal
                    )
                    if legal:
                        self.assertEqual(result["bandwidths"][(0, receiver)], 10e6)
                    else:
                        self.assertEqual(result["bandwidths"], {})
            with self.subTest(link_type="S2U", distance=distance):
                result = self._s2u_boundary_result(distance)
                env = result["env"]
                self.assertEqual(env.is_s2u_in_range(0, 0), legal)
                self.assertEqual((0, 0) in env.active_s2u_capacities, legal)
                self.assertEqual(result["service_bits"] > 0.0, legal)
                self.assertEqual(bool(env.active_link_diagnostics), legal)
                if legal:
                    self.assertEqual(result["bandwidths"][("S2U", 0, 0)], 10e6)
                else:
                    self.assertEqual(result["bandwidths"], {})

    def test_horizontal_distance_below_limit_is_illegal_when_3d_distance_exceeds_it(self):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        env.GS_pos = np.array([0.0, 0.0, 0.0])
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (
            399.0,
            0.0,
            50.0,
        )
        env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (
            0.0,
            0.0,
            80.0,
        )
        env.SR_teams[0].x, env.SR_teams[0].y, env.SR_teams[0].z = (0.0, 0.0, 0.0)
        env.update_u2u_channels()
        env.update_u2g_channels()

        self.assertLess(399.0, COMMUNICATION_RANGE_M)
        self.assertGreater(np.hypot(399.0, 50.0), COMMUNICATION_RANGE_M)
        self.assertGreater(np.hypot(399.0, 30.0), COMMUNICATION_RANGE_M)
        self.assertFalse(env.is_u2g_in_range(0))
        self.assertFalse(env.is_u2u_in_range(0, 1))
        self.assertFalse(env.is_s2u_in_range(0, 0))
        self.assertEqual(env.gs_capacity[0], 0.0)
        self.assertEqual(env.Capacity_matrix[0, 1], 0.0)

    def test_com_generation_activates_on_first_entry_and_then_persists(self):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        env.source_uavs = set()
        env.multi_tasks = {uav_id: [] for uav_id in range(env.num_UAV)}
        env.multi_tasks[0] = [{"task_type": "COM", "target_obj_id": 0}]
        env.SR_teams[0].assigned_gt_id = 0
        env.SR_teams[0].x, env.SR_teams[0].y, env.SR_teams[0].z = (0.0, 0.0, 0.0)
        engine = PacketEngine(num_uav=10)
        for slot, expected in ((0, 0), (1, 1), (2, 2)):
            env.uav_dict[0].x_u = 400.0 if slot == 1 else 400.001
            env.uav_dict[0].y_u = 0.0
            env.uav_dict[0].z_u = 0.0
            engine.inject_packets(
                env,
                delay_bound_steps=20,
                current_time=slot * 0.25,
                step_time=0.25,
                rate_overrides={"FOV": 0.0, "COM": 4.0},
            )
            self.assertEqual(engine.generated_packet_counts["COM"], expected)
        self.assertTrue(engine.com_sessions[0]["session_active"])
        self.assertEqual(len(engine.sr_queues[0]), 2)
        env.prepare_channel_routing_slot(0)
        _, bandwidths = env.allocate_active_link_capacities(
            {}, s2u_links=engine.active_s2u_links(env)
        )
        self.assertEqual(bandwidths, {})
        waiting = engine.serve_active_links(env, {}, {}, current_time=0.5)
        self.assertEqual(waiting["transmitted_bits_by_link"], {})
        self.assertEqual(len(engine.sr_queues[0]), 2)

    def test_s2u_inclusive_3d_boundary_and_range_independent_rng_draws(self):
        first = Simulator(num_UAV=10)
        second = Simulator(num_UAV=10)
        first.apply_scenario_entry(self.scenario)
        second.apply_scenario_entry(self.scenario)
        for env in (first, second):
            env.SR_teams[0].x = 0.0
            env.SR_teams[0].y = 0.0
            env.SR_teams[0].z = 0.0
            env.uav_dict[0].x_u = 0.0
            env.uav_dict[0].y_u = 0.0
            env.uav_dict[0].z_u = 400.0
            env.uav_dict[1].x_u = 0.0
            env.uav_dict[1].y_u = 0.0
            env.uav_dict[1].z_u = 400.001
        second.uav_dict[0].z_u = 400.001
        self.assertTrue(first.is_s2u_in_range(0, 0))
        self.assertFalse(second.is_s2u_in_range(0, 0))
        before_first = first.channel.small_scale_normal_draw_count
        before_second = second.channel.small_scale_normal_draw_count
        first.prepare_channel_routing_slot(0)
        second.prepare_channel_routing_slot(0)
        self.assertEqual(
            first.channel.small_scale_normal_draw_count - before_first,
            second.channel.small_scale_normal_draw_count - before_second,
        )
        self.assertEqual(first.channel._profile_keys, second.channel._profile_keys)
        self.assertEqual(first.channel.profile_generation_count, second.channel.profile_generation_count)
        self.assertEqual(
            first.channel.small_scale_rng.bit_generator.state,
            second.channel.small_scale_rng.bit_generator.state,
        )
        np.testing.assert_array_equal(
            first.channel._gain_matrix,
            second.channel._gain_matrix,
        )

    def test_out_of_range_partial_hop_waits_and_resumes_without_restart(self):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(self.scenario)
        env.uav_dict[0].x_u, env.uav_dict[0].y_u, env.uav_dict[0].z_u = (0.0, 0.0, 100.0)
        env.uav_dict[1].x_u, env.uav_dict[1].y_u, env.uav_dict[1].z_u = (100.0, 0.0, 100.0)
        env.update_u2u_channels()
        engine = PacketEngine(
            num_uav=10,
            enable_packet_diagnostic_artifacts=True,
        )
        pkt = engine.create_packet(0, "COM", 1_000.0, 0.0)
        pkt["hop_service_start_time"] = 0.0
        engine.record_hop_transmission(pkt, 0, 1, 400.0)
        self.assertEqual(pkt["hop_receiver"], 1)
        self.assertEqual(pkt["rem_bits"], 600.0)

        env.uav_dict[1].x_u = 400.001
        env.update_u2u_channels()
        effective = engine.get_effective_action_mask(
            env, 0, env.get_routing_action_mask(0).astype(bool)
        )
        self.assertEqual(set(np.flatnonzero(effective)), {0})
        engine.serve_active_links(
            env,
            {0: 0},
            {},
            current_time=0.0,
            start_of_slot_effective_masks_by_sender={0: effective},
        )
        self.assertEqual(
            pkt["locked_receiver_out_of_range_wait_slot_count"], 1
        )
        capacities, bandwidths = env.allocate_active_link_capacities({0: 1})
        self.assertEqual(capacities, {})
        self.assertEqual(bandwidths, {})
        self.assertEqual(pkt["rem_bits"], 600.0)
        self.assertEqual(pkt["hop_receiver"], 1)

        env.uav_dict[1].x_u = 400.0
        env.update_u2u_channels()
        capacity_unavailable = np.zeros(env.num_UAV + 1, dtype=bool)
        capacity_unavailable[0] = True
        effective = engine.get_effective_action_mask(
            env, 0, capacity_unavailable
        )
        engine.serve_active_links(
            env,
            {0: 0},
            {},
            current_time=0.25,
            start_of_slot_effective_masks_by_sender={0: effective},
        )
        self.assertEqual(
            pkt["locked_receiver_out_of_range_wait_slot_count"], 1
        )
        capacities, _ = env.allocate_active_link_capacities({0: 1})
        engine.serve_active_links(
            env,
            {0: 1},
            capacities,
            current_time=0.5,
            block_capacity_profiles=env.active_link_capacity_profiles_mbps,
        )
        self.assertEqual(pkt["path"][-1], 1)
        self.assertEqual(pkt["current"], 1)


class RoutingCreditAndRewardContractTest(unittest.TestCase):
    def test_reward_uses_frozen_other_backlog_not_elapsed_wait(self):
        engine = PacketEngine(num_uav=1)
        pkt = engine.create_packet(0, "COM", 200.0, 0.0)
        env = SimpleNamespace(GS_ID=1)
        first = engine.routing_local_reward(
            env, 0, 1, 0.001, pkt=pkt, current_time=0.0,
            total_backlog_bits=600.0,
        )
        second = engine.routing_local_reward(
            env, 0, 1, 0.001, pkt=pkt, current_time=100.0,
            total_backlog_bits=600.0,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            engine.routing_local_reward(
                env, 0, 0, 0.0, pkt=pkt, current_time=0.0,
                total_backlog_bits=600.0,
            ),
            -1.0,
        )

    def test_transition_is_not_sampleable_until_packet_reference_closes(self):
        engine = PacketEngine(num_uav=1)
        pkt = engine.create_packet(0, "COM", 100.0, 0.0)
        replay = ReplayBufferDiscrete(2, 2, max_size=8, n_step=1)
        ledger = RoutingTransitionLedger()
        transition_id = ledger.create(
            agent_id=0,
            state=np.array([1.0, 0.0], dtype=np.float32),
            action=0,
            tag_gt=2,
        )
        ledger.set_reward(transition_id, 0.25)
        engine._set_packet_routing_transition(pkt, transition_id)
        ledger.finalize_causality(
            {0: np.array([0.0, 1.0], dtype=np.float32)}, {0: pkt}
        )
        self.assertEqual(
            ledger.commit_ready(
                replay, engine.routing_transition_reference_counts()
            ),
            [],
        )
        event = engine._mark_deadline_violation(
            pkt, 0.25, sender=0, reason="deadline"
        )
        self.assertEqual(event["routing_transition_id"], transition_id)
        self.assertEqual(engine.routing_constraint_counts(), (1, 1))
        self.assertEqual(engine.system_qos_counts(), (1, 1))
        self.assertTrue(engine.assert_violation_credit_conservation())
        self.assertEqual(engine.unattributed_transition_violation_count, 0)
        self.assertTrue(ledger.add_cost(event["routing_transition_id"], 1.0))
        self.assertEqual(
            ledger.commit_ready(
                replay, engine.routing_transition_reference_counts()
            ),
            [transition_id],
        )
        self.assertEqual(replay.transition_id[0], transition_id)
        self.assertEqual(replay.cost[0, 0], 1.0)

    def test_packet_a_cost_cannot_move_to_later_packet_b_transition(self):
        engine = PacketEngine(num_uav=1)
        packet_a = engine.create_packet(0, "COM", 100.0, 0.0)
        packet_b = engine.create_packet(0, "COM", 100.0, 0.0)
        replay = ReplayBufferDiscrete(2, 2, max_size=8, n_step=1)
        ledger = RoutingTransitionLedger()
        transition_a = ledger.create(
            agent_id=0, state=[1.0, 0.0], action=0, tag_gt=2
        )
        ledger.set_reward(transition_a, 0.0)
        engine._set_packet_routing_transition(packet_a, transition_a)
        ledger.finalize_causality({0: [0.0, 1.0]}, {0: packet_a})
        transition_b = ledger.create(
            agent_id=0, state=[0.0, 1.0], action=0, tag_gt=2
        )
        ledger.set_reward(transition_b, 0.0)
        engine._set_packet_routing_transition(packet_b, transition_b)
        self.assertEqual(engine.routing_transition_reference_counts(), {0: 1, 1: 1})
        event_a = engine._mark_deadline_violation(
            packet_a, 0.5, sender=0, reason="deadline"
        )
        self.assertEqual(event_a["routing_transition_id"], transition_a)
        ledger.add_cost(event_a["routing_transition_id"], 1.0)
        ledger.finalize_causality({}, {}, terminal=True)
        engine.mark_packet_done(packet_b, current_time=0.5, reason="dropped")
        ledger.commit_ready(replay, engine.routing_transition_reference_counts())
        costs = {
            int(replay.transition_id[index]): float(replay.cost[index, 0])
            for index in range(replay.size)
        }
        self.assertEqual(costs, {transition_a: 1.0, transition_b: 0.0})

    def test_one_wait_transition_can_reference_multiple_packets(self):
        engine = PacketEngine(num_uav=1)
        packets = [
            engine.create_packet(0, "COM", 100.0, 0.0)
            for _ in range(2)
        ]
        transition_id = 7
        for pkt in packets:
            engine._set_packet_routing_transition(pkt, transition_id)
        self.assertEqual(
            engine.routing_transition_reference_counts(), {transition_id: 2}
        )
        engine._set_packet_routing_transition(packets[0], 8)
        self.assertEqual(
            engine.routing_transition_reference_counts(), {transition_id: 1, 8: 1}
        )

    def test_max_hop_violation_returns_its_stable_transition_event(self):
        engine = PacketEngine(num_uav=1)
        packet = engine.create_packet(0, "COM", 100.0, 0.0)
        packet["hops"] = 20
        engine._set_packet_routing_transition(packet, 12)

        events = engine.drop_expired_packets(0.25)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["routing_transition_id"], 12)
        self.assertEqual(engine.system_qos_counts(), (1, 1))
        self.assertEqual(engine.routing_constraint_counts(), (1, 1))
        self.assertTrue(engine.assert_violation_credit_conservation())


class AtomicFovAndAggregationContractTest(unittest.TestCase):
    def test_search_uavs_observe_same_precommit_bitmap(self):
        env = Simulator(num_UAV=10)
        env.apply_scenario_entry(
            generate_manifest("test", 9402, 1).episodes[0]
        )
        env._search_phase_over = False
        env.visited_bitmap[:] = False
        env.multi_tasks = {uid: [] for uid in range(env.num_UAV)}
        env.multi_tasks[0] = [{"task_type": "Search"}]
        env.multi_tasks[1] = [{"task_type": "Search"}]
        env.uav_dict[1].x_u = env.uav_dict[0].x_u
        env.uav_dict[1].y_u = env.uav_dict[0].y_u
        env.uav_dict[1].z_u = env.uav_dict[0].z_u
        transitions = _mark_search_observations(env)
        self.assertEqual(len(transitions), env.num_UAV)
        self.assertEqual(transitions[0].raw_unvisited, 1.0)
        self.assertEqual(transitions[1].raw_unvisited, 1.0)
        self.assertTrue(transitions[0].map_changed)
        self.assertTrue(transitions[1].map_changed)
        footprint = transitions[0].current_footprint
        self.assertTrue(
            env.visited_bitmap[
                footprint[0] : footprint[1] + 1,
                footprint[2] : footprint[3] + 1,
            ].all()
        )

    def test_cross_seed_mean_uses_seed_ratios_not_pooled_episodes(self):
        def row(seed, violations, eligible, goodput, energy):
            return {
                "training_seed": seed,
                "timely_goodput_mbits": goodput,
                "total_mobility_energy_j": energy,
                "fov_delivered_packets": 1,
                "fov_delivered_e2e_delay_sum_seconds": 1.0,
                "fov_violation_packets": violations,
                "fov_eligible_packets": eligible,
                "com_delivered_packets": 0,
                "com_delivered_e2e_delay_sum_seconds": 0.0,
                "com_violation_packets": 0,
                "com_eligible_packets": 0,
            }

        per_seed, cross_seed = canonical_aggregation(
            [row(1, 1, 1, 1.0, 1.0), row(2, 0, 9, 9.0, 9.0)]
        )
        violation = next(
            item for item in cross_seed
            if item["metric"] == "violation_probability"
            and item["task_type"] == "FOV"
        )
        self.assertEqual(violation["mean"], 0.5)
        self.assertEqual(violation["pooled_numerator"], 1.0)
        self.assertEqual(violation["pooled_denominator"], 10.0)
        com = next(
            item for item in cross_seed
            if item["metric"] == "violation_probability"
            and item["task_type"] == "COM"
        )
        self.assertTrue(com["missing"])
        self.assertEqual(com["valid_training_seed_count"], 0)
        self.assertEqual(len(per_seed), 12)

    def test_within_seed_probability_delay_and_ee_are_ratios_of_sums(self):
        rows = [
            {
                "training_seed": 1,
                "timely_goodput_mbits": 1.0,
                "total_mobility_energy_j": 1.0,
                "fov_delivered_packets": 1,
                "fov_delivered_e2e_delay_sum_seconds": 1.0,
                "fov_violation_packets": 1,
                "fov_eligible_packets": 1,
                "com_delivered_packets": 0,
                "com_delivered_e2e_delay_sum_seconds": 0.0,
                "com_violation_packets": 0,
                "com_eligible_packets": 0,
            },
            {
                "training_seed": 1,
                "timely_goodput_mbits": 0.0,
                "total_mobility_energy_j": 99.0,
                "fov_delivered_packets": 99,
                "fov_delivered_e2e_delay_sum_seconds": 198.0,
                "fov_violation_packets": 0,
                "fov_eligible_packets": 99,
                "com_delivered_packets": 0,
                "com_delivered_e2e_delay_sum_seconds": 0.0,
                "com_violation_packets": 0,
                "com_eligible_packets": 0,
            },
        ]
        per_seed, _ = canonical_aggregation(rows)
        keyed = {
            (row["metric"], row["task_type"]): row for row in per_seed
        }
        self.assertEqual(
            keyed[("violation_probability", "FOV")]["value"], 0.01
        )
        self.assertEqual(
            keyed[("energy_efficiency_mbit_per_j", None)]["value"], 0.01
        )
        self.assertEqual(
            keyed[("average_e2e_delay_seconds", "FOV")]["value"], 1.99
        )
        paper_rows = aggregate_paper_point_metrics(
            "method",
            "fixed_roi",
            {"point_id": "roi_2", "x_value": 2, "x_unit": "RoIs"},
            rows,
        )
        paper_keyed = {
            (row["metric"], row["task_type"]): row for row in paper_rows
        }
        for identity, row in keyed.items():
            self.assertEqual(paper_keyed[identity]["value"], row["value"])


if __name__ == "__main__":
    unittest.main()
