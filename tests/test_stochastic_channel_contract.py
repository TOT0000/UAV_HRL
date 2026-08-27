import copy
from types import SimpleNamespace
import unittest

import numpy as np

from Channel_model import (
    A2G_LOS_EXCESS_DB,
    A2G_NLOS_EXCESS_DB,
    CHANNEL_MODEL_VERSION,
    FADING_BLOCK_SECONDS,
    FADING_BLOCKS_PER_ROUTING_SLOT,
    RICIAN_K_LINEAR,
    ROUTING_SLOT_SECONDS,
    ChannelLifecycle,
    a2g_conditional_path_loss_db,
    a2g_expected_path_loss_db,
    a2g_free_space_path_loss_db,
    a2g_los_probability_from_elevation_deg,
    block_capacity_profile_mbps,
    effective_capacity_mbps,
    expected_fading_capacity_mbps,
    sample_fading_power_gains,
    slot_service_bits,
    validate_channel_time_grid,
)
from HRL_task_aware import ROUTING_STATE_DIM
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from experiment_config import METHOD_REGISTRY, MethodSpec, comparison_method_configuration
from rng_contract import NamedRNGStreams, RNG_STREAM_IDS


def make_environment(seed=20260817, *, evaluation=False):
    env = Simulator(
        10,
        rng_streams=NamedRNGStreams(seed),
        evaluation=evaluation,
    )
    env.num_GT = 2
    env.reset_environment()
    return env


def channel_geometry(num_uav=3, num_sr=2):
    uav = np.column_stack(
        (
            np.linspace(100.0, 300.0, num_uav),
            np.linspace(50.0, 150.0, num_uav),
            np.linspace(80.0, 120.0, num_uav),
        )
    )
    sr = np.column_stack(
        (
            np.linspace(0.0, 500.0, num_sr),
            np.linspace(500.0, 0.0, num_sr),
            np.zeros(num_sr),
        )
    )
    return uav, sr, np.zeros(3, dtype=float)


class ChannelMathTest(unittest.TestCase):
    def test_time_grid_is_exact_and_nonintegral_grid_fails_fast(self):
        self.assertEqual(validate_channel_time_grid(), 50)
        self.assertEqual(FADING_BLOCKS_PER_ROUTING_SLOT, 50)
        self.assertEqual(FADING_BLOCK_SECONDS, 0.005)
        self.assertEqual(ROUTING_SLOT_SECONDS, 0.25)
        self.assertEqual(
            FADING_BLOCKS_PER_ROUTING_SLOT * FADING_BLOCK_SECONDS,
            ROUTING_SLOT_SECONDS,
        )
        with self.assertRaisesRegex(ValueError, "integer number"):
            validate_channel_time_grid(0.25, 0.006)

    def test_rician_and_rayleigh_power_means_are_normalized(self):
        rng = np.random.default_rng(91)
        rician = sample_fading_power_gains(
            rng, (300_000,), fading="rician"
        )
        rayleigh = sample_fading_power_gains(
            rng, (300_000,), fading="rayleigh"
        )
        self.assertEqual(RICIAN_K_LINEAR, 10.0)
        self.assertAlmostEqual(float(rician.mean()), 1.0, delta=0.01)
        self.assertAlmostEqual(float(rayleigh.mean()), 1.0, delta=0.01)

    def test_samples_are_seed_reproducible_and_seed_sensitive(self):
        first = sample_fading_power_gains(
            np.random.default_rng(8), (5, 50), fading="rician"
        )
        repeated = sample_fading_power_gains(
            np.random.default_rng(8), (5, 50), fading="rician"
        )
        different = sample_fading_power_gains(
            np.random.default_rng(9), (5, 50), fading="rician"
        )
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))

    def test_expected_capacity_matches_monte_carlo_for_both_fading_laws(self):
        path_loss_db = 100.0
        bandwidth_hz = 1e6
        transmit_power_dbm = 30.0
        rng = np.random.default_rng(101)
        for fading in ("rayleigh", "rician"):
            with self.subTest(fading=fading):
                gains = sample_fading_power_gains(
                    rng, (400_000,), fading=fading
                )
                monte_carlo = float(
                    block_capacity_profile_mbps(
                        path_loss_db,
                        bandwidth_hz,
                        transmit_power_dbm,
                        gains,
                    ).mean()
                )
                deterministic = float(
                    expected_fading_capacity_mbps(
                        path_loss_db,
                        bandwidth_hz,
                        transmit_power_dbm,
                        fading=fading,
                    )
                )
                self.assertAlmostEqual(deterministic, monte_carlo, delta=0.04)
                self.assertEqual(
                    deterministic,
                    float(
                        expected_fading_capacity_mbps(
                            path_loss_db,
                            bandwidth_hz,
                            transmit_power_dbm,
                            fading=fading,
                        )
                    ),
                )

    def test_expected_capacity_monotonicity_units_and_finiteness(self):
        base = float(
            expected_fading_capacity_mbps(100.0, 1e6, 23.0, fading="rician")
        )
        worse_loss = float(
            expected_fading_capacity_mbps(110.0, 1e6, 23.0, fading="rician")
        )
        more_power = float(
            expected_fading_capacity_mbps(100.0, 1e6, 30.0, fading="rician")
        )
        more_bandwidth = float(
            expected_fading_capacity_mbps(100.0, 5e6, 23.0, fading="rician")
        )
        self.assertGreater(base, worse_loss)
        self.assertGreater(more_power, base)
        self.assertGreater(more_bandwidth, base)
        for value in (base, worse_loss, more_power, more_bandwidth):
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        profile = np.full(50, 2.0)
        self.assertEqual(float(slot_service_bits(profile)), 500_000.0)
        self.assertEqual(float(effective_capacity_mbps(profile)), 2.0)

    def test_los_probability_and_conditional_path_loss_contract(self):
        probabilities = a2g_los_probability_from_elevation_deg(
            np.linspace(0.0, 90.0, 181)
        )
        self.assertTrue(np.all(np.diff(probabilities) >= 0.0))
        target_probability = float(a2g_los_probability_from_elevation_deg(30.0))
        draws = np.random.default_rng(19).random(300_000) < target_probability
        self.assertAlmostEqual(
            float(draws.mean()), target_probability, delta=0.004
        )
        aerial, ground = (100.0, 20.0, 80.0), (0.0, 0.0, 0.0)
        free_space = float(a2g_free_space_path_loss_db(aerial, ground))
        los = float(a2g_conditional_path_loss_db(aerial, ground, True))
        nlos = float(a2g_conditional_path_loss_db(aerial, ground, False))
        expected = float(a2g_expected_path_loss_db(aerial, ground))
        self.assertAlmostEqual(los, free_space + A2G_LOS_EXCESS_DB)
        self.assertAlmostEqual(nlos, free_space + A2G_NLOS_EXCESS_DB)
        self.assertGreater(expected, los)
        self.assertLess(expected, nlos)


class ChannelLifecycleTest(unittest.TestCase):
    def _lifecycle(self, seed, namespace="training"):
        streams = NamedRNGStreams(seed)
        prefix = "evaluation_" if namespace == "evaluation" else ""
        lifecycle = ChannelLifecycle(
            3,
            3,
            large_scale_rng=streams.numpy(
                f"{prefix}channel_large_scale_state"
            ),
            small_scale_rng=streams.numpy(
                f"{prefix}channel_small_scale_fading"
            ),
            namespace=namespace,
        )
        geometry = channel_geometry()
        lifecycle.reset_episode(
            uav_positions=geometry[0],
            sr_positions=geometry[1],
            gs_position=geometry[2],
            episode_identity="scenario-1",
        )
        return streams, lifecycle, geometry

    def test_large_scale_state_is_held_for_four_slots_and_u2u_is_always_rician(self):
        _, lifecycle, geometry = self._lifecycle(22)
        u2g = lifecycle.u2g_los_state.copy()
        s2u = lifecycle.s2u_los_state.copy()
        draw_count = lifecycle.large_scale_draw_count
        for slot in range(4):
            lifecycle.prepare_routing_slot(slot)
            np.testing.assert_array_equal(lifecycle.u2g_los_state, u2g)
            np.testing.assert_array_equal(lifecycle.s2u_los_state, s2u)
            self.assertTrue(lifecycle.a2g_state("U2U", 0, 1))
        self.assertEqual(lifecycle.large_scale_draw_count, draw_count)
        self.assertFalse(
            lifecycle.begin_movement_interval(
                0,
                uav_positions=geometry[0],
                sr_positions=geometry[1],
                gs_position=geometry[2],
            )
        )
        self.assertTrue(
            lifecycle.begin_movement_interval(
                1,
                uav_positions=geometry[0],
                sr_positions=geometry[1],
                gs_position=geometry[2],
            )
        )
        self.assertEqual(
            lifecycle.large_scale_draw_count,
            draw_count + 3 * (2 + 1),
        )

    def test_all_potential_links_are_generated_once_independent_of_actions(self):
        _, first, _ = self._lifecycle(31)
        _, second, _ = self._lifecycle(31)
        first.prepare_routing_slot(0)
        second.prepare_routing_slot(0)
        self.assertEqual(first.potential_link_keys(), second.potential_link_keys())
        np.testing.assert_array_equal(first._gain_matrix, second._gain_matrix)
        before = first.small_scale_normal_draw_count
        self.assertFalse(first.prepare_routing_slot(0))
        self.assertEqual(first.small_scale_normal_draw_count, before)
        # Different active-link choices never enter the channel generator.
        for key in first.potential_link_keys()[::3]:
            first.gain_profile(*key)
        first.prepare_routing_slot(1)
        second.prepare_routing_slot(1)
        np.testing.assert_array_equal(first._gain_matrix, second._gain_matrix)

    def test_episode_reset_clears_expired_slot_profile_and_initializes_a2g_state(self):
        _, lifecycle, geometry = self._lifecycle(32)
        lifecycle.prepare_routing_slot(0)
        self.assertIsNotNone(lifecycle._gain_matrix)
        lifecycle.reset_episode(
            uav_positions=geometry[0],
            sr_positions=geometry[1],
            gs_position=geometry[2],
            episode_identity="scenario-2",
        )
        self.assertIsNone(lifecycle._gain_matrix)
        self.assertIsNone(lifecycle.routing_slot_index)
        self.assertEqual(lifecycle.u2g_los_state.shape, (3,))
        self.assertEqual(lifecycle.s2u_los_state.shape, (2, 3))

    def test_channel_checkpoint_resume_restores_next_state_and_profile(self):
        streams, lifecycle, geometry = self._lifecycle(44)
        lifecycle.prepare_routing_slot(0)
        saved_rng = streams.state_dict()
        saved_channel = lifecycle.state_dict()
        lifecycle.begin_movement_interval(
            1,
            uav_positions=geometry[0],
            sr_positions=geometry[1],
            gs_position=geometry[2],
        )
        lifecycle.prepare_routing_slot(4)
        expected_u2g = lifecycle.u2g_los_state.copy()
        expected_s2u = lifecycle.s2u_los_state.copy()
        expected_gains = lifecycle._gain_matrix.copy()

        restored_streams = NamedRNGStreams(44)
        restored_streams.load_state_dict(saved_rng)
        restored = ChannelLifecycle(
            3,
            3,
            large_scale_rng=restored_streams.numpy("channel_large_scale_state"),
            small_scale_rng=restored_streams.numpy("channel_small_scale_fading"),
            namespace="training",
        )
        restored.load_state_dict(saved_channel)
        restored.begin_movement_interval(
            1,
            uav_positions=geometry[0],
            sr_positions=geometry[1],
            gs_position=geometry[2],
        )
        restored.prepare_routing_slot(4)
        np.testing.assert_array_equal(restored.u2g_los_state, expected_u2g)
        np.testing.assert_array_equal(restored.s2u_los_state, expected_s2u)
        np.testing.assert_array_equal(restored._gain_matrix, expected_gains)

    def test_training_and_evaluation_channel_streams_are_isolated(self):
        baseline = NamedRNGStreams(55)
        mixed = NamedRNGStreams(55)
        mixed.numpy("evaluation_channel_large_scale_state").random(100)
        mixed.numpy("evaluation_channel_small_scale_fading").standard_normal(100)
        np.testing.assert_array_equal(
            baseline.numpy("channel_large_scale_state").random(20),
            mixed.numpy("channel_large_scale_state").random(20),
        )
        np.testing.assert_array_equal(
            baseline.numpy("channel_small_scale_fading").standard_normal(20),
            mixed.numpy("channel_small_scale_fading").standard_normal(20),
        )
        for name in (
            "channel_large_scale_state",
            "channel_small_scale_fading",
            "evaluation_channel_large_scale_state",
            "evaluation_channel_small_scale_fading",
        ):
            self.assertIn(name, RNG_STREAM_IDS)

    def test_profile_generation_is_vectorized_and_fast_enough_for_one_slot(self):
        streams = NamedRNGStreams(71)
        lifecycle = ChannelLifecycle(
            10,
            10,
            large_scale_rng=streams.numpy("channel_large_scale_state"),
            small_scale_rng=streams.numpy("channel_small_scale_fading"),
        )
        geometry = channel_geometry(10, 8)
        lifecycle.reset_episode(
            uav_positions=geometry[0],
            sr_positions=geometry[1],
            gs_position=geometry[2],
        )
        lifecycle.prepare_routing_slot(0)
        self.assertEqual(lifecycle._gain_matrix.shape, (180, 50))
        self.assertLess(lifecycle.last_profile_generation_seconds, 1.0)


class ObservationFdmaAndPacketServiceTest(unittest.TestCase):
    def test_observation_uses_expected_csi_and_does_not_consume_fading_rng(self):
        env = make_environment(80)
        engine = PacketEngine(10)
        env.update_source_uavs()
        engine.create_packet(0, "COM", 256.0, 0.0)
        env.update_u2u_channels()
        env.update_u2g_channels()
        rng_before = copy.deepcopy(env.channel_small_scale_rng.bit_generator.state)
        state_before = engine.get_state_ta(
            env,
            0,
            backlog_bits=engine.backlog_bits,
            action_mask=env.get_routing_action_mask(0).astype(bool),
        )
        self.assertEqual(env.channel_small_scale_rng.bit_generator.state, rng_before)
        env.prepare_channel_routing_slot(0)
        env.channel._gain_matrix *= 1000.0
        state_after = engine.get_state_ta(
            env,
            0,
            backlog_bits=engine.backlog_bits,
            action_mask=env.get_routing_action_mask(0).astype(bool),
        )
        np.testing.assert_array_equal(state_before, state_after)
        self.assertEqual(state_before.shape, (ROUTING_STATE_DIM,))

    def test_physical_reference_normalization_has_no_fixed_200_saturation(self):
        env = make_environment(81)
        u2u_reference = env.routing_capacity_reference_mbps(0, 1)
        u2g_reference = env.routing_capacity_reference_mbps(0, env.GS_ID)
        ratios = np.concatenate(
            (
                env.Capacity_matrix[~np.eye(10, dtype=bool)] / u2u_reference,
                env.gs_capacity[:10] / u2g_reference,
            )
        )
        self.assertTrue(np.all(np.isfinite(ratios)))
        self.assertTrue(np.all((ratios >= 0.0) & (ratios <= 1.0 + 1e-12)))
        self.assertLess(float(np.mean(ratios >= 1.0 - 1e-12)), 0.1)
        mask = env.get_routing_action_mask(0).astype(bool)
        expected = np.array(
            [
                True if receiver == 0 else env.is_routing_link_in_range(0, receiver)
                for receiver in range(env.num_UAV + 1)
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_equal_fdma_recomputes_profiles_and_reuses_slot_gains(self):
        env = make_environment(82)
        env.uav_dict[2].x_u = 100.0
        env.uav_dict[2].y_u = 0.0
        env.uav_dict[2].z_u = 100.0
        env.update_u2g_channels()
        env.prepare_channel_routing_slot(0)
        gains_before = env.channel.gain_profile("U2U", 0, 1).copy()
        draw_count = env.channel.small_scale_normal_draw_count
        capacities, bandwidths = env.allocate_active_link_capacities(
            {0: 1, 2: env.GS_ID}, s2u_links={0: 3}
        )
        self.assertLessEqual(sum(bandwidths.values()), env.B_tot + 1e-6)
        self.assertEqual(set(capacities), {(0, 1), (2, env.GS_ID)})
        for profile in (
            list(env.active_link_capacity_profiles_mbps.values())
            + list(env.active_s2u_capacity_profiles_mbps.values())
        ):
            self.assertEqual(profile.shape, (50,))
            self.assertAlmostEqual(
                float(profile.mean()) * ROUTING_SLOT_SECONDS * 1e6,
                float(slot_service_bits(profile)),
            )
        env.allocate_active_link_capacities({0: 1})
        np.testing.assert_array_equal(
            env.channel.gain_profile("U2U", 0, 1), gains_before
        )
        self.assertEqual(env.channel.small_scale_normal_draw_count, draw_count)

    def test_packet_completion_time_uses_the_completing_block(self):
        env = SimpleNamespace(
            GS_ID=1,
            GS_pos=(0.0, 0.0, 0.0),
            uav_dict={0: SimpleNamespace(x_u=10.0, y_u=0.0)},
        )
        engine = PacketEngine(1)
        packet = engine.create_packet(0, "COM", 2.5, 0.0)
        profile = np.zeros(50, dtype=float)
        profile[1] = 0.001  # 1000 bit/s; 2.5 bits complete 2.5 ms into block 1.
        result = engine.serve_active_links(
            env,
            actions={0: env.GS_ID},
            capacities={(0, env.GS_ID): float(profile.mean())},
            block_capacity_profiles={(0, env.GS_ID): profile},
            current_time=0.0,
        )
        self.assertAlmostEqual(packet["finish_time"], 0.0075, places=12)
        self.assertAlmostEqual(packet["e2e_delay_ms"], 7.5, places=9)
        self.assertAlmostEqual(
            result["transmitted_bits_by_link"][(0, env.GS_ID)], 2.5
        )
        self.assertEqual(engine.link_slot_budget_violations, 0)

    def test_every_registered_method_publishes_the_same_channel_contract(self):
        configurations = {
            method_id: comparison_method_configuration(MethodSpec.parse(method_id))
            for method_id in METHOD_REGISTRY
        }
        self.assertGreaterEqual(len(configurations), 16)
        for method_id, configuration in configurations.items():
            with self.subTest(method=method_id):
                self.assertEqual(
                    configuration["channel_model_version"], CHANNEL_MODEL_VERSION
                )
                self.assertEqual(
                    configuration["fading_blocks_per_routing_slot"], 50
                )
                self.assertEqual(
                    configuration["channel_configuration"]["routing_csi"],
                    "deterministic-distributional-expected-capacity-v1",
                )


if __name__ == "__main__":
    unittest.main()
