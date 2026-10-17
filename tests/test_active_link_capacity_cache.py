import copy
import unittest
from unittest import mock

import numpy as np

import Simulator as simulator_module
from Simulator import Simulator
from rng_contract import NamedRNGStreams


class ActiveLinkCapacityCacheTest(unittest.TestCase):
    def _environment(self, seed=20260817, *, cache_enabled=True):
        env = Simulator(16, rng_streams=NamedRNGStreams(seed))
        env._active_link_capacity_cache_enabled = bool(cache_enabled)
        env.num_GT = 2
        env.reset_environment()
        for uav_id, uav in env.uav_dict.items():
            uav.x_u = 100.0 + 10.0 * (uav_id % 4)
            uav.y_u = 100.0 + 10.0 * (uav_id // 4)
            uav.z_u = 100.0
        for sr_id, team in enumerate(env.SR_teams):
            team.x = 105.0 + 5.0 * sr_id
            team.y = 105.0
            team.z = 0.0
        env.update_u2u_channels()
        env.update_u2g_channels()
        env.prepare_channel_routing_slot(0)
        return env

    @staticmethod
    def _snapshot(env, capacities, bandwidths):
        return {
            "capacities": dict(capacities),
            "bandwidths": dict(bandwidths),
            "uav_profiles": {
                key: value.copy()
                for key, value in env.active_link_capacity_profiles_mbps.items()
            },
            "s2u_capacities": dict(env.active_s2u_capacities),
            "s2u_profiles": {
                key: value.copy()
                for key, value in env.active_s2u_capacity_profiles_mbps.items()
            },
            "diagnostics": copy.deepcopy(env.active_link_diagnostics),
        }

    def assertAllocationEqual(self, first, second):
        self.assertEqual(first["capacities"], second["capacities"])
        self.assertEqual(first["bandwidths"], second["bandwidths"])
        self.assertEqual(first["s2u_capacities"], second["s2u_capacities"])
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        self.assertEqual(set(first["uav_profiles"]), set(second["uav_profiles"]))
        self.assertEqual(set(first["s2u_profiles"]), set(second["s2u_profiles"]))
        for key in first["uav_profiles"]:
            np.testing.assert_array_equal(
                first["uav_profiles"][key], second["uav_profiles"][key]
            )
        for key in first["s2u_profiles"]:
            np.testing.assert_array_equal(
                first["s2u_profiles"][key], second["s2u_profiles"][key]
            )

    def test_repeated_inputs_are_exact_and_return_isolated_profiles(self):
        env = self._environment()
        with mock.patch.object(
            simulator_module,
            "block_capacity_profile_mbps",
            wraps=simulator_module.block_capacity_profile_mbps,
        ) as compute:
            capacities, bandwidths = env.allocate_active_link_capacities({0: 1})
            first = self._snapshot(env, capacities, bandwidths)
            env.active_link_capacity_profiles_mbps[(0, 1)][0] = -123.0

            capacities, bandwidths = env.allocate_active_link_capacities({0: 1})
            second = self._snapshot(env, capacities, bandwidths)

        self.assertEqual(compute.call_count, 1)
        self.assertAllocationEqual(first, second)
        self.assertEqual(
            env.active_link_capacity_cache_statistics(),
            {"enabled": True, "hits": 1, "misses": 1, "entries": 1},
        )

    def test_four_five_four_reuses_profiles_when_bandwidth_returns(self):
        env = self._environment()
        four_links = {0: 1, 2: 3, 4: 5, 6: 7}
        five_links = {**four_links, 8: 9}
        with mock.patch.object(
            simulator_module,
            "block_capacity_profile_mbps",
            wraps=simulator_module.block_capacity_profile_mbps,
        ) as compute:
            first = self._snapshot(
                env, *env.allocate_active_link_capacities(four_links)
            )
            env.allocate_active_link_capacities(five_links)
            repeated = self._snapshot(
                env, *env.allocate_active_link_capacities(four_links)
            )

        self.assertEqual(compute.call_count, 9)
        self.assertAllocationEqual(first, repeated)
        self.assertEqual(
            set(repeated["bandwidths"].values()), {env.B_tot / 4.0}
        )
        stats = env.active_link_capacity_cache_statistics()
        self.assertEqual((stats["hits"], stats["misses"]), (4, 9))

    def test_same_count_changed_set_reuses_only_unchanged_links(self):
        env = self._environment()
        first_links = {0: 1, 2: 3, 4: 5, 6: 7}
        changed_links = {0: 1, 2: 3, 4: 5, 8: 9}
        with mock.patch.object(
            simulator_module,
            "block_capacity_profile_mbps",
            wraps=simulator_module.block_capacity_profile_mbps,
        ) as compute:
            env.allocate_active_link_capacities(first_links)
            capacities, bandwidths = env.allocate_active_link_capacities(
                changed_links
            )

        self.assertEqual(compute.call_count, 5)
        self.assertEqual(set(capacities), set(changed_links.items()))
        self.assertEqual(set(bandwidths), set(changed_links.items()))
        self.assertEqual(
            set(env.active_link_capacity_profiles_mbps),
            set(changed_links.items()),
        )
        self.assertEqual(env.active_s2u_capacity_profiles_mbps, {})
        stats = env.active_link_capacity_cache_statistics()
        self.assertEqual((stats["hits"], stats["misses"]), (3, 5))

    def test_link_type_direction_and_endpoints_have_distinct_entries(self):
        env = self._environment()
        uav_links = {0: 1, 1: 0, 2: env.GS_ID}
        s2u_links = {0: 3}
        env.allocate_active_link_capacities(uav_links, s2u_links=s2u_links)
        first_stats = env.active_link_capacity_cache_statistics()
        env.allocate_active_link_capacities(uav_links, s2u_links=s2u_links)
        second_stats = env.active_link_capacity_cache_statistics()

        self.assertEqual(first_stats["misses"], 4)
        self.assertEqual(first_stats["entries"], 4)
        self.assertEqual(second_stats["hits"], 4)
        self.assertFalse(
            np.array_equal(
                env.active_link_capacity_profiles_mbps[(0, 1)],
                env.active_link_capacity_profiles_mbps[(1, 0)],
            )
        )
        self.assertIn((0, 3), env.active_s2u_capacity_profiles_mbps)

    def test_slot_geometry_load_and_episode_boundaries_invalidate(self):
        env = self._environment()
        link = {0: 1}
        env.allocate_active_link_capacities(link)
        env.allocate_active_link_capacities(link)
        self.assertEqual(
            (env._active_link_capacity_cache_hits, env._active_link_capacity_cache_misses),
            (1, 1),
        )

        env.prepare_channel_routing_slot(1)
        env.allocate_active_link_capacities(link)
        self.assertEqual(env._active_link_capacity_cache_misses, 2)

        env.uav_dict[1].x_u += 25.0
        env.update_u2u_channels()
        env.allocate_active_link_capacities(link)
        self.assertEqual(env._active_link_capacity_cache_misses, 3)

        channel_state = env.channel_state_dict()
        env.load_channel_state_dict(channel_state)
        env.allocate_active_link_capacities(link)
        self.assertEqual(env._active_link_capacity_cache_misses, 4)

        env.reset_environment()
        env.prepare_channel_routing_slot(0)
        env.allocate_active_link_capacities({0: env.GS_ID})
        self.assertEqual(
            env.active_link_capacity_cache_statistics(),
            {"enabled": True, "hits": 0, "misses": 1, "entries": 1},
        )

    def test_cached_and_uncached_sequences_are_exact_and_rng_neutral(self):
        cached = self._environment(seed=31415, cache_enabled=True)
        uncached = self._environment(seed=31415, cache_enabled=False)
        sequence = (
            ({0: 1, 2: 3, 4: 5, 6: 7}, {}),
            ({0: 1, 2: 3, 4: 5, 6: 7, 8: 9}, {}),
            ({0: 1, 2: 3, 4: 5, 6: 7}, {}),
            ({0: 1, 1: 0, 2: cached.GS_ID}, {0: 3}),
        )
        cached_rng_before = copy.deepcopy(
            cached.channel_small_scale_rng.bit_generator.state
        )
        uncached_rng_before = copy.deepcopy(
            uncached.channel_small_scale_rng.bit_generator.state
        )
        self.assertEqual(cached_rng_before, uncached_rng_before)

        for uav_links, s2u_links in sequence:
            cached_result = self._snapshot(
                cached,
                *cached.allocate_active_link_capacities(
                    uav_links, s2u_links=s2u_links
                ),
            )
            uncached_result = self._snapshot(
                uncached,
                *uncached.allocate_active_link_capacities(
                    uav_links, s2u_links=s2u_links
                ),
            )
            self.assertAllocationEqual(cached_result, uncached_result)

        self.assertEqual(
            cached.channel_small_scale_rng.bit_generator.state,
            uncached.channel_small_scale_rng.bit_generator.state,
        )
        self.assertGreater(
            cached.active_link_capacity_cache_statistics()["hits"], 0
        )
        self.assertEqual(
            uncached.active_link_capacity_cache_statistics()["hits"], 0
        )

    def test_physical_geometry_cache_is_exact_invalidated_and_rng_neutral(self):
        cached = self._environment(seed=2718)
        reference = self._environment(seed=2718)
        reference._routing_geometry_cache_enabled = False

        cached_result = self._snapshot(
            cached,
            *cached.allocate_active_link_capacities(
                {0: 1, 2: cached.GS_ID}, s2u_links={0: 3}
            ),
        )
        reference_result = self._snapshot(
            reference,
            *reference.allocate_active_link_capacities(
                {0: 1, 2: reference.GS_ID}, s2u_links={0: 3}
            ),
        )
        self.assertAllocationEqual(cached_result, reference_result)
        self.assertEqual(
            cached.channel_small_scale_rng.bit_generator.state,
            reference.channel_small_scale_rng.bit_generator.state,
        )
        stats = cached.routing_geometry_cache_statistics()
        self.assertEqual(stats["builds"], 1)
        self.assertGreater(stats["hits"], 0)

        # Existing tools/tests may update object coordinates directly. Exact
        # endpoint validation must reject the stale matrix without requiring
        # a rounded coordinate key.
        cached.uav_dict[1].x_u += 900.0
        expected = cached.distance_3d(
            cached.uav_dict[0].get_position(),
            cached.uav_dict[1].get_position(),
        ) <= simulator_module.COMMUNICATION_RANGE_M
        self.assertEqual(cached.is_u2u_in_range(0, 1), expected)
        self.assertEqual(
            cached.routing_geometry_cache_statistics()["builds"], 2
        )

        channel_state = cached.channel_state_dict()
        cached.load_channel_state_dict(channel_state)
        self.assertFalse(
            cached.routing_geometry_cache_statistics()["populated"]
        )
        cached.reset_environment()
        self.assertEqual(
            cached.routing_geometry_cache_statistics(),
            {"enabled": True, "builds": 0, "hits": 0, "populated": False},
        )

    def test_zero_wait_and_out_of_range_links_do_not_populate_cache(self):
        env = self._environment()
        capacities, bandwidths = env.allocate_active_link_capacities({0: 0})
        self.assertEqual((capacities, bandwidths), ({}, {}))
        self.assertEqual(env.active_link_capacity_cache_statistics()["entries"], 0)

        env.allocate_active_link_capacities({0: 1})
        capacities, bandwidths = env.allocate_active_link_capacities({0: 0})
        self.assertEqual((capacities, bandwidths), ({}, {}))
        self.assertEqual(env.active_link_capacities, {})
        self.assertEqual(env.active_link_capacity_profiles_mbps, {})
        self.assertEqual(env.active_s2u_capacities, {})
        self.assertEqual(env.active_s2u_capacity_profiles_mbps, {})

        env.uav_dict[0].x_u = 0.0
        env.uav_dict[0].y_u = 0.0
        env.uav_dict[1].x_u = 1000.0
        env.uav_dict[1].y_u = 1000.0
        env.update_u2u_channels()
        capacities, bandwidths = env.allocate_active_link_capacities({0: 1})
        self.assertEqual((capacities, bandwidths), ({}, {}))
        self.assertEqual(env.active_link_capacity_cache_statistics()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
