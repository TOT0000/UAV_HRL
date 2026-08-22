import unittest

import numpy as np

from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from scenario_manifest import generate_manifest


class EpisodeScopedRoutingStateTest(unittest.TestCase):
    def test_scenario_b_first_observation_is_independent_of_scenario_a(self):
        manifest = generate_manifest("test", 4401, 2)
        scenario_a, scenario_b = manifest.episodes
        env = Simulator(num_UAV=16)
        reused = PacketEngine(num_uav=16, step_time=0.25)

        env.apply_scenario_entry(scenario_a)
        for _ in range(3):
            reused.get_state_ta(
                env,
                0,
                backlog_bits=reused.backlog_bits,
                action_mask=env.get_routing_action_mask(0),
            )
        self.assertEqual(reused.fov_ema, {})
        reused.update_fov_ema(env, "scenario-a-map")
        self.assertIn(0, reused.fov_ema)
        self.assertTrue(any(value != 0.0 for value in reused.fov_ema[0].values()))

        env.apply_scenario_entry(scenario_b)
        reused.reset_packet_state()
        reset_state = reused.get_state_ta(
            env,
            0,
            backlog_bits=reused.backlog_bits,
            action_mask=env.get_routing_action_mask(0),
        )

        fresh = PacketEngine(num_uav=16, step_time=0.25)
        fresh_state = fresh.get_state_ta(
            env,
            0,
            backlog_bits=fresh.backlog_bits,
            action_mask=env.get_routing_action_mask(0),
        )

        self.assertEqual(reset_state.shape, (126,))
        np.testing.assert_array_equal(reset_state, fresh_state)
        self.assertEqual(reused.fov_ema, fresh.fov_ema)

    def test_reset_clears_only_episode_packet_and_observation_state(self):
        engine = PacketEngine(num_uav=16, step_time=0.25)
        norm_cfg = {"ema_alpha": 0.7, "sentinel": 123}
        engine.norm_cfg = norm_cfg
        engine.fov_ema = {
            4: {"overlap": 0.5, "unvisited": 0.4, "frontier": 0.3}
        }

        engine.reset_packet_state()

        self.assertEqual(engine.fov_ema, {})
        self.assertIs(engine.norm_cfg, norm_cfg)


if __name__ == "__main__":
    unittest.main()
