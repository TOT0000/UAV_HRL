import unittest
from types import SimpleNamespace

import numpy as np

from object import SRTeam, UAV, straight_line_route
from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator


class CommunicationSafetyLifecycleTest(unittest.TestCase):
    def test_identical_physical_states_apply_identical_safety_for_all_uav_ids(self):
        mobility = {
            "comm_safety": {
                "enable": True,
                "mode": "gs_only",
                "gs_pos": (0.0, 0.0, 0.0),
                "r_soft": 180.0,
                "r_hard": 200.0,
            }
        }
        proposals = []
        for uav_id in (0, 2, 7, 15):
            uav = UAV(uav_id, 190.0, 0.0, 50.0)
            proposals.append(
                uav.propose_movement(
                    10.0,
                    0.0,
                    0.0,
                    mobility_params=mobility,
                )["new_position"]
            )
        for proposal in proposals[1:]:
            np.testing.assert_allclose(proposal, proposals[0])


class FovEmaLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=16)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.env.update_source_uavs()
        self.env.update_u2u_channels()
        self.env.update_u2g_channels()
        self.engine = PacketEngine(num_uav=16)

    def _state(self, engine=None):
        engine = engine or self.engine
        return engine.get_state_ta(
            self.env,
            0,
            action_mask=self.env.get_routing_action_mask(0),
        )

    def test_getters_are_pure_and_map_transition_updates_once(self):
        before = self.engine.fov_ema_state()
        first = self._state()
        second = self._state()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.engine.fov_ema_state(), before)

        self.env.visited_bitmap[0, 0] = True
        self.assertTrue(self.engine.update_fov_ema(self.env, "map-transition-1"))
        after_update = self.engine.fov_ema_state()
        self.assertEqual(after_update["update_count"], 1)
        self.assertFalse(self.engine.update_fov_ema(self.env, "map-transition-1"))
        self._state()
        self._state()
        self.assertEqual(self.engine.fov_ema_state(), after_update)

    def test_checkpoint_round_trip_preserves_exact_ema_sequence_state(self):
        self.env.visited_bitmap[0, 0] = True
        self.engine.update_fov_ema(self.env, "map-transition-1")
        saved = self.engine.fov_ema_state()
        restored = PacketEngine(num_uav=16)
        restored.load_fov_ema_state(saved)
        self.assertEqual(restored.fov_ema_state(), saved)
        np.testing.assert_array_equal(self._state(self.engine), self._state(restored))

    def test_hol_slack_uses_instance_deadline_override(self):
        overridden = PacketEngine(
            num_uav=16,
            task_deadlines_seconds={"FOV": 0.5, "COM": 1.0},
        )
        overridden.create_packet(0, "FOV", 100.0, 0.0)
        self.env.current_time = 0.25
        state = self._state(overridden)
        self.assertAlmostEqual(state[-2], 0.5)


class SrRouteLifecycleTest(unittest.TestCase):
    def test_common_simulator_uses_the_shared_route_and_no_duplicate_samples(self):
        env = Simulator(num_UAV=16)
        env.num_GT = 2
        env.reset_environment()
        for index, team in enumerate(env.SR_teams):
            team.x, team.y, team.z = (0.0, 0.0, 0.0) if index == 0 else (100.0, 100.0, 0.0)
            env.sr_trajectory[team.id] = [[team.x, team.y, team.z]]
        target = SimpleNamespace(id=0, x=2.5, y=0.0, assigned=False)
        assigned = env.SR_team_gogo(target)
        self.assertIs(assigned, env.SR_teams[0])
        self.assertEqual(assigned.path[-1], (2.5, 0.0))
        env.advance_sr_teams()
        saved = env.sr_route_state()
        restored = Simulator(num_UAV=16)
        restored.num_GT = 2
        restored.reset_environment()
        restored.load_sr_route_state(saved)
        self.assertEqual(restored.sr_route_state(), saved)
        while assigned.active:
            env.advance_sr_teams()
            restored.advance_sr_teams()
            self.assertEqual(restored.sr_route_state(), env.sr_route_state())
        self.assertEqual(assigned.get_position(), (2.5, 0.0, 0.0))
        trajectory = env.sr_trajectory[assigned.id]
        self.assertTrue(
            all(first != second for first, second in zip(trajectory, trajectory[1:]))
        )

    def test_short_route_has_no_duplicate_start_and_arrives_on_first_update(self):
        team = SRTeam(0)
        team.assign_mission(7, (0.5, 0.0), speed=1.0)
        self.assertEqual(team.path, [(0.5, 0.0)])
        self.assertNotEqual(team.path[0], (0.0, 0.0))
        team.step_forward()
        self.assertEqual(team.get_position(), (0.5, 0.0, 0.0))
        self.assertTrue(team.arrived)
        self.assertFalse(team.active)

    def test_multi_step_route_ends_exactly_and_resume_keeps_cursor(self):
        self.assertEqual(
            straight_line_route((0.0, 0.0), (2.5, 0.0), speed=1.0),
            [(1.0, 0.0), (2.0, 0.0), (2.5, 0.0)],
        )
        original = SRTeam(3)
        original.assign_mission(9, (2.5, 0.0), speed=1.0)
        original.step_forward()
        saved = original.route_state()
        restored = SRTeam(3)
        restored.load_route_state(saved)
        self.assertEqual(restored.route_state(), saved)

        while original.active:
            original.step_forward()
            restored.step_forward()
            self.assertEqual(restored.route_state(), original.route_state())
        self.assertEqual(original.get_position(), (2.5, 0.0, 0.0))
        self.assertTrue(original.arrived)


if __name__ == "__main__":
    unittest.main()
