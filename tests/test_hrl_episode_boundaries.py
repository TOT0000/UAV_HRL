import unittest

import numpy as np

from HRL_task_aware import (
    MOVEMENT_CONTROL_INTERVAL,
    _create_active_replay_buffers,
    _is_episode_end,
    _is_last_movement_decision,
    _search_transition_done,
)


class ActiveHRLEpisodeBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.routing, self.search, self.fov = _create_active_replay_buffers(
            state_dim=2, routing_dim=2, max_size=8
        )
        self.state = np.zeros(2, dtype=np.float32)
        self.next_state = np.ones(2, dtype=np.float32)
        self.action = np.zeros(3, dtype=np.float32)

    def test_active_buffers_use_one_step_returns(self):
        self.assertEqual(self.routing.n_step, 1)
        self.assertEqual(self.search.n_step, 1)
        self.assertEqual(self.fov.n_step, 1)

    def test_last_movement_decision_is_before_final_slot(self):
        total_slots = 240

        self.assertFalse(_is_last_movement_decision(232, total_slots))
        self.assertTrue(_is_last_movement_decision(236, total_slots))
        self.assertFalse(_is_last_movement_decision(239, total_slots))
        self.assertEqual(MOVEMENT_CONTROL_INTERVAL, 4)

    def test_search_completion_and_movement_boundary_cut_bootstrap(self):
        cases = (
            (False, False, 1.0),
            (True, False, 0.0),
            (False, True, 0.0),
        )
        for search_done, movement_end, expected_not_done in cases:
            with self.subTest(search_done=search_done, movement_end=movement_end):
                self.search.add(
                    self.state,
                    self.action,
                    self.next_state,
                    reward=1.0,
                    done=_search_transition_done(search_done, movement_end),
                )
                self.assertEqual(
                    self.search.not_done[self.search.size - 1, 0], expected_not_done
                )
                self.assertEqual(self.search.n_step_buffer, [])

    def test_fov_and_routing_episode_boundaries_do_not_stage_across_reset(self):
        movement_episode_end = _is_last_movement_decision(236, 240)
        episode_end = _is_episode_end(239, 240)
        self.assertFalse(_is_episode_end(238, 240))

        self.fov.add(
            self.state,
            self.action,
            self.next_state,
            reward=1.0,
            done=movement_episode_end,
        )
        self.routing.add(
            self.state,
            0,
            self.next_state,
            reward=1.0,
            cost=0.5,
            done=episode_end,
        )

        self.assertEqual(self.fov.not_done[0, 0], 0.0)
        self.assertEqual(self.routing.not_done[0, 0], 0.0)
        self.assertEqual(self.fov.n_step_buffer, [])
        self.assertEqual(self.routing.n_step_buffer, [])

        # The first transition after reset is independent and non-terminal.
        self.fov.add(
            self.next_state,
            self.action,
            self.state,
            reward=2.0,
            done=False,
        )
        self.routing.add(
            self.next_state,
            1,
            self.state,
            reward=2.0,
            cost=1.0,
            done=False,
        )

        self.assertEqual(self.fov.not_done[1, 0], 1.0)
        self.assertEqual(self.routing.not_done[1, 0], 1.0)
        self.assertEqual(self.fov.n_step_buffer, [])
        self.assertEqual(self.routing.n_step_buffer, [])


if __name__ == "__main__":
    unittest.main()
