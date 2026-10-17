import unittest

import numpy as np

from routing_transition_ledger import RoutingTransitionLedger
from utils_update_v2 import ReplayBufferDiscrete


STATE_DIM = 4
ACTION_DIM = 3
BUFFER_CAPACITY = 3


def _replay():
    return ReplayBufferDiscrete(
        STATE_DIM,
        ACTION_DIM,
        max_size=BUFFER_CAPACITY,
        n_step=1,
        gamma=0.99,
    )


def _add_reward_transitions(replay, count, *, first_transition_id=0):
    for offset in range(int(count)):
        transition_id = int(first_transition_id + offset)
        state = np.full(STATE_DIM, transition_id, dtype=np.float32)
        replay.add(
            state,
            transition_id % ACTION_DIM,
            state + 1.0,
            reward=float(transition_id),
            cost=0.0,
            done=True,
            transition_id=transition_id,
        )


def _commit_packet_path_batch(replay, count, *, first_transition_id=0):
    """Exercise the production ledger-to-path-cost replay write path."""

    ledger = RoutingTransitionLedger(next_transition_id=first_transition_id)
    ledger.begin_episode()
    for offset in range(int(count)):
        packet_id = int(first_transition_id + offset)
        state = np.full(STATE_DIM, packet_id, dtype=np.float32)
        transition_id = ledger.create(
            agent_id=offset % 2,
            state=state,
            action=offset % ACTION_DIM,
            tag_gt=4,
            packet_id=packet_id,
        )
        ledger.set_reward(transition_id, float(offset))
        ledger.finalize_packet(packet_id, violated=bool(offset % 2))
    ledger.finalize_causality({}, {}, terminal=True)
    committed = ledger.commit_ready(replay)
    diagnostics = ledger.episode_diagnostics()
    ledger.assert_drained()
    return committed, diagnostics


class RoutingReplayDiagnosticCounterTest(unittest.TestCase):
    def test_reward_write_delta_handles_capacity_boundaries_and_overwrites(self):
        cases = (
            (0, 2),  # below capacity
            (2, 2),  # crosses capacity
            (3, 2),  # already full
            (3, 8),  # one episode overwrites the buffer multiple times
        )
        for initial_count, episode_count in cases:
            with self.subTest(
                initial_count=initial_count,
                episode_count=episode_count,
            ):
                replay = _replay()
                _add_reward_transitions(replay, initial_count)
                total_added_before = replay.total_added

                _add_reward_transitions(
                    replay,
                    episode_count,
                    first_transition_id=100,
                )

                self.assertEqual(
                    replay.total_added - total_added_before,
                    episode_count,
                )
                self.assertEqual(
                    replay.size,
                    min(initial_count + episode_count, BUFFER_CAPACITY),
                )

    def test_packet_path_cost_delta_uses_formal_write_path_after_overwrites(self):
        cases = (
            (0, 2),
            (2, 2),
            (3, 2),
            (3, 8),
        )
        for initial_count, episode_count in cases:
            with self.subTest(
                initial_count=initial_count,
                episode_count=episode_count,
            ):
                replay = _replay()
                if initial_count:
                    _commit_packet_path_batch(replay, initial_count)
                total_added_before = replay.path_cost_total_added

                committed, diagnostics = _commit_packet_path_batch(
                    replay,
                    episode_count,
                    first_transition_id=100,
                )

                self.assertEqual(len(committed), episode_count)
                self.assertEqual(
                    diagnostics["packet_path_cost_transition_count"],
                    episode_count,
                )
                self.assertEqual(
                    replay.path_cost_total_added - total_added_before,
                    episode_count,
                )
                self.assertEqual(
                    replay.path_cost_size,
                    min(initial_count + episode_count, BUFFER_CAPACITY),
                )

    def test_episode_snapshots_do_not_double_count_previous_writes(self):
        replay = _replay()

        first_before = replay.total_added
        _add_reward_transitions(replay, 5)
        self.assertEqual(replay.total_added - first_before, 5)

        second_before = replay.total_added
        _add_reward_transitions(replay, 2, first_transition_id=100)
        self.assertEqual(replay.total_added - second_before, 2)
        self.assertEqual(replay.size, BUFFER_CAPACITY)

        first_cost_before = replay.path_cost_total_added
        _commit_packet_path_batch(replay, 5, first_transition_id=200)
        self.assertEqual(replay.path_cost_total_added - first_cost_before, 5)

        second_cost_before = replay.path_cost_total_added
        _commit_packet_path_batch(replay, 2, first_transition_id=300)
        self.assertEqual(replay.path_cost_total_added - second_cost_before, 2)
        self.assertEqual(replay.path_cost_size, BUFFER_CAPACITY)


if __name__ == "__main__":
    unittest.main()
