import unittest

import numpy as np

from Packet_scheduler_v1 import PacketEngine
from Simulator import Simulator
from centralized_movement import (
    MOVEMENT_STATE_DIM,
    aggregate_coverage_map,
    calculate_movement_potentials,
    get_global_movement_state,
)


class CoverageAggregationTest(unittest.TestCase):
    def test_false_true_and_shape(self):
        false_map = np.zeros((500, 500), dtype=bool)
        true_map = np.ones((500, 500), dtype=bool)
        self.assertEqual(aggregate_coverage_map(false_map).shape, (256,))
        self.assertTrue(np.all(aggregate_coverage_map(false_map) == 0.0))
        self.assertTrue(np.all(aggregate_coverage_map(true_map) == 1.0))

    def test_partial_map_uses_row_major_macro_cell_and_loses_no_cells(self):
        bitmap = np.zeros((500, 500), dtype=bool)
        row_groups = np.array_split(np.arange(500), 16)
        col_groups = np.array_split(np.arange(500), 16)
        bitmap[np.ix_(row_groups[2], col_groups[3])] = True
        macro = aggregate_coverage_map(bitmap).reshape(16, 16)
        self.assertEqual(macro[2, 3], 1.0)
        self.assertEqual(np.count_nonzero(macro), 1)

        recovered_true = 0.0
        recovered_cells = 0
        for row_index, rows in enumerate(row_groups):
            for col_index, cols in enumerate(col_groups):
                cell_count = rows.size * cols.size
                recovered_true += macro[row_index, col_index] * cell_count
                recovered_cells += cell_count
        self.assertEqual(recovered_cells, bitmap.size)
        self.assertAlmostEqual(recovered_true, float(bitmap.sum()))


class CentralizedMovementStateTest(unittest.TestCase):
    def setUp(self):
        self.env = Simulator(num_UAV=10)
        self.env.num_GT = 2
        self.env.reset_environment()
        self.packet_engine = PacketEngine(num_uav=10, step_time=0.25)

    def test_state_is_429_finite_side_effect_free_and_routing_is_101(self):
        positions_before = [uav.get_position() for uav in self.env.UAVs]
        tasks_before = {
            uid: [dict(task) for task in tasks]
            for uid, tasks in self.env.multi_tasks.items()
        }
        bitmap_before = self.env.visited_bitmap.copy()
        random_before = np.random.get_state()

        state = get_global_movement_state(
            self.env,
            self.packet_engine,
            self.packet_engine.backlog_bits,
            1.0,
            remaining_time=0.75,
        )

        self.assertEqual(state.shape, (MOVEMENT_STATE_DIM,))
        self.assertTrue(np.isfinite(state).all())
        self.assertEqual(state[428], 0.75)
        self.assertEqual(positions_before, [uav.get_position() for uav in self.env.UAVs])
        self.assertEqual(tasks_before, self.env.multi_tasks)
        np.testing.assert_array_equal(bitmap_before, self.env.visited_bitmap)
        random_after = np.random.get_state()
        self.assertEqual(random_before[0], random_after[0])
        np.testing.assert_array_equal(random_before[1], random_after[1])
        self.assertEqual(random_before[2:], random_after[2:])

        routing_state = self.packet_engine.get_state_ta(
            self.env, 0, backlog_bits=self.packet_engine.backlog_bits
        )
        self.assertEqual(routing_state.shape, (101,))

    def test_search_potential_is_full_boolean_map_mean(self):
        self.env.visited_bitmap[:] = False
        self.env.visited_bitmap[:100, :250] = True
        phi_search, _, _ = calculate_movement_potentials(self.env, 1.0)
        self.assertAlmostEqual(phi_search, self.env.visited_bitmap.mean())

    def test_duplicate_fov_and_com_targets_fail_fast(self):
        for task_type, targets in (
            ("FOV", self.env.gts),
            ("COM", self.env.SR_teams),
        ):
            with self.subTest(task_type=task_type):
                self.env.multi_tasks[0] = [
                    {
                        "task_type": task_type,
                        "target_id": index,
                        "target_obj_id": index,
                        "target_pos": target.get_position(),
                    }
                    for index, target in enumerate(targets[:2])
                ]
                with self.assertRaisesRegex(
                    ValueError, rf"UAV 0 has duplicate {task_type} targets: \[0, 1\]"
                ):
                    get_global_movement_state(
                        self.env,
                        self.packet_engine,
                        self.packet_engine.backlog_bits,
                        1.0,
                        remaining_time=1.0,
                    )

    def test_remaining_time_is_explicit_and_terminal_state_is_zero(self):
        start = get_global_movement_state(
            self.env,
            self.packet_engine,
            self.packet_engine.backlog_bits,
            1.0,
            remaining_time=1.0,
        )
        terminal = get_global_movement_state(
            self.env,
            self.packet_engine,
            self.packet_engine.backlog_bits,
            1.0,
            remaining_time=0.0,
        )
        self.assertEqual(start[428], 1.0)
        self.assertEqual(terminal[428], 0.0)
        with self.assertRaisesRegex(ValueError, "remaining_time"):
            get_global_movement_state(
                self.env,
                self.packet_engine,
                self.packet_engine.backlog_bits,
                1.0,
                remaining_time=1.01,
            )


if __name__ == "__main__":
    unittest.main()
