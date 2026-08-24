import unittest

import numpy as np

from DDQN import DDQN
from experiment_config import (
    METHOD_REGISTRY,
    MethodSpec,
    exploration_schedule_configuration,
    movement_agent_configuration,
    routing_agent_configuration,
)
from HRL_task_aware import _select_routing_actions, formal_training_config
from routing_agents import ControlledDQN, RandomRoutingController, create_routing_agent
from routing_lifecycle import RoutingLearnerLifecycle
from utils_update_v2 import ReplayBufferDiscrete


class SizedReplay:
    def __init__(self, size=0):
        self.size = int(size)


class CountingRoutingAgent:
    routing_agent_kind = "dqn"

    def __init__(self):
        self.num_training = 0
        self.target_update_count = 0
        self.reward_optimizer_update_count = 0
        self.reward_target_update_count = 0

    def train(self, replay, batch_size):
        del replay, batch_size
        self.num_training += 1
        self.reward_optimizer_update_count += 1

    def update_target(self):
        self.target_update_count += 1
        self.reward_target_update_count += 1


def routing_replay(state_dim=42, action_dim=3):
    replay = ReplayBufferDiscrete(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=8,
        n_step=1,
        gamma=0.99,
    )
    state = np.zeros(state_dim, dtype=np.float32)
    state[0] = 1.0
    state[10 : 10 + action_dim] = 1.0
    replay.add(state, 0, state, reward=1.0, cost=0.5, done=False)
    return replay


class RoutingCadenceTest(unittest.TestCase):
    def test_four_slot_boundaries_are_global_not_per_uav(self):
        lifecycle = RoutingLearnerLifecycle(warmup_transitions=1)
        agent = CountingRoutingAgent()
        replay = SizedReplay(size=16)

        updates = []
        for _ in range(8):
            updates.append(lifecycle.complete_slot(agent, replay, batch_size=64))

        self.assertEqual(updates, [False, False, False, True] * 2)
        self.assertEqual(agent.num_training, 2)
        self.assertEqual(agent.target_update_count, 2)
        self.assertEqual(lifecycle.optimizer_update_count, 2)
        self.assertEqual(lifecycle.target_update_count, 2)

    def test_replay_warmup_waits_for_next_fixed_boundary(self):
        lifecycle = RoutingLearnerLifecycle(warmup_transitions=1000)
        agent = CountingRoutingAgent()
        replay = SizedReplay()

        observed = []
        for size in (300, 700, 1000, 1000):
            replay.size = size
            observed.append(lifecycle.complete_slot(agent, replay, batch_size=64))

        self.assertEqual(observed, [False, False, False, True])
        self.assertEqual(lifecycle.epsilon_decay_start_slot, 3)
        self.assertEqual(agent.num_training, 1)

    def test_warmup_and_epsilon_marker_use_replay_completion(self):
        lifecycle = RoutingLearnerLifecycle(warmup_transitions=1000)
        agent = CountingRoutingAgent()
        replay = SizedReplay(999)

        self.assertEqual(lifecycle.epsilon(100), 1.0)
        lifecycle.complete_slot(agent, replay, batch_size=64)
        replay.size = 1000
        lifecycle.complete_slot(agent, replay, batch_size=64)
        self.assertEqual(lifecycle.epsilon_decay_start_slot, 2)
        self.assertEqual(lifecycle.epsilon(100), 1.0)
        lifecycle.complete_slot(agent, replay, batch_size=64)
        self.assertAlmostEqual(lifecycle.epsilon(100), 0.9905)

    def test_same_slot_shares_one_epsilon_across_active_uavs(self):
        calls = []

        class RecordingAgent:
            def select_action(self, state, uav_id, mask, **kwargs):
                del state
                calls.append((uav_id, kwargs["epsilon"]))
                return int(np.flatnonzero(mask)[0])

        states = {7: np.zeros(2), 1: np.zeros(2), 4: np.zeros(2)}
        masks = {uid: np.array([False, True]) for uid in states}
        _select_routing_actions(RecordingAgent(), states, masks, epsilon=0.625)
        self.assertEqual(calls, [(1, 0.625), (4, 0.625), (7, 0.625)])

    def test_safe_ddqn_updates_reward_cost_and_both_targets_once(self):
        agent = DDQN(42, 3, hidden_dim=8)
        lifecycle = RoutingLearnerLifecycle(warmup_transitions=1)
        replay = routing_replay()
        for _ in range(4):
            lifecycle.complete_slot(agent, replay, batch_size=1)
        self.assertEqual(agent.reward_optimizer_update_count, 1)
        self.assertEqual(agent.cost_optimizer_update_count, 1)
        self.assertEqual(agent.reward_target_update_count, 1)
        self.assertEqual(agent.cost_target_update_count, 1)
        self.assertEqual(agent.cost_multiplier_update_count, 0)

    def test_controlled_dqn_updates_reward_and_target_only(self):
        agent = ControlledDQN(42, 3, hidden_dim=8)
        lifecycle = RoutingLearnerLifecycle(warmup_transitions=1)
        replay = routing_replay()
        for _ in range(4):
            lifecycle.complete_slot(agent, replay, batch_size=1)
        self.assertEqual(agent.reward_optimizer_update_count, 1)
        self.assertEqual(agent.reward_target_update_count, 1)
        self.assertFalse(hasattr(agent, "cost_network"))
        self.assertFalse(hasattr(agent, "lambda_cost"))

    def test_non_boundary_state_round_trip_preserves_next_update(self):
        replay = SizedReplay(1000)
        first = RoutingLearnerLifecycle(
            global_slot_count=6,
            optimizer_update_count=1,
            target_update_count=1,
            epsilon_decay_start_slot=3,
            last_optimizer_update_slot=4,
        )
        restored = RoutingLearnerLifecycle.from_state(first.state_dict())
        agent = CountingRoutingAgent()
        agent.num_training = 1
        agent.target_update_count = 1
        agent.reward_optimizer_update_count = 1
        agent.reward_target_update_count = 1
        self.assertFalse(restored.complete_slot(agent, replay, 64))
        self.assertTrue(restored.complete_slot(agent, replay, 64))
        self.assertEqual(restored.global_slot_count, 8)

    def test_incomplete_or_inconsistent_checkpoint_state_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            RoutingLearnerLifecycle.from_state({})
        state = RoutingLearnerLifecycle(global_slot_count=3).state_dict()
        state["routing_update_phase"] = 0
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            RoutingLearnerLifecycle.from_state(state)


class StrategyScopeTest(unittest.TestCase):
    def test_every_registry_method_uses_strategy_driven_scope(self):
        config = formal_training_config(1500)
        self.assertEqual(len(METHOD_REGISTRY), 16)
        for method_key in METHOD_REGISTRY:
            with self.subTest(method=method_key):
                method = MethodSpec.parse(method_key)
                movement = movement_agent_configuration(method, config)
                routing = routing_agent_configuration(method, config)
                exploration = exploration_schedule_configuration(config, method)
                self.assertEqual(
                    movement["behavior_exploration_enabled"],
                    method.agent in {"td3", "ddpg"},
                )
                self.assertEqual(
                    routing["routing_learner_enabled"],
                    method.routing in {"safe_ddqn", "dqn"},
                )
                self.assertEqual(
                    exploration["routing_epsilon_enabled"],
                    method.routing in {"safe_ddqn", "dqn"},
                )
                if method.routing == "random":
                    agent = create_routing_agent(method, 42, 3)
                    self.assertIsInstance(agent, RandomRoutingController)
                    self.assertFalse(hasattr(agent, "q_network"))
                    self.assertIsNone(routing["batch_size"])
                elif method.routing == "dqn":
                    agent = create_routing_agent(method, 42, 3)
                    self.assertIsInstance(agent, ControlledDQN)
                    self.assertFalse(hasattr(agent, "cost_network"))
                else:
                    self.assertEqual(method.routing, "safe_ddqn")

    def test_assignment_objective_and_observation_do_not_change_cadence(self):
        config = formal_training_config(1500)
        keys = (
            "td3_dinkelbach",
            "td3_ratio",
            "km_td3_dinkelbach",
            "random_assignment_td3_dinkelbach",
            "td3_dinkelbach_wo_ta",
        )
        configs = [
            routing_agent_configuration(MethodSpec.parse(key), config)
            for key in keys
        ]
        lifecycle_fields = (
            "routing_optimizer_update_scope",
            "routing_update_interval_slots",
            "routing_gradient_steps_per_update",
            "routing_warmup_transitions",
        )
        self.assertEqual(
            [{field: item[field] for field in lifecycle_fields} for item in configs],
            [{field: configs[0][field] for field in lifecycle_fields}] * len(configs),
        )


if __name__ == "__main__":
    unittest.main()
