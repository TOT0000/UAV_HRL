from pathlib import Path
import inspect

import pytest

from Simulator import Simulator
from Task_assignment import Task
from HRL_task_aware import TrainingConfig, _interval_reward
from centralized_movement import (
    MOVEMENT_STATE_DIM,
    calculate_movement_potentials,
    movement_state_feature_schema,
)
from experiment_config import ASSIGNMENT_CONTRACT_VERSION
from training_checkpoint import CHECKPOINT_SCHEMA_VERSION
from utils_update_v2 import ReplayBufferJoint


@pytest.mark.parametrize(
    "strategy,rounds",
    [("k_km", 2), ("km", 1), ("random_one_to_one", 2)],
)
def test_every_assignment_strategy_materializes_only_service_or_fallback_roles(
    strategy, rounds
):
    env = Simulator(16)
    env.assignment_strategy = strategy
    env.assignment_rounds = rounds
    env.num_GT = 2
    env.reset_environment()
    for gt in env.gts:
        gt.is_found = True
        sr = env.SR_teams[gt.id]
        sr.assigned_gt_id = gt.id
        env.task_list.extend(
            [
                Task(len(env.task_list), "FOV", gt, gt.id),
                Task(len(env.task_list) + 1, "COM", sr, sr.id),
            ]
        )

    env.assign_tasks()

    allowed = {"Search", "FOV", "COM", "Hovering"}
    roles = [
        task["task_type"]
        for tasks in env.multi_tasks.values()
        for task in tasks
    ]
    assert roles
    assert set(roles) <= allowed
    assert all(task.task_type in {"FOV", "COM"} for task in env.task_list)
    assert "relay" not in repr(env.assignment_metadata()).lower()


def test_service_only_observation_potential_and_checkpoint_contracts():
    schema = movement_state_feature_schema()
    names = [feature["name"] for feature in schema["features"]]
    assert MOVEMENT_STATE_DIM == 531
    assert len(names) == MOVEMENT_STATE_DIM
    assert not any("relay" in name.lower() for name in names)
    assert CHECKPOINT_SCHEMA_VERSION == 31
    assert "service-only" in ASSIGNMENT_CONTRACT_VERSION

    env = Simulator(16)
    env.num_GT = 2
    env.reset_environment()
    assert len(calculate_movement_potentials(env, 1.0)) == 3


def test_reward_is_the_previous_reward_minus_relay_shaping_only():
    config = TrainingConfig(total_episodes=1)
    current = (0.2, 0.3, 0.4)
    following = (0.5, 0.6, 0.7)
    reward = _interval_reward(
        delivered_mbits=4.0,
        energy=2.0,
        current_lambda=0.25,
        gamma=1.0,
        potentials_t=current,
        potentials_t1=following,
        done=False,
        config=config,
    )
    unchanged_objective_and_three_potentials = 3.5 + 3.0 * 0.3 * 3
    hypothetical_retired_relay_shaping = 3.0 * (0.8 - 0.1)
    previous_reward = (
        unchanged_objective_and_three_potentials
        + hypothetical_retired_relay_shaping
    )
    assert reward == pytest.approx(
        previous_reward - hypothetical_retired_relay_shaping
    )
    assert "beta_relay" not in inspect.signature(
        ReplayBufferJoint._reward_numpy
    ).parameters
    assert "relay_shaping_enabled" not in inspect.signature(
        _interval_reward
    ).parameters


def test_retired_explicit_role_modules_are_absent():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "relay_contract.py").exists()
    assert not (root / "relay_diagnostics.py").exists()
