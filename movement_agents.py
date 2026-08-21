"""Movement-agent factory shared by training, evaluation, and smoke runs."""

import numpy as np

from centralized_ddpg import CentralizedDDPG, RandomMovementController
from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    movement_agent_configuration,
)
from td3 import TD3


def sample_random_joint_action(action_dim):
    """Sample the common continuous joint-action domain with the seeded RNG."""

    return np.random.uniform(-1.0, 1.0, size=int(action_dim)).astype(np.float32)


def create_movement_agent(method_spec, state_dim, action_dim, config):
    hyperparameters = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]
    effective = movement_agent_configuration(method_spec)
    common = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": hyperparameters["max_action"],
        "gamma": hyperparameters["gamma"],
    }
    if method_spec.agent == "td3":
        td3 = hyperparameters["td3"]
        return TD3(
            **common,
            policy_delay=effective["policy_delay"],
            policy_noise=effective["target_policy_noise"],
            noise_clip=effective["target_noise_clip"],
            tau=hyperparameters["tau"],
            actor_lr=hyperparameters["actor_learning_rate"],
            critic_lr=hyperparameters["critic_learning_rate"],
        )
    if method_spec.agent == "ddpg":
        return CentralizedDDPG(
            **common,
            tau=hyperparameters["tau"],
            actor_lr=hyperparameters["actor_learning_rate"],
            critic_lr=hyperparameters["critic_learning_rate"],
        )
    if method_spec.agent == "random":
        return RandomMovementController(gamma=hyperparameters["gamma"])
    raise ValueError(f"unsupported movement agent: {method_spec.agent}")
