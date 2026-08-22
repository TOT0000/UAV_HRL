from experiment_config import (
    FORMAL_EXPERIMENT_DEFAULTS,
    MOVEMENT_EXPLORATION_DECAY_EPISODES,
    ROUTING_EPSILON_DECAY_EPISODES,
    ROUTING_EPSILON_END,
    ROUTING_EPSILON_START,
)


_MOVEMENT_HYPERPARAMETERS = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]
MOVEMENT_NOISE_START = _MOVEMENT_HYPERPARAMETERS["exploration_noise_start"]
MOVEMENT_NOISE_END = _MOVEMENT_HYPERPARAMETERS["exploration_noise_end"]
TD3_NOISE_START = MOVEMENT_NOISE_START
TD3_NOISE_END = MOVEMENT_NOISE_END
DDQN_EPSILON_START = ROUTING_EPSILON_START
DDQN_EPSILON_END = ROUTING_EPSILON_END


def _linear_decay(start, end, step, decay_steps):
    if step < 0:
        raise ValueError(f"schedule step must be non-negative, got {step}")
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be positive, got {decay_steps}")
    fraction = min(float(step) / float(decay_steps), 1.0)
    return float(max(end, start - (start - end) * fraction))


def movement_behavior_noise(post_warmup_transition, decay_steps, evaluation=False):
    if evaluation:
        return 0.0
    return _linear_decay(
        MOVEMENT_NOISE_START,
        MOVEMENT_NOISE_END,
        post_warmup_transition,
        decay_steps,
    )


def td3_behavior_noise(post_warmup_transition, decay_steps, evaluation=False):
    """Backward-compatible name for the shared TD3/DDPG schedule."""

    return movement_behavior_noise(
        post_warmup_transition, decay_steps, evaluation=evaluation
    )


def ddqn_epsilon(post_warmup_routing_slot, decay_steps, evaluation=False):
    if evaluation:
        return 0.0
    return _linear_decay(
        DDQN_EPSILON_START,
        DDQN_EPSILON_END,
        post_warmup_routing_slot,
        decay_steps,
    )


def evaluation_exploration_settings():
    return {
        "td3_behavior_noise": 0.0,
        "ddqn_epsilon": 0.0,
        "ddqn_logits_noise_std": 0.0,
    }


def movement_decay_steps(
    episode_seconds,
    movement_interval_seconds=1.0,
    decay_episodes=MOVEMENT_EXPLORATION_DECAY_EPISODES,
):
    transitions_per_episode = int(
        round(float(episode_seconds) / float(movement_interval_seconds))
    )
    if transitions_per_episode <= 0 or int(decay_episodes) <= 0:
        raise ValueError("movement decay horizon must be positive")
    return int(decay_episodes) * transitions_per_episode


def td3_decay_steps(total_episodes, episode_seconds, warmup_transitions):
    """Backward-compatible name for the transition-derived movement horizon."""

    del total_episodes, warmup_transitions
    return movement_decay_steps(episode_seconds)


def ddqn_decay_steps(
    episode_seconds,
    routing_slot_seconds=0.25,
    decay_episodes=ROUTING_EPSILON_DECAY_EPISODES,
):
    slots_per_episode = int(
        round(float(episode_seconds) / float(routing_slot_seconds))
    )
    if slots_per_episode <= 0 or int(decay_episodes) <= 0:
        raise ValueError("routing decay horizon must be positive")
    return int(decay_episodes) * slots_per_episode
