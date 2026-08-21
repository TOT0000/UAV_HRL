from experiment_config import FORMAL_EXPERIMENT_DEFAULTS


_MOVEMENT_HYPERPARAMETERS = FORMAL_EXPERIMENT_DEFAULTS["movement_hyperparameters"]
MOVEMENT_NOISE_START = _MOVEMENT_HYPERPARAMETERS["exploration_noise_start"]
MOVEMENT_NOISE_END = _MOVEMENT_HYPERPARAMETERS["exploration_noise_end"]
TD3_NOISE_START = MOVEMENT_NOISE_START
TD3_NOISE_END = MOVEMENT_NOISE_END
DDQN_EPSILON_START = 1.0
DDQN_EPSILON_END = 0.05
FORMAL_DECAY_FRACTION = 0.80


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


def ddqn_epsilon(global_routing_slot, decay_steps, evaluation=False):
    if evaluation:
        return 0.0
    return _linear_decay(
        DDQN_EPSILON_START,
        DDQN_EPSILON_END,
        global_routing_slot,
        decay_steps,
    )


def evaluation_exploration_settings():
    return {
        "td3_behavior_noise": 0.0,
        "ddqn_epsilon": 0.0,
        "ddqn_logits_noise_std": 0.0,
    }


def movement_decay_steps(total_episodes, episode_seconds, warmup_transitions):
    total_transitions = int(total_episodes) * int(episode_seconds)
    post_warmup = max(total_transitions - int(warmup_transitions), 0)
    return max(1, int(FORMAL_DECAY_FRACTION * post_warmup))


def td3_decay_steps(total_episodes, episode_seconds, warmup_transitions):
    """Backward-compatible name for the transition-derived movement horizon."""

    return movement_decay_steps(
        total_episodes, episode_seconds, warmup_transitions
    )


def ddqn_decay_steps(total_episodes, episode_seconds, slots_per_interval):
    total_slots = (
        int(total_episodes) * int(episode_seconds) * int(slots_per_interval)
    )
    return max(1, int(FORMAL_DECAY_FRACTION * total_slots))
