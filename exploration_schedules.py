TD3_NOISE_START = 0.20
TD3_NOISE_END = 0.05
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


def td3_behavior_noise(post_warmup_transition, decay_steps, evaluation=False):
    if evaluation:
        return 0.0
    return _linear_decay(
        TD3_NOISE_START,
        TD3_NOISE_END,
        post_warmup_transition,
        decay_steps,
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


def td3_decay_steps(total_episodes, episode_seconds, warmup_transitions):
    total_transitions = int(total_episodes) * int(episode_seconds)
    post_warmup = max(total_transitions - int(warmup_transitions), 0)
    return max(1, int(FORMAL_DECAY_FRACTION * post_warmup))


def ddqn_decay_steps(total_episodes, episode_seconds, slots_per_interval):
    total_slots = (
        int(total_episodes) * int(episode_seconds) * int(slots_per_interval)
    )
    return max(1, int(FORMAL_DECAY_FRACTION * total_slots))
