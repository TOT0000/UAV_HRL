import numpy as np

from Channel_model import (
    CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
    CHANNEL_MODEL_VERSION,
)


def initialized_channel_lifecycle_state(num_uav=10, num_sr=2):
    """Small valid terminal channel snapshot for checkpoint contract tests."""

    return {
        "channel_model_version": CHANNEL_MODEL_VERSION,
        "channel_environment_contract_version": (
            CHANNEL_ENVIRONMENT_CONTRACT_VERSION
        ),
        "namespace": "training",
        "episode_identity": "fixture",
        "num_uav": int(num_uav),
        "num_sr": int(num_sr),
        "gs_id": int(num_uav),
        "movement_interval_index": 0,
        "routing_slot_index": None,
        "u2g_los_state": np.ones(int(num_uav), dtype=bool),
        "s2u_los_state": np.ones((int(num_sr), int(num_uav)), dtype=bool),
        "u2g_los_probability": np.ones(int(num_uav), dtype=float),
        "s2u_los_probability": np.ones(
            (int(num_sr), int(num_uav)), dtype=float
        ),
        "profile_keys": [],
        "gain_matrix": None,
        "large_scale_draw_count": int(num_uav) * (int(num_sr) + 1),
        "small_scale_normal_draw_count": 0,
        "profile_generation_count": 0,
        "checkpoint_scope": "current-channel-lifecycle-snapshot",
    }
