from experiment_config import FOV_EMA_LIFECYCLE_VERSION, NUM_UAV


def initialized_fov_ema_state(marker="fixture-map-transition"):
    return {
        "lifecycle_version": FOV_EMA_LIFECYCLE_VERSION,
        "values": {
            str(uav_id): {
                "overlap": 0.0,
                "unvisited": 0.0,
                "frontier": 0.0,
            }
            for uav_id in range(NUM_UAV)
        },
        "initialized_uav_ids": list(range(NUM_UAV)),
        "previous_footprints": {
            str(uav_id): [0, 0, 0, 0] for uav_id in range(NUM_UAV)
        },
        "transition_marker": str(marker),
        "footprint_transition_marker": str(marker),
        "update_count": 1,
    }
