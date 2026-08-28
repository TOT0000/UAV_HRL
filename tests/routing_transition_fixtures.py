from experiment_config import COM_SESSION_LIFECYCLE_VERSION, NUM_UAV
from Packet_scheduler_v1 import PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION
from routing_transition_ledger import RoutingTransitionLedger


def drained_routing_transition_state(next_transition_id=0):
    return RoutingTransitionLedger(next_transition_id).state_dict()


def terminal_packet_engine_state(*, next_packet_id=0):
    return {
        "schema_version": PACKET_ENGINE_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_scope": "episode_boundary_terminal_snapshot",
        "mid_episode_checkpoint_supported": False,
        "next_packet_id": int(next_packet_id),
        "active_packets": [],
        "inject_buffer": {},
        "source_buffer": {},
        "uav_queue_packet_ids": {str(uid): [] for uid in range(NUM_UAV)},
        "sr_queue_packet_ids": {},
        "routing_transition_reference_counts": {},
        "com_session_state": {
            "lifecycle_version": COM_SESSION_LIFECYCLE_VERSION,
            "sessions": {},
        },
        "generated_packet_counts": {"FOV": 0, "COM": 0},
        "eligible_packet_counts": {"FOV": 0, "COM": 0},
        "raw_final_hop_bits": 0.0,
        "timely_goodput_bits": 0.0,
        "fov_generated_raw_bits": 0.0,
        "fov_timely_delivered_raw_bits": 0.0,
        "fov_timely_useful_bits": 0.0,
        "fov_capture_coverage_sum": 0.0,
        "fov_capture_coverage_count": 0,
        "fov_zero_coverage_packet_count": 0,
        "com_timely_delivered_bits": 0.0,
        "total_timely_useful_bits": 0.0,
        "pending_terminal_violation_events": [],
        "system_qos_eligible_packet_count": 0,
        "system_qos_violation_count": 0,
        "routing_credit_eligible_packet_count": 0,
        "routing_credit_violation_count": 0,
        "replay_attributed_violation_cost_count": 0.0,
        "unattributed_transition_violation_count": 0,
        "unattributed_pre_routing_violation_count": 0,
    }


def routing_transition_checkpoint_fixture(next_transition_id=0):
    return {
        "routing_transition_state": drained_routing_transition_state(
            next_transition_id
        ),
        "packet_engine_state": terminal_packet_engine_state(),
    }
