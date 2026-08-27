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
        "uav_queue_packet_ids": {str(uid): [] for uid in range(NUM_UAV)},
        "sr_queue_packet_ids": {},
        "routing_transition_reference_counts": {},
        "com_session_state": {
            "lifecycle_version": COM_SESSION_LIFECYCLE_VERSION,
            "sessions": {},
        },
        "pending_terminal_violation_events": [],
        "unattributed_transition_violation_count": 0,
    }


def routing_transition_checkpoint_fixture(next_transition_id=0):
    return {
        "routing_transition_state": drained_routing_transition_state(
            next_transition_id
        ),
        "packet_engine_state": terminal_packet_engine_state(),
    }
