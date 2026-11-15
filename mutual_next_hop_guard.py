"""Evaluation-only sequential routing mask guard.

The guard changes only the external action mask supplied to an existing routing
policy.  It is deterministic, consumes no random numbers, and never mutates the
effective masks produced by the packet engine.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from DDQN import forwarding_or_forced_wait_mask


MUTUAL_NEXT_HOP_GUARD_VERSION = "evaluation-mrv-no-mutual-v1"
MUTUAL_NEXT_HOP_GUARD_SPEC = {
    "scope": "evaluation_only",
    "ordering": "minimum_remaining_forwarding_candidates",
    "tie_break": "rotating_uav_order_from_slot_index",
    "constraint": "exclude_i_to_j_when_selected_action_j_equals_i",
    "forced_wait": "self_only_when_no_forwarding_candidate",
    "cycle_scope": "two_node_only",
}
MUTUAL_NEXT_HOP_GUARD_FINGERPRINT = hashlib.sha256(
    json.dumps(
        MUTUAL_NEXT_HOP_GUARD_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

EVALUATION_ROUTING_STRATEGY_SCHEMA_VERSION = "evaluation-routing-strategy-v1"
LEGACY_EVALUATION_ROUTING_STRATEGY_VERSION = "evaluation-fixed-order-independent-v1"
LEGACY_EVALUATION_ROUTING_STRATEGY_SPEC = {
    "scope": "evaluation_only",
    "ordering": "fixed_uav_id_order",
    "constraint": "independent_action_selection",
    "forced_wait": "self_only_when_no_forwarding_candidate",
}
LEGACY_EVALUATION_ROUTING_STRATEGY_FINGERPRINT = hashlib.sha256(
    json.dumps(
        LEGACY_EVALUATION_ROUTING_STRATEGY_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def evaluation_routing_strategy_info(enabled):
    """Return stable evaluation-policy provenance for both guard modes."""

    enabled = normalize_mutual_next_hop_guard(enabled)
    return {
        "schema_version": EVALUATION_ROUTING_STRATEGY_SCHEMA_VERSION,
        "scope": "evaluation_only",
        "enabled": bool(enabled),
        "mutual_next_hop_guard_enabled": bool(enabled),
        "strategy_version": (
            MUTUAL_NEXT_HOP_GUARD_VERSION
            if enabled
            else LEGACY_EVALUATION_ROUTING_STRATEGY_VERSION
        ),
        "strategy_fingerprint": (
            MUTUAL_NEXT_HOP_GUARD_FINGERPRINT
            if enabled
            else LEGACY_EVALUATION_ROUTING_STRATEGY_FINGERPRINT
        ),
        "strategy": (
            dict(MUTUAL_NEXT_HOP_GUARD_SPEC)
            if enabled
            else dict(LEGACY_EVALUATION_ROUTING_STRATEGY_SPEC)
        ),
    }


def normalize_mutual_next_hop_guard(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise ValueError("mutual-next-hop-guard must be 'off' or 'on'")


def forwarding_candidate_count(mask, sender_uav_id):
    values = np.asarray(mask, dtype=bool).copy()
    if 0 <= int(sender_uav_id) < len(values):
        values[int(sender_uav_id)] = False
    return int(np.count_nonzero(values))


def guarded_effective_mask(
    original_effective_mask,
    sender_uav_id,
    selected_actions,
):
    """Return a copied effective mask after all current-slot mutual exclusions."""

    sender = int(sender_uav_id)
    result = np.asarray(original_effective_mask, dtype=bool).copy()
    excluded = []
    for selected_sender, selected_action in selected_actions.items():
        selected_sender = int(selected_sender)
        if int(selected_action) != sender:
            continue
        if selected_sender < 0 or selected_sender >= len(result):
            raise ValueError("selected sender is outside the routing action space")
        if bool(result[selected_sender]):
            result[selected_sender] = False
            excluded.append(selected_sender)
    return result, tuple(sorted(excluded))


def actual_selection_mask(effective_mask, sender_uav_id):
    values = np.asarray(effective_mask, dtype=bool)
    return forwarding_or_forced_wait_mask(values, sender_uav_id, len(values))


def rotating_priority(sender_uav_id, slot_index, num_uav):
    if int(num_uav) <= 0:
        raise ValueError("num_uav must be positive")
    start = int(slot_index) % int(num_uav)
    return (int(sender_uav_id) - start) % int(num_uav)


def assert_no_mutual_next_hops(actions, num_uav):
    for sender in range(int(num_uav)):
        receiver = int(actions[sender])
        if receiver < 0 or receiver >= int(num_uav) or receiver == sender:
            continue
        if int(actions[receiver]) == sender:
            raise AssertionError(
                f"mutual next-hop guard failed: {sender}->{receiver} and "
                f"{receiver}->{sender}"
            )
