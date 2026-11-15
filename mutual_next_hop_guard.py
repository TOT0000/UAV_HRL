"""Sequential routing mask guard shared by training and evaluation.

The guard changes only the external action mask supplied to an existing routing
policy.  It is deterministic, consumes no random numbers, and never mutates the
effective masks produced by the packet engine.
"""

from __future__ import annotations

import numpy as np

from experiment_config import (
    MUTUAL_NEXT_HOP_GUARD_FINGERPRINT,
    MUTUAL_NEXT_HOP_GUARD_SPEC,
    MUTUAL_NEXT_HOP_GUARD_VERSION,
)

import hashlib
import json
from pathlib import Path

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
        "scope": (
            "training_and_evaluation" if enabled else "explicit_legacy_control"
        ),
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


def checkpoint_training_routing_strategy(checkpoint_dir):
    """Read and validate the actual routing strategy of a full checkpoint."""

    metadata_path = Path(checkpoint_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            "full-resume checkpoint lacks routing strategy metadata: "
            f"{metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    experiment = metadata.get("experiment") or {}
    saved = experiment.get("evaluation_routing_strategy")
    if not isinstance(saved, dict):
        raise RuntimeError(
            "full-resume checkpoint lacks the actual mutual-next-hop guard "
            "training contract; start a new run instead of assuming the new default"
        )
    try:
        enabled = saved["mutual_next_hop_guard_enabled"]
    except KeyError as exc:
        raise RuntimeError(
            "full-resume checkpoint routing strategy has no actual enabled state"
        ) from exc
    if not isinstance(enabled, bool):
        raise RuntimeError(
            "full-resume checkpoint routing strategy enabled state is invalid"
        )
    expected = evaluation_routing_strategy_info(enabled)
    required = (
        "schema_version",
        "scope",
        "enabled",
        "mutual_next_hop_guard_enabled",
        "strategy_version",
        "strategy_fingerprint",
        "strategy",
    )
    mismatches = {
        key: (saved.get(key), expected[key])
        for key in required
        if saved.get(key) != expected[key]
    }
    aliases = {
        "mutual_next_hop_guard_enabled": enabled,
        "evaluation_routing_strategy_version": expected["strategy_version"],
        "evaluation_routing_strategy_fingerprint": expected[
            "strategy_fingerprint"
        ],
        "evaluation_routing_strategy_scope": expected["scope"],
    }
    mismatches.update({
        key: (experiment.get(key), value)
        for key, value in aliases.items()
        if experiment.get(key) != value
    })
    if mismatches:
        raise RuntimeError(
            "full-resume checkpoint routing strategy contract is incompatible: "
            f"{mismatches}"
        )
    return expected


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
    from DDQN import forwarding_or_forced_wait_mask

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
