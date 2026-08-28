"""Bounded or streaming persistence for per-packet episode outcomes."""

from __future__ import annotations

import json
from pathlib import Path


PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION = "uav-hrl-packet-outcomes-jsonl-v3"
PACKET_OUTCOME_MODE_DISABLED = "disabled"
PACKET_OUTCOME_MODE_BOUNDED = "bounded_memory"
PACKET_OUTCOME_MODE_STREAMING = "stream_jsonl"
PACKET_OUTCOME_ARTIFACT_MODES = frozenset(
    {
        PACKET_OUTCOME_MODE_DISABLED,
        PACKET_OUTCOME_MODE_BOUNDED,
        PACKET_OUTCOME_MODE_STREAMING,
    }
)
MAX_BOUNDED_PACKET_OUTCOME_EPISODES = 16


def packet_outcome_episode_record(scenario_id, summary, packet_outcomes):
    """Build one traceable episode record without copying packet dictionaries."""

    if not isinstance(packet_outcomes, list):
        raise TypeError("packet outcomes must be an episode-local list")
    if not isinstance(summary, dict):
        raise TypeError("packet outcome summary must be a dictionary")
    return {
        "artifact_schema_version": PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "summary": summary,
        "packet_outcomes": packet_outcomes,
    }


class PacketOutcomeJsonlWriter:
    """Write and flush one complete episode per JSONL record."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None
        self.episode_count = 0

    @property
    def closed(self):
        return self._handle is None or self._handle.closed

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("packet outcome writer is already open")
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def write_episode(self, record):
        if self._handle is None or self._handle.closed:
            raise RuntimeError("packet outcome writer is not open")
        if not isinstance(record, dict):
            raise TypeError("packet outcome episode record must be a dictionary")
        if record.get("artifact_schema_version") != (
            PACKET_OUTCOME_ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError("packet outcome artifact schema version is invalid")
        for field in ("scenario_id", "summary", "packet_outcomes"):
            if field not in record:
                raise ValueError(
                    f"packet outcome episode record lacks required field: {field}"
                )
        json.dump(
            record,
            self._handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()
        self.episode_count += 1

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._handle.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
