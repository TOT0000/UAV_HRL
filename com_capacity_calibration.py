"""Compatibility metadata for the canonical COM demand-satisfaction scale.

The former sampled P95/raw-capacity normalization has been retired.  COM
features, utilities, and potentials now use min(C_ref / R_required, 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_config import (
    COM_PACKET_RATE_PER_SECOND,
    COM_PACKET_SIZE_BITS,
    COM_REQUIRED_RATE_BPS,
    NUM_UAV,
    REFERENCE_COM_BANDWIDTH_HZ,
    ROI_COUNT_MAX,
)


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "config" / "com_capacity_calibration.json"
COM_NORMALIZATION_SCHEMA = "demand-satisfaction-v1"


def demand_satisfaction_metadata(required_rate_packets_per_second=None):
    packet_rate = (
        COM_PACKET_RATE_PER_SECOND
        if required_rate_packets_per_second is None
        else float(required_rate_packets_per_second)
    )
    required_rate_bps = packet_rate * COM_PACKET_SIZE_BITS
    return {
        "schema": COM_NORMALIZATION_SCHEMA,
        "normalization": "min(reference_capacity_bps / required_rate_bps, 1)",
        "reference_bandwidth_hz": REFERENCE_COM_BANDWIDTH_HZ,
        "reference_bandwidth_denominator": NUM_UAV + ROI_COUNT_MAX,
        "total_bandwidth_hz": 10e6,
        "packet_rate_packets_per_second": packet_rate,
        "packet_size_bits": COM_PACKET_SIZE_BITS,
        "required_rate_bps": required_rate_bps,
        "zero_demand_satisfaction": 1.0,
        # Retained name for checkpoint API compatibility; it is a demand rate,
        # not a sampled raw-capacity percentile.
        "c_ref_com": required_rate_bps / 1e6,
    }


def calibrate_com_capacity(env=None, seed=None, sample_count=None):
    del env, seed, sample_count
    return demand_satisfaction_metadata()


def save_calibration(calibration, path=DEFAULT_ARTIFACT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return path


def load_com_capacity_reference(path=DEFAULT_ARTIFACT):
    del path
    metadata = demand_satisfaction_metadata()
    return COM_REQUIRED_RATE_BPS / 1e6, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Write canonical COM demand-satisfaction metadata"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    metadata = demand_satisfaction_metadata()
    output = save_calibration(metadata, args.output)
    print(json.dumps(metadata, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
