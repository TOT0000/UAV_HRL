"""Metadata for the fixed theoretical S2U capacity normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_config import (
    COM_UTILITY_CONTRACT_VERSION,
    COM_PACKET_RATE_PER_SECOND,
    COM_PACKET_SIZE_BITS,
    NUM_UAV,
    REFERENCE_COM_BANDWIDTH_HZ,
    ROI_COUNT_MAX,
)
from Channel_model import reference_s2u_max_capacity_mbps


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "config" / "com_capacity_calibration.json"
COM_NORMALIZATION_SCHEMA = COM_UTILITY_CONTRACT_VERSION


def fixed_reference_capacity_metadata(offered_rate_packets_per_second=None):
    packet_rate = (
        COM_PACKET_RATE_PER_SECOND
        if offered_rate_packets_per_second is None
        else float(offered_rate_packets_per_second)
    )
    offered_rate_bps = packet_rate * COM_PACKET_SIZE_BITS
    reference_maximum_mbps = reference_s2u_max_capacity_mbps(
        REFERENCE_COM_BANDWIDTH_HZ
    )
    return {
        "schema": COM_NORMALIZATION_SCHEMA,
        "normalization": (
            "clip(conditional expected capacity / fixed LoS-Rician expected "
            "theoretical maximum, 0, 1)"
        ),
        "reference_bandwidth_hz": REFERENCE_COM_BANDWIDTH_HZ,
        "reference_bandwidth_denominator": NUM_UAV + ROI_COUNT_MAX,
        "total_bandwidth_hz": 10e6,
        "packet_rate_packets_per_second": packet_rate,
        "packet_size_bits": COM_PACKET_SIZE_BITS,
        "offered_rate_bps": offered_rate_bps,
        "reference_geometry": {
            "horizontal_distance_m": 0.0,
            "uav_agl_m": 50.0,
            "sr_agl_m": 0.0,
        },
        "transmit_power_dbm": 23.0,
        "reference_large_scale_state": "LoS",
        "reference_small_scale_fading": "Rician-K-linear-10",
        "routing_csi": "deterministic-distributional-expected-capacity",
        "reference_s2u_max_capacity_mbps": reference_maximum_mbps,
        "c_ref_com": reference_maximum_mbps,
        "rate_sweep_invariant": True,
    }


def calibrate_com_capacity(env=None, seed=None, sample_count=None):
    del env, seed, sample_count
    return fixed_reference_capacity_metadata()


def save_calibration(calibration, path=DEFAULT_ARTIFACT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return path


def load_com_capacity_reference(path=DEFAULT_ARTIFACT):
    del path
    metadata = fixed_reference_capacity_metadata()
    return metadata["reference_s2u_max_capacity_mbps"], metadata


def main():
    parser = argparse.ArgumentParser(
        description="Write canonical fixed-reference COM utility metadata"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    metadata = fixed_reference_capacity_metadata()
    output = save_calibration(metadata, args.output)
    print(json.dumps(metadata, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
