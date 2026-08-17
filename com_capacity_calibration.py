import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SEED = 20260817
DEFAULT_SAMPLE_COUNT = 20000
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "config" / "com_capacity_calibration.json"


def calibrate_com_capacity(env, seed=DEFAULT_SEED, sample_count=DEFAULT_SAMPLE_COUNT):
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not env.uav_dict or not env.SR_teams:
        raise ValueError("Simulator must be reset before COM calibration")

    rng = np.random.default_rng(int(seed))
    uav = env.uav_dict[0]
    sr = env.SR_teams[0]
    original_uav = uav.get_position()
    original_sr = sr.get_position()
    z_min = float(getattr(uav, "min_AGL", 50.0))
    z_max = float(getattr(uav, "max_AGL", 150.0))
    capacities_mbps = np.empty(int(sample_count), dtype=np.float64)
    distances_m = np.empty(int(sample_count), dtype=np.float64)
    attempts = 0

    try:
        index = 0
        while index < int(sample_count):
            attempts += 1
            sr.x = float(rng.uniform(0.0, env.env_width))
            sr.y = float(rng.uniform(0.0, env.env_height))
            sr.z = 0.0
            uav.x_u = float(rng.uniform(0.0, env.env_width))
            uav.y_u = float(rng.uniform(0.0, env.env_height))
            uav.z_u = float(rng.uniform(z_min, z_max))
            distance_m = float(
                np.linalg.norm(
                    np.asarray(uav.get_position(), dtype=float)
                    - np.asarray(sr.get_position(), dtype=float)
                )
            )
            if distance_m > float(env.SR_UAV_MAX_RANGE_M):
                continue
            capacities_mbps[index] = env.get_sr_uav_capacity_mbps(0, 0)
            distances_m[index] = distance_m
            index += 1
    finally:
        uav.x_u, uav.y_u, uav.z_u = original_uav
        sr.x, sr.y, sr.z = original_sr

    if not np.all(np.isfinite(capacities_mbps)) or np.any(capacities_mbps <= 0):
        raise ValueError("COM calibration produced non-positive or non-finite capacity")
    stats = {
        "min_mbps": float(np.min(capacities_mbps)),
        "median_mbps": float(np.median(capacities_mbps)),
        "p95_mbps": float(np.percentile(capacities_mbps, 95.0)),
        "max_mbps": float(np.max(capacities_mbps)),
    }
    c_ref_com = stats["p95_mbps"]
    if not 0.1 <= c_ref_com <= 1000.0:
        raise RuntimeError(
            "COM P95 calibration is outside the required 0.1--1000 Mbps range: "
            f"{c_ref_com} Mbps"
        )
    return {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "candidate_attempts": int(attempts),
        "feasible_only": True,
        "capacity_unit": "Mbps",
        "carrier_frequency_ghz": float(env.SR_UAV_CARRIER_GHZ),
        "bandwidth_hz": float(env.SR_UAV_BANDWIDTH_HZ),
        "transmit_power_dbm": float(env.SR_UAV_TX_POWER_DBM),
        "noise_density_dbm_per_hz": float(env.SR_UAV_NOISE_DBM_PER_HZ),
        "maximum_link_range_m": float(env.SR_UAV_MAX_RANGE_M),
        "sampled_distance_range_m": [
            float(np.min(distances_m)),
            float(np.max(distances_m)),
        ],
        "x_range_m": [0.0, float(env.env_width)],
        "y_range_m": [0.0, float(env.env_height)],
        "uav_altitude_range_m": [z_min, z_max],
        "sr_altitude_m": 0.0,
        "percentile": 95.0,
        "c_ref_com": c_ref_com,
        "capacity_statistics": stats,
    }


def save_calibration(calibration, path=DEFAULT_ARTIFACT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return path


def load_com_capacity_reference(path=DEFAULT_ARTIFACT):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"COM capacity calibration is missing: {path}. "
            "Run com_capacity_calibration.py before training."
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    value = float(calibration["c_ref_com"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid c_ref_com in {path}: {value}")
    return value, calibration


def main():
    parser = argparse.ArgumentParser(description="Calibrate the shared COM capacity reference")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    from Simulator import Simulator

    env = Simulator(num_UAV=16)
    env.num_GT = 4
    env.reset_environment()
    calibration = calibrate_com_capacity(env, seed=args.seed, sample_count=args.samples)
    output = save_calibration(calibration, args.output)
    print(json.dumps(calibration, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
