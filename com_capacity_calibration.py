import argparse
import json
from pathlib import Path

import numpy as np

from Channel_model import ChannelModel


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
    z_max = float(getattr(uav, "max_AGL", 200.0))
    capacities_mbps = np.empty(int(sample_count), dtype=np.float64)

    try:
        for index in range(int(sample_count)):
            sr.x = float(rng.uniform(0.0, env.env_width))
            sr.y = float(rng.uniform(0.0, env.env_height))
            sr.z = 0.0
            uav.x_u = float(rng.uniform(0.0, env.env_width))
            uav.y_u = float(rng.uniform(0.0, env.env_height))
            uav.z_u = float(rng.uniform(z_min, z_max))
            snr = env.get_snr(0, 0)
            capacities_mbps[index] = float(
                ChannelModel.C_ug(B_ug=10e6, SNR_ug_t=snr)
            )
    finally:
        uav.x_u, uav.y_u, uav.z_u = original_uav
        sr.x, sr.y, sr.z = original_sr

    c_ref_com = float(np.percentile(capacities_mbps, 95.0))
    if not np.isfinite(c_ref_com) or c_ref_com <= 0:
        raise ValueError(f"invalid COM P95 calibration result: {c_ref_com}")
    return {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "capacity_unit": "Mbps",
        "bandwidth_hz": 10e6,
        "x_range_m": [0.0, float(env.env_width)],
        "y_range_m": [0.0, float(env.env_height)],
        "uav_altitude_range_m": [z_min, z_max],
        "sr_altitude_m": 0.0,
        "percentile": 95.0,
        "c_ref_com": c_ref_com,
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
