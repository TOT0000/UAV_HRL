"""Canonical deterministic channel model shared by every experiment method."""

from __future__ import annotations

import numpy as np


NOISE_PSD_DBM_PER_HZ = -169.0
A2G_CARRIER_GHZ = 2.0
A2A_CARRIER_GHZ = 2.4
A2G_LOS_A = 11.95
A2G_LOS_B = 0.136
A2G_LOS_EXCESS_DB = 2.0
A2G_NLOS_EXCESS_DB = 20.0
S2U_TX_POWER_DBM = 23.0
U2U_U2G_TX_POWER_DBM = 30.0
NUMERICAL_CAPACITY_EPS_MBPS = np.finfo(np.float64).eps


def noise_power_dbm(bandwidth_hz):
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be positive and finite")
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(bandwidth)


def shannon_capacity_mbps(path_loss_db, bandwidth_hz, transmit_power_dbm):
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    path_loss = np.asarray(path_loss_db, dtype=float)
    if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be positive and finite")
    snr = 10.0 ** (
        (float(transmit_power_dbm) - path_loss - noise_power_dbm(bandwidth)) / 10.0
    )
    capacity = bandwidth * np.log2(1.0 + np.maximum(snr, 0.0)) / 1e6
    return np.where(np.isfinite(capacity), np.maximum(capacity, 0.0), 0.0)


def a2g_path_loss_db(aerial_position, ground_position):
    """Expected A2G loss using distance in metres and frequency in GHz."""

    aerial = np.asarray(aerial_position, dtype=float)
    ground = np.asarray(ground_position, dtype=float)
    delta = aerial - ground
    distance = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
    altitude = np.maximum(delta[..., 2], 0.0)
    elevation = np.degrees(np.arcsin(np.clip(altitude / distance, -1.0, 1.0)))
    los_probability = 1.0 / (
        1.0
        + A2G_LOS_A * np.exp(-A2G_LOS_B * (elevation - A2G_LOS_A))
    )
    free_space = (
        20.0 * np.log10(distance)
        + 20.0 * np.log10(A2G_CARRIER_GHZ)
        + 32.44
    )
    return (
        free_space
        + los_probability * A2G_LOS_EXCESS_DB
        + (1.0 - los_probability) * A2G_NLOS_EXCESS_DB
    )


def a2g_capacity_mbps(
    aerial_position, ground_position, bandwidth_hz, transmit_power_dbm
):
    return shannon_capacity_mbps(
        a2g_path_loss_db(aerial_position, ground_position),
        bandwidth_hz,
        transmit_power_dbm,
    )


def u2u_path_loss_db(sender_position, receiver_position):
    """Directed paper A2A loss using the sender's absolute AGL altitude."""

    sender = np.asarray(sender_position, dtype=float)
    receiver = np.asarray(receiver_position, dtype=float)
    sender_altitude = np.maximum(sender[..., 2], 1e-3)
    distance = np.maximum(np.linalg.norm(sender - receiver, axis=-1), 1.0)
    distance_coefficient = np.maximum(
        23.9 - 1.8 * np.log10(sender_altitude), 20.0
    )
    frequency_term = 20.0 * np.log10(40.0 * np.pi * A2A_CARRIER_GHZ / 3.0)
    return distance_coefficient * np.log10(distance) + frequency_term


def u2u_capacity_mbps(sender_position, receiver_position, bandwidth_hz):
    return shannon_capacity_mbps(
        u2u_path_loss_db(sender_position, receiver_position),
        bandwidth_hz,
        U2U_U2G_TX_POWER_DBM,
    )


class ChannelModel:
    """Compatibility facade; canonical code calls the functions above."""

    @staticmethod
    def PL_uu(sender_altitude_agl, d_3d, f_c=A2A_CARRIER_GHZ):
        altitude = np.asarray(sender_altitude_agl, dtype=float)
        distance = np.asarray(d_3d, dtype=float)
        coefficient = np.maximum(
            23.9 - 1.8 * np.log10(np.maximum(altitude, 1e-3)), 20.0
        )
        return coefficient * np.log10(np.maximum(distance, 1.0)) + 20.0 * np.log10(
            40.0 * np.pi * float(f_c) / 3.0
        )

    @staticmethod
    def PL_ug(distances_ug, f_c=A2G_CARRIER_GHZ):
        distance = np.maximum(np.asarray(distances_ug, dtype=float), 1e-3)
        return 20.0 * np.log10(distance) + 20.0 * np.log10(float(f_c)) + 32.44

    @staticmethod
    def SNR_uu(power_dbm, noise_psd_dbm_hz, path_loss_db, bandwidth_hz):
        del noise_psd_dbm_hz
        return 10.0 ** (
            (
                float(power_dbm)
                - np.asarray(path_loss_db)
                - noise_power_dbm(bandwidth_hz)
            )
            / 10.0
        )

    SNR_ug = SNR_uu

    @staticmethod
    def C_uu(bandwidth_hz, snr):
        return (
            np.asarray(bandwidth_hz)
            * np.log2(1.0 + np.maximum(snr, 0.0))
            / 1e6
        )

    C_ug = C_uu
