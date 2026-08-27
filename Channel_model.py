"""Authoritative stochastic channel model shared by every experiment method.

The routing policy observes deterministic distributional expected capacity. The
environment privately samples one common-random-number fading profile for every
potential link and routing slot, and packet service consumes that profile.
"""

from __future__ import annotations

import copy
import time

import numpy as np


CHANNEL_MODEL_VERSION = "sampled-a2g-conditional-5ms-block-fading-v1"
CHANNEL_ENVIRONMENT_CONTRACT_VERSION = (
    "boundary-aligned-one-second-a2g-state-fifty-block-service-v2"
)
CHANNEL_FAIRNESS_CONTRACT_VERSION = "all-potential-links-fixed-order-crn-v1"
CHANNEL_NORMALIZATION_VERSION = "link-type-fading-aware-physical-reference-v1"
ROUTING_CSI_CONTRACT = "deterministic-distributional-expected-capacity-v1"
ACTUAL_TRANSMISSION_CONTRACT = "fifty-block-cumulative-service-v1"

NOISE_PSD_DBM_PER_HZ = -169.0
A2G_CARRIER_GHZ = 2.0
A2A_CARRIER_GHZ = 2.4
A2G_LOS_A = 11.95
A2G_LOS_B = 0.136
A2G_LOS_EXCESS_DB = 2.0
A2G_NLOS_EXCESS_DB = 20.0
S2U_TX_POWER_DBM = 23.0
U2U_U2G_TX_POWER_DBM = 30.0
RICIAN_K_LINEAR = 10.0
RICIAN_K_DB = 10.0 * np.log10(RICIAN_K_LINEAR)
LARGE_SCALE_STATE_SECONDS = 1.0
FADING_BLOCK_SECONDS = 0.005
ROUTING_SLOT_SECONDS = 0.25
FADING_BLOCKS_PER_ROUTING_SLOT = 50
EXPECTED_CAPACITY_QUADRATURE_ORDER = 64
NUMERICAL_CAPACITY_EPS_MBPS = np.finfo(np.float64).eps
MINIMUM_A2A_DISTANCE_M = 1.0
REFERENCE_UAV_MIN_AGL_M = 50.0
CHANNEL_TIME_SLOT_REFERENCE = {
    "citation": (
        "Masoud Ghazikor, Keenan Roach, Kenny Cheung, and Morteza Hashemi, "
        "Channel-Aware Distributed Transmission Control and Video Streaming "
        "in UAV Networks, IEEE Transactions on Communications, 2026"
    ),
    "doi": "10.1109/TCOMM.2025.3650376",
    "scope": "5 ms block duration only",
}


def validate_channel_time_grid(
    routing_slot_seconds=ROUTING_SLOT_SECONDS,
    fading_block_seconds=FADING_BLOCK_SECONDS,
):
    routing_slot_seconds = float(routing_slot_seconds)
    fading_block_seconds = float(fading_block_seconds)
    if not np.isfinite(routing_slot_seconds) or routing_slot_seconds <= 0.0:
        raise ValueError("routing slot duration must be positive and finite")
    if not np.isfinite(fading_block_seconds) or fading_block_seconds <= 0.0:
        raise ValueError("fading block duration must be positive and finite")
    ratio = routing_slot_seconds / fading_block_seconds
    rounded = int(round(ratio))
    if rounded <= 0 or not np.isclose(
        rounded * fading_block_seconds,
        routing_slot_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "routing slot must contain an integer number of fading blocks: "
            f"slot={routing_slot_seconds}, block={fading_block_seconds}"
        )
    return rounded


if validate_channel_time_grid() != FADING_BLOCKS_PER_ROUTING_SLOT:
    raise RuntimeError("canonical channel time grid must contain exactly 50 blocks")


def noise_power_dbm(bandwidth_hz):
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be positive and finite")
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(bandwidth)


def average_snr_linear(path_loss_db, bandwidth_hz, transmit_power_dbm):
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    path_loss = np.asarray(path_loss_db, dtype=np.float64)
    if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be positive and finite")
    snr = 10.0 ** (
        (float(transmit_power_dbm) - path_loss - noise_power_dbm(bandwidth)) / 10.0
    )
    return np.where(np.isfinite(snr), np.maximum(snr, 0.0), 0.0)


def shannon_capacity_mbps(path_loss_db, bandwidth_hz, transmit_power_dbm):
    """No-small-scale-fading reference capacity; not the stochastic main model."""

    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    snr = average_snr_linear(path_loss_db, bandwidth, transmit_power_dbm)
    capacity = bandwidth * np.log2(1.0 + snr) / 1e6
    return np.where(np.isfinite(capacity), np.maximum(capacity, 0.0), 0.0)


def a2g_los_probability_from_elevation_deg(elevation_angle_deg):
    elevation = np.asarray(elevation_angle_deg, dtype=np.float64)
    probability = 1.0 / (
        1.0 + A2G_LOS_A * np.exp(-A2G_LOS_B * (elevation - A2G_LOS_A))
    )
    return np.clip(probability, 0.0, 1.0)


def _a2g_geometry(aerial_position, ground_position):
    aerial = np.asarray(aerial_position, dtype=np.float64)
    ground = np.asarray(ground_position, dtype=np.float64)
    delta = aerial - ground
    distance = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
    altitude = np.maximum(delta[..., 2], 0.0)
    elevation = np.degrees(np.arcsin(np.clip(altitude / distance, -1.0, 1.0)))
    return distance, elevation


def a2g_los_probability(aerial_position, ground_position):
    """Paper equation (13), with elevation angle expressed in degrees."""

    _, elevation = _a2g_geometry(aerial_position, ground_position)
    return a2g_los_probability_from_elevation_deg(elevation)


def a2g_free_space_path_loss_db(aerial_position, ground_position):
    distance, _ = _a2g_geometry(aerial_position, ground_position)
    return (
        20.0 * np.log10(distance)
        + 20.0 * np.log10(A2G_CARRIER_GHZ)
        + 32.44
    )


def a2g_conditional_path_loss_db(aerial_position, ground_position, los_state):
    """Conditional A2G path loss after the one-second state has been sampled."""

    free_space = a2g_free_space_path_loss_db(aerial_position, ground_position)
    los_state = np.asarray(los_state, dtype=bool)
    excess = np.where(los_state, A2G_LOS_EXCESS_DB, A2G_NLOS_EXCESS_DB)
    return np.asarray(free_space, dtype=np.float64) + excess


def a2g_expected_path_loss_db(aerial_position, ground_position):
    """Equation (14) diagnostic/reference; never used by the stochastic runtime."""

    free_space = a2g_free_space_path_loss_db(aerial_position, ground_position)
    probability = a2g_los_probability(aerial_position, ground_position)
    return (
        free_space
        + probability * A2G_LOS_EXCESS_DB
        + (1.0 - probability) * A2G_NLOS_EXCESS_DB
    )


def a2g_path_loss_db(aerial_position, ground_position):
    """Compatibility alias for the deterministic expected-path-loss reference."""

    return a2g_expected_path_loss_db(aerial_position, ground_position)


def u2u_path_loss_db(sender_position, receiver_position):
    """Directed paper A2A loss using the sender's absolute AGL altitude."""

    sender = np.asarray(sender_position, dtype=np.float64)
    receiver = np.asarray(receiver_position, dtype=np.float64)
    sender_altitude = np.maximum(sender[..., 2], 1e-3)
    distance = np.maximum(
        np.linalg.norm(sender - receiver, axis=-1), MINIMUM_A2A_DISTANCE_M
    )
    distance_coefficient = np.maximum(
        23.9 - 1.8 * np.log10(sender_altitude), 20.0
    )
    frequency_term = 20.0 * np.log10(40.0 * np.pi * A2A_CARRIER_GHZ / 3.0)
    return distance_coefficient * np.log10(distance) + frequency_term


def fading_power_gains_from_normals(standard_normals, los_rician):
    """Vectorized CN(0,1) transform for Rayleigh or K=10 Rician power gain."""

    normals = np.asarray(standard_normals, dtype=np.float64)
    if normals.shape[-1] != 2 or not np.all(np.isfinite(normals)):
        raise ValueError("standard_normals must be finite with final dimension 2")
    rayleigh = 0.5 * (normals[..., 0] ** 2 + normals[..., 1] ** 2)
    scatter = np.sqrt(1.0 / (2.0 * (RICIAN_K_LINEAR + 1.0)))
    specular = np.sqrt(RICIAN_K_LINEAR / (RICIAN_K_LINEAR + 1.0))
    rician = (
        (specular + scatter * normals[..., 0]) ** 2
        + (scatter * normals[..., 1]) ** 2
    )
    selector = np.asarray(los_rician, dtype=bool)
    while selector.ndim < rayleigh.ndim:
        selector = selector[..., None]
    return np.where(selector, rician, rayleigh)


def sample_fading_power_gains(rng, shape, *, fading="rician"):
    """Sample normalized gains without any Python link-by-block inner loop."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("fading sampling requires a local numpy Generator")
    fading = str(fading).lower()
    if fading not in {"rician", "rayleigh"}:
        raise ValueError(f"unsupported fading distribution: {fading}")
    normals = rng.standard_normal(tuple(shape) + (2,))
    return fading_power_gains_from_normals(normals, fading == "rician")


_LAGUERRE_NODES, _LAGUERRE_BASE_WEIGHTS = np.polynomial.laguerre.laggauss(
    EXPECTED_CAPACITY_QUADRATURE_ORDER
)
_RAYLEIGH_GAIN_NODES = _LAGUERRE_NODES.astype(np.float64)
_RAYLEIGH_WEIGHTS = _LAGUERRE_BASE_WEIGHTS.astype(np.float64)
_RICIAN_GAIN_NODES = _LAGUERRE_NODES / (RICIAN_K_LINEAR + 1.0)
_RICIAN_WEIGHTS = (
    _LAGUERRE_BASE_WEIGHTS
    * np.exp(-RICIAN_K_LINEAR)
    * np.i0(2.0 * np.sqrt(RICIAN_K_LINEAR * _LAGUERRE_NODES))
)
_RICIAN_WEIGHTS = _RICIAN_WEIGHTS / _RICIAN_WEIGHTS.sum()


def expected_fading_capacity_mbps(
    path_loss_db,
    bandwidth_hz,
    transmit_power_dbm,
    *,
    fading="rician",
):
    """Deterministic E[B log2(1+mean_snr*G)] by fixed Gauss-Laguerre rule."""

    fading = str(fading).lower()
    if fading == "rician":
        gain_nodes, weights = _RICIAN_GAIN_NODES, _RICIAN_WEIGHTS
    elif fading == "rayleigh":
        gain_nodes, weights = _RAYLEIGH_GAIN_NODES, _RAYLEIGH_WEIGHTS
    else:
        raise ValueError(f"unsupported fading distribution: {fading}")
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    snr = average_snr_linear(path_loss_db, bandwidth, transmit_power_dbm)
    spectral_efficiency = np.sum(
        np.log2(1.0 + snr[..., None] * gain_nodes) * weights,
        axis=-1,
    )
    capacity = bandwidth * spectral_efficiency / 1e6
    return np.where(np.isfinite(capacity), np.maximum(capacity, 0.0), 0.0)


def block_capacity_profile_mbps(
    path_loss_db,
    bandwidth_hz,
    transmit_power_dbm,
    power_gains,
):
    gains = np.asarray(power_gains, dtype=np.float64)
    if np.any(gains < 0.0) or not np.all(np.isfinite(gains)):
        raise ValueError("fading power gains must be finite and non-negative")
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    snr = average_snr_linear(path_loss_db, bandwidth, transmit_power_dbm)
    capacity = bandwidth * np.log2(1.0 + snr[..., None] * gains) / 1e6
    return np.where(np.isfinite(capacity), np.maximum(capacity, 0.0), 0.0)


def effective_capacity_mbps(block_capacity_profile):
    profile = np.asarray(block_capacity_profile, dtype=np.float64)
    if profile.shape[-1] != FADING_BLOCKS_PER_ROUTING_SLOT:
        raise ValueError("capacity profile must contain exactly 50 fading blocks")
    if np.any(profile < 0.0) or not np.all(np.isfinite(profile)):
        raise ValueError("capacity profile must be finite and non-negative")
    return np.mean(profile, axis=-1)


def slot_service_bits(block_capacity_profile):
    profile = np.asarray(block_capacity_profile, dtype=np.float64)
    if profile.shape[-1] != FADING_BLOCKS_PER_ROUTING_SLOT:
        raise ValueError("capacity profile must contain exactly 50 fading blocks")
    return np.sum(profile * 1e6 * FADING_BLOCK_SECONDS, axis=-1)


def a2g_expected_capacity_mbps(
    aerial_position,
    ground_position,
    bandwidth_hz,
    transmit_power_dbm,
    los_state,
):
    path_loss = a2g_conditional_path_loss_db(
        aerial_position, ground_position, los_state
    )
    if np.ndim(los_state) == 0:
        return expected_fading_capacity_mbps(
            path_loss,
            bandwidth_hz,
            transmit_power_dbm,
            fading="rician" if bool(los_state) else "rayleigh",
        )
    los_state = np.asarray(los_state, dtype=bool)
    rician = expected_fading_capacity_mbps(
        path_loss, bandwidth_hz, transmit_power_dbm, fading="rician"
    )
    rayleigh = expected_fading_capacity_mbps(
        path_loss, bandwidth_hz, transmit_power_dbm, fading="rayleigh"
    )
    return np.where(los_state, rician, rayleigh)


def a2g_capacity_mbps(
    aerial_position, ground_position, bandwidth_hz, transmit_power_dbm
):
    """Deterministic expected-path-loss reference retained for diagnostics."""

    return shannon_capacity_mbps(
        a2g_expected_path_loss_db(aerial_position, ground_position),
        bandwidth_hz,
        transmit_power_dbm,
    )


def u2u_capacity_mbps(sender_position, receiver_position, bandwidth_hz):
    return expected_fading_capacity_mbps(
        u2u_path_loss_db(sender_position, receiver_position),
        bandwidth_hz,
        U2U_U2G_TX_POWER_DBM,
        fading="rician",
    )


def reference_s2u_max_capacity_mbps(bandwidth_hz):
    return float(
        a2g_expected_capacity_mbps(
            (0.0, 0.0, REFERENCE_UAV_MIN_AGL_M),
            (0.0, 0.0, 0.0),
            bandwidth_hz,
            S2U_TX_POWER_DBM,
            True,
        )
    )


def reference_u2g_max_capacity_mbps(system_bandwidth_hz):
    return float(
        a2g_expected_capacity_mbps(
            (0.0, 0.0, REFERENCE_UAV_MIN_AGL_M),
            (0.0, 0.0, 0.0),
            system_bandwidth_hz,
            U2U_U2G_TX_POWER_DBM,
            True,
        )
    )


def reference_u2u_max_capacity_mbps(system_bandwidth_hz):
    return float(
        u2u_capacity_mbps(
            (0.0, 0.0, REFERENCE_UAV_MIN_AGL_M),
            (MINIMUM_A2A_DISTANCE_M, 0.0, REFERENCE_UAV_MIN_AGL_M),
            system_bandwidth_hz,
        )
    )


def normalized_s2u_capacity_utility(
    aerial_position,
    ground_position,
    bandwidth_hz,
    *,
    los_state=True,
):
    capacity = float(
        a2g_expected_capacity_mbps(
            aerial_position,
            ground_position,
            bandwidth_hz,
            S2U_TX_POWER_DBM,
            bool(los_state),
        )
    )
    maximum = reference_s2u_max_capacity_mbps(bandwidth_hz)
    if not np.isfinite(capacity) or maximum <= 0.0 or not np.isfinite(maximum):
        raise RuntimeError("canonical S2U reference capacity is invalid")
    return float(np.clip(capacity / maximum, 0.0, 1.0))


def channel_configuration_metadata():
    return {
        "channel_model_version": CHANNEL_MODEL_VERSION,
        "channel_environment_contract_version": CHANNEL_ENVIRONMENT_CONTRACT_VERSION,
        "a2a_carrier_frequency_ghz": A2A_CARRIER_GHZ,
        "a2g_carrier_frequency_ghz": A2G_CARRIER_GHZ,
        "a2g_los_probability_parameters": {"a": A2G_LOS_A, "b": A2G_LOS_B},
        "a2g_excess_loss_db": {
            "los": A2G_LOS_EXCESS_DB,
            "nlos": A2G_NLOS_EXCESS_DB,
        },
        "a2g_large_scale_state_interval_seconds": LARGE_SCALE_STATE_SECONDS,
        "a2g_large_scale_state_model": (
            "one-second large-scale block-state Monte Carlo approximation"
        ),
        "large_scale_state_sampling_geometry": (
            "post-boundary UAV/SR geometry before interval policy observation"
        ),
        "movement_replay_boundary_semantics": (
            "next_state equals the next movement policy observation"
        ),
        "routing_replay_boundary_semantics": (
            "bootstrap next_state finalized at the next start-of-slot decision observation"
        ),
        "a2g_path_loss_mode": "conditional-after-sampled-state",
        "fading_block_seconds": FADING_BLOCK_SECONDS,
        "routing_slot_seconds": ROUTING_SLOT_SECONDS,
        "fading_blocks_per_routing_slot": FADING_BLOCKS_PER_ROUTING_SLOT,
        "rician_k_linear": RICIAN_K_LINEAR,
        "rician_k_db": RICIAN_K_DB,
        "u2u_fading": "always-LoS-Rician",
        "a2g_nlos_fading": "Rayleigh",
        "routing_csi": ROUTING_CSI_CONTRACT,
        "actual_transmission": ACTUAL_TRANSMISSION_CONTRACT,
        "expected_capacity_quadrature": {
            "method": "fixed-order-Gauss-Laguerre",
            "order": EXPECTED_CAPACITY_QUADRATURE_ORDER,
        },
        "communication_range_cutoff": None,
        "minimum_capacity_cutoff": None,
        "outage_per_harq": "disabled",
        "common_random_number_contract": CHANNEL_FAIRNESS_CONTRACT_VERSION,
        "normalization_contract": CHANNEL_NORMALIZATION_VERSION,
        "bandwidth_allocation": "equal-FDMA-over-one-shared-10MHz-pool",
        "time_slot_reference": dict(CHANNEL_TIME_SLOT_REFERENCE),
    }


def validate_channel_lifecycle_state(state, *, num_uav=None):
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint channel lifecycle state is missing")
    if state.get("channel_model_version") != CHANNEL_MODEL_VERSION:
        raise RuntimeError("checkpoint channel lifecycle contract is incompatible")
    if (
        state.get("channel_environment_contract_version")
        != CHANNEL_ENVIRONMENT_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "checkpoint channel boundary lifecycle contract is incompatible"
        )
    if state.get("checkpoint_scope") != "current-channel-lifecycle-snapshot":
        raise RuntimeError("checkpoint channel lifecycle scope is incompatible")
    resolved_num_uav = int(state.get("num_uav", -1))
    resolved_num_sr = int(state.get("num_sr", -1))
    if resolved_num_uav <= 0 or resolved_num_sr < 0:
        raise RuntimeError("checkpoint channel lifecycle dimensions are invalid")
    if num_uav is not None and resolved_num_uav != int(num_uav):
        raise RuntimeError("checkpoint channel UAV dimension is incompatible")
    u2g = np.asarray(state.get("u2g_los_state"), dtype=bool)
    s2u = np.asarray(state.get("s2u_los_state"), dtype=bool)
    if u2g.shape != (resolved_num_uav,) or s2u.shape != (
        resolved_num_sr,
        resolved_num_uav,
    ):
        raise RuntimeError("checkpoint channel A2G state shape is invalid")
    profile_keys = list(state.get("profile_keys", []))
    gains = state.get("gain_matrix")
    if gains is not None and np.asarray(gains).shape != (
        len(profile_keys),
        FADING_BLOCKS_PER_ROUTING_SLOT,
    ):
        raise RuntimeError("checkpoint channel gain profile shape is invalid")
    return copy.deepcopy(state)


class ChannelLifecycle:
    """One authoritative A2G-state and private slot-profile lifecycle."""

    def __init__(
        self,
        num_uav,
        gs_id,
        *,
        large_scale_rng,
        small_scale_rng,
        namespace="training",
    ):
        if not isinstance(large_scale_rng, np.random.Generator):
            raise TypeError("large-scale channel RNG must be a local Generator")
        if not isinstance(small_scale_rng, np.random.Generator):
            raise TypeError("small-scale channel RNG must be a local Generator")
        self.num_uav = int(num_uav)
        self.gs_id = int(gs_id)
        self.large_scale_rng = large_scale_rng
        self.small_scale_rng = small_scale_rng
        self.namespace = str(namespace)
        self.episode_identity = None
        self.num_sr = 0
        self.movement_interval_index = None
        self.routing_slot_index = None
        self.u2g_los_state = None
        self.s2u_los_state = None
        self.u2g_los_probability = None
        self.s2u_los_probability = None
        self._profile_keys = ()
        self._profile_index = {}
        self._gain_matrix = None
        self.large_scale_draw_count = 0
        self.small_scale_normal_draw_count = 0
        self.profile_generation_count = 0
        self.last_profile_generation_seconds = None

    def clear_episode(self, *, num_sr, episode_identity=None):
        """Clear episode-local channel state without consuming either RNG.

        Simulator uses this phase while constructing episode geometry.  Interval
        zero is sampled only after the canonical slot-0 SR boundary movement.
        """

        self.episode_identity = episode_identity
        self.num_sr = int(num_sr)
        if self.num_sr < 0:
            raise ValueError("channel lifecycle SR count must be non-negative")
        self.movement_interval_index = None
        self.routing_slot_index = None
        self.u2g_los_state = None
        self.s2u_los_state = None
        self.u2g_los_probability = None
        self.s2u_los_probability = None
        self._profile_keys = ()
        self._profile_index = {}
        self._gain_matrix = None
        self.last_profile_generation_seconds = None

    def reset_episode(
        self,
        *,
        uav_positions,
        sr_positions,
        gs_position,
        episode_identity=None,
    ):
        self.clear_episode(
            num_sr=len(sr_positions), episode_identity=episode_identity
        )
        self.begin_movement_interval(
            0,
            uav_positions=uav_positions,
            sr_positions=sr_positions,
            gs_position=gs_position,
            force=True,
        )

    def begin_movement_interval(
        self,
        interval_index,
        *,
        uav_positions,
        sr_positions,
        gs_position,
        force=False,
    ):
        interval_index = int(interval_index)
        if (
            not force
            and self.movement_interval_index == interval_index
            and self.u2g_los_state is not None
        ):
            return False
        uav_positions = np.asarray(uav_positions, dtype=np.float64)
        sr_positions = np.asarray(sr_positions, dtype=np.float64)
        gs_position = np.asarray(gs_position, dtype=np.float64)
        if uav_positions.shape != (self.num_uav, 3):
            raise ValueError("channel lifecycle received invalid UAV geometry")
        if sr_positions.shape != (self.num_sr, 3):
            raise ValueError("channel lifecycle received invalid SR geometry")
        self.u2g_los_probability = np.asarray(
            a2g_los_probability(uav_positions, gs_position), dtype=np.float64
        )
        self.u2g_los_state = (
            self.large_scale_rng.random(self.num_uav)
            < self.u2g_los_probability
        )
        if self.num_sr:
            self.s2u_los_probability = np.asarray(
                a2g_los_probability(
                    uav_positions[None, :, :], sr_positions[:, None, :]
                ),
                dtype=np.float64,
            )
            self.s2u_los_state = (
                self.large_scale_rng.random((self.num_sr, self.num_uav))
                < self.s2u_los_probability
            )
        else:
            self.s2u_los_probability = np.zeros(
                (0, self.num_uav), dtype=np.float64
            )
            self.s2u_los_state = np.zeros((0, self.num_uav), dtype=bool)
        self.large_scale_draw_count += self.num_uav * (self.num_sr + 1)
        self.movement_interval_index = interval_index
        self.routing_slot_index = None
        self._profile_keys = ()
        self._profile_index = {}
        self._gain_matrix = None
        return True

    def potential_link_keys(self):
        keys = [
            ("S2U", sr_id, uav_id)
            for sr_id in range(self.num_sr)
            for uav_id in range(self.num_uav)
        ]
        keys.extend(("U2G", uav_id, self.gs_id) for uav_id in range(self.num_uav))
        keys.extend(
            ("U2U", sender, receiver)
            for sender in range(self.num_uav)
            for receiver in range(self.num_uav)
            if sender != receiver
        )
        return tuple(keys)

    def prepare_routing_slot(self, routing_slot_index):
        routing_slot_index = int(routing_slot_index)
        if self.u2g_los_state is None or self.s2u_los_state is None:
            raise RuntimeError("large-scale A2G state is not initialized")
        if (
            self.routing_slot_index == routing_slot_index
            and self._gain_matrix is not None
        ):
            return False
        started = time.perf_counter()
        keys = self.potential_link_keys()
        normals = self.small_scale_rng.standard_normal(
            (len(keys), FADING_BLOCKS_PER_ROUTING_SLOT, 2)
        )
        rician_mask = np.empty(len(keys), dtype=bool)
        for index, (link_type, sender, receiver) in enumerate(keys):
            if link_type == "U2U":
                rician_mask[index] = True
            elif link_type == "U2G":
                rician_mask[index] = bool(self.u2g_los_state[sender])
            else:
                rician_mask[index] = bool(self.s2u_los_state[sender, receiver])
        self._gain_matrix = fading_power_gains_from_normals(normals, rician_mask)
        self._profile_keys = keys
        self._profile_index = {key: index for index, key in enumerate(keys)}
        self.routing_slot_index = routing_slot_index
        self.small_scale_normal_draw_count += int(normals.size)
        self.profile_generation_count += 1
        self.last_profile_generation_seconds = time.perf_counter() - started
        return True

    def gain_profile(self, link_type, sender_id, receiver_id):
        if self._gain_matrix is None:
            raise RuntimeError("routing-slot fading profiles are not prepared")
        key = (str(link_type), int(sender_id), int(receiver_id))
        try:
            index = self._profile_index[key]
        except KeyError as exc:
            raise KeyError(f"unknown potential channel link: {key}") from exc
        return self._gain_matrix[index]

    def a2g_state(self, link_type, sender_id, receiver_id):
        link_type = str(link_type)
        if link_type == "U2G":
            return bool(self.u2g_los_state[int(sender_id)])
        if link_type == "S2U":
            return bool(self.s2u_los_state[int(sender_id), int(receiver_id)])
        if link_type == "U2U":
            return True
        raise KeyError(f"unknown channel link type: {link_type}")

    def state_dict(self):
        return {
            "channel_model_version": CHANNEL_MODEL_VERSION,
            "channel_environment_contract_version": (
                CHANNEL_ENVIRONMENT_CONTRACT_VERSION
            ),
            "namespace": self.namespace,
            "episode_identity": self.episode_identity,
            "num_uav": self.num_uav,
            "num_sr": self.num_sr,
            "gs_id": self.gs_id,
            "movement_interval_index": self.movement_interval_index,
            "routing_slot_index": self.routing_slot_index,
            "u2g_los_state": copy.deepcopy(self.u2g_los_state),
            "s2u_los_state": copy.deepcopy(self.s2u_los_state),
            "u2g_los_probability": copy.deepcopy(self.u2g_los_probability),
            "s2u_los_probability": copy.deepcopy(self.s2u_los_probability),
            "profile_keys": list(self._profile_keys),
            "gain_matrix": copy.deepcopy(self._gain_matrix),
            "large_scale_draw_count": int(self.large_scale_draw_count),
            "small_scale_normal_draw_count": int(
                self.small_scale_normal_draw_count
            ),
            "profile_generation_count": int(self.profile_generation_count),
            "checkpoint_scope": "current-channel-lifecycle-snapshot",
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("channel lifecycle state must be an object")
        if state.get("channel_model_version") != CHANNEL_MODEL_VERSION:
            raise RuntimeError("channel lifecycle contract is incompatible")
        if (
            state.get("channel_environment_contract_version")
            != CHANNEL_ENVIRONMENT_CONTRACT_VERSION
        ):
            raise RuntimeError("channel boundary lifecycle contract is incompatible")
        if (
            int(state.get("num_uav", -1)) != self.num_uav
            or int(state.get("gs_id", -1)) != self.gs_id
            or str(state.get("namespace")) != self.namespace
        ):
            raise RuntimeError("channel lifecycle dimensions/namespace are incompatible")
        self.episode_identity = state.get("episode_identity")
        self.num_sr = int(state["num_sr"])
        self.movement_interval_index = state.get("movement_interval_index")
        self.routing_slot_index = state.get("routing_slot_index")
        self.u2g_los_state = np.asarray(state["u2g_los_state"], dtype=bool).copy()
        self.s2u_los_state = np.asarray(state["s2u_los_state"], dtype=bool).copy()
        self.u2g_los_probability = np.asarray(
            state["u2g_los_probability"], dtype=np.float64
        ).copy()
        self.s2u_los_probability = np.asarray(
            state["s2u_los_probability"], dtype=np.float64
        ).copy()
        if self.u2g_los_state.shape != (self.num_uav,) or self.s2u_los_state.shape != (
            self.num_sr,
            self.num_uav,
        ):
            raise RuntimeError("channel lifecycle A2G state shape is incompatible")
        self._profile_keys = tuple(tuple(item) for item in state.get("profile_keys", []))
        self._profile_index = {
            key: index for index, key in enumerate(self._profile_keys)
        }
        gain_matrix = state.get("gain_matrix")
        self._gain_matrix = (
            None if gain_matrix is None else np.asarray(gain_matrix, dtype=np.float64).copy()
        )
        if self._gain_matrix is not None and self._gain_matrix.shape != (
            len(self._profile_keys),
            FADING_BLOCKS_PER_ROUTING_SLOT,
        ):
            raise RuntimeError("channel fading profile shape is incompatible")
        self.large_scale_draw_count = int(state["large_scale_draw_count"])
        self.small_scale_normal_draw_count = int(
            state["small_scale_normal_draw_count"]
        )
        self.profile_generation_count = int(state["profile_generation_count"])
        self.last_profile_generation_seconds = None


class ChannelModel:
    """Compatibility facade; formal code uses the explicit helpers above."""

    @staticmethod
    def PL_uu(sender_altitude_agl, d_3d, f_c=A2A_CARRIER_GHZ):
        altitude = np.asarray(sender_altitude_agl, dtype=np.float64)
        distance = np.asarray(d_3d, dtype=np.float64)
        coefficient = np.maximum(
            23.9 - 1.8 * np.log10(np.maximum(altitude, 1e-3)), 20.0
        )
        return coefficient * np.log10(np.maximum(distance, 1.0)) + 20.0 * np.log10(
            40.0 * np.pi * float(f_c) / 3.0
        )

    @staticmethod
    def PL_ug(distances_ug, f_c=A2G_CARRIER_GHZ):
        distance = np.maximum(np.asarray(distances_ug, dtype=np.float64), 1e-3)
        return 20.0 * np.log10(distance) + 20.0 * np.log10(float(f_c)) + 32.44

    @staticmethod
    def SNR_uu(power_dbm, noise_psd_dbm_hz, path_loss_db, bandwidth_hz):
        del noise_psd_dbm_hz
        return average_snr_linear(path_loss_db, bandwidth_hz, power_dbm)

    SNR_ug = SNR_uu

    @staticmethod
    def C_uu(bandwidth_hz, snr):
        return np.asarray(bandwidth_hz) * np.log2(1.0 + np.maximum(snr, 0.0)) / 1e6

    C_ug = C_uu
