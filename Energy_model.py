import math

import numpy as np

from experiment_config import PROPULSION_MODEL_ID, PROPULSION_PARAMETERS

class EnergyConsumptionModel:
    def __init__(self, E_max, N_u=16, mobility_params=None):
        """
        初始化能量模型
        :param E_max: 最大能量 (J)
        :param N_u: UAV 數量
        :param mobility_params: 預設移動能耗模型參數
        """
        self.E_max = E_max
        self.N_u = N_u
        # 預設參數
        canonical = dict(PROPULSION_PARAMETERS)
        if mobility_params is not None and dict(mobility_params) != canonical:
            raise ValueError(
                "the canonical propulsion model has fixed parameters; legacy "
                "P0/P1 mobility parameters are incompatible"
            )
        self.mobility_params = canonical
        self.model_id = PROPULSION_MODEL_ID

    def propulsion_power(self, velocity_vector):
        """Return canonical 3-D quadrotor propulsion power in watts."""

        velocity = np.asarray(velocity_vector, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity_vector must contain three finite m/s values")
        p = self.mobility_params
        n_r = float(p["n_r"])
        rho = float(p["rho"])
        area = float(p["A"])
        mass = float(p["m"])
        gravity = float(p["g"])
        speed = float(np.linalg.norm(velocity))
        direction = np.zeros(3, dtype=float) if speed == 0.0 else velocity / speed
        sin_climb = 0.0 if speed == 0.0 else float(velocity[2] / speed)
        gravity_vector = np.asarray((0.0, 0.0, -gravity), dtype=float)
        rotor_thrust = float(
            np.linalg.norm(
                mass * gravity_vector
                - 0.5 * rho * speed**2 * float(p["S_FP"]) * direction
            )
            / n_r
        )
        profile = (
            float(p["delta"])
            / 8.0
            * (rotor_thrust / (float(p["c_T"]) * rho * area) + 3.0 * speed**2)
            * math.sqrt(
                rotor_thrust
                * rho
                * float(p["c_s"]) ** 2
                * area
                / float(p["c_T"])
            )
        )
        induced_inner = (
            math.sqrt(
                rotor_thrust**2 / (4.0 * rho**2 * area**2)
                + speed**4 / 4.0
            )
            - speed**2 / 2.0
        )
        if induced_inner < 0.0 or not math.isfinite(induced_inner):
            raise RuntimeError("canonical induced-power radicand is invalid")
        induced = (
            (1.0 + float(p["c_f"]))
            * rotor_thrust
            * math.sqrt(induced_inner)
        )
        climb = mass * gravity * speed / n_r * sin_climb
        parasite = (
            0.5
            * float(p["d_0"])
            * speed**3
            * rho
            * float(p["c_s"])
            * area
        )
        power = n_r * (profile + induced + climb + parasite)
        if not math.isfinite(power) or power < 0.0:
            raise RuntimeError(
                "canonical propulsion power is non-finite or negative; "
                "the action envelope/formula is invalid"
            )
        return float(power)

    def mobility_power(self, v_h, v_v, **kwargs):
        """Compatibility facade using a heading-invariant velocity vector."""

        if kwargs and dict(kwargs) != self.mobility_params:
            raise ValueError("non-canonical propulsion parameters are not supported")
        return self.propulsion_power((float(v_h), 0.0, float(v_v)))

    def compute_mobility_energy(
        self, uav_idx, v_h=None, v_v=None, t=1.0, velocity_vector=None
    ):
        """計算移動能耗，並更新內部狀態"""
        del uav_idx
        if velocity_vector is None:
            velocity_vector = (float(v_h), 0.0, float(v_v))
        P_mob = self.propulsion_power(velocity_vector)
        E = P_mob * t
        if not math.isfinite(E) or E < 0.0:
            raise RuntimeError("canonical propulsion energy is invalid")
        return float(E)

    def metadata(self):
        return {"model_id": self.model_id, "parameters": dict(self.mobility_params)}

    # def compute_comm_energy(self, uav_idx, PL_dB, data_rate, sigma_sq, B, t):
    #     """計算通訊能耗（單次）"""
    #     PL_linear = 10 ** (PL_dB / 10)
    #     noise_power_dBm = sigma_sq + 10 * math.log10(B)
    #     noise_linear = 10 ** ((noise_power_dBm - 30) / 10)  # dBm to W
    #     E = noise_linear * PL_linear * (2**(data_rate / B) - 1) * t
    #     # self.energy_levels[uav_idx] = max(self.energy_levels[uav_idx] - E, 0)
    #     return E

    # def energy_remaining(self, uav_idx):
    #     """回傳剩餘能量"""
    #     return self.energy_levels.get(uav_idx, 0)
