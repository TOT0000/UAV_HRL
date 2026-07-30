import numpy as np
import math

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
        self.mobility_params = mobility_params or {
            "P0": 99.66, "P1": 120.16, "s_tip": 120, "s0": 0.002,
            "d0": 0.48, "rho": 1.225, "mu0": 0.0001, "Z": 0.5,
            "m": 2.0, "g": 9.8, "n_r": 1  #如論文參數
        }

    def mobility_power(self, v_h, v_v, **kwargs):
        """計算移動功率"""
        P0 = kwargs["P0"]; P1 = kwargs["P1"]; s_tip = kwargs["s_tip"]; s0 = kwargs["s0"]
        d0 = kwargs["d0"]; rho = kwargs["rho"]; mu0 = kwargs["mu0"]; Z = kwargs["Z"]
        m= kwargs["m"]; g= kwargs["g"]; n_r=kwargs["n_r"]
        eta_up = n_r
        eta_down = 3.0  #無人機下降的成本避免讓RL鑽漏洞認為下降0成本
        term1 = P0 * (1 + (3 * v_h**2) / (s_tip**2))
        inner = math.sqrt(1 + (v_h**4) / (4 * (s0**4))) - (v_h**2) / (2 * (s0**2))
        term2 = P1 *math.sqrt(max(inner, 0.0))  #防數值爆炸
        term3 = 0.5 * d0 * rho * mu0 * Z * v_h**3
        if v_v >= 0:
            term4 = (m * g * v_v) / eta_up
        else:
            term4 = (m * g * (-v_v)) / eta_down
        return float(term1 + term2 + term3 + term4)

    def compute_mobility_energy(self, uav_idx, v_h, v_v, t):
        """計算移動能耗，並更新內部狀態"""
        P_mob = self.mobility_power(v_h, v_v, **self.mobility_params)
        E = P_mob * t
        return E

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
