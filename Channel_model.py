import math
import numpy as np
from scipy.integrate import quad
from scipy.special import iv  # Modified Bessel function of the first kind
import scipy.io



# 讀取 MATLAB .mat 檔案
# mat_data = scipy.io.loadmat('Rician_fading.mat')
# # 提取 b_values 和 g_small_scale
# b_values = mat_data['b_values']  
# g_small_scale = mat_data['g_small_scale']

class ChannelModel:
    def __init__(self):
        pass
    
    @staticmethod
    def PL_uu(H_u, d_3D, f_c):
        """
        向量化計算 UAV-to-UAV 路徑損耗
        支援單值或 NumPy 陣列輸入
        :param H_u: 無人機間高度差 (m)，可為 scalar 或矩陣
        :param d_3D: UAV-to-UAV 3D 距離 (m)，可為 scalar 或矩陣
        :param f_c: 無人機通信頻率 (GHz)
        :return: PL_uu (dB)
        """
        H_u = np.asarray(H_u)
        d_3D = np.asarray(d_3D)

        # term1 根據 H_u 是否大於 0 選擇
        term1 = np.where(H_u > 0, 23.9 - 1.8 * np.log10(np.maximum(H_u, 1e-3)), 20)
        term1 = np.maximum(term1, 20)
        term2 = 20 * np.log10(40 * np.pi * f_c / 3)
        # 完整公式
        PL = term1 * np.log10(np.maximum(d_3D, 1.0)) + term2
        return PL

    @staticmethod
    def SNR_uu(P_u, sigma_sq, PL_uu_t, B_uu):
        """
        向量化計算 UAV-to-UAV SNR (線性值)
        :param P_u: 發射功率 (dBm)
        :param sigma_sq: 雜訊功率 (dBm/Hz)
        :param PL_uu_t: UAV-to-UAV 路徑損耗 (dB)
        :param B_uu: 頻寬 (Hz)
        :return: SNR_uu (線性值)
        """
        PL_uu_t = np.asarray(PL_uu_t)
        signal_linear = 10 ** ((P_u - PL_uu_t) / 10)
        noise_linear = 10 ** ((sigma_sq + 10 * np.log10(B_uu)) / 10)
        return signal_linear / noise_linear

    @staticmethod
    
    def C_uu(B_uu, SNR_uu_t):
        """
        向量化計算 UAV-to-UAV 通道容量 (Mbps)
        :param B_uu: 頻寬 (Hz)
        :param SNR_uu_t: SNR (線性值)
        :return: C_uu (Mbps)
        """
        SNR_uu_t = np.asarray(SNR_uu_t)
        C = B_uu * np.log2(1 + np.maximum(SNR_uu_t, 0))
        return C / 1e6  # Mbps

    @staticmethod
    def PL_ug(distances_ug, f_c):

        distances_ug = np.asarray(distances_ug, dtype=float)
        f_c = np.asarray(f_c, dtype=float)
        term1 = 20 * np.log10(distances_ug)
        term2 = 20 * np.log10(f_c)
        return 32.44 + term1 + term2

    @staticmethod
    def SNR_ug(P_u, sigma_sq, PL_ug_t, B_ug):
        PL_ug_t = np.asarray(PL_ug_t)
        signal_linear = 10 ** ((P_u - PL_ug_t) / 10)
        noise_linear = 10 ** ((sigma_sq + 10 * np.log10(B_ug)) / 10)
        return signal_linear / noise_linear

    @staticmethod
    def C_ug(B_ug, SNR_ug_t):
        SNR_ug_t = np.asarray(SNR_ug_t)
        SNR_ug_t = np.maximum(SNR_ug_t, 0)
        C = B_ug * np.log2(1.0 + SNR_ug_t)
        return C / 1e6  # Mbps


