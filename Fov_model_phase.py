import math
import numpy as np


class FovModel:
    def __init__(self, f, wl, z_u, i_l, gamma_g):
        """
        初始化視場角模型參數
        :param f: 相機焦距 (focal length)
        :param wl: 視場的寬度 (field width)
        :param il: 視場的長度 (field length)
        :param zg: 無人機高度 (相機的相對高度)
        :param gamma_g: 地面目標的半徑
        """
        self.f = f
        self.wl = wl
        self.z_u = z_u
        self.il = i_l
        self.gamma_g = gamma_g

    

    # def calculate_coverage(self, x_u, y_u, z_u, x_g, y_g, z_g):
    #     """
    #     計算地面目標覆蓋比值 I
    #     :param x_u: 無人機的 x 坐標
    #     :param y_u: 無人機的 y 坐標
    #     :param x_g: 地面目標的 x 坐標
    #     :param y_g: 地面目標的 y 坐標
    #     :return: 覆蓋比值 I
    #     """
        
    #     # if isinstance(z_u, (int, float)):
    #     #     z_u = [z_u] * len(x_u)
    #     # if isinstance(z_g, (int, float)):
    #     #     z_g = [z_g] * len(x_g)
    #     num_uav = 8  # 無人機數量
    #     num_targets = 4  # 地面目標數量
        
    #     # 初始化距離矩陣
    #     distances = np.zeros((num_uav, num_targets))
        
    #     I = np.zeros((num_uav, num_targets))

    #     for i, (u_x, u_y, u_z) in enumerate(zip(x_u, y_u, z_u)):
    #         for j,  (g_x, g_y, g_z) in enumerate(zip(x_g, y_g, z_g)):
    #             # 計算無人機到地面目標的距離
    #             distances[i,j] = math.sqrt((u_x - g_x) ** 2 + (u_y - g_y) ** 2 + (u_z-g_z) ** 2 )
    #             # 計算 I 的分子和分母
    #             numerator = u_z**2 - self.wl**2 * distances / (4 * self.f**2)
    #             denominator = self.wl*self.il * ((distances + u_z**2)**(3/2)) * u_z**3
    #             # 計算覆蓋比值
    #             I  = (self.f**2 * math.pi * self.gamma_g**2 * numerator**2) / denominator
    #     return I
    
    def calculate_fov_single(self, x_u, y_u, z_u, x_g, y_g, z_g):
        fc = self.f      # focal length
        wc = self.wl     # image plane width
        lc = self.il     # image plane length
        rg = self.gamma_g  # ground target radius

        # 幾何量
        zu = float(z_u - z_g)
        r2 = (x_u - x_g)**2 + (y_u - y_g)**2
        R2 = r2 + zu**2
        R32 = R2**1.5

        # 式(1)
        # I = [ f_c^2 * pi * r_g^2 * ( z_u^2 - (w_c^2 * r^2)/(4 f_c^2) )^2 ] /
        #     [ w_c * l_c * ( (x-xg)^2 + (y-yg)^2 + z_u^2 )^{3/2} * z_u^3 ]
        num_term = zu**2 - (wc**2 * r2) / (4.0 * fc**2)
        numerator = (fc**2) * math.pi * (rg**2) * (num_term**2)
        denominator = wc * lc * R32 * (zu**3)

        if denominator <= 0 or not math.isfinite(denominator):
            I_raw = 0.0
        else:
            I_raw = numerator / denominator
            if not math.isfinite(I_raw) or I_raw < 0:
                I_raw = 0.0

        distance = math.sqrt(R2)
        return I_raw, distance

    # def calculate_distance(self, x_u, y_u, x_s, y_s):
    #     num_uav = len(x_u)  # 無人機數量
    #     num_SR = len(x_s)
    #     distances_us = np.zeros((num_uav, num_SR))
    #     for i, (u_x, u_y) in enumerate(zip(x_u, y_u)):
    #         for j, (s_x, s_y) in enumerate(zip(x_s, y_s)):
    #             distances_us[i,j] = math.sqrt((u_x - s_x) ** 2 + (u_y - s_y) ** 2 )
    #     return distances_us

    def get_ground_fov_size(self, z_u, f=0.008, i_l=0.024):
        """
        根據 UAV 高度與相機參數，計算照射地面範圍大小 (fov_w, fov_h)
        """
        # Ground FOV = z_u * (i_l / f)
        fov_size = z_u * (i_l / f)  # 正方形 sensor
        return fov_size, fov_size  # width, height
        

        

# 測試範例
# if __name__ == "__main__":
#     # 初始化模型參數(單位皆為公尺)
#     #焦距越長，物體在照片中的顯示比例越大，適合特寫。
#     #短焦距則讓物體看起來更小，並包含更多背景。

#     f = 0.035  # 相機焦距(短焦距) 
#     wl = 0.0156  # 視場寬度
#     i_l = 0.0235  # 視場長度
#     z_u = 100  # 無人機高度
#     gamma_g = 5  # 地面目標半徑

#     # 創建模型實例
#     model = FovModel(f, wl, z_u, i_l, gamma_g)

#     # 設定無人機和地面目標的坐標
#     x_u, y_u = 0, 0  # 無人機坐標
#     x_g, y_g = 10, 10  # 地面目標坐標

    
#     # 計算覆蓋比值
#     I = model.calculate_coverage(x_u, y_u, x_g, y_g)
#     print(f"地面目標的覆蓋比值 I: {I:.3f}")
    
    
