import numpy as np
import math
import matplotlib.pyplot as plt
from object import straight_line_route
# from Environment import UAVEnvironment
# from Fov_model import FovModel
# from Channel_model import ChannelModel


class UAVTask:
    def __init__(self, uav_id, x_u, y_u, z_u, task_type="Unknown", target =None, task_id=None, speed = 20 ):
        self.id = id
        self.x_u = x_u
        self.y_u = y_u
        self.z_u = z_u
        self.task = None  # UAV 當前的任務
        self.target = target  # 地面目標的位置
        self.uav_id = uav_id
        self.task_id = task_id
        self.speed = speed
        # self.task_id = task_id
        self.task_type = task_type
        self.path = []  #  確保 path 變數被初始化
    


    # def assign_task(self, x_g, y_g, z_g, task_type ):
    #     """分派 UAV 的任務"""
    #     self.target = (x_g, y_g, z_g)
    #     self.task = task_type  # 設置 UAV 的任務類型
    #     print(f"UAV {self.id} 被指派到 {self.target} 執行 {self.task}")
    # def move_to_target_for_fov(self, Fov, time=90):
    #     """
    #     逐步移動 UAV 以提升 FOV，直到達到 0.9
    #     """
    #     step_size = 5  # UAV 每次移動的步長
    #     max_iterations = time  # 以防無窮迴圈
    #     iteration = 0
    #     target_x, target_y, target_z = self.target
    #     self.path.append((self.x_u, self.y_u, self.z_u))  # 記錄初始位置
        
    #     while iteration < max_iterations:
    #         # 計算當前 FOV
    #         current_fov = Fov.calculate_coverage([self.x_u], [self.y_u], self.z_u, [target_x], [target_y], target_z)[0, 0]

    #         print(f"UAV {self.uav_id} 當前 FOV: {current_fov:.3f}")

    #         # 如果 FOV = 1，則停止移動
    #         if current_fov == 1:
    #             print(f" UAV {self.uav_id} 已成功達到 FOV = 1，最終位置 ({self.x_u:.2f}, {self.y_u:.2f})")
    #             break
    #         # 計算 UAV 移動方向
    #         direction_x = target_x - self.x_u
    #         direction_y = target_y - self.y_u
    #         direction_z = target_z - self.z_u
    #         norm = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)

    #         # 確保 UAV 按照步長移動
    #         if norm > 0 :
    #             self.x_u += (direction_x / norm) * self.speed
    #             self.y_u += (direction_y / norm) * self.speed
    #             # 檢查 z 軸變化，確保不低於 0
    #             new_z = self.z_u + (direction_z / norm) * self.speed
    #             if new_z >= 5:
    #                 self.z_u = new_z  # 只有當 new_z ≥ 0 才更新 z_u
    #             self.path.append((self.x_u, self.y_u, self.z_u))  # 記錄移動過程
    #         # print(self.path)
    #         # 完成一次迴圈
    #         iteration += 1
    #     return current_fov, self.uav_id
    
    # def get_path(self):
    #     """返回 UAV 的移動路徑"""
    #     return np.array(self.path)
    
    # @staticmethod
    # def euclidean_distance(a, b):
    #     """ 計算兩點間的歐幾里得距離 """
    #     distances = np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=2)
    #     return distances
    
    @staticmethod
    def move_towards_target(start, goal, v_max=1, time=360):
        """搜救隊移動路徑(靠近地面目標)"""
        del time
        route = straight_line_route(start, goal, speed=v_max)
        return np.asarray(route, dtype=float).reshape((-1, 2))
    
    
