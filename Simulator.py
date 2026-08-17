import numpy as np
import random
import math
from Channel_model import ChannelModel
from UAV_task import UAVTask
from Fov_model_phase import FovModel
from collections import defaultdict
from Energy_model import EnergyConsumptionModel
from Task_assignment import UAVAssigner, Task
from object import UAV, SRTeam, GroundTarget


class Simulator:
    SR_UAV_CARRIER_GHZ = 2.0
    SR_UAV_BANDWIDTH_HZ = 2e6
    SR_UAV_TX_POWER_DBM = 23.0
    SR_UAV_NOISE_DBM_PER_HZ = -169.0
    SR_UAV_MAX_RANGE_M = 200.0
    SR_UAV_LOS_EXCESS_DB = 2.0
    SR_UAV_NLOS_EXCESS_DB = 2.4

    def __init__(self, num_UAV, p_u = 30): #初始化
        self.dt = 0.25
        self.sr_update_interval = int(1.0 / self.dt)
        self.sim_step_count = 0
        self.B_tot = 10e6
        self.E_max = 10000
        self.num_UAV = num_UAV
        self.p_u = np.full(num_UAV, p_u)
        self.energy_model = EnergyConsumptionModel(E_max=self.E_max, N_u=self.num_UAV)
        # self.last_energy = np.full(self.num_UAV, 1000.0)  # 初始為滿能
        self.PL_uu_t= np.zeros( (self.num_UAV , self.num_UAV))
        self.SNR_uu_t= np.zeros( (self.num_UAV , self.num_UAV))
        self.Capacity_matrix = np.zeros( (self.num_UAV , self.num_UAV))
        self.gs_capacity = np.zeros(self.num_UAV+1)
        self.u2u_nominal_capacity = self.Capacity_matrix.copy()
        self.u2g_nominal_capacity = self.gs_capacity.copy()
        self.active_link_capacities = {}
        self.active_link_bandwidths = {}
        self.GS_pos = (0, 0, 0)
        self.GS_ID = self.num_UAV
        self.num_GT = None
        self.N_u = self.num_UAV
        self.UAVs = []
        self.SR_teams=[]
        self.num_SR = 4
        self.env_width = 1000
        self.env_height= 1000
        self.bit_resolution = 2   # 每 5 公尺為 1 像素單位
        self.grid_size = 5  # 每格 20m，取決於你的 FOV 粒度
        self.map_width = self.env_width // self.bit_resolution
        self.map_height = self.env_height // self.bit_resolution
        self.visited_bitmap = np.zeros((self.map_width, self.map_height), dtype=bool)
        self.uav_list = []
        self.task_list = []
        self.explorer_id_map = np.full((self.map_width, self.map_height), -1)
        self.uav_path = {}
        self.source_uavs = set()
        # 每台無人機開始跳點到GS結束的跳點計算陣列
        self.hop_count = np.zeros(num_UAV, dtype=int)
        # 設定對角線設定對角線
        np.fill_diagonal(self.Capacity_matrix, 0)
        # 初始化延遲違反機率 
        self.violation_prob_matrix = np.zeros((num_UAV + 2, num_UAV + 2))
        self.fov_uavs = []  # 存放 FOV 任務 UAV
        self.cap_uavs = []  # 存放 Capacity 任務 UAV
        self.search_uavs = []
        # ==============設定地面站位置================
        self.x, self.y, self.z = 0, 0, 0
        self.uav_paths = {}  

        # ===== Path-loss cache (for faster per-packet reward) =====
        # These caches are refreshed whenever you update channels.
        # PacketEngine.calculate_packet_reward_fast() can use them to avoid
        # expensive geometry/path-loss recomputation per hop.
        self.PL_uu_cache = np.zeros((self.num_UAV, self.num_UAV), dtype=float)
        self.PL_ug_cache = np.zeros(self.num_UAV, dtype=float)
        self.mobility_params = {
            "comm_safety": {
                "enable": True,
                "mode": "gs_only",
                "gs_pos": (0.0, 0.0, 0.0),
                "r_soft": 180.0,
                "r_hard": 200.0,
                "only_uav_ids": [0,1,4],
            }
            }
        
    def add_uav_path(self, uav_id, path): # 記錄 UAV 路徑
        """存儲 UAV 移動軌跡"""
        self.uav_paths[uav_id] = path  
    # ======================建立無人機列表(左邊集合的頂點)=========================
    def get_available_uav_ids(self):
        """
        取得目前處於 Search 狀態、電量正常、可參與任務分配的 UAV ID 清單
        """
        uav_ids = []
        for uav in self.UAVs:
            uav_ids.append(uav.id)
        return uav_ids

    # ==================任務分配===========================
    def assign_tasks(self):
        uav_id_list = self.get_available_uav_ids() #建立左邊的頂點
        assigner = UAVAssigner(self)
        assigner.assign_tasks(uav_id_list, self.task_list, mode="KM")
        assigner.build_uav_tasks_from_assignment()# 分配結果改成任務列表
    # ====================更新探索區域=====================
    def update_visited_grid(self, uav_id):
        """
        根據 UAV 的 FOV 更新 visited_grid，並判斷是否有 Ground Target 被發現。
        """
        uav = self.uav_dict[uav_id]
        

         
        # === 檢查每個 Ground Target 是否在此 UAV 的 FOV 中 ===
        for gt in self.gts:
            # print(f"  └─ [GT {gt.id}] 位置=({gt.x}, {gt.y}), is_found={gt.is_found}")

            if (not gt.is_found) and self.is_visible(uav_id, gt):
                # print(f"[觸發] UAV {uav_id}  發現 GT {gt.id}，觸發 FOV + COM")
                sr = self.SR_team_gogo(gt)
                gt.mark_found(uav_id)
                 #  直接給 bonus reward
                if  not gt.rewarded:
                    gt.rewarded = True
                    uav.explore_reward_bonus = getattr(uav, "explore_reward_bonus", 0) + 10.0

                # #  K-KM方法: 移除 task_list 中最早的 Search 任務（任選一個）
            #     for i, t in enumerate(self.task_list):
            #         if t.task_type == "Search":
            #             del self.task_list[i]
            #             break
                #  KM方法: 移除 task_list 中最早的兩個 Search 任務
                removed = 0
                for i, t in enumerate(list(self.task_list)):
                    if t.task_type == "Search":
                        del self.task_list[i]
                        removed += 1
                        if removed >= 1:
                            break
                # 新增 FOV 與 COM 任務
                self.task_list.append(Task(
                                    task_id = len(self.task_list),
                                    task_type = "FOV",
                                    target_obj = gt,
                                    target_obj_id = gt.id)
                                    )
                self.task_list.append(Task(
                                    task_id= len(self.task_list), 
                                    task_type="COM", 
                                    target_obj=sr,
                                    target_obj_id = sr.id)
                                    )
                self.need_reassign = True

    def mark_search_coverage(self, uav_id):
        """Mark the boolean camera footprint for centralized Search coverage."""
        uav = self.uav_dict[uav_id]
        model = FovModel(
            f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80
        )
        fov_w, fov_h = model.get_ground_fov_size(uav.z_u)
        bx_min, bx_max, by_min, by_max, _, _ = self.fov_to_indices_and_patch(
            uav.x_u,
            uav.y_u,
            fov_w,
            fov_h,
            self.env_width,
            self.env_height,
            self.bit_resolution,
            self.visited_bitmap,
        )
        if bx_max < bx_min or by_max < by_min:
            return
        self.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True
        uav.last_box_idx = (bx_min, bx_max, by_min, by_max)
        
        # if (not self.search_completed) and self.is_search_done(cov_th=0.8, min_found=4):
        #     self.search_completed = True
        #     print(f" Search done in env: cov={self.visited_bitmap.mean():.3f}, found={self.count_found_targets()}")
        #     self.task_list = [t for t in self.task_list if t.task_type != "Search"]
        #     for uid in getattr(self, "multi_tasks", {}):
        #         self.multi_tasks[uid] = [t for t in self.multi_tasks[uid] if t["task_type"] != "Search"]

    # ===============判斷TG是否有被發現=====================
    def is_visible(self, uav_id, target):
        # =========方法一===============
        uav = self.uav_dict[uav_id]
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)
        # print(fov_w, fov_h)
        fc = self.FovModel.f
        wc = self.FovModel.wl
        lc = self.FovModel.il

        # 幾何位置
        x_u, y_u, z_u = uav.get_position()
        x_g, y_g, z_g = target.x, target.y, getattr(target, "z", 0.0)
        r_g = getattr(target, "radius", self.FovModel.gamma_g)  # 目標半徑
        zu = z_u - z_g

        if zu <= 0:
            return False  # 相機不在目標上方，直接不可見

        zu = float(z_u - z_g)
        # ===== 檢查式(2)：FoV 圓錐/半徑約束（確保目標落在視野範圍內）=====
        # 式(2) 等價 r <= (2 f_c / w_c) * z_u
        r = math.hypot(x_u - x_g, y_u - y_g)
        alpha = (2.0 * fc) / wc
        beta  = (2.0 * fc) / lc
        # if r > alpha * zu + 1e-9:
        #     # print(f"[WARN] r={r:.3f} > r_max={r_max:.3f}  (units wrong or (2) not enforced)")
        #     return False

        # ===== 檢查式(3)：目標整個圓要被影像平面包住 =====
        # --- 式(3) 嚴格幾何版
        # 水平左右邊距
        dL = (zu*zu + r*r) / (alpha * zu + r)
        dR = (zu*zu + r*r) / max(alpha * zu - r, 1e-9)  # 防除以零

        # 垂直上下邊距（用 ℓ_c 與同型式；若要更嚴謹可把 r 換成沿 y 方向的投影距離）
        dB = (zu*zu + r*r) / (beta * zu + r)
        dT = (zu*zu + r*r) / max(beta * zu - r, 1e-9)

        if min(dL, dR, dB, dT) < r_g:
            return False
            

        # 額外：可在這裡計算 I_raw（原式(1)），作為權重或拍攝質量參考
        # I_raw, _ = self.calculate_fov_single(x_u, y_u, z_u, x_g, y_g, z_g)
        # 例如你要避免「幾何合法但 I 極小」的觸發，也能在這裡加門檻：
        # if I_raw < 1e-3: return False

        return (not target.is_found)


       
    def count_found_targets(self) -> int:
        return sum(1 for gt in self.gts if gt.is_found)

    # def is_search_done(self, cov_th: float = 0.8, min_found: int = 4) -> bool:
    #     cov = float(self.visited_bitmap.mean())
    #     return (cov >= cov_th) and (self.count_found_targets() >= min_found)
    def begin_step(self):
        """每個 step 開始時呼叫：初始化 pending flag，建立步前地圖快照。"""
        self._pending_search_done = False
        self._pending_reason = None
        # 公平：所有 UAV 本步的 newly_explored 皆相對同一張快照計算
        self._pre_map = self.visited_bitmap.copy()
        if self.sim_step_count % self.sr_update_interval == 0:
            self.advance_sr_teams()

        self.sim_step_count += 1

    def convert_search_to_hovering(self):
        """將所有 Search 任務轉換成 Hovering，避免 UAV 被重新分配到 FOV/COM。"""
        # 全域 task_list 內的 Search 改為 Hovering
        for t in self.task_list:
            if t.task_type == "Search":
                t.task_type = "Hovering"
                # Hovering 的目標就設為 UAV 目前位置
                uav = self.uav_dict[t.target_obj_id]
                t.target_obj = uav
                t.target_obj_id = uav.id

        # 每台 UAV 的 multi_tasks 裡也同步修改
        for uid in list(self.multi_tasks.keys()):
            new_tasks = []
            for task in self.multi_tasks[uid]:
                if task["task_type"] == "Search":
                    # 改成 Hovering，位置維持 UAV 當前位置
                    uav = self.uav_dict[uid]
                    new_tasks.append({
                        "task_type": "Hovering",
                        "target_id": task["target_id"],
                        "target_obj_id": uid,
                        "target_pos": uav.get_position()
                    })
                else:
                    new_tasks.append(task)
            self.multi_tasks[uid] = new_tasks

        # 更新 UAV 狀態
        for uid, uav in self.uav_dict.items():
            tasks = self.multi_tasks.get(uid, [])
            if tasks:
                uav.active_task_index = uav.active_task_index % len(tasks)
                # 如果第一個是 Hovering，就直接設定
                uav.task_type = tasks[0]["task_type"]
                uav.target_position = tasks[0]["target_pos"]
    # =============搜救隊出發===============================
    def SR_team_gogo(self, gt):
        """
        找出距離目標最近的 SR team，並更新其移動目標
        """
        gt_pos = np.array([gt.x, gt.y])
        sr_positions = np.array([
            (sr.x, sr.y) for sr in self.SR_teams if not sr.active
        ])
        available_indices = [
            i for i, sr in enumerate(self.SR_teams) if not sr.active
        ]

        # 找距離 GT 最近的 SR 成員
        if len(sr_positions) > 0:
            dists = np.linalg.norm(sr_positions - gt_pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_sr_idx = available_indices[nearest_idx]

            sr = self.SR_teams[nearest_sr_idx]
            start = (sr.x, sr.y)
            goal = (gt.x, gt.y)
            sr.path = list(
                UAVTask.move_towards_target(
                    start, goal, v_max=float(getattr(self, "sr_speed_mps", 1.0))
                )
            )
            # print(sr.path)
            # print(sr.x, sr.y)
            sr.active= True
            sr.assigned_gt_id = gt.id
            sr.current_step = 0          # 從路徑頭開始
            sr.arrived = False
            self.SR_paths = sr.path
            gt.assigned = True
            # print(f"[SR Team] 指派 SR {sr.id} 前往 GT {gt.id}")
            # print(self.SR_paths)
            return sr
    def advance_sr_teams(self):
        for sr in self.SR_teams:
            # 先把「推進前」的位置也記一筆，確保軌跡連續
            self.sr_trajectory[sr.id].append([sr.x, sr.y, sr.z])
            sr.step_forward()
            # 推進後再記一筆
            self.sr_trajectory[sr.id].append([sr.x, sr.y, sr.z])
    def get_unexplored_ratio(self, uav_id):
        uav = self.uav_dict[uav_id]
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)

        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)
        x, y, _ = uav.get_position()

        x_min = max(0, x - fov_w / 2)
        x_max = min(self.env_width, x + fov_w / 2)
        y_min = max(0, y - fov_h / 2)
        y_max = min(self.env_height, y + fov_h / 2)

        bx_min = int(x_min / self.bit_resolution)
        bx_max = int(x_max / self.bit_resolution)
        by_min = int(y_min / self.bit_resolution)
        by_max = int(y_max / self.bit_resolution)

        bx_min = min(max(0, bx_min), self.map_width - 1)
        bx_max = min(max(0, bx_max), self.map_width - 1)
        by_min = min(max(0, by_min), self.map_height - 1)
        by_max = min(max(0, by_max), self.map_height - 1)

        submap = self.visited_bitmap[bx_min:bx_max + 1, by_min:by_max + 1]
        total = submap.size
        unexplored = np.sum(~submap)
        return unexplored / total if total > 0 else 0.0

    #=====================通訊如何======================== 
    def _get_sr_uav_link_metrics(self, uav_id, sr_id):
        uav = self.uav_dict[uav_id]
        sr = self.SR_teams[sr_id]

        # 位置
        uav_pos = np.array([uav.x_u, uav.y_u, uav.z_u])
        sr_pos = np.array([sr.x, sr.y, sr.z])
        vec = uav_pos - sr_pos
        d_3d = float(np.linalg.norm(vec))
        
        # 距離與高度
        d_safe = max(d_3d, 1e-3)
        H_u = float(abs(vec[2]))                      # 垂直距離（基本上就是 UAV 高度）

        # 通訊半徑：超出半徑時 SNR 與容量皆為零。
        if d_safe > self.SR_UAV_MAX_RANGE_M:
            return 0.0, 0.0

        # 仰角
        ratio = np.clip(H_u / d_safe, -1.0, 1.0)
        elevation_angle = np.degrees(np.arcsin(ratio))

        # LoS 機率
        LoS_prob = 1.0 / (1.0 + 4.88 * np.exp(-0.429 * (elevation_angle - 4.88)))

        # ===== Path Loss（向量化） =====
        FSPL = ChannelModel.PL_ug(d_safe, self.SR_UAV_CARRIER_GHZ)

        expected_pl = (
            FSPL
            + LoS_prob * self.SR_UAV_LOS_EXCESS_DB
            + (1 - LoS_prob) * self.SR_UAV_NLOS_EXCESS_DB
        )

        # ===== SNR + Capacity =====
        snr_us = float(
            ChannelModel.SNR_ug(
                self.SR_UAV_TX_POWER_DBM,
                self.SR_UAV_NOISE_DBM_PER_HZ,
                expected_pl,
                self.SR_UAV_BANDWIDTH_HZ,
            )
        )
        capacity_mbps = float(
            ChannelModel.C_ug(self.SR_UAV_BANDWIDTH_HZ, snr_us)
        )
        return snr_us, capacity_mbps

    def get_snr(self, uav_id, sr_id):
        """Return the SR-UAV link SNR as a linear ratio."""

        snr_us, _ = self._get_sr_uav_link_metrics(uav_id, sr_id)
        return snr_us

    def get_sr_uav_capacity_mbps(self, uav_id, sr_id):
        """Return the canonical SR-UAV link capacity in Mbps."""

        _, capacity_mbps = self._get_sr_uav_link_metrics(uav_id, sr_id)
        return capacity_mbps

    # =====================U2U channel model================================
    def update_u2u_channels(self):
        P_u = 30
        sigma_sq = -169  # 噪聲功率 (dBm/Hz)
        self.B_tot = 10e6       # 頻寬 10 MHz
        f_c = 2.4        # GHz
        d_a2a = 400.0
        cap_eps_mbps = 0.1  # 可行的鏈路
        # =====  收集所有 UAV 的 (x, y, z) 位置 =====
        pos = np.array([[u.x_u, u.y_u, u.z_u] for u in self.UAVs])  # shape (N, 3)
        N = pos.shape[0]

        # =====  批次計算所有配對間距 =====
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
        d_3D = np.linalg.norm(diff, axis=2)                   # (N, N)
        dx = diff[:, :, 0]
        dy = diff[:, :, 1]
        dz = diff[:, :, 2]
        d_2D = np.sqrt(dx*dx + dy*dy)          # (N, N) 水平距離
        d_3D = np.sqrt(dx*dx + dy*dy + dz*dz)  # (N, N) 3D 距離
        # A2A 通訊半徑遮罩
        in_range = d_2D<= d_a2a         

        d_safe = np.maximum(d_3D, 1e-3)
        H_u = np.abs(diff[:, :, 2])                           # 垂直距離 (N, N)

        # 避免對角線為 0 導致除以零
        np.fill_diagonal(d_safe, 1e6)
        np.fill_diagonal(H_u, 0)


        # =====  批次計算路徑損耗 / SNR / 通道容量 =====
        PL = ChannelModel.PL_uu(H_u, d_safe, f_c)
        SNR = ChannelModel.SNR_uu(P_u, sigma_sq, PL, self.B_tot)
        capacity = ChannelModel.C_uu(self.B_tot, SNR)

        try:
            self.PL_uu_cache = np.array(PL, dtype=float)
            np.fill_diagonal(self.PL_uu_cache, 0.0)
        except Exception:
            # If PL is not array-like for any reason, skip caching.
            pass

        # =====  對角線容量設為 0，自身不通訊 =====
        np.fill_diagonal(capacity, 0.0)

        # 半徑外的 link 容量設為 0
        capacity[~in_range] = 0.0

        # ========計算可行的連接================
        feasible = (capacity > cap_eps_mbps)
        # 不算自己
        np.fill_diagonal(feasible, False)
        # 每台 UAV 的可行鄰居數 k_u(t)
        k_u = feasible.sum(axis=1)  
        self.k_u_u2u = k_u 
        # 平均可行鄰居數 k_bar(t)
        k_bar = float(k_u.mean())
        # 存到 env
        self.k_bar_u2u = k_bar
        # Candidate actions use nominal full-pool quality. Actual slot capacity
        # is computed only after all proposed links are known.
        self.B_eff_u2u = np.full(N, self.B_tot, dtype=float)
        self.u2u_nominal_capacity = np.array(capacity, dtype=float)
        self.Capacity_matrix = self.u2u_nominal_capacity.copy()

    # ==========================U2G channel model============================
    # 無人機與地面站
    def update_u2g_channels(self):
        P_u = 30
        sigma_sq = -169  # dBm/Hz
        B_ug = 10e6
        f_c = 2
        d_A2G = 200.0 
        eta_LoS = 2
        eta_NLoS = 2.4

        # UAV positions (N, 3)
        pos = np.array([[u.x_u, u.y_u, u.z_u] for u in self.UAVs])
        gs_pos = np.array([self.x, self.y, self.z])

        # ===== 計算距離 =====
        diff = pos - gs_pos                 # (N, 3)

        # 3D 距離
        d_3D = np.linalg.norm(diff, axis=1)
        d_safe = np.maximum(d_3D, 1e-3)

        # 水平距離
        # d_2D = np.linalg.norm(diff[:, :2], axis=1)  # (N,)
        # 高度（仍可用於仰角、LoS 機率）
        H_u = np.abs(diff[:, 2])

        # 用3D距離判斷是否可通訊
        in_range = d_3D <= d_A2G

        # 仰角
        elevation_angle = np.degrees(
            np.arcsin(np.clip(H_u / d_safe, -1.0, 1.0))
        )

        # ===== LoS 機率（向量化） =====
        LoS_prob = 1.0 / (1.0 + 4.88 * np.exp(-0.429 * (elevation_angle - 4.88)))

        # ===== Path Loss（向量化） =====
        FSPL = ChannelModel.PL_ug(d_safe, f_c)

        self.Expected_PL = FSPL + LoS_prob * eta_LoS + (1-LoS_prob) * eta_NLoS
        self.PL_ug_cache = np.array(self.Expected_PL, dtype=float)

        # ===== SNR + Capacity =====
        SNR_ug = ChannelModel.SNR_ug(P_u, sigma_sq, self.Expected_PL,  B_ug)
        C_mix = ChannelModel.C_ug(B_ug, SNR_ug)

        # 半徑外的 UAV 容量設為 0
        C_mix[~in_range] = 0.0

        self.u2g_nominal_capacity = np.array(C_mix, dtype=float)
        self.gs_capacity = self.u2g_nominal_capacity.copy()
        # print(self.gs_capacity)

    def allocate_active_link_capacities(self, proposed_links):
        """Allocate independent 10 MHz pools across actual U2U/U2G links."""

        active_links = [
            (int(sender), int(receiver))
            for sender, receiver in sorted(proposed_links.items())
            if int(receiver) != int(sender)
        ]
        u2u_links = [link for link in active_links if link[1] < self.num_UAV]
        u2g_links = [link for link in active_links if link[1] == self.GS_ID]
        u2u_bandwidth = self.B_tot / len(u2u_links) if u2u_links else 0.0
        u2g_bandwidth = self.B_tot / len(u2g_links) if u2g_links else 0.0

        capacities = {}
        bandwidths = {}
        for sender, receiver in u2u_links:
            bandwidths[(sender, receiver)] = float(u2u_bandwidth)
            snr = ChannelModel.SNR_uu(
                float(self.p_u[sender]),
                -169.0,
                float(self.PL_uu_cache[sender, receiver]),
                u2u_bandwidth,
            )
            capacities[(sender, receiver)] = float(
                ChannelModel.C_uu(u2u_bandwidth, snr)
            )
        for sender, receiver in u2g_links:
            bandwidths[(sender, receiver)] = float(u2g_bandwidth)
            snr = ChannelModel.SNR_ug(
                float(self.p_u[sender]),
                -169.0,
                float(self.PL_ug_cache[sender]),
                u2g_bandwidth,
            )
            capacities[(sender, receiver)] = float(
                ChannelModel.C_ug(u2g_bandwidth, snr)
            )

        self.active_link_capacities = capacities
        self.active_link_bandwidths = bandwidths
        return capacities, bandwidths

    #===========回傳不可選擇的節點======================== 
    def get_routing_action_mask(self, from_uav_id, cap_eps=0.1):

        num_uav = self.num_UAV
        num_actions = num_uav + 1
        mask = np.zeros(num_actions, dtype=np.float32)
        # The sender's own index is the explicit Wait action.
        mask[from_uav_id] = 1.0

        # UAV → UAV link
        if self.Capacity_matrix is not None:
            for to_id in range(num_uav):
                if to_id == from_uav_id:
                    continue
                cap = float(self.Capacity_matrix[from_uav_id, to_id])
                if cap > cap_eps:
                    mask[to_id] = 1.0

        # UAV → GS
        if self.gs_capacity is not None:
            cap_gs = float(self.gs_capacity[from_uav_id])
            if cap_gs > cap_eps:
                mask[num_uav] = 1.0

        return mask


    # =================辨認封包來源的無人機編號=========================
    def update_source_uavs(self):
        self.source_uavs = set()
        for uav in self.UAVs:
            uav_id = uav.id
            task_list = self.multi_tasks.get(uav_id, [])
            for task in task_list:
                if task["task_type"] in ["FOV", "COM"]:
                    self.source_uavs.add(uav_id)
                    break  # 一旦有一個符合就可以加入，跳出這台 UAV 的任務迴圈
        # print(f"[DEBUG] Source UAVs: {sorted(self.source_uavs)}")
    
    def calculate_fov_reward(self, uav_id, lamda_EE, E_mob=None):

        uav = self.uav_dict[uav_id]

        if E_mob is None:
            E_mob = float(getattr(uav, "move_energy_step", 0.0))

        # =========================================================
        # 1) 計算目前 FOV
        # =========================================================
        tx, ty, tz = uav.target_position
        # print(tx, ty, tz)
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(
                f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80
            )
        else:
            if hasattr(self.FovModel, "z_u"):
                self.FovModel.z_u = uav.z_u

        fov, _ = self.FovModel.calculate_fov_single(
            uav.x_u, uav.y_u, uav.z_u, tx, ty, tz
        )
        fov = float(fov)

        # =========================================================
        # 2) FOV 主 reward：越接近 1 越好
        #    >1 的 overshoot 要罰更重，避免只靠降高度硬修
        # =========================================================
        err = abs(fov - 1.0)

        if fov > 1.0:
            r_fov = 1.0 - 1.8 * (err ** 1.5)
        else:
            r_fov = 1.0 - 1.0 * (err ** 1.2)

        r_fov = float(np.clip(r_fov, -4.0, 1.2))

        # =========================================================
        # 3) XY 對位 reward：離 target 越近越好
        # =========================================================
        dist_xy = float(np.hypot(uav.x_u - tx, uav.y_u - ty))
        env_diag = float(np.hypot(self.env_width, self.env_height))
        dist_xy_norm = dist_xy / (env_diag + 1e-9)

        # 靜態距離懲罰
        r_xy = -1.5 * dist_xy_norm

        # 動態進步獎勵：比上一刻更接近就加分
        prev_dist_xy = getattr(uav, "prev_dist_xy", None)
        if prev_dist_xy is None:
            delta_dist = 0.0
        else:
            delta_dist = float(prev_dist_xy - dist_xy)

        uav.prev_dist_xy = dist_xy

        # 正向靠近比負向遠離更重要
        r_xy_progress = 4.0 * delta_dist
        r_xy_progress = float(np.clip(r_xy_progress, -1.0, 1.0))

        # =========================================================
        # 4) 水平幾乎不動的懲罰
        # =========================================================
        move_xy = float(np.hypot(getattr(uav, "last_dx", 0.0), getattr(uav, "last_dy", 0.0)))
        r_stall_xy = -0.20 if move_xy < 0.15 else 0.0

        # =========================================================
        # 5) 高度 shaping
        #    FOV 任務偏中低高度，但不要只靠高度解問題
        # =========================================================
        z = float(uav.z_u)
        z_min = float(getattr(uav, "min_AGL", 50.0))
        z_max = float(getattr(uav, "max_AGL", 150.0))
        z_norm = (z - z_min) / (z_max - z_min + 1e-9)

        z_mid = 0.30
        r_alt = -0.35 * ((z_norm - z_mid) ** 2)

        # =========================================================
        # 6) 邊界懲罰
        # =========================================================
        edge_penalty = 0.0
        if z_norm > 0.90:
            edge_penalty -= 2.0 * ((z_norm - 0.90) / 0.10) ** 2
        elif z_norm < 0.10:
            edge_penalty -= 1.2 * ((0.10 - z_norm) / 0.10) ** 2

        # =========================================================
        # 7) 能耗項
        # =========================================================
        E_ref = float(max(getattr(self, "_E_move_ref", 500.0), 1e-3))
        energy_term = float(np.clip(E_mob / E_ref, 0.0, 10.0))
        r_energy = -lamda_EE * energy_term

        # =========================================================
        # 8) 合成 reward
        # =========================================================
        reward = (
            2.2 * r_fov
            + 1.2 * r_xy
            + 1.5 * r_xy_progress
            + r_stall_xy
            + r_alt
            + edge_penalty
            + r_energy
        )

        reward = float(np.clip(reward, -6.0, 6.0))

        return reward, err, fov
    def calculate_fov_reward_wo_Dinkel(self, uav_id, E_mob=None, debug=False):
        uav = self.uav_dict[uav_id]

        # --- 能耗正規化 ---
        if E_mob is None:
            E_mob = float(getattr(uav, "move_energy_step", 0.0))
        if not hasattr(self, "_E_move_ref"):
            self._E_move_ref = max(E_mob, 1.0)
        else:
            self._E_move_ref = 0.9*self._E_move_ref + 0.1*max(E_mob, 1e-3)
        e_norm = max(E_mob / self._E_move_ref, 1e-3)

        # --- FOV ---
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
        else:
            if hasattr(self.FovModel, "z_u"):
                self.FovModel.z_u = uav.z_u

        tx, ty, tz = uav.target_position
        fov, _ = self.FovModel.calculate_fov_single(uav.x_u, uav.y_u, uav.z_u, tx, ty, tz)

        # --- 以 1 為目標的對稱形狀獎勵（峰值在 1，兩側都下降）---
        # 高斯形狀：r_shape ∈ (0, A]，err=0 時達到 A
        A = 2.5          # 形狀峰值（每步不宜太大，避免主導整體）
        sigma = 0.45    # 允許帶寬；越小越尖銳
        err = abs(fov - 1.0)
        r_shape = A * np.exp(-(err / sigma)**2)

        # **基準扣除**：讓 FOV=0.5 的回饋≈0，避免 0.5 成為「穩賺」點
        baseline = A * np.exp(-((1.0 - 0.3) / sigma)**2)
        r_centered = r_shape - baseline   # 1.0 附近為正、0.5 附近為 0、遠離為負

        # 進步獎勵（限制幅度，避免一次大跳過衝）
        last_err = getattr(uav, "last_fov_err", None)
        uav.last_fov_err = err
        d_err = 0.0 if last_err is None else (last_err - err)
        r_gain = 0.5 * max(0.0, min(d_err, 0.2))  # 小步前進才加分

        # overshoot 懲罰（柔化，避免一次過衝災難性負分）
        overshoot = max(fov - 1.0, 0.0)
        r_overpen = 1 * np.sqrt(overshoot)      # 原本是 2*x，改為 sqrt 以降低斜率

        # 能耗（保持很輕）
        r_energy = 0.1 * max(np.log(e_norm), 0.0)
        if r_energy <= 0:
            reward = -1.0   # 或一個小負值
        else:
            reward = (r_centered + r_gain - r_overpen) / r_energy

        # reward = (r_centered + r_gain - r_overpen) / r_energy

        # 近目標帶小獎金（幫助穩在 1），但不要太大
        if err < 0.15: reward += 0.2
        if err < 0.05: reward += 0.6

        # 不要太緊的截斷，避免壓掉「接近 1」的區分度
        reward = float(np.clip(reward, -2.0, 4.0))

        if debug:
            print(f"[UAV {uav_id}] FOV={fov:.3f}, err={err:.3f}, e_norm={e_norm:.3f}")
            print(f"  r_shape={r_shape:.3f}, baseline={baseline:.3f}, r_centered={r_centered:.3f}, "
                f"r_gain={r_gain:.3f}, r_overpen={r_overpen:.3f}, r_energy={r_energy:.3f}, "
                f"Total={reward:.3f}")

        return reward, err, fov
    def calculate_search_reward(self, uav_id, lamda_EE, E_mob=None):
        uav = self.uav_dict[uav_id]
        if E_mob is None:
            E_mob = float(getattr(uav, "move_energy_step", 0.0))

        # === FOV 邊界（連續→格點）===
        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)
        x, y, _ = uav.get_position()
        bx_min, bx_max, by_min, by_max, patch, fov_cells = self.fov_to_indices_and_patch(
            x, y, fov_w, fov_h,
            self.env_width, self.env_height,
            self.bit_resolution, self.visited_bitmap
        )
        if not (bx_max >= bx_min and by_max >= by_min):
            return 0.0, 0.0, 0.0
        fov_cells = max(1, fov_cells)

        # --- 新探索：只看「本次FOV」區塊 ---
        cur_map = getattr(self, "_pre_map", self.visited_bitmap)
        cur = cur_map[bx_min:bx_max+1, by_min:by_max+1]
        newly_explored = int((~cur).sum())
        p = newly_explored / float(fov_cells)   # 0~1

        # --- 與上一張FOV的局部重疊，用來懲罰 ---
        if uav.last_box_idx is not None:
            lbx0, lbx1, lby0, lby1 = uav.last_box_idx
            ix0 = max(bx_min, lbx0)
            iy0 = max(by_min, lby0)
            ix1 = min(bx_max, lbx1)
            iy1 = min(by_max, lby1)
            inter_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1) if (ix1 >= ix0 and iy1 >= iy0) else 0
            overlap_rate_local = inter_cells / float(fov_cells)
        else:
            overlap_rate_local = 0.0

        # --- 局部重疊懲罰 ---
        overlap_penalty = max(0.0, overlap_rate_local - 0.5) ** 2

        # --- 幾乎沒動的懲罰 ---
        no_move_penalty = 0.1 if overlap_rate_local >= 0.95 else 0.0

        # --- 能耗項 ---
        E_ref = float(max(getattr(self, "_E_move_ref", 500.0), 1e-3))
        energy_term = np.clip(E_mob / E_ref, 0.0, 10.0)

        # =========================================================
        # 1) 原本的 Search 主回饋
        # =========================================================
        explore = 1.2 * (p ** 0.6) + 0.6 * p
        raw = (explore - overlap_penalty - no_move_penalty) - lamda_EE * energy_term

        # =========================================================
        # 2) Search 高度 shaping：偏好較高，但不要貼頂
        # =========================================================
        z = float(uav.z_u)
        z_min = float(getattr(uav, "min_AGL", 50.0))
        z_max = float(getattr(uav, "max_AGL", 150.0))
        z_norm = (z - z_min) / (z_max - z_min + 1e-9)

        # Search 希望在偏高高度帶，而不是直接衝到最上界
        z_target = 0.72
        z_band = 0.22
        r_alt_search = -0.45 * ((z_norm - z_target) / z_band) ** 2
        r_alt_search = float(np.clip(r_alt_search, -0.8, 0.0))

        # =========================================================
        # 3) 邊界懲罰：避免吸到上下界
        # =========================================================
        edge_penalty = 0.0
        if z_norm > 0.90:
            edge_penalty -= 1.5 * ((z_norm - 0.90) / 0.10) ** 2
        elif z_norm < 0.10:
            edge_penalty -= 1.5 * ((0.10 - z_norm) / 0.10) ** 2

        # =========================================================
        # 4) 合成 reward
        # =========================================================
        reward = raw + r_alt_search + edge_penalty
        reward = float(np.clip(reward, -3.5, 3.5))

        # 外部事件獎勵（例如找到 GT）
        evt_ext = float(getattr(uav, "explore_reward_bonus", 0.0))
        total_reward = reward + evt_ext

        # --- 狀態更新 ---
        self.visited_bitmap[bx_min:bx_max+1, by_min:by_max+1] = True
        uav.last_box_idx = (bx_min, bx_max, by_min, by_max)
        uav.explore_reward_bonus = 0.0

        return total_reward, 0.0, 0.0
    
    def calculate_search_reward_wo_Dinkel(self, uav_id, t, E_mob=None):
        uav = self.uav_dict[uav_id]
        if E_mob is None:
            E_mob = float(getattr(uav, "move_energy_step", 0.0))

        # === FOV 邊界（連續→格點）===
        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)
        x, y, _ = uav.get_position()
        bx_min, bx_max, by_min, by_max, patch, fov_cells = self.fov_to_indices_and_patch(
            x, y, fov_w, fov_h,
            self.env_width, self.env_height,
            self.bit_resolution, self.visited_bitmap
        )
        if not (bx_max >= bx_min and by_max >= by_min):
            return 0.0, 0.0, 0.0
        fov_cells = max(1, fov_cells)

        # --- 新探索：只看「本次FOV」區塊（關鍵）---
        cur_map = getattr(self, "_pre_map", self.visited_bitmap)
        cur = cur_map[bx_min:bx_max+1, by_min:by_max+1]
        newly_explored = int((~cur).sum())
        p = newly_explored / float(fov_cells)   # 0~1

        # --- 與上一張FOV的局部重疊，用來懲罰 ---
        if uav.last_box_idx is not None:
            lbx0, lbx1, lby0, lby1 = uav.last_box_idx
            ix0 = max(bx_min, lbx0);  iy0 = max(by_min, lby0)
            ix1 = min(bx_max, lbx1);  iy1 = min(by_max, lby1)
            inter_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1) if (ix1 >= ix0 and iy1 >= iy0) else 0
            overlap_rate_local = inter_cells / float(fov_cells)
        else:
            overlap_rate_local = 0.0

        # 局部重疊懲罰（>0.5 才懲罰）
        overlap_penalty = max(0.0, overlap_rate_local - 0.5) ** 2

        # 「幾乎沒動」的極小懲罰（可關掉）
        no_move_penalty = 0.1 if overlap_rate_local >= 0.95 else 0.0

        # --- 能耗項（保持量級穩定）---
        E_ref = float(max(getattr(self, "_E_move_ref", 500.0), 1e-3))
        energy_term = np.clip(E_mob / E_ref, 0.0, 10.0)

        # --- Stationary shaping（拿掉 early_scale 與 tanh）---
        explore = 1.2 * (p ** 0.6) + 0.6 * p       # 典型 0~1.8
        if energy_term  <= 0:
            raw = -1.0   # 或一個小負值
        else:
            raw = (explore -  overlap_penalty - no_move_penalty) / energy_term
        # raw = (explore -  overlap_penalty - no_move_penalty) /  energy_term 
        reward = float(np.clip(raw, -3.0, 3.0))
        # print(lamda_EE * energy_term)
        # 外部事件獎勵（建議量級 10~20，而非 100）
        evt_ext = float(getattr(uav, "explore_reward_bonus", 0.0))
        total_reward = reward + evt_ext

        # --- 狀態更新 ---
        self.visited_bitmap[bx_min:bx_max+1, by_min:by_max+1] = True
        uav.last_box_idx = (bx_min, bx_max, by_min, by_max)
        # 當步把 bonus 吃完就清零，避免跨步殘留
        uav.explore_reward_bonus = 0.0
        # 事件獎勵是否清除，建議在「事件觸發處」清；若要在此清：
        # uav.explore_reward_bonus = 0.0

        return total_reward, 0.0, 0.0

        
    def fov_to_indices_and_patch(self, x, y, fov_w, fov_h,
                             env_width, env_height,
                             bit_resolution, visited_bitmap):
        # 連續座標 → 區域邊界
        x_min = max(0.0, x - fov_w / 2)
        x_max = min(env_width,  x + fov_w / 2)
        y_min = max(0.0, y - fov_h / 2)
        y_max = min(env_height, y + fov_h / 2)

        br = float(bit_resolution)
        # floor 切格（右上角 -1e-6 與你的寫法一致）
        bx_min = int(x_min // br)
        bx_max = int((x_max - 1e-6) // br)
        by_min = int(y_min // br)
        by_max = int((y_max - 1e-6) // br)

        # 夾在合法格點範圍
        BX, BY = visited_bitmap.shape
        bx_min = max(0, min(bx_min, BX - 1))
        bx_max = max(bx_min, min(bx_max, BX - 1))
        by_min = max(0, min(by_min, BY - 1))
        by_max = max(by_min, min(by_max, BY - 1))

        # 取 patch（np view，不複製）
        if bx_max >= bx_min and by_max >= by_min:
            patch = visited_bitmap[bx_min:bx_max+1, by_min:by_max+1]
            fov_cells = (bx_max - bx_min + 1) * (by_max - by_min + 1)
        else:
            patch = visited_bitmap[0:0, 0:0]  # 空 view
            fov_cells = 0

        return bx_min, bx_max, by_min, by_max, patch, fov_cells   
    
    def fixed_boundary_points(self, w, h):
        return [
            (0, h / 2),      # West
            (w, h / 2),      # East
            (w / 2, 0),      # South
            (w / 2, h),      # North
        ]

    # ================隨機GT版本=============================
    def reset_environment(self, scenario_entry=None):
        if scenario_entry is not None:
            from scenario_manifest import validate_scenario_entry

            validate_scenario_entry(scenario_entry)
            self.num_GT = int(scenario_entry["num_GT"])
            self.active_scenario_id = str(scenario_entry["scenario_id"])
            self.active_scenario_seed = int(scenario_entry["scenario_seed"])
            self.traffic_primitives = dict(scenario_entry["traffic_primitives"])
        else:
            self.active_scenario_id = None
            self.active_scenario_seed = None
            self.traffic_primitives = {
                "load_factor": 1.0,
                "base_fov_packets_per_second": 5.0,
                "base_com_packets_per_second": 50.0,
                "generation_model": "task-and-fov-gated-rate-accumulator-v1",
            }
        self.load_factor = float(self.traffic_primitives["load_factor"])
        self.need_reassign = True   
        self.UAVs.clear()
        self.current_time = 0
        self.uav_tasks = {}
        self.multi_tasks = {}
        self._search_phase_over: bool = False   # 是否已經完成搜尋相位（環境內部 guard）
        self._pending_search_done: bool = False # 本 step 是否有人達標（步末集中處理）
        self._pending_reason = None  
        self.visited_bitmap  = np.zeros((self.env_width // self.bit_resolution, self.env_height // self.bit_resolution), dtype=bool)
        self.task_list = []
        # self.UAVs = []
        self.explorer_id_map[:, :] = -1
        self.search_completed = False
        # self.last_energy = np.full(self.energy_model.N_u, self.energy_model.E_max)
        self.source_uavs = set()  #  清除封包來源
        self.forwarding_rate = {uav_id: 0 for uav_id in range(self.num_UAV)}  # if needed
        self.source_buffer = defaultdict(float)  #  封包累積用 buffer 重置
        self.num_SR_team=self.num_GT
        # ==================初始化無人機位置=============================
        self.UAVs = []           # UAV list（順序）
        self.uav_dict = {}       # id → UAV 的查表 dict
        self.uav_tasks = {}      # 任務查表（維持原本）
        if scenario_entry is None:
            uav_initial_data = [
                {
                    "uav_id": index,
                    "position": [x_u, y_u, random.uniform(80, 120)],
                    "energy_j": self.E_max,
                }
                for index, (x_u, y_u) in enumerate(
                    (x, y)
                    for y in (100, 300, 500, 700)
                    for x in (100, 300, 500, 700)
                )
            ]
        else:
            uav_initial_data = sorted(
                scenario_entry["uavs"], key=lambda item: int(item["uav_id"])
            )
        for initial in uav_initial_data:
            i = int(initial["uav_id"])
            x_u, y_u, z_u = map(float, initial["position"])
            uav = UAV(id=i, x=x_u, y=y_u, z=z_u)
            uav.energy = float(initial["energy_j"])
            self.UAVs.append(uav)
            uav.last_energy = uav.energy
            uav.update_battery(uav.energy, self.E_max)
            self.uav_dict[uav.id] = uav
        self.uav_paths = {i: [] for i in range(self.num_UAV)}
        for uav in self.UAVs:
            self.uav_paths[uav.id].append([uav.x_u, uav.y_u, uav.z_u])
        # ============= 初始化 Ground Target =============
        # ============= 初始化 Ground Target（不重疊） =============
        self.gts = []
        radius = 80
        W = getattr(self, "env_width", 1000)
        H = getattr(self, "env_height", 1000)

        d_min = int(2 * radius)   # 最小中心距離；2*radius 最嚴格，1.6~1.8*radius較易收斂
        max_tries_per_point = 200   # 單點嘗試上限
        relax_ratio = 0.9           # 若太擠，逐步放寬 d_min（避免死循環）

        if scenario_entry is None:
            pts, tries = [], 0
            gs_x, gs_y = 0.0, 0.0
            while len(pts) < self.num_GT:
                if tries > max_tries_per_point:
                    d_min = max(int(d_min * relax_ratio), radius)
                    tries = 0
                x = random.uniform(radius, W - radius)
                y = random.uniform(radius, H - radius)
                if (x - gs_x) ** 2 + (y - gs_y) ** 2 < 200**2:
                    tries += 1
                    continue
                if any(
                    ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < d_min
                    for px, py in pts
                ):
                    tries += 1
                    continue
                pts.append((x, y))
                tries = 0
            gt_initial_data = [
                {
                    "gt_id": index,
                    "position": [x, y, 0.0],
                    "radius_m": radius,
                }
                for index, (x, y) in enumerate(pts)
            ]
        else:
            gt_initial_data = sorted(
                scenario_entry["ground_targets"],
                key=lambda item: int(item["gt_id"]),
            )

        self.gts = []
        for initial in gt_initial_data:
            i = int(initial["gt_id"])
            x, y, z = map(float, initial["position"])
            gt = GroundTarget(
                id=i,
                x=x,
                y=y,
                z=z,
                radius=float(initial["radius_m"]),
            )
            gt.is_found = False
            gt.found_by = None
            self.gts.append(gt)
        # print(f"Generated {len(self.gts)}")
        # =============== SR team 初始位置 =====================
        self.SR_teams = []
        if scenario_entry is None:
            start_points = self.fixed_boundary_points(
                self.env_width, self.env_height
            )
            sr_initial_data = [
                {
                    "sr_id": index,
                    "position": [*start_points[index % 4], 0.0],
                    "movement_primitive": {"speed_mps": 1.0},
                }
                for index in range(self.num_GT)
            ]
        else:
            sr_initial_data = sorted(
                scenario_entry["sr_teams"],
                key=lambda item: int(item["sr_id"]),
            )
        self.sr_speed_mps = float(
            sr_initial_data[0]["movement_primitive"]["speed_mps"]
        )
        for initial in sr_initial_data:
            i = int(initial["sr_id"])
            x, y, z = map(float, initial["position"])
            sr = SRTeam(id=i)
            sr.x, sr.y, sr.z = x, y, z
            sr.assigned_gt_id = None
            sr.path = []
            sr.active = False
            sr.arrived = False
            sr.current_step = 0
            self.SR_teams.append(sr)
        self.sr_trajectory = {i: [] for i in range(self.num_SR_team)}
        for sr in self.SR_teams:
            self.sr_trajectory[sr.id].append([sr.x, sr.y, sr.z])  # 初始位置
        for uav in self.UAVs:
            uav.task_type = None
            uav.assigned_target_id = None
        
        self.task_list = []
        for uav in self.UAVs:
            task = Task(
                task_id=len(self.task_list),
                task_type="Search",
                target_obj=uav,
                target_obj_id = uav.id
            )
            self.task_list.append(task)
        self.assign_tasks()

    def generate_scenario_entry(self, split, manifest_seed, episode_index):
        """Generate exogenous episode data without consuming global RNG state."""

        from scenario_manifest import generate_scenario_entry

        return generate_scenario_entry(split, manifest_seed, episode_index)

    def apply_scenario_entry(self, scenario_entry):
        """Reset the corrected environment from one manifest episode entry."""

        if self.num_UAV != 16:
            raise RuntimeError(
                "scenario manifest requires the corrected 16-UAV environment"
            )
        self.reset_environment(scenario_entry=scenario_entry)
        self.validate_applied_scenario(scenario_entry)

    def validate_applied_scenario(self, scenario_entry):
        """Fail fast if applied exogenous state differs from the manifest."""

        from scenario_manifest import validate_scenario_entry

        validate_scenario_entry(scenario_entry)
        if self.active_scenario_id != str(scenario_entry["scenario_id"]):
            raise RuntimeError("applied scenario identity mismatch")
        if self.num_GT != int(scenario_entry["num_GT"]):
            raise RuntimeError("applied scenario num_GT mismatch")
        actual_uavs = [
            [uav.x_u, uav.y_u, uav.z_u, uav.energy]
            for uav in sorted(self.UAVs, key=lambda item: item.id)
        ]
        expected_uavs = [
            [*map(float, item["position"]), float(item["energy_j"])]
            for item in sorted(
                scenario_entry["uavs"], key=lambda item: int(item["uav_id"])
            )
        ]
        actual_gts = [
            [gt.x, gt.y, gt.z, gt.radius]
            for gt in sorted(self.gts, key=lambda item: item.id)
        ]
        expected_gts = [
            [*map(float, item["position"]), float(item["radius_m"])]
            for item in sorted(
                scenario_entry["ground_targets"],
                key=lambda item: int(item["gt_id"]),
            )
        ]
        actual_sr = [
            [sr.x, sr.y, sr.z]
            for sr in sorted(self.SR_teams, key=lambda item: item.id)
        ]
        expected_sr = [
            list(map(float, item["position"]))
            for item in sorted(
                scenario_entry["sr_teams"],
                key=lambda item: int(item["sr_id"]),
            )
        ]
        if not np.allclose(actual_uavs, expected_uavs):
            raise RuntimeError("applied UAV initial state mismatch")
        if not np.allclose(actual_gts, expected_gts):
            raise RuntimeError("applied GT/RoI initial state mismatch")
        if not np.allclose(actual_sr, expected_sr):
            raise RuntimeError("applied SR initial state mismatch")
        if not np.isclose(
            self.load_factor,
            float(scenario_entry["traffic_primitives"]["load_factor"]),
        ):
            raise RuntimeError("applied traffic primitive mismatch")
        return True
    
    

        
        
        
        
        
        
    
        
