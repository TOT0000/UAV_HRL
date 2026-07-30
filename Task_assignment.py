from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from Fov_model_phase import FovModel
from Channel_model import ChannelModel

class UAVAssigner:
    def __init__(self, env):
        self.env = env
    def assign_tasks(self, uav_id_list, task_list,K=2, alpha=0.6, beta=0.4, mode="KM"):
        """
        根據 mode 來分配任務:
        - mode="KM"     → K 次 KM 演算法 (K=1 時就是單次 KM)
        - mode="Random" → 隨機分配
        """

        if mode == "KM":
            self.assign_uav_tasks_k_times(uav_id_list, task_list, K, alpha=alpha, beta=beta)
        elif mode == "Random":
            self.random_assign_tasks(uav_id_list, task_list)
        else:
            raise ValueError(f"Unknown assignment mode: {mode}")
        return self.assignments
    def assign_uav_tasks_k_times(self, uav_list, task_list, K=2, alpha=0.6, beta=0.4):
        """
        執行 K 次 KM 任務分配，每次指派一組 UAV→任務 配對。
        支援多任務被指派給同一 UAV（例如 FOV + COM）
        """
        self._snapshot_tasks = list(task_list)

        num_uav = len(uav_list)
        num_task = len(task_list)

        incompatible_task_pairs = [
        ("Search", "FOV"),
        ("FOV", "Search"),
        ("Search", "Search")
    ]
         # 初始化 assignment 結構：每台 UAV 對應一組任務清單
        self.assignments = {uid: [] for uid in uav_list}
        available_task_ids = set(range(num_task))  # 以 task list index 編號

        # Step 1: 建立對應矩陣
        # 1) 矩陣預設為 -1e9，作為不可選遮罩
        FOV_matrix = np.full((num_uav, num_task), -1e9, dtype=float)
        Cap_matrix = np.full((num_uav, num_task), -1e9, dtype=float)

        # 將電量加入權重
        explore_weights = np.array([
            self.env.get_unexplored_ratio(uav_id)
            for uav_id in uav_list
        ]).reshape(-1, 1)

        for i, uav_id in enumerate(uav_list):
            uav = self.env.uav_dict[uav_id]
            for j, task in enumerate(task_list):
                # 搜完後：直接遮罩所有 Search 任務
                if getattr(self.env, "search_completed", False) and task.task_type == "Search":
                    continue  # 保持 -1e9
                if task.task_type == "FOV":
                    gt_id = task.target_obj_id 
                    gt = self.env.gts[gt_id]
                    
                    if not gt.is_found:
                        # FOV_matrix[i, j] = 0
                        continue
                    # 計算 FOV
                    x_g, y_g, z_g = gt.x, gt.y, gt.z
                    fov_model = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
                    fov,_ = fov_model.calculate_fov_single(uav.x_u, uav.y_u, uav.z_u, x_g, y_g, z_g)
                    eps = 1e-6 
                    if fov <= 1.0 + eps:
                        FOV_matrix[i, j] = fov
                    else:
                        continue  # 保持 -1e9，不可行就不給選
                elif task.task_type == "COM":
                    sr_id = task.target_obj_id 
                    sr = self.env.SR_teams[sr_id]
                    if not sr.active:
                        # Cap_matrix[i, j] = 0  
                        continue
                    # 計算 Capacity
                    SNR = self.env.get_snr(uav_id, sr_id)     # SNR (dB)
                    # print(SNR)
                    capacity = ChannelModel.C_ug(B_ug=10e6, SNR_ug_t=SNR)
                    Cap_matrix[i, j] = capacity / 1e6            # 換算為 Mbps
                elif task.task_type == "Hovering":
                    # 分數直接給一個固定值（例如鼓勵保持原地懸停）
                    FOV_matrix[i, j] = 0.1
                elif task.task_type == "Search":
                    # 尚未搜完時：允許 Search，被當作可選（給個基礎分數）
                    if not getattr(self.env, "search_completed", False):
                        FOV_matrix[i, j] = 0.05  # 或用 self.env.get_unexplored_ratio(uav_id)


        W = (alpha * FOV_matrix + beta * Cap_matrix)

        for round in range(K):
            if not available_task_ids:
                break

            W_filtered = W.copy()

            for i, uav_id in enumerate(uav_list):
                uav_prev_tasks = [task_type for (_, task_type, _) in self.assignments[uav_id]]

                for j in range(num_task):
                    task = task_list[j]

                    if j not in available_task_ids:
                        W_filtered[i, j] = -1e9
                        continue

                    # 只從第2輪之後開始檢查不相容組合
                    if round > 0:
                        for prev_task_type in uav_prev_tasks:
                            if prev_task_type == task.task_type:
                                W_filtered[i, j] = -1e9
                        for prev_task_type in uav_prev_tasks:
                            if (prev_task_type, task.task_type) in incompatible_task_pairs:
                                W_filtered[i, j] = -1e9
                                # print(f"🚫 UAV {uav_id} 任務組合衝突: {prev_task_type} + {task.task_type} → 已遮罩")
                                break

            # 執行匈牙利指派
            cost_matrix = -W_filtered
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if j not in available_task_ids:
                    continue

                task = task_list[j]
                score = FOV_matrix[i, j] if task.task_type == "FOV" else Cap_matrix[i, j]
                self.assignments[uav_list[i]].append((j, task.task_type, score))
                available_task_ids.remove(j)

        return self.assignments
    def random_assign_tasks(self, uav_list, task_list):
        """
        隨機分配 UAV → 任務，並建立 self.assignments
        """
        task_ids = list(range(len(task_list)))
        random.shuffle(task_ids)

        # ✅ 建立與 KM 相同格式的 self.assignments
        self.assignments = {uid: [] for uid in uav_list}
        for uav_id, task_id in zip(uav_list, task_ids):
            task = task_list[task_id]
            # 隨機分配就不需要分數，給個 0.0
            self.assignments[uav_id].append((task_id, task.task_type, 0.0))
        return self.assignments
    def build_uav_tasks_from_assignment(self):
        # ✅ 用 assign 時的快照來還原任務，避免 env.task_list 被修改導致錯位
        snapshot = getattr(self, "_snapshot_tasks", None)
        if snapshot is None:
            snapshot = self.env.task_list  # 後備（不建議）
        self.env.multi_tasks = {}

        for uav_id, task_list in self.assignments.items():
            uav = self.env.uav_dict[uav_id]
            task_entries = []
            for task_id, task_type, _ in task_list:
                # 根據任務類型取得目標位置
                task = snapshot[task_id]
                if task_type == "FOV":
                    gt = self.env.gts[task.target_obj_id]
                    x_tgt, y_tgt, z_tgt = gt.get_position()
                elif task_type == "COM":
                    sr = self.env.SR_teams[task.target_obj_id]
                    x_tgt, y_tgt, z_tgt = sr.get_position()
                else:  # Hovering or fallback
                    x_tgt, y_tgt, z_tgt = uav.get_position()

                # 新增任務進 UAV 任務列表
                task_entries.append({
                    "task_type": task_type,
                    "target_id": task_id,
                    "target_pos": (x_tgt, y_tgt, z_tgt)
                })

            # 儲存到環境中
            self.env.multi_tasks[uav_id] = task_entries

            # ✅ 選擇第一個任務作為主任務（方便原架構使用）
            if task_entries:
                primary = task_entries[0]
                uav.task_type = primary["task_type"]
                uav.assigned_target_id = primary["target_id"]
                uav.target_position = primary["target_pos"]
        # print("\n📋 [Debug] UAV 任務分配結果：")
        # for uav_id, task_entries in self.env.multi_tasks.items():
        #     print(f"UAV {uav_id}:")
        #     for i, task in enumerate(task_entries):
        #         print(f"  Task {i}: Type = {task['task_type']}, Target ID = {task['target_id']}, Target Pos = {task['target_pos']}")
                

class Task:
    def __init__(self, task_id, task_type, target_obj, target_obj_id ):
        self.task_id = task_id
        self.task_type = task_type  # "Search", "FOV", "COM"
        self.target_obj = target_obj  # 可以是 UAV、GT、SR team 等
        self.target_obj_id = target_obj_id          # ✅ 額外存一份物件 ID
        self.target_type = type(target_obj).__name__  # eg. "GroundTarget", "SRTeam", "UAV"
        self.is_assigned = False