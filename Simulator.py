import numpy as np
import random
import math
from dataclasses import dataclass
from Channel_model import (
    A2G_CARRIER_GHZ,
    A2G_LOS_A,
    A2G_LOS_B,
    A2G_LOS_EXCESS_DB,
    A2G_NLOS_EXCESS_DB,
    NOISE_PSD_DBM_PER_HZ,
    S2U_TX_POWER_DBM,
    U2U_U2G_TX_POWER_DBM,
    a2g_capacity_mbps,
    a2g_path_loss_db,
    shannon_capacity_mbps,
    u2u_path_loss_db,
    normalized_s2u_capacity_utility,
    reference_s2u_max_capacity_mbps,
)
from Fov_model_phase import FovModel
from collections import defaultdict
from Energy_model import EnergyConsumptionModel
from Task_assignment import UAVAssigner, Task
from object import UAV, SRTeam, GroundTarget
from experiment_config import (
    FOV_COM_PAIR_MAX_DISTANCE_M,
    COM_OFFERED_RATE_BPS,
    NUM_UAV,
    REFERENCE_COM_BANDWIDTH_HZ,
    RESERVED_SEARCH_UAV_IDS,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    SR_ROUTE_LIFECYCLE_VERSION,
    SEARCH_COVERAGE_THRESHOLD,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
)


@dataclass(frozen=True)
class FovCoverageTransition:
    """Immutable Search-map transition before its current footprint is committed."""

    uav_id: int
    previous_footprint: tuple[int, int, int, int] | None
    current_footprint: tuple[int, int, int, int]
    map_changed: bool


class Simulator:
    SR_UAV_CARRIER_GHZ = A2G_CARRIER_GHZ
    SR_UAV_TX_POWER_DBM = S2U_TX_POWER_DBM
    SR_UAV_NOISE_DBM_PER_HZ = NOISE_PSD_DBM_PER_HZ
    A2G_LOS_A = A2G_LOS_A
    A2G_LOS_B = A2G_LOS_B
    SR_UAV_LOS_EXCESS_DB = A2G_LOS_EXCESS_DB
    SR_UAV_NLOS_EXCESS_DB = A2G_NLOS_EXCESS_DB

    def __init__(self, num_UAV, p_u=30, rng_streams=None, evaluation=False): #初始化
        self.dt = 0.25
        self.rng_streams = rng_streams
        environment_stream = (
            "evaluation_environment" if evaluation else "environment_dynamics"
        )
        assignment_stream = (
            "evaluation_random_assignment" if evaluation else "random_assignment"
        )
        self.environment_rng = (
            rng_streams.numpy(environment_stream)
            if rng_streams is not None
            else np.random.default_rng(0)
        )
        self.assignment_rng = (
            rng_streams.numpy(assignment_stream)
            if rng_streams is not None
            else np.random.default_rng(0)
        )
        self.sr_update_interval = int(1.0 / self.dt)
        self.sim_step_count = 0
        if int(num_UAV) != NUM_UAV:
            raise ValueError(
                f"canonical Simulator requires exactly {NUM_UAV} UAVs, got {num_UAV}"
            )
        self.B_tot = TOTAL_COMMUNICATION_BANDWIDTH_HZ
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
        self.active_s2u_capacities = {}
        self.active_link_diagnostics = []
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

        # ===== Path-loss diagnostics cache =====
        # Refreshed with channel geometry. Canonical routing reward consumes
        # allocated capacities; the legacy per-hop reward entry point fails fast.
        self.PL_uu_cache = np.zeros((self.num_UAV, self.num_UAV), dtype=float)
        self.PL_ug_cache = np.zeros(self.num_UAV, dtype=float)
        self.mobility_params = dict(self.energy_model.mobility_params)
        self.assignment_strategy = "k_km"
        self.assignment_rounds = 2
        self.fov_com_pair_max_distance_m = FOV_COM_PAIR_MAX_DISTANCE_M
        self.search_coverage_threshold = SEARCH_COVERAGE_THRESHOLD
        self.reserved_search_uav_ids = RESERVED_SEARCH_UAV_IDS
        self.com_offered_rate_bps = COM_OFFERED_RATE_BPS
        self.search_release_time = None
        self.search_release_coverage = None
        self.assignment_invocations = 0
        self.search_to_hover_conversions = 0

    def configure_method(self, method_spec):
        """Install comparison strategies before any episode reset."""

        self.assignment_strategy = str(method_spec.assignment)
        self.assignment_rounds = int(method_spec.assignment_rounds)
        self.fov_com_pair_max_distance_m = FOV_COM_PAIR_MAX_DISTANCE_M
        self.search_coverage_threshold = SEARCH_COVERAGE_THRESHOLD

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
        coverage = float(np.asarray(self.visited_bitmap, dtype=bool).mean())
        search_active = not self._search_phase_over and coverage < self.search_coverage_threshold
        reserved = set(self.reserved_search_uav_ids) if search_active else set()
        uav_id_list = [
            uav_id for uav_id in self.get_available_uav_ids() if uav_id not in reserved
        ]
        assigner = UAVAssigner(self)
        assigner.assign_tasks(
            uav_id_list,
            self.task_list,
            K=self.assignment_rounds,
            strategy=self.assignment_strategy,
            max_distance_m=self.fov_com_pair_max_distance_m,
            coverage_threshold=self.search_coverage_threshold,
        )
        assigner.build_uav_tasks_from_assignment()# 分配結果改成任務列表
    # ====================更新探索區域=====================
        self.assignment_invocations += 1
        self.last_assignment = assigner
        self.last_assignment_metadata = self.assignment_metadata()

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

    def fov_footprint_indices(self, uav_id):
        """Return one UAV's current camera footprint without mutating state."""

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
            return None
        return (int(bx_min), int(bx_max), int(by_min), int(by_max))

    def mark_search_coverage(self, uav_id):
        """Apply coverage and return the uncommitted footprint transition."""

        uav = self.uav_dict[uav_id]
        current = self.fov_footprint_indices(uav_id)
        if current is None:
            return None
        previous = getattr(uav, "last_box_idx", None)
        previous = tuple(int(value) for value in previous) if previous is not None else None
        bx_min, bx_max, by_min, by_max = current
        patch = self.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1]
        map_changed = bool((~patch).any())
        self.visited_bitmap[bx_min : bx_max + 1, by_min : by_max + 1] = True
        return FovCoverageTransition(
            uav_id=int(uav_id),
            previous_footprint=previous,
            current_footprint=current,
            map_changed=map_changed,
        )
        
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

    def _convert_search_to_hovering_phase(self):
        if self._search_phase_over and self.search_completed:
            return
        self._search_phase_over = True
        self.search_completed = True
        self.search_to_hover_conversions += 1
        self.search_release_time = float(getattr(self, "current_time", 0.0))
        self.search_release_coverage = float(self.visited_bitmap.mean())
        self.task_list = [task for task in self.task_list if task.task_type != "Search"]
        self.assign_tasks()
        if any(
            task["task_type"] == "Search"
            for entries in self.multi_tasks.values()
            for task in entries
        ):
            raise AssertionError("Search assignment survived the 99% release event")

    def assignment_metadata(self):
        return {
            "strategy": self.assignment_strategy,
            "invocation": int(self.assignment_invocations),
            "reserved_search_uav_ids": list(self.reserved_search_uav_ids),
            "search_release_time_seconds": self.search_release_time,
            "search_release_coverage": self.search_release_coverage,
            "assignments": {
                str(uav_id): sorted(
                    [dict(task) for task in self.multi_tasks.get(uav_id, [])],
                    key=lambda task: (
                        task["task_type"],
                        -1 if task.get("target_obj_id") is None else task["target_obj_id"],
                    ),
                )
                for uav_id in range(self.num_UAV)
            },
        }

    def convert_search_to_hovering(self):
        """Apply the guarded Search-to-Hover phase conversion exactly once."""

        return self._convert_search_to_hovering_phase()
    # =============搜救隊出發===============================
    def SR_team_gogo(self, gt):
        """
        找出距離目標最近的 SR team，並更新其移動目標
        """
        gt_pos = np.array([gt.x, gt.y])
        sr_positions = np.array([
            (sr.x, sr.y)
            for sr in self.SR_teams
            if sr.assigned_gt_id is None
        ])
        available_indices = [
            i
            for i, sr in enumerate(self.SR_teams)
            if sr.assigned_gt_id is None
        ]

        # 找距離 GT 最近的 SR 成員
        if len(sr_positions) > 0:
            dists = np.linalg.norm(sr_positions - gt_pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_sr_idx = available_indices[nearest_idx]

            sr = self.SR_teams[nearest_sr_idx]
            sr.assign_mission(
                gt.id,
                (gt.x, gt.y),
                speed=float(getattr(self, "sr_speed_mps", 1.0)),
            )
            self.SR_paths = list(sr.path)
            gt.assigned = True
            # print(f"[SR Team] 指派 SR {sr.id} 前往 GT {gt.id}")
            # print(self.SR_paths)
            return sr
    def advance_sr_teams(self):
        for sr in self.SR_teams:
            sr.step_forward()
            point = [sr.x, sr.y, sr.z]
            if not self.sr_trajectory[sr.id] or self.sr_trajectory[sr.id][-1] != point:
                self.sr_trajectory[sr.id].append(point)

    def sr_route_state(self):
        return {
            "lifecycle_version": SR_ROUTE_LIFECYCLE_VERSION,
            "teams": [
                team.route_state()
                for team in sorted(self.SR_teams, key=lambda item: item.id)
            ],
            "trajectory": {
                str(team_id): [list(map(float, point)) for point in points]
                for team_id, points in sorted(
                    getattr(self, "sr_trajectory", {}).items()
                )
            },
            "checkpoint_scope": "episode_boundary_terminal_snapshot",
            "mid_episode_checkpoint_supported": False,
        }

    def load_sr_route_state(self, state):
        if (state or {}).get("lifecycle_version") != SR_ROUTE_LIFECYCLE_VERSION:
            raise RuntimeError("checkpoint SR route lifecycle is incompatible")
        if (
            state.get("checkpoint_scope")
            != "episode_boundary_terminal_snapshot"
            or bool(state.get("mid_episode_checkpoint_supported"))
        ):
            raise RuntimeError("checkpoint SR route scope is incompatible")
        teams = list(state.get("teams", []))
        if len(teams) != len(self.SR_teams):
            raise RuntimeError("checkpoint SR route team count is incompatible")
        by_id = {int(team.id): team for team in self.SR_teams}
        for team_state in teams:
            team_id = int(team_state.get("sr_id", -1))
            if team_id not in by_id:
                raise RuntimeError("checkpoint SR route team id is incompatible")
            by_id[team_id].load_route_state(team_state)
        trajectory = state.get("trajectory") or {}
        if set(map(int, trajectory)) != set(by_id):
            raise RuntimeError("checkpoint SR trajectory ids are incompatible")
        restored_trajectory = {}
        for team_id, points in trajectory.items():
            restored_points = []
            for point in points:
                values = np.asarray(point, dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise RuntimeError("checkpoint SR trajectory point is invalid")
                restored_points.append(values.tolist())
            restored_trajectory[int(team_id)] = restored_points
        self.sr_trajectory = restored_trajectory
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
    def _get_sr_uav_link_metrics(self, uav_id, sr_id, bandwidth_hz=None):
        """Return canonical S2U SNR/capacity without an arbitrary range cutoff."""

        bandwidth_hz = float(
            REFERENCE_COM_BANDWIDTH_HZ if bandwidth_hz is None else bandwidth_hz
        )
        uav_position = self.uav_dict[int(uav_id)].get_position()
        sr_position = self.SR_teams[int(sr_id)].get_position()
        path_loss = float(a2g_path_loss_db(uav_position, sr_position))
        capacity_mbps = float(
            a2g_capacity_mbps(
                uav_position,
                sr_position,
                bandwidth_hz,
                S2U_TX_POWER_DBM,
            )
        )
        noise_dbm = NOISE_PSD_DBM_PER_HZ + 10.0 * math.log10(bandwidth_hz)
        snr = 10.0 ** ((S2U_TX_POWER_DBM - path_loss - noise_dbm) / 10.0)
        return float(snr), capacity_mbps

    def get_snr(self, uav_id, sr_id):
        """Return the SR-UAV link SNR as a linear ratio."""

        snr_us, _ = self._get_sr_uav_link_metrics(uav_id, sr_id)
        return snr_us

    def get_sr_uav_capacity_mbps(self, uav_id, sr_id):
        """Return reference-bandwidth S2U capacity for decision features."""

        _, capacity_mbps = self._get_sr_uav_link_metrics(uav_id, sr_id)
        return capacity_mbps

    get_sr_uav_reference_capacity_mbps = get_sr_uav_capacity_mbps

    def get_sr_uav_normalized_utility(self, uav_id, sr_id):
        """Return the shared fixed-reference COM utility for one link."""

        return normalized_s2u_capacity_utility(
            self.uav_dict[int(uav_id)].get_position(),
            self.SR_teams[int(sr_id)].get_position(),
            REFERENCE_COM_BANDWIDTH_HZ,
        )

    @property
    def reference_s2u_max_capacity_mbps(self):
        return reference_s2u_max_capacity_mbps(REFERENCE_COM_BANDWIDTH_HZ)

    # =====================U2U channel model================================
    def update_u2u_channels(self):
        positions = np.asarray([uav.get_position() for uav in self.UAVs], dtype=float)
        count = len(positions)
        path_loss = np.zeros((count, count), dtype=float)
        capacity = np.zeros((count, count), dtype=float)
        for sender in range(count):
            for receiver in range(count):
                if sender == receiver:
                    continue
                path_loss[sender, receiver] = float(
                    u2u_path_loss_db(positions[sender], positions[receiver])
                )
                capacity[sender, receiver] = float(
                    shannon_capacity_mbps(
                        path_loss[sender, receiver],
                        self.B_tot,
                        U2U_U2G_TX_POWER_DBM,
                    )
                )
        self.PL_uu_cache = path_loss
        feasible = np.isfinite(capacity) & (capacity > 0.0)
        np.fill_diagonal(feasible, False)
        self.k_u_u2u = feasible.sum(axis=1)
        self.k_bar_u2u = float(self.k_u_u2u.mean())
        self.B_eff_u2u = np.full(count, self.B_tot, dtype=float)
        self.u2u_nominal_capacity = capacity
        self.Capacity_matrix = capacity.copy()

    # ==========================U2G channel model============================
    # 無人機與地面站
    def update_u2g_channels(self):
        positions = np.asarray([uav.get_position() for uav in self.UAVs], dtype=float)
        gs_position = np.asarray(self.GS_pos, dtype=float)
        self.Expected_PL = np.asarray(
            [a2g_path_loss_db(position, gs_position) for position in positions],
            dtype=float,
        )
        self.PL_ug_cache = self.Expected_PL.copy()
        self.u2g_nominal_capacity = np.asarray(
            [
                shannon_capacity_mbps(
                    loss, self.B_tot, U2U_U2G_TX_POWER_DBM
                )
                for loss in self.Expected_PL
            ],
            dtype=float,
        )
        self.gs_capacity = self.u2g_nominal_capacity.copy()

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        """Equal-FDMA allocation over one shared S2U/U2U/U2G 10 MHz pool."""

        active_links = [
            (int(sender), int(receiver))
            for sender, receiver in sorted(proposed_links.items())
            if int(receiver) != int(sender)
        ]
        s2u_links = {
            int(sr_id): int(uav_id)
            for sr_id, uav_id in dict(s2u_links or {}).items()
        }
        total_links = len(active_links) + len(s2u_links)
        shared_bandwidth = self.B_tot / total_links if total_links else 0.0

        capacities = {}
        bandwidths = {}
        diagnostics = []
        for sender, receiver in active_links:
            link_type = "U2G" if receiver == self.GS_ID else "U2U"
            path_loss = (
                float(self.PL_ug_cache[sender])
                if link_type == "U2G"
                else float(self.PL_uu_cache[sender, receiver])
            )
            bandwidths[(sender, receiver)] = float(shared_bandwidth)
            capacities[(sender, receiver)] = float(
                shannon_capacity_mbps(
                    path_loss, shared_bandwidth, U2U_U2G_TX_POWER_DBM
                )
            )
            diagnostics.append(
                {
                    "link_type": link_type,
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "bandwidth_hz": float(shared_bandwidth),
                    "capacity_mbps": capacities[(sender, receiver)],
                }
            )
        s2u_capacities = {}
        for sr_id, uav_id in sorted(s2u_links.items()):
            _, capacity = self._get_sr_uav_link_metrics(
                uav_id, sr_id, bandwidth_hz=shared_bandwidth
            )
            key = ("S2U", sr_id, uav_id)
            bandwidths[key] = float(shared_bandwidth)
            s2u_capacities[(sr_id, uav_id)] = float(capacity)
            diagnostics.append(
                {
                    "link_type": "S2U",
                    "sender_id": sr_id,
                    "receiver_id": uav_id,
                    "bandwidth_hz": float(shared_bandwidth),
                    "capacity_mbps": float(capacity),
                }
            )

        if sum(bandwidths.values()) > self.B_tot + 1e-6:
            raise AssertionError("active link bandwidth exceeds the shared 10 MHz pool")

        self.active_link_capacities = capacities
        self.active_link_bandwidths = bandwidths
        self.active_s2u_capacities = s2u_capacities
        self.active_link_diagnostics = diagnostics
        return capacities, bandwidths

    #===========回傳不可選擇的節點======================== 
    def get_routing_action_mask(self, from_uav_id):

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
                if np.isfinite(cap) and cap > 0.0:
                    mask[to_id] = 1.0

        # UAV → GS
        if self.gs_capacity is not None:
            cap_gs = float(self.gs_capacity[from_uav_id])
            if np.isfinite(cap_gs) and cap_gs > 0.0:
                mask[num_uav] = 1.0

        return mask


    # =================辨認封包來源的無人機編號=========================
    def update_source_uavs(self):
        self.source_uavs = set()
        for uav in self.UAVs:
            uav_id = uav.id
            task_list = self.multi_tasks.get(uav_id, [])
            for task in task_list:
                # FOV data originates at the UAV. COM data originates at its SR
                # and enters a UAV queue only after a complete S2U upload.
                if task["task_type"] == "FOV":
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
        if not ROI_COUNT_MIN <= int(self.num_GT) <= ROI_COUNT_MAX:
            raise ValueError(
                f"environment num_GT must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
            )
        self.load_factor = float(self.traffic_primitives["load_factor"])
        self.need_reassign = True   
        self.assignment_invocations = 0
        self.search_to_hover_conversions = 0
        self.search_release_time = None
        self.search_release_coverage = None
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
                    "position": [x_u, y_u, self.environment_rng.uniform(80, 120)],
                    "energy_j": self.E_max,
                }
                for index, (x_u, y_u) in enumerate(
                    (x, y)
                    for y in (250, 750)
                    for x in (100, 300, 500, 700, 900)
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
                x = self.environment_rng.uniform(radius, W - radius)
                y = self.environment_rng.uniform(radius, H - radius)
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
            sr.reset_lifecycle()
            self.SR_teams.append(sr)
        self.sr_trajectory = {i: [] for i in range(self.num_SR_team)}
        for sr in self.SR_teams:
            self.sr_trajectory[sr.id].append([sr.x, sr.y, sr.z])  # 初始位置
        for uav in self.UAVs:
            uav.task_type = None
            uav.assigned_target_id = None
        
        # Search is orchestration fallback, not a solver candidate or utility.
        self.task_list = []
        self.assign_tasks()

    def generate_scenario_entry(self, split, manifest_seed, episode_index):
        """Generate exogenous episode data without consuming global RNG state."""

        from scenario_manifest import generate_scenario_entry

        return generate_scenario_entry(split, manifest_seed, episode_index)

    def apply_scenario_entry(self, scenario_entry):
        """Reset the corrected environment from one manifest episode entry."""

        if self.num_UAV != NUM_UAV:
            raise RuntimeError(
                f"scenario manifest requires the corrected {NUM_UAV}-UAV environment"
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
    
    

        
        
        
        
        
        
    
        
