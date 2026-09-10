import copy
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
    ChannelLifecycle,
    FADING_BLOCKS_PER_ROUTING_SLOT,
    NOISE_PSD_DBM_PER_HZ,
    ROUTING_SLOT_SECONDS,
    S2U_TX_POWER_DBM,
    U2U_U2G_TX_POWER_DBM,
    a2g_conditional_path_loss_db,
    a2g_expected_capacity_mbps,
    average_snr_linear,
    block_capacity_profile_mbps,
    effective_capacity_mbps,
    expected_fading_capacity_mbps,
    reference_u2g_max_capacity_mbps,
    reference_u2u_max_capacity_mbps,
    slot_service_bits,
    u2u_path_loss_db,
    normalized_s2u_capacity_utility,
    reference_s2u_max_capacity_mbps,
)
from visual_sensing import search_footprint
from collections import defaultdict
from Energy_model import EnergyConsumptionModel
from Task_assignment import UAVAssigner, Task
from object import UAV, SRTeam, GroundTarget
from experiment_config import (
    CANONICAL_UAV_INITIAL_XY_M,
    COMMUNICATION_RANGE_M,
    FOV_COM_PAIR_MAX_DISTANCE_M,
    FOV_ASSIGNMENT_UTILITY_VERSION,
    FOV_QUALITY_TRANSFORM,
    COM_OFFERED_RATE_BPS,
    GROUND_STATION_POSITION_M,
    GS_GATEWAY_CONTRACT_VERSION,
    GS_GATEWAY_HARD_RADIUS_M,
    GS_GATEWAY_PROJECTION_MODE,
    GS_GATEWAY_SOFT_RADIUS_M,
    GS_GATEWAY_SOFT_RADIUS_OPERATIONAL,
    NUM_UAV,
    PERMANENT_GS_GATEWAY_UAV_ID,
    REFERENCE_COM_BANDWIDTH_HZ,
    RESERVED_SEARCH_UAV_IDS,
    ROI_COUNT_MAX,
    ROI_COUNT_MIN,
    RELAY_POTENTIAL_WEIGHT,
    RELAY_TASK_CONTRACT_VERSION,
    SR_ROUTE_LIFECYCLE_VERSION,
    SEARCH_COVERAGE_THRESHOLD,
    TOTAL_COMMUNICATION_BANDWIDTH_HZ,
    TASK_COMPATIBILITY_POLICY,
)
from relay_contract import empty_plan, relay_snapshot, refresh_relay_targets, COUNT_RULE


@dataclass(frozen=True)
class FovCoverageTransition:
    """Immutable all-participant sample from one pre-commit map snapshot."""

    uav_id: int
    previous_footprint: tuple[int, int, int, int] | None
    current_footprint: tuple[int, int, int, int] | None
    map_changed: bool
    raw_overlap: float | None = None
    raw_unvisited: float | None = None
    raw_frontier: float | None = None
    coverage_contributor: bool = True


class Simulator:
    SR_UAV_CARRIER_GHZ = A2G_CARRIER_GHZ
    SR_UAV_TX_POWER_DBM = S2U_TX_POWER_DBM
    SR_UAV_NOISE_DBM_PER_HZ = NOISE_PSD_DBM_PER_HZ
    A2G_LOS_A = A2G_LOS_A
    A2G_LOS_B = A2G_LOS_B
    SR_UAV_LOS_EXCESS_DB = A2G_LOS_EXCESS_DB
    SR_UAV_NLOS_EXCESS_DB = A2G_NLOS_EXCESS_DB

    def __init__(self, num_UAV, p_u=30, rng_streams=None, evaluation=False): #初始化
        self.dt = ROUTING_SLOT_SECONDS
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
        channel_large_scale_stream = (
            "evaluation_channel_large_scale_state"
            if evaluation
            else "channel_large_scale_state"
        )
        channel_small_scale_stream = (
            "evaluation_channel_small_scale_fading"
            if evaluation
            else "channel_small_scale_fading"
        )
        self.channel_large_scale_rng = (
            rng_streams.numpy(channel_large_scale_stream)
            if rng_streams is not None
            else np.random.default_rng(60)
        )
        self.channel_small_scale_rng = (
            rng_streams.numpy(channel_small_scale_stream)
            if rng_streams is not None
            else np.random.default_rng(61)
        )
        self.channel_namespace = "evaluation" if evaluation else "training"
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
        self.u2u_range_mask = np.ones(
            (self.num_UAV, self.num_UAV), dtype=bool
        )
        np.fill_diagonal(self.u2u_range_mask, False)
        self.u2g_range_mask = np.ones(self.num_UAV, dtype=bool)
        self.active_link_capacities = {}
        self.active_link_bandwidths = {}
        self.active_s2u_capacities = {}
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_link_diagnostics = []
        self.GS_pos = GROUND_STATION_POSITION_M
        self.GS_ID = self.num_UAV
        self.channel = ChannelLifecycle(
            self.num_UAV,
            self.GS_ID,
            large_scale_rng=self.channel_large_scale_rng,
            small_scale_rng=self.channel_small_scale_rng,
            namespace=self.channel_namespace,
        )
        self.defer_initial_channel_boundary = False
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
        self.permanent_gs_gateway_uav_id = PERMANENT_GS_GATEWAY_UAV_ID
        self.com_offered_rate_bps = COM_OFFERED_RATE_BPS
        self.search_release_time = None
        self.search_release_coverage = None
        self.search_release_reassignment_pending = False
        self.assignment_invocations = 0
        self.search_to_hover_conversions = 0
        self.assignment_backlog_snapshot = {
            uav_id: 0.0 for uav_id in range(self.num_UAV)
        }
        self.relay_position_history = []
        self.relay_shaping_history = []
        self.relay_plan = empty_plan()
        self.assignment_history = []
        self.relay_role_change_count = 0
        self._previous_relay_uav_ids = set()

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
    def set_assignment_backlog_snapshot(self, backlog_bits):
        snapshot = {}
        for uav_id in range(self.num_UAV):
            value = float(
                backlog_bits.get(uav_id, backlog_bits.get(str(uav_id), 0.0))
            )
            snapshot[uav_id] = value if np.isfinite(value) and value > 0.0 else 0.0
        self.assignment_backlog_snapshot = snapshot

    def assign_tasks(self):
        if self.channel.movement_interval_index is None:
            raise RuntimeError(
                "task assignment requires an initialized movement-interval channel state"
            )
        previous_roles = {
            uid: sorted(task["task_type"] for task in self.multi_tasks.get(uid, []))
            for uid in range(self.num_UAV)
        }
        service_tasks = [task for task in self.task_list if task.task_type != "Relay"]
        coverage = float(np.asarray(self.visited_bitmap, dtype=bool).mean())
        search_active = not self._search_phase_over and coverage < self.search_coverage_threshold
        reserved = (
            set(self.reserved_search_uav_ids) if search_active else set()
        )
        reserved.add(self.permanent_gs_gateway_uav_id)
        uav_id_list = [
            uav_id for uav_id in self.get_available_uav_ids() if uav_id not in reserved
        ]
        assigner = UAVAssigner(self)
        assigner.assign_tasks(
            uav_id_list,
            service_tasks,
            K=self.assignment_rounds,
            strategy=self.assignment_strategy,
            max_distance_m=self.fov_com_pair_max_distance_m,
            coverage_threshold=self.search_coverage_threshold,
        )
        self.relay_plan = assigner.relay_plan
        self.task_list = list(assigner._snapshot_tasks)
        assigner.build_uav_tasks_from_assignment()
    # ====================更新探索區域=====================
        self.assignment_invocations += 1
        self.last_assignment = assigner
        selected_relays = set(assigner.selected_relay_uav_ids)
        role_changes = len(
            self._previous_relay_uav_ids.symmetric_difference(selected_relays)
        )
        self.relay_role_change_count += role_changes
        self._previous_relay_uav_ids = selected_relays
        self.assignment_history.append(
            {
                "invocation": int(self.assignment_invocations),
                "discovered_roi_count": int(self.count_found_targets()),
                "requested_relay_count": int(assigner.requested_relay_count),
                "assigned_relay_count": len(selected_relays),
                "selected_relay_uav_ids": sorted(selected_relays),
                "relay_role_changes": int(role_changes),
                "role_changes": [
                    {"uav_id": uid, "before": previous_roles[uid], "after": current}
                    for uid in range(self.num_UAV)
                    if (current := sorted(task["task_type"] for task in self.multi_tasks[uid]))
                    != previous_roles[uid]
                ],
                "cumulative_relay_role_change_count": int(
                    self.relay_role_change_count
                ),
                "relay_handling_mode": assigner.relay_handling_mode,
                "planning": relay_snapshot(self),
            }
        )
        self.last_assignment_metadata = self.assignment_metadata()

    def update_visited_grid(self, uav_id, *, footprint=None, coverage_contributor=True):
        """
        根據 UAV 的 FOV 更新 visited_grid，並判斷是否有 Ground Target 被發現。
        """
        uav = self.uav_dict[uav_id]
        

         
        # === 檢查每個 Ground Target 是否在此 UAV 的 FOV 中 ===
        for gt in self.gts:
            # print(f"  └─ [GT {gt.id}] 位置=({gt.x}, {gt.y}), is_found={gt.is_found}")

            if (not gt.is_found) and self.is_visible(uav_id, gt, footprint=footprint, coverage_contributor=coverage_contributor):
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

    def fov_footprint_indices(self, uav_id, *, footprint=None):
        """Rasterize the current Search footprint; VS never uses this API."""

        uav = self.uav_dict[uav_id]
        if not self.is_search_contributor(uav_id):
            return None
        footprint = footprint if footprint is not None else self.search_footprint(uav_id)
        if footprint is None:
            return None
        fov_w, fov_h = footprint.width, footprint.height
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

    def mark_search_coverage(
        self,
        uav_id,
        *,
        visited_snapshot=None,
        commit=True,
        coverage_contributor=True,
        footprint=None,
    ):
        """Freeze one FOV sample; only Search contributors may mutate coverage."""

        uav = self.uav_dict[uav_id]
        coverage_contributor = bool(coverage_contributor and self.is_search_contributor(uav_id))
        current = self.fov_footprint_indices(uav_id, footprint=footprint) if coverage_contributor else None
        previous = getattr(uav, "last_box_idx", None)
        previous = tuple(int(value) for value in previous) if previous is not None else None
        snapshot = (
            self.visited_bitmap
            if visited_snapshot is None
            else np.asarray(visited_snapshot, dtype=bool)
        )
        if snapshot.shape != self.visited_bitmap.shape:
            raise ValueError("Search pre-commit bitmap shape is incompatible")
        if current is None:
            return FovCoverageTransition(
                uav_id=int(uav_id),
                previous_footprint=previous,
                current_footprint=None,
                map_changed=False,
                raw_overlap=0.0,
                raw_unvisited=0.0,
                raw_frontier=0.0,
                coverage_contributor=bool(coverage_contributor),
            )
        bx_min, bx_max, by_min, by_max = current
        patch = snapshot[bx_min : bx_max + 1, by_min : by_max + 1]
        map_changed = bool((~patch).any())
        raw_unvisited = float((~patch).mean()) if patch.size else 0.0
        if patch.size:
            border = np.concatenate(
                [patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]]
            )
            raw_frontier = float((~border).mean())
        else:
            raw_frontier = 0.0
        if previous is None:
            raw_overlap = 0.0
        else:
            lbx_min, lbx_max, lby_min, lby_max = previous
            ix_min, iy_min = max(bx_min, lbx_min), max(by_min, lby_min)
            ix_max, iy_max = min(bx_max, lbx_max), min(by_max, lby_max)
            intersection = (
                (ix_max - ix_min + 1) * (iy_max - iy_min + 1)
                if ix_max >= ix_min and iy_max >= iy_min
                else 0
            )
            raw_overlap = intersection / float(max(patch.size, 1))
        if commit and coverage_contributor:
            self.visited_bitmap[
                bx_min : bx_max + 1, by_min : by_max + 1
            ] = True
        return FovCoverageTransition(
            uav_id=int(uav_id),
            previous_footprint=previous,
            current_footprint=current,
            map_changed=map_changed,
            raw_overlap=float(raw_overlap),
            raw_unvisited=float(raw_unvisited),
            raw_frontier=float(raw_frontier),
            coverage_contributor=bool(coverage_contributor),
        )
        
        # if (not self.search_completed) and self.is_search_done(cov_th=0.8, min_found=4):
        #     self.search_completed = True
        #     print(f" Search done in env: cov={self.visited_bitmap.mean():.3f}, found={self.count_found_targets()}")
        #     self.task_list = [t for t in self.task_list if t.task_type != "Search"]
        #     for uid in getattr(self, "multi_tasks", {}):
        #         self.multi_tasks[uid] = [t for t in self.multi_tasks[uid] if t["task_type"] != "Search"]

    # ===============判斷TG是否有被發現=====================
    def is_search_contributor(self, uav_id):
        return (
            not self._search_phase_over
            and int(uav_id) != int(self.permanent_gs_gateway_uav_id)
            and any(
                task.get("task_type") == "Search"
                for task in self.multi_tasks.get(uav_id, ())
            )
        )

    def search_footprint(self, uav_id):
        return search_footprint(self.uav_dict[uav_id].get_position())

    def is_visible(self, uav_id, target, *, footprint=None, coverage_contributor=True):
        if not coverage_contributor or not self.is_search_contributor(uav_id):
            return False
        footprint = footprint if footprint is not None else self.search_footprint(uav_id)
        return bool(footprint and not target.is_found and footprint.contains(target.x, target.y))

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

    def _convert_search_to_hovering_phase(self, *, defer_assignment=False):
        if self._search_phase_over and self.search_completed:
            return
        self._search_phase_over = True
        self.search_completed = True
        self.search_to_hover_conversions += 1
        self.search_release_time = float(getattr(self, "current_time", 0.0))
        self.search_release_coverage = float(self.visited_bitmap.mean())
        self.task_list = [task for task in self.task_list if task.task_type != "Search"]
        # Coverage release changes only Search fallback roles. Complete service
        # and Relay reassignment remains exclusively a new-RoI boundary event.
        for uid, tasks in self.multi_tasks.items():
            for task in tasks:
                if task["task_type"] == "Search":
                    task["task_type"] = "Hovering"
                    task["reserved_search"] = False
                    self.uav_dict[uid].task_type = "Hovering"
        self.search_release_reassignment_pending = False
        if not self.need_reassign:
            self._validate_search_release_assignment()

    def _validate_search_release_assignment(self):
        """Validate the post-99% assignment only after its real boundary commit."""

        if not self._search_phase_over or not self.search_completed:
            raise AssertionError("Search release assignment validated before phase completion")
        if self.need_reassign or self.search_release_reassignment_pending:
            raise AssertionError("Search release reassignment is still pending")
        if any(
            task["task_type"] == "Search"
            for entries in self.multi_tasks.values()
            for task in entries
        ):
            raise AssertionError("Search assignment survived the 99% release event")
        gateway_tasks = self.multi_tasks.get(self.permanent_gs_gateway_uav_id, ())
        if [task["task_type"] for task in gateway_tasks] != ["Hovering"]:
            raise AssertionError(
                "permanent GS gateway must enter Hovering after Search release"
            )
        assigned_service_targets = set()
        for uav_id, entries in self.multi_tasks.items():
            task_types = [task["task_type"] for task in entries]
            if "Relay" in task_types and task_types != ["Relay"]:
                raise AssertionError("Relay must remain exclusive after Search release")
            for task in entries:
                task_type = task["task_type"]
                if task_type not in {"Relay", "FOV", "COM", "Hovering"}:
                    raise AssertionError(
                        f"invalid task after Search release: UAV {uav_id} {task_type}"
                    )
                if task_type in {"FOV", "COM"}:
                    key = (task_type, int(task["target_obj_id"]))
                    if key in assigned_service_targets:
                        raise AssertionError(
                            "duplicate FOV/COM target after Search release"
                        )
                    assigned_service_targets.add(key)

    def assignment_metadata(self):
        assigner = getattr(self, "last_assignment", None)
        selected = sorted(getattr(assigner, "selected_relay_uav_ids", []))
        planning = relay_snapshot(self)
        requested = planning["required_before_budget"]
        return {
            "strategy": self.assignment_strategy,
            "invocation": int(self.assignment_invocations),
            "channel_movement_interval_index": (
                None
                if self.channel.movement_interval_index is None
                else int(self.channel.movement_interval_index)
            ),
            "reserved_search_uav_ids": list(self.reserved_search_uav_ids),
            "permanent_gs_gateway_uav_id": self.permanent_gs_gateway_uav_id,
            "gs_gateway_contract_version": GS_GATEWAY_CONTRACT_VERSION,
            "gs_gateway_projection_mode": GS_GATEWAY_PROJECTION_MODE,
            "gs_gateway_soft_radius_m": GS_GATEWAY_SOFT_RADIUS_M,
            "gs_gateway_soft_radius_operational": (
                GS_GATEWAY_SOFT_RADIUS_OPERATIONAL
            ),
            "gs_gateway_hard_radius_m": GS_GATEWAY_HARD_RADIUS_M,
            "fov_com_pair_max_distance_m": self.fov_com_pair_max_distance_m,
            "fov_com_pair_distance_gate": "disabled",
            "task_compatibility_policy": TASK_COMPATIBILITY_POLICY,
            "fov_assignment_utility_version": FOV_ASSIGNMENT_UTILITY_VERSION,
            "fov_quality_transform": FOV_QUALITY_TRANSFORM,
            "relay_task_contract_version": RELAY_TASK_CONTRACT_VERSION,
            "relay_count_rule": COUNT_RULE,
            "relay_potential_weight": RELAY_POTENTIAL_WEIGHT,
            "discovered_roi_count": int(self.count_found_targets()),
            "requested_relay_count": int(requested),
            "assigned_relay_count": len(selected),
            "selected_relay_uav_ids": selected,
            "relay_handling_mode": getattr(assigner, "relay_handling_mode", None),
            "relay_role_change_count": int(self.relay_role_change_count),
            "relay_planning": planning,
            "relay_assignment_history": copy.deepcopy(self.assignment_history),
            "relay_position_history": copy.deepcopy(self.relay_position_history),
            "relay_shaping_history": copy.deepcopy(self.relay_shaping_history),
            "relay_reassignment_pending": bool(self.need_reassign),
            "search_release_reassignment_pending": bool(
                self.search_release_reassignment_pending
            ),
            "search_phase_over": bool(self._search_phase_over),
            "search_completed": bool(self.search_completed),
            "search_release_assignment_applied": bool(
                self._search_phase_over
                and not self.search_release_reassignment_pending
            ),
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

    def convert_search_to_hovering(self, *, defer_assignment=False):
        """Apply the guarded Search-to-Hover phase conversion exactly once."""

        return self._convert_search_to_hovering_phase(
            defer_assignment=defer_assignment
        )
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
        indices = self.fov_footprint_indices(uav_id)
        if indices is None:
            return 0.0
        bx_min, bx_max, by_min, by_max = indices
        submap = self.visited_bitmap[bx_min:bx_max+1, by_min:by_max+1]
        return float((~submap).mean()) if submap.size else 0.0

    #=====================通訊如何======================== 
    def _channel_geometry(self):
        return (
            np.asarray([uav.get_position() for uav in self.UAVs], dtype=float),
            np.asarray([sr.get_position() for sr in self.SR_teams], dtype=float),
            np.asarray(self.GS_pos, dtype=float),
        )

    def begin_channel_movement_interval(self, interval_index):
        """Sample every potential A2G large-scale state once per second."""

        uav_positions, sr_positions, gs_position = self._channel_geometry()
        return self.channel.begin_movement_interval(
            interval_index,
            uav_positions=uav_positions,
            sr_positions=sr_positions,
            gs_position=gs_position,
        )

    def initialize_channel_episode_boundary(self):
        """Sample interval zero after episode geometry and slot-0 SR movement."""

        if self.channel.movement_interval_index is not None:
            raise RuntimeError("episode channel boundary is already initialized")
        uav_positions, sr_positions, gs_position = self._channel_geometry()
        self.channel.reset_episode(
            uav_positions=uav_positions,
            sr_positions=sr_positions,
            gs_position=gs_position,
            episode_identity=self.active_scenario_id,
        )
        self.update_u2u_channels()
        self.update_u2g_channels()
        self.assign_tasks()
        self.need_reassign = False

    def prepare_initial_movement_interval(self):
        """Apply slot-0 SR movement, then sample/assign interval zero once."""

        if self.channel.movement_interval_index is not None:
            raise RuntimeError("initial movement interval is already prepared")
        self.advance_sr_teams()
        self.initialize_channel_episode_boundary()

    def advance_channel_boundary(self, next_interval_index):
        """Install the next interval state before assignment and observations."""

        next_interval_index = int(next_interval_index)
        if self.channel.movement_interval_index != next_interval_index - 1:
            raise RuntimeError(
                "channel boundary interval is not the successor of the active interval"
            )
        changed = self.begin_channel_movement_interval(next_interval_index)
        if not changed:
            raise RuntimeError("channel boundary failed to sample exactly once")
        # U2U geometry is already current after the fourth movement substep;
        # U2G expected CSI must be refreshed for the newly sampled A2G state.
        self.update_u2g_channels()
        return True

    def prepare_next_movement_interval(self, next_interval_index):
        """Authoritative non-terminal one-second geometry/channel boundary."""

        self.advance_sr_teams()
        self.advance_channel_boundary(next_interval_index)
        assignment_performed = False
        if self.need_reassign:
            self.assign_tasks()
            self.need_reassign = False
            released_search = self.search_release_reassignment_pending
            self.search_release_reassignment_pending = False
            if released_search:
                self._validate_search_release_assignment()
            assignment_performed = True
        refresh_relay_targets(self)
        return assignment_performed

    def prepare_channel_routing_slot(self, routing_slot_index):
        """Privately generate all potential-link gains in fixed CRN order."""

        return self.channel.prepare_routing_slot(routing_slot_index)

    def channel_state_dict(self):
        return self.channel.state_dict()

    def load_channel_state_dict(self, state):
        self.channel.load_state_dict(state)

    def _get_sr_uav_link_metrics(self, uav_id, sr_id, bandwidth_hz=None):
        """Return conditional mean SNR and deterministic expected S2U capacity."""

        bandwidth_hz = float(
            REFERENCE_COM_BANDWIDTH_HZ if bandwidth_hz is None else bandwidth_hz
        )
        uav_position = self.uav_dict[int(uav_id)].get_position()
        sr_position = self.SR_teams[int(sr_id)].get_position()
        los_state = self.channel.a2g_state("S2U", int(sr_id), int(uav_id))
        path_loss = float(
            a2g_conditional_path_loss_db(
                uav_position, sr_position, los_state
            )
        )
        capacity_mbps = float(
            a2g_expected_capacity_mbps(
                uav_position,
                sr_position,
                bandwidth_hz,
                S2U_TX_POWER_DBM,
                los_state,
            )
        )
        snr = float(average_snr_linear(path_loss, bandwidth_hz, S2U_TX_POWER_DBM))
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
        """Return prospective COM utility without the hard service cutoff."""

        return normalized_s2u_capacity_utility(
            self.uav_dict[int(uav_id)].get_position(),
            self.SR_teams[int(sr_id)].get_position(),
            REFERENCE_COM_BANDWIDTH_HZ,
            los_state=self.channel.a2g_state("S2U", int(sr_id), int(uav_id)),
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
                    expected_fading_capacity_mbps(
                        path_loss[sender, receiver],
                        self.B_tot,
                        U2U_U2G_TX_POWER_DBM,
                        fading="rician",
                    )
                )
        self.PL_uu_cache = path_loss
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        self.u2u_range_mask = distances <= COMMUNICATION_RANGE_M
        np.fill_diagonal(self.u2u_range_mask, False)
        feasible = (
            self.u2u_range_mask
            & np.isfinite(capacity)
            & (capacity > 0.0)
        )
        self.k_u_u2u = feasible.sum(axis=1)
        self.k_bar_u2u = float(self.k_u_u2u.mean())
        self.B_eff_u2u = np.full(count, self.B_tot, dtype=float)
        self.u2u_nominal_capacity = capacity
        self.Capacity_matrix = np.where(
            self.u2u_range_mask, capacity, 0.0
        )

    # ==========================U2G channel model============================
    # 無人機與地面站
    def update_u2g_channels(self):
        positions = np.asarray([uav.get_position() for uav in self.UAVs], dtype=float)
        gs_position = np.asarray(self.GS_pos, dtype=float)
        los_state = np.asarray(self.channel.u2g_los_state, dtype=bool)
        self.Conditional_PL_ug = np.asarray(
            a2g_conditional_path_loss_db(positions, gs_position, los_state),
            dtype=float,
        )
        self.PL_ug_cache = self.Conditional_PL_ug.copy()
        self.u2g_nominal_capacity = np.asarray(
            [
                a2g_expected_capacity_mbps(
                    positions[index],
                    gs_position,
                    self.B_tot,
                    U2U_U2G_TX_POWER_DBM,
                    bool(los_state[index]),
                )
                for index in range(len(positions))
            ],
            dtype=float,
        )
        distances = np.linalg.norm(positions - gs_position[None, :], axis=1)
        self.u2g_range_mask = distances <= COMMUNICATION_RANGE_M
        self.gs_capacity = np.where(
            self.u2g_range_mask, self.u2g_nominal_capacity, 0.0
        )

    @staticmethod
    def distance_3d(first_position, second_position):
        return float(
            np.linalg.norm(
                np.asarray(first_position, dtype=np.float64)
                - np.asarray(second_position, dtype=np.float64)
            )
        )

    def is_u2u_in_range(self, sender_id, receiver_id):
        sender_id, receiver_id = int(sender_id), int(receiver_id)
        if sender_id == receiver_id:
            return False
        if not hasattr(self, "uav_dict"):
            return bool(self.u2u_range_mask[sender_id, receiver_id])
        return self.distance_3d(
            self.uav_dict[sender_id].get_position(),
            self.uav_dict[receiver_id].get_position(),
        ) <= COMMUNICATION_RANGE_M

    def is_u2g_position_in_range(self, position):
        """Canonical U2G eligibility for both physical and virtual UAV positions."""
        return self.distance_3d(position, self.GS_pos) <= COMMUNICATION_RANGE_M

    def is_u2g_in_range(self, sender_id):
        if not hasattr(self, "uav_dict"):
            return bool(self.u2g_range_mask[int(sender_id)])
        return self.is_u2g_position_in_range(self.uav_dict[int(sender_id)].get_position())

    def is_s2u_in_range(self, sr_id, uav_id):
        if not hasattr(self, "uav_dict") or not self.SR_teams:
            return False
        return self.distance_3d(
            self.SR_teams[int(sr_id)].get_position(),
            self.uav_dict[int(uav_id)].get_position(),
        ) <= COMMUNICATION_RANGE_M

    def is_routing_link_in_range(self, sender_id, receiver_id):
        receiver_id = int(receiver_id)
        if receiver_id == self.GS_ID:
            return self.is_u2g_in_range(sender_id)
        return self.is_u2u_in_range(sender_id, receiver_id)

    def allocate_active_link_capacities(self, proposed_links, s2u_links=None):
        """Equal-FDMA then recompute 50 capacities using one cached gain profile."""

        if self.channel._gain_matrix is None:
            if self.channel.u2g_los_state is None:
                raise RuntimeError(
                    "channel state must be initialized before bandwidth allocation"
                )
            self.prepare_channel_routing_slot(
                0 if self.channel.routing_slot_index is None else self.channel.routing_slot_index
            )

        active_links = [
            (int(sender), int(receiver))
            for sender, receiver in sorted(proposed_links.items())
            if int(receiver) != int(sender)
            and self.is_routing_link_in_range(sender, receiver)
        ]
        s2u_links = {
            int(sr_id): int(uav_id)
            for sr_id, uav_id in dict(s2u_links or {}).items()
            if self.is_s2u_in_range(sr_id, uav_id)
        }
        total_links = len(active_links) + len(s2u_links)
        shared_bandwidth = self.B_tot / total_links if total_links else 0.0

        capacities = {}
        capacity_profiles = {}
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
            gains = self.channel.gain_profile(link_type, sender, receiver)
            profile = np.asarray(
                block_capacity_profile_mbps(
                    path_loss,
                    shared_bandwidth,
                    U2U_U2G_TX_POWER_DBM,
                    gains,
                ),
                dtype=float,
            )
            capacity_profiles[(sender, receiver)] = profile
            capacities[(sender, receiver)] = float(effective_capacity_mbps(profile))
            diagnostics.append(
                {
                    "link_type": link_type,
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "bandwidth_hz": float(shared_bandwidth),
                    "capacity_mbps": capacities[(sender, receiver)],
                    "fading_blocks": FADING_BLOCKS_PER_ROUTING_SLOT,
                    "slot_service_bits": float(slot_service_bits(profile)),
                    "gain_profile_reused": True,
                    "range_eligible": True,
                    "distance_3d_m": (
                        self.distance_3d(
                            self.uav_dict[sender].get_position(),
                            self.GS_pos
                            if link_type == "U2G"
                            else self.uav_dict[receiver].get_position(),
                        )
                        if hasattr(self, "uav_dict")
                        else None
                    ),
                }
            )
        s2u_capacities = {}
        s2u_capacity_profiles = {}
        for sr_id, uav_id in sorted(s2u_links.items()):
            uav_position = self.uav_dict[int(uav_id)].get_position()
            sr_position = self.SR_teams[int(sr_id)].get_position()
            los_state = self.channel.a2g_state("S2U", sr_id, uav_id)
            path_loss = float(
                a2g_conditional_path_loss_db(
                    uav_position, sr_position, los_state
                )
            )
            gains = self.channel.gain_profile("S2U", sr_id, uav_id)
            profile = np.asarray(
                block_capacity_profile_mbps(
                    path_loss,
                    shared_bandwidth,
                    S2U_TX_POWER_DBM,
                    gains,
                ),
                dtype=float,
            )
            capacity = float(effective_capacity_mbps(profile))
            key = ("S2U", sr_id, uav_id)
            bandwidths[key] = float(shared_bandwidth)
            s2u_capacities[(sr_id, uav_id)] = float(capacity)
            s2u_capacity_profiles[(sr_id, uav_id)] = profile
            diagnostics.append(
                {
                    "link_type": "S2U",
                    "sender_id": sr_id,
                    "receiver_id": uav_id,
                    "bandwidth_hz": float(shared_bandwidth),
                    "capacity_mbps": float(capacity),
                    "fading_blocks": FADING_BLOCKS_PER_ROUTING_SLOT,
                    "slot_service_bits": float(slot_service_bits(profile)),
                    "gain_profile_reused": True,
                    "range_eligible": True,
                    "distance_3d_m": self.distance_3d(
                        sr_position, uav_position
                    ),
                }
            )

        if sum(bandwidths.values()) > self.B_tot + 1e-6:
            raise AssertionError("active link bandwidth exceeds the shared 10 MHz pool")

        self.active_link_capacities = capacities
        self.active_link_capacity_profiles_mbps = capacity_profiles
        self.active_link_bandwidths = bandwidths
        self.active_s2u_capacities = s2u_capacities
        self.active_s2u_capacity_profiles_mbps = s2u_capacity_profiles
        self.active_link_diagnostics = diagnostics
        return capacities, bandwidths

    def routing_capacity_reference_mbps(self, sender_id, receiver_id):
        del sender_id
        if int(receiver_id) == self.GS_ID:
            return reference_u2g_max_capacity_mbps(self.B_tot)
        return reference_u2u_max_capacity_mbps(self.B_tot)

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
                if (
                    bool(self.u2u_range_mask[from_uav_id, to_id])
                    and np.isfinite(cap)
                    and cap > 0.0
                ):
                    mask[to_id] = 1.0

        # UAV → GS
        if self.gs_capacity is not None:
            cap_gs = float(self.gs_capacity[from_uav_id])
            if (
                bool(self.u2g_range_mask[from_uav_id])
                and np.isfinite(cap_gs)
                and cap_gs > 0.0
            ):
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
                "generation_model": "assigned-fov-rate-accumulator-v2",
            }
        if not ROI_COUNT_MIN <= int(self.num_GT) <= ROI_COUNT_MAX:
            raise ValueError(
                f"environment num_GT must be in [{ROI_COUNT_MIN}, {ROI_COUNT_MAX}]"
            )
        self.load_factor = float(self.traffic_primitives["load_factor"])
        self.need_reassign = True   
        self.assignment_invocations = 0
        self.search_to_hover_conversions = 0
        self.assignment_backlog_snapshot = {
            uav_id: 0.0 for uav_id in range(self.num_UAV)
        }
        self.relay_position_history = []
        self.relay_shaping_history = []
        self.relay_plan = empty_plan()
        self.assignment_history = []
        self.relay_role_change_count = 0
        self._previous_relay_uav_ids = set()
        self.last_assignment = None
        self.search_release_time = None
        self.search_release_coverage = None
        self.search_release_reassignment_pending = False
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
        self.active_link_capacities = {}
        self.active_link_bandwidths = {}
        self.active_s2u_capacities = {}
        self.active_link_capacity_profiles_mbps = {}
        self.active_s2u_capacity_profiles_mbps = {}
        self.active_link_diagnostics = []
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
                    CANONICAL_UAV_INITIAL_XY_M
                )
            ]
        else:
            uav_initial_data = sorted(
                scenario_entry["uavs"], key=lambda item: int(item["uav_id"])
            )
        from scenario_manifest import (
            validate_initial_communication_topology,
            validate_permanent_gateway_initial_position,
        )

        validate_initial_communication_topology(
            uav_initial_data,
            scenario_id=(
                self.active_scenario_id
                if self.active_scenario_id is not None
                else "simulator-fallback"
            ),
            gs_position=self.GS_pos,
        )
        validate_permanent_gateway_initial_position(
            uav_initial_data,
            scenario_id=(
                self.active_scenario_id
                if self.active_scenario_id is not None
                else "simulator-fallback"
            ),
            gs_position=self.GS_pos,
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

        # Clear prior-episode profiles without consuming channel RNG. Formal
        # training completes slot-0 SR movement before sampling interval zero.
        self.channel.clear_episode(
            num_sr=len(self.SR_teams), episode_identity=self.active_scenario_id
        )

        # Search is orchestration fallback, not a solver candidate or utility.
        self.task_list = []
        if not self.defer_initial_channel_boundary:
            self.initialize_channel_episode_boundary()

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
    
    

        
        
        
        
        
        
    
        
