from collections import deque
import numpy as np
from Fov_model_phase import FovModel
from Energy_model import EnergyConsumptionModel

class UAV:
    def __init__(self, id, x, y, z, battery=100, task_type=None, max_buffer_bits=1e6, assigned_target_id=None, E_max=10000):
        self.energy = E_max
        self.last_energy = self.energy
        self.battery = 100  #設定電量
        self.update_battery(self.energy, E_max)
        self.id = id          # 無人機 ID
        self.x_u = x          # 目前 x 座標
        self.y_u = y          # 目前 y 座標
        self.z_u = z          # 目前 z 座標
        self.task = None
        self.speed = 10       # m/s
        self.energy_step = 1e-6
        self.buffer = []      # 可用來存放封包
        self.max_buffer_bits= max_buffer_bits 
        self.task_type = task_type 
        self.assigned_sr = None  # 初始化為未分配狀態
        self.assigned_target_id = assigned_target_id
        self.current_buffer_bits = 0   # 目前 buffer 佔用大小
        self.min_AGL = 50
        self.max_AGL = 150
        self.active_task_index = 0  
        self.last_box_idx = None
        self.recent_fov_history = deque(maxlen=8)  # 8步的短期記憶
        self.bit_resolution=2
        
    # ========== 移動與位置 ==========
    def get_position(self):
        return self.x_u, self.y_u, self.z_u
    def move_to(self, x_new, y_new, z_new, env_width=1000.0, env_height=1000.0):
        self.x_u = min(max(x_new, 0), float(env_width))
        self.y_u = min(max(y_new, 0), float(env_height))
        self.z_u = np.clip(z_new, self.min_AGL, self.max_AGL)
        # print(f"[MOVE DEBUG] UAV {self.id} actual object id: {id(self)}")

    def propose_movement(
        self,
        dx,
        dy,
        dz,
        step_time=1.0,
        mobility_params=None,
        env_width=1000.0,
        env_height=1000.0,
        v_max_phys=10.0,
        max_step_ratio=0.60,
        dz_cap=10.0,
    ):
        """Build a deterministic movement proposal without mutating the UAV."""
        step_time = float(step_time)
        dx_m = float(dx) * step_time
        dy_m = float(dy) * step_time
        dz_m = float(np.clip(float(dz) * step_time, -dz_cap, dz_cap))

        fov_model = FovModel(
            f=0.004, wl=0.008, i_l=0.012, z_u=self.z_u, gamma_g=80
        )
        fov_w, _ = fov_model.get_ground_fov_size(self.z_u)
        horizontal_cap = min(max_step_ratio * fov_w, v_max_phys * step_time)
        horizontal_distance = float(np.hypot(dx_m, dy_m))
        if horizontal_distance > horizontal_cap > 0:
            scale = horizontal_cap / horizontal_distance
            dx_m *= scale
            dy_m *= scale

        raw_x = float(self.x_u) + dx_m
        raw_y = float(self.y_u) + dy_m
        raw_z = float(self.z_u) + dz_m

        comm_safe = mobility_params.get("comm_safety", {}) if mobility_params else {}
        only_ids = comm_safe.get("only_uav_ids", [0])
        apply_safe = comm_safe.get("enable", False) and self.id in only_ids
        is_stationary = bool(
            np.isclose(dx_m, 0.0) and np.isclose(dy_m, 0.0) and np.isclose(dz_m, 0.0)
        )
        if apply_safe and not is_stationary:
            raw_x, raw_y, raw_z = self._project_to_comm_safe_region(
                raw_x, raw_y, raw_z, mobility_params
            )

        new_x = float(np.clip(raw_x, 0.0, float(env_width)))
        new_y = float(np.clip(raw_y, 0.0, float(env_height)))
        new_z = float(np.clip(raw_z, self.min_AGL, self.max_AGL))
        return {
            "old_position": (float(self.x_u), float(self.y_u), float(self.z_u)),
            "new_position": (new_x, new_y, new_z),
            "env_width": float(env_width),
            "env_height": float(env_height),
        }

    def apply_movement_proposal(self, proposal, energy_model, step_time=1.0):
        """Apply a precomputed proposal and charge energy from actual displacement."""
        old_x, old_y, old_z = map(float, proposal["old_position"])
        if not np.allclose((self.x_u, self.y_u, self.z_u), (old_x, old_y, old_z)):
            raise RuntimeError(
                f"UAV {self.id} changed after its movement proposal was created"
            )

        new_x, new_y, new_z = map(float, proposal["new_position"])
        step_time = float(step_time)
        self.last_position = (old_x, old_y, old_z)
        self.move_to(
            new_x,
            new_y,
            new_z,
            env_width=proposal["env_width"],
            env_height=proposal["env_height"],
        )

        actual_dx = float(self.x_u) - old_x
        actual_dy = float(self.y_u) - old_y
        actual_dz = float(self.z_u) - old_z
        v_h = float(np.hypot(actual_dx, actual_dy) / max(step_time, 1e-9))
        v_v = float(actual_dz / max(step_time, 1e-9))
        energy = float(
            energy_model.compute_mobility_energy(
                uav_idx=self.id, v_h=v_h, v_v=v_v, t=step_time
            )
        )
        self.last_energy = self.energy
        self.energy = max(self.energy - energy, 0.0)
        self.move_energy_step = energy
        self.update_battery(self.energy, energy_model.E_max)
        return energy
    # ==============無人機移動限制模型===========================
    def apply_movement(self, dx, dy, dz, speed=20, terrain_func=None, energy_model=None,
                   step_time=1, mobility_params=None, 
                   v_max_phys=10,            # 物理最大速度 (m/s)
                   min_step_ratio=0.30,        # η：最小步長比例
                   max_step_ratio=0.60,        # κ：最大步長比例
                   dz_cap=10.0,):
        dx_m = dx * step_time
        dy_m = dy * step_time
        dz_m = dz * step_time
        # 2) 依當前高度計 FOV 寬，設定上下限
        self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=self.z_u, gamma_g=80)
        fov_w, _ = self.FovModel.get_ground_fov_size(self.z_u)
        min_step = min_step_ratio * fov_w
        phys_cap = float(v_max_phys) * float(step_time)
        max_step = min(max_step_ratio * fov_w, phys_cap)

        # 3) 對水平位移套「最小/最大步長」
        horiz = np.hypot(dx_m, dy_m)
        if horiz < 1e-3:  
            # 幾乎完全靜止，才強制給一個小步
            theta = np.random.uniform(0, 2*np.pi)
            dx_m = (0.2 * min_step) * np.cos(theta)   
            dy_m = (0.2 * min_step) * np.sin(theta)
        else:
            # 正常情況 → 只做上界限制，不要下界
            target = min(horiz, max_step)
            if not np.isclose(target, horiz):
                scale = target / horiz
                dx_m *= scale
                dy_m *= scale
        # 4) 限制垂直位移（避免高度亂跳）
        dz_m = float(np.clip(dz_m, -dz_cap, dz_cap))

        # ====== 以下維持你原本流程 ======
        old_x, old_y, old_z = self.x_u, self.y_u, self.z_u
        self.last_position = (old_x, old_y, old_z)

        raw_x = self.x_u + dx_m
        raw_y = self.y_u + dy_m
        raw_z = self.z_u + dz_m

        comm_safe = mobility_params.get("comm_safety", {}) if mobility_params else {}
        only_ids = comm_safe.get("only_uav_ids", [0])
        apply_safe = comm_safe.get("enable", False) and (getattr(self, "id", None) in only_ids)

        if apply_safe:
            # 原本的投影邏輯不變
            proj_x, proj_y, proj_z = self._project_to_comm_safe_region(raw_x, raw_y, raw_z, mobility_params)
        else:
            proj_x, proj_y, proj_z = raw_x, raw_y, raw_z
        # proj_x, proj_y = raw_x, raw_y  # 預設不投影


        # 更新位置
        self.x_u, self.y_u = proj_x, proj_y
        self.z_u = proj_z

        new_x, new_y, new_z = proj_x, proj_y, raw_z

        terrain_uav_z = terrain_func(new_x, new_y) if terrain_func else 0.0
        self.min_AGL = terrain_uav_z + 50
        self.max_AGL = terrain_uav_z + 200

        self.move_to(new_x, new_y, new_z)

        # 能耗用「實際位移 / step_time」計
        v_h = np.hypot(dx_m, dy_m) / step_time
        v_v = dz_m / step_time

        if energy_model is not None:
            E_mob = energy_model.compute_mobility_energy(uav_idx=self.id, v_h=v_h, v_v=v_v, t=step_time)
            self.last_energy = self.energy
            self.energy = max(self.energy - E_mob, 0)
            self.move_energy_step = E_mob
            self.update_battery(self.energy, energy_model.E_max)
        else:
            E_mob = 0.0

        return E_mob
    # ========== 電量與狀態 ==========
    def update_battery(self, remaining_energy, E_max):
        self.battery = max(0, min(100.0, remaining_energy / E_max * 100))
    def is_low_battery(self, threshold=30):
        return self.battery < threshold
    def is_on_task(self):
        return self.task_type is not None
    
    def _project_to_comm_safe_region(self, x, y, z, mobility_params=None):

        if mobility_params is None:
            return x, y, z

        cs = mobility_params.get("comm_safety", None)
        if not cs or not cs.get("enable", False):
            return x, y, z

        mode = cs.get("mode", "gs_only")

        def project_to_sphere(cx, cy, cz, r_soft, r_hard, px, py, pz):
            vx, vy, vz = px - cx, py - cy, pz - cz
            d = float(np.sqrt(vx**2 + vy**2 + vz**2))

            if d < 1e-9:
                return px, py, pz

            # hard bound: 超過就直接投影到球面上
            if d >= r_hard:
                s = r_hard / d
                return cx + vx * s, cy + vy * s, cz + vz * s

            # soft band: 接近邊界就漸進縮放
            if d > r_soft:
                alpha = (d - r_soft) / max(r_hard - r_soft, 1e-9)
                target = (1 - alpha) * d + alpha * r_soft
                s = target / d
                return cx + vx * s, cy + vy * s, cz + vz * s

            return px, py, pz

        def within(cx, cy, cz, r, px, py, pz):
            return (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2 <= r ** 2

        gs_pos = cs.get("gs_pos", (0.0, 0.0, 0.0))
        r_soft = float(cs.get("r_soft", 180.0))
        r_hard = float(cs.get("r_hard", 200.0))

        # --- GS only：強制留在 GS 安全球內 ---
        if mode == "gs_only":
            return project_to_sphere(
                gs_pos[0], gs_pos[1], gs_pos[2],
                r_soft, r_hard,
                x, y, z
            )

        # --- GS or Relay：只要能連 GS 或至少一個 relay 就 OK ---
        relay_positions = cs.get("relay_positions", []) or []
        rrs = float(cs.get("r_relay_soft", 360.0))
        rrh = float(cs.get("r_relay_hard", 400.0))

        # 先檢查是否已經在任何可行球內
        if within(gs_pos[0], gs_pos[1], gs_pos[2], r_hard, x, y, z):
            return x, y, z

        for (rx, ry, rz) in relay_positions:
            if within(rx, ry, rz, rrh, x, y, z):
                return x, y, z

        # 不滿足 -> 投影到最近的一個可行球（GS 或 relay）
        px, py, pz = project_to_sphere(
            gs_pos[0], gs_pos[1], gs_pos[2],
            r_soft, r_hard,
            x, y, z
        )
        best = (px, py, pz, (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)

        for (rx, ry, rz) in relay_positions:
            qx, qy, qz = project_to_sphere(rx, ry, rz, rrs, rrh, x, y, z)
            cost = (x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2
            if cost < best[3]:
                best = (qx, qy, qz, cost)

        return best[0], best[1], best[2]
    # ============將封包加入buffer=====================
    def add_packet_to_buffer(self, pkt, pkt_bits):
        current_bits = sum(p['bits'] for p in self.buffer)
        if current_bits + pkt_bits <= self.max_buffer_bits:
            pkt['bits'] = pkt_bits
            self.buffer.append(pkt)
            return True
        return False
    # ================離開的封包======================
    def remove_packet(self, pkt_id):
        self.buffer = [p for p in self.buffer if p['id'] != pkt_id]
    # =============取得buffer情況======================
    def get_buffer_bits(self):
        return sum(p['bits'] for p in self.buffer)

class GroundTarget:
    def __init__(self, id, x, y, z=0, radius=80):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.is_found = False
        self.found_by = None  
        self.assigned = False
        self.rewarded = False 
    def get_position(self):
        return self.x, self.y, self.z
    def mark_found(self, uav_id):
        self.is_found = True
        self.found_by = uav_id
        # # ✅ 印出觸發資訊
        # import traceback
        # print(f"\n[TRACE] 🚨 GT {self.id} 被設定為 is_found = True by UAV {uav_id}")
        # traceback.print_stack(limit=4)
        

class SRTeam:
    def __init__(self, id):
        self.id = id
        self.assigned_gt_id = None
        self.path = []           # 儲存從目前到 GT 的移動路徑
        self.active = False      # 是否出動中
        self.arrived = False
        self.current_step = 0
        self.x, self.y, self.z = 0.0, 0.0, 0.0
    def get_position(self):
        return self.x, self.y, self.z
    def assign_mission(self, gt_id, gt_pos, speed=1.0):
        """
        ✅ 由環境分派 GT 任務並啟動移動
        """
        self.assigned_gt_id = gt_id
        self.path = self.plan_path(self.get_position(), gt_pos, speed)
        self.active = True
        self.arrived = False
        self.current_step = 0
    def step_forward(self):
        if not self.active:
            return
        if self.path is None or len(self.path) == 0:
            return
        if self.current_step < len(self.path):
            nx, ny = self.path[self.current_step]
            self.x, self.y = float(nx), float(ny)
            self.z = 0.0
            self.current_step += 1
        else:
            self.active = False
            self.arrived = True
