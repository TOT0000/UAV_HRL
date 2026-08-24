from collections import deque
import numpy as np
from Fov_model_phase import FovModel
from Energy_model import EnergyConsumptionModel


def straight_line_route(start, goal, speed=1.0):
    """Return movement points after start, ending exactly at the target."""

    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("SR route speed must be finite and positive")
    start_xy = np.asarray(start[:2], dtype=float)
    goal_xy = np.asarray(goal[:2], dtype=float)
    if not np.isfinite(start_xy).all() or not np.isfinite(goal_xy).all():
        raise ValueError("SR route endpoints must be finite")
    delta = goal_xy - start_xy
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return []
    direction = delta / distance
    step_count = int(np.ceil(distance / speed))
    route = [
        tuple(start_xy + direction * min(step * speed, distance))
        for step in range(1, step_count + 1)
    ]
    route[-1] = tuple(goal_xy)
    return route

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
                uav_idx=self.id,
                v_h=v_h,
                v_v=v_v,
                t=step_time,
                velocity_vector=(
                    actual_dx / max(step_time, 1e-9),
                    actual_dy / max(step_time, 1e-9),
                    actual_dz / max(step_time, 1e-9),
                ),
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

        new_x = float(np.clip(raw_x, 0.0, 1000.0))
        new_y = float(np.clip(raw_y, 0.0, 1000.0))
        new_z = float(np.clip(raw_z, self.min_AGL, self.max_AGL))

        terrain_uav_z = terrain_func(new_x, new_y) if terrain_func else 0.0
        self.min_AGL = terrain_uav_z + 50
        self.max_AGL = terrain_uav_z + 200

        self.move_to(new_x, new_y, new_z)

        # 能耗用「實際位移 / step_time」計
        actual_dx = float(self.x_u) - float(old_x)
        actual_dy = float(self.y_u) - float(old_y)
        actual_dz = float(self.z_u) - float(old_z)
        v_h = np.hypot(actual_dx, actual_dy) / step_time
        v_v = actual_dz / step_time

        if energy_model is not None:
            E_mob = energy_model.compute_mobility_energy(
                uav_idx=self.id,
                v_h=v_h,
                v_v=v_v,
                t=step_time,
                velocity_vector=(actual_dx / step_time, actual_dy / step_time, v_v),
            )
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
        self.arrived = False
        self.current_step = 0
        self.x, self.y, self.z = 0.0, 0.0, 0.0

    @property
    def is_moving(self):
        return self.assigned_gt_id is not None and not self.arrived

    @property
    def com_source_enabled(self):
        return self.assigned_gt_id is not None

    @property
    def active(self):
        """Read-only compatibility alias for the derived moving state."""

        return self.is_moving

    def reset_lifecycle(self):
        self.assigned_gt_id = None
        self.arrived = False
        self.path = []
        self.current_step = 0
    def get_position(self):
        return self.x, self.y, self.z
    def assign_mission(self, gt_id, gt_pos, speed=1.0):
        """
        ✅ 由環境分派 GT 任務並啟動移動
        """
        if self.assigned_gt_id is not None:
            raise RuntimeError(
                f"SR {self.id} is permanently assigned to GT "
                f"{self.assigned_gt_id} for this episode"
            )
        self.assigned_gt_id = int(gt_id)
        self.path = straight_line_route(self.get_position(), gt_pos, speed)
        self.arrived = not self.path
        self.current_step = 0
    def step_forward(self):
        if not self.is_moving:
            return
        if self.path is None or len(self.path) == 0:
            self.arrived = True
            return
        if self.current_step < len(self.path):
            nx, ny = self.path[self.current_step]
            self.x, self.y = float(nx), float(ny)
            self.z = 0.0
            self.current_step += 1
        if self.current_step >= len(self.path):
            self.arrived = True

    def route_state(self):
        """Serialize the exact waypoint cursor used by deterministic resume."""

        return {
            "sr_id": int(self.id),
            "assigned_gt_id": self.assigned_gt_id,
            "path": [[float(x), float(y)] for x, y in self.path],
            "current_step": int(self.current_step),
            "position": [float(self.x), float(self.y), float(self.z)],
            "arrived": bool(self.arrived),
        }

    def load_route_state(self, state):
        """Restore position, route, cursor, and arrival flags without replanning."""

        if int(state.get("sr_id", -1)) != int(self.id):
            raise RuntimeError("SR route state belongs to a different team")
        path = [tuple(map(float, point)) for point in state.get("path", [])]
        if any(len(point) != 2 or not np.isfinite(point).all() for point in path):
            raise RuntimeError("SR route checkpoint path is invalid")
        current_step = int(state.get("current_step", -1))
        if not 0 <= current_step <= len(path):
            raise RuntimeError("SR route checkpoint cursor is invalid")
        position = np.asarray(state.get("position", []), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise RuntimeError("SR route checkpoint position is invalid")
        arrived = bool(state.get("arrived", False))
        assigned_gt_id = state.get("assigned_gt_id")
        idle = assigned_gt_id is None
        expected_arrived = bool(assigned_gt_id is not None and current_step >= len(path))
        if (
            (idle and (arrived or path or current_step != 0))
            or (not idle and arrived != expected_arrived)
        ):
            raise RuntimeError("SR route checkpoint lifecycle flags are inconsistent")
        self.assigned_gt_id = assigned_gt_id
        self.path = path
        self.current_step = current_step
        self.x, self.y, self.z = position.tolist()
        self.arrived = arrived
