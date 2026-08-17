import math
from collections import defaultdict, deque
from Energy_model import EnergyConsumptionModel
from Fov_model_phase import FovModel
from Channel_model import ChannelModel
from centralized_movement import vs_data_valid
import numpy as np


def final_hop_delivered_bits(to_target, gs_id, bits_tx_used):
    if int(to_target) != int(gs_id):
        return 0.0
    return max(float(bits_tx_used), 0.0)


MAX_PACKET_HOPS = 20
PACKET_EPS = 1e-9
TASK_DEADLINE_SECONDS = {"FOV": 1.5, "COM": 1.0}



class PacketEngine:
    def __init__(self, num_uav, step_time=0.25, E_max=10000):  # ★ 改成可傳入 UAV 數
        self.step_time = step_time
        self.num_UAV = num_uav                                 # ★ 存起來，避免外部依賴
        self._next_pkt_id = 0
        self._active_idx = []

        # PacketManager
        self.packet_pool = []
        self.inject_buffer = defaultdict(float)
        self.source_buffer = defaultdict(float)

        # The global pool is only a lifecycle/statistics index. Forwarding order
        # is owned by one aggregate FIFO per UAV and may mix FOV and COM packets.
        self.uav_queues = {uav_id: deque() for uav_id in range(num_uav)}

        # === Dual-Queue backlog tracking (no weights) ===
        # backlog_bits_by_task[node_id][task] stores SUM of remaining bits at that node for that task.
        # task ∈ {"FOV","COM"}; unknown tasks are mapped to "COM".
        self.backlog_bits = defaultdict(float)

        # 能量模型（假設你的 EnergyConsumptionModel 需要 N_u）
        self.energy_model = EnergyConsumptionModel(E_max=E_max, N_u=self.num_UAV)

        # LinkEstimator / 統計
        self.buffer_info = {}
        self.actual_backlog = {}
        self.forwarding_rate = {}
        self.total_delivered = 0
        self.total_violated = 0
        self.fov_delivered= 0
        self.com_delivered= 0
        self.energy = E_max
        self.last_energy = E_max
        self.MAX_ACTIVE_HIGH = 50000
        self.MAX_ACTIVE_LOW  = 20000
        self.backpressure_on = False

        self.done_delay_buf = []
        self.done_type_buf  = []
        self.DELAY_CHUNK = 50000
        self.delay_chunk_id = 0
        self.fov_delivered = 0
        self.com_delivered = 0
        self.fov_violated = 0
        self.com_violated = 0
        self.partial_transmissions = 0

        self.bp_skip_inject_steps = 0
        # self.max_packets = 3000

        # （可選）保留 GC 參數，未使用也無妨
        self._active_gc_every = 50
        # （可選）保留 GC 參數，未使用也無妨
        self._active_gc_every = 50
        self.delay_log = []  
        self.type_delay_accum = {
            "FOV": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
            "COM": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
        }
    # （可選）若要用到才保留；否則刪掉它避免 self.num_UAV 未定義
    def initialize_packet_buffer(self, num_pkt):
        self.packet_buffer = {
            uav_id: [{"source": uav_id + 1, "hops": 0, "done": False}
                     for _ in range(num_pkt)]
            for uav_id in range(self.num_UAV + 1)              # ★ 用 self.num_UAV
        }
        self.source_uavs = set(self.packet_buffer.keys())

    def _sync_backlog(self, uav_id):
        if not (0 <= int(uav_id) < self.num_UAV):
            return 0.0
        queue = self.uav_queues[int(uav_id)]
        total = sum(
            max(float(pkt.get("rem_bits", 0.0)), 0.0)
            for pkt in queue
            if pkt is not None and not pkt.get("done", False)
        )
        self.backlog_bits[int(uav_id)] = max(float(total), 0.0)
        return self.backlog_bits[int(uav_id)]

    def _remove_from_queue(self, pkt, uav_id=None):
        candidate_ids = (
            [int(uav_id)] if uav_id is not None else range(self.num_UAV)
        )
        removed = False
        for node_id in candidate_ids:
            if not (0 <= node_id < self.num_UAV):
                continue
            queue = self.uav_queues[node_id]
            try:
                queue.remove(pkt)
                removed = True
            except ValueError:
                pass
            self._sync_backlog(node_id)
            if removed:
                break
        return removed

    def enqueue_packet(self, pkt, uav_id, queue_enter_time):
        """Append a fully arrived packet to a UAV's aggregate FIFO."""

        uav_id = int(uav_id)
        if not (0 <= uav_id < self.num_UAV):
            raise ValueError(f"invalid UAV queue id: {uav_id}")
        pkt["current"] = uav_id
        pkt["queue_enter_time"] = float(queue_enter_time)
        self.uav_queues[uav_id].append(pkt)
        self._sync_backlog(uav_id)

    def create_packet(self, source, task_type, size_bits, generation_time):
        """Create and enqueue one packet; shared by injection and unit tests."""

        source = int(source)
        task_type = self._task_norm(task_type)
        size_bits = float(size_bits)
        generation_time = float(generation_time)
        pool_idx = len(self.packet_pool)
        pkt = {
            "id": self._next_pkt_id,
            "_pool_idx": pool_idx,
            "source": source,
            "current": source,
            "arrival_time": generation_time,
            "generation_time": generation_time,
            "queue_enter_time": generation_time,
            "deadline": 8,
            "done": False,
            "hops": 0,
            "task_type": task_type,
            "size_bits": size_bits,
            "rem_bits": size_bits,
            "path": [source],
            "e2e_delay_ms": 0.0,
            "bn_path_mbps": float("inf"),
            "bn_final_mbps": None,
            "bn_counted": False,
            "hop_receiver": None,
            "hop_bits_sent": 0.0,
        }
        self.packet_pool.append(pkt)
        self._active_idx.append(pool_idx)
        self.enqueue_packet(pkt, source, generation_time)
        self._next_pkt_id += 1
        return pkt

    def get_queue_packets(self, uav_id):
        return [
            pkt
            for pkt in self.uav_queues[int(uav_id)]
            if pkt is not None and not pkt.get("done", False)
        ]

    def get_hol_packet(self, uav_id):
        queue = self.uav_queues[int(uav_id)]
        while queue and (queue[0] is None or queue[0].get("done", False)):
            queue.popleft()
        self._sync_backlog(int(uav_id))
        return queue[0] if queue else None

    def nonempty_uav_ids(self):
        return [
            uav_id
            for uav_id in range(self.num_UAV)
            if self.get_hol_packet(uav_id) is not None
        ]

    def get_effective_action_mask(self, env, uav_id, physical_mask=None):
        """Return physical legal links plus Wait, restricted by a hop lock."""

        uav_id = int(uav_id)
        if physical_mask is None:
            physical_mask = env.get_routing_action_mask(uav_id)
        mask = np.asarray(physical_mask, dtype=bool).copy()
        if mask.shape != (self.num_UAV + 1,):
            raise ValueError(
                f"routing mask has shape {mask.shape}, expected "
                f"({self.num_UAV + 1},)"
            )
        mask[uav_id] = True
        hol = self.get_hol_packet(uav_id)
        locked_receiver = hol.get("hop_receiver") if hol is not None else None
        if locked_receiver is not None:
            locked_receiver = int(locked_receiver)
            locked_is_physical = bool(mask[locked_receiver])
            mask[:] = False
            mask[uav_id] = True
            if locked_is_physical:
                mask[locked_receiver] = True
        return mask

    def record_hop_transmission(self, pkt, sender, receiver, bits):
        """Apply partial hop service while keeping the packet at its sender."""

        sender = int(sender)
        receiver = int(receiver)
        bits = max(float(bits), 0.0)
        if pkt is not self.get_hol_packet(sender):
            raise AssertionError("only the aggregate FIFO HOL packet may transmit")
        locked_receiver = pkt.get("hop_receiver")
        if locked_receiver is not None and int(locked_receiver) != receiver:
            raise AssertionError(
                f"packet {pkt['id']} hop is locked to {locked_receiver}, not {receiver}"
            )
        if bits <= 0.0:
            return False
        if locked_receiver is None:
            pkt["hop_receiver"] = receiver
        remaining = max(float(pkt["rem_bits"]), 0.0)
        used = min(bits, remaining)
        pkt["rem_bits"] = max(remaining - used, 0.0)
        pkt["hop_bits_sent"] = float(pkt.get("hop_bits_sent", 0.0)) + used
        if pkt["rem_bits"] > PACKET_EPS:
            self.partial_transmissions += 1
        self._sync_backlog(sender)
        return pkt["rem_bits"] <= PACKET_EPS

    def detach_completed_hop(self, pkt, sender, receiver, completion_time):
        """Remove a full hop from the sender and return a pending relay arrival."""

        sender = int(sender)
        receiver = int(receiver)
        locked_receiver = pkt.get("hop_receiver")
        if locked_receiver is None or int(locked_receiver) != receiver:
            raise AssertionError("completed hop does not match the receiver lock")
        if float(pkt.get("rem_bits", 0.0)) > PACKET_EPS:
            raise AssertionError("cannot detach an incomplete hop")
        if not self._remove_from_queue(pkt, sender):
            raise AssertionError("completed packet was not in the sender FIFO")
        pkt["hops"] = int(pkt.get("hops", 0)) + 1
        pkt.setdefault("path", []).append(receiver)
        pkt["current"] = receiver
        pkt["hop_receiver"] = None
        pkt["hop_bits_sent"] = 0.0
        return {
            "packet": pkt,
            "receiver": receiver,
            "completion_time": float(completion_time),
            "sender": sender,
            "packet_id": int(pkt["id"]),
        }

    def enqueue_relay_arrivals(self, arrivals):
        """Append full relay arrivals deterministically after slot service ends."""

        ordered = sorted(
            arrivals,
            key=lambda item: (
                float(item["completion_time"]),
                int(item["sender"]),
                int(item["packet_id"]),
            ),
        )
        for arrival in ordered:
            pkt = arrival["packet"]
            pkt["rem_bits"] = float(pkt["size_bits"])
            self.enqueue_packet(
                pkt, arrival["receiver"], arrival["completion_time"]
            )
        return ordered

    def inject_packets(self, env, delay_bound_steps, current_time, step_time=0.25,
                       base_fov_rate=5, base_ctrl_rate=50):
        # if self.target_total_packets is not None and self.total_injected_packets >= self.target_total_packets:
        #     return
        # 這些屬性在 __init__ 都建好了，以下保守檢查可留可去
        if not hasattr(self, "inject_buffer"):
            self.inject_buffer = defaultdict(float)
        if not hasattr(self, "packet_pool"):
            self.packet_pool = []
        if not hasattr(self, "_active_idx"):
            self._active_idx = []
        # if len(self.packet_pool) >= getattr(self, "max_packets", 3000):
        #     # print(f"超過3000")
        #     return
        load_factor = getattr(env, "load_factor", 1.0)
        base_fov_rate  = base_fov_rate  * load_factor
        base_ctrl_rate = base_ctrl_rate * load_factor

        for uav_id in env.source_uavs:
            task_list = env.multi_tasks.get(uav_id, [])
            for task in task_list:
                task_type = task["task_type"]
                if task_type not in ["FOV", "COM"]:
                    continue
                if task_type == "FOV" and not vs_data_valid(env, uav_id, task):
                    # Existing queued/relayed VS packets remain untouched; only new
                    # source generation is gated by current geometry and full ROI coverage.
                    continue
                rate = (base_fov_rate if task_type == "FOV" else base_ctrl_rate)
                
                # === 基於速率積分的封包計數 ===
                key = f"{uav_id}_{task_type}"
                self.inject_buffer[key] += rate * step_time
                num_packets = int(self.inject_buffer[key])
                if num_packets <= 0:
                    continue
                self.inject_buffer[key] -= num_packets

                # === 檢查剩餘封包名額 ===
                # if self.target_total_packets is not None:
                #     remain = self.target_total_packets - self.total_injected_packets
                #     if remain <= 0:
                #         return
                #     if num_packets > remain:
                #         num_packets = remain

                if task_type == "FOV":
                    uav = env.uav_dict[uav_id]
                    x_tgt, y_tgt, z_tgt = task["target_pos"]
                    # 確保有 FovModel
                    if not hasattr(self, "FovModel"):
                        self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
                    self.FovModel.z_u = uav.z_u

                    # 依當下幾何關係計算 FOV 面積（地面投影像素/覆蓋）
                    current_fov, _ = self.FovModel.calculate_fov_single(
                        uav.x_u, uav.y_u, uav.z_u, x_tgt, y_tgt, z_tgt
                    )
                    max_fov = min (1, current_fov)
                    # 你的 FOV→bits 公式（與 backlog 現用一致，以免前後不一）
                    wl, i_l, tau = 0.008, 0.012, 3.9e-6
                    pkt_bits = 0.005 * (wl * i_l / (tau ** 2)) * max_fov
                else:
                    pkt_bits = 256

                for _ in range(num_packets):
                    self.create_packet(
                        uav_id, task_type, pkt_bits, current_time
                    )
                    # self.total_injected_packets += 1
                    # if self.total_injected_packets >= self.target_total_packets:
                    #     print(f"✅ Packet quota reached: {self.total_injected_packets}")
                    #     return

    def mark_packet_done(self, pkt, current_time=None, reason=None):
        """
        ✅ Confirmed for your current engine structure:
        - self.packet_pool: list (pool index -> pkt or None)
        - self._active_idx: list of pool indices (ints)
        - pkt contains "_pool_idx"
        - remaining bits is pkt["rem_bits"]

        Behavior:
        1) mark pkt done + reason + finish_time
        2) cleanup backlog_bits at current node using rem_bits (safe clamp)
        3) remove pool_idx from _active_idx
        4) set packet_pool[pool_idx] = None
        """
        if pkt is None:
            return

        # 1) mark done
        pkt["done"] = True
        if reason is not None:
            pkt["reason"] = reason
        else:
            pkt.setdefault("reason", "done")
        if current_time is not None:
            pkt["finish_time"] = current_time

        # 2) remove the packet from whichever per-UAV FIFO currently owns it.
        self._remove_from_queue(pkt)

        # 3) remove from active indices
        pi = pkt.get("_pool_idx", None)
        if pi is None:
            return
        try:
            pi = int(pi)
        except Exception:
            return

        # try fast remove once; fallback remove-all (safer if duplicates ever exist)
        try:
            self._active_idx.remove(pi)
        except ValueError:
            # remove-all fallback
            self._active_idx = [x for x in self._active_idx if x != pi]

        # 4) clear pool slot
        if 0 <= pi < len(self.packet_pool):
            self.packet_pool[pi] = None



    def drop_expired_packets(self, current_time):
        """
        Drop packets that are expired by deadline or exceed max hops.
        Safe against: packet_pool entries being None, active_idx changing, etc.
        """
        dropped = 0

        for idx in list(self._active_idx):
            # --- 安全檢查：索引合法性 ---
            if idx is None or idx < 0 or idx >= len(self.packet_pool):
                try:
                    self._active_idx.remove(idx)
                except ValueError:
                    pass
                continue

            pkt = self.packet_pool[idx]
            if pkt is None:
                try:
                    self._active_idx.remove(idx)
                except ValueError:
                    pass
                continue

            if pkt.get("done", False):
                try:
                    self._active_idx.remove(idx)
                except ValueError:
                    pass
                continue

            # =========================
            # (A) DEADLINE EXPIRED DROP
            # =========================
            # if current_time > pkt.get("deadline", float("inf")):
            #     # backlog 扣掉（避免 backlog 虛胖）
            #     if cur >= 0 and rem_bits > 0:
            #         self.backlog_bits[cur] = max(0.0, self.backlog_bits[cur] - rem_bits)
            #     # 清掉 active / pool
            #     try:
            #         self._active_idx.remove(idx)
            #     except ValueError:
            #         pass
            #     self.packet_pool[idx] = None
            #     dropped += 1
            #     continue

            # =========================
            # (B) MAX HOPS DROP
            # =========================
            hops = int(pkt.get("hops", 0))
            if hops >= MAX_PACKET_HOPS:
                self.mark_packet_done(
                    pkt, current_time=current_time, reason="max_hops"
                )
                dropped += 1
                continue

        return dropped


    
    # ===== Dual-Queue helpers (no weights) =====
    def _task_norm(self, task_type) -> str:
        t = str(task_type).upper()
        return "FOV" if "FOV" in t else "COM"

    def get_backlog_total(self, node_id:int)->float:
        return float(self.backlog_bits[node_id])

    def get_backlog_task(self, node_id:int, task_type:str)->float:
        # 單一 backlog 沒有 per-task queue：回傳 total 即可（避免舊介面壞掉）
        return float(self.backlog_bits[node_id])

    def active_count(self):
        """回傳目前 active packet 數量（用 _active_idx 管理）"""
        return len(self._active_idx)

    def get_active_packets(self):
        # ★ 安全檢查：邊界 + None（若未來釋放記憶體改成 None 也可）
        pool = self.packet_pool
        idxs = self._active_idx
        safe_idxs = [i for i in idxs if 0 <= i < len(pool) and pool[i] is not None]
        # 若出現壞索引，順手同步修正 _active_idx
        if len(safe_idxs) != len(idxs):
            self._active_idx = safe_idxs
        # 過濾 done
        return [pool[i] for i in safe_idxs if not pool[i]["done"]]

    def reset_packet_state(self):
        # 封包本體
        self.packet_pool = []
        # 活躍索引
        self._active_idx = []
        # 流水號歸零（或保留原值也行）
        self._next_pkt_id = 0
        # 速率積分緩衝清空
        self.inject_buffer.clear()
        self.source_buffer.clear()
        # Dual-Queue backlog tracking
        self.backlog_bits.clear()
        self.uav_queues = {
            uav_id: deque() for uav_id in range(self.num_UAV)
        }
        # 其他快取
        self.buffer_info = {}
        self.actual_backlog = {}
        self.forwarding_rate = {}
        self.total_delivered = 0
        self.total_violated = 0
        self.fov_delivered= 0
        self.com_delivered= 0
        self.partial_transmissions = 0
        self.delay_log = []  # 每跳記錄
        self.type_delay_accum = {
            "FOV": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
            "COM": {"sum_queue": 0.0, "sum_tx": 0.0, "sum_total": 0.0, "count": 0},
        }

        # 總延遲時間
    def log_hop_delay(self, env, pkt, current_node, next_hop, link_capacity_mbps, current_time, pkt_bits, backlog_bits):
        """用這顆封包的 bits 計算該 hop 的 queue + tx 延遲，並記錄（不在這裡加 GS +0.1）。"""
        # slot_time = getattr(self, "step_time", 0.25)
        if link_capacity_mbps <= 0.1:
            return 0.0
        service_bps = max(float(link_capacity_mbps) * 1e6 , 1e-6)
        # print(backlog_bits)
        # print(f"UAV ID = {current_node} task type = {pkt["task_type"]} UAV total bits = {backlog_bits_total} ")
        queue_delay_s = (backlog_bits / service_bps) if backlog_bits > 0.0 else 0.0

        # 傳輸延遲
        tx_delay_s = (pkt_bits / service_bps) if pkt_bits > 0.0 else 0.0
        # 總延遲時間
        hop_delay_ms = (queue_delay_s + tx_delay_s)*1e3 
        pkt["e2e_delay_ms"] = pkt.get("e2e_delay_ms", 0.0) + hop_delay_ms
        pkt.setdefault("per_hop", []).append({
        "from": current_node, "to": next_hop,
        "queue_s": queue_delay_s, "tx_s": tx_delay_s, "delay_ms": hop_delay_ms
        })
        # print(
        #     f"[PKT] id={pkt['id']} "
        #     f"type={pkt.get('task_type')} "
        #     f"{current_node}->{next_hop} | "
        #     f"queue={queue_delay_s*1e3:.3f} ms, "
        #     f"tx={tx_delay_s*1e3:.3f} ms, "
        #     f"hop={hop_delay_ms:.3f} ms"
        # )
        # if pkt.get("bits", 0.0) <= 1e-9:
        #     print(f"[ZERO_BITS] id={pkt['id']} cur={pkt['current']} next={next_hop} bits={pkt.get('bits')} done={pkt.get('done')} path={pkt.get('path')[-5:]}")
        if getattr(env, "enable_delay_log", False):
            self.delay_log.append({
                "time": current_time,
                "pkt_id": pkt["id"],
                "task_type": pkt.get("task_type", "UNK"),
                "uav_id": current_node,
                "next_hop": next_hop,
                "capacity_mbps": float(link_capacity_mbps),
                "queue_delay_s": queue_delay_s,
                "tx_delay_s": tx_delay_s,
                "hop_delay_ms": hop_delay_ms,
            })
        
        return hop_delay_ms

    def get_type_delay_stats(self):
        out = {}
        for tt, acc in self.type_delay_accum.items():
            c = max(acc["count"], 1)
            out[tt] = {
                "avg_queue_s": acc["sum_queue"] / c,
                "avg_tx_s":    acc["sum_tx"]    / c,
                "avg_total_s": acc["sum_total"] / c,
                "hops_count":  acc["count"],
            }
        return out

    
    def get_per_packet_e2e(self):
        # 若你相信 pkt["e2e_delay_s"] 已在到達 GS 時累加完成，也可以直接讀 pkt["e2e_delay_s"]
        e2e = defaultdict(lambda: {"task_type": None, "sum_hop_ms": 0.0, "hops": 0})
        for r in self.delay_log:
            pid = r["pkt_id"]
            e2e[pid]["task_type"] = r["task_type"]
            e2e[pid]["sum_hop_ms"] += r["hop_delay_ms"]
            e2e[pid]["hops"]+= 1
        return [{"pkt_id": pid, **v} for pid, v in e2e.items()]
    def debug_print_state_v2(self, uav_id, state, env):
        """
        統一版 State Debug Print，格式類似你原本的版本
        """
        N = env.num_UAV
        L = N + 1

        idx = 0
        print("\n=== State Breakdown ===")

        # UAV one-hot
        uav_one_hot = state[idx:idx+N]; idx += N
        print("UAV one-hot ID:", uav_one_hot)

        # Energy / My backlog
        energy_norm = state[idx]; idx += 1
        my_backlog = state[idx]; idx += 1
        print("Normalized energy:", energy_norm)
        print("My backlog (log-norm):", my_backlog)

        # Task flags / FOV flag
        task_flags = state[idx:idx+4]; idx += 4
        fov_task_flag = state[idx]; idx += 1
        print("Task flags:", task_flags, "FOV_task_flag:", fov_task_flag)

        # Link info
        link_valid_mask = state[idx:idx+L]; idx += L
        link_delay_norm = state[idx:idx+L]; idx += L
        link_capacity_norm = state[idx:idx+L]; idx += L
        next_hop_backlog_norm = state[idx:idx+L]; idx += L
        print("Link valid mask:", link_valid_mask)
        print("Link delay vector (norm):", link_delay_norm)
        print("Link capacity vector (norm):", link_capacity_norm)
        print("Next hop backlog vector (log-norm):", next_hop_backlog_norm)

        # Geometry / GS
        uav_position = state[idx:idx+3]; idx += 3
        dist_to_GS_norm = state[idx]; idx += 1
        eta_to_GS_slots_norm = state[idx]; idx += 1
        print("UAV position:", uav_position)
        print("Dist to GS (norm):", dist_to_GS_norm)
        print("ETA to GS (slots, norm):", eta_to_GS_slots_norm)

        # FOV features
        overlap_ema = state[idx]; idx += 1
        unvisited_ema = state[idx]; idx += 1
        frontier_ema = state[idx]; idx += 1
        print("FOV overlap EMA:", overlap_ema)
        print("FOV unvisited EMA:", unvisited_ema)
        print("FOV frontier EMA:", frontier_ema)

        # 與任務資訊
        print(f"[STATE DEBUG] UAV {uav_id} → task_type = {env.uav_dict[uav_id].task_type}")
        print(f"[STATE DEBUG] UAV {uav_id} → multi_tasks = {[t['task_type'] for t in env.multi_tasks.get(uav_id, [])]}")

        print("=== End of State ===\n")

    def get_state(self, env, uav_id, visited_nodes=None, backlog_bits=None):
        """
        Task-aware routing state with dimension 6N+30 (N=16 gives 126).

        The original 6N+26 fields keep their order. Four HOL packet fields
        [is_FOV, is_COM, normalized_slack, normalized_remaining] are appended.
        """
        

        # ---------- 常數/正規化上限（可依環境微調） ----------
        if not hasattr(self, "norm_cfg"):
            self.norm_cfg = dict(
                C_MAX=200.0,   # 容量上限 (Mbps)
                D_MAX=3.0,     # 延遲上限 (s)
                B_MAX=2e6,     # backlog 上限 (bits)
                ETA_MAX=60.0,  # 估計回GS所需slot數的上限
                ema_alpha=0.7  # FOV特徵EMA係數
            )
        C_MAX = float(self.norm_cfg["C_MAX"])
        D_MAX = float(self.norm_cfg["D_MAX"])
        B_MAX = float(self.norm_cfg["B_MAX"])
        ETA_MAX = float(self.norm_cfg["ETA_MAX"])
        ALPHA = float(self.norm_cfg["ema_alpha"])

        # ---------- 基本物件 ----------
        N = env.num_UAV
        L = N + 1                           # 含 GS（GS 索引固定為 env.GS_ID）
        uav = env.uav_dict[uav_id]
        x_u, y_u, z_u = uav.get_position()
        uav_position = np.array([x_u, y_u, z_u], dtype=float)

        # ---------- Energy ----------
        energy_remaining = env.uav_dict[uav_id].energy
        energy_norm = float(energy_remaining / env.E_max)
        # ---------- Backlog (自己) ----------
        max_bits_norm = 5e7
        if backlog_bits is None:
            # 兼容舊寫法：若沒傳，才退回掃描（但訓練時我們會傳，避免走到這裡）
            my_bits_raw = sum(
                pkt["rem_bits"] for pkt in self.packet_pool
                if pkt["current"] == uav_id and not pkt.get("done", False)
            )
        else:
            my_bits_raw = backlog_bits.get(uav_id, 0)
        my_bits = min(my_bits_raw / max_bits_norm, 1.0)
        # # ---------- 任務旗標(規0) ----------
        task_flags = np.zeros(4, dtype=float)     # [Search, FOV, COM, Hovering] 全 0
        fov_task_flag = 0.0                       # 固定 0
        task_feat = np.concatenate([task_flags, np.array([fov_task_flag])], axis=0)  # 4 維

        # ---------- FOV 三特徵（與你現有計法一致，後續做EMA） ----------
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)

        bx_min, bx_max, by_min, by_max, patch, fov_cells = env.fov_to_indices_and_patch(
            x_u, y_u, fov_w, fov_h,
            env.env_width, env.env_height,
            env.bit_resolution, env.visited_bitmap
        )

        if bx_max >= bx_min and by_max >= by_min:
            patch = env.visited_bitmap[bx_min:bx_max+1, by_min:by_max+1]
            fov_cells = max(1, (bx_max - bx_min + 1) * (by_max - by_min + 1))

            unvisited_ratio_patch = float((~patch).mean()) if patch.size > 0 else 0.0
            if patch.size > 0:
                border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
                frontier_density = float((~border).mean())
            else:
                frontier_density = 0.0

            if getattr(uav, "last_box_idx", None) is not None:
                lbx0, lbx1, lby0, lby1 = uav.last_box_idx
                ix0, iy0 = max(bx_min, lbx0), max(by_min, lby0)
                ix1, iy1 = min(bx_max, lbx1), min(by_max, lby1)
                inter_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1) if (ix1 >= ix0 and iy1 >= iy0) else 0
                overlap_rate_local = inter_cells / float(fov_cells)
            else:
                overlap_rate_local = 0.0
        else:
            unvisited_ratio_patch = 0.0
            frontier_density = 0.0
            overlap_rate_local = 0.0

        # FOV EMA 緩存
        if not hasattr(self, "fov_ema"):
            self.fov_ema = {}
        if uav_id not in self.fov_ema:
            self.fov_ema[uav_id] = dict(overlap=0.0, unvisited=0.0, frontier=0.0)
        self.fov_ema[uav_id]["overlap"]  = ALPHA * self.fov_ema[uav_id]["overlap"]  + (1-ALPHA) * overlap_rate_local
        self.fov_ema[uav_id]["unvisited"]= ALPHA * self.fov_ema[uav_id]["unvisited"]+ (1-ALPHA) * unvisited_ratio_patch
        self.fov_ema[uav_id]["frontier"] = ALPHA * self.fov_ema[uav_id]["frontier"] + (1-ALPHA) * frontier_density
        overlap_ema   = float(self.fov_ema[uav_id]["overlap"])
        unvisited_ema = float(self.fov_ema[uav_id]["unvisited"])
        frontier_ema  = float(self.fov_ema[uav_id]["frontier"])

        # ---------- 通訊向量（含 GS）：用 mask+正規化，取代「延遲=1 代表無效」 ----------
        link_valid_mask           = np.zeros(L, dtype=float)
        link_delay_norm           = np.zeros(L, dtype=float)
        link_capacity_norm        = np.zeros(L, dtype=float)
        next_hop_backlog_log_norm = np.zeros(L, dtype=float)

        # 逐鏈路（UAV->UAV），使用 buffer_info
        for link_info in self.buffer_info.get(uav_id, []):
            nh = int(link_info["next_hop"])
            if nh == uav_id:   # 禁止自連結
                continue
            delay = float(link_info["delay"])
            cap   = float(link_info["channel_capacity"])  # Mbps

            link_valid_mask[nh]    = 1.0
            link_delay_norm[nh]    = min(max(delay, 0.0), D_MAX) / D_MAX
            link_capacity_norm[nh] = min(max(cap,   0.0), C_MAX) / C_MAX

            if nh < N:
                if backlog_bits is None:
                    nh_bits = sum(pkt["rem_bits"] for pkt in self.packet_pool if pkt["current"] == nh and not pkt.get("done", False))
                else:
                    nh_bits = backlog_bits.get(nh, 0)
                    
                next_hop_backlog_log_norm[nh] = np.log1p(float(nh_bits)) / np.log1p(B_MAX)

        # GS（單一節點）
        GS = env.GS_ID  # 請確保 GS 索引固定為 N
        if hasattr(env, "gs_capacity"):
            cap_gs = float(env.gs_capacity[uav_id])
            if cap_gs > 0:
                link_valid_mask[GS]    = 1.0
                link_capacity_norm[GS] = min(cap_gs, C_MAX) / C_MAX
                # 若沒有明確的GS延遲估計，給一個保守上限（不偏好也不懲罰）
                if hasattr(env, "gs_delay"):
                    d_gs = float(env.gs_delay[uav_id])
                else:
                    d_gs = D_MAX
                link_delay_norm[GS] = min(max(d_gs, 0.0), D_MAX) / D_MAX

        # ---------- 幾何/回傳緊迫度 ----------
        # 取得 GS 座標（請依你的環境擇一實作）
        if hasattr(env, "gs_position"):
            gs_x, gs_y, gs_z = env.gs_position
        elif hasattr(env, "GS_POS"):
            gs_x, gs_y, gs_z = env.GS_POS
        else:
            gs_x, gs_y, gs_z = 0.0, 0.0, 0.0  # fallback

        # 2D 距離（通常回傳走水平面）
        dist_to_GS = float(np.hypot(x_u - gs_x, y_u - gs_y))
        Dg_max = float(np.hypot(env.env_width, env.env_height))
        dist_to_GS_norm = min(dist_to_GS, Dg_max) / Dg_max
        #lambda_EE
        lambda_EE = float(getattr(env, "lambda_EE_global", 0.2))
        lambda_norm = min(lambda_EE / 0.3, 1.0)

        # 估 ETA (slots) = 距離 / 速度 / dt
        dt = float(getattr(env, "dt", 1.0))
        speed = float(getattr(uav, "speed", getattr(env, "cruise_speed", 1.0)))  # 請對接你的速度欄位
        eta_slots = dist_to_GS / max(speed, 1e-6) / max(dt, 1e-6)
        eta_to_GS_slots_norm = min(eta_slots, ETA_MAX) / ETA_MAX

        # ---------- UAV 身分 one-hot ----------
        uav_id_one_hot = np.zeros(N, dtype=float)
        uav_id_one_hot[uav_id] = 1.0

        # ---------- 最終 state 串接 ----------
        state = np.concatenate([
            uav_id_one_hot,                                    # N
            np.array([energy_norm, my_bits]),      # 2            
            task_feat,                         # 4

            link_valid_mask,                                   # L
            link_delay_norm,                                   # L
            link_capacity_norm,                                # L
            next_hop_backlog_log_norm,                         # L       => B: 4L

            uav_position,                                      # 3
            np.array([dist_to_GS_norm, eta_to_GS_slots_norm]), # 2       => C: 5
            np.array([lambda_norm], dtype=float),

            np.array([overlap_ema, unvisited_ema, frontier_ema])  # 3     => D: 3
        ], dtype=float)

        # 維度檢查：5N + 19
        expected = 5 * N + 20
        assert state.shape[0] == expected, f"State dim mismatch: got {state.shape[0]}, expect {expected}"
        # self.debug_print_state_v2(uav_id, state, env)
        return state

    def get_state_ta(
        self,
        env,
        uav_id,
        visited_nodes=None,
        backlog_bits=None,
        action_mask=None,
    ):
        """
        State v2 (維度 = 5N + 19, N=num_UAV, L=N+1(含GS)):
        A 個體/任務: [uav_one_hot(N), energy(1), my_backlog(1), task_flags(4), fov_task_flag(1)]
        B 通訊(逐鏈路; 含GS): [link_valid_mask(L), link_delay_norm(L), link_capacity_norm(L), next_hop_backlog_log_norm(L)]
        C 幾何/回傳緊迫度: [uav_position(3), dist_to_GS_norm(1), eta_to_GS_slots_norm(1)]
        D FOV 探索(EMA): [overlap_ema(1), unvisited_ema(1), frontier_ema(1)]
        """

        

        # ---------- 常數/正規化上限（可依環境微調） ----------
        if not hasattr(self, "norm_cfg"):
            self.norm_cfg = dict(
                C_MAX=200.0,   # 容量上限 (Mbps)
                D_MAX=3.0,     # 延遲上限 (s)
                B_MAX=2e6,     # backlog 上限 (bits)
                ETA_MAX=60.0,  # 估計回GS所需slot數的上限
                ema_alpha=0.7  # FOV特徵EMA係數
            )
        C_MAX = float(self.norm_cfg["C_MAX"])
        D_MAX = float(self.norm_cfg["D_MAX"])
        B_MAX = float(self.norm_cfg["B_MAX"])
        ETA_MAX = float(self.norm_cfg["ETA_MAX"])
        ALPHA = float(self.norm_cfg["ema_alpha"])

        # ---------- 基本物件 ----------
        N = env.num_UAV
        L = N + 1                           # 含 GS（GS 索引固定為 env.GS_ID）
        uav = env.uav_dict[uav_id]
        x_u, y_u, z_u = uav.get_position()
        uav_position = np.array([x_u, y_u, z_u], dtype=float)

        # ---------- Altitude features (task-conditioned vertical control) ----------
        # Normalize altitude to [0,1] for stable learning
        z_min = float(getattr(uav, "min_AGL", 50.0))
        z_max = float(getattr(uav, "max_AGL", 200.0))
        z_norm = (float(z_u) - z_min) / max(z_max - z_min, 1e-6)
        z_norm = float(np.clip(z_norm, 0.0, 1.0))

        # Vertical velocity (normalized by dz_cap, which caps per-step vertical motion)
        last = getattr(uav, "last_position", (x_u, y_u, z_u))
        dz = float(z_u - float(last[2]))
        dz_cap_state = float(getattr(env, "dz_cap", 5.0))
        z_vel_norm = float(np.clip(dz / max(dz_cap_state, 1e-6), -1.0, 1.0))

        # Task-dependent target altitude (Search higher, FOV lower)
        if getattr(uav, "task_type", None) == "FOV":
            z_tgt_norm = 0.2
        elif getattr(uav, "task_type", None) == "Search":
            z_tgt_norm = 0.8
        else:
            z_tgt_norm = 0.5
        z_err_norm = float(np.clip(z_norm - z_tgt_norm, -1.0, 1.0))

        # FOV quality features (only meaningful for FOV task; otherwise zeros)
        fov_now_clip = 0.0
        fov_err_clip = 0.0
        try:
            if getattr(uav, "task_type", None) == "FOV" and hasattr(uav, "target_position"):
                tx, ty, tz = uav.target_position
                if not hasattr(self, "_state_fov_model"):
                    self._state_fov_model = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=float(z_u), gamma_g=80)
                else:
                    if hasattr(self._state_fov_model, "z_u"):
                        self._state_fov_model.z_u = float(z_u)
                fov_now, _ = self._state_fov_model.calculate_fov_single(float(x_u), float(y_u), float(z_u), tx, ty, tz)
                fov_now_clip = float(np.clip(fov_now, 0.0, 3.0))
                fov_err_clip = float(np.clip(fov_now - 1.0, -3.0, 3.0))
        except Exception:
            fov_now_clip = 0.0
            fov_err_clip = 0.0
        # ---------- Energy ----------
        energy_remaining = env.uav_dict[uav_id].energy
        energy_norm = float(energy_remaining / env.E_max)
        # ---------- Backlog (自己) ----------
        max_bits_norm = 5e7
        if backlog_bits is None:
            # 兼容舊寫法：若沒傳，才退回掃描（但訓練時我們會傳，避免走到這裡）
            my_bits_raw = sum(
                pkt["rem_bits"] for pkt in self.packet_pool
                if pkt["current"] == uav_id and not pkt.get("done", False)
            )
        else:
            my_bits_raw = backlog_bits.get(uav_id, 0)

        my_bits = min(my_bits_raw / max_bits_norm, 1.0)

        # ---------- 任務旗標 ----------
        task_types = ["Search", "FOV", "COM", "Hovering"]
        task_flags = np.zeros(len(task_types), dtype=float)
        for task in env.multi_tasks.get(uav_id, []):
            if task["task_type"] in task_types:
                task_flags[task_types.index(task["task_type"])] = 1.0
        fov_task_flag = 1.0 if uav.task_type == "FOV" else 0.0  # 原本的 "Current FOV: 0/1"
        # 新增：是否為來源 UAV
        is_source_flag = 1.0 if uav_id in env.source_uavs else 0.0
        # ---------- FOV 三特徵（與你現有計法一致，後續做EMA） ----------
        if not hasattr(self, "FovModel"):
            self.FovModel = FovModel(f=0.004, wl=0.008, i_l=0.012, z_u=uav.z_u, gamma_g=80)
        fov_w, fov_h = self.FovModel.get_ground_fov_size(uav.z_u)

        bx_min, bx_max, by_min, by_max, patch, fov_cells = env.fov_to_indices_and_patch(
            x_u, y_u, fov_w, fov_h,
            env.env_width, env.env_height,
            env.bit_resolution, env.visited_bitmap
        )

        if bx_max >= bx_min and by_max >= by_min:
            patch = env.visited_bitmap[bx_min:bx_max+1, by_min:by_max+1]
            fov_cells = max(1, (bx_max - bx_min + 1) * (by_max - by_min + 1))

            unvisited_ratio_patch = float((~patch).mean()) if patch.size > 0 else 0.0
            if patch.size > 0:
                border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
                frontier_density = float((~border).mean())
            else:
                frontier_density = 0.0

            if getattr(uav, "last_box_idx", None) is not None:
                lbx0, lbx1, lby0, lby1 = uav.last_box_idx
                ix0, iy0 = max(bx_min, lbx0), max(by_min, lby0)
                ix1, iy1 = min(bx_max, lbx1), min(by_max, lby1)
                inter_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1) if (ix1 >= ix0 and iy1 >= iy0) else 0
                overlap_rate_local = inter_cells / float(fov_cells)
            else:
                overlap_rate_local = 0.0
        else:
            unvisited_ratio_patch = 0.0
            frontier_density = 0.0
            overlap_rate_local = 0.0

        # FOV EMA 緩存
        if not hasattr(self, "fov_ema"):
            self.fov_ema = {}
        if uav_id not in self.fov_ema:
            self.fov_ema[uav_id] = dict(overlap=0.0, unvisited=0.0, frontier=0.0)
        self.fov_ema[uav_id]["overlap"]  = ALPHA * self.fov_ema[uav_id]["overlap"]  + (1-ALPHA) * overlap_rate_local
        self.fov_ema[uav_id]["unvisited"]= ALPHA * self.fov_ema[uav_id]["unvisited"]+ (1-ALPHA) * unvisited_ratio_patch
        self.fov_ema[uav_id]["frontier"] = ALPHA * self.fov_ema[uav_id]["frontier"] + (1-ALPHA) * frontier_density
        overlap_ema   = float(self.fov_ema[uav_id]["overlap"])
        unvisited_ema = float(self.fov_ema[uav_id]["unvisited"])
        frontier_ema  = float(self.fov_ema[uav_id]["frontier"])

        # ---------- 通訊向量（含 GS）：用 mask+正規化，取代「延遲=1 代表無效」 ----------
        link_valid_mask           = np.zeros(L, dtype=float)
        link_delay_norm           = np.zeros(L, dtype=float)
        link_capacity_norm        = np.zeros(L, dtype=float)
        next_hop_backlog_log_norm = np.zeros(L, dtype=float)
        next_hop_is_fov = np.zeros(L, dtype=float)
        effective_mask = self.get_effective_action_mask(
            env, uav_id, action_mask
        )
        link_valid_mask[:] = effective_mask.astype(float)

        # Nominal full-pool qualities describe candidate links. Slot-specific
        # allocated capacities are deliberately kept out of the next state.
        for nh in range(N):
            if nh == uav_id:
                continue
            cap = float(env.Capacity_matrix[uav_id, nh])
            link_capacity_norm[nh] = min(max(cap, 0.0), C_MAX) / C_MAX
            if backlog_bits is None:
                nh_bits = self.backlog_bits.get(nh, 0.0)
            else:
                nh_bits = backlog_bits.get(nh, 0.0)
            next_hop_backlog_log_norm[nh] = (
                np.log1p(float(nh_bits)) / np.log1p(B_MAX)
            )
            is_fov = any(
                task["task_type"] == "FOV"
                for task in env.multi_tasks.get(nh, [])
            )
            next_hop_is_fov[nh] = 1.0 if is_fov else 0.0

        GS = env.GS_ID
        if hasattr(env, "gs_capacity"):
            cap_gs = float(env.gs_capacity[uav_id])
            link_capacity_norm[GS] = min(max(cap_gs, 0.0), C_MAX) / C_MAX
            if cap_gs > 0.0:
                d_gs = (
                    float(env.gs_delay[uav_id])
                    if hasattr(env, "gs_delay")
                    else D_MAX
                )
                link_delay_norm[GS] = min(max(d_gs, 0.0), D_MAX) / D_MAX

        # ---------- 幾何/回傳緊迫度 ----------
        # 取得 GS 座標（請依你的環境擇一實作）
        if hasattr(env, "gs_position"):
            gs_x, gs_y, gs_z = env.gs_position
        elif hasattr(env, "GS_POS"):
            gs_x, gs_y, gs_z = env.GS_POS
        else:
            gs_x, gs_y, gs_z = 0.0, 0.0, 0.0  # fallback

        # 2D 距離（通常回傳走水平面）
        dist_to_GS = float(np.hypot(x_u - gs_x, y_u - gs_y))
        Dg_max = float(np.hypot(env.env_width, env.env_height))
        dist_to_GS_norm = min(dist_to_GS, Dg_max) / Dg_max

        # 估 ETA (slots) = 距離 / 速度 / dt
        dt = float(getattr(env, "dt", 1))
        speed = float(getattr(uav, "speed", getattr(env, "cruise_speed", 1.0)))  # 請對接你的速度欄位
        eta_slots = dist_to_GS / max(speed, 1e-6) / max(dt, 1e-6)
        eta_to_GS_slots_norm = min(eta_slots, ETA_MAX) / ETA_MAX
        #lambda_EE
        lambda_EE = float(getattr(env, "lambda_EE_global", 0.2))
        lambda_norm = min(lambda_EE / 0.3, 1.0)

        # ---------- UAV 身分 one-hot ----------
        uav_id_one_hot = np.zeros(N, dtype=float)
        uav_id_one_hot[uav_id] = 1.0

        hol = self.get_hol_packet(uav_id)
        hol_context = np.zeros(4, dtype=float)
        if hol is not None:
            hol_task = self._task_norm(hol.get("task_type", "COM"))
            hol_context[0] = 1.0 if hol_task == "FOV" else 0.0
            hol_context[1] = 1.0 if hol_task == "COM" else 0.0
            deadline_abs = hol.get("deadline_abs")
            if deadline_abs is not None:
                deadline_window = TASK_DEADLINE_SECONDS[hol_task]
                hol_context[2] = np.clip(
                    (float(deadline_abs) - float(getattr(env, "current_time", 0.0)))
                    / deadline_window,
                    0.0,
                    1.0,
                )
            hol_context[3] = np.clip(
                float(hol.get("rem_bits", 0.0))
                / max(float(hol.get("size_bits", 0.0)), PACKET_EPS),
                0.0,
                1.0,
            )

        # ---------- 最終 state 串接 ----------
        state = np.concatenate([
            uav_id_one_hot,                                    # N
            np.array([energy_norm, my_bits]),      # 2
            task_flags,                                        # 4
            np.array([fov_task_flag, is_source_flag]),         # 2       => A: N+7

            link_valid_mask,                                   # L
            link_delay_norm,                                   # L
            link_capacity_norm,                                # L
            next_hop_backlog_log_norm,                         # L       => B: 4L
            next_hop_is_fov,                                   # L

            np.array([x_u, y_u, z_norm], dtype=float),             # 3 (z normalized)
            np.array([z_err_norm, z_vel_norm], dtype=float),      # 2 (altitude error & vertical speed)
            np.array([fov_now_clip, fov_err_clip], dtype=float),  # 2 (FOV quality)

            np.array([dist_to_GS_norm, eta_to_GS_slots_norm]), # 2       => C: 5
            np.array([lambda_norm], dtype=float),

            np.array([overlap_ema, unvisited_ema, frontier_ema]), # 3     => D: 3
            hol_context,                                        # 4
        ], dtype=float)

        expected = 6 * N + 30
        assert state.shape[0] == expected, f"State dim mismatch: got {state.shape[0]}, expect {expected}"
        # self.debug_print_state_v2(uav_id, state, env)
        return state
    # ===== 放在原類別內（與原函式並存），或直接取代 =====
    def calculate_packet_reward_fast(
    self, env, pkt, hop_delay_ms, from_uav, to_target, t, backlog, mode="uav",
    channel_capacity=None
    ):
        # -----------------------
        # Constants / thresholds
        # -----------------------
        cap_eps = 0.1      # Mbps, link too weak threshold
        f_c = 2e9
        d_0 = 1
        sigma_sq = -169    # dBm/Hz

        # -----------------------
        # Basic fields
        # -----------------------
        task_type = pkt.get("task_type", "UNKNOWN")
        GS_ID = env.GS_ID
        is_gs = (to_target == GS_ID)

        # bandwidth
        if is_gs:
            B = float(env.B_tot)
        else:
            B = float(env.B_eff_u2u[from_uav]) if hasattr(env, "B_eff_u2u") else float(env.B_tot)

        # -----------------------
        # Channel capacity (Mbps)
        # -----------------------
        if channel_capacity is None:
            if is_gs and (getattr(env, "gs_capacity", None) is not None):
                channel_capacity = float(env.gs_capacity[from_uav])
            elif (not is_gs) and (getattr(env, "Capacity_matrix", None) is not None):
                channel_capacity = float(env.Capacity_matrix[from_uav, to_target])
            else:
                channel_capacity = 0.0

        if channel_capacity < cap_eps:
            # 注意：你的 e2e 欄位是 e2e_delay_ms
            return task_type, 0.0, False, float(pkt.get("e2e_delay_ms", 0.0)), False, 0.0, 0.0, 0.0, False

        # -----------------------
        # Current accumulated E2E (ms)
        # -----------------------
        pkt_e2e_ms = float(pkt.get("e2e_delay_ms", 0.0))

        # -----------------------
        # Path-loss (prefer env cache)
        # -----------------------
        uav_from = env.uav_dict[from_uav]
        PL = None
        if is_gs and hasattr(env, "PL_ug_cache"):
            try:
                PL = float(env.PL_ug_cache[from_uav])
            except Exception:
                PL = None
        elif (not is_gs) and hasattr(env, "PL_uu_cache"):
            try:
                PL = float(env.PL_uu_cache[from_uav, to_target])
            except Exception:
                PL = None

        if PL is None:
            # fallback geometry
            if is_gs:
                gx, gy, gz = env.GS_pos
                dx = uav_from.x_u - gx
                dy = uav_from.y_u - gy
                dz = uav_from.z_u - gz
                d_3D = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
                ratio = uav_from.z_u / d_3D
                ratio = max(-1.0, min(1.0, ratio))
                angle = math.degrees(math.asin(ratio))

                LOSPL  = ChannelModel.PL_ug(d_3D, d_0, f_c=f_c, mu=2.0)
                NLOSPL = ChannelModel.PL_ug(d_3D, d_0, f_c=f_c, mu=2.4)
                Los_prob = 1.0 / (1.0 + 4.88 * math.exp(-0.429 * (angle - 4.88)))
                PL = Los_prob * LOSPL + (1.0 - Los_prob) * NLOSPL
            else:
                uav_to = env.uav_dict[to_target]
                dx = uav_from.x_u - uav_to.x_u
                dy = uav_from.y_u - uav_to.y_u
                dz = uav_from.z_u - uav_to.z_u
                d_3D = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
                H_u = abs(dz)
                PL = ChannelModel.PL_uu(H_u=H_u, d_3D=d_3D, f_c=2.4)

        # -----------------------
        # Transmission (store-and-forward, bits mean "remaining-to-GS")
        # -----------------------
        dt = float(self.step_time)

        size_bits = float(pkt.get("size_bits", pkt.get("bits", 0.0)))
        rem_bits  = float(pkt.get("rem_bits", size_bits))

        cap_bits_step = float(channel_capacity) * 1e6 * dt
        bits_tx_used  = min(cap_bits_step, rem_bits)
        # --- Update bottleneck (bn_path_mbps) per hop ---
        if bits_tx_used > 0:
            cap_mbps = float(channel_capacity)
            if not np.isfinite(cap_mbps):
                cap_mbps = 0.0
            prev_bn = float(pkt.get("bn_path_mbps", float("inf")))
            pkt["bn_path_mbps"] = min(prev_bn, cap_mbps)
        # ----------------------------------------------
        # 扣「本 hop」剩餘
        rem_bits = max(0.0, rem_bits - bits_tx_used)
        pkt["rem_bits"] = rem_bits
        moved = False
        if rem_bits <= 1e-9 and bits_tx_used > 0:
            # 這一跳傳完了，才允許 forward
            pkt["current"] = to_target
            pkt["hops"] = int(pkt.get("hops", 0)) + 1
            pkt.setdefault("path", []).append(to_target)
            moved = True

            if to_target == env.GS_ID:
                pkt["done"] = True
                pkt["reason"] = "GS"
                pkt["bn_final_mbps"] = float(pkt.get("bn_path_mbps", 0.0))  
                # print("[SET DONE] id=", id(pkt), "done=", pkt.get("done"), "reason=", pkt.get("reason"))
                self.mark_packet_done(pkt, current_time=t, reason="delivered")
                # print(f"[t={t}] 📦 Delivered | task={pkt.get('task_type')} | e2e={pkt.get('e2e_delay_ms', 0.0):.2f} ms | hops={pkt.get('hops',0)}")
            else:
                # 到了下一個中繼點，下一跳要再傳一次完整封包
                pkt["rem_bits"] = size_bits


        # 從 from_uav 的 queue 扣掉本 step 真的送出去的 bits
        if bits_tx_used > 0:
            self.backlog_bits[from_uav] = max(0.0, self.backlog_bits[from_uav] - float(bits_tx_used))

        # 若 hop 完成且 forward 到下一個 UAV，下一個 UAV 的 queue 要「新增一整包 size_bits」
        if moved and (to_target != env.GS_ID):
            self.backlog_bits[to_target] += float(size_bits)
        # -----------------------
        # Energy (comm)
        # -----------------------
        # data_rate_bps_eff = bits_tx_used / max(dt, 1e-9)  # effective bps
        # E_comm = env.energy_model.compute_comm_energy(
        #     uav_idx=from_uav, PL_dB=PL, data_rate=data_rate_bps_eff,
        #     sigma_sq=sigma_sq, B=B, t=dt
        # )
        # if E_comm < 1e-9:
        #     E_comm = 1e-9
        

        # Update UAV energy
        # uav_from.last_energy = uav_from.energy
        # uav_from.energy = max(uav_from.energy - E_comm, 0.0)
        # uav_from.update_battery(uav_from.energy, env.E_max)

        # -----------------------
        # Cost: violation probability (use ms thresholds!)
        # -----------------------
        
        # (A) progress-to-GS shaping (small, stable)
        gx, gy, gz = getattr(env, "GS_pos", (0.0, 0.0, 0.0))
        prev_d = math.hypot(uav_from.x_u - gx, uav_from.y_u - gy)
        if is_gs:
            new_d = 0.0
        else:
            uav_to = env.uav_dict[to_target]
            new_d = math.hypot(uav_to.x_u - gx, uav_to.y_u - gy)

        # kappa = 5e-3
        progress = (prev_d - new_d)  # >0 means closer to GS

        # (B) congestion penalty: prefer next hop with smaller backlog
        # use env-side backlog estimate (already maintained)
        nh_backlog = float(self.backlog_bits.get(to_target, 0.0)) if (to_target != env.GS_ID) else 0.0
        # log makes it smooth and scale-invariant
        cong_pen = math.log1p(nh_backlog / 1e5)

        # (C) deadline-risk penalty: if we are close to tau, penalize extra hop delay more
        pkt_e2e_ms = float(pkt.get("e2e_delay_ms", 0.0))
        if task_type == "FOV":
            tau_ms = float(getattr(env, "tau_FOV_ms", getattr(env, "tau_D_ms", 15.0)))
        else:
            tau_ms = float(getattr(env, "tau_COM_ms", getattr(env, "tau_D_ms", 10.0)))

        # risk in [0,1], becomes large when e2e approaches tau
        risk = min(1.0, max(0.0, pkt_e2e_ms / max(tau_ms, 1e-6)))

        # ---- final dense reward (stable) ----
        # weights: start small to avoid exploding TD-targets
        w_prog = 0.02
        w_cong = 0.01
        w_risk = 0.02

        reward_step = (w_prog * progress) - (w_cong * cong_pen) - (w_risk * risk * hop_delay_ms)
        # -----------------------
        # Cost: per-packet violation (stationary!)
        # -----------------------
        violated = False
        cost = 0.0
        pkt_done = bool(pkt.get("done", False))

        if pkt_done:
            
            e2e_delay_ms = pkt_e2e_ms
            violated = (e2e_delay_ms > tau_ms)
            cost = 1.0 if violated else 0.0

            # terminal bonus for delivering to GS (stable, not too sparse)
            if to_target == env.GS_ID:
                reward_step += 1.0  # you can tune 0.5~2.0
                

            return task_type, reward_step, True, e2e_delay_ms, violated, cost, bits_tx_used, None, moved
        
        return task_type, reward_step, False, pkt_e2e_ms, False, 0.0, bits_tx_used, None, moved

    def calculate_packet_reward_fast_Dinkel(
        self, env, pkt, hop_delay_ms, from_uav, to_target, t,
        backlog, mode="uav", channel_capacity=None
    ):
        # 1) 先跑 baseline：確保 rem_bits/backlog_bits/moved/done/violated 全部一致
        task_type, r_base, pkt_done, e2e_ms, violated, cost, bits_tx_used, E_comm, moved = \
            self.calculate_packet_reward_fast(
                env, pkt, hop_delay_ms,
                from_uav=from_uav,
                to_target=to_target,
                t=t,
                backlog=backlog,
                mode=mode,
                channel_capacity=channel_capacity
            )

        # 2) Dinkelbach 的 lambda（由你的 training 週期性更新）
        lam = float(getattr(env, "lambda_EE_global", 0.0))

        # 3) 能量項要做尺度化（不然 reward 會被能量項淹沒）
        #    你可以在 env 設 env.E_COMM_SCALE，例如 10 或 100，視 E_comm 量級而定
        E_SCALE = float(getattr(env, "E_COMM_SCALE", 10.0))
        E_pen = float(E_comm) / max(E_SCALE, 1e-9)

        # 4) ✅ Dinkel reward：R - λE
        r = float(r_base) - lam * E_pen

        return task_type, r, pkt_done, e2e_ms, violated, cost, bits_tx_used, E_comm, moved
    def Rand_selected_test(self, uav_id: int, mask=None, visited_nodes=None):
        """
        回傳 action = next_hop id (int)，介面完全對齊 DDQN.select_action()
        state 只是為了對齊介面，Rand 不用它。
        """
        action_dim = self.num_UAV + 1  # UAV(0..num_uav-1) + GS(num_uav or gs_id 依你定義)

        # 1) 建 mask
        if mask is None:
            final_mask = np.ones(action_dim, dtype=bool)
        else:
            final_mask = np.asarray(mask, dtype=bool)
        
        # 2) 排除自己
        if 0 <= uav_id < action_dim:
            final_mask[uav_id] = False

        # 3) 排除 visited
        if visited_nodes is not None:
            for node in visited_nodes:
                if 0 <= node < action_dim:
                    final_mask[node] = False

        # 4) 確保至少一個可選
        if not final_mask.any():
            final_mask[:] = True
            if 0 <= uav_id < action_dim:
                final_mask[uav_id] = False

        avail = np.flatnonzero(final_mask)
        return int(np.random.choice(avail))
