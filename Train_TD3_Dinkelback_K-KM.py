import numpy as np
from Simulator import Simulator
from Packet_scheduler_v1 import PacketEngine
import matplotlib.pyplot as plt
from td3 import TD3
from DDQN import DDQN
import matplotlib.pyplot as plt
import os
import utils_update_v2  
import pandas as pd
from collections import defaultdict
import time
import json

def train():
    print("✅ 累狗!")
    env = Simulator(num_UAV=16)
    rout = PacketEngine(num_uav=16, step_time=0.25)
    num_uav = env.num_UAV
    state_dim = 100
    routing_dim = num_uav + 1
    moving_dim = 3
    max_action = 1
    Model_TD3_search = TD3(state_dim, moving_dim, max_action )
    Model_TD3_fov = TD3(state_dim, moving_dim, max_action )
    Model_DDQN = DDQN(state_dim, routing_dim)
    routing_buffer = utils_update_v2.ReplayBufferDiscrete(state_dim, action_dim=routing_dim, max_size=int(2e5), n_step=3, gamma=0.99)
    movement_buffer_search = utils_update_v2.ReplayBufferContinuous(state_dim, action_dim=3, max_size=int(2e5), n_step=3, gamma=0.99)
    movement_buffer_fov = utils_update_v2.ReplayBufferContinuous(state_dim, action_dim=3, max_size=int(2e5), n_step=3, gamma=0.99)
    # replay_buffer = utils.ReplayBufferNStep(state_dim, action_dim=4, max_size=int(1e6), n_step=3, gamma=0.99)
    T_sec = 60
    total_episodes = 6
    episode_times = []
    warmup_episodes = 0
    step_time = 0.25
    T_slot = int(T_sec / step_time)
    delay_bound_steps = int(5.0 / step_time)
    violation_rate_log = {
        "FOV": [],
        "COM": []
    }
    pkt_loss_log={
        "loss_bits":[],
        "pkt_count":[]
    }
    routing_reward_log = []   
    search_reward_log = []  
    fov_reward_log = []
    reward_log= []
    fov_log = []
    next_hop_by_uav = {} 
    lambda_EE_global = 0.1

    # ===== Checkpoint / Resume =====
    ckpt_root = "checkpoints_K-KM_td3_dinkel"
    CKPT_EVERY = 2  # save every N episodes
    resume_dir = None #"checkpoints_td3_dinkel/ep_0800"  # e.g., "checkpoints_ddpg_dinkel/ep_0200" to resume from ep=200
    # print(resume_dir)
    start_episode = 0
    if resume_dir is not None and os.path.isdir(resume_dir):
        meta_path = os.path.join(resume_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            lambda_EE_global = float(meta.get("lambda_EE_global", lambda_EE_global))
            start_episode = int(meta.get("episode", -1)) + 1

        # load model weights (use your existing .load() API)
        Model_TD3_search.load(os.path.join(resume_dir, "uav_search"))
        Model_TD3_fov.load(os.path.join(resume_dir, "uav_fov"))
        Model_DDQN.load(os.path.join(resume_dir, "uav_ddqn"))

        print(f"🔁 Resume from {resume_dir} | start_episode={start_episode} | lambda={lambda_EE_global:.4f}")
    else:
        os.makedirs(ckpt_root, exist_ok=True)
        # env.set_uav_tasks(uav_task_list)
    for episode in range(start_episode, total_episodes):
        start = time.perf_counter()
        env.debug_reward = True
        GT_min, GT_max = 2, 10
        if episode < warmup_episodes:
            env.num_GT = 4
        else:
            env.num_GT = np.random.randint(GT_min, GT_max)
        # env.num_GT = 10
        num_gt = env.num_GT
        fov_accum, fov_count = 0, 0  # 累積所有 time slot 的平均 FOV
        violation_stats = {
            "FOV": {"delivered": 0, "violated": 0},
            "COM": {"delivered": 0, "violated": 0},
            "Search": {"delivered": 0, "violated": 0}
        }
        pkt_stats = {
            "loss_bits": 0,
            "pkt_count": 0
        }
        # env.assign_tasks()
        previous_lambda_EE = lambda_EE_global
        lambda_EE = lambda_EE_global
        env.reset_environment()
        rout.reset_packet_state()
        env.lambda_EE_global = float(lambda_EE_global)
        routing_mask_cache = {}
        env._pending_search_done = False
        env._search_phase_over = False
        state_cache = {}
        # total_bits = 0
        total_energy = 0
        episode_total_reward= 0
        episode_routing_reward=0
        episode_search_reward = 0
        episode_fov_reward = 0
        fov_accum, fov_count = 0, 0  # 累積所有 time slot 的平均 FOV
        total_bn_rate_bps=0.0
        SEARCH_COVERAGE_TH = 0.99
        EPS = 1e-4
        for t in range(T_slot):
            slot_r = defaultdict(float)
            slot_c = defaultdict(float)
            env.begin_step()   
            env.current_time = t * step_time
            rout.drop_expired_packets(env.current_time)
            rout.inject_packets(env, delay_bound_steps, env.current_time, step_time)
            active_pkts_now = rout.get_active_packets()
            backlog_bits = defaultdict(float, rout.backlog_bits)
            uavs_with_pkts = [u for u, b in backlog_bits.items() if b > 0]
            uavs_with_pkts = [uid for uid in uavs_with_pkts if uid != env.GS_ID]
            active_uav_ids = list(range(env.num_UAV)) 
            # active_uav_ids = uavs_with_pkts  # 只算有封包的
            if (t == 0) or (t % 4 == 0):
                state_cache = {uid: rout.get_state(env, uid, backlog_bits=backlog_bits) for uid in active_uav_ids}
            if getattr(env, "need_reassign", False):
                env.assign_tasks()
                env.need_reassign = False
            if t% 4 == 0:
                for uav_id in active_uav_ids:
                    task_list = env.multi_tasks.get(uav_id, [])
                    if not task_list:
                        continue
                    for task in task_list:
                        # print(task)
                        task_type = task["task_type"]
                        # print(f"task_type={task_type}")
                        if  task_type not in ["FOV", "Search"]:
                            continue
                        
                        # 如果已經搜完，跳過所有 Search 任務，讓其他任務繼續
                        if task_type == "Search" and getattr(env, "_search_phase_over", False):
                            continue
                        # 執行目前輪到的任務
                        uav = env.uav_dict[uav_id]
                        uav.task_type = task_type
                        uav.target_position = task["target_pos"]
                        uav.assigned_target_id = task["target_id"]
                        #  取得狀態並讓 actor 選擇動作
                        state = rout.get_state(env, uav_id, backlog_bits=backlog_bits)
                        if task_type == "Search":
                            raw_action = Model_TD3_search.select_action(state, uav_id)
                            movement_action = Model_TD3_search.decode_action(raw_action)
                            dx, dy, dz = movement_action
                            E_mob =uav.apply_movement(dx, dy, dz, energy_model=env.energy_model, step_time=1.0)
                            env.update_visited_grid(uav_id)
                            search_reward, _, current_fov = env.calculate_search_reward(uav_id, lambda_EE, E_mob)
                            episode_total_reward += search_reward
                            episode_search_reward += search_reward
                            total_energy += E_mob
                            next_state =  rout.get_state(env, uav_id, backlog_bits=backlog_bits)
                            global_cov = float(env.visited_bitmap.mean())
                            search_done = (global_cov >= SEARCH_COVERAGE_TH- EPS) 
                            movement_buffer_search.add(state, raw_action, next_state, search_reward, done=search_done, tag_gt=env.num_GT)
                            env._pending_search_done |= bool(search_done)
                        elif task_type == "FOV":
                            raw_action = Model_TD3_fov.select_action(state, episode=episode)
                            movement_action = Model_TD3_fov.decode_action(raw_action)
                            dx, dy, dz = movement_action
                            E_mob = uav.apply_movement(dx, dy, dz, energy_model=env.energy_model, step_time=1.0)
                            fov_reward, _, current_fov = env.calculate_fov_reward(uav_id, lambda_EE, E_mob)
                            episode_total_reward += fov_reward
                            episode_fov_reward += fov_reward
                            total_energy += E_mob
                            next_state =  rout.get_state(env, uav_id, backlog_bits=backlog_bits)
                            movement_buffer_fov.add(state, raw_action, next_state, fov_reward, done=False, tag_gt=env.num_GT)
                            fov_accum += current_fov
                            fov_count += 1
                        elif task_type == "Hovering":
                            dx, dy, dz =0, 0, 0
                            E_hover = uav.apply_movement(dx, dy, dz, energy_model=env.energy_model)
                            total_energy += E_hover
                    # 🔁 輪替下一個任務
                    uav.active_task_index = (uav.active_task_index + 1) % max(1, len(task_list))
            # ======= 這裡是「step 末端」統一轉場與清除 Search 任務 =======
            if getattr(env, "_pending_search_done", False) and not getattr(env, "_search_phase_over", False):
                env._search_phase_over = True   # 內部 guard，不暴露給 agent
                cov = float(env.visited_bitmap.mean())
                print(f"Search done in env: cov={cov:.3f}, found={env.count_found_targets()}")
                env.convert_search_to_hovering()   # ←← 在這裡清掉所有 Search 任務
            # 清掉暫存旗標，進下一步
            env.update_source_uavs()
            # start = time.perf_counter()
            next_hop_by_uav.clear()
            routing_mask_cache.clear()
            # if env.source_uavs:
            if t % 4 == 0:   # 每 1 秒更新一次 (0.25 × 4)
                env.update_u2u_channels()
                env.update_u2g_channels()
                # if t % 20 == 0:  # 低頻印
                #     gs_reachable = np.where(env.gs_capacity > 0.1)[0] if env.gs_capacity is not None else []
                #     print(f"[t={t}] GS-reachable UAVs: {len(gs_reachable)}/{env.num_UAV}  ids={gs_reachable[:10]}")
                #     print("GS xy =", (env.x, env.y), "UAV0 xy =", (env.UAVs[0].x_u, env.UAVs[0].y_u))
                #     print("d2d max =", np.max(np.linalg.norm(np.array([[u.x_u,u.y_u] for u in env.UAVs]) - np.array([env.x,env.y]), axis=1)))

                cap_ok = (env.Capacity_matrix > 0.1)
                np.fill_diagonal(cap_ok, False)
                gs_ok = (env.gs_capacity is not None) & (env.gs_capacity > 0.1)

                for uid in active_uav_ids:
                    m = np.zeros(num_uav + 1, dtype=bool)
                    m[:num_uav] = cap_ok[uid]
                    m[env.GS_ID] = bool(gs_ok[uid]) if env.gs_capacity is not None else False
                    routing_mask_cache[uid] = m

            # ===== (A) 每台 UAV、每種type 只做一次 DDQN 推論 =====
            for uid in uavs_with_pkts:
                state_u = state_cache[uid]
                mask_u = routing_mask_cache.get(uid, None)
                if mask_u is None:
                    mask_u = np.ones(num_uav + 1, dtype=bool)
                    mask_u[uid] = False

                a = Model_DDQN.select_action(state_u, uid, mask_u, visited_nodes=None)
                next_hop_by_uav[uid] = int(a)
            
            for pkt in active_pkts_now:
                if pkt.get("done", False):
                    continue

                u = pkt["current"]
                if u == env.GS_ID:
                    continue
                next_hop = next_hop_by_uav.get(u, env.GS_ID)

                # --- channel capacity ---
                if next_hop == env.GS_ID:
                    cap_mbps = float(env.gs_capacity[u]) if env.gs_capacity is not None else 0.0
                else:
                    cap_mbps = float(env.Capacity_matrix[u, next_hop])

                if cap_mbps <= 0:
                    continue

                # --- queue delay：用 node backlog 近似 ---
                my_bits = float(pkt.get("rem_bits", pkt.get("size_bits", 0.0)))
                node_backlog = float(rout.backlog_bits.get(u, 0.0))
                queue_bits_wo_self = max(node_backlog - my_bits, 0.0)

                # ====== Method A: backlog 平均分到可用出口數 ======
                if next_hop == env.GS_ID:
                    k_out = 1  # 走 GS 當作單一路徑（先別分）
                else:
                    # env.k_u_u2u 是 update_u2u_channels() 裡算的 feasible 鄰居數
                    k_out = int(getattr(env, "k_u_u2u", np.ones(env.num_UAV))[u]) if hasattr(env, "k_u_u2u") else 1
                k_out = max(k_out, 1)

                queue_bits_eff = queue_bits_wo_self / k_out

                # --- 本 hop 預計可送 bits ---
                cap_bits_step = cap_mbps * 1e6 * step_time
                bits_used_pred = min(cap_bits_step, my_bits)

                # --- 記錄 hop delay ---
                hop_delay_ms = rout.log_hop_delay(
                    env, pkt,
                    current_node=u,
                    next_hop=next_hop,
                    link_capacity_mbps=cap_mbps,
                    current_time=env.current_time,
                    pkt_bits=bits_used_pred,
                    backlog_bits=queue_bits_eff
                )

                # --- 更新封包 / reward / backlog ---
                task_type, route_reward, pkt_done, _, violated, cost, bits_used, E_comm, _ = \
                    rout.calculate_packet_reward_fast(
                        env, pkt, hop_delay_ms,
                        from_uav=u,
                        to_target=next_hop,
                        t=env.current_time,
                        backlog=queue_bits_wo_self,
                        mode="uav",
                        channel_capacity=cap_mbps
                    )
                
                bits_used = float(bits_used)

                if pkt.get("done", False) and (not pkt.get("bn_counted", False)) and (pkt.get("bn_final_mbps", None) is not None):
                    total_bn_rate_bps += float(pkt["bn_final_mbps"]) * 1e6
                    pkt["bn_counted"] = True

                # total_energy += E_comm

                # slot reward / cost
                scaled_route_reward = route_reward / max(num_gt, 1)
                episode_routing_reward += scaled_route_reward / 10
                episode_total_reward += scaled_route_reward / 10

                slot_r[u] += float(route_reward)
                slot_c[u] += float(cost)

                if pkt_done and task_type in violation_stats:
                    vs = violation_stats[task_type]
                    vs["delivered"] += 1
                    if violated:
                        vs["violated"] += 1
            active_pkts_after = rout.get_active_packets()
            backlog_bits_after = defaultdict(float)
            for p in active_pkts_after:
                if p.get("done", False):
                    continue
                uu = p["current"]
                backlog_bits_after[uu] += float(p.get("rem_bits", p.get("size_bits", 0.0)))

            ns_cache = {
                uid: rout.get_state(env, uid, backlog_bits=backlog_bits_after)
                for uid in uavs_with_pkts
            }

            # ====== add one transition per UAV per slot ======
            add_buf = routing_buffer.add
            for uid in uavs_with_pkts:
                state = state_cache[uid]
                action = int(next_hop_by_uav.get(uid, uid))   
                next_state = ns_cache.get(uid, state)

                reward = float(slot_r[uid])  
                cost   = float(slot_c[uid])  

                done_flag = False  
                add_buf(state, action, next_state, reward, cost, done_flag, tag_gt=env.num_GT)
                
        # ===== Dinkelbach Method: 更新 λ_EE =====
        WARMUP_LAMBDA = 200
        K_UPDATE = 50
        BETA_LAMBDA = 0.01

        # 你原本 lambda clip
        LAMBDA_MIN, LAMBDA_MAX = 0.0, 0.3

        # ratio clip (放寬一點，避免太早打頂)
        RATIO_MIN, RATIO_MAX = 0.0, 0.3

        #  新增：尺度放大（先用 100，讓 0.001 → 0.1）
        SCALE_LAMBDA = 400.0

        if (episode >= WARMUP_LAMBDA) and (episode % K_UPDATE == 0) and (total_energy > 1e-6):

            ratio_raw = (total_bn_rate_bps / 1e9) / total_energy     # 原始 EE proxy
            ratio = SCALE_LAMBDA * ratio_raw                           # 放大讓 TD3 感受到能量成本
            ratio = float(np.clip(ratio, RATIO_MIN, RATIO_MAX))

            lambda_EE_global = (1 - BETA_LAMBDA) * lambda_EE_global + BETA_LAMBDA * ratio
            lambda_EE_global = float(np.clip(lambda_EE_global, LAMBDA_MIN, LAMBDA_MAX))
            env.lambda_EE_global = float(lambda_EE_global) 
            lambda_EE = lambda_EE_global

            print(f"[λ-update] ep={episode} "
                f"ratio_raw={ratio_raw*1e3:.6f} Mbits/J"
                f"ratio_scaled={ratio:.4f} "
                f"lambda={lambda_EE_global:.4f} "
                f"energy={total_energy:.2f} "
                f"bnGbps={total_bn_rate_bps/1e9:.4f}")
        # ==============蒐集迭代後資訊========================
        # === Episode 結束計時 ===
        end = time.perf_counter()
        episode_duration = end - start
        episode_times.append(episode_duration)
        
        avg_time = np.mean(episode_times)
        eta = avg_time * (total_episodes - (episode + 1))
        
        print(f"[Episode {episode+1}/{total_episodes}] 用時: {episode_duration:.2f} 秒, 預估剩餘: {eta/60:.1f} 分鐘")
        search_reward_log.append(episode_search_reward)
        fov_reward_log.append(episode_fov_reward)
        routing_reward_log.append(episode_routing_reward)
        reward_log.append(episode_total_reward)
        avg_fov_episode = (fov_accum / fov_count) if fov_count > 0 else 0.0
        fov_log.append(avg_fov_episode)
        

        for task in ["FOV", "COM"]:
            delivered = violation_stats[task]["delivered"]
            violated = violation_stats[task]["violated"]
            rate = violated / delivered if delivered > 0 else 0
            # print(f"[Episode {episode+1}] {task} violation rate = {rate:.3f}")
            violation_rate_log[task].append((episode+1, rate))
        
        pkt_loss_log["pkt_count"].append(pkt_stats["pkt_count"])
        pkt_loss_log["loss_bits"].append(pkt_stats["loss_bits"])
    
        # 前 400–800 集保守探索，保持敏感度。等策略初步穩定、loss 有下降後，再提升 batch size 穩定訓練。
        # if episode < 600:
        #     current_batch_size = 64
        # else:
        current_batch_size = 64
        if episode >= warmup_episodes:
            if movement_buffer_search.size > current_batch_size:
                Model_TD3_search.update(movement_buffer_search, env.num_GT, current_batch_size)
            if movement_buffer_fov.size > current_batch_size:
                Model_TD3_fov.update(movement_buffer_fov, env.num_GT, current_batch_size)
            if routing_buffer.size > current_batch_size:
                Model_DDQN.train(routing_buffer, current_batch_size)
        target_update_freq=20
        if episode % target_update_freq == 0:
            Model_DDQN.update_target()
        # print(uav_path_log)
        print(f"[Episode {episode+1}] Routing_reward: {episode_routing_reward:.2f} ,Search_reward: {episode_search_reward:.2f}, Fov_reward: {episode_fov_reward:.2f}")
        print(f"Average FOV: {avg_fov_episode:.3f}, lambda為:{lambda_EE:.3f}")

        # ===== Save checkpoint =====
        if ((episode + 1) % CKPT_EVERY) == 0:
            save_dir = os.path.join(ckpt_root, f"ep_{episode+1:04d}")
            os.makedirs(save_dir, exist_ok=True)

            # model files
            Model_TD3_search.save(os.path.join(save_dir, "uav_search"))
            Model_TD3_fov.save(os.path.join(save_dir, "uav_fov"))
            Model_DDQN.save(os.path.join(save_dir, "uav_ddqn"))

            # meta (critical: keep lambda + episode)
            meta = {
                "episode": episode,
                "lambda_EE_global": float(lambda_EE_global),
                "SCALE_LAMBDA": float(SCALE_LAMBDA),
                "BETA_LAMBDA": float(BETA_LAMBDA),
                "K_UPDATE": int(K_UPDATE),
                "WARMUP_LAMBDA": int(WARMUP_LAMBDA),
            }
            with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f" Saved checkpoint: {save_dir}")
            

        # 畫routing_reward和movement_reward
    window_size = 2
    x_episode = list(range(total_episodes))
    routing_reward = routing_reward_log
    search_reward = search_reward_log
    fov_reward = fov_reward_log
    total_reward = reward_log

    # 平滑處理
    def smooth_rewards(rewards, window_size):
        smoothed_rewards = []
        for i in range(len(rewards)):
            if i < window_size:
                smoothed_rewards.append(np.mean(rewards[:i+1]))  # 前5次做一次平均
            else:
                smoothed_rewards.append(np.mean(rewards[i - window_size:i]))  # 正常的滑動窗口
        return smoothed_rewards

    smoothed_routing = smooth_rewards(routing_reward, window_size)
    smoothed_search = smooth_rewards(search_reward, window_size)
    smoothed_fov = smooth_rewards(fov_reward, window_size)
    smoothed_total = smooth_rewards(total_reward, window_size)
    smoothed_x = list(range(len(smoothed_total)))# 確保 x 軸與 smoothed_total 對應

    # ========== 第一張：Total Reward ==========
    plt.figure(figsize=(8, 6))
    plt.plot(x_episode, total_reward, label="Total Reward", color="lightcoral", linewidth=1, alpha=0.2)
    plt.plot(smoothed_x, smoothed_total, label="AVG Total Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("K-KM TD3 Dinkelbach Training - Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Total_reward.png")  
    plt.show()

    # ========== 第二張：Routing Reward ==========
    plt.figure(figsize=(8, 6))
    plt.plot(x_episode, routing_reward, label="Routing Reward", color="lightcoral", linewidth=1, alpha=0.2 )
    plt.plot(smoothed_x, smoothed_routing, label="AVG Routing Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Routing Reward")
    # plt.title("TD3-DDQN Training - Routing Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Routing_reward.png")
    plt.show()

    # ========== 第三張：Search Reward ==========
    plt.figure(figsize=(8, 6))
    plt.plot(x_episode, search_reward, label="Search Reward", color="lightcoral", linewidth=1, alpha=0.2)
    plt.plot(smoothed_x, smoothed_search, label="AVG Search Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Search Reward")
    # plt.title("TD3-DDQN Training - Search Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Task_Search_reward.png")
    plt.show()

    # ========== 第四張：FOV Reward ==========
    plt.figure(figsize=(8, 6))
    plt.plot(x_episode, fov_reward, label="FOV Reward", color="lightcoral", linewidth=1, alpha=0.2)
    plt.plot(smoothed_x, smoothed_fov, label="AVG FOV Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("FOV Reward")
    # plt.title("TD3-DDQN Training - FOV Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Task_FOV_reward.png")
    plt.show()

    # 儲存資料
    csv_dir = os.path.join("results")
    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame({
    "xs": x_episode,
    "Val": total_reward
    }).to_csv(os.path.join(csv_dir,"K-KM_TD3_results_total.csv"), index=False)
    pd.DataFrame({
    "xs": smoothed_x,
    "Val": smoothed_total
    }).to_csv(os.path.join(csv_dir,"K-KM_TD3_AVG_results_total.csv"), index=False)

    #===================== 畫Avg fov coverage=============================
    x_episode = list(range(total_episodes))
    start_idx = 2
    fov_log_cut = fov_log[start_idx:]
    x_cut = x_episode[start_idx:]

    # 平滑（注意這時長度會縮短：len = len(fov_log_cut) - window_size + 1）
    smoothed_fov = np.convolve(fov_log_cut, np.ones(window_size)/window_size, mode='valid')
    x_smooth = x_cut[:len(smoothed_fov)]  # 對齊長度

    # 畫圖
    plt.plot(x_cut, fov_log_cut, label="Avg fov", color="lightblue", linewidth=1)
    plt.plot(x_smooth, smoothed_fov, label="smooth Avg fov", color="blue", linewidth=2)
    

    plt.xlabel("Episode")
    plt.ylabel("Average FOV Coverage")
    plt.title("K-KM TD3 with Dinkelback Learning Trend of FOV Coverage")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 將 step_rewards 轉為 DataFrame
    for task in ["FOV", "COM"]:
        episodes, rates = zip(*violation_rate_log[task])

        if task == "FOV":
            label_name = "FOV"
        elif task == "COM":
            label_name = "COM"

        # 先畫原始數據線，記錄目前顏色
        line_raw, = plt.plot(episodes, rates, label=f"{label_name} Violation Rate", alpha=0.2, linewidth=1)
        color = line_raw.get_color()  # 取得 matplotlib 自動分配的顏色
        # 平滑曲線
        smoothed_rates = np.convolve(rates, np.ones(window_size)/window_size, mode='valid')
        smoothed_eps = episodes[len(episodes) - len(smoothed_rates):]
        plt.plot(smoothed_eps, smoothed_rates, label=f"{label_name} Avg", linewidth=2, color=color)  # 用相同顏色

    plt.xlabel("Episode")
    plt.ylabel("Violation Rate")
    plt.title("K-KM TD3 with Dinkelbach Violation Rate over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("violation_rate_trend.png")
    plt.show()


if __name__ == "__main__":
    train()

