from collections import defaultdict
from dataclasses import dataclass
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DDQN import DDQN
from Packet_scheduler_v1 import PacketEngine, final_hop_delivered_bits
from Simulator import Simulator
from centralized_movement import (
    JOINT_ACTION_DIM,
    MOVEMENT_STATE_DIM,
    apply_joint_movement_proposals,
    build_joint_movement_proposals,
    calculate_movement_potentials,
    get_global_movement_state,
    project_joint_action,
)
from com_capacity_calibration import load_com_capacity_reference
from td3 import TD3
import utils_update_v2


MOVEMENT_CONTROL_INTERVAL = 4
ROUTING_STATE_DIM = 122
PRODUCTION_WARMUP_TRANSITIONS = 1000
PRODUCTION_BATCH_SIZE = 64
PRODUCTION_POLICY_DELAY = 2


def _is_movement_decision(slot, interval=MOVEMENT_CONTROL_INTERVAL):
    return slot % interval == 0


def _is_last_movement_decision(
    slot, total_slots, interval=MOVEMENT_CONTROL_INTERVAL
):
    return _is_movement_decision(slot, interval) and slot + interval >= total_slots


def _is_episode_end(slot, total_slots):
    return slot == total_slots - 1


def _search_transition_done(search_done, movement_episode_end):
    return bool(search_done or movement_episode_end)


def _create_active_replay_buffers(state_dim, routing_dim, max_size=int(2e5)):
    """Legacy test helper; the active movement path uses ReplayBufferJoint."""
    routing_buffer = utils_update_v2.ReplayBufferDiscrete(
        state_dim,
        action_dim=routing_dim,
        max_size=max_size,
        n_step=1,
        gamma=0.99,
    )
    movement_buffer_search = utils_update_v2.ReplayBufferContinuous(
        state_dim, action_dim=3, max_size=max_size, n_step=1, gamma=0.99
    )
    movement_buffer_fov = utils_update_v2.ReplayBufferContinuous(
        state_dim, action_dim=3, max_size=max_size, n_step=1, gamma=0.99
    )
    return routing_buffer, movement_buffer_search, movement_buffer_fov


@dataclass
class TrainingConfig:
    total_episodes: int = 6
    episode_seconds: int = 60
    routing_slot_seconds: float = 0.25
    warmup_joint_transitions: int = PRODUCTION_WARMUP_TRANSITIONS
    batch_size: int = PRODUCTION_BATCH_SIZE
    policy_delay: int = PRODUCTION_POLICY_DELAY
    replay_max_size: int = int(2e5)
    beta_search: float = 1.0
    beta_vs: float = 1.0
    beta_com: float = 1.0
    search_coverage_threshold: float = 0.99
    checkpoint_every: int = 2
    checkpoint_root: str = "checkpoints_centralized_td3"
    resume_dir: str | None = None
    enable_checkpoints: bool = True
    enable_plots: bool = True
    enable_csv: bool = True
    random_seed: int | None = None

    def __post_init__(self):
        slots = 1.0 / float(self.routing_slot_seconds)
        if not np.isclose(slots, MOVEMENT_CONTROL_INTERVAL):
            raise ValueError("movement interval must contain exactly four routing slots")
        if self.episode_seconds <= 0 or self.total_episodes <= 0:
            raise ValueError("episode_seconds and total_episodes must be positive")
        if self.warmup_joint_transitions < 0 or self.batch_size <= 0:
            raise ValueError("warmup must be non-negative and batch_size positive")


def _routing_masks(env):
    num_uav = env.num_UAV
    capacity_ok = env.Capacity_matrix > 0.1
    np.fill_diagonal(capacity_ok, False)
    gs_ok = (
        env.gs_capacity > 0.1
        if env.gs_capacity is not None
        else np.zeros(num_uav, dtype=bool)
    )
    masks = {}
    for uav_id in range(num_uav):
        mask = np.zeros(num_uav + 1, dtype=bool)
        mask[:num_uav] = capacity_ok[uav_id]
        mask[env.GS_ID] = bool(gs_ok[uav_id])
        masks[uav_id] = mask
    return masks


def _active_backlog(packet_engine):
    return defaultdict(float, packet_engine.backlog_bits)


def _run_routing_slot(
    env,
    packet_engine,
    ddqn,
    routing_buffer,
    routing_masks,
    current_time,
    done,
    delay_bound_steps,
    violation_stats,
):
    step_time = float(packet_engine.step_time)
    env.current_time = float(current_time)
    packet_engine.drop_expired_packets(env.current_time)
    packet_engine.inject_packets(
        env, delay_bound_steps, env.current_time, step_time=step_time
    )
    active_packets = packet_engine.get_active_packets()
    backlog_before = _active_backlog(packet_engine)
    uavs_with_packets = sorted(
        uid
        for uid, bits in backlog_before.items()
        if uid != env.GS_ID and bits > 0.0
    )

    states = {
        uid: packet_engine.get_state_ta(
            env, uid, backlog_bits=backlog_before
        )
        for uid in uavs_with_packets
    }
    for uid, state in states.items():
        if state.shape != (ROUTING_STATE_DIM,):
            raise AssertionError(
                f"routing state for UAV {uid} has shape {state.shape}, expected (122,)"
            )

    next_hops = {}
    for uid in uavs_with_packets:
        next_hops[uid] = int(
            ddqn.select_action(
                states[uid], uid, routing_masks[uid], visited_nodes=None
            )
        )

    slot_reward = defaultdict(float)
    slot_cost = defaultdict(float)
    delivered_bits = 0.0
    for packet in active_packets:
        if packet.get("done", False):
            continue
        from_uav = int(packet["current"])
        if from_uav == env.GS_ID:
            continue
        to_target = int(next_hops.get(from_uav, env.GS_ID))
        if to_target == env.GS_ID:
            capacity_mbps = float(env.gs_capacity[from_uav])
        else:
            capacity_mbps = float(env.Capacity_matrix[from_uav, to_target])
        if capacity_mbps <= 0.0:
            continue

        packet_bits = float(packet.get("rem_bits", packet.get("size_bits", 0.0)))
        node_backlog = float(packet_engine.backlog_bits.get(from_uav, 0.0))
        queue_bits_without_packet = max(node_backlog - packet_bits, 0.0)
        if to_target == env.GS_ID:
            outgoing_links = 1
        else:
            outgoing_links = max(int(env.k_u_u2u[from_uav]), 1)
        effective_queue_bits = queue_bits_without_packet / outgoing_links
        predicted_bits = min(
            capacity_mbps * 1e6 * step_time, packet_bits
        )
        hop_delay_ms = packet_engine.log_hop_delay(
            env,
            packet,
            current_node=from_uav,
            next_hop=to_target,
            link_capacity_mbps=capacity_mbps,
            current_time=env.current_time,
            pkt_bits=predicted_bits,
            backlog_bits=effective_queue_bits,
        )
        (
            task_type,
            route_reward,
            packet_done,
            _,
            violated,
            cost,
            bits_used,
            _,
            _,
        ) = packet_engine.calculate_packet_reward_fast(
            env,
            packet,
            hop_delay_ms,
            from_uav=from_uav,
            to_target=to_target,
            t=env.current_time,
            backlog=queue_bits_without_packet,
            mode="uav",
            channel_capacity=capacity_mbps,
        )
        delivered_bits += final_hop_delivered_bits(
            to_target, env.GS_ID, bits_used
        )
        slot_reward[from_uav] += float(route_reward)
        slot_cost[from_uav] += float(cost)
        if packet_done and task_type in violation_stats:
            violation_stats[task_type]["delivered"] += 1
            if violated:
                violation_stats[task_type]["violated"] += 1

    backlog_after = _active_backlog(packet_engine)
    next_states = {
        uid: packet_engine.get_state_ta(
            env, uid, backlog_bits=backlog_after
        )
        for uid in uavs_with_packets
    }
    for uid in uavs_with_packets:
        routing_buffer.add(
            states[uid],
            int(next_hops.get(uid, uid)),
            next_states.get(uid, states[uid]),
            float(slot_reward[uid]),
            float(slot_cost[uid]),
            bool(done),
            tag_gt=env.num_GT,
        )
    return delivered_bits, float(sum(slot_reward.values()))


def _mark_search_observations(env):
    if getattr(env, "_search_phase_over", False):
        return
    for uav_id in range(env.num_UAV):
        has_search = any(
            task.get("task_type") == "Search"
            for task in env.multi_tasks.get(uav_id, [])
        )
        if has_search:
            env.update_visited_grid(uav_id)
            env.mark_search_coverage(uav_id)


def _dinkelbach_update(delivered_mbits, total_energy, previous_lambda):
    if total_energy <= 1e-12:
        return float(previous_lambda)
    return float(delivered_mbits / total_energy)


def _interval_reward(
    delivered_mbits,
    energy,
    current_lambda,
    gamma,
    potentials_t,
    potentials_t1,
    done,
    config,
):
    next_values = (0.0, 0.0, 0.0) if done else potentials_t1
    shaping = (
        config.beta_search * (gamma * next_values[0] - potentials_t[0])
        + config.beta_vs * (gamma * next_values[1] - potentials_t[1])
        + config.beta_com * (gamma * next_values[2] - potentials_t[2])
    )
    return float(delivered_mbits - current_lambda * energy + shaping)


def train(config=None):
    config = TrainingConfig() if config is None else config
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    c_ref_com, calibration = load_com_capacity_reference()
    env = Simulator(num_UAV=16)
    packet_engine = PacketEngine(num_uav=16, step_time=config.routing_slot_seconds)
    centralized_td3 = TD3(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_action=1.0,
        policy_delay=config.policy_delay,
    )
    ddqn = DDQN(ROUTING_STATE_DIM, env.num_UAV + 1)
    joint_replay = utils_update_v2.ReplayBufferJoint(
        MOVEMENT_STATE_DIM,
        JOINT_ACTION_DIM,
        max_size=config.replay_max_size,
    )
    routing_replay = utils_update_v2.ReplayBufferDiscrete(
        ROUTING_STATE_DIM,
        env.num_UAV + 1,
        max_size=config.replay_max_size,
        n_step=1,
        gamma=0.99,
    )

    lambda_ee = 0.1
    start_episode = 0
    if config.resume_dir is not None:
        if not os.path.isdir(config.resume_dir):
            raise FileNotFoundError(f"centralized checkpoint not found: {config.resume_dir}")
        meta_path = os.path.join(config.resume_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("movement_state_dim") != MOVEMENT_STATE_DIM or metadata.get(
            "joint_action_dim"
        ) != JOINT_ACTION_DIM:
            raise RuntimeError(
                "checkpoint is not compatible with the 531-D/48-D centralized TD3"
            )
        centralized_td3.load(os.path.join(config.resume_dir, "centralized_td3"))
        ddqn.load(os.path.join(config.resume_dir, "uav_ddqn"))
        lambda_ee = float(metadata["lambda_EE_global"])
        start_episode = int(metadata["episode"]) + 1

    reward_log = []
    delivered_log = []
    energy_log = []
    lambda_log = []
    total_joint_transitions = 0
    duplicate_target_assertions = 0
    environment_actor_calls = 0
    proposal_batches = 0
    energy_evaluations = 0
    terminal_joint_transitions = 0
    routing_slots_executed = 0
    delay_bound_steps = int(5.0 / config.routing_slot_seconds)

    for episode in range(start_episode, start_episode + config.total_episodes):
        env.num_GT = int(np.random.randint(2, 10))
        env.reset_environment()
        packet_engine.reset_packet_state()
        env.lambda_EE_global = float(lambda_ee)
        episode_lambda = float(lambda_ee)
        episode_delivered_mbits = 0.0
        episode_energy = 0.0
        episode_reward = 0.0
        episode_routing_reward = 0.0
        violation_stats = {
            "FOV": {"delivered": 0, "violated": 0},
            "COM": {"delivered": 0, "violated": 0},
        }

        for interval in range(config.episode_seconds):
            if interval == 0:
                # Match the existing slot-0 SR update. Later SR updates are part
                # of the preceding one-second transition so S_{t+1} == S_t next.
                env.advance_sr_teams()
            if getattr(env, "need_reassign", False):
                env.assign_tasks()
                env.need_reassign = False

            backlog_before = _active_backlog(packet_engine)
            try:
                state = get_global_movement_state(
                    env, packet_engine, backlog_before, c_ref_com
                )
                potentials_t = calculate_movement_potentials(env, c_ref_com)
            except ValueError as exc:
                if "duplicate" in str(exc):
                    duplicate_target_assertions += 1
                raise

            if total_joint_transitions < config.warmup_joint_transitions:
                raw_joint_action = np.random.uniform(
                    -1.0, 1.0, size=JOINT_ACTION_DIM
                ).astype(np.float32)
            else:
                raw_joint_action = centralized_td3.select_action(
                    state, episode=episode, add_noise=True
                )
                environment_actor_calls += 1
            projected_action = project_joint_action(raw_joint_action, state)

            # Phase 1 is read-only: all sixteen proposals are built from one snapshot.
            proposals = build_joint_movement_proposals(
                env, centralized_td3, projected_action
            )
            proposal_batches += 1
            # Phase 2 mutates positions only after every proposal exists.
            interval_energies = apply_joint_movement_proposals(env, proposals)
            energy_evaluations += int(interval_energies.size)
            interval_energy = float(interval_energies.sum())

            _mark_search_observations(env)
            if getattr(env, "need_reassign", False):
                env.assign_tasks()
                env.need_reassign = False
            if (
                not getattr(env, "_search_phase_over", False)
                and float(env.visited_bitmap.mean())
                >= config.search_coverage_threshold
            ):
                env._search_phase_over = True
                env.convert_search_to_hovering()
            env.update_source_uavs()
            env.update_u2u_channels()
            env.update_u2g_channels()
            masks = _routing_masks(env)

            interval_delivered_bits = 0.0
            for routing_slot in range(MOVEMENT_CONTROL_INTERVAL):
                routing_slots_executed += 1
                absolute_slot = interval * MOVEMENT_CONTROL_INTERVAL + routing_slot
                final_slot = (
                    interval == config.episode_seconds - 1
                    and routing_slot == MOVEMENT_CONTROL_INTERVAL - 1
                )
                delivered_bits, routing_reward = _run_routing_slot(
                    env,
                    packet_engine,
                    ddqn,
                    routing_replay,
                    masks,
                    current_time=absolute_slot * config.routing_slot_seconds,
                    done=final_slot,
                    delay_bound_steps=delay_bound_steps,
                    violation_stats=violation_stats,
                )
                interval_delivered_bits += delivered_bits
                episode_routing_reward += routing_reward

            interval_delivered_mbits = interval_delivered_bits / 1e6
            done = interval == config.episode_seconds - 1
            if not done:
                env.advance_sr_teams()
            backlog_after = _active_backlog(packet_engine)
            potentials_t1 = calculate_movement_potentials(env, c_ref_com)
            next_state = get_global_movement_state(
                env, packet_engine, backlog_after, c_ref_com
            )
            terminal_joint_transitions += int(done)
            joint_replay.add(
                state,
                projected_action,
                next_state,
                done=done,
                delivered_mbits=interval_delivered_mbits,
                total_mobility_energy=interval_energy,
                phi_search_t=potentials_t[0],
                phi_search_t1=potentials_t1[0],
                phi_vs_t=potentials_t[1],
                phi_vs_t1=potentials_t1[1],
                phi_com_t=potentials_t[2],
                phi_com_t1=potentials_t1[2],
            )
            total_joint_transitions += 1
            episode_delivered_mbits += interval_delivered_mbits
            episode_energy += interval_energy
            episode_reward += _interval_reward(
                interval_delivered_mbits,
                interval_energy,
                episode_lambda,
                centralized_td3.gamma,
                potentials_t,
                potentials_t1,
                done,
                config,
            )

            if (
                total_joint_transitions >= config.warmup_joint_transitions
                and joint_replay.size >= config.batch_size
            ):
                centralized_td3.update_joint(
                    joint_replay,
                    current_lambda=lambda_ee,
                    batch_size=config.batch_size,
                    beta_search=config.beta_search,
                    beta_vs=config.beta_vs,
                    beta_com=config.beta_com,
                )

        lambda_ee = _dinkelbach_update(
            episode_delivered_mbits, episode_energy, lambda_ee
        )
        env.lambda_EE_global = lambda_ee
        if routing_replay.size >= config.batch_size:
            ddqn.train(routing_replay, config.batch_size)
        ddqn.update_target()

        reward_log.append(episode_reward)
        delivered_log.append(episode_delivered_mbits)
        energy_log.append(episode_energy)
        lambda_log.append(lambda_ee)
        print(
            f"[Episode {episode + 1}] joint_transitions={config.episode_seconds} "
            f"reward={episode_reward:.6f} delivered={episode_delivered_mbits:.6f} Mbit "
            f"energy={episode_energy:.6f} J lambda={lambda_ee:.9f} "
            f"routing_reward={episode_routing_reward:.6f} "
            f"critic_updates={centralized_td3.num_critic_update_iteration} "
            f"actor_updates={centralized_td3.num_actor_update_iteration}"
        )

        if (
            config.enable_checkpoints
            and config.checkpoint_every > 0
            and (episode + 1) % config.checkpoint_every == 0
        ):
            save_dir = os.path.join(config.checkpoint_root, f"ep_{episode + 1:04d}")
            os.makedirs(save_dir, exist_ok=True)
            centralized_td3.save(os.path.join(save_dir, "centralized_td3"))
            ddqn.save(os.path.join(save_dir, "uav_ddqn"))
            with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "episode": episode,
                        "lambda_EE_global": lambda_ee,
                        "movement_state_dim": MOVEMENT_STATE_DIM,
                        "routing_state_dim": ROUTING_STATE_DIM,
                        "joint_action_dim": JOINT_ACTION_DIM,
                        "c_ref_com": c_ref_com,
                    },
                    handle,
                    indent=2,
                )

    if config.enable_csv:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(
            {
                "episode": np.arange(len(reward_log)),
                "reward": reward_log,
                "delivered_mbits": delivered_log,
                "mobility_energy": energy_log,
                "lambda": lambda_log,
            }
        ).to_csv("results/centralized_td3_training.csv", index=False)
    if config.enable_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(reward_log, label="Centralized TD3 reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "episodes": config.total_episodes,
        "joint_transitions": total_joint_transitions,
        "critic_updates": centralized_td3.num_critic_update_iteration,
        "actor_updates": centralized_td3.num_actor_update_iteration,
        "routing_state_dim": ROUTING_STATE_DIM,
        "movement_state_dim": MOVEMENT_STATE_DIM,
        "joint_action_dim": JOINT_ACTION_DIM,
        "joint_replay_size": joint_replay.size,
        "routing_replay_size": routing_replay.size,
        "lambda": lambda_ee,
        "calibration": calibration,
        "duplicate_target_assertions": duplicate_target_assertions,
        "environment_actor_calls": environment_actor_calls,
        "proposal_batches": proposal_batches,
        "energy_evaluations": energy_evaluations,
        "terminal_joint_transitions": terminal_joint_transitions,
        "routing_slots_executed": routing_slots_executed,
        "reward_log": reward_log,
        "delivered_log": delivered_log,
        "energy_log": energy_log,
    }


if __name__ == "__main__":
    train()
