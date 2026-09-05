"""Controlled centralized DDPG baseline for the shared movement flow."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import copy

from centralized_movement import (
    project_action_domain,
    project_joint_action,
    project_local_action,
)
from rng_contract import NamedRNGStreams, build_torch_module
from td3 import Actor, Critic, device


class CentralizedDDPG:
    """Single-critic DDPG with the shared derived centralized architecture."""

    agent_kind = "ddpg"

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=1.0,
        tau=0.005,
        actor_lr=6e-5,
        critic_lr=2e-4,
        rng_streams=None,
        master_seed=0,
    ):
        self.rng_streams = rng_streams or NamedRNGStreams(master_seed)
        init_seed = self.rng_streams.master_seed
        self.actor = build_torch_module(
            lambda: Actor(state_dim, action_dim, max_action),
            init_seed,
            "movement_actor_init",
            device,
        )
        self.critic = build_torch_module(
            lambda: Critic(state_dim, action_dim),
            init_seed,
            "movement_critic1_init",
            device,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(actor_lr)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(critic_lr)
        )
        self.max_action = float(max_action)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = 1
        self.target_policy_noise = None
        self.target_noise_clip = None
        self.twin_critics = False
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, episode=0, add_noise=True, noise_std=None):
        state_tensor = torch.as_tensor(
            np.asarray(state).reshape(1, -1), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        if add_noise:
            if noise_std is None:
                noise_std = max(0.05, 0.20 * (1.0 - float(episode) / 4000.0))
            if float(noise_std) < 0.0:
                raise ValueError("noise_std must be non-negative")
            action = action + self.rng_streams.numpy("movement_exploration").normal(
                0.0, float(noise_std), action.shape
            )
        action = action.astype(np.float32, copy=False)
        return (
            project_action_domain(action)
            if action.shape[-1] % 3 == 0
            else action
        )

    @staticmethod
    def decode_action(raw_action):
        v_scalar, theta_scalar, phi_scalar = project_local_action(raw_action)
        v_xy = (v_scalar + 1.0) * 5.0
        theta = theta_scalar * np.pi
        return np.asarray(
            [v_xy * np.cos(theta), v_xy * np.sin(theta), phi_scalar * 2.0],
            dtype=np.float32,
        )

    def update_joint(
        self,
        replay_memory,
        current_lambda,
        batch_size=64,
        beta_search=1.0,
        beta_vs=1.0,
        beta_com=1.0,
        beta_relay=1.0,
        reward_mode="dinkelbach",
        task_potential_enabled=True,
    ):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            current_movement_mask,
            next_movement_mask,
        ) = replay_memory.sample(
            batch_size=batch_size,
            current_lambda=current_lambda,
            gamma=self.gamma,
            beta_search=beta_search,
            beta_vs=beta_vs,
            beta_com=beta_com,
            beta_relay=beta_relay,
            reward_mode=reward_mode,
            task_potential_enabled=task_potential_enabled,
            include_movement_masks=True,
        )
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        not_done = not_done.to(device)
        current_movement_mask = current_movement_mask.to(device)
        next_movement_mask = next_movement_mask.to(device)

        with torch.no_grad():
            next_action = project_joint_action(
                self.actor_target(next_state), movement_mask=next_movement_mask
            )
            target_q = reward + not_done * self.gamma * self.critic_target(
                next_state, next_action
            )

        critic_loss = F.mse_loss(self.critic(state, action), target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = project_joint_action(
            self.actor(state), movement_mask=current_movement_mask
        )
        actor_loss = -self.critic(state, actor_action).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        with torch.no_grad():
            for online, target in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target.data.mul_(1.0 - self.tau).add_(self.tau * online.data)
            for online, target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target.data.mul_(1.0 - self.tau).add_(self.tau * online.data)

        self.num_critic_update_iteration += 1
        self.num_actor_update_iteration += 1
        self.num_training += 1
        self.last_joint_update = {
            "state": state.detach().cpu(),
            "next_state": next_state.detach().cpu(),
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_action": actor_action.detach().cpu(),
            "target_actor_action": next_action.detach().cpu(),
            "current_movement_mask": current_movement_mask.detach().cpu(),
            "next_movement_mask": next_movement_mask.detach().cpu(),
        }
        return True


class RandomMovementController:
    """Decode-only controller; action generation remains in the shared loop."""

    agent_kind = "random"
    num_critic_update_iteration = 0
    num_actor_update_iteration = 0
    num_training = 0

    def __init__(self, gamma=1.0):
        self.gamma = float(gamma)
        self.policy_delay = None
        self.target_policy_noise = None
        self.target_noise_clip = None
        self.twin_critics = False

    decode_action = staticmethod(CentralizedDDPG.decode_action)
