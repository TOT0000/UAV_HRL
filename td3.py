import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy

from centralized_movement import (
    project_action_domain,
    project_joint_action,
    project_local_action,
)
from rng_contract import NamedRNGStreams, build_torch_module

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


def _bellman_target(immediate_value, next_value, not_done, discount):
    return immediate_value + not_done * discount * next_value


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)          
		self.l5 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		a = F.relu(self.l4(a))          
		return self.max_action * torch.tanh(self.l5(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)          
		self.l5 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = F.relu(self.l4(q1))          
		q1 = self.l5(q1)

		return q1


class TD3():
    agent_kind = "td3"

    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.10,
        noise_clip=0.25,
        policy_delay=2,
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
        self.critic_1 = build_torch_module(
            lambda: Critic(state_dim, action_dim),
            init_seed,
            "movement_critic1_init",
            device,
        )
        self.critic_2 = build_torch_module(
            lambda: Critic(state_dim, action_dim),
            init_seed,
            "movement_critic2_init",
            device,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        # self.cost_1 = Critic(state_dim, action_dim).to(device)
        # self.cost_1_target = Critic(state_dim, action_dim).to(device)
        # self.cost_2 = Critic(state_dim, action_dim).to(device)
        # self.cost_2_target = Critic(state_dim, action_dim).to(device)        

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # self.cost_1_target.load_state_dict(self.cost_1.state_dict())
        # self.cost_2_target.load_state_dict(self.cost_2.state_dict())        

        # self.lambd_lr = 1e-2
        # self.C = 4e-2
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(
        self, state, episode=0, add_noise=True, noise_std=None
    ):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            raw_action = self.actor(state).cpu().numpy().flatten()

        if add_noise:
            # Active centralized training supplies an explicit transition-based
            # schedule. Retain the legacy fallback for inactive entry points.
            if noise_std is None:
                noise_std = max(0.1, 0.5 * (1 - episode / 4000))
            if float(noise_std) < 0.0:
                raise ValueError(f"noise_std must be non-negative, got {noise_std}")
            noise = self.rng_streams.numpy("movement_exploration").normal(
                0, noise_std, size=raw_action.shape
            )
            raw_action = raw_action + noise

        raw_action = raw_action.astype(np.float32, copy=False)
        return (
            project_action_domain(raw_action)
            if raw_action.shape[-1] % 3 == 0
            else raw_action
        )


    def decode_action(self, raw_action):
        v_scalar, theta_scalar, phi_scalar = project_local_action(raw_action)

        max_speed_xy = 10.0
        max_dz = 2.0

        v_xy = (v_scalar + 1.0) / 2.0 * max_speed_xy
        theta = theta_scalar * np.pi

        dx = v_xy * np.cos(theta)
        dy = v_xy * np.sin(theta)
        dz = phi_scalar * max_dz

        return np.array([dx, dy, dz], dtype=np.float32)


    def Testing_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            raw_action = self.actor(state).cpu().numpy().flatten()

        return self.decode_action(raw_action)

    def update(self,replay_memory, num_GT, batch_size=64):

        raise RuntimeError(
            "legacy per-tag TD3.update is disabled; use canonical update_joint"
        )

        # if self.num_training % 500 == 0:
        #     print("====================================")
        #     print("model has been trained for {} times...".format(self.num_training))
        #     print("====================================")

        x, y, u, r, d, ng = replay_memory.sample_by_tag(batch_size, curr_tag=num_GT, p_same=0.6, p_neighbor=0.2)
        state = x.to(device)
        action = y.to(device)
        next_state = u.to(device)
        reward = r.to(device)
        not_done = d.to(device)
        num_gt = ng.to(device)
        # cost = c.to(device)

        # Select next action according to target policy:
        noise = torch.randn(
            action.shape,
            dtype=action.dtype,
            device=action.device,
            generator=self.rng_streams.torch(
                "td3_target_policy_noise", device=action.device
            ),
        ) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_state) + noise)
        next_action = next_action.clamp(-self.max_action, self.max_action)

        # Compute target Q-value:
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = _bellman_target(reward, target_Q, not_done, self.gamma).detach()

        # Optimize Critic 1:
        current_Q1 = self.critic_1(state, action)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        current_Q2 = self.critic_2(state, action)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # # Compute target cost Q-value:
        # target_C1 = self.cost_1_target(next_state, next_action)
        # target_C2 = self.cost_2_target(next_state, next_action)
        # target_C = torch.min(target_C1, target_C2)
        # target_C = cost + ((1 - done) * self.gamma * target_C).detach()

        # # Optimize Cost 1:
        # current_C1 = self.cost_1(state, action)
        # loss_C1 = F.mse_loss(current_C1, target_C)
        # self.cost_1_optimizer.zero_grad()
        # loss_C1.backward()
        # self.cost_1_optimizer.step()
        # # Optimize Cost 2:
        # current_C2 = self.cost_2(state, action)
        # loss_C2 = F.mse_loss(current_C2, target_C)
        # self.cost_2_optimizer.zero_grad()
        # loss_C2.backward()
        # self.cost_2_optimizer.step()

        # Delayed policy updates:
        if self.num_training % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()

            # Polyak update
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

            self.num_actor_update_iteration += 1

        self.num_critic_update_iteration += 1
        self.num_training += 1

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
            raw_target_action = self.actor_target(next_state)
            noise = torch.randn(
                action.shape,
                dtype=action.dtype,
                device=action.device,
                generator=self.rng_streams.torch(
                    "td3_target_policy_noise", device=action.device
                ),
            ) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            expanded_mask = next_movement_mask.unsqueeze(-1).expand(
                *next_movement_mask.shape, 3
            ).reshape_as(noise)
            noise = noise * expanded_mask.to(dtype=noise.dtype)
            next_action = project_joint_action(
                raw_target_action + noise, movement_mask=next_movement_mask
            )
            target_actor_action = project_joint_action(
                raw_target_action, movement_mask=next_movement_mask
            )
            target_q = torch.min(
                self.critic_1_target(next_state, next_action),
                self.critic_2_target(next_state, next_action),
            )
            target_q = _bellman_target(
                reward, target_q, not_done, self.gamma
            )

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        self.num_critic_update_iteration += 1
        self.num_training += 1
        actor_updated = False
        actor_action = None
        actor_loss_value = None
        if self.num_critic_update_iteration % self.policy_delay == 0:
            actor_action = project_joint_action(
                self.actor(state), movement_mask=current_movement_mask
            )
            actor_loss = -self.critic_1(state, actor_action).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            actor_updated = True

            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

            self.num_actor_update_iteration += 1

        self.last_joint_update = {
            "state": state.detach().cpu(),
            "next_state": next_state.detach().cpu(),
            "target_actor_action": target_actor_action.detach().cpu(),
            "target_smoothed_action": next_action.detach().cpu(),
            "target_policy_noise": noise.detach().cpu(),
            "actor_action": None if actor_action is None else actor_action.detach().cpu(),
            "critic_1_loss": float(critic_1_loss.detach().cpu()),
            "critic_2_loss": float(critic_2_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "current_movement_mask": current_movement_mask.detach().cpu(),
            "next_movement_mask": next_movement_mask.detach().cpu(),
        }
        return actor_updated

    def save(self, save_dir="param"):
        os.makedirs(save_dir, exist_ok=True) #建立資料夾
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')
        torch.save(self.actor_target.state_dict(), f'{save_dir}/actor_target.pth')
        torch.save(self.critic_1.state_dict(), f'{save_dir}/critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), f'{save_dir}/critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), f'{save_dir}/critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), f'{save_dir}/critic_2_target.pth')
        print("====================================")
        print(f"Model has been saved to [{save_dir}]...")
        print("====================================")

    def load(self, load_dir="param"):
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth', map_location='cpu'))
        self.actor_target.load_state_dict(torch.load(f'{load_dir}/actor_target.pth', map_location='cpu'))
        self.critic_1.load_state_dict(torch.load(f'{load_dir}/critic_1.pth', map_location='cpu'))
        self.critic_1_target.load_state_dict(torch.load(f'{load_dir}/critic_1_target.pth', map_location='cpu'))
        self.critic_2.load_state_dict(torch.load(f'{load_dir}/critic_2.pth', map_location='cpu'))
        self.critic_2_target.load_state_dict(torch.load(f'{load_dir}/critic_2_target.pth', map_location='cpu'))
        print("====================================")
        print(f"Model has been loaded from [{load_dir}]...")
        print("====================================")

