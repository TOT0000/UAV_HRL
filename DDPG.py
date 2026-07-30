import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
	def __init__(self, state_dim, movement_dim, max_movement):
		super(Actor, self).__init__()

		self.shared = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
		self.movement_head = nn.Linear(300, movement_dim)
		nn.init.uniform_(self.movement_head.weight, -0.003, 0.003)
		nn.init.zeros_(self.movement_head.bias)
		self.max_movement = max_movement

	
	def forward(self, state):
		x = self.shared(state)
		movement = self.max_movement * torch.tanh(self.movement_head(x))  # shape = [batch_size, 3]
		return  movement


class Critic(nn.Module):
	def __init__(self, state_dim,  movement_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 +  movement_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, movement_action):
		x = F.relu(self.l1(state))
		x = torch.cat([x, movement_action], dim=1)
		x = F.relu(self.l2(x))
		return self.l3(x)


class DDPG(object):
	def __init__(self, state_dim, movement_dim, max_movement, discount=0.95, tau=0.005):
		self.actor = Actor(state_dim, movement_dim, max_movement).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, movement_dim).to(device)
		self.cost_critic = Critic(state_dim, movement_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.cost_critic_target =  copy.deepcopy(self.cost_critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=5e-3)
		self.cost_critic_optimizer =  torch.optim.Adam(self.cost_critic.parameters(), lr=3e-4, weight_decay=5e-3)

		self.actor_reward_log = []
		self.actor_cost_log = []
		self.actor_loss_log = []
		self.total_it = 0
		self.lambda_param = 1
		self.discount = discount
		self.tau = tau
		# ✅ 新增這行初始化 loss history
		self.actor_reward_log = []
		self.actor_cost_log = []
		self.actor_loss_log = []
		self.critic_loss_history =[]
		self.critic_cost_loss_history=[]

		self.movement_dim = movement_dim
		self.max_action =  max_movement	

	def select_action(self, state, episode=0, add_noise=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		with torch.no_grad():
			raw_action = self.actor(state).cpu().numpy().flatten()

		if add_noise:
			noise_std = max(0.1, 0.5 * (1 - episode / 4000))
			noise = np.random.normal(0, noise_std, size=raw_action.shape)
			raw_action = raw_action + noise

		raw_action = np.clip(raw_action, -1.0, 1.0)
		return raw_action

	def decode_action(self, raw_action):
		raw_action = np.asarray(raw_action, dtype=np.float32)
		v_scalar, theta_scalar, phi_scalar = np.clip(raw_action, -1.0, 1.0)

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
			movement_action = self.actor(state)

		# ---------- Movement ----------
		movement_action_raw = movement_action.cpu().numpy().flatten()
		v_scalar, theta_scalar, phi_scalar = np.clip(movement_action_raw, -1, 1)
		max_speed_xy = 10
		max_dz = 5.0
		v_xy = (v_scalar + 1) / 2 * max_speed_xy
		theta = theta_scalar * np.pi
		dx = v_xy * np.cos(theta)
		dy = v_xy * np.sin(theta)
		dz = phi_scalar * max_dz
		movement_action = np.array([dx, dy, dz]) 

		return  movement_action

	def train(self, replay_buffer, num_GT, batch_size=64):
		if replay_buffer.size < batch_size:
			return  # 等待 replay buffer 有足夠資料

		self.total_it += 1

		# 1️⃣ Sample replay buffer
		state, action, next_state, reward, not_done, num_gt = replay_buffer.sample_by_tag(
			batch_size, curr_tag=num_GT, p_same=0.6, p_neighbor=0.2
		)
	
		# 2️⃣ Compute target Q value
		with torch.no_grad():
			next_action = self.actor_target(next_state)
			target_Q = self.critic_target(next_state, next_action)
			target_Q = reward + not_done * self.discount * target_Q

		# 3️⃣ Q-value 預測
		current_Q = self.critic(state, action)

		# 4️⃣ Critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.critic_loss_history.append(critic_loss.item())

		# 5️⃣ Actor loss
		actor_movement = self.actor(state) 
		actor_loss = -self.critic(state, actor_movement).mean()
		self.actor_loss_log.append(actor_loss.item())

		# 6️⃣ Update actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# 7️⃣ Polyak target update
		with torch.no_grad():
			for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
				tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
			for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
				tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)



	def save(self, save_dir="param"):
		os.makedirs(save_dir, exist_ok=True) #建立資料夾
		torch.save(self.critic.state_dict(), save_dir + "_critic")
		torch.save(self.critic_optimizer.state_dict(), save_dir + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), save_dir + "_actor")
		torch.save(self.actor_optimizer.state_dict(), save_dir + "_actor_optimizer")
		print("====================================")
		print(f"Model has been saved to [{save_dir}]...")
		print("====================================")

	def load(self, load_dir="param"):
		self.critic.load_state_dict(torch.load(load_dir + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(load_dir + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)
 
		self.actor.load_state_dict(torch.load(load_dir + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(load_dir + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
	def plot_actor_losses(agent):
		plt.figure(figsize=(10, 5))
		plt.plot(agent.actor_cost_log, label="Cost Penalty", color='red')
		plt.plot(agent.actor_reward_log, label="Reward Value", color='green')
		plt.plot(agent.actor_loss_log, label="Total Actor Loss", color='black')
		plt.xlabel("Policy Update Steps")
		plt.ylabel("Loss Value")
		plt.title("Actor Loss Components Over Time")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()
