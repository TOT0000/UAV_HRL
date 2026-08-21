"""Routing-agent factory for safe-DDQN, controlled DQN, and random routing."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from DDQN import DDQN, QNetwork, device, routing_action_mask_from_state


class ControlledDQN:
    """Standard masked DQN with no cost critic and no Double-DQN target."""

    routing_agent_kind = "dqn"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
    ):
        self.q_network = QNetwork(action_dim, state_dim, hidden_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.action_dim = int(action_dim)
        self.loss_log = []
        self.num_training = 0
        self.target_update_count = 0

    def select_action(
        self,
        state,
        uav_id,
        mask=None,
        visited_nodes=None,
        epsilon=0.5,
        logits_noise_std=0.0,
        eta=None,
    ):
        del uav_id, visited_nodes, eta
        state_tensor = torch.as_tensor(
            np.asarray(state).reshape(1, -1), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().reshape(-1)
        legal = np.ones(self.action_dim, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if legal.shape != (self.action_dim,):
            raise ValueError("routing mask shape does not match controlled DQN actions")
        if not legal.any():
            raise ValueError("routing action mask has no legal action")
        if np.random.rand() < float(epsilon):
            return int(np.random.choice(np.flatnonzero(legal)))
        masked_values = q_values.copy()
        masked_values[~legal] = -np.inf
        if float(logits_noise_std) > 0.0:
            noise = np.random.normal(0.0, float(logits_noise_std), q_values.shape)
            masked_values[legal] += noise[legal]
        return int(np.argmax(masked_values))

    def _routing_action_mask(self, next_state):
        return routing_action_mask_from_state(next_state, self.action_dim)

    @torch.no_grad()
    def _standard_targets(self, next_state, reward, not_done):
        legal = self._routing_action_mask(next_state)
        target_values = self.target_q_network(next_state).masked_fill(
            ~legal, float("-inf")
        )
        next_values = target_values.max(dim=1).values
        return reward.squeeze(1) + not_done.squeeze(1) * self.gamma * next_values

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, _cost, not_done = replay_buffer.sample(
            batch_size
        )
        values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        targets = self._standard_targets(next_state, reward, not_done)
        loss = F.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_log.append(float(loss.item()))
        self.num_training += 1

    def update_target(self):
        for parameter, target_parameter in zip(
            self.q_network.parameters(), self.target_q_network.parameters()
        ):
            target_parameter.data.copy_(
                self.tau * parameter.data
                + (1.0 - self.tau) * target_parameter.data
            )
        self.target_update_count += 1


class RandomRoutingController:
    """Seed-controlled uniform sampling over the current effective mask."""

    routing_agent_kind = "random"

    def __init__(self, gamma=0.99):
        self.gamma = float(gamma)
        self.tau = None
        self.num_training = 0
        self.target_update_count = 0
        self.loss_log = []

    def select_action(
        self,
        state,
        uav_id,
        mask=None,
        visited_nodes=None,
        epsilon=0.0,
        logits_noise_std=0.0,
        eta=None,
    ):
        del state, uav_id, visited_nodes, epsilon, logits_noise_std, eta
        legal = np.asarray(mask, dtype=bool)
        if legal.ndim != 1 or not legal.any():
            raise ValueError("random routing requires at least one legal action")
        return int(np.random.choice(np.flatnonzero(legal)))

    def train(self, replay_buffer, batch_size=64):
        del replay_buffer, batch_size

    def update_target(self):
        return None


def create_routing_agent(method_spec, state_dim, action_dim):
    if method_spec.routing == "safe_ddqn":
        return DDQN(state_dim, action_dim)
    if method_spec.routing == "dqn":
        return ControlledDQN(state_dim, action_dim)
    if method_spec.routing == "random":
        return RandomRoutingController()
    raise ValueError(f"unsupported routing policy: {method_spec.routing}")
