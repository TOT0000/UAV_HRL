import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

# 深度 Q 網絡（DQN）類別
class DQN(nn.Module):
    def __init__(self, n_features, n_actions, learning_rate=1e-3, reward_decay=0.99, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, e_greedy_increment=None, hidden_dim=20):
        super( DQN, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.learn_step_counter = 0

        # 初始化回放緩衝區
        self.memory = []

        # 定義reward Q networks
        self.eval_net = self._build_net(hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.eval_net).to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # 定義cost Q networks
        self.eval_cost_net = self._build_net(hidden_dim).to(self.device)
        self.target_cost_net = copy.deepcopy(self.eval_cost_net).to(self.device)
        self.cost_optimizer = optim.Adam(self.eval_cost_net.parameters(), lr=self.lr)
        self.cost_loss_fn = nn.MSELoss()

        # logs (optional)
        self.reward_loss_log = []
        self.cost_loss_log = []

        self.cost_his = []

    def _build_net(self,hidden_dim=20):
        # 定義 Q 網絡結構
        return nn.Sequential(
            nn.Linear(self.n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )
    @torch.no_grad()
    def _forward_reward_q(self, state_np):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).view(1, -1)
        return self.eval_net(s).detach().cpu().numpy().flatten()

    def select_action(self, state, uav_id, mask=None, visited_nodes=None, epsilon=None, logits_noise_std=0,lambda_ee=None, eta=0.01):
        q_r = self._forward_reward_q(state)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            q_c = self.eval_cost_net(s).detach().cpu().numpy().flatten()
        # if np.random.rand() < 0.001:  # 0.1% 機率印，避免爆log
        #     q_c_eff = np.maximum(q_c, 0.0)
        #     qt = q_r - eta * q_c_eff
        #     # qt = q_r - float(eta) * q_c
        #     print(f"[lag] eta={eta:.4f} "
        #         f"q_r(min,max)=({q_r.min():.3f},{q_r.max():.3f}) "
        #         f"q_c(min,max)=({q_c_eff.min():.3f},{q_c_eff.max():.3f}) "
        #         f"q_t(min,max)=({qt.min():.3f},{qt.max():.3f})")
        q_c_eff = np.maximum(q_c, 0.0)
        q_values = q_r - float(eta) * q_c
        
        if epsilon is None:
            epsilon = self.epsilon

        # ---------- mask ----------
        if mask is None:
            final_mask = np.ones_like(q_values, dtype=bool)
        else:
            final_mask = np.array(mask, dtype=bool)

        # exclude self
        if 0 <= uav_id < len(final_mask):
            final_mask[uav_id] = False

        # exclude visited nodes
        if visited_nodes:
            for node in visited_nodes:
                if 0 <= node < len(final_mask):
                    final_mask[node] = False

        # ensure at least one action
        if not final_mask.any():
            final_mask[:] = True
            if 0 <= uav_id < len(final_mask):
                final_mask[uav_id] = False

        q_values_masked = q_values.copy()
        q_values_masked[~final_mask] = -1e9

        if logits_noise_std and logits_noise_std > 0:
            q_values_masked = q_values_masked + np.random.normal(
                0, logits_noise_std, size=q_values_masked.shape
            )

        # epsilon-greedy
        if np.random.rand() < epsilon:
            available = np.where(final_mask)[0]
            action = int(np.random.choice(available))
        else:
            action = int(np.argmax(q_values_masked))

        return action

    def select_action_test(
        self,
        state,
        uav_id,
        mask=None,
        visited_nodes=None,
        *,
        epsilon=0.0,            # 測試預設不隨機
        logits_noise_std=0.0,   # 測試不加 noise
        seed=None,
        return_debug=True
    ):
        """
        Test-only action selection.
        Deterministic greedy by default.
        """

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # ---------- forward ----------
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).view(1, -1)

        with torch.no_grad():
            q = self.eval_net(state_t)[0].detach().cpu().numpy()  # (A,)

        action_dim = q.shape[0]

        # ---------- build mask ----------
        if mask is None:
            final_mask = np.ones(action_dim, dtype=bool)
        else:
            final_mask = np.asarray(mask, dtype=bool)

        # exclude self
        if 0 <= uav_id < action_dim:
            final_mask[uav_id] = False

        # exclude visited
        if visited_nodes:
            for v in visited_nodes:
                if 0 <= v < action_dim:
                    final_mask[v] = False

        # fallback (avoid all False)
        if not final_mask.any():
            final_mask[:] = True
            if 0 <= uav_id < action_dim:
                final_mask[uav_id] = False

        # ---------- mask Q ----------
        q_masked = q.copy()
        q_masked[~final_mask] = -1e9

        # optional noise (test default off)
        if logits_noise_std > 0:
            q_masked += np.random.normal(
                0.0, logits_noise_std, size=q_masked.shape
            )

        # ---------- epsilon-greedy ----------
        if np.random.rand() < epsilon:
            avail = np.where(final_mask)[0]
            action = int(np.random.choice(avail))
            mode = "random"
        else:
            action = int(np.argmax(q_masked))
            mode = "greedy"

        if return_debug:
            debug = {
                "q_raw": q,
                "q_masked": q_masked,
                "final_mask": final_mask,
                "mode": mode,
                "uav_id": uav_id,
                "visited_nodes": visited_nodes,
                "epsilon": epsilon,
            }
            return action, debug

        return action

    def train(self, replay_buffer, batch_size=64):
        """
        replay_buffer.sample(batch_size) should return:
          state, action, next_state, reward, cost, dones
        where each is a torch.Tensor on CPU (we'll move to device).
        """
        state, action, next_state, reward, cost, dones = replay_buffer.sample(batch_size)

        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device).long().view(-1)            # (B,)
        reward = reward.to(self.device).view(-1)                   # (B,)
        cost = cost.to(self.device).view(-1)                       # (B,)
        dones = dones.to(self.device).view(-1).float()             # (B,)

        # -------- reward Q update --------
        q_eval_all = self.eval_net(state)                          # (B, A)
        q_eval = q_eval_all.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0]         # (B,)
            q_target = reward + self.gamma * q_next * (1.0 - dones)

        loss_r = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss_r.backward()
        self.optimizer.step()
        self.reward_loss_log.append(float(loss_r.item()))

        # -------- cost Q update --------
        c_eval_all = self.eval_cost_net(state)                     # (B, A)
        c_eval = c_eval_all.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            c_next = self.target_cost_net(next_state).max(1)[0]    # (B,)
            c_target = cost + self.gamma * c_next * (1.0 - dones)

        loss_c = self.cost_loss_fn(c_eval, c_target)
        self.cost_optimizer.zero_grad()
        loss_c.backward()
        self.cost_optimizer.step()
        self.cost_loss_log.append(float(loss_c.item()))

        # -------- hard target update --------
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_cost_net.load_state_dict(self.eval_cost_net.state_dict())

        self.learn_step_counter += 1

        # epsilon schedule (optional)
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)

        return loss_r.item(), loss_c.item()

    def save(self, path_prefix):
        torch.save(self.eval_net.state_dict(), path_prefix + "_qnet.pth")
        torch.save(self.optimizer.state_dict(), path_prefix + "_opt.pth")

        torch.save(self.eval_cost_net.state_dict(), path_prefix + "_cnet.pth")
        torch.save(self.cost_optimizer.state_dict(), path_prefix + "_copt.pth")

    def load(self, path_prefix):
        self.eval_net.load_state_dict(torch.load(path_prefix + "_qnet.pth", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(path_prefix + "_opt.pth", map_location=self.device))
        self.target_net = copy.deepcopy(self.eval_net)

        # cost nets (若你之前存的是舊版不含 cost，這邊會噴錯；那就先用 try/except)
        self.eval_cost_net.load_state_dict(torch.load(path_prefix + "_cnet.pth", map_location=self.device))
        self.cost_optimizer.load_state_dict(torch.load(path_prefix + "_copt.pth", map_location=self.device))
        self.target_cost_net = copy.deepcopy(self.eval_cost_net)
