# import gym
import torch
import copy
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _bellman_target(immediate_value, next_value, not_done, discount):
    return immediate_value + not_done * discount * next_value


"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim):
        super(QNetworkCNN, self).__init__()

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(8960, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def forward(self, inp):
        inp = inp.view((1, 3, 210, 160))
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1


class DDQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
        eta=0.1,
    ):
        self.q_network = QNetwork(action_dim, state_dim, hidden_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.cost_network = QNetwork(action_dim, state_dim, hidden_dim).to(device)
        self.target_cost_network = copy.deepcopy(self.cost_network)
        self.cost_optimizer = torch.optim.Adam(self.cost_network.parameters(), lr=lr)


        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.eta = eta

        self.loss_log = []
        self.cost_loss_log = []
        self.num_training = 0
    def select_action(self, state, uav_id, mask=None, visited_nodes=None, epsilon=0.5, logits_noise_std=0.5, eta=None):
        if eta is None:
            eta = self.eta
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            q_r = self.q_network(state_t).cpu().numpy().flatten()      # reward Q
            q_c = self.cost_network(state_t).cpu().numpy().flatten()   # cost Q
        # if np.random.rand() < 0.001:  # 0.1% 機率印，避免爆log
        #     q_c_eff = np.maximum(q_c, 0.0)
        #     qt = q_r - eta * q_c_eff
        #     # qt = q_r - float(eta) * q_c
        #     print(f"[lag] eta={eta:.4f} "
        #         f"q_r(min,max)=({q_r.min():.3f},{q_r.max():.3f}) "
        #         f"q_c(min,max)=({q_c_eff.min():.3f},{q_c_eff.max():.3f}) "
        #         f"q_t(min,max)=({qt.min():.3f},{qt.max():.3f})")
        # Lagrangian 合成（越大越好，所以 cost 用減的）
        q_c_eff = np.maximum(q_c, 0.0)
        q_values = q_r - float(eta) * q_c_eff
        # ----------  mask ----------
        # 1. 先從「外部傳入的 mask」開始，如果沒有就預設全 1
        if mask is None:
            final_mask = np.ones_like(q_values, dtype=bool)
        else:
            final_mask = np.array(mask, dtype=bool)

        # 2. 排除自己
        if 0 <= uav_id < len(final_mask):
            final_mask[uav_id] = False

        # 3. 排除已走過的無人機
        if visited_nodes:
            for node in visited_nodes:
                if 0 <= node < len(final_mask):
                    final_mask[node] = False

        # 4. 確保至少還有一個動作（避免全 False）
        if not final_mask.any():
            # fallback：如果真的全不能選，先恢復成全 True 再排除自己
            final_mask[:] = True
            if 0 <= uav_id < len(final_mask):
                final_mask[uav_id] = False

        # ---------- 套用 mask 到 Q 值 ----------
        # 非法動作的 Q 值設成極小
        q_values_masked = q_values.copy()
        q_values_masked[~final_mask] = -1e9

        # 加上 noise（可選）
        if logits_noise_std > 0:
            q_values_masked = q_values_masked + np.random.normal(
                0, logits_noise_std, size=q_values_masked.shape
            )

        # ---------- ε-greedy ----------
        if np.random.rand() < epsilon:
            available_actions = np.where(final_mask)[0]
            action = int(np.random.choice(available_actions))
        else:
            action = int(np.argmax(q_values_masked))

        return action

    def _routing_action_mask(self, next_state):
        num_uav = self.action_dim - 1
        state_dim = next_state.shape[1]

        if state_dim == 5 * num_uav + 20:
            mask_start = num_uav + 7
        elif state_dim == 6 * num_uav + 26:
            mask_start = num_uav + 8
        else:
            raise ValueError(
                f"Unsupported routing state layout: state_dim={state_dim}, "
                f"action_dim={self.action_dim}"
            )

        action_mask = next_state[
            :, mask_start : mask_start + self.action_dim
        ].bool()

        # Preserve select_action()'s existing empty-mask fallback.
        empty_rows = ~action_mask.any(dim=1)
        if empty_rows.any():
            action_mask = action_mask.clone()
            action_mask[empty_rows] = True
            uav_ids = next_state[empty_rows, :num_uav].argmax(dim=1)
            action_mask[empty_rows, uav_ids] = False

        return action_mask

    @torch.no_grad()
    def _safe_targets(self, next_state, reward, cost, not_done):
        next_q_online = self.q_network(next_state)
        next_c_online = self.cost_network(next_state)
        next_c_eff = torch.clamp(next_c_online, min=0.0)
        safe_scores = next_q_online - self.eta * next_c_eff

        action_mask = self._routing_action_mask(next_state)
        safe_scores = safe_scores.masked_fill(~action_mask, float("-inf"))
        next_actions = safe_scores.argmax(dim=1, keepdim=True)

        next_q_target = self.target_q_network(next_state).gather(
            1, next_actions
        ).squeeze(1)
        next_c_target = self.target_cost_network(next_state).gather(
            1, next_actions
        ).squeeze(1)

        target_q = _bellman_target(
            reward.squeeze(1), next_q_target, not_done.squeeze(1), self.gamma
        )
        target_c = _bellman_target(
            cost.squeeze(1), next_c_target, not_done.squeeze(1), self.gamma
        )
        return target_q, target_c, next_actions

    def train(self, replay_buffer, batch_size=64):

        state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

        q_values = self.q_network(state)
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        target_q, target_c, _ = self._safe_targets(
            next_state, reward, cost, not_done
        )
        
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_log.append(loss.item())

        # === cost critic 訓練 ===
        c_values = self.cost_network(state)
        c_values = c_values.gather(1, action.unsqueeze(1)).squeeze(1)

        cost_loss = F.mse_loss(c_values, target_c)
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()
        self.cost_loss_log.append(cost_loss.item())
        self.num_training += 1
        # print("q_values.shape =", q_values.shape)
        # print("target_q.shape =", target_q.shape)


    def update_target(self):
        """
        Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
        episode reward.
        """
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.cost_network.parameters(), self.target_cost_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_parameters(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def save(self, load_dir="param"):
        torch.save(self.q_network.state_dict(), load_dir + "_qnet.pth")
        torch.save(self.optimizer.state_dict(), load_dir + "_opt.pth")

    def load(self, path_prefix):
        self.q_network.load_state_dict(torch.load(path_prefix + "_qnet.pth"))
        self.optimizer.load_state_dict(torch.load(path_prefix + "_opt.pth"))
        self.target_q_network = copy.deepcopy(self.q_network)

    def plot_loss(self):
        plt.plot(self.loss_log, label="DDQN Reward Training Loss", linewidth=1, color = 'red')
        plt.plot(self.cost_loss_log, label="DDQN Cost Training Loss", linewidth=1, color = 'black')
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss (MSE)")
        plt.title("DDQN Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# def main(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.998, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
#          num_episodes=3000, seed=42, max_memory_size=5000, lr_gamma=1, lr_step=100, measure_step=100,
#          measure_repeats=100, hidden_dim=64, env_name='CartPole-v1', cnn=False, horizon=np.inf, render=True, render_step=50):
#     """
#     Remark: Convergence is slow. Wait until around episode 2500 to see good performance.

#     :param gamma: reward discount factor
#     :param lr: learning rate for the Q-Network
#     :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
#     :param eps: probability to take a random action during training
#     :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
#     :param eps_min: minimal value of "eps"
#     :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
#     batch of size "batch_size" from the memory.
#     :param batch_size: see above
#     :param update_repeats: see above
#     :param num_episodes: the number of episodes played in total
#     :param seed: random seed for reproducibility
#     :param max_memory_size: size of the replay memory
#     :param lr_gamma: learning rate decay for the Q-Network
#     :param lr_step: every "lr_step" episodes we decay the learning rate
#     :param measure_step: every "measure_step" episode the performance is measured
#     :param measure_repeats: the amount of episodes played in to asses performance
#     :param hidden_dim: hidden dimensions for the Q_network
#     :param env_name: name of the gym environment
#     :param cnn: set to "True" when using environments with image observations like "Pong-v0"
#     :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
#     :param render: if "True" renders the environment every "render_step" episodes
#     :param render_step: see above
#     :return: the trained Q-Network and the measured performances
#     """
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     env = gym.make(env_name)
#     env.seed(seed)

#     if cnn:
#         Q_1 = QNetworkCNN(action_dim=env.action_space.n).to(device)
#         Q_2 = QNetworkCNN(action_dim=env.action_space.n).to(device)
#     else:
#         Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
#                        hidden_dim=hidden_dim).to(device)
#         Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
#                        hidden_dim=hidden_dim).to(device)
#     # transfer parameters from Q_1 to Q_2
#     update_parameters(Q_1, Q_2)

#     # we only train Q_1
#     for param in Q_2.parameters():
#         param.requires_grad = False

#     optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
#     scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

#     memory = Memory(max_memory_size)
#     performance = []

#     for episode in range(num_episodes):
#         # display the performance
#         if (episode % measure_step == 0) and episode >= min_episodes:
#             performance.append([episode, evaluate(Q_1, env, measure_repeats)])
#             print("Episode: ", episode)
#             print("rewards: ", performance[-1][1])
#             print("lr: ", scheduler.get_last_lr()[0])
#             print("eps: ", eps)

#         state = env.reset()
#         memory.state.append(state)

#         done = False
#         i = 0
#         while not done:
#             i += 1
#             action = select_action(Q_2, env, state, eps)
#             state, reward, done, _ = env.step(action)

#             if i > horizon:
#                 done = True

#             # render the environment if render == True
#             if render and episode % render_step == 0:
#                 env.render()

#             # save state, action, reward sequence
#             memory.update(state, action, reward, done)

#         if episode >= min_episodes and episode % update_step == 0:
#             for _ in range(update_repeats):
#                 train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

#             # transfer new parameter from Q_1 to Q_2
#             update_parameters(Q_1, Q_2)

#         # update learning rate and eps
#         scheduler.step()
#         eps = max(eps*eps_decay, eps_min)

#     return Q_1, performance


# if __name__ == '__main__':
#     main()
