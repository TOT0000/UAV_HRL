import numpy as np
import torch

def _to_np_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.float32, copy=False)

class ReplayBufferContinuous:
    def __init__(self, state_dim, action_dim, max_size=int(2e5), n_step=3, gamma=0.99):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        

        # 預配置為 float32，節省記憶體
        self.state      = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done   = np.zeros((self.max_size, 1), dtype=np.float32)
        self.tag_gt = np.full((self.max_size, 1), -1, dtype=np.int16)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffer = []  # 只放純數值/np.float32

    @torch.no_grad()
    def add(self, state, action, next_state, reward, done, tag_gt=None):
        s  = _to_np_float32(state)
        a  = _to_np_float32(action)
        ns = _to_np_float32(next_state)
        r  = float(reward)
        # c  = float(cost)
        d  = bool(done)
        tg = int(tag_gt) if tag_gt is not None else -1
        
        self.n_step_buffer.append((s, a, ns, r, d, tg))
        if len(self.n_step_buffer) < self.n_step:
            return

        R, next_s, done_flag = 0.0, None, False
        for idx, (_, _, ns_i, r_i, d_i, _) in enumerate(self.n_step_buffer):
            g = self.gamma ** idx
            R += g * float(r_i)
            # C += g * float(c_i)
            if d_i:
                done_flag = True
                next_s = ns_i
                break
        if not done_flag:
            next_s = self.n_step_buffer[-1][2]

        s0, a0, _, _, _, tg0 = self.n_step_buffer[0]

        i = self.ptr
        self.state[i]      = s0
        self.action[i]     = a0
        self.next_state[i] = next_s
        self.reward[i, 0]  = R
        # self.cost[i, 0]    = C
        self.not_done[i, 0]= 1.0 - float(done_flag)
        self.tag_gt[i, 0]   = tg0 

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if d:
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.pop(0)

    # @torch.no_grad()
    # def add_TD3(self, state, action, next_state, reward, done, cost):
    #     # 和 add() 一樣，先轉成乾淨數值
    #     s  = _to_np_float32(state)
    #     a  = _to_np_float32(action)
    #     ns = _to_np_float32(next_state)
    #     r  = float(reward)
    #     d  = bool(done)
    #     c  = float(cost)

    #     self.n_step_buffer.append((s, a, ns, r, d, c))
    #     if len(self.n_step_buffer) < self.n_step:
    #         return

    #     R, next_s, done_flag = 0.0, None, False
    #     for idx, (_, _, ns_i, r_i, d_i, _) in enumerate(self.n_step_buffer):
    #         R += (self.gamma ** idx) * float(r_i)
    #         if d_i:
    #             done_flag = True
    #             next_s = ns_i
    #             break
    #     if not done_flag:
    #         next_s = self.n_step_buffer[-1][2]

    #     s0, a0, _, _, _, c0 = self.n_step_buffer[0]
    #     i = self.ptr
    #     self.state[i]       = s0
    #     self.action[i]      = a0
    #     self.next_state[i]  = next_s
    #     self.reward[i, 0]   = R
    #     self.not_done[i, 0] = 1.0 - float(done_flag)
    #     self.cost[i, 0]     = c0  # 或者依需求放入累積 cost

    #     self.ptr  = (self.ptr + 1) % self.max_size
    #     self.size = min(self.size + 1, self.max_size)

    #     if d:
    #         self.n_step_buffer.clear()
    #     else:
    #         self.n_step_buffer.pop(0)
    def sample_by_tag(self, batch_size, curr_tag, neighbor_step=2,
                      p_same=0.6, p_neighbor=0.2, include_cost=False):
        size = self.size
        assert size > 0, "Replay buffer is empty."
        n_same = int(batch_size * p_same)
        n_nei  = int(batch_size * p_neighbor)
        n_mix  = batch_size - n_same - n_nei

        tags = self.tag_gt[:size, 0]
        all_idx = np.arange(size)
        same_idx = all_idx[tags == int(curr_tag)]
        neigh_tags = {int(curr_tag - neighbor_step), int(curr_tag + neighbor_step)}
        neigh_idx = all_idx[np.isin(tags, list(neigh_tags))]

        def _pick(pool, k):
            if k <= 0:
                return np.empty((0,), dtype=np.int64)
            if len(pool) == 0:
                return np.random.randint(0, size, size=k)
            replace = len(pool) < k
            return np.random.choice(pool, size=k, replace=replace)

        ind = np.concatenate([
            _pick(same_idx, n_same),
            _pick(neigh_idx, n_nei),
            np.random.randint(0, size, size=n_mix),
        ])
        return self._gather(ind)

    def _gather(self, ind):
        s  = torch.from_numpy(self.state[ind]).to(self.device)
        a  = torch.from_numpy(self.action[ind]).to(self.device)
        ns = torch.from_numpy(self.next_state[ind]).to(self.device)
        r  = torch.from_numpy(self.reward[ind]).to(self.device)
        nd = torch.from_numpy(self.not_done[ind]).to(self.device)
        ng = torch.from_numpy(self.tag_gt[ind]).to(self.device)
        
        return s, a, ns, r, nd, ng


class ReplayBufferJoint:
    """One-step joint replay with rewards reconstructed from current lambda."""

    def __init__(self, state_dim, action_dim, max_size=int(2e5)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.n_step = 1
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.delivered_mbits = np.zeros((self.max_size, 1), dtype=np.float32)
        self.total_mobility_energy = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_search_t = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_search_t1 = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_vs_t = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_vs_t1 = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_com_t = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi_com_t1 = np.zeros((self.max_size, 1), dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def add(
        self,
        state,
        action,
        next_state,
        done,
        delivered_mbits,
        total_mobility_energy,
        phi_search_t,
        phi_search_t1,
        phi_vs_t,
        phi_vs_t1,
        phi_com_t,
        phi_com_t1,
    ):
        index = self.ptr
        self.state[index] = _to_np_float32(state)
        self.action[index] = _to_np_float32(action)
        self.next_state[index] = _to_np_float32(next_state)
        self.not_done[index, 0] = 1.0 - float(bool(done))
        self.delivered_mbits[index, 0] = float(delivered_mbits)
        self.total_mobility_energy[index, 0] = float(total_mobility_energy)
        self.phi_search_t[index, 0] = float(phi_search_t)
        self.phi_search_t1[index, 0] = float(phi_search_t1)
        self.phi_vs_t[index, 0] = float(phi_vs_t)
        self.phi_vs_t1[index, 0] = float(phi_vs_t1)
        self.phi_com_t[index, 0] = float(phi_com_t)
        self.phi_com_t1[index, 0] = float(phi_com_t1)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _reward_numpy(
        self,
        indices,
        current_lambda,
        gamma,
        beta_search=1.0,
        beta_vs=1.0,
        beta_com=1.0,
        reward_mode="dinkelbach",
        task_potential_enabled=True,
    ):
        not_done = self.not_done[indices]
        delivered = self.delivered_mbits[indices]
        energy = self.total_mobility_energy[indices]
        if reward_mode == "dinkelbach":
            objective = delivered - float(current_lambda) * energy
        elif reward_mode == "ratio":
            objective = np.divide(
                delivered,
                energy,
                out=np.zeros_like(delivered),
                where=np.isfinite(energy) & (energy > 0.0),
            )
            objective[~np.isfinite(objective)] = 0.0
        else:
            raise ValueError(f"unsupported reward mode: {reward_mode}")
        shaping_scale = 1.0 if task_potential_enabled else 0.0
        reward = objective + shaping_scale * (
            float(beta_search)
            * (float(gamma) * not_done * self.phi_search_t1[indices] - self.phi_search_t[indices])
            + float(beta_vs)
            * (float(gamma) * not_done * self.phi_vs_t1[indices] - self.phi_vs_t[indices])
            + float(beta_com)
            * (float(gamma) * not_done * self.phi_com_t1[indices] - self.phi_com_t[indices])
        )
        return reward.astype(np.float32, copy=False)

    def sample(
        self,
        batch_size,
        current_lambda,
        gamma,
        beta_search=1.0,
        beta_vs=1.0,
        beta_com=1.0,
        reward_mode="dinkelbach",
        task_potential_enabled=True,
    ):
        if self.size <= 0:
            raise ValueError("Replay buffer is empty")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        reward = self._reward_numpy(
            indices,
            current_lambda=current_lambda,
            gamma=gamma,
            beta_search=beta_search,
            beta_vs=beta_vs,
            beta_com=beta_com,
            reward_mode=reward_mode,
            task_potential_enabled=task_potential_enabled,
        )
        return (
            torch.from_numpy(self.state[indices]).to(self.device),
            torch.from_numpy(self.action[indices]).to(self.device),
            torch.from_numpy(self.next_state[indices]).to(self.device),
            torch.from_numpy(reward).to(self.device),
            torch.from_numpy(self.not_done[indices]).to(self.device),
        )


class ReplayBufferDiscrete:
    def __init__(self, state_dim, action_dim, max_size=int(2e5), n_step=3, gamma=0.99):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.tag_gt = np.full((self.max_size, 1), -1, dtype=np.int16)

        self.state      = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((self.max_size,), dtype=np.int64)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1), dtype=np.float32)
        self.cost       = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done   = np.zeros((self.max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffer = []

    @torch.no_grad()
    def add(self, state, action, next_state, reward, cost, done, tag_gt=None):
        s  = _to_np_float32(state)
        a  = _to_np_float32(action)
        ns = _to_np_float32(next_state)
        r  = float(reward)
        d  = bool(done)
        c  = float(cost)
        tg = int(tag_gt) if tag_gt is not None else -1

        self.n_step_buffer.append((s, a, ns, r, c, d, tg))
        if len(self.n_step_buffer) < self.n_step:
            return

        R, C, next_s, done_flag = 0.0, 0.0, None, False
        for idx, (_, _, ns_i, r_i, c_i, d_i, _) in enumerate(self.n_step_buffer):
            g = self.gamma ** idx
            R += g * float(r_i)
            C += g * float(c_i)
            if d_i:
                done_flag = True
                next_s = ns_i
                break
        if not done_flag:
            next_s = self.n_step_buffer[-1][2]

        s0, a0, _, _, _, _, tg0 = self.n_step_buffer[0]

        i = self.ptr
        self.state[i]       = s0
        self.action[i]      = a0
        self.next_state[i]  = next_s
        self.reward[i, 0]   = R
        self.cost[i, 0]     = C
        self.not_done[i, 0] = 1.0 - float(done_flag)
        self.tag_gt[i, 0]   = tg0  # ★ 寫入場景標籤

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if d:
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.pop(0)


    def sample(self, batch_size, include_cost=False):
        ind = np.random.randint(0, self.size, size=batch_size)
        return self._gather(ind, include_cost=include_cost)

    def sample_by_tag(self, batch_size, curr_tag, neighbor_step=2,
                      p_same=0.6, p_neighbor=0.2, include_cost=False):
        size = self.size
        assert size > 0, "Replay buffer is empty."
        n_same = int(batch_size * p_same)
        n_nei  = int(batch_size * p_neighbor)
        n_mix  = batch_size - n_same - n_nei

        tags = self.tag_gt[:size, 0]
        all_idx = np.arange(size)
        same_idx = all_idx[tags == int(curr_tag)]
        neigh_tags = {int(curr_tag - neighbor_step), int(curr_tag + neighbor_step)}
        neigh_idx = all_idx[np.isin(tags, list(neigh_tags))]

        def _pick(pool, k):
            if k <= 0:
                return np.empty((0,), dtype=np.int64)
            if len(pool) == 0:
                return np.random.randint(0, size, size=k)
            replace = len(pool) < k
            return np.random.choice(pool, size=k, replace=replace)

        ind = np.concatenate([
            _pick(same_idx, n_same),
            _pick(neigh_idx, n_nei),
            np.random.randint(0, size, size=n_mix),
        ])
        return self._gather(ind, include_cost=include_cost)

    def _gather(self, ind, include_cost=False):
        s  = torch.from_numpy(self.state[ind]).to(self.device)
        a  = torch.from_numpy(self.action[ind]).to(self.device)
        ns = torch.from_numpy(self.next_state[ind]).to(self.device)
        r  = torch.from_numpy(self.reward[ind]).to(self.device)
        c  = torch.from_numpy(self.cost[ind]).to(self.device)
        nd = torch.from_numpy(self.not_done[ind]).to(self.device)
        if include_cost:
            return s, a, ns, r, c, nd
        else:
            return s, a, ns, r, c, nd
