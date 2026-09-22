import numpy as np
import torch

from centralized_movement import (
    MOVEMENT_STATE_DIM,
    NUM_UAV,
    movement_mask_from_state,
    project_joint_action,
    validate_movement_mask,
)
from experiment_config import (
    TASK_POTENTIAL_BETA_COM,
    TASK_POTENTIAL_BETA_SEARCH,
    TASK_POTENTIAL_BETA_VS,
)


def _to_np_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.float32, copy=False)

class ReplayBufferContinuous:
    def __init__(self, state_dim, action_dim, max_size=50_000, n_step=3, gamma=0.99, rng=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        self.rng = rng or np.random.default_rng(0)

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
        self.total_added += 1

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
                return self.rng.integers(0, size, size=k)
            replace = len(pool) < k
            return self.rng.choice(pool, size=k, replace=replace)

        ind = np.concatenate([
            _pick(same_idx, n_same),
            _pick(neigh_idx, n_nei),
            self.rng.integers(0, size, size=n_mix),
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
    """One-step joint replay supporting Dinkelbach reward reconstruction and stored terminal ratio objectives."""

    def __init__(self, state_dim, action_dim, max_size=50_000, rng=None):
        self.state_dim = int(state_dim)
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        self.rng = rng or np.random.default_rng(0)
        self.n_step = 1
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.current_movement_mask = np.zeros(
            (self.max_size, NUM_UAV), dtype=bool
        )
        self.next_movement_mask = np.zeros(
            (self.max_size, NUM_UAV), dtype=bool
        )
        self.movement_mask_valid = np.zeros((self.max_size, 1), dtype=bool)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.delivered_mbits = np.zeros((self.max_size, 1), dtype=np.float32)
        self.total_mobility_energy = np.zeros((self.max_size, 1), dtype=np.float32)
        self.ratio_objective_reward = np.zeros(
            (self.max_size, 1), dtype=np.float32
        )
        self.c9_penalty = np.zeros((self.max_size, 1), dtype=np.float32)
        self.c10_penalty = np.zeros((self.max_size, 1), dtype=np.float32)
        self.com_range_penalty = np.zeros((self.max_size, 1), dtype=np.float32)
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
        c9_penalty,
        c10_penalty,
        com_range_penalty,
        ratio_objective_reward=0.0,
        current_movement_mask=None,
        next_movement_mask=None,
    ):
        index = self.ptr
        state_array = _to_np_float32(state)
        next_state_array = _to_np_float32(next_state)
        self.state[index] = state_array
        self.next_state[index] = next_state_array
        if (current_movement_mask is None) != (next_movement_mask is None):
            raise ValueError(
                "current_movement_mask and next_movement_mask must be provided together"
            )
        if current_movement_mask is None and self.state_dim == MOVEMENT_STATE_DIM:
            current_movement_mask = movement_mask_from_state(state_array)
            next_movement_mask = movement_mask_from_state(next_state_array)
        if current_movement_mask is not None:
            current_mask = validate_movement_mask(current_movement_mask)
            next_mask = validate_movement_mask(next_movement_mask)
            if current_mask.ndim != 1 or next_mask.ndim != 1:
                raise ValueError("one replay transition requires unbatched movement masks")
            self.current_movement_mask[index] = np.asarray(current_mask, dtype=bool)
            self.next_movement_mask[index] = np.asarray(next_mask, dtype=bool)
            self.movement_mask_valid[index, 0] = True
            self.action[index] = project_joint_action(
                action, movement_mask=current_mask
            )
        else:
            self.current_movement_mask[index] = False
            self.next_movement_mask[index] = False
            self.movement_mask_valid[index, 0] = False
            self.action[index] = _to_np_float32(action)
        self.not_done[index, 0] = 1.0 - float(bool(done))
        self.delivered_mbits[index, 0] = float(delivered_mbits)
        self.total_mobility_energy[index, 0] = float(total_mobility_energy)
        self.ratio_objective_reward[index, 0] = float(ratio_objective_reward)
        self.c9_penalty[index, 0] = float(c9_penalty)
        self.c10_penalty[index, 0] = float(c10_penalty)
        self.com_range_penalty[index, 0] = float(com_range_penalty)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_added += 1

    def diagnostics(self):
        if self.size == 0:
            oldest = newest = None
        elif self.size < self.max_size:
            oldest, newest = 0, self.size - 1
        else:
            oldest, newest = self.ptr, (self.ptr - 1) % self.max_size
        return {
            "capacity": int(self.max_size),
            "size": int(self.size),
            "write_pointer": int(self.ptr),
            "total_added": int(self.total_added),
            "wrapped": bool(self.total_added > self.max_size),
            "oldest_physical_index": oldest,
            "newest_physical_index": newest,
            "oldest_age": self.size - 1 if self.size else None,
            "newest_age": 0 if self.size else None,
        }

    def _reward_numpy(
        self,
        indices,
        current_lambda,
        gamma,
        beta_search=TASK_POTENTIAL_BETA_SEARCH,
        beta_vs=TASK_POTENTIAL_BETA_VS,
        beta_com=TASK_POTENTIAL_BETA_COM,
        reward_mode="dinkelbach",
        task_potential_enabled=True,
    ):
        not_done = self.not_done[indices]
        delivered = self.delivered_mbits[indices]
        energy = self.total_mobility_energy[indices]
        if reward_mode == "dinkelbach":
            objective = delivered - float(current_lambda) * energy
        elif reward_mode == "ratio":
            objective = self.ratio_objective_reward[indices].copy()
            objective[~np.isfinite(objective)] = 0.0
        else:
            raise ValueError(f"unsupported reward mode: {reward_mode}")
        del not_done, gamma, beta_search, beta_vs, beta_com
        penalty_scale = 1.0 if task_potential_enabled else 0.0
        reward = objective - penalty_scale * (
            self.c9_penalty[indices]
            + self.c10_penalty[indices]
            + self.com_range_penalty[indices]
        )
        return reward.astype(np.float32, copy=False)

    def sample(
        self,
        batch_size,
        current_lambda,
        gamma,
        beta_search=TASK_POTENTIAL_BETA_SEARCH,
        beta_vs=TASK_POTENTIAL_BETA_VS,
        beta_com=TASK_POTENTIAL_BETA_COM,
        reward_mode="dinkelbach",
        task_potential_enabled=True,
        include_movement_masks=False,
    ):
        if self.size <= 0:
            raise ValueError("Replay buffer is empty")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
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
        batch = (
            torch.from_numpy(self.state[indices]).to(self.device),
            torch.from_numpy(self.action[indices]).to(self.device),
            torch.from_numpy(self.next_state[indices]).to(self.device),
            torch.from_numpy(reward).to(self.device),
            torch.from_numpy(self.not_done[indices]).to(self.device),
        )
        if not include_movement_masks:
            return batch
        if not self.movement_mask_valid[indices].all():
            raise RuntimeError(
                "sampled joint replay transitions lack authoritative movement projection masks"
            )
        return batch + (
            torch.from_numpy(self.current_movement_mask[indices]).to(self.device),
            torch.from_numpy(self.next_movement_mask[indices]).to(self.device),
        )


class ReplayBufferDiscrete:
    def __init__(self, state_dim, action_dim, max_size=50_000, n_step=3, gamma=0.99, rng=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        self.rng = rng or np.random.default_rng(0)
        self.tag_gt = np.full((self.max_size, 1), -1, dtype=np.int16)
        self.transition_id = np.full((self.max_size,), -1, dtype=np.int64)

        self.state      = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((self.max_size,), dtype=np.int64)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1), dtype=np.float32)
        self.cost       = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done   = np.zeros((self.max_size, 1), dtype=np.float32)
        self.cost_next_state = np.zeros(
            (self.max_size, state_dim), dtype=np.float32
        )
        self.cost_not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.cost_ready = np.zeros((self.max_size, 1), dtype=bool)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffer = []
        self.latest_index_by_agent = {}
        self.index_by_transition_id = {}

    @torch.no_grad()
    def add(
        self, state, action, next_state, reward, cost, done, tag_gt=None,
        agent_id=None, transition_id=None,
    ):
        s  = _to_np_float32(state)
        a  = _to_np_float32(action)
        ns = _to_np_float32(next_state)
        r  = float(reward)
        d  = bool(done)
        c  = float(cost)
        tg = int(tag_gt) if tag_gt is not None else -1
        tid = int(transition_id) if transition_id is not None else -1

        self.n_step_buffer.append((s, a, ns, r, c, d, tg, tid))
        if len(self.n_step_buffer) < self.n_step:
            return

        R, C, next_s, done_flag = 0.0, 0.0, None, False
        for idx, (_, _, ns_i, r_i, c_i, d_i, _, _) in enumerate(self.n_step_buffer):
            g = self.gamma ** idx
            R += g * float(r_i)
            C += g * float(c_i)
            if d_i:
                done_flag = True
                next_s = ns_i
                break
        if not done_flag:
            next_s = self.n_step_buffer[-1][2]

        s0, a0, _, _, _, _, tg0, tid0 = self.n_step_buffer[0]

        i = self.ptr
        previous_transition_id = int(self.transition_id[i])
        if previous_transition_id >= 0:
            self.index_by_transition_id.pop(previous_transition_id, None)
        self.state[i]       = s0
        self.action[i]      = a0
        self.next_state[i]  = next_s
        self.reward[i, 0]   = R
        self.cost[i, 0]     = C
        self.not_done[i, 0] = 1.0 - float(done_flag)
        self.tag_gt[i, 0]   = tg0  # ★ 寫入場景標籤
        self.transition_id[i] = tid0
        self.cost_next_state[i] = next_s if tid0 < 0 else 0.0
        self.cost_not_done[i, 0] = (
            1.0 - float(done_flag) if tid0 < 0 else 0.0
        )
        self.cost_ready[i, 0] = tid0 < 0
        if tid0 >= 0:
            self.index_by_transition_id[tid0] = i
        if agent_id is not None:
            self.latest_index_by_agent[int(agent_id)] = i

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_added += 1

        if d:
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.pop(0)

    @property
    def cost_ready_size(self):
        return int(np.count_nonzero(self.cost_ready[: self.size, 0]))

    def attach_cost_transition(self, transition_id, next_state, cost, done):
        """Attach packet-chain cost causality to an existing reward row."""

        transition_id = int(transition_id)
        index = self.index_by_transition_id.get(transition_id)
        if index is None:
            matches = np.flatnonzero(
                self.transition_id[: self.size] == transition_id
            )
            if matches.size != 1:
                return False
            index = int(matches[0])
            self.index_by_transition_id[transition_id] = index
        if int(self.transition_id[index]) != transition_id:
            self.index_by_transition_id.pop(transition_id, None)
            return False
        if bool(self.cost_ready[index, 0]):
            raise AssertionError("packet-chain cost transition was attached twice")
        next_state_array = _to_np_float32(next_state)
        if next_state_array.shape != self.cost_next_state[index].shape:
            raise ValueError("cost next-state shape is incompatible with replay")
        value = float(cost)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("packet-chain cost must be finite and non-negative")
        self.cost_next_state[index] = next_state_array
        self.cost[index, 0] = value
        self.cost_not_done[index, 0] = 1.0 - float(bool(done))
        self.cost_ready[index, 0] = True
        return True

    def sample(self, batch_size, include_cost=False):
        ind = self.rng.integers(0, self.size, size=batch_size)
        return self._gather(ind, include_cost=include_cost)

    def sample_cost(self, batch_size):
        ready = np.flatnonzero(self.cost_ready[: self.size, 0])
        if ready.size == 0:
            raise ValueError("routing replay has no packet-chain cost transitions")
        indices = self.rng.choice(ready, size=int(batch_size), replace=True)
        return (
            torch.from_numpy(self.state[indices]).to(self.device),
            torch.from_numpy(self.action[indices]).to(self.device),
            torch.from_numpy(self.cost_next_state[indices]).to(self.device),
            torch.from_numpy(self.cost[indices]).to(self.device),
            torch.from_numpy(self.cost_not_done[indices]).to(self.device),
        )

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
                return self.rng.integers(0, size, size=k)
            replace = len(pool) < k
            return self.rng.choice(pool, size=k, replace=replace)

        ind = np.concatenate([
            _pick(same_idx, n_same),
            _pick(neigh_idx, n_nei),
            self.rng.integers(0, size, size=n_mix),
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
