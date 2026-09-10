# Relay snapshot 規劃與 TD3 movement shaping 變更

日期：2026-09-10。分支：`feature/centralized-td3`。
修改前 HEAD：`5119e4efb28844a3f2d22166ed78b376ce17f283`。

## 修改前的 production 行為

- Relay 數量為 `floor(discovered_roi_count / 2)`，task 沒有位置目標。
- K-KM 先以 Relay utility 做 Hungarian 配額分配，再執行最多兩輪 FOV／COM KM。
- KM 將 Relay／FOV／COM 放入同一輪 KM；Random 原先只有一輪混合隨機分配。
- Relay utility 使用 backlog-weighted receive capacity 與 shortest-path forward score；movement shaping 使用 receive／forward 的容量與距離進度混合。
- 發現 RoI 以外，Search coverage release 也會觸發完整重分配。
- Movement observation 為 675-D，checkpoint schema 為 27。

## 修改後的資料流

1. 保留初始化的任務設定；之後只在新 RoI 的既有 movement assignment boundary 完整重分配。
2. 先分配 FOV／COM：K-KM 最多兩輪、KM 一輪、Random 兩輪。保留 gateway、availability、Search reservation 與 FOV+COM 相容性限制。
3. Planning sources 為有效且未過期的 in-air FOV／COM queue owners，加上本輪 prospective service UAV。SR ground backlog 不計；prospective zero backlog 權重為零。
4. 使用全部合法 routing UAV 節點，以 3D 400 m 圖與 reverse BFS 求 GS reachability。U2G 判斷共用原 production 的距離 eligibility，不參考 channel samples。
5. 各斷聯來源取最近的實體 source／GS component endpoints，建立 `ceil(d/400)-1` 個等距 candidates；以固定順序逐一測試 zero-loss deletion。
6. 從 deterministic witnesses 收集固定 neighbors。兩個以上必要鄰居即標記 shared；以 bounded minimax 更新位置。數值失敗有 deterministic fallback；規劃時無效 merge 會恢復 candidates，必要時回到原始 bridge-chain witnesses。
7. 不足時每一步重新計算 marginal backlog loss，按 loss、斷聯來源數、slot ID 刪除。保留 budget 前 required、available、assigned、shortage 四個不同數值。
8. K-KM／KM 按 slot 重要性，依 raw 3D distance greedy 配對剩餘 UAV；Random 使用正式 assignment RNG。Relay 永遠 exclusive，沒有 Relay utility matrix／Hungarian。
9. 兩次 RoI boundary 間固定 slot／UAV／anchor／neighbor 身分，只更新位置。Search release 只轉 Hover，不再重新分配服務或 Relay。

## Movement 與 routing

每架 UAV 的舊 Relay 8 個欄位改為 3 個 normalized target displacement，非 Relay 與 masked observation 為零。Movement observation 現為 **595-D，schema 6**；checkpoint schema **28**，舊 checkpoint 在 restore weights 前拒絕並要求重新訓練。

Relay potential 為 `0.3 * exp(-distance/400) + 0.7 * min(clip(C_bar/C_ref,0,1))`。U2U capacity 為 deterministic Rician expectation；GS neighbor 使用 expected-path-loss A2G estimate。Reference 是 `reference_u2u_max_capacity_mbps(10 MHz)`，各 slot diagnostics 記錄其數值。Virtual neighbor 的 link term 使用分配到該 slot 的實體 UAV 位置。

Relay 外層權重與 COM 相同，沿用既有 `gamma*Phi_next-Phi_current`、boundary replay continuity 與 terminal-zero convention。沒有額外 per-step、arrival 或 distance bonus；`no_task_potential` 可完整停用。

Safe-DDQN 的 143-D observation、48-D joint movement action、routing actions／masks、next-hop semantics、FDMA、FIFO、deadlines 與 routing reward 均保留。唯一與 GS API 相關的整理是把相同的距離 eligibility 提取成可接受虛擬位置的共用方法。

## Diagnostics 與限制

`relay_diagnostics.json` v3 包含完整 assignment planning events、candidate removal tests、budget loss tests、mapping／raw assignment distances、角色變更，以及各 movement boundary 的 compact targets／potential／feasibility／physical reachability snapshots。

Evaluation aggregation 分別加總 episode-final required、available、assigned、shortage，並統計 disconnected source-boundary 與 infeasible slot-boundary observations。零 Relay 的各方法仍使用一致 schema 與輸出檔案。Validation 檢查 count／mapping／summary 一致性及有限值。

這是 snapshot greedy heuristic，沒有全域最少 Relay 保證，也不預測 UAV 移動後的圖。Physical witness 中的 UAV 自己也可能移動；anchors 分離後可能永久斷聯，直到下一個新 RoI。Budget pruning 可能留下需要已移除 virtual neighbor 的 slot，此缺失 link 為零並記錄 infeasible。若沒有任何實體 GS component，來源記為 unsupported，不發明 projection／gateway 規則。

## 主要修改檔案

| 檔案 | 用途 |
| --- | --- |
| `relay_contract.py` | 圖、planning、minimax、pruning、greedy pairing、位置更新與 potential |
| `Task_assignment.py`、`Simulator.py` | Service-first assignment、exclusive Relay、RoI boundary 與 Search release |
| `centralized_movement.py`、`movement_feature_schema.py`、`observation_strategy.py` | 595-D observation、mask 與 Relay shaping |
| `HRL_task_aware.py`、`experiment_config.py`、`training_checkpoint.py` | 有效 backlog、既有 replay lifecycle、COM 共用權重、schema gating |
| `relay_diagnostics.py`、`evaluation_aggregation.py` | 一致 artifacts、validation、aggregation |
| `tests/test_virtual_relay_planning.py`、`tests/test_relay_task_contract.py` | 新演算法及 training／evaluation／replay regression |
| 其他相關 tests、`EXPERIMENTS.md` | 更新舊 schema、dimension 與 release 行為的契約 |

## 驗證方式

使用既有 `C:/Users/user/anaconda3/envs/LLM_HRL/python.exe`，pytest 關閉 cache provider。未執行長時間正式訓練。

- Targeted suite：Relay planning／potential／artifacts、assignment、UAV16、channel boundary、centralized training。
- Training smoke：1 episode × 3 movement seconds，warmup=0、batch=1，實際更新 TD3 actor 與 critic，且具有 Relay 任務。
- Evaluation smoke：K-KM／KM／Random 三方法各先產生小型 model checkpoint，再經正式 checkpoint loading 執行 1-second evaluation，驗證零 Relay artifact round trip。
- 額外固定 seed 的 80 組稀疏 3D snapshots 檢查，涵蓋最多 12 個 budget 前 Relay slots。
- 全部 regression：`python -u -m pytest tests -q -p no:cacheprovider --tb=short --disable-warnings`。

最終完整 regression：**661 passed、373 subtests passed**，355.58 秒。
Targeted related suite：**93 passed、27 subtests passed**；另執行含最新三鄰居共用情境的完整 Relay planner suite：**37 passed**。
Preflight／assignment fixture 修正後：**24 passed、15 subtests passed**。
完整 suite 包含上述 training 與 evaluation smoke；沒有未解決的測試失敗。
Pytest 另輸出 228 個第三方／既有 warning，未停用其收集，只以 `--disable-warnings` 隱藏冗長列表。
`git diff --check` 通過。過程中曾失敗的舊 schema／release／preflight fixture 已依新契約更新。
