# 第七块实验冻结与前置检查报告

**生成时间:** 2026-07-15 00:10 CST
**评估器版本:** `netkeeper-evaluation-v3`
**本步状态:** 未修改文件，未运行正式 test

---

## 1. Checkpoint Gate 结果

### 判定: ✅ PASSED

正式 checkpoint `runs/rl-f27e74f349/best.pt` 通过全部硬门槛验证：

| 检查项 | 状态 |
|---|---|
| 来自 train split (`scenarios/train.jsonl`) | ✅ |
| 仅以 validation_total_reward 选择 | ✅ |
| 保存 resolved Experiment config | ✅ |
| 保存 dataset manifest SHA-256 | ✅ |
| 保存 seed (20260714) | ✅ |
| model_version = `rl-coma-v2` | ✅ |
| schema_version = `netkeeper-sim.schema.v1` | ✅ |
| strict load 成功 | ✅ |
| greedy dispatch 在 validation fixture 验证通过 | ✅ |
| checkpoint_status = `formal_validation_selected` | ✅ |
| test_split_accessed = false | ✅ |
| checkpoint SHA-256 已冻结 | ✅ |
| 非 `debug_unconverged` / 随机初始化 / 缺失来源 | ✅ |

**选出的 checkpoint 详情:**

- **路径:** `runs/rl-f27e74f349/best.pt`
- **SHA-256:** `6ec7a3fbba370dab7656916a1d6b7737b8a29ce3f9925be94b5c71038afeb728`
- **训练 episodes:** 100 (Experiment config，非 Debug)
- **选出 episode:** 59 (validation_total_reward = 0.02668009693884514)
- **训练配置:** Experiment (`rl_experiment.yaml`)，hidden_dim=64, gnn_layers=2, transformer_layers=2, heads=4, batch_size=4, max_steps=50
- **Resolved config SHA-256:** `6fd819c324df15ee94cf71078c10e61283aaaa0d2cb09fd41202fef7b3069195`
- **Adapter version:** `rl-coma-v2-adapter.2`, config hash `f5c198c346e63c884da1545f93bbf05201c038d19128c61105f76e066b4d7a5e`

**Validation 历史（每 20 episodes）:**

| Episode | Validation Total Reward |
|--:|--:|
| 19 | -0.0100 |
| 39 | 0.0233 |
| **59** | **0.0267** ← best |
| 79 | -0.0100 |
| 99 | -0.0200 |

---

## 2. 所有 Hash 汇总

### 2.1 评估 Manifest

| 项目 | 值 |
|---|---|
| 文件路径 | `configs/frozen_evaluation_manifest.json` |
| 文件 SHA-256 | `6ac39f3b742d057daf9a7039e39ffef8d9c9072834a265b3c325a210330da45e` |
| manifest_hash (内嵌) | `c271d19cc16277712cc46e8d854597de2b80b0efad80849516ad80549151e01c` |
| manifest_version | `netkeeper-evaluation-manifest-v2` |
| schema_version | `netkeeper-sim.schema.v1` |
| generation git_commit | `58ea8c9a45dd4fa441669cf0ae46c15ef314347d` |

### 2.2 数据源

| 项目 | 文件 | SHA-256 |
|---|---|---|
| Dataset manifest | `metadata/manifest.json` | `89420a8d43c75ddec9ef7937fb91afa31194eb1a49ea2a8863cde828bdd34397` |
| Static scenarios | `scenarios/test.jsonl` | `335496386f2f54704be6fca50a026deb20ef591176129c0fef3aacecb595b9cd` |
| Dynamic sequences | `dynamic_sequences/test.jsonl` | `0704bdd63058336471f7b895a15ac2ea0e1dd35c5078731efc75db55e3491220` |

### 2.3 方法版本与 Config Hash

| 方法 | Version | Config Hash |
|---|---|---|
| No Update | `baseline-v1` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| Random | `baseline-v1` | `7263f9fd4090b05cab0fea7081e7b70452831469a18170c012adaf39bcd105f9` |
| OSPF Default | `baseline-v1` | `c82f71493fa62c93c049389d7a171a38a68191e99126b3c393a644dc3aa79d2a` |
| Local Search OSPF | `baseline-v1` | `c0ddc2d301b272b4aa39f8229145c265dc2857cfb0518b4fa5147ecccb0ebb25` |
| NetKeeper Checkpoint | `rl-coma-v2-adapter.2` | `f5c198c346e63c884da1545f93bbf05201c038d19128c61105f76e066b4d7a5e` |

### 2.4 评估器 Config Hash

| 模式 | Config Hash |
|---|---|
| Static evaluator | `6868c853b55b704a8d13f1008339e5c1c1b075c562ac70ad14d9857bfbc1ad4e` |
| Dynamic evaluator | `ad26515846abad8dee0b14f9c94ad63c046017da36b6f547fd3435a75810b5f9` |

---

## 3. Frozen Evaluation Manifest 详情

### 3.1 静态场景 (500)

| 拓扑 | 场景数 |
|---|---|
| `zoo:belnet2006:49d8165b` | 125 |
| `zoo:darkstrand:aecb48e6` | 125 |
| `zoo:garr201005:f6fa51f4` | 125 |
| `zoo:garr:c6b0c695` | 125 |

**按难度:**

| 难度 | 数量 |
|---|--:|
| Easy | 175 |
| Medium | 200 |
| Hard | 125 |

### 3.2 动态序列 (100)

- 覆盖 6 种事件类型: `policy_add`, `policy_remove`, `traffic_scale`, `hotspot_change`, `link_failure_recovery`, `node_failure_recovery`
- 每个序列 max_steps=240，recovery_budget_steps=30
- 4 个测试拓扑，每个 25 条序列

### 3.3 Seeds

| 类型 | Seeds |
|---|---|
| Deterministic | `[20260714]` |
| Random | `[20260714, 20260715, 20260716]` |

---

## 4. 负载分组（互斥且完整）

四组互斥、覆盖全部 500 场景，无交叠：

| 负载组 | 规则 | 数量 |
|---|---|---|
| **Normal** | `load_level == Normal` + `traffic_pattern ∈ {gravity, diurnal}` | **166** |
| | └ gravity: 66, diurnal: 100 | |
| **Hotspot** | `load_level == Low` + `traffic_pattern == hotspot` | **125** |
| **Burst** | `load_level == Low` + `traffic_pattern == burst` | **42** |
| **High-load** | `load_level == High` (load_multiplier == 3.0) | **167** |
| | └ gravity: 84, burst: 83 | |

- 交叠验证: 四组两两交集均为 0 ✅
- 并集大小: 500 = 全部场景 ✅
- 定义已在 frozen manifest `analysis_groups.load` 中固定

---

## 5. 成功阈值与统计规则

| 参数 | 值 |
|---|---|
| max_steps (static) | 50 |
| max_steps (dynamic) | 240 |
| hold_steps (连续成功窗口) | 3 |
| recovery_budget_steps | 30 |
| Local Search candidate budget | 64 |
| Local Search deltas | ±1, ±2, ±4, ±8 |
| Traffic Shift 主版本 | paper-v1 |
| Traffic Shift 辅助版本 | project-v1 |
| 成功判定 | 连续 3 状态 PC=1.0（static）或回到 pre-event 水平（dynamic） |
| 失败处理 | convergence/recovery = null + censored |
| Random 统计 | 区分 scenario 间方差和 Random seed 内方差 |

---

## 6. 所有 Checkpoint 清单

| 路径 | SHA-256 | Episodes | 训练配置 | Metadata | 状态 |
|---|---|---|---|---|---|
| `checkpoints/rl_debug.pt` | `1506875b...` | ~20 | Debug (hidden_dim=32) | `debug_unconverged` | ❌ 不合格 |
| `checkpoints/rl_debug_review.pt` | `1d0b7b3d...` | ~20 | Debug | `debug_unconverged` | ❌ 不合格 |
| `runs/rl-122f3f3e29/latest.pt` | `dd5e7c04...` | 2 | Experiment 启动 smoke | 缺失 manifest | ❌ 不合格 |
| `runs/rl-f27e74f349/latest.pt` | `68b7b2a5...` | 100 | Experiment | episode 100 final | ⚠️ 非 best |
| **`runs/rl-f27e74f349/best.pt`** | **`6ec7a3fb...`** | **100** | **Experiment** | **formal_validation_selected** | **✅ 合格** |

---

## 7. 正式实验矩阵

### A. 方法 (5)

1. NetKeeper Checkpoint (`coma_dispatcher`, `rl-coma-v2-adapter.2`)
2. No Update (`baseline-v1`)
3. Random (`baseline-v1`, 3 seeds)
4. OSPF Default (`baseline-v1`)
5. Local Search OSPF (`baseline-v1`)

### B. 静态配置质量

- 全部 500 static test 场景
- 按 Easy/Medium/Hard 分组
- 主指标：final/best Policy Consistency、static success rate、convergence steps
- 辅助指标：MLU、paper-v1/project-v1 Traffic Shift、配置修改比例、runtime、failures

### C. 负载实验

- 使用 frozen manifest 互斥四组（见第 4 节）
- 主指标：final/best/worst MLU 和 congestion/success
- 同时报告 PC，避免以降低 MLU 换取策略违反

### D. Traffic Shift

- 使用相同 static trajectories
- 报告 paper-v1 和 project-v1 的 step/total/peak/mean
- 同时报告 PC 和配置修改比例
- 不挑成功场景

### E. 动态适应

- 全部 100 dynamic test sequences
- 6 种事件类型，每逻辑事件最多 30 recovery steps
- 按事件类型报告 recovered rate、recovery steps、pre/worst/recovered PC/MLU、Traffic Shift、配置修改、runtime、censored
- 主结果包含全部 100 条序列

### F. Seeds

- 确定性方法 + checkpoint: seed `20260714`
- Random: 预冻结 seeds `[20260714, 20260715, 20260716]`
- 汇总区分 scenario 间方差和 Random 重复 seed

---

## 8. 任务数预估

| 实验 | 方法数 | 场景/序列数 | Seeds | 任务数 |
|---|---|---|---|---|
| Static | 5 | 500 | 1 (det) + 3 (random) | **3,500** |
| Dynamic | 5 | 100 | 1 (det) + 3 (random) | **700** |
| **总计** | | | | **4,200** |

展开:
- Static: `no_update`(500) + `random`(1500) + `ospf_default`(500) + `local_search_ospf`(500) + `checkpoint`(500) = 3,500
- Dynamic: `no_update`(100) + `random`(300) + `ospf_default`(100) + `local_search_ospf`(100) + `checkpoint`(100) = 700

---

## 9. 当前运行状态

### Static formal v3 (`runs/block7-static-formal-v3`)

- **进度:** 1,230 / 3,500 (35.1%)
- 已完成: `no_update` 全 500 + `random` 730/1500
- 待完成: `random`(770) + `ospf_default`(500) + `local_search_ospf`(500) + `checkpoint`(500) = 2,270
- 方式: 8 worker shard 并行，v3 inter-process lock
- 无 dynamic task 启动

### Dynamic formal

- 尚未开始，等待 static 完成并验证

---

## 10. 环境资源

| 资源 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.2 GB VRAM |
| CUDA | 12.4 (PyTorch 2.6.0+cu124) |
| CUDA available | True |
| CPU cores | 16 |
| RAM | 15 GB (12 GB available) |
| Disk | 1 TB, 936 GB available (3% used) |
| Evaluator device | `cpu`（正式配置） |
| AMP | 训练时 `true`，推理时不适用（greedy/eval/no-grad） |

### Local Search 计算成本估算

- 每步最多 64 candidates + 1 no-op reference，每个 candidate 一次 sandbox simulator call
- Smoke 数据: 7 场景 pilot 记录 3,250 次 candidate evaluations
- 500 场景 × 50 steps ≈ 每 run 最多 500×50×65 = 1,625,000 simulator calls（实际远少，因为 early convergence 和no_update）
- Pilot 7 场景平均每 run ~464 candidate calls → 500 场景预计 ~33,000 candidate calls

---

## 11. P0/P1 评估

### P0（阻塞实验正确性）

无已知 P0。所有强制验证项已通过。

### P1（可能影响解释但非阻塞）

| 项目 | 说明 |
|---|---|
| 训练收敛有限 | Best validation reward = 0.027，训练 reward 全程为负。模型可能仅学到保守策略（倾向于 no_update），对复杂场景的优化能力有限 |
| 仅有 4 个测试拓扑 | 所有 500 test 场景分布在 4 个拓扑，泛化评估范围窄 |
| AMP 训练但未见 CUDA 验证数据 | Block 5 报告注明 "No manual CUDA forward/update ... was supplied" |
| 100 episodes vs 500 planned | `rl_experiment.yaml` 目标是 500 episodes，实际只跑了 100（best 在 ep 59 选出）。可能未充分训练 |

**以上 P1 不阻塞实验执行，但需要在最终论文中如实报告。**

---

## 12. 运行顺序与预计时间

### 推荐顺序

1. **继续 static formal v3** — 已在运行中（35.1%），预计还需 2-4 小时（取决于 Local Search 速度）
2. **validate-results + aggregate** — static 完成后立即执行
3. **Dynamic formal v3** — 700 tasks，预计 1-2 小时
4. **validate-results + aggregate** — dynamic 完成后
5. **生成表格和曲线** — 第八块

### Resume 计划

- 所有运行使用 `--resume`，只跳过已有 terminal key 的 task
- v3 inter-process lock 保证并发安全
- 失败 run 保留原始日志，不静默重试

---

## 13. Test 不调参承诺与 Bug 修复规则

1. **不调参承诺:** 不以 test 指标选择 checkpoint，也不在看过 test 结果后回去改训练超参数再替换结果
2. **Bug 修复规则:** 任何正确性修复（affecting evaluator 或 method）:
   - 递增 evaluator 或 method version
   - 失效受影响 run key
   - 重新运行所有受影响 method/scenario/seed cell
   - 不得修改 frozen checkpoint、seeds、groups、thresholds、budgets 或 method hyperparameters
3. **不变项:** frozen manifest、checkpoint、seeds、hold_steps=3、recovery_budget=30、Local Search budget=64/deltas=[1,2,4,8]

---

## 14. 预期输出文件

```
runs/block7-static-formal-v3/evaluation-6868c853b55b/
├── resolved_evaluation_config.json
├── run_manifest.json
├── episodes.jsonl          (3,500 lines)
├── steps.jsonl             (N lines)
├── event_recovery.jsonl    (空，static 无 event)
├── aggregate.json
├── aggregate.csv
├── mean_plus_std.csv
└── reports/                (若生成)

runs/block7-dynamic-formal-v3/evaluation-<dynamic-config-hash>/
├── (同上结构)
├── episodes.jsonl          (700 lines)
└── event_recovery.jsonl    (动态恢复行)
```

---

## 15. 验收矩阵

| 验收项 | 状态 |
|---|---|
| Frozen evaluation manifest 路径/数量/hash 已验证 | ✅ |
| 500 static + 100 dynamic 数量确认 | ✅ |
| 四组负载互斥完整、ID 列明确 | ✅ |
| 成功阈值 (hold=3, max_steps=50/240, recovery=30) | ✅ |
| 四种基线 method version/config hash | ✅ |
| Checkpoint gate 全部检查通过 | ✅ |
| Checkpoint strict load + greedy dispatch smoke 通过 | ✅ |
| Checkpoint SHA-256 已冻结 | ✅ |
| 无 P0 阻塞项 | ✅ |
| CUDA/disk 资源充足 | ✅ |
| Static formal v3 运行中 (35.1%) | ✅ |
| Test 不调参承诺确立 | ✅ |
| Bug 修复版本处理规则确立 | ✅ |
| 预期输出文件结构明确 | ✅ |

---

**本步未修改文件，也未运行正式 test。Static 正式运行已在 v3 评估器下进行中。Dynamic 正式运行在 static 完成并验证后启动。**

下一步：等待 static formal v3 完成（当前 1,230/3,500），运行 `validate-results` 和 `aggregate`，然后启动 dynamic formal v3。
