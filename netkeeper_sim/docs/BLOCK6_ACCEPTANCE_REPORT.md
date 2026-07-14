# 第六块验收报告：基线方法与统一评估框架

## 结论

**有条件通过。** 第六块已实现统一方法接口、四种基线、Block 5 checkpoint
适配器、静态/动态 episode runner、冻结 manifest、可恢复批量 CLI、结果
Schema、原始日志校验和聚合。完整回归已取得最终汇总：**199 passed, 33
skipped in 37.46s**。

条件项仅为实验结论而非框架正确性：尚未运行正式 500 条 static test 或 100
条 dynamic test；现有 Block 5 checkpoint 仍为 `debug_unconverged`，不能作为
收敛训练模型或论文比较结论。

## 已交付能力

- `EvaluationMethod` / `EvaluationContext` / `MethodDecision`：metadata 含
  方法名、版本、config/checkpoint hash、确定性、权限、lookahead 和 checkpoint
  状态。
- 所有正式状态变化仅经 `UnifiedNetworkEnvironment.step(JointAction)`；方法不能
  直接修改 NetworkX、traffic、configuration 或 metrics。
- 静态 runner 保存 pre/post snapshot hash、action、字段 diff、RewardBreakdown、
  Metrics、终止状态、错误、decision/simulator/wall time。
- 动态 runner 按第二块 Event schedule 执行；支持 `logical_events` 的 down/up
  合并恢复单元，记录 pre-event、worst、recovery value、recovered 与 censored
  recovery step。
- 结果目录为 `evaluation-<config-hash>/`，原子写入 resolved config、run
  manifest、steps/episodes/failures JSONL、aggregate JSON/CSV、mean±std CSV。
- `run`、`aggregate`、`validate-results` CLI；默认串行，每任务重新构造方法和
  环境；dry-run 不执行环境，resume 只跳过已有终态 key。

## 基线与公平协议

| 方法 | 允许参数 | 正式每步预算 | Lookahead |
|---|---|---:|---:|
| No Update | 无 | 0 | 否 |
| Random | 与 Block 5 相同的 OSPF/BGP/Performance 参数 | 每 agent 1 | 否 |
| OSPF Default | `ospf_weight` | 1 | 否 |
| Local Search OSPF | `ospf_weight` | 1 | 是 |
| Checkpoint | Block 5 action adapter 参数 | 每 agent 1 | 否 |

Random 直接复用 `snapshot_to_graph`、`action_masks` 和
`candidate_to_joint_action`，在每 agent 的合法 mask（含 no-op）内以局部 RNG
均匀采样。OSPF Default 固定默认权重为 1，按稳定 `link_id` 每步提交一个仍需
默认化的合法链接，绝不修改 BGP/Performance 或故障后重新优化。

Local Search 只生成当前 OSPF mask 内的单参数候选，邻域固定为 ±1/±2/±4/±8、
候选值 1..64、每步最多 64 个候选与一个 no-op reference。目标严格词典序：
最大 PC、最小 MLU、最小 project-v1 shift、最小字段 diff，随后以稳定
`(link_id, value)` 决胜。只有严格优于 no-op 才提交。每个候选在新的统一环境
中从同一 immutable snapshot 单步回放；其 simulator calls、candidate count、
decision time 和目标前后值写入日志。没有跨 snapshot 指标缓存。

## 指标、结果与校验

PC 使用 schema `satisfied/enabled`；MLU 为 schema 最大 directed utilization；
paper-v1/project-v1 的 step/total Traffic Shift 都保留。配置修改比例为：

`最终与初始不同的合法原子字段数 / 初始字段 universe 数`

字段 universe 覆盖 OSPF、每路由 BGP local-pref/AS-path/MED、Performance、
link/node state，因而不会把并行链路或 synthetic BGP route 合并。累计 action
count 单独统计，不能以 configuration version 替代。成功要求连续 3 个状态；
未达到的 convergence/recovery 为 `null + censored`。

validator 验证 run key、manifest method/version/scenario/seed、一任务一终态、
step 连续与 snapshot 链、同 scenario/seed 跨方法 initial snapshot hash、非有限
JSON 值和 aggregate 从原始 episodes/failures 的重算一致性。失败 run 始终保留。

## Smoke 验收

仅使用 validation/fixture：Easy、Medium、Hard 各一条代表场景，固定一 seed、
每条最多两步，四种基线各覆盖；动态使用合成 contract fixture。dry-run 共计划
15 task；首次/恢复执行后结果为：

```text
planned=15, terminal=15, valid=true
```

其中 checkpoint 缺失路径 smoke 形成 3 条 retained failure；真实
`runs/rl-122f3f3e29/latest.pt` 另完成单场景、单步统一 evaluator smoke：
`completed`，但 metadata 为 `debug_unconverged`。Local Search 的两步 smoke
每 run 记录 130 次 sandbox simulator calls。上述结果只验证 contract，不排名、
不调参、不代表论文实验。

## 测试与已知限制

- 第六块专项及关键统一环境/API/dataset/RL 回归：18 passed。
- 完整 `pytest -q --durations=20`：199 passed, 33 skipped in 37.46s。
- 缓慢项主要是完整 intent dataset validator（15.46s）、intent evaluator
  resume（4.57s）和小 split deterministic generation（4.14s）；不是失败或卡死。
- 尚未实现 timeout 自动中断/重试策略；当前异常会作为 failure 留存，不会改变
  seed 或静默重试。
- 设备参数目前主要供记录和 checkpoint 配置使用；批量执行默认串行 CPU，未宣称
  并行 deterministic 性。

## 第七块固定入口

先生成或保存冻结 manifest，再使用其 hash 校验后运行：

```bash
python -m netkeeper_sim.evaluation.cli run \
  --dataset-root ../data/netkeeper_lite --output RUNS \
  --evaluation-manifest frozen-evaluation-manifest.json \
  --config configs/evaluation.yaml
```

正式运行不得因 test 结果改变 seeds、场景顺序、success/recovery 阈值、Local
Search 邻域/预算、统计规则或 checkpoint 状态；完成后运行
`validate-results` 和 `aggregate`。本报告不包含任何正式 test 优劣结论。
