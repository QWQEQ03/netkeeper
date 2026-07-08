# NetKeeper Sim

`netkeeper_sim` 是为了复现 NetKeeper 论文相关思想而搭建的轻量级网络仿真内核。目前项目只关注可测试、可扩展的网络基础能力：真实拓扑加载、OSPF/ECMP 转发、流量传播、链路负载指标和简化 BGP 出口选择。

本项目当前不是强化学习环境，也不是完整网络控制器。它的定位是后续封装 Gymnasium/RL 环境之前的确定性仿真内核。

## 当前已实现能力

- 加载 Internet Topology Zoo 的 GraphML 和 GML 拓扑文件。
- 将 Topology Zoo 原始属性规范化为内部节点和链路模型。
- 使用可配置默认值补齐 Topology Zoo 中缺失的仿真属性。
- 基于链路 `ospf_weight` 计算 OSPF 最短路径。
- 保留所有等价最短路径下一跳，用于 ECMP。
- 支持链路故障、链路恢复和 OSPF 权重更新后重新计算路由。
- 支持 CSV 长表格式流量矩阵：`source,destination,demand`。
- 支持 NumPy 矩阵和普通二维列表形式的流量矩阵。
- 支持随机生成可复现的流量矩阵。
- 基于 OSPF/ECMP 转发表逐跳传播流量。
- ECMP 下一跳按等比例分流。
- 统计每条链路的 `total_load` 和 `utilization`。
- 计算 `maximum_link_utilization`。
- 检测不可达需求和丢弃流量。
- 校验流量守恒：`total_input_traffic == delivered_traffic + dropped_traffic`。
- 实现简化 BGP 路径选择。
- BGP 选择 prefix 出口后，继续使用 OSPF/ECMP 将流量转发到出口。
- 实现 NetKeeper 三个评价指标：策略一致性、最大链路利用率封装和流量迁移率。
- 支持更新前后 forwarding snapshot 的捕获和比较。
- 提供 CLI 示例和完整单元测试。

## 明确未实现内容

当前阶段故意不实现以下内容：

- 强化学习算法。
- 神经网络模型、Graph Transformer 编码器和 COMA 训练算法。
- LLM、LangChain、DSL 或意图翻译。
- Mininet、FRRouting、Containerlab 或真实路由器部署。
- 完整 BGP 协议。
- BGP 会话状态机、报文、定时器、withdraw、route reflection、community、策略语言。
- 复杂队列模型、真实丢包模型或时延仿真。

## 多智能体 RL 环境适配层

`netkeeper_sim/rl/` 提供一个轻量级多智能体强化学习环境适配层，只封装现有确定性仿真内核，不重新实现 OSPF、BGP、流量传播、策略和指标逻辑。

核心入口：

```python
from netkeeper_sim.rl import MultiAgentNetworkEnvironment

env = MultiAgentNetworkEnvironment(topology=topology, traffic_matrix=traffic, policies=policies)
state, observations = env.reset(seed=7)
result = env.step(joint_action)
```

`NetworkGraphState` 包含：

- `node_features`: `[num_nodes, 17]`
- `edge_index`: `[2, num_directed_edges]`
- `edge_features`: `[num_directed_edges, 11]`
- `node_mask`
- `edge_mask`
- `node_ids`
- `edge_ids`
- `parameter_masks`
- `policy_observation`
- `utilization_observation`

节点顺序使用 `sorted(topology.nodes)`，物理链路顺序使用 `sorted(topology.links)`。无向物理链路会转换成两条 directed edge，平行链路保持现有 `Topology.links` 的物理链路语义。

当前节点特征是工程版可解释编码，不是论文未公开的 20 维 Network Sketch 原始实现。特征包括节点类型 one-hot、相邻 OSPF 权重统计、BGP route 参数统计、相邻带宽/容量/队列/丢包率统计、active ratio、degree、BGP speaker 标记和 policy endpoint 标记。边特征包括 edge type one-hot、OSPF weight、bandwidth、capacity、queue length、loss rate、active 和 utilization。连续值按 `RLConfig` 中的参数上界归一化并裁剪到合理范围。

三个 agent 的局部观测：

- OSPF Agent: `[逐策略满足 0/1, 按稳定 link_id 排序的链路利用率]`
- BGP Agent: `[逐策略满足 0/1]`
- Performance Agent: `[按稳定 link_id 排序的链路利用率]`

联合动作采用 factorized 离散动作，不构造整张图参数的笛卡尔积：

```python
{
    "ospf": {"ospf_weight": [...]},
    "bgp": {
        "local_preference": [...],
        "as_path_length": [...],
        "med": [...],
    },
    "performance": {
        "bandwidth": [...],
        "capacity": [...],
        "queue_length": [...],
    },
}
```

`ospf_weight`、`bandwidth`、`capacity`、`queue_length` 按物理 `link_ids` 排序；BGP 参数按稳定的 `(router, prefix, route_index)` 排序。值为 `0` 表示该位置 no-op。合法范围来自论文参数区间：OSPF/BGP 参数为 `[1, 64]`，Performance 参数为 `[65, 128]`。无效位置或越界动作不会改变环境。

mask 字段包括：

- `ospf_weight_mask`
- `local_preference_mask`
- `as_path_length_mask`
- `med_mask`
- `bandwidth_mask`
- `capacity_mask`
- `queue_length_mask`

奖励复用现有三个指标，支持 `reward_mode="paper"` 和 `reward_mode="normalized"`：

```text
R_pol = K * policy_consistency + stationary_reward + dynamic_reward
R_res = K * ((1 - normalized_load) + (1 - traffic_shift))

R_OSPF = R_pol + R_res
R_BGP = R_pol
R_PERF = R_res
```

其中 `stationary_reward` 对连续两步完全相同动作惩罚 `-1`；`dynamic_reward` 对每条策略从不满足变为满足奖励 `+1`，从满足变为不满足惩罚 `-1`。`paper` 模式保留原始公式中的未裁剪负载项，`normalized` 模式会对 maximum link utilization 做可配置归一化/裁剪。

适配层优先返回 PyTorch tensor。当前运行环境未安装 PyTorch 时，会使用 NumPy ndarray 作为 tensor-like fallback；安装可选依赖后会自动返回 `torch.Tensor`：

```bash
pip install "netkeeper-sim[rl]"
```

### 共享 Graph Transformer 与 decentralized actors

`netkeeper_sim/rl/networks/` 实现了神经网络前向组件，但不包含 COMA loss、critic 更新或训练循环。

已实现模块：

- `SharedGraphTransformerEncoder`
- `ParameterIDEmbedding`
- `OSPFActor`
- `BGPActor`
- `PerformanceActor`
- `MultiAgentActor`

`SharedGraphTransformerEncoder` 输入：

- `node_features`
- `edge_index`
- `edge_features`
- `batch`
- `node_mask`

输出：

- `node_embeddings`: `[num_nodes, hidden_dim]`
- `graph_embedding`: `[batch_size, hidden_dim]`

配置位于 `GraphNetworkConfig`：

- debug: `hidden_dim=64`, `gcn_layers=2`, `transformer_layers=2`
- paper: `hidden_dim=128`, `gcn_layers=8`, `transformer_layers=8`

当前实现与论文 GraphTrans 的差异：

- 论文没有完整公开 Network Sketch 20 维特征和 GraphTrans 细节；当前实现使用工程版可解释图特征。
- GCN-like 层使用 PyG `GINEConv`，以支持 edge features。
- Transformer 层使用 PyG `TransformerConv`，并传入 edge features。
- graph pooling 使用显式 masked mean pooling。
- 当前实现包含 centralized critic、COMA baseline 和 debug 训练循环，但仍不是论文级完整实验复现。

7 种参数 ID 使用可训练 embedding：

- `ospf_weight`
- `bandwidth`
- `capacity`
- `queue_length`
- `local_preference`
- `as_path_length`
- `med`

Actor 输出是 logits，不在 forward 内 `argmax`：

- OSPF: `ospf_weight -> [num_links, 64]`
- BGP: `local_preference/as_path_length/med -> [num_bgp_routes, 64]`
- Performance: `bandwidth/capacity/queue_length -> [num_links, 64]`

无效实体会通过 action mask 将 logits 设置为极小值。采样和贪心选择由 `sample_masked_actions()` 与 `argmax_masked_actions()` 完成，训练时应使用 `Categorical` 采样，不应把 `argmax` 放进训练路径。

### Centralized critic 与 COMA

`netkeeper_sim/rl/networks/centralized_critic.py` 和 `netkeeper_sim/rl/algorithms/` 提供 COMA 训练所需的最小工程实现。

Critic 使用独立 `SharedGraphTransformerEncoder`，不与 actor 共享参数。对指定 agent、parameter 和实体集合，critic 输出所有候选动作的 Q-values：

```text
Q_i,k: [num_entities, 64]
```

COMA counterfactual baseline：

```text
baseline = sum_a pi_i,k(a | o_i) * Q_i,k(s, a)
advantage = Q_i,k(s, selected_action) - baseline
advantage = advantage * entity_mask
```

无效实体不会进入 baseline；actor 概率按合法实体重新处理，`selected_q` 使用 `gather`。默认 actor loss 中 advantage 会 detach，避免 actor loss 反向更新 critic。

Actor loss：

```text
-mean(log pi(selected_action) * advantage.detach())
```

Critic target 当前使用标准一阶 TD 目标：

```text
target = reward + gamma * (1 - done) * target_Q_next
```

论文对 critic target 和 loss 的完整公式描述有限，因此这里没有声称该 TD 目标是论文原始公式。critic loss 默认使用 Huber loss，也支持 MSE。

Target critic 支持：

- hard update
- soft update with configurable `tau`

论文表中给出 target update interval 为 16 timestep，但未公开 soft update 系数；默认实现每 16 step hard copy，`tau` 仅作为可配置工程选项。

训练配置：

- `configs/rl_debug.yaml`: 小网络，2 episodes，5 steps，batch size 2。
- `configs/rl_paper.yaml`: 保留论文表中的主要超参数。

运行 debug 训练：

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages:. \
python3 -m netkeeper_sim.rl.train \
  --config configs/rl_debug.yaml \
  --checkpoint checkpoints/rl_debug.pt
```

## 项目结构

```text
netkeeper_sim/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   └── samples/
│       └── traffic.csv
├── netkeeper_sim/
│   ├── cli.py
│   ├── topology/
│   │   ├── loader.py
│   │   ├── model.py
│   │   └── normalizer.py
│   ├── routing/
│   │   ├── ospf.py
│   │   ├── ecmp.py
│   │   └── bgp.py
│   ├── traffic/
│   │   ├── matrix.py
│   │   └── propagation.py
│   ├── metrics/
│   │   ├── load.py
│   │   ├── traffic_shift.py
│   │   └── evaluation.py
│   ├── policies/
│   │   ├── model.py
│   │   └── evaluator.py
│   └── simulator/
│       └── environment.py
└── tests/
    ├── test_topology_loader.py
    ├── test_ospf.py
    ├── test_ecmp.py
    ├── test_failures_and_updates.py
    ├── test_traffic_matrix.py
    ├── test_traffic_propagation.py
    ├── test_bgp.py
    ├── test_bgp_forwarding.py
    ├── test_policy_consistency.py
    ├── test_traffic_shift.py
    ├── test_evaluation_metrics.py
    └── test_environment.py
```

## 安装依赖

进入项目目录：

```bash
cd /home/dministrator/projects/netkeeper/netkeeper_sim
```

推荐使用虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

如果当前环境已经安装依赖，也可以直接运行测试和示例。

## 运行测试

```bash
python3 -m pytest
```

当前测试覆盖：

- Topology Zoo GraphML/GML 加载。
- 缺失链路属性使用配置默认值补齐。
- 单路径 OSPF 最短路。
- ECMP 菱形拓扑。
- 三路 ECMP 下一跳保留。
- 非等价路径不分流。
- 链路故障和恢复。
- OSPF 权重更新。
- CSV、NumPy 和二维列表流量矩阵。
- 逐跳流量传播。
- ECMP 等比例分流。
- 重复 demand 的路径和负载聚合。
- 不可达需求记录。
- 流量守恒。
- 简化 BGP 的每条路径选择规则。
- BGP prefix 流量通过 OSPF/ECMP 转发到出口。
- 策略一致性、ECMP 策略判断和 isolation 简化语义。
- 更新前后 forwarding snapshot 的 traffic shift。
- 三个 NetKeeper 指标在环境类中的端到端集成。

最近一次完整测试结果：

```text
65 passed
```

## 真实 Topology Zoo 示例

使用 Abilene GraphML 拓扑运行 OSPF、ECMP 统计和随机流量传播：

```bash
python3 -m netkeeper_sim.cli \
  --topology ../InternetTopologyZoo/graphml/Abilene.graphml \
  --random-traffic \
  --seed 7 \
  --max-demand 20 \
  --density 0.25
```

一次示例输出：

```text
topology: ../InternetTopologyZoo/graphml/Abilene.graphml
nodes: 11
edges: 14
reachable_node_pairs: 110
unreachable_node_pairs: 0
ecmp_node_pairs: 15
traffic_demands: 33
total_input_traffic: 286.079082
delivered_traffic: 286.079082
dropped_traffic: 0.000000
unreachable_demands: 0
flow_conserved: True
maximum_link_utilization: 0.913705
```

NetKeeper 指标演示中，使用同一 Abilene 拓扑和随机流量，先保存初始 forwarding snapshot，再将链路 `0-1` 的 OSPF 权重更新为 10：

```text
initial_policy_consistency=1.000000
initial_maximum_link_utilization=0.913705
updated_policy_consistency=0.000000
updated_maximum_link_utilization=0.973520
traffic_shift_ratio=0.118182
traffic_shift_changed_entries=13
traffic_shift_total_entries=110
```

## Topology Zoo 加载与属性规范化

支持两种真实 Topology Zoo 文件格式：

- `.graphml`
- `.gml`

加载后会统一规范化为内部 `Topology` 对象：

- `Topology.nodes`: `dict[str, Node]`
- `Topology.links`: `dict[str, Link]`
- `Topology.graph`: `networkx.MultiGraph`
- `Topology.metadata`: 原始图级元数据

节点模型 `Node` 包含：

- `node_id`
- `name`
- `latitude`
- `longitude`
- `node_type`
- `as_number`
- `raw_attributes`

链路模型 `Link` 包含：

- `link_id`
- `source`
- `target`
- `ospf_weight`
- `bandwidth`
- `capacity`
- `queue_length`
- `loss_rate`
- `propagation_delay`
- `is_active`
- `raw_attributes`

Topology Zoo 原始文件通常包含节点标签、经纬度、国家、链路标签等真实元数据，但不稳定包含 OSPF 权重、容量、队列、丢包率、传播时延等仿真字段。因此这些字段由配置默认值补齐。

默认配置位于：

```text
configs/default.yaml
```

当前默认值：

```yaml
topology_defaults:
  ospf_weight: 1.0
  bandwidth: 100.0
  capacity: 100.0
  queue_length: 100
  loss_rate: 0.0
  propagation_delay: 1.0
  use_link_speed_as_capacity: false
```

这些默认值是仿真假设，不代表真实网络测量值。

## OSPF 与 ECMP

OSPF 模块位于：

```text
netkeeper_sim/routing/ospf.py
```

核心接口：

```python
compute_ospf_routes(topology)
```

返回的转发表结构：

```python
ForwardingTable = dict[str, dict[str, ForwardingEntry]]
```

每个 `ForwardingEntry` 包含：

```python
ForwardingEntry(
    cost=2.0,
    next_hops=["R2", "R3"],
    reachable=True,
)
```

含义：

- `cost`: 源到目的的最短路径代价。
- `next_hops`: 所有满足最短代价的下一跳。
- `reachable`: 目的是否可达。

ECMP 不只保留一条最短路径，而是保留所有等价下一跳。例如菱形拓扑中，`R1 -> R4` 可以同时保留 `R2` 和 `R3` 作为下一跳。

当前拓扑内部使用 `MultiGraph` 保留平行链路。用于 OSPF 计算时，会把平行链路按最小 OSPF 权重折叠为路由图；如果多条平行链路权重相同，流量统计时会在这些等价物理链路之间平均分担。

## 链路故障和权重更新

`Topology` 支持：

```python
topology.fail_link("R1", "R2")
topology.restore_link("R1", "R2")
topology.update_ospf_weight("R1", "R2", 10)
```

`NetworkSimulationEnvironment` 也提供对应方法：

```python
env.fail_link("R1", "R2")
env.restore_link("R1", "R2")
env.update_ospf_weight("R1", "R2", 10)
```

链路状态或权重发生变化后，需要重新计算 OSPF 转发表。

## 流量矩阵

流量矩阵模块位于：

```text
netkeeper_sim/traffic/matrix.py
```

支持 CSV 长表：

```csv
source,destination,demand
R1,R5,100
R2,R6,50
```

读取方式：

```python
traffic = TrafficMatrix.from_csv("data/samples/traffic.csv", topology.nodes)
```

支持 NumPy 矩阵：

```python
traffic = TrafficMatrix.from_numpy(matrix, nodes=["R1", "R2", "R3"])
```

支持随机矩阵：

```python
traffic = TrafficMatrix.random(
    sorted(topology.nodes),
    seed=7,
    max_demand=20.0,
    density=0.25,
)
```

校验规则：

- source 必须存在于拓扑节点中。
- destination 必须存在于拓扑节点中。
- demand 不允许为负数。
- `source == destination` 默认忽略。
- demand 为 0 默认忽略。

## 流量传播

流量传播模块位于：

```text
netkeeper_sim/traffic/propagation.py
```

核心接口：

```python
result = propagate_traffic(topology, forwarding_table, traffic_matrix)
```

传播规则：

- 每个 demand 从 source 出发。
- 如果目的已经到达，计入 delivered traffic。
- 如果目的不可达，计入 dropped traffic 和 unreachable demands。
- 如果存在多个 ECMP 下一跳，按下一跳数量等比例分流。
- 每走过一条边，记录路径流量和链路负载。
- 如果检测到环路或超过最大 hop 限制，将对应流量记为 dropped。

结果对象 `PropagationResult` 包含：

- `flow_paths`
- `link_loads`
- `unreachable_demands`
- `delivered_traffic`
- `dropped_traffic`
- `total_input_traffic`
- `is_flow_conserved()`

`flow_paths` 示例：

```python
{
    ("R1", "R4"): {
        ("R1", "R2"): 50.0,
        ("R2", "R4"): 50.0,
        ("R1", "R3"): 50.0,
        ("R3", "R4"): 50.0,
    }
}
```

## 链路负载和利用率

指标模块位于：

```text
netkeeper_sim/metrics/load.py
```

核心接口：

```python
metrics = calculate_link_load_metrics(topology, result.link_loads)
```

输出：

- `total_load`: 每条 `link_id` 上累计流量。
- `utilization`: `total_load / capacity`。
- `maximum_link_utilization`: 所有链路中的最大利用率。

如果链路容量为 0 且有流量，利用率记为 `inf`；如果容量为 0 且无流量，利用率记为 0。

## NetKeeper 评价指标

当前实现的三个论文指标位于：

```text
netkeeper_sim/policies/
netkeeper_sim/metrics/traffic_shift.py
netkeeper_sim/metrics/evaluation.py
```

### Policy Consistency

策略一致性定义为：

```text
policy_consistency = satisfied_policy_count / total_policy_count
```

没有策略时，当前实现返回 `consistency=1.0` 且 `total=0`，表示没有发现违反项，而不是网络天然满足任何业务意图。

支持的策略类型：

- `ForwardPolicy(source, destination, required_next_hop)`：`forward(A, B, C)`，表示 A 到 B 的转发表下一跳集合必须包含 C。
- `ReachablePolicy(source, destination, must_pass)`：`reachable(A, B, C)`，表示 A 到 B 的有效转发路径必须经过 C。
- `IsolationPolicy(...)`：当前复现中的最小语义，不声称覆盖论文完整 isolation 语言。

ECMP 假设：

- `ForwardPolicy` 使用下一跳集合判断，不依赖列表顺序。
- `ReachablePolicy` 默认使用 `any_path`：只要至少一条 ECMP 有效路径经过 `must_pass` 即满足。
- `ReachablePolicy(mode="all_paths")` 表示所有 ECMP 有效路径都必须经过 `must_pass`。
- 路径判断基于现有 OSPF 转发表的 `next_hops`，不重新计算另一套路由。
- 路径枚举会防止环路、去重，并通过 `max_paths` 限制路径爆炸；超过上限会返回不满足并给出原因。

Isolation 当前假设：

- `mode="forbidden_node"`：两组转发关系不能同时存在经过同一个 `forbidden_node` 的有效路径。
- `mode="path_disjoint"`：两组转发关系的任意有效路径对不能共享中间节点。
- 如果任一转发关系不可达，isolation 策略判为不满足。

### Maximum Link Utilization

论文中的负载指标 `rho` 使用现有 `metrics/load.py` 的 `maximum_link_utilization`：

```text
utilization(link) = total_load(link) / capacity(link)
rho = max(utilization(link))
```

统一评价结果中同时返回：

- `maximum_link_utilization`
- `average_link_utilization`
- `link_utilizations`
- `overloaded_links`

其中 `maximum_link_utilization` 的计算含义没有改变。

### Traffic Shift

流量迁移率基于更新前后的 forwarding plane snapshot：

```text
traffic_shift = changed_forwarding_entries / compared_forwarding_entries
```

每个 snapshot entry 显式保存：

```python
ForwardingState(
    reachable=True,
    next_hops=frozenset({"R2", "R3"}),
)
```

比较规则：

- `reachable=False -> reachable=False`：不算变化。
- `reachable=False -> reachable=True`：算变化。
- `reachable=True -> reachable=False`：算变化。
- 都可达但 `next_hops` 集合不同：算变化。
- 都可达且 `next_hops` 集合相同：不算变化。

ECMP 下一跳按集合比较，因此 `{R2, R3}` 和 `{R3, R2}` 不算变化，`{R2, R3}` 到 `{R2}` 算变化。

Denominator 模式：

- `union`：默认模式，比较前后 snapshot 条目的并集；新增、删除和可达性变化都计入 traffic shift。
- `intersection`：只比较前后都存在的条目，适合单纯配置调整分析。

当前实现为了支持链路故障、设备故障和拓扑变化，默认采用 `union`。这与论文中 `total_devices * total_reachable_prefixes` 的静态分母不同，是本仿真内核的复现假设。

BGP prefix snapshot 使用选中的 BGP `next_hop` 作为 prefix forwarding entry 的下一跳集合；OSPF node snapshot 默认不包含 `source == destination` 的本地条目。

### Environment 示例

```python
from netkeeper_sim.policies import ForwardPolicy
from netkeeper_sim.simulator.environment import NetworkSimulationEnvironment
from netkeeper_sim.traffic.matrix import TrafficDemand, TrafficMatrix

env = NetworkSimulationEnvironment()
env.load_topology("../InternetTopologyZoo/graphml/Abilene.graphml")
env.set_traffic_matrix(TrafficMatrix((TrafficDemand("0", "10", 100.0),)))
env.set_policies([ForwardPolicy("p1", "0", "10", "1")])

env.compute_ospf_routes()
env.propagate_traffic()
before = env.capture_forwarding_snapshot()
initial = env.evaluate_metrics()

env.update_ospf_weight("0", "1", 10)
env.compute_ospf_routes()
env.propagate_traffic()
updated = env.evaluate_metrics(previous_snapshot=before)
```

`updated` 是统一 `EvaluationResult`，包含：

- `policy_consistency`
- `maximum_link_utilization`
- `average_link_utilization`
- `link_utilizations`
- `overloaded_links`
- `traffic_shift`

## 简化 BGP

BGP 模块位于：

```text
netkeeper_sim/routing/bgp.py
```

`BGPRoute` 数据结构包含：

- `prefix`
- `next_hop`
- `local_preference`
- `as_path`
- `med`
- `origin_router`
- `learned_from`
- `igp_cost_to_next_hop`

路径选择优先级严格为：

1. Local Preference 越大越优。
2. AS Path 越短越优。
3. MED 越小越优。
4. 到 next hop 的 OSPF cost 越小越优。
5. 使用确定性的 router ID / 字符串排序打破平局。

核心接口：

```python
best = select_best_route(routes)
selected = select_best_routes(candidate_routes, forwarding_table)
```

`candidate_routes` 结构：

```python
{
    "R1": {
        "203.0.113.0/24": [route1, route2],
    }
}
```

BGP prefix 流量传播：

```python
result = propagate_bgp_traffic(
    topology,
    forwarding_table,
    selected_bgp_routes,
    prefix_traffic_matrix,
)
```

此过程只做两件事：

1. BGP 为 prefix 选择出口 next hop。
2. OSPF/ECMP 将流量从源节点转发到该 next hop。

## 统一环境封装

环境封装位于：

```text
netkeeper_sim/simulator/environment.py
```

当前提供：

```python
env = NetworkSimulationEnvironment()
env.load_topology(path)
env.set_traffic_matrix(matrix)
env.compute_ospf_routes()
env.compute_bgp_routes(candidate_routes)
env.propagate_traffic()
env.calculate_metrics()
env.fail_link(u, v)
env.restore_link(u, v)
env.update_ospf_weight(u, v, weight)
env.reset()
```

当前环境还不是 Gymnasium 环境，但已经把 topology、routing、traffic propagation、metrics 分开保存，后续可以自然封装为：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## 设计假设和限制

- Topology Zoo 拓扑默认按无向图处理。
- 无向链路可双向转发。
- OSPF 使用静态 `ospf_weight`。
- ECMP 默认按下一跳等比例分流。
- 平行链路在路由计算中按最小权重折叠。
- 等权平行链路在负载统计中平均分担。
- 流量传播不模拟包级行为。
- 链路容量默认是仿真参数，不是实际链路测量值。
- BGP 只做本地最佳路径选择，不模拟控制平面传播。
- 当前指标只覆盖负载、利用率、不可达和守恒，不包含复杂 QoS。

## 后续可扩展方向

- 增加 Gymnasium 兼容环境接口。
- 将 OSPF weight、BGP 属性、链路容量等作为 action space。
- 将链路利用率、不可达需求、策略满足情况编码为 observation。
- 增加 reward 函数，例如最小化最大链路利用率、减少 dropped traffic。
- 增加更多真实 Topology Zoo 拓扑的批量实验脚本。
- 增加策略约束，例如 waypoint、isolation、forwarding policy。
- 扩展更真实的 BGP 策略模型，但仍与完整协议实现保持边界。
