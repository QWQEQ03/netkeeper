# NetKeeper Lite 数据集生成

第二块产物位于仓库根目录的 `data/netkeeper_lite/`。它是对第一块统一运行时 Schema 的离线数据表示，不引入第二套拓扑、配置、策略或事件模型。

## 架构和运行时接口

```
Topology Zoo (.graphml/.gml，只读)
  -> schemas.load_schema_topology() -> Topology / Link JSON
  -> traffic .npy + TrafficMatrix 元数据
  -> 静态 JSONL (NetworkConfiguration + Policy + traffic 引用)
  -> ScenarioDataset (逐行惰性读取)
  -> NetworkScenario
  -> UnifiedNetworkEnvironment.reset() / step()
```

动态 JSONL 在同一静态 test 场景上附加统一 `Event` 记录；可用
`dynamic_scenario()` 生成带事件的 `NetworkScenario`。`step` 是事件的调度步（文档中的 `at_step` 语义），不依赖自然语言或 API 表示。

## 从零生成与验证

在 `netkeeper_sim/` 目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m netkeeper_sim.dataset.cli generate-release \
  --source-root ../InternetTopologyZoo \
  --output-root ../data/netkeeper_lite \
  --selection-seed 20260713

PYTHONDONTWRITEBYTECODE=1 python3 -m netkeeper_sim.dataset.cli validate-release \
  --output-root ../data/netkeeper_lite
```

抽样查看静态场景和动态序列：

```bash
sed -n '1p' ../data/netkeeper_lite/scenarios/test.jsonl
sed -n '1p' ../data/netkeeper_lite/dynamic_sequences/test.jsonl
```

`validate-release` 会验证完整 JSONL、引用、内容哈希、split 隔离、正式数量，并对覆盖所有拓扑和 traffic pattern/load/difficulty 的代表场景执行 `reset()` 和空 `JointAction` 的 `step()`。

## 可复现性

根 seed 为 `20260713`。派生 seed 使用 SHA-256 的前 8 字节：

```
int.from_bytes(sha256(f"{root_seed}:{namespace}:{part1}:...").digest()[:8], "big")
```

使用的 namespace 为 `topology`、`scenario`、`traffic`、`policy`、`event`；不使用 Python 的 `hash()`，每个生成器使用局部 NumPy Generator，不污染全局随机状态。完整配置和可计算的 seed 规则在 `metadata/generation_config.yaml` 与 `metadata/random_seeds.json`。

交通矩阵按统一拓扑的 `node_order` 保存为非 pickle 的 `.npy` 基础矩阵；Low/Normal/High 通过同一基础矩阵的 0.5/1.0/3.0 `load_multiplier` 重放。所有引用均为相对路径。

## 范围

本数据集仅覆盖第二块：Topology Zoo 标准化、流量、初始配置、策略、静态场景和结构化动态事件。没有自然语言到 API 数据、LLM/DeepSeek 调用、Actor/Critic/COMA 改动或正式训练。第三块可直接消费 `ScenarioDataset`、`NetworkScenario`、`Event` 和 `UnifiedNetworkEnvironment`，无需转换为另一种运行时 Schema。
