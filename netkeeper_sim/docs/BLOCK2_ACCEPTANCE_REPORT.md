# 第二块验收报告

## 结论

**通过。** 已逐项审查代码、正式数据和生成链路；发现并修正了一处 P1 级元数据问题：`generation_config.yaml` 曾声明未实际使用的 `split` seed namespace，现已移除并重新生成元数据。未发现 P0/P1 遗留问题。本次未进入第三块，未实现 API/LLM/模型训练。

## 实现摘要与架构

第二块使用第一块的 `Topology`、`Link`、`NetworkConfiguration`、`TrafficMatrix`、`Policy`、`Event` 和 `NetworkScenario`。离线 JSON/JSONL/NPY 仅是这些统一对象的可校验序列化：`scenario_from_record()` 反序列化真实统一对象，`ScenarioDataset` 按行惰性读取，随后直接进入 `UnifiedNetworkEnvironment.reset()` 和 `step()`；不存在转换脚本掩盖的平行运行时 Schema。

Topology Zoo 原始文件保持只读。稳定自然排序后，节点被映射为 `R0…`，保留 `original_label`；并行链路通过稳定序号产生 `link_id`，不被简化图丢弃。容量采用已有的速度/单位解析，坐标有效时估算时延，否则采用配置中的显式默认值；JSON 不写 NaN 或缺失值。

流量实现四种真实模式：gravity（按节点容量质量的 OD 引力矩阵）、diurnal（基础矩阵乘以存储的 phase/time-index 周期因子）、hotspot（确定性热点入/出流量增益）、burst（确定性少量 OD 突发增益）。基础矩阵以 Normal 目标 MLU 校准，并由相同矩阵的 Low/Normal/High=0.5/1.0/3.0 倍率重放。Topology Zoo 不提供 BGP 语义，因此初始 BGP 为可验证的简化模型合成配置，记录 `synthetic=true`。

静态场景仅从本 split 拓扑抽取，无 failures/events。Easy/Medium/Hard 分别为 Reachable、Forward、Isolation 各 2/4/8 条；冲突检测和有限重采样失败会显式报错。初始一致性过滤为 `0 < consistency < 1`，因此没有强行让所有策略预先满足。动态序列只引用 static test 场景，包含策略增加/删除、流量缩放/热点变化、链路和节点故障/恢复；每个逻辑变化后的恢复预算为 30 step。

## 新增/修改的主要文件

- 数据生成：`netkeeper_sim/netkeeper_sim/dataset/topologies.py`、`traffic.py`、`scenarios.py`、`dynamic_sequences.py`、`publication.py`、`cli.py`、`__init__.py`
- 统一 Schema/加载复用：`netkeeper_sim/netkeeper_sim/schemas/loader.py`，以及现有 `schemas/models.py`、`simulator/unified_environment.py`
- 配置与说明：`netkeeper_sim/configs/dataset_traffic.yaml`、`dataset_scenarios.yaml`、`dataset_dynamic.yaml`、`netkeeper_sim/docs/TRAFFIC_DATASET.md`、`STATIC_SCENARIOS.md`、`DYNAMIC_SEQUENCES.md`、`DATASET_GENERATION.md`
- 测试：`netkeeper_sim/tests/test_dataset_topologies.py`、`test_dataset_traffic.py`、`test_dataset_scenarios.py`、`test_dataset_dynamic_sequences.py`
- 正式数据：`data/netkeeper_lite/`（以下目录树）

## 最终数据目录

```
data/netkeeper_lite/
├── topologies/{train,validation,test}/*.json       # 12 / 3 / 4
├── configurations/*.json                           # 19 个初始配置
├── traffic/*.npy                                   # 76 个基础矩阵
├── scenarios/{train,validation,test}.jsonl         # 3000 / 400 / 500
├── dynamic_sequences/test.jsonl                    # 100
└── metadata/
    ├── topology_split.json
    ├── topology_candidates.json
    ├── traffic_manifest.json
    ├── scenario_manifest.json
    ├── dynamic_sequences_manifest.json
    ├── generation_config.yaml
    ├── random_seeds.json
    ├── dataset_statistics.json
    └── manifest.json
```

发布 manifest 覆盖 127 个正式文件（约 19 MB），记录相对路径、字节数、SHA-256 与 Schema 版本。自身通过明确的 `self_hash_rule` 排除；smoke fixture 不属于发布集，也被明确排除。

目录体积为：`scenarios/` 约 16 MB、`traffic/` 约 832 KB、`topologies/` 约 780 KB、`metadata/` 约 656 KB、`configurations/` 约 392 KB、`dynamic_sequences/` 约 272 KB；JSONL 行数为 3,000 + 400 + 500 + 100。

## 拓扑 split

| Split | topology_id | 节点 | 边 |
| --- | --- | ---: | ---: |
| train | Arpanet19706 | 9 | 10 |
| train | Belnet2004 | 23 | 43 |
| train | Noel | 19 | 25 |
| train | BtNorthAmerica | 36 | 76 |
| train | Geant2009 | 34 | 52 |
| train | Bren | 37 | 38 |
| train | Gridnet | 9 | 20 |
| train | Belnet2003 | 23 | 43 |
| train | HostwayInternational | 16 | 21 |
| train | Cesnet200304 | 29 | 33 |
| train | Cesnet200603 | 39 | 44 |
| train | Navigata | 13 | 17 |
| validation | Nordu2005 | 9 | 10 |
| validation | Bics | 33 | 48 |
| validation | Garr2009 | 54 | 68 |
| test | Darkstrand | 28 | 31 |
| test | Belnet2006 | 23 | 44 |
| test | Garr2010 | 55 | 72 |
| test | Garr | 56 | 74 |

选择由固定 seed、内容哈希排序和分层规则决定，与文件系统枚举顺序无关。topology_id 在三个 split 间交集为空。训练节点数为 9–39；验证为 9–54；测试为 23–56，均不超过 80。

## 关键统计

| 项目 | train | validation | test |
| --- | ---: | ---: | ---: |
| 拓扑数 | 12 | 3 | 4 |
| 场景数 | 3,000 | 400 | 500 |
| gravity / diurnal / hotspot / burst | 900 / 600 / 750 / 750 | 120 / 80 / 100 / 100 | 150 / 100 / 125 / 125 |
| Low / Normal / High | 1,000 / 1,000 / 1,000 | 133 / 133 / 134 | 167 / 166 / 167 |
| Easy / Medium / Hard | 1,350 / 1,350 / 300 | 160 / 160 / 80 | 175 / 200 / 125 |
| 各策略类型条目数 | 10,500 | 1,600 | 2,150 |
| 初始 MLU 范围 | 0.125–0.750 | 0.125–0.750 | 0.125–0.750 |
| 初始策略一致性均值 | 0.578 | 0.594 | 0.531 |

静态策略重采样尝试均值为 train 0.0047、validation 0.015、test 0；没有策略采样耗尽。动态序列共 100 条，均为 `expected_valid=true`，每条 6 个逻辑事件（8 个底层 Event 记录）：policy_add、policy_remove、traffic_scale、hotspot_change、link failure/recovery、node failure/recovery 各覆盖 100 条。无意外永久分区和动态无效重采样均为 0。

## Seed、生成、验证与复现

根 seed 为 `20260713`。派生使用 SHA-256 的前 8 字节，namespace 为 `topology`、`scenario`、`traffic`、`policy`、`event`；不使用 Python `hash()`，随机数为局部 Generator。完整规则在 `metadata/generation_config.yaml` 和 `metadata/random_seeds.json`。

生成、验证和抽样命令见 [DATASET_GENERATION.md](DATASET_GENERATION.md)。相同 seed 已在独立临时输出目录从零重生成，并比较 `topology_split.json`、`generation_config.yaml`、`dataset_statistics.json` 与最终 `manifest.json`；规范化内容和 manifest 一致。

## 测试与验收结果

- 全量发布验证：有效；静态场景 3,900、动态序列 100、环境抽样 25 条，所有引用、哈希、split 和数量验证无错误。
- 集成：代表场景执行 `UnifiedNetworkEnvironment.reset()` 后执行空 `JointAction` 的 `step()`；覆盖 19 个拓扑及四种流量、三档负载、三档难度所需组合，无 Schema、实体或路由错误。
- 单元/集成回归：`127 passed, 8 skipped in 4.06s`。
- 数据测试覆盖稳定重命名、单位转换、缺省属性、MultiGraph 并行边、连通性过滤、split 无泄漏、矩阵不变量/缩放/重放、策略数量/冲突、JSONL 惰性读取、事件配对和环境调度。

## 已知限制与后续块接口

- Topology Zoo 不携带可用 BGP 业务语义，故 BGP 为明确标记的合成简化配置；不是原始 Zoo 路由策略。
- 发布集仅含有效动态序列；可选负例/策略冲突子集未混入 100 条测试序列。
- 流量 NPY 是基础矩阵，加载方需应用记录中的 `load_multiplier`；不使用 pickle。
- smoke 文件为开发夹具，未计入发布 manifest。

第三块若开始，应直接使用 `ScenarioDataset`/`scenario_from_record()`、`dynamic_scenario()`、统一 `Event` 和 `UnifiedNetworkEnvironment`。它不应创建自然语言/API 或另一套 Schema 作为本块数据的前置条件。
