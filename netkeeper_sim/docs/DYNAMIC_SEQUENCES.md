# 测试专用动态事件序列

动态数据仅引用 `scenarios/smoke/test.jsonl` 中的 test scenario。每条默认有 6 个逻辑事件、8 个统一 `Event` records：策略增加/删除、流量缩放/hotspot、更安全的链路 down/up 和节点 down/up。不存在自然语言或 API 作为执行依赖。

`Event.step=t` 会在环境对 snapshot `t` 调用 `step()` 时、动作校验前生效。底层 event times 固定为 `0, 30, 60, 90, 120, 150, 180, 210`；每次变化之间保留 30 个 agent action 窗口，动态运行时 `max_steps=240`。链路候选按“移除该物理 link 后仍连通”筛选；节点候选按“移除后仍连通且不属于初始策略端点/waypoint”筛选，因此主 100 条均为 `expected_valid=true`、`expected_partition=false`。冲突/分区负例不进入本数据集。
