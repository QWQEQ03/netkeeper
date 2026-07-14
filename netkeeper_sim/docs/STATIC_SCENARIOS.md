# 静态 MADRL smoke 场景

本步只生成 smoke 数据：每个 split 12 条；正式目标固定为 train 3,000、validation 400、test 500，尚未生成。难度配额按最大余数法固定：训练 45/45/10，验证 40/40/20，测试 35/40/25（Easy/Medium/Hard）。每个难度分别包含 Reachable、Forward、Isolation 各 2/4/8 条。

流量模式配额目标是 gravity 30%、diurnal 20%、hotspot 25%、burst 25%；三个负载等级等比例。场景仅引用同 split 的拓扑、配置与流量矩阵，路径均相对 dataset root，且每条记录有 canonical content SHA-256。

策略采样以有限重试生成结构有效且无重复的约束；静态初始评估必须满足 `0 < consistency < 1`，并且没有 invalid/conflict/infeasible 条目。该过滤只依赖初始确定性内核，不依赖训练模型。场景默认 `events=[]`、`failures=[]`、`max_steps=50`。
