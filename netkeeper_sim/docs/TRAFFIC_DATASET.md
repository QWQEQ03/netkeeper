# NetKeeper-Lite 流量矩阵

所有基础矩阵均为 `float64`、`N×N`、单位 bps；节点顺序严格等于统一 `Topology.nodes` 的 `R0…` 顺序，且对角线恒为零。需求是有向 OD 需求，因此矩阵不要求对称。

`gravity` 以节点相邻链路容量（Mbps）为质量 `w_i`，生成 `D_ij ∝ w_i w_j`，并对每个 OD 施加 `[0.90, 1.10]` 的局部确定性扰动。`diurnal` 在 gravity 矩阵上按源节点相位乘以 `1 + 0.35 sin(2π(phase + time_index + offset_i)/24)`；phase、time_index 和每节点 offset 均写入 manifest。`hotspot` 确定性选择 `max(1, floor(N/10))` 个热点，并将相关入/出 OD 乘以 4。`burst` 确定性选择 `max(1, floor(N/8))` 个非对角 OD，并将这些 OD 乘以 8。

每一种模式在默认初始 `NetworkConfiguration` 下通过确定性内核一次性校准到 Normal 目标 MLU 0.25。Low/Normal/High 分别作为同一未缩放 `.npy` 基础矩阵的 `TrafficMatrix.load_multiplier` 0.5/1.0/3.0；不会再次改写基础矩阵。Zoo 未提供 BGP 语义，因此每拓扑生成一条显式标记 `synthetic=true` 的简化 BGP 路由。
