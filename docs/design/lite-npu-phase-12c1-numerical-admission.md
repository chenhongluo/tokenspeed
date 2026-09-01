# Lite NPU 阶段 12C.1：Featurewise-beta 数值准入修订

## 1. 修订原因

阶段 12C 原设计要求 Decode 的 BF16 output 与 FP32 recurrent state 相对现有 tensor oracle
逐元素 bitwise exact。仓外 Triton-Ascend 原型证明 output 可以保持 exact，但 state 无法在更换
reduction 实现后保持 bitwise exact：现有路径的两次 `torch.matmul` 与候选 kernel 的向量归约采用
不同但合法的 FP32 累加顺序。

这不是数学或 dtype 变化。候选仍按同一顺序执行 Q/K FP32 L2Norm、featurewise-beta scale、forget
gate、K-major FP32 state decay/delta/update 和 readout。把向量归约改成 `tl.dot` 仍有末位差，并使
BS1/BS2 graph 延迟从约 `25/25 us` 退化到 `30/46 us`，因此不能通过更慢的 reduction 伪造
bitwise 等价。

本修订只替换不成立的 bitwise state 判据；数学、state/cache ABI、性能门槛和 8P8D 最终验收均不变。

## 2. 已有上板证据

固定在同一张 Ascend 910B、exact Phase12A source 和同一公开 KDA artifact 上：

| 路径 | 结果 |
|---|---:|
| Prefill T32/T256/T1024 output/state rel-L2 | `0 / 0` |
| Prefill 完整链提升 | `12.44% / 15.33% / 15.14%` |
| Decode graph BS1 | `131.35 -> 24.81 us` |
| Decode graph BS2 | `171.32 -> 24.89 us` |
| Decode 单步/changed-input BF16 output | bitwise exact |
| Decode 单步 FP32 state max abs | `1.49e-8` |
| Decode 128 步 FP32 state rel-L2 / max abs | `2.25e-8 / 2.98e-8` |
| Decode 128 步 BF16 output 最大 abs | `3.05e-5` |
| padding、page 0、未选邻页 | bitwise unchanged |

`tl.dot` 对照的 state max abs 仍为 `1.49e-8--2.98e-8`，说明差异来自 reduction 取整而非
候选公式遗漏。

## 3. 修订后的数值门槛

### 3.1 标准算子验证

生产实现必须通过 `triton-op-verifier` 的浮点计算类三项 AND 判定；测试输出同时返回 BF16 output
与更新后的 FP32 state，不能只验证最终 BF16 张量而隐藏 recurrent state。

### 3.2 Prefill

保持原门槛不变：

- T32/T256/T1024 output rel-L2 `<0.01`、state rel-L2 `<0.005`；
- 全部 finite；empty request compact、initial state 不原地修改和 final-state scatter 语义不变。

### 3.3 Decode 单步与 graph

- BS1/BS2 eager 和 graph output max abs `<=1e-4`、rel-L2 `<=1e-3`；
- FP32 state max abs `<=1e-6`、rel-L2 `<=1e-6`；
- changed-input replay 满足同一门槛；
- padding output 为零，padding/page0/未选邻页必须 bitwise unchanged；
- 所有 output/state finite。

### 3.4 Decode 连续轨迹

至少四个独立 seed、每个 BS2 连续 128 步：

- 每步 BF16 output max abs `<=1e-4`、rel-L2 `<=1e-3`；
- 最终 FP32 state max abs `<=1e-6`、rel-L2 `<=1e-6`；
- 全轨迹无 NaN/Inf，page0 和未选邻页 bitwise unchanged。

四个独立 seed 的 128 步补板中，output 最坏 rel-L2 为 `4.49e-4`，而 max abs 仍不超过
`1e-4`。因此 rel-L2 采用 `1e-3`，避免小范数 output 放大相对指标；它仍比 verifier 的 BF16
通用阈值 `7.81e-3` 严约八倍。FP32 state 门槛保持不变。

## 4. 未改变的性能与系统门槛

- Prefill T32/T256/T1024 至少两个 shape 的完整链提升 `>=3%`；
- Decode BS1、BS2 graph replay 都提升 `>=5%` 且每层至少 `5 us`；
- 不增加第二套 state、cache、graph executor、host sync 或第三方依赖；
- production wrapper 只做 shape/dtype/stride 校验、输出分配和 Triton launch，不用 PyTorch 拼计算；
- executable source 改变后必须完成 exact-source focused NPU、8P8D smoke、GSM8K first100 和资源清理。

任一门槛失败即回退到阶段 12A executable source `5da19b9b`，不保留不可达 kernel 或实验开关。
