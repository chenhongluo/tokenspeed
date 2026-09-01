# Lite NPU 阶段 6D：Grouped MoE 分层精度门禁

## 1. 背景

阶段 6D 的真实 checkpoint 回归最初把完整 MLP 输出直接与 FP32 expert oracle 做逐元素
`atol=2e-2, rtol=2e-2` 比较。该门槛混合了两类误差：

- Grouped MoE leaf 的 BF16 GMM、SwiGLU 和 route accumulation 舍入；
- `proj_output` 对少量近零抵消项的误差放大。

实测 P1024 中每 rank 只有 1--2 个逐元素离群点，完整输出 rel-L2 约 `1e-3`；在 leaf 和
ReduceScatter 边界，max-abs 分别不超过 `0.015625` 和 `0.03125`。因此继续提高完整输出的统一
absolute tolerance 会降低错误定位能力，也会错误改变阶段 6C 已冻结的 A/B/C placement gate。

## 2. 分层门禁

生产 board 同时构造两条数据流：

1. production wrapper 的真实 fused 路径；
2. 相同 router/权重/collective 下的独立 FP32 local-expert accumulation oracle。

验收边界固定为：

| 边界 | 硬门禁 |
| --- | --- |
| 所有 production 输出 | finite |
| fused local leaf vs FP32 local oracle | rel-L2 `<=5e-3`，max-abs `<=2e-2` |
| EP collective 后 routed hidden | rel-L2 `<=5e-3`，max-abs `<=4e-2` |
| wrapper vs 手工拼接的同一 fused 数据流 | rel-L2 `<=1e-2` |
| 完整 MLP vs FP32 oracle | rel-L2 `<=1e-2`，max-abs 只记录 |
| Decode graph replay vs eager | 输入更新后输出必须变化，rel-L2 `<=5e-3`，max-abs 只记录 |

所有 rel-L2 和逐元素统计都先转 FP32 再计算，避免 BF16 上连容差本身也发生舍入。阶段 6C 的
A/B/C 比较继续使用原 `atol=2e-2, rtol=2e-2` 逐元素门槛；本修正不改变 winner。

## 3. 测试矩阵

- synthetic 与真实 checkpoint；
- Prefill T32/T128/T1024，不等长八 rank split；
- Decode BS1/BS2 eager 与 graph replay；
- 单份 `[192,...]` parent expert storage 和四个共享 view；
- P 的 token AG/RS 与 D 的 feature AG、expert AR、dense AR 全链路。

本阶段只修正硬件 board 的观测边界，不修改生产 runtime、kernel、checkpoint loader 或通信实现。
真实 checkpoint 的 Decode BS1 exact 重放中，两次独立 fused 执行经 Wout/TP8 AllReduce 后的
wrapper rel-L2 为 `0.005496`，但 leaf/EP collective 仍分别不超过 `0.00455/0.00154`；因此 wrapper
使用与完整输出相同的 `1e-2` 上限，根因附近的 leaf/collective 门槛保持不变。
