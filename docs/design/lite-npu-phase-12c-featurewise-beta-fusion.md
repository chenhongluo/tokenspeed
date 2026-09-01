# Lite NPU 阶段 12C：Featurewise-beta 融合准入

## 1. 目标与边界

本阶段只评估 Lite KDA 中 featurewise-beta 引入的额外计算能否在 Ascend 910B 上获得可复现收益：

- Prefill：Q/K L2Norm、`sqrt(sigmoid(beta_logits) + 1e-10)` 与 K/V featurewise scale；
- Decode：上述 prepare、forget gate、K-major FP32 recurrence、readout 与 live-slot 发布。

现有数学、TP8 rank-local shape、K-major state、PD one-copy、Decode graph BS1/BS2、P/D overlap 和
8P8D 拓扑均不改变。本阶段不引入第二套 state layout、cache、graph executor、runtime registry 或依赖，
也不把 output epilogue、causal convolution或 projection 重新纳入融合范围。

候选未通过数值、graph 或性能门槛时，只提交负验证记录，不保留死代码或生产开关。

## 2. 当前生产路径

目标 TP8 rank-local shape 为 `H=4, K=V=128`。

### 2.1 Prefill

现有 adapter 在公开 KDA core 前执行：

```text
q_prepared = L2Norm(q.fp32).bf16
scale      = sqrt(sigmoid(beta_logits.fp32) + 1e-10)
k_prepared = (L2Norm(k.fp32) * scale).bf16
v_prepared = (v.fp32 * scale).bf16
beta_core  = ones([1,T,H], bf16)
gate       = public KdaGateCumsum(g_raw, A_log, dt_bias)
out,state  = public ChunkKdaFwd(q_prepared, k_prepared, v_prepared,
                                gate, beta_core, initial_state)
```

公开 `KdaGateCumsum + ChunkKdaFwd` 只接受 head-scalar beta。Lite 通过在 core 外完成 featurewise
scale，再传 unit beta 保持数学等价。公开 core 的约 2.5 ms 固定成本不属于本阶段可消除部分。

### 2.2 Decode

现有 `torch_kda_paged_decode` 已在 TokenSpeed NPUGraph 内以 batch tensor 执行完整单 token路径：

```text
metadata active/safe-index
Q/K FP32 L2Norm
featurewise beta sigmoid/sqrt 与 K/V scale
forget gate
state gather + decay + read/correction + outer-product update
live-slot index_copy publication
BF16 output
```

它没有 request Python loop、`.item()`、D2H 或 host metadata sync。阶段 4C 的 exact-source
BS1/BS2 graph 基线为约 `127.8/166.4 us`。公开 RecurrentKda 的 K-major 路径约 2 ms；V-major
TorchAir adapter 仍约 `269.1/257.8 us` 且不能嵌套 TokenSpeed NPUGraph，均不是生产候选。

## 3. 先测上界，再决定是否写 kernel

遵循最小实现原则，本阶段先测“可消除部分”的理论上界，而不是先维护一个新 AscendC 工程。

### 3.1 Prefill 上界板

在 T32/T256/T1024 分别测量：

1. 完整生产 adapter；
2. 单独 prepare；
3. 预先准备 Q/K/V 和复用 unit-beta 后的公开 gate/chunk core；
4. profiler 中 prepare 的 operator 数、device self time 与临时 tensor bytes。

若 prepare 在至少两个 shape 上占完整链 `<3%`，则即使理想零成本 kernel 也达不到链路准入门槛，
直接负准入。若达到门槛，再比较当前环境中已有的公开/已安装融合能力；只有已有能力可复用或最小
prototype 有明确收益时才进入生产实现。

unit-beta 先验证模块级或 graph-stable storage 复用。它只有 `[1,T,H]`，不能为了删除这个小分配修改
公开 ChunkKda ABI；除非 profiler 证明它是主因，否则保持当前显式契约。

### 3.2 Decode 上界板

BS1/BS2 分别测量：

1. 完整生产 graph replay；
2. profiler 中 normalize、sigmoid/sqrt/scale、gate、state gather/publication 和 matmul/outer-product；
3. 本地 probe 中预先准备 Q/K/V/gate 后的 recurrence-only 下界；
4. 交替顺序的重复 graph trial，避免固定顺序和 warmup 偏差。

该下界只用于判断新 kernel 的最大可得收益，不接入 runtime。若 prepare/gate 的理想消除上界仍不能
跨过门槛，或现有公开 recurrent core 本身已慢于生产路径，则不开发新的 production kernel。

## 4. 候选复用顺序

只按以下顺序尝试，命中即停止：

1. 现有生产 tensor 路径和已有公开 KDA artifact；
2. 当前环境已安装且可由统一 kernel 边界直接调用的原生能力；
3. 不改变 state/graph ABI 的最小 prototype；
4. 只有前三项证明收益后，才讨论独立 AscendC kernel。

当前审计已经排除以下伪候选：

- 给约 2 ms 的公开 K-major RecurrentKda 增加 featurewise-beta 参数；
- 为 V-major TorchAir adapter 增加第二套 state 或 graph executor；
- 仅用 Python helper 包住多条 Torch op 并称为融合；
- 为 unit-beta 小 tensor 单独修改公开 ChunkKda ABI；
- 把 output norm/gate、causal conv 或 projection 跨阶段重新合并。

## 5. 数值与状态契约

任何候选必须保持以下顺序：

```text
q0 = q / ||q||2
k0 = k / ||k||2
s  = sqrt(sigmoid(beta_logits) + 1e-10)
k1 = k0 * s
v1 = v * s
```

并保持：

- recurrent accumulation/state 为 FP32，输出最后转回输入 dtype；
- `A_log`、`dt_bias`、`lower_bound` 的现有 gate 公式不变；
- active row 只发布到对应 live write slot；
- padding row 输出为零，page 0 和未选邻页 bitwise 不变；
- BS2 changed-input replay 不冻结 Q/K/V/beta/gate/index；
- Prefill empty request compact、输入 state 不原地修改、final state scatter 语义不变。

Prefill 对量化后逐 token oracle 要求 output rel-L2 `<0.01`、state rel-L2 `<0.005`；Decode 对当前
graph-safe tensor oracle 要求 output/state exact。所有输入、输出和 state 必须 finite。

## 6. 性能准入门槛

计时固定在同一张 910B、同一进程、同一 source/artifact，先 warmup，再采用交替顺序的至少 11 轮重复
trial，以中位数作判断。

### Prefill

- T32/T256/T1024 中至少两个 shape 的完整 prepare+gate+chunk 链提升 `>=3%`；
- 不增加大于现有 prepared Q/K/V 的峰值临时量；
- 不增加公开 core launch 数或 host/device 同步。

### Decode

- BS1 和 BS2 graph replay 都必须提升 `>=5%` 且每层至少 `5 us`；
- changed-input replay、padding、read/write 换页和 live-slot publication 全部通过；
- 不新增 graph fallback、二次 executor、state transpose 或 host sync。

任何一项失败即负准入。孤立 eager 小算子收益不能替代 graph replay 门槛，也不外推为服务收益。

## 7. 实施与提交顺序

1. 提交本文，冻结边界、上界板和门槛；
2. exact-source NPU0 运行 Prefill/Decode 上界板与 profiler；
3. 仅当上界和可复用候选均满足门槛时，提交最小实现与 focused tests；
4. 提交独立验证记录；若无 executable source 变化，不重复 8P8D/GSM8K；
5. 若有生产变化，按总体 Phase 12 的完整 8P8D、GSM8K first100 和清理门禁验证。

回退点是阶段 12A 已验证的 executable source `5da19b9b`。阶段 12B 和本设计只增加文档，不改变该
执行源码。
