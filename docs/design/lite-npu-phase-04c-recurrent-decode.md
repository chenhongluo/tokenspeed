# Lite NPU 阶段 4C：Recurrent Decode

## 1. 目的

本阶段为 Lite featurewise-beta KDA Decode 建立可被 TokenSpeed 固定 batch NPUGraph 捕获的 recurrent
实现。它必须保持阶段 2 的独立 `state_in/state_out` page、尾部 padding 和 K-major FP32 state，不改变
runtime/backend、cache 几何、PD 协议或 checkpoint 参数。

阶段 4C 分成三个独立提交：

1. 本文：冻结数学、state 发布、kernel selection、公开算子拒绝证据和上板门槛；
2. Ascend graph-safe tensor 实现与 focused tests；
3. exact-source NPU 数值、graph replay 和性能验证记录。

本阶段不接 Prefill chunk KDA，不接 output epilogue，不启动完整模型服务。

## 2. 复用边界

统一入口继续是 `tokenspeed_kernel.ops.attention.kda_paged_decode`：

```text
q/k/g_raw       [1,B,H,K]       BF16/FP16
v               [1,B,H,V]       BF16/FP16
beta_logits     [1,B,H] or [1,B,H,K]
A_log           [H]             FP32
dt_bias         [H,K] or [H*K]  FP32
state_pool      [pages,H,K,V]   FP32, K-major
read_indices    [B]             device int32/int64
write_indices   [B]             device int32/int64
cu_seqlens      [B+1]           device int32/int64
```

返回 `[1,B,H,V]`，只向 active `write_indices` 发布 final state。runtime 不导入 `torch_npu`、TorchAir、
公开 artifact loader 或 vendor op；Ascend 实现继续通过现有 kernel registry 注入。

Decode 的 graph padding 同时满足：

- `read_indices/write_indices=-1`；
- 对应 `cu_seqlens` segment 长度为 0；
- padding 位于 batch 尾部。

page 0 是逻辑零页，不能因 padding 的安全索引替换被修改。

## 3. Lite recurrent 数学

### 3.1 Featurewise beta

Lite beta logits 与 Q/K 同形，不能压缩成 head scalar：

```text
s       = sqrt(sigmoid(beta_logits) + 1e-10)
q0      = L2Norm(q)
k0      = L2Norm(k) * s
v0      = v * s
decay   = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
S_decay = exp(decay)[...,K,1] * S_read
delta   = v0 - k0^T * S_decay
S_write = S_decay + k0 * delta
output  = q0^T * S_write / sqrt(K)
```

全部 recurrent accumulation 和 state 保持 FP32，output 最后转换为输入 dtype。该表达式与阶段 3
reference 相同，只把每 request Python 循环改成 batch tensor 运算。

### 3.2 Scalar beta 兼容

同一 Ascend registration 还需保持现有 Kimi scalar-beta ABI：Q/K 只做 L2Norm，`sigmoid(beta)` 乘在
`delta` 上。这样替换 NPU reference 时不会让一个 Lite 优化破坏已有 KDA 模型。

## 4. Graph-safe state 发布

实现只使用 device tensor 运算：

```text
segment_active = cu_seqlens[1:] > cu_seqlens[:-1]
active         = segment_active & (read >= 0) & (write >= 0)
safe_read      = where(active, read, 0)
safe_write     = where(active, write, 0)
state          = state_pool.index_select(0, safe_read)
next_state     = recurrent(state, q, k, v, gate, beta)
destination    = state_pool.index_select(0, safe_write)
published      = where(active, next_state, destination)
state_pool.index_copy_(0, safe_write, published)
output         = where(active, output, 0)
```

active write page 必须唯一；该契约已由统一 reference 的 metadata 校验保证。padding 全部安全映射到
page 0，但发布值严格等于 page 0 的原值，所以捕获和 replay 都不得污染它。实现中禁止 `.item()`、
`.tolist()`、device-to-host copy、Python request 循环和运行时 tensor shape 分支。

## 5. 公开 RecurrentKda 审计

### 5.1 数值适配成立

公开算子只接受 head-scalar beta。Lite 的等价适配是在 kernel 外完成 featurewise prepare，再传 FP32
head-scalar ones，并关闭 kernel 内 Q/K norm 和 beta sigmoid。NPU0 验证结果为：

| 场景 | output rel-L2 | state rel-L2 | 结论 |
| --- | ---: | ---: | --- |
| BS1 单步，对生产 FP32 prepare | `0.00371` | `0.00233` | 通过 |
| BS2 单步，对生产 FP32 prepare | `0.00405` | `0.00239` | 通过 |
| BS2 连续 128 步 | `0.00317` | `0.00153` | finite，无单调放大 |

对“prepare 后转 BF16”的独立 oracle，output bitwise 一致，state 最大绝对差不超过 `9.31e-10`。
`read!=write`、BS2 中 1 个 active、更新输入/index/边界的 graph replay、page 0 和 neighbor isolation
全部通过。因此公开候选的拒绝原因不是精度。

### 5.2 当前执行模型下不准入

公开 binding 的 K-major eager/NPUGraph 路径都保留每次约 2 ms 的调度/tiling 成本：

| BS | 公开 K-major NPUGraph | graph-safe tensor NPUGraph |
| ---: | ---: | ---: |
| 1 | `2049.7 us` | `130.0 us` |
| 2 | `2095.8 us` | `172.2 us` |
| 8 | `2187.3 us` | `231.5 us` |
| 64 | `12685.4 us` | `885.7 us` |
| 256 | `46342.8 us` | `2903.8 us` |

公开 kernel 的高性能路径要求 TorchAir converter 和 V-major state。active state 做
K-major→V-major→K-major compact 后，TorchAir 完整 adapter 的 BS1/2 时间为 `269.1/257.8 us`，仍分别
慢于 tensor NPUGraph 约 `2.07x/1.50x`。此外，TorchAir compiled graph 在稳定 pool、index、stream 和
四次同 stream warmup 后仍不能嵌套 TokenSpeed NPUGraph capture；capture 内 `LoadGraph` 会触发被
runtime 禁止的 H2D memcpy。

因此本阶段不注册公开 recurrent kernel，也不为它增加第二套 graph executor、TorchAir runtime 依赖、
state layout 或 cache。阶段 4A 的 artifact 保留，供 Prefill 和后续架构评估复用。

只有满足以下任一条件时才重新评估：

- TokenSpeed NPU Decode 统一迁移到 TorchAir graph executor；
- 公开 op 可被 NPUGraph 原生捕获且 BS1/2 完整 adapter 快于 tensor 路径；
- 出现直接消费 K-major 双页 state 的新融合 kernel。

## 6. Kernel selection

Ascend 的 `kda_paged_decode` 默认选择 graph-safe tensor implementation，traits 保持：

```text
indexed_state=True
single_token=True
recurrent_layout=k_major
beta_mode={scalar,featurewise}
```

现有 kernel 名和统一 API 不变；只把 Ascend registration 从会 host-sync 的 portable reference 切到 NPU
graph-safe 实现。CPU、非 Ascend 和显式 reference 继续使用阶段 3 实现。由于公开 recurrent 未注册，显式
选择 `solution=public_kda` 必须 fail loud，不能静默回落到 tensor 路径。

## 7. 测试矩阵

### 7.1 CPU/meta

- featurewise/scalar beta 分别对齐独立 reference；
- BS1/2/8、`read==write`、`read!=write`、padding、page 0 与 neighbor isolation；
- 重复 active write、单边负 index、非单 token segment 和错误 state layout fail loud；
- Ascend selection traits 命中 graph-safe implementation，显式公开 solution 不可用时 fail loud；
- 现有 Kimi/Lite recurrent、cache recipe 和 loader tests 全部回归。

### 7.2 NPU0

- Lite TP8 `H=4,K=V=128`，BS1/2/8 与连续 1/2/8/32/128 步；
- scalar-beta 至少一个 BS2 回归；
- NPUGraph capture/replay 后更新 Q/K/V/gate/beta、read/write index 和 padding 边界；
- active output/state 对 FP32 reference 通过，未选 page bitwise 不变，所有张量 finite；
- BS1/2/8/64/256 graph replay 计时；至少不得复现公开 K-major 的约 2 ms 固定成本。

性能是记录项，不设吞吐目标；任何 NaN/Inf、page 0 mutation、state 漂移放大或 graph value 冻结都直接
拒绝实现。

## 8. 回退与后续

实现失败时只恢复 Ascend registration 指向阶段 3 reference，不改 runtime/cache。实现通过后进入阶段
4D Prefill gate/chunk KDA；公开 recurrent 的 TorchAir/V-major实验不进入生产提交。

阶段 4C 明确不包含：

- causal-conv + gate + recurrent 的新融合 kernel；
- TorchAir/NPUGraph 双 executor 管理；
- Prefill gate-cumsum/chunk KDA；
- output epilogue、speculative verify/replay；
- 8P8D、PD one-copy、CP8/KVP8 或完整服务。
