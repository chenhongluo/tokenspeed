# Lite NPU 阶段 4E：KDA Output Epilogue

## 1. 目的

本阶段把 Lite KDA core 后的逐算子 RMSNorm、sigmoid output gate 和逐元素乘，替换为仓库已有的
`rmsnorm_gated_sigmoid` Triton kernel。该调用位于 Lite 模型层、KDA backend 返回之后，因此同一条
路径同时覆盖 Prefill 与 Decode，并保持后续 `o_proj` 不变。

阶段 4E 分成三个独立提交：

1. 本文：冻结复用边界、数值语义、CPU 回退、graph 和测试门槛；
2. Lite model 最小接线与 focused tests；
3. exact-source NPU 数值、graph replay、性能和回退验证记录。

本阶段不新增 kernel，不复制公开 FLA/vLLM-Ascend epilogue，不改 KDA backend/cache/PD，也不启动完整
8P8D 服务。

## 2. 当前数据流

Lite TP8 rank-local output epilogue 的输入为：

```text
core_output   [T,H,D]       BF16，H=4，D=128
output_gate   [T,H*D]       BF16，最后一维 stride=1
norm_weight   [D]           BF16
eps           scalar        config.rms_norm_eps
```

当前模型依次执行：

```text
x_fp32     = fp32(core_output)
rms        = x_fp32 * rsqrt(mean(x_fp32^2, dim=D) + eps)
normalized = bf16(rms * fp32(norm_weight))
gate       = bf16(sigmoid(fp32(output_gate)))
gated      = normalized * gate
output     = linear(flatten(gated), o_proj)
```

这里 RMS reduction 按 head 独立进行，`norm_weight[D]` 在所有 head 间共享；不能把 `[H,D]` 展成单个
`H*D` RMSNorm，也不能把 gate 提前传入只覆盖 Decode 的 backend 分支。

## 3. 复用边界

仓库已有 `tokenspeed_kernel.ops.activation.triton.rmsnorm_gated_sigmoid`：

- 单个 Triton kernel 完成每 head FP32 RMS reduction、weight、sigmoid gate 和最终乘；
- 输入 `x` 为连续 `[T,H*D]`；
- gate 只要求形状相同且最后一维 stride 为 1，支持 future packed projection 的行跨步 slice；
- Kimi K3 模型和 hybrid KDA Decode 已复用同一个实现；
- CUDA 与 Ascend Triton 后端均可执行，返回 dtype 与 `x` 相同。

Lite 只在 `LiteKDAParameters.forward` 的现有 epilogue seam 调用它：

```text
core_output [T,H,D]
    -> flatten(1).contiguous()
    -> rmsnorm_gated_sigmoid(output_gate, o_norm.weight, eps, H, D)
    -> o_proj
```

不把 output gate 传入 `ctx.attn_backend.forward`。原因是当前 backend 的 fused gate seam 只在部分
Decode 实现消费，Prefill 返回 core output/final state 后仍需模型侧 epilogue；在模型侧接一次可以同时
覆盖 P/D，避免两套选择和重复 gate。

不增加统一 kernel registry entry。该 kernel 已是模型与 backend 直接复用的 portable Triton leaf；为
一个现有单实现再包一层 registry 不会增加回退能力。

## 4. 数值语义

融合 kernel 采用更直接的语义：

```text
y = bf16(
    fp32(x) * rsqrt(mean(fp32(x)^2) + eps)
    * fp32(weight)
    * sigmoid(fp32(gate))
)
```

它只在最终落盘时转 BF16；旧路径在 norm 和 sigmoid 后分别转 BF16，再做 BF16 乘。因此两者不要求
bitwise 一致，差异来自旧路径的额外舍入，不是数学变化。

910B NPU0、`H=4,D=128`、行跨步 gate 的 discovery 结果如下：

| tokens | 对旧手工路径 max abs | 对旧手工路径 rel-L2 | 对一次 FP32 语义 rel-L2 |
| ---: | ---: | ---: | ---: |
| 1 | `0.0078125` | `0.003152` | `0` |
| 2 | `0.0078125` | `0.003375` | `0` |
| 8 | `0.015625` | `0.003232` | `0` |
| 32 | `0.015625` | `0.003351` | `0` |
| 256 | `0.015625` | `0.003344` | `2.10e-5` |
| 1024 | `0.015625` | `0.003353` | `9.20e-6` |

所有结果 finite。实现门槛固定为：

- 对旧手工 BF16 路径：`max_abs <= 0.02` 且 `rel_L2 <= 0.005`；
- 对一次 FP32 语义再转 BF16：`max_abs <= 0.004` 且 `rel_L2 <= 1e-4`；
- 不允许输出 dtype、shape 或 per-head norm 轴变化。

## 5. CPU/meta 回退

现有 Lite CPU focused test 直接执行 model forward，Triton kernel 不能消费 CPU tensor。因此只在
`core_output.device.type` 为 `cuda` 或 `npu` 时调用融合 kernel；CPU/meta 和其它未验证 device 保留当前
Torch 表达式。

该分支由 tensor device 静态决定：NPU Decode graph capture/replay 内不会改变，不引入 host sync、
shape 判断或 graph 输入冻结。CPU 回退是测试/reference 能力，不参加 NPU kernel selection，也不增加
用户配置开关。

## 6. Decode graph 与布局

Decode BS1/2 discovery 已完成 NPUGraph capture、replay、更新 `x/gate/weight` 后再次 replay：两次结果
均对旧手工路径最大差 `0.0078125`，且全部 finite，证明 kernel 没有把输入值冻结在 capture 时刻。

生产接线必须保持：

- `core_output.flatten(1).contiguous()`，避免 backend 返回 view 时违反 kernel 的连续输入契约；
- `output_gate` 不做无条件 `.contiguous()`，保留 kernel 对 packed projection 行跨步 slice 的支持；
- gate 的最后一维 stride 必须为 1，现有独立 `F.linear` 输出和后续 packed slice 都满足；
- kernel 输出直接作为 `o_proj` 输入，不 reshape 后再产生一次 copy。

Prefill 不使用 graph，但采用同一 epilogue；T32/256/1024 覆盖短 prompt、常见 chunk 和长 packed
token 形状。

## 7. 性能记录

NPU0 同步 steady-state discovery 的单层 epilogue 时间为：

| tokens | fused | 当前手工路径 |
| ---: | ---: | ---: |
| 1 | `41.37 us` | `100.57 us` |
| 2 | `41.04 us` | `100.06 us` |
| 8 | `40.68 us` | `95.23 us` |
| 32 | `40.41 us` | `107.85 us` |
| 256 | `41.75 us` | `108.28 us` |
| 1024 | `78.86 us` | `110.07 us` |

性能只作准入证据，不设服务吞吐目标。目标是删除同一层的多次 launch/中间 tensor；本阶段不把
`o_proj` GEMM 合入 epilogue，也不开发 KDA core + epilogue mega-kernel。

## 8. 测试矩阵

### 8.1 CPU/meta

- 现有 Lite model forward 继续验证 featurewise beta、backend 参数和 output gate 数学；
- CPU 回退对独立 per-head RMSNorm + sigmoid oracle；
- 空 token 仍在进入 backend/epilogue 前原样返回；
- gate 不传入 backend，Prefill/Decode 继续共享模型侧语义；
- 现有 Kimi、hybrid KDA 和 Phase 3/4 focused tests 全部回归。

### 8.2 NPU0

- Lite TP8 `H=4,D=128`，T1/2/8/32/256/1024；
- 连续 gate 与行跨步 gate，最后一维 stride=1；
- 对旧 BF16 路径和一次 FP32 语义分别执行上述数值门槛，所有输出 finite；
- Decode BS1/2 NPUGraph capture/replay，更新 `x/gate/weight` 后结果随输入变化；
- T1/2/8/32/256/1024 同步 steady-state A/B，性能只记录；
- exact-source focused suite，不依赖 Phase 4A 自定义 OPP artifact。

## 9. 准入与回退

实现提交要求本地 focused、全量 `pre-commit run --all-files` 和 NPU0 exact-source focused 全部通过；
model 侧仅一处 accelerator 分支；NPU/CUDA 不再执行旧的多算子链；CPU/reference 行为不回归；Decode
graph replay 不冻结值。

若任一 production shape、row-strided gate、数值或 NPUGraph 门槛失败，只恢复 Lite model 的旧 epilogue；
不修改现有共享 kernel、不复制第三方实现，也不在 backend 增加 P/D 分支。

阶段 4E 明确不包含：

- output epilogue 与 KDA recurrent、causal-conv 或 `o_proj` 的新 mega-kernel；
- packed QKV/gate/forget/beta projection；
- Prefill graph、PD one-copy、CP8/KVP8 或完整 8P8D 服务；
- Grouped MoE、MLA 或 OE 优化。
