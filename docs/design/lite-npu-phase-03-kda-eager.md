# Lite NPU 阶段 3：KDA Eager Baseline

## 1. 目的

本阶段在阶段 2 已冻结的混合 cache 上接通 Lite KDA 子模块的 eager 数值路径，建立后续公开 NPU
kernel、Decode graph 和 8P8D 服务共同使用的正确性基线。本阶段只加载一个被测 KDA 层或缩小后的
等价子模块，不加载完整模型，不追求性能，也不引入 graph、融合、CP/KVP 或 PD 传输。

阶段 3 分成三个独立提交：

1. 本文：冻结数学、复用边界、kernel ABI 和测试矩阵；
2. Lite KDA eager 实现及 focused tests；
3. CPU/NPU exact-source 验证记录。

## 2. 与现有 Kimi KDA 的差异

两种 KDA 共用以下语义：

- Q/K/V 投影后执行独立的 width-4 depthwise causal convolution 和 SiLU；
- decay gate 为每个 head、每个 key channel 独立的 safe gate；
- recurrent state、conv state、page index 和 Prefill/Decode 生命周期相同；
- 输出执行 per-head RMSNorm、sigmoid output gate 和 OProj。

不能复用的是 beta 的宽度和进入 recurrence 前的处理：

| 项目 | Kimi KDA | Lite KDA |
| --- | --- | --- |
| beta projection | 一次投影，`[T, H]` | 两次无 bias 投影，`[T, D] -> [T, H*D]` |
| beta 语义 | 每个 head 一个标量 | 每个 head、每个 channel 一个值 |
| recurrence 输入 | kernel 内执行 `sigmoid(beta)` | 先把 `sqrt(sigmoid(beta))` 分别乘入 K/V |
| output gate | full-rank `[T,H,D]` | full-rank `[T,H,D]`，语义相同 |

因此不能把 Lite beta reshape 成 `[T,H]`，也不能让 scalar-beta kernel 再执行一次 sigmoid。cache
几何不受 beta 宽度影响，仍复用阶段 2 的 `KimiK3Recipe` 和 `HybridKDATokenToKVPool`。

## 3. Rank-local 形状

设 hidden size 为 `M`，全局 KDA head 数为 `H`，head dim 为 `D`，KDA TP 为 `P`：

```text
H_local = H / P
K_local = H_local * D

Q/K/V/output_gate = [T, K_local]
forget_down         = [T, D]
forget_raw          = [T, H_local, D]
beta_down           = [T, D]
beta_logits         = [T, H_local, D]
packed_qkv          = [T, 3 * K_local]
conv_state          = [slots, 3 * K_local, 3]
recurrent_state     = [slots, H_local, D, D]
```

目标 TP8 下 `H=32`、`D=128`，因此 `H_local=4`、`K_local=512`，packed QKV 宽度为 1536。
`forget_down` 和 `beta_down` 的第一层权重均在 rank 间复制；第二层权重、Q/K/V、output gate、conv、
`A_log`、`dt_bias` 和 OProj 均按 KDA TP 切分。

## 4. 精确数学

### 4.1 Projection 与 causal convolution

对 post-norm hidden state `x`：

```text
q0 = linear(x, Wq)
k0 = linear(x, Wk)
v0 = linear(x, Wv)

f0 = linear(x, Wf0)
f_raw = linear(f0, Wf1).reshape(T, H_local, D)

b0 = linear(x, Wb0)
beta_logits = linear(b0, Wb1).reshape(T, H_local, D)

output_gate = linear(x, Wg).reshape(T, H_local, D)
```

`Wf0/Wf1` 与 `Wb0/Wb1` 都无 bias，两层之间没有激活。Q/K/V 在 token 维分别执行相同边界的
depthwise causal convolution：

```text
y[t,c] = SiLU(sum(i=0..3, w[c,i] * window[t,c,i]))
```

每个 request 的 `window` 由 cache 中前三个原始 projection 值和当前输入组成；写回的 conv state
是最后三个原始 projection 值，不是 SiLU 输出。实现允许把三组输入/权重拼为一个 packed 调用，但
Torch oracle 必须能逐组计算并得到相同结果。

### 4.2 Featurewise beta 与 decay gate

对 convolution 后的 `q/k/v`：

```text
beta_scale = sqrt(sigmoid(beta_logits) + 1e-10)
q_hat = l2_normalize(q, dim=-1)
k_hat = l2_normalize(k, dim=-1) * beta_scale
v_hat = v * beta_scale

gate = lower_bound * sigmoid(
    exp(A_log)[..., None] * (f_raw + dt_bias.reshape(H_local, D))
)
alpha = exp(gate)
```

`lower_bound=-5.0`。所有非线性和 recurrence 累加先转 FP32；回写 recurrent state 保持 FP32。
beta 已吸收到 K/V 后，底层 delta-rule 不再应用 scalar beta。

### 4.3 Delta-rule recurrence

对每个 request、head 和 token，采用 key-major 记忆矩阵 `S[D,D]`：

```text
S_decay = alpha[..., None] * S
delta_v = v_hat - k_hat @ S_decay
S_next = S_decay + outer(k_hat, delta_v)
o = (q_hat * D**-0.5) @ S_next
```

若具体 kernel 使用 value-major `[D,D]` 布局，只能在统一 attention API 边界转置；模型和 cache
manager 不持有第二份 state。Prefill 对每个 packed segment 顺序执行同一 token recurrence，Decode
限制每个 segment 最多一个新 token。两种模式必须得到相同的逐 token 输出和最终 state。

### 4.4 Output epilogue

```text
normalized = RMSNorm(o, W_norm, eps=rms_norm_eps, per_head=True)
gated = normalized * sigmoid(output_gate)
local_output = linear(gated.reshape(T, K_local), Wo)
```

阶段 3 复用统一 `rmsnorm` 入口，并用普通 sigmoid/multiply 完成 gate；融合 epilogue 留到阶段 4。
TP8 的 OProj reduce 留在并行阶段，本阶段的单 rank 子模块验证只检查 rank-local 数值。

## 5. Runtime 与 kernel 边界

### 5.1 模型侧

`LiteKDAParameters` 从参数占位类变为真实 KDA 子模块，但继续保留阶段 1 已冻结的参数名和 loader
target。它执行 projection、调用 `ctx.attn_backend`、执行 output epilogue，并将 featurewise
`beta_logits` 原样传给 backend；不继承或复制 Kimi 整模型。

### 5.2 Hybrid backend

继续使用现有 `KdaAttnBackend`：

- 由 beta tensor 的尾维严格区分 scalar `[T,H_local]` 与 featurewise `[T,H_local*D]`；其他宽度拒绝；
- scalar 路径保持现有 Kimi 行为；
- featurewise 路径不进入只支持 scalar beta 的 fused Decode/verify kernel；
- Prefill/Decode 仍从阶段 2 的 `state_in/state_out` page index 读取和原地发布 conv/recurrent state；
- empty batch 在进入 projection/backend 前返回，不创建临时 state。

不得新增 Lite backend、request-to-slot map、cache pool 或完整 state staging。

### 5.3 Torch reference

统一 kernel package 提供两个可在 CPU/NPU 上运行的低优先级 reference seam：

1. packed causal-conv + SiLU + indexed state update；
2. featurewise-beta Prefill/Decode KDA recurrence。

attention 注册按 `beta_mode=featurewise` 选择 reference，避免误选 scalar-beta kernel。reference 只使用
PyTorch 标准算子，不直接 import `torch_npu`；阶段 4 的公开融合算子通过同一入口替换，runtime
模型代码不变。Python token/segment 循环是本阶段明确接受的 correctness ceiling，不能用于性能结论。

## 6. 状态发布规则

- fresh request 从已清零的 `state_in` 读取；non-zero history 从已有 page 读取；
- Prefill 将每个 segment 的最后 conv window 和 recurrent state 写入 `state_out`；
- Decode 必须支持 `state_in != state_out`，禁止先覆盖共享输入 page；
- 同一 batch 的非空 request 必须使用不同的输出 slot；pad/empty segment 不读写 state；
- 写回前所有输出和 state 必须 finite；发现 NaN/Inf 时 reference fail loud，并报告最早的语义阶段；
- repeated state reuse 必须只改变目标 slot，null page、neighbor slot 和其他 KDA group 不变。

## 7. 测试矩阵

### 7.1 CPU oracle

- 4-token、单 request、zero history，逐级比较 projection、conv、beta scale、gate、每 token 输出和 state；
- 4-token Prefill 与四次连续 Decode 从同一初态开始，逐 token 输出和最终 state 一致；
- non-zero history、`state_in != state_out`、slot reuse 和 neighbor isolation；
- packed variable length，包括 `[4]`、`[1,3]`、`[2,0,2]`；
- empty batch、empty segment、重复 active slot、越界 slot、错误 beta 宽度和错误 dtype/shape；
- 对 Q/K/V、gate、beta、output 和 state 注入 NaN/Inf，确认在最早 reference 边界拒绝。

独立 oracle 不调用 production recurrence helper，避免两条路径共享同一错误。

### 7.2 NPU focused

只使用一张 NPU，先跑 TP8 rank-local production shape 的 4-token case，再跑 variable-length、
multi-request、non-zero history 和连续 Decode。检查：

- 所有语义 tap 与 FP32/Torch oracle 在冻结阈值内；
- Prefill 与逐 token Decode 在同一设备上的逐 token state fingerprint 一致；
- conv/recurrent state 只写目标 page；
- 输出和 state 全部 finite；
- scalar Kimi regression 仍选择原有 ABI，不受 featurewise 路径影响。

NPU focused 不启动服务、不加载完整 checkpoint，不占用多 rank。

## 8. 准入与非目标

阶段 3 准入要求：focused tests 和完整 `pre-commit run --all-files` 通过；exact source 在 NPU 上完成
4-token、变长、多请求、连续 Decode 和 state reuse；每个 token 的输出/state 与 reference 对齐；
Kimi scalar-beta 回归通过。验证记录独立提交后才进入阶段 4。

本阶段不实现或声明以下能力：

- fused causal-conv、recurrent KDA、chunk KDA 或 fused output epilogue；
- Decode graph、overlap、speculative verify/replay；
- CP8、KVP8、PD transfer 或 8P8D 完整服务；
- KDA projection packing、量化或性能收益。
