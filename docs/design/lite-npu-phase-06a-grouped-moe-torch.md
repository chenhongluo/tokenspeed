# Lite NPU 阶段 6A：Grouped MoE Torch 语义与 packed owner

## 1. 目标与边界

本子阶段只建立可在 CPU/meta 独立验证的 Grouped MoE 数学和 canonical packed 权重加载，作为后续
Ascend leaf、EP8 collective 与 A/B/C placement 实验的统一 oracle。

本阶段实现：

- 四组独立 softmax、selection-only correction bias、TopK12 和 route scale；
- real/identity expert 分流、shared expert 单次计算与外围 projection；
- canonical `w13/w2` 参数 owner；
- 复用通用 MoE loader 的 EP contiguous ownership 和 gate/up/down packing；
- N=1 Torch reference forward、CPU/meta 与 strict loader 测试。

本阶段不实现：

- Ascend fused TopK、routing/GMM/SwiGLU/finalize；
- Prefill AG/RS、Decode local-expert/AllReduce 或 NPUGraph；
- A/B/C physical placement；
- N=8 Grouped MoE forward、量化、Weight-NZ 或服务接线。

因此 N=8 仍可构造和加载参数，但调用 Grouped MoE forward 必须明确 fail closed。只有 N=1 reference
forward 在本阶段可执行。

## 2. 复用与最小改动

### 2.1 Packed owner

不继续维护当前逐 expert `ModuleDict`。Lite 建立一个无执行计划的 packed owner，并直接复用：

- `MoELayerSpec` 描述 `TP=1`、`EP=N`、local expert 数和 H/I；
- `create_dense_weight_pair` 创建 canonical unquant 参数及已有 `weight_loader`。

固定 canonical shape：

```text
w13_weight = [num_local_experts, 2 * I, D]
w2_weight  = [num_local_experts, D, I]
```

其中 `D=hidden_size/group_count`。`w13` 前半为 gate、后半为 up，与通用 loader 的 `w1/w3` 默认顺序
一致。6A 不实例化完整 `MoELayer`，因为后者在构造期选择设备 kernel plan；kernel plan 属于 6B，参数
owner 不应因此依赖可用加速器。

### 2.2 Loader

严格 loader 继续逐 source key 验证 exact name、shape、dtype 和 duplicate。expert source 通过同一个
`build_moe_checkpoint_loader`：

```text
experts.{id}.gate_proj.weight -> experts.w13_weight[local_id, :I]
experts.{id}.up_proj.weight   -> experts.w13_weight[local_id, I:]
experts.{id}.down_proj.weight -> experts.w2_weight[local_id]
```

通用 loader 的 global expert plan 识别所有 rank 的 source key；只有 contiguous EP owner 的 key 写入本地
参数。外层 strict loader 对 source 仍只允许一次，但允许 gate/up 两个 source 合法写入同一个 packed target。
阶段结束时每层的 `w13_weight` 和 `w2_weight` 都必须出现在 loaded-target coverage 中。

### 2.3 CPU/meta 导入边界

当前 `tokenspeed.runtime.layers.moe` 和 unquant weight helper 因 eager import 量化/设备 kernel，使只使用
schema、loader 或 dense packed 参数也要求加速器。这不是 Lite 数学要求。

最小修正为：

1. MoE package export 对执行模块使用 module-level lazy import；
2. `create_layer_weights` 只在选中的 quant kind 分支导入相应 creator；
3. FP8 quant helper 只在 `scale_dst` 实际存在且 dtype 需要量化时导入。

已有 accelerator 调用仍在第一次访问相同对象时加载同一实现；不新增 fallback、不改变 kernel 选择。

## 3. N=1 数学

### 3.1 Projection 与分组

输入 `x:[T,H]`：

```text
p = linear(x, W_in)                                      # [T,H]
n = bf16(fp32(p) * rsqrt(mean(fp32(p)^2) + eps) * norm) # [T,H]
u = view(n * grouped_moe_norm_scale, [T,G,D])            # [T,G,D]
```

N=1 无 dense collective。先对完整 H 做 RMSNorm，再 reshape；不能对 group 独立归一化。

### 3.2 Router

每组 `g` 独立计算：

```text
logits_g = linear(fp32(u[:,g,:]), fp32(W_router_g))       # [T,E+Z]
prob_g   = softmax(logits_g, dim=-1)
ids_g    = topk(prob_g + correction_bias_g, K)
weight_g = gather(prob_g, ids_g) * routed_scaling_factor
```

要求：

- correction bias 只改变选择，不进入输出 weight；
- 不对 TopK weight 重新归一化；
- route weight 保持 FP32，ID 转为 INT32；
- 测试避开 tie，ID 必须逐元素一致。

### 3.3 Real 与 identity

组内 `id<E` 是 real expert，flattened group-major ID 为 `g*E+id`。N=1 reference 只遍历当前输入实际
命中的 real expert：

```text
h13 = linear(u_selected, w13[flat_id])
gate, up = split(h13, I)
y = linear(silu(gate) * up, w2[flat_id])
group_out[token] += bf16(weight) * y
```

组内 `E<=id<E+Z` 是 identity expert：

```text
identity_weight = sum(weight where id >= E, dim=topk)
group_out += bf16(identity_weight) * u
```

identity route 不索引 packed real expert，也不改变 real route 的权重。

### 3.4 输出与 shared expert

```text
routed = linear(view(group_out,[T,H]), W_out)
shared = linear(silu(linear(x,W_shared_gate)) * linear(x,W_shared_up), W_shared_down)
output = routed + shared
```

shared expert 始终消费原始 `x`，每层只执行一次。空 token 输入直接返回同 shape tensor，不进入 TopK 或
expert 循环。

## 4. 参数名和 strict coverage

替换前后的每层 target：

| checkpoint source | 旧 target | 新 target |
| --- | --- | --- |
| 1536 个 gate weight | 1536 个独立参数 | `experts.w13_weight` 前半 |
| 1536 个 up weight | 1536 个独立参数 | `experts.w13_weight` 后半 |
| 1536 个 down weight | 1536 个独立参数 | `experts.w2_weight` |

checkpoint source 数、source shape/dtype、router/dense/shared/OE target 均不改变。只减少 runtime 参数对象
数量，不改变总元素数。

对于小配置 `G=4,E=8,EP8,D=24,I=16`，rank3 应持有 flattened expert `12..15`，但 packed local index
固定为 `0..3`；loader 测试必须分别向 gate/up/down 写入不同 sentinel，以证明顺序和 local ID 正确。

## 5. 测试

### 5.1 CPU

- package/schema/loader/unquant packed owner 在无加速器进程可导入；
- packed shape、BF16 dtype、EP rank0/middle/last ownership；
- gate/up/down sentinel 正确写入 W13 前半、后半和 W2；
- 非本 rank expert source 被识别但不写入，本 rank target coverage 完整；
- missing/duplicate/unexpected/shape-dtype strict failure 保持不变；
- correction bias 改变 ID 但 weight 来自无 bias softmax，scale=6 且不 renorm；
- all-real、all-identity、mixed、duplicate route 与 shared-once 对独立 loop oracle；
- T0/T1/T4 输出 shape、dtype、finite。

### 5.2 Meta

- N=1 和 N=8 构造不分配 payload；
- packed/local expert shape 与 dense/router shape 正确；
- strict loader 在 meta target 上完成 source/target coverage，不执行 copy。

### 5.3 回归

- 既有 Lite config/layout/KDA/MLA tests；
- 通用 MoE loader/expert tests；
- 全仓 pre-commit。

## 6. 准入与回退

准入要求：

- route ID exact；route weight `atol=1e-6,rtol=1e-5`；
- packed full output 对独立 FP32/BF16 boundary oracle `atol=2e-2,rtol=2e-2`；
- 所有输出 finite，identity/shared 分支均有独立断言；
- 每个本地 packed target source shard 无重无漏；
- 无加速器导入不加载设备 kernel，现有加速器路径的 import/API 不变。

若 lazy import 回归既有 MoE API，则回退该导入整理并把 6A CPU loader 测试迁到 meta-safe isolated module；
不能因此复制第二套 loader。若 packed loader coverage 不成立，则保持本阶段不接 forward，先修通用 loader
映射，不恢复逐 expert `ModuleDict`。

## 7. 提交边界

1. 本设计文档独立提交并推送；
2. lazy import、packed owner、loader、N=1 reference 与 tests 作为一个实现提交；
3. exact-source CPU/meta/NPU import 回归结果写入独立验证记录并提交。

6A 完成后才进入 6B Ascend fused TopK 和 local-expert leaf。
