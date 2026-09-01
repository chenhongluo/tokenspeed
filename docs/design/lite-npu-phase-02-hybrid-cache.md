# Lite NPU 阶段 2：混合 Cache 与 Graph 稳定元数据

## 1. 目的

本阶段把 Lite 的 21 层 KDA state 和 7 层 MLA latent KV 接入 TokenSpeed 已有的混合 cache
基础设施，并冻结后续 KDA/MLA kernel、PD 和 Decode graph 共同依赖的内存契约。本阶段不实现
attention 数值 forward，也不引入 Lite 专用 allocator、recipe 或 graph runner。

阶段 2 分成三个独立提交：

1. 本文：冻结复用边界、byte layout、容量和生命周期契约；
2. Lite architecture 注册及 focused tests；
3. CPU/meta/NPU 验证记录。

## 2. 复用边界

Lite 与 Kimi K3 的 cache 数学相同：MLA 保存压缩 latent history，KDA 保存 width-4 causal-conv
状态和 recurrent matrix state。现有 `KimiK3Recipe` 已由 config 推导层数、KDA TP 宽度和 MLA
几何，并只对 93 层 Kimi 启用 69/24 的特例校验；Lite 的 28 层、21/7 分布已经满足其通用路径。

因此采用以下最小接入：

- 将 `FLASHLocalForCausalLM` 注册为现有 hybrid MLA+KDA architecture；
- 继续使用内部 cache family 标识 `kimi_k3`，把它视为稳定的存储布局 ID，而不是模型身份；
- 复用 `KimiK3Recipe`、`CacheArena`、`HybridKDATokenToKVPool`、
  `HybridLinearAttnBackend` 和已有 scheduler cache-group contract；
- 不新增 `LiteRecipe`、cache family alias、allocator、metadata class 或 page table。

Lite 特有的 featurewise beta 只改变进入 KDA recurrence 前的数据准备，不改变 conv/recurrent state
形状或生命周期，因此留在阶段 3/4。CP8、KVP8、PD direct-to-live one-copy 和 NPU graph kernel
分别留在后续并行、传输和算子阶段，本阶段不提前实现。

## 3. Cache 组与字段

### 3.1 逻辑分组

`prefix_granularity` 固定为 128 token。28 层按 config 的 3:1 分布映射为四个 cache group：

| Group | 层 | Family | Retention | PD policy |
| --- | --- | --- | --- | --- |
| `full_attention` | 7 个 MLA 层 | history | full history | `full_suffix` |
| `linear_attention_0` | 前 7 个 KDA 层 | state | latest snapshot | `latest_snapshot` |
| `linear_attention_1` | 中间 7 个 KDA 层 | state | latest snapshot | `latest_snapshot` |
| `linear_attention_2` | 后 7 个 KDA 层 | state | latest snapshot | `latest_snapshot` |

三个 KDA group 是独立的 block-table 地址域；同组的 7 层共享 block ID，但每层绑定不同的 arena
plane。任何两个 layer 的可写 field 不能成为同一 tensor slice。

### 3.2 每层字段公式

MLA 每个 block 保存 128 个压缩 latent row：

```text
latent_width = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
latent_shape = [128, 1, 576]
latent_bytes = 128 * 1 * 576 * sizeof(BF16) = 147,456
```

KDA 的 rank-local 状态只由 linear-attention TP 切分；MLA projection 复制与 cache 形状无关：

```text
conv_dim = 2 * num_k_heads * head_k_dim
           + num_v_heads * head_v_dim
conv_shape = [conv_dim / kda_tp, conv_kernel_size - 1]
recurrent_shape = [num_v_heads / kda_tp, head_v_dim, head_k_dim]

conv_bytes = product(conv_shape) * sizeof(BF16)
recurrent_bytes = product(recurrent_shape) * sizeof(FP32)
state_bytes = conv_bytes + recurrent_bytes
```

目标 `num_k_heads=num_v_heads=32`、`head_k_dim=head_v_dim=128`、`conv_kernel_size=4`。精确结果为：

| KDA TP | Conv shape / bytes | Recurrent shape / bytes | State bytes/layer/snapshot |
| ---: | ---: | ---: | ---: |
| 8 | `[1536, 3]` / 9,216 | `[4, 128, 128]` / 262,144 | 271,360 |
| 1 | `[12288, 3]` / 73,728 | `[32, 128, 128]` / 2,097,152 | 2,170,880 |

N=1 只用于 layout、CPU/meta 或小子模块测试，不表示完整模型可在单卡加载。

## 4. 单 Arena Packing

每个 MLA 层提供一个 physical plane。KDA state 使用带 runtime stride 的 view 放入同一 plane，不为
三个 state group 额外分配 arena。最小 packing 为：

```text
mla_packing = ceil(state_bytes / latent_bytes)
state_packing = floor(mla_packing * latent_bytes / state_bytes), lower-bounded by 1
plane_bytes = mla_packing * latent_bytes
lcm_block_bytes = 7 * plane_bytes
```

| KDA TP | `full_attention` packing | 每个 state group packing | Plane bytes | LCM parent bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 1 | 294,912 | 2,064,384 |
| 1 | 15 | 1 | 2,211,840 | 15,482,880 |

arena 只分配一次：`arena_bytes = (num_lcm_blocks + 1) * lcm_block_bytes`。额外的一个 parent 是
保留的 null parent；逻辑 block 0 永不分配给 live request。每个 field view 的 dtype、offset、stride
和 shape 全部来自 `CacheMemoryPlan`，pool 不再次推导几何。

## 5. 容量与请求状态

MLA history capacity 和 KDA state admission 使用不同单位：

- MLA 按 token history 消耗 block；
- 每个 KDA group 为每个 live request 保留两个 snapshot block，分别作为当前输入和下一次输出；
- overlap depth 为 1 时，MLA 还保护一个 in-flight Decode span；
- KDA state 数量随 `max_live_requests` 变化，不随 prompt token 总量线性变化；
- recipe 在相同 parent budget 下对 token capacity 做单调反查，不能把最大 child-page 数直接当成
  可用 history token 数。

对 token capacity `C`、page token 数 `P=128`、最大 live request 数 `R` 和 Decode 输入宽度 `D`：

```text
protected_history_pages = R * ceil(overlap_depth * D / P)
mla_child_pages = ceil(C / P) + R - 1 + protected_history_pages
each_state_group_child_pages = 2 * R
parents = ceil(mla_child_pages / mla_packing)
          + sum(ceil(2 * R / state_packing), for three state groups)
```

容量测试必须分别断言 history token capacity 和 state supported-request capacity，不能用一个
“KV token 数”掩盖 state slot 上限。

## 6. 生命周期与清零

所有生命周期继续由通用 scheduler/coordinator 驱动：

1. fresh admission 为四个 group 分配 block；
2. data plane 在同一执行序列中先完成旧 owner 的 write-back，再清零新 block，再执行 load/transfer，
   最后启动 forward；
3. KDA backend 从每个 state group 的 block table 一次计算 `state_in`/`state_out`，每层按
   `state_group_by_layer` 选择对应地址；
4. abort 或 retract 只通过通用 cache lifecycle 释放或保存 block，不在模型侧维护 request-to-slot map；
5. block 被新请求复用前必须经 `HybridKDATokenToKVPool.zero_new_blocks()` 清零，避免旧 recurrent
   state、conv tail 或 null-page 污染；
6. late admission 与同轮释放遵守 write-back → zero → load → forward 顺序。

阶段 2 只验证 arena/pool 的清零、复用和地址契约。完整 scheduler 的 abort、retract、readmission 已由
通用 cache tests 覆盖；Lite 不复制这些状态机。

## 7. Decode Graph 元数据

现有 hybrid backend 为每个 graph batch size 和每个 KDA state group 预分配一对 int32 buffer：

```text
state_in_by_group[group_id][bs - 1]
state_out_by_group[group_id][bs - 1]
```

capture 时 buffer 绑定固定 storage 并填入 pad slot；replay 只原地更新 block ID，不替换 tensor、storage
或 Python 容器。MLA full-attention metadata 和 KDA state metadata 继续由同一个 hybrid wrapper 顺序
初始化，模型层不持有 graph metadata。

本阶段的 pointer-stability 测试使用真实 Lite cache contract 验证：

- 三个 group 的 storage 彼此独立；
- capture/replay 前后每个 buffer 的 `data_ptr` 不变；
- padding row 始终指向 pad/null slot；
- 新 forward operation 只刷新值，旧 metadata 不能跨 operation 使用。

真实 NPU graph capture 需要阶段 4 的 KDA NPU kernel 后才能成为数值门禁；本阶段只冻结 graph-safe
storage ABI，不能宣称 Lite Decode graph 已端到端跑通。

## 8. 实现范围

生产实现只修改 attention architecture registry，使 Lite 进入现有 hybrid MLA+KDA backend 和
`kimi_k3` cache family。测试新增一个 Lite focused 文件，直接复用阶段 1 的 config 语义和现有 cache
helpers；若通用 Kimi 测试已经覆盖某个状态机，不为 Lite 再写一套等价 scheduler fixture。

不新增依赖，不直接 import `torch_npu`，不修改 kernel package，不添加 CP/KVP/PD rank fragment，
不增加第二份 transfer state，也不启动完整模型服务。

## 9. 测试与准入

### 9.1 CPU/meta

- Lite architecture 生成 MLA + `LinearAttnConfig`，选择 hybrid backend 和 `kimi_k3` cache family；
- N=8/N=1 下 21/7 layer/group 分布、field dtype/shape/byte、packing 和 LCM parent byte 精确匹配；
- arena allocation byte 数与 `CacheMemoryPlan.arena_bytes` 相等；
- 21 层都绑定独立 conv/recurrent view，state group 恰好为 7/7/7；
- 新 block 清零、写入后释放/复用再清零，其他 block 和其他 layer 不受影响；
- graph capture/replay 的三组双索引 buffer 地址稳定且值按 group 独立刷新；
- 非整除 KDA 层数、缺失 MLA、非法 cache dtype 和错误 config 继续 fail closed。

### 9.2 NPU focused

在目标环境只使用一个 NPU 运行相同的 registry、recipe、arena bind、清零和 metadata pointer 测试；
不得加载完整 checkpoint 或启动完整服务。NPU0 是单测资源，不改变最终仅支持 8P8D 服务的约束。

### 9.3 准入门槛

focused tests 和完整 `pre-commit run --all-files` 全部通过；计划 bytes 与真实 allocation 完全一致，
所有 state tensor finite/zero-initialized，跨 layer/group 无可写别名，capture/replay 地址不变。验证记录
独立提交后才能进入阶段 3 的 Lite KDA eager baseline。
