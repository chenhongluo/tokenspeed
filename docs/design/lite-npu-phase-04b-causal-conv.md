# Lite NPU 阶段 4B：Causal-Conv

## 1. 目的

本阶段在阶段 4A 已固定的公开算子 artifact/ABI 上，为 Lite width-4 packed QKV causal-conv 建立统一
kernel 入口，并替换阶段 3 逐 token Python reference。目标同时覆盖 Prefill eager、Decode fixed-BS
NPU Graph、独立 `state_in/state_out` page 和 padding；不改变阶段 2 的 cache 几何或 PD 协议。

阶段 4B 分成三个独立提交：

1. 本文：冻结布局适配、kernel selection、状态发布和上板门槛；
2. 统一 kernel API、Ascend 实现、Lite 调用和 focused tests；
3. exact-source NPU 数值、graph、性能 A/B 和回退验证记录。

本阶段不接入 recurrent/chunk KDA，不改变 beta/gate 数学，也不启动完整模型服务。

## 2. 复用边界

### 2.1 保持不变的部分

- Lite 继续把 Q/K/V projection 拼成一次 `[T,C]` 输入，其中 TP8 的 `C=1536`；
- checkpoint 权重的逻辑布局保持 `[C,4]`；
- cache 中 conv state 保持 `[pages,C,3]`；
- `KdaAttnBackend` 继续从统一 metadata 取得独立的 `read_indices/write_indices`；
- Prefill 继续由 `_prepare_cache_prefill_state_inputs` 将输入 checkpoint 复制到工作输出页；
- page 0 保持只读逻辑零页，padding index 保持 `-1`；
- portable reference 继续作为 CPU、非 Ascend 和显式 debug oracle。

不得为了迁就一个算子，把整个 cache 改成 `[pages,3,C]`。该改动会同时扩散到 allocator、PD schema、
Kimi/GDN fallback 和既有 Triton ABI，而本阶段只需搬运 active rows 的 `B*C*3` 个元素。

### 2.2 新增的唯一公共入口

`tokenspeed_kernel.ops.attention.kda_causal_conv1d` 接收标准布局：

```text
projected       [T,C]             BF16/FP16
weight          [C,W]             BF16/FP16，W=4
conv_state      [pages,C,W-1]     与 projected 同 dtype
read_indices    [B]               int32/int64
write_indices   [B]               int32/int64
cu_seqlens      [B+1]             device int32/int64
cu_seqlens_cpu  [B+1] or None     host int64 Prefill 边界
has_initial     [B] or None       Prefill 初态掩码
bias            [C] or None
decode          bool
```

返回 `[T,C]`，并只向 `write_indices` 发布最后 `W-1` 个原始 projection。runtime 不导入
`torch_npu`、公开 artifact loader 或 vendor op；这些内容只存在于 kernel package 的 Ascend 注册中。

## 3. 公开算子 ABI 审计

阶段 4A 固定的 `causal_conv1d` schema 使用：

```text
x             [T,C] or [B,S,C]
weight        [W,C]
conv_state    [slots,W-1,C]
cache_indices [B]
run_mode      0=Prefill, 1=Decode
```

它只有一个 `cache_indices`：同一 slot 既是输入又是输出；`initial_state_mode` 仅在 Prefill 有效；Decode
始终从该 slot 读取已有 history。公开 kernel 支持 width 2--4、BF16/FP16、2D varlen Prefill、2D/3D
Decode、SiLU、pad sentinel 和可选 bias。

这与 TokenSpeed 有两个不能忽略的差异：

1. state 的最后两维相反；
2. TokenSpeed 允许 `read_page != write_page`，公开 ABI 不允许。

把 `[pages,C,3]` 直接转置成非连续 `[pages,3,C]` view 虽然不会报错，但公开 op 的 contiguous
materialization 不会把 mutation 写回原 pool。这是静默 state 丢失，禁止作为生产路径。

## 4. State 适配

公开 Prefill 的 large-batch 路径只搬运 active batch rows：

```text
active       = (read >= 0) & (write >= 0)
safe_read    = active ? read  : 0
safe_write   = active ? write : 0
compact      = state[safe_read].transpose(1,2).contiguous()   # [B,3,C]
local_index  = active ? arange(B) : -1
output       = public_op(..., compact, cache_indices=local_index)
state.index_copy_(0, safe_write, compact.transpose(1,2))
```

该 compact buffer 同时解决布局和双页语义。padding row 从 page 0 读取，公开 op 因 `-1` 跳过，回写值
仍等于原 page 0；即使多个 padding row 映射到 page 0，也不得改变其内容。公开 binding 的输出必须
zero-init，保证 varlen 尾部和跳过行不会把未初始化值送入后续 split/gate。

Prefill 现有准备逻辑通常已令 `read==write==state_out`，但公共 API 仍保留双索引，避免在调用点制造
第二套状态规则。Decode 的 Torch 路径直接 gather `read`、计算、再 scatter `write`，不需要 compact
布局转换。

## 5. 数值实现

### 5.1 Decode graph-safe Torch 路径

Decode 每个 request 只有一个 token：

```text
history = state[safe_read]                         # [B,C,3]
window  = cat(history, projected[...,None])        # [B,C,4]
value   = sum(fp32(window) * fp32(weight), dim=-1)
output  = silu(value + bias).to(projected.dtype)
next    = window[...,1:]
state.index_copy_(0, safe_write, next)
```

padding row 的 `next` 强制保持原 page-0 history，output 强制为零。整条路径只使用 tensor op，没有
`.item()`、device-to-host copy 或 Python token 循环，可被 fixed-BS NPU Graph 捕获。

### 5.2 Prefill small-batch Torch 路径

Prefill 从 scheduler 已有的 host `cu_seqlens_cpu` 遍历 request，而不是读取 device boundary。每个
非空 request 将 `[C,3]` history 与 `[C,L]` 输入拼接，然后调用一次标准 depthwise `conv1d`：

```text
signal = cat(history, projected_segment.T, dim=-1)
value  = conv1d(signal[None], weight[:,None,:], groups=C)
output = silu(value).T
next   = signal[:,-3:]
```

这是 eager Prefill，不要求 graph；循环次数等于 request 数，不再按 token 循环。fresh request 在
conv 前用 `has_initial=false` 将 history 置零。

### 5.3 Prefill large-batch 公开路径

公开 op 一次消费 packed varlen batch，并执行 depthwise width-4 convolution、SiLU 与 compact state
更新。标准 `[C,4]` 权重只在该分支转为 contiguous `[4,C]`；不为此引入第二种 checkpoint 参数或
platform-specific model 字段。

Lite 当前每次 forward 都重新 `cat` 三个 conv weight。阶段 4B 复用模型 loader 已有的
`process_weights_after_loading` hook，把标准 `[C,4]` packed weight 缓存一次；未经过 loader 的小型
单测仍允许按原表达式构造 fallback。该缓存是公共 causal-conv 输入，不包含 NPU 类型或 vendor 状态。

## 6. Kernel selection

Ascend 注册两个实现：

| 模式 | request 数 | 默认实现 | 原因 |
| --- | ---: | --- | --- |
| Decode | 任意 fixed BS | graph-safe Torch | 当前 cache 布局下全范围更稳定，BS1/2/8/256 更快 |
| Prefill | `<16` | depthwise `conv1d` | 避免公开 op 的固定 host/tiling 开销 |
| Prefill | `>=16` | 公开 fused op | 多 request 时一次 varlen 调用明显减少 launch 数 |

selection trait 固定为 `forward_mode={prefill,decode}` 和
`batch_class={small,large}`，large 定义为 `B>=16`。公开 artifact 不可用或 schema 校验失败时，large
Prefill 必须自动回退同一个 Torch 实现；不能让可选 artifact 成为功能依赖。显式 kernel override
仍需 fail loud，便于测试指定实现。

阈值只依据阶段 4B 同口径 NPU0 A/B，不推导为其他 SoC 的通用结论。若后续完整服务 profile 显示
交叉点改变，应改 registration/selection 测试和本文，而不是在模型 forward 中散落条件分支。

## 7. 上板证据与决策

在 910B、TP8 rank-local `C=1536,W=4` 上得到：

### 7.1 正确性

- compact 公开 Prefill/Decode 对 FP32 oracle 的 output/state 最大绝对误差均为 0；
- `read!=write`、fresh history、empty segment、padding 和 page-0 isolation 通过；
- 非连续 state view 调用不报错但不修改源 pool，证明必须 compact/scatter；
- 公开 op eager 接受无 bias；Prefill 生产路径无需为 graph 创建零 bias 参数。

### 7.2 Decode NPU Graph（微秒/次）

| BS | Torch | 公开 op + compact | 决策 |
| ---: | ---: | ---: | --- |
| 1 | 55--59 | 76--81 | Torch |
| 2 | 77 | 93--95 | Torch |
| 8 | 101--106 | 110--115 | Torch |
| 64 | 166--176 | 152--159 | 不为单一交叉点增加生产分支 |
| 256 | 270--274 | 282--295 | Torch |

公开 op eager 本体约 1.21--1.30 ms，compact 端到端约 1.35--1.54 ms；Torch eager 为
0.16--0.27 ms。graph 后公开 op 只在 BS64 的孤立点领先，不能抵消双实现、布局搬运和边界测试成本。

### 7.3 Prefill eager（微秒/次）

| requests | tokens | depthwise `conv1d` | 公开 op + compact |
| ---: | ---: | ---: | ---: |
| 1 | 64 | 196 | 1417 |
| 4 | 250 | 732 | 1435 |
| 8 | 500 | 1449 | 1432 |
| 16 | 1000 | 2900 | 1441 |

单 request 下 token 数从 64 增至 1024 时，Torch 路径仅从约 0.17 ms 增至 0.21 ms，公开 op 保持约
1.35--1.44 ms。阶段 3 的逐 token reference 在 T64 约 40--41 ms。因此阶段 4B 默认阈值取保守的
16，而不是在误差范围内的 8。

## 8. 测试矩阵

### 8.1 CPU/meta

- 统一 API 对 4-token、`[1,3]`、`[2,0,2]`、fresh/resume、`read!=write` 与 padding 对齐独立 oracle；
- 错误 width、channel、dtype、boundary、index shape 和重复 active write fail loud；
- 非 Ascend 或无注册 kernel 时保持阶段 3 portable reference；
- Lite post-load 只缓存一次标准 packed weight，参数名和 strict loader coverage 不变；
- selection 测试冻结 Decode=Torch、Prefill B15=Torch、B16=public，以及 artifact-missing fallback。

### 8.2 NPU0

- Prefill B1/B4/B16，包含变长、empty、fresh/resume、双页与 page-0 sentinel；
- Decode BS1/2/8/64/256 eager 与 NPU Graph 两次 replay；第二次更新 projected/index 内容，证明 graph
  不冻结 value；
- output、目标 state 与 FP32 oracle 在 BF16 阈值内，未选 state/page0/neighbor bitwise 不变；
- 所有 output/state finite；
- 强制 public/Torch override 各自运行，默认 selection 与阈值一致；
- artifact 缺失时同一 focused case 自动回退且结果不变。

性能 A/B 只比较同步后的 steady-state；graph 单独报告 replay latency，不拿 eager host/tiling 开销冒充
设备 kernel 时间。

## 9. 准入与回退

实现提交准入要求：本地 focused、全量 `pre-commit run --all-files`、NPU0 exact-source focused 全部
通过；Prefill 大小 batch 和 Decode graph selection 命中本文；没有 NaN/Inf、page0 mutation 或
neighbor write；公开 op 缺失仍能运行。

若公开 large-Prefill 任一 correctness/availability 门禁失败，只删除其 performant registration，保留
统一 API 和 Torch 路径；不得回退到阶段 3 逐 token NPU reference。若后续完整服务 profile 证明
large-Prefill 公开 op 无收益，再按同样方式移除 registration，不改 runtime 或 cache。

阶段 4B 明确不包含：

- Decode conv+gate+recurrent 的新融合 kernel；
- Prefill gate-cumsum/chunk KDA；
- output epilogue 融合；
- speculative verify/replay；
- Decode overlap、CP8/KVP8、PD one-copy 或 8P8D 服务。
