# Lite NPU 阶段 4D：Prefill Gate 与 Chunk KDA

## 1. 目的

本阶段在阶段 4A 固定的公开算子 artifact 上，为 Lite featurewise-beta KDA Prefill 接入
`KdaGateCumsum + ChunkKdaFwd`。目标是替换阶段 3 按 token 扫描的 Torch reference，同时保持统一
`kda_paged_prefill` API、K-major FP32 recurrent state、变长 packed batch 和空请求语义。

阶段 4D 分成三个独立提交：

1. 本文：冻结 artifact 加载根因、Lite 数学适配、空请求 compact、kernel selection 与上板门槛；
2. loader/manifest、Ascend Prefill adapter、registration 和 focused tests；
3. exact-source NPU 数值、变长、性能和回退验证记录。

本阶段不改 runtime/backend/cache/PD，不接 Decode graph，不开发或修改上游 AscendC 源码，也不启动
完整模型服务。

## 2. 复用边界

统一入口继续是 `tokenspeed_kernel.ops.attention.kda_paged_prefill`：

```text
q/k/g_raw       [1,T,H,K]       BF16/FP16
v               [1,T,H,V]       BF16/FP16
beta_logits     [1,T,H,K]       Lite featurewise beta
A_log           [H]             FP32
dt_bias         [H,K]           FP32
initial_state   [B,H,K,V]       FP32, K-major
cu_seqlens      [B+1]           device int32/int64
cu_seqlens_cpu  [B+1]           host int64
```

返回 `KdaPrefillResult(out=[1,T,H,V], final_state=[B,H,K,V])`。scheduler 已提供 host
`cu_seqlens_cpu`，因此 chunk plan 必须从 host copy 构造；禁止每层从 device boundary 做 D2H。

阶段 4D 只在 `tokenspeed-kernel-npu` 增加 Ascend 实现和 artifact 加载能力。vendor-neutral kernel facade、
runtime backend、state page 生命周期和 checkpoint 参数保持不变。

## 3. Lite 数学适配

Lite 的 beta 是每个 key channel 一个值，不能直接传给只接受 head-scalar beta 的公开 core。等价变换为：

```text
s         = sqrt(sigmoid(beta_logits) + 1e-10)
q_public  = L2Norm(q)
k_public  = L2Norm(k) * s
v_public  = v * s
beta_core = ones([1,T,H])
gate      = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
```

`KdaGateCumsum` 对每个 sequence 的每个 64-token chunk 计算 `gate * log2(e)` 的 chunk-local prefix sum；
`ChunkKdaFwd` 消费 prepared Q/K/V、unit beta、gate prefix 和 K-major FP32 initial state，执行 delta-rule
chunk scan并返回 output/final state。公开 core 的 `scale` 固定为 `K**-0.5`。

Q/K/V 在公开 ABI 边界量化回输入 dtype；因此正确性分别对两种 oracle 报告：

- 对同样量化后的独立逐 token oracle，验证公开 core 本身；
- 对阶段 3 FP32 prepare reference，验证生产可接受误差和连续 state 稳定性。

本阶段 registration 只声明 `beta_mode=featurewise`。scalar-beta Kimi 继续走现有 reference，避免为 Lite
顺带扩张另一个模型的数值边界。

## 4. Artifact tiling library 根因与修复

### 4.1 根因

阶段 4A loader 只在 `torch_npu` 已初始化后设置 `ASCEND_CUSTOM_OPP_PATH`，随后加载 Torch binding 和
`libcust_opapi.so`。这足以注册 schema 和 op-api，却不会让 CANN 重新扫描自定义 host-tiling library。

实际进程 maps 只包含系统 built-in/flash OPP 的 `liboptiling.so`，不包含 artifact 的
`custom_transformer/liboptiling.so`。结果是：

- `KdaGateCumsum` 找不到自定义 tiling registration；
- CANN 错误回退到通用 AutoTiling JSON parser；
- 公开 kernel JSON 的 `compileInfo={}` 不含 `_pattern`，调用在任何数学计算前失败。

在启动 Python 前把 artifact vendor 加入 `ASCEND_CUSTOM_OPP_PATH`，原始未修改 artifact 立即通过；在
`torch_npu` 已导入后，以 `RTLD_GLOBAL` 显式加载 artifact `liboptiling.so` 也得到相同结果。因此根因不是
shape、Gate/Chunk 算法、JSON 或上游 C++ tiling 实现。

### 4.2 最小生产修复

不维护上游源码 patch。artifact contract 增加一个经过 SHA-256 校验的 `vendor_op_tiling` 路径；loader
按以下顺序执行：

1. 校验 source/CATLASS revision、binding、op-api 和 op-tiling 路径及 digest；
2. 将 vendor 置于 `ASCEND_CUSTOM_OPP_PATH` 首位；
3. 用标准库 `ctypes.CDLL(..., RTLD_GLOBAL)` 加载 op-tiling library，并保留 handle；
4. 加载 TokenSpeed Torch binding；
5. 核对四个 schema。

所有路径与 digest 必须在第一次 `dlopen` 前完成校验。tiling 或 binding 任一加载失败时恢复原环境并返回
unavailable；已 `RTLD_GLOBAL` 加载的 tiling library 无法从进程安全卸载，因此保留 handle，但不得继续选择
public kernel。manifest schema 随新增的可执行文件校验升级，旧 artifact 必须重建，不能静默执行未校验
tiling library。

## 5. 空请求 compact 与 host chunk plan

公开 Gate/Chunk 接受 varlen boundary，但空 segment 会静默错位 final state。NPU0 的 `[32,0,32]` 直接
调用虽然 finite，state rel-L2 达 `1.01`；compact 后为 `0.00155`。因此生产 adapter 必须先移除空请求：

```text
lengths       = diff(cu_seqlens_cpu)
keep          = indices(length > 0)
compact_cu    = prefix_sum(lengths[keep])
compact_state = initial_state[keep]
chunk_indices = [(sequence, chunk) for each compact real chunk]
```

空 segment 不占 packed token，所以 Q/K/V/gate 无需重排。公开 op 返回后：

```text
final_state = initial_state.clone()
final_state[keep] = compact_final_state
```

从而空请求 state bitwise 不变。若全部 segment 为空，直接返回零 output 和 initial state clone，不调用要求
至少一个 chunk 的公开 op。`compact_cu` 和 `chunk_indices` 都由已有 host boundary 用 Python tuple 构造，
不增加 device-to-host 同步，也不复用会读取 device tensor 的 Triton `prepare_chunk_indices`。

## 6. Kernel selection 与回退

Ascend 增加一个 specialized registration：

```text
solution=public_kda
beta_mode=featurewise
recurrent_layout=k_major
priority=SPECIALIZED
```

NPU0 完整 adapter A/B（含 featurewise prepare、Gate、Chunk 和 final state）如下：

| tokens | 公开链 | 阶段 3 reference | 加速比 |
| ---: | ---: | ---: | ---: |
| 4 | `0.802 ms` | `1.261 ms` | `1.57x` |
| 64 | `0.851 ms` | `5.708 ms` | `6.71x` |
| 256 | `0.886 ms` | `19.016 ms` | `21.47x` |
| 1024 | `1.487 ms` | `73.798 ms` | `49.62x` |

公开链在最小 T4 已领先，因此不增加 token 阈值分支。artifact 缺失、校验失败、tiling/binding 加载失败或
public-specific shape 不满足时，默认选择回退阶段 3 reference；显式 `solution=public_kda` 或 exact
override 必须 fail loud。

facade 只在选中的 kernel spec 属于 `public_kda` 时传递 `require_public`，不向 CUDA/ROCm/其它 Ascend
实现扩散 vendor-specific 参数。

## 7. 已有上板证据

原始 exact artifact 在 910B、Lite TP8 rank-local `H=4,K=V=128` 上得到：

| 场景 | output rel-L2 | state rel-L2 | finite |
| --- | ---: | ---: | --- |
| T4 | `0.00407` | `0.00145` | 是 |
| T32 | `0.00390` | `0.00156` | 是 |
| T63 | `0.00415` | `0.00161` | 是 |
| T64 | `0.00412` | `0.00158` | 是 |
| `[1,63]` | `0.00394` | `0.00137` | 是 |
| T65 | `0.00411` | `0.00159` | 是 |
| T127 | `0.00403` | `0.00158` | 是 |
| `[65,127]` | `0.00412` | `0.00163` | 是 |
| `[32,0,32]` compact | `0.00406` | `0.00155` | 是 |

最大 output 绝对误差不超过 `2.44e-4`；非空场景最大 state 绝对误差不超过 `1.37e-3`。短序列、非整
chunk、跨 chunk、变长、多请求和 empty compact 均通过。

## 8. 测试矩阵

### 8.1 CPU/meta

- manifest 缺 tiling path、越界 path、digest 错误均在任何环境修改或 `dlopen` 前失败；
- loader 调用顺序冻结为 tiling→binding，成功只加载一次，失败恢复原 `ASCEND_CUSTOM_OPP_PATH`；
- host chunk plan 覆盖 T0/T1/T63/T64/T65、`[1,63]`、`[32,0,32]` 和全部为空；
- mock public op 验证 featurewise prepare、unit beta、chunk64、K-major state、compact/scatter；
- public artifact unavailable 时默认回退 reference，显式 public fail loud；
- scalar beta 不命中 public registration；现有 Kimi/Lite/selection tests 全部回归。

### 8.2 NPU0

- 使用原始 pinned artifact，不应用 C++/JSON patch；不预设启动前 `ASCEND_CUSTOM_OPP_PATH`；
- loader 动态加载后 T4/32/63/64/65/127/256/1024；
- `[1,63]`、`[65,127]`、`[32,0,32]` 和全部空 segment；
- output/final state 对量化 oracle 与 FP32 production reference，所有张量 finite；
- 空 state、未选 state 和 input state 不被修改；
- artifact 缺失、tiling digest 损坏及显式 public 的失败路径；
- T4/64/256/1024 同步 steady-state A/B，性能只记录，不设目标。

## 9. 准入与回退

实现提交准入要求：本地 focused、全量 `pre-commit run --all-files`、NPU0 exact-source focused 全部
通过；原始 artifact 在不依赖启动脚本 export 的进程内完成 tiling 注册；empty segment state 不变；没有
NaN/Inf 或 neighbor write；默认 unavailable 回退与显式 fail-loud 同时成立。

若公开链任一生产 shape、compact 或 loader 门禁失败，只删除 specialized registration，保留 loader
完整性修复和阶段 3 reference；不得引入私有 CANN patch，也不得把 artifact 安装变成完整服务的硬依赖。

阶段 4D 明确不包含：

- 修改上游 Gate/LayoutSwap/Chunk AscendC 或 compile-info JSON；
- Prefill graph、CP8、PD state transfer 或 8P8D 服务；
- Decode recurrent、causal-conv 或 output epilogue；
- KDA projection/conv/beta prepare 的新融合 kernel。
