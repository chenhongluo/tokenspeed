# Lite NPU 适配总体计划

## 1. 目标

在不破坏 TokenSpeed 现有模块边界的前提下，为 Lite 文本模型增加完整的昇腾 NPU 支持。最终实现需要满足：

- 能加载 Lite checkpoint，并严格还原 KDA:MLA = 3:1 的层分布；
- 支持 featurewise-beta KDA、带 output gate 的 NoPE MLA、EveryLayer Grouped MoE、
  shared/identity-zero expert 和 n-gram embedding（OE）；
- 唯一完整服务形态为 8 个 Prefill rank + 8 个 Decode rank，共 16 rank，不使用 PP；
- Prefill 使用 eager，Decode 使用固定 batch 的 NPU graph，并启用 Decode overlap；
- 首轮先用 EP8 和已有权重切分完成静态权重装载、有限上下文 Eager 服务和精度闭环；
- CP/KVP 分布式 KV Cache、P/D 分片传输和 KDA direct-to-live 统一放到最后 TODO，
  不作为首轮服务和融合算子验收的前置条件；
- 补齐参考实现中已经准入的全部优化，并新增 Lite KDA 前置 projection 合并、
  packed QKV causal convolution 和 featurewise-beta 融合；
- 对 Grouped MoE 执行三组受控实验，分别隔离四组 executor 合并收益和 group-to-rank 物理放置收益，
  目标候选为一个 flattened EP8 executor 加 group-major 的“两 rank 一组”布局；
- TokenSpeed runtime 保持厂商无关：所有直接的 `torch_npu` 和第三方 kernel 调用都放在
  `tokenspeed-kernel` / `tokenspeed-kernel-npu` 边界后面；
- 每个非平凡阶段都留下可运行的单元测试、算子上板测试、服务测试、精度检查、内存检查和性能检查。

本文只是实施计划，不代表已经授权实施、提交、推送、部署或修改外部服务。

## 2. 当前基础与复用边界

TokenSpeed 的 [PR #1281](https://github.com/lightseekorg/tokenspeed/pull/1281)
已经为 Qwen 建立了通用 NPU 基座，包括：

- device-neutral runtime；
- HCCL 通信后端；
- NPU graph 基础设施；
- paged MHA；
- RMSNorm 和 fused-add-RMSNorm；
- RoPE；
- embedding dispatch；
- NPU package 安装和测试框架。

当前仓库还已经具备较成熟的 Kimi K3 路径，包含大部分与具体模型无关的 KDA/MLA 能力：

- `KimiLinearKDA` 和单 GEMM 的 KDA 输入 projection；
- packed Q/K/V causal-conv 权重布局；
- hybrid MLA/KDA backend 路由；
- paged BF16 conv state 和 FP32 recurrent state；
- graph-stable recurrent-state metadata；
- paged KDA Decode/Prefill 以及 verify/replay API；
- NoPE MLA 和 output gate；
- MLA latent page 与 KDA state group 共用的 cache recipe；
- 通用 cache transfer contract 和 rank fragment planner。

Lite 应在这些接口上做最小增量，不应复制第二套 hybrid attention 框架。

### 2.1 单卡能力边界

“单卡”只用于算子、单层/子模块和 CPU/meta-device 构造验证，不能用于拉起完整 Lite 服务。
按目标 checkpoint 的实测账本：

| 项目 | GiB |
| --- | ---: |
| 未切分 checkpoint 静态权重 | 129.113 |
| 放到 host 的 OE 主表 | -27.000 |
| 单卡仍需驻留的静态权重 | 102.113 |
| 910B 扣除服务启动前占用后的可用 HBM | 60.649 |
| 静态权重缺口（尚未计 cache/runtime） | 41.464 |

因此，完整模型不设计单卡或四卡过渡服务，直接使用 16 rank 的 8P8D；`N=1` 仅允许加载被测
shard/子模块或使用小尺寸合成 tensor。目标 checkpoint 的完整加载账本已给出更严格的
EP8 后单 rank 口径：

| 角色 | Parameter storage | Post-load derived | 逻辑 resident static | 910B 可用 HBM 中的静态余量 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 17.234249 GiB | 0.054928 GiB | **17.289177 GiB** | 约 43.360 GiB |
| Decode | 13.440303 GiB | 0.054928 GiB | **13.495231 GiB** | 约 47.154 GiB |

其中 routed expert 由 EP8 切分，MLA 权重仍是每 rank 复制，OE 主表仍完整留在 host。因此
**EP8 已足以解决静态权重能否装入 64 GiB 910B 的问题**，并可为有限上下文 Eager/smoke
留出充足空间。该结论不包含 2M 长上下文或高并发容量；这两项仍需要最后 TODO 中的
分布式 KV Cache。

| 能力 | TokenSpeed 当前状态 | Lite 需要补齐 |
| --- | --- | --- |
| NPU device、HCCL、graph、MHA、Norm、Embedding | PR #1281 已支持 | 复用并扩展测试 |
| Hybrid KDA/MLA 模型和 state cache | Kimi K3 已支持 | 新增 Lite architecture/config 和数学差异 |
| KDA beta | 每个 head 一个标量 | 两层 learnable projection 和逐 channel beta 缩放 |
| NoPE MLA + output gate | 已有模型路径 | 补充 Ascend MLA kernel 和目标 shape |
| Grouped MoE | 已有 grouped top-k 和 zero-expert 映射基础 | 四组路由、执行和 NPU kernel |
| OE / n-gram embedding | Qwen4-Exp 有相近工程实现 | Lite hash、排除规则、归一化和 host residency |
| recurrent state Decode graph | 已有通用 capture/replay | Lite NPU kernel graph-safe 化和 BS1/BS2 验证 |
| P/D cache transfer | 已有通用 contract | 最后 TODO：CP/KVP 分片和 KDA direct-to-live |
| P 侧 CP、D 侧 KVP | MLA CachePD 尚未支持 | 最后 TODO：page ownership、partial attention、LSE merge 和 transfer mapping |

## 3. 模型契约

模型实现必须以 checkpoint/config 为维度和层分布的唯一真值来源。第一个实现提交需要显式校验所有影响模型数学的字段，不能依赖静默默认值。

### 3.1 KDA

复用 Kimi KDA 的 state equation 和 cache 布局，但需要实现以下 Lite 差异：

- `b_a: hidden -> beta_rank` 和 `b_b: beta_rank -> heads * head_dim`
  是两层可学习、无 bias 的 projection；
- `beta_scale = sqrt(sigmoid(beta_logits) + epsilon)`，并且 beta 是 featurewise；
- Q 和 K 先做 L2Norm；
- K 和 V 再乘 `beta_scale`；
- 完成上述变换后，标准 recurrent KDA kernel 的 beta 输入使用全 1；
- forget gate 保持 per-channel，并保留配置中的 lower bound；
- output gate 为 full rank，在 per-head RMSNorm 后应用；
- Q/K/V 使用 width=4 的 depthwise causal convolution 和 SiLU；
- recurrent state 保持 FP32，conv state 保持 BF16。

归一化顺序是必须固定的数学契约：

`normalize(K) * beta_scale` 不等于 `normalize(K * beta_scale)`。

只有在 kernel ABI 能保持这一顺序时，才能直接复用现有 Kimi kernel。

### 3.2 MLA

复用 Kimi 的 NoPE MLA 和 output-gate 模型路径。不能因为环境里存在 MLA HW prolog 就直接启用：
必须先用 Lite 的真实 hidden/latent shape 做算子上板能力测试。shape 不支持时继续使用已验证的 fallback。

### 3.3 Grouped MoE

每个物理层都包含一个 Grouped MoE，必须保持：

- 每个 expert group 独立路由；
- global expert id、group id、EP-local expert id 的精确映射；
- zero expert 的 identity 语义；
- shared expert contribution 只累加一次；
- 某个 EP rank 没有收到 token 时仍能正确完成；
- Prefill、eager Decode 和 graph Decode 的数学完全一致。

目标参数为 `G=4`、每组 `E_g=384` 个 real experts、每组独立 `TopK=12`，因此共有
`E=G*E_g=1536` 个 real experts。需要把两个概念严格分开：

- 路由语义：每个 token 在四组内分别做 TopK，保证每组都有固定数量的候选 route；
- 物理放置：把逻辑 expert 映射到 EP rank，决定上述组级约束能否转化为设备负载约束。

最终候选使用一个 flattened EP8 executor，不给 routed expert 叠加 TP。保持 group-major global ID 时，
每 rank 持有 `1536/8=192` 个 real experts，每个 `384`-expert group 恰好映射到两个连续 rank：
`g0 -> r0/r1`、`g1 -> r2/r3`、`g2 -> r4/r5`、`g3 -> r6/r7`。这不会改变每 rank 的 routed-expert
静态权重规模。identity-zero expert 不进入 GMM，shared expert 继续独立计算。

该布局是待上板验证的目标候选，不靠推导直接准入。阶段 6 必须用三组实验分别测量 executor 合并与
物理放置，且记录 pair 级和 rank 级真实 route 数；每组固定 TopK 只能约束 rank pair 的总候选 route，
zero-expert 命中和 pair 内 hot expert 仍可能造成实际 GMM 负载偏斜。

通用 `MoELayer` 当前不支持同一层同时使用 TP 和 EP。目标方案中 routed expert 只使用 EP，
不需要为此新增 mixed-TP/EP 抽象。

### 3.4 OE

可以复用 Qwen4-Exp 在历史 token 读取、graph-safe metadata、分片表加载和紧凑存储方面的工程模式，
但需要独立实现 Lite 的：

- `ngram_embedding` checkpoint key；
- `ngram_vocab_size_ratio`、`emb_neighbor_num`、`emb_split_num`；
- special token 排除和作用域；
- 固定归一化系数；
- block projection / aggregation；
- 全量 embedding table 放在 host，仅将当前 batch 需要的行或 block staging 到 device。

Lite OE 与 Qwen4-Exp PLE 的数学不同，只复用存储、加载和历史管理模式。

## 4. 首轮并行与本地 Cache 设计

每个角色固定使用 `N=8` 个 rank，共 16 rank，并保持 `PP=1`、`DP=1`。本计划不实现或验收
4P4D、单角色 4 rank 或完整模型单卡服务。

| 角色 | MLA | KDA | MoE | Dense/shared 权重 |
| --- | --- | --- | --- | --- |
| Prefill | 权重每 rank 复制；eager 使用有限的本地/replicated history | linear-attention TP8 | EP8；阶段 6 选择 flattened placement | TP1 |
| Decode | 权重每 rank 复制；eager 使用有限的本地/replicated history | linear-attention TP8 | EP8；与 P 使用同一 expert placement | TP8 |

首轮只使用上表中的权重/计算切分：EP8 负责将 1536 个 routed expert 均分为
192 个/rank，KDA 使用已有 linear-attention TP8，P/D 继续使用已验证的 dense/shared 映射。
MLA 的有限上下文页和 KDA request state 先使用本地、有界的 correctness 布局，不在首轮
实现跨 rank KV page ownership、partial output/LSE merge 或 P/D cache fragment 转换。

`Mapping` 虽已能表达 attention TP/CP/DP、linear-attention TP 和 MoE EP，但这不等于 CachePD
已经支持 CP/KVP。相关工作不删除，统一移到文末“最后 TODO：分布式 KV Cache”，
并与首轮功能/精度验收解耦。

## 5. NPU Kernel 策略

每个新增 runtime 操作都需要：

1. 在 `tokenspeed-kernel` 中提供通用 API；
2. 在 `tokenspeed-kernel-npu` 中注册 Ascend solution；
3. 保留 Torch/reference solution 作为正确性 fallback。

公开第三方源码只能在完成 license/provenance 审计后引入，并补齐所需 notices。
正常安装 TokenSpeed NPU extra 后应能直接使用，不能让 vLLM 或 vLLM-Ascend 成为 runtime 依赖。

缺少融合算子时，先使用若干 PyTorch/NPU 小算子组成 baseline；融合实现与 baseline 使用同一 API，
最终由 kernel selection 替换，不能要求模型路径重写。

| 算子族 | Baseline | 最终 NPU 路径 |
| --- | --- | --- |
| Packed QKV causal conv | Torch depthwise conv + SiLU + 显式 state update | 单次 graph-safe packed QKV width-4 kernel |
| KDA Decode | Torch/reference recurrence | 适配公开 recurrent KDA，再融合 Lite beta |
| KDA Prefill | Torch/reference scan | gate preprocessing + 公开 chunk KDA |
| Featurewise beta | 标准 KDA 前的 fused pointwise prepare | Decode 融入 recurrent kernel；Prefill 保留一个 prepare launch |
| MLA | Torch/reference projection 和 attention | Ascend latent Prefill/Decode、LSE、value projection、output gate |
| MoE routing | Torch softmax/top-k/remap | 四组独立 Ascend fused top-k + global ID 映射 |
| MoE expert core | 四个独立 EP8 executor 的 reference baseline | 一个 flattened EP8 executor，复用 NPU GMM/collective |
| OE | Torch id、lookup、aggregate | block projection + host staged lookup |
| Norm/residual | 现有非融合路径 | 复用 PR #1281 fused-add-RMSNorm |

## 6. 实施阶段与提交边界

每个阶段必须可以独立 review。阶段实施前先新增或更新对应设计文档；需要上板的阶段还要单独保存结果。
建议的提交结构为：

1. `docs: define <stage> contract`
2. `feat/fix: implement <stage> with tests`
3. `docs: record <stage> NPU validation`

每个 commit 都需要 signed-off。提交前必须执行完整的：

`pre-commit run --all-files`

不能用局部 lint 替代。

### 阶段 0：冻结基线和公开算子清单

交付：

- 记录 TokenSpeed base SHA 和 PR #1281 的能力边界；
- 冻结 Lite config/key manifest，但 repo 中不记录私有路径；
- 在本地 artifact 中冻结参考请求、token id、关键 hidden、logits 和最终输出；
- 记录每个公开 NPU kernel 候选的 license 和源码 revision；
- 分别定义 BF16 projection、FP32 recurrent state、routing weight、logits 和 token 的精度阈值。

准入门槛：不存在未解释的 checkpoint key、config 字段或 kernel license。

### 阶段 1：Config、Architecture 和 Checkpoint Loader

交付：

- 新增 Lite config 和 model registry entry；
- 新增 text-only Lite model/decoder；
- 数学相同的部分直接复用 Kimi MLA/KDA component；
- 实现 3:1 layer selection 和 EveryLayer Grouped MoE；
- 先用未优化的独立参数完成权重加载，包括 OE 和 beta；
- 校验 tensor shape 和 loaded-key 完整性。

测试：

- config round-trip 和必填字段拒绝；
- layer pattern、KDA/MLA/MoE module 数量；
- checkpoint name mapping、TP/EP shard 范围和 missing-shard failure；
- N=1/8 的 CPU/meta-device 构造；其中 N=1 不加载完整实权重。

准入门槛：每个参数只加载一次，无 unexpected key，也不存在未加载而保持全零的 shard。

### 阶段 2：Hybrid Cache Recipe 和 Graph-stable Metadata

交付：

- 泛化并复用 `KimiK3Recipe`，不复制 Lite 专用 recipe；
- 声明 MLA latent page、BF16 conv state 和 FP32 recurrent state；
- 配置层分布允许时保留三个 KDA state group；
- 为每个 KDA group 建立固定地址的 Decode graph metadata；
- KDA state slot 按最大 live request 数分配，与普通 token KV capacity 分开计算。

本阶段只定义单 rank/有界请求的 tensor 几何、本地 page/state 生命周期和 graph-stable
metadata，不实现跨 rank MLA KV page 存储、CP/KVP owner 或 PD fragment transfer。

测试：

- N=1/8 下 byte-exact field shape、packing、capacity、block zeroing、page reuse 和 pointer stability；
- fresh request、abort、retract、slot reuse、late admission。

准入门槛：cache sizing 与真实 tensor allocation 一致，不同 layer 不会意外共享可写 state plane。

### 阶段 3：Lite KDA Eager Baseline

交付：

- 实现 featurewise-beta projection 和精确的 normalize/scale 顺序；
- 使用未融合的 packed-QKV convolution reference；
- 增加 Decode recurrent 和 Prefill scan reference；
- 通过已有 hybrid backend 原地更新 conv/recurrent state。

测试：

- 先跑 4-token 算子测试，再跑 variable-length 和 multi-request；
- zero history、non-zero history、连续 Decode、state reuse、empty batch；
- 与冻结的 Torch/reference tensor 逐级对比；
- 全链路 NaN/Inf 检查。

准入门槛：单 rank、仅加载被测 KDA 层/子模块的 eager 结果与 reference 对齐，每个 token 后的
state fingerprint 一致；此处不加载完整模型。

### 阶段 4：KDA 公开 NPU Kernel

按以下顺序逐项准入：

1. packed width-4 QKV causal conv + SiLU + state update；
2. Decode recurrent KDA；
3. Prefill gate preprocessing + chunk KDA；
4. per-head RMSNorm + full-rank output-gate epilogue。

每项都需要 Torch oracle、4-token NPU board、variable-length board、适用时的 graph capture 测试和 off/on A/B。
单项失败只能关闭该项，不能影响其他 fallback。

准入门槛：输出和 state 都 finite、精度正确；目标 Decode/Prefill shape 下不得比 baseline 更慢，
除非明确仅作为 correctness fallback。

### 阶段 5：MLA NPU Baseline 和 Output Gate

交付：

- 注册 MLA query normalize/project、本地 latent Prefill/Decode attention 和 value projection 的 Ascend 实现；
- 复用现有 in-place sigmoid output gate；
- MLA HW prolog 必须先通过目标 shape 上板。

测试：

- 单 rank、仅加载被测 MLA 层/子模块的 Prefill/Decode 对比 Torch oracle；
- current token、history-only、empty history、page boundary、output gate on/off；
- 不支持的 HW prolog shape 必须 fail closed。

准入门槛：单 rank 子模块 eager MLA 对齐；HW prolog 未通过精度和收益测试时保持关闭。
partitioned attention 的 output/LSE 和跨 rank merge 属于分布式 KV Cache TODO。

### 阶段 6：Grouped MoE、Shared Expert 和 Zero Expert

交付：

- 使用通用 top-k API 构建四个独立 group route；
- 实现 logical/group/physical/EP-local id 映射，路由输出统一为 `[4T,12]` global expert ID；
- 实现 identity zero expert，shared expert 只累加一次；
- 注册 NPU fused top-k；
- 接入目标 shape 支持的 grouped matmul/collective；
- 实现一个加载 1536 个 real experts、每 rank 192 个 local experts 的 flattened EP8 executor；
- 保留 Torch fallback；
- EP rank 输入 token 数为 0 时正常完成。

三组实验固定相同 checkpoint、输入、四组 TopK、EP8、每 rank 192 个 real experts 和数值阈值，只改变
executor 边界或物理 expert ID：

| 实验 | Executor | 每 rank expert 放置 | 用途 |
| --- | --- | --- | --- |
| A：reference | 4 个独立 `E=384` EP8 executor | 每组 48 个，共 192 个 | 当前 NPU 组合基线 |
| B：flattened-interleaved | 1 个 `E=1536` EP8 executor | 每组 48 个，共 192 个 | 相对 A 单独隔离四次执行合为一次的收益 |
| C：flattened-group-major | 1 个 `E=1536` EP8 executor | 仅一个 group 的 192 个；两 rank 合成完整一组 | 相对 B 单独隔离 group-to-rank 放置收益，作为目标候选 |

实验 B 需要显式 logical-to-physical remap 保持每 rank 四组各 48 个；实验 C 直接使用 group-major
连续 real-expert ID。三组的 identity-zero ID 都放在 real-expert 域外并保持原组语义，不能把 identity
route 计入 GMM token。实验只复用现有 routing、GMM、SwiGLU、finalize 和 HCCL 能力，不先开发新的
端到端 Grouped MoE kernel。A/B/C 选择只存在于测试/准入代码；生产路径最终只保留胜出布局和 Torch
correctness fallback，不长期维护三个运行模式。

测试：

- 手工构造 winner 的 router-only oracle；
- identity-zero、all-zero、duplicate route、boundary id、empty-rank；
- N=8 distributed leaf，覆盖 P eager 的 T32/T128/T1024 和 D eager/graph 的 BS1/BS2；
- A/B/C 使用相同输入逐项比较 output、route、expert token count 和 shared/zero contribution；
- 记录每层每 rank real-route 数、pair 聚合 route 数、`max/mean`、变异系数、空 expert/rank、GMM
  active tokens、collective 次数/字节/耗时、workspace/HBM 和端到端 Prefill/Decode 指标；
- Prefill/Decode 数学一致，Decode graph 地址稳定且不产生额外 specialization。

准入门槛：三组分布式结果都与同一单 rank dense oracle 对齐，所有 rank 执行匹配的 collective 序列；
B 相对 A 证明四次 dispatch/GMM/finalize 合并为一次且无性能或内存回退；C 相对 B 在保持一次 executor
和相同静态权重的条件下，pair 级负载符合四组约束，rank 级负载、HCCL 和 TPOT/Prefill throughput
没有不可接受回退。若 C 未通过则保留 B，不能为了理论均衡牺牲实测主路径。

### 阶段 7：OE 正确性和 Host Residency

交付：

- Lite n-gram id、special-token 行为、lookup、normalize 和 block projection；
- split OE key 加载到确定性的 host-resident table；
- 只 staging 当前 batch 需要的 row/block；
- request history 和 graph metadata 固定地址、slot-safe。

测试：

- 手工计算 n-gram id 和 special-token scope；
- Prefill/Decode continuation 等价；
- slot reuse 和 concurrent request；
- host/device memory 账本证明每张 NPU 上没有完整 OE table。

准入门槛：OE-on 与 reference 对齐；OE-off 产生预期且可解释的输出差异。

### 阶段 8：8P8D 权重和角色内 Eager 切分

交付：

- 基于 PR #1281 Qwen launcher 增加 public-safe Lite NPU 启动脚本；
- 直接建立 8P + 8D 共 16 rank 的进程、HCCL group、checkpoint shard 和角色 placement；
- 验证 KDA TP8、MoE EP8、dense/shared mapping 和 P/D checkpoint shard；
- 使用阶段 6 胜出的 MoE expert placement，并复核每 rank 192 个 real experts、route 分布和权重账本；
- 在尚未连接 P/D transfer 前，分别用冻结输入验证 P、D 角色内 eager 计算；
- 通过显式 debug flag 提供 layerwise NaN/Inf 检查。

准入门槛：16 rank 全部完成目标 shard 加载，P/D 角色内 eager 输出与冻结 reference 一致；每 rank
权重和 collective 符合 8P8D 声明，静态权重通过实际内存账本。此阶段不宣称 HTTP 服务已端到端可用。

### 阶段 9：8P8D 有限上下文 Eager 完整服务

交付：

- eager 模式拉起唯一目标拓扑 8P8D 的完整 Lite HTTP 服务；
- 使用适合 smoke 的小型本地 KV pool，MLA history 不做 CP/KVP 跨 rank 分片；
- 如 P/D 分离需要 cache 传输，先复用现有非分片整页/1:1-rank correctness contract，
  不引入 fragment ownership 或 direct-to-live 内存优化；
- health、model-list、确定性 completion、multi-turn、concurrent smoke；
- 对冻结请求执行全链路 NaN/Inf 和逐层定位开关。

准入门槛：8P8D 服务稳定且输出确定，冻结请求与 reference 对齐；每 rank 静态权重、cache 和 runtime
均通过实际内存账本。此阶段仍关闭 graph 和 overlap，也不声称 2M/高并发容量。

### 阶段 10：Decode Graph 和 Overlap

交付：

- Prefill 保持 eager，Prefill overlap 关闭；
- 首先 capture Decode BS1/BS2；
- 只有配置的最大并发需要时才增加更大固定 BS；
- graph metadata/state index 原地更新；
- graph 内不能出现 `.item()`、动态 host 分支或地址变化；
- graph/eager 对齐后再启用 Decode scheduler overlap。

测试：

- eager vs graph 的 token、logprob、logits、MLA page、conv state、recurrent state；
- sequential BS1、steady BS2、late-admission BS2；
- graph compile count 和 pointer stability；
- overlap off/on 对齐；
- 本地 cache slot admission 期间不存在 use-after-free。

准入门槛：graph 和 overlap 只能改变性能，不能改变输出或 state。

### 阶段 11：累计准入参考实现已有优化

不重复实现前面已经完成的内容。本阶段新增剩余的 loader/layer 优化，并进行累计服务 A/B。

| 优化 | 实现阶段 |
| --- | --- |
| Fused add + RMSNorm | 阶段 11 |
| P/D fused MoE top-k | 阶段 6 |
| Decode-only Weight-NZ | 阶段 11 |
| Packed conv、recurrent KDA、chunk KDA | 阶段 4 |
| Grouped MoE flattened executor 与 group-to-rank placement | 阶段 6 |
| OE block projection | 阶段 7 |
| In-place MLA output gate | 阶段 5 |
| KDA direct-to-live PD state | 最后 TODO |
| P eager/overlap-off，D graph/overlap-on | 阶段 10 |

Fused add + RMSNorm 和 Weight-NZ 分别使用独立实现/测试提交和独立服务 A/B。

MLA HW prolog 在目标 shape 不支持时继续 fail closed。Continuous Decode steps 和 Prefill graph 不属于本计划。

准入门槛：每项优化必须在目标 shape 上带来可测收益，或显著降低内存，并且没有精度回退。
被拒绝的候选需要记录结论，不能以半启用状态留在生产路径。

### 阶段 12：Lite 新增优化

#### 12A：单次合并 Hidden-state Projection

扩展现有 Kimi packed projection/loader，将以下权重放入一个 rank-local buffer：

`[Q | K | V | output_gate | forget_a | beta_a]`

head-sharded row 和 replicated low-rank row 保持原有 shard 语义。
`forget_b` 和 `beta_b` 的输入不同，继续使用两个独立 GEMM，不强行合并。

准入门槛：

- 与 unpacked projection 精确一致；
- projection latency 降低；
- Prefill live activation 增量可接受。

#### 12B：Packed QKV Causal Conv

从 projection、权重加载、conv state、kernel call、本地 cache state 一直到 graph capture，
全链路保持 packed QKV。禁止在每个 Decode step 临时执行三路 `torch.cat`。

准入门槛：只有一次 conv launch，state/output 对齐，并带来 Decode latency 收益。

#### 12C：Featurewise-beta 融合

- 首先新增一个 fused prepare kernel，完成 Q/K L2Norm、`sqrt(sigmoid(beta))` 和 K/V scale；
- Prefill chunk KDA 前保留这一个 prepare launch；
- 扩展 Decode recurrent KDA ABI，使其直接接收 featurewise beta logits；
- Decode 中将 prepare 折入 recurrent kernel；
- 绝不能在一个会再次 normalize K 的 kernel 前预先缩放 K。

准入门槛：

- normalize 顺序精确正确；
- 与 reference 对齐；
- 不产生额外的大临时 tensor；
- Decode 消除独立 beta-prepare launch。

### 阶段 13：首轮最终验收

最终验收必须运行累计配置，不能只验证孤立 leaf。

- 完成 unit/kernel test suite；
- 执行完整 `pre-commit run --all-files`；
- 完成 N=8 operator-board matrix；
- 8P8D 有限上下文服务 readiness、确定性 smoke、有界并发、late admission、retraction 和清理；
- 使用与冻结 reference 完全相同的 EvalScope prompt/evaluator 跑 GSM8K 前 100 条；
- 记录逐条输出、完成率、API error 和 NaN/Inf；
- 记录 Prefill throughput、TTFT、Decode TPOT、graph compile count、HBM 拆分、
  本地 cache capacity、host OE memory，以及 MoE 的 per-rank/pair route 与 GMM active-token 分布；
- 对阶段 11/12 所有优化做累计 off/on A/B；
- 保存 exact source SHA、launcher 参数、artifact manifest，并确保进程和端口清理干净。

最终门槛：

- GSM8K 100/100 请求完成，无 API error、空输出；
- 确定性输出相对冻结 reference 无回退；
- P 使用 eager，D 使用已准入 graph batch 和 overlap；
- 唯一目标拓扑 8P8D 完成有限上下文功能、精度和优化验收；
- 每张 NPU 上不存在完整 OE table；
- 每项启用的优化都有独立证据，以及 rollback switch 或 kernel-selection fallback。

## 7. 依赖顺序

关键路径固定为：

`model/load -> 本地 cache metadata -> KDA/MLA/MoE/OE 单卡子模块 eager
-> EP8 为核心的 8P8D 权重/角色内切分 -> 有限上下文 8P8D 全模型 eager 服务
-> graph/overlap -> fusion -> 首轮最终验收 -> 分布式 KV Cache TODO`

在 eager state 未对齐前不开始 graph 调试。CP8/KVP8、跨 rank page ownership 和 PD fragment transfer
不再阻塞 Eager、graph 或融合算子验收；它们只在首轮闭环后进入最后 TODO。
Projection packing 和 conv packing 放在功能 kernel 之后，确保问题能够局部归因。

## 8. 已知风险与停止条件

| 风险 | 最早判别实验 | 处理方式 |
| --- | --- | --- |
| 公开 KDA kernel ABI 只支持 scalar beta | 4-token featurewise-beta board | 保留标准 recurrence + Lite fused prepare，再增量扩展 ABI |
| Ascend MLA 无法返回可用 LSE | 最后 TODO 的两路 partial attention board | 不阻塞本地 MLA；做 KVP 前补 registered MLA solution |
| CachePD 的 CP 限制不仅是 rank planner | 最后 TODO 的 8P8D synthetic transfer contract | 扩展通用 topology/fragment 层，不新增模型侧 transfer |
| EP8 grouped collective 不支持目标 shape | router + identity expert distributed leaf | 保留 fixed-shape native HCCL fallback，记录 fused path 不支持 |
| group-major 两卡一组只改善 pair 级、却放大 pair 内 hot-expert 偏斜 | 阶段 6 的 A/B/C route 与 GMM 计数 | C 未优于 B 时保留 flattened-interleaved，不引入未验证的 EPLB |
| MLA HW prolog 不支持目标 hidden size | 真实 shape direct board | 保持关闭 |
| Host OE staging 成为 Prefill 瓶颈 | row/block hit rate 和 transfer profile | 只有测到瓶颈后才增加 block batching/prefetch |
| Graph 覆盖新接收的 KDA state | 最后 TODO 的 late-admission state fingerprint | 修复 live-slot publication/fence，禁止恢复第二份完整 staging cache |

## 9. 首轮最终验收的非目标

- Prefill graph；
- 完整模型单卡服务；
- continuous multi-step Decode scheduling；
- speculative decoding 或 draft model；
- 新的通用 parallelism framework；
- 2M 级等效 KV 容量与高并发容量准入；
- Prefill CP8/SP、Decode KVP8 以及与之配套的分布式 KV page storage；
- Megatron golden 对齐；
- checkpoint 本身未提供的强制低比特模型转换。

这些工作在 BF16/等价 checkpoint 路径以及全部目标优化验收完成后再单独规划。

## 10. 最后 TODO：分布式 KV Cache 与 PD 内存优化

以下工作保留为独立后续，不阻塞阶段 9--13 的有限上下文服务、graph、融合和
GSM8K 前 100 条验收。每项仍必须先有独立设计，再实现、上板和提交。

### TODO A：Prefill CP/SP 分布式 MLA Cache

- 将 packed prompt token 分散到 8 个 rank，保持全局 causal position 和 request segment 边界；
- 每 rank 只写自己负责的 MLA history page；
- 返回 partial output + LSE，通过统一 `attn_merge_state` 合并；
- 验证 chunk boundary、empty owner、page ownership 和 non-partitioned oracle。

### TODO B：Decode KVP8 分布式 MLA Cache

- 复制 query/current-token projection，historical MLA page 按 owner 分散到 8 rank；
- 每 rank 计算本地 partial attention/LSE，empty owner 贡献 zero output/-inf LSE；
- 用同一份 ownership metadata 驱动 cache recipe、block table 和 transfer planner；
- 与 non-partitioned oracle 对齐并验证固定 KVP8。

### TODO C：P/D Fragment Transfer 和 KDA One-copy

- 扩展通用 CachePD topology/transfer planner，表达 P-CP8 owner 到 D-KVP8 owner 的分片映射；
- MLA page 只传给目标 D owner；
- Decode live KDA conv/recurrent slot 直接作为接收目标，删除第二份完整 staging state；
- 验证 completion/publication、abort/retry/reuse、late admission 和 graph stream ordering。

### TODO D：长上下文容量验收

- 按 2M 口径重做每 rank 静态权重、MLA KV、KDA state、graph/runtime/workspace 账本；
- 运行长上下文、高并发、retraction、late admission 和内存回收；
- 只有 TODO A--C 的精度、one-copy 和容量证据都通过后，才声称 2M/高并发生产准入。
