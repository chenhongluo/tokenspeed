# Lite NPU 阶段 8A：Decoder 与模型数值执行接线

## 1. 目标与边界

阶段 3--7 已分别完成 KDA、MLA output gate、Grouped MoE EP8 和 OE 的生产模块，但 Lite 顶层模型仍只
拥有参数骨架，decoder 与 causal-LM `forward` 明确未接线。本阶段只完成从 token 到 logits 的数值数据流，
并用小尺寸模型在 CPU 与单 NPU 上验证各模块组合后的 residual、归约和输出语义。

本阶段不做以下工作：

- 不加载完整 checkpoint，不拉起 8P8D HTTP 服务；
- 不实现 Prefill CP8 的 MLA page 分片、Decode KVP8 或 LSE partial merge；
- 不实现 PD transfer、Decode graph、overlap、Weight-NZ 或新融合算子；
- 不复制 Kimi、Qwen 或 LongCat decoder，也不引入新的通用模型基类。

完整 P8/D8 checkpoint shard、collective 与 16-rank 内存账本属于阶段 8B；CP/KVP attention ownership
属于阶段 9。阶段 8A 的 N=8 分支只冻结接口和 collective 归属，不宣称完整角色输出已经上板通过。

## 2. 复用边界

实现直接复用以下已有组件：

| 功能 | 复用组件 | Lite 增量 |
| --- | --- | --- |
| embedding | `VocabParallelEmbedding` | TP 维度取 `mapping.dense` |
| lm-head | `ParallelLMHead` | TP 维度取 `mapping.dense` |
| logits | `LogitsProcessor` | rank/size/group 同样取 `mapping.dense` |
| norm | `RMSNorm` | 仅替换无 forward 的参数占位类 |
| collective | `distributed.comm_ops` | KDA output 只增加一次 linear-TP AllReduce |
| attention | `LiteKDAParameters` / `LiteMLAParameters` | 不复制数学或 cache backend |
| MLP | `LiteGroupedMoEParameters` | 只透传 P 的 token split metadata |
| OE | `LiteNgramParameters` / `LiteOEStatePreparer` | 消费已准备的 fixed staging |

不直接继承 `BaseCausalLM`。该基类按 `mapping.attn` 决定 embedding/lm-head/logits 的分片，而 Lite D
把同一个八 rank 域同时解释为 MLA attention-DP8、dense/KDA TP8 和 MoE EP8。若沿用基类，D 会创建
replicated full-vocab head，但 checkpoint loader 实际只加载 `vocab/8`，参数 ownership 与 logits token
ID 都会错误。Lite 保留现有严格 loader，仅把现成的 vocab-parallel 组件按 dense mapping 装入原层级。

## 3. 顶层数据流

令输入 token 数为 `T`、hidden size 为 `H`。模型入口固定为：

```text
input_ids [T]
  -> dense-vocab-parallel embedding [T,H]
  -> merge(prepared OE [T,12,256]) [T,H]
  -> decoder layer 0 ... layer 27
  -> final RMSNorm [T,H]
  -> dense-vocab-parallel lm-head
  -> LogitsProcessorOutput
```

OE 不在 model forward 中重新执行 CPU hash 或 host lookup。`ModelExecutor` 已在 forward 前调用
`prepare_external_inputs()`；model 只读取 `prepared_raw_oe(T)`，并与当前 `input_ids` 一起执行已验证的
projection、special-token bypass 和 `/sqrt(13)` merge。prepared 行数与 model 输入不一致时立即失败，
不能静默截断真实 token 或读取旧 staging。

N=1 使用同一条生产数据流；测试不会增加 `raw_oe=`、`skip_oe=` 等仅供测试的 forward 分支。

## 4. Decoder residual 语义

Lite 每层使用标准双 residual，不使用 Kimi AttnRes。令层输入为 `x`：

```text
a_in = RMSNorm(x)
a    = Attention(a_in)
r    = x + reduce_attention(a)
m_in = RMSNorm(r)
m    = GroupedMoE(m_in)
y    = r + m
```

这与 reference implementation 的
`post_attention_layernorm(attention_output, residual)` 后再执行 `residual + moe_output` 等价。阶段 8A
先使用非融合 add/norm，阶段 13 再独立验证 fused add+RMSNorm；不能为了提前优化改变 residual 的
舍入点。

`ctx.forward_mode.is_idle()` 不在本阶段发明新的跨 rank 行为。单 NPU probe 可验证纯 idle no-op；D
角色在真正 KVP 接线前不能依赖通用 attention-DP scheduler 让部分 rank 独立 idle，因为 dense TP8 和
MoE EP8 要求八个 rank 执行一致 collective 序列。这一服务级约束在阶段 9/11 验证。

## 5. Attention output 的归约归属

KDA 与 MLA 的输出 placement 不同，decoder 不能统一 AllReduce：

- KDA 的 `o_proj` 输入和权重按 linear TP8 切分，输出是 `[T,H]` partial；N=8 时必须沿
  `mapping.linear_attn.tp_group` AllReduce 一次，N=1 为 no-op；
- 当前 MLA 权重和 output 均完整复制，`LiteMLAParameters` 返回完整 `[T,H]`，decoder 不再归约；
- 阶段 9A/9B 将在 attention backend 内生成 CP/KVP partial output 与 LSE，并经
  `attn_merge_state` 合并。该 merge 不能伪装成普通 output AllReduce，也不在阶段 8A 预建占位接口。

测试必须对 KDA/MLA 分别记录 collective 调用，证明 KDA 恰好一次、MLA 零次，防止重复求和或漏求和。

## 6. Grouped MoE 调用契约

`LiteGroupedMoEParameters` 已拥有 P/D 的全部内部通信，decoder 不再套 `CommManager` 的 MLP
pre/post collective：

- P mapping (`CP8, dense TP1, MoE EP8`) 将 `ctx.global_num_tokens` 原样传为
  `global_sp_num_tokens`；列表必须是 rank 顺序的八项 token split；
- D mapping (`attention-DP8, dense TP8, MoE EP8`) 传 `None`，由模块执行 feature AG、expert AR 和
  dense AR；
- N=1 传 `None`，使用 portable reference；
- MLP 输出已是完整 residual contribution，decoder 只执行一次 `r + m`。

本阶段不改变阶段 6C 选定的 A 布局，也不重新引入 B/C production 开关。

## 7. Embedding、lm-head 与 logits

embedding 和 lm-head 均沿 dense mapping：

| 角色 | dense TP | embedding | lm-head/logits |
| --- | ---: | --- | --- |
| P | 1 | 每 rank 完整权重，消费本地 SP rows | 完整 vocab；阶段 9 再处理 SP hidden gather |
| D | 8 | vocab shard masked lookup + dense AR | vocab shard GEMM + dense AG |
| N=1 | 1 | portable full vocab | portable full vocab |

目标 vocab 必须能被 dense TP 和现有 vocab padding 规则整除。严格 checkpoint loader 继续校验实际
Parameter shape；不为不对齐的测试 config 增加 padding-copy 特例。

`LogitsProcessor` 继续负责 `gather_ids`、Prefill logprob 和 `LogitsProcessorOutput`，Lite 不实现第二套
logits slicing。它的 TP rank/size/group必须与 lm-head 的 dense mapping 完全相同；构造期不一致立即失败。

## 8. 权重与 post-load 生命周期

组件替换后参数名字保持不变：

```text
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

因此 `LiteCheckpointLayout`、host OE adoption 和 expert loader 不增加 alias 表。已有 loader 在加载结束后
遍历模块调用 `process_weights_after_loading()`，继续负责 KDA packed conv、MLA absorbed 权重和 Ascend
MoE plan；model forward 不延迟创建新 Parameter，也不保存第二份 derived weight。

## 9. Debug finite probe

阶段 8A 的层级 NaN/Inf 定位是验证工具，不是生产热路径。新增 probe 提供显式 `--check-finite`：

- 使用 forward hooks 检查 embedding/OE merge、每层 attention、attention residual、MoE、layer output、
  final norm 和 logits；
- 发现第一个非 finite tensor 时报告匿名化 stage/layer/tensor shape并失败；
- 开关关闭时不注册 hook，生产 forward 不出现 `.item()`、D2H 或额外同步；
- 最终服务仍复用已有 logits NaN guard做请求级 containment。

probe 不保存真实 prompt、checkpoint 路径或 tensor payload到提交记录。

## 10. 测试矩阵

实现提交至少包含：

1. 参数层级：替换 embedding/head/norm 后 source/target 名字与 strict loader coverage 不变；
2. residual oracle：确定性 fake attention/MoE 对上述双 residual 公式逐元素对齐；
3. reduction ownership：N=8 fake mapping 下 KDA 恰好一次 linear-TP AR、MLA 零 AR；
4. MoE metadata：P 精确透传八项 `global_num_tokens`，D/N=1 传 `None`；
5. OE/model：prepared rows、special bypass、embedding merge、final norm 和 logits shape；
6. causal-LM：Decode 与 Extend 的 `LogitsMetadata`/`gather_ids` 走既有 `LogitsProcessor`；
7. negative：OE 未准备、行数不符、dense vocab shape不符、非法 mapping、non-finite probe；
8. 累计回归：阶段 1--7 的 loader/cache/KDA/MLA/Grouped MoE/OE focused tests全部继续通过。

单 NPU exact-source 验证使用小尺寸、同数据流的 KDA 与 MLA 层分别运行 Prefill-like 和 Decode-like
输入，比较独立 FP32 oracle，要求所有 tap finite；BF16 门槛沿用各 leaf 已冻结的门槛，不用宽松的最终
logits误差掩盖中间层问题。验证时关闭 graph 和 overlap。

## 11. 提交与停止条件

提交顺序固定为：

1. 本设计文档独立提交并推送；
2. 最小 production 接线、focused tests 和 probe提交并推送；
3. exact-source NPU0 与累计回归写入独立验证记录，再提交并推送。

出现以下任一情况时不进入阶段 8B：parameter name/shape 漂移、KDA/MLA 归约次数错误、P token split
未透传、OE staging 被重复生成、任一必需 tap出现 NaN/Inf、或 NPU 与 oracle 超出既有 leaf 门槛。
