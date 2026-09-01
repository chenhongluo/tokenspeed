# Lite NPU 阶段 6D：Grouped MoE Production 接线

## 1. 目标与边界

本阶段只把阶段 6C 的唯一 winner **A：四个独立 EP8 local leaf** 接入 Lite 模型生产模块，并完成
单层真实 checkpoint 的 exact-source 回归。完整 decoder、OE、CP/KVP attention、PD、HTTP 服务和
GSM8K 仍按总计划留在后续阶段。

生产拓扑与已验证的 Fluent 8P8D 保持一致：

| 角色 | token/cache 并行 | routed expert | dense/shared |
| --- | --- | --- | --- |
| Prefill | CP/SP8，rank 间 token 不同 | EP8 | TP1，权重复制 |
| Decode | KVP8，current token 在 rank 间复制 | EP8 | TP8 |

阶段 1 的参数骨架曾让 P/D 都使用 dense TP8，以便先冻结 shape 和严格 loader；阶段 6D 以真实角色
拓扑为准，P 改为 dense TP1，D 保持 dense TP8。完整模型仍只支持每个角色八个 rank。

## 2. Winner 权重放置

令 `G=4`、每组 real expert `E=384`、`EP=8`，每 rank 每组持有 `48` 个 expert。逻辑 expert
`(g,e)` 的 owner 和本地 packed 下标为：

```text
owner(g,e) = floor(e / 48)
local(g,e) = g * 48 + (e mod 48)
```

因此 rank `r` 的 canonical packed 权重顺序固定为：

```text
[g * 384 + r * 48 + j for g in 0..3 for j in 0..47]
```

实现只给通用 MoE checkpoint loader 增加可选的本地 expert ID 顺序；默认连续 owner 行为完全不变。
Lite 不复制第二套 loader，也不保留 B/C production 配置。

Ascend post-load 只转换一份 `[192,...]` packed storage。四个 leaf 是该 storage 的四个 `[48,...]`
view，并分别声明 `num_experts=384`、`ep_rank=r`、`ep_size=8`。不能注册四份重复 Parameter，也不能
为 leaf 再分配四份权重。

## 3. 公共 rank-local 数学

输入 `x` 先经过角色对应的 `proj_input`：

```text
p = proj_input(x)
p = RMSNorm(p) * grouped_moe_norm_scale
u = view(p, [T, 4, 768])
```

四个 router 保持独立 softmax、selection-only correction bias 和 TopK。NPU tensor 使用已经准入的
`moe_softmax_bias_topk`；CPU/reference 继续使用 Torch 实现。每组调用阶段 6B 的同一 `moe_apply`
plan 和本组 48-expert view。

identity route 不进入 leaf。四组 real partial 完成 EP collective 后才把 identity contribution 加一次，
然后执行 `proj_output`；shared expert 始终消费原始 `x`，每 token 只累加一次。

## 4. Prefill 数据流

P 的 dense/shared 权重复制，输入为本 rank 的 SP token rows。`global_sp_num_tokens` 是一个长度为 8 的
显式 split，阶段 6D 由 Grouped MoE forward 接口消费；其在完整执行上下文中的生产由阶段 9A 负责。

```text
local x
  -> replicated W_in / RMSNorm / four local routers
  -> token AG(grouped input, route weight, route ID)
  -> four EP8 local leaves
  -> token RS(concatenated real partial)
  -> add local identity
  -> replicated W_out
  -> add replicated shared expert(local x)
```

route 在 AG 前按本地 token 计算，避免八个 rank 对完整 prompt 重复 router GEMM。四个 group 共用一次
grouped-input AG、一次 route-weight AG、一次 route-ID AG 和一次 concatenated-output RS；不会为四个
leaf 重复 HCCL collective。split 支持不等长和零 token rank。

P 不使用 dense TP8，因此没有 feature AllGather 或 dense AllReduce。这一点同时避免把不同 SP token
rows 错误送入 dense AllReduce。

## 5. Decode 数据流

D 的 KVP8 contract 要求八个 rank 消费相同 current-token rows；它不是通用 request-DP。dense/shared
使用 TP8：

```text
replicated x
  -> local W_in shard
  -> feature AllGather(dim=-1)
  -> RMSNorm / four routers / four EP8 local leaves
  -> expert AllReduce
  -> add identity once
  -> local W_out input slice and local shared expert partial
  -> dense AllReduce
```

expert AllReduce 和 dense AllReduce 不能合并：前者先补齐每个 768 维 group output，后者才可对
`W_out` 的 384-column shard 与 shared partial 求和。直接把 local expert partial 送入本 rank 的
`W_out` shard 会漏掉“其他 rank expert × 本 rank W_out columns”的交叉项。

BS1/BS2 路径不分配动态 workspace，不读取 host tensor value，并继续由阶段 6B leaf 支持 NPUGraph。
graph capture/replay 需要覆盖 feature AG、四个 leaf、两次 AllReduce 和 identity/shared 合并。

## 6. 通信归属

Grouped MoE 模块拥有上述内部通信，后续 decoder 不再对该 MLP 调用通用 `CommManager` 的
`pre_mlp_comm/post_mlp_comm`，避免双重 collective。原因是通用 manager 只区分 AR 与 token AG/RS，
不能表达 P 的 replicated dense 与 D 的 KVP-replicated current token，也不能表达 expert reduction 和
dense reduction 两个不同的代数边界。

所有通信只调用 `distributed.comm_ops`；Lite runtime 不直接依赖 HCCL 或 `torch_npu`。N=1 的 portable
reference 不进入任何 collective。

## 7. Fail-closed 约束

- N=8 P 只接受 `CP=8, attention-DP=1, dense TP=1, KDA TP=8, MoE EP=8`；
- N=8 D 只接受 `CP=1, attention-DP=8, dense TP=8, KDA TP=8, MoE EP=8`；
- P 缺少八项 `global_sp_num_tokens`、split 总数与输入不符或 rank 顺序不一致时立即失败；
- D 不接受 rank-local request-DP 输入；该复制契约由阶段 9B 的 KVP 集成测试证明；
- Ascend leaf 未 post-process、expert view 非连续或 source expert 无重无漏检查失败时立即失败；
- EPLB、冗余 expert、量化、MC2 和 B/C production layout 继续不支持。

## 8. 测试与提交

实现提交至少覆盖：

1. CPU/meta：P/D mapping、P replicated dense 与 D TP8 shape、A placement 1536 expert 双射；
2. loader：八个 rank 的 source expert 无重无漏，每 rank 四组各 48 个，默认通用 loader 行为不变；
3. Torch 数学：real/identity/shared、重复 route、零 token、不等长 P split、D TP8 两级 reduction；
4. NPU8 synthetic：P T32/T128/T1024，D BS1/BS2 eager/graph，完整 production wrapper 对独立 FP32
   oracle；
5. NPU8 checkpoint：真实单层 router/expert/projection/shared 权重，按
   `lite-npu-phase-06d-precision-gate.md` 分别验证 leaf、collective、wrapper、完整输出和 graph；
6. exact-source：提交 SHA、archive SHA、目标文件 SHA、本地/tracking/远端 SHA 和清理状态写入独立
   验证记录。

提交顺序固定为：本设计文档；实现与测试；exact-source 验证记录。每个提交通过全量 pre-commit、
signed-off，并立即推送 `hz/lite`。

## 9. 本阶段明确不做

- 不实现完整 Lite decoder 或模型 forward；
- 不提前实现 CP/KVP attention metadata、PD cache transfer 或 OE；
- 不增加 role/layout 用户开关，角色由唯一 8P8D mapping 决定；
- 不把 6C 离线 A/B/C harness 改造成 production executor；
- 不为后续融合、overlap、Weight-NZ 或量化预建抽象。
