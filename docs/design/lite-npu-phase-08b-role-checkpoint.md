# Lite NPU 阶段 8B：8P8D 完整 Checkpoint 与 Role-local Eager 设计

## 1. 目标与边界

阶段 8A 已连接 Lite decoder、model 和 logits 的数值 forward。阶段 8B 只完成以下闭环：

1. 用生产 loader 在两个独立八 rank HCCL world 中同时加载完整 checkpoint；
2. 固化 P8/D8 的参数 ownership、local fingerprint 和静态 HBM 账本；
3. 复用阶段 0 的冻结 tensor reference，对 cache-independent role-local eager 数据流做逐层数值检查；
4. 提供显式 `--check-finite`，定位第一个非 finite stage；
5. 提供公开安全、只接受命令行路径的 8P8D checkpoint probe 启动入口。

本阶段不实现 MLA history page 的 CP8/KVP8 owner、partial output/LSE merge、PD transfer、HTTP
服务、graph、overlap 或新融合算子。完整 28 层连续 forward 中的 MLA attention 依赖阶段 9，因此
8B 不会用注入 reference attention 的 replay 冒充完整服务推理。

## 2. 审计结论与最小生产增量

### 2.1 已有能力

- `LiteCheckpointLayout` 已覆盖 129,870 个 source tensor，并为 dense、KDA、EP expert、replicated
  MLA、host OE 和 OE projection 定义唯一 placement；
- `DefaultModelLoader` 已负责 safetensors 迭代、模型构造、`load_weights()` 和所有 post-load hook；
- `Mapping` 已原生支持独立的 `linear_attn_tp_size`，distributed initializer 也已创建该 TP group；
- 阶段 6D 已验证 P/D Grouped MoE 的真实 checkpoint 单层 EP8 执行；阶段 3--7 已分别验证 KDA、MLA、
  Grouped MoE 和 OE leaf；
- 阶段 0 已冻结 Prefill 一步和 Decode 三步共 460 个语义 tap，不需要生成新的模型基准。

### 2.2 唯一缺失的生产配置接线

`Mapping` 虽支持 `linear_attn_tp_size`，`ServerArgs` 尚未暴露或传递它。现有默认值跟随 attention TP；
Lite 的 P 是 MLA `TP1+CP8`，D 是 MLA `TP1+DP8`，两者却都要求 KDA TP8。因此只补一个通用
`--linear-attn-tp-size` 参数并传给已有 `Mapping` 构造，不增加 Lite 专用 mapping 类或隐式模型名覆盖。

启动前必须验证最终 resolved mapping，而不是只验证原始参数。缺少该参数时 Lite 模型现有
`_validate_mapping()` 会在分配完整权重前 fail closed。

## 3. 固定 8P8D 拓扑

P 和 D 是两个独立八 rank process group；它们并行存在于同一 16-NPU 节点，但不组成一个 world-size
16 的模型并行域。

| mapping | Prefill world | Decode world |
| --- | --- | --- |
| world / PP | 8 / 1 | 8 / 1 |
| MLA attention | TP1 + CP8 + DP1 | TP1 + CP1 + DP8 |
| dense/shared/vocab | TP1 | TP8 |
| KDA | linear TP8 | linear TP8 |
| routed expert | TP1 + EP8 | TP1 + EP8 |
| MLA weights | 每 rank 完整复制 | 每 rank 完整复制 |
| OE table | host mmap | host mmap |
| OE projection | 每 rank完整 18 MiB | 每 rank完整 18 MiB |

P 启动沿用已有 `ENABLE_CP` 入口把 attention width 解释为 CP；D 使用 attention DP8。probe 为 P/D
分别选择独立 rendezvous port，并把 role-local rank 0--7 映射到互不重叠的 device 0--7 和 8--15。

## 4. 完整 Checkpoint 加载契约

probe 必须调用生产 `DefaultModelLoader` 数据流，不能用一个只为测试写的 safetensors copy loop 代替：

```text
ModelConfig + LoadConfig + resolved Mapping
  -> construct FLASHLocalForCausalLM on target device
  -> DefaultModelLoader safetensors iterator
  -> strict Lite load_weights
  -> KDA / MLA / Grouped MoE post-load hooks
  -> eval model
```

完整 source stream 的 missing、unexpected、duplicate、shape/dtype 或 local target coverage 错误继续由
现有 strict loader 拒绝。阶段 8B 不增加 shard 文件重排、第二份 checkpoint、模型专用 prefetcher或新的
load format。

加载后每个 rank 生成匿名化 local manifest：

- role、local rank、参数名/shape/dtype/category 的聚合计数；
- KDA TP shard、dense shard、192 个 local real expert、replicated MLA 和 OE projection 的独立
  fingerprint；
- 12 张 OE table 的 storage pointer 只用于本机 file-backed/PSS 检查，不写入结果 artifact；
- 所有 fingerprint 与由标准 safetensors slicing 独立生成的冻结 expected manifest 比较。

同一 local rank 的 P/D expert、KDA、MLA 和 OE projection fingerprint 应一致；P/D dense fingerprint
因 TP1/TP8 ownership 不同，分别比较自己的 expected manifest。

## 5. 静态权重公式与预期

以下数值由当前真实 config、实际 parameter object 和 post-load derived buffer shape计算；OE 主表不计
入 HBM。`derived` 包含 KDA packed conv 和 MLA `w_kc/w_vc`，不包含 184-byte non-persistent
`ignore_tokens`、cache、HCCL buffer、算子 workspace 或 allocator reserve。其公式为
`21 * 3 * 512 * 4 * 2 + 7 * 2 * 32 * 128 * 512 * 2 = 58,978,304 bytes`。

| 模块 | P bytes/rank | P GiB/rank | D bytes/rank | D GiB/rank |
| --- | ---: | ---: | ---: | ---: |
| KDA parameters | 369,143,376 | 0.343792 | 369,143,376 | 0.343792 |
| MLA parameters | 634,023,936 | 0.590481 | 634,023,936 | 0.590481 |
| Grouped MoE | 15,469,475,840 | 14.407072 | 13,157,365,760 | 12.253752 |
| OE projection | 18,874,368 | 0.017578 | 18,874,368 | 0.017578 |
| embedding/head/outer norm | 2,013,616,128 | 1.875326 | 252,008,448 | 0.234701 |
| **Parameter storage** | **18,505,133,648** | **17.234249** | **14,431,415,888** | **13.440303** |
| post-load derived | 58,978,304 | 0.054928 | 58,978,304 | 0.054928 |
| **逻辑 resident static** | **18,564,111,952** | **17.289177** | **14,490,394,192** | **13.495231** |

P 比 D 多出的主要部分不是 attention，而是 dense/shared Grouped MoE 和完整 embedding/lm-head；D 的
OE 主表也保持 host resident，因此不能套用旧的 D/rank 额外 27 GiB/8 表分片口径。

每个进程仍会 mmap 28,991,102,976 bytes 的 OE payload。其虚拟 mapping/RSS 不能算作 NPU 静态权重；
16 进程的 page PSS 总和必须继续接近一份实际 resident page set，且每张 NPU 的 table-adoption
allocated 增量为零。

## 6. HBM 账本

每个 worker 在同步点记录：

1. NPU context + HCCL group 建立后 baseline；
2. model parameter object 构造后；
3. strict checkpoint load 完成后；
4. 所有 post-load hook 完成并 synchronize 后；
5. role-local eager replay 后的 current/peak；
6. 进程退出后的设备清理状态。

每个点同时记录 `memory_allocated`、`memory_reserved`、`max_memory_allocated` 和设备进程内存。判定规则：

- `sum(parameter storage bytes)` 必须逐 byte 等于第 5 节公式；
- derived buffer 的逻辑 byte 必须逐 byte等于 58,978,304；
- 八个同 role rank 的逻辑 resident byte 必须一致；
- OE table adoption 不得产生 GiB 级 HBM 增量；
- allocated/reserved 与逻辑 storage 的差值单独归为 runtime/allocator，不通过修改公式隐藏；
- 任一 rank OOM、出现未解释的约 27 GiB table copy或明显的 rank 内存倾斜时停止，不进入数值 replay。

阶段 8B 不分配最终 token KV pool，因此本账本是“权重 + loader/post-load + probe runtime”，不是最终
服务内存。cache/runtime 的生产账本仍在阶段 11。

## 7. Role-local Eager 验证

### 7.1 为什么不能在 8B 声明完整连续 forward

阶段 9 之前没有 MLA CP8/KVP8 page owner和 partial LSE merge。若在 8B 直接运行 28 层连续 P/D
forward，只能复制完整 MLA history或注入 reference attention；前者不是目标 placement，后者不是真实
attention。两者都不能作为完整服务正确性证据。

### 7.2 本阶段实际检查的数据流

8B 使用阶段 0 已冻结的 layer-boundary tensor，按真实 P/D mapping 分段重放：

1. embedding、OE merge、final norm 和 dense logits head使用真实 checkpoint 权重；
2. 每个 KDA layer 运行真实 TP8 projection、KDA core、output gate、o-proj 和 AllReduce，并比较冻结
   KDA attention tap；
3. 每个 MLA layer 运行阶段 5 已准入的 projection、local non-partitioned attention oracle和 output gate，
   但不把它记作 CP/KVP；
4. 每层以冻结 attention output进入真实 residual、Grouped MoE 和第二个 residual，比较 attention
   residual、MoE 和 layer output tap；
5. P 侧用冻结 rank token split执行 SP-aware Grouped MoE，D 侧用复制 current-token执行 dense TP8路径；
6. logits 只比较冻结行和 token，不拉起 sampler/HTTP。

这套分段 replay 会发现错误 checkpoint shard、重复/缺失 collective、residual 顺序、P/D dense
ownership、MoE placement、OE merge 和 logits 拼接问题。阶段 9 再把 MLA partial attention 与 cache
owner补入同一层级 tap；阶段 11 才运行不注入 boundary tensor 的完整连续请求。

### 7.3 数值门槛

- route ID、token ID、shape、owner 和 collective count：exact；
- KDA state：沿用阶段 0 的逐 token fingerprint；
- Grouped MoE leaf/collective/wrapper：沿用阶段 6D 分层 rel-L2 门槛；
- OE、MLA epilogue 和最终 BF16 tensor：沿用各 leaf 已冻结门槛；
- 所有输出与 state 必须 finite；不能只检查最终 logits。

## 8. `--check-finite` 设计

finite 检查只属于 probe，不进入生产 forward 热路径。开关打开时，在以下 boundary 检查 tensor：

```text
embedding -> OE merge
layer.N.attention input / local output / reduced output
layer.N.attention residual
layer.N.grouped_moe local leaf / collective output / final output
layer.N.output
final norm -> local logits -> gathered logits
```

首个 NaN/Inf 立即失败，只记录 role、rank、stage、layer、shape、dtype和 non-finite count；不保存真实
prompt、checkpoint 路径或 tensor payload。开关关闭时不注册 hook、不执行 `.item()` 或 D2H。

## 9. Probe 与测试文件边界

最小实现预计只需要：

1. `ServerArgs` 增加一个通用 linear-attention TP 参数及 focused test；
2. 一个 `test/ci_system` 下的 checkpoint role worker/controller；
3. 一个 shell 入口同时启动两个独立八 rank `torchrun`，只负责参数转发、PID trap 和结果聚合；
4. 一个小尺寸 CPU/meta test，覆盖静态 byte、topology、manifest、finite failure和命令构造。

脚本只接受 `--checkpoint`、`--reference`、`--output-root` 和 rendezvous/device 参数；不写默认私有路径、
主机、凭据或端口。checkpoint/reference 不存在、目标 device 重叠、端口冲突或进程残留时在分配权重前
失败。

## 10. 上板顺序与硬门槛

1. 本地 CPU/meta：P/D mapping、参数 byte公式、CLI、manifest schema和 negative cases；
2. NPU8 synthetic：先 P 后 D，验证 HCCL group和role-local collective序列；
3. NPU8 exact：先单 role完整 checkpoint，确认 loader/HBM/fingerprint；
4. NPU16 exact：P8/D8 同时加载并停在 barrier，采集 HBM/PSS；
5. NPU16 replay：打开 `--check-finite`，完成分段 reference 对齐；
6. 累计阶段 1--8B focused tests；
7. 精确停止两个 role world，确认 16 张卡无进程并删除明确的临时源码目录。

任一 rank strict coverage、fingerprint、logical byte、finite、numerical gate或清理失败，阶段 8B 均不准入。

## 11. 提交顺序

1. 本设计文档独立提交并推送；
2. linear-attention TP CLI 接线与 tests独立提交并推送；
3. probe/controller、静态账本和分段 replay tests独立提交并推送；
4. exact-source NPU8/NPU16 验证记录独立提交并推送。

阶段 8B 闭环后才进入阶段 9A。不会在同一提交混入 CP/KVP、PD、服务 launcher 或 graph。
