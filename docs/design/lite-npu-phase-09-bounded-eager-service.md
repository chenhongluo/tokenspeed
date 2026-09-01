# Lite NPU 阶段 9：有界 Cache 的 8P8D Eager 服务

## 1. 目标

本阶段首次把 Lite 的完整 decoder、scheduler、CachePD 和 OpenAI 兼容入口连接成可请求的
8P8D 服务。Prefill 与 Decode 各使用一个 8-rank world，均只运行 eager；服务先覆盖单请求和
BS2、有界 4K 上下文，不承担最终 2M token 容量或性能结论。

阶段 9 的主要约束是：跨 rank KV page ownership 已被后置，但 TokenSpeed 现有 Lite mapping 只接受
Prefill CP8 或 Decode attention-DP8。前者被 CachePD 明确拒绝，后者会让 scheduler 把请求分给不同
DP rank，不能和 Lite 的 dense TP8、KDA TP8 collective 直接组成一个 lockstep forward。因此不能只写
启动脚本，也不能把 role-local checkpoint probe 外推成服务已经可用。

本阶段增加一个有界、可删除的正确性拓扑：8 个 rank 对同一 batch lockstep 执行，MLA 权重和完整
latent KV 在每个 rank 复制；dense/KDA 使用 TP8，routed expert 使用 EP8。该拓扑只复用现有 cache
和 collective，不新增分布式 KV 协议。

## 2. 明确不做

- 不实现 Prefill CP/SP、Decode KVP、跨 rank KV page ownership 或 partial LSE merge；
- 不实现 PD fragment transfer、2M token pool、高并发或 KDA one-copy 的最终容量优化；
- 不启用 Prefill graph、Decode graph、overlap、投机推理、量化或新增融合算子；
- 不修改非 Lite 模型的 parallelism、attention backend 或 PD 行为；
- 不把服务依赖安装进基础 Python，目标机只使用阶段目录内的精确版本依赖。

## 3. 8-rank lockstep 拓扑

Prefill 与 Decode 使用相同的模型 ownership，避免 CachePD 两侧布局不一致：

| 部分 | 每个 role 的阶段 9 布局 | 说明 |
| --- | --- | --- |
| scheduler / execution group | TP8 lockstep | 8 rank 接收相同 token rows |
| MLA 参数 | replicated | 沿用 checkpoint loader 的既有 ownership |
| MLA latent cache | replicated | 每 rank 保存完整有界 history |
| dense/shared/vocab | TP8 | 沿用 Decode 已验证的 weight/collective 路径 |
| KDA | linear TP8 | 每 rank 保存本地 head 的 conv/recurrent state |
| routed expert | EP8 | 每组每 rank 48 个 real expert |
| OE 主表 | Host replicated mmap | HBM 只保存本轮 projection/staging |

CLI mapping 仍以 `attn_tp_size=8` 建立 lockstep process group 和一一对应的 PD rank topology，但 Lite
MLA component 的计算几何必须显式保持 `attn_tp_size=1`：每 rank 消费全部 32 个 head、全部 replicated
MLA 参数和完整 latent page。这个 override 只在 architecture 为 `FLASHLocalForCausalLM` 且 resolved
mapping 精确命中阶段 9 拓扑时生效；其他模型和 Lite 的 CP8/attention-DP8 目标拓扑不变。

Lite mapping validation 新增且只新增下面的合法组合：

```text
world=8
attention TP/CP/DP = 8/1/1
dense TP = 8
linear-attention TP = 8
MoE TP/EP = 1/8
PP = 1
```

该组合下 Grouped MoE 对 Prefill 和 Decode 都复用现有 local-expert + EP AllReduce 路径。它已经支持
任意 token 行数，不需要为 Prefill 复制一套 wrapper；CP8 专用的 token AllGather/ReduceScatter 路径
仍只在最终 CP 实现中使用。

## 4. Cache 与 PD 传输

阶段 9 不新增 allocator 或 transfer schema。`LiteRecipe` 已声明：

- MLA full-history group 使用 `full_suffix`；
- 三个 KDA recurrent-state group 使用 `latest_snapshot`；
- Lite OE 最近三个 token state 使用 `latest_snapshot`。

P/D 都是 CP1、TP8，因此通过现有 `PDParallelTopology`。每个 Prefill local rank 只向相同 local rank 的
Decode peer 发送一个完整 manifest：MLA 的八份源 cache 内容相同，KDA state 是对应 head shard，OE
state 相同。阶段 9 使用 `disaggregation_layerwise_interval=0`，先验证一次 final transfer；layerwise
streaming 和 transfer/compute overlap 后置。

默认有界容量固定为：

| 参数 | 默认值 |
| --- | ---: |
| `max_model_len` | 4,096 |
| `max_total_tokens` | 8,192 / rank |
| `max_num_seqs` | 2 |
| `chunked_prefill_size` | 1,024 |
| `prefix_granularity` | 64 |

MLA cache 在 rank 间复制，所以 8,192 是服务逻辑总容量，不乘 8。启动时必须记录 recipe 反解出的实际
token capacity、LCM parent 数和各 cache group bytes；实际容量小于 8,192 时 fail closed，不调整门槛
掩盖内存问题。

## 5. Eager 角色策略

两个 role 都固定以下运行策略：

```text
device=npu
enforce_eager=true
disable_prefill_graph=true
disable_pdl=true
disable_overlap_schedule=true
disable_autotune=true
sampling_backend=greedy
temperature=0
```

Prefill role 本来会自动强制 eager，但启动脚本仍显式传入相同开关，使 P/D 配置可审计。Decode graph
和 overlap 在阶段 10 分别启用，不允许阶段 9 的隐藏默认值提前改变数值基线。

## 6. 启动入口

新增一个公开安全的 `test/ci_system` 8P8D launcher，复用既有 PD smoke 脚本的三个进程边界：

1. Prefill gRPC engine 使用可见设备 0--7；
2. Decode gRPC engine 使用可见设备 8--15；
3. SMG gateway 暴露单个 OpenAI 兼容 HTTP endpoint。

脚本只从环境或参数接收 model、端口、日志目录和容量，不包含 checkpoint、主机或凭据默认值。它在
启动前完成：16 张卡不重叠、全部端口可绑定、model/config/tokenizer 可读、精确 Python 包可导入、
P/D 参数完全一致。任一 engine 未进入 `SERVING` 时停止另外两个进程并返回非零。

目标环境若缺少 `tokenspeed-smg*`，只按 `python/pyproject.toml` 的精确 pin 安装到阶段目录；不安装
TokenSpeed 第二份 runtime，也不修改系统 site-packages。source checkout、kernel package 和阶段依赖
都由显式 `PYTHONPATH` 组成。

## 7. 实现拆分

### 9A：replicated MLA lockstep topology

- Lite mapping 接受新的唯一组合；
- Lite MLA config 在该组合下使用 replica head geometry；
- P/D cache contract 必须 byte-identical；
- 非 Lite、CP8 和 attention-DP8 行为保持不变。

### 9B：8P8D eager launcher

- 复用既有 PD launcher 的 readiness、PID trap 和 gateway 接线；
- 增加 NPU 可见卡、CANN、端口、capacity 和依赖 preflight；
- 提供 `--check`/dry-run 测试入口，不启动设备即可验证命令构造。

### 9C：exact-source NPU16 admission

- exact source 与阶段目录依赖 SHA 固定；
- P/D 16 rank 同时加载，endpoint readiness 通过；
- 运行确定性 4-token smoke 和 BS2 late admission；
- 检查所有输出、KDA state、MLA cache 和 OE state finite；
- 拉回匿名日志、cache/HBM 账本和 request summary，精确清理全部进程与端口。

9A、9B、9C 分别提交实现/测试；NPU16 结果另写独立验证记录。任一子阶段失败不跳过到下一阶段。

## 8. 测试与门槛

### 本地 focused tests

- 新 mapping 的全部 size/group 坐标精确匹配；相邻非法组合 fail closed；
- Lite MLA backend 为 32 个 replica head，非 Lite TP8 仍为本地 head 数；
- P/D `CacheTransferContract`、field order、dtype、offset 和 byte size exact；
- Prefill 多 token 使用 lockstep Grouped MoE 路径，CPU oracle 与 N=1 语义对齐；
- launcher dry-run 覆盖 card overlap、端口冲突、缺依赖和容量非法负例。

### NPU16 门槛

- 16/16 worker ready，P/D 分别形成且只形成一个 8-rank HCCL world；
- exact checkpoint strict coverage，OE 主表 HBM 增量为 0；
- 单请求 4-token greedy smoke 返回非空、无错误、token 数正确；
- BS2 两请求均完成，late admission 不复用错误的 KDA/MLA/OE state；
- P/D transfer manifest、每组 block count、目标 slot 和 completion status exact；
- 所有 activation/cache/state/logits finite；
- 停止后 engine/gateway/worker 为 0、全部测试端口可绑定、NPU0--15 无遗留进程。

阶段 9 通过只证明有界 eager 服务的功能闭环。GSM8K 前 100 条、Decode graph/overlap、融合累计准入
和最终内存/性能分别留在后续阶段；分布式 KV 仍只在最后 TODO 实现。

## 9. 回退

阶段 9 的 replicated MLA topology 是新增的独立合法组合，不改变原 CP8/attention-DP8 组合。若它在
服务上出现数值、PD 或内存失败，回退只移除该组合及 launcher；阶段 1--8 的 model、kernel、loader、
role probe 和官方 dispatch/combine 准入实验均不受影响。禁止用关闭 strict loader、跳过 cache state、
复用 V2 MoE 通信或注入 reference activation 的方式让服务假通过。
