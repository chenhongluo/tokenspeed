# Lite NPU 阶段 6C：EP8 Grouped MoE Placement Board 设计

## 1. 目标与边界

本阶段只在八张 Ascend 910B 上比较三种 Lite Grouped MoE executor/placement：

- A：四个独立 EP8 executor；
- B：一个 flattened-interleaved EP8 executor；
- C：一个 flattened-group-major EP8 executor。

实验固定同一输入、路由、逻辑 expert 权重、BF16 数学和 EP8 group，得到数值、负载、延迟、通信与
HBM 证据，并选出唯一 production winner。Lite decoder、dense/shared projection、完整 checkpoint
decoder、P/D 服务和 GSM8K 不在本阶段接线；它们只在 6D 复用 winner。

A/B/C 只存在于离线 board harness。production runtime 不增加 layout 开关、三套 executor 或第二套
process-group 抽象。

## 2. 实现追踪后的通信边界修正

阶段 6 总体设计把 A 描述成四套 expert collective、B/C 描述成一套。继续追踪 TokenSpeed 真实调用链后，
该边界需要在 6C 修正：

```text
decoder
  -> CommManager.pre_moe_comm
  -> 整个 MLP / Grouped MoE
  -> CommManager.post_moe_comm
```

collective 包住整个 MLP，而不是位于每个 expert executor 内。P 的 token-aware AG/RS 和 D 的 AllReduce
都必须对拼回的完整 `[T,3072]` Grouped MoE lane 调用一次。A/B/C 的真实差异是 local leaf 数从四次降为
一次，以及 real expert 的 rank placement；不能为了放大 B 的收益而让 A 重复 production 不会执行的
collective。

因此 board 固定：

| 模式 | P collective | D collective | local leaf |
| --- | --- | --- | ---: |
| A | 1 次 token AG + 1 次 token RS | 1 次 AllReduce | 4 |
| B | 1 次 token AG + 1 次 token RS | 1 次 AllReduce | 1 |
| C | 1 次 token AG + 1 次 token RS | 1 次 AllReduce | 1 |

阶段 6 总体设计中“B 必须把四次 collective 合为一次”的门槛据此改为“B 必须把四次 local leaf 合为
一次且 collective 次数不变”。后续 6D 以本设计为 production seam。

## 3. 固定模型参数

| 参数 | 值 |
| --- | ---: |
| group `G` | 4 |
| real expert / group | 384 |
| identity expert / group | 32 |
| TopK / group | 12 |
| group hidden `D` | 768 |
| expert intermediate `I` | 512 |
| EP | 8 |
| local real experts / rank | 192 |

router 仍按四组独立执行 `softmax(logits)`、`topk(prob+bias)` 和无 bias probability gather。board 缓存一次
逻辑 `(group,expert)` 路由，A/B/C 只做 physical ID remap，不重新选择 expert。

## 4. 三种 placement

令 `0<=g<4`、`0<=e<384`。

### 4.1 A：四个独立 EP8 executor

```text
rank_A(g,e)  = floor(e / 48)
local_A(g,e) = e mod 48
```

每 rank 每组 48 个 expert。每组输入 `[T,768]` 分别调用一次 6B leaf，四个 partial output 拼成
`[T,3072]` 后才进入唯一 collective。

### 4.2 B：flattened-interleaved

```text
rank_B(g,e)     = floor(e / 48)
local_B(g,e)    = g * 48 + (e mod 48)
physical_B(g,e) = rank_B * 192 + local_B
```

每 rank 仍持有每组 48 个 expert。输入按 group-major 展平为 `[4T,768]`，四组 route remap 后只调用
一次 leaf，再恢复为 `[T,3072]`。

### 4.3 C：flattened-group-major

```text
physical_C(g,e) = g * 384 + e
rank_C(g,e)     = g * 2 + floor(e / 192)
local_C(g,e)    = e mod 192
```

每 rank 只持有一个 group 的 192 个 expert；相邻两个 rank 覆盖完整一组。输入/output tensor layout与 B
相同，只改变 real route physical ID 和 local weight owner。

每种 placement 都生成正反映射并验证 1536 个逻辑 expert 无重、无漏、无重复 owner。

## 5. 权重与路由输入

### 5.1 Synthetic deterministic

synthetic 权重由共享 base matrix 和 logical expert ID scale 构造。同一 `(g,e)` 在 A/B/C 中逐元素一致，
每 rank 只 materialize 自己的 192 个 expert。每轮只保留一个 layout，结束后释放，避免三份约
`453 MiB/rank` expert 权重同时驻留。

路由包含：

- 均匀 real route；
- hot expert / pair 内偏斜；
- identity-only、real-only 和 mixed；
- duplicate route、首末 expert ID；
- 让部分 rank 的 local active route 为零但仍进入相同 collective。

identity route 不进入 GMM。它在 leaf 中映射成 non-local documented input；identity contribution 必须在
EP collective 之后只加一次，避免 D AllReduce 把 replicated identity 乘八。

### 5.2 Checkpoint single-layer replay

harness 通过环境变量接收 checkpoint 目录和 layer ID，不把私有路径写入源码、文档或日志。它读取
`model.safetensors.index.json`，只打开 router 与当前 layout/rank 拥有的 expert tensors：

```text
model.layers.{layer}.mlp.expert_groups.{g}.router.*
model.layers.{layer}.mlp.experts.{g*384+e}.{gate,up,down}_proj.weight
```

router 在相同 grouped hidden 上只运行一次；A/B/C 复用同一逻辑 ID/weight。所有 rank 在加载完成后
barrier，再进入相同顺序的实验，禁止某个 rank 因空 route 提前退出。

## 6. P 与 D 数据流

### 6.1 P eager

每 rank 初始持有不等长 local token rows `[T_r,3072]`：

1. 使用既有 HCCL `token_all_gather` 一次，得到 rank-major 全局 `[T,3072]`；
2. reshape 为 `[T,4,768]`，执行缓存的四组路由和 A/B/C local leaf；
3. partial group output 恢复为 `[T,3072]`；
4. 使用既有 `token_reduce_scatter` 一次，返回 `[T_r,3072]`；
5. 在 local rows 上加入一次 identity contribution。

总 token 与不等长 split 固定为：

| 总 token | rank0..7 split |
| ---: | --- |
| 32 | `1,3,4,5,2,6,7,4` |
| 128 | `13,15,17,19,11,21,14,18` |
| 1024 | `125,127,129,131,123,133,126,130` |

### 6.2 D eager / graph

八个 rank 消费相同 `[BS,4,768]` input 和路由：

1. A 调四次 local leaf；B/C 对 `[4BS,768]` 调一次；
2. 恢复 partial `[BS,3072]`；
3. 使用既有 HCCL `all_reduce` 一次；
4. 加入一次 replicated identity contribution。

BS1/BS2 先 eager 预热，再 capture 完整 local leaf + AllReduce + identity 链。replay 前原地更新 hidden、
route weight 和 route ID；输出必须变化且不创建新 specialization。若目标 HCCL 不支持在当前 NPUGraph
边界捕获，必须记录准确失败阶段，不允许以 eager 结果冒充 graph。

## 7. Harness 与复用边界

最小实现只增加一个可由 pytest/`torchrun` 启动的离线 harness：

- `Mapping(..., moe_ep_size=8)` 生成唯一 `mapping.moe.ep_group`；
- `ProcessGroupManager` 初始化并登记 HCCL group；
- `token_all_gather`、`token_reduce_scatter`、`all_reduce` 走现有 comm facade；
- expert 计算只调用阶段 6B 的 `moe_plan/moe_process_weights/moe_apply`；
- A/B/C placement、checkpoint 读取与统计只存在于 test harness。

不修改 Lite runtime、loader、kernel registry、launcher 或 comm backend。board 选出 winner 后，6D 才把
一个 mapping/remap 接入 Lite wrapper。

## 8. 观测与计时

每个 case 记录所有 rank 的：

- 每 group real/identity route 数；
- 每 rank 和相邻 rank-pair 的 real route 数、`max/mean`、变异系数、零 route rank；
- local leaf 调用数、GMM active rows；
- collective 调用数和按 tensor shape 计算的 payload bytes；
- local leaf NPU Event 时间；
- 含 collective 的同步 wall latency，取八 rank 的 max；
- `max_memory_allocated` 与 `max_memory_reserved`。

每个 layout/case 先 warmup 20 次，再记录 50 次。计时窗口前后 barrier 与 NPU synchronize；统计使用
独立小 tensor collective，不混入被测窗口。A/B/C 在同一进程、相同 stream 和相同 case 顺序相邻执行。

## 9. 正确性与 winner 门槛

- placement 1536 expert owner 双射 exact；
- 非 tie route ID exact，weight `atol=1e-6,rtol=1e-5`；
- A 对 synthetic FP32 accumulation oracle满足 `atol=2e-2,rtol=2e-2`、rel-L2 `<=5e-3`；
- B/C 对 A 的 partial/collective output 满足相同阈值，完整 Grouped MoE lane rel-L2 `<=1e-2`；
- 全部中间值和输出 finite；identity、duplicate、boundary、zero-local-route 全通过；
- P 三个 token split、D eager BS1/2 和 D graph BS1/2 全通过；
- B 的 local leaf 调用从 4 降为 1，P/D 稳态均不慢于 A，HBM peak 相对 A 不增加超过 5%；
- C 的 P/D 稳态相对 B 回退不超过 3%，HBM 不增加；route/pair 负载必须完整记录，不能仅凭 expert
  数量宣称均衡。

选择顺序保持单调：B 任一硬门槛失败则 winner=A；B 通过而 C 失败则 winner=B；只有 C 全部门槛通过
才选 C。失败 layout 不进入 production，但负结果保留在独立验证记录。

## 10. 资源与运行约束

board 每次从既有 NPU 资源入口动态解析当前作业，不缓存地址。进程只暴露设备 0--7，并以：

```text
torchrun --standalone --nproc-per-node=8 -m pytest ...
```

启动。实验前检查八张卡无本任务遗留进程；结束后销毁 process group，并再次确认无 board marker。
第二组 8 卡不参与本阶段。

## 11. 提交与完成条件

1. 本设计文档独立 signed-off 提交并推送；
2. harness 与 CPU/placement checks 独立提交并推送；
3. 从真实实现提交生成 exact archive，运行 synthetic 与 checkpoint NPU8 board；
4. winner、原始统计摘要、负结果和 archive SHA 写入独立验证记录，提交并推送。

只有 winner 已唯一确定、NPU8 进程清理完成、local/tracking/remote SHA 一致后，阶段 6C 才关闭并进入
6D production 接线。
